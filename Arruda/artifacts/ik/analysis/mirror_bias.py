"""Reference-free test of the IK single-observation network: is it mirror-equivariant?

The reflection (h, a1, a2, a3) -> (-h, -a1, -a2, -a3) maps x = (x1, x2) to Mx = (-x1, x2)
and leaves the prior invariant, so exactly p(g | Mx) = p(-g | x). Hence for every mirror
pair, logit P(g>0 | x) + logit P(g>0 | Mx) = 0. What the network returns instead is a
per-pair branch log-odds bias. Composition multiplies likelihoods, so if the network's
error is a branch reweighting the biases add: the composed log-odds on a mirror-paired
dataset (exact answer: 0, i.e. 50/50) is predicted to be the sum of the pair biases.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "Arruda")]
sys.argv = [sys.argv[0], "--benchmark", "ik", "--stage", "sample", "--weights-dir", str(ROOT / "Arruda/artifacts/ik")]
import run_arruda as R  # applies the 3-thread CPU cap on import

args = R.argparse.Namespace(benchmark="ik", weights_dir=str(ROOT / "Arruda/artifacts/ik"), tag="diag", retrain=False,
                            train_samples=16_384, batch_size=64, epochs=0, seed=20260923)
approximator = R.train_or_load(args, None)
DRAWS = 8_000
logit = lambda p: np.log(p) - np.log1p(-p)
arruda = {(r["n"], r["bridge"]): float(np.mean(np.load(r["samples_path"]) > 0))
          for r in json.loads((ROOT / "Arruda/artifacts/ik/mirror_paired/results.json").read_text())}
rows = []
for n in (2, 4, 8, 16, 32, 64, 128):
    rng = np.random.default_rng(args.seed + n)                      # identical to ik_tall_problem("mirror_paired")
    half = R.ik_study.selected_dataset(n // 2, true_angle=R.IK_TRUE_ANGLE, epsilon=0.15, rng=rng)
    x = np.concatenate([half, half * np.array([-1.0, 1.0])]).astype("float32")
    g = np.concatenate([np.asarray(approximator.sample(num_samples=DRAWS, conditions={"x": x[i:i + 16]})["g"])[..., 0]
                        for i in range(0, n, 16)])                  # (n, DRAWS), chunked to fit GPU memory
    p = np.clip((g > 0).mean(axis=1), 0.5 / DRAWS, 1 - 0.5 / DRAWS)
    pair_bias = logit(p[: n // 2]) + logit(p[n // 2:])              # 0 for an equivariant network
    mean_bias = g[: n // 2].mean(1) + g[n // 2:].mean(1)            # E[g|x] + E[g|Mx], 0 if equivariant
    total = float(pair_bias.sum())
    rows.append(dict(n=n, mean_P_pos=float(p.mean()), pair_bias_mean=float(pair_bias.mean()),
                     pair_bias_sd=float(pair_bias.std()), mean_location_bias=float(mean_bias.mean()),
                     predicted_composed_P_pos=float(1 / (1 + np.exp(-total))),
                     arruda_P_pos_d1=arruda.get((n, 1.0)), arruda_P_pos_d01=arruda.get((n, 0.1))))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
(HERE / "mirror_bias.json").write_text(json.dumps(rows, indent=2) + "\n")
