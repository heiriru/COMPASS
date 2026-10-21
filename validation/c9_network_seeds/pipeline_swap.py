"""E-C9.3 through the real pipeline: ModelTransfuser.compare on two seeds of one model.

    .COMPASS/bin/python -m validation.c9_network_seeds.pipeline_swap --family gauss

Two networks trained on the same simulator (2^16 sims, seeds 0 and 1) are registered
as two "models" and compared on 32 disjoint datasets of 16 rows from that family's
test set (each row its own theta: flat model, k_local = 2). The full default
pipeline runs: posterior sampling, MAP, PF-ODE MLE and counted AICc (-> AIC fallback).
The right answer is 50/50. The per-dataset log-likelihood difference is also
recorded at the MLE, to compare with the true-theta swap test in analyze_swap.py.
"""

import validation.common.runtime_gpu  # noqa: F401

import argparse
import contextlib
import io
import json
import time
import os

import torch

from compass.ModelTransfuser import ModelTransfuser
from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm
from validation.b5_b6_pfode_accuracy.evaluate import test_set
from validation.b5_b6_pfode_accuracy.families import Family
from validation.b5_b6_pfode_accuracy.train import model_dir

OUT = os.path.join(os.path.dirname(__file__), "results", "pipeline_swap")
N_ROWS, N_SETS = 16, 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--seeds", type=int, nargs=2, default=(0, 1))
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{args.family}_s{args.seeds[0]}_vs_s{args.seeds[1]}.jsonl")
    done = sum(1 for _ in open(path)) if os.path.exists(path) else 0

    family = Family(args.family, 0.5)
    _, x = test_set(family)
    models = {f"seed{s}": SBIm.load(os.path.join(model_dir(args.family, 0.5, 65536, s),
                                                 "final.pt"), device="cuda")
              for s in args.seeds}
    for m in models.values():
        m.model.eval()
    names = list(models)
    for k in range(done, N_SETS):
        xs = x[k * N_ROWS:(k + 1) * N_ROWS]
        t0 = time.time()
        mtf = ModelTransfuser()
        torch.manual_seed(1000 + k)
        with contextlib.redirect_stdout(io.StringIO()):
            mtf.add_models(models)
            mtf.trained_models = True
            # Lighter MLE than the default (4 starts x 50 iterations): with the default a
            # single two-model comparison of 16 rows takes > 10 min, and B6 showed 20
            # iterations reach the maximum at 2^16 while more mainly exploit network error.
            mtf.compare(x=xs, num_samples=256, timesteps=50, device="cuda", verbose=False,
                        mle_starts=2, mle_iterations=20)
        st = mtf.stats
        rec = {"set": k, "seconds": time.time() - t0,
               "prob_first": st[names[0]]["model_prob"],
               "delta_loglik_mle": float(st[names[0]]["log_probs"].sum()
                                         - st[names[1]]["log_probs"].sum()),
               "delta_loglik_map": float(st[names[0]]["log_probs_at_map"].sum()
                                         - st[names[1]]["log_probs_at_map"].sum()),
               "criterion": st[names[0]].get("criterion_used")}
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)


if __name__ == "__main__":
    main()
