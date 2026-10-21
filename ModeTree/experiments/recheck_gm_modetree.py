#!/usr/bin/env python
"""Re-measure the Gaussian-mixture ModeTree arm after the acceptance-sign fix.

``_refine_candidates`` negated the precision ``refine_batch`` returns, so every
genuine mode was rejected and every saddle admitted. The learned-arm result in
``gaussian_mixture_modetree.json`` ("no branches found" at every n) was recorded
through that path and is not trustworthy. This re-runs the ModeTree arm only --
the baselines are unaffected and take hours -- on both score sources.

Run:  python ModeTree/experiments/recheck_gm_modetree.py --observations 8 16
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "Gaussian_mixtures_compositional_score_modelling"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402

import numpy as np  # noqa: E402

import datasets     # noqa: E402
import model as M   # noqa: E402

from experiment_gaussian_mixture import (  # noqa: E402
    load_source, mixture_density, reference, run_modetree, w1_from_density,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--sources", nargs="+", default=("learned", "oracle"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    mixture = M.DEFAULT
    rows = []
    for kind in args.sources:
        source = load_source(kind, mixture)
        for n in args.observations:
            rng = np.random.default_rng(
                [args.seed, datasets.REGIMES.index("asymmetric"), n, 0])
            data = datasets.make_dataset("asymmetric", n, rng, mixture=mixture,
                                         branch_mass=0.75, spread=1.0)
            x = np.asarray(data["x"], dtype=np.float64)
            grid, exact = reference(mixture, x)
            exact_weights = np.asarray(mixture.branch_weights(x))
            exact_means = np.asarray(mixture.branch_means(x))

            posterior, seconds, tree, row_source = run_modetree(source, x, mixture)
            if posterior is None:
                print(f"{kind:8s} n={n:4d}  no branches found  {seconds:.1f}s",
                      flush=True)
                rows.append({"score_source": kind, "n": n, "seconds": seconds,
                             "error": "no branches found"})
                continue
            order = np.argsort([float(b.global_state) for b in posterior.branches])
            means = np.array([float(posterior.branches[i].global_state)
                              for i in order])
            sds = np.array([float(posterior.branches[i].global_covariance.sqrt())
                            for i in order])
            weights = np.array([float(posterior.weights[i]) for i in order])
            w1 = w1_from_density(grid, mixture_density(grid, means, sds, weights),
                                 exact)
            aligned = np.zeros(2)
            for position, mean in enumerate(means):
                aligned[int(np.argmin(np.abs(exact_means - mean)))] += weights[position]
            tv = 0.5 * float(np.abs(aligned - exact_weights).sum())
            print(f"{kind:8s} n={n:4d}  branches={len(means)}  W1={w1:.4f}  "
                  f"TV={tv:.4f}  means={np.round(means, 3).tolist()}  "
                  f"exact={np.round(exact_means, 3).tolist()}  {seconds:.1f}s",
                  flush=True)
            rows.append({"score_source": kind, "n": n, "seconds": seconds,
                         "branches": len(means), "w1": w1, "weight_tv": tv,
                         "means": means.tolist(), "sds": sds.tolist(),
                         "weights": weights.tolist(),
                         "exact_means": exact_means.tolist(),
                         "exact_weights": exact_weights.tolist()})
    with open(ARTIFACTS / "gm_modetree_recheck.json", "w") as handle:
        json.dump(rows, handle, indent=1)
    print(f"\nwrote {ARTIFACTS / 'gm_modetree_recheck.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
