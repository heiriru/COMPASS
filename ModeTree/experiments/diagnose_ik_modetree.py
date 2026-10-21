#!/usr/bin/env python
"""Why does ModeTree find no branches on the IK network?

Two questions, asked separately:

1. **Does the clean composed field have maxima where the reference says the
   modes are?**  Seed the Newton refinement at the ABC reference's own mode
   locations and report where it lands, its residual, and whether the curvature
   test passes.  If this fails, no amount of mode *discovery* can help -- the
   target ModeTree is shown simply does not have the modes.

2. **If it does, where in the tree are they lost?**  Run the tree with event
   logging and print the prune reasons.

Run:  python ModeTree/experiments/diagnose_ik_modetree.py --observations 4
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IK = ROOT / "inverse_kinematics"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(IK))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

import hierarchical as H       # noqa: E402
import train_model as trainer  # noqa: E402

from compass.priors import GaussianPrior               # noqa: E402
from modetree.compass_adapter import CompassRowSource  # noqa: E402
from modetree.diffusion import ComposedField           # noqa: E402
from modetree.refine import refine_mode                # noqa: E402
from modetree.tree import ModeTree, ModeTreeConfig, TimeComposer  # noqa: E402
from experiment_ik import CHECKPOINT, make_dataset, reference  # noqa: E402
from bimodality_study import bimodality                # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def build(model, x):
    prior = GaussianPrior(
        torch.as_tensor(H.GLOBAL_PRIOR_MEAN, dtype=torch.float64),
        torch.as_tensor(H.GLOBAL_PRIOR_STD ** 2, dtype=torch.float64),
    )
    source = CompassRowSource(model, x, H.HIERARCHY, H.LOCAL_INDICES,
                              H.OBSERVED_INDICES, prior)
    field = ComposedField(source, "gauss_jacobian", jacobian_refresh=4)
    return source, field


def seeded_refinement(field, x, modes, t_values, iterations=200):
    """Newton from the reference modes, at several terminal times."""
    n = x.shape[0]
    d_g, d_l = len(H.HIERARCHY), len(H.LOCAL_INDICES)
    local_seed = torch.as_tensor(H.LOCAL_PRIOR_MEAN, dtype=torch.float64)
    rows = []
    for t in t_values:
        clean = TimeComposer(field, float(t))
        for mode in modes:
            state = torch.cat([
                torch.full((d_g,), float(mode), dtype=torch.float64),
                local_seed.repeat(n),
            ])
            result = refine_mode(clean, state, tol=1e-11, max_iter=iterations)
            precision = result.blocks.global_precision()
            row = {
                "t": float(t), "seed": float(mode),
                "landed": float(result.theta[0]),
                "score_norm": float(result.score_norm),
                "iterations": int(result.iterations),
                "is_local_maximum": bool(
                    result.diagnostics["is_local_maximum"]),
                "global_precision": float(precision.reshape(-1)[0]),
            }
            if result.blocks.C is not None:
                row["min_eig_local"] = float(
                    torch.linalg.eigvalsh(-result.blocks.C).min())
            rows.append(row)
            print(f"  t={t:<8g} seed {mode:+.3f} -> {row['landed']:+.4f}  "
                  f"||S||={row['score_norm']:.2e}  Pg={row['global_precision']:+.3f}"
                  f"  min eig(-C)={row.get('min_eig_local', float('nan')):+.3f}"
                  f"  max={row['is_local_maximum']}", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=4)
    parser.add_argument("--regime", default="bimodal")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--reference-draws", type=int, default=20_000)
    parser.add_argument("--times", type=float, nargs="+",
                        default=(1e-3, 1e-2, 3e-2, 0.1))
    parser.add_argument("--run-tree", action="store_true")
    args = parser.parse_args(argv)

    model = trainer.load_to(CHECKPOINT, "cpu")
    truth, x = make_dataset(args.regime, args.observations, args.seed)
    grid, exact = reference(x, args.reference_draws, 0.05, 201, args.seed)
    shape = bimodality(grid, exact)
    print(f"reference modes {[round(m, 3) for m in shape['modes']]} "
          f"dip {shape['dip_ratio']:.3f}  (truth {truth:+.3f})", flush=True)

    source, field = build(model, x)
    print("\nNewton seeded at the reference modes:")
    rows = seeded_refinement(field, x, shape["modes"], args.times)

    events = None
    if args.run_tree:
        config = ModeTreeConfig(
            levels=16, probe_iterations=30, clean_refine_iterations=60,
            sentinel_checkpoints=2, sentinel_sobol=4, jacobian_stride=4,
            switch_to_clean_t=1e-3, lambda_max=8.0, return_diagnostics=True,
            verbose=True)
        tree = ModeTree(field, TimeComposer(field, 1e-3), config,
                        prior_std=torch.as_tensor(H.GLOBAL_PRIOR_STD,
                                                  dtype=torch.float64))
        branches = tree.run()
        print(f"\nbranches: {len(branches)}  counters: {tree.counters}")
        events = [e for e in tree.events if e.get("event") == "prune"]
        for event in events[:20]:
            print(" prune:", event)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "ik_modetree_diagnosis.json", "w") as handle:
        json.dump({"observations": args.observations, "regime": args.regime,
                   "truth": truth, "reference_shape": shape,
                   "seeded": rows, "prunes": events}, handle, indent=1)
    print(f"\nwrote {ARTIFACTS / 'ik_modetree_diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
