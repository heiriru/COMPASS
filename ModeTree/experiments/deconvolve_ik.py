#!/usr/bin/env python
"""Remove the known lambda^2 diffusion smoothing from ModeTree's IK branches.

ModeTree stops the clean composition at ``switch_to_clean_t`` because below it
the network's Jacobian is not Tweedie-admissible, so its target is the
*diffused* posterior ``p_t`` rather than ``p_0``. For a Gaussian branch that is
an exact statement -- ``Var_t = Var_0 + lambda(t)^2`` -- so the bias is
removable in closed form rather than merely bounded.

This re-grades the stored branches with ``sd^2 -> sd^2 - lambda^2`` and reports
both numbers, which separates "ModeTree found the wrong mode" from "ModeTree
found the right mode at the wrong resolution". No re-run: means, standard
deviations and weights are already in the artifacts.

Run:  python ModeTree/experiments/deconvolve_ik.py --tag ik_mt_t003 --t 0.03
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "inverse_kinematics"))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402

import numpy as np  # noqa: E402

from modetree.diffusion import VESchedule           # noqa: E402
from experiment_ik import mixture_density, w1_density  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="ik_mt_t003")
    parser.add_argument("--t", type=float, default=0.03)
    parser.add_argument("--sde-sigma", type=float, default=8.0)
    args = parser.parse_args(argv)

    lam = VESchedule(sigma=args.sde_sigma).lam(args.t)
    print(f"terminal t = {args.t}  ->  lambda = {lam:.4f}  "
          f"(variance removed: {lam ** 2:.5f})\n")

    with open(ARTIFACTS / f"{args.tag}_modetree.json") as handle:
        rows = json.load(handle)
    with open(ARTIFACTS / f"{args.tag}_panels.json") as handle:
        panels = json.load(handle)

    out = []
    for row in rows:
        if row["method"] != "modetree" or "error" in row:
            continue
        key = f"{row['regime']}|{row['n']}"
        grid = np.asarray(panels[key]["grid"])
        exact = np.asarray(panels[key]["exact"])
        means = np.asarray(row["branch_means"])
        sds = np.asarray(row["branch_sds"])
        weights = np.asarray(row["branch_weights"])

        variance = sds ** 2 - lam ** 2
        floored = int((variance <= 0).sum())
        # A branch narrower than the smoothing carries no recoverable width:
        # report it rather than silently flooring and calling the result clean.
        corrected = np.sqrt(np.clip(variance, 1e-6, None))
        w1_raw = w1_density(grid, mixture_density(grid, means, sds, weights), exact)
        w1_deconv = w1_density(
            grid, mixture_density(grid, means, corrected, weights), exact)
        reference_sd = float(np.sqrt(np.trapezoid(
            grid ** 2 * exact, grid) - np.trapezoid(grid * exact, grid) ** 2))
        entry = {
            "regime": row["regime"], "n": row["n"],
            "w1_over_sigma_raw": row["w1_over_sigma"],
            "w1_over_sigma_deconvolved": w1_deconv / reference_sd,
            "width_ratio_raw": row["width_ratio"],
            "sd_raw": sds.tolist(), "sd_deconvolved": corrected.tolist(),
            "branches_floored": floored, "lambda": lam,
        }
        out.append(entry)
        print(f"{row['regime']:8s} n={row['n']:4d}  "
              f"W1/sd {row['w1_over_sigma']:.4f} -> {entry['w1_over_sigma_deconvolved']:.4f}"
              f"   sd {np.round(sds, 4).tolist()} -> {np.round(corrected, 4).tolist()}"
              + ("  [floored]" if floored else ""))

    with open(ARTIFACTS / f"{args.tag}_deconvolved.json", "w") as handle:
        json.dump(out, handle, indent=1)
    print(f"\nwrote {ARTIFACTS / (args.tag + '_deconvolved.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
