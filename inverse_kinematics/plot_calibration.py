#!/usr/bin/env python3
"""Calibration of the composed posterior, shared coordinate and locals together.

Reads ``artifacts/bimodal/composed_evaluation.json`` and draws the rank ECDF
difference for every parameter and every composition rule -- the same diagnostic
BayesFlow's ``calibration_ecdf`` shows, so the two can be read side by side.

A calibrated sampler produces ranks uniform on (0, 1), so the plotted difference
``ECDF(r) - r`` hugs zero inside the band. The shapes name their failure:

    drifts positive then negative   posterior too wide  (ranks pile in the middle)
    drifts negative then positive   posterior too narrow (ranks pile at the tails)
    monotone offset                 posterior biased

The band is the pointwise 95% envelope for the number of *independent* draws.
For the shared angle that is the number of replicates. For the locals it is also
the number of replicates, **not** the number of ranks: the three local
coordinates of the eight observations in one dataset all condition on the same
composed shared draw, so they are not independent samples, and using their raw
count would draw a band about `sqrt(24)` times too tight and manufacture
failures.

Usage:
    python inverse_kinematics/plot_calibration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import hierarchical as H  # noqa: E402
from plots import INK, RAIL, _style_axis  # noqa: E402
from plots_bimodal import METHOD_COLOURS  # noqa: E402

ROOT = Path(__file__).resolve().parent
BAND = "#DFE4E8"


def ecdf_difference(ranks, grid):
    ranks = np.sort(np.asarray(ranks, dtype=np.float64))
    ecdf = np.searchsorted(ranks, grid, side="right") / len(ranks)
    return ecdf - grid


def plot(summary, replicates, output):
    names = ["angle_1 (shared)"] + [f"{n} (local)" for n in H.LOCAL_NAMES]
    methods = list(summary)
    grid = np.linspace(0.0, 1.0, 400)
    # Pointwise 95% envelope for `replicates` independent uniforms.
    envelope = 1.96 * np.sqrt(grid * (1 - grid) / max(replicates, 1))

    figure, axes = plt.subplots(
        1, 4, figsize=(13, 3.2), layout="constrained", sharey=True,
    )
    for index, (axis, title) in enumerate(zip(axes, names)):
        axis.fill_between(grid, -envelope, envelope, color=BAND, linewidth=0,
                          zorder=0, label="95% band" if index == 0 else None)
        axis.axhline(0.0, color=RAIL, linewidth=1.0, zorder=1)
        for method in methods:
            entry = summary[method]
            if index == 0:
                ranks, count = entry["ranks"], replicates
            else:
                local = entry.get("local", {}).get(H.LOCAL_NAMES[index - 1])
                if not local:
                    continue
                ranks, count = local["ranks"], replicates
            axis.plot(grid, ecdf_difference(ranks, grid),
                      color=METHOD_COLOURS.get(method, INK), linewidth=1.7,
                      zorder=3,
                      label=method.replace("_", " ") if index == 0 else None)
        _style_axis(axis)
        axis.set_xlim(0, 1)
        axis.set_xlabel("Fractional rank", color=INK)
        axis.set_title(title, fontsize=10, color=INK)
    axes[0].set_ylabel("ECDF difference", color=INK)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    figure.suptitle(
        f"Calibration of the composed posterior — {replicates} replicates, "
        "8 observations, plain prior predictive\n"
        "inside the band = ranks consistent with Uniform(0,1)",
        fontsize=11, color=INK,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv=None):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    arguments = result.parse_args(argv)

    source = arguments.artifacts / "bimodal" / "composed_evaluation.json"
    payload = json.loads(source.read_text())
    if payload.get("select_x1", 0) > 0:
        raise SystemExit(
            "This evaluation used --select-x1 > 0, where SBC ranks are not "
            "interpretable (see evaluate_composed.py). Re-run with "
            "--select-x1 0 before plotting calibration."
        )
    figures = arguments.artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = plot(payload["summary"], payload["replicates"],
                figures / "17_calibration_global_local.png")

    print(f"{'rule':>22}{'parameter':>22}{'mean rank':>11}{'cov50':>8}"
          f"{'cov90':>8}{'KS':>8}{'uniform?':>10}")
    for method, entry in payload["summary"].items():
        rows = [("angle_1 (shared)", entry["mean_rank"], entry["coverage_50"],
                 entry["coverage_90"], entry["ks_distance"],
                 entry["ks_critical_95"])]
        for name, local in entry.get("local", {}).items():
            rows.append((f"{name} (local)", local["mean_rank"],
                         local["coverage_50"], local["coverage_90"],
                         local["ks_distance"], local["ks_critical_95"]))
        for label, mean, c50, c90, ks, critical in rows:
            print(f"{method:>22}{label:>22}{mean:>11.3f}{c50:>8.2f}"
                  f"{c90:>8.2f}{ks:>8.3f}"
                  f"{('yes' if ks < critical else 'NO'):>10}")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
