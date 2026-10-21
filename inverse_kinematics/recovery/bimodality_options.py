#!/usr/bin/env python3
"""Two routes to a bimodal IK target, checked on the exact posterior before any network is trained.

1. **Wider height prior, natural data.** The mirror twin costs `2 x1 (x1 - h) / sigma_h^2` per
   observation; widening sigma_h shrinks that by its square. Data are drawn from the *same* wider
   prior (no filtering), so there is no selection effect.
2. **Closer modes, filtered data.** True angle_1 = 0.25 instead of 0.6, rows filtered as before
   (|x1| < 0.15, or |x1 - h| < 0.05), stock prior.

angle_1 is the only global; exact grid posterior via ik_exact.log_marginal_a1 (no MCMC needed
for a 1-D target, and it is the same density the MCMC was validated against).

Usage (from COMPASS/): .COMPASS/bin/python inverse_kinematics/recovery/bimodality_options.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import hierarchical as H  # noqa: E402
import ik_exact as E  # noqa: E402
import simulator as sim  # noqa: E402
from bimodality import shape  # noqa: E402

OUT = HERE / "artifacts" / "bimodality"
NS = (1, 2, 8, 32, 128)
GRID = np.linspace(-2.0, 2.0, 4001)
SD_HS = (0.25, 0.5, 1.0, 2.0)
TRUTHS = (0.25, 0.6)
COUNT = 50


def draw_rows(rng, truth, sd_h, rows=128, filter_=None):
    """``rows`` observations with a1 = truth, h ~ N(0, sd_h), a2, a3 ~ N(0, .5); optional filter."""
    kept = []
    while sum(len(k) for k in kept) < rows:
        th = np.column_stack([np.full(20_000, truth), rng.normal(0, sd_h, 20_000),
                              rng.normal(0, E.SD_A2, 20_000), rng.normal(0, E.SD_A3, 20_000)])
        x = H.forward(th)
        if filter_ == "x1sel":
            x = x[np.abs(x[:, 0]) < 0.15]
        elif filter_ == "rail":
            x = x[np.abs(x[:, 0] - th[:, 1]) < 0.05]
        kept.append(x)
    return np.concatenate(kept)[:rows]


def posteriors(x, sd_h):
    """Exact p(a1 | first n rows) on GRID for every n in NS."""
    table = np.stack([E.log_marginal_a1(GRID, x[j, 0], x[j, 1], nodes=32, wraps=(0,), sd_h=sd_h)
                      for j in range(len(x))], 1)
    out = {}
    for n in NS:
        lp = E.log_normal(GRID, E.SD_A1) + table[:, :n].sum(1)
        p = np.exp(lp - lp.max())
        out[n] = p / np.trapezoid(p, GRID)
    return out


def summarise(curves, truth):
    rows = {}
    for n in NS:
        stats = [shape(GRID, c[n]) for c in curves]
        sds = [np.sqrt(np.trapezoid((GRID - np.trapezoid(GRID * c[n], GRID)) ** 2 * c[n], GRID)) for c in curves]
        bimodal = [s for s in stats if s["bimodal"]]
        near = lambda s, v: any(abs(p - v) < 0.1 for p in s["peaks"])  # noqa: E731
        rows[n] = {"bimodal_fraction": float(np.mean([s["bimodal"] for s in stats])),
                   "both_truth_and_mirror_fraction": float(np.mean([near(s, truth) and near(s, -truth) for s in stats])),
                   "median_posterior_sd": float(np.median(sds)),
                   "median_p_positive": float(np.median([s["p_positive"] for s in stats])),
                   "median_peak_separation": float(np.median([max(s["peaks"]) - min(s["peaks"]) for s in bimodal]))
                   if bimodal else None}
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"wider_height_prior": {}, "closer_modes_filtered": {}}
    examples = {}
    for sd_h in SD_HS:
        for truth in TRUTHS:
            rng = sim.generator(f"options-sdh{sd_h}-a{truth}")
            curves = [posteriors(draw_rows(rng, truth, sd_h), sd_h) for _ in range(COUNT)]
            examples[(sd_h, truth)] = curves[0]
            rows = summarise(curves, truth)
            report["wider_height_prior"][f"sd_h={sd_h}, a1={truth}"] = rows
            for n, r in rows.items():
                print(f"natural  sd_h {sd_h:4.2f} a1 {truth:4.2f} n {n:3d}: bimodal {r['bimodal_fraction']:.2f} "
                      f"(both +-truth {r['both_truth_and_mirror_fraction']:.2f}) sd {r['median_posterior_sd']:.3f} "
                      f"P+ {r['median_p_positive']:.2f} sep {r['median_peak_separation']}", flush=True)
    for filter_ in ("x1sel", "rail"):
        rng = sim.generator(f"options-{filter_}-a0.25")
        curves = [posteriors(draw_rows(rng, 0.25, E.SD_H, filter_=filter_), E.SD_H) for _ in range(10)]
        examples[(filter_, 0.25)] = curves[0]
        rows = summarise(curves, 0.25)
        report["closer_modes_filtered"][filter_] = rows
        for n, r in rows.items():
            print(f"filtered {filter_:5s} a1 0.25 n {n:3d}: bimodal {r['bimodal_fraction']:.2f} "
                  f"(both +-0.25 {r['both_truth_and_mirror_fraction']:.2f}) sd {r['median_posterior_sd']:.3f} "
                  f"P+ {r['median_p_positive']:.2f} sep {r['median_peak_separation']}", flush=True)
    (OUT / "bimodality_options.json").write_text(json.dumps(report, indent=1))

    keys = [(s, 0.25) for s in SD_HS] + [("x1sel", 0.25), ("rail", 0.25)]
    fig, axes = plt.subplots(len(keys), len(NS), figsize=(2.3 * len(NS), 1.45 * len(keys)), constrained_layout=True,
                             sharex=True, squeeze=False)
    for r, key in enumerate(keys):
        for c, n in enumerate(NS):
            ax = axes[r, c]
            ax.fill_between(GRID, examples[key][n], color="#2a78d6", alpha=0.35, lw=0)
            ax.plot(GRID, examples[key][n], color="#2a78d6", lw=1)
            for v in (-key[1], key[1]):
                ax.axvline(v, color="#52514e", lw=0.7, ls=(0, (2, 2)))
            ax.set_yticks([])
            ax.set_xlim(-1.2, 1.2)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(f"n = {n}", fontsize=9)
        label = f"natural data\nsigma_h = {key[0]}" if isinstance(key[0], float) else \
            {"x1sel": "filtered |x1| < 0.15", "rail": "filtered |x1 - h| < 0.05"}[key[0]]
        axes[r, 0].set_ylabel(label, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("angle_1")
    fig.suptitle("Exact angle_1 posterior, one example dataset per row, true angle_1 = 0.25 (dotted: +-0.25)",
                 fontsize=10)
    fig.savefig(OUT / "bimodality_options.png", dpi=170)
    print("wrote", OUT / "bimodality_options.png")


if __name__ == "__main__":
    main()
