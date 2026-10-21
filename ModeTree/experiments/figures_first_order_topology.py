#!/usr/bin/env python
"""Figures and tables for the first-order topology diagnostic.

Three panels per (problem, n): the reconstructed profile posterior against the
reference, the score field with its roots classified, and -- where an oracle
exists -- the learned-minus-oracle score difference.

Also emits the comparison table, folding in the already-computed
``gauss_jacobian`` and F-NPSE numbers rather than re-running them.

Run:  python ModeTree/experiments/figures_first_order_topology.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
OUT = ARTIFACTS / "first_order_topology"
STYLE = {"figure.dpi": 140, "font.size": 9, "axes.grid": True,
         "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False}
COLOUR = {"oracle": "#1f77b4", "learned": "#2ca02c"}
XLIM = {"ik": (-1.2, 1.2), "gm": (-3.5, 3.5)}


def load(problem):
    with open(OUT / f"{problem}_rows.json") as handle:
        rows = json.load(handle)
    with open(OUT / f"{problem}_curves.json") as handle:
        curves = json.load(handle)
    return rows, curves


def best_t(rows, kind, n):
    """The terminal time that minimizes W1 for this source -- reported, not hidden.

    The oracle improves monotonically as t falls; the learned score does not, so
    "the best t" is itself a measurement and differs between the two.
    """
    candidates = [r for r in rows if r["score_source"] == kind and r["n"] == n]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["w1_over_sigma"])


def figure(problem, rows, curves, n, times):
    sources = sorted({r["score_source"] for r in rows})
    have_oracle = "oracle" in sources
    panels = 3 if have_oracle else 2
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, panels, figsize=(4.6 * panels, 3.4),
                                 squeeze=False)
        reference_drawn = False
        for kind in sources:
            row = best_t(rows, kind, n)
            if row is None:
                continue
            key = f"{kind}|{n}|{row['t']:g}"
            curve = curves[key]
            grid = np.asarray(curve["grid"])
            if not reference_drawn:
                axes[0][0].plot(grid, curve["reference"], color="#22262B",
                                lw=2.0, zorder=5, label="reference")
                axes[0][1].plot(grid, curve["reference_score"], color="#22262B",
                                lw=1.6, zorder=5, label="reference score")
                for mode in row["reference"]["modes"]:
                    axes[0][0].axvline(mode, color="#22262B", ls=":", lw=0.8)
                reference_drawn = True
            label = f"{kind} (t={row['t']:g})"
            axes[0][0].plot(grid, curve["density"], color=COLOUR[kind], ls="--",
                            lw=1.5, label=f"{label} profile")
            axes[0][1].plot(grid, curve["score"], color=COLOUR[kind], lw=1.3,
                            label=label)
            for root in row["roots"]:
                axes[0][1].plot([root["root"]], [0.0],
                                marker="o" if root["type"] == "mode" else "x",
                                ms=5, color=COLOUR[kind],
                                mfc="none" if root["type"] == "valley" else None)
        axes[0][1].axhline(0.0, color="#888", lw=0.8)
        axes[0][0].set_title(f"{problem} n={n}: profile posterior", fontsize=9)
        axes[0][0].set_xlabel("global parameter")
        axes[0][0].set_ylabel("density")
        axes[0][0].legend(fontsize=6.5)
        axes[0][1].set_title("first-order score  (o = mode, x = valley)",
                             fontsize=9)
        axes[0][1].set_xlabel("global parameter")
        axes[0][1].set_ylabel("S(g)")
        axes[0][1].legend(fontsize=6.5)

        if have_oracle:
            oracle_row = best_t(rows, "oracle", n)
            learned_row = best_t(rows, "learned", n)
            if oracle_row and learned_row:
                # Compare at a common t so the difference is the score's, not
                # two different amounts of diffusion smoothing.
                shared = sorted({r["t"] for r in rows
                                 if r["n"] == n and r["score_source"] == "oracle"}
                                & {r["t"] for r in rows
                                   if r["n"] == n and r["score_source"] == "learned"})
                if shared:
                    t = min(shared)
                    o = np.asarray(curves[f"oracle|{n}|{t:g}"]["score"])
                    l = np.asarray(curves[f"learned|{n}|{t:g}"]["score"])
                    g = np.asarray(curves[f"oracle|{n}|{t:g}"]["grid"])
                    axes[0][2].plot(g, l - o, color="#B5179E", lw=1.3)
                    axes[0][2].axhline(0.0, color="#888", lw=0.8)
                    axes[0][2].set_title(f"learned - oracle score (t={t:g})",
                                         fontsize=9)
                    axes[0][2].set_xlabel("global parameter")
        for axis in axes[0]:
            axis.set_xlim(*XLIM.get(problem, (-3, 3)))
        fig.tight_layout()
        fig.savefig(OUT / f"{problem}_topology_n{n}.png")
        plt.close(fig)


def table(problem, rows):
    lines = ["| n | t | source | # modes | mode locations | valley | "
             "mode-loc error | dip | W1/sd |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n in sorted({r["n"] for r in rows}):
        reference = next(r for r in rows if r["n"] == n)["reference"]
        lines.append(
            f"| {n} | - | reference | {len(reference['modes'])} | "
            f"{', '.join(f'{m:+.3f}' for m in reference['modes'])} | "
            f"{reference.get('valley', float('nan')):+.3f} | - | "
            f"{reference['dip_ratio']:.3f} | - |")
        for row in sorted((r for r in rows if r["n"] == n),
                          key=lambda r: (r["score_source"], -r["t"])):
            modes = row["modes_found"]
            error = ""
            if len(modes) == len(reference["modes"]):
                error = f"{max(abs(a - b) for a, b in zip(sorted(modes), sorted(reference['modes']))):.3f}"
            lines.append(
                f"| {n} | {row['t']:g} | {row['score_source']} | {len(modes)} | "
                f"{', '.join(f'{m:+.3f}' for m in modes)} | "
                f"{', '.join(f'{v:+.3f}' for v in row['valleys_found'])} | "
                f"{error} | {row['reconstructed']['dip_ratio']:.3f} | "
                f"{row['w1_over_sigma']:.3f} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs="+", default=("ik", "gm"))
    args = parser.parse_args(argv)
    for problem in args.problems:
        try:
            rows, curves = load(problem)
        except FileNotFoundError:
            print(f"skipping {problem}: no artifacts yet")
            continue
        times = sorted({r["t"] for r in rows})
        for n in sorted({r["n"] for r in rows}):
            figure(problem, rows, curves, n, times)
            print(f"wrote {problem}_topology_n{n}.png")
        (OUT / f"{problem}_table.md").write_text(table(problem, rows) + "\n")
        print(f"wrote {problem}_table.md")


if __name__ == "__main__":
    main()
