#!/usr/bin/env python
"""Tables and figures comparing the retrained GM arms against the rescue gate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "gm_arms"
# Part IX's development gate, not a claim of statistical validity.
GATE = {"w1_small_n": 0.10, "w1_large_n": 0.15, "mode_error_over_sd": 0.10,
        "weight_tv": 0.10, "width_low": 0.9, "width_high": 1.1}


def passes(row):
    limit = GATE["w1_small_n"] if row["n"] <= 32 else GATE["w1_large_n"]
    if row["w1_over_sd"] > limit or row["mode_recall"] < 1.0:
        return False
    if row["mode_error_over_sd"] is None:
        return False
    return (row["mode_error_over_sd"] <= GATE["mode_error_over_sd"]
            and row["weight_tv"] <= GATE["weight_tv"]
            and GATE["width_low"] <= row["width_ratio"] <= GATE["width_high"])


def table(arms, lam):
    lines = [f"### lambda = {lam:g}"
             + ("  *(below the trained floor -- extrapolation)*"
                if lam < 0.03167 else ""),
             "",
             "| n | " + " | ".join(
                 f"{name}: W1/sd (recall, TV, width)" for name in arms) + " |",
             "|---|" + "---|" * len(arms)]
    ladder = sorted({row["n"] for row in next(iter(arms.values()))})
    for n in ladder:
        cells = []
        for name, rows in arms.items():
            row = next(r for r in rows if r["n"] == n and r["lambda"] == lam)
            mark = " **PASS**" if passes(row) else ""
            cells.append(f"{row['w1_over_sd']:.3f} ({row['mode_recall']:.2f}, "
                         f"{row['weight_tv']:.2f}, {row['width_ratio']:.2f})"
                         + mark)
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    arms = json.load(open(OUT / "arm_rows.json"))
    order = [a for a in ("A", "Csim", "Csimbig", "B", "Bpre", "C", "Cbig",
                         "D", "Dbig") if a in arms]
    arms = {name: arms[name] for name in order}
    lambdas = sorted({row["lambda"] for row in next(iter(arms.values()))},
                     reverse=True)
    text = ["# Retrained GM arms, graded by direct score integration", "",
            "`W1/sd (mode recall, mode-weight TV, width ratio)` against the "
            "exact posterior. **PASS** marks a cell meeting every development "
            "gate in Part IX.", ""]
    for lam in lambdas:
        text.append(table(arms, lam))
    (OUT / "arms.md").write_text("\n".join(text))

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, rows in arms.items():
        for axis, lam in zip(axes, (0.1, 0.05, 0.03167)):
            selected = sorted([r for r in rows if r["lambda"] == lam],
                              key=lambda r: r["n"])
            axis.plot([r["n"] for r in selected],
                      [r["relative_network_error"] for r in selected], "o-",
                      label=name)
    for axis, lam in zip(axes, (0.1, 0.05, 0.03167)):
        axis.axhline(0.1, color="crimson", ls="--", lw=0.9)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("n")
        axis.set_title(f"lambda = {lam:g}", fontsize=9)
    axes[0].set_ylabel("relative composed field error")
    axes[0].legend(fontsize=7)
    figure.suptitle("Gaussian mixture: relative error of the composed "
                    "first-order field, by arm", fontsize=11)
    figure.tight_layout()
    figure.savefig(OUT / "arms.png", dpi=150)
    print(f"wrote {OUT / 'arms.md'} and {OUT / 'arms.png'}")


if __name__ == "__main__":
    main()
