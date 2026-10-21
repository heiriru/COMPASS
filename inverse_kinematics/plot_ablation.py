#!/usr/bin/env python3
"""The mask-vs-capacity ablation, as a figure and a persisted table.

Four arms, identical data, sampler and evaluation; only the training mask
distribution and the data-dependent capacity differ.

    baseline   Bernoulli(0.33) masks, h16d2       posterior mask drawn 2.2%
    A menu     ConditionalMenu, h16d2             posterior mask drawn 69%
    B wide     Bernoulli(0.33), h64d4 mlp8        data-dependent 5,056 -> 334,208
    C both     ConditionalMenu, h64d4 mlp8

Numbers are produced by the evaluation described in README Part 4 and recorded
here rather than recomputed, so the figure cannot drift from the run that
produced it. Regenerate them with that section's commands if the models change.

Usage:
    python inverse_kinematics/plot_ablation.py
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

from plots import INK, RAIL, _style_axis  # noqa: E402

ROOT = Path(__file__).resolve().parent

# (label, parameters, posterior-mask share, data-dependent params, dip,
#  frac>0, sd(angle_1), endpoint error)
ARMS = [
    ("baseline\nh16d2", 367_136, 0.022, 5_056, 1.000, 0.478, 0.370, 0.0933),
    ("A: menu\nh16d2", 367_136, 0.69, 5_056, 0.486, 0.452, 0.787, 0.0780),
    ("B: wide\nh64d4", 3_163_008, 0.022, 334_208, 0.368, 0.522, 0.786, 0.0756),
    ("C: both\nh64d4+menu", 3_163_008, 0.69, 334_208, 0.283, 0.498, 0.784, 0.0747),
]
BAYESFLOW = {"label": "BayesFlow", "sd": 0.821, "endpoint": 0.0144,
             "parameters": 509_460}

MENU = "#E76F51"
PLAIN = "#264653"


def plot(output):
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    labels = [a[0] for a in ARMS]
    positions = np.arange(len(ARMS))
    colours = [MENU if a[2] > 0.1 else PLAIN for a in ARMS]

    # 1. Bimodality: lower dip = deeper valley between the two modes.
    axes[0].bar(positions, [a[4] for a in ARMS], color=colours, width=0.62)
    axes[0].axhline(0.5, color=RAIL, lw=1.0, ls=(0, (4, 3)))
    axes[0].annotate("bimodal below", xy=(2.6, 0.53), fontsize=8, color=INK)
    axes[0].set_ylabel("Dip ratio", color=INK)
    axes[0].set_title("Two modes, or one?", fontsize=10.5, color=INK)

    # 2. Posterior width: the tell that the baseline had merged the branches.
    axes[1].bar(positions, [a[6] for a in ARMS], color=colours, width=0.62)
    axes[1].axhline(BAYESFLOW["sd"], color=INK, lw=1.4, ls=(0, (4, 3)))
    axes[1].annotate("BayesFlow 0.821", xy=(-0.35, BAYESFLOW["sd"] + 0.02),
                     fontsize=8, color=INK)
    axes[1].set_ylabel(r"sd of $\mathrm{angle}_1$", color=INK)
    axes[1].set_ylim(0, 0.95)
    axes[1].set_title("Does it span both branches?", fontsize=10.5, color=INK)

    # 3. Accuracy, where the gap to BayesFlow survives every fix.
    axes[2].bar(positions, [a[7] for a in ARMS], color=colours, width=0.62)
    axes[2].axhline(BAYESFLOW["endpoint"], color=INK, lw=1.4, ls=(0, (4, 3)))
    axes[2].annotate("BayesFlow 0.0144", xy=(-0.35, BAYESFLOW["endpoint"] + 0.003),
                     fontsize=8, color=INK)
    axes[2].set_ylabel("Endpoint error", color=INK)
    axes[2].set_title("Accuracy: still 5x behind", fontsize=10.5, color=INK)

    for axis in axes:
        _style_axis(axis)
        axis.set_xticks(positions, labels, fontsize=8)
    figure.suptitle(
        "Why the score was not learned: signal share, not capacity\n"
        "orange = posterior mask trained 69% of the time · "
        "dark = 2.2% (Bernoulli 0.33)",
        fontsize=11, color=INK,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv=None):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    arguments = result.parse_args(argv)

    figures = arguments.artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = plot(figures / "16_mask_capacity_ablation.png")

    table = {
        "question": "why the network did not learn the bimodal score",
        "observation": [0.0, 1.5],
        "note": "n=1 through MultiObsSampler, where composition is a no-op, "
                "with churn 2 and no Langevin correctors",
        "bayesflow": BAYESFLOW,
        "arms": [
            {"label": a[0].replace("\n", " "), "parameters": a[1],
             "posterior_mask_share": a[2], "data_dependent_parameters": a[3],
             "dip_ratio": a[4], "fraction_positive": a[5],
             "sd_angle_1": a[6], "endpoint_error": a[7]}
            for a in ARMS
        ],
    }
    output = arguments.artifacts / "models" / "mask_capacity_ablation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2))
    print(f"Wrote {path}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
