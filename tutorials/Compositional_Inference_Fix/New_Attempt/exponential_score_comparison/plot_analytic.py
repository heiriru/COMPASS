#!/usr/bin/env python3
"""The rule-versus-sampler picture, from artifacts already on disk.

``analytic_compare.py`` establishes that on this problem the composition rule and
the sampler are separable, and that the usual reading of the headline figures
conflates them:

* With **exact** single-observation scores, ``gauss_jacobian``'s composed score is
  the most accurate of the three rules at every noise level -- by 7x to 50x over
  F-NPSE (panel 1, from ``oracle_diagnostics/analytic_composition.csv``).
* Yet under predictor-only DPM2 its draws are the *worst* of the three, because a
  deterministic probability-flow map returns the pushforward of its initial law
  and has no mechanism to relax onto the field it is given (panels 2 and 3).
* Give the same field a sampler that does relax -- annealed Langevin, exactly
  what F-NPSE uses -- and the ordering inverts to match the score panel.

So hue is the **composition rule** and line style is the **sampler**: those are
the two dimensions the experiment varies, and the figure is legible only if they
are encoded separately. The rule hues are the ones compare.py already uses, so
these panels sit alongside ``01_*``/``02_*`` without a second vocabulary.

This script only reads ``artifacts/*.npz``; it runs no sampler and touches no
GPU, so it is safe to run while ``analytic_compare.py`` is still working. Arms
whose archive is absent are skipped, so it is also safe to run early.

Usage:
    python plot_analytic.py
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

# compare.py's palette, minus BLUE: under the Vienot dichromat simulation
# TEAL/BLUE separate by only 3.7 (OKLab dE x100) for tritanopia, far under the 8
# floor, and the two Jacobian arms would have been exactly that pair. Encoding
# the sampler as line style instead keeps every hue pair above the floor
# (worst: TEAL/CORAL at 7.9 under deuteranopia, which is why every series is
# also direct-labelled and marker-coded).
NAVY, TEAL, CORAL, GOLD = "#17223B", "#2A9D8F", "#EF476F", "#FFB703"
MUTED = "#8A8F98"

RULES = {
    "gauss_jacobian": {"colour": TEAL, "label": "gauss_jacobian"},
    "gauss_hierarchical": {"colour": CORAL, "label": "gauss_hierarchical"},
    "fnpe": {"colour": GOLD, "label": "F-NPSE"},
}

SAMPLERS = {
    "predictor": {"style": ":", "marker": "o", "label": "DPM2, predictor only"},
    "corrected2": {"style": "-.", "marker": "D", "label": "DPM2 + 2 correctors"},
    "corrected": {"style": "-", "marker": "s", "label": "DPM2 + 10 correctors"},
    "langevin": {"style": "--", "marker": "^", "label": "annealed Langevin"},
}

# (archive stem, rule, sampler, whether the row score was exact)
ARMS = [
    ("gauss_jacobian_analytic_predictor_only", "gauss_jacobian", "predictor", True),
    ("gauss_hierarchical_analytic_predictor_only", "gauss_hierarchical",
     "predictor", True),
    ("langevin_fnpe_analytic", "fnpe", "langevin", True),
    ("gauss_jacobian_analytic_langevin", "gauss_jacobian", "langevin", True),
    ("gauss_jacobian_analytic", "gauss_jacobian", "corrected", True),
    ("gauss_jacobian_analytic_correctors2", "gauss_jacobian", "corrected2", True),
    ("gauss_hierarchical_analytic", "gauss_hierarchical", "corrected", True),
    ("gauss_jacobian_predictor_only", "gauss_jacobian", "predictor", False),
    ("gauss_hierarchical_predictor_only", "gauss_hierarchical", "predictor", False),
    ("langevin_fnpe", "fnpe", "langevin", False),
    ("gauss_jacobian", "gauss_jacobian", "corrected", False),
    ("gauss_jacobian_correctors2", "gauss_jacobian", "corrected2", False),
    ("gauss_hierarchical", "gauss_hierarchical", "corrected", False),
]

# analytic_composition.csv's method names, in this figure's vocabulary.
COMPOSITION_ROWS = {
    "analytic_gauss_jacobian": "gauss_jacobian",
    "analytic_gauss_hierarchical": "gauss_hierarchical",
    "analytic_fnpe": "fnpe",
}


def configure_style():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 240, "font.size": 10,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def load_arms(directory):
    """Every arm whose archive exists, keyed by stem."""
    loaded = {}
    for stem, rule, sampler, exact in ARMS:
        path = directory / f"{stem}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as archive:
            stored = {key: archive[key] for key in archive.files}
        loaded[stem] = {
            "rule": rule, "sampler": sampler, "exact": exact,
            "metrics": json.loads(str(stored["metrics_json"])),
            "shared": np.asarray(stored["kept_shared"]).reshape(-1),
            "score_grid": np.asarray(stored["score_grid"]),
            "implied_score": np.asarray(stored["implied_score"]),
            "exact_score": np.asarray(stored["exact_score"]),
            "global_grid": np.asarray(stored["global_grid"]),
            "global_density": np.asarray(stored["global_density"]),
            "global_truth": float(stored["global_truth"]),
            "x": np.asarray(stored["x"]).reshape(-1),
        }
    return loaded


def load_composition(path):
    """Composed-score error per rule and noise level, exact row scores."""
    if not path.exists():
        return {}
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "overall" or row["method"] not in COMPOSITION_ROWS:
                continue
            rule = COMPOSITION_ROWS[row["method"]]
            rows.setdefault(rule, []).append(
                (float(row["lambda"]), float(row["shared_rel_rmse"]))
            )
    return {rule: sorted(values) for rule, values in rows.items()}


def label_for(entry):
    return f"{RULES[entry['rule']]['label']}, {SAMPLERS[entry['sampler']]['label']}"


def plot(arms, composition, output):
    exact_arms = {stem: entry for stem, entry in arms.items() if entry["exact"]}
    if not exact_arms:
        raise SystemExit("No analytic archives found; run analytic_compare.py first.")
    reference = next(iter(exact_arms.values()))
    grid, density = reference["global_grid"], reference["global_density"]
    mean = float((grid * density).sum() / density.sum())
    std = float(np.sqrt(
        (density * (grid - mean) ** 2).sum() / density.sum()
    ))
    wall = float(reference["x"].min())

    fig, axes = plt.subplots(1, 4, figsize=(18.6, 5.0))
    fig.suptitle(
        "Composition rule and sampler are separable: the most accurate composed "
        "score is not the best posterior"
        "\n(exponential hierarchy g → ℓⱼ → xⱼ, exact single-observation score "
        "throughout; hue = rule, line style = sampler)",
        fontsize=13.5, fontweight="bold", color=NAVY,
    )

    # Panel 1 -- the field each rule produces, before any sampler runs. Exact
    # row scores, exact quadrature target, so this is the rule and nothing else.
    for rule, values in composition.items():
        lambdas = [value[0] for value in values]
        errors = [value[1] for value in values]
        axes[0].plot(lambdas, errors, "o-", lw=2, ms=5,
                     color=RULES[rule]["colour"], label=RULES[rule]["label"])
        axes[0].annotate(RULES[rule]["label"], (lambdas[-1], errors[-1]),
                         textcoords="offset points", xytext=(6, 0), fontsize=8.5,
                         color=NAVY, va="center")
    axes[0].axhline(1.0, color=MUTED, ls=":", lw=1.2)
    axes[0].set(xscale="log", yscale="log", xlabel="noise level λ",
                ylabel="relative RMS error of the composed shared score",
                title="1. The rule alone")
    # Headroom on the right for the direct labels, which sit past the last rung.
    axes[0].set_xlim(0.04, 14.0)
    axes[0].legend(fontsize=8.5, loc="lower left")
    axes[0].text(0.5, -0.26,
                 "exact row scores, exact quadrature target;\n"
                 "gauss_jacobian is the most accurate field at every λ",
                 transform=axes[0].transAxes, ha="center", va="top",
                 color=NAVY, fontsize=8)

    # Panel 2 -- what each sampler makes of that field.
    axes[1].plot(grid, density, color=NAVY, ls="-", lw=2.2, label="exact",
                 zorder=5)
    for stem, entry in exact_arms.items():
        axes[1].hist(entry["shared"], bins=70, density=True, histtype="step",
                     lw=1.8, color=RULES[entry["rule"]]["colour"],
                     linestyle=SAMPLERS[entry["sampler"]]["style"],
                     label=label_for(entry))
    axes[1].axvline(wall, color=NAVY, ls="-.", lw=1.4, label="wall at min$_j$ x$_j$")
    axes[1].set_xlim(mean - 6.5 * std, mean + 3.0 * std)
    axes[1].set(xlabel="global parameter g", ylabel="density",
                title="2. The sampled posterior")
    axes[1].legend(fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    # Panel 3 -- the same draws read as a score, against the closed form.
    axes[2].plot(reference["score_grid"], reference["exact_score"], color=NAVY,
                 lw=2.2, label="exact d/dg log p(g|x)", zorder=5)
    for stem, entry in exact_arms.items():
        axes[2].plot(entry["score_grid"], entry["implied_score"], lw=1.8,
                     color=RULES[entry["rule"]]["colour"],
                     linestyle=SAMPLERS[entry["sampler"]]["style"],
                     label=f"{label_for(entry)} "
                           f"({entry['metrics']['marginal_score_rel_rmse']:.2f})")
    axes[2].set(xlabel="global parameter g", ylabel="d/dg log p(g | x$_{1:N}$)",
                title="3. Marginal score implied by the draws")
    axes[2].legend(fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    # Panel 4 -- learned against exact row score, per configuration. The point
    # is the *absence* of an improvement: removing the network changes little,
    # and never in the direction that would explain the predictor-only failure.
    configurations, values_exact, values_learned, colours = [], [], [], []
    seen = []
    for stem, entry in arms.items():
        key = (entry["rule"], entry["sampler"])
        if key in seen:
            continue
        seen.append(key)
        pair = {
            other["exact"]: other["metrics"]["shared_w1_over_sigma"]
            for other in arms.values()
            if (other["rule"], other["sampler"]) == key
        }
        if True not in pair:
            continue
        configurations.append(
            f"{RULES[key[0]]['label']}\n{SAMPLERS[key[1]]['label']}"
        )
        values_exact.append(pair[True])
        values_learned.append(pair.get(False, np.nan))
        colours.append(RULES[key[0]]["colour"])

    # Horizontal bars: the configuration names are long, and rotated tick labels
    # under four narrow panels collide.
    order = list(np.argsort(values_exact)[::-1])
    positions = np.arange(len(order))
    height = 0.38
    floor = 0.04
    for offset, (values, hatch, name) in enumerate([
        (values_learned, "///", "learned score"),
        (values_exact, None, "exact row score"),
    ]):
        widths = [values[index] for index in order]
        axes[3].barh(
            positions + (0.5 - offset) * height,
            [floor if not np.isfinite(value) else value for value in widths],
            height=height * 0.92, left=floor,
            color=[colours[index] for index in order],
            alpha=0.5 if offset == 0 else 1.0, hatch=hatch,
            edgecolor="white", linewidth=1.2, label=name,
        )
        for row, value in zip(positions, widths):
            if not np.isfinite(value):
                axes[3].text(floor * 1.6, row + (0.5 - offset) * height,
                             "no learned counterpart", va="center", fontsize=6.8,
                             color=MUTED)
    axes[3].set_yticks(positions)
    axes[3].set_yticklabels(
        [configurations[index].replace("\n", ", ") for index in order], fontsize=7.6
    )
    axes[3].axvline(1.0, color=NAVY, ls=":", lw=1.2)
    axes[3].set_xscale("log")
    axes[3].set_xlim(floor, 12.0)
    axes[3].set(xlabel="W1 of the shared marginal / posterior σ",
                title="4. Was it the network?")
    axes[3].grid(axis="y", visible=False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, alpha=0.5, hatch="///",
                      edgecolor="white", label="learned score"),
        plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor="white",
                      label="exact row score"),
    ]
    axes[3].legend(handles=handles, fontsize=8, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, -0.20))
    axes[3].text(0.5, -0.34,
                 "while transport bias dominates the network is irrelevant;\n"
                 "once a sampler relaxes onto the field, it is worth 5-11x",
                 transform=axes[3].transAxes, ha="center", va="top",
                 color=NAVY, fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    configure_style()
    arms = load_arms(arguments.artifacts)
    print("Arms found: " + ", ".join(sorted(arms)))
    # Panel 1 grades the composition rule at fixed noise levels, with no sampler
    # in the loop, so it is the same table for any timestep count. A run
    # directory that has no diagnostics of its own falls back to the default
    # artifacts tree rather than dropping the panel.
    composition_path = (
        arguments.artifacts / "oracle_diagnostics" / "analytic_composition.csv"
    )
    if not composition_path.exists():
        composition_path = ARTIFACTS / "oracle_diagnostics" / "analytic_composition.csv"
    composition = load_composition(composition_path)
    output = arguments.output or arguments.artifacts / "03_rule_versus_sampler.png"
    plot(arms, composition, output)


if __name__ == "__main__":
    main()
