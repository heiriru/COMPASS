"""Figures for the gauss_jacobian validation experiments.

Reads the CSVs written by experiments.py and writes PNG + PDF to artifacts/.
Each figure is a grid: one row per score-error level, one column per metric.
The score-error row is the one that matters -- with a *perfect* score there is
little for any correction to fix, and `uncorrected` is asymptotically exact.

The CSVs are the table view: every number plotted here is readable there, which
is what lets the third series carry its own direct label rather than relying on
colour contrast alone.
"""
import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import csv  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Categorical slots 1-3 of the validated default palette, in fixed order.
# Colour follows the method, never its rank, so a method keeps its hue across
# every panel, every row and both experiments.
COLOURS = {
    "uncorrected": "#2a78d6",
    "gauss_hierarchical": "#eb6834",
    "gauss_jacobian": "#1baf7a",
}
LABELS = {
    "uncorrected": "uncorrected",
    "gauss_hierarchical": "gauss_hierarchical (exact pilot)",
    "gauss_jacobian": "gauss_jacobian (no pilot)",
}
ORDER = ["uncorrected", "gauss_hierarchical", "gauss_jacobian"]

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"


def read(name):
    path = ARTIFACTS / f"{name}_metrics.csv"
    if not path.exists():
        return None
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        noise = float(row.get("score_noise") or 0.0)
        grouped[(noise, int(row["n"]), row["method"])].append(row)
    return grouped


def mean(values):
    values = [value for value in values if value == value]      # drop NaN
    return sum(values) / len(values) if values else float("nan")


def aggregate(grouped, noise, field):
    counts = sorted({key[1] for key in grouped if key[0] == noise})
    series = {
        method: [
            mean([float(row[field])
                  for row in grouped.get((noise, count, method), [])])
            for count in counts
        ]
        for method in ORDER
    }
    return counts, series


def format_value(value):
    """Counts read as integers; metrics keep three decimals."""
    if value != value:
        return ""
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.3f}"


def grouped_bars(axis, counts, series, ylabel, reference=None):
    width = 0.26
    positions = range(len(counts))
    for index, method in enumerate(ORDER):
        offsets = [position + (index - 1) * width for position in positions]
        bars = axis.bar(
            offsets, series[method], width * 0.9, label=LABELS[method],
            color=COLOURS[method], linewidth=0,
        )
        for rectangle, value in zip(bars, series[method]):
            if value != value:
                continue
            axis.annotate(
                format_value(value),
                (rectangle.get_x() + rectangle.get_width() / 2,
                 rectangle.get_height()),
                textcoords="offset points", xytext=(0, 3), ha="center",
                fontsize=7, color=MUTED,
            )
    if reference is not None:
        axis.axhline(reference, color=MUTED, linewidth=1, linestyle=(0, (4, 3)),
                     zorder=0)
        axis.annotate("exact", xy=(1.012, reference),
                      xycoords=("axes fraction", "data"), fontsize=7,
                      color=MUTED, va="center", ha="left",
                      annotation_clip=False)
    axis.margins(y=0.18)
    axis.set_xticks(list(positions))
    axis.set_xticklabels([f"n = {count}" for count in counts])
    axis.set_ylabel(ylabel, color=INK, fontsize=8.5)
    axis.tick_params(colors=MUTED, labelsize=8, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)


def figure(name, grouped, title, subtitle, panels):
    noises = sorted({key[0] for key in grouped})
    rows, columns = len(noises), len(panels)
    fig, axes = plt.subplots(rows, columns, figsize=(4.7 * columns, 3.9 * rows),
                             squeeze=False)

    for row_index, noise in enumerate(noises):
        for column_index, (field, ylabel, reference) in enumerate(panels):
            counts, series = aggregate(grouped, noise, field)
            grouped_bars(axes[row_index][column_index], counts, series, ylabel,
                         reference)
        tag = ("perfect score  (eps = 0)" if noise == 0
               else f"imperfect score  (eps = {noise:g})")
        axes[row_index][0].annotate(
            tag, xy=(-0.20, 0.5), xycoords="axes fraction", rotation=90,
            ha="center", va="center", fontsize=9.5, color=INK,
        )

    top = 1 - 0.24 / rows
    fig.subplots_adjust(left=0.075, right=0.962, top=top,
                        bottom=0.115 / rows + 0.02, hspace=0.34, wspace=0.28)
    fig.text(0.012, 0.985, title, fontsize=13, color=INK, ha="left", va="top")
    fig.text(0.012, 0.985 - 0.052 / rows * 1.6, subtitle, fontsize=8.5,
             color=MUTED, ha="left", va="top", linespacing=1.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=MUTED, bbox_to_anchor=(0.5, 0.004))
    for suffix in ("png", "pdf"):
        fig.savefig(ARTIFACTS / f"{name}.{suffix}", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {ARTIFACTS / (name + '.png')}")


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    produced = False

    gaussian = read("gaussian")
    if gaussian:
        figure(
            "01_gaussian_reduction", gaussian,
            "Exactly Gaussian posteriors: the null test",
            "With a perfect score gauss_jacobian must TIE gauss_hierarchical - "
            "the Tweedie construction reduces to the pilot form algebraically.\n"
            "With an imperfect score the pilot wins slightly: a constant "
            "covariance is the right answer here, and averaging it beats "
            "differentiating a noisy network.",
            [
                ("mean_error_sigma", "shared-parameter mean error  (/ sigma)", None),
                ("width_ratio", "posterior width / exact width", 1.0),
                ("network_calls", "score-network calls", None),
            ],
        )
        produced = True

    nongaussian = read("nongaussian")
    if nongaussian:
        figure(
            "02_nongaussian_advantage", nongaussian,
            "Non-Gaussian posteriors: the discriminating test",
            "Per-observation mixtures with opposite-sign g/l correlation - no "
            "single covariance describes them; gauss_hierarchical still gets "
            "the exact moment-matched one.\n"
            "Read the bottom row: with a perfect score there is little to "
            "correct and the unweighted sum is asymptotically exact, so only "
            "the imperfect-score row reflects real use.",
            [
                ("wasserstein", "W1 to exact shared posterior", None),
                ("mean_error_sigma", "shared-parameter mean error  (/ sigma)", None),
                ("width_ratio", "posterior width / exact width", 1.0),
            ],
        )
        produced = True

    if not produced:
        raise SystemExit("No metrics CSVs found; run experiments.py first.")


if __name__ == "__main__":
    main()
