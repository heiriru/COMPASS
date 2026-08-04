"""Visualize observed DDM data while varying one global parameter at a time."""

import argparse
from pathlib import Path
import sys


GLOBAL_LABELS = (
    r"$\mu_\nu$",
    r"$\mu_{\log \alpha}$",
    r"$\mu_{\log t_0}$",
    r"$\log \sigma_\nu$",
    r"$\log \sigma_{\log \alpha}$",
    r"$\log \sigma_{\log t_0}$",
    r"$\beta_{\mathrm{raw}}$",
)

GLOBAL_DESCRIPTIONS = (
    "population mean drift",
    "population mean threshold",
    "population mean non-decision time",
    "between-subject drift variability",
    "between-subject threshold variability",
    "between-subject non-decision-time variability",
    "starting-point bias",
)

INK = "#152238"
MUTED = "#667085"
PAPER = "#F7F8FA"
PANEL = "#FFFFFF"
GRID = "#DDE3EA"
COLORS = ("#2878B5", "#E8743B", "#775DA6")


def _parser():
    parser = argparse.ArgumentParser(
        description="Plot observational sensitivity to the seven global parameters."
    )
    parser.add_argument(
        "--artifact", type=Path, required=True,
        help="Recovery artifact supplying the baseline global parameter vector.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=48)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser


def _simulate_sweep(baseline, prior_std, subjects, trials, seed, np):
    from Partial_Pooling.simulators.ddm_sde import simulate_hierarchical

    offsets = np.asarray((-1.5, -0.75, 0.0, 0.75, 1.5))
    latent_rng = np.random.default_rng(seed)
    subject_latents = latent_rng.normal(size=(subjects, 3))
    results = []

    for parameter in range(7):
        values = baseline[parameter] + offsets * prior_std[parameter]
        outcomes = []
        for level, value in enumerate(values):
            globals_ = baseline.copy()
            globals_[parameter] = value
            locals_ = (
                globals_[None, :3]
                + np.exp(globals_[None, 3:6]) * subject_latents
            )
            simulation_rng = np.random.default_rng(
                seed + 10_000 * parameter + level
            )
            raw = simulate_hierarchical(
                globals_, locals_, simulation_rng, trials=trials,
            )
            outcomes.append({
                "choice": raw[..., 0].mean(axis=1),
                "rt": np.median(raw[..., 1], axis=1),
                "rt_tail": np.quantile(raw[..., 1], 0.90, axis=1),
            })
        results.append({"values": values, "outcomes": outcomes})
    return results


def _style_axis(axis):
    axis.set_facecolor(PANEL)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.grid(axis="y", color=GRID, lw=0.7, alpha=0.75)
    axis.set_axisbelow(True)


def _violin_panel(axis, sweep, outcome, color, baseline, np):
    values = sweep["values"]
    distributions = [entry[outcome] for entry in sweep["outcomes"]]
    spacing = float(np.min(np.diff(values)))
    violins = axis.violinplot(
        distributions, positions=values, widths=0.62 * spacing,
        showmeans=False, showmedians=False, showextrema=False,
        points=80, bw_method=0.35,
    )
    for body in violins["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_linewidth(0.7)
        body.set_alpha(0.32)

    medians = np.asarray([np.median(distribution) for distribution in distributions])
    lower = np.asarray([np.quantile(distribution, 0.10) for distribution in distributions])
    upper = np.asarray([np.quantile(distribution, 0.90) for distribution in distributions])
    axis.fill_between(values, lower, upper, color=color, alpha=0.08, linewidth=0)
    axis.plot(values, medians, color=color, lw=2.1, marker="o", ms=4.5, zorder=4)
    axis.scatter(
        values[:, None].repeat(len(distributions[0]), axis=1).ravel(),
        np.asarray(distributions).ravel(), s=3.5, color=color,
        alpha=0.10, edgecolors="none", zorder=2,
    )
    axis.axvline(baseline, color=INK, lw=1.0, ls=(0, (3, 3)), alpha=0.65)
    axis.set_xticks(values, [f"{value:.2g}" for value in values])
    axis.set_xlim(values[0] - 0.55 * spacing, values[-1] + 0.55 * spacing)
    _style_axis(axis)


def create_figure(sweeps, baseline, subjects, trials, output, plt, np):
    figure, axes = plt.subplots(
        7, 3, figsize=(15.5, 18.5), facecolor=PAPER,
        constrained_layout=False, sharey="col",
    )
    figure.subplots_adjust(
        left=0.19, right=0.985, top=0.90, bottom=0.045,
        hspace=0.72, wspace=0.30,
    )

    column_specs = (
        ("choice", COLORS[0], "Choice behaviour", r"subject $P(\mathrm{choice}=1)$"),
        ("rt", COLORS[1], "Reaction times", "subject median RT (s)"),
        ("rt_tail", COLORS[2], "Slow-response tail",
         "subject 90th-percentile RT (s)"),
    )
    for parameter, sweep in enumerate(sweeps):
        for column, (outcome, color, title, ylabel) in enumerate(column_specs):
            axis = axes[parameter, column]
            _violin_panel(
                axis, sweep, outcome, color, baseline[parameter], np,
            )
            if parameter == 0:
                axis.set_title(title, color=color, fontsize=14,
                               fontweight="bold", pad=13)
            axis.set_ylabel(ylabel, color=MUTED, fontsize=8.5)
            axis.set_xlabel(
                f"{GLOBAL_LABELS[parameter]} value", color=MUTED, fontsize=8.5,
            )
            if outcome == "choice":
                axis.set_ylim(-0.035, 1.035)

        axes[parameter, 0].text(
            -0.32, 0.52,
            f"{GLOBAL_LABELS[parameter]}\n{GLOBAL_DESCRIPTIONS[parameter]}",
            transform=axes[parameter, 0].transAxes, ha="right", va="center",
            color=INK, fontsize=10.5, fontweight="bold", linespacing=1.35,
        )

    figure.suptitle(
        "How global parameters shape the observed data",
        x=0.51, y=0.982, color=INK, fontsize=24, fontweight="bold",
    )
    figure.text(
        0.51, 0.957,
        "One global parameter varies per row; all others remain at the dataset-0 "
        "baseline. Each violin is the distribution across simulated subjects.",
        ha="center", va="top", color=MUTED, fontsize=11,
    )
    figure.text(
        0.51, 0.012,
        f"{subjects} subjects × {trials} trials per setting  •  dots: subjects  •  "
        "line: median  •  shading: 10th–90th percentile  •  dashed line: baseline",
        ha="center", va="bottom", color=MUTED, fontsize=9.5,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main(argv=None):
    args = _parser().parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from Partial_Pooling.simulators.priors import GLOBAL_PRIOR_STD

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    baseline = artifact["truth_globals"].numpy().astype(float)
    sweeps = _simulate_sweep(
        baseline, GLOBAL_PRIOR_STD, args.subjects, args.trials, args.seed, np,
    )
    create_figure(
        sweeps, baseline, args.subjects, args.trials, args.output, plt, np,
    )
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
