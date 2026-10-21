#!/usr/bin/env python3
"""Figures for the observation sweep produced by ``run_scaling.py``.

``posterior_vs_analytic``
    The headline: every rule's composed shared posterior drawn against the
    **exact** ``p(g | x_1..n)``, one panel per (regime, n). The reference is the
    only filled curve and the only one in ink, so it never competes with a
    method for identity.

``w1_scaling``
    Accuracy against ``n``: W1 to the exact posterior in units of that
    posterior's own sd, median over replicates with the interquartile range as a
    band. The grey band is the ``exact_mc`` arm -- draws from the true posterior
    graded by the same code -- and is the floor of the measurement. A method
    inside it is indistinguishable from perfect at this sample size; a
    difference below it is not a result.

``branch_mass``
    The failure mode this benchmark was built to expose: the posterior mass each
    rule puts on the positive branch, against the exact weight (dashed). A rule
    that summarises each observation by a mean and a covariance cannot carry
    unequal branch weights, and this is where that shows.

``calibration``
    Width ratio, mean error, and the accuracy of the *local* parameters, all
    against ``n``.

Usage:
    python plot_scaling.py --tag main main_n32
    python plot_scaling.py --tag diagnostic diagnostic_n32
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit()

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import datasets  # noqa: E402
import methods  # noqa: E402
from model import DEFAULT  # noqa: E402
from plot_data import FAINT, INK, RAIL, style  # noqa: E402

ROOT = Path(__file__).resolve().parent
FLOOR_COLOUR = "#B8BEC6"


def load(directory):
    with (directory / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["observations"] = int(row["observations"])
        row["replicate"] = int(row["replicate"])
        for key, value in list(row.items()):
            if key in ("regime", "method", "failed") or value == "":
                continue
            if key not in ("observations", "replicate"):
                row[key] = float(value)
    datasets_used = json.loads((directory / "datasets.json").read_text())
    draws = dict(np.load(directory / "draws.npz"))
    return rows, datasets_used, draws


def load_all(artifacts, tags):
    """Load one or more sweeps, later tags replacing earlier ones cell by cell.

    A cell is a ``(regime, n)`` pair, and a later tag replaces *every* method in
    it, never a single arm: a rerun exists because something in that cell had to
    be redone (at ``n = 32`` the Jacobian rule exhausts an 11 GB card at 4,000
    draws and the whole cell was rerun at 2,000), and mixing draw counts inside
    one cell would put the arms on different Monte-Carlo floors.
    """
    rows, records, draws = [], [], {}
    for tag in tags:
        directory = Path(artifacts) / "scaling" / tag
        if not (directory / "metrics.csv").exists():
            raise SystemExit(f"No sweep at {directory}.")
        new_rows, new_records, new_draws = load(directory)
        replaced = {(row["regime"], row["observations"]) for row in new_rows}
        rows = [row for row in rows
                if (row["regime"], row["observations"]) not in replaced]
        records = [record for record in records
                   if (record["regime"], record["observations"]) not in replaced]
        draws = {key: value for key, value in draws.items()
                 if not any(key.startswith(f"{regime}_n{n}_")
                            for regime, n in replaced)}
        rows += new_rows
        records += new_records
        draws.update(new_draws)
    return rows, records, draws


def present(rows, key):
    """Ordered, de-duplicated values of ``key`` -- the sweep's own axes."""
    seen = []
    for row in rows:
        if row[key] not in seen:
            seen.append(row[key])
    return seen


def series(rows, regime, method, key, observations):
    """Median and IQR of ``key`` over replicates, per ``n``."""
    median, low, high = [], [], []
    for n in observations:
        values = [row[key] for row in rows
                  if row["regime"] == regime and row["method"] == method
                  and row["observations"] == n and not row.get("failed")
                  and row.get(key, "") != ""]
        values = np.asarray([v for v in values if np.isfinite(v)],
                            dtype=np.float64)
        if not len(values):
            median.append(np.nan), low.append(np.nan), high.append(np.nan)
            continue
        median.append(float(np.median(values)))
        low.append(float(np.percentile(values, 25)))
        high.append(float(np.percentile(values, 75)))
    return np.array(median), np.array(low), np.array(high)


def method_order(rows):
    names = [name for name in methods.ALL
             if any(row["method"] == name for row in rows)]
    return names


def kde(values, grid, bandwidth):
    """A KDE at a *prescribed absolute* bandwidth, shared by every arm.

    Scott's rule is a function of the sample's own spread, so a rule that
    returns one wide blob would be drawn with a wide kernel and a rule that
    returns two sharp branches with a narrow one -- the smoothing would then
    differ between the very curves being compared, and the exact-posterior
    sample would be drawn oversmoothed enough to look bimodal-free. Fixing the
    bandwidth to a fraction of the exact within-branch sd makes every curve on a
    panel, the sampled reference included, carry the same smoothing bias.
    """
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=np.float64)
    spread = float(values.std())
    if not np.isfinite(spread) or spread <= 0.0:
        return np.zeros_like(grid)
    density = gaussian_kde(values, bw_method=bandwidth / spread)(grid)
    area = np.trapezoid(density, grid)
    return density / area if area > 0 else density


# ---------------------------------------------------------------------------
# The headline figure
# ---------------------------------------------------------------------------

def posterior_figure(output, rows, dataset_records, draws, mixture=DEFAULT,
                     regimes=None, observations=None, title=None):
    lookup = {(record["regime"], record["observations"]): np.asarray(record["x"])
              for record in dataset_records}
    regimes = regimes or present(rows, "regime")
    observations = observations or sorted(present(rows, "observations"))
    names = method_order(rows)

    figure, axes = plt.subplots(
        len(regimes), len(observations), squeeze=False,
        # A single-regime figure gets taller panels: at 2.35 inches over six
        # columns the densities are drawn in a strip too flat to read.
        figsize=(2.9 * len(observations),
                 (2.35 if len(regimes) > 1 else 3.4) * len(regimes)),
        layout="constrained",
    )
    for row, regime in enumerate(regimes):
        for column, n in enumerate(observations):
            axis = axes[row][column]
            x = lookup.get((regime, n))
            if x is None:
                axis.set_axis_off()
                continue
            grid = np.linspace(-4.2, 4.2, 1201)
            exact = mixture.global_density(grid, x)
            bandwidth = 0.35 * float(np.sqrt(mixture.posterior_variance(n)))
            axis.fill_between(grid, exact, color=INK, alpha=0.10, linewidth=0)
            axis.plot(grid, exact, color=INK, linewidth=2.0, zorder=5,
                      label="exact $p(g \\mid x_{1:n})$")
            for name in names:
                key = f"{regime}_n{n}_{name}"
                if key not in draws:
                    continue
                axis.plot(grid, kde(draws[key], grid, bandwidth),
                          color=methods.METHODS[name]["colour"], linewidth=1.5,
                          alpha=0.95, label=methods.METHODS[name]["label"])
            if f"{regime}_n{n}_exact_mc" in draws:
                axis.plot(grid,
                          kde(draws[f"{regime}_n{n}_exact_mc"], grid, bandwidth),
                          color=FLOOR_COLOUR, linewidth=1.3, linestyle=(0, (4, 2)),
                          zorder=6, label="exact posterior, sampled")
            if row == 0:
                axis.set_title(f"$n = {n}$", fontsize=11, color=INK)
            if column == 0:
                axis.set_ylabel(regime, fontsize=10, color=INK)
            if row == len(regimes) - 1:
                axis.set_xlabel("$g$", fontsize=10, color=INK)
            axis.set_yticks([])
            style(axis)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False,
                  ncols=min(len(labels), 3), fontsize=9.5, labelcolor=INK)
    figure.suptitle(
        title or "Composed shared posterior vs the exact one",
        fontsize=13, color=INK,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(f"{output}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


# ---------------------------------------------------------------------------
# Scaling curves
# ---------------------------------------------------------------------------

def _curve_panel(axis, rows, regime, key, observations, names, log=True):
    floor_median, floor_low, floor_high = series(
        rows, regime, "exact_mc", key, observations)
    if np.isfinite(floor_median).any():
        axis.fill_between(observations, floor_low, floor_high,
                          color=FLOOR_COLOUR, alpha=0.45, linewidth=0,
                          zorder=1, label="Monte-Carlo floor (exact posterior)")
        axis.plot(observations, floor_median, color=FLOOR_COLOUR, linewidth=1.2,
                  linestyle=(0, (4, 2)), zorder=2)
    for name in names:
        median, low, high = series(rows, regime, name, key, observations)
        colour = methods.METHODS[name]["colour"]
        axis.fill_between(observations, low, high, color=colour, alpha=0.14,
                          linewidth=0)
        axis.plot(observations, median, color=colour, linewidth=1.9,
                  marker="o", markersize=4.5,
                  label=methods.METHODS[name]["label"])
    axis.set_xscale("log", base=2)
    axis.set_xticks(observations)
    axis.set_xticklabels([str(n) for n in observations])
    # A cell whose every arm failed leaves nothing positive to log-scale, and
    # matplotlib raises rather than falling back.
    plotted = np.concatenate([
        series(rows, regime, name, key, observations)[0]
        for name in list(names) + ["exact_mc"]
    ])
    if log and np.isfinite(plotted).any() and np.nanmax(plotted) > 0:
        axis.set_yscale("log")
    style(axis, grid=True)


ABLATION_COLOUR = "#7A5C3E"


def ablation_curve(artifacts):
    """``prior_ablation.json`` if it exists: perfect network, Gaussian prior."""
    path = Path(artifacts) / "scaling" / "prior_ablation.json"
    if not path.exists():
        return {}
    return {(entry["regime"], entry["observations"]): entry["gaussian"]
            for entry in json.loads(path.read_text())}


def scaling_figure(output, rows, key, ylabel, title, log=True, regimes=None,
                   observations=None, ablation=None):
    regimes = regimes or present(rows, "regime")
    observations = observations or sorted(present(rows, "observations"))
    names = method_order(rows)
    figure, axes = plt.subplots(
        1, len(regimes), squeeze=False, sharey=True,
        figsize=(3.4 * len(regimes), 3.6), layout="constrained",
    )
    for column, regime in enumerate(regimes):
        axis = axes[0][column]
        _curve_panel(axis, rows, regime, key, observations, names, log=log)
        if ablation:
            # The same composition with EXACT per-observation posteriors and the
            # moment-matched Gaussian prior: what the rule would return with a
            # perfect network. Where it lands on top of the measured curves, the
            # prior term is the whole story and the network is not the problem.
            # At n = 1 there is no ``(n-1)`` term, so this curve is exactly
            # zero there by construction and carries no information; plotting it
            # would stretch a log axis over eight empty decades.
            values = [ablation.get((regime, n), np.nan) for n in observations]
            values = [v if (v is not None and v > 1e-6) else np.nan
                      for v in values]
            axis.plot(observations, values, color=ABLATION_COLOUR,
                      linewidth=1.8, linestyle=(0, (1, 1.6)), zorder=6,
                      marker="s", markersize=3.5,
                      label="exact scores + Gaussian prior")
        axis.set_title(regime, fontsize=11, color=INK)
        axis.set_xlabel("observations $n$", fontsize=9.5, color=INK)
    axes[0][0].set_ylabel(ylabel, fontsize=9.5, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False,
                  ncols=min(len(labels), 4), fontsize=9.5, labelcolor=INK)
    figure.suptitle(title, fontsize=13, color=INK)
    for suffix in ("png", "pdf"):
        figure.savefig(f"{output}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def source_comparison_figure(output, rows, oracle_rows, regimes=None,
                            observations=None, ablation=None):
    """Trained network (solid) against the exact score (dashed), per rule.

    The two together answer the question the sweep alone cannot: how much of the
    composed error is the network's and how much is the rule's. Where a dashed
    curve sits on its solid one, the rule fails identically with a perfect score
    and the network is not the cause; where they separate, that gap is the
    network's contribution, amplified by the composition.
    """
    regimes = regimes or present(rows, "regime")
    observations = observations or sorted(present(rows, "observations"))
    names = method_order(rows)
    figure, axes = plt.subplots(
        1, len(regimes), squeeze=False, sharey=True,
        figsize=(3.4 * len(regimes), 3.8), layout="constrained",
    )
    for column, regime in enumerate(regimes):
        axis = axes[0][column]
        floor, low, high = series(rows, regime, "exact_mc", "w1_over_sigma",
                                  observations)
        axis.fill_between(observations, low, high, color=FLOOR_COLOUR,
                          alpha=0.45, linewidth=0,
                          label="Monte-Carlo floor (exact posterior)")
        for name in names:
            colour = methods.METHODS[name]["colour"]
            median, _, _ = series(rows, regime, name, "w1_over_sigma",
                                  observations)
            axis.plot(observations, median, color=colour, linewidth=1.9,
                      marker="o", markersize=4.5,
                      label=f"{methods.METHODS[name]['label']} (network)")
            median, _, _ = series(oracle_rows, regime, name, "w1_over_sigma",
                                  observations)
            axis.plot(observations, median, color=colour, linewidth=1.6,
                      linestyle=(0, (5, 2)), marker="^", markersize=4.5,
                      markerfacecolor="white",
                      label=f"{methods.METHODS[name]['label']} (exact score)")
        if ablation:
            values = [ablation.get((regime, n), np.nan) for n in observations]
            values = [v if (v is not None and v > 1e-6) else np.nan
                      for v in values]
            axis.plot(observations, values, color=ABLATION_COLOUR,
                      linewidth=1.8, linestyle=(0, (1, 1.6)), zorder=6,
                      marker="s", markersize=3.5,
                      label="analytic: exact scores + Gaussian prior")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(observations)
        axis.set_xticklabels([str(n) for n in observations])
        axis.set_title(regime, fontsize=11, color=INK)
        axis.set_xlabel("observations $n$", fontsize=9.5, color=INK)
        style(axis, grid=True)
    axes[0][0].set_ylabel("$W_1$ / exact sd", fontsize=9.5, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False,
                  ncols=2, fontsize=9, labelcolor=INK)
    figure.suptitle("Is it the network or the rule?", fontsize=13, color=INK)
    for suffix in ("png", "pdf"):
        figure.savefig(f"{output}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def branch_mass_figure(output, rows, regimes=None, observations=None):
    """Absolute branch mass, with the exact weight drawn as the target.

    Only regimes whose exact weight is the *same in every replicate* are drawn.
    Under ``natural`` data the target weight is a property of the replicate's own
    ``g`` -- it is 0 for one draw and 1 for the next -- so a median over
    replicates would be an average of two different targets and would read as
    wild oscillation rather than as any property of a method.
    """
    regimes = regimes or present(rows, "regime")
    observations = observations or sorted(present(rows, "observations"))
    regimes = [regime for regime in regimes
               if _target_is_constant(rows, regime, observations)]
    if not regimes:
        return None
    names = method_order(rows)
    figure, axes = plt.subplots(
        1, len(regimes), squeeze=False, sharey=True,
        figsize=(3.4 * len(regimes), 3.6), layout="constrained",
    )
    for column, regime in enumerate(regimes):
        axis = axes[0][column]
        exact, _, _ = series(rows, regime, "exact_mc",
                             "reference_branch_mass", observations)
        axis.plot(observations, exact, color=INK, linewidth=1.8,
                  linestyle=(0, (4, 2)), zorder=5, label="exact $w_+$")
        floor, low, high = series(rows, regime, "exact_mc", "branch_mass",
                                  observations)
        axis.fill_between(observations, low, high, color=FLOOR_COLOUR,
                          alpha=0.45, linewidth=0,
                          label="Monte-Carlo floor (exact posterior)")
        for name in names:
            median, low, high = series(rows, regime, name, "branch_mass",
                                       observations)
            colour = methods.METHODS[name]["colour"]
            axis.fill_between(observations, low, high, color=colour, alpha=0.14,
                              linewidth=0)
            axis.plot(observations, median, color=colour, linewidth=1.9,
                      marker="o", markersize=4.5,
                      label=methods.METHODS[name]["label"])
        axis.set_xscale("log", base=2)
        axis.set_xticks(observations)
        axis.set_xticklabels([str(n) for n in observations])
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(regime, fontsize=11, color=INK)
        axis.set_xlabel("observations $n$", fontsize=9.5, color=INK)
        style(axis, grid=True)
    axes[0][0].set_ylabel("posterior mass on the $+$ branch", fontsize=9.5,
                          color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False,
                  ncols=min(len(labels), 3), fontsize=9.5, labelcolor=INK)
    figure.suptitle("Branch mass: what a Gaussian summary cannot carry",
                    fontsize=13, color=INK)
    for suffix in ("png", "pdf"):
        figure.savefig(f"{output}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _target_is_constant(rows, regime, observations):
    for n in observations:
        values = {round(row["reference_branch_mass"], 12) for row in rows
                  if row["regime"] == regime and row["observations"] == n
                  and not row.get("failed")}
        if len(values) > 1:
            return False
    return True


def calibration_figure(output, rows, regimes=None, observations=None):
    regimes = regimes or present(rows, "regime")
    observations = observations or sorted(present(rows, "observations"))
    names = method_order(rows)
    panels = [
        ("width_ratio", "width / exact width", False, 1.0),
        ("mean_error_sigma", "mean error [exact sd]", False, 0.0),
        ("local_w1_over_sigma", "local $W_1$ / sd", True, None),
    ]
    figure, axes = plt.subplots(
        len(panels), len(regimes), squeeze=False, sharex=True,
        figsize=(3.4 * len(regimes), 2.8 * len(panels)), layout="constrained",
    )
    for row, (key, ylabel, log, target) in enumerate(panels):
        for column, regime in enumerate(regimes):
            axis = axes[row][column]
            _curve_panel(axis, rows, regime, key, observations, names, log=log)
            if target is not None:
                axis.axhline(target, color=INK, linewidth=1.4,
                             linestyle=(0, (4, 2)), zorder=5,
                             label="exact value")
            if row == 0:
                axis.set_title(regime, fontsize=11, color=INK)
            if row == len(panels) - 1:
                axis.set_xlabel("observations $n$", fontsize=9.5, color=INK)
            if column == 0:
                axis.set_ylabel(ylabel, fontsize=9.5, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False,
                  ncols=min(len(labels), 4), fontsize=9.5, labelcolor=INK)
    figure.suptitle("Shape, location and the local parameters", fontsize=13,
                    color=INK)
    for suffix in ("png", "pdf"):
        figure.savefig(f"{output}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tag", nargs="+", default=["main"],
                        help="One or more sweeps to draw. Later tags replace "
                             "earlier ones cell by cell, which is how a rerun "
                             "of a single n is folded in.")
    result.add_argument("--regimes", nargs="+", default=None)
    result.add_argument("--oracle-tag", nargs="+", default=None,
                        help="Sweep(s) run with --oracle. Given these, the "
                             "network-vs-exact-score comparison is drawn too.")
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    rows, dataset_records, draws = load_all(arguments.artifacts, arguments.tag)
    regimes = arguments.regimes or [r for r in datasets.REGIMES
                                    if r in present(rows, "regime")]
    output = Path(arguments.artifacts) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    suffix = "" if arguments.tag[0] == "main" else f"_{arguments.tag[0]}"

    print(posterior_figure(output / f"posterior_vs_analytic{suffix}", rows,
                           dataset_records, draws, regimes=regimes))
    for regime in regimes:
        print(posterior_figure(
            output / f"posterior_{regime}{suffix}", rows, dataset_records,
            draws, regimes=[regime],
            title=f"Composed vs exact $p(g \\mid x_{{1:n}})$ -- {regime} data",
        ))
    print(scaling_figure(output / f"w1_scaling{suffix}", rows,
                         "w1_over_sigma", "$W_1$ / exact sd",
                         "Accuracy of the composed shared posterior",
                         regimes=regimes,
                         ablation=ablation_curve(arguments.artifacts)))
    if arguments.oracle_tag:
        oracle_rows, _, _ = load_all(arguments.artifacts, arguments.oracle_tag)
        print(source_comparison_figure(
            output / f"network_vs_oracle{suffix}", rows, oracle_rows,
            regimes=regimes, ablation=ablation_curve(arguments.artifacts),
        ))
    print(branch_mass_figure(output / f"branch_mass{suffix}", rows,
                             regimes=regimes))
    print(calibration_figure(output / f"calibration{suffix}", rows,
                             regimes=regimes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
