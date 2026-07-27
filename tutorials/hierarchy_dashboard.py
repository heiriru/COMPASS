"""Publication-style hierarchy validation dashboard."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

INK = "#17223B"
MUTED = "#65758B"
GRID = "#DCE3EC"
PAPER = "#F5F7FB"
WHITE = "#FFFFFF"
BLUE = "#4361EE"
CORAL = "#E76F51"
TEAL = "#159A8C"
AMBER = "#E9A23B"
PURPLE = "#7C5CFC"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _select(rows: Sequence[dict[str, str]], **filters: object) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if all(str(row[key]) == str(value) for key, value in filters.items()):
            selected.append(row)
    return selected


def _summary(rows: Sequence[dict[str, str]], metric: str) -> tuple[float, float, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    if not len(values):
        raise ValueError(f"No values available for {metric}")
    return float(values.mean()), float(np.quantile(values, 0.16)), float(np.quantile(values, 0.84))


def _series(
    rows: Sequence[dict[str, str]], n_values: Sequence[int], metric: str,
    **filters: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    summaries = [
        _summary(_select(rows, n_observations=n, **filters), metric)
        for n in n_values
    ]
    return tuple(np.asarray(values) for values in zip(*summaries))


def _setup_axis(axis: mpl.axes.Axes, letter: str, title: str, subtitle: str) -> None:
    axis.set_facecolor(WHITE)
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.72)
    axis.grid(axis="x", visible=False)
    axis.tick_params(colors=MUTED, labelsize=9)
    for spine in axis.spines.values():
        spine.set_color("#D5DDE8")
        spine.set_linewidth(0.8)
    axis.text(
        -0.02, 1.13, letter, transform=axis.transAxes, ha="left", va="center",
        fontsize=10, fontweight="bold", color=WHITE,
        bbox={"boxstyle": "round,pad=0.34", "facecolor": INK, "edgecolor": "none"},
    )
    axis.text(0.075, 1.13, title, transform=axis.transAxes, ha="left", va="center",
              fontsize=12.5, fontweight="bold", color=INK)
    axis.text(0.075, 1.055, subtitle, transform=axis.transAxes, ha="left", va="center",
              fontsize=8.7, color=MUTED)


def _format_n_axis(axis: mpl.axes.Axes, n_values: Sequence[int]) -> None:
    axis.set_xscale("log")
    axis.set_xticks(n_values, labels=[str(value) for value in n_values])
    axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axis.set_xlabel("Number of observations, N", color=INK, labelpad=8)


def _band_line(
    axis: mpl.axes.Axes, n_values: Sequence[int], values: tuple[np.ndarray, np.ndarray, np.ndarray],
    color: str, label: str, marker: str,
) -> None:
    centre, lower, upper = values
    axis.fill_between(n_values, lower, upper, color=color, alpha=0.13, linewidth=0)
    axis.plot(
        n_values, centre, color=color, linewidth=2.5, marker=marker,
        markersize=6, markerfacecolor=WHITE, markeredgewidth=1.8, label=label,
    )


def plot_hierarchy_dashboard(csv_path: Path, output: Path) -> None:
    rows = _read(csv_path)
    n_values = sorted({int(row["n_observations"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6))
    fig.patch.set_facecolor(PAPER)
    # A dedicated header band and generous row spacing prevent title collisions.
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.82,
                        wspace=0.24, hspace=0.52)
    fig.suptitle(
        "Global–local hierarchy validation", x=0.055, y=0.975, ha="left",
        fontsize=19, fontweight="bold", color=INK,
    )
    fig.text(
        0.055, 0.932,
        "Correct sharing pools global information while preserving observation-specific local structure",
        ha="left", va="top", fontsize=10.5, color=MUTED,
    )

    truth_specs = [
        (0.0, "linear", BLUE, "Linear truth  ·  c = 0", "o"),
        (0.6, "quadratic", CORAL, "Quadratic truth  ·  c = 0.6", "s"),
    ]
    for c_true, model, color, label, marker in truth_specs:
        common = {"configuration": "correct", "c_true": c_true, "model": model}
        global_values = _series(rows, n_values, "global_parameter_rmse", **common)
        local_values = _series(rows, n_values, "b_rmse", **common)
        weight_values = _series(rows, n_values, "bic_weight", **common)
        _band_line(axes[0, 0], n_values, global_values, color, label, marker)
        _band_line(axes[0, 1], n_values, local_values, color, label, marker)
        _band_line(axes[1, 1], n_values, weight_values, color, label, marker)

    _setup_axis(axes[0, 0], "A", "Global recovery", "Mean RMSE; shaded band is the central 68% across mock datasets")
    _format_n_axis(axes[0, 0], n_values)
    axes[0, 0].set_ylabel("Global-parameter RMSE", color=INK, labelpad=8)
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)

    _setup_axis(axes[0, 1], "B", "Local recovery", "Each observation retains its own intercept bᵢ")
    _format_n_axis(axes[0, 1], n_values)
    axes[0, 1].set_ylabel("Local-intercept RMSE", color=INK, labelpad=8)

    sharing_axis = axes[1, 0]
    sharing_specs = [
        ("all_local", AMBER, "All local", "-", 2),
        ("all_global", PURPLE, "All global", "-", 3),
        ("correct", TEAL, "Correct hierarchy", "--", 8),
    ]
    sharing_floor = 1e-8
    for configuration, color, label, linestyle, zorder in sharing_specs:
        centre, lower, upper = _series(
            rows, n_values, "a_sharing_range", configuration=configuration, model="quadratic",
        )
        centre = np.maximum(centre, sharing_floor)
        lower = np.maximum(lower, sharing_floor)
        upper = np.maximum(upper, sharing_floor)
        sharing_axis.fill_between(
            n_values, lower, upper, color=color,
            alpha=0.10 if configuration != "correct" else 0.16,
            linewidth=0, zorder=zorder - 1,
        )
        sharing_axis.plot(
            n_values, centre, color=color, linestyle=linestyle,
            linewidth=2.7 if configuration == "correct" else 2.2,
            marker="o", markersize=5.5, markerfacecolor=WHITE,
            markeredgewidth=1.5, label=label, zorder=zorder,
        )
    _setup_axis(sharing_axis, "C", "Sharing constraint", "Zero means the inferred global value is identical for every observation")
    _format_n_axis(sharing_axis, n_values)
    sharing_axis.set_yscale("log")
    sharing_axis.set_ylabel("Range of inferred global a", color=INK, labelpad=8)
    sharing_axis.legend(loc="center right", frameon=False, fontsize=8.5, labelcolor=INK)
    sharing_axis.text(
        0.03, 0.08, "exact sharing", transform=sharing_axis.transAxes,
        color=TEAL, fontsize=8.5, fontweight="bold",
    )

    weight_axis = axes[1, 1]
    weight_axis.axhspan(0.5, 1.0, color=TEAL, alpha=0.045, zorder=0)
    weight_axis.axhline(0.5, color=MUTED, linestyle=(0, (2, 3)), linewidth=1.2, zorder=1)
    _setup_axis(weight_axis, "D", "Model identification", "BIC support assigned to the model that generated the data")
    _format_n_axis(weight_axis, n_values)
    weight_axis.set_ylim(-0.03, 1.03)
    weight_axis.set_ylabel("BIC weight of generating model", color=INK, labelpad=8)
    weight_axis.text(
        0.98, 0.54, "more support than alternative", transform=weight_axis.transAxes,
        ha="right", va="bottom", fontsize=8.2, color=TEAL,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {output}")


def plot_model_pairplot(
    output: Path,
    data_output: Path | None = None,
    seed: int = 42,
    samples_per_model: int = 500,
) -> None:
    """Visualize the linear and quadratic observation geometries with random draws."""
    rng = np.random.default_rng(seed)
    t_values = np.asarray([-1.0, 0.0, 1.0])
    model_specs = [
        ("Linear · c = 0", 0.0, BLUE),
        ("Quadratic · c = 0.6", 0.6, CORAL),
    ]
    generated: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for label, curvature, _ in model_specs:
        a = rng.normal(0.0, 1.0, samples_per_model)
        b = rng.normal(0.0, 1.5, samples_per_model)
        noise = rng.normal(0.0, 0.15, (samples_per_model, len(t_values)))
        observations = (
            b[:, None] + a[:, None] * t_values[None, :]
            + curvature * t_values[None, :] ** 2 + noise
        )
        contrast = 0.5 * (observations[:, 0] + observations[:, 2]) - observations[:, 1]
        arrays[label] = np.column_stack([observations, contrast])
        for index in range(samples_per_model):
            generated.append({
                "model": label, "a": a[index], "b": b[index], "c": curvature,
                "x_t_minus_1": observations[index, 0],
                "x_t_0": observations[index, 1],
                "x_t_plus_1": observations[index, 2],
                "curvature_contrast": contrast[index],
            })

    feature_labels = (
        "x(t = −1)", "x(t = 0)", "x(t = +1)",
        "curvature contrast\n½[x(−1)+x(+1)]−x(0)",
    )
    fig, axes = plt.subplots(4, 4, figsize=(12.2, 11.2))
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.84,
                        wspace=0.08, hspace=0.08)
    fig.suptitle(
        "Linear versus quadratic observation geometry", x=0.06, y=0.975,
        ha="left", fontsize=19, fontweight="bold", color=INK,
    )
    fig.text(
        0.06, 0.935,
        "Random draws from  x(t)=b+a·t+c·t²+ε  with  a~N(0,1),  b~N(0,1.5²),  ε~N(0,0.15²)",
        ha="left", fontsize=10.5, color=MUTED,
    )

    handles = []
    for label, _, color in model_specs:
        handles.append(mpl.lines.Line2D(
            [], [], linestyle="", marker="o", markersize=7,
            markerfacecolor=color, markeredgecolor=WHITE, label=label,
        ))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.977),
               frameon=False, fontsize=9.5)

    for row in range(4):
        for column in range(4):
            axis = axes[row, column]
            axis.set_facecolor(WHITE)
            axis.grid(color=GRID, linewidth=0.65, alpha=0.55)
            for spine in axis.spines.values():
                spine.set_color("#D5DDE8")
                spine.set_linewidth(0.7)
            if row == column:
                combined = np.concatenate([
                    arrays[label][:, column] for label, _, _ in model_specs
                ])
                padding = 0.06 * max(float(np.ptp(combined)), 1e-6)
                grid = np.linspace(
                    float(combined.min() - padding),
                    float(combined.max() + padding),
                    400,
                )
                for label, _, color in model_specs:
                    density = gaussian_kde(arrays[label][:, column])(grid)
                    axis.fill_between(grid, 0.0, density, color=color, alpha=0.18)
                    axis.plot(grid, density, color=color, linewidth=1.8)
                axis.set_yticks([])
            else:
                for label, _, color in model_specs:
                    axis.scatter(
                        arrays[label][:, column], arrays[label][:, row],
                        s=8, alpha=0.22, color=color, edgecolors="none", rasterized=True,
                    )
            if row == 3:
                axis.set_xlabel(feature_labels[column], color=INK, fontsize=9, labelpad=7)
            else:
                axis.set_xticklabels([])
            if column == 0 and row != 0:
                axis.set_ylabel(feature_labels[row], color=INK, fontsize=9, labelpad=7)
            elif column != 0:
                axis.set_yticklabels([])
            axis.tick_params(colors=MUTED, labelsize=7.5)

    fig.text(
        0.5, 0.025,
        "The curvature contrast concentrates near 0 for the linear model and near 0.6 for the quadratic model.",
        ha="center", color=MUTED, fontsize=9.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {output}")

    if data_output is not None:
        data_output.parent.mkdir(parents=True, exist_ok=True)
        with data_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(generated[0]))
            writer.writeheader()
            writer.writerows(generated)
        print(f"Wrote {data_output}")
