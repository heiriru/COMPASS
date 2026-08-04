"""Regenerate partial-pooling figures from several completed inference runs."""

import argparse
import json
import re
import sys
from pathlib import Path


METHOD_STYLES = {
    "dpm2_damped_sum": ("DPM-Solver-2 + damped sum", "#0072B2"),
    "dpm2_gaussian": ("DPM-Solver-2 + Gaussian", "#D55E00"),
    "dpm2_full_gaussian": ("DPM-Solver-2 + full Gaussian", "#CC79A7"),
    "dpm2_gauss_global_local_moment": (
        "DPM-Solver-2 + global/local Gaussian moments", "#E69F00",
    ),
    "langevin_fnpse": ("Langevin + F-NPSE", "#009E73"),
}


def parser():
    result = argparse.ArgumentParser(
        description=(
            "Create method-specific and combined partial-pooling plots from "
            "completed recovery artifacts; no inference is run."
        )
    )
    result.add_argument(
        "--preset", choices=("smoke", "full", "large"), default="full",
    )
    result.add_argument("--sde-type", choices=("vesde", "vpsde"), default="vesde")
    result.add_argument("--beta-min", type=float, default=0.1)
    result.add_argument("--beta-max", type=float, default=20.0)
    result.add_argument("--root", type=Path)
    result.add_argument("--seed", type=int)
    result.add_argument(
        "--run-signature", action="append", dest="run_signatures", required=True,
        help=(
            "Completed recovery signature to include. Repeat once per method, "
            "in the desired legend order."
        ),
    )
    result.add_argument(
        "--output-signature",
        help=(
            "Figure-directory signature. Defaults to the first --run-signature."
        ),
    )
    return result


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def method_metadata(artifact, run_config=None):
    """Return a stable key, readable label, and color for old and new runs."""
    run_config = run_config or {}
    method = artifact.get("inference_method") or run_config.get("inference_method")
    correction = artifact.get("correction") or run_config.get("correction")
    sampler = artifact.get("sampler") or run_config.get("sampler")
    order = artifact.get("order") or run_config.get("order")
    if method in METHOD_STYLES:
        key = method
    elif correction == "damped_sum" and sampler in (None, "dpm"):
        key = "dpm2_damped_sum" if order in (None, 2) else f"dpm{order}_damped_sum"
    elif correction == "gauss" and sampler == "dpm":
        key = f"dpm{order or 2}_gaussian"
    elif correction == "full_gaussian" and sampler == "dpm":
        key = f"dpm{order or 2}_full_gaussian"
    elif correction == "fnpe" and sampler == "langevin":
        key = "langevin_fnpse"
    else:
        key = _slug(method or f"{sampler or 'unknown'}_{correction or 'unknown'}")
    default_label, color = METHOD_STYLES.get(
        key, (key.replace("_", " ").title(), "#CC79A7"),
    )
    return {
        "key": key,
        "label": default_label,
        "color": color,
        "point_estimator": artifact.get("joint_map", {}).get(
            "settings", {}
        ).get("estimator", artifact.get("joint_map", {}).get("estimator", "joint_map")),
    }


def _load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _load_artifacts(directory, signature, torch):
    artifacts = []
    candidates = {
        *directory.glob("dataset-*.pt"),
        *directory.glob("*-dataset-*.pt"),
    }
    for path in sorted(candidates):
        artifact = torch.load(path, map_location="cpu")
        if artifact.get("inference_signature") == signature:
            artifacts.append(artifact)
    return artifacts


def _load_run(paths, preset, signature, torch):
    parent = paths.root / "partial_pooling_recovery" / preset
    exact = parent / signature
    matches = [path for path in parent.glob(f"*-{signature}") if path.is_dir()]
    if exact.is_dir():
        run_directory = exact
    elif len(matches) == 1:
        run_directory = matches[0]
    elif not matches:
        raise FileNotFoundError(
            f"No recovery directory ends in signature {signature!r} below {parent}."
        )
    else:
        raise RuntimeError(
            f"Several recovery directories match signature {signature!r}: {matches}"
        )
    config_candidates = sorted(run_directory.glob("*-run_config.json"))
    run_config_path = (
        config_candidates[0] if len(config_candidates) == 1
        else run_directory / "run_config.json"
    )
    run_config = _load_json(run_config_path)
    artifacts = _load_artifacts(run_directory, signature, torch)
    if not artifacts:
        raise RuntimeError(f"No completed datasets found in {run_directory}")
    metadata = method_metadata(artifacts[0], run_config)
    subjects = int(artifacts[0]["subjects"])
    sweep_directory = run_directory / "observation_sweep"
    manifest_candidates = sorted(sweep_directory.glob("*-manifest.json"))
    sweep_manifest_path = (
        manifest_candidates[0] if len(manifest_candidates) == 1
        else sweep_directory / "manifest.json"
    )
    sweep_manifest = _load_json(sweep_manifest_path)
    counts = tuple(int(count) for count in sweep_manifest.get("counts", ()))
    if not counts:
        counts = tuple(sorted({
            int(path.name.removeprefix("subjects-"))
            for path in sweep_directory.glob("subjects-*") if path.is_dir()
        } | {subjects}))
    sweep = {}
    for count in counts:
        directory = (
            run_directory if count == subjects
            else sweep_directory / f"subjects-{count:04d}"
        )
        loaded = _load_artifacts(directory, signature, torch)
        if loaded:
            sweep[count] = loaded
    return {
        "signature": signature,
        "directory": run_directory,
        "config": run_config,
        "artifacts": artifacts,
        "sweep": sweep,
        **metadata,
    }


def _dataset_ids(artifacts):
    return tuple(sorted(int(artifact["dataset_id"]) for artifact in artifacts))


def _validate_runs(runs, config_signature):
    keys = [run["key"] for run in runs]
    if len(set(keys)) != len(keys):
        raise RuntimeError(
            "Each run must represent a distinct method; got " + ", ".join(keys)
        )
    expected_datasets = _dataset_ids(runs[0]["artifacts"])
    expected_counts = tuple(sorted(runs[0]["sweep"]))
    if not expected_counts:
        raise RuntimeError(f"Run {runs[0]['signature']} has no observation sweep.")
    for run in runs:
        artifact_signature = run["artifacts"][0].get("config_signature")
        if artifact_signature != config_signature:
            raise RuntimeError(
                f"Run {run['signature']} uses configuration {artifact_signature}, "
                f"not {config_signature}."
            )
        if _dataset_ids(run["artifacts"]) != expected_datasets:
            raise RuntimeError(
                "All methods must contain the same completed base datasets."
            )
        counts = tuple(sorted(run["sweep"]))
        if counts != expected_counts:
            raise RuntimeError(
                f"Observation counts differ for {run['signature']}: "
                f"{counts} != {expected_counts}."
            )
        for count in expected_counts:
            if _dataset_ids(run["sweep"][count]) != expected_datasets:
                raise RuntimeError(
                    f"Run {run['signature']} is incomplete at {count} subjects."
                )


def _rows(run):
    return (
        [row for artifact in run["artifacts"] for row in artifact["global_rows"]],
        [row for artifact in run["artifacts"] for row in artifact["local_rows"]],
    )


def _plot_method_specific(runs, output_directory, global_names, local_names, plt):
    from Partial_Pooling.infer_partial_pooling import (
        _plot_error_distributions,
        _plot_global_density_grid,
        _plot_local_shrinkage,
        _plot_parity,
        _plot_residuals,
    )

    outputs = []
    by_method = {}
    for run in runs:
        suffix = f"_{run['key']}"
        global_rows, local_rows = _rows(run)
        method_outputs = []
        method_outputs.extend(_plot_parity(
            global_rows, local_rows, global_names, local_names,
            output_directory, plt, filename_suffix=suffix,
            method_label=run["label"],
        ))
        method_outputs.extend(_plot_residuals(
            global_rows, local_rows, global_names, local_names,
            output_directory, plt, filename_suffix=suffix,
            method_label=run["label"],
        ))
        method_outputs.append(_plot_global_density_grid(
            run["sweep"], global_names,
            output_directory / f"global_forest{suffix}.png", plt,
            method_label=run["label"],
        ))
        method_outputs.append(_plot_local_shrinkage(
            global_rows, local_rows, local_names,
            output_directory / f"local_shrinkage{suffix}.png", plt,
            method_label=run["label"],
        ))
        method_outputs.append(_plot_error_distributions(
            global_rows, local_rows, global_names, local_names,
            output_directory / f"error_distributions{suffix}.png", plt,
            method_label=run["label"],
        ))
        by_method[run["key"]] = [path.name for path in method_outputs]
        outputs.extend(method_outputs)
    return outputs, by_method


def _finish(fig, output, plt):
    from Partial_Pooling.infer_partial_pooling import FIGURE_DPI

    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_combined_error_distributions(
    runs, global_names, local_names, output, plt,
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), constrained_layout=True)
    method_count = len(runs)
    group_width = 0.78
    box_width = group_width / method_count
    for axis, names, scope, title in (
        (axes[0], global_names, "global", "Global point-estimate errors"),
        (axes[1], local_names, "local", "Local point-estimate errors"),
    ):
        centers = list(range(1, len(names) + 1))
        for method_index, run in enumerate(runs):
            global_rows, local_rows = _rows(run)
            rows = global_rows if scope == "global" else local_rows
            values = [[
                float(row["joint_map"]) - float(row["truth"])
                for row in rows if row["parameter"] == name
            ] for name in names]
            positions = [
                center - group_width / 2 + box_width * (method_index + 0.5)
                for center in centers
            ]
            boxes = axis.boxplot(
                values, positions=positions, widths=box_width * 0.82,
                patch_artist=True, showmeans=True, manage_ticks=False,
                meanprops={
                    "marker": "D", "markerfacecolor": run["color"],
                    "markeredgecolor": "black", "markersize": 4,
                },
                medianprops={"color": "black", "linewidth": 1},
            )
            for box in boxes["boxes"]:
                box.set(facecolor=run["color"], alpha=0.42)
            for parameter_index, parameter_values in enumerate(values):
                jitter = [
                    ((index % 11) - 5) * box_width * 0.018
                    for index in range(len(parameter_values))
                ]
                axis.scatter(
                    [positions[parameter_index] + value for value in jitter],
                    parameter_values, s=8, color=run["color"], alpha=0.22,
                    edgecolors="none",
                )
        axis.axhline(0.0, color="black", lw=1, ls="--")
        axis.set(
            title=title, ylabel="point estimate - truth",
            xticks=centers, xticklabels=names,
        )
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    fig.legend(
        handles=[
            *[
                Patch(facecolor=run["color"], alpha=0.42, label=run["label"])
                for run in runs
            ],
            Line2D([0], [0], color="black", ls="--", label="zero error"),
        ],
        loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=True,
    )
    fig.suptitle("Recovery error distributions: all methods", fontsize=14)
    return _finish(fig, output, plt)


def _observation_errors(sweep, theta_scale):
    import numpy as np

    scale = theta_scale.detach().cpu()
    global_scale = scale[:7]
    local_scale = scale[7:]
    global_values = []
    local_values = []
    for count in sorted(sweep):
        global_errors = []
        local_errors = []
        for artifact in sweep[count]:
            global_error = (
                artifact["joint_map"]["globals"] - artifact["truth_globals"]
            ) / global_scale
            local_error = (
                artifact["joint_map"]["locals"] - artifact["truth_locals"]
            ) / local_scale
            global_errors.append(float(global_error.square().mean().sqrt()))
            local_errors.append(float(local_error.square().mean().sqrt()))
        global_values.append(global_errors)
        local_values.append(local_errors)
    return {
        "global_mean": np.asarray([np.mean(values) for values in global_values]),
        "global_std": np.asarray([np.std(values) for values in global_values]),
        "local_mean": np.asarray([np.mean(values) for values in local_values]),
        "local_std": np.asarray([np.std(values) for values in local_values]),
    }


def _plot_combined_error_by_observation_count(
    runs, theta_scale, output, plt, trials_per_subject,
):
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    counts = tuple(sorted(runs[0]["sweep"]))
    x = np.asarray(counts)
    fig, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for run in runs:
        values = _observation_errors(run["sweep"], theta_scale)
        axis.plot(
            x, values["global_mean"], marker="o", lw=2,
            color=run["color"], ls="-",
        )
        axis.fill_between(
            x, np.maximum(0.0, values["global_mean"] - values["global_std"]),
            values["global_mean"] + values["global_std"],
            color=run["color"], alpha=0.12,
        )
        axis.plot(
            x, values["local_mean"], marker="s", lw=1.8,
            color=run["color"], ls="--",
        )
        axis.fill_between(
            x, np.maximum(0.0, values["local_mean"] - values["local_std"]),
            values["local_mean"] + values["local_std"],
            color=run["color"], alpha=0.055,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, [str(count) for count in counts])
    axis.set(
        xlabel=f"subjects used ({trials_per_subject} trials per subject)",
        ylabel="normalized point-estimate RMSE",
        title="Recovery error versus observations: all methods",
    )
    axis.grid(alpha=0.22)
    fig.legend(
        handles=[
            *[
                Line2D([0], [0], color=run["color"], lw=2, label=run["label"])
                for run in runs
            ],
            Line2D([0], [0], color="black", marker="o", lw=2,
                   ls="-", label="global RMSE"),
            Line2D([0], [0], color="black", marker="s", lw=1.8,
                   ls="--", label="local RMSE"),
            Patch(facecolor="black", alpha=0.12,
                  label="+/- 1 SD across datasets"),
        ],
        loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=True,
    )
    return _finish(fig, output, plt)


def _artifact_for_dataset(artifacts, dataset_id):
    return next(
        artifact for artifact in artifacts
        if int(artifact["dataset_id"]) == dataset_id
    )


def _plot_combined_global_density_grid(runs, global_names, output, plt):
    import numpy as np
    from matplotlib.lines import Line2D

    from Partial_Pooling.infer_partial_pooling import (
        _robust_density_limits,
        _safe_1d_kde,
        density_panel_counts,
    )

    counts = density_panel_counts(tuple(sorted(runs[0]["sweep"])), columns=4)
    common_datasets = set(_dataset_ids(runs[0]["sweep"][counts[0]]))
    for run in runs[1:]:
        common_datasets &= set(_dataset_ids(run["sweep"][counts[0]]))
    if not common_datasets:
        raise RuntimeError("The method sweeps do not share a dataset for density plots.")
    dataset_id = min(common_datasets)
    selected = {
        run["key"]: {
            count: _artifact_for_dataset(run["sweep"][count], dataset_id)
            for count in counts
        }
        for run in runs
    }
    trials_per_subject = int(
        selected[runs[0]["key"]][counts[0]].get("trials_per_subject", 30)
    )
    fig, axes = plt.subplots(
        len(global_names), len(counts),
        figsize=(4.1 * len(counts), 2.55 * len(global_names)),
        squeeze=False, constrained_layout=True,
    )
    for parameter, name in enumerate(global_names):
        reference = selected[runs[0]["key"]][counts[0]]
        truth = float(reference["truth_globals"][parameter])
        value_sets = []
        all_maps = []
        all_medians = []
        for run in runs:
            for count in counts:
                artifact = selected[run["key"]][count]
                values = artifact["posterior"]["globals"][:, parameter].numpy()
                value_sets.append(values)
                all_maps.append(float(artifact["joint_map"]["globals"][parameter]))
                all_medians.append(float(np.median(values)))
        lower, upper = _robust_density_limits(
            value_sets, anchors=(*all_maps, *all_medians, truth),
        )
        grid = np.linspace(lower, upper, 300)
        for column, count in enumerate(counts):
            axis = axes[parameter, column]
            for run in runs:
                artifact = selected[run["key"]][count]
                values = artifact["posterior"]["globals"][:, parameter].numpy()
                density = _safe_1d_kde(values, grid)
                if density is not None:
                    axis.fill_between(
                        grid, density, color=run["color"], alpha=0.08,
                    )
                    axis.plot(grid, density, color=run["color"], lw=1.5)
                else:
                    axis.axvline(float(values[0]), color=run["color"], lw=1.5)
                axis.axvline(
                    float(artifact["joint_map"]["globals"][parameter]),
                    color=run["color"], lw=1.25, ls="--",
                )
                axis.axvline(
                    float(np.median(values)), color=run["color"],
                    lw=1.15, ls="-.",
                )
                outside = int(
                    np.count_nonzero((values < lower) | (values > upper))
                )
                if outside:
                    axis.text(
                        0.98, 0.94 - 0.08 * runs.index(run),
                        f"{run['key']}: {outside}/{values.size} outside",
                        transform=axis.transAxes, ha="right", va="top",
                        fontsize=6.5, color=run["color"],
                    )
            axis.axvline(truth, color="black", lw=1.4, ls=":")
            if parameter == 0:
                axis.set_title(
                    f"{count} subject{'s' if count != 1 else ''}\n"
                    f"({count * trials_per_subject} trials)"
                )
            if column == 0:
                axis.set_ylabel(f"{name}\ndensity")
            if parameter == len(global_names) - 1:
                axis.set_xlabel("parameter value")
            axis.set_xlim(grid[0], grid[-1])
            axis.set_yticks([])
            axis.grid(axis="x", alpha=0.16)
    fig.legend(
        handles=[
            *[
                Line2D([0], [0], color=run["color"], lw=1.8, label=run["label"])
                for run in runs
            ],
            Line2D([0], [0], color="black", lw=1.5, ls="-",
                   label="posterior KDE"),
            Line2D([0], [0], color="black", lw=1.25, ls="--",
                   label="point estimate"),
            Line2D([0], [0], color="black", lw=1.15, ls="-.",
                   label="posterior median"),
            Line2D([0], [0], color="black", lw=1.4, ls=":",
                   label="simulator truth"),
        ],
        loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=True,
    )
    fig.suptitle(
        f"Global posterior densities and point estimates: all methods "
        f"(dataset {dataset_id})",
        fontsize=14,
    )
    return _finish(fig, output, plt)


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from Partial_Pooling.config import get_config
    from Partial_Pooling.io_utils import atomic_json
    from Partial_Pooling.model_pipeline import load_normalizers
    from Partial_Pooling.paths import BenchmarkPaths
    from Partial_Pooling.schema import GLOBAL_NAMES, LOCAL_NAMES

    if len(args.run_signatures) < 2:
        raise ValueError("Repeat --run-signature for at least two methods.")
    if len(set(args.run_signatures)) != len(args.run_signatures):
        raise ValueError("Duplicate --run-signature values are not allowed.")
    config = get_config(
        args.preset, args.seed, args.sde_type,
        args.beta_min, args.beta_max,
    )
    paths = BenchmarkPaths(args.root) if args.root else BenchmarkPaths.default()
    runs = [
        _load_run(paths, args.preset, signature, torch)
        for signature in args.run_signatures
    ]
    _validate_runs(runs, config.signature)
    output_signature = args.output_signature or args.run_signatures[0]
    matching_run = next(
        (run for run in runs if run["signature"] == output_signature), None,
    )
    output_name = (
        f"{matching_run['key']}-{output_signature}"
        if matching_run else f"method_comparison-{output_signature}"
    )
    output_directory = (
        paths.figures / "partial_pooling" / args.preset / output_name
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    theta_scale = load_normalizers(config, paths)["sde_joint_theta"].scale
    trials_per_subject = int(runs[0]["artifacts"][0].get("trials_per_subject", 30))
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    })

    outputs, method_figures = _plot_method_specific(
        runs, output_directory, GLOBAL_NAMES, LOCAL_NAMES, plt,
    )
    combined = [
        _plot_combined_error_distributions(
            runs, GLOBAL_NAMES, LOCAL_NAMES,
            output_directory / "error_distributions.png", plt,
        ),
        _plot_combined_error_by_observation_count(
            runs, theta_scale, output_directory / "error_vs_observations.png",
            plt, trials_per_subject,
        ),
        _plot_combined_global_density_grid(
            runs, GLOBAL_NAMES, output_directory / "global_forest.png", plt,
        ),
    ]
    outputs.extend(combined)
    atomic_json(output_directory / "comparison_manifest.json", {
        "preset": args.preset,
        "output_signature": output_signature,
        "runs": [
            {
                "signature": run["signature"],
                "method": run["key"],
                "label": run["label"],
                "point_estimator": run["point_estimator"],
                "completed_datasets": len(run["artifacts"]),
                "observation_counts": list(sorted(run["sweep"])),
            }
            for run in runs
        ],
        "method_specific_figures": method_figures,
        "combined_figures": [path.name for path in combined],
    })
    print(
        f"Created {len(outputs)} comparison figures in {output_directory}",
        flush=True,
    )
    return outputs


def main(argv=None):
    args = parser().parse_args(argv)
    # Apply the shared CPU/GPU resource guard before importing numerical packages.
    from runtime import configure_runtime
    configure_runtime()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
