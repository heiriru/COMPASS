#!/usr/bin/env python3
"""Run isolated terminal-damping sweeps for selected DPM-2 corrections.

This driver intentionally does not call ``damping_factor_comparison.main``.
It calculates only the selected method's missing d1 cells, resumes from its
own per-cell cache, and writes one method-specific CSV and figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import damping_factor_comparison as base


METHODS = {
    "dpm2_damping": {
        "variant_key": "dpm2_damping_c5",
        "output_stem": "learned_score_damping_d1_sweep_dpm2_damping",
    },
    "dpm2_gaussian_damping": {
        "variant_key": "dpm2_gauss_damping",
        "output_stem": "learned_score_damping_d1_sweep_dpm2_gaussian_damping",
    },
    "dpm2_hybrid_damping": {
        "variant_key": "dpm2_hybrid_damping",
        "output_stem": "learned_score_damping_d1_sweep_dpm2_hybrid_damping",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", required=True, choices=tuple(METHODS))
    parser.add_argument("--output-dir", type=Path, default=base.DEFAULT_OUTPUT)
    parser.add_argument(
        "--shared-checkpoint", type=Path, default=base.DEFAULT_SHARED_CHECKPOINT,
    )
    parser.add_argument("--scaling-raw", type=Path, default=base.DEFAULT_SCALING_RAW)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--posterior-samples", type=int, default=3_000)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--n-values", type=base.parse_n_values, default=base.DEFAULT_N_VALUES,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--adaptive-abs-tol", type=float, default=0.002576)
    parser.add_argument("--adaptive-rel-tol", type=float, default=0.1)
    parser.add_argument("--adaptive-safety", type=float, default=0.9)
    parser.add_argument("--adaptive-exponent", type=float, default=0.9)
    parser.add_argument("--adaptive-max-evals", type=int, default=10_000)
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="recalculate only this method's dedicated d1-sweep cache cells",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="require all selected cache cells and only rebuild the CSV/figure",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def solver_variant(variant_key: str) -> dict[str, Any]:
    return next(
        dict(variant)
        for variant in base.SOLVER_VARIANTS
        if variant["key"] == variant_key
    )


def dedicated_cell_path(
    output_dir: Path, method: str, damping_key: str, repeat: int, n: int,
) -> Path:
    return (
        output_dir / "cache" / "d1_method_sweeps" / method / damping_key
        / f"repeat_{repeat}_n_{n}.npz"
    )


def cache_matches(path: Path, config: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    cached = base.load_npz(path)
    try:
        cached_config = json.loads(str(cached["config_json"].item()))
    except (KeyError, json.JSONDecodeError):
        return False
    return cached_config == config


def failure_path(cell_path: Path) -> Path:
    return cell_path.with_suffix(".failure.json")


def load_matching_failure(
    cell_path: Path, config: dict[str, Any],
) -> dict[str, Any] | None:
    path = failure_path(cell_path)
    if not path.exists():
        return None
    try:
        failure = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if failure.get("config_json") != base.jsonable_config(config):
        return None
    return failure


def is_numerical_divergence(error: RuntimeError) -> bool:
    message = str(error)
    return (
        "produced non-finite learned-score samples" in message
        or "produced a non-finite score-ascent MAP" in message
        or "composed precision became non-positive" in message
    )


def save_failure(
    spec: dict[str, Any], method_label: str, error: RuntimeError,
) -> dict[str, Any]:
    failure = {
        "status": "numerical_divergence",
        "method": method_label,
        "damping_key": spec["damping_key"],
        "damping_label": spec["damping_label"],
        "damping_at_noise": spec["damping_at_noise"],
        "repeat": spec["repeat"],
        "n_observations": spec["n"],
        "error": str(error),
        "config_json": base.jsonable_config(spec["config"]),
    }
    path = failure_path(spec["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failure, indent=2) + "\n")
    return failure


def cell_spec(
    args: argparse.Namespace,
    method_variant: dict[str, Any],
    damping_key: str,
    damping_label: str,
    repeat: int,
    n: int,
    damping_at_noise: float,
) -> tuple[Path, dict[str, Any]]:
    """Select a dedicated cell, reusing a matching baseline when possible."""
    sweep_variant = {
        **method_variant,
        "key": f"{method_variant['key']}__{damping_key}",
        "label": f"{method_variant['label']}; {damping_label}",
    }
    dedicated = dedicated_cell_path(
        args.output_dir, args.method, damping_key, repeat, n,
    )

    if damping_key == "d1_inverse_sqrt_n" and not args.force_new:
        baseline = base.learned_cell_path(
            args.output_dir, "learned_solver", method_variant["key"], repeat, n,
        )
        baseline_config = base.learned_config(
            args, method_variant, repeat, n, damping_at_noise,
        )
        if cache_matches(baseline, baseline_config):
            return baseline, method_variant

    return dedicated, sweep_variant


def collect_specs(
    args: argparse.Namespace, method_variant: dict[str, Any],
) -> list[dict[str, Any]]:
    specs = []
    for repeat in range(args.repeats):
        for n in args.n_values:
            for damping_key, damping_label, endpoint in base.SWEEP_VARIANTS:
                damping_at_noise = float(endpoint(n))
                path, variant = cell_spec(
                    args, method_variant, damping_key, damping_label,
                    repeat, n, damping_at_noise,
                )
                specs.append({
                    "repeat": repeat,
                    "n": n,
                    "damping_key": damping_key,
                    "damping_label": damping_label,
                    "damping_at_noise": damping_at_noise,
                    "path": path,
                    "variant": variant,
                    "config": base.learned_config(
                        args, variant, repeat, n, damping_at_noise,
                    ),
                })
    return specs


def main() -> None:
    args = build_parser().parse_args()
    if args.posterior_samples < 2 or args.repeats < 1 or args.timesteps < 2:
        raise ValueError("posterior samples/repeats/timesteps must be at least 2/1/2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    method = METHODS[args.method]
    method_variant = solver_variant(method["variant_key"])
    specs = collect_specs(args, method_variant)
    missing_or_stale = [
        spec for spec in specs
        if (
            not cache_matches(spec["path"], spec["config"])
            and load_matching_failure(spec["path"], spec["config"]) is None
        )
    ]
    if args.plot_only and missing_or_stale:
        raise FileNotFoundError(
            "--plot-only requested, but "
            f"{len(missing_or_stale)} selected cache cells are missing or stale"
        )

    needs_compute = bool(missing_or_stale) or args.force_new
    model = None
    runtime_device = args.device
    if needs_compute:
        base.ensure_runtime(args.device, needs_compute=True)
        model = base.load_score_checkpoint(args.shared_checkpoint, args.device)

    pools = base.load_npz(args.scaling_raw)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in specs:
        recorded_failure = load_matching_failure(spec["path"], spec["config"])
        if recorded_failure is not None and not args.force_new:
            failures.append(recorded_failure)
            print(
                "Skipping recorded numerical divergence: "
                f"{spec['damping_label']}, repeat={spec['repeat']}, N={spec['n']}"
            )
            continue

        repeat = spec["repeat"]
        n = spec["n"]
        observations = pools[f"x_pool_r{repeat}"][:n]
        truth_mean, truth_std = base.learned_truth(observations)
        try:
            result = base.run_learned_cell(
                model, observations, spec["variant"], repeat, n,
                spec["damping_at_noise"], spec["path"], args, runtime_device,
                allow_compute=True,
            )
        except RuntimeError as error:
            if not is_numerical_divergence(error):
                raise
            failure = save_failure(spec, method_variant["label"], error)
            failures.append(failure)
            print(
                "Recorded numerical divergence and continuing: "
                f"{spec['damping_label']}, repeat={repeat}, N={n}"
            )
            if runtime_device == "cuda":
                base.torch.cuda.empty_cache()
            continue

        marker = failure_path(spec["path"])
        if marker.exists():
            marker.unlink()
        rows.append(base.learned_metrics(
            result, truth_mean, truth_std,
            spec["damping_key"], spec["damping_label"], repeat, n,
        ))

    output_stem = method["output_stem"]
    base.write_rows(args.output_dir / f"{output_stem}_metrics.csv", [
        {key: value for key, value in row.items() if key != "calibration"}
        for row in rows
    ])
    failure_csv = args.output_dir / f"{output_stem}_failures.csv"
    if failures:
        base.write_rows(failure_csv, [
            {key: value for key, value in failure.items() if key != "config_json"}
            for failure in failures
        ])
    elif failure_csv.exists():
        failure_csv.unlink()

    available_variants = [
        (key, label) for key, label, _ in base.SWEEP_VARIANTS
        if any(row["variant"] == key for row in rows)
    ]
    if not available_variants:
        raise RuntimeError("No finite sweep cells are available to plot.")
    base.configure_style()
    base.plot_three_panel(
        rows,
        available_variants,
        args.output_dir / f"{output_stem}.png",
        f"{method_variant['label']}: terminal damping sweep",
    )
    if failures:
        print(
            f"Completed with {len(failures)} numerically divergent cells; "
            f"details are in {failure_csv}"
        )

    if model is not None and runtime_device == "cuda":
        base.torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
