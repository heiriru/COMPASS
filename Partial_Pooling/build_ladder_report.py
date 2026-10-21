#!/usr/bin/env python3
"""Compare ladder arms on the partial-pooling recovery metric contract.

Each arm is one (model, inference method) pair: a completed recovery run written
by ``infer_partial_pooling.py`` (directly, or through
``run_churn_comparison.py``). The report answers the ladder's success criterion
-- better than the 5,957,984-parameter legacy model on posterior accuracy, with
no material calibration regression, at under 1,000,000 parameters -- and it
answers it the way the plan's metric contract demands: global and local
parameter error reported separately, calibration as a hard constraint rather
than as another number to average in, and the parameter count next to both.

Errors are standardized by the *prior* standard deviation of each parameter, so
they can be averaged across parameters whose raw scales differ by two orders of
magnitude. A standardized error of 1.0 means "no better than the prior".

Usage:
    python Partial_Pooling/build_ladder_report.py \
        --arm 'legacy-5.96M/400k:dpm2_gauss_jacobian_newton:churn100/jacobian_newton/artifacts_churn_eta2_refresh20' \
        --arm 'iid-trial-98k/100k:dpm2_gauss_jacobian_newton:ladder/tok_trial_h16d2/artifacts_churn_eta2_refresh20' \
        --output Partial_Pooling/artifacts_ladder_report
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit(3)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
PRESET = "small"

sys.path.insert(0, str(ROOT.parent))

from Partial_Pooling.simulators.priors import (  # noqa: E402
    GLOBAL_PRIOR_STD, LOCAL_PRIOR_STD,
)
from Partial_Pooling.schema import GLOBAL_NAMES, LOCAL_NAMES  # noqa: E402

PRIOR_STD = {
    **{name: float(value) for name, value in zip(GLOBAL_NAMES, GLOBAL_PRIOR_STD)},
    **{name: float(value) for name, value in zip(LOCAL_NAMES, LOCAL_PRIOR_STD)},
}
TARGET_COVERAGE = 0.95
# A calibration regression is "material" when 95% intervals miss their nominal
# level by more than this, averaged over the parameters of a scope. The
# threshold is a decision rule, declared before the numbers are read.
CALIBRATION_TOLERANCE = 0.05


def parse_arm(text):
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--arm wants LABEL:METHOD:ROOT, got {text!r}."
        )
    label, method, root = parts
    return {"label": label, "method": method, "root": root}


def locate(arm):
    root = Path(arm["root"])
    if not root.is_absolute():
        root = ROOT / root
    parent = root / "partial_pooling_recovery" / PRESET
    matches = sorted(
        path for path in parent.glob(f"{arm['method']}-*") if path.is_dir()
    )
    if len(matches) != 1:
        raise SystemExit(
            f"{arm['label']}: expected exactly one {arm['method']!r} run below "
            f"{parent}; found {[p.name for p in matches]}"
        )
    return matches[0]


def _checkpoint_parameters(path):
    if not path.exists():
        return None
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return int(sum(
        value.numel() for value in blob["model_state_dict"].values()
    ))


def checkpoint_metadata(run_config):
    """Parameter count and representation of the model behind a run."""
    tag = run_config.get("checkpoint_tag")
    if not tag:
        return {}
    manifest = (
        ROOT / "artifacts" / "checkpoints" / run_config.get("preset", PRESET)
        / tag / f"{tag}-manifest.json"
    )
    if not manifest.exists():
        return {"checkpoint_tag": tag}
    payload = json.loads(manifest.read_text())
    parameters = payload.get("parameters")
    if parameters is None:
        # Manifests written before the ladder do not record the count; take it
        # from the checkpoint itself rather than leaving the column empty.
        parameters = _checkpoint_parameters(manifest.with_name(payload["checkpoint"]))
    return {
        "checkpoint_tag": tag,
        "parameters": parameters,
        "arch": payload.get("arch", "legacy"),
        "tokenization": payload.get("tokenization", "scalar"),
        "tokens": payload.get("tokens", 100),
        "hidden_size": payload.get("hidden_size"),
        "depth": payload.get("depth"),
        "training_samples": payload.get("training_samples"),
        "validation_loss": (
            min(payload["validation_loss"]) if payload.get("validation_loss") else None
        ),
        "epochs": len(payload.get("train_loss") or []),
    }


def scope_metrics(entries):
    """Standardized error and calibration for one scope (global or local)."""
    standardized_rmse, standardized_mae, coverage, bias = [], [], [], []
    for name, values in sorted(entries.items()):
        scale = PRIOR_STD[name]
        standardized_rmse.append(values["rmse"] / scale)
        standardized_mae.append(values["mae"] / scale)
        coverage.append(values["coverage_95"])
        bias.append(values["mean_posterior_z_error"])
    deviations = [abs(value - TARGET_COVERAGE) for value in coverage]
    return {
        "parameters_reported": len(coverage),
        "standardized_rmse": float(np.mean(standardized_rmse)),
        "standardized_mae": float(np.mean(standardized_mae)),
        "worst_standardized_rmse": float(np.max(standardized_rmse)),
        "mean_coverage_95": float(np.mean(coverage)),
        "mean_abs_coverage_error": float(np.mean(deviations)),
        "worst_coverage_95": float(min(coverage)),
        "mean_abs_posterior_z_bias": float(np.mean(np.abs(bias))),
        "per_parameter": {
            name: {
                "standardized_rmse": values["rmse"] / PRIOR_STD[name],
                "rmse": values["rmse"],
                "coverage_95": values["coverage_95"],
            }
            for name, values in sorted(entries.items())
        },
    }


def read_arm(arm):
    directory = locate(arm)
    summary_path = directory / f"{arm['method']}-summary.json"
    config_path = directory / f"{arm['method']}-run_config.json"
    if not summary_path.exists():
        raise SystemExit(f"{arm['label']}: no summary at {summary_path}")
    summary = json.loads(summary_path.read_text())
    run_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    record = {
        "label": arm["label"],
        "method": arm["method"],
        "run_directory": str(directory),
        "completed_datasets": summary.get("completed_datasets"),
        "subjects": run_config.get("subjects"),
        "draws": run_config.get("draws"),
        "timesteps": run_config.get("timesteps"),
        "sampling_runtime_seconds": summary.get("total_sampling_runtime_seconds"),
        "map_runtime_seconds": summary.get("total_map_runtime_seconds"),
        **checkpoint_metadata(run_config),
    }
    record["global"] = scope_metrics(summary["summary"]["global"])
    record["local"] = scope_metrics(summary["summary"]["local"])
    return record


def verdict(record, baseline):
    """Did this arm beat the baseline under the plan's success criterion?"""
    if baseline is None or record["label"] == baseline["label"]:
        return {}
    calibration = max(
        record["global"]["mean_abs_coverage_error"],
        record["local"]["mean_abs_coverage_error"],
    )
    baseline_calibration = max(
        baseline["global"]["mean_abs_coverage_error"],
        baseline["local"]["mean_abs_coverage_error"],
    )
    parameters = record.get("parameters")
    return {
        "global_rmse_ratio": (
            record["global"]["standardized_rmse"]
            / baseline["global"]["standardized_rmse"]
        ),
        "local_rmse_ratio": (
            record["local"]["standardized_rmse"]
            / baseline["local"]["standardized_rmse"]
        ),
        "better_global": (
            record["global"]["standardized_rmse"]
            < baseline["global"]["standardized_rmse"]
        ),
        "better_local": (
            record["local"]["standardized_rmse"]
            < baseline["local"]["standardized_rmse"]
        ),
        "calibration_regression": (
            calibration > baseline_calibration + CALIBRATION_TOLERANCE
        ),
        "under_parameter_budget": (
            parameters < 1_000_000 if parameters is not None else None
        ),
    }


def _cell(value, spec=""):
    if value is None:
        return "?"
    return format(value, spec)


def render(records, baseline_label):
    by_method = {}
    for record in records:
        by_method.setdefault(record["method"], []).append(record)

    lines = []
    for method, group in by_method.items():
        reference = next(
            (item for item in group if item["label"] == baseline_label), None,
        )
        lines.append("")
        lines.append(f"### {method}")
        lines.append("")
        lines.append(
            "| model | params | tokens | train | global std-RMSE | local std-RMSE "
            "| global cov95 | local cov95 | sampling s | vs baseline |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for record in group:
            record["verdict"] = verdict(record, reference)
            note = []
            if record["verdict"]:
                note.append(
                    f"global x{record['verdict']['global_rmse_ratio']:.3f}"
                )
                note.append(
                    f"local x{record['verdict']['local_rmse_ratio']:.3f}"
                )
                if record["verdict"]["calibration_regression"]:
                    note.append("**calibration regression**")
            else:
                note.append("baseline")
            lines.append(" | ".join([
                f"| {record['label']}",
                _cell(record.get("parameters"), ",d"),
                _cell(record.get("tokens"), "d"),
                _cell(record.get("training_samples"), ",d"),
                f"{record['global']['standardized_rmse']:.4f}",
                f"{record['local']['standardized_rmse']:.4f}",
                f"{record['global']['mean_coverage_95']:.3f}",
                f"{record['local']['mean_coverage_95']:.3f}",
                _cell(record.get("sampling_runtime_seconds"), ",.0f"),
                ", ".join(note) + " |",
            ]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ladder recovery comparison")
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument(
        "--baseline", default=None,
        help="Label of the arm every other arm is measured against.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    records = [read_arm(arm) for arm in args.arm]
    baseline_label = args.baseline or records[0]["label"]
    table = render(records, baseline_label)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ladder_report.json").write_text(json.dumps(
        {"baseline": baseline_label,
         "calibration_tolerance": CALIBRATION_TOLERANCE,
         "prior_std": PRIOR_STD, "arms": records},
        indent=2, sort_keys=True,
    ))
    (args.output / "ladder_report.md").write_text(
        "# Partial-pooling ladder\n\n"
        f"Baseline: `{baseline_label}`. Errors are MAP errors standardized by "
        "the prior standard deviation of each parameter (1.0 = no better than "
        "the prior). Calibration is a constraint, not a score: a regression is "
        f"declared when mean |coverage - 0.95| exceeds the baseline's by more "
        f"than {CALIBRATION_TOLERANCE}.\n" + table + "\n"
    )
    rows = []
    for record in records:
        rows.append({
            "label": record["label"], "method": record["method"],
            "parameters": record.get("parameters"),
            "tokens": record.get("tokens"),
            "training_samples": record.get("training_samples"),
            "completed_datasets": record["completed_datasets"],
            "global_standardized_rmse": record["global"]["standardized_rmse"],
            "local_standardized_rmse": record["local"]["standardized_rmse"],
            "global_mean_coverage_95": record["global"]["mean_coverage_95"],
            "local_mean_coverage_95": record["local"]["mean_coverage_95"],
            "global_mean_abs_coverage_error": record["global"]["mean_abs_coverage_error"],
            "local_mean_abs_coverage_error": record["local"]["mean_abs_coverage_error"],
            "sampling_runtime_seconds": record["sampling_runtime_seconds"],
            **{f"verdict_{key}": value
               for key, value in record.get("verdict", {}).items()},
        })
    fieldnames = sorted({key for row in rows for key in row})
    with (args.output / "ladder_report.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(table)
    print(f"\nWrote {args.output}/ladder_report.{{json,md,csv}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
