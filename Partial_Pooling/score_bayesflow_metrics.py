#!/usr/bin/env python3
"""Score COMPASS recovery runs with BayesFlow's own metric definitions.

`bayesflow-org/diffusion-experiments` case_study4 -- the partial-pooling case
study of Arruda et al., "Diffusion Models in Simulation-Based Inference: A
Tutorial Review" (arXiv:2512.20685) -- reports

    NRMSE                = bayesflow.diagnostics.metrics.root_mean_squared_error
    calibration error    = bayesflow.diagnostics.metrics.calibration_error

against a pinned ``bayesflow==2.0.13``. Our own reports use a different summary
(MAP error standardized by the prior standard deviation, plus 95% coverage), so
the two are not comparable as printed. This script recomputes our runs in *their*
metric, transcribing the 2.0.13 implementations exactly:

* ``root_mean_squared_error``: the RMSE is taken between every posterior *draw*
  and the ground truth -- not between the posterior mean and the ground truth --
  then divided by a normalizer and aggregated with ``np.median`` across datasets.
  The 2.0.13 default ``normalize="prior"`` divides by the RMSE a *prior-only*
  predictor would achieve, bootstrapped from the empirical ground truths, so
  0 means "point mass at the truth" and 1 means "no better than the prior".
  ``normalize="range"`` (the pre-2.0.13 default) divides by max - min instead;
  both are reported because the paper's figure does not say which it used.
* ``calibration_error``: the median over 20 credible levels spanning
  [0.005, 0.995] of |empirical coverage - nominal coverage|.

Caveat that no amount of metric alignment removes: these are *our* test datasets
and *our* trained networks. The paper publishes case study 4 as a figure
(`plots/pooling_metrics.pdf`) built from pickles that are not in the repository,
so its numeric values cannot be read off and are not reproduced here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit(3)

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from Partial_Pooling.schema import GLOBAL_NAMES, LOCAL_NAMES  # noqa: E402

PRESET = "small"
BOOTSTRAP_SEED = 20250718
BOOTSTRAP_REPEATS = 64


def nrmse(estimates, targets, normalize="prior", aggregation=np.median,
          rng=None):
    """bayesflow 2.0.13 ``root_mean_squared_error``, transcribed.

    ``estimates``: (num_datasets, num_draws, num_variables)
    ``targets``:   (num_datasets, num_variables)
    """
    err = estimates - targets[:, None, :]
    rmse = np.sqrt(np.mean(err**2, axis=1))

    if normalize in (None, False):
        normalizer = np.array(1.0)
    elif normalize == "range":
        normalizer = targets.max(axis=0) - targets.min(axis=0)
    elif normalize == "std":
        normalizer = np.std(targets, axis=0, ddof=0)
    elif normalize == "prior":
        # The 2.0.13 default draws one bootstrap; it is stochastic, so average
        # the normalizer over repeats to keep the reported number stable.
        count, draws, _ = estimates.shape
        rng = rng or np.random.default_rng(BOOTSTRAP_SEED)
        samples = []
        for _ in range(BOOTSTRAP_REPEATS):
            index = rng.integers(0, count, size=(count, draws))
            prior_err = targets[index] - targets[:, None, :]
            samples.append(aggregation(
                np.sqrt(np.mean(prior_err**2, axis=1)), axis=0,
            ))
        normalizer = np.mean(samples, axis=0)
    else:
        raise ValueError(f"Unknown normalization mode: {normalize}")

    return aggregation(rmse / normalizer, axis=0)


def calibration_error(estimates, targets, resolution=20, aggregation=np.median,
                      min_quantile=0.005, max_quantile=0.995):
    """bayesflow 2.0.13 ``calibration_error``, transcribed."""
    alphas = np.linspace(start=min_quantile, stop=max_quantile, num=resolution)
    regions = 1 - alphas
    lowers = regions / 2
    uppers = 1 - lowers
    quantiles = np.quantile(estimates, [lowers, uppers], axis=1)
    lower_bounds, upper_bounds = quantiles[0], quantiles[1]
    inlier = np.logical_and(
        lower_bounds <= targets[None, ...], upper_bounds >= targets[None, ...],
    )
    alpha_pred = np.mean(inlier, axis=1)
    return aggregation(np.abs(alpha_pred - alphas[:, None]), axis=0)


def coverage(estimates, targets, level=0.95):
    """Empirical coverage of the central ``level`` credible interval.

    ``calibration_error`` takes an absolute value, so it cannot say whether a
    posterior is too wide or too narrow. Over-coverage is the direction that
    matters for the group scales: sigma is identified from the spread of the
    subject-level estimates, so systematically wide subject posteriors inflate
    it. This reports the signed quantity.
    """
    lower = (1 - level) / 2
    bounds = np.quantile(estimates, [lower, 1 - lower], axis=1)
    inside = np.logical_and(bounds[0] <= targets, bounds[1] >= targets)
    return inside.mean(axis=0)


def interval_width(estimates, targets, level=0.95):
    """Mean central-interval width, in units of the spread of the truths."""
    lower = (1 - level) / 2
    bounds = np.quantile(estimates, [lower, 1 - lower], axis=1)
    return (bounds[1] - bounds[0]).mean(axis=0) / np.std(targets, axis=0, ddof=0)


def median_correlation(estimates, targets):
    """Pearson r between the posterior median and the truth, per variable.

    This is what case_study4's recovery figure reports for the *global*
    parameters -- ``bayesflow.diagnostics.plots.recovery`` defaults to
    ``point_agg=np.median`` and prints ``r`` in each panel. The paper publishes
    no global NRMSE, so this is the only global quantity of theirs that can be
    compared against at all.
    """
    point = np.median(estimates, axis=1)
    return np.array([
        np.corrcoef(point[:, i], targets[:, i])[0, 1]
        for i in range(targets.shape[1])
    ])


def load_run(root, method, limit=None):
    directory = Path(root)
    if not directory.is_absolute():
        directory = ROOT / directory
    parent = directory / "partial_pooling_recovery" / PRESET
    matches = sorted(p for p in parent.glob(f"{method}-*") if p.is_dir())
    if len(matches) != 1:
        raise SystemExit(
            f"expected one {method!r} run below {parent}, found "
            f"{[p.name for p in matches]}"
        )
    run = matches[0]

    global_draws, global_truth, local_draws, local_truth = [], [], [], []
    # Dataset files are named by index, so truncating a sorted listing takes the
    # *same* test datasets from every arm -- which is what makes a partially
    # finished run comparable with a finished one.
    paths = sorted(run.glob(f"{method}-dataset-*.pt"),
                   key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    for path in paths[:limit] if limit else paths:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        global_draws.append(blob["posterior"]["globals"].numpy())
        global_truth.append(blob["truth_globals"].numpy())
        # (draws, subjects, 3) -> one "dataset" per subject, as the paper's
        # local model is evaluated per subject.
        locals_ = blob["posterior"]["locals"].numpy()
        local_draws.append(np.transpose(locals_, (1, 0, 2)))
        local_truth.append(blob["truth_locals"].numpy())

    if not global_draws:
        raise SystemExit(f"no completed datasets in {run}")
    return {
        "run": str(run),
        "datasets": len(global_draws),
        "global_estimates": np.stack(global_draws),
        "global_targets": np.stack(global_truth),
        "local_estimates": np.concatenate(local_draws),
        "local_targets": np.concatenate(local_truth),
    }


def score(payload):
    result = {"datasets": payload["datasets"],
              "subject_units": int(payload["local_estimates"].shape[0])}
    for scope, names in (("global", GLOBAL_NAMES), ("local", LOCAL_NAMES)):
        estimates = payload[f"{scope}_estimates"]
        targets = payload[f"{scope}_targets"]
        entry = {}
        for label, mode in (("nrmse_prior", "prior"), ("nrmse_range", "range")):
            values = nrmse(estimates, targets, normalize=mode)
            entry[label] = {
                "per_parameter": dict(zip(names, map(float, values))),
                "mean": float(np.mean(values)),
            }
        correlations = median_correlation(estimates, targets)
        entry["median_correlation"] = {
            "per_parameter": dict(zip(names, map(float, correlations))),
            "mean": float(np.mean(correlations)),
        }
        errors = calibration_error(estimates, targets)
        entry["calibration_error"] = {
            "per_parameter": dict(zip(names, map(float, errors))),
            "mean": float(np.mean(errors)),
        }
        for label, values in (
            ("coverage_95", coverage(estimates, targets, 0.95)),
            ("width_95", interval_width(estimates, targets, 0.95)),
        ):
            entry[label] = {
                "per_parameter": dict(zip(names, map(float, values))),
                "mean": float(np.mean(values)),
            }
        result[scope] = entry
    return result


def parse_arm(text):
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--arm wants LABEL:METHOD:ROOT, got {text!r}")
    return dict(zip(("label", "method", "root"), parts))


def main(argv=None):
    parser = argparse.ArgumentParser(description="BayesFlow-metric scoring")
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--datasets", type=int,
        help=(
            "Score only the first N test datasets. Applied to every arm, so a "
            "run still in progress can be compared against a finished one on "
            "exactly the datasets both have."
        ),
    )
    args = parser.parse_args(argv)

    records = []
    for arm in args.arm:
        payload = load_run(arm["root"], arm["method"], limit=args.datasets)
        record = {"label": arm["label"], "method": arm["method"], **score(payload)}
        records.append(record)

    header = (
        f"{'model':22s} {'method':26s} {'g NRMSE':>9s} {'g NRMSE':>9s} {'g ECE':>7s} "
        f"{'l NRMSE':>9s} {'l NRMSE':>9s} {'l ECE':>7s} {'g cov95':>8s} "
        f"{'l cov95':>8s} {'l w95':>7s}"
    )
    print(header)
    print(f"{'':22s} {'':26s} {'(prior)':>9s} {'(range)':>9s} {'':>7s} "
          f"{'(prior)':>9s} {'(range)':>9s} {'':>7s}")
    for record in records:
        g, l = record["global"], record["local"]
        print(
            f"{record['label'][:22]:22s} {record['method'][:26]:26s} "
            f"{g['nrmse_prior']['mean']:9.4f} {g['nrmse_range']['mean']:9.4f} "
            f"{g['calibration_error']['mean']:7.4f} "
            f"{l['nrmse_prior']['mean']:9.4f} {l['nrmse_range']['mean']:9.4f} "
            f"{l['calibration_error']['mean']:7.4f} "
            f"{g['coverage_95']['mean']:8.4f} {l['coverage_95']['mean']:8.4f} "
            f"{l['width_95']['mean']:7.3f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "metric_source": "bayesflow==2.0.13 diagnostics.metrics",
            "reference": "arXiv:2512.20685 case_study4",
            "note": (
                "Computed on COMPASS test datasets and COMPASS checkpoints. "
                "The paper's own values are published only as a figure and are "
                "not reproduced here."
            ),
            "arms": records,
        }, indent=2, sort_keys=True))
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
