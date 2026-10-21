#!/usr/bin/env python3
"""Validate ordinary posterior inference when all four arm parameters are local.

Every row is an independent prior-predictive pair (theta_i, x_i). No hierarchy,
shared latent, MultiObsSampler, or score composition is used. The standard
single-observation posterior is run separately for every row.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit

configure_cpu_limit()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import kstest  # noqa: E402

import hierarchical as H  # noqa: E402
import simulator as sim  # noqa: E402
import train_model as trainer  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = {
    "compact_tokenized": ROOT / (
        "artifacts/models/tokenized_h32d3_n200000/Model_checkpoint.pt"
    ),
    "wide_legacy": ROOT / (
        "artifacts/models/legacy_h64d4wide_n200000/Model_checkpoint.pt"
    ),
}


def parse_checkpoint(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def weighted_quantiles(values, weights, probabilities):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(weights)
    cumulative = (cumulative - 0.5 * weights) / cumulative[-1]
    return np.interp(probabilities, cumulative, values)


def abc_comparison(samples, observations, reference_theta, reference_x,
                   bandwidth, case_indices):
    rows = []
    probabilities = (
        np.arange(samples.shape[1], dtype=np.float64) + 0.5
    ) / samples.shape[1]
    for index in case_indices:
        squared = np.sum(
            (reference_x - observations[index][None, :]) ** 2, axis=1
        )
        log_weight = -squared / (2.0 * bandwidth**2)
        log_weight -= log_weight.max()
        weight = np.exp(log_weight)
        weight /= weight.sum()
        ess = 1.0 / np.sum(weight**2)
        row = {"observation_index": int(index), "abc_ess": float(ess)}
        for parameter, name in enumerate(H.NODE_NAMES[:4]):
            reference_values = reference_theta[:, parameter]
            reference_mean = float(np.sum(weight * reference_values))
            reference_sd = float(np.sqrt(np.sum(
                weight * (reference_values - reference_mean) ** 2
            )))
            reference_q = weighted_quantiles(
                reference_values, weight, probabilities
            )
            sample_q = np.quantile(samples[index, :, parameter], probabilities)
            w1 = float(np.mean(np.abs(sample_q - reference_q)))
            row[name] = {
                "w1": w1,
                "w1_over_reference_sd": w1 / max(reference_sd, 1e-12),
                "sample_mean": float(samples[index, :, parameter].mean()),
                "reference_mean": reference_mean,
                "sample_sd": float(samples[index, :, parameter].std(ddof=1)),
                "reference_sd": reference_sd,
            }
        angle_weight = float(np.sum(weight[reference_theta[:, 0] > 0]))
        sample_angle_weight = float(np.mean(samples[index, :, 0] > 0))
        row["angle_1_positive_mass"] = {
            "sample": sample_angle_weight,
            "reference": angle_weight,
            "absolute_error": abs(sample_angle_weight - angle_weight),
        }
        rows.append(row)
    return rows


def posterior_metrics(samples, truth, observations):
    ranks = np.mean(samples < truth[:, None, :], axis=1)
    summary = {}
    rank_rows = []
    for parameter, name in enumerate(H.NODE_NAMES[:4]):
        parameter_ranks = ranks[:, parameter]
        posterior_mean = samples[:, :, parameter].mean(axis=1)
        posterior_sd = samples[:, :, parameter].std(axis=1, ddof=1)
        ks_result = kstest(parameter_ranks, "uniform")
        rmse = float(np.sqrt(np.mean(
            (posterior_mean - truth[:, parameter]) ** 2
        )))
        entry = {
            "mean_rank": float(parameter_ranks.mean()),
            "rank_sd": float(parameter_ranks.std(ddof=1)),
            "rank_ks_statistic": float(ks_result.statistic),
            "rank_ks_pvalue": float(ks_result.pvalue),
            "posterior_mean_rmse": rmse,
            "posterior_mean_rmse_over_prior_sd": rmse / H.PRIOR_STD[parameter],
            "median_posterior_sd": float(np.median(posterior_sd)),
            "median_posterior_sd_over_prior_sd": float(
                np.median(posterior_sd) / H.PRIOR_STD[parameter]
            ),
        }
        for mass in (0.50, 0.80, 0.95):
            tail = (1.0 - mass) / 2.0
            lower, upper = np.quantile(
                samples[:, :, parameter], [tail, 1.0 - tail], axis=1
            )
            entry[f"coverage_{int(100 * mass)}"] = float(np.mean(
                (lower <= truth[:, parameter])
                & (truth[:, parameter] <= upper)
            ))
        summary[name] = entry
        for observation, rank in enumerate(parameter_ranks):
            rank_rows.append({
                "observation": observation,
                "parameter": name,
                "rank_fraction": float(rank),
            })

    predicted = H.forward(samples.reshape(-1, 4)).reshape(
        samples.shape[0], samples.shape[1], 2
    )
    distance = np.linalg.norm(
        predicted - observations[:, None, :], axis=-1
    )
    endpoint = {
        "mean_distance": float(distance.mean()),
        "median_distance": float(np.median(distance)),
        "p90_distance": float(np.quantile(distance, 0.90)),
        "fraction_within_0.05": float(np.mean(distance < 0.05)),
        "fraction_within_0.10": float(np.mean(distance < 0.10)),
    }
    return summary, endpoint, rank_rows


def sample_model(model, observations, args):
    chunks = []
    started = time.perf_counter()
    for start in range(0, len(observations), args.observation_batch):
        x = torch.as_tensor(
            observations[start:start + args.observation_batch],
            dtype=torch.float32,
        )
        torch.manual_seed(args.seed + start)
        draws = model.sample(
            x=x, num_samples=args.posterior_draws, timesteps=args.timesteps,
            method="dpm", order=2, churn_eta=args.churn_eta,
            corrector_steps=0, final_corrector_steps=0,
            device=args.device, verbose=False,
            terminal_denoise=args.terminal_denoise,
        )
        chunks.append(draws.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0), time.perf_counter() - started


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, default=None,
        help="NAME=PATH; repeat to compare checkpoints.",
    )
    result.add_argument("--observations", type=int, default=48)
    result.add_argument("--posterior-draws", type=int, default=512)
    result.add_argument("--observation-batch", type=int, default=8)
    result.add_argument("--timesteps", type=int, default=80)
    # Pure churn: the corrector sampler is no longer this repository's recipe.
    result.add_argument("--churn-eta", type=float, default=2.0,
                        help="Karras churn strength (compass.churn); "
                             "0 is the stock deterministic predictor.")
    result.add_argument(
        "--terminal-denoise", action=argparse.BooleanOptionalAction,
        default=True,
        help="Take the final probability-flow step from eps to zero noise.",
    )
    result.add_argument("--abc-cases", type=int, default=6)
    result.add_argument("--abc-prior-draws", type=int, default=400_000)
    result.add_argument("--abc-bandwidth", type=float, default=0.05)
    result.add_argument("--seed", type=int, default=812)
    result.add_argument("--device", default="cuda")
    result.add_argument(
        "--artifacts", type=Path,
        default=ROOT / "artifacts/all_local_inference",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.device.startswith("cuda"):
        # autocvd blocks until a GPU is *idle*, which on a shared host can be
        # never. An explicit CUDA_VISIBLE_DEVICES is a deliberate choice of
        # device and must win over the search.
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            print("CUDA_VISIBLE_DEVICES="
                  f"{os.environ['CUDA_VISIBLE_DEVICES']} set; skipping autocvd.")
        else:
            from autocvd import autocvd
            autocvd(num_gpus=1, interval=1)
        args.device = "cuda"

    checkpoints = dict(args.checkpoint or DEFAULT_CHECKPOINTS.items())
    truth_tensor, x_tensor = H.simulate(
        args.observations, sim.generator("all-local-test", args.seed)
    )
    truth = truth_tensor.numpy()
    observations = x_tensor.numpy()

    reference_theta_tensor, reference_x_tensor = H.simulate(
        args.abc_prior_draws,
        sim.generator("all-local-abc-reference", args.seed),
    )
    reference_theta = reference_theta_tensor.numpy()
    reference_x = reference_x_tensor.numpy()
    case_indices = np.linspace(
        0, args.observations - 1,
        min(args.abc_cases, args.observations), dtype=int,
    )

    result = {
        "experiment": "all parameters local; ordinary non-compositional sample",
        "observations": args.observations,
        "posterior_draws": args.posterior_draws,
        "timesteps": args.timesteps,
        "churn_eta": float(args.churn_eta),
        "terminal_denoise": args.terminal_denoise,
        "seed": args.seed,
        "abc": {
            "prior_draws": args.abc_prior_draws,
            "bandwidth": args.abc_bandwidth,
            "case_indices": case_indices.tolist(),
        },
        "models": {},
    }
    all_samples = {}
    all_rank_rows = []
    for name, checkpoint in checkpoints.items():
        model = trainer.load_to(checkpoint, args.device)
        samples, seconds = sample_model(model, observations, args)
        parameter_metrics, endpoint_metrics, rank_rows = posterior_metrics(
            samples, truth, observations
        )
        abc_rows = abc_comparison(
            samples, observations, reference_theta, reference_x,
            args.abc_bandwidth, case_indices,
        )
        result["models"][name] = {
            "checkpoint": str(checkpoint),
            "seconds": seconds,
            "parameters": parameter_metrics,
            "posterior_predictive_endpoint": endpoint_metrics,
            "abc_comparison": abc_rows,
        }
        for row in rank_rows:
            all_rank_rows.append({"model": name, **row})
        all_samples[name] = samples
        print(
            f"{name}: {seconds:.1f}s, endpoint median "
            f"{endpoint_metrics['median_distance']:.3f}",
            flush=True,
        )

    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    with (args.artifacts / "sbc_ranks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rank_rows[0]))
        writer.writeheader()
        writer.writerows(all_rank_rows)
    np.savez_compressed(
        args.artifacts / "samples.npz",
        truth=truth, observations=observations,
        **{f"samples_{name}": values
           for name, values in all_samples.items()},
    )
    print(f"Wrote {args.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
