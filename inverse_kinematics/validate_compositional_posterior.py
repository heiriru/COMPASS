#!/usr/bin/env python3
"""Lightweight held-out validation of the learned bimodal composition.

This is deliberately a pilot, not a publication-scale benchmark: two held-out
bimodal datasets, one nested observation ladder, six prior-predictive SBC
replicates, and a small ABC bandwidth/draw sensitivity grid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit

configure_cpu_limit()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hierarchical as H  # noqa: E402
import run_inference as inference  # noqa: E402
import simulator as sim  # noqa: E402
import train_model as trainer  # noqa: E402
from bimodality_study import bimodality, selected_dataset  # noqa: E402
from run_bimodal import kde_shape  # noqa: E402

ROOT = Path(__file__).resolve().parent
METHODS = tuple(inference.METHODS)


def branch_stats(draws):
    draws = np.asarray(draws, dtype=np.float64)
    result = {"fraction_positive": float(np.mean(draws > 0))}
    for label, selected in (
        ("negative", draws[draws < 0]), ("positive", draws[draws >= 0])
    ):
        result[f"{label}_count"] = int(len(selected))
        result[f"{label}_sd"] = (
            float(selected.std(ddof=1)) if len(selected) > 1 else float("nan")
        )
    return result


def draw(model, x, method, beta, seed, samples, pilot_draws, device):
    args = SimpleNamespace(
        seed=seed, draws=samples, pilot_draws=pilot_draws,
        denoise_clamp=5.0, churn_eta=2.0, jacobian_refresh=5,
        composition_temperature=beta,
    )
    values, _, seconds = inference.draw(
        model, torch.as_tensor(x, dtype=torch.float32), method, args, device
    )
    return values[0, :, H.HIERARCHY[0]].astype(np.float64), seconds


def reference_metrics(x, draws, bandwidth, seed):
    grid = np.linspace(-2.0, 2.0, 801)
    _, density = H.reference_global(
        x, grid=grid, draws=draws, bandwidth=bandwidth, seed=seed
    )
    mean, std, mode = H.grid_moments(grid, density)
    shape = bimodality(grid, density)
    positive = float(np.trapezoid(
        density[grid >= 0], grid[grid >= 0]
    ))
    branch_sd = {}
    for label, keep in (("negative", grid < 0), ("positive", grid >= 0)):
        branch_grid = grid[keep]
        branch_density = density[keep]
        mass = float(np.trapezoid(branch_density, branch_grid))
        branch_mean = float(
            np.trapezoid(branch_grid * branch_density, branch_grid) / mass
        )
        branch_variance = float(np.trapezoid(
            (branch_grid - branch_mean) ** 2 * branch_density, branch_grid
        ) / mass)
        branch_sd[f"{label}_sd"] = float(np.sqrt(max(branch_variance, 0.0)))
    return grid, density, {
        "mean": mean, "sd": std, "mode": mode,
        "fraction_positive": positive, **branch_sd, **shape,
    }


def sample_metrics(values, grid, density, reference_sd):
    _, shape = kde_shape(values, grid)
    return {
        "w1_over_reference_sd": inference.wasserstein_to_grid(
            values, grid, density
        ) / reference_sd,
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "modes": shape["modes"],
        "dip_ratio": shape["dip_ratio"],
        **branch_stats(values),
    }


def central_coverage(values, truth, mass):
    tail = (1.0 - mass) / 2.0
    lo, hi = np.quantile(values, [tail, 1.0 - tail])
    return bool(lo <= truth <= hi)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / (
            "artifacts/models/bernoulli_focus002_tokenized_h32d3_n100000/"
            "Model_checkpoint.pt"
        ),
    )
    result.add_argument("--beta", type=float, default=0.4)
    result.add_argument("--draws", type=int, default=384)
    result.add_argument("--pilot-draws", type=int, default=192)
    result.add_argument("--timesteps", type=int, default=50)
    result.add_argument("--reference-draws", type=int, default=30_000)
    result.add_argument("--sbc-replicates", type=int, default=6)
    result.add_argument("--device", default="cuda")
    result.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/compositional_posterior_validation.json",
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
    inference.COMMON["timesteps"] = args.timesteps
    model = trainer.load_to(args.checkpoint, args.device)
    output = {
        "checkpoint": str(args.checkpoint),
        "beta": args.beta,
        "methods": list(METHODS),
        "heldout": [],
        "contraction": [],
        "sbc": {},
        "reference_sensitivity": [],
    }

    # Two datasets not used by either training stream or beta calibration.
    for seed in (101, 103):
        x = selected_dataset(
            8, 0.6, 0.15, sim.generator("bimodal-dataset", seed)
        )
        grid, density, ref = reference_metrics(
            x, args.reference_draws, 0.05, seed
        )
        for method in METHODS:
            values, seconds = draw(
                model, x, method, args.beta, seed, args.draws,
                args.pilot_draws, args.device,
            )
            output["heldout"].append({
                "seed": seed, "method": method, "seconds": seconds,
                "reference": ref,
                **sample_metrics(values, grid, density, ref["sd"]),
            })
            print(f"heldout seed={seed} {method}", flush=True)

    # Nested selected-data ladder; report within-branch SD because total SD of
    # a separating bimodal posterior need not shrink.
    for count in (1, 2, 4, 8, 16):
        seed = 211
        x = selected_dataset(
            count, 0.6, 0.15, sim.generator("bimodal-dataset", seed)
        )
        grid, density, ref = reference_metrics(
            x, args.reference_draws, 0.05, seed
        )
        for method in METHODS:
            values, seconds = draw(
                model, x, method, args.beta, seed, args.draws,
                args.pilot_draws, args.device,
            )
            output["contraction"].append({
                "observations": count, "method": method, "seconds": seconds,
                "reference": ref,
                **sample_metrics(values, grid, density, ref["sd"]),
            })
            print(f"contraction n={count} {method}", flush=True)

    # Prior-predictive pilot SBC. Six ranks are enough to catch gross bias, not
    # enough to claim uniformity; coverage is reported descriptively.
    sbc_rows = {method: [] for method in METHODS}
    for replicate in range(args.sbc_replicates):
        seed = 500 + replicate
        truth, _, x = H.simulate_dataset(
            4, sim.generator("pilot-sbc", seed)
        )
        for method in METHODS:
            values, seconds = draw(
                model, x, method, args.beta, seed, 256,
                args.pilot_draws, args.device,
            )
            sbc_rows[method].append({
                "replicate": replicate,
                "truth": truth,
                "rank_fraction": float(np.mean(values < truth)),
                "coverage_50": central_coverage(values, truth, 0.50),
                "coverage_80": central_coverage(values, truth, 0.80),
                "coverage_95": central_coverage(values, truth, 0.95),
                "seconds": seconds,
            })
        print(f"SBC replicate {replicate + 1}/{args.sbc_replicates}", flush=True)
    for method, rows in sbc_rows.items():
        ranks = [row["rank_fraction"] for row in rows]
        output["sbc"][method] = {
            "replicates": len(rows),
            "mean_rank": float(np.mean(ranks)),
            "rank_sd": float(np.std(ranks, ddof=1)),
            "coverage_50": float(np.mean([r["coverage_50"] for r in rows])),
            "coverage_80": float(np.mean([r["coverage_80"] for r in rows])),
            "coverage_95": float(np.mean([r["coverage_95"] for r in rows])),
            "rows": rows,
        }

    # ABC reference stability on the original selected seed.
    x = selected_dataset(
        8, 0.6, 0.15, sim.generator("bimodal-dataset", 99)
    )
    for reference_draws in (20_000, 40_000):
        for bandwidth in (0.04, 0.05, 0.07):
            _, _, metrics = reference_metrics(
                x, reference_draws, bandwidth, 99
            )
            output["reference_sensitivity"].append({
                "draws": reference_draws,
                "bandwidth": bandwidth,
                **metrics,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
