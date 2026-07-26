#!/usr/bin/env python3
"""Analytic multi-observation sampler study.

This tutorial uses the exact single-observation score of the linear-Gaussian
problem from ``tests/test_multiobs_analytic.py``.  Consequently any posterior
error comes from score composition or numerical sampling, not score-network
training. It compares two prior/likelihood scale configurations and saves one
observation-scaling figure plus its underlying metrics for each configuration under
``tutorials/output/compositional_score_analytic``.

Run (after reserving a GPU in shared environments):
    python tutorials/compositional_score_analytic.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Set this before importing libraries that may create native thread pools.
CPU_FRACTION = 0.06
CPU_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def limit_cpus() -> None:
    """Limit this process and child processes to at most 6% of host CPUs."""
    total = os.cpu_count() or 1
    limit = int(total * CPU_FRACTION)
    if limit < 1:
        raise RuntimeError(f"6% of {total} logical CPUs is fewer than one CPU.")
    available = sorted(os.sched_getaffinity(0))
    selected = available[:limit]
    if not selected:
        raise RuntimeError("No CPUs are available in the current affinity mask.")
    os.sched_setaffinity(0, selected)
    for variable in CPU_VARS:
        os.environ[variable] = str(len(selected))
    print(f"Using {len(selected)} logical CPU(s) ({100 * len(selected) / total:.1f}% of host capacity).")


limit_cpus()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compass.MultiObsSampler import MultiObsSampler
from compass.Sampler import Sampler
from compass.SDE import VESDE


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "compositional_score_analytic"
N_VALUES = (1, 2, 5, 10, 25, 50, 100, 200)
COLORS = {"DPM + uncorrected": "#0072B2", "DPM + Gaussian": "#56B4E9",
          "Langevin + Gaussian": "#009E73", "Langevin + uncorrected": "#D55E00",
          "Langevin + F-NPSE": "#CC79A7"}


@dataclass(frozen=True)
class Variant:
    label: str
    method: str
    correction: str
    equation: str = "reverse_sde"
    estimate_precision: bool = False
    inference_time_grid: str = "log_sigma"
    corrector_steps: int = 5


VARIANTS = (
    Variant("DPM + uncorrected", "dpm", "uncorrected"),
    Variant("DPM + Gaussian", "dpm", "gauss"),
    Variant("Langevin + Gaussian", "langevin", "gauss", corrector_steps=8),
    Variant("Langevin + uncorrected", "langevin", "uncorrected", corrector_steps=8),
    Variant("Langevin + F-NPSE", "langevin", "fnpe", corrector_steps=8),
)


class MockSBIm:
    """Minimal score-based model wrapper used by the COMPASS samplers."""

    def __init__(self, device: torch.device, mu0: torch.Tensor, sig0: torch.Tensor, sigx: torch.Tensor):
        self.sde = VESDE(sigma=25.0)
        self.sde.sigma = self.sde.sigma.to(device)
        self.model = LinearGaussianScore(self.sde, mu0, sig0, sigx)
        self.sampler = Sampler(self)
        self.nodes_size = 4

    def output_scale_function(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x / self.sde.marginal_prob_std(t).to(x.device)


class LinearGaussianScore(torch.nn.Module):
    """Exact score of p_t(theta | x) for the two-dimensional Gaussian toy problem."""

    def __init__(self, sde: VESDE, mu0: torch.Tensor, sig0: torch.Tensor, sigx: torch.Tensor):
        super().__init__()
        self.sde = sde
        self.register_buffer("mu0", mu0)
        self.register_buffer("sig0", sig0)
        self.register_buffer("sigx", sigx)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, return_attn_weights: bool = False):
        std_t = self.sde.marginal_prob_std(t).to(x.device)
        theta_t, x_obs = x[:, :2], x[:, 2:]
        precision = 1 / self.sig0**2 + 1 / self.sigx**2
        mean = (self.mu0 / self.sig0**2 + x_obs / self.sigx**2) / precision
        score = -(theta_t - mean) / (1 / precision + std_t**2)
        output = torch.zeros_like(x)
        output[:, :2] = std_t * score
        return (output, torch.zeros(1, device=x.device)) if return_attn_weights else output



def posterior(x: torch.Tensor, mu0: torch.Tensor, sig0: torch.Tensor, sigx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact posterior mean and marginal standard deviation for all observations."""
    precision = 1 / sig0**2 + x.shape[0] / sigx**2
    mean = (mu0 / sig0**2 + x.sum(0) / sigx**2) / precision
    return mean, torch.sqrt(1 / precision)


def seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_problem(n: int, seed_value: int, device: torch.device, mu0: torch.Tensor, sig0: torch.Tensor, sigx: torch.Tensor):
    seed(seed_value)
    theta_true = mu0 + sig0 * torch.randn(2, device=device)
    observations = theta_true + sigx * torch.randn(n, 2, device=device)
    return theta_true, observations, posterior(observations, mu0, sig0, sigx)


def sample_multi(observations: torch.Tensor, variant: Variant, seed_value: int, device: torch.device,
                 mu0: torch.Tensor, sig0: torch.Tensor, sigx: torch.Tensor, samples: int, timesteps: int) -> tuple[torch.Tensor, float]:
    seed(seed_value)
    sbim = MockSBIm(device, mu0, sig0, sigx)
    precision = None
    if variant.correction == "gauss" and not variant.estimate_precision:
        # Validation occurs before MultiObsSampler moves its prior to CUDA, so
        # supplied precision metadata must remain on CPU at this boundary.
        precision = (1 / sig0**2 + 1 / sigx**2).repeat(observations.shape[0], 1).cpu()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    values = MultiObsSampler(sbim).sample(
        world_size=1, data=observations, condition_mask=torch.tensor([0., 0., 1., 1.], device=device),
        timesteps=timesteps, num_samples=samples, hierarchy=[0, 1],
        prior=(mu0.cpu(), sig0.cpu()), correction=variant.correction,
        posterior_precision=precision, precision_est_samples=min(500, samples), method=variant.method,
        equation=variant.equation, inference_time_grid=variant.inference_time_grid,
        corrector_steps=variant.corrector_steps, device=device, verbose=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    return values[0, :, :2], runtime


def metric_row(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, runtime: float) -> dict[str, float]:
    normalized_error = (values.mean(0) - mean) / std
    return {"mean_error_sigma": float(torch.linalg.vector_norm(normalized_error)),
            "std_ratio": float((values.std(0) / std).mean()), "runtime_seconds": runtime}


def benchmark(variants: tuple[Variant, ...], device: torch.device, mu0: torch.Tensor, sig0: torch.Tensor,
              sigx: torch.Tensor, samples: int, timesteps: int, repeats: int) -> list[dict]:
    rows: list[dict] = []
    for n in N_VALUES:
        for repeat in range(repeats):
            _, observations, (mean, std) = make_problem(n, 1000 + 101 * n + repeat, device, mu0, sig0, sigx)
            for index, variant in enumerate(variants):
                draws, runtime = sample_multi(observations, variant, 2000 + 101 * n + 10 * repeat + index,
                                               device, mu0, sig0, sigx, samples, timesteps)
                rows.append({"variant": variant.label, "n_observations": n, "repeat": repeat,
                             **metric_row(draws, mean, std, runtime)})
    return rows



def summarise(rows: list[dict], label: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(N_VALUES)
    values = [np.asarray([row[metric] for row in rows if row["variant"] == label and row["n_observations"] == n]) for n in xs]
    return xs, np.asarray([item.mean() for item in values]), np.asarray([item.std(ddof=0) for item in values])



def style() -> None:
    mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 220, "font.size": 10, "axes.titlesize": 12,
                         "axes.grid": True, "grid.alpha": .22, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.fontsize": 8})


def finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_scaling(rows: list[dict], output: Path, filename: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes = axes.ravel()
    for variant in VARIANTS:
        color = COLORS[variant.label]
        for ax, metric in zip(axes[:3], ("mean_error_sigma", "std_ratio", "runtime_seconds")):
            xs, ys, spread = summarise(rows, variant.label, metric)
            ax.plot(xs, ys, "o-", color=color, label=variant.label)
            ax.fill_between(xs, ys - spread, ys + spread, color=color, alpha=.13)
        xs, errors, _ = summarise(rows, variant.label, "mean_error_sigma")
        _, runtime, _ = summarise(rows, variant.label, "runtime_seconds")
        axes[3].plot(runtime, errors, "o-", color=color, label=variant.label)
        for n_observations, x, y in zip(xs, runtime, errors):
            axes[3].annotate(str(n_observations), (x, y), xytext=(3, 3),
                             textcoords="offset points", fontsize=7, color=color)
    axes[0].set(title="Posterior-centre accuracy",
                ylabel="Euclidean normalized mean error")
    axes[1].set(title="Posterior-width accuracy", ylabel="sample std / analytic std")
    axes[1].axhline(1, color="black", ls="--", lw=1)
    axes[2].set(title="End-to-end sampling cost", ylabel="inference time (s)", yscale="log")
    axes[3].set(title="Accuracy-cost trade-off (labels show N)",
                xlabel="inference time (s)", ylabel="Euclidean normalized mean error",
                xscale="log", yscale="log")
    for ax in axes[:3]:
        ax.set(xscale="log", xlabel="observations N")
    axes[0].legend(loc="upper left")
    fig.suptitle(title)
    finish(fig, output / filename)



def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reserve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    try:
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=1)
    except ImportError as exc:
        raise RuntimeError("autocvd is required before this GPU tutorial runs.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable to PyTorch.")
    return torch.device("cuda")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-samples", type=int, default=3_000)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.num_samples < 2 or args.repeats < 1:
        raise ValueError("--num-samples must be at least 2 and --repeats at least 1.")

    device = reserve_device(args.device)
    style()
    args.output.mkdir(parents=True, exist_ok=True)
    mu0 = torch.tensor([-2.3, -2.89], device=device)
    configurations = (
        ("prior1p0_likelihood1p0", 1.0, 1.0),
        ("prior0p3_likelihood0p5", 0.3, 0.5),
    )

    for config_index, (name, prior_std, likelihood_std) in enumerate(configurations):
        sig0 = torch.full((2,), prior_std, device=device)
        sigx = torch.full((2,), likelihood_std, device=device)

        # Warm up lazy CUDA/PyTorch initialization outside measured regions.
        if device.type == "cuda":
            _, warmup_observations, _ = make_problem(
                1, 999 + 100 * config_index, device, mu0, sig0, sigx
            )
            for index, variant in enumerate(VARIANTS):
                sample_multi(
                    warmup_observations, variant, 998 + 100 * config_index + index,
                    device, mu0, sig0, sigx, min(64, args.num_samples), 5,
                )

        rows = benchmark(
            VARIANTS, device, mu0, sig0, sigx,
            args.num_samples, args.timesteps, args.repeats,
        )
        for row in rows:
            row["prior_std"] = prior_std
            row["likelihood_std"] = likelihood_std

        write_rows(rows, args.output / f"sampler_metrics_{name}.csv")
        np.savez_compressed(
            args.output / f"raw_metrics_{name}.npz",
            rows=np.array(rows, dtype=object),
        )
        plot_scaling(
            rows,
            args.output,
            f"samplers_vs_observation_count_{name}.png",
            f"Prior std={prior_std:g}, likelihood std={likelihood_std:g}",
        )

    print(f"Saved two analytic compositional-score plots to {args.output}")


if __name__ == "__main__":
    main()
