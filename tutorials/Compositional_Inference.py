#!/usr/bin/env python3
"""Validation experiments for compositional (multi-observation) inference.

This is the script counterpart of ``Compositional_Inference.ipynb``.  It is
deliberately an experiment suite rather than a linear notebook export: each
experiment has a quantitative reference, writes a CSV table, saves every raw
plot input in a compressed ``raw_plot_data.npz`` archive, and writes one or more
plots under ``tutorials/output/compositional_inference``.

The suite answers six questions:

1. Does the multi-observation API preserve shapes, synchronize shared samples,
   agree with N=1 inference, and recover a known Gaussian posterior?
2. How do DPM and annealed Langevin perform with Gaussian-corrected and raw
   composition?  F-NPSE/Langevin is included as its method-native baseline.
3. How do those sampler variants scale as the number of observations grows?
4. What is the effect of uniform, log-sigma, and mixture training-time sampling?
5. Does the best setting (mixture + DPM + Gaussian correction) improve over
   separate single-observation posteriors, analytic truth, and MCMC?
6. Are global and local parameters handled correctly in two hierarchical tests:
   (a) unknown global Normal centre and scale and (b) a global offset plus one
   local offset per observation?

No experiment is run merely by importing this module.  Typical invocations are:

    python tutorials/Compositional_Inference.py --experiments all
    python tutorials/Compositional_Inference.py --experiments contract,samplers
    python tutorials/Compositional_Inference.py --quick --experiments all
    python tutorials/Compositional_Inference.py --plots-only --experiments all

``--quick`` is only a wiring smoke test; its small networks and chains are not
large enough for scientific conclusions. Existing checkpoints are reused.
``--plots-only`` reads the saved CSV/NPZ files and never initializes a model,
trains, samples, or runs MCMC.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Callable, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm, wasserstein_distance
import torch

try:
    from autocvd import autocvd
except ImportError:
    autocvd = None

from compass import ScoreBasedInferenceModel as SBIm


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "compositional_inference"

# Shared Gaussian toy problem from the notebook.
MU0, S0, SX = 0.0, 1.0, 1.0

# Shared + local Gaussian hierarchy.
MUG, S0G, S0L, SXH = 0.0, 1.0, 1.0, 0.5

# Population model: x_i ~ Normal(mu, variance), with Gaussian priors on mu/log variance.
POP_MU_MEAN, POP_MU_STD = 0.0, 1.5
POP_LOGV_MEAN, POP_LOGV_STD = math.log(0.7**2), 0.7

TIME_SAMPLINGS = ("uniform", "log_sigma", "mixture")
N_VALUES = (1, 2, 5, 10, 25, 50)


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    device: str
    seed: int = 7
    train_samples: int = 50_000
    validation_samples: int = 3_000
    max_epochs: int = 80
    patience: int = 15
    batch_size: int = 256
    posterior_samples: int = 3_000
    timesteps: int = 100
    repeats: int = 3
    mcmc_steps: int = 40_000
    mcmc_burn: int = 8_000
    mcmc_thin: int = 8


@dataclass(frozen=True)
class SamplerVariant:
    key: str
    label: str
    method: str
    correction: str
    corrector_steps: int
    snr: float = 0.1


SAMPLER_VARIANTS = (
    SamplerVariant("dpm_gauss", "DPM + Gaussian", "dpm", "gauss", 5),
    SamplerVariant("dpm_raw", "DPM + uncorrected", "dpm", "uncorrected", 5),
    SamplerVariant("langevin_gauss", "Langevin + Gaussian", "langevin", "gauss", 8),
    SamplerVariant("langevin_raw", "Langevin + uncorrected", "langevin", "uncorrected", 8),
    # F-NPSE is a bridging-density score and is valid only with annealed Langevin.
    SamplerVariant("langevin_fnpe", "Langevin + F-NPSE", "langevin", "fnpe", 8),
)


def configure_plot_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "legend.fontsize": 8,
    })


def select_device(requested: str) -> str:
    """Reserve one GPU before Torch work, or explicitly fall back to CPU."""
    if requested == "cpu":
        return "cpu"
    if autocvd is None:
        print("autocvd is unavailable; using CPU.")
        return "cpu"
    try:
        selected = autocvd(num_gpus=1, interval=1, timeout=30, progress=False)
    except (OSError, TimeoutError, subprocess.SubprocessError):
        print("No GPU became free within 30 seconds; using CPU.")
        return "cpu"
    if torch.cuda.is_available():
        print(f"Reserved GPU {selected[0]} with autocvd.")
        return "cuda"
    print("autocvd selected a GPU but CUDA is unavailable to Torch; using CPU.")
    return "cpu"


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_raw_data(path: Path, **arrays: object) -> None:
    """Save all plot inputs in a portable, compressed NumPy archive.

    Torch tensors are detached and moved to CPU. Each archive is intentionally
    self-contained so plotting can be redone without loading a COMPASS model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    converted = {}
    for name, value in arrays.items():
        if isinstance(value, torch.Tensor):
            value = tensor_numpy(value)
        converted[name] = np.asarray(value)
    np.savez_compressed(path, **converted)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def simulate_shared(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    theta = MU0 + S0 * torch.randn(n, 1)
    x = theta + SX * torch.randn(n, 1)
    return theta, x


def shared_observations(n: int, theta_true: float) -> torch.Tensor:
    return theta_true + SX * torch.randn(n, 1)


def analytic_shared(x: torch.Tensor) -> tuple[float, float]:
    precision = 1.0 / S0**2 + x.shape[0] / SX**2
    mean = (MU0 / S0**2 + x.sum().item() / SX**2) / precision
    return mean, precision**-0.5


def simulate_hierarchical_pairs(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    global_theta = MUG + S0G * torch.randn(n, 1)
    local_theta = S0L * torch.randn(n, 1)
    x = global_theta + local_theta + SXH * torch.randn(n, 1)
    return torch.cat([global_theta, local_theta], dim=1), x


def simulate_population_pairs(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    mu = POP_MU_MEAN + POP_MU_STD * torch.randn(n, 1)
    log_variance = POP_LOGV_MEAN + POP_LOGV_STD * torch.randn(n, 1)
    x = mu + torch.exp(0.5 * log_variance) * torch.randn(n, 1)
    return torch.cat([mu, log_variance], dim=1), x


def build_model(nodes: int, device: str, hierarchical: bool = False) -> SBIm:
    return SBIm(
        nodes_size=nodes,
        sde_type="vesde",
        sigma=8.0,
        hidden_size=32,
        depth=3 if hierarchical else 2,
        num_heads=4,
        mlp_ratio=2,
        device=device,
    )


def load_or_train(
    cfg: RunConfig,
    key: str,
    nodes: int,
    simulator: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    time_sampling: str = "mixture",
    hierarchical: bool = False,
) -> SBIm:
    model_dir = cfg.output_dir / "models" / key
    checkpoint = model_dir / "Model_checkpoint.pt"
    if checkpoint.exists():
        print(f"Loading {key}: {checkpoint}")
        return SBIm.load(str(checkpoint), device=cfg.device)

    seed_all(cfg.seed)
    theta_train, x_train = simulator(cfg.train_samples)
    theta_val, x_val = simulator(cfg.validation_samples)
    model = build_model(nodes, cfg.device, hierarchical=hierarchical)
    print(f"Training {key} with time_sampling={time_sampling!r} on {cfg.device}...")
    started = time.perf_counter()
    model.train(
        theta=theta_train,
        x=x_train,
        theta_val=theta_val,
        x_val=x_val,
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.patience,
        time_sampling=time_sampling,
        device=cfg.device,
        verbose=False,
        path=str(model_dir),
    )
    print(f"  trained in {(time.perf_counter() - started) / 60:.1f} minutes")
    return model


def known_shared_precision(n: int = 1) -> torch.Tensor:
    value = 1.0 / S0**2 + 1.0 / SX**2
    return torch.full((n, 1), value)


def sample_shared(
    model: SBIm,
    x: torch.Tensor,
    cfg: RunConfig,
    variant: SamplerVariant,
    seed: int,
    automatic_precision: bool = False,
) -> tuple[torch.Tensor, float]:
    seed_all(seed)
    started = time.perf_counter()
    samples = model.sample(
        x=x,
        multi_obs_inference=True,
        hierarchy=[0],
        prior=([MU0], [S0]),
        correction=variant.correction,
        posterior_precision=(
            None if automatic_precision or variant.correction != "gauss"
            else known_shared_precision(x.shape[0])
        ),
        precision_est_samples=min(750, cfg.posterior_samples),
        num_samples=cfg.posterior_samples,
        timesteps=cfg.timesteps,
        method=variant.method,
        corrector_steps=variant.corrector_steps,
        snr=variant.snr,
        device=cfg.device,
        verbose=False,
    )
    return samples, time.perf_counter() - started


def scalar_metrics(samples: torch.Tensor, truth_mean: float, truth_std: float) -> dict:
    values = tensor_numpy(samples).reshape(-1)
    return {
        "sample_mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "mean_error_sigma": float(abs(values.mean() - truth_mean) / truth_std),
        "std_ratio": float(values.std(ddof=1) / truth_std),
    }


def random_walk_metropolis(
    log_prob: Callable[[np.ndarray], float],
    initial: Sequence[float],
    proposal_scale: Sequence[float],
    steps: int,
    burn: int,
    thin: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Small dependency-free MCMC reference; reports acceptance for diagnostics."""
    rng = np.random.default_rng(seed)
    state = np.asarray(initial, dtype=float)
    scale = np.asarray(proposal_scale, dtype=float)
    current = float(log_prob(state))
    kept: list[np.ndarray] = []
    accepted = 0
    for step in range(steps):
        proposal = state + scale * rng.normal(size=state.shape)
        candidate = float(log_prob(proposal))
        if math.log(rng.random()) < candidate - current:
            state, current = proposal, candidate
            accepted += 1
        if step >= burn and (step - burn) % thin == 0:
            kept.append(state.copy())
    return np.asarray(kept), accepted / steps


def shared_mcmc(x: torch.Tensor, cfg: RunConfig, seed: int) -> tuple[np.ndarray, float]:
    observations = tensor_numpy(x).reshape(-1)

    def log_prob(theta: np.ndarray) -> float:
        return -0.5 * ((theta[0] - MU0) / S0) ** 2 \
            - 0.5 * np.sum(((observations - theta[0]) / SX) ** 2)

    mean, std = analytic_shared(x)
    return random_walk_metropolis(
        log_prob, [mean], [0.8 * std], cfg.mcmc_steps,
        cfg.mcmc_burn, cfg.mcmc_thin, seed,
    )


def experiment_contract(model: SBIm, cfg: RunConfig) -> None:
    """API invariants plus analytic moment checks, including auto precision."""
    out = cfg.output_dir / "01_contract"
    seed_all(cfg.seed + 10)
    theta_true = float(MU0 + S0 * torch.randn(()))
    x = shared_observations(8, theta_true)
    truth_mean, truth_std = analytic_shared(x)
    variant = SAMPLER_VARIANTS[0]
    samples, runtime = sample_shared(
        model, x, cfg, variant, cfg.seed + 11, automatic_precision=True,
    )

    # Shared dimensions must be sample-wise identical across observation rows.
    synchronization_error = float((samples[:, :, 0] - samples[0:1, :, 0]).abs().max())
    metrics = scalar_metrics(samples[0, :, 0], truth_mean, truth_std)
    metrics.update({
        "n_observations": x.shape[0],
        "shape": str(tuple(samples.shape)),
        "shared_synchronization_max_abs": synchronization_error,
        "runtime_seconds": runtime,
        "truth_mean": truth_mean,
        "truth_std": truth_std,
    })

    # At N=1, compositional and ordinary inference target the same posterior.
    x_one = x[:1]
    seed_all(cfg.seed + 12)
    ordinary = model.sample(
        x=x_one, num_samples=cfg.posterior_samples, timesteps=cfg.timesteps,
        method="dpm", device=cfg.device, verbose=False,
    )[0, :, 0]
    composed_one, _ = sample_shared(model, x_one, cfg, variant, cfg.seed + 13)
    metrics["n1_wasserstein_ordinary_vs_composed"] = wasserstein_distance(
        tensor_numpy(ordinary), tensor_numpy(composed_one[0, :, 0]),
    )
    write_rows(out / "contract_metrics.csv", [metrics])
    save_raw_data(
        out / "raw_plot_data.npz",
        x_observed=x,
        theta_true=theta_true,
        analytic_mean=truth_mean,
        analytic_std=truth_std,
        multi_samples=samples,
        n1_ordinary_samples=ordinary,
        n1_composed_samples=composed_one,
    )

    grid = np.linspace(truth_mean - 4 * truth_std, truth_mean + 4 * truth_std, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(tensor_numpy(samples[0, :, 0]), bins=50, density=True, alpha=0.6,
                 label="multi-observation")
    axes[0].plot(grid, norm.pdf(grid, truth_mean, truth_std), "k--", label="analytic")
    axes[0].axvline(theta_true, color="tab:red", label="true theta")
    axes[0].set(title="N=8 analytic recovery", xlabel="theta", ylabel="density")
    axes[0].legend()
    axes[1].hist(tensor_numpy(ordinary), bins=45, density=True, alpha=0.5,
                 label="ordinary N=1")
    axes[1].hist(tensor_numpy(composed_one[0, :, 0]), bins=45, density=True, alpha=0.5,
                 label="composed N=1")
    axes[1].set(title="N=1 API equivalence", xlabel="theta", ylabel="density")
    axes[1].legend()
    save_figure(fig, out / "contract_checks.png")

    # Fail loudly only for structural invariants; statistical quality stays in CSV.
    if samples.shape != (x.shape[0], cfg.posterior_samples, 1):
        raise AssertionError(f"Unexpected multi-observation shape: {samples.shape}")
    if synchronization_error > 1e-6:
        raise AssertionError(f"Shared samples are not synchronized: {synchronization_error}")


def evaluate_sampler_grid(
    model: SBIm,
    cfg: RunConfig,
    n_values: Iterable[int],
    repeats: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    raw: dict[str, np.ndarray] = {}
    max_n = max(n_values)
    for repeat in range(repeats):
        seed_all(cfg.seed + 100 * repeat)
        theta_true = float(MU0 + S0 * torch.randn(()))
        pool = shared_observations(max_n, theta_true)
        raw[f"x_pool_r{repeat}"] = tensor_numpy(pool)
        raw[f"theta_true_r{repeat}"] = np.asarray(theta_true)
        for n in n_values:
            x = pool[:n]
            truth_mean, truth_std = analytic_shared(x)
            for index, variant in enumerate(SAMPLER_VARIANTS):
                samples, runtime = sample_shared(
                    model, x, cfg, variant,
                    seed=cfg.seed + 10_000 * repeat + 100 * n + index,
                )
                row = scalar_metrics(samples[0, :, 0], truth_mean, truth_std)
                row.update({
                    "repeat": repeat,
                    "n_observations": n,
                    "variant": variant.key,
                    "label": variant.label,
                    "method": variant.method,
                    "correction": variant.correction,
                    "runtime_seconds": runtime,
                    "truth_mean": truth_mean,
                    "truth_std": truth_std,
                    "theta_true": theta_true,
                })
                rows.append(row)
                raw[f"samples_r{repeat}_n{n}_{variant.key}"] = tensor_numpy(samples[0, :, 0])
                raw[f"analytic_r{repeat}_n{n}"] = np.asarray([truth_mean, truth_std])
                print(f"N={n:>2} {variant.label:<24} "
                      f"mean={row['mean_error_sigma']:.2f} sigma "
                      f"width={row['std_ratio']:.2f} time={runtime:.1f}s")
    return rows, raw


def aggregate(rows: Sequence[dict], group_keys: Sequence[str], value: str) -> dict:
    grouped: dict[tuple, list[float]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(float(row[value]))
    return {key: (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            for key, values in grouped.items()}


def experiment_samplers(model: SBIm, cfg: RunConfig) -> None:
    out = cfg.output_dir / "02_sampler_comparison"
    rows, raw = evaluate_sampler_grid(model, cfg, n_values=(10,), repeats=cfg.repeats)
    write_rows(out / "sampler_metrics.csv", rows)
    save_raw_data(out / "raw_plot_data.npz", **raw)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    labels = [variant.label for variant in SAMPLER_VARIANTS]
    x_positions = np.arange(len(labels))
    for ax, key, title in zip(
        axes,
        ("mean_error_sigma", "std_ratio", "runtime_seconds"),
        ("Mean error / analytic std", "Sample std / analytic std", "Runtime (seconds)"),
    ):
        means = [np.mean([r[key] for r in rows if r["variant"] == v.key])
                 for v in SAMPLER_VARIANTS]
        errors = [np.std([r[key] for r in rows if r["variant"] == v.key], ddof=1)
                  if cfg.repeats > 1 else 0.0 for v in SAMPLER_VARIANTS]
        ax.bar(x_positions, means, yerr=errors, capsize=3)
        ax.set(title=title, xticks=x_positions, xticklabels=labels)
        ax.tick_params(axis="x", rotation=35)
    axes[1].axhline(1.0, color="black", linestyle="--")
    save_figure(fig, out / "sampler_comparison_n10.png")


def experiment_observation_scaling(model: SBIm, cfg: RunConfig) -> None:
    out = cfg.output_dir / "03_observation_scaling"
    rows, raw = evaluate_sampler_grid(model, cfg, n_values=N_VALUES, repeats=cfg.repeats)
    write_rows(out / "observation_scaling_metrics.csv", rows)
    save_raw_data(out / "raw_plot_data.npz", **raw)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for variant in SAMPLER_VARIANTS:
        chosen = [row for row in rows if row["variant"] == variant.key]
        for ax, value in zip(axes, ("mean_error_sigma", "std_ratio", "runtime_seconds")):
            summary = aggregate(chosen, ("n_observations",), value)
            xs = np.array(N_VALUES)
            ys = np.array([summary[(n,)][0] for n in N_VALUES])
            es = np.array([summary[(n,)][1] for n in N_VALUES])
            ax.plot(xs, ys, marker="o", label=variant.label)
            ax.fill_between(xs, ys - es, ys + es, alpha=0.12)
    axes[0].set(xscale="log", xlabel="observations N", ylabel="mean error / analytic std")
    axes[1].set(xscale="log", xlabel="observations N", ylabel="sample std / analytic std")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[2].set(xscale="log", yscale="log", xlabel="observations N", ylabel="runtime (s)")
    axes[0].legend()
    save_figure(fig, out / "samplers_vs_observation_count.png")


def experiment_time_sampling(cfg: RunConfig) -> None:
    out = cfg.output_dir / "04_time_sampling"
    models = {
        scheme: load_or_train(
            cfg, f"shared_{scheme}", 2, simulate_shared, time_sampling=scheme,
        )
        for scheme in TIME_SAMPLINGS
    }
    rows: list[dict] = []
    raw: dict[str, np.ndarray] = {}
    variant = SAMPLER_VARIANTS[0]
    for repeat in range(cfg.repeats):
        seed_all(cfg.seed + 500 + repeat)
        theta_true = float(MU0 + S0 * torch.randn(()))
        pool = shared_observations(max(N_VALUES), theta_true)
        raw[f"x_pool_r{repeat}"] = tensor_numpy(pool)
        raw[f"theta_true_r{repeat}"] = np.asarray(theta_true)
        for n in N_VALUES:
            x = pool[:n]
            truth_mean, truth_std = analytic_shared(x)
            for index, (scheme, model) in enumerate(models.items()):
                samples, runtime = sample_shared(
                    model, x, cfg, variant,
                    cfg.seed + 30_000 * repeat + 100 * n + index,
                )
                row = scalar_metrics(samples[0, :, 0], truth_mean, truth_std)
                row.update({"repeat": repeat, "n_observations": n,
                            "time_sampling": scheme, "runtime_seconds": runtime})
                rows.append(row)
                raw[f"samples_r{repeat}_n{n}_{scheme}"] = tensor_numpy(samples[0, :, 0])
                raw[f"analytic_r{repeat}_n{n}"] = np.asarray([truth_mean, truth_std])
    write_rows(out / "time_sampling_metrics.csv", rows)
    save_raw_data(out / "raw_plot_data.npz", **raw)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for scheme in TIME_SAMPLINGS:
        chosen = [row for row in rows if row["time_sampling"] == scheme]
        for ax, value in zip(axes, ("mean_error_sigma", "std_ratio")):
            summary = aggregate(chosen, ("n_observations",), value)
            ys = [summary[(n,)][0] for n in N_VALUES]
            ax.plot(N_VALUES, ys, marker="o", label=scheme)
    axes[0].set(xscale="log", xlabel="observations N", ylabel="mean error / analytic std")
    axes[1].set(xscale="log", xlabel="observations N", ylabel="sample std / analytic std")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[0].legend(title="training time sampling")
    save_figure(fig, out / "training_time_sampling_effect.png")


def experiment_multi_vs_individual(model: SBIm, cfg: RunConfig) -> None:
    out = cfg.output_dir / "05_multi_vs_individual"
    seed_all(cfg.seed + 700)
    n = 10
    theta_true = float(MU0 + S0 * torch.randn(()))
    x = shared_observations(n, theta_true)
    truth_mean, truth_std = analytic_shared(x)
    multi, multi_runtime = sample_shared(model, x, cfg, SAMPLER_VARIANTS[0], cfg.seed + 701)
    multi_values = tensor_numpy(multi[0, :, 0])

    individual_values: list[np.ndarray] = []
    individual_rows: list[dict] = []
    started = time.perf_counter()
    for index in range(n):
        seed_all(cfg.seed + 800 + index)
        samples = model.sample(
            x=x[index:index + 1], num_samples=cfg.posterior_samples,
            timesteps=cfg.timesteps, method="dpm", device=cfg.device, verbose=False,
        )[0, :, 0]
        values = tensor_numpy(samples)
        individual_values.append(values)
        single_mean, single_std = analytic_shared(x[index:index + 1])
        individual_rows.append({
            "observation": index,
            "x": float(x[index]),
            **scalar_metrics(samples, single_mean, single_std),
        })
    individual_runtime = time.perf_counter() - started
    mcmc, acceptance = shared_mcmc(x, cfg, cfg.seed + 900)
    mcmc_values = mcmc[:, 0]

    analytic_draws = np.random.default_rng(cfg.seed).normal(
        truth_mean, truth_std, size=max(cfg.posterior_samples, len(mcmc_values)),
    )
    summary = [{
        "n_observations": n,
        "multi_mean_error_sigma": abs(multi_values.mean() - truth_mean) / truth_std,
        "multi_std_ratio": multi_values.std(ddof=1) / truth_std,
        "multi_wasserstein_to_analytic": wasserstein_distance(multi_values, analytic_draws),
        "multi_wasserstein_to_mcmc": wasserstein_distance(multi_values, mcmc_values),
        "pooled_individual_wasserstein_to_analytic": wasserstein_distance(
            np.concatenate(individual_values), analytic_draws,
        ),
        "multi_runtime_seconds": multi_runtime,
        "all_individual_runtime_seconds": individual_runtime,
        "mcmc_acceptance": acceptance,
    }]
    write_rows(out / "comparison_summary.csv", summary)
    write_rows(out / "individual_posterior_metrics.csv", individual_rows)
    save_raw_data(
        out / "raw_plot_data.npz",
        x_observed=x,
        theta_true=theta_true,
        analytic_mean=truth_mean,
        analytic_std=truth_std,
        multi_samples=multi_values,
        individual_samples=np.stack(individual_values),
        mcmc_samples=mcmc_values,
        analytic_reference_draws=analytic_draws,
        mcmc_acceptance=acceptance,
    )

    grid = np.linspace(truth_mean - 5 * truth_std, truth_mean + 5 * truth_std, 500)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for index, values in enumerate(individual_values):
        kde = gaussian_kde(values)
        ax.plot(grid, kde(grid), color="0.7", alpha=0.55,
                label="individual posteriors" if index == 0 else None)
    ax.plot(grid, norm.pdf(grid, truth_mean, truth_std), "k--", linewidth=2.4,
            label="analytic multi-observation")
    ax.plot(grid, gaussian_kde(mcmc_values)(grid), color="tab:orange", label="MCMC")
    ax.plot(grid, gaussian_kde(multi_values)(grid), color="tab:blue", linewidth=2.4,
            label="COMPASS multi-observation")
    ax.axvline(theta_true, color="tab:red", label="true theta")
    ax.set(xlabel="theta", ylabel="density", title="Multi-observation vs separate inference")
    ax.legend()
    save_figure(fig, out / "multi_vs_individual_vs_references.png")


def population_log_prob(parameters: np.ndarray, observations: np.ndarray) -> float:
    mu, log_variance = parameters
    if not -10.0 < log_variance < 6.0:
        return -math.inf
    variance = math.exp(log_variance)
    prior = -0.5 * ((mu - POP_MU_MEAN) / POP_MU_STD) ** 2 \
        - 0.5 * ((log_variance - POP_LOGV_MEAN) / POP_LOGV_STD) ** 2
    likelihood = -0.5 * len(observations) * log_variance \
        - 0.5 * np.sum((observations - mu) ** 2 / variance)
    return float(prior + likelihood)


def experiment_population_globals(cfg: RunConfig) -> None:
    """Infer which Normal population generated the data: global centre and scale."""
    out = cfg.output_dir / "06a_population_globals"
    model = load_or_train(
        cfg, "population_mu_logvariance_mixture", 3, simulate_population_pairs,
        time_sampling="mixture", hierarchical=True,
    )
    seed_all(cfg.seed + 1_000)
    n = 30
    truth = np.array([0.65, math.log(0.45**2)])
    x = truth[0] + math.exp(0.5 * truth[1]) * torch.randn(n, 1)
    observations = tensor_numpy(x).reshape(-1)
    mcmc, acceptance = random_walk_metropolis(
        lambda value: population_log_prob(value, observations),
        initial=[float(np.mean(observations)), math.log(float(np.var(observations)))],
        proposal_scale=[0.08, 0.10],
        steps=cfg.mcmc_steps,
        burn=cfg.mcmc_burn,
        thin=cfg.mcmc_thin,
        seed=cfg.seed + 1_001,
    )
    seed_all(cfg.seed + 1_002)
    started = time.perf_counter()
    samples = model.sample(
        x=x, multi_obs_inference=True, hierarchy=[0, 1],
        prior=([POP_MU_MEAN, POP_LOGV_MEAN], [POP_MU_STD, POP_LOGV_STD]),
        # Estimate each single-observation posterior precision. Reusing the
        # multi-observation MCMC covariance here would leak the reference answer
        # into COMPASS and would also count the N observations twice.
        correction="gauss", posterior_precision=None,
        precision_est_samples=min(750, cfg.posterior_samples),
        num_samples=cfg.posterior_samples, timesteps=cfg.timesteps,
        method="dpm", corrector_steps_interval=1, corrector_steps=10, snr=0.2,
        device=cfg.device, verbose=False,
    )
    runtime = time.perf_counter() - started
    compass_values = tensor_numpy(samples[0, :, :2])
    rows = []
    for index, name in enumerate(("mu", "log_variance")):
        rows.append({
            "parameter": name,
            "truth": truth[index],
            "compass_mean": compass_values[:, index].mean(),
            "compass_std": compass_values[:, index].std(ddof=1),
            "mcmc_mean": mcmc[:, index].mean(),
            "mcmc_std": mcmc[:, index].std(ddof=1),
            "wasserstein_to_mcmc": wasserstein_distance(
                compass_values[:, index], mcmc[:, index],
            ),
            "mcmc_acceptance": acceptance,
            "runtime_seconds": runtime,
        })
    write_rows(out / "population_global_metrics.csv", rows)
    save_raw_data(
        out / "raw_plot_data.npz",
        x_observed=x,
        truth_mu_logvariance=truth,
        compass_samples_mu_logvariance=compass_values,
        mcmc_samples_mu_logvariance=mcmc,
        mcmc_acceptance=acceptance,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for index, (name, truth_value) in enumerate(zip(("mu", "log variance"), truth)):
        axes[index].hist(mcmc[:, index], bins=45, density=True, alpha=0.45, label="MCMC")
        axes[index].hist(compass_values[:, index], bins=45, density=True, alpha=0.45,
                         label="COMPASS")
        axes[index].axvline(truth_value, color="tab:red", label="truth")
        axes[index].set(xlabel=name, ylabel="density")
    axes[0].legend()
    axes[2].scatter(mcmc[:, 0], np.exp(mcmc[:, 1]), s=5, alpha=0.15, label="MCMC")
    axes[2].scatter(compass_values[:, 0], np.exp(compass_values[:, 1]), s=5,
                    alpha=0.15, label="COMPASS")
    axes[2].scatter([truth[0]], [math.exp(truth[1])], marker="*", s=130,
                    color="tab:red", label="truth")
    axes[2].set(xlabel="population centre mu", ylabel="population variance")
    axes[2].legend()
    save_figure(fig, out / "population_mu_variance_vs_mcmc.png")


def exact_hierarchical_posterior(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Joint posterior for [global, local_1, ..., local_N]."""
    observations = tensor_numpy(x).reshape(-1)
    n = len(observations)
    precision = np.zeros((n + 1, n + 1))
    precision[0, 0] = 1.0 / S0G**2 + n / SXH**2
    precision[1:, 1:] = np.eye(n) * (1.0 / S0L**2 + 1.0 / SXH**2)
    precision[0, 1:] = 1.0 / SXH**2
    precision[1:, 0] = 1.0 / SXH**2
    natural = np.zeros(n + 1)
    natural[0] = MUG / S0G**2 + observations.sum() / SXH**2
    natural[1:] = observations / SXH**2
    covariance = np.linalg.inv(precision)
    return covariance @ natural, covariance


def experiment_shared_local(cfg: RunConfig) -> None:
    out = cfg.output_dir / "06b_shared_local"
    model = load_or_train(
        cfg, "shared_local_mixture", 3, simulate_hierarchical_pairs,
        time_sampling="mixture", hierarchical=True,
    )
    seed_all(cfg.seed + 1_200)
    n = 30
    global_true = float(MUG + S0G * torch.randn(()))
    local_true = S0L * torch.randn(n)
    x = global_true + local_true[:, None] + SXH * torch.randn(n, 1)
    exact_mean, exact_covariance = exact_hierarchical_posterior(x)
    per_observation_global_precision = 1.0 / S0G**2 + 1.0 / (S0L**2 + SXH**2)
    seed_all(cfg.seed + 1_201)
    samples = model.sample(
        x=x, multi_obs_inference=True, hierarchy=[0], prior=([MUG], [S0G]),
        correction="gauss",
        posterior_precision=torch.full((n, 1), per_observation_global_precision),
        num_samples=cfg.posterior_samples, timesteps=cfg.timesteps,
        method="dpm", corrector_steps_interval=1, corrector_steps=10, snr=0.2,
        device=cfg.device, verbose=False,
    )
    global_samples = tensor_numpy(samples[0, :, 0])
    local_samples = tensor_numpy(samples[:, :, 1])
    synchronization_error = float((samples[:, :, 0] - samples[0:1, :, 0]).abs().max())
    rows = [{
        "parameter": "global",
        "index": -1,
        "truth": global_true,
        "analytic_mean": exact_mean[0],
        "analytic_std": math.sqrt(exact_covariance[0, 0]),
        "compass_mean": global_samples.mean(),
        "compass_std": global_samples.std(ddof=1),
        "shared_synchronization_max_abs": synchronization_error,
    }]
    for index in range(n):
        rows.append({
            "parameter": "local",
            "index": index,
            "truth": float(local_true[index]),
            "analytic_mean": exact_mean[index + 1],
            "analytic_std": math.sqrt(exact_covariance[index + 1, index + 1]),
            "compass_mean": local_samples[index].mean(),
            "compass_std": local_samples[index].std(ddof=1),
            "shared_synchronization_max_abs": synchronization_error,
        })
    write_rows(out / "shared_local_metrics.csv", rows)
    save_raw_data(
        out / "raw_plot_data.npz",
        x_observed=x,
        global_truth=global_true,
        local_truth=local_true,
        exact_joint_mean=exact_mean,
        exact_joint_covariance=exact_covariance,
        compass_global_samples=global_samples,
        compass_local_samples=local_samples,
        shared_synchronization_max_abs=synchronization_error,
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    grid = np.linspace(exact_mean[0] - 4 * math.sqrt(exact_covariance[0, 0]),
                       exact_mean[0] + 4 * math.sqrt(exact_covariance[0, 0]), 400)
    axes[0].hist(global_samples, bins=50, density=True, alpha=0.6, label="COMPASS")
    axes[0].plot(grid, norm.pdf(grid, exact_mean[0], math.sqrt(exact_covariance[0, 0])),
                 "k--", label="analytic")
    axes[0].axvline(global_true, color="tab:red", label="truth")
    axes[0].set(xlabel="global parameter", ylabel="density", title="Shared posterior")
    axes[0].legend()
    indices = np.arange(n)
    local_means = local_samples.mean(axis=1)
    local_stds = local_samples.std(axis=1, ddof=1)
    axes[1].errorbar(indices, local_means, yerr=local_stds, fmt="o", ms=3,
                     label="COMPASS mean +/- std")
    axes[1].plot(indices, exact_mean[1:], "k_", ms=8, label="analytic mean")
    axes[1].scatter(indices, tensor_numpy(local_true), marker="x", color="tab:red",
                    label="true local")
    axes[1].set(xlabel="observation", ylabel="local parameter", title="Local posteriors")
    axes[1].legend()
    axes[2].scatter(exact_mean[1:], local_means, c=indices, cmap="viridis")
    lo = min(exact_mean[1:].min(), local_means.min())
    hi = max(exact_mean[1:].max(), local_means.max())
    axes[2].plot([lo, hi], [lo, hi], "k--")
    axes[2].set(xlabel="analytic local mean", ylabel="COMPASS local mean",
                title="Local-parameter recovery")
    save_figure(fig, out / "shared_and_local_recovery.png")

    if synchronization_error > 1e-6:
        raise AssertionError(f"Shared hierarchical samples are not synchronized: {synchronization_error}")


def read_metric_rows(path: Path) -> list[dict]:
    """Load a metric CSV, converting numeric fields back to floats."""
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            converted = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def load_raw_data(path: Path) -> dict[str, np.ndarray]:
    """Load a self-contained plot archive without requiring Torch or a model."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def replot_contract(output_dir: Path) -> None:
    out = output_dir / "01_contract"
    raw = load_raw_data(out / "raw_plot_data.npz")
    mean, std = float(raw["analytic_mean"]), float(raw["analytic_std"])
    grid = np.linspace(mean - 4 * std, mean + 4 * std, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(raw["multi_samples"][0, :, 0], bins=50, density=True, alpha=0.6,
                 label="multi-observation")
    axes[0].plot(grid, norm.pdf(grid, mean, std), "k--", label="analytic")
    axes[0].axvline(float(raw["theta_true"]), color="tab:red", label="true theta")
    axes[0].set(title="N=8 analytic recovery", xlabel="theta", ylabel="density")
    axes[0].legend()
    axes[1].hist(raw["n1_ordinary_samples"], bins=45, density=True, alpha=0.5,
                 label="ordinary N=1")
    axes[1].hist(raw["n1_composed_samples"][0, :, 0], bins=45, density=True, alpha=0.5,
                 label="composed N=1")
    axes[1].set(title="N=1 API equivalence", xlabel="theta", ylabel="density")
    axes[1].legend()
    save_figure(fig, out / "contract_checks.png")


def replot_samplers(output_dir: Path) -> None:
    out = output_dir / "02_sampler_comparison"
    rows = read_metric_rows(out / "sampler_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    labels = [variant.label for variant in SAMPLER_VARIANTS]
    positions = np.arange(len(labels))
    for ax, key, title in zip(
        axes, ("mean_error_sigma", "std_ratio", "runtime_seconds"),
        ("Mean error / analytic std", "Sample std / analytic std", "Runtime (seconds)"),
    ):
        groups = [[r[key] for r in rows if r["variant"] == variant.key]
                  for variant in SAMPLER_VARIANTS]
        means = [np.mean(values) for values in groups]
        errors = [np.std(values, ddof=1) if len(values) > 1 else 0.0 for values in groups]
        ax.bar(positions, means, yerr=errors, capsize=3)
        ax.set(title=title, xticks=positions, xticklabels=labels)
        ax.tick_params(axis="x", rotation=35)
    axes[1].axhline(1.0, color="black", linestyle="--")
    save_figure(fig, out / "sampler_comparison_n10.png")


def replot_scaling(output_dir: Path) -> None:
    out = output_dir / "03_observation_scaling"
    rows = read_metric_rows(out / "observation_scaling_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for variant in SAMPLER_VARIANTS:
        chosen = [row for row in rows if row["variant"] == variant.key]
        for ax, value in zip(axes, ("mean_error_sigma", "std_ratio", "runtime_seconds")):
            summary = aggregate(chosen, ("n_observations",), value)
            xs = np.asarray(N_VALUES, dtype=float)
            ys = np.asarray([summary[(float(n),)][0] for n in N_VALUES])
            errors = np.asarray([summary[(float(n),)][1] for n in N_VALUES])
            ax.plot(xs, ys, marker="o", label=variant.label)
            ax.fill_between(xs, ys - errors, ys + errors, alpha=0.12)
    axes[0].set(xscale="log", xlabel="observations N", ylabel="mean error / analytic std")
    axes[1].set(xscale="log", xlabel="observations N", ylabel="sample std / analytic std")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[2].set(xscale="log", yscale="log", xlabel="observations N", ylabel="runtime (s)")
    axes[0].legend()
    save_figure(fig, out / "samplers_vs_observation_count.png")


def replot_time_sampling(output_dir: Path) -> None:
    out = output_dir / "04_time_sampling"
    rows = read_metric_rows(out / "time_sampling_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for scheme in TIME_SAMPLINGS:
        chosen = [row for row in rows if row["time_sampling"] == scheme]
        for ax, value in zip(axes, ("mean_error_sigma", "std_ratio")):
            summary = aggregate(chosen, ("n_observations",), value)
            ys = [summary[(float(n),)][0] for n in N_VALUES]
            ax.plot(N_VALUES, ys, marker="o", label=scheme)
    axes[0].set(xscale="log", xlabel="observations N", ylabel="mean error / analytic std")
    axes[1].set(xscale="log", xlabel="observations N", ylabel="sample std / analytic std")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[0].legend(title="training time sampling")
    save_figure(fig, out / "training_time_sampling_effect.png")


def replot_individual(output_dir: Path) -> None:
    out = output_dir / "05_multi_vs_individual"
    raw = load_raw_data(out / "raw_plot_data.npz")
    mean, std = float(raw["analytic_mean"]), float(raw["analytic_std"])
    grid = np.linspace(mean - 5 * std, mean + 5 * std, 500)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for index, values in enumerate(raw["individual_samples"]):
        ax.plot(grid, gaussian_kde(values)(grid), color="0.7", alpha=0.55,
                label="individual posteriors" if index == 0 else None)
    ax.plot(grid, norm.pdf(grid, mean, std), "k--", linewidth=2.4,
            label="analytic multi-observation")
    ax.plot(grid, gaussian_kde(raw["mcmc_samples"])(grid), color="tab:orange", label="MCMC")
    ax.plot(grid, gaussian_kde(raw["multi_samples"])(grid), color="tab:blue", linewidth=2.4,
            label="COMPASS multi-observation")
    ax.axvline(float(raw["theta_true"]), color="tab:red", label="true theta")
    ax.set(xlabel="theta", ylabel="density", title="Multi-observation vs separate inference")
    ax.legend()
    save_figure(fig, out / "multi_vs_individual_vs_references.png")


def replot_population(output_dir: Path) -> None:
    out = output_dir / "06a_population_globals"
    raw = load_raw_data(out / "raw_plot_data.npz")
    truth = raw["truth_mu_logvariance"]
    compass_values = raw["compass_samples_mu_logvariance"]
    mcmc = raw["mcmc_samples_mu_logvariance"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for index, name in enumerate(("mu", "log variance")):
        axes[index].hist(mcmc[:, index], bins=45, density=True, alpha=0.45, label="MCMC")
        axes[index].hist(compass_values[:, index], bins=45, density=True, alpha=0.45,
                         label="COMPASS")
        axes[index].axvline(truth[index], color="tab:red", label="truth")
        axes[index].set(xlabel=name, ylabel="density")
    axes[0].legend()
    axes[2].scatter(mcmc[:, 0], np.exp(mcmc[:, 1]), s=5, alpha=0.15, label="MCMC")
    axes[2].scatter(compass_values[:, 0], np.exp(compass_values[:, 1]), s=5,
                    alpha=0.15, label="COMPASS")
    axes[2].scatter([truth[0]], [np.exp(truth[1])], marker="*", s=130,
                    color="tab:red", label="truth")
    axes[2].set(xlabel="population centre mu", ylabel="population variance")
    axes[2].legend()
    save_figure(fig, out / "population_mu_variance_vs_mcmc.png")


def replot_hierarchy(output_dir: Path) -> None:
    out = output_dir / "06b_shared_local"
    raw = load_raw_data(out / "raw_plot_data.npz")
    exact_mean, covariance = raw["exact_joint_mean"], raw["exact_joint_covariance"]
    global_samples, local_samples = raw["compass_global_samples"], raw["compass_local_samples"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    global_std = math.sqrt(covariance[0, 0])
    grid = np.linspace(exact_mean[0] - 4 * global_std, exact_mean[0] + 4 * global_std, 400)
    axes[0].hist(global_samples, bins=50, density=True, alpha=0.6, label="COMPASS")
    axes[0].plot(grid, norm.pdf(grid, exact_mean[0], global_std), "k--", label="analytic")
    axes[0].axvline(float(raw["global_truth"]), color="tab:red", label="truth")
    axes[0].set(xlabel="global parameter", ylabel="density", title="Shared posterior")
    axes[0].legend()
    indices = np.arange(local_samples.shape[0])
    local_means, local_stds = local_samples.mean(1), local_samples.std(1, ddof=1)
    axes[1].errorbar(indices, local_means, yerr=local_stds, fmt="o", ms=3,
                     label="COMPASS mean +/- std")
    axes[1].plot(indices, exact_mean[1:], "k_", ms=8, label="analytic mean")
    axes[1].scatter(indices, raw["local_truth"], marker="x", color="tab:red", label="true local")
    axes[1].set(xlabel="observation", ylabel="local parameter", title="Local posteriors")
    axes[1].legend()
    axes[2].scatter(exact_mean[1:], local_means, c=indices, cmap="viridis")
    lo, hi = min(exact_mean[1:].min(), local_means.min()), max(exact_mean[1:].max(), local_means.max())
    axes[2].plot([lo, hi], [lo, hi], "k--")
    axes[2].set(xlabel="analytic local mean", ylabel="COMPASS local mean",
                title="Local-parameter recovery")
    save_figure(fig, out / "shared_and_local_recovery.png")


REPLOTTERS = {
    "contract": replot_contract,
    "samplers": replot_samplers,
    "scaling": replot_scaling,
    "time": replot_time_sampling,
    "individual": replot_individual,
    "population": replot_population,
    "hierarchy": replot_hierarchy,
}


def parse_experiments(value: str) -> list[str]:
    aliases = {
        "contract": "contract",
        "samplers": "samplers",
        "scaling": "scaling",
        "time": "time",
        "individual": "individual",
        "population": "population",
        "hierarchy": "hierarchy",
    }
    if value.strip() == "all":
        return list(aliases.values())
    chosen = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(chosen) - set(aliases))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown experiments: {', '.join(unknown)}")
    return [aliases[item] for item in chosen]


def build_config(args: argparse.Namespace, device: str) -> RunConfig:
    values = dict(output_dir=args.output_dir.resolve(), device=device, seed=args.seed)
    if args.quick:
        values.update(
            train_samples=4_000,
            validation_samples=500,
            max_epochs=3,
            patience=2,
            posterior_samples=250,
            timesteps=12,
            repeats=1,
            mcmc_steps=3_000,
            mcmc_burn=500,
            mcmc_thin=5,
        )
    return RunConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default="all",
                        help="all or comma-separated: contract,samplers,scaling,time,individual,population,hierarchy")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true",
                        help="tiny wiring test; do not use results for conclusions")
    parser.add_argument("--plots-only", action="store_true",
                        help="regenerate figures from saved CSV/NPZ data; run no calculations")
    args = parser.parse_args()
    experiments = parse_experiments(args.experiments)
    if args.plots_only:
        configure_plot_style()
        for name in experiments:
            print(f"Regenerating {name} plots from saved data...")
            REPLOTTERS[name](args.output_dir.resolve())
        return
    device = select_device(args.device)
    cfg = build_config(args, device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    seed_all(cfg.seed)
    with (cfg.output_dir / "run_config.json").open("w") as handle:
        payload = asdict(cfg)
        payload["output_dir"] = str(payload["output_dir"])
        payload["experiments"] = experiments
        json.dump(payload, handle, indent=2)

    shared_model = None
    needs_shared = bool(set(experiments) & {"contract", "samplers", "scaling", "individual"})
    if needs_shared:
        shared_model = load_or_train(
            cfg, "shared_mixture", 2, simulate_shared, time_sampling="mixture",
        )
    runners = {
        "contract": lambda: experiment_contract(shared_model, cfg),
        "samplers": lambda: experiment_samplers(shared_model, cfg),
        "scaling": lambda: experiment_observation_scaling(shared_model, cfg),
        "time": lambda: experiment_time_sampling(cfg),
        "individual": lambda: experiment_multi_vs_individual(shared_model, cfg),
        "population": lambda: experiment_population_globals(cfg),
        "hierarchy": lambda: experiment_shared_local(cfg),
    }
    for name in experiments:
        print(f"\n=== Running experiment: {name} ===")
        started = time.perf_counter()
        runners[name]()
        print(f"=== {name} finished in {(time.perf_counter() - started) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
