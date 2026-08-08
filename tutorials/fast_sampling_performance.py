#!/usr/bin/env python3
"""Controlled fast-sampling ablations for Gaussian and banana posteriors.

Run without arguments:

    python tutorials/fast_sampling_performance.py

Outputs are written below ``tutorials/output/fast_sampling_performance``, with
separate ``gaussian`` and ``banana`` subdirectories.  Each experiment trains
matched models with uniform, log-sigma, and mixture diffusion-time sampling,
then uses
several observations and sampling seeds to quantify the effects of:

* mixture training-time sampling;
* sigma-space rather than time-space DPM prediction;
* a log-sigma rather than uniform-time integration grid; and
* the stochastic term in Euler--Maruyama.

"""

from __future__ import annotations

import os


CPU_THREAD_LIMIT = 3
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_cpu_usage_limit(
    max_threads: int = CPU_THREAD_LIMIT,
) -> tuple[int, tuple[int, ...]]:
    """Hard-limit this process and inherited children before native imports."""
    logical_cpus = os.cpu_count()
    if logical_cpus is None:
        raise RuntimeError("Cannot enforce the CPU cap: os.cpu_count() is unavailable.")
    cpu_limit = min(int(max_threads), logical_cpus)
    if cpu_limit < 1:
        raise RuntimeError(
            f"Cannot enforce a {max_threads}-thread CPU limit on a {logical_cpus}-CPU "
            "host: one logical CPU would exceed the limit."
        )
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError(
            "Cannot enforce the CPU cap: OS affinity controls are unavailable."
        )

    allowed_cpus = tuple(sorted(os.sched_getaffinity(0)))
    selected_cpus = allowed_cpus[:cpu_limit]
    if not selected_cpus:
        raise RuntimeError("The process has no CPUs available in its affinity mask.")
    os.sched_setaffinity(0, selected_cpus)
    active_cpus = tuple(sorted(os.sched_getaffinity(0)))
    if active_cpus != selected_cpus:
        raise RuntimeError(
            "Failed to enforce CPU affinity: requested "
            f"{selected_cpus}, active {active_cpus}."
        )
    if len(active_cpus) > max_threads:
        raise RuntimeError("The active CPU affinity exceeds the configured 3-thread cap.")
    for variable in CPU_THREAD_ENV_VARS:
        os.environ[variable] = str(len(active_cpus))
    return logical_cpus, active_cpus


# This has to run before Matplotlib, NumPy, pandas, or PyTorch initialize their
# native thread pools. Child processes inherit both affinity and thread limits.
CPU_LIMIT_INFO = configure_cpu_usage_limit()
_logical_cpus, _active_cpus = CPU_LIMIT_INFO
print(
    "[setup] CPU limited to {} of {} logical CPUs ({:.2%} capacity)".format(
        len(_active_cpus), _logical_cpus, len(_active_cpus) / _logical_cpus,
    ),
    flush=True,
)

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import subprocess
import argparse
import time
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns
import torch

try:
    from autocvd import autocvd
except ImportError:  # Allows a CPU-only Python environment to run the tutorial.
    autocvd = None

from compass import ScoreBasedInferenceModel as SBIm


# The script intentionally has no required command-line inputs. Anchoring output
# to this file makes the result location independent of the launch directory.
OUTPUT_ROOT = (
    Path(__file__).resolve().parent / "output" / "fast_sampling_performance"
)
GAUSSIAN_OUTPUT_DIR = OUTPUT_ROOT / "gaussian"
BANANA_OUTPUT_DIR = OUTPUT_ROOT / "banana"
DEVICE = "cpu"  # Updated by select_device() before any model is created.
SEED = 0
TRAIN_SAMPLES = 20_000
VALIDATION_SAMPLES = 2_000
MAX_EPOCHS = 80
BATCH_SIZE = 256
BANANA_TRAIN_SAMPLES = 100_000
BANANA_VALIDATION_SAMPLES = 10_000
BANANA_BATCH_SIZE = 64
BANANA_REFERENCE_SAMPLES = 25_000
EARLY_STOPPING_PATIENCE = 10
# Independent initialisations are necessary before interpreting a training-time
# sampling difference as an effect rather than a lucky/unlucky model fit.
MODEL_SEEDS = (0, 1, 2)
NUM_OBSERVATIONS = 4
NUM_REPEATS = 2
NUM_POSTERIOR_SAMPLES = 750
CALIBRATION_OBSERVATIONS = 16
# Include the complete calibration curve, including the degenerate 0% interval
# and the unbounded 100% interval.
CALIBRATION_LEVELS = tuple(np.linspace(0.0, 1.0, 11))
CALIBRATION_TIMESTEPS = (10, 50)
TIMESTEPS = (5, 10, 20, 50)
PARETO_TIMESTEPS = (10, 50, 100)
RUN_TIMESTEPS = tuple(sorted(set(TIMESTEPS) | set(PARETO_TIMESTEPS)))

A_MIX = torch.tensor([[1.0, 0.5], [0.3, 1.0]])
NOISE_STD = 0.3
COLORS = {
    "uniform": "#9C755F",
    "log_sigma": "#F58518",
    "mixture": "#4C78A8",
    "sigma": "#72B7B2",
    "log": "#54A24B",
    "euler": "#F58518",
}


TRAINING_TIME_SCHEMES = ("uniform", "mixture")
DPM2_TRAINING_TIME_SCHEMES = ("uniform", "log_sigma", "mixture")
TRAINING_TIME_LABELS = {
    "uniform": "Uniform",
    "log_sigma": "Log-sigma",
    "mixture": "Mixed",
}


def select_device() -> str:
    """Reserve one free GPU with autocvd, otherwise use CPU without waiting.

    ``autocvd`` sets ``CUDA_VISIBLE_DEVICES`` itself.  Its timeout is deliberate:
    this benchmark should use a GPU if one is free now, but must not queue for a
    busy GPU when CPU execution is an acceptable fallback.
    """
    if autocvd is None:
        print("autocvd is unavailable; using CPU.")
        return "cpu"
    try:
        selected = autocvd(
            num_gpus=1, interval=1, timeout=1, progress=False,
        )
    except (OSError, TimeoutError, subprocess.SubprocessError):
        print("No free GPU found; using CPU.")
        return "cpu"
    if torch.cuda.is_available():
        print(f"Reserved GPU {selected[0]} with autocvd; using CUDA.")
        return "cuda"
    print("autocvd selected a GPU but CUDA is unavailable to PyTorch; using CPU.")
    return "cpu"


@dataclass(frozen=True)
class SamplerVariant:
    """One controlled sampling setting in the ablation suite."""

    key: str
    label: str
    family: str
    method: str
    order: int | None = None
    grid: str = "log_sigma"
    corrector_steps: int = 0
    color: str = "#4C78A8"
    marker: str = "o"


VARIANTS = (
    SamplerVariant(
        "sigma_uniform_dpm", "Sigma-space DPM-1, uniform t", "DPM predictor",
        "dpm", order=1, grid="uniform_t", color=COLORS["sigma"], marker="s",
    ),
    SamplerVariant(
        "sigma_log_dpm", "Sigma-space DPM-1, log sigma", "DPM predictor",
        "dpm", order=1, grid="log_sigma", color=COLORS["log"], marker="o",
    ),
    SamplerVariant(
        "dpm2_uniform", "DPM-2 + corrector, uniform t", "Overall sampler",
        "dpm", order=2, grid="uniform_t", corrector_steps=5,
        color="#7F6D9A", marker="h",
    ),
    SamplerVariant(
        "dpm2_log", "DPM-2 + corrector, log sigma", "Overall sampler",
        "dpm", order=2, grid="log_sigma", corrector_steps=5,
        color="#B279A2", marker="D",
    ),
    SamplerVariant(
        "em_uniform", "Euler--Maruyama, uniform t", "Euler update",
        "euler", grid="uniform_t", color=COLORS["euler"], marker="^",
    ),
    SamplerVariant(
        "em_log", "Euler--Maruyama, log sigma", "Euler update",
        "euler", grid="log_sigma", color="#FFBF79", marker="v",
    ),
)


def configure_plot_style() -> None:
    """Set a compact, colour-blind-friendly visual style for all figures."""
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 220, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.7,
        "lines.linewidth": 2.1, "lines.markersize": 6,
    })


def simulate(n: int, noise: float = NOISE_STD) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw parameter/simulation pairs from the linear-Gaussian toy model."""
    theta = torch.randn(n, 2)
    return theta, theta @ A_MIX.T + noise * torch.randn(n, 2)


def analytic_posterior(x_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact posterior mean and covariance for one observation."""
    noise_variance = NOISE_STD**2
    covariance = torch.linalg.inv(torch.eye(2) + A_MIX.T @ A_MIX / noise_variance)
    mean = covariance @ A_MIX.T @ x_obs / noise_variance
    return mean, covariance


def train_model(
    time_sampling: str, theta_train: torch.Tensor, x_train: torch.Tensor,
    theta_val: torch.Tensor, x_val: torch.Tensor, output_dir: Path, seed: int,
) -> SBIm:
    """Load an existing model checkpoint, or train it when it is missing."""
    model_dir = output_dir / "models" / time_sampling / f"seed_{seed}"
    checkpoint = model_dir / "Model_checkpoint.pt"
    if checkpoint.exists():
        print(f"Loading existing {time_sampling!r} checkpoint: {checkpoint}")
        return SBIm.load(str(checkpoint), device=DEVICE)


    torch.manual_seed(seed)
    model = SBIm(
        nodes_size=4, sde_type="vesde", sigma=4.0, hidden_size=32, depth=2,
        num_heads=2, mlp_ratio=2, device=DEVICE,
    )
    print(f"Training {time_sampling!r} model on {DEVICE}...")
    started = time.perf_counter()
    model.train(
        theta=theta_train, x=x_train, theta_val=theta_val, x_val=x_val,
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE, device=DEVICE,
        verbose=False, path=str(model_dir),
        time_sampling=time_sampling,
    )
    print(f"  finished in {time.perf_counter() - started:.1f} s")
    return model


def uniform_time_grid(sampler: object):
    """Build a PFODE-compatible grid with nodes equally spaced in diffusion time."""
    def grid(timesteps: int, eps: float, device: str, descending: bool = False,
             dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        ts = torch.linspace(float(eps), 1.0, timesteps, device=device, dtype=dtype)
        if descending:
            ts = torch.flip(ts, dims=(0,))
        return sampler.sde.lambda_t(ts), ts
    return grid


@contextmanager
def sampler_grid_mode(model: SBIm, variant: SamplerVariant) -> Iterator[None]:
    """Temporarily select the requested integration grid."""
    sampler = model.sampler
    original_grid = sampler.pfode.lambda_grid
    try:
        if variant.grid == "uniform_t":
            sampler.pfode.lambda_grid = uniform_time_grid(sampler)
        yield
    finally:
        sampler.pfode.lambda_grid = original_grid


def posterior_errors(samples: torch.Tensor, mean: torch.Tensor,
                     covariance: torch.Tensor) -> tuple[float, float, float]:
    """Return direct errors against supplied reference-posterior moments.

    Keeping mean and covariance errors separate makes sampler bias and
    dispersion error directly visible without fitting another Gaussian or
    constructing a combined divergence.
    """
    empirical_mean = samples.mean(0)
    empirical_covariance = torch.cov(samples.T)
    mean_error = torch.linalg.vector_norm(empirical_mean - mean).item()
    covariance_error = torch.linalg.matrix_norm(
        empirical_covariance - covariance
    ).item()
    posterior_width_ratio = torch.sqrt(
        torch.trace(empirical_covariance) / torch.trace(covariance)
    ).item()
    return mean_error, covariance_error, posterior_width_ratio


def sample_once(model: SBIm, x_obs: torch.Tensor, variant: SamplerVariant,
                timesteps: int) -> tuple[torch.Tensor, float]:
    """Run one sampler configuration and return samples plus wall time."""
    kwargs: dict[str, object] = {
        "x": x_obs.unsqueeze(0), "num_samples": NUM_POSTERIOR_SAMPLES,
        "timesteps": timesteps, "method": variant.method, "device": DEVICE,
        "verbose": False, "corrector_steps": variant.corrector_steps,
        "final_corrector_steps": 3 if variant.corrector_steps else 0,
    }
    if variant.order is not None:
        kwargs["order"] = variant.order
    started = time.perf_counter()
    with sampler_grid_mode(model, variant):
        samples = model.sample(**kwargs)[0].detach().cpu()
    return samples, time.perf_counter() - started


def run_ablation(models: dict[tuple[str, int], SBIm], observations: torch.Tensor) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    """Evaluate all matched model/sampler/observation/repeat combinations."""
    rows: list[dict[str, object]] = []
    exemplars: dict[str, torch.Tensor] = {}
    for (training_scheme, model_seed), model in models.items():
        for observation_index, x_obs in enumerate(observations):
            mean, covariance = analytic_posterior(x_obs)
            for variant in VARIANTS:
                for steps in RUN_TIMESTEPS:
                    for repeat in range(NUM_REPEATS):
                        torch.manual_seed(SEED + 10_000 * observation_index + 100 * steps + repeat)
                        samples, elapsed = sample_once(model, x_obs, variant, steps)
                        (
                            mean_error, covariance_error, posterior_width_ratio,
                        ) = posterior_errors(samples, mean, covariance)
                        rows.append({
                            "training_scheme": training_scheme, "variant": variant.key,
                            "variant_label": variant.label, "family": variant.family,
                            "timesteps": steps, "model_seed": model_seed,
                            "observation": observation_index,
                            "repeat": repeat, "runtime_s": elapsed,
                            "runtime_per_sample_ms": elapsed * 1e3 / NUM_POSTERIOR_SAMPLES,
                            "runtime_device": DEVICE,
                            "mean_error": mean_error, "covariance_error": covariance_error,
                            "posterior_width_ratio": posterior_width_ratio,
                        })
                        if model_seed == MODEL_SEEDS[0] and observation_index == 0 and repeat == 0 and steps == 20:
                            exemplars[f"{training_scheme}:{variant.key}"] = samples
    return pd.DataFrame(rows), exemplars


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Median and 10--90% bands for direct reference-posterior errors."""
    group_columns = [
        "training_scheme", "variant", "variant_label", "family", "timesteps",
    ]
    grouped = results.groupby(group_columns, as_index=False)
    aggregations: dict[str, tuple[str, object]] = {
        "runtime_s_median": ("runtime_s", "median"),
        "mean_error_median": ("mean_error", "median"),
        "mean_error_lo": ("mean_error", lambda x: x.quantile(0.1)),
        "mean_error_hi": ("mean_error", lambda x: x.quantile(0.9)),
        "covariance_error_median": ("covariance_error", "median"),
        "covariance_error_lo": (
            "covariance_error", lambda x: x.quantile(0.1)
        ),
        "covariance_error_hi": (
            "covariance_error", lambda x: x.quantile(0.9)
        ),
        "posterior_width_ratio_median": (
            "posterior_width_ratio", "median"
        ),
        "posterior_width_ratio_lo": (
            "posterior_width_ratio", lambda x: x.quantile(0.1)
        ),
        "posterior_width_ratio_hi": (
            "posterior_width_ratio", lambda x: x.quantile(0.9)
        ),
    }
    if "sliced_wasserstein" in results.columns:
        aggregations.update({
            "sliced_wasserstein_median": (
                "sliced_wasserstein", "median"
            ),
            "sliced_wasserstein_lo": (
                "sliced_wasserstein", lambda x: x.quantile(0.1)
            ),
            "sliced_wasserstein_hi": (
                "sliced_wasserstein", lambda x: x.quantile(0.9)
            ),
        })
    if {
        "theta_map_squared_error", "observation_map_squared_error",
    }.issubset(results.columns):
        aggregations.update({
            "theta_map_squared_error_median": (
                "theta_map_squared_error", "median"
            ),
            "theta_map_squared_error_lo": (
                "theta_map_squared_error", lambda x: x.quantile(0.1)
            ),
            "theta_map_squared_error_hi": (
                "theta_map_squared_error", lambda x: x.quantile(0.9)
            ),
            "observation_map_squared_error_median": (
                "observation_map_squared_error", "median"
            ),
            "observation_map_squared_error_lo": (
                "observation_map_squared_error", lambda x: x.quantile(0.1)
            ),
            "observation_map_squared_error_hi": (
                "observation_map_squared_error", lambda x: x.quantile(0.9)
            ),
        })
    return grouped.agg(**aggregations)


def evaluate_training_score_accuracy(
    models: dict[tuple[str, int], SBIm],
    observations: torch.Tensor,
    num_draws: int = 256,
    grid_points: int = 31,
) -> pd.DataFrame:
    """Compare learned posterior scores with the analytic score over noise.

    For the VESDE toy problem,
    theta_t | x is Gaussian with covariance Sigma_post + sigma_m(t)^2 I.
    Its score is therefore available in closed form at every noise level.
    """
    device = torch.device(DEVICE)
    reference_model = next(iter(models.values()))
    lam_min = reference_model.sde.lambda_t(torch.full((1,), 1e-3))
    lam_max = reference_model.sde.lambda_t(torch.ones(1))
    sigmas = torch.logspace(
        torch.log10(lam_min).item(), torch.log10(lam_max).item(),
        grid_points, device=device,
    )
    times = reference_model.sde.time_of_lambda(sigmas.cpu()).to(device)
    condition_mask = torch.tensor(
        [0.0, 0.0, 1.0, 1.0], device=device,
    ).repeat(num_draws, 1)
    identity = torch.eye(2, device=device)
    rows: list[dict[str, object]] = []

    for (scheme, model_seed), model in models.items():
        model.model.to(device).eval()
        for observation_index, x_obs_cpu in enumerate(observations):
            x_obs = x_obs_cpu.to(device)
            mean, covariance = analytic_posterior(x_obs_cpu)
            mean = mean.to(device)
            covariance = covariance.to(device)
            for grid_index, (sigma, time_value) in enumerate(zip(sigmas, times)):
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    50_000 + 1_000 * observation_index + grid_index
                )
                covariance_t = covariance + sigma.square() * identity
                standard_normal = torch.randn(
                    num_draws, 2, generator=generator, device=device,
                )
                theta_t = mean + (
                    standard_normal @ torch.linalg.cholesky(covariance_t).T
                )
                exact_score = -(theta_t - mean) @ torch.linalg.inv(covariance_t).T
                state = torch.cat(
                    [theta_t, x_obs.repeat(num_draws, 1)], dim=1,
                )
                time_batch = time_value.reshape(1, 1).repeat(num_draws, 1)
                with torch.no_grad():
                    raw_score = model.model(
                        x=state, t=time_batch, c=condition_mask,
                    )
                    predicted_score = model.output_scale_function(
                        time_batch, raw_score,
                    )[:, :2]
                residual_power = torch.mean(
                    torch.sum((predicted_score - exact_score).square(), dim=1)
                )
                score_power = torch.mean(
                    torch.sum(exact_score.square(), dim=1)
                )
                rows.append({
                    "training_scheme": scheme,
                    "model_seed": model_seed,
                    "observation": observation_index,
                    "sigma_m": sigma.item(),
                    "diffusion_time": time_value.item(),
                    "relative_score_rmse": torch.sqrt(
                        residual_power / score_power
                    ).item(),
                    "scaled_score_rmse": (
                        sigma * torch.sqrt(residual_power / 2)
                    ).item(),
                })
    return pd.DataFrame(rows)


def evaluate_training_calibration(
    models: dict[tuple[str, int], SBIm],
    observations: torch.Tensor,
) -> pd.DataFrame:
    """Evaluate marginal credible-interval coverage against the exact posterior.

    Each predicted interval is fitted to generated samples.  We then compute
    analytically how much of the true Gaussian posterior lies in that interval.
    A calibrated sampler has analytical coverage equal to its nominal level.
    """
    selected = tuple(
        item for item in VARIANTS
        if item.key in ("sigma_log_dpm", "dpm2_log")
    )
    normal = torch.distributions.Normal(0.0, 1.0)
    rows: list[dict[str, object]] = []

    for (scheme, model_seed), model in models.items():
        for observation_index, x_obs in enumerate(observations):
            analytic_mean, analytic_covariance = analytic_posterior(x_obs)
            analytic_std = torch.sqrt(torch.diag(analytic_covariance))
            for variant in selected:
                for steps in CALIBRATION_TIMESTEPS:
                    torch.manual_seed(
                        SEED + 200_000 + 10_000 * observation_index
                        + 100 * steps + 1_000 * model_seed
                    )
                    samples, _ = sample_once(model, x_obs, variant, steps)
                    predicted_mean = samples.mean(0)
                    predicted_std = torch.sqrt(torch.diag(torch.cov(samples.T)))
                    for nominal_coverage in CALIBRATION_LEVELS:
                        if nominal_coverage == 0.0:
                            exact_coverage = torch.zeros_like(analytic_mean)
                        elif nominal_coverage == 1.0:
                            exact_coverage = torch.ones_like(analytic_mean)
                        else:
                            probability = torch.tensor(
                                (1.0 + nominal_coverage) / 2.0,
                                dtype=samples.dtype,
                            )
                            z_value = normal.icdf(probability)
                            lower = predicted_mean - z_value * predicted_std
                            upper = predicted_mean + z_value * predicted_std
                            exact_coverage = torch.special.ndtr(
                                (upper - analytic_mean) / analytic_std
                            ) - torch.special.ndtr(
                                (lower - analytic_mean) / analytic_std
                            )
                        rows.append({
                            "training_scheme": scheme,
                            "variant": variant.key,
                            "variant_label": variant.label,
                            "timesteps": steps,
                            "model_seed": model_seed,
                            "observation": observation_index,
                            "nominal_coverage": nominal_coverage,
                            "analytical_coverage": exact_coverage.mean().item(),
                        })
    return pd.DataFrame(rows)


def summarise_training_calibration(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise exact conditional coverage across seeds and observations."""
    grouped = results.groupby(
        ["training_scheme", "variant", "variant_label", "timesteps",
         "nominal_coverage"],
        as_index=False,
    )
    return grouped.agg(
        analytical_coverage_median=("analytical_coverage", "median"),
        analytical_coverage_lo=(
            "analytical_coverage", lambda x: x.quantile(0.1)
        ),
        analytical_coverage_hi=(
            "analytical_coverage", lambda x: x.quantile(0.9)
        ),
    )


def summarise_training_score_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise analytical score errors across seeds and observations."""
    grouped = results.groupby(
        ["training_scheme", "sigma_m", "diffusion_time"], as_index=False,
    )
    return grouped.agg(
        relative_score_rmse_median=("relative_score_rmse", "median"),
        relative_score_rmse_lo=(
            "relative_score_rmse", lambda x: x.quantile(0.1)
        ),
        relative_score_rmse_hi=(
            "relative_score_rmse", lambda x: x.quantile(0.9)
        ),
        scaled_score_rmse_median=("scaled_score_rmse", "median"),
        scaled_score_rmse_lo=(
            "scaled_score_rmse", lambda x: x.quantile(0.1)
        ),
        scaled_score_rmse_hi=(
            "scaled_score_rmse", lambda x: x.quantile(0.9)
        ),
    )


ANALYTIC_ERROR_PANELS = (
    (
        "mean_error", "Posterior mean",
        r"$\|\hat{\mu} - \mu_{\mathrm{reference}}\|_2$",
    ),
    (
        "covariance_error", "Posterior covariance",
        r"$\|\hat{\Sigma} - \Sigma_{\mathrm{reference}}\|_F$",
    ),
)


def line_with_band(ax: plt.Axes, data: pd.DataFrame, label: str, color: str,
                   metric: str, marker: str = "o",
                   linestyle: str = "-",
                   markerfacecolor: str | None = None) -> None:
    """Draw one direct analytic-error curve and its 10--90% band."""
    data = data[data.timesteps.isin(TIMESTEPS)].sort_values("timesteps")
    x = data["timesteps"].to_numpy()
    ax.plot(
        x, data[f"{metric}_median"], label=label, color=color,
        marker=marker, linestyle=linestyle,
        markerfacecolor=markerfacecolor or color,
        markeredgecolor=color, markeredgewidth=1.4,
    )
    ax.fill_between(
        x, data[f"{metric}_lo"], data[f"{metric}_hi"],
        color=color, alpha=0.11, linewidth=0,
    )


def finish_error_axes(axes: np.ndarray, title: str) -> None:
    """Label paired direct-error panels against the reference posterior."""
    for ax, (_, panel_title, ylabel) in zip(axes, ANALYTIC_ERROR_PANELS):
        ax.set(
            title=panel_title, xlabel="Sampling steps",
            ylabel=ylabel, xscale="log",
        )
        ax.legend(loc="best", frameon=True)
    axes[0].figure.suptitle(title, fontsize=13, fontweight="bold")


def plot_training_time_sampling(
    summary: pd.DataFrame, output_path: Path,
) -> None:
    """Compare uniform and mixture training for log-noise DPM-1 and DPM-2."""
    configure_plot_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(13.2, 5.1), constrained_layout=True,
    )
    samplers = (
        ("sigma_log_dpm", "DPM-1", COLORS["log"], "o"),
        ("dpm2_log", "DPM-2 + corrector", "#B279A2", "D"),
    )
    for ax, (metric, _, _) in zip(axes, ANALYTIC_ERROR_PANELS):
        for variant_key, sampler_label, color, marker in samplers:
            for scheme, linestyle in (
                ("uniform", (0, (6, 3))), ("mixture", "-"),
            ):
                data = summary[
                    (summary.training_scheme == scheme)
                    & (summary.variant == variant_key)
                ]
                line_with_band(
                    ax, data, f"{sampler_label} — {scheme} training",
                    color, metric, marker, linestyle,
                    "white" if scheme == "uniform" else color,
                )
    finish_error_axes(
        axes,
        "Training-time distribution: uniform vs mixture "
        "(sigma-space predictors, log-noise grid)",
    )
    legend_title = (
        "Dashed + hollow: uniform training\n"
        "Solid + filled: mixture training"
    )
    for ax in axes:
        ax.legend(
            title=legend_title, loc="best", frameon=True,
            handlelength=4.0, handletextpad=0.9,
        )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_calibration(
    calibration_summary: pd.DataFrame,
    output_path: Path,
    coverage_ylabel: str = (
        "Analytical posterior mass in predicted interval"
    ),
    figure_title: str = (
        "Posterior calibration: uniform vs mixture time training"
    ),
) -> None:
    """Plot exact posterior coverage for matched uniform/mixture comparisons."""
    configure_plot_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(13.2, 5.1), constrained_layout=True,
    )
    samplers = (
        ("sigma_log_dpm", "DPM-1", COLORS["log"], "o"),
        ("dpm2_log", "DPM-2 + corrector", "#B279A2", "D"),
    )
    for ax, steps in zip(axes, CALIBRATION_TIMESTEPS):
        ax.plot(
            [0.0, 1.0], [0.0, 1.0], color="#202020", linestyle="--",
            linewidth=1.4, label="Perfect calibration",
        )
        for variant_key, sampler_label, color, marker_style in samplers:
            for scheme, linestyle, fill in (
                ("uniform", (0, (6, 3)), "white"),
                ("mixture", "-", color),
            ):
                data = calibration_summary[
                    (calibration_summary.training_scheme == scheme)
                    & (calibration_summary.variant == variant_key)
                    & (calibration_summary.timesteps == steps)
                ].sort_values("nominal_coverage")
                ax.plot(
                    data.nominal_coverage, data.analytical_coverage_median,
                    label=f"{sampler_label} — {scheme} training",
                    color=color, linestyle=linestyle, marker=marker_style,
                    markerfacecolor=fill, markeredgecolor=color,
                )
                ax.fill_between(
                    data.nominal_coverage, data.analytical_coverage_lo,
                    data.analytical_coverage_hi, color=color,
                    alpha=0.10, linewidth=0,
                )
        ax.set(
            title=f"{steps} log-sigma sampling steps",
            xlabel="Nominal marginal credible-interval coverage",
            ylabel=coverage_ylabel,
            xlim=(0.0, 1.0), ylim=(0.0, 1.0),
        )
        ax.legend(
            loc="upper left", frameon=True, fontsize=7.5,
            handlelength=3.7,
        )
    fig.suptitle(
        figure_title,
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_noise_diagnostic(
    score_summary: pd.DataFrame,
    posterior_summary: pd.DataFrame,
    output_path: Path,
    score_panel_title: str = "Analytical posterior-score accuracy",
    figure_title: str = "What mixture time sampling changes",
) -> None:
    """Show where mixture training helps and whether posteriors sharpen."""
    configure_plot_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(19.2, 5.1), constrained_layout=True,
    )

    ax = axes[0]
    for scheme, color, linestyle, marker_style in (
        ("uniform", COLORS["uniform"], (0, (6, 3)), "o"),
        ("mixture", COLORS["mixture"], "-", "o"),
    ):
        data = score_summary[
            score_summary.training_scheme == scheme
        ].sort_values("sigma_m")
        ax.plot(
            data.sigma_m, data.relative_score_rmse_median,
            label=f"{scheme.title()} training", color=color,
            linestyle=linestyle, marker=marker_style, markevery=5,
            markerfacecolor="white" if scheme == "uniform" else color,
            markeredgecolor=color,
        )
        ax.fill_between(
            data.sigma_m, data.relative_score_rmse_lo,
            data.relative_score_rmse_hi, color=color, alpha=0.12,
            linewidth=0,
        )
    ax.axvspan(
        score_summary.sigma_m.min(), 0.2, color="#54A24B", alpha=0.05,
        label=r"small noise ($\sigma_m<0.2$)",
    )
    ax.axvspan(
        1.0, score_summary.sigma_m.max(), color="#9D9D9D", alpha=0.07,
        label=r"large noise ($\sigma_m>1$)",
    )
    ax.set(
        title=score_panel_title,
        xlabel=r"Marginal noise scale $\sigma_m$",
        ylabel="Relative score RMSE", xscale="log", yscale="log",
    )
    ax.legend(frameon=True, handlelength=3.5)

    ax = axes[1]
    variants = (
        ("sigma_log_dpm", "DPM-1"),
        ("dpm2_log", "DPM-2 + corrector"),
    )
    positions = np.arange(len(variants))
    for scheme, color, offset, marker_style in (
        ("uniform", COLORS["uniform"], -0.12, "o"),
        ("mixture", COLORS["mixture"], 0.12, "D"),
    ):
        medians, lower, upper = [], [], []
        for variant_key, _ in variants:
            row = posterior_summary[
                (posterior_summary.training_scheme == scheme)
                & (posterior_summary.variant == variant_key)
                & (posterior_summary.timesteps == 50)
            ].iloc[0]
            medians.append(row.posterior_width_ratio_median)
            lower.append(
                row.posterior_width_ratio_median
                - row.posterior_width_ratio_lo
            )
            upper.append(
                row.posterior_width_ratio_hi
                - row.posterior_width_ratio_median
            )
        ax.errorbar(
            positions + offset, medians, yerr=[lower, upper],
            label=f"{scheme.title()} training", color=color,
            marker=marker_style, linestyle="none", capsize=4,
            markersize=7,
            markerfacecolor="white" if scheme == "uniform" else color,
            markeredgewidth=1.5,
        )
    ax.axhline(
        1.0, color="#202020", linestyle="--", linewidth=1.4,
        label="Analytical posterior width",
    )
    ax.set(
        title="Posterior sharpness at 50 sampling steps",
        ylabel="RMS posterior width / analytical width",
        xticks=positions,
        xticklabels=[label for _, label in variants],
    )
    ax.legend(frameon=True)

    ax = axes[2]
    variants = (
        ("sigma_log_dpm", "DPM-1"),
        ("em_log", "Euler--Maruyama"),
        ("dpm2_log", "DPM-2 + corrector"),
    )
    positions = np.arange(len(variants))
    for scheme, color, offset, marker_style in (
        ("uniform", COLORS["uniform"], -0.12, "o"),
        ("mixture", COLORS["mixture"], 0.12, "D"),
    ):
        medians, lower, upper = [], [], []
        for variant_key, _ in variants:
            row = posterior_summary[
                (posterior_summary.training_scheme == scheme)
                & (posterior_summary.variant == variant_key)
                & (posterior_summary.timesteps == 50)
            ].iloc[0]
            medians.append(row.mean_error_median)
            lower.append(row.mean_error_median - row.mean_error_lo)
            upper.append(row.mean_error_hi - row.mean_error_median)
        ax.errorbar(
            positions + offset, medians, yerr=[lower, upper],
            label=f"{scheme.title()} training", color=color,
            marker=marker_style, linestyle="none", capsize=4,
            markersize=7,
            markerfacecolor="white" if scheme == "uniform" else color,
            markeredgewidth=1.5,
        )
    ax.axhline(
        0.0, color="#202020", linestyle="--", linewidth=1.4,
        label="Reference posterior mean",
    )
    ax.set(
        title="Posterior mean distance at 50 sampling steps",
        ylabel="Distance to reference posterior mean",
        xticks=positions,
        xticklabels=[label for _, label in variants],
    )
    ax.legend(frameon=True)
    fig.suptitle(
        figure_title,
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_dpm_ablation(summary: pd.DataFrame, output_path: Path) -> None:
    """Compare DPM updates and grids against the analytic posterior."""
    configure_plot_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(13.2, 5.1), constrained_layout=True,
    )
    dpm_keys = {
        "sigma_uniform_dpm", "sigma_log_dpm",
        "dpm2_uniform", "dpm2_log",
    }
    for ax, (metric, _, _) in zip(axes, ANALYTIC_ERROR_PANELS):
        for variant in (item for item in VARIANTS if item.key in dpm_keys):
            data = summary[
                (summary.training_scheme == "mixture")
                & (summary.variant == variant.key)
            ]
            line_with_band(
                ax, data, variant.label, variant.color,
                metric, variant.marker,
            )
    finish_error_axes(
        axes, "DPM update and time-grid ablation (mixture-trained models)",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_euler_ablation(summary: pd.DataFrame, output_path: Path) -> None:
    """Compare Euler updates directly with the analytic posterior."""
    configure_plot_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(13.2, 5.1), constrained_layout=True,
    )
    for ax, (metric, _, _) in zip(axes, ANALYTIC_ERROR_PANELS):
        for variant in (
            item for item in VARIANTS if item.family == "Euler update"
        ):
            data = summary[
                (summary.training_scheme == "mixture")
                & (summary.variant == variant.key)
            ]
            line_with_band(
                ax, data, variant.label, variant.color,
                metric, variant.marker,
            )
    finish_error_axes(
        axes, "Euler update and time-grid ablation (mixture-trained models)",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot runtime against analytic errors at the labelled step budgets."""
    configure_plot_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(13.4, 5.3), constrained_layout=True,
    )
    required_steps = set(PARETO_TIMESTEPS)
    pareto_series = [(variant, "mixture") for variant in VARIANTS]
    pareto_series.append((
        next(variant for variant in VARIANTS if variant.key == "dpm2_log"),
        "uniform",
    ))
    for ax, (metric, panel_title, ylabel) in zip(
        axes, ANALYTIC_ERROR_PANELS,
    ):
        for variant, training_scheme in pareto_series:
            data = summary[
                (summary.training_scheme == training_scheme)
                & (summary.variant == variant.key)
                & (summary.timesteps.isin(PARETO_TIMESTEPS))
            ].sort_values("timesteps")
            missing_steps = required_steps - set(data.timesteps.astype(int))
            if missing_steps:
                raise ValueError(
                    f"Missing Pareto results for {variant.key}: "
                    f"{sorted(missing_steps)}"
                )
            display_label = (
                "DPM-1, " + variant.label.split(", ", 1)[1]
                if variant.key.startswith("sigma_")
                else variant.label
            )
            if training_scheme == "uniform":
                display_label += " — uniform training"
            ax.plot(
                data.runtime_s_median, data[f"{metric}_median"],
                label=display_label, color=variant.color, marker=variant.marker,
                linestyle="--" if training_scheme == "uniform" else "-",
                markerfacecolor="white" if training_scheme == "uniform" else variant.color,
            )
            for row in data.itertuples(index=False):
                ax.annotate(
                    str(int(row.timesteps)),
                    (row.runtime_s_median, getattr(row, f"{metric}_median")),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=6.5, color=variant.color,
                    bbox={
                        "facecolor": "white", "edgecolor": "none",
                        "alpha": 0.68, "pad": 0.5,
                    },
                )
        ax.set(
            title=panel_title, xlabel="Median sampling time [s]",
            ylabel=ylabel, xscale="log", yscale="log",
        )
        ax.legend(loc="best", fontsize=7.5, frameon=True)
    fig.suptitle(
        "Accuracy--runtime Pareto view (mixture-trained models, plus "
        "uniform-trained DPM-2 + corrector on the log-sigma grid; "
        "labels = sampling steps)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_dpm2_training_grid_pareto(
    summary: pd.DataFrame, output_path: Path,
) -> bool:
    """Compare the six DPM-2/corrector training-grid combinations.

    Colour identifies the training-time distribution; line style and marker
    identify the inference-time grid.  Returning ``False`` lets replot-only
    mode leave legacy result directories untouched until all six conditions
    have been benchmarked.
    """
    variants = (
        ("dpm2_uniform", "uniform t", "--", "s"),
        ("dpm2_log", "log-sigma", "-", "o"),
    )
    series = []
    for scheme in DPM2_TRAINING_TIME_SCHEMES:
        for variant_key, grid_label, linestyle, marker in variants:
            data = summary[
                (summary.training_scheme == scheme)
                & (summary.variant == variant_key)
                & (summary.timesteps.isin(PARETO_TIMESTEPS))
            ].sort_values("timesteps")
            if set(data.timesteps.astype(int)) != set(PARETO_TIMESTEPS):
                return False
            series.append((scheme, grid_label, linestyle, marker, data))

    configure_plot_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(13.4, 5.3), constrained_layout=True,
    )
    for ax, (metric, panel_title, ylabel) in zip(
        axes, ANALYTIC_ERROR_PANELS,
    ):
        for scheme, grid_label, linestyle, marker, data in series:
            color = COLORS[scheme]
            ax.plot(
                data.runtime_s_median, data[f"{metric}_median"],
                label=(
                    f"{TRAINING_TIME_LABELS[scheme]} training — {grid_label}"
                ),
                color=color, marker=marker, linestyle=linestyle,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.8,
            )
            for row in data.itertuples(index=False):
                ax.annotate(
                    str(int(row.timesteps)),
                    (row.runtime_s_median, getattr(row, f"{metric}_median")),
                    xytext=(5, 4), textcoords="offset points", fontsize=6.5,
                    color=color,
                )
        ax.set(
            title=panel_title, xlabel="Median sampling time [s]",
            ylabel=ylabel, xscale="log", yscale="log",
        )
        ax.legend(loc="best", fontsize=7.2, frameon=True)
    fig.suptitle(
        "DPM-2 + corrector: training-time sampling × inference-time grid "
        "(labels = sampling steps)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_banana_map_pareto(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot MAP accuracy against runtime in parameter and observation space."""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3), constrained_layout=True)
    panels = (
        (
            "theta_map_squared_error", "Parameter-space MAP",
            r"$\|\hat{\theta}_{\mathrm{MAP}} - "
            r"\theta_{\mathrm{MAP,reference}}\|_2^2$",
        ),
        (
            "observation_map_squared_error", "Observation-space MAP",
            r"$\|f(\hat{\theta}_{\mathrm{MAP}}) - "
            r"f(\theta_{\mathrm{MAP,reference}})\|_2^2$",
        ),
    )
    series = [(variant, "mixture") for variant in VARIANTS]
    series.append((next(v for v in VARIANTS if v.key == "dpm2_log"), "uniform"))
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for variant, training_scheme in series:
            data = summary[(summary.training_scheme == training_scheme)
                           & (summary.variant == variant.key)
                           & (summary.timesteps.isin(PARETO_TIMESTEPS))].sort_values("timesteps")
            label = ("DPM-1, " + variant.label.split(", ", 1)[1]
                     if variant.key.startswith("sigma_") else variant.label)
            if training_scheme == "uniform":
                label += " — uniform training"
            ax.plot(data.runtime_s_median, data[f"{metric}_median"], label=label,
                    color=variant.color, marker=variant.marker,
                    linestyle="--" if training_scheme == "uniform" else "-",
                    markerfacecolor="white" if training_scheme == "uniform" else variant.color)
            for row in data.itertuples(index=False):
                ax.annotate(str(int(row.timesteps)),
                            (row.runtime_s_median, getattr(row, f"{metric}_median")),
                            xytext=(5, 4), textcoords="offset points", fontsize=6.5)
        ax.set(title=title, xlabel="Median sampling time [s]", ylabel=ylabel,
               xscale="log", yscale="log")
        ax.legend(loc="best", fontsize=7.5, frameon=True)
    fig.suptitle("MAP accuracy--runtime Pareto view (labels = sampling steps)",
                  fontsize=13, fontweight="bold")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


PAIRPLOT_VARIANTS = (
    ("sigma_log_dpm", "Sigma-space DPM-1 + log sigma"),
    ("dpm2_log", "DPM-2 + corrector, log sigma"),
    ("em_log", "Euler--Maruyama + log sigma"),
)


def plot_sampler_pairplot(
    exemplars: dict[str, torch.Tensor],
    reference_samples: torch.Tensor,
    output_path: Path,
    title: str,
    max_points_per_source: int = 2_000,
) -> None:
    """Plot reference and sampler posteriors in the supplied Seaborn style."""
    frames = [
        pd.DataFrame({
            r"$\theta_1$": reference_samples[
                :max_points_per_source, 0
            ].numpy(),
            r"$\theta_2$": reference_samples[
                :max_points_per_source, 1
            ].numpy(),
            "Sampler": "Reference posterior",
        })
    ]
    for variant_key, label in PAIRPLOT_VARIANTS:
        samples = exemplars[f"mixture:{variant_key}"][
            :max_points_per_source
        ]
        frames.append(pd.DataFrame({
            r"$\theta_1$": samples[:, 0].numpy(),
            r"$\theta_2$": samples[:, 1].numpy(),
            "Sampler": label,
        }))
    grid = sns.pairplot(
        pd.concat(frames, ignore_index=True),
        vars=(r"$\theta_1$", r"$\theta_2$"),
        hue="Sampler",
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 3},
    )
    grid.fig.suptitle(title, y=1.02)
    grid.fig.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(grid.fig)


def add_covariance_ellipse(ax: plt.Axes, mean: torch.Tensor, covariance: torch.Tensor,
                           n_std: float) -> None:
    values, vectors = np.linalg.eigh(covariance.numpy())
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    ax.add_patch(Ellipse(mean.numpy(), 2 * n_std * np.sqrt(values[0]),
                         2 * n_std * np.sqrt(values[1]), angle=angle,
                         fill=False, color="#4C78A8", linewidth=1.4))


def plot_posterior_examples(exemplars: dict[str, torch.Tensor], observation: torch.Tensor,
                            output_path: Path) -> None:
    """Show intuitive posterior samples for the current sampler variants."""
    configure_plot_style()
    selected = [
        ("mixture:sigma_log_dpm", "Sigma-space DPM-1 + log sigma"),
        ("mixture:em_log", "Euler--Maruyama + log sigma"),
    ]
    mean, covariance = analytic_posterior(observation)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, selected):
        samples = exemplars[key]
        ax.scatter(samples[:, 0], samples[:, 1], s=4, color="#F58518", alpha=0.18, rasterized=True)
        add_covariance_ellipse(ax, mean, covariance, 1)
        add_covariance_ellipse(ax, mean, covariance, 2)
        ax.plot(*mean.numpy(), marker="+", color="#4C78A8", ms=12, mew=2.2)
        ax.set_title(title)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
    fig.suptitle("One observation at 20 sampling steps\norange: generated samples; blue: analytic posterior", fontsize=14, fontweight="bold")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_error_distributions(
    results: pd.DataFrame, output_path: Path,
) -> None:
    """Show direct reference errors at the 50-step sampling budget."""
    configure_plot_style()
    selected = [
        "sigma_log_dpm", "em_log", "dpm2_log",
    ]
    labels = [
        next(v.label for v in VARIANTS if v.key == key) for key in selected
    ]
    data = results[
        (results.training_scheme == "mixture")
        & (results.timesteps == 50)
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(14.2, 5.2), constrained_layout=True,
    )
    for ax, (metric, panel_title, ylabel) in zip(
        axes, ANALYTIC_ERROR_PANELS,
    ):
        values = [
            data[data.variant == key][metric].to_numpy() for key in selected
        ]
        box = ax.boxplot(
            values, tick_labels=labels, patch_artist=True, showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.5},
        )
        for patch, key in zip(box["boxes"], selected):
            patch.set(
                facecolor=next(
                    v.color for v in VARIANTS if v.key == key
                ),
                alpha=0.70,
            )
        ax.set(title=panel_title, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle(
        "50-step sampling: direct errors against the reference posterior",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_banana_shape_ablation(
    summary: pd.DataFrame, output_path: Path,
) -> None:
    """Plot the nonlinear full-distribution error for every sampler variant."""
    configure_plot_style()
    fig, ax = plt.subplots(
        1, 1, figsize=(8.2, 5.6), constrained_layout=True,
    )
    for variant in VARIANTS:
        data = summary[
            (summary.training_scheme == "mixture")
            & (summary.variant == variant.key)
        ]
        line_with_band(
            ax, data, variant.label, variant.color,
            "sliced_wasserstein", variant.marker,
        )
    ax.set(
        title="Banana posterior shape error",
        xlabel="Sampling steps",
        ylabel="Sliced Wasserstein distance to rejection reference",
        xscale="log", yscale="log",
    )
    ax.legend(loc="best", fontsize=7.5, frameon=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


BANANA_NOISE_STD = 0.3


@dataclass(frozen=True)
class BananaData:
    """Raw and normalized splits for the Banana_posterior.ipynb problem."""

    theta_train: torch.Tensor
    x_train: torch.Tensor
    theta_validation: torch.Tensor
    x_validation: torch.Tensor
    theta_mean: torch.Tensor
    theta_std: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor

    def normalize_theta(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.theta_mean) / self.theta_std

    def denormalize_theta(self, theta: torch.Tensor) -> torch.Tensor:
        return theta * self.theta_std + self.theta_mean

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std

    def denormalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.x_std + self.x_mean


def simulate_banana(
    n: int, generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the nonlinear posterior problem from Banana_posterior.ipynb."""
    theta = torch.randn(n, 2, generator=generator)
    noise = BANANA_NOISE_STD * torch.randn(n, 2, generator=generator)
    x = torch.stack(
        [theta[:, 0], theta[:, 0].square() + theta[:, 1]], dim=1,
    ) + noise
    return theta, x


def make_banana_data() -> BananaData:
    theta_train, x_train = simulate_banana(BANANA_TRAIN_SAMPLES)
    theta_validation, x_validation = simulate_banana(
        BANANA_VALIDATION_SAMPLES,
    )
    return BananaData(
        theta_train=theta_train,
        x_train=x_train,
        theta_validation=theta_validation,
        x_validation=x_validation,
        theta_mean=theta_train.mean(0),
        theta_std=theta_train.std(0),
        x_mean=x_train.mean(0),
        x_std=x_train.std(0),
    )


def train_banana_model(
    time_sampling: str,
    data: BananaData,
    output_dir: Path,
    seed: int,
) -> SBIm:
    """Load or train one notebook-matched banana posterior model."""
    model_dir = output_dir / "models" / time_sampling / f"seed_{seed}"
    checkpoint = model_dir / "Model_checkpoint.pt"
    if checkpoint.exists():
        print(f"Loading existing banana {time_sampling!r} checkpoint: {checkpoint}")
        return SBIm.load(str(checkpoint), device=DEVICE)

    torch.manual_seed(seed)
    model = SBIm(
        nodes_size=4, sde_type="vesde", sigma=2.0,
        hidden_size=128, depth=4, num_heads=4, device=DEVICE,
    )
    print(f"Training banana {time_sampling!r} model on {DEVICE}...")
    started = time.perf_counter()
    model.train(
        theta=data.normalize_theta(data.theta_train),
        x=data.normalize_x(data.x_train),
        theta_val=data.normalize_theta(data.theta_validation),
        x_val=data.normalize_x(data.x_validation),
        batch_size=BANANA_BATCH_SIZE, max_epochs=MAX_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        device=DEVICE, verbose=False, path=str(model_dir),
        time_sampling=time_sampling,
    )
    print(f"  finished in {time.perf_counter() - started:.1f} s")
    return model


def banana_log_likelihood(
    theta: torch.Tensor, x_obs: torch.Tensor,
) -> torch.Tensor:
    """Unnormalised log likelihood; its maximum is zero."""
    residual_1 = x_obs[0] - theta[:, 0]
    residual_2 = x_obs[1] - theta[:, 0].square() - theta[:, 1]
    return -0.5 * (
        residual_1.square() + residual_2.square()
    ) / BANANA_NOISE_STD**2


def banana_observation_from_theta(theta: torch.Tensor) -> torch.Tensor:
    """Return the noiseless banana simulator output for parameter values."""
    return torch.stack(
        [theta[..., 0], theta[..., 0].square() + theta[..., 1]], dim=-1,
    )


def banana_reference_map(x_obs: torch.Tensor) -> torch.Tensor:
    """Return the exact posterior MAP for one banana observation."""
    noise_variance = BANANA_NOISE_STD**2
    x_1, x_2 = x_obs.detach().cpu().numpy()
    roots = np.roots([
        2.0 * noise_variance,
        (1.0 + noise_variance)**2 - 2.0 * noise_variance * x_2,
        0.0,
        -(1.0 + noise_variance) * x_1,
    ])
    theta_1 = np.real(roots[np.isclose(roots.imag, 0.0)])
    theta_2 = (x_2 - np.square(theta_1)) / (1.0 + noise_variance)
    candidates = torch.as_tensor(
        np.column_stack((theta_1, theta_2)), dtype=x_obs.dtype,
    )
    log_posterior = (
        banana_log_likelihood(candidates, x_obs)
        - 0.5 * candidates.square().sum(dim=1)
    )
    return candidates[torch.argmax(log_posterior)]


def sample_map(samples: torch.Tensor) -> torch.Tensor:
    """Estimate a two-dimensional sample-distribution MAP with a Gaussian KDE."""
    dimension = samples.shape[1]
    scale = samples.std(dim=0).mean().clamp_min(torch.finfo(samples.dtype).eps)
    bandwidth = scale * samples.shape[0] ** (-1.0 / (dimension + 4))
    distances = torch.cdist(samples, samples).square()
    density = torch.exp(-0.5 * distances / bandwidth.square()).sum(dim=1)
    return samples[torch.argmax(density)]


def banana_reference_samples(
    x_obs: torch.Tensor,
    num_samples: int,
    seed: int,
    cache_path: Path,
) -> torch.Tensor:
    """Draw an exact posterior reference by rejection from the Gaussian prior."""
    if cache_path.exists():
        payload = torch.load(
            cache_path, map_location="cpu", weights_only=True,
        )
        if (
            isinstance(payload, dict)
            and "observation" in payload
            and "samples" in payload
            and torch.allclose(payload["observation"], x_obs)
            and len(payload["samples"]) >= num_samples
        ):
            return payload["samples"][:num_samples]

    generator = torch.Generator().manual_seed(seed)
    accepted: list[torch.Tensor] = []
    accepted_count = 0
    proposal_count = 0
    while accepted_count < num_samples:
        remaining = num_samples - accepted_count
        batch_size = max(100_000, min(1_000_000, 20 * remaining))
        proposals = torch.randn(batch_size, 2, generator=generator)
        log_acceptance = banana_log_likelihood(proposals, x_obs)
        log_uniform = torch.log(torch.rand(batch_size, generator=generator))
        batch_accepted = proposals[log_uniform < log_acceptance]
        if len(batch_accepted):
            accepted.append(batch_accepted)
            accepted_count += len(batch_accepted)
        proposal_count += batch_size
        if proposal_count > 100_000_000 and accepted_count < num_samples:
            raise RuntimeError(
                "Banana rejection reference acceptance is unexpectedly low "
                f"for observation {x_obs.tolist()}."
            )
    samples = torch.cat(accepted, dim=0)[:num_samples]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "observation": x_obs.detach().cpu(),
        "samples": samples,
    }, cache_path)
    return samples


def sliced_wasserstein_distance(
    samples: torch.Tensor,
    reference: torch.Tensor,
    num_directions: int = 32,
    num_quantiles: int = 128,
) -> float:
    """Measure nonlinear shape error through deterministic 1D projections."""
    angles = torch.arange(num_directions, dtype=samples.dtype)
    angles = angles * torch.pi / num_directions
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    quantiles = torch.linspace(
        0.0, 1.0, num_quantiles, dtype=samples.dtype,
    )
    sample_quantiles = torch.quantile(
        samples @ directions.T, quantiles, dim=0,
    )
    reference_quantiles = torch.quantile(
        reference @ directions.T, quantiles, dim=0,
    )
    return torch.mean(torch.abs(sample_quantiles - reference_quantiles)).item()


def banana_sample_once(
    model: SBIm,
    x_obs_normalized: torch.Tensor,
    data: BananaData,
    variant: SamplerVariant,
    timesteps: int,
) -> tuple[torch.Tensor, float]:
    samples_normalized, elapsed = sample_once(
        model, x_obs_normalized, variant, timesteps,
    )
    return data.denormalize_theta(samples_normalized), elapsed


def run_banana_ablation(
    models: dict[tuple[str, int], SBIm],
    observations_normalized: torch.Tensor,
    references: list[torch.Tensor],
    data: BananaData,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    """Run the full sampler ablation against nonlinear posterior references."""
    rows: list[dict[str, object]] = []
    exemplars: dict[str, torch.Tensor] = {}
    reference_moments = [
        (reference.mean(0), torch.cov(reference.T))
        for reference in references
    ]
    reference_maps = [
        banana_reference_map(data.denormalize_x(x_obs.unsqueeze(0))[0])
        for x_obs in observations_normalized
    ]
    for (training_scheme, model_seed), model in models.items():
        for observation_index, x_obs in enumerate(observations_normalized):
            mean, covariance = reference_moments[observation_index]
            reference = references[observation_index]
            reference_map = reference_maps[observation_index]
            for variant in VARIANTS:
                for steps in RUN_TIMESTEPS:
                    for repeat in range(NUM_REPEATS):
                        torch.manual_seed(
                            SEED + 10_000 * observation_index
                            + 100 * steps + repeat
                        )
                        samples, elapsed = banana_sample_once(
                            model, x_obs, data, variant, steps,
                        )
                        (
                            mean_error, covariance_error,
                            posterior_width_ratio,
                        ) = posterior_errors(samples, mean, covariance)
                        estimated_map = sample_map(samples)
                        theta_map_squared_error = torch.sum(
                            (estimated_map - reference_map).square()
                        ).item()
                        observation_map_squared_error = torch.sum((
                            banana_observation_from_theta(estimated_map)
                            - banana_observation_from_theta(reference_map)
                        ).square()).item()
                        rows.append({
                            "training_scheme": training_scheme,
                            "variant": variant.key,
                            "variant_label": variant.label,
                            "family": variant.family,
                            "timesteps": steps,
                            "model_seed": model_seed,
                            "observation": observation_index,
                            "repeat": repeat,
                            "runtime_s": elapsed,
                            "runtime_per_sample_ms": (
                                elapsed * 1e3 / NUM_POSTERIOR_SAMPLES
                            ),
                            "runtime_device": DEVICE,
                            "mean_error": mean_error,
                            "covariance_error": covariance_error,
                            "posterior_width_ratio": posterior_width_ratio,
                            "theta_map_squared_error": theta_map_squared_error,
                            "observation_map_squared_error": (
                                observation_map_squared_error
                            ),
                            "sliced_wasserstein": (
                                sliced_wasserstein_distance(
                                    samples, reference,
                                )
                            ),
                        })
                        if (
                            model_seed == MODEL_SEEDS[0]
                            and observation_index == 0
                            and repeat == 0
                            and steps == 20
                        ):
                            exemplars[
                                f"{training_scheme}:{variant.key}"
                            ] = samples
    return pd.DataFrame(rows), exemplars


def evaluate_banana_training_score_accuracy(
    models: dict[tuple[str, int], SBIm],
    observations_normalized: torch.Tensor,
    references: list[torch.Tensor],
    data: BananaData,
    num_draws: int = 256,
    grid_points: int = 31,
) -> pd.DataFrame:
    """Compare learned scores with matched denoising-score targets over noise."""
    device = torch.device(DEVICE)
    reference_model = next(iter(models.values()))
    lam_min = reference_model.sde.lambda_t(torch.full((1,), 1e-3))
    lam_max = reference_model.sde.lambda_t(torch.ones(1))
    sigmas = torch.logspace(
        torch.log10(lam_min).item(), torch.log10(lam_max).item(),
        grid_points, device=device,
    )
    times = reference_model.sde.time_of_lambda(sigmas.cpu()).to(device)
    condition_mask = torch.tensor(
        [0.0, 0.0, 1.0, 1.0], device=device,
    ).repeat(num_draws, 1)
    rows: list[dict[str, object]] = []

    normalized_references = [
        data.normalize_theta(reference) for reference in references
    ]
    for (scheme, model_seed), model in models.items():
        model.model.to(device).eval()
        for observation_index, x_obs_cpu in enumerate(
            observations_normalized
        ):
            x_obs = x_obs_cpu.to(device)
            reference = normalized_references[observation_index]
            for grid_index, (sigma, time_value) in enumerate(
                zip(sigmas, times)
            ):
                seed = 50_000 + 1_000 * observation_index + grid_index
                cpu_generator = torch.Generator().manual_seed(seed)
                indices = torch.randint(
                    len(reference), (num_draws,), generator=cpu_generator,
                )
                clean = reference[indices].to(device)
                noise_generator = torch.Generator(device=device)
                noise_generator.manual_seed(seed + 1_000_000)
                standard_normal = torch.randn(
                    num_draws, 2, generator=noise_generator, device=device,
                )
                theta_t = clean + sigma * standard_normal
                denoising_target = -standard_normal / sigma
                state = torch.cat(
                    [theta_t, x_obs.repeat(num_draws, 1)], dim=1,
                )
                time_batch = time_value.reshape(1, 1).repeat(num_draws, 1)
                with torch.no_grad():
                    raw_score = model.model(
                        x=state, t=time_batch, c=condition_mask,
                    )
                    predicted_score = model.output_scale_function(
                        time_batch, raw_score,
                    )[:, :2]
                residual_power = torch.mean(torch.sum(
                    (predicted_score - denoising_target).square(), dim=1,
                ))
                target_power = torch.mean(torch.sum(
                    denoising_target.square(), dim=1,
                ))
                rows.append({
                    "training_scheme": scheme,
                    "model_seed": model_seed,
                    "observation": observation_index,
                    "sigma_m": sigma.item(),
                    "diffusion_time": time_value.item(),
                    "relative_score_rmse": torch.sqrt(
                        residual_power / target_power
                    ).item(),
                    "scaled_score_rmse": (
                        sigma * torch.sqrt(residual_power / 2)
                    ).item(),
                })
    return pd.DataFrame(rows)


def evaluate_banana_training_calibration(
    models: dict[tuple[str, int], SBIm],
    observations_normalized: torch.Tensor,
    references: list[torch.Tensor],
    data: BananaData,
) -> pd.DataFrame:
    """Measure generated marginal intervals against exact reference samples."""
    selected = tuple(
        item for item in VARIANTS
        if item.key in ("sigma_log_dpm", "dpm2_log")
    )
    rows: list[dict[str, object]] = []
    for (scheme, model_seed), model in models.items():
        for observation_index, x_obs in enumerate(observations_normalized):
            reference = references[observation_index]
            for variant in selected:
                for steps in CALIBRATION_TIMESTEPS:
                    torch.manual_seed(
                        SEED + 200_000 + 10_000 * observation_index
                        + 100 * steps + 1_000 * model_seed
                    )
                    samples, _ = banana_sample_once(
                        model, x_obs, data, variant, steps,
                    )
                    for nominal_coverage in CALIBRATION_LEVELS:
                        tail = (1.0 - nominal_coverage) / 2.0
                        lower = torch.quantile(samples, tail, dim=0)
                        upper = torch.quantile(samples, 1.0 - tail, dim=0)
                        reference_coverage = (
                            (reference >= lower)
                            & (reference <= upper)
                        ).float().mean(0)
                        rows.append({
                            "training_scheme": scheme,
                            "variant": variant.key,
                            "variant_label": variant.label,
                            "timesteps": steps,
                            "model_seed": model_seed,
                            "observation": observation_index,
                            "nominal_coverage": nominal_coverage,
                            "analytical_coverage": (
                                reference_coverage.mean().item()
                            ),
                        })
    return pd.DataFrame(rows)


def plot_banana_posterior_examples(
    exemplars: dict[str, torch.Tensor],
    reference: torch.Tensor,
    observation: torch.Tensor,
    output_path: Path,
) -> None:
    """Show the curved reference posterior beside principal sampler contrasts."""
    configure_plot_style()
    selected = [
        ("reference", "Exact rejection reference", reference),
        *[
            (
                f"mixture:{key}",
                label,
                exemplars[f"mixture:{key}"],
            )
            for key, label in PAIRPLOT_VARIANTS
        ],
    ]
    fig, axes = plt.subplots(
        2, 3, figsize=(12.5, 8.2), sharex=True, sharey=True,
        constrained_layout=True,
    )
    theta_1_limits = torch.quantile(
        reference[:, 0], torch.tensor([0.005, 0.995]),
    )
    ridge_theta_1 = torch.linspace(
        theta_1_limits[0], theta_1_limits[1], 300,
    )
    ridge_theta_2 = observation[1] - ridge_theta_1.square()
    for ax, (_, title, samples) in zip(axes.flat, selected):
        ax.scatter(
            samples[:, 0], samples[:, 1], s=4, color="#4C78A8",
            alpha=0.18, rasterized=True,
        )
        ax.plot(
            ridge_theta_1, ridge_theta_2, color="#E45756",
            linestyle="--", linewidth=1.4,
        )
        ax.set_title(title)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
    axes.flat[-1].axis("off")
    fig.suptitle(
        "Banana posterior at 20 sampling steps\n"
        "blue: samples; red: noiseless simulator ridge",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    output_dir: Path,
    problem_name: str,
    reference_description: str,
    score_description: str,
    shape_metric: bool = False,
) -> None:
    """Document each problem-specific result directory."""
    shape_text = (
        " Banana rows additionally report sliced_wasserstein, which compares "
        "the full curved distribution through 32 one-dimensional projections."
        if shape_metric else ""
    )
    shape_plot_text = (
        "shape_ablation.png plots that sliced-Wasserstein distance so curved-"
        "distribution quality is visible directly.\n\n"
        if shape_metric else ""
    )
    (output_dir / "README.md").write_text(
        f"# Fast sampling performance: {problem_name}\n\n"
        "raw_results.csv contains one row per training scheme, sampler variant, "
        "model seed, observation, step budget, and sampling seed. summary.csv "
        "reports median direct errors in the generated posterior mean and "
        "covariance, together with 10--90% bands. "
        f"The reference is {reference_description}.{shape_text}\n\n"
        "sampler_pairplot.png uses the same Seaborn pairplot style as "
        "gaussian_hypotheses_pairplot.png: KDE diagonals, small translucent "
        "scatter points, and a categorical sampler legend.\n\n"
        f"{shape_plot_text}"
        "training_time_sampling.png compares uniform and mixture diffusion-time "
        "training for sigma-space DPM-1 and DPM-2 on the log-noise grid. "
        f"The score diagnostic uses {score_description}.\n\n"
    )


def save_benchmark_tables(
    output_dir: Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    score_results: pd.DataFrame,
    score_summary: pd.DataFrame,
    calibration_results: pd.DataFrame,
    calibration_summary: pd.DataFrame,
) -> None:
    results.to_csv(output_dir / "raw_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    score_results.to_csv(
        output_dir / "training_score_accuracy.csv", index=False,
    )
    score_summary.to_csv(
        output_dir / "training_score_accuracy_summary.csv", index=False,
    )
    calibration_results.to_csv(
        output_dir / "training_calibration.csv", index=False,
    )
    calibration_summary.to_csv(
        output_dir / "training_calibration_summary.csv", index=False,
    )


def plot_common_outputs(
    output_dir: Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    banana: bool = False,
) -> None:
    plot_training_time_sampling(
        summary, output_dir / "training_time_sampling.png",
    )
    plot_training_noise_diagnostic(
        score_summary,
        summary,
        output_dir / "training_noise_diagnostic.png",
        score_panel_title=(
            "Denoising-score target error"
            if banana else "Analytical posterior-score accuracy"
        ),
        figure_title=(
            "What mixture time sampling changes — banana posterior"
            if banana else "What mixture time sampling changes — Gaussian posterior"
        ),
    )
    plot_training_calibration(
        calibration_summary,
        output_dir / "training_calibration_comparison.png",
        coverage_ylabel=(
            "Reference posterior mass in predicted interval"
            if banana
            else "Analytical posterior mass in predicted interval"
        ),
        figure_title=(
            "Banana posterior calibration: uniform vs mixture time training"
            if banana
            else "Gaussian posterior calibration: uniform vs mixture time training"
        ),
    )
    plot_dpm_ablation(summary, output_dir / "dpm_ablation.png")
    plot_euler_ablation(summary, output_dir / "euler_ablation.png")
    plot_pareto(summary, output_dir / "pareto.png")
    if banana:
        plot_dpm2_training_grid_pareto(
            summary,
            output_dir / "dpm2_training_grid_pareto.png",
        )
    if banana and {
        "theta_map_squared_error", "observation_map_squared_error",
    }.issubset(summary.columns):
        plot_banana_map_pareto(summary, output_dir / "pareto_map.png")
    plot_error_distributions(
        results, output_dir / "low_step_distributions.png",
    )
    if banana:
        plot_banana_shape_ablation(
            summary, output_dir / "shape_ablation.png",
        )


def run_gaussian_benchmark(output_dir: Path) -> None:
    """Run and save the original closed-form Gaussian benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_overview.png").unlink(missing_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    theta_train, x_train = simulate(TRAIN_SAMPLES)
    theta_val, x_val = simulate(VALIDATION_SAMPLES)
    _, observations = simulate(NUM_OBSERVATIONS)
    calibration_generator = torch.Generator().manual_seed(SEED + 20_000)
    calibration_observations = (
        torch.randn(CALIBRATION_OBSERVATIONS, 2, generator=calibration_generator)
        @ A_MIX.T
        + NOISE_STD * torch.randn(
            CALIBRATION_OBSERVATIONS, 2, generator=calibration_generator,
        )
    )

    models = {
        (scheme, model_seed): train_model(
            scheme, theta_train, x_train, theta_val, x_val,
            output_dir, model_seed,
        )
        for scheme in TRAINING_TIME_SCHEMES
        for model_seed in MODEL_SEEDS
    }
    results, exemplars = run_ablation(models, observations)
    summary = summarise(results)
    score_results = evaluate_training_score_accuracy(models, observations)
    score_summary = summarise_training_score_accuracy(score_results)
    calibration_results = evaluate_training_calibration(
        models, calibration_observations,
    )
    calibration_summary = summarise_training_calibration(calibration_results)
    save_benchmark_tables(
        output_dir, results, summary, score_results, score_summary,
        calibration_results, calibration_summary,
    )
    plot_common_outputs(
        output_dir, results, summary, score_summary, calibration_summary,
    )
    plot_posterior_examples(
        exemplars, observations[0], output_dir / "posterior_examples.png",
    )
    pairplot_generator = torch.Generator().manual_seed(SEED + 400_000)
    pairplot_mean, pairplot_covariance = analytic_posterior(observations[0])
    pairplot_reference = (
        torch.randn(2_000, 2, generator=pairplot_generator)
        @ torch.linalg.cholesky(pairplot_covariance).T
        + pairplot_mean
    )
    plot_sampler_pairplot(
        exemplars,
        pairplot_reference,
        output_dir / "sampler_pairplot.png",
        "Gaussian posterior: reference and sampler variants",
    )
    write_readme(
        output_dir,
        "Gaussian",
        "the closed-form conditional Gaussian posterior",
        "the exact conditional score at every marginal noise scale",
    )
    print(f"Saved Gaussian benchmark tables and figures to {output_dir}")


def run_banana_benchmark(output_dir: Path) -> None:
    """Run the same suite on the nonlinear Banana_posterior.ipynb problem."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED + 1_000)
    np.random.seed(SEED + 1_000)
    data = make_banana_data()
    observation_generator = torch.Generator().manual_seed(SEED + 10_000)
    _, observations = simulate_banana(
        NUM_OBSERVATIONS, generator=observation_generator,
    )
    calibration_generator = torch.Generator().manual_seed(SEED + 20_000)
    _, calibration_observations = simulate_banana(
        CALIBRATION_OBSERVATIONS, generator=calibration_generator,
    )
    observations_normalized = data.normalize_x(observations)
    calibration_observations_normalized = data.normalize_x(
        calibration_observations,
    )
    pd.DataFrame(
        observations.numpy(), columns=["x_1", "x_2"],
    ).to_csv(output_dir / "benchmark_observations.csv", index=False)
    pd.DataFrame({
        "quantity": ["theta_mean_1", "theta_mean_2", "theta_std_1",
                     "theta_std_2", "x_mean_1", "x_mean_2",
                     "x_std_1", "x_std_2"],
        "value": torch.cat([
            data.theta_mean, data.theta_std, data.x_mean, data.x_std,
        ]).numpy(),
    }).to_csv(output_dir / "normalization.csv", index=False)

    reference_dir = output_dir / "references"
    references = [
        banana_reference_samples(
            observation,
            BANANA_REFERENCE_SAMPLES,
            SEED + 300_000 + index,
            reference_dir / f"observation_{index}.pt",
        )
        for index, observation in enumerate(observations)
    ]
    calibration_references = [
        banana_reference_samples(
            observation,
            BANANA_REFERENCE_SAMPLES,
            SEED + 350_000 + index,
            reference_dir / f"calibration_observation_{index}.pt",
        )
        for index, observation in enumerate(calibration_observations)
    ]
    models = {
        (scheme, model_seed): train_banana_model(
            scheme, data, output_dir, model_seed,
        )
        for scheme in DPM2_TRAINING_TIME_SCHEMES
        for model_seed in MODEL_SEEDS
    }
    results, exemplars = run_banana_ablation(
        models, observations_normalized, references, data,
    )
    summary = summarise(results)
    score_results = evaluate_banana_training_score_accuracy(
        models, observations_normalized, references, data,
    )
    score_summary = summarise_training_score_accuracy(score_results)
    calibration_results = evaluate_banana_training_calibration(
        models,
        calibration_observations_normalized,
        calibration_references,
        data,
    )
    calibration_summary = summarise_training_calibration(
        calibration_results,
    )
    save_benchmark_tables(
        output_dir, results, summary, score_results, score_summary,
        calibration_results, calibration_summary,
    )
    plot_common_outputs(
        output_dir, results, summary, score_summary, calibration_summary,
        banana=True,
    )
    plot_banana_posterior_examples(
        exemplars, references[0], observations[0],
        output_dir / "posterior_examples.png",
    )
    plot_sampler_pairplot(
        exemplars,
        references[0],
        output_dir / "sampler_pairplot.png",
        "Banana posterior: reference and sampler variants",
    )
    write_readme(
        output_dir,
        "Banana",
        "exact rejection sampling from the Gaussian prior using likelihood-only acceptance",
        "matched denoising-score targets drawn from the exact reference posterior",
        shape_metric=True,
    )
    print(f"Saved Banana benchmark tables and figures to {output_dir}")


def replot_saved_outputs() -> None:
    """Regenerate metric-derived figures from saved CSVs without sampling."""
    for output_dir, banana in (
        (GAUSSIAN_OUTPUT_DIR, False),
        (BANANA_OUTPUT_DIR, True),
    ):
        results = pd.read_csv(output_dir / "raw_results.csv")
        results = results[results.variant.isin({item.key for item in VARIANTS})]
        summary = summarise(results)
        score_summary = pd.read_csv(
            output_dir / "training_score_accuracy_summary.csv"
        )
        calibration_summary = pd.read_csv(
            output_dir / "training_calibration_summary.csv"
        )
        plot_common_outputs(
            output_dir, results, summary, score_summary, calibration_summary,
            banana=banana,
        )
        # Posterior draws were not saved with the results, so these two figures
        # cannot be regenerated without running the sampler again.
        for filename in ("posterior_examples.png", "sampler_pairplot.png"):
            (output_dir / filename).unlink(missing_ok=True)
        write_readme(
            output_dir,
            "Banana" if banana else "Gaussian",
            (
                "exact rejection sampling from the Gaussian prior using "
                "likelihood-only acceptance"
                if banana else "the closed-form conditional Gaussian posterior"
            ),
            (
                "matched denoising-score targets drawn from the exact reference posterior"
                if banana else "the exact conditional score at every marginal noise scale"
            ),
            shape_metric=banana,
        )
        print(f"Regenerated metric-derived figures in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replot-only", action="store_true",
        help="Regenerate metric-derived figures from saved CSVs without sampling.",
    )
    args = parser.parse_args()
    if args.replot_only:
        replot_saved_outputs()
        return

    global DEVICE
    configure_plot_style()
    DEVICE = select_device()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "README.md").write_text(
        "# Fast sampling performance\n\n"
        "Gaussian outputs are stored in gaussian/. Banana-posterior outputs "
        "are stored in banana/. Run tutorials/fast_sampling_performance.py "
        "to regenerate both suites. Files directly in this directory "
        "predate the split and are not read by the current benchmark.\n"
    )
    run_gaussian_benchmark(GAUSSIAN_OUTPUT_DIR)
    run_banana_benchmark(BANANA_OUTPUT_DIR)


if __name__ == "__main__":
    main()
