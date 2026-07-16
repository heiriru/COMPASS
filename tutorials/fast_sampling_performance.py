#!/usr/bin/env python3
"""Controlled fast-sampling ablations for the linear-Gaussian COMPASS toy problem.

Run without arguments:

    python tutorials/fast_sampling_performance.py

Outputs are written to ``output/fast_sampling_performance``.  The experiment
trains matched models with uniform and mixture diffusion-time sampling, then
uses several observations and sampling seeds to quantify the effects of:

* mixture training-time sampling;
* sigma-space rather than time-space DPM prediction;
* a log-sigma rather than uniform-time integration grid; and
* the stochastic term in Euler--Maruyama.

The ``legacy`` variants below are explicit reference implementations in this
tutorial.  They reproduce the prior update *classes* for a controlled ablation;
they are not public COMPASS sampler modes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
import types
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import torch

try:
    from autocvd import autocvd
except ImportError:  # Allows a CPU-only Python environment to run the tutorial.
    autocvd = None

from compass import ScoreBasedInferenceModel as SBIm


# The script intentionally has no required command-line inputs.
OUTPUT_DIR = Path("output/fast_sampling_performance")
DEVICE = "cpu"  # Updated by select_device() before any model is created.
SEED = 0
TRAIN_SAMPLES = 20_000
VALIDATION_SAMPLES = 2_000
MAX_EPOCHS = 80
BATCH_SIZE = 256
EARLY_STOPPING_PATIENCE = 10
# Independent initialisations are necessary before interpreting a training-time
# sampling difference as an effect rather than a lucky/unlucky model fit.
MODEL_SEEDS = (0, 1, 2)
NUM_OBSERVATIONS = 4
NUM_REPEATS = 2
NUM_POSTERIOR_SAMPLES = 750
TIMESTEPS = (5, 10, 20, 50)

A_MIX = torch.tensor([[1.0, 0.5], [0.3, 1.0]])
NOISE_STD = 0.3
COLORS = {
    "uniform": "#9C755F",
    "mixture": "#4C78A8",
    "legacy": "#E45756",
    "sigma": "#72B7B2",
    "log": "#54A24B",
    "euler": "#F58518",
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
    reference_update: str | None = None
    corrector_steps: int = 0
    color: str = "#4C78A8"
    marker: str = "o"


VARIANTS = (
    SamplerVariant(
        "legacy_time_dpm", "Legacy time-space DPM-1", "DPM predictor",
        "dpm", order=1, grid="uniform_t", reference_update="time_dpm",
        color="#E45756", marker="X",
    ),
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
        "legacy_drift_euler", "Legacy drift-only Euler", "Euler update",
        "euler", grid="uniform_t", reference_update="drift_euler",
        color="#9D755D", marker="P",
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


def uniform_time_grid(sampler: object) -> types.FunctionType:
    """Build a PFODE-compatible grid with nodes equally spaced in diffusion time."""
    def grid(timesteps: int, eps: float, device: str, descending: bool = False,
             dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        ts = torch.linspace(float(eps), 1.0, timesteps, device=device, dtype=dtype)
        if descending:
            ts = torch.flip(ts, dims=(0,))
        return sampler.sde.lambda_t(ts), ts
    return grid


def legacy_time_dpm_step(self: object, data: torch.Tensor, t: torch.Tensor,
                         t_next: torch.Tensor, condition_mask: torch.Tensor) -> torch.Tensor:
    """First-order probability-flow update in time coordinates (reference only)."""
    # For this VESDE, g(t)^2 = sigma_base ** (2t).  The current predictor uses
    # the exactly equivalent sigma-space integral rather than this t-space Euler
    # approximation, which is inaccurate on coarse grids.
    score = self._get_score(data, t, condition_mask, self.cfg_alpha)
    g_squared = self.sde.sigma.to(data.device) ** (2 * t)
    dt = t - t_next
    return data + 0.5 * dt * g_squared * score * (1 - condition_mask)


def legacy_drift_euler(self: object, data: torch.Tensor,
                       condition_mask: torch.Tensor) -> torch.Tensor:
    """Pre-improvement Euler reference: correct drift, deliberately no noise."""
    for index in range(self.timesteps - 1):
        t = self.timesteps_list[index].reshape(-1, 1)
        t_next = self.timesteps_list[index + 1].reshape(-1, 1)
        score = self._get_score(data, t, condition_mask, self.cfg_alpha)
        dvar = self.sde.lambda_t(t)**2 - self.sde.lambda_t(t_next)**2
        data = data + dvar * score * (1 - condition_mask)
    return data.detach()


@contextmanager
def sampler_reference_mode(model: SBIm, variant: SamplerVariant) -> Iterator[None]:
    """Temporarily install one tutorial-only grid/update reference implementation."""
    sampler = model.sampler
    original_grid = sampler.pfode.lambda_grid
    original_dpm = sampler._dpm_solver_1_step
    original_euler = sampler._basic_sampler
    try:
        if variant.grid == "uniform_t":
            sampler.pfode.lambda_grid = uniform_time_grid(sampler)
        if variant.reference_update == "time_dpm":
            sampler._dpm_solver_1_step = types.MethodType(legacy_time_dpm_step, sampler)
        if variant.reference_update == "drift_euler":
            sampler._basic_sampler = types.MethodType(legacy_drift_euler, sampler)
        yield
    finally:
        sampler.pfode.lambda_grid = original_grid
        sampler._dpm_solver_1_step = original_dpm
        sampler._basic_sampler = original_euler


def posterior_errors(samples: torch.Tensor, mean: torch.Tensor,
                     covariance: torch.Tensor) -> tuple[float, float, float]:
    """Return mean/covariance errors and symmetrised Gaussian KL divergence.

    The exact posterior is Gaussian.  Evaluating the Gaussian fitted to each
    sampled posterior gives a proper distributional metric: it strongly
    penalises both an over-dispersed posterior and the variance collapse caused
    by the legacy drift-only Euler update.
    """
    mean_error = torch.linalg.vector_norm(samples.mean(0) - mean).item()
    empirical_mean = samples.mean(0)
    empirical_covariance = torch.cov(samples.T)
    covariance_error = torch.linalg.matrix_norm(empirical_covariance - covariance).item()
    dimension = mean.numel()
    # The ridge is negligible at the posterior scale but prevents a numerical
    # singularity if a pathological sampler collapses almost completely.
    ridge = 1e-6 * torch.eye(dimension, dtype=samples.dtype)
    empirical_covariance = empirical_covariance + ridge
    covariance = covariance + ridge

    def gaussian_kl(first_mean: torch.Tensor, first_covariance: torch.Tensor,
                    second_mean: torch.Tensor, second_covariance: torch.Tensor) -> torch.Tensor:
        delta = second_mean - first_mean
        precision_delta = torch.linalg.solve(second_covariance, delta)
        trace = torch.trace(torch.linalg.solve(second_covariance, first_covariance))
        return 0.5 * (
            trace + delta @ precision_delta - dimension
            + torch.linalg.slogdet(second_covariance).logabsdet
            - torch.linalg.slogdet(first_covariance).logabsdet
        )

    symmetric_kl = 0.5 * (
        gaussian_kl(empirical_mean, empirical_covariance, mean, covariance)
        + gaussian_kl(mean, covariance, empirical_mean, empirical_covariance)
    )
    return mean_error, covariance_error, symmetric_kl.item()


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
    with sampler_reference_mode(model, variant):
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
                for steps in TIMESTEPS:
                    for repeat in range(NUM_REPEATS):
                        torch.manual_seed(SEED + 10_000 * observation_index + 100 * steps + repeat)
                        samples, elapsed = sample_once(model, x_obs, variant, steps)
                        mean_error, covariance_error, posterior_error = posterior_errors(samples, mean, covariance)
                        rows.append({
                            "training_scheme": training_scheme, "variant": variant.key,
                            "variant_label": variant.label, "family": variant.family,
                            "timesteps": steps, "model_seed": model_seed,
                            "observation": observation_index,
                            "repeat": repeat, "runtime_s": elapsed,
                            "runtime_per_sample_ms": elapsed * 1e3 / NUM_POSTERIOR_SAMPLES,
                            "mean_error": mean_error, "covariance_error": covariance_error,
                            "posterior_error": posterior_error,
                        })
                        if model_seed == MODEL_SEEDS[0] and observation_index == 0 and repeat == 0 and steps == 20:
                            exemplars[f"{training_scheme}:{variant.key}"] = samples
    return pd.DataFrame(rows), exemplars


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Median and 10--90% interval, preserving one row per plotted condition."""
    group_columns = ["training_scheme", "variant", "variant_label", "family", "timesteps"]
    grouped = results.groupby(group_columns, as_index=False)
    summary = grouped.agg(
        posterior_error_median=("posterior_error", "median"),
        posterior_error_lo=("posterior_error", lambda x: x.quantile(0.1)),
        posterior_error_hi=("posterior_error", lambda x: x.quantile(0.9)),
        runtime_s_median=("runtime_s", "median"),
        mean_error_median=("mean_error", "median"),
        covariance_error_median=("covariance_error", "median"),
    )
    return summary


def line_with_band(ax: plt.Axes, data: pd.DataFrame, label: str, color: str,
                   marker: str = "o") -> None:
    """Draw an error curve with its observation/repetition uncertainty band."""
    data = data.sort_values("timesteps")
    x = data["timesteps"].to_numpy()
    y = data["posterior_error_median"].to_numpy()
    ax.plot(x, y, label=label, color=color, marker=marker)
    ax.fill_between(x, data["posterior_error_lo"], data["posterior_error_hi"], color=color, alpha=0.14, linewidth=0)


def finish_error_axis(ax: plt.Axes, title: str) -> None:
    """Apply shared labels for every separate ablation figure."""
    ax.set(title=title, xlabel="Sampling steps", ylabel="Symmetrised Gaussian KL", xscale="log")
    ax.legend(loc="best", frameon=True)


def plot_training_time_sampling(summary: pd.DataFrame, output_path: Path) -> None:
    """Compare training-time distributions using the same current sampler."""
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
    for scheme, color in (("uniform", COLORS["uniform"]), ("mixture", COLORS["mixture"])):
        data = summary[(summary.training_scheme == scheme) & (summary.variant == "dpm2_log")]
        line_with_band(ax, data, f"{scheme.replace('_', ' ').title()} training", color, "D")
    finish_error_axis(ax, "Training-time sampling: matched DPM-2 + corrector sampler")
    ax.legend(title="Score-model training\n(three independent seeds)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_dpm_ablation(summary: pd.DataFrame, output_path: Path) -> None:
    """Separate DPM update/grid plot, including the default order-two sampler."""
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    dpm_keys = {
        "legacy_time_dpm", "sigma_uniform_dpm", "sigma_log_dpm",
        "dpm2_uniform", "dpm2_log",
    }
    for variant in (item for item in VARIANTS if item.key in dpm_keys):
        data = summary[(summary.training_scheme == "mixture") & (summary.variant == variant.key)]
        line_with_band(ax, data, variant.label, variant.color, variant.marker)
    finish_error_axis(ax, "DPM update and time-grid ablation (mixture-trained models)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_euler_ablation(summary: pd.DataFrame, output_path: Path) -> None:
    """Separate Euler update/grid plot using a divergence that penalises collapse."""
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.3, 5.4), constrained_layout=True)
    for variant in (item for item in VARIANTS if item.family == "Euler update"):
        data = summary[(summary.training_scheme == "mixture") & (summary.variant == variant.key)]
        line_with_band(ax, data, variant.label, variant.color, variant.marker)
    finish_error_axis(ax, "Euler update and time-grid ablation (mixture-trained models)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot runtime/accuracy trade-offs with a unique colour for every variant."""
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    for variant in VARIANTS:
        data = summary[(summary.training_scheme == "mixture") & (summary.variant == variant.key)]
        ax.plot(data.runtime_s_median, data.posterior_error_median, label=variant.label,
                color=variant.color, marker=variant.marker)
    ax.set(title="Accuracy--runtime Pareto view (mixture-trained models)", xlabel="Median sampling time [s]", ylabel="Symmetrised Gaussian KL", xscale="log", yscale="log")
    ax.legend(loc="upper right", fontsize=7.5, frameon=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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
    """Show intuitive posterior samples for the principal legacy/new contrasts."""
    configure_plot_style()
    selected = [
        ("mixture:legacy_time_dpm", "Legacy time-space DPM-1"),
        ("mixture:sigma_log_dpm", "Sigma-space DPM-1 + log sigma"),
        ("mixture:legacy_drift_euler", "Legacy drift-only Euler"),
        ("mixture:em_log", "Euler--Maruyama + log sigma"),
    ]
    mean, covariance = analytic_posterior(observation)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, selected):
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


def plot_error_distributions(results: pd.DataFrame, output_path: Path) -> None:
    """Show repeat/observation variability at the low-budget 10-step stress test."""
    configure_plot_style()
    selected = ["legacy_time_dpm", "sigma_log_dpm", "legacy_drift_euler", "em_log", "dpm2_log"]
    labels = [next(v.label for v in VARIANTS if v.key == key) for key in selected]
    data = results[(results.training_scheme == "mixture") & (results.timesteps == 10)]
    values = [data[data.variant == key].posterior_error.to_numpy() for key in selected]
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    box = ax.boxplot(values, labels=labels, patch_artist=True, showfliers=False, medianprops={"color": "#202020", "linewidth": 1.5})
    for patch, key in zip(box["boxes"], selected):
        patch.set(facecolor=next(v.color for v in VARIANTS if v.key == key), alpha=0.70)
    ax.set(title="Low-step stress test: variation across observations and sampling seeds", ylabel="Symmetrised Gaussian KL")
    ax.tick_params(axis="x", rotation=16)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_readme(output_dir: Path) -> None:
    """Document the output columns and the causal interpretation of each ablation."""
    (output_dir / "README.md").write_text(
        "# Fast sampling performance outputs\n\n"
        "`raw_results.csv` contains one row per training scheme, sampler variant, "
        "model seed, observation, step budget, and sampling seed. `summary.csv` "
        "reports the median symmetrised Gaussian KL and 10--90% bands across those "
        "rows.\n\n"
        "The legacy time-space DPM and drift-only Euler settings are tutorial-only "
        "reference implementations. They permit one-change-at-a-time comparisons "
        "against the current sigma-space DPM and Euler--Maruyama implementations.\n"
    )


def main() -> None:
    global DEVICE
    configure_plot_style()
    DEVICE = select_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # This file belonged to the previous combined-figure layout; remove it so a
    # run of the current script presents only the separate requested figures.
    (OUTPUT_DIR / "ablation_overview.png").unlink(missing_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    theta_train, x_train = simulate(TRAIN_SAMPLES)
    theta_val, x_val = simulate(VALIDATION_SAMPLES)
    _, observations = simulate(NUM_OBSERVATIONS)

    models = {
        (scheme, model_seed): train_model(
            scheme, theta_train, x_train, theta_val, x_val, OUTPUT_DIR, model_seed,
        )
        for scheme in ("uniform", "mixture")
        for model_seed in MODEL_SEEDS
    }
    results, exemplars = run_ablation(models, observations)
    summary = summarise(results)
    results.to_csv(OUTPUT_DIR / "raw_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    plot_training_time_sampling(summary, OUTPUT_DIR / "training_time_sampling.png")
    plot_dpm_ablation(summary, OUTPUT_DIR / "dpm_ablation.png")
    plot_euler_ablation(summary, OUTPUT_DIR / "euler_ablation.png")
    plot_pareto(summary, OUTPUT_DIR / "pareto.png")
    plot_posterior_examples(exemplars, observations[0], OUTPUT_DIR / "posterior_examples.png")
    plot_error_distributions(results, OUTPUT_DIR / "low_step_distributions.png")
    write_readme(OUTPUT_DIR)
    print(f"Saved benchmark tables and figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
