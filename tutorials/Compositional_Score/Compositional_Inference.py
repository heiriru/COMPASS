#!/usr/bin/env python3
"""Validation experiments for compositional (multi-observation) inference.

This is the script counterpart of ``Compositional_Inference.ipynb``.  It is
deliberately an experiment suite rather than a linear notebook export: each
experiment has a quantitative reference, writes a CSV table, saves every raw
plot input in a compressed ``raw_plot_data.npz`` archive, and writes one or more
plots under ``tutorials/output/compositional_inference``.

The suite answers eight questions:

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
7. Can deliberately small models infer a mean that is global both when a set
   of Gaussian observations is created and during inference?  For a mean of
   -4, three training/sampling configurations are compared at N=5,10,25,50.
8. Can COMPASS recover a shared geometric offset and local curve positions for
   unshifted, vertically shifted, and horizontally shifted parabola models,
   and can it select the generating geometry?

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

import os


CPU_USAGE_LIMIT_FRACTION = 0.06
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_cpu_usage_limit(fraction: float = CPU_USAGE_LIMIT_FRACTION) -> tuple[int, tuple[int, ...]]:
    """Restrict this script and inherited children to at most 6% of host CPUs."""
    logical_cpus = os.cpu_count() or 1
    cpu_limit = int(logical_cpus * fraction)
    if cpu_limit < 1:
        raise RuntimeError(
            f"Cannot enforce a {fraction:.1%} CPU limit on a {logical_cpus}-CPU host."
        )
    allowed_cpus = sorted(os.sched_getaffinity(0))
    selected_cpus = tuple(allowed_cpus[:cpu_limit])
    if not selected_cpus:
        raise RuntimeError("The process has no CPUs available in its affinity mask.")
    os.sched_setaffinity(0, selected_cpus)
    for variable in CPU_THREAD_ENV_VARS:
        os.environ[variable] = str(len(selected_cpus))
    return logical_cpus, selected_cpus


CPU_LIMIT_INFO = None
if __name__ == "__main__":
    CPU_LIMIT_INFO = configure_cpu_usage_limit()


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
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, norm, wasserstein_distance
import torch

try:
    from autocvd import autocvd
except ImportError:
    autocvd = None

from compass import ScoreBasedInferenceModel as SBIm
from compass import ModelTransfuser as MTf


ROOT = Path(__file__).resolve().parent
# Keep outputs under tutorials/output regardless of the current working directory.
DEFAULT_OUTPUT = ROOT.parent / "output" / "compositional_inference"

# Shared Gaussian toy problem from the notebook.
MU0, S0, SX = 0.0, 1.0, 1.0

# Shared + local Gaussian hierarchy.
MUG, S0G, S0L, SXH = 0.0, 1.0, 1.0, 0.5

# Population model: x_i ~ Normal(mu, variance), with Gaussian priors on mu/log variance.
POP_MU_MEAN, POP_MU_STD = 0.0, 1.5
POP_LOGV_MEAN, POP_LOGV_STD = math.log(0.7**2), 0.7

TIME_SAMPLINGS = ("uniform", "log_sigma", "mixture")
INFERENCE_TIME_GRIDS = ("log_sigma", "uniform_t")
N_VALUES = (1, 2, 5, 10, 25, 50, 100, 200)
PAIRPLOT_SAMPLES = 4_000

# Publication-quality settings for the coupled global/local score model.
# The versioned checkpoint key prevents reuse of the earlier 32-wide network.
SHARED_LOCAL_HQ_TRAIN_SAMPLES = 200_000
SHARED_LOCAL_HQ_VALIDATION_SAMPLES = 20_000
SHARED_LOCAL_HQ_MAX_EPOCHS = 300
SHARED_LOCAL_HQ_PATIENCE = 40
SHARED_LOCAL_HQ_BATCH_SIZE = 512
SHARED_LOCAL_HQ_LR = 3e-4
SHARED_LOCAL_HQ_MODEL_KWARGS = {
    "sde_type": "vesde", "sigma": 8.0, "hidden_size": 128,
    "depth": 6, "num_heads": 8, "mlp_ratio": 4,
}

# Deliberately small global-mean Gaussian test (experiment 07).
GAUSSIAN_TEST_MEANS = (-4.0, 4.0)
GAUSSIAN_TEST_OBSERVATION_STD = 1.0
GAUSSIAN_TEST_PRIOR_MEAN = 0.0
GAUSSIAN_TEST_PRIOR_STD = 5.0
GAUSSIAN_TEST_GROUP_SIZE = 20
GAUSSIAN_TEST_N_VALUES = (5, 10, 25, 50)

# Parabola recovery and model-selection test (experiment 08).
PARABOLA_FAMILIES = ("original", "vertical", "horizontal")
PARABOLA_LABELS = {
    "original": "Original",
    "vertical": "Vertical shift",
    "horizontal": "Horizontal shift",
}
PARABOLA_NOISE_STD = 0.1
PARABOLA_OFFSET_PRIOR_MEAN = 0.0
PARABOLA_OFFSET_PRIOR_STD = 1.0
PARABOLA_TEST_OFFSET = 10.0 * PARABOLA_NOISE_STD
PARABOLA_T_MIN, PARABOLA_T_MAX = -2.0, 2.0
PARABOLA_GROUP_SIZE = 50
PARABOLA_N_VALUES = (5, 10, 25, 50)


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


def checkpoint_dir(cfg: RunConfig, key: str) -> Path:
    """Keep quick smoke-test checkpoints separate from full-quality checkpoints."""
    quality = "quick" if cfg.train_samples <= 4_000 and cfg.max_epochs <= 3 else "full"
    return cfg.output_dir / "models" / f"{key}_{quality}"


@dataclass(frozen=True)
class SamplerVariant:
    key: str
    label: str
    method: str
    correction: str
    corrector_steps: int
    snr: float = 0.1
    equation: str = "reverse_sde"


@dataclass(frozen=True)
class TimeSamplingSamplerVariant:
    """A controlled method/grid setting for the training-time ablation."""

    key: str
    label: str
    method: str
    correction: str
    corrector_steps: int
    inference_time_grid: str
    linestyle: object


TIME_SAMPLING_SAMPLER_VARIANTS = (
    TimeSamplingSamplerVariant(
        "dpm_gauss_log_sigma", "DPM-2 + Gaussian, log σ", "dpm", "gauss", 5,
        "log_sigma", "-",
    ),
    TimeSamplingSamplerVariant(
        "dpm_gauss_uniform_t", "DPM-2 + Gaussian, uniform t", "dpm", "gauss", 5,
        "uniform_t", (0, (5, 4)),
    ),
    TimeSamplingSamplerVariant(
        "langevin_gauss_log_sigma", "Langevin + Gaussian, log σ", "langevin", "gauss", 8,
        "log_sigma", "-",
    ),
    TimeSamplingSamplerVariant(
        "langevin_gauss_uniform_t", "Langevin + Gaussian, uniform t", "langevin", "gauss", 8,
        "uniform_t", (0, (5, 4)),
    ),
    TimeSamplingSamplerVariant(
        "langevin_fnpe_log_sigma", "Langevin + F-NPSE, log σ", "langevin", "fnpe", 8,
        "log_sigma", "-",
    ),
    TimeSamplingSamplerVariant(
        "langevin_fnpe_uniform_t", "Langevin + F-NPSE, uniform t", "langevin", "fnpe", 8,
        "uniform_t", (0, (5, 4)),
    ),
)
TRAINING_TIME_COLORS = {
    "uniform": "tab:blue",
    "log_sigma": "tab:orange",
    "mixture": "tab:green",
}


SAMPLER_VARIANTS = (
    SamplerVariant("dpm_gauss", "DPM-2 + Gaussian", "dpm", "gauss", 5),
    SamplerVariant("dpm_gauss_full", "DPM-2 + full Gaussian", "dpm", "gauss_full", 5),
    SamplerVariant("dpm_raw", "DPM-2 + uncorrected", "dpm", "uncorrected", 5),
    SamplerVariant("langevin_gauss", "Langevin + Gaussian", "langevin", "gauss", 8),
    SamplerVariant("langevin_raw", "Langevin + uncorrected", "langevin", "uncorrected", 8),
    # F-NPSE is a bridging-density score and is valid only with annealed Langevin.
    SamplerVariant("langevin_fnpe", "Langevin + F-NPSE", "langevin", "fnpe", 8),
)

PFODE_GAUSS_VARIANT = SamplerVariant(
    "pfode_gauss", "PF-ODE + Gaussian", "heun", "gauss", 0,
    equation="probability_flow_ode",
)

# The observation-scaling plot compares PF-ODE + Gaussian in place of the
# otherwise redundant DPM-2 + full Gaussian curve.
OBSERVATION_SCALING_VARIANTS = (
    SAMPLER_VARIANTS[0],
    PFODE_GAUSS_VARIANT,
    *SAMPLER_VARIANTS[2:],
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


def save_problem_pairplot(path: Path, data: dict[str, object], title: str) -> None:
    """Save a dense corner pairplot of the generative problem being inferred."""
    variables = list(data)
    arrays = {name: np.asarray(values, dtype=float).reshape(-1) for name, values in data.items()}
    if len(variables) < 2 or min(map(len, arrays.values())) < 2:
        raise ValueError("A problem pairplot needs at least two variables and two rows.")
    n = min(PAIRPLOT_SAMPLES, min(map(len, arrays.values())))
    arrays = {name: values[:n] for name, values in arrays.items()}
    n_variables = len(variables)
    fig, axes = plt.subplots(
        n_variables, n_variables,
        figsize=(3.8 * n_variables, 3.8 * n_variables),
        squeeze=False,
    )
    color = "#5B9BD5"
    for row, y_name in enumerate(variables):
        for column, x_name in enumerate(variables):
            axis = axes[row, column]
            if row < column:
                axis.axis("off")
                continue
            if row == column:
                axis.hist(
                    arrays[x_name], bins=35, color=color, alpha=0.85,
                    edgecolor="black", linewidth=0.8,
                )
            else:
                axis.scatter(
                    arrays[x_name], arrays[y_name],
                    s=8, alpha=0.35, color=color, edgecolors="none",
                )
            axis.grid(True, color="0.90", linewidth=0.8)
            axis.set_axisbelow(True)
            if row == n_variables - 1:
                axis.set_xlabel(x_name)
            else:
                axis.tick_params(labelbottom=False)
            if column == 0:
                axis.set_ylabel(y_name)
            else:
                axis.tick_params(labelleft=False)
    fig.suptitle(title, y=0.995)
    save_figure(fig, path)


def parabola_mean_numpy(
    t: np.ndarray, b: np.ndarray | float, family: str,
) -> np.ndarray:
    """Return x=(x1,x2) for one of the three parabola hypotheses."""
    t_values, b_values = np.broadcast_arrays(
        np.asarray(t, dtype=float), np.asarray(b, dtype=float),
    )
    if family == "original":
        return np.stack([t_values, t_values**2], axis=-1)
    if family == "vertical":
        return np.stack([t_values, t_values**2 + b_values], axis=-1)
    if family == "horizontal":
        return np.stack([t_values + b_values, t_values**2], axis=-1)
    raise ValueError(f"Unknown parabola family {family!r}")


def plot_parabola_data_separation(path: Path) -> None:
    """Visualize matched noisy data and the 10-sigma translated curves."""
    rng = np.random.default_rng(7)
    t_curve = np.linspace(PARABOLA_T_MIN, PARABOLA_T_MAX, 500)
    t_data = rng.uniform(PARABOLA_T_MIN, PARABOLA_T_MAX, 400)
    noise = PARABOLA_NOISE_STD * rng.normal(size=(len(t_data), 2))
    colors = {"original": "black", "vertical": "tab:blue", "horizontal": "tab:orange"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    for axis, family in zip(axes, PARABOLA_FAMILIES):
        offset = 0.0 if family == "original" else PARABOLA_TEST_OFFSET
        curve = parabola_mean_numpy(t_curve, offset, family)
        observations = parabola_mean_numpy(t_data, offset, family) + noise
        axis.plot(curve[:, 0], curve[:, 1], color=colors[family], linewidth=2.2)
        axis.scatter(observations[:, 0], observations[:, 1], s=8, alpha=0.25,
                     color=colors[family], edgecolors="none")
        axis.set(title=f"{PARABOLA_LABELS[family]}: b={offset:g}",
                 xlabel="x1", ylabel="x2")
        axis.set_aspect("equal", adjustable="box")
    fig.suptitle(
        f"Parabola hypotheses: shifted cases are {PARABOLA_TEST_OFFSET / PARABOLA_NOISE_STD:g}σ from baseline"
    )
    save_figure(fig, path)



def save_problem_pairplot_for_experiment(output_dir: Path, experiment: str) -> None:
    """Plot the known prior-predictive problem, independent of inference results."""
    rng = np.random.default_rng(7)
    n = PAIRPLOT_SAMPLES
    path = output_dir / "problem_pairplot.png"

    if experiment == "parabola":
        plot_parabola_data_separation(path)
        return
    if experiment in {"contract", "samplers", "scaling", "time", "time-grid", "individual"}:
        theta = rng.normal(MU0, S0, n)
        observations = theta + SX * rng.normal(size=n)
        data = {"true theta": theta, "observed x": observations}
        titles = {
            "contract": "Contract: shared parameter and observations",
            "samplers": "Sampler comparison: shared parameter and observations",
            "scaling": "Observation scaling: shared parameter and observations",
            "time": "Training-time sampling: shared parameter and observations",
            "time-grid": "Training/inference-time sampling: shared parameter and observations",
            "individual": "Multi-observation inference: shared parameter and observations",
        }
        title = titles[experiment]
    elif experiment == "population":
        mu = rng.normal(POP_MU_MEAN, POP_MU_STD, n)
        log_variance = rng.normal(POP_LOGV_MEAN, POP_LOGV_STD, n)
        observations = mu + np.exp(0.5 * log_variance) * rng.normal(size=n)
        data = {
            "population mu": mu,
            "log population variance": log_variance,
            "observed x": observations,
        }
        title = "Population model: global parameters and observations"
    elif experiment == "hierarchy":
        global_parameter = rng.normal(MUG, S0G, n)
        local_parameter = rng.normal(0.0, S0L, n)
        observations = global_parameter + local_parameter + SXH * rng.normal(size=n)
        data = {
            "global parameter": global_parameter,
            "local parameter": local_parameter,
            "observed x": observations,
        }
        title = "Hierarchical model: global/local parameters and observations"
    elif experiment == "gaussian":
        global_means = np.full(n, GAUSSIAN_TEST_MEANS[0])
        observations = global_means + GAUSSIAN_TEST_OBSERVATION_STD * rng.normal(
            size=len(global_means)
        )
        data = {
            "global mean": global_means,
            "observed x": observations,
        }
        title = "Global-mean Gaussian test: Model 1 mean=-4"
    else:
        raise ValueError(f"Unknown pairplot experiment {experiment!r}")
    save_problem_pairplot(path, data, title)


PAIRPLOT_FOLDERS = {
    "contract": "01_contract",
    "samplers": "02_sampler_comparison",
    "scaling": "03_observation_scaling",
    "time": "04_time_sampling",
    "time-grid": "04_time_sampling",
    "individual": "05_multi_vs_individual",
    "population": "06a_population_globals",
    "hierarchy": "06b_shared_local",
    "gaussian": "07_gaussian_test",
    "parabola": "08_parabola",
}


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


def simulate_grouped_global_gaussian(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create flattened training pairs from sets that share one global mean.

    Each group first draws one mean from the training prior, repeats that exact
    value for all observations in the group, and then draws unit-variance
    observations. Flattening is required by the ordinary SBI training API; the
    repeated values ensure the parameter is genuinely global at creation.
    """
    groups = math.ceil(n / GAUSSIAN_TEST_GROUP_SIZE)
    group_means = (
        GAUSSIAN_TEST_PRIOR_MEAN
        + GAUSSIAN_TEST_PRIOR_STD * torch.randn(groups, 1)
    )
    theta = torch.repeat_interleave(
        group_means, GAUSSIAN_TEST_GROUP_SIZE, dim=0,
    )[:n]
    x = theta + GAUSSIAN_TEST_OBSERVATION_STD * torch.randn(n, 1)
    return theta, x


def simulate_mtf_global_gaussian(
    n: int,
    centre: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create grouped data for one MTF candidate's global mean offset."""
    groups = math.ceil(n / GAUSSIAN_TEST_GROUP_SIZE)
    global_offset = torch.randn(groups, 1)
    theta = torch.repeat_interleave(
        global_offset, GAUSSIAN_TEST_GROUP_SIZE, dim=0,
    )[:n]
    x = centre + theta + GAUSSIAN_TEST_OBSERVATION_STD * torch.randn(n, 1)
    return theta, x


def parabola_mean_torch(t: torch.Tensor, b: torch.Tensor, family: str) -> torch.Tensor:
    """Torch equivalent of :func:`parabola_mean_numpy`."""
    if family == "original":
        return torch.cat([t, t.square()], dim=1)
    if family == "vertical":
        return torch.cat([t, t.square() + b], dim=1)
    if family == "horizontal":
        return torch.cat([t + b, t.square()], dim=1)
    raise ValueError(f"Unknown parabola family {family!r}")


def simulate_parabola_pairs(n: int, family: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Create flattened single-pair training data with grouped global offsets."""
    if family not in PARABOLA_FAMILIES:
        raise ValueError(f"Unknown parabola family {family!r}")
    t = PARABOLA_T_MIN + (PARABOLA_T_MAX - PARABOLA_T_MIN) * torch.rand(n, 1)
    if family == "original":
        b = torch.zeros(n, 1)
        theta = t
    else:
        groups = math.ceil(n / PARABOLA_GROUP_SIZE)
        group_offsets = (
            PARABOLA_OFFSET_PRIOR_MEAN
            + PARABOLA_OFFSET_PRIOR_STD * torch.randn(groups, 1)
        )
        b = torch.repeat_interleave(group_offsets, PARABOLA_GROUP_SIZE, dim=0)[:n]
        theta = torch.cat([b, t], dim=1)
    x = parabola_mean_torch(t, b, family) + PARABOLA_NOISE_STD * torch.randn(n, 2)
    return theta, x


def matched_parabola_cases(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Generate matched latent positions/noise for the three test hypotheses."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(PARABOLA_T_MIN, PARABOLA_T_MAX, n)
    noise = PARABOLA_NOISE_STD * rng.normal(size=(n, 2))
    observations = {}
    for family in PARABOLA_FAMILIES:
        b = 0.0 if family == "original" else PARABOLA_TEST_OFFSET
        observations[family] = parabola_mean_numpy(t, b, family) + noise
    return t, noise, observations


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


def build_gaussian_test_model(device: str) -> SBIm:
    """Smallest practical transformer used by the one-dimensional test."""
    return SBIm(
        nodes_size=2,  # one global parameter plus one observed value
        sde_type="vesde",
        sigma=8.0,
        hidden_size=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2,
        device=device,
    )


def load_or_train_gaussian_test(cfg: RunConfig, time_sampling: str) -> SBIm:
    """Load or train Model 1's compact network for one time-sampling scheme."""
    if time_sampling not in {"uniform", "mixture"}:
        raise ValueError(f"Unsupported Gaussian-test time sampling: {time_sampling}")
    key = (
        "gaussian_model_1_global_mean"
        if time_sampling == "mixture"
        else "gaussian_model_1_global_mean_uniform"
    )
    model_dir = checkpoint_dir(cfg, key)
    checkpoint = model_dir / "Model_checkpoint.pt"
    if checkpoint.exists():
        print(f"Loading Gaussian Model 1 ({time_sampling}): {checkpoint}")
        return SBIm.load(str(checkpoint), device=cfg.device)

    seed_all(cfg.seed + 1_400)
    theta_train, x_train = simulate_grouped_global_gaussian(cfg.train_samples)
    theta_val, x_val = simulate_grouped_global_gaussian(cfg.validation_samples)
    model = build_gaussian_test_model(cfg.device)
    print(
        f"Training Gaussian Model 1 (compact, {time_sampling}) on {cfg.device}..."
    )
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


def load_or_train_gaussian_mtf_candidate(
    cfg: RunConfig,
    model_index: int,
    centre: float,
) -> SBIm:
    """Train one compact candidate likelihood for ModelTransfuser."""
    model_dir = checkpoint_dir(cfg, f"gaussian_mtf_model_{model_index}")
    checkpoint = model_dir / "Model_checkpoint.pt"
    if checkpoint.exists():
        print(f"Loading Gaussian MTF Model {model_index}: {checkpoint}")
        return SBIm.load(str(checkpoint), device=cfg.device)

    seed_all(cfg.seed + 1_700 + model_index)
    theta_train, x_train = simulate_mtf_global_gaussian(
        cfg.train_samples, centre,
    )
    theta_val, x_val = simulate_mtf_global_gaussian(
        cfg.validation_samples, centre,
    )
    model = build_gaussian_test_model(cfg.device)
    print(
        f"Training Gaussian MTF Model {model_index} "
        f"(centre={centre:g}) on {cfg.device}..."
    )
    started = time.perf_counter()
    model.train(
        theta=theta_train,
        x=x_train,
        theta_val=theta_val,
        x_val=x_val,
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.patience,
        time_sampling="mixture",
        device=cfg.device,
        verbose=False,
        path=str(model_dir),
    )
    print(f"  trained in {(time.perf_counter() - started) / 60:.1f} minutes")
    return model


def build_parabola_model(family: str, device: str) -> SBIm:
    """Use the divergence-head parabola architecture with the required latent width."""
    return SBIm(
        nodes_size=3 if family == "original" else 4,
        sde_type="vesde",
        sigma=3.0,
        hidden_size=40,
        depth=4,
        num_heads=4,
        mlp_ratio=4,
        device=device,
    )


def load_or_train_parabola_models(cfg: RunConfig) -> dict[str, SBIm]:
    """Load or train one surrogate for each parabola hypothesis."""
    models = {}
    quick_suffix = "_quick" if cfg.train_samples <= 4_000 and cfg.max_epochs <= 3 else ""
    for family_index, family in enumerate(PARABOLA_FAMILIES):
        key = f"parabola_{family}_mixture{quick_suffix}"
        model_dir = checkpoint_dir(cfg, key)
        checkpoint = model_dir / "Model_checkpoint.pt"
        if checkpoint.exists():
            print(f"Loading {PARABOLA_LABELS[family]} parabola: {checkpoint}")
            models[family] = SBIm.load(str(checkpoint), device=cfg.device)
            continue
        seed_all(cfg.seed + 1_800 + family_index)
        theta_train, x_train = simulate_parabola_pairs(cfg.train_samples, family)
        theta_val, x_val = simulate_parabola_pairs(cfg.validation_samples, family)
        model = build_parabola_model(family, cfg.device)
        print(f"Training {PARABOLA_LABELS[family]} parabola on {cfg.device}...")
        started = time.perf_counter()
        model.train(
            theta=theta_train,
            x=x_train,
            theta_val=theta_val,
            x_val=x_val,
            batch_size=cfg.batch_size,
            max_epochs=cfg.max_epochs,
            early_stopping_patience=cfg.patience,
            time_sampling="mixture",
            device=cfg.device,
            verbose=False,
            path=str(model_dir),
        )
        print(f"  trained in {(time.perf_counter() - started) / 60:.1f} minutes")
        models[family] = model
    return models


def load_or_train(
    cfg: RunConfig,
    key: str,
    nodes: int,
    simulator: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    time_sampling: str = "mixture",
    hierarchical: bool = False,
) -> SBIm:
    model_dir = checkpoint_dir(cfg, key)
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


def load_or_train_shared_local(cfg: RunConfig) -> SBIm:
    """Load or train the high-capacity score model for p(g, l | x)."""
    if cfg.train_samples <= 4_000 and cfg.max_epochs <= 3:
        return load_or_train(
            cfg, "shared_local_mixture", 3, simulate_hierarchical_pairs,
            time_sampling="mixture", hierarchical=True,
        )

    model_dir = checkpoint_dir(cfg, "shared_local_mixture_hq_v1")
    checkpoint = model_dir / "Model_checkpoint.pt"
    if checkpoint.exists():
        print(f"Loading high-quality shared/local model: {checkpoint}")
        return SBIm.load(str(checkpoint), device=cfg.device)

    seed_all(cfg.seed + 1_190)
    theta_train, x_train = simulate_hierarchical_pairs(SHARED_LOCAL_HQ_TRAIN_SAMPLES)
    theta_val, x_val = simulate_hierarchical_pairs(SHARED_LOCAL_HQ_VALIDATION_SAMPLES)
    model = SBIm(nodes_size=3, device=cfg.device, **SHARED_LOCAL_HQ_MODEL_KWARGS)
    print(
        "Training publication-quality shared/local score model "
        f"on {SHARED_LOCAL_HQ_TRAIN_SAMPLES:,} simulations..."
    )
    started = time.perf_counter()
    model.train(
        theta=theta_train, x=x_train, theta_val=theta_val, x_val=x_val,
        batch_size=SHARED_LOCAL_HQ_BATCH_SIZE,
        max_epochs=SHARED_LOCAL_HQ_MAX_EPOCHS,
        early_stopping_patience=SHARED_LOCAL_HQ_PATIENCE,
        lr=SHARED_LOCAL_HQ_LR, time_sampling="mixture",
        device=cfg.device, verbose=False, path=str(model_dir),
    )
    print(f"  trained in {(time.perf_counter() - started) / 60:.1f} minutes")
    return model


def known_shared_precision(n: int = 1) -> torch.Tensor:
    value = 1.0 / S0**2 + 1.0 / SX**2
    return torch.full((n, 1), value)


def known_shared_precision_full(n: int = 1) -> torch.Tensor:
    return torch.diag_embed(known_shared_precision(n))


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
            None if automatic_precision or variant.correction not in ("gauss", "gauss_full")
            else (
                known_shared_precision(x.shape[0]) if variant.correction == "gauss"
                else known_shared_precision_full(x.shape[0])
            )
        ),
        precision_est_samples=min(750, cfg.posterior_samples),
        num_samples=cfg.posterior_samples,
        timesteps=cfg.timesteps,
        order=2,
        method=variant.method,
        equation=variant.equation,
        corrector_steps=variant.corrector_steps,
        snr=variant.snr,
        device=cfg.device,
        verbose=False,
    )
    return samples, time.perf_counter() - started


def sample_shared_time_grid(
    model: SBIm,
    x: torch.Tensor,
    cfg: RunConfig,
    variant: TimeSamplingSamplerVariant,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Sample with the requested composition method and inference-time grid."""
    seed_all(seed)
    started = time.perf_counter()
    samples = model.sample(
        x=x,
        multi_obs_inference=True,
        hierarchy=[0],
        prior=([MU0], [S0]),
        correction=variant.correction,
        posterior_precision=(
            known_shared_precision(x.shape[0]) if variant.correction == "gauss" else None
        ),
        num_samples=cfg.posterior_samples,
        timesteps=cfg.timesteps,
        order=2,
        method=variant.method,
        corrector_steps=variant.corrector_steps,
        snr=0.1,
        inference_time_grid=variant.inference_time_grid,
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
    variants: Sequence[SamplerVariant] = SAMPLER_VARIANTS,
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
            all_seed_variants = (*SAMPLER_VARIANTS, PFODE_GAUSS_VARIANT)
            for variant in variants:
                index = next(
                    index for index, candidate in enumerate(all_seed_variants)
                    if candidate.key == variant.key
                )
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    axes = axes.ravel()
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


def experiment_observation_scaling(
    model: SBIm,
    cfg: RunConfig,
    variants: Sequence[SamplerVariant] = OBSERVATION_SCALING_VARIANTS,
    incremental: bool = False,
) -> None:
    out = cfg.output_dir / "03_observation_scaling"
    rows, raw = evaluate_sampler_grid(
        model, cfg, n_values=N_VALUES, repeats=cfg.repeats, variants=variants,
    )
    metrics_path = out / "observation_scaling_metrics.csv"
    raw_path = out / "raw_plot_data.npz"
    if incremental:
        if not metrics_path.exists() or not raw_path.exists():
            raise FileNotFoundError(
                "--sampler-variants updates an existing scaling run, but its CSV/NPZ "
                f"files are missing under {out}."
            )
        selected_keys = {variant.key for variant in variants}
        previous_rows = read_metric_rows(metrics_path)
        rows = [
            row for row in previous_rows if str(row["variant"]) not in selected_keys
        ] + rows
        raw = {**load_raw_data(raw_path), **raw}
    write_rows(metrics_path, rows)
    save_raw_data(raw_path, **raw)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    axes = axes.ravel()
    runtime_linestyles = ("-", "-", "-", "-", "-", (0, (1, 6)))
    for variant_index, variant in enumerate(OBSERVATION_SCALING_VARIANTS):
        chosen = [row for row in rows if row["variant"] == variant.key]
        for axis_index, (ax, value) in enumerate(
            zip(axes[:3], ("mean_error_sigma", "std_ratio", "runtime_seconds"))
        ):
            summary = aggregate(chosen, ("n_observations",), value)
            xs = np.array(N_VALUES)
            ys = np.array([summary[(n,)][0] for n in N_VALUES])
            es = np.array([summary[(n,)][1] for n in N_VALUES])
            linestyle = runtime_linestyles[variant_index] if axis_index == 2 else "-"
            plot_kwargs = {"linestyle": linestyle}
            if axis_index == 2 and variant_index == 5:
                plot_kwargs.update(
                    linewidth=mpl.rcParams["lines.linewidth"] * 1.4,
                    dash_capstyle="round",
                )
            line, = ax.plot(xs, ys, marker="o", label=variant.label, **plot_kwargs)
            ax.fill_between(xs, ys - es, ys + es, color=line.get_color(), alpha=0.12)
        error_summary = aggregate(chosen, ("n_observations",), "mean_error_sigma")
        runtime_summary = aggregate(chosen, ("n_observations",), "runtime_seconds")
        mean_errors = np.asarray([error_summary[(n,)][0] for n in N_VALUES])
        runtimes = np.asarray([runtime_summary[(n,)][0] for n in N_VALUES])
        tradeoff_kwargs = {"linestyle": runtime_linestyles[variant_index]}
        if variant_index == 5:
            tradeoff_kwargs.update(
                linewidth=mpl.rcParams["lines.linewidth"] * 1.4,
                dash_capstyle="round",
            )
        tradeoff_line, = axes[3].plot(
            runtimes, mean_errors, marker="o", label=variant.label, **tradeoff_kwargs
        )
        for n, runtime, mean_error in zip(N_VALUES, runtimes, mean_errors):
            axes[3].annotate(
                str(n), (runtime, mean_error), xytext=(4, 3),
                textcoords="offset points", color=tradeoff_line.get_color(),
                fontsize=5, clip_on=True,
            )
    axes[0].set(xscale="log", xlabel="observations N", ylabel="mean error / analytic std")
    axes[1].set(xscale="log", xlabel="observations N", ylabel="sample std / analytic std")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[2].set(xscale="log", yscale="log", xlabel="observations N", ylabel="runtime (s)")
    axes[3].set(
        xscale="log", yscale="log", xlabel="runtime (s)",
        ylabel="mean error / analytic std",
    )
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


def experiment_time_sampling_sampler_grid(
    models: dict[str, SBIm], cfg: RunConfig, out: Path,
) -> None:
    """Cross training-time draws with three composition methods and two grids."""
    rows: list[dict] = []
    raw: dict[str, np.ndarray] = {}
    for repeat in range(cfg.repeats):
        seed_all(cfg.seed + 50_000 + repeat)
        theta_true = float(MU0 + S0 * torch.randn(()))
        pool = shared_observations(max(N_VALUES), theta_true)
        raw[f"x_pool_r{repeat}"] = tensor_numpy(pool)
        raw[f"theta_true_r{repeat}"] = np.asarray(theta_true)
        for n in N_VALUES:
            x = pool[:n]
            truth_mean, truth_std = analytic_shared(x)
            for train_index, (training_time_sampling, model) in enumerate(models.items()):
                for variant_index, variant in enumerate(TIME_SAMPLING_SAMPLER_VARIANTS):
                    samples, runtime = sample_shared_time_grid(
                        model, x, cfg, variant,
                        cfg.seed + 500_000 * repeat + 10_000 * n
                        + 100 * train_index + variant_index,
                    )
                    row = scalar_metrics(samples[0, :, 0], truth_mean, truth_std)
                    row.update({
                        "repeat": repeat,
                        "n_observations": n,
                        "training_time_sampling": training_time_sampling,
                        "method": variant.method,
                        "correction": variant.correction,
                        "inference_time_grid": variant.inference_time_grid,
                        "variant": variant.key,
                        "label": variant.label,
                        "runtime_seconds": runtime,
                    })
                    rows.append(row)
                    raw[
                        f"samples_r{repeat}_n{n}_{training_time_sampling}_{variant.key}"
                    ] = tensor_numpy(samples[0, :, 0])
                    print(
                        f"N={n:>2} train={training_time_sampling:<9} "
                        f"{variant.label:<34} "
                        f"mean={row['mean_error_sigma']:.2f} sigma "
                        f"width={row['std_ratio']:.2f} time={runtime:.1f}s"
                    )
    write_rows(out / "training_time_sampler_grid_metrics.csv", rows)
    save_raw_data(out / "training_time_sampler_grid_raw_plot_data.npz", **raw)
    plot_time_sampling_sampler_grid(rows, out / "training_time_sampler_grid_effect.png")
    plot_selected_time_sampling_sampler_grid(
        rows, out / "training_time_sampler_grid_selected_effect.png",
    )


def plot_time_sampling_sampler_grid(rows: Sequence[dict], path: Path) -> None:
    """Plot each composition method as a row; colour=train draw and dash=grid."""
    fig, axes = plt.subplots(3, 4, figsize=(17, 11.0), constrained_layout=True)
    method_specs = (
        ("dpm", "gauss", "DPM-2 + Gaussian"),
        ("langevin", "gauss", "Langevin + Gaussian"),
        ("langevin", "fnpe", "Langevin + F-NPSE"),
    )
    metrics = (
        ("mean_error_sigma", "mean error / analytic std"),
        ("std_ratio", "sample std / analytic std"),
        ("runtime_seconds", "runtime (s)"),
    )
    for row_index, (method, correction, method_label) in enumerate(method_specs):
        method_rows = [
            row for row in rows
            if row["method"] == method and row["correction"] == correction
        ]
        for training_time_sampling in TIME_SAMPLINGS:
            for grid in INFERENCE_TIME_GRIDS:
                chosen = [
                    row for row in method_rows
                    if row["training_time_sampling"] == training_time_sampling
                    and row["inference_time_grid"] == grid
                ]
                variant = next(
                    variant for variant in TIME_SAMPLING_SAMPLER_VARIANTS
                    if variant.method == method and variant.correction == correction
                    and variant.inference_time_grid == grid
                )
                label = f"train: {training_time_sampling}; grid: {grid}"
                summaries = {
                    key: aggregate(chosen, ("n_observations",), key)
                    for key, _ in metrics
                }
                for axis_index, (key, _) in enumerate(metrics):
                    ys = np.asarray([summaries[key][(n,)][0] for n in N_VALUES])
                    errors = np.asarray([summaries[key][(n,)][1] for n in N_VALUES])
                    line, = axes[row_index, axis_index].plot(
                        N_VALUES, ys, marker="o",
                        color=TRAINING_TIME_COLORS[training_time_sampling],
                        linestyle=variant.linestyle, label=label,
                    )
                    axes[row_index, axis_index].fill_between(
                        N_VALUES, ys - errors, ys + errors,
                        color=line.get_color(), alpha=0.10,
                    )
                mean_errors = np.asarray([
                    summaries["mean_error_sigma"][(n,)][0] for n in N_VALUES
                ])
                runtimes = np.asarray([
                    summaries["runtime_seconds"][(n,)][0] for n in N_VALUES
                ])
                tradeoff_line, = axes[row_index, 3].plot(
                    runtimes, mean_errors, marker="o",
                    color=TRAINING_TIME_COLORS[training_time_sampling],
                    linestyle=variant.linestyle, label=label,
                )
                for n, runtime, mean_error in zip(N_VALUES, runtimes, mean_errors):
                    axes[row_index, 3].annotate(
                        str(n), (runtime, mean_error), xytext=(3, 3),
                        textcoords="offset points", color=tradeoff_line.get_color(),
                        fontsize=5, clip_on=True,
                    )
        for axis_index, (_, ylabel) in enumerate(metrics):
            axes[row_index, axis_index].set(
                xscale="log", xlabel="observations N", ylabel=ylabel,
                title=f"{method_label}: {ylabel}",
            )
        axes[row_index, 1].axhline(1.0, color="black", linestyle="--")
        axes[row_index, 2].set(yscale="log")
        axes[row_index, 3].set(
            xscale="log", yscale="log", xlabel="runtime (s)",
            ylabel="mean error / analytic std",
            title=f"{method_label}: accuracy/runtime",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, fontsize=8,
        title="colour = training-time sampling; line = inference-time grid",
    )
    save_figure(fig, path)


def plot_selected_time_sampling_sampler_grid(rows: Sequence[dict], path: Path) -> None:
    """Compare the two matched training/inference time-sampling settings."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    method_specs = (
        ("dpm", "gauss", "DPM-2 + Gaussian"),
        ("langevin", "gauss", "Langevin + Gaussian"),
        ("langevin", "fnpe", "Langevin + F-NPSE"),
    )
    settings = (
        ("mixture", "log_sigma", "train: mixture; grid: log σ", "-"),
        ("uniform", "uniform_t", "train: uniform; grid: uniform t", "--"),
    )
    metrics = (
        ("mean_error_sigma", "mean error / analytic std"),
        ("std_ratio", "sample std / analytic std"),
        ("runtime_seconds", "runtime (s)"),
    )
    series_colors = (
        "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
    )
    for method_index, (method, correction, method_label) in enumerate(method_specs):
        for setting_index, (training_time_sampling, grid, setting_label, linestyle) in enumerate(settings):
            chosen = [
                row for row in rows
                if row["method"] == method
                and row["correction"] == correction
                and row["training_time_sampling"] == training_time_sampling
                and row["inference_time_grid"] == grid
            ]
            label = f"{method_label}\n{setting_label}"
            summaries = {
                key: aggregate(chosen, ("n_observations",), key)
                for key, _ in metrics
            }
            color = series_colors[method_index * len(settings) + setting_index]
            for axis_index, (key, _) in enumerate(metrics):
                ys = np.asarray([summaries[key][(n,)][0] for n in N_VALUES])
                errors = np.asarray([summaries[key][(n,)][1] for n in N_VALUES])
                line, = axes[axis_index].plot(
                    N_VALUES, ys, marker="o", color=color, linestyle=linestyle,
                    label=label,
                )
                axes[axis_index].fill_between(
                    N_VALUES, ys - errors, ys + errors,
                    color=line.get_color(), alpha=0.07,
                )
            mean_errors = np.asarray([
                summaries["mean_error_sigma"][(n,)][0] for n in N_VALUES
            ])
            runtimes = np.asarray([
                summaries["runtime_seconds"][(n,)][0] for n in N_VALUES
            ])
            tradeoff_line, = axes[3].plot(
                runtimes, mean_errors, marker="o", color=color, linestyle=linestyle,
                label=label,
            )
            for n, runtime, mean_error in zip(N_VALUES, runtimes, mean_errors):
                axes[3].annotate(
                    str(n), (runtime, mean_error), xytext=(3, 3),
                    textcoords="offset points", color=tradeoff_line.get_color(),
                    fontsize=5, clip_on=True,
                )
    for axis_index, (_, ylabel) in enumerate(metrics):
        axes[axis_index].set(
            xscale="log", xlabel="observations N", ylabel=ylabel,
        )
        axes[axis_index].grid(True, which="major", color="0.88", linewidth=0.8)
        axes[axis_index].set_axisbelow(True)
    axes[0].set_title("Accuracy")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_title("Posterior width")
    axes[2].set(yscale="log")
    axes[2].set_title("Runtime")
    axes[3].set(
        xscale="log", yscale="log", xlabel="runtime (s)",
        ylabel="mean error / analytic std",
    )
    axes[3].set_title("Accuracy–runtime trade-off")
    axes[3].grid(True, which="major", color="0.88", linewidth=0.8)
    axes[3].set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=3, fontsize=7.5, frameon=False, columnspacing=2.4,
        title="Each colour identifies one method/setting combination",
        title_fontsize=8.5,
    )
    fig.subplots_adjust(bottom=0.31, left=0.05, right=0.995, top=0.90, wspace=0.16)
    save_figure(fig, path)


def experiment_multi_vs_individual(model: SBIm, cfg: RunConfig) -> None:
    out = cfg.output_dir / "05_multi_vs_individual"
    seed_all(cfg.seed + 700)
    n = 10
    theta_true = float(MU0 + S0 * torch.randn(()))
    x = shared_observations(n, theta_true)
    truth_mean, truth_std = analytic_shared(x)
    multi, multi_runtime = sample_shared(model, x, cfg, SAMPLER_VARIANTS[0], cfg.seed + 701)
    multi_values = tensor_numpy(multi[0, :, 0])
    multi_fnpe, multi_fnpe_runtime = sample_shared(
        model, x, cfg, SAMPLER_VARIANTS[5], cfg.seed + 702,
    )
    multi_fnpe_values = tensor_numpy(multi_fnpe[0, :, 0])

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
        "multi_fnpe_mean_error_sigma": abs(multi_fnpe_values.mean() - truth_mean) / truth_std,
        "multi_fnpe_std_ratio": multi_fnpe_values.std(ddof=1) / truth_std,
        "multi_fnpe_wasserstein_to_analytic": wasserstein_distance(
            multi_fnpe_values, analytic_draws,
        ),
        "multi_fnpe_wasserstein_to_mcmc": wasserstein_distance(
            multi_fnpe_values, mcmc_values,
        ),
        "multi_fnpe_runtime_seconds": multi_fnpe_runtime,
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
        multi_fnpe_samples=multi_fnpe_values,
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
            label="COMPASS multi-observation (DPM-2 + Gaussian)")
    ax.plot(grid, gaussian_kde(multi_fnpe_values)(grid), color="tab:green", linewidth=2.4,
            label="COMPASS multi-observation (Langevin + F-NPSE)")
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
    """Validate learned p(g, local | x) against its exact Gaussian posterior."""
    out = cfg.output_dir / "06b_shared_local"
    out.mkdir(parents=True, exist_ok=True)
    quick_run = cfg.train_samples <= 4_000 and cfg.max_epochs <= 3
    model_specification = {
        "generative_model": "g ~ N(0,1); local_i ~ N(0,1); x_i = g + local_i + epsilon_i",
        "observation_noise": "epsilon_i ~ N(0, 0.5^2)",
        "learned_conditional": "p(g, local_i | x_i)",
        "inference_target": "p(g, local_1, ..., local_N | x_1, ..., x_N)",
        "global_indices": [0], "local_indices": [1],
        "score_architecture": (
            {"sde_type": "vesde", "sigma": 8.0, "hidden_size": 32,
             "depth": 3, "num_heads": 4, "mlp_ratio": 2}
            if quick_run else SHARED_LOCAL_HQ_MODEL_KWARGS
        ),
        "training_simulations": (cfg.train_samples if quick_run else SHARED_LOCAL_HQ_TRAIN_SAMPLES),
        "validation_simulations": (cfg.validation_samples if quick_run else SHARED_LOCAL_HQ_VALIDATION_SAMPLES),
        "quality": "quick" if quick_run else "high",
        "training_time_sampling": "mixture",
        "note": "Quick mode deliberately uses a smaller smoke-test model.",
    }
    (out / "model_specification.json").write_text(
        json.dumps(model_specification, indent=2) + "\n"
    )
    model = load_or_train_shared_local(cfg)
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
    global_row = rows[0]
    local_rows = rows[1:]
    quality_summary = {
        "global_mean_error_in_analytic_std": float(
            abs(global_row["compass_mean"] - global_row["analytic_mean"])
            / global_row["analytic_std"]
        ),
        "global_std_ratio": float(global_row["compass_std"] / global_row["analytic_std"]),
        "local_mean_absolute_error_in_analytic_std": float(np.mean([
            abs(row["compass_mean"] - row["analytic_mean"]) / row["analytic_std"]
            for row in local_rows
        ])),
        "local_mean_std_ratio": float(np.mean([
            row["compass_std"] / row["analytic_std"] for row in local_rows
        ])),
        "shared_synchronization_max_abs": synchronization_error,
    }
    (out / "quality_summary.json").write_text(
        json.dumps(quality_summary, indent=2) + "\n"
    )
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

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    axes = axes.ravel()
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


def gaussian_test_analytic_posterior(x: torch.Tensor) -> tuple[float, float]:
    """Posterior moments for the shared mean under the experiment-07 prior."""
    prior_precision = 1.0 / GAUSSIAN_TEST_PRIOR_STD**2
    likelihood_precision = 1.0 / GAUSSIAN_TEST_OBSERVATION_STD**2
    precision = prior_precision + x.shape[0] * likelihood_precision
    mean = (
        GAUSSIAN_TEST_PRIOR_MEAN * prior_precision
        + x.sum().item() * likelihood_precision
    ) / precision
    return mean, precision**-0.5


def plot_gaussian_test(raw: dict[str, np.ndarray], path: Path) -> None:
    """Plot the three requested samplers by four observation counts."""
    samples = raw["posterior_samples"]
    true_mean = float(raw["true_global_mean"])
    n_values = raw["n_values"].astype(int)
    row_labels = [str(value) for value in raw["row_labels"]]
    analytic_means = raw["analytic_means"]
    analytic_stds = raw["analytic_stds"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5), sharex=False, sharey=False)
    for row_index, row_label in enumerate(row_labels):
        for column_index, n_observations in enumerate(n_values):
            axis = axes[row_index, column_index]
            posterior_grid = np.linspace(
                min(
                    samples[row_index, column_index].min(),
                    analytic_means[column_index] - 5 * analytic_stds[column_index],
                ),
                max(
                    samples[row_index, column_index].max(),
                    analytic_means[column_index] + 5 * analytic_stds[column_index],
                ),
                400,
            )
            axis.hist(
                samples[row_index, column_index],
                bins=42,
                density=True,
                alpha=0.62,
                label="COMPASS",
            )
            axis.plot(
                posterior_grid,
                norm.pdf(
                    posterior_grid,
                    analytic_means[column_index],
                    analytic_stds[column_index],
                ),
                "k--",
                linewidth=2,
                label="analytic posterior",
            )
            axis.axvline(true_mean, color="tab:red", linewidth=2, label="true mean")
            if row_index == 0:
                axis.set_title(f"N = {n_observations} observations")
            if column_index == 0:
                axis.set_ylabel(f"{row_label}\n\ndensity")
            if row_index == len(row_labels) - 1:
                axis.set_xlabel("global mean")
            if row_index == 0 and column_index == len(n_values) - 1:
                axis.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Global mean recovery for Model 1: "
        f"xᵢ ~ N(μ, 1), shared μ = {true_mean:g}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path)


def run_gaussian_mtf_comparisons(cfg: RunConfig) -> None:
    """Run MTF once for observations from each candidate Gaussian model."""
    out = cfg.output_dir / "07_gaussian_test" / "mtf_compare"
    candidate_names = ("Model 1", "Model 2")
    candidate_models = {
        name: load_or_train_gaussian_mtf_candidate(cfg, index, centre)
        for index, (name, centre) in enumerate(
            zip(candidate_names, GAUSSIAN_TEST_MEANS), start=1,
        )
    }
    summary_rows = []
    raw_observations = {}
    n_observations = max(GAUSSIAN_TEST_N_VALUES)

    for true_index, (true_name, centre) in enumerate(
        zip(candidate_names, GAUSSIAN_TEST_MEANS), start=1,
    ):
        case_dir = out / f"true_model_{true_index}"
        case_dir.mkdir(parents=True, exist_ok=True)
        seed_all(cfg.seed + 1_800 + true_index)
        global_offset = torch.zeros(n_observations, 1)
        observations = (
            centre
            + global_offset
            + GAUSSIAN_TEST_OBSERVATION_STD
            * torch.randn(n_observations, 1)
        )
        raw_observations[f"true_model_{true_index}"] = tensor_numpy(observations)

        mtf = MTf(path=str(case_dir))
        mtf.add_models(candidate_models)
        seed_all(cfg.seed + 1_900 + true_index)
        mtf.compare(
            x=observations,
            timesteps=cfg.timesteps,
            num_samples=min(cfg.posterior_samples, 1_000),
            multi_obs_inference=True,
            hierarchy=[0],
            prior=([0.0], [1.0]),
            correction="gauss",
            order=2,
            snr=0.1,
            corrector_steps_interval=1,
            corrector_steps=5,
            final_corrector_steps=3,
            device=cfg.device,
            verbose=False,
            method="dpm",
            equation="reverse_sde",
            likelihood_method="pfode",
            map_method="score",
            criterion="aic",
            log_prob_timesteps=cfg.timesteps,
            log_prob_divergence="exact",
        )
        seed_all(cfg.seed + 2_000 + true_index)
        mtf.plot_comparison(
            stats_dict=mtf.stats,
            path=str(case_dir),
            show=False,
            sort="none",
        )

        for candidate_name in candidate_names:
            stats = mtf.stats[candidate_name]
            map_values = np.asarray(stats["MAP"], dtype=float)
            summary_rows.append({
                "true_model": true_name,
                "candidate_model": candidate_name,
                "n_observations": n_observations,
                "true_centre": centre,
                "true_global_offset": 0.0,
                "model_probability": float(stats["model_prob"]),
                "aicc": float(torch.as_tensor(stats["AICc"])),
                "bic": float(torch.as_tensor(stats["BIC"])),
                "neg2_log_likelihood": float(
                    torch.as_tensor(stats["neg2_log_likelihood"])
                ),
                "mean_log_likelihood": float(
                    torch.as_tensor(stats["log_probs"]).mean()
                ),
                "mean_map_global_offset": float(map_values[:, 0, :].mean()),
                "mean_posterior_std": float(map_values[:, 1, :].mean()),
                "selected_as_best": (
                    stats["model_prob"]
                    == max(
                        mtf.stats[name]["model_prob"] for name in candidate_names
                    )
                ),
            })

    write_rows(out / "mtf_comparison_summary.csv", summary_rows)
    save_raw_data(out / "comparison_observations.npz", **raw_observations)


def experiment_gaussian_global(cfg: RunConfig) -> None:
    """Compare three composition settings as the observation count increases."""
    out = cfg.output_dir / "07_gaussian_test"
    models = {
        scheme: load_or_train_gaussian_test(cfg, scheme)
        for scheme in ("uniform", "mixture")
    }
    variants = (
        {
            "key": "dpm_gauss_uniform_uniform_t",
            "label": "DPM-2 + Gaussian\nuniform train / uniform t",
            "training": "uniform",
            "method": "dpm",
            "correction": "gauss",
            "grid": "uniform_t",
            "corrector_steps": 5,
        },
        {
            "key": "dpm_gauss_mixture_log_sigma",
            "label": "DPM-2 + Gaussian\nmixture train / log σ",
            "training": "mixture",
            "method": "dpm",
            "correction": "gauss",
            "grid": "log_sigma",
            "corrector_steps": 5,
        },
        {
            "key": "langevin_fnpe_mixture_log_sigma",
            "label": "Langevin + F-NPSE\nmixture train / log σ",
            "training": "mixture",
            "method": "langevin",
            "correction": "fnpe",
            "grid": "log_sigma",
            "corrector_steps": 8,
        },
    )
    rows = []
    posterior_samples = np.empty(
        (len(variants), len(GAUSSIAN_TEST_N_VALUES), cfg.posterior_samples),
        dtype=np.float32,
    )
    analytic_means = []
    analytic_stds = []
    prior_precision = 1.0 / GAUSSIAN_TEST_PRIOR_STD**2
    single_observation_precision = (
        prior_precision + 1.0 / GAUSSIAN_TEST_OBSERVATION_STD**2
    )

    true_mean = GAUSSIAN_TEST_MEANS[0]
    seed_all(cfg.seed + 1_500)
    global_mean = torch.full((max(GAUSSIAN_TEST_N_VALUES), 1), true_mean)
    all_observations = (
        global_mean
        + GAUSSIAN_TEST_OBSERVATION_STD
        * torch.randn(max(GAUSSIAN_TEST_N_VALUES), 1)
    )
    for n_observations in GAUSSIAN_TEST_N_VALUES:
        x = all_observations[:n_observations]
        analytic_mean, analytic_std = gaussian_test_analytic_posterior(x)
        analytic_means.append(analytic_mean)
        analytic_stds.append(analytic_std)
    for row_index, variant in enumerate(variants):
        model = models[variant["training"]]
        for column_index, n_observations in enumerate(GAUSSIAN_TEST_N_VALUES):
            x = all_observations[:n_observations]
            seed_all(cfg.seed + 1_510 + 100 * row_index + column_index)
            started = time.perf_counter()
            samples = model.sample(
                x=x,
                multi_obs_inference=True,
                hierarchy=[0],
                prior=([GAUSSIAN_TEST_PRIOR_MEAN], [GAUSSIAN_TEST_PRIOR_STD]),
                correction=variant["correction"],
                posterior_precision=(
                    torch.full((n_observations, 1), single_observation_precision)
                    if variant["correction"] == "gauss"
                    else None
                ),
                num_samples=cfg.posterior_samples,
                timesteps=cfg.timesteps,
                order=2,
                method=variant["method"],
                corrector_steps_interval=1,
                corrector_steps=variant["corrector_steps"],
                snr=0.1,
                inference_time_grid=variant["grid"],
                device=cfg.device,
                verbose=False,
            )
            runtime = time.perf_counter() - started
            shared_samples = tensor_numpy(samples[0, :, 0])
            posterior_samples[row_index, column_index] = shared_samples
            synchronization_error = float(
                (samples[:, :, 0] - samples[0:1, :, 0]).abs().max()
            )
            analytic_mean = analytic_means[column_index]
            analytic_std = analytic_stds[column_index]
            rows.append({
                "variant": variant["key"],
                "training_time_sampling": variant["training"],
                "method": variant["method"],
                "correction": variant["correction"],
                "inference_time_grid": variant["grid"],
                "true_global_mean": true_mean,
                "observation_variance": GAUSSIAN_TEST_OBSERVATION_STD**2,
                "n_observations": n_observations,
                "creation_global_mean_max_abs_deviation": float(
                    (global_mean[:n_observations] - global_mean[0]).abs().max()
                ),
                "analytic_mean": analytic_mean,
                "analytic_std": analytic_std,
                "compass_mean": float(shared_samples.mean()),
                "compass_std": float(shared_samples.std(ddof=1)),
                "mean_error_in_analytic_std": float(
                    abs(shared_samples.mean() - analytic_mean) / analytic_std
                ),
                "std_ratio_to_analytic": float(
                    shared_samples.std(ddof=1) / analytic_std
                ),
                "inference_shared_max_abs_deviation": synchronization_error,
                "runtime_seconds": runtime,
            })
            if synchronization_error > 1e-6:
                raise AssertionError(
                    f"{variant['key']} N={n_observations} samples are not "
                    f"synchronized: {synchronization_error}"
                )

    write_rows(out / "gaussian_global_metrics.csv", rows)
    raw = {
        "observations": tensor_numpy(all_observations).reshape(-1),
        "posterior_samples": posterior_samples,
        "true_global_mean": np.asarray(true_mean),
        "n_values": np.asarray(GAUSSIAN_TEST_N_VALUES),
        "row_labels": np.asarray([variant["label"] for variant in variants]),
        "analytic_means": np.asarray(analytic_means),
        "analytic_stds": np.asarray(analytic_stds),
    }
    save_raw_data(out / "raw_plot_data.npz", **raw)
    plot_gaussian_test(raw, out / "global_mean_recovery.png")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "experiment_config.json").open("w") as handle:
        json.dump({
            "trained_models": ["Model 1"],
            "training_time_samplings": ["uniform", "mixture"],
            "model_1_mean": GAUSSIAN_TEST_MEANS[0],
            "observation_variance": GAUSSIAN_TEST_OBSERVATION_STD**2,
            "mean_is_global_during_creation": True,
            "mean_is_global_during_inference": True,
            "training_group_size": GAUSSIAN_TEST_GROUP_SIZE,
            "n_observations": list(GAUSSIAN_TEST_N_VALUES),
            "variants": [
                {
                    "key": variant["key"],
                    "training_time_sampling": variant["training"],
                    "method": variant["method"],
                    "correction": variant["correction"],
                    "inference_time_grid": variant["grid"],
                }
                for variant in variants
            ],
            "network": {
                "nodes_size": 2,
                "hidden_size": 16,
                "depth": 2,
                "num_heads": 2,
                "mlp_ratio": 2,
            },
        }, handle, indent=2)
    run_gaussian_mtf_comparisons(cfg)


def parabola_quadrature(points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return t/b grids and normalized trapezoidal log-prior weights."""
    t_grid = np.linspace(PARABOLA_T_MIN, PARABOLA_T_MAX, points)
    t_step = t_grid[1] - t_grid[0]
    t_weights = np.full(points, t_step)
    t_weights[[0, -1]] *= 0.5
    log_t_weights = np.log(t_weights / (PARABOLA_T_MAX - PARABOLA_T_MIN))
    b_points = max(401, points // 2 * 2 + 1)
    b_grid = np.linspace(-5.0, 5.0, b_points)
    b_step = b_grid[1] - b_grid[0]
    b_weights = np.full(b_points, b_step)
    b_weights[[0, -1]] *= 0.5
    log_b_weights = norm.logpdf(
        b_grid, PARABOLA_OFFSET_PRIOR_MEAN, PARABOLA_OFFSET_PRIOR_STD,
    ) + np.log(b_weights)
    log_b_weights -= logsumexp(log_b_weights)
    return t_grid, log_t_weights, b_grid, log_b_weights


def parabola_log_likelihood_grid(
    observation: np.ndarray,
    family: str,
    t_grid: np.ndarray,
    b_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate log p(x|t,b,M) on the requested quadrature grid."""
    x = np.asarray(observation, dtype=float).reshape(1, 1, 2)
    if family == "original":
        means = parabola_mean_numpy(t_grid, 0.0, family).reshape(1, -1, 2)
    else:
        if b_grid is None:
            raise ValueError("Shifted parabola likelihoods require a b grid.")
        means = parabola_mean_numpy(t_grid[None, :], b_grid[:, None], family)
    residual = (x - means) / PARABOLA_NOISE_STD
    return (
        -0.5 * np.square(residual).sum(axis=-1)
        - 2.0 * math.log(PARABOLA_NOISE_STD * math.sqrt(2.0 * math.pi))
    )


def exact_parabola_evidence_series(
    observations: np.ndarray,
    family: str,
    n_values: Sequence[int],
    points: int,
) -> dict[int, float]:
    """Integrate local t and, where present, global b for every nested N."""
    t_grid, log_t_weights, b_grid, log_b_weights = parabola_quadrature(points)
    per_observation = []
    for observation in observations[:max(n_values)]:
        log_likelihood = parabola_log_likelihood_grid(
            observation, family, t_grid, None if family == "original" else b_grid,
        )
        per_observation.append(logsumexp(log_likelihood + log_t_weights, axis=-1))
    integrated = np.stack(per_observation, axis=-1)
    results = {}
    for n in n_values:
        cumulative = integrated[..., :n].sum(axis=-1)
        results[int(n)] = float(
            np.asarray(cumulative).squeeze() if family == "original"
            else logsumexp(log_b_weights + cumulative)
        )
    if not all(np.isfinite(list(results.values()))):
        raise AssertionError(f"Non-finite exact evidence for {family}.")
    return results


def exact_parabola_posterior(
    observations: np.ndarray, family: str, points: int,
) -> dict[str, np.ndarray | float]:
    """Numerically integrate posterior moments for global b and every local t."""
    t_grid, log_t_weights, b_grid, log_b_weights = parabola_quadrature(points)
    observations = np.asarray(observations, dtype=float)
    if family == "original":
        local_means, local_stds, log_terms = [], [], []
        for observation in observations:
            log_likelihood = parabola_log_likelihood_grid(
                observation, family, t_grid,
            ).reshape(-1)
            log_joint = log_t_weights + log_likelihood
            log_normalizer = logsumexp(log_joint)
            weights = np.exp(log_joint - log_normalizer)
            mean = float(np.sum(weights * t_grid))
            variance = float(np.sum(weights * np.square(t_grid - mean)))
            local_means.append(mean)
            local_stds.append(math.sqrt(max(variance, 0.0)))
            log_terms.append(log_normalizer)
        return {
            "log_evidence": float(np.sum(log_terms)), "b_mean": np.nan,
            "b_std": np.nan, "local_means": np.asarray(local_means),
            "local_stds": np.asarray(local_stds),
        }
    full_log_likelihoods, integrated = [], []
    for observation in observations:
        log_likelihood = parabola_log_likelihood_grid(
            observation, family, t_grid, b_grid,
        )
        full_log_likelihoods.append(log_likelihood)
        integrated.append(logsumexp(log_likelihood + log_t_weights, axis=1))
    integrated_array = np.stack(integrated, axis=1)
    total_by_b = integrated_array.sum(axis=1)
    log_evidence = float(logsumexp(log_b_weights + total_by_b))
    b_weights = np.exp(log_b_weights + total_by_b - log_evidence)
    b_mean = float(np.sum(b_weights * b_grid))
    b_variance = float(np.sum(b_weights * np.square(b_grid - b_mean)))
    local_means, local_stds = [], []
    for index, log_likelihood in enumerate(full_log_likelihoods):
        other_observations = total_by_b - integrated_array[:, index]
        log_joint = (
            log_b_weights[:, None] + log_t_weights[None, :]
            + log_likelihood + other_observations[:, None]
        )
        weights = np.exp(log_joint - logsumexp(log_joint))
        mean = float(np.sum(weights * t_grid[None, :]))
        variance = float(np.sum(weights * np.square(t_grid[None, :] - mean)))
        local_means.append(mean)
        local_stds.append(math.sqrt(max(variance, 0.0)))
    return {
        "log_evidence": log_evidence, "b_mean": b_mean,
        "b_std": math.sqrt(max(b_variance, 0.0)),
        "local_means": np.asarray(local_means),
        "local_stds": np.asarray(local_stds),
    }


def infer_parabola_candidate(
    model: SBIm,
    observations: np.ndarray,
    family: str,
    cfg: RunConfig,
    seed: int,
) -> dict[str, object]:
    """Infer candidate parameters, estimate their MAP, and evaluate learned likelihoods."""
    x = torch.as_tensor(observations, dtype=torch.float32)
    seed_all(seed)
    started = time.perf_counter()
    if family == "original":
        samples = model.sample(
            x=x, num_samples=cfg.posterior_samples, timesteps=cfg.timesteps,
            order=2, method="dpm", device=cfg.device, verbose=False,
        )
        theta_mean = samples.mean(dim=1)
        joint_init = torch.cat([theta_mean, x.to(samples.device)], dim=1)
        map_mask = torch.tensor([0.0, 1.0, 1.0])
    else:
        samples = model.sample(
            x=x, multi_obs_inference=True, hierarchy=[0],
            prior=([PARABOLA_OFFSET_PRIOR_MEAN], [PARABOLA_OFFSET_PRIOR_STD]),
            correction="gauss", posterior_precision=None,
            precision_est_samples=min(500, cfg.posterior_samples),
            precision_est_timesteps=cfg.timesteps,
            num_samples=cfg.posterior_samples, timesteps=cfg.timesteps,
            order=2, method="dpm", corrector_steps_interval=1,
            corrector_steps=10, snr=0.2, device=cfg.device, verbose=False,
        )
        shared_b = samples[0, :, 0].mean().expand(len(x), 1)
        local_mean = samples[:, :, 1].mean(dim=1, keepdim=True)
        joint_init = torch.cat([shared_b, local_mean, x.to(samples.device)], dim=1)
        map_mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    joint_map = model.map_estimate(
        joint_init, map_mask, sigma_start=0.5,
        timesteps=max(12, cfg.timesteps // 2), iterations_per_level=2,
        device=cfg.device,
    )
    theta_width = 1 if family == "original" else 2
    theta_map = joint_map[:, :theta_width]
    joint_evaluation = torch.cat([theta_map, x.to(theta_map.device)], dim=1)
    likelihood_mask = torch.cat([torch.ones(theta_width), torch.zeros(2)])
    log_likelihoods = model.log_prob(
        joint_evaluation, condition_mask=likelihood_mask,
        timesteps=cfg.timesteps, divergence="exact",
        device=cfg.device, verbose=False,
    )
    return {
        "samples": samples,
        "theta_map": theta_map,
        "log_likelihoods": tensor_numpy(log_likelihoods),
        "runtime_seconds": time.perf_counter() - started,
    }


def information_criterion_weights(
    log_likelihoods: dict[str, float], criterion: str, n: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return conventional AICc/BIC scores and normalized support weights."""
    scores = {}
    for family, log_likelihood in log_likelihoods.items():
        k = 1 if family == "original" else 2
        if criterion == "aicc":
            score = 2 * k - 2 * log_likelihood + 2 * k * (k + 1) / (n - k - 1)
        elif criterion == "bic":
            score = k * math.log(n) - 2 * log_likelihood
        else:
            raise ValueError(f"Unknown information criterion {criterion!r}")
        scores[family] = float(score)
    ordered = np.asarray([scores[family] for family in PARABOLA_FAMILIES])
    weights = np.exp(-0.5 * (ordered - ordered.min()))
    weights /= weights.sum()
    return scores, dict(zip(PARABOLA_FAMILIES, map(float, weights)))


def record_parabola_recovery(
    rows: list[dict],
    raw: dict[str, np.ndarray],
    result: dict[str, object],
    reference: dict[str, np.ndarray | float],
    family: str,
    n: int,
    true_t: np.ndarray,
) -> None:
    """Record global/local posterior moments for one matched recovery case."""
    samples = tensor_numpy(result["samples"])
    if family == "original":
        local_samples = samples[:, :, 0]
        synchronization_error = 0.0
    else:
        b_samples = samples[0, :, 0]
        local_samples = samples[:, :, 1]
        synchronization_error = float(np.max(np.abs(samples[:, :, 0] - samples[0:1, :, 0])))
        if synchronization_error > 1e-6:
            raise AssertionError(
                f"{family} N={n} shared b samples are not synchronized: {synchronization_error}"
            )
        exact_b_mean = float(reference["b_mean"])
        exact_b_std = float(reference["b_std"])
        rows.append({
            "family": family, "n_observations": n, "parameter": "global", "index": -1,
            "truth": PARABOLA_TEST_OFFSET, "exact_mean": exact_b_mean,
            "exact_std": exact_b_std, "compass_mean": float(b_samples.mean()),
            "compass_std": float(b_samples.std(ddof=1)),
            "mean_error_in_exact_std": float(abs(b_samples.mean() - exact_b_mean) / max(exact_b_std, 1e-12)),
            "std_ratio_to_exact": float(b_samples.std(ddof=1) / max(exact_b_std, 1e-12)),
            "shared_synchronization_max_abs": synchronization_error,
            "runtime_seconds": float(result["runtime_seconds"]),
        })
        raw[f"b_samples_{family}_n{n}"] = b_samples
    exact_local_means = np.asarray(reference["local_means"])
    exact_local_stds = np.asarray(reference["local_stds"])
    raw[f"local_samples_{family}_n{n}"] = local_samples
    raw[f"exact_local_means_{family}_n{n}"] = exact_local_means
    raw[f"exact_local_stds_{family}_n{n}"] = exact_local_stds
    for index in range(n):
        compass_values = local_samples[index]
        exact_std = max(float(exact_local_stds[index]), 1e-12)
        rows.append({
            "family": family, "n_observations": n, "parameter": "local", "index": index,
            "truth": float(true_t[index]), "exact_mean": float(exact_local_means[index]),
            "exact_std": float(exact_local_stds[index]),
            "compass_mean": float(compass_values.mean()),
            "compass_std": float(compass_values.std(ddof=1)),
            "mean_error_in_exact_std": float(abs(compass_values.mean() - exact_local_means[index]) / exact_std),
            "std_ratio_to_exact": float(compass_values.std(ddof=1) / exact_std),
            "shared_synchronization_max_abs": synchronization_error,
            "runtime_seconds": float(result["runtime_seconds"]),
        })


def plot_parabola_recovery(rows: Sequence[dict], path: Path) -> None:
    """Plot global and local posterior diagnostics across observation counts."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    axes = axes.ravel()
    for family in PARABOLA_FAMILIES:
        color = {"original": "black", "vertical": "tab:blue", "horizontal": "tab:orange"}[family]
        local_errors, local_widths = [], []
        for n in PARABOLA_N_VALUES:
            local = [
                row for row in rows if row["family"] == family
                and int(row["n_observations"]) == n and row["parameter"] == "local"
            ]
            local_errors.append(float(np.mean([row["mean_error_in_exact_std"] for row in local])))
            local_widths.append(float(np.mean([row["std_ratio_to_exact"] for row in local])))
        axes[2].plot(PARABOLA_N_VALUES, local_errors, marker="o", color=color,
                     label=PARABOLA_LABELS[family])
        axes[3].plot(PARABOLA_N_VALUES, local_widths, marker="o", color=color,
                     label=PARABOLA_LABELS[family])
        if family != "original":
            global_rows = [row for row in rows if row["family"] == family and row["parameter"] == "global"]
            axes[0].plot(
                PARABOLA_N_VALUES,
                [row["mean_error_in_exact_std"] for row in global_rows],
                marker="o", color=color, label=PARABOLA_LABELS[family],
            )
            axes[1].plot(
                PARABOLA_N_VALUES,
                [row["std_ratio_to_exact"] for row in global_rows],
                marker="o", color=color, label=PARABOLA_LABELS[family],
            )
    axes[0].set(xscale="log", xlabel="observations N", ylabel="|b mean - exact| / exact std",
                title="Global-offset mean")
    axes[1].set(xscale="log", xlabel="observations N", ylabel="COMPASS std / exact std",
                title="Global-offset width")
    axes[2].set(xscale="log", yscale="log", xlabel="observations N",
                ylabel="mean local error / exact std", title="Local-position means")
    axes[3].set(xscale="log", xlabel="observations N", ylabel="mean COMPASS std / exact std",
                title="Local-position widths")
    axes[1].axhline(1.0, color="0.3", linestyle="--")
    axes[3].axhline(1.0, color="0.3", linestyle="--")
    axes[0].legend()
    axes[2].legend()
    save_figure(fig, path)


def plot_parabola_model_selection(rows: Sequence[dict], path: Path) -> None:
    """Plot accuracy curves and maximum-N confusion matrices for all criteria."""
    criteria = ("aicc", "bic", "exact_evidence")
    criterion_labels = {"aicc": "AICc", "bic": "BIC", "exact_evidence": "Exact evidence"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.0))
    accuracy_axis = axes[0, 0]
    for criterion in criteria:
        accuracy = []
        for n in PARABOLA_N_VALUES:
            selected = [
                row for row in rows if row["criterion"] == criterion
                and int(row["n_observations"]) == n and int(row["selected"]) == 1
            ]
            accuracy.append(np.mean([int(row["correct"]) for row in selected]))
        accuracy_axis.plot(PARABOLA_N_VALUES, accuracy, marker="o",
                           label=criterion_labels[criterion])
    accuracy_axis.set(xscale="log", ylim=(-0.03, 1.03), xlabel="observations N",
                      ylabel="selection accuracy", title="Generating-model recovery")
    accuracy_axis.legend()
    for axis, criterion in zip(axes.ravel()[1:], criteria):
        matrix = np.zeros((len(PARABOLA_FAMILIES), len(PARABOLA_FAMILIES)))
        selected = [
            row for row in rows if row["criterion"] == criterion
            and int(row["n_observations"]) == max(PARABOLA_N_VALUES)
            and int(row["selected"]) == 1
        ]
        for row in selected:
            matrix[PARABOLA_FAMILIES.index(row["generating_family"]),
                   PARABOLA_FAMILIES.index(row["candidate_family"])] += 1
        row_totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        for row_index in range(3):
            for column_index in range(3):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}",
                          ha="center", va="center")
        axis.set(xticks=range(3), yticks=range(3),
                 xticklabels=[PARABOLA_LABELS[name] for name in PARABOLA_FAMILIES],
                 yticklabels=[PARABOLA_LABELS[name] for name in PARABOLA_FAMILIES],
                 xlabel="selected", ylabel="generated", title=f"{criterion_labels[criterion]}, N=50")
        axis.tick_params(axis="x", rotation=25)
    fig.colorbar(image, ax=axes.ravel()[1:].tolist(), shrink=0.7, label="row-normalized frequency")
    save_figure(fig, path)


def experiment_parabola(cfg: RunConfig) -> None:
    """Recover global/local parabola parameters and select among three geometries."""
    out = cfg.output_dir / "08_parabola"
    out.mkdir(parents=True, exist_ok=True)
    models = load_or_train_parabola_models(cfg)
    quadrature_points = 401 if cfg.posterior_samples <= 250 else 801
    recovery_rows: list[dict] = []
    selection_rows: list[dict] = []
    raw: dict[str, np.ndarray] = {
        "n_values": np.asarray(PARABOLA_N_VALUES),
        "families": np.asarray(PARABOLA_FAMILIES),
    }

    for repeat in range(cfg.repeats):
        true_t, noise, cases = matched_parabola_cases(
            max(PARABOLA_N_VALUES), cfg.seed + 1_900 + repeat,
        )
        if not np.all((true_t >= PARABOLA_T_MIN) & (true_t <= PARABOLA_T_MAX)):
            raise AssertionError("Generated parabola positions left the configured prior bounds.")
        vertical_delta = cases["vertical"] - cases["original"]
        horizontal_delta = cases["horizontal"] - cases["original"]
        if not np.allclose(vertical_delta, (0.0, PARABOLA_TEST_OFFSET), atol=1e-7):
            raise AssertionError("Vertical observations are not exactly 10 sigma from baseline.")
        if not np.allclose(horizontal_delta, (PARABOLA_TEST_OFFSET, 0.0), atol=1e-7):
            raise AssertionError("Horizontal observations are not exactly 10 sigma from baseline.")
        if repeat == 0:
            raw["true_t"] = true_t
            raw["matched_noise"] = noise
            for family in PARABOLA_FAMILIES:
                raw[f"observations_{family}"] = cases[family]

        exact_evidences = {
            generating_family: {
                candidate_family: exact_parabola_evidence_series(
                    cases[generating_family], candidate_family,
                    PARABOLA_N_VALUES, quadrature_points,
                )
                for candidate_family in PARABOLA_FAMILIES
            }
            for generating_family in PARABOLA_FAMILIES
        }

        for generating_index, generating_family in enumerate(PARABOLA_FAMILIES):
            for n_index, n in enumerate(PARABOLA_N_VALUES):
                observations = cases[generating_family][:n]
                candidate_results = {}
                learned_log_likelihoods = {}
                for candidate_index, candidate_family in enumerate(PARABOLA_FAMILIES):
                    result = infer_parabola_candidate(
                        models[candidate_family], observations, candidate_family, cfg,
                        cfg.seed + 2_000_000 * repeat + 100_000 * generating_index
                        + 10_000 * n_index + 100 * candidate_index,
                    )
                    candidate_results[candidate_family] = result
                    learned_log_likelihoods[candidate_family] = float(
                        np.sum(result["log_likelihoods"])
                    )

                if repeat == 0:
                    reference = exact_parabola_posterior(
                        observations, generating_family, quadrature_points,
                    )
                    record_parabola_recovery(
                        recovery_rows, raw, candidate_results[generating_family],
                        reference, generating_family, n, true_t[:n],
                    )

                criterion_payloads = {}
                for criterion in ("aicc", "bic"):
                    scores, weights = information_criterion_weights(
                        learned_log_likelihoods, criterion, n,
                    )
                    criterion_payloads[criterion] = (scores, weights)
                exact_values = {
                    candidate: exact_evidences[generating_family][candidate][n]
                    for candidate in PARABOLA_FAMILIES
                }
                exact_ordered = np.asarray([exact_values[name] for name in PARABOLA_FAMILIES])
                exact_weights_array = np.exp(exact_ordered - exact_ordered.max())
                exact_weights_array /= exact_weights_array.sum()
                criterion_payloads["exact_evidence"] = (
                    {name: -2.0 * exact_values[name] for name in PARABOLA_FAMILIES},
                    dict(zip(PARABOLA_FAMILIES, map(float, exact_weights_array))),
                )

                for criterion, (scores, weights) in criterion_payloads.items():
                    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
                        raise AssertionError(f"{criterion} model weights do not sum to one.")
                    selected_family = max(weights, key=weights.get)
                    for candidate_family in PARABOLA_FAMILIES:
                        selection_rows.append({
                            "repeat": repeat,
                            "n_observations": n,
                            "generating_family": generating_family,
                            "candidate_family": candidate_family,
                            "criterion": criterion,
                            "score": scores[candidate_family],
                            "support_weight": weights[candidate_family],
                            "selected": int(candidate_family == selected_family),
                            "correct": int(selected_family == generating_family),
                            "learned_log_likelihood": learned_log_likelihoods[candidate_family],
                            "exact_log_evidence": exact_values[candidate_family],
                            "runtime_seconds": float(candidate_results[candidate_family]["runtime_seconds"]),
                        })
                print(
                    f"repeat={repeat} generated={generating_family:<10} N={n:>2} "
                    + " ".join(
                        f"{criterion}={max(payload[1], key=payload[1].get)}"
                        for criterion, payload in criterion_payloads.items()
                    )
                )

    summary_rows = []
    for criterion in ("aicc", "bic", "exact_evidence"):
        for n in PARABOLA_N_VALUES:
            for generating_family in PARABOLA_FAMILIES:
                selected = [
                    row for row in selection_rows
                    if row["criterion"] == criterion
                    and row["n_observations"] == n
                    and row["generating_family"] == generating_family
                    and row["selected"] == 1
                ]
                summary_rows.append({
                    "criterion": criterion, "n_observations": n,
                    "generating_family": generating_family,
                    "trials": len(selected),
                    "accuracy": float(np.mean([row["correct"] for row in selected])),
                })

    write_rows(out / "parabola_recovery_metrics.csv", recovery_rows)
    write_rows(out / "parabola_model_selection_metrics.csv", selection_rows)
    write_rows(out / "parabola_model_selection_summary.csv", summary_rows)
    save_raw_data(out / "raw_plot_data.npz", **raw)
    plot_parabola_data_separation(out / "three_model_data_separation.png")
    plot_parabola_recovery(recovery_rows, out / "global_local_posterior_recovery.png")
    plot_parabola_model_selection(selection_rows, out / "model_selection.png")
    with (out / "experiment_config.json").open("w") as handle:
        json.dump({
            "families": list(PARABOLA_FAMILIES),
            "simulators": {
                "original": "x=(t,t^2)+eps",
                "vertical": "x=(t,t^2+b)+eps",
                "horizontal": "x=(t+b,t^2)+eps",
            },
            "t_prior": {"distribution": "uniform", "low": PARABOLA_T_MIN, "high": PARABOLA_T_MAX},
            "b_prior": {"distribution": "normal", "mean": PARABOLA_OFFSET_PRIOR_MEAN,
                        "std": PARABOLA_OFFSET_PRIOR_STD},
            "epsilon_std": PARABOLA_NOISE_STD,
            "test_offset": PARABOLA_TEST_OFFSET,
            "offset_in_epsilon_std": PARABOLA_TEST_OFFSET / PARABOLA_NOISE_STD,
            "n_observations": list(PARABOLA_N_VALUES),
            "repeats": cfg.repeats,
            "quick_smoke_configuration": cfg.train_samples <= 4_000 and cfg.max_epochs <= 3,
            "quadrature_points": quadrature_points,
            "selection_criteria": ["aicc", "bic", "exact_evidence"],
            "network": {"sde_type": "vesde", "sigma": 3.0, "hidden_size": 40,
                        "depth": 4, "num_heads": 4, "mlp_ratio": 4},
        }, handle, indent=2)


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
        ax.tick_params(axis="x", rotation=70)
    axes[1].axhline(1.0, color="black", linestyle="--")
    save_figure(fig, out / "sampler_comparison_n10.png")


def replot_scaling(output_dir: Path) -> None:
    out = output_dir / "03_observation_scaling"
    rows = read_metric_rows(out / "observation_scaling_metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    axes = axes.ravel()
    runtime_linestyles = ("-", "-", "-", "-", "-", (0, (1, 6)))
    for variant_index, variant in enumerate(OBSERVATION_SCALING_VARIANTS):
        chosen = [row for row in rows if row["variant"] == variant.key]
        if not chosen:
            continue
        for axis_index, (ax, value) in enumerate(
            zip(axes[:3], ("mean_error_sigma", "std_ratio", "runtime_seconds"))
        ):
            summary = aggregate(chosen, ("n_observations",), value)
            xs = np.asarray(N_VALUES, dtype=float)
            ys = np.asarray([summary[(float(n),)][0] for n in N_VALUES])
            errors = np.asarray([summary[(float(n),)][1] for n in N_VALUES])
            linestyle = runtime_linestyles[variant_index] if axis_index == 2 else "-"
            plot_kwargs = {"linestyle": linestyle}
            if axis_index == 2 and variant_index == 5:
                plot_kwargs.update(
                    linewidth=mpl.rcParams["lines.linewidth"] * 1.4,
                    dash_capstyle="round",
                )
            line, = ax.plot(xs, ys, marker="o", label=variant.label, **plot_kwargs)
            ax.fill_between(
                xs, ys - errors, ys + errors, color=line.get_color(), alpha=0.12
            )
        error_summary = aggregate(chosen, ("n_observations",), "mean_error_sigma")
        runtime_summary = aggregate(chosen, ("n_observations",), "runtime_seconds")
        mean_errors = np.asarray(
            [error_summary[(float(n),)][0] for n in N_VALUES]
        )
        runtimes = np.asarray(
            [runtime_summary[(float(n),)][0] for n in N_VALUES]
        )
        tradeoff_kwargs = {"linestyle": runtime_linestyles[variant_index]}
        if variant_index == 5:
            tradeoff_kwargs.update(
                linewidth=mpl.rcParams["lines.linewidth"] * 1.4,
                dash_capstyle="round",
            )
        tradeoff_line, = axes[3].plot(
            runtimes, mean_errors, marker="o", label=variant.label, **tradeoff_kwargs
        )
        for n, runtime, mean_error in zip(N_VALUES, runtimes, mean_errors):
            axes[3].annotate(
                str(n), (runtime, mean_error), xytext=(4, 3),
                textcoords="offset points", color=tradeoff_line.get_color(),
                fontsize=5, clip_on=True,
            )
    axes[0].set(xscale="log", xlabel="observations N", ylabel="mean error / analytic std")
    axes[1].set(xscale="log", xlabel="observations N", ylabel="sample std / analytic std")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[2].set(xscale="log", yscale="log", xlabel="observations N", ylabel="runtime (s)")
    axes[3].set(
        xscale="log", yscale="log", xlabel="runtime (s)",
        ylabel="mean error / analytic std",
    )
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
    grid_metrics = out / "training_time_sampler_grid_metrics.csv"
    if grid_metrics.exists():
        grid_rows = read_metric_rows(grid_metrics)
        plot_time_sampling_sampler_grid(
            grid_rows, out / "training_time_sampler_grid_effect.png",
        )
        plot_selected_time_sampling_sampler_grid(
            grid_rows, out / "training_time_sampler_grid_selected_effect.png",
        )


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
            label="COMPASS multi-observation (DPM-2 + Gaussian)")
    if "multi_fnpe_samples" in raw:
        ax.plot(grid, gaussian_kde(raw["multi_fnpe_samples"])(grid), color="tab:green",
                linewidth=2.4, label="COMPASS multi-observation (Langevin + F-NPSE)")
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


def replot_gaussian(output_dir: Path) -> None:
    out = output_dir / "07_gaussian_test"
    plot_gaussian_test(
        load_raw_data(out / "raw_plot_data.npz"),
        out / "global_mean_recovery.png",
    )


def replot_parabola(output_dir: Path) -> None:
    out = output_dir / "08_parabola"
    recovery_rows = read_metric_rows(out / "parabola_recovery_metrics.csv")
    selection_rows = read_metric_rows(out / "parabola_model_selection_metrics.csv")
    plot_parabola_data_separation(out / "three_model_data_separation.png")
    plot_parabola_recovery(recovery_rows, out / "global_local_posterior_recovery.png")
    plot_parabola_model_selection(selection_rows, out / "model_selection.png")


REPLOTTERS = {
    "contract": replot_contract,
    "samplers": replot_samplers,
    "scaling": replot_scaling,
    "time": replot_time_sampling,
    "time-grid": replot_time_sampling,
    "individual": replot_individual,
    "population": replot_population,
    "hierarchy": replot_hierarchy,
    "gaussian": replot_gaussian,
    "parabola": replot_parabola,
}


def parse_experiments(value: str) -> list[str]:
    aliases = {
        "contract": "contract",
        "samplers": "samplers",
        "scaling": "scaling",
        "time": "time",
        "time-grid": "time-grid",
        "individual": "individual",
        "population": "population",
        "hierarchy": "hierarchy",
        "gaussian": "gaussian",
        "parabola": "parabola",
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
                        help="all or comma-separated: contract,samplers,scaling,time,time-grid,individual,population,hierarchy,gaussian,parabola")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true",
                        help="tiny wiring test; do not use results for conclusions")
    parser.add_argument("--plots-only", action="store_true",
                        help="regenerate figures from saved CSV/NPZ data; run no calculations")
    parser.add_argument(
        "--sampler-variants",
        help=("comma-separated scaling sampler keys to recalculate in place; "
              "for example: pfode_gauss"),
    )
    args = parser.parse_args()
    logical_cpus, selected_cpus = CPU_LIMIT_INFO
    print(
        f"CPU limited to {len(selected_cpus)} of {logical_cpus} logical CPUs "
        f"({len(selected_cpus) / logical_cpus:.2%} capacity)."
    )
    experiments = parse_experiments(args.experiments)
    scaling_variants = OBSERVATION_SCALING_VARIANTS
    if args.sampler_variants:
        if experiments != ["scaling"] or args.plots_only:
            parser.error("--sampler-variants requires --experiments scaling")
        variants_by_key = {
            variant.key: variant for variant in OBSERVATION_SCALING_VARIANTS
        }
        requested_keys = [
            key.strip() for key in args.sampler_variants.split(",") if key.strip()
        ]
        unknown_keys = sorted(set(requested_keys) - set(variants_by_key))
        if unknown_keys:
            parser.error(
                "unknown scaling sampler variants: " + ", ".join(unknown_keys)
            )
        scaling_variants = tuple(variants_by_key[key] for key in requested_keys)
    if args.plots_only:
        configure_plot_style()
        for name in experiments:
            print(f"Regenerating {name} plots from saved data...")
            REPLOTTERS[name](args.output_dir.resolve())
            save_problem_pairplot_for_experiment(args.output_dir.resolve() / PAIRPLOT_FOLDERS[name], name)
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
        if args.sampler_variants:
            payload["sampler_variants"] = [variant.key for variant in scaling_variants]
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
        "scaling": lambda: experiment_observation_scaling(
            shared_model, cfg, scaling_variants,
            incremental=bool(args.sampler_variants),
        ),
        "time": lambda: experiment_time_sampling(cfg),
        "time-grid": lambda: experiment_time_sampling_sampler_grid(
            {
                scheme: load_or_train(
                    cfg, f"shared_{scheme}", 2, simulate_shared, time_sampling=scheme,
                )
                for scheme in TIME_SAMPLINGS
            },
            cfg,
            cfg.output_dir / "04_time_sampling",
        ),
        "individual": lambda: experiment_multi_vs_individual(shared_model, cfg),
        "population": lambda: experiment_population_globals(cfg),
        "hierarchy": lambda: experiment_shared_local(cfg),
        "gaussian": lambda: experiment_gaussian_global(cfg),
        "parabola": lambda: experiment_parabola(cfg),
    }
    for name in experiments:
        print(f"\n=== Running experiment: {name} ===")
        started = time.perf_counter()
        runners[name]()
        print(f"=== {name} finished in {(time.perf_counter() - started) / 60:.1f} min ===")
        save_problem_pairplot_for_experiment(cfg.output_dir / PAIRPLOT_FOLDERS[name], name)


if __name__ == "__main__":
    main()
