#!/usr/bin/env python3
"""Validate global/local MAP inference with a linear-Gaussian hierarchy.

The two trained candidates have the same single-observation simulator.  They
become distinguishable only when Model 1's first parameter is constrained to be
global across observations while both parameters of Model 2 remain local.

This is intentionally a standalone experiment.  It does not run on import.
Typical invocations are::

    python tutorials/07_gaussian_test.py
    python tutorials/07_gaussian_test.py --quick

The experiment uses COMPASS's multi-observation sampler to enforce global
coordinates and its score-based MAP refinement for local coordinates.  The
current ModelTransfuser API accepts one hierarchy and counts only the latent
width, so it cannot represent per-model roles or ``d_G + N d_L`` in one call.
Consequently, this script evaluates the same COMPASS PF-ODE likelihood directly
and calculates the required information criteria with explicit role metadata.
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
    """Apply the repository's hard CPU cap before native libraries import."""
    logical_cpus = os.cpu_count() or 1
    cpu_limit = min(int(max_threads), logical_cpus)
    if cpu_limit < 1:
        raise RuntimeError(
            f"A {max_threads}-thread CPU cap permits no CPU on this {logical_cpus}-CPU host."
        )
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    selected = allowed[:cpu_limit]
    if not selected:
        raise RuntimeError("The process has no CPUs available in its affinity mask.")
    os.sched_setaffinity(0, selected)
    for variable in CPU_THREAD_ENV_VARS:
        os.environ[variable] = str(len(selected))
    return logical_cpus, selected


CPU_LIMIT_INFO = configure_cpu_usage_limit() if __name__ == "__main__" else None

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from autocvd import autocvd
except ImportError:
    autocvd = None

from compass import ScoreBasedInferenceModel as SBIm


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "hierarchical_gaussian_map"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "compositional_inference" / "07_hierarchical_gaussian_map"

MODEL_NAMES = ("Shared Global", "Fully Local")
CONFIGURATIONS = ("correct_hierarchy", "all_local", "all_global")
CONFIGURATION_LABELS = {
    "correct_hierarchy": "Correct hierarchy",
    "all_local": "All local",
    "all_global": "All global",
}

SIGMA_G = 1.5
SIGMA_LOCAL = 1.0
SIGMA_X = 0.15
TRAIN_N = 100_000
VAL_N = 5_000
N_OBSERVATIONS_LIST = (2, 5, 10, 25, 50)
N_MOCK_RUNS = 100

TRAIN_BATCH_SIZE = 256
TRAIN_MAX_EPOCHS = 500
TRAIN_LR = 1e-3
TRAIN_EARLY_STOPPING_PATIENCE = 25
GLOBAL_SHARING_ATOL = 1e-6
AMBIGUOUS_WEIGHT_THRESHOLD = 0.65

MODEL_KWARGS = {
    "sde_type": "vesde",
    "sigma": 3.0,
    "hidden_size": 32,
    "depth": 4,
    "num_heads": 4,
    "mlp_ratio": 4,
}


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    device: str
    seed: int
    quick: bool
    train_n: int
    val_n: int
    n_observations: tuple[int, ...]
    n_runs: int
    num_samples: int
    timesteps: int
    max_epochs: int
    patience: int
    force_data: bool
    force_train: bool


@dataclass(frozen=True)
class Normalization:
    """One common affine transform for [theta_0, theta_1, x_1, x_2]."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, values_physical: np.ndarray) -> np.ndarray:
        values = np.asarray(values_physical, dtype=np.float32)
        if values.shape[-1] != 4:
            raise ValueError(f"Expected joint width 4, received {values.shape}.")
        return (values - self.mean) / self.std

    def inverse(self, values_normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(values_normalized, dtype=np.float32)
        if values.shape[-1] != 4:
            raise ValueError(f"Expected joint width 4, received {values.shape}.")
        return values * self.std + self.mean

    def normalize_theta(self, theta_physical: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta_physical, dtype=np.float32)
        if theta.shape[-1] != 2:
            raise ValueError(f"Expected theta width 2, received {theta.shape}.")
        return (theta - self.mean[:2]) / self.std[:2]

    def normalize_x(self, x_physical: np.ndarray) -> np.ndarray:
        x = np.asarray(x_physical, dtype=np.float32)
        if x.shape[-1] != 2:
            raise ValueError(f"Expected observation width 2, received {x.shape}.")
        return (x - self.mean[2:]) / self.std[2:]

    def theta_to_physical(self, theta_normalized: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta_normalized, dtype=np.float32)
        if theta.shape[-1] != 2:
            raise ValueError(f"Expected theta width 2, received {theta.shape}.")
        return theta * self.std[:2] + self.mean[:2]

    def x_log_abs_det(self) -> float:
        return float(np.log(self.std[2:]).sum())


@dataclass(frozen=True)
class MapResult:
    global_indices: tuple[int, ...]
    local_indices: tuple[int, ...]
    global_map: np.ndarray
    local_map: np.ndarray
    expanded_map: np.ndarray
    posterior_sample_std: np.ndarray


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> str:
    """Reserve one GPU through autocvd, or gracefully use CPU."""
    if requested == "cpu":
        return "cpu"
    if autocvd is None:
        if requested == "cuda":
            print("autocvd is unavailable; requested CUDA cannot be reserved, using CPU.")
        else:
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
    print("autocvd reserved a GPU but CUDA is unavailable to Torch; using CPU.")
    return "cpu"


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def simulate_pairs(n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw independent single-observation theta/x pairs in physical units."""
    global_parameter = SIGMA_G * torch.randn(n, 1, generator=generator)
    local_parameter = SIGMA_LOCAL * torch.randn(n, 1, generator=generator)
    noise = SIGMA_X * torch.randn(n, 2, generator=generator)
    theta = torch.cat([global_parameter, local_parameter], dim=1)
    x = torch.cat(
        [global_parameter + local_parameter, global_parameter - local_parameter],
        dim=1,
    ) + noise
    assert theta.shape == (n, 2)
    assert x.shape == (n, 2)
    return theta, x


def data_signature(cfg: ExperimentConfig) -> dict[str, object]:
    return {
        "version": 1,
        "seed": cfg.seed,
        "train_n": cfg.train_n,
        "val_n": cfg.val_n,
        "sigma_g": SIGMA_G,
        "sigma_local": SIGMA_LOCAL,
        "sigma_x": SIGMA_X,
        "models": list(MODEL_NAMES),
    }


def generate_datasets(
    cfg: ExperimentConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], Normalization]:
    """Generate separate named splits and fit one shared training normalization."""
    datasets: dict[str, dict[str, torch.Tensor]] = {}
    training_joints = []
    for model_index, model_name in enumerate(MODEL_NAMES):
        train_generator = torch.Generator().manual_seed(cfg.seed + 100 * model_index)
        val_generator = torch.Generator().manual_seed(cfg.seed + 100 * model_index + 1)
        theta_train, x_train = simulate_pairs(cfg.train_n, train_generator)
        theta_val, x_val = simulate_pairs(cfg.val_n, val_generator)
        datasets[model_name] = {
            "theta_train_physical": theta_train,
            "x_train_physical": x_train,
            "theta_val_physical": theta_val,
            "x_val_physical": x_val,
        }
        training_joints.append(torch.cat([theta_train, x_train], dim=1))

    common_joint = torch.cat(training_joints, dim=0)
    mean = common_joint.mean(dim=0).numpy().astype(np.float32)
    std = common_joint.std(dim=0, unbiased=False).numpy().astype(np.float32)
    if np.any(std <= 1e-8):
        raise RuntimeError(f"Degenerate normalization standard deviation: {std}.")
    normalization = Normalization(mean=mean, std=std)
    return datasets, normalization


def save_datasets(
    path: Path,
    metadata_path: Path,
    datasets: Mapping[str, Mapping[str, torch.Tensor]],
    normalization: Normalization,
    signature: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "joint_mean": normalization.mean,
        "joint_std": normalization.std,
    }
    for model_name, splits in datasets.items():
        slug = model_slug(model_name)
        for key, value in splits.items():
            arrays[f"{slug}__{key}"] = value.numpy()
    np.savez_compressed(path, **arrays)
    metadata_path.write_text(json.dumps(dict(signature), indent=2) + "\n")


def load_datasets(
    path: Path,
) -> tuple[dict[str, dict[str, torch.Tensor]], Normalization]:
    with np.load(path) as archive:
        normalization = Normalization(
            mean=np.asarray(archive["joint_mean"], dtype=np.float32),
            std=np.asarray(archive["joint_std"], dtype=np.float32),
        )
        datasets: dict[str, dict[str, torch.Tensor]] = {}
        for model_name in MODEL_NAMES:
            slug = model_slug(model_name)
            datasets[model_name] = {
                key: torch.as_tensor(
                    np.asarray(archive[f"{slug}__{key}"], dtype=np.float32)
                )
                for key in (
                    "theta_train_physical",
                    "x_train_physical",
                    "theta_val_physical",
                    "x_val_physical",
                )
            }
    return datasets, normalization


def prepare_datasets(
    cfg: ExperimentConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], Normalization, Path]:
    mode = "quick" if cfg.quick else "full"
    data_path = cfg.data_dir / f"datasets_{mode}.npz"
    metadata_path = cfg.data_dir / f"datasets_{mode}.json"
    signature = data_signature(cfg)
    compatible = False
    if data_path.exists() and metadata_path.exists() and not cfg.force_data:
        try:
            compatible = json.loads(metadata_path.read_text()) == signature
        except (json.JSONDecodeError, OSError):
            compatible = False
    if compatible:
        print(f"Loading compatible data: {data_path}")
        datasets, normalization = load_datasets(data_path)
    else:
        print(f"Generating separate datasets for {', '.join(MODEL_NAMES)}.")
        datasets, normalization = generate_datasets(cfg)
        save_datasets(data_path, metadata_path, datasets, normalization, signature)
    return datasets, normalization, data_path


def normalized_splits(
    splits: Mapping[str, torch.Tensor], normalization: Normalization,
) -> dict[str, torch.Tensor]:
    result = {
        "theta_train": torch.as_tensor(
            normalization.normalize_theta(splits["theta_train_physical"].numpy()),
            dtype=torch.float32,
        ),
        "x_train": torch.as_tensor(
            normalization.normalize_x(splits["x_train_physical"].numpy()),
            dtype=torch.float32,
        ),
        "theta_val": torch.as_tensor(
            normalization.normalize_theta(splits["theta_val_physical"].numpy()),
            dtype=torch.float32,
        ),
        "x_val": torch.as_tensor(
            normalization.normalize_x(splits["x_val_physical"].numpy()),
            dtype=torch.float32,
        ),
    }
    for key, value in result.items():
        expected = 2
        if value.ndim != 2 or value.shape[1] != expected:
            raise AssertionError(f"{key} has invalid shape {tuple(value.shape)}.")
    return result


def checkpoint_signature(
    cfg: ExperimentConfig,
    model_name: str,
    normalization: Normalization,
) -> dict[str, object]:
    return {
        "version": 1,
        "model_name": model_name,
        "nodes_size": 4,
        "model_kwargs": MODEL_KWARGS,
        "training": {
            "seed": cfg.seed,
            "train_n": cfg.train_n,
            "val_n": cfg.val_n,
            "batch_size": TRAIN_BATCH_SIZE,
            "max_epochs": cfg.max_epochs,
            "lr": TRAIN_LR,
            "patience": cfg.patience,
            "time_sampling": "mixture",
        },
        "joint_mean": normalization.mean.tolist(),
        "joint_std": normalization.std.tolist(),
    }


def build_model(device: str) -> SBIm:
    return SBIm(nodes_size=4, device=device, **MODEL_KWARGS)


def load_or_train_model(
    cfg: ExperimentConfig,
    model_name: str,
    splits: Mapping[str, torch.Tensor],
    normalization: Normalization,
) -> SBIm:
    mode = "quick" if cfg.quick else "full"
    model_dir = cfg.data_dir / "checkpoints" / mode / model_slug(model_name)
    checkpoint = model_dir / "Model_checkpoint.pt"
    metadata_path = model_dir / "metadata.json"
    signature = checkpoint_signature(cfg, model_name, normalization)
    compatible = False
    if checkpoint.exists() and metadata_path.exists() and not cfg.force_train:
        try:
            compatible = json.loads(metadata_path.read_text()) == signature
        except (json.JSONDecodeError, OSError):
            compatible = False
    if compatible:
        print(f"Loading {model_name}: {checkpoint}")
        return SBIm.load(str(checkpoint), device=cfg.device)

    normalized = normalized_splits(splits, normalization)
    seed_all(cfg.seed + 1_000 + MODEL_NAMES.index(model_name))
    model = build_model(cfg.device)
    print(f"Training {model_name} on {cfg.device}; checkpoint={model_dir}")
    model.train(
        theta=normalized["theta_train"],
        x=normalized["x_train"],
        theta_val=normalized["theta_val"],
        x_val=normalized["x_val"],
        batch_size=TRAIN_BATCH_SIZE,
        max_epochs=cfg.max_epochs,
        lr=TRAIN_LR,
        early_stopping_patience=cfg.patience,
        time_sampling="mixture",
        device=cfg.device,
        verbose=False,
        path=str(model_dir),
        name="Model",
    )
    if not checkpoint.exists():
        raise RuntimeError(f"Training did not create expected checkpoint {checkpoint}.")
    metadata_path.write_text(json.dumps(signature, indent=2) + "\n")
    return SBIm.load(str(checkpoint), device=cfg.device)


def generate_mock_dataset(
    n_observations: int,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Draw one shared g, N local values, and N observation pairs."""
    rng = np.random.default_rng(seed)
    global_true = float(rng.normal(0.0, SIGMA_G))
    local_true = rng.normal(0.0, SIGMA_LOCAL, size=n_observations)
    noise = rng.normal(0.0, SIGMA_X, size=(n_observations, 2))
    observations = np.column_stack(
        [global_true + local_true, global_true - local_true]
    ) + noise
    assert observations.shape == (n_observations, 2)
    return global_true, local_true.astype(np.float32), observations.astype(np.float32)


def analytic_maps(observations_physical: np.ndarray) -> dict[str, np.ndarray | float]:
    """Return ML references and exact prior-aware Gaussian posterior MAP values."""
    observations = np.asarray(observations_physical, dtype=np.float64)
    z = 0.5 * (observations[:, 0] + observations[:, 1])
    d = 0.5 * (observations[:, 0] - observations[:, 1])
    global_precision = 1.0 / SIGMA_G**2 + 2.0 * len(z) / SIGMA_X**2
    local_precision = 1.0 / SIGMA_LOCAL**2 + 2.0 / SIGMA_X**2
    global_map = ((2.0 / SIGMA_X**2) * z.sum()) / global_precision
    local_map = ((2.0 / SIGMA_X**2) * d) / local_precision
    return {
        "global_ml": float(z.mean()),
        "local_ml": d.astype(np.float32),
        "global_map": float(global_map),
        "local_map": local_map.astype(np.float32),
        "global_posterior_std": float(global_precision**-0.5),
    }


def roles_for(model_name: str, configuration: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if configuration == "correct_hierarchy":
        global_indices = (0,) if model_name == "Shared Global" else ()
    elif configuration == "all_local":
        global_indices = ()
    elif configuration == "all_global":
        global_indices = (0, 1)
    else:
        raise ValueError(f"Unknown configuration {configuration!r}.")
    local_indices = tuple(index for index in range(2) if index not in global_indices)
    return global_indices, local_indices


def normalized_prior(
    global_indices: Sequence[int], normalization: Normalization,
) -> tuple[list[float], list[float]]:
    prior_stds_physical = np.asarray([SIGMA_G, SIGMA_LOCAL], dtype=np.float32)
    indices = np.asarray(global_indices, dtype=int)
    means = (-normalization.mean[:2] / normalization.std[:2])[indices]
    stds = (prior_stds_physical / normalization.std[:2])[indices]
    return means.tolist(), stds.tolist()


def normalized_single_observation_precision(
    global_indices: Sequence[int], normalization: Normalization,
) -> torch.Tensor:
    """Exact one-pair posterior precision for the Gaussian composition correction."""
    physical_precision = np.asarray(
        [
            1.0 / SIGMA_G**2 + 2.0 / SIGMA_X**2,
            1.0 / SIGMA_LOCAL**2 + 2.0 / SIGMA_X**2,
        ],
        dtype=np.float32,
    )
    normalized_precision = physical_precision * np.square(normalization.std[:2])
    return torch.as_tensor(normalized_precision[list(global_indices)], dtype=torch.float32)


def infer_hierarchical_map(
    model: SBIm,
    observations_physical: np.ndarray,
    global_indices: tuple[int, ...],
    local_indices: tuple[int, ...],
    normalization: Normalization,
    cfg: ExperimentConfig,
    inference_seed: int,
) -> MapResult:
    """Use hierarchical sampling and score refinement without copying globals."""
    x_normalized = torch.as_tensor(
        normalization.normalize_x(observations_physical), dtype=torch.float32
    )
    n_observations = x_normalized.shape[0]
    seed_all(inference_seed)

    sample_kwargs = dict(
        x=x_normalized,
        num_samples=cfg.num_samples,
        timesteps=cfg.timesteps,
        order=2,
        method="dpm",
        corrector_steps_interval=1,
        corrector_steps=5,
        final_corrector_steps=3,
        snr=0.1,
        device=cfg.device,
        verbose=False,
    )
    if global_indices:
        precision = normalized_single_observation_precision(
            global_indices, normalization
        ).repeat(n_observations, 1)
        samples = model.sample(
            **sample_kwargs,
            multi_obs_inference=True,
            hierarchy=list(global_indices),
            prior=normalized_prior(global_indices, normalization),
            correction="gauss",
            posterior_precision=precision,
        )
    else:
        # No shared dimensions means N independent ordinary posteriors.  This is
        # the intentional all-local optimization, not a post-hoc expansion.
        samples = model.sample(**sample_kwargs, multi_obs_inference=False)

    if tuple(samples.shape) != (n_observations, cfg.num_samples, 2):
        raise AssertionError(f"Unexpected posterior sample shape {tuple(samples.shape)}.")
    samples_cpu = samples.detach().cpu()
    posterior_mean = samples_cpu.mean(dim=1)
    posterior_std = samples_cpu.std(dim=1, unbiased=False)

    # MultiObsSampler itself keeps hierarchy coordinates synchronized throughout
    # inference.  We verify that result before extracting the first shared value;
    # inconsistent values are never averaged or silently repaired.
    for index in global_indices:
        synchronization_error = float(
            (samples_cpu[:, :, index] - samples_cpu[0:1, :, index]).abs().max()
        )
        if synchronization_error > GLOBAL_SHARING_ATOL:
            raise AssertionError(
                f"Global sample coordinate {index} is not shared: "
                f"max deviation={synchronization_error:.3e}."
            )

    joint_init = torch.cat([posterior_mean, x_normalized], dim=1)
    condition_mask = torch.tensor([0.0, 0.0, 1.0, 1.0])
    if global_indices:
        condition_mask[list(global_indices)] = 1.0

    if local_indices:
        sigma_start = max(2.0 * float(posterior_std[:, list(local_indices)].max()), 1e-3)
        joint_map = model.map_estimate(
            joint_init,
            condition_mask,
            sigma_start=sigma_start,
            timesteps=max(12, cfg.timesteps // 2),
            iterations_per_level=2,
            device=cfg.device,
        )
    else:
        # All theta dimensions are shared and already jointly inferred; there is
        # no local coordinate for the single-observation score ascent to update.
        joint_map = joint_init

    expanded_normalized = joint_map[:, :2].detach().cpu().numpy()
    expanded_physical = normalization.theta_to_physical(expanded_normalized)
    for index in global_indices:
        coordinate_range = float(np.ptp(expanded_physical[:, index]))
        if coordinate_range > GLOBAL_SHARING_ATOL:
            raise AssertionError(
                f"Global MAP coordinate {index} is not shared: range={coordinate_range:.3e}."
            )

    global_map = (
        expanded_physical[0, list(global_indices)].copy()
        if global_indices
        else np.empty((0,), dtype=np.float32)
    )
    local_map = (
        expanded_physical[:, list(local_indices)].copy()
        if local_indices
        else np.empty((n_observations, 0), dtype=np.float32)
    )
    return MapResult(
        global_indices=global_indices,
        local_indices=local_indices,
        global_map=global_map,
        local_map=local_map,
        expanded_map=expanded_physical,
        posterior_sample_std=posterior_std.numpy(),
    )


def evaluate_likelihood(
    model: SBIm,
    map_result: MapResult,
    observations_physical: np.ndarray,
    normalization: Normalization,
    cfg: ExperimentConfig,
) -> np.ndarray:
    """Evaluate physical-unit log p(x_i | theta_MAP,i) with the COMPASS PF-ODE."""
    theta_normalized = normalization.normalize_theta(map_result.expanded_map)
    x_normalized = normalization.normalize_x(observations_physical)
    joint = torch.as_tensor(
        np.concatenate([theta_normalized, x_normalized], axis=1),
        dtype=torch.float32,
    )
    assert joint.shape == (len(observations_physical), 4)
    log_prob_normalized = model.log_prob(
        joint,
        condition_mask=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        timesteps=cfg.timesteps,
        divergence="exact",
        device=cfg.device,
        verbose=False,
    )
    # Convert density in normalized x coordinates to physical x coordinates.
    return log_prob_normalized.detach().cpu().numpy() - normalization.x_log_abs_det()


def effective_parameter_count(
    n_observations: int,
    global_indices: Sequence[int],
    local_indices: Sequence[int],
) -> int:
    return len(global_indices) + n_observations * len(local_indices)


def information_criteria(log_likelihood: float, k: int, n: int) -> dict[str, float]:
    """Use N independent observation pairs as the information-criterion sample size."""
    aic = 2.0 * k - 2.0 * log_likelihood
    # AICc is genuinely undefined when n <= k + 1; retain NaN rather than hiding it.
    aicc = (
        aic + 2.0 * k * (k + 1) / (n - k - 1)
        if n > k + 1
        else math.nan
    )
    bic = k * math.log(n) - 2.0 * log_likelihood
    return {"aic": float(aic), "aicc": float(aicc), "bic": float(bic)}


def normalized_weights(scores: Sequence[float], lower_is_better: bool) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    logits = -0.5 * values if lower_is_better else values
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def map_metrics(
    model_name: str,
    map_result: MapResult,
    global_true: float,
    local_true: np.ndarray,
    references: Mapping[str, np.ndarray | float],
) -> dict[str, float]:
    global_values = map_result.expanded_map[:, 0]
    local_values = map_result.expanded_map[:, 1]
    analytic_global = float(references["global_map"])
    analytic_local = np.asarray(references["local_map"])
    estimated_global = (
        float(global_values[0]) if 0 in map_result.global_indices else math.nan
    )
    return {
        "estimated_global": estimated_global,
        "global_error": float(np.sqrt(np.mean(np.square(global_values - global_true)))),
        "global_analytic_error": float(
            np.sqrt(np.mean(np.square(global_values - analytic_global)))
        ),
        "local_rmse": float(np.sqrt(np.mean(np.square(local_values - local_true)))),
        "local_analytic_rmse": float(
            np.sqrt(np.mean(np.square(local_values - analytic_local)))
        ),
        "sharing_range": float(np.ptp(global_values)),
        "sharing_std": float(np.std(global_values)),
        "local_estimate_std": float(np.std(local_values)),
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty table {path}.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "global_error",
        "global_analytic_error",
        "local_rmse",
        "local_analytic_rmse",
        "sharing_range",
        "sharing_std",
        "local_estimate_std",
        "log_likelihood",
        "effective_k",
        "aic",
        "bic",
        "aic_weight",
        "bic_weight",
        "no_penalty_weight",
    )
    groups: dict[tuple[int, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (int(row["n_observations"]), str(row["model"]), str(row["configuration"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (n, model_name, configuration), group in sorted(groups.items()):
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                stats = [math.nan] * 7
            else:
                stats = [
                    float(finite.mean()),
                    float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                    float(np.median(finite)),
                    float(np.quantile(finite, 0.05)),
                    float(np.quantile(finite, 0.25)),
                    float(np.quantile(finite, 0.75)),
                    float(np.quantile(finite, 0.95)),
                ]
            summary.append({
                "n_observations": n,
                "model": model_name,
                "configuration": configuration,
                "metric": metric,
                "mean": stats[0],
                "std": stats[1],
                "median": stats[2],
                "q05": stats[3],
                "q25": stats[4],
                "q75": stats[5],
                "q95": stats[6],
                "count": int(finite.size),
            })
    return summary


def rows_for(
    rows: Sequence[Mapping[str, object]],
    *,
    model: str | None = None,
    configuration: str | None = None,
    n: int | None = None,
) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if (model is None or row["model"] == model)
        and (configuration is None or row["configuration"] == configuration)
        and (n is None or int(row["n_observations"]) == n)
    ]


def mean_metric(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def configure_plot_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 240,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "legend.fontsize": 8,
    })


def save_figure(fig: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_by_configuration(
    rows: Sequence[Mapping[str, object]],
    n_values: Sequence[int],
    metric: str,
    ylabel: str,
    path: Path,
    analytic_reference: bool = False,
    dashed_ablations: bool = False,
) -> None:
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    for configuration, marker in zip(CONFIGURATIONS, ("o", "s", "^")):
        means = [
            mean_metric(
                rows_for(rows, model="Shared Global", configuration=configuration, n=n),
                metric,
            )
            for n in n_values
        ]
        axis.plot(
            n_values,
            means,
            marker=marker,
            linestyle="--" if dashed_ablations and configuration != "correct_hierarchy" else "-",
            label=CONFIGURATION_LABELS[configuration],
        )
    if analytic_reference:
        posterior_std = [
            (1.0 / SIGMA_G**2 + 2.0 * n / SIGMA_X**2) ** -0.5
            for n in n_values
        ]
        axis.plot(n_values, posterior_std, "k--", label="Analytic posterior std")
    axis.set(xscale="log", yscale="log", xlabel="Number of observation pairs N", ylabel=ylabel)
    axis.set_xticks(n_values, labels=[str(n) for n in n_values])
    axis.legend()
    save_figure(fig, path)


def plot_sharing_error(
    rows: Sequence[Mapping[str, object]], n_values: Sequence[int], path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    for configuration, marker in zip(CONFIGURATIONS, ("o", "s", "^")):
        values = [
            mean_metric(
                rows_for(rows, model="Shared Global", configuration=configuration, n=n),
                "sharing_range",
            )
            for n in n_values
        ]
        axis.plot(n_values, values, marker=marker, label=CONFIGURATION_LABELS[configuration])
    axis.axhline(GLOBAL_SHARING_ATOL, color="black", linestyle="--", label="sharing tolerance")
    axis.set_yscale("symlog", linthresh=GLOBAL_SHARING_ATOL / 10)
    axis.set(xscale="log", xlabel="Number of observation pairs N",
             ylabel="Global-coordinate range")
    axis.set_xticks(n_values, labels=[str(n) for n in n_values])
    axis.legend()
    save_figure(fig, path)


def plot_model_weights(
    rows: Sequence[Mapping[str, object]],
    invalid_rows: Sequence[Mapping[str, object]],
    n_values: Sequence[int],
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.5))
    for configuration, marker in (("correct_hierarchy", "o"), ("all_local", "s")):
        values = [
            mean_metric(
                rows_for(rows, model="Shared Global", configuration=configuration, n=n),
                "bic_weight",
            )
            for n in n_values
        ]
        axis.plot(n_values, values, marker=marker, label=CONFIGURATION_LABELS[configuration])
    invalid_values = [
        float(np.mean([
            float(row["invalid_shared_bic_weight"])
            for row in invalid_rows if int(row["n_observations"]) == n
        ]))
        for n in n_values
    ]
    axis.plot(
        n_values,
        invalid_values,
        marker="x",
        linestyle="--",
        color="tab:red",
        label="INVALID: all-local optimization with shared-model penalty",
    )
    axis.axhline(0.5, color="0.4", linestyle=":")
    axis.set(xscale="log", ylim=(-0.03, 1.03), xlabel="Number of observation pairs N",
             ylabel="BIC weight for Shared Global")
    axis.set_xticks(n_values, labels=[str(n) for n in n_values])
    axis.legend()
    save_figure(fig, path)


def plot_log_likelihood_difference(
    rows: Sequence[Mapping[str, object]], n_values: Sequence[int], path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.5))
    for configuration, marker in zip(CONFIGURATIONS, ("o", "s", "^")):
        differences = []
        for n in n_values:
            shared = rows_for(rows, model="Shared Global", configuration=configuration, n=n)
            local = rows_for(rows, model="Fully Local", configuration=configuration, n=n)
            shared_by_run = {int(row["run"]): float(row["log_likelihood"]) for row in shared}
            local_by_run = {int(row["run"]): float(row["log_likelihood"]) for row in local}
            differences.append(float(np.mean([
                shared_by_run[index] - local_by_run[index]
                for index in sorted(shared_by_run)
            ])))
        axis.plot(n_values, differences, marker=marker, label=CONFIGURATION_LABELS[configuration])
    axis.axhline(0.0, color="black", linestyle="--")
    axis.set(xscale="log", xlabel="Number of observation pairs N",
             ylabel="log L(shared) - log L(local)")
    axis.set_xticks(n_values, labels=[str(n) for n in n_values])
    axis.legend()
    save_figure(fig, path)


def plot_effective_parameter_count(n_values: Sequence[int], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.6, 4.4))
    axis.plot(n_values, [1 + n for n in n_values], "o-", label="Shared Global: 1 + N")
    axis.plot(n_values, [2 * n for n in n_values], "s-", label="Fully Local: 2N")
    axis.set(xscale="log", xlabel="Number of observation pairs N", ylabel="Effective fitted parameters k")
    axis.set_xticks(n_values, labels=[str(n) for n in n_values])
    axis.legend()
    save_figure(fig, path)


def plot_example_reconstruction(example: Mapping[str, np.ndarray | float], path: Path) -> None:
    indices = np.arange(len(np.asarray(example["local_true"])))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].axhline(float(example["global_true"]), color="tab:red", label="true g")
    axes[0].axhline(float(example["analytic_global"]), color="black", linestyle="--", label="analytic MAP")
    axes[0].axhline(float(example["compass_global"]), color="tab:blue", linestyle="-.", label="COMPASS MAP")
    axes[0].set(xlabel="Observation", ylabel="Global parameter", title="Shared global reconstruction")
    axes[0].legend()
    axes[1].plot(indices, example["local_true"], "x", color="tab:red", label="true local")
    axes[1].plot(indices, example["analytic_local"], "_", color="black", markersize=10, label="analytic MAP")
    axes[1].plot(indices, example["compass_local"], "o", color="tab:blue", fillstyle="none", label="COMPASS MAP")
    axes[1].set(xlabel="Observation index", ylabel="Local parameter", title="Observation-specific reconstruction")
    axes[1].legend()
    save_figure(fig, path)


def plot_confusion_table(
    rows: Sequence[Mapping[str, object]], n_max: int, path: Path,
) -> None:
    table = np.zeros((len(CONFIGURATIONS), 3), dtype=float)
    labels = ("Shared selected", "Local selected", "Ambiguous")
    for row_index, configuration in enumerate(CONFIGURATIONS):
        shared_rows = rows_for(
            rows, model="Shared Global", configuration=configuration, n=n_max
        )
        for row in shared_rows:
            weight = float(row["bic_weight"])
            if max(weight, 1.0 - weight) < AMBIGUOUS_WEIGHT_THRESHOLD:
                table[row_index, 2] += 1
            elif weight >= 0.5:
                table[row_index, 0] += 1
            else:
                table[row_index, 1] += 1
    row_totals = table.sum(axis=1, keepdims=True)
    table = np.divide(table, row_totals, out=np.zeros_like(table), where=row_totals > 0)
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    image = axis.imshow(table, vmin=0.0, vmax=1.0, cmap="Blues")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            axis.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center")
    axis.set(
        xticks=range(3),
        yticks=range(len(CONFIGURATIONS)),
        xticklabels=labels,
        yticklabels=[CONFIGURATION_LABELS[name] for name in CONFIGURATIONS],
        title=f"BIC decisions at N={n_max}; ambiguous if max weight < {AMBIGUOUS_WEIGHT_THRESHOLD}",
    )
    fig.colorbar(image, ax=axis, label="Fraction of mock runs")
    save_figure(fig, path)


def create_plots(
    rows: Sequence[Mapping[str, object]],
    invalid_rows: Sequence[Mapping[str, object]],
    example: Mapping[str, np.ndarray | float],
    cfg: ExperimentConfig,
) -> None:
    plot_dir = cfg.output_dir / "plots"
    plot_metric_by_configuration(
        rows, cfg.n_observations, "global_error", "Global-coordinate RMSE",
        plot_dir / "global_rmse.png", analytic_reference=True,
    )
    plot_metric_by_configuration(
        rows, cfg.n_observations, "local_rmse", "Local-parameter RMSE",
        plot_dir / "local_rmse.png", dashed_ablations=True,
    )
    plot_sharing_error(rows, cfg.n_observations, plot_dir / "sharing_error.png")
    plot_model_weights(rows, invalid_rows, cfg.n_observations, plot_dir / "model_weight.png")
    plot_log_likelihood_difference(
        rows, cfg.n_observations, plot_dir / "log_likelihood_difference.png"
    )
    plot_effective_parameter_count(
        cfg.n_observations, plot_dir / "effective_parameter_count.png"
    )
    plot_example_reconstruction(example, plot_dir / "example_map_reconstruction.png")
    plot_confusion_table(
        rows, max(cfg.n_observations), plot_dir / "confusion_style_table.png"
    )


def validate_results(
    rows: Sequence[Mapping[str, object]],
    invalid_rows: Sequence[Mapping[str, object]],
    cfg: ExperimentConfig,
) -> dict[str, bool]:
    n_min, n_max = min(cfg.n_observations), max(cfg.n_observations)
    correct_shared = rows_for(rows, model="Shared Global", configuration="correct_hierarchy")
    all_local_shared = rows_for(rows, model="Shared Global", configuration="all_local")
    all_global_shared = rows_for(rows, model="Shared Global", configuration="all_global")

    expected_counts = all(
        int(row["effective_k"]) == (
            1 + int(row["n_observations"])
            if row["model"] == "Shared Global" and row["configuration"] == "correct_hierarchy"
            else 2 * int(row["n_observations"])
            if row["configuration"] == "all_local"
            or (row["model"] == "Fully Local" and row["configuration"] == "correct_hierarchy")
            else 2
        )
        for row in rows
    )
    invalid_is_separate = all(
        row.get("configuration") != "INVALID: all-local optimization with shared-model penalty"
        for row in rows
    ) and bool(invalid_rows)

    correct_local_std = mean_metric(correct_shared, "local_estimate_std")
    all_global_local_std = max(float(row["local_estimate_std"]) for row in all_global_shared)
    results = {
        "global_sharing_enforced": max(float(row["sharing_range"]) for row in correct_shared) < GLOBAL_SHARING_ATOL,
        "global_rmse_improves": mean_metric(rows_for(correct_shared, n=n_max), "global_error")
        < mean_metric(rows_for(correct_shared, n=n_min), "global_error"),
        "local_parameters_distinct": correct_local_std > 10.0 * GLOBAL_SHARING_ATOL,
        "shared_bic_improves": mean_metric(rows_for(correct_shared, n=n_max), "bic_weight")
        > mean_metric(rows_for(correct_shared, n=n_min), "bic_weight"),
        "correct_selected_more_than_all_local": mean_metric(
            rows_for(correct_shared, n=n_max), "selected_by_bic"
        ) > mean_metric(rows_for(all_local_shared, n=n_max), "selected_by_bic"),
        "all_local_sharing_broken": mean_metric(all_local_shared, "sharing_range") > GLOBAL_SHARING_ATOL,
        "all_local_models_ambiguous": abs(mean_metric(all_local_shared, "bic_weight") - 0.5) < 0.25,
        "all_global_local_collapse": all_global_local_std < GLOBAL_SHARING_ATOL,
        "all_global_reconstruction_degrades": mean_metric(all_global_shared, "local_rmse")
        > mean_metric(correct_shared, "local_rmse"),
        "effective_counts_exact": expected_counts,
        "invalid_penalty_separate": invalid_is_separate,
        "invalid_penalty_bias_demonstrated": float(np.mean([
            float(row["invalid_shared_bic_weight"]) for row in invalid_rows
        ])) > mean_metric(all_local_shared, "bic_weight"),
    }
    structural = (
        results["global_sharing_enforced"]
        and results["all_global_local_collapse"]
        and results["effective_counts_exact"]
        and results["invalid_penalty_separate"]
    )
    if not structural:
        failed = [name for name, passed in results.items() if not passed]
        raise AssertionError(f"Structural hierarchical validation failed: {failed}.")
    return results


def print_validation_summary(results: Mapping[str, bool], cfg: ExperimentConfig) -> None:
    status = lambda key: "PASS" if results[key] else "FAIL"
    print("\nHierarchical Gaussian MAP validation")
    print("------------------------------------")
    print("Correct hierarchy:")
    print(f"  global sharing enforced: {status('global_sharing_enforced')}")
    print(f"  global RMSE improves with N: {status('global_rmse_improves')}")
    print(f"  local parameters remain distinct: {status('local_parameters_distinct')}")
    print(f"  shared model BIC selection improves with N: {status('shared_bic_improves')}")
    print(
        "  selected more often than under all-local inference: "
        f"{status('correct_selected_more_than_all_local')}"
    )
    print("\nAll-local ablation:")
    print(f"  global sharing intentionally broken: {status('all_local_sharing_broken')}")
    print(f"  models become approximately equivalent: {status('all_local_models_ambiguous')}")
    print("\nAll-global ablation:")
    print(f"  local parameters collapse: {status('all_global_local_collapse')}")
    print(f"  reconstruction degrades: {status('all_global_reconstruction_degrades')}")
    print("\nInvalid penalty demonstration:")
    print(
        "  over-flexible MAP plus under-counted k produces biased selection: "
        + ("DEMONSTRATED" if results["invalid_penalty_bias_demonstrated"] else "NOT DEMONSTRATED")
    )
    print("\nOutputs:")
    print(f"  checkpoints: {cfg.data_dir / 'checkpoints'}")
    print(f"  detailed CSV: {cfg.output_dir / 'hierarchical_gaussian_map_detailed.csv'}")
    print(f"  summary CSV: {cfg.output_dir / 'hierarchical_gaussian_map_summary.csv'}")
    print(f"  configuration JSON: {cfg.output_dir / 'experiment_config.json'}")
    print(f"  plots: {cfg.output_dir / 'plots'}")


def run_experiment(
    cfg: ExperimentConfig,
    models: Mapping[str, SBIm],
    normalization: Normalization,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray | float]]:
    detailed_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    example: dict[str, np.ndarray | float] | None = None

    for n_index, n_observations in enumerate(cfg.n_observations):
        for run in range(cfg.n_runs):
            mock_seed = cfg.seed + 10_000 + 10_000 * n_index + run
            global_true, local_true, observations = generate_mock_dataset(
                n_observations, mock_seed
            )
            references = analytic_maps(observations)
            for configuration in CONFIGURATIONS:
                print(
                    f"\nComparison N={n_observations}, run={run}, "
                    f"configuration={CONFIGURATION_LABELS[configuration]}"
                )
                comparison: dict[str, dict[str, object]] = {}
                for model_index, model_name in enumerate(MODEL_NAMES):
                    global_indices, local_indices = roles_for(model_name, configuration)
                    print(
                        f"  {model_name}: global={list(global_indices)}, "
                        f"local={list(local_indices)}"
                    )
                    inference_seed = (
                        cfg.seed
                        + 100_000
                        + 10_000 * n_index
                        + 100 * run
                        + 10 * model_index
                        + sum(global_indices)
                        + 3 * len(global_indices)
                    )
                    started = time.perf_counter()
                    result = infer_hierarchical_map(
                        models[model_name], observations, global_indices, local_indices,
                        normalization, cfg, inference_seed,
                    )
                    log_likelihoods = evaluate_likelihood(
                        models[model_name], result, observations, normalization, cfg
                    )
                    total_log_likelihood = float(log_likelihoods.sum())
                    k = effective_parameter_count(
                        n_observations, global_indices, local_indices
                    )
                    criteria = information_criteria(total_log_likelihood, k, n_observations)
                    metrics = map_metrics(
                        model_name, result, global_true, local_true, references
                    )
                    print(f"    effective k={k}")
                    if run == 0:
                        print(
                            f"    global MAP={result.global_map.tolist()}, "
                            f"local MAP shape={result.local_map.shape}"
                        )
                    comparison[model_name] = {
                        "result": result,
                        "log_likelihoods": log_likelihoods,
                        "log_likelihood": total_log_likelihood,
                        "effective_k": k,
                        "criteria": criteria,
                        "metrics": metrics,
                        "runtime_seconds": time.perf_counter() - started,
                    }

                aic_weights = normalized_weights(
                    [comparison[name]["criteria"]["aic"] for name in MODEL_NAMES],
                    lower_is_better=True,
                )
                bic_weights = normalized_weights(
                    [comparison[name]["criteria"]["bic"] for name in MODEL_NAMES],
                    lower_is_better=True,
                )
                likelihood_weights = normalized_weights(
                    [comparison[name]["log_likelihood"] for name in MODEL_NAMES],
                    lower_is_better=False,
                )
                for model_index, model_name in enumerate(MODEL_NAMES):
                    item = comparison[model_name]
                    result = item["result"]
                    row = {
                        "n_observations": n_observations,
                        "run": run,
                        "model": model_name,
                        "configuration": configuration,
                        "global_indices": json.dumps(result.global_indices),
                        "local_indices": json.dumps(result.local_indices),
                        "true_global": global_true,
                        "estimated_global": item["metrics"]["estimated_global"],
                        "analytic_global_map": references["global_map"],
                        "global_error": item["metrics"]["global_error"],
                        "global_analytic_error": item["metrics"]["global_analytic_error"],
                        "local_rmse": item["metrics"]["local_rmse"],
                        "local_analytic_rmse": item["metrics"]["local_analytic_rmse"],
                        "sharing_range": item["metrics"]["sharing_range"],
                        "sharing_std": item["metrics"]["sharing_std"],
                        "local_estimate_std": item["metrics"]["local_estimate_std"],
                        "per_observation_log_likelihoods": json.dumps(
                            np.asarray(item["log_likelihoods"]).tolist()
                        ),
                        "log_likelihood": item["log_likelihood"],
                        "effective_k": item["effective_k"],
                        "aic": item["criteria"]["aic"],
                        "aicc": item["criteria"]["aicc"],
                        "bic": item["criteria"]["bic"],
                        "aic_weight": float(aic_weights[model_index]),
                        "bic_weight": float(bic_weights[model_index]),
                        "no_penalty_weight": float(likelihood_weights[model_index]),
                        "selected_by_aic": int(aic_weights[model_index] == aic_weights.max()),
                        "selected_by_bic": int(bic_weights[model_index] == bic_weights.max()),
                        "runtime_seconds": item["runtime_seconds"],
                    }
                    detailed_rows.append(row)

                if configuration == "all_local":
                    shared = comparison["Shared Global"]
                    local = comparison["Fully Local"]
                    invalid_shared_k = 1 + n_observations
                    invalid_shared_bic = information_criteria(
                        float(shared["log_likelihood"]), invalid_shared_k, n_observations
                    )["bic"]
                    local_bic = float(local["criteria"]["bic"])
                    invalid_weight = normalized_weights(
                        [invalid_shared_bic, local_bic], lower_is_better=True
                    )[0]
                    invalid_rows.append({
                        "label": "INVALID: all-local optimization with shared-model penalty",
                        "n_observations": n_observations,
                        "run": run,
                        "shared_all_local_log_likelihood": shared["log_likelihood"],
                        "local_all_local_log_likelihood": local["log_likelihood"],
                        "invalid_shared_k": invalid_shared_k,
                        "valid_shared_all_local_k": 2 * n_observations,
                        "local_k": 2 * n_observations,
                        "invalid_shared_bic": invalid_shared_bic,
                        "local_bic": local_bic,
                        "invalid_shared_bic_weight": float(invalid_weight),
                    })

                if (
                    example is None
                    and n_observations == 10
                    and run == 0
                    and configuration == "correct_hierarchy"
                ):
                    shared_result = comparison["Shared Global"]["result"]
                    example = {
                        "global_true": global_true,
                        "analytic_global": float(references["global_map"]),
                        "compass_global": float(shared_result.global_map[0]),
                        "local_true": local_true.copy(),
                        "analytic_local": np.asarray(references["local_map"]).copy(),
                        "compass_local": shared_result.expanded_map[:, 1].copy(),
                        "observations": observations.copy(),
                    }

    if example is None:
        # A custom observation list may omit N=10.  Use the first correct case so
        # the required reconstruction plot remains available and label its true N.
        first = rows_for(
            detailed_rows, model="Shared Global", configuration="correct_hierarchy"
        )[0]
        fallback_n = int(first["n_observations"])
        global_true, local_true, observations = generate_mock_dataset(
            fallback_n, cfg.seed + 10_000
        )
        references = analytic_maps(observations)
        roles = roles_for("Shared Global", "correct_hierarchy")
        result = infer_hierarchical_map(
            models["Shared Global"], observations, roles[0], roles[1], normalization,
            cfg, cfg.seed + 999_999,
        )
        example = {
            "global_true": global_true,
            "analytic_global": float(references["global_map"]),
            "compass_global": float(result.global_map[0]),
            "local_true": local_true,
            "analytic_local": np.asarray(references["local_map"]),
            "compass_local": result.expanded_map[:, 1],
            "observations": observations,
        }
    return detailed_rows, invalid_rows, example


def save_configuration(
    cfg: ExperimentConfig,
    normalization: Normalization,
    data_path: Path,
) -> None:
    payload = asdict(cfg)
    payload["data_dir"] = str(cfg.data_dir)
    payload["output_dir"] = str(cfg.output_dir)
    payload["n_observations"] = list(cfg.n_observations)
    payload.update({
        "model_names": list(MODEL_NAMES),
        "sigma_g": SIGMA_G,
        "sigma_local": SIGMA_LOCAL,
        "sigma_x": SIGMA_X,
        "model_kwargs": MODEL_KWARGS,
        "training_batch_size": TRAIN_BATCH_SIZE,
        "training_lr": TRAIN_LR,
        "joint_mean": normalization.mean.tolist(),
        "joint_std": normalization.std.tolist(),
        "dataset_path": str(data_path),
        "global_sharing_atol": GLOBAL_SHARING_ATOL,
        "ambiguous_weight_threshold": AMBIGUOUS_WEIGHT_THRESHOLD,
        "map_method": "hierarchical sampling plus score-based local refinement",
        "likelihood_method": "COMPASS probability-flow ODE, exact divergence",
        "effective_sample_size_definition": "N independent observation pairs",
        "parameter_roles": {
            configuration: {
                model: {
                    "global_indices": list(roles_for(model, configuration)[0]),
                    "local_indices": list(roles_for(model, configuration)[1]),
                }
                for model in MODEL_NAMES
            }
            for configuration in CONFIGURATIONS
        },
    })
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "experiment_config.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_config(args: argparse.Namespace, device: str) -> ExperimentConfig:
    if args.n_runs is not None and args.n_runs < 1:
        raise ValueError("--n-runs must be positive.")
    if args.num_samples is not None and args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2.")
    if args.timesteps is not None and args.timesteps < 2:
        raise ValueError("--timesteps must be at least 2.")
    return ExperimentConfig(
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        device=device,
        seed=args.seed,
        quick=args.quick,
        train_n=10_000 if args.quick else TRAIN_N,
        val_n=1_000 if args.quick else VAL_N,
        n_observations=(2, 5, 10) if args.quick else N_OBSERVATIONS_LIST,
        n_runs=args.n_runs if args.n_runs is not None else (5 if args.quick else N_MOCK_RUNS),
        num_samples=args.num_samples if args.num_samples is not None else (256 if args.quick else 1_000),
        timesteps=args.timesteps if args.timesteps is not None else (20 if args.quick else 50),
        max_epochs=50 if args.quick else TRAIN_MAX_EPOCHS,
        patience=10 if args.quick else TRAIN_EARLY_STOPPING_PATIENCE,
        force_data=args.force_data,
        force_train=args.force_train,
    )


def main() -> None:
    args = parse_args()
    if CPU_LIMIT_INFO is None:
        raise RuntimeError("CPU cap was not configured before scientific imports.")
    logical_cpus, active_cpus = CPU_LIMIT_INFO
    active_now = tuple(sorted(os.sched_getaffinity(0)))
    if active_now != active_cpus or len(active_now) > CPU_THREAD_LIMIT:
        raise RuntimeError("The active CPU affinity no longer satisfies the 3-thread hard cap.")
    print(
        f"CPU limited to {len(active_cpus)} of {logical_cpus} logical CPUs "
        f"({len(active_cpus) / logical_cpus:.2%} capacity)."
    )
    device = select_device(args.device)
    cfg = build_config(args, device)
    seed_all(cfg.seed)
    configure_plot_style()

    datasets, normalization, data_path = prepare_datasets(cfg)
    models = {
        model_name: load_or_train_model(
            cfg, model_name, datasets[model_name], normalization
        )
        for model_name in MODEL_NAMES
    }
    save_configuration(cfg, normalization, data_path)
    rows, invalid_rows, example = run_experiment(cfg, models, normalization)

    detailed_path = cfg.output_dir / "hierarchical_gaussian_map_detailed.csv"
    summary_path = cfg.output_dir / "hierarchical_gaussian_map_summary.csv"
    invalid_path = cfg.output_dir / "invalid_counterfactual.csv"
    write_rows(detailed_path, rows)
    write_rows(summary_path, build_summary_rows(rows))
    write_rows(invalid_path, invalid_rows)
    np.savez_compressed(cfg.output_dir / "example_map_reconstruction.npz", **example)
    create_plots(rows, invalid_rows, example, cfg)
    validation = validate_results(rows, invalid_rows, cfg)
    (cfg.output_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print_validation_summary(validation, cfg)


if __name__ == "__main__":
    main()
