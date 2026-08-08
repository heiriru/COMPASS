#!/usr/bin/env python3
"""Hierarchical linear-versus-quadratic COMPASS mini-experiment.

This standalone experiment trains one linear and one quadratic COMPASS model,
then reuses those networks under several global/local MAP hierarchies.  The
current ``ModelTransfuser.compare`` interface accepts one hierarchy for every
candidate and counts raw latent width, so it cannot represent the model-specific
roles or ``d_G + N d_L`` required here.  We therefore call the same public
COMPASS sampling, score-MAP, and PF-ODE likelihood APIs directly and use
``ModelTransfuser``'s information-criterion implementations with explicit
effective parameter counts.

Run the validation-sized experiment with::

    python tutorials/Hierarchical_Linear_Quadratic.py --quick
"""

from __future__ import annotations

import os


CPU_THREAD_LIMIT = 3
CPU_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_cpu_limit() -> tuple[int, tuple[int, ...]]:
    """Enforce the repository CPU cap before importing native libraries."""
    logical = os.cpu_count() or 1
    limit = min(CPU_THREAD_LIMIT, logical)
    if limit < 1:
        raise RuntimeError(
            f"The {CPU_THREAD_LIMIT}-thread CPU cap permits no CPU on a {logical}-CPU host."
        )
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    selected = allowed[:limit]
    if not selected:
        raise RuntimeError("No logical CPUs are available in this process affinity mask.")
    os.sched_setaffinity(0, selected)
    for name in CPU_ENV_VARS:
        os.environ[name] = str(len(selected))
    return logical, selected


CPU_LIMIT = configure_cpu_limit() if __name__ == "__main__" else None

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

from compass import ModelTransfuser
from compass import ScoreBasedInferenceModel as SBIm


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    ROOT / "output" / "compositional_inference" / "08_miniexperiment"
)

T_VALUES = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
C_TRUE_VALUES = (0.0, 0.6)
N_OBSERVATIONS_LIST = (2, 5, 10, 25, 50)
N_MOCK_RUNS = 50
TRAIN_N = 100_000
VAL_N = 5_000
SIGMA_X = 0.15
SEED = 42

# The problem specifies priors for a and b.  A unit Normal training prior for c
# covers both requested truth values and makes the normalized hierarchy prior
# exactly standard Normal in every latent coordinate.
PARAMETER_PRIOR_STD = {"a": 1.0, "c": 1.0, "b": 1.5}
MODEL_NAMES = ("linear", "quadratic")
PARAMETER_NAMES = {"linear": ("a", "b"), "quadratic": ("a", "c", "b")}
CONFIGURATIONS = ("correct", "curvature_local", "all_local", "all_global")
CONFIG_LABELS = {
    "correct": "Correct hierarchy",
    "curvature_local": "Curvature local",
    "all_local": "All local",
    "all_global": "All global",
}
GLOBAL_ATOL = 1e-6

TRAIN_BATCH_SIZE = 256
TRAIN_MAX_EPOCHS = 250
TRAIN_PATIENCE = 25
TRAIN_LR = 1e-3
MODEL_KWARGS = {
    "sde_type": "vesde", "sigma": 3.0, "hidden_size": 48,
    "depth": 4, "num_heads": 4, "mlp_ratio": 3,
}


@dataclass(frozen=True)
class Config:
    output_dir: Path
    device: str
    seed: int
    quick: bool
    train_n: int
    val_n: int
    n_values: tuple[int, ...]
    n_runs: int
    posterior_samples: int
    timesteps: int
    max_epochs: int
    patience: int
    force_train: bool


@dataclass(frozen=True)
class Normalization:
    theta_std: np.ndarray
    x_std: np.ndarray

    def theta_normalize(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) / self.theta_std

    def theta_physical(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) * self.theta_std

    def x_normalize(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) / self.x_std

    @property
    def x_log_jacobian(self) -> float:
        return float(np.log(self.x_std).sum())


@dataclass(frozen=True)
class MapResult:
    global_indices: tuple[int, ...]
    local_indices: tuple[int, ...]
    expanded_theta: np.ndarray
    posterior_std: np.ndarray


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> str:
    """Reserve one free GPU before training/inference, falling back to CPU."""
    if requested == "cpu":
        return "cpu"
    if autocvd is None:
        print("autocvd is unavailable; using CPU.")
        return "cpu"
    try:
        selected = autocvd(num_gpus=1, interval=1, timeout=60, progress=False)
    except (OSError, TimeoutError, subprocess.SubprocessError) as error:
        print(f"GPU reservation failed ({error}); using CPU.")
        return "cpu"
    if torch.cuda.is_available():
        print(f"Reserved GPU {selected[0]} through autocvd.")
        return "cuda"
    print("autocvd selected a GPU but CUDA is unavailable to Torch; using CPU.")
    return "cpu"


def parameter_roles(model: str, configuration: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return actual global/local latent coordinates for one candidate."""
    width = len(PARAMETER_NAMES[model])
    if configuration == "correct":
        global_indices = (0,) if model == "linear" else (0, 1)
    elif configuration == "curvature_local":
        global_indices = (0,)
    elif configuration == "all_local":
        global_indices = ()
    elif configuration == "all_global":
        global_indices = tuple(range(width))
    else:
        raise ValueError(f"Unknown hierarchy configuration {configuration!r}.")
    local_indices = tuple(index for index in range(width) if index not in global_indices)
    return global_indices, local_indices


def normalization_for(model: str) -> Normalization:
    names = PARAMETER_NAMES[model]
    theta_std = np.asarray([PARAMETER_PRIOR_STD[name] for name in names], dtype=np.float32)
    design = design_matrix(model)
    x_variance = np.square(design * theta_std[None, :]).sum(axis=1) + SIGMA_X**2
    return Normalization(theta_std=theta_std, x_std=np.sqrt(x_variance).astype(np.float32))


def design_matrix(model: str) -> np.ndarray:
    if model == "linear":
        return np.column_stack([T_VALUES, np.ones_like(T_VALUES)]).astype(np.float32)
    if model == "quadratic":
        return np.column_stack([T_VALUES, T_VALUES**2, np.ones_like(T_VALUES)]).astype(np.float32)
    raise ValueError(model)


def simulate_training(model: str, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    norm = normalization_for(model)
    theta = rng.normal(size=(n, len(PARAMETER_NAMES[model]))).astype(np.float32)
    theta *= norm.theta_std
    x = theta @ design_matrix(model).T
    x += rng.normal(0.0, SIGMA_X, size=x.shape).astype(np.float32)
    return (
        torch.as_tensor(norm.theta_normalize(theta), dtype=torch.float32),
        torch.as_tensor(norm.x_normalize(x), dtype=torch.float32),
    )


def checkpoint_signature(cfg: Config, model: str) -> dict[str, object]:
    norm = normalization_for(model)
    return {
        "version": 1,
        "model": model,
        "parameter_names": list(PARAMETER_NAMES[model]),
        "train_n": cfg.train_n,
        "val_n": cfg.val_n,
        "seed": cfg.seed,
        "batch_size": TRAIN_BATCH_SIZE,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "learning_rate": TRAIN_LR,
        "model_kwargs": MODEL_KWARGS,
        "theta_std": norm.theta_std.tolist(),
        "x_std": norm.x_std.tolist(),
    }


def load_or_train(cfg: Config, model_name: str) -> SBIm:
    mode = "quick" if cfg.quick else "full"
    model_dir = cfg.output_dir / "models" / mode / model_name
    checkpoint = model_dir / "Model_checkpoint.pt"
    metadata = model_dir / "metadata.json"
    signature = checkpoint_signature(cfg, model_name)
    compatible = False
    if checkpoint.exists() and metadata.exists() and not cfg.force_train:
        try:
            compatible = json.loads(metadata.read_text()) == signature
        except (json.JSONDecodeError, OSError):
            compatible = False
    if compatible:
        print(f"Loading {model_name} checkpoint: {checkpoint}")
        return SBIm.load(str(checkpoint), device=cfg.device)

    theta_train, x_train = simulate_training(model_name, cfg.train_n, cfg.seed + 100)
    theta_val, x_val = simulate_training(model_name, cfg.val_n, cfg.seed + 200)
    seed_all(cfg.seed + MODEL_NAMES.index(model_name))
    model = SBIm(
        nodes_size=len(PARAMETER_NAMES[model_name]) + len(T_VALUES),
        device=cfg.device,
        **MODEL_KWARGS,
    )
    print(f"Training {model_name} model on {cfg.device}: {model_dir}")
    model.train(
        theta_train, x_train, theta_val=theta_val, x_val=x_val,
        batch_size=TRAIN_BATCH_SIZE, max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.patience, lr=TRAIN_LR,
        time_sampling="mixture", device=cfg.device, verbose=False,
        path=str(model_dir), name="Model",
    )
    if not checkpoint.exists():
        raise RuntimeError(f"Training did not create {checkpoint}.")
    metadata.write_text(json.dumps(signature, indent=2) + "\n")
    return SBIm.load(str(checkpoint), device=cfg.device)


def generate_mock(max_n: int, c_true: float, seed: int) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    a_true = float(rng.normal())
    b_true = rng.normal(0.0, PARAMETER_PRIOR_STD["b"], size=max_n)
    mean = b_true[:, None] + a_true * T_VALUES[None, :] + c_true * T_VALUES[None, :] ** 2
    x = mean + rng.normal(0.0, SIGMA_X, size=mean.shape)
    return {
        "a_true": a_true,
        "c_true": float(c_true),
        "b_true": b_true.astype(np.float32),
        "x": x.astype(np.float32),
    }


def analytic_references(x: np.ndarray) -> dict[str, np.ndarray | float]:
    a_ref = float(np.mean((x[:, 2] - x[:, 0]) / 2.0))
    c_ref = float(np.mean((x[:, 0] + x[:, 2]) / 2.0 - x[:, 1]))
    return {"a_ref": a_ref, "c_ref": c_ref, "b_ref": x[:, 1].copy()}


def posterior_covariance(model: str, global_indices: Sequence[int], norm: Normalization) -> torch.Tensor:
    """Exact single-observation marginal posterior covariance in normalized theta."""
    design = design_matrix(model).astype(np.float64)
    prior_precision = np.diag(1.0 / np.square(norm.theta_std.astype(np.float64)))
    full_precision = prior_precision + design.T @ design / SIGMA_X**2
    covariance = np.linalg.inv(full_precision)
    indices = np.asarray(global_indices, dtype=int)
    marginal_covariance = covariance[np.ix_(indices, indices)]
    normalized_covariance = marginal_covariance / np.outer(
        norm.theta_std[indices], norm.theta_std[indices]
    )
    return torch.as_tensor(normalized_covariance, dtype=torch.float32)


def infer_map(
    model: SBIm,
    model_name: str,
    x_physical: np.ndarray,
    configuration: str,
    cfg: Config,
    inference_seed: int,
) -> MapResult:
    """Run COMPASS hierarchical inference and refine only actual local coordinates."""
    norm = normalization_for(model_name)
    global_indices, local_indices = parameter_roles(model_name, configuration)
    x = torch.as_tensor(norm.x_normalize(x_physical), dtype=torch.float32)
    n = len(x)
    seed_all(inference_seed)
    common = dict(
        x=x, num_samples=cfg.posterior_samples, timesteps=cfg.timesteps,
        method="dpm", order=2, corrector_steps_interval=1,
        corrector_steps=3 if cfg.quick else 7, final_corrector_steps=2,
        snr=0.12, device=cfg.device, verbose=False,
    )
    if global_indices:
        samples = model.sample(
            **common, multi_obs_inference=True, hierarchy=list(global_indices),
            prior=([0.0] * len(global_indices), [1.0] * len(global_indices)),
            correction="full_gaussian",
            posterior_covariance=posterior_covariance(model_name, global_indices, norm),
        )
    else:
        samples = model.sample(**common, multi_obs_inference=False)

    width = len(PARAMETER_NAMES[model_name])
    expected_shape = (n, cfg.posterior_samples, width)
    if tuple(samples.shape) != expected_shape:
        raise AssertionError(f"Expected posterior shape {expected_shape}, got {tuple(samples.shape)}.")
    samples = samples.detach().cpu()
    for index in global_indices:
        deviation = float((samples[:, :, index] - samples[0:1, :, index]).abs().max())
        assert deviation <= GLOBAL_ATOL, (
            f"COMPASS did not share {model_name}.{PARAMETER_NAMES[model_name][index]}: "
            f"max deviation={deviation:.3e}."
        )

    posterior_mean = samples.mean(dim=1)
    posterior_std = samples.std(dim=1, unbiased=False)
    joint_init = torch.cat([posterior_mean, x], dim=1)
    condition_mask = torch.cat([torch.zeros(width), torch.ones(len(T_VALUES))])
    if global_indices:
        condition_mask[list(global_indices)] = 1.0
    if local_indices:
        sigma_start = max(
            2.0 * float(posterior_std[:, list(local_indices)].max()), 1e-3
        )
        joint_map = model.map_estimate(
            joint_init, condition_mask, sigma_start=sigma_start,
            timesteps=max(10, cfg.timesteps // 2), iterations_per_level=2,
            device=cfg.device,
        )
    else:
        joint_map = joint_init
    theta = norm.theta_physical(joint_map[:, :width].numpy())
    for index in global_indices:
        assert float(np.ptp(theta[:, index])) <= GLOBAL_ATOL
    return MapResult(
        global_indices=global_indices,
        local_indices=local_indices,
        expanded_theta=theta,
        posterior_std=posterior_std.numpy(),
    )


def evaluate_log_likelihood(
    model: SBIm, model_name: str, result: MapResult,
    x_physical: np.ndarray, cfg: Config,
) -> tuple[np.ndarray, float]:
    norm = normalization_for(model_name)
    theta = norm.theta_normalize(result.expanded_theta)
    x = norm.x_normalize(x_physical)
    joint = torch.as_tensor(np.concatenate([theta, x], axis=1), dtype=torch.float32)
    width = theta.shape[1]
    mask = torch.cat([torch.ones(width), torch.zeros(len(T_VALUES))])
    normalized = model.log_prob(
        joint, condition_mask=mask, timesteps=cfg.timesteps,
        divergence="exact", device=cfg.device, verbose=False,
    ).detach().cpu().numpy()
    physical = normalized - norm.x_log_jacobian
    return physical, float(physical.sum())


def effective_k(model: str, configuration: str, n: int) -> int:
    global_indices, local_indices = parameter_roles(model, configuration)
    return len(global_indices) + n * len(local_indices)


def expected_k(model: str, configuration: str, n: int) -> int:
    if model == "linear" and configuration in ("correct", "curvature_local"):
        return 1 + n
    if model == "quadratic" and configuration == "correct":
        return 2 + n
    if model == "quadratic" and configuration == "curvature_local":
        return 1 + 2 * n
    if configuration == "all_local":
        return (2 if model == "linear" else 3) * n
    if configuration == "all_global":
        return 2 if model == "linear" else 3
    raise ValueError((model, configuration))


def criteria(engine: ModelTransfuser, log_likelihood: float, k: int, n: int) -> dict[str, float]:
    ll = torch.tensor(log_likelihood, dtype=torch.float32)
    return {
        "aic": float(engine._aic(ll, k)),
        "bic": float(engine._bic(ll, k, n)),
    }


def weights(score_by_model: Mapping[str, float]) -> dict[str, float]:
    values = np.asarray([score_by_model[name] for name in MODEL_NAMES], dtype=np.float64)
    logits = -0.5 * (values - values.min())
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return dict(zip(MODEL_NAMES, map(float, probabilities)))


def parameter_metrics(
    model: str, result: MapResult, truth: Mapping[str, object],
    refs: Mapping[str, object],
) -> dict[str, float]:
    names = PARAMETER_NAMES[model]
    theta = result.expanded_theta
    a = theta[:, names.index("a")]
    if "c" in names:
        c = theta[:, names.index("c")]
    else:
        c = np.zeros(len(theta), dtype=np.float32)
    b = theta[:, names.index("b")]
    a_true = float(truth["a_true"])
    c_true = float(truth["c_true"])
    b_true = np.asarray(truth["b_true"])
    return {
        "a_error": float(np.sqrt(np.mean(np.square(a - a_true)))),
        "a_reference_error": float(np.sqrt(np.mean(np.square(a - float(refs["a_ref"]))))),
        "c_error": float(np.sqrt(np.mean(np.square(c - c_true)))),
        "c_reference_error": float(np.sqrt(np.mean(np.square(c - float(refs["c_ref"]))))),
        "b_rmse": float(np.sqrt(np.mean(np.square(b - b_true)))),
        "b_reference_rmse": float(np.sqrt(np.mean(np.square(b - np.asarray(refs["b_ref"]))))),
        "a_sharing_range": float(np.ptp(a)),
        "a_sharing_std": float(np.std(a)),
        "c_sharing_range": float(np.ptp(c)) if "c" in names else math.nan,
        "c_sharing_std": float(np.std(c)) if "c" in names else math.nan,
        "b_inferred_std": float(np.std(b)),
        "global_parameter_rmse": float(
            np.sqrt(np.mean([np.mean(np.square(a - a_true)), np.mean(np.square(c - c_true))]))
            if model == "quadratic" else np.sqrt(np.mean(np.square(a - a_true)))
        ),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grouped_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "a_error", "a_reference_error", "c_error", "c_reference_error",
        "b_rmse", "b_reference_rmse", "a_sharing_range", "a_sharing_std",
        "c_sharing_range", "c_sharing_std", "b_inferred_std",
        "global_parameter_rmse", "total_log_likelihood", "effective_k",
        "aic", "bic", "aic_weight", "bic_weight", "selected_aic", "selected_bic",
        "mispenalized_aic_weight", "mispenalized_bic_weight",
    )
    groups: dict[tuple[float, int, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            float(row["c_true"]), int(row["n_observations"]),
            str(row["configuration"]), str(row["model"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (c_true, n, configuration, model), group in sorted(groups.items()):
        for metric in metrics:
            values = np.asarray([float(row.get(metric, math.nan)) for row in group])
            values = values[np.isfinite(values)]
            output.append({
                "c_true": c_true, "n_observations": n,
                "configuration": configuration, "model": model, "metric": metric,
                "mean": float(values.mean()) if values.size else math.nan,
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else math.nan,
                "median": float(np.median(values)) if values.size else math.nan,
                "q05": float(np.quantile(values, 0.05)) if values.size else math.nan,
                "q95": float(np.quantile(values, 0.95)) if values.size else math.nan,
                "count": int(values.size),
            })
    return output


def subset(
    rows: Sequence[Mapping[str, object]], *, c_true: float | None = None,
    n: int | None = None, configuration: str | None = None,
    model: str | None = None,
) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if (c_true is None or math.isclose(float(row["c_true"]), c_true))
        and (n is None or int(row["n_observations"]) == n)
        and (configuration is None or row["configuration"] == configuration)
        and (model is None or row["model"] == model)
    ]


def mean_metric(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def run_experiment(
    cfg: Config, models: Mapping[str, SBIm],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    example: dict[str, object] | None = None
    comparison_engine = ModelTransfuser(path=None)
    max_n = max(cfg.n_values)
    for c_index, c_true in enumerate(C_TRUE_VALUES):
        for run in range(cfg.n_runs):
            # Nested prefixes reduce Monte Carlo noise in the requested N trends.
            full_truth = generate_mock(max_n, c_true, cfg.seed + 10_000 * (c_index + 1) + run)
            for n_index, n in enumerate(cfg.n_values):
                truth = {
                    "a_true": full_truth["a_true"], "c_true": c_true,
                    "b_true": np.asarray(full_truth["b_true"])[:n],
                }
                x = np.asarray(full_truth["x"])[:n]
                refs = analytic_references(x)
                for configuration in CONFIGURATIONS:
                    print(
                        f"c={c_true:g}, N={n}, run={run + 1}/{cfg.n_runs}, "
                        f"hierarchy={configuration}"
                    )
                    results: dict[str, dict[str, object]] = {}
                    for model_index, model_name in enumerate(MODEL_NAMES):
                        started = time.perf_counter()
                        result = infer_map(
                            models[model_name], model_name, x, configuration, cfg,
                            cfg.seed + 100_000 + 10_000 * c_index + 1_000 * run
                            + 100 * n_index + 10 * CONFIGURATIONS.index(configuration)
                            + model_index,
                        )
                        log_probs, total_ll = evaluate_log_likelihood(
                            models[model_name], model_name, result, x, cfg
                        )
                        k = effective_k(model_name, configuration, n)
                        assert k == expected_k(model_name, configuration, n), (
                            model_name, configuration, n, k,
                            expected_k(model_name, configuration, n),
                        )
                        results[model_name] = {
                            "map": result, "log_probs": log_probs,
                            "total_ll": total_ll, "k": k,
                            "criteria": criteria(comparison_engine, total_ll, k, n),
                            "metrics": parameter_metrics(model_name, result, truth, refs),
                            "runtime": time.perf_counter() - started,
                        }

                    aic_weights = weights({m: results[m]["criteria"]["aic"] for m in MODEL_NAMES})
                    bic_weights = weights({m: results[m]["criteria"]["bic"] for m in MODEL_NAMES})
                    invalid_aic_weights = {m: math.nan for m in MODEL_NAMES}
                    invalid_bic_weights = {m: math.nan for m in MODEL_NAMES}
                    invalid_selected_aic = ""
                    invalid_selected_bic = ""
                    if configuration == "curvature_local":
                        invalid_k = {"linear": 1 + n, "quadratic": 2 + n}
                        invalid_scores = {
                            criterion_name: {
                                m: criteria(
                                    comparison_engine, float(results[m]["total_ll"]),
                                    invalid_k[m], n,
                                )[criterion_name]
                                for m in MODEL_NAMES
                            }
                            for criterion_name in ("aic", "bic")
                        }
                        invalid_aic_weights = weights(invalid_scores["aic"])
                        invalid_bic_weights = weights(invalid_scores["bic"])
                        invalid_selected_aic = max(invalid_aic_weights, key=invalid_aic_weights.get)
                        invalid_selected_bic = max(invalid_bic_weights, key=invalid_bic_weights.get)

                    selected_aic = max(aic_weights, key=aic_weights.get)
                    selected_bic = max(bic_weights, key=bic_weights.get)
                    for model_name in MODEL_NAMES:
                        item = results[model_name]
                        result = item["map"]
                        row = {
                            "c_true": c_true, "n_observations": n, "run": run,
                            "configuration": configuration, "model": model_name,
                            "global_indices": json.dumps(result.global_indices),
                            "local_indices": json.dumps(result.local_indices),
                            "a_true": truth["a_true"], "a_reference": refs["a_ref"],
                            "c_reference": refs["c_ref"],
                            **item["metrics"],
                            "per_unit_log_likelihood": json.dumps(item["log_probs"].tolist()),
                            "total_log_likelihood": item["total_ll"],
                            "effective_k": item["k"],
                            "aic": item["criteria"]["aic"], "bic": item["criteria"]["bic"],
                            "aic_weight": aic_weights[model_name],
                            "bic_weight": bic_weights[model_name],
                            "selected_model_aic": selected_aic,
                            "selected_model_bic": selected_bic,
                            "selected_aic": int(selected_aic == model_name),
                            "selected_bic": int(selected_bic == model_name),
                            "mispenalized_aic_weight": invalid_aic_weights[model_name],
                            "mispenalized_bic_weight": invalid_bic_weights[model_name],
                            "mispenalized_selected_model_aic": invalid_selected_aic,
                            "mispenalized_selected_model_bic": invalid_selected_bic,
                            "runtime_seconds": item["runtime"],
                        }
                        rows.append(row)

                    if example is None and n == 10 and run == 0 and configuration == "correct":
                        quad = results["quadratic"]["map"].expanded_theta
                        example = {
                            "c_true": c_true, "a_true": truth["a_true"],
                            "b_true": np.asarray(truth["b_true"]).copy(),
                            "x": x.copy(), "a_ref": refs["a_ref"], "c_ref": refs["c_ref"],
                            "b_ref": np.asarray(refs["b_ref"]).copy(),
                            "theta_map": quad.copy(),
                        }
    if example is None:
        raise AssertionError("N=10 must be present so the required reconstruction can be plotted.")
    return rows, example


def configure_plots() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 220, "axes.grid": True,
        "grid.alpha": 0.22, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9,
    })


def save_figure(fig: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_metric(
    rows: Sequence[Mapping[str, object]], metric: str, ylabel: str,
    path: Path, models: Sequence[str] = MODEL_NAMES,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    for axis, c_true in zip(axes, C_TRUE_VALUES):
        for configuration, marker in zip(CONFIGURATIONS, ("o", "s", "^", "D")):
            for model in models:
                values = [
                    mean_metric(subset(rows, c_true=c_true, n=n, configuration=configuration, model=model), metric)
                    for n in sorted({int(row["n_observations"]) for row in rows})
                ]
                axis.plot(
                    sorted({int(row["n_observations"]) for row in rows}), values,
                    marker=marker, linestyle="-" if model == "quadratic" else "--",
                    label=f"{CONFIG_LABELS[configuration]}, {model}",
                )
        axis.set(title=f"True c={c_true:g}", xlabel="N", xscale="log")
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=6, ncol=2)
    save_figure(fig, path)


def plot_weights(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True, sharey=True)
    n_values = sorted({int(row["n_observations"]) for row in rows})
    for row_axes, c_true in zip(axes, C_TRUE_VALUES):
        for axis, criterion in zip(row_axes, ("aic", "bic")):
            for configuration, marker in zip(CONFIGURATIONS, ("o", "s", "^", "D")):
                quadratic = [
                    mean_metric(subset(rows, c_true=c_true, n=n, configuration=configuration, model="quadratic"), f"{criterion}_weight")
                    for n in n_values
                ]
                axis.plot(n_values, quadratic, marker=marker, label=CONFIG_LABELS[configuration])
            axis.axhline(0.5, color="0.4", linestyle=":")
            axis.set(title=f"{criterion.upper()}, true c={c_true:g}", xscale="log", ylim=(-0.03, 1.03))
    axes[1, 0].set_xlabel("N")
    axes[1, 1].set_xlabel("N")
    axes[0, 0].set_ylabel("Quadratic weight\n(linear weight = 1 - value)")
    axes[1, 0].set_ylabel("Quadratic weight\n(linear weight = 1 - value)")
    axes[0, 1].legend(fontsize=7)
    save_figure(fig, path)


def confusion_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for c_true in C_TRUE_VALUES:
        expected = "linear" if c_true == 0.0 else "quadratic"
        for configuration in CONFIGURATIONS:
            for criterion in ("aic", "bic"):
                decisions = [
                    row for row in rows
                    if row["model"] == "linear" and row["configuration"] == configuration
                    and math.isclose(float(row["c_true"]), c_true)
                ]
                for n in sorted({int(row["n_observations"]) for row in decisions}):
                    group = [row for row in decisions if int(row["n_observations"]) == n]
                    linear_count = sum(row[f"selected_model_{criterion}"] == "linear" for row in group)
                    quadratic_count = len(group) - linear_count
                    output.append({
                        "c_true": c_true, "expected_model": expected,
                        "configuration": configuration, "criterion": criterion,
                        "n_observations": n, "linear_selected": linear_count,
                        "quadratic_selected": quadratic_count,
                        "accuracy": sum(row[f"selected_model_{criterion}"] == expected for row in group) / len(group),
                    })
    return output


def plot_confusion(table: Sequence[Mapping[str, object]], path: Path) -> None:
    labels, values = [], []
    max_n = max(int(row["n_observations"]) for row in table)
    for c_true in C_TRUE_VALUES:
        for configuration in CONFIGURATIONS:
            group = [
                row for row in table if float(row["c_true"]) == c_true
                and row["configuration"] == configuration and row["criterion"] == "bic"
                and int(row["n_observations"]) == max_n
            ]
            labels.append(f"c={c_true:g} / {CONFIG_LABELS[configuration]}")
            values.append(float(group[0]["accuracy"]))
    matrix = np.asarray(values).reshape(2, len(CONFIGURATIONS))
    fig, axis = plt.subplots(figsize=(8.2, 3.3))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    axis.set(
        xticks=range(len(CONFIGURATIONS)), xticklabels=[CONFIG_LABELS[c] for c in CONFIGURATIONS],
        yticks=(0, 1), yticklabels=("True linear (c=0)", "True quadratic (c=0.6)"),
        title=f"BIC selection accuracy at N={max_n}",
    )
    fig.colorbar(image, ax=axis, label="Accuracy")
    save_figure(fig, path)


def plot_reconstruction(example: Mapping[str, object], path: Path) -> None:
    x = np.asarray(example["x"])
    theta = np.asarray(example["theta_map"])
    reconstructed = theta[:, 2, None] + theta[:, 0, None] * T_VALUES + theta[:, 1, None] * T_VALUES**2
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for index in range(len(x)):
        axes[0].plot(T_VALUES, x[index], "o", alpha=0.55)
        axes[0].plot(T_VALUES, reconstructed[index], "-", alpha=0.55)
    axes[0].set(xlabel="t", ylabel="x", title="N=10 observations and quadratic MAP fits")
    axes[1].plot(np.asarray(example["b_true"]), "x", label="true b")
    axes[1].plot(np.asarray(example["b_ref"]), "_", ms=10, label="analytic b reference")
    axes[1].plot(theta[:, 2], "o", fillstyle="none", label="COMPASS b MAP")
    axes[1].set(xlabel="Unit", ylabel="b", title="Local-intercept reconstruction")
    axes[1].legend()
    save_figure(fig, path)


def validate(rows: Sequence[Mapping[str, object]], cfg: Config) -> dict[str, bool | float]:
    n_min, n_max = min(cfg.n_values), max(cfg.n_values)
    correct = subset(rows, configuration="correct")
    all_global = subset(rows, configuration="all_global")
    counts_exact = all(
        int(row["effective_k"]) == expected_k(
            str(row["model"]), str(row["configuration"]), int(row["n_observations"])
        )
        for row in rows
    )
    correct_global_shared = all(
        float(row["a_sharing_range"]) <= GLOBAL_ATOL
        and (row["model"] == "linear" or float(row["c_sharing_range"]) <= GLOBAL_ATOL)
        for row in correct
    )
    all_global_collapsed = all(float(row["b_inferred_std"]) <= GLOBAL_ATOL for row in all_global)
    global_min = mean_metric(subset(correct, n=n_min, model="quadratic"), "global_parameter_rmse")
    global_max = mean_metric(subset(correct, n=n_max, model="quadratic"), "global_parameter_rmse")
    linear_weight_min = mean_metric(
        subset(correct, c_true=0.0, n=n_min, model="linear"), "bic_weight"
    )
    linear_weight_max = mean_metric(
        subset(correct, c_true=0.0, n=n_max, model="linear"), "bic_weight"
    )
    quadratic_weight_min = mean_metric(
        subset(correct, c_true=0.6, n=n_min, model="quadratic"), "bic_weight"
    )
    quadratic_weight_max = mean_metric(
        subset(correct, c_true=0.6, n=n_max, model="quadratic"), "bic_weight"
    )
    report: dict[str, bool | float] = {
        "correct_globals_shared": correct_global_shared,
        "all_global_intercepts_collapse": all_global_collapsed,
        "effective_parameter_counts_exact": counts_exact,
        "global_map_accuracy_improves": global_max < global_min,
        "linear_selection_improves_for_c_zero": linear_weight_max > linear_weight_min,
        "quadratic_selection_improves_for_c_nonzero": quadratic_weight_max > quadratic_weight_min,
        "global_rmse_n_min": global_min, "global_rmse_n_max": global_max,
        "linear_bic_weight_n_min": linear_weight_min,
        "linear_bic_weight_n_max": linear_weight_max,
        "quadratic_bic_weight_n_min": quadratic_weight_min,
        "quadratic_bic_weight_n_max": quadratic_weight_max,
    }
    required = (
        "correct_globals_shared", "all_global_intercepts_collapse",
        "effective_parameter_counts_exact", "global_map_accuracy_improves",
        "linear_selection_improves_for_c_zero",
        "quadratic_selection_improves_for_c_nonzero",
    )
    failed = [name for name in required if not report[name]]
    if failed:
        raise AssertionError(f"Hierarchical experiment assertions failed: {failed}; report={report}")
    return report


def save_config(cfg: Config) -> None:
    payload = asdict(cfg)
    payload["output_dir"] = str(cfg.output_dir)
    payload["n_values"] = list(cfg.n_values)
    payload.update({
        "t_values": T_VALUES.tolist(), "c_true_values": list(C_TRUE_VALUES),
        "sigma_x": SIGMA_X, "parameter_prior_std": PARAMETER_PRIOR_STD,
        "model_kwargs": MODEL_KWARGS, "global_tolerance": GLOBAL_ATOL,
        "effective_parameter_definition": "k = d_G + N*d_L",
        "likelihood": "COMPASS PF-ODE exact-divergence likelihood in physical x units",
        "map": "COMPASS hierarchical sampling plus score MAP refinement of local coordinates",
        "parameter_roles": {
            configuration: {
                model: {
                    "global": list(parameter_roles(model, configuration)[0]),
                    "local": list(parameter_roles(model, configuration)[1]),
                }
                for model in MODEL_NAMES
            }
            for configuration in CONFIGURATIONS
        },
    })
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "experiment_config.json").write_text(json.dumps(payload, indent=2) + "\n")


def create_outputs(
    rows: Sequence[Mapping[str, object]], example: Mapping[str, object], cfg: Config,
) -> None:
    write_csv(cfg.output_dir / "hierarchical_linear_quadratic_detailed.csv", rows)
    write_csv(cfg.output_dir / "hierarchical_linear_quadratic_summary.csv", grouped_summary(rows))
    table = confusion_rows(rows)
    write_csv(cfg.output_dir / "model_selection_confusion.csv", table)
    plot_dir = cfg.output_dir / "plots"
    plot_metric(rows, "global_parameter_rmse", "Global parameter RMSE", plot_dir / "global_parameter_rmse_vs_n.png")
    plot_metric(rows, "b_rmse", "Local intercept RMSE", plot_dir / "local_intercept_rmse_vs_n.png")
    plot_metric(rows, "a_sharing_range", "Range of inferred a", plot_dir / "sharing_error_vs_n.png")
    plot_weights(rows, plot_dir / "model_weights_vs_n.png")
    plot_confusion(table, plot_dir / "model_selection_confusion.png")
    plot_reconstruction(example, plot_dir / "example_reconstruction_n10.png")
    np.savez_compressed(cfg.output_dir / "example_reconstruction_n10.npz", **example)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--posterior-samples", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_config(args: argparse.Namespace, device: str) -> Config:
    if args.n_runs is not None and args.n_runs < 1:
        raise ValueError("--n-runs must be positive.")
    if args.posterior_samples is not None and args.posterior_samples < 2:
        raise ValueError("--posterior-samples must be at least 2.")
    if args.timesteps is not None and args.timesteps < 2:
        raise ValueError("--timesteps must be at least 2.")
    return Config(
        output_dir=args.output_dir.resolve(), device=device, seed=SEED,
        quick=args.quick, train_n=10_000 if args.quick else TRAIN_N,
        val_n=1_000 if args.quick else VAL_N,
        n_values=(2, 5, 10) if args.quick else N_OBSERVATIONS_LIST,
        n_runs=args.n_runs if args.n_runs is not None else (3 if args.quick else N_MOCK_RUNS),
        posterior_samples=args.posterior_samples if args.posterior_samples is not None else (96 if args.quick else 1_000),
        timesteps=args.timesteps if args.timesteps is not None else (14 if args.quick else 50),
        max_epochs=50 if args.quick else TRAIN_MAX_EPOCHS,
        patience=10 if args.quick else TRAIN_PATIENCE,
        force_train=args.force_train,
    )


def main() -> None:
    if CPU_LIMIT is None:
        raise RuntimeError("CPU affinity was not configured before scientific imports.")
    logical, selected = CPU_LIMIT
    active = tuple(sorted(os.sched_getaffinity(0)))
    if active != selected or len(active) > CPU_THREAD_LIMIT:
        raise RuntimeError("The active CPU affinity violates the repository's hard 3-thread cap.")
    print(
        f"CPU limited to {len(selected)} of {logical} logical CPUs "
        f"({len(selected) / logical:.2%})."
    )
    args = parse_args()
    device = select_device(args.device)
    cfg = build_config(args, device)
    seed_all(cfg.seed)
    configure_plots()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg)
    models = {model: load_or_train(cfg, model) for model in MODEL_NAMES}
    rows, example = run_experiment(cfg, models)
    create_outputs(rows, example, cfg)
    report = validate(rows, cfg)
    (cfg.output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    print(f"Outputs written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
