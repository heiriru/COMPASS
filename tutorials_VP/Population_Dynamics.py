#!/usr/bin/env python
# coding: utf-8
"""Compact population-dynamics COMPASS pipeline: create data, train, evaluate."""
from __future__ import annotations

from autocvd import autocvd
import csv
import json
import sys
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from compass import ModelTransfuser as MTf
from compass import ScoreBasedInferenceModel as SBIm

OUTPUT_DIR = SCRIPT_DIR / "output" / "population_dynamics_new"
MODEL_DIR = SCRIPT_DIR / "data" / "Population_dynamics_new"
POPULATION_DATA_PATH = MODEL_DIR / "population_dynamics_training_and_mock_data.pt"
CHECKPOINT_METADATA_PATH = MODEL_DIR / "checkpoint_metadata.json"


NORMALIZATION_METHOD = "log1p"  # choose from: "log1p", "asinh", "classical", "minmax" with which one to do normalization: 
"""
# log1p: z = log(1+x) -> so mu = mean(log(1+x_all))
# x_norm = (log(1+x) - mu) / std
# inverse: x = exp(z * std + mu) - 1

# asinh: z = asinh(x / ASINH_SCALE) -> so mu = mean(asinh(x_all / ASINH_SCALE))
# x_norm = (asinh(x / ASINH_SCALE) - mu) / std
# inverse: x = sinh(z * std + mu) * ASINH_SCALE

# classical: z = x -> so mu = mean(x_all)
# x_norm = (x - mu) / std
# inverse: x = x_norm * std + mu


"""
NORMALIZATION_METHODS_FOR_DIAGNOSTICS = ["classical", "log1p", "asinh", "minmax"] 
CONFUSION_N = 50
CONFUSION_N_OBSERVATIONS = 8
CONFUSION_TIMESTEPS = 60
CONFUSION_NUM_SAMPLES = 1000
TRAIN_N = 50_000
VAL_N = 2_000
TRAIN_MAX_EPOCHS = 500
TRAIN_BATCH_SIZE = 256
TRAIN_LR = 1e-3
TRAIN_EARLY_STOPPING_PATIENCE = 20
TRAIN_VERBOSE = True
DATA_SCHEMA_VERSION = 3

# Simulator/model constants.
TIMESTEPS_PER_SERIES = 20
CONFUSION_TIMESTEPS_PER_SERIES = TIMESTEPS_PER_SERIES
SIMULATION_T_MAX = 20.0
DT = 0.01
ASINH_SCALE = 100.0
NORMALIZATION_EPS = 1e-6
MAX_POPULATION = 1e6
DATA_BATCH_SIZE = 5_000
MAX_DATA_ATTEMPTS = 200
MOCK_INITIAL_STATE = torch.tensor([[30.0, 1.0]])
MOCK_PARAMS_LOG = torch.tensor([[-0.1, -3.0, -0.1, -3.0]])
LOG_PARAM_MEANS = torch.tensor([-0.125, -3.0, -0.125, -3.0])
LOG_PARAM_STDS = torch.tensor([0.5, 0.5, 0.5, 0.5])
MODEL_NAMES = ["Logistic Prey", "Satiated Predator", "Rosenzweig-MacArthur", "Lotka-Volterra"]
EXPECTED_NODES_SIZE = 4 + 2 * TIMESTEPS_PER_SERIES

MODEL_KWARGS = dict(
    sde_type="vesde",
    sigma=3,
    hidden_size=32,
    depth=4,
    num_heads=4,
    mlp_ratio=4,
)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def savefig(fig, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved data: {path}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved data: {path}")


# simulator models
def lotka_volterra(state: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    prey, pred = state.T
    alpha, beta, gamma, delta = p.T
    return torch.stack([alpha * prey - beta * prey * pred, delta * prey * pred - gamma * pred]).T


def logistic_prey(state: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    prey, pred = state.T
    alpha, beta, gamma, delta = p.T
    capacity = delta * 1000
    return torch.stack([
        alpha * prey * (1 - prey / capacity) - beta * prey * pred,
        0.5 * beta * prey * pred - gamma * pred,
    ]).T


def satiated_predator(state: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    prey, pred = state.T
    alpha, beta, gamma, delta = p.T
    consumption = beta * prey / (1 + beta * delta * prey)
    return torch.stack([alpha * prey - consumption * pred, 0.5 * consumption * pred - gamma * pred]).T


def rosenzweig_macarthur(state: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    prey, pred = state.T
    alpha, beta, gamma, delta = p.T
    capacity = delta * 1000
    consumption = beta * prey / (1 + beta * 0.1 * prey)
    return torch.stack([
        alpha * prey * (1 - prey / capacity) - consumption * pred,
        0.5 * consumption * pred - gamma * pred,
    ]).T


MODELS = {
    "Logistic Prey": logistic_prey,
    "Satiated Predator": satiated_predator,
    "Rosenzweig-MacArthur": rosenzweig_macarthur,
    "Lotka-Volterra": lotka_volterra,
}


def sample_theta(n: int) -> torch.Tensor:
    return LOG_PARAM_MEANS + LOG_PARAM_STDS * torch.randn(n, 4)


def simulate(model_name: str, theta_log: torch.Tensor, n_time: int = TIMESTEPS_PER_SERIES) -> torch.Tensor:
    params = torch.exp(theta_log)
    state = MOCK_INITIAL_STATE.to(device=theta_log.device, dtype=theta_log.dtype).repeat(theta_log.shape[0], 1)
    sample_steps = torch.round(torch.linspace(0, SIMULATION_T_MAX, n_time) / DT).long()
    history = torch.zeros(theta_log.shape[0], n_time, 2, device=theta_log.device, dtype=theta_log.dtype)
    history[:, 0] = state
    out = 1
    zero = torch.zeros(2, device=theta_log.device, dtype=theta_log.dtype)
    for step in range(1, int(round(SIMULATION_T_MAX / DT)) + 1):
        state = torch.maximum(state + MODELS[model_name](state, params) * DT, zero)
        while out < n_time and step >= sample_steps[out].item():
            history[:, out] = state
            out += 1
    return history.flatten(1)


def stable_dataset(model_name: str, n: int, label: str, n_time: int = TIMESTEPS_PER_SERIES) -> tuple[torch.Tensor, torch.Tensor]:
    theta_parts, x_parts, accepted, proposed = [], [], 0, 0
    for _ in range(MAX_DATA_ATTEMPTS):
        draw_n = max(n - accepted, DATA_BATCH_SIZE)
        theta = sample_theta(draw_n)
        x = simulate(model_name, theta, n_time=n_time)
        mask = torch.isfinite(x).all(1) & (x >= 0).all(1) & (x <= MAX_POPULATION).all(1)
        proposed += draw_n
        if mask.any():
            theta_parts.append(theta[mask])
            x_parts.append(x[mask])
            accepted += int(mask.sum())
        print(f"{label}: accepted {min(accepted, n)}/{n} after proposing {proposed}")
        if accepted >= n:
            theta_all, x_all = torch.cat(theta_parts)[:n], torch.cat(x_parts)[:n]
            print(f"{label}: theta={tuple(theta_all.shape)}, x={tuple(x_all.shape)}, x_max={x_all.max().item():.4g}")
            return theta_all, x_all
    raise RuntimeError(f"Could only generate {accepted}/{n} stable trajectories for {label}.")


# normalization methods
def transform(x: torch.Tensor, method: str = NORMALIZATION_METHOD) -> torch.Tensor:
    if method == "log1p":
        return torch.log1p(x)
    if method == "asinh":
        return torch.asinh(x / ASINH_SCALE)
    if method == "classical":
        return x
    if method == "minmax":
        return x
    raise ValueError(f"Unknown normalization method: {method}")


def inverse_transform(z: torch.Tensor, method: str = NORMALIZATION_METHOD) -> torch.Tensor:
    if method == "log1p":
        return torch.expm1(z)
    if method == "asinh":
        return torch.sinh(z) * ASINH_SCALE
    if method == "classical":
        return z
    if method == "minmax":
        return z
    raise ValueError(f"Unknown normalization method: {method}")


def norm_stats(xs: list[torch.Tensor], method: str = NORMALIZATION_METHOD) -> tuple[torch.Tensor, torch.Tensor]:
    z = transform(torch.cat(xs), method)
    if method == "minmax":
        z_min = z.min().reshape(1, 1)
        z_range = (z.max() - z.min()).reshape(1, 1) + NORMALIZATION_EPS
        return z_min, z_range
    return z.mean().reshape(1, 1), z.std().reshape(1, 1) + NORMALIZATION_EPS


def distribution_moments(values: torch.Tensor) -> tuple[float, float]:
    values = values.flatten().float()
    centered = values - values.mean()
    scaled = centered / (values.std() + NORMALIZATION_EPS)
    skew = torch.mean(scaled ** 3).item()
    excess_kurtosis = (torch.mean(scaled ** 4) - 3).item()
    return skew, excess_kurtosis


def normalization_stats_like(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return mu.to(device=x.device, dtype=x.dtype), std.to(device=x.device, dtype=x.dtype)


def normalize_x(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, method: str = NORMALIZATION_METHOD) -> torch.Tensor:
    mu, std = normalization_stats_like(x, mu, std)
    if method == "minmax":
        return 2 * (transform(x, method) - mu) / std - 1
    return (transform(x, method) - mu) / std


def unnormalize_x(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, method: str = NORMALIZATION_METHOD) -> torch.Tensor:
    mu, std = normalization_stats_like(x, mu, std)
    if method == "minmax":
        return inverse_transform((x + 1) * std / 2 + mu, method)
    return inverse_transform(x * std + mu, method)


def plot_normalization_diagnostics(raw_train: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
    xs = [x for _, x in raw_train.values()]
    time = torch.linspace(0, SIMULATION_T_MAX, TIMESTEPS_PER_SERIES)
    colors = dict(zip(MODEL_NAMES, sns.color_palette("deep", len(MODEL_NAMES))))
    views = [("raw population", None, False)]
    for method in NORMALIZATION_METHODS_FOR_DIAGNOSTICS:
        mu, std = norm_stats(xs, method)
        print(f"{method}: mu={mu.item():.6g}, std={std.item():.6g}")
        if method != "classical":
            views.append((f"{method} transform", method, False))
        views.append((f"{method} normalized", method, True))

    fig, axes = plt.subplots(len(views), 2, figsize=(13, 2.4 * len(views)), sharex=True, dpi=300)
    for row, (label, method, do_normalize) in enumerate(views):
        for model_name in MODEL_NAMES:
            x = raw_train[model_name][1]
            if method is None:
                y = x
            elif do_normalize:
                mu, std = norm_stats(xs, method)
                y = normalize_x(x, mu, std, method)
            else:
                y = transform(x, method)
            y = y.reshape(-1, TIMESTEPS_PER_SERIES, 2)
            for species, ax in enumerate(axes[row]):
                values = y[:, :, species]
                lo, med, hi = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9]), dim=0)
                color = colors[model_name]
                ax.plot(time, med, color=color, lw=1.8, label=model_name if row == 0 and species == 0 else None)
                ax.fill_between(time, lo, hi, color=color, alpha=0.13, linewidth=0)
        axes[row, 0].set_ylabel(label)
        for ax in axes[row]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.15)
    axes[0, 0].set_title("Prey: median and 10-90% band")
    axes[0, 1].set_title("Predator: median and 10-90% band")
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    fig.legend(loc="upper center", ncol=len(MODEL_NAMES), frameon=False)
    fig.suptitle(f"Population trajectory normalization diagnostics; selected: {NORMALIZATION_METHOD}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig(fig, "population_dynamics_normalization_diagnostics.png")

    x_all = torch.cat(xs)
    x_plot = x_all[torch.randperm(len(x_all))[: min(len(x_all), 20_000)]]
    fig, axes = plt.subplots(len(NORMALIZATION_METHODS_FOR_DIAGNOSTICS), 2, figsize=(11, 12), dpi=300)
    for row, method in enumerate(NORMALIZATION_METHODS_FOR_DIAGNOSTICS):
        mu, std = norm_stats(xs, method)
        skew, excess_kurtosis = distribution_moments(transform(x_all, method))
        for col, (values, title, xlabel) in enumerate([
            (transform(x_plot, method), f"{method}: transformed distribution", "transformed value"),
            (normalize_x(x_plot, mu, std, method), f"{method}: normalized distribution", "normalized value"),
        ]):
            values = values.flatten().detach().cpu()
            axes[row, col].hist(values.numpy(), bins=80, color="#4c78a8")
            axes[row, col].set_yscale("log")
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel(xlabel)
            axes[row, col].set_ylabel("log count")
            axes[row, col].text(
                0.98,
                0.94,
                f"skew={skew:.2g}\nexcess kurtosis={excess_kurtosis:.2g}",
                transform=axes[row, col].transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
            )
            axes[row, col].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Population value distributions after transforms")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig(fig, "population_dynamics_normalization_histograms.png")

# data creation
def config_payload(mu: torch.Tensor, std: torch.Tensor, data_generation_id: str) -> dict:
    return {
        "model_names": MODEL_NAMES,
        "train_n": TRAIN_N,
        "val_n": VAL_N,
        "timesteps_per_series": TIMESTEPS_PER_SERIES,
        "simulation_t_max": SIMULATION_T_MAX,
        "normalization_method": NORMALIZATION_METHOD,
        "normalization_scope": "one scalar over all models, trajectories, species, and time points",
        "x_mu": mu.tolist(),
        "x_std": std.tolist(),
        "asinh_scale": ASINH_SCALE,
        "max_population": MAX_POPULATION,
        "confusion_n": CONFUSION_N,
        "confusion_n_observations": CONFUSION_N_OBSERVATIONS,
        "confusion_timesteps_per_series": CONFUSION_TIMESTEPS_PER_SERIES,
        "confusion_timesteps": CONFUSION_TIMESTEPS,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "data_generation_id": data_generation_id,
        "confusion_num_samples": CONFUSION_NUM_SAMPLES,
        "model_kwargs": MODEL_KWARGS,
        "model_dir": str(MODEL_DIR),
    }


def expected_confusion_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "theta": (CONFUSION_N, CONFUSION_N_OBSERVATIONS, 4),
        "raw": (CONFUSION_N, CONFUSION_N_OBSERVATIONS, 2 * CONFUSION_TIMESTEPS_PER_SERIES),
        "normalized": (CONFUSION_N, CONFUSION_N_OBSERVATIONS, 2 * CONFUSION_TIMESTEPS_PER_SERIES),
    }


def confusion_data_matches(data: dict) -> bool:
    try:
        return all(
            tuple(data["confusion_mocks"][model_name][key].shape) == expected_shape
            for model_name in MODEL_NAMES
            for key, expected_shape in expected_confusion_shapes().items()
        )
    except (KeyError, TypeError):
        return False


def create_confusion_mocks(mu: torch.Tensor, std: torch.Tensor) -> dict:
    confusion = {}
    for model_name in MODEL_NAMES:
        total_observations = CONFUSION_N * CONFUSION_N_OBSERVATIONS
        theta, raw = stable_dataset(
            model_name,
            total_observations,
            f"confusion {model_name}",
            n_time=CONFUSION_TIMESTEPS_PER_SERIES,
        )
        theta_runs = theta.reshape(CONFUSION_N, CONFUSION_N_OBSERVATIONS, 4)
        raw_runs = raw.reshape(
            CONFUSION_N,
            CONFUSION_N_OBSERVATIONS,
            2 * CONFUSION_TIMESTEPS_PER_SERIES,
        )
        confusion[model_name] = {
            "theta": theta_runs,
            "raw": raw_runs,
            "normalized": torch.stack([normalize_x(x, mu, std) for x in raw_runs]),
        }
    return confusion


def write_data_configs(data: dict) -> None:
    config = config_payload(data["x_mu"], data["x_std"], data["data_generation_id"])
    write_json(OUTPUT_DIR / "experiment_config.json", config)
    write_json(MODEL_DIR / "normalization_config.json", config)


def create_population_data(force: bool = False) -> dict:
    if POPULATION_DATA_PATH.exists() and not force:
        print(f"Loading existing data: {POPULATION_DATA_PATH}")
        data = torch.load(POPULATION_DATA_PATH, map_location="cpu")
        validate_saved_data(data, check_confusion=False)
        if not confusion_data_matches(data):
            print(
                "Refreshing confusion mocks only: "
                f"{CONFUSION_N} run(s) x {CONFUSION_N_OBSERVATIONS} observation(s)."
            )
            data["confusion_mocks"] = create_confusion_mocks(data["x_mu"], data["x_std"])
            torch.save(data, POPULATION_DATA_PATH)
            write_data_configs(data)
            print(f"Updated confusion mocks without changing training data: {POPULATION_DATA_PATH}")
        return data

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    POPULATION_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_train = {m: stable_dataset(m, TRAIN_N, f"raw train {m}") for m in MODEL_NAMES}
    raw_val = {m: stable_dataset(m, VAL_N, f"raw validation {m}") for m in MODEL_NAMES}
    plot_normalization_diagnostics(raw_train)

    mu, std = norm_stats([x for _, x in raw_train.values()])
    print(f"Selected normalization={NORMALIZATION_METHOD}; mu={mu.item():.6g}; std={std.item():.6g}")
    train = {m: (t, normalize_x(x, mu, std)) for m, (t, x) in raw_train.items()}
    val = {m: (t, normalize_x(x, mu, std)) for m, (t, x) in raw_val.items()}

    data_generation_id = str(uuid.uuid4())
    notebook_raw = simulate("Lotka-Volterra", MOCK_PARAMS_LOG)
    confusion = create_confusion_mocks(mu, std)

    payload = {
        "training_data": train,
        "validation_data": val,
        "raw_training_data": raw_train,
        "raw_validation_data": raw_val,
        "notebook_mock": {"raw": notebook_raw, "normalized": normalize_x(notebook_raw, mu, std)},
        "confusion_mocks": confusion,
        "x_mu": mu,
        "x_std": std,
        "normalization_method": NORMALIZATION_METHOD,
        "model_names": MODEL_NAMES,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "data_generation_id": data_generation_id,
    }
    torch.save(payload, POPULATION_DATA_PATH)
    print(f"Saved data: {POPULATION_DATA_PATH}")
    write_data_configs(payload)
    return payload


def validate_saved_data(data: dict, check_confusion: bool = True) -> None:
    if data.get("normalization_method") != NORMALIZATION_METHOD:
        raise RuntimeError("Saved data normalization does not match NORMALIZATION_METHOD. Regenerate with --force.")
    if (
        data.get("data_schema_version") != DATA_SCHEMA_VERSION
        or not data.get("data_generation_id")
        or tuple(data["x_mu"].shape) != (1, 1)
    ):
        raise RuntimeError("Saved data was created by an older normalization schema. Regenerate with --force.")
    if check_confusion and not confusion_data_matches(data):
        raise RuntimeError(
            "Saved confusion mocks do not match the current CONFUSION_N or "
            "CONFUSION_N_OBSERVATIONS. Run `python Population_Dynamics_create_data.py` "
            "without --force to refresh only confusion mocks; model retraining is not required."
        )


def load_population_data() -> dict:
    if not POPULATION_DATA_PATH.exists():
        raise FileNotFoundError(f"Run Population_Dynamics_create_data.py first. Missing: {POPULATION_DATA_PATH}")
    data = torch.load(POPULATION_DATA_PATH, map_location="cpu")
    validate_saved_data(data)
    return data


# training models
def checkpoint_metadata(data: dict) -> dict:
    return {
        "data_generation_id": data["data_generation_id"],
        "normalization_method": data["normalization_method"],
        "x_mu": data["x_mu"].tolist(),
        "x_std": data["x_std"].tolist(),
        "model_kwargs": MODEL_KWARGS,
        "nodes_size": EXPECTED_NODES_SIZE,
    }


def checkpoints_match_data(data: dict) -> bool:
    if not CHECKPOINT_METADATA_PATH.exists():
        return False
    with CHECKPOINT_METADATA_PATH.open() as handle:
        return json.load(handle) == checkpoint_metadata(data)


def checkpoint_path(model_name: str) -> Path:
    return MODEL_DIR / f"{model_name}.pt"


def promote_checkpoint(model_name: str) -> Path:
    final, tmp = checkpoint_path(model_name), MODEL_DIR / f"{model_name}_checkpoint.pt"
    if tmp.exists():
        tmp.replace(final)
        return final
    if final.exists():
        return final
    raise FileNotFoundError(f"Missing checkpoint for {model_name}: {final}")


def train_models_from_saved_data(dev: str | None = None) -> MTf:
    data = load_population_data()
    dev = dev or device()
    missing = [m for m in MODEL_NAMES if not checkpoint_path(m).exists()]
    models_to_train = MODEL_NAMES if not checkpoints_match_data(data) else missing
    if models_to_train:
        mtf = MTf(path=str(MODEL_DIR))
        for m in MODEL_NAMES:
            mtf.add_data(m, *data["training_data"][m], *data["validation_data"][m])
        mtf.init_models(**MODEL_KWARGS)
        for m in models_to_train:
            print(f"Training {m}")
            theta, x = data["training_data"][m]
            val_theta, val_x = data["validation_data"][m]
            mtf.models_dict[m].train(
                theta=theta,
                x=x,
                theta_val=val_theta,
                x_val=val_x,
                batch_size=TRAIN_BATCH_SIZE,
                max_epochs=TRAIN_MAX_EPOCHS,
                lr=TRAIN_LR,
                device=dev,
                verbose=TRAIN_VERBOSE,
                path=str(MODEL_DIR),
                name=m,
                early_stopping_patience=TRAIN_EARLY_STOPPING_PATIENCE,
            )
            promote_checkpoint(m)
            torch.cuda.empty_cache()
        write_json(CHECKPOINT_METADATA_PATH, checkpoint_metadata(data))
    else:
        print(f"All checkpoints match the saved data in {MODEL_DIR}")
    return load_transfuser(dev)


def load_transfuser(device: str) -> MTf:
    data = load_population_data()
    if not checkpoints_match_data(data):
        raise RuntimeError(
            "Checkpoints do not match the current generated data and normalization. Run "
            "`python Population_Dynamics_train_models.py` before evaluation."
        )
    mtf = MTf(path=str(MODEL_DIR))
    for m in MODEL_NAMES:
        model = SBIm.load(str(promote_checkpoint(m)), device=device)
        if model.nodes_size != EXPECTED_NODES_SIZE:
            raise RuntimeError(f"{m} checkpoint has nodes_size={model.nodes_size}; expected {EXPECTED_NODES_SIZE}")
        mtf.add_model(m, model)
    return mtf


# evaluations
def stat_tensor(value) -> torch.Tensor:
    return torch.as_tensor(value).detach().cpu().float()


def stat_scalar(value) -> torch.Tensor:
    return torch.as_tensor(value).detach().cpu().reshape(())


def print_probs(mtf: MTf, title: str) -> None:
    print(f"\n{title}")
    for m in MODEL_NAMES:
        prob = stat_scalar(mtf.stats[m]["model_prob"]).item()
        print(f"  {m:<22}: {100 * prob:8.2f}%")


def plot_map_trajectories(mtf: MTf, test_raw: torch.Tensor) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5), constrained_layout=True, dpi=300)
    t_obs = torch.linspace(0, SIMULATION_T_MAX, TIMESTEPS_PER_SERIES)
    colors = sns.color_palette("deep", len(MODEL_NAMES))
    for color, m in zip(colors, MODEL_NAMES):
        theta = stat_tensor(mtf.stats[m]["MAP"][0, 0]).reshape(1, -1)
        full_x = simulate_full(m, theta)
        full_t = torch.linspace(0, SIMULATION_T_MAX, full_x.shape[1])
        axes[0].plot(full_t, full_x[0, :, 0], color=color, label=m)
        axes[1].plot(full_t, full_x[0, :, 1], color=color, label=m)
    axes[0].scatter(t_obs, test_raw[0, ::2], color="k", s=10, zorder=10)
    axes[1].scatter(t_obs, test_raw[0, 1::2], color="k", s=10, zorder=10)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=len(MODEL_NAMES), frameon=False)
    axes[0].set_ylabel("Prey")
    axes[1].set_ylabel("Predator")
    axes[1].set_xlabel("Time")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "population_dynamics_map_trajectories.png")


def simulate_full(model_name: str, theta_log: torch.Tensor) -> torch.Tensor:
    params = torch.exp(theta_log)
    state = MOCK_INITIAL_STATE.to(device=theta_log.device, dtype=theta_log.dtype).repeat(theta_log.shape[0], 1)
    hist = torch.zeros(theta_log.shape[0], int(round(SIMULATION_T_MAX / DT)) + 1, 2, device=theta_log.device, dtype=theta_log.dtype)
    hist[:, 0] = state
    zero = torch.zeros(2, device=theta_log.device, dtype=theta_log.dtype)
    for step in range(1, hist.shape[1]):
        state = torch.maximum(state + MODELS[model_name](state, params) * DT, zero)
        hist[:, step] = state
    return hist


def plot_posterior_predictive(mtf: MTf, test_raw: torch.Tensor, data: dict, device: str) -> None:
    mu, std = data["x_mu"], data["x_std"]
    t = torch.linspace(0, SIMULATION_T_MAX, TIMESTEPS_PER_SERIES)
    obs = test_raw.reshape(1, TIMESTEPS_PER_SERIES, 2)[0]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5.5), constrained_layout=True, dpi=300)
    colors = sns.color_palette("deep", len(MODEL_NAMES))
    for color, m in zip(colors, MODEL_NAMES):
        theta = stat_tensor(mtf.stats[m]["MAP"][0, 0])
        err = stat_tensor(mtf.stats[m]["MAP"][0, 1])
        raw_samples = unnormalize_x(
            mtf.models_dict[m].sample(theta=theta, err=err, device=device, timesteps=CONFUSION_TIMESTEPS, method="dpm", order=2)[0],
            mu,
            std,
        )
        trajectories = raw_samples.detach().cpu().reshape(-1, TIMESTEPS_PER_SERIES, 2)
        for species, ax in enumerate(axes):
            values = trajectories[:, :, species]
            q05, q25, q50, q75, q95 = torch.quantile(values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]), dim=0)
            ax.fill_between(t, q05, q95, color=color, alpha=0.08, linewidth=0)
            ax.fill_between(t, q25, q75, color=color, alpha=0.18, linewidth=0)
            ax.plot(t, q50, color=color, lw=1.8, label=m if species == 0 else None)
    axes[0].scatter(t, obs[:, 0], color="k", s=18, zorder=10, label="Observed")
    axes[1].scatter(t, obs[:, 1], color="k", s=18, zorder=10)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, frameon=False)
    axes[0].set_ylabel("Prey")
    axes[1].set_ylabel("Predator")
    axes[1].set_xlabel("Time")
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "population_dynamics_posterior_predictive_samples.png")


def validation_features(raw_val: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    features = {}
    eps = torch.tensor(1e-6)
    for m in MODEL_NAMES:
        x = raw_val[m][1].reshape(-1, TIMESTEPS_PER_SERIES, 2)
        prey, pred = x[:, :, 0], x[:, :, 1]
        features[m] = {
            "Prey final / initial": prey[:, -1] / (prey[:, 0] + eps),
            "Predator final / initial": pred[:, -1] / (pred[:, 0] + eps),
            "Peak prey": prey.max(dim=1).values,
            "Peak predator": pred.max(dim=1).values,
        }
    return features


def plot_validation_diagnostics(raw_val: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5), constrained_layout=True, dpi=300)
    t_obs = torch.linspace(0, SIMULATION_T_MAX, TIMESTEPS_PER_SERIES)
    colors = sns.color_palette("deep", len(MODEL_NAMES))
    for color, m in zip(colors, MODEL_NAMES):
        theta = raw_val[m][0][0].reshape(1, -1)
        observed = raw_val[m][1][0].reshape(TIMESTEPS_PER_SERIES, 2)
        full_x = simulate_full(m, theta)
        full_t = torch.linspace(0, SIMULATION_T_MAX, full_x.shape[1])
        axes[0].plot(full_t, full_x[0, :, 0], color=color, label=m)
        axes[1].plot(full_t, full_x[0, :, 1], color=color, label=m)
        axes[0].scatter(t_obs, observed[:, 0], color="k", s=10, zorder=10)
        axes[1].scatter(t_obs, observed[:, 1], color="k", s=10, zorder=10)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=len(MODEL_NAMES), frameon=False)
    axes[0].set_ylabel("Prey")
    axes[1].set_ylabel("Predator")
    axes[1].set_xlabel("Time")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "population_dynamics_validation_trajectories.png")

    colors = dict(zip(MODEL_NAMES, colors))
    feature_sets = validation_features(raw_val)
    feature_names = list(next(iter(feature_sets.values())).keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True, dpi=300)
    positions = torch.arange(1, len(MODEL_NAMES) + 1).float()
    for ax, feature_name in zip(axes.flatten(), feature_names):
        values = [feature_sets[m][feature_name].detach().cpu() for m in MODEL_NAMES]
        violin = ax.violinplot(
            values,
            positions=positions.tolist(),
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, m in zip(violin["bodies"], MODEL_NAMES):
            body.set_facecolor(colors[m])
            body.set_edgecolor(colors[m])
            body.set_alpha(0.16)
        box = ax.boxplot(
            values,
            positions=positions.tolist(),
            widths=0.34,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
        )
        for patch, m in zip(box["boxes"], MODEL_NAMES):
            patch.set_facecolor(colors[m])
            patch.set_alpha(0.28)
            patch.set_edgecolor(colors[m])
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.2)
        for pos, vals, m in zip(positions, values, MODEL_NAMES):
            subset = vals[torch.randperm(vals.numel())[: min(vals.numel(), 120)]]
            jitter = (torch.rand(subset.numel()) - 0.5) * 0.28
            ax.scatter((pos + jitter).numpy(), subset.numpy(), s=7, color=colors[m], alpha=0.22, linewidth=0)
        ax.set_yscale("log")
        if "final / initial" in feature_name:
            ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_ylabel("Fold change (log scale)")
        else:
            ax.set_ylabel("Population (log scale)")
        ax.set_title(feature_name)
        ax.set_xticks(positions.tolist())
        ax.set_xticklabels(MODEL_NAMES, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "population_dynamics_validation_summary.png")


def plot_confusion_misprediction(
    mtf: MTf,
    raw_observations: torch.Tensor,
    true_model: str,
    predicted_model: str,
    run_index: int,
) -> None:
    observations = raw_observations.reshape(-1, TIMESTEPS_PER_SERIES, 2).detach().cpu()
    num_observations = observations.shape[0]
    fig, axes = plt.subplots(
        num_observations,
        2,
        sharex=True,
        squeeze=False,
        figsize=(12, max(3.0 * num_observations, 4.5)),
        constrained_layout=True,
        dpi=300,
    )
    t_obs = torch.linspace(0, SIMULATION_T_MAX, TIMESTEPS_PER_SERIES)
    model_colors = {true_model: "#4c78a8", predicted_model: "#e45756"}
    for observation_index, observation in enumerate(observations):
        for model_name in (true_model, predicted_model):
            maps = mtf.stats[model_name]["MAP"]
            if len(maps) != num_observations:
                raise RuntimeError(
                    f"Expected {num_observations} MAP estimates for {model_name}, got {len(maps)}."
                )
            theta = stat_tensor(maps[observation_index, 0]).reshape(1, -1)
            full_x = simulate_full(model_name, theta)
            full_t = torch.linspace(0, SIMULATION_T_MAX, full_x.shape[1])
            label = f"{model_name} MAP" if observation_index == 0 else None
            axes[observation_index, 0].plot(
                full_t, full_x[0, :, 0], color=model_colors[model_name], lw=1.8, label=label
            )
            axes[observation_index, 1].plot(
                full_t, full_x[0, :, 1], color=model_colors[model_name], lw=1.8
            )
        axes[observation_index, 0].scatter(
            t_obs, observation[:, 0], color="k", s=14, zorder=10,
            label="Observed data" if observation_index == 0 else None,
        )
        axes[observation_index, 1].scatter(t_obs, observation[:, 1], color="k", s=14, zorder=10)
        axes[observation_index, 0].set_ylabel(f"Observation {observation_index + 1}")
        for ax in axes[observation_index]:
            ax.grid(alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("Prey")
    axes[0, 1].set_title("Predator")
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False)
    fig.suptitle(f"True model: {true_model} | Selected model: {predicted_model}")
    output_dir = OUTPUT_DIR / "misprediction_trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_true = true_model.lower().replace(" ", "_").replace("-", "_")
    safe_predicted = predicted_model.lower().replace(" ", "_").replace("-", "_")
    path = output_dir / f"run_{run_index:03d}_true_{safe_true}_predicted_{safe_predicted}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved misprediction trajectory: {path}")


def plot_first_run_all_model_trajectories(
    mtf: MTf,
    raw_observations: torch.Tensor,
    true_model: str,
    run_index: int,
) -> None:
    observations = raw_observations.reshape(-1, TIMESTEPS_PER_SERIES, 2).detach().cpu()
    num_observations = observations.shape[0]
    fig, axes = plt.subplots(
        num_observations,
        2,
        sharex=True,
        squeeze=False,
        figsize=(12, max(3.0 * num_observations, 4.5)),
        constrained_layout=True,
        dpi=300,
    )
    t_obs = torch.linspace(0, SIMULATION_T_MAX, TIMESTEPS_PER_SERIES)
    colors = dict(zip(MODEL_NAMES, sns.color_palette("deep", len(MODEL_NAMES))))
    for observation_index, observation in enumerate(observations):
        for model_name in MODEL_NAMES:
            maps = mtf.stats[model_name]["MAP"]
            if len(maps) != num_observations:
                raise RuntimeError(
                    f"Expected {num_observations} MAP estimates for {model_name}, got {len(maps)}."
                )
            theta = stat_tensor(maps[observation_index, 0]).reshape(1, -1)
            full_x = simulate_full(model_name, theta)
            full_t = torch.linspace(0, SIMULATION_T_MAX, full_x.shape[1])
            label = f"{model_name} MAP" if observation_index == 0 else None
            axes[observation_index, 0].plot(
                full_t, full_x[0, :, 0], color=colors[model_name], lw=1.8, label=label
            )
            axes[observation_index, 1].plot(
                full_t, full_x[0, :, 1], color=colors[model_name], lw=1.8
            )
        axes[observation_index, 0].scatter(
            t_obs, observation[:, 0], color="k", s=14, zorder=10,
            label="Observed data" if observation_index == 0 else None,
        )
        axes[observation_index, 1].scatter(t_obs, observation[:, 1], color="k", s=14, zorder=10)
        axes[observation_index, 0].set_ylabel(f"Observation {observation_index + 1}")
        for ax in axes[observation_index]:
            ax.grid(alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("Prey")
    axes[0, 1].set_title("Predator")
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=False)
    fig.suptitle(f"True model: {true_model} | All model MAP trajectories | Run {run_index}")
    output_dir = OUTPUT_DIR / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_true = true_model.lower().replace(" ", "_").replace("-", "_")
    path = output_dir / f"run_{run_index:03d}_true_{safe_true}_all_models.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved all-model trajectory diagnostic: {path}")


def run_confusion_matrix(mtf: MTf, data: dict, device: str) -> torch.Tensor:
    rows, per_true = [], {}
    for true_m in MODEL_NAMES:
        probs_for_true = []
        for run in range(CONFUSION_N):
            x = data["confusion_mocks"][true_m]["normalized"][run]
            mtf.compare(x=x, device=device, timesteps=CONFUSION_TIMESTEPS, num_samples=CONFUSION_NUM_SAMPLES, method="dpm", order=2, verbose=True, multi_obs_inference=False)
            probs = torch.stack([stat_scalar(mtf.stats[m]["model_prob"]) for m in MODEL_NAMES])
            pred = MODEL_NAMES[int(probs.argmax())]
            if run == 0:
                plot_first_run_all_model_trajectories(
                    mtf,
                    data["confusion_mocks"][true_m]["raw"][run],
                    true_m,
                    run + 1,
                )
            if pred != true_m:
                plot_confusion_misprediction(
                    mtf,
                    data["confusion_mocks"][true_m]["raw"][run],
                    true_m,
                    pred,
                    run + 1,
                )
            probs_for_true.append(probs)
            rows.append({
                "true_model": true_m,
                "run_index": run + 1,
                "selected_model": pred,
                "correct": pred == true_m,
                **{f"probability_percent_{m}": 100 * p.item() for m, p in zip(MODEL_NAMES, probs)},
                **{f"aic_{m}": stat_scalar(mtf.stats[m]["AIC"]).item() for m in MODEL_NAMES},
            })
            print(f"true={true_m}, run={run + 1}, selected={pred}")
        per_true[true_m] = torch.stack(probs_for_true)
    matrix = torch.stack([per_true[m].mean(0) for m in MODEL_NAMES])
    fields = ["true_model", "run_index", "selected_model", "correct", *[f"probability_percent_{m}" for m in MODEL_NAMES], *[f"aic_{m}" for m in MODEL_NAMES]]
    write_csv(OUTPUT_DIR / "confusion_run_probabilities.csv", rows, fields)
    write_csv(OUTPUT_DIR / "confusion_matrix_probabilities.csv", [
        {"true_model": m, **{f"probability_percent_{mm}": 100 * p.item() for mm, p in zip(MODEL_NAMES, row)}}
        for m, row in zip(MODEL_NAMES, matrix)
    ], ["true_model", *[f"probability_percent_{m}" for m in MODEL_NAMES]])
    torch.save({"model_names": MODEL_NAMES, "mean_probs": matrix, "per_model_probs": per_true}, OUTPUT_DIR / "confusion_probabilities.pt")
    return matrix


def plot_confusion_matrix(matrix: torch.Tensor) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    sns.heatmap(matrix.detach().cpu().numpy(), vmin=0, vmax=1, annot=True, fmt=".2f", cmap="Blues", xticklabels=MODEL_NAMES, yticklabels=MODEL_NAMES, ax=ax)
    ax.set(xlabel="Compared model", ylabel="True model", title=f"Mean model probability, {CONFUSION_N} runs")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    savefig(fig, "population_dynamics_confusion_matrix.png")


def main() -> None:
    autocvd(num_gpus=1, interval=1)
    data = load_population_data()
    dev = device()
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Data/checkpoints: {MODEL_DIR}")
    print(f"Device: {dev}")
    print(f"Normalization: {NORMALIZATION_METHOD}, mu={data['x_mu'].item():.6g}, std={data['x_std'].item():.6g}")

    mtf = load_transfuser(dev)
    mtf.compare(x=data["notebook_mock"]["normalized"], device=dev, timesteps=CONFUSION_TIMESTEPS, method="dpm", order=2, verbose=True)
    print_probs(mtf, "Notebook-style Lotka-Volterra mock probabilities")
    test_raw = data["notebook_mock"]["raw"]
    plot_map_trajectories(mtf, test_raw)
    plot_posterior_predictive(mtf, test_raw, data, dev)
    plot_validation_diagnostics(data["raw_validation_data"])
    matrix = run_confusion_matrix(mtf, data, dev)
    plot_confusion_matrix(matrix)
    comparison_dir = OUTPUT_DIR / "mtf_plot_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    mtf.plot_comparison(path=str(comparison_dir), show=False)
    print(f"Saved ModelTransfuser comparison plots to: {comparison_dir}")


if __name__ == "__main__":
    main()
