from autocvd import autocvd
from copy import deepcopy
from pathlib import Path
from shutil import copyfile

import torch
import torch.nn as nn
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm.auto import tqdm

from compass import ScoreBasedInferenceModel as SBIm
from compass import ModelTransfuser as MTf


TUTORIAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TUTORIAL_DIR / "output" / "nested_circle_models_VP_fixed_prior_sigma=0.1"
MODEL_DIR = TUTORIAL_DIR / "data" / "nested_circle_models_VP_fixed_prior_sigma=0.1"
TRAIN_N = 200_000
VAL_N = 2_000
CONFUSION_N = 30
CONFUSION_N_OBSERVATIONS = 50
CONFUSION_TIMESTEPS = 100
CONFUSION_NUM_SAMPLES = 800
PPC_N_CASES = 2
PPC_NUM_SAMPLES = 800
PPC_POSTERIOR_CASES = 2
PPC_POSTERIOR_THETA_SAMPLES = 128
PPC_X_SAMPLES_PER_THETA = 2
RAW_SIMULATOR_PLOT_N = 2_000
FIXED_THETA_TRUE_N = 1_000
FIXED_THETA_LEARNED_N = 1_000
FIXED_THETA_MAP_OBS_N = 64
DIAGNOSTIC_MAX_RUNS = 2
LOW_CONFIDENCE_THRESHOLD = 0.65
TRAIN_BATCH_SIZE = 256
TRAIN_MAX_EPOCHS = 500
TRAIN_LR = 1e-3
TRAIN_EARLY_STOPPING_PATIENCE = 20
TRAIN_VERBOSE = True
PRETRAINED_ONLY = False
EXPECTED_X_DIM = 2
MODEL_KWARGS = dict(
    sde_type="vesde",
    sigma=3,
    hidden_size=40,
    depth=4,
    num_heads=4,
    mlp_ratio=4,
)
OBSERVATION_SIGMA = 0.1
TAU_A = 1.0
TAU_B = 1.0
MOCK_MIN_ABS_DEFORMATION = 0.5
MODEL_SPECS = [
    {"name": "Circle 1p", "family": "circle", "description": "one-parameter noisy unit circle"},
    {"name": "Horizontally scaled circle 2p", "family": "circle_a", "description": "circle horizontally scaled by a"},
    {"name": "Axis-scaled circle 3p", "family": "circle_ab", "description": "circle horizontally scaled by a and vertically scaled by b"},
]
BASELINE_N_TEST = 120
BASELINE_TRUE_MARGINAL_K = 50_000
BASELINE_CLASSIFIER_N = 100_000
BASELINE_CLASSIFIER_EPOCHS = 100
BASELINE_CONVERGENCE_RUNS = 50
BASELINE_SWEEP_TIMESTEPS = [50, 100, 200, 300, 400, 500]
BASELINE_SWEEP_ORDERS = [1]


def sample_deformation(n, mean, std, excluded_value=None, mock=False):
    values = mean + std * torch.randn(n)
    if mock and excluded_value is not None:
        too_close = (values - excluded_value).abs() < MOCK_MIN_ABS_DEFORMATION
        while too_close.any():
            values[too_close] = mean + std * torch.randn(int(too_close.sum()))
            too_close = (values - excluded_value).abs() < MOCK_MIN_ABS_DEFORMATION
    return values


def circle_mean(theta, family):
    angle = theta[:, 0]
    a = theta[:, 1] if family in {"circle_a", "circle_ab"} else torch.ones_like(angle)
    b = theta[:, 2] if family == "circle_ab" else torch.ones_like(angle)
    return torch.stack([a * torch.sin(angle), b * torch.cos(angle)], dim=1)


def gen_nested_circle(n, theta=None, mock=False, family="circle"):
    theta_dim = {"circle": 1, "circle_a": 2, "circle_ab": 3}[family]
    if theta is None:
        columns = [torch.pi * torch.rand(n)]
        if theta_dim >= 2:
            columns.append(sample_deformation(n, 1.0, TAU_A, excluded_value=1.0, mock=mock))
        if theta_dim == 3:
            columns.append(sample_deformation(n, 1.0, TAU_B, excluded_value=1.0, mock=mock))
        theta = torch.stack(columns, dim=1)
    else:
        theta = torch.as_tensor(theta, dtype=torch.float).reshape(n, theta_dim)
    mean = circle_mean(theta, family)
    x = mean + OBSERVATION_SIGMA * torch.randn(n, EXPECTED_X_DIM)
    return theta, x


def make_generator(family):
    if family in {"circle", "circle_a", "circle_ab"}:
        return lambda n, theta=None, mock=False: gen_nested_circle(n, theta=theta, mock=mock, family=family)
    raise ValueError(f"Unknown nested circle family: {family}")


for spec in MODEL_SPECS:
    spec["generator"] = make_generator(spec["family"])


def generate_single_theta_observations(spec, n_observations, mock=False):
    theta_config, _ = spec["generator"](1, mock=mock)
    theta = theta_config.repeat(n_observations, 1)
    _, x = spec["generator"](n_observations, theta=theta)
    return theta, x


def generate_confusion_mock_sets(spec):
    return [
        spec["generator"](CONFUSION_N_OBSERVATIONS, mock=True)
        for _ in range(CONFUSION_N)
    ]


def concatenate_datasets(datasets):
    theta = torch.cat([theta for theta, _ in datasets], dim=0)
    x = torch.cat([x for _, x in datasets], dim=0)
    return theta, x


def compute_normalization(*datasets):
    all_x = torch.cat([x for _, x in datasets], dim=0)
    data_mean = all_x.mean(0)
    data_std = all_x.std(0)
    data_std = torch.where(data_std == 0, torch.ones_like(data_std), data_std)
    return data_mean, data_std


def normalize(theta, x, data_mean, data_std):
    return theta, normalize_x(x, data_mean, data_std)


def normalize_x(x, data_mean, data_std):
    return (x - data_mean) / data_std


def unnormalize_x(x_norm, data_mean, data_std):
    return x_norm * data_std + data_mean


def log(message):
    print(message, flush=True)


def progress(iterable, description, leave=False):
    return tqdm(iterable, desc=description, leave=leave, dynamic_ncols=True)


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] saved {path}")


def plot_information_criterion_parameter_counts(mtf, model_names):
    dummy_x = torch.zeros(1, EXPECTED_X_DIM)
    rows = []
    log("[model] k values used for AIC/AICc/BIC penalties:")
    for model_name in model_names:
        model = mtf.models_dict[model_name]
        k = mtf._model_parameter_count(model, dummy_x)
        rows.append((model_name, int(k), int(model.nodes_size)))
        log(f"[model]   {model_name}: k={int(k)} (nodes_size={model.nodes_size}, x_dim={EXPECTED_X_DIM})")

    names = [row[0] for row in rows]
    k_values = [row[1] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, k_values, color="#4C78A8", edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Free parameters k")
    ax.set_title("Model parameter counts used for AIC/AICc/BIC")
    ax.set_ylim(0, max(k_values) + 1)
    ax.tick_params(axis="x", rotation=25)
    for bar, k in zip(bars, k_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(k), ha="center", va="bottom")
    fig.tight_layout()
    save_figure(fig, "gaussian_model_parameter_counts.png")


def save_experiment_config(data_mean=None, data_std=None, theta_dims=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "role_model": "Gaussian_models_baselines.ipynb",
        "model_dir": str(MODEL_DIR),
        "output_dir": str(OUTPUT_DIR),
        "theta_dims": theta_dims,
        "x_dim": EXPECTED_X_DIM,
        "theta_prior": "theta~Uniform(0, pi), a~N(1,tau_a^2), b~N(1,tau_b^2)",
        "train_n": TRAIN_N,
        "validation_n": VAL_N,
        "mock_sets_per_model": CONFUSION_N,
        "mock_observations_per_set": CONFUSION_N_OBSERVATIONS,
        "observation_sigma": OBSERVATION_SIGMA,
        "tau_a": TAU_A,
        "tau_b": TAU_B,
        "mock_min_distance_from_unit_scale": MOCK_MIN_ABS_DEFORMATION,
        "model_descriptions": {
            spec["name"]: spec["description"] for spec in MODEL_SPECS
        },
        "confusion_n": CONFUSION_N,
        "confusion_n_observations": CONFUSION_N_OBSERVATIONS,
        "confusion_timesteps": CONFUSION_TIMESTEPS,
        "confusion_num_samples": CONFUSION_NUM_SAMPLES,
        "model_kwargs": MODEL_KWARGS,
        "baseline_n_test": BASELINE_N_TEST,
        "baseline_true_marginal_k": BASELINE_TRUE_MARGINAL_K,
        "baseline_classifier_n": BASELINE_CLASSIFIER_N,
        "baseline_classifier_epochs": BASELINE_CLASSIFIER_EPOCHS,
        "baseline_convergence_runs": BASELINE_CONVERGENCE_RUNS,
        "baseline_sweep_timesteps": BASELINE_SWEEP_TIMESTEPS,
        "baseline_sweep_orders": BASELINE_SWEEP_ORDERS,
    }
    if data_mean is not None and data_std is not None:
        payload["data_mean"] = to_numpy(data_mean).tolist()
        payload["data_std"] = to_numpy(data_std).tolist()
    path = OUTPUT_DIR / "experiment_config.json"
    with path.open("w") as config_file:
        json.dump(payload, config_file, indent=2)
    log(f"[config] saved {path}")


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def format_theta(theta_row):
    values = to_numpy(theta_row).reshape(-1)
    return "theta=(" + ", ".join(f"{value:.2f}" for value in values) + ")"


def format_theta_mean(theta):
    means = to_numpy(theta).mean(axis=0).reshape(-1)
    return "theta mean=(" + ", ".join(f"{value:.2f}" for value in means) + ")"


def add_covariance_guide(ax, spec, theta, color):
    theta = torch.as_tensor(theta, dtype=torch.float).reshape(1, -1)
    mean = circle_mean(theta, spec["family"])[0]
    ax.scatter(mean[0], mean[1], marker="x", s=90, linewidth=2.2, color=color, zorder=5)


def sample_theta_configs(spec, n_configs, mock):
    theta, _ = spec["generator"](n_configs, mock=mock)
    return theta


def samples_for_theta(spec, theta, n_samples):
    theta_batch = theta.unsqueeze(0).repeat(n_samples, 1)
    _, x = spec["generator"](n_samples, theta=theta_batch)
    return x


def plot_raw_simulator_samples(raw_data_by_model):
    log("[plot] creating raw simulator sample check")
    n_models = len(MODEL_SPECS)
    fig, axes = plt.subplots(1, n_models, figsize=(4.3 * n_models, 4.2), sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    all_x = torch.cat([x[:RAW_SIMULATOR_PLOT_N] for _, x in raw_data_by_model.values()], dim=0)
    pad = 1.0
    x_min, y_min = all_x[:, :2].min(dim=0).values - pad
    x_max, y_max = all_x[:, :2].max(dim=0).values + pad

    for ax, spec in zip(axes, MODEL_SPECS):
        theta, x = raw_data_by_model[spec["name"]]
        x = x[:RAW_SIMULATOR_PLOT_N]
        theta = theta[:RAW_SIMULATOR_PLOT_N]
        color_values = theta[:, 0]
        scatter = ax.scatter(
            x[:, 0],
            x[:, 1],
            c=color_values,
            cmap="viridis",
            s=6,
            alpha=0.35,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(spec["name"])
        ax.set_xlabel("x_1")
        ax.set_xlim(float(x_min), float(x_max))
        ax.set_ylim(float(y_min), float(y_max))
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="0.88", lw=0.8, zorder=0)
        ax.axvline(0, color="0.88", lw=0.8, zorder=0)
        ax.grid(True, color="0.92", linewidth=0.8)
        sns.despine(ax=ax)

    axes[0].set_ylabel("x_2")
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("theta_1")
    fig.suptitle("Raw simulator samples for the three nested circle hypotheses")
    fig.tight_layout(rect=(0, 0, 0.94, 0.93))
    save_figure(fig, "gaussian_raw_simulator_samples.png")


def plot_fixed_theta_true_vs_learned(mtf, models_by_name, data_mean, data_std, device):
    log("[plot] creating true-theta vs inferred-MAP diagnostic")
    n_models = len(MODEL_SPECS)
    fig, axes = plt.subplots(n_models, 2, figsize=(9.4, 3.8 * n_models), sharex=False, sharey=False)
    if n_models == 1:
        axes = axes[None, :]

    colors = {"true": "tab:orange", "learned": "tab:blue"}
    for row, spec in enumerate(MODEL_SPECS):
        model_name = spec["name"]
        model = models_by_name[model_name]
        theta_true = sample_theta_configs(spec, 1, mock=True)[0]
        true_x = samples_for_theta(spec, theta_true, FIXED_THETA_TRUE_N)

        x_obs = true_x[:FIXED_THETA_MAP_OBS_N]
        x_obs_norm = normalize_x(x_obs, data_mean, data_std)
        model_condition_mask = mtf._condition_mask_for_model(model, x_obs_norm, None)
        with torch.no_grad():
            posterior_theta = model.sample(
                x=x_obs_norm,
                condition_mask=model_condition_mask,
                timesteps=CONFUSION_TIMESTEPS,
                num_samples=CONFUSION_NUM_SAMPLES,
                method="dpm",
                order=1,
                device=device,
                verbose=False,
                multi_obs_inference=False,
            )

        theta_dim = posterior_theta.shape[-1]
        posterior_theta_flat = to_numpy(posterior_theta).reshape(-1, theta_dim)
        theta_map_np, theta_map_std_np = mtf._map_kde(posterior_theta_flat)
        theta_map = torch.tensor(theta_map_np, dtype=torch.float)
        theta_map_std = torch.tensor(theta_map_std_np, dtype=torch.float)

        theta_batch = theta_map.unsqueeze(0).repeat(FIXED_THETA_LEARNED_N, 1)
        with torch.no_grad():
            learned_x_norm = model.sample(
                theta=theta_batch,
                timesteps=CONFUSION_TIMESTEPS,
                num_samples=1,
                method="dpm",
                order=1,
                device=device,
                verbose=False,
            )
        learned_x_norm = learned_x_norm.reshape(FIXED_THETA_LEARNED_N, EXPECTED_X_DIM)
        learned_x = unnormalize_x(
            learned_x_norm,
            data_mean.to(learned_x_norm.device),
            data_std.to(learned_x_norm.device),
        )

        x_both = torch.cat([true_x, learned_x.cpu()], dim=0)
        pad = 1.0
        x_min, y_min = x_both[:, :2].min(dim=0).values - pad
        x_max, y_max = x_both[:, :2].max(dim=0).values + pad

        panels = (
            ("true simulator", true_x, theta_true, None),
            ("learned from inferred MAP", learned_x.cpu(), theta_map, theta_map_std),
        )
        for col, (label, samples, theta_display, theta_std) in enumerate(panels):
            ax = axes[row, col]
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                s=7,
                alpha=0.28,
                color=colors["true" if col == 0 else "learned"],
                edgecolors="none",
                rasterized=True,
            )
            add_covariance_guide(ax, spec, theta_display, "black")
            if col == 0:
                title = f"{model_name}: {label}\ntrue {format_theta(theta_true)}"
            else:
                std_text = ", ".join(f"{v:.2f}" for v in to_numpy(theta_std).reshape(-1))
                title = (
                    f"{model_name}: {label}\n"
                    f"MAP {format_theta(theta_map)}\n"
                    f"true {format_theta(theta_true)}; MAP std=({std_text})"
                )
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x_1")
            if col == 0:
                ax.set_ylabel("x_2")
            ax.set_xlim(float(x_min), float(x_max))
            ax.set_ylim(float(y_min), float(y_max))
            ax.set_aspect("equal", adjustable="box")
            ax.axhline(0, color="0.88", lw=0.8, zorder=0)
            ax.axvline(0, color="0.88", lw=0.8, zorder=0)
            ax.grid(True, color="0.92", linewidth=0.8)
            sns.despine(ax=ax)

    fig.suptitle(
        "True-theta vs inferred-MAP diagnostic: simulator samples and learned likelihood samples"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "gaussian_fixed_theta_true_vs_learned.png")


def make_pairplot(val_data_by_model, mock=True):
    plotted_variables = ["x_1", "x_2", "theta"]
    deformation_settings = [
        (0.5, "gaussian_hypotheses_pairplot.png"),
        (0.3, "gaussian_hypotheses_pairplot_ab_0p3.png"),
        (0.0, "gaussian_hypotheses_pairplot_ab_0.png"),
    ]

    for deformation, filename in deformation_settings:
        log(f"[plot] creating validation pairplot with a=b={deformation:g}")
        frames = []
        for spec in MODEL_SPECS:
            model_name = spec["name"]
            validation_theta, _ = val_data_by_model[model_name]
            theta = validation_theta.clone()
            if spec["family"] in {"circle_a", "circle_ab"}:
                theta[:, 1] = deformation
            if spec["family"] == "circle_ab":
                theta[:, 2] = deformation

            _, x = spec["generator"](len(theta), theta=theta)
            x_np = to_numpy(x)
            frames.append(pd.DataFrame({
                "x_1": x_np[:, 0],
                "x_2": x_np[:, 1],
                "theta": to_numpy(theta[:, 0]),
                "Hypothesis": model_name,
            }))

        combined_df = pd.concat(frames, axis=0, ignore_index=True)
        pairplot = sns.pairplot(
            combined_df,
            vars=plotted_variables,
            diag_kind="kde",
            hue="Hypothesis",
            plot_kws=dict(alpha=0.5, s=3),
        )
        pairplot.fig.suptitle(
            f"Joint latent–observation space (a=b={deformation:g} where applicable)",
            y=1.02,
        )
        save_figure(pairplot.fig, filename)

def plot_mock_data_overview(data_by_model):
    log("[plot] creating mock-data overview")
    n_models = len(data_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4), sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]
    palette = dict(zip(data_by_model.keys(), sns.color_palette("deep", len(data_by_model))))

    for ax, (model_name, (theta, x)) in zip(axes, progress(data_by_model.items(), "[plot] overview panels")):
        theta_np = to_numpy(theta)
        x_np = to_numpy(x)
        color_values = theta_np[:, 0]
        ax.scatter(x_np[:, 0], x_np[:, 1], c=color_values, s=5, alpha=0.45, cmap="viridis")
        ax.set_title(model_name)
        ax.set_xlabel("x_1")
        ax.axhline(0, color="0.85", linewidth=0.8)
        ax.axvline(0, color="0.85", linewidth=0.8)
        ax.text(
            0.02,
            0.98,
            f"{format_theta_mean(theta)}\nx mean=({x_np[:, 0].mean():.2f}, {x_np[:, 1].mean():.2f})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="0.8"),
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor((*palette[model_name], 0.04))

    axes[0].set_ylabel("x_2")
    fig.suptitle("Mock data generated from the three nested circle hypotheses")
    fig.tight_layout()
    save_figure(fig, "gaussian_mock_data_overview.png")


def plot_likelihood_predictive_checks(models_by_name, val_data_by_model, device):
    log("[ppc] running learned likelihood checks p(x|theta)")
    fig, axes = plt.subplots(
        len(models_by_name),
        PPC_N_CASES,
        figsize=(2.5 * PPC_N_CASES, 2.6 * len(models_by_name)),
        sharex=True,
        sharey=True,
    )
    if len(models_by_name) == 1:
        axes = axes[None, :]

    for row, (model_name, model) in enumerate(
        progress(models_by_name.items(), "[ppc] likelihood models")
    ):
        log(f"[ppc] {model_name}: sampling likelihood predictions for {PPC_N_CASES} theta cases")
        val_theta, val_x = val_data_by_model[model_name]
        theta_cases = val_theta[:PPC_N_CASES]
        x_cases = val_x[:PPC_N_CASES]

        with torch.no_grad():
            predictive = model.sample(
                theta=theta_cases,
                timesteps=CONFUSION_TIMESTEPS,
                num_samples=PPC_NUM_SAMPLES,
                method="dpm",
                order=1,
                device=device,
                verbose=False,
            )
        predictive_np = to_numpy(predictive)
        x_cases_np = to_numpy(x_cases)
        theta_cases_np = to_numpy(theta_cases)

        for col in range(PPC_N_CASES):
            ax = axes[row, col]
            ax.scatter(
                predictive_np[col, :, 0],
                predictive_np[col, :, 1],
                s=4,
                alpha=0.18,
                color="tab:blue",
                label="model p(x|theta)",
            )
            ax.scatter(
                x_cases_np[col, 0],
                x_cases_np[col, 1],
                s=48,
                color="black",
                marker="x",
                linewidth=1.8,
                label="held-out simulator x",
            )
            ax.set_title(format_theta(theta_cases_np[col]), fontsize=9)
            if col == 0:
                ax.set_ylabel(model_name)
            if row == len(models_by_name) - 1:
                ax.set_xlabel("x_1")
            ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Predictive check: learned likelihood samples at held-out theta")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "gaussian_likelihood_predictive_check.png")


def plot_posterior_predictive_checks(models_by_name, val_data_by_model, device):
    log("[ppc] running posterior predictive checks x_rep from p(theta|x_obs)")
    fig, axes = plt.subplots(
        len(models_by_name),
        PPC_POSTERIOR_CASES,
        figsize=(3.2 * PPC_POSTERIOR_CASES, 2.9 * len(models_by_name)),
        sharex=True,
        sharey=True,
    )
    if len(models_by_name) == 1:
        axes = axes[None, :]

    for row, (model_name, model) in enumerate(
        progress(models_by_name.items(), "[ppc] posterior predictive models")
    ):
        log(f"[ppc] {model_name}: sampling posterior theta and replicated x")
        _, val_x = val_data_by_model[model_name]
        x_cases = val_x[:PPC_POSTERIOR_CASES]

        with torch.no_grad():
            posterior_theta = model.sample(
                x=x_cases,
                timesteps=CONFUSION_TIMESTEPS,
                num_samples=PPC_POSTERIOR_THETA_SAMPLES,
                method="dpm",
                order=1,
                device=device,
                verbose=False,
            )
        theta_dim = posterior_theta.shape[-1]
        posterior_theta = posterior_theta.reshape(-1, theta_dim)

        with torch.no_grad():
            replicated_x = model.sample(
                theta=posterior_theta,
                timesteps=CONFUSION_TIMESTEPS,
                num_samples=PPC_X_SAMPLES_PER_THETA,
                method="dpm",
                order=1,
                device=device,
                verbose=False,
            )

        replicated_x_np = to_numpy(replicated_x).reshape(PPC_POSTERIOR_CASES, -1, EXPECTED_X_DIM)
        x_cases_np = to_numpy(x_cases)

        for col in range(PPC_POSTERIOR_CASES):
            ax = axes[row, col]
            ax.scatter(
                replicated_x_np[col, :, 0],
                replicated_x_np[col, :, 1],
                s=4,
                alpha=0.16,
                color="tab:green",
                label="posterior predictive x_rep",
            )
            ax.scatter(
                x_cases_np[col, 0],
                x_cases_np[col, 1],
                s=58,
                color="black",
                marker="x",
                linewidth=1.9,
                label="observed x",
            )
            ax.set_title(f"held-out obs {col}", fontsize=9)
            if col == 0:
                ax.set_ylabel(model_name)
            if row == len(models_by_name) - 1:
                ax.set_xlabel("x_1")
            ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Posterior predictive check before model comparison")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "gaussian_posterior_predictive_check.png")


def sample_likelihood_from_compare_maps(mtf, model_names, device):
    samples_by_model = {}
    for model_name in progress(model_names, "[diagnostic] likelihood clouds"):
        log(f"[diagnostic] sampling likelihood cloud for {model_name}")
        model = mtf.models_dict[model_name]
        theta_hat = torch.tensor(mtf.stats[model_name]["MAP"][:, 0], dtype=torch.float)
        theta_err = torch.tensor(mtf.stats[model_name]["MAP"][:, 1], dtype=torch.float)
        with torch.no_grad():
            samples_by_model[model_name] = model.sample(
                theta=theta_hat,
                err=theta_err,
                timesteps=CONFUSION_TIMESTEPS,
                num_samples=CONFUSION_NUM_SAMPLES,
                method="dpm",
                order=1,
                device=device,
                verbose=False,
            )
    return samples_by_model


def plot_wrong_prediction_map_summary(
    mtf,
    model_names,
    true_model_name,
    observations,
    data_mean,
    data_std,
    run_index,
):
    probs = torch.tensor([mtf.stats[model_name]["model_prob"] for model_name in model_names])
    predicted_model = model_names[probs.argmax().item()]
    if predicted_model == true_model_name:
        return

    log(
        f"[diagnostic] creating MAP overlay for wrong prediction: "
        f"true={true_model_name}, predicted={predicted_model}, run={run_index}"
    )
    observations_raw = unnormalize_x(observations, data_mean, data_std)
    observations_np = to_numpy(observations_raw)
    colors = dict(zip(model_names, sns.color_palette("deep", len(model_names))))

    fig, axes = plt.subplots(1, len(model_names), figsize=(5.0 * len(model_names), 4.8), sharex=True, sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    x_min, y_min = observations_raw[:, :2].min(dim=0).values - 1.5
    x_max, y_max = observations_raw[:, :2].max(dim=0).values + 1.5
    guide_alpha = min(0.32, max(0.06, 8.0 / len(observations_raw)))

    for ax, model_name in zip(axes, model_names):
        spec = next(spec for spec in MODEL_SPECS if spec["name"] == model_name)
        color = colors[model_name]
        map_theta = torch.tensor(mtf.stats[model_name]["MAP"][:, 0], dtype=torch.float)
        map_mean = map_theta.mean(dim=0)
        map_std = map_theta.std(dim=0)

        ax.scatter(
            observations_np[:, 0],
            observations_np[:, 1],
            s=22,
            alpha=0.55,
            color="0.2",
            edgecolors="white",
            linewidth=0.35,
            label="mock observations",
            zorder=2,
        )

        for theta in map_theta:
            add_covariance_guide(ax, spec, theta, color)
            for patch in ax.patches[-2:]:
                patch.set_alpha(guide_alpha)
                patch.set_linewidth(1.1)
            for line in ax.lines[-1:]:
                line.set_alpha(guide_alpha)
                line.set_linewidth(0.8)
            for collection in ax.collections[1:]:
                collection.set_alpha(guide_alpha)

        add_covariance_guide(ax, spec, map_mean, "black")
        ax.set_title(
            f"{model_name}\nP={probs[model_names.index(model_name)].item():.2f}\n"
            f"MAP mean {format_theta(map_mean)}\n"
            f"MAP std=({', '.join(f'{v:.2f}' for v in to_numpy(map_std).reshape(-1))})",
            fontsize=9,
        )
        ax.set_xlabel("x_1")
        ax.set_xlim(float(x_min), float(x_max))
        ax.set_ylim(float(y_min), float(y_max))
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="0.88", lw=0.8, zorder=0)
        ax.axvline(0, color="0.88", lw=0.8, zorder=0)
        ax.grid(True, color="0.92", linewidth=0.8)
        sns.despine(ax=ax)

    axes[0].set_ylabel("x_2")
    fig.suptitle(
        f"Wrong comparison diagnostic: true={true_model_name}, predicted={predicted_model}, run={run_index}\n"
        "Raw mock data with each model's per-observation MAP-implied centers/covariances"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    filename = (
        "misspecification_runs"
        f"/wrong_prediction_map_overlay_{true_model_name.replace(' ', '_')}_run_{run_index:02d}.png"
    )
    save_figure(fig, filename)


def plot_compare_diagnostic(mtf, model_names, true_model_name, observations, run_index, device):
    log(f"[diagnostic] creating compare diagnostic for true={true_model_name}, run={run_index}")
    obs_probs = torch.stack([mtf.stats[name]["obs_probs"] for name in model_names])
    predictions = obs_probs.argmax(dim=0)
    true_index = model_names.index(true_model_name)
    wrong_mask = predictions != true_index
    if wrong_mask.any():
        selected = torch.where(wrong_mask)[0][:9]
    else:
        true_probs = obs_probs[true_index]
        selected = torch.argsort(true_probs)[:9]

    likelihood_samples = sample_likelihood_from_compare_maps(mtf, model_names, device)
    observations_np = to_numpy(observations)

    fig, axes = plt.subplots(3, 3, figsize=(11, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    colors = dict(zip(model_names, sns.color_palette("deep", len(model_names))))

    for ax, obs_idx in zip(axes, selected.tolist()):
        for model_name in model_names:
            samples = to_numpy(likelihood_samples[model_name][obs_idx])
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                s=3,
                alpha=0.13,
                color=colors[model_name],
                label=model_name,
            )
        ax.scatter(
            observations_np[obs_idx, 0],
            observations_np[obs_idx, 1],
            s=55,
            marker="x",
            color="black",
            linewidth=2,
            label="observed x",
        )
        prob_text = "\n".join(
            f"{name}: {obs_probs[i, obs_idx].item():.2f}"
            for i, name in enumerate(model_names)
        )
        ax.text(
            0.02,
            0.98,
            prob_text,
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="0.8"),
        )
        ax.set_title(f"obs {obs_idx}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(selected):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(model_names) + 1, frameon=False)
    fig.suptitle(
        f"Compare diagnostic for {true_model_name}, run {run_index}: "
        "likelihood samples at MAP theta versus observed x"
    )
    fig.supxlabel("x_1")
    fig.supylabel("x_2")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    filename = f"compare_diagnostic_{true_model_name.replace(' ', '_')}_run_{run_index:02d}.png"
    save_figure(fig, filename)


def should_save_compare_diagnostic(mtf, model_names, true_model_name):
    probs = torch.tensor([mtf.stats[model_name]["model_prob"] for model_name in model_names])
    predicted = model_names[probs.argmax().item()]
    true_prob = probs[model_names.index(true_model_name)].item()
    return predicted != true_model_name or true_prob < LOW_CONFIDENCE_THRESHOLD


def model_checkpoint_paths(model_names):
    return {model_name: MODEL_DIR / f"{model_name}.pt" for model_name in model_names}


def promote_best_checkpoint(model_name, overwrite=False):
    final_path = MODEL_DIR / f"{model_name}.pt"
    checkpoint_path = MODEL_DIR / f"{model_name}_checkpoint.pt"
    if final_path.exists() and not overwrite:
        return final_path
    if checkpoint_path.exists():
        copyfile(checkpoint_path, final_path)
        log(f"[train] promoted best checkpoint {checkpoint_path.name} -> {final_path.name}")
        return final_path
    raise FileNotFoundError(
        f"Expected either {final_path} or {checkpoint_path}, but neither exists. "
        "Training likely stopped before the first checkpoint was saved."
    )



def train_missing_models(mtf, missing_models, device):
    log(
        f"[train] missing checkpoints in {MODEL_DIR}: {missing_models}. "
        "Training missing nested circle models now."
    )
    log(
        f"[train] batch_size={TRAIN_BATCH_SIZE}, max_epochs={TRAIN_MAX_EPOCHS}, "
        f"lr={TRAIN_LR}, early_stopping_patience={TRAIN_EARLY_STOPPING_PATIENCE}"
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in progress(missing_models, "[train] models", leave=True):
        log(f"[train] training {model_name}")
        model = mtf.models_dict[model_name]
        data = mtf.data_dict[model_name]
        model.train(
            theta=data["train_theta"],
            x=data["train_x"],
            theta_val=data.get("val_theta"),
            x_val=data.get("val_x"),
            batch_size=TRAIN_BATCH_SIZE,
            max_epochs=TRAIN_MAX_EPOCHS,
            lr=TRAIN_LR,
            device=device,
            verbose=TRAIN_VERBOSE,
            path=str(MODEL_DIR),
            name=model_name,
            early_stopping_patience=TRAIN_EARLY_STOPPING_PATIENCE,
        )
        promote_best_checkpoint(model_name, overwrite=True)
        torch.cuda.empty_cache()

    log(f"[train] finished training; checkpoints saved in {MODEL_DIR}")


def load_or_train_models(mtf, model_names, device):
    checkpoint_paths = model_checkpoint_paths(model_names)
    missing_models = [
        model_name
        for model_name, checkpoint_path in checkpoint_paths.items()
        if not checkpoint_path.exists()
    ]
    if missing_models:
        if PRETRAINED_ONLY:
            missing_paths = [str(checkpoint_paths[name]) for name in missing_models]
            raise FileNotFoundError(
                "PRETRAINED_ONLY=True, but required checkpoints are missing: "
                + ", ".join(missing_paths)
            )
        train_missing_models(mtf, missing_models, device)
    else:
        log(f"[model] found all pretrained checkpoints in {MODEL_DIR}; loading without training")

    models_by_name = {}
    for model_name, checkpoint_path in progress(
        checkpoint_paths.items(), "[model] loading checkpoints"
    ):
        checkpoint_path = promote_best_checkpoint(model_name)
        log(f"[model] loading {checkpoint_path}")
        model = SBIm.load(str(checkpoint_path), device=device)
        expected_theta_dim = mtf.data_dict[model_name]["train_theta"].shape[1]
        expected_nodes_size = expected_theta_dim + EXPECTED_X_DIM
        if model.nodes_size != expected_nodes_size:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has nodes_size={model.nodes_size}, "
                f"but {model_name} expects nodes_size={expected_nodes_size}. "
                "Delete the checkpoint or use the nested_circle_models_VP model directory."
            )
        checkpoint_kwargs = {
            "sde_type": model.sde_type,
            "sigma": model.sigma,
            "hidden_size": model.hidden_size,
            "depth": model.depth,
            "num_heads": model.num_heads,
            "mlp_ratio": model.mlp_ratio,
        }
        if checkpoint_kwargs != MODEL_KWARGS:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} was trained with {checkpoint_kwargs}, "
                f"but this script now expects {MODEL_KWARGS}. "
                "Retrain the Gaussian checkpoints before evaluation."
            )
        models_by_name[model_name] = model

    return models_by_name



def model_log_likelihood(x, theta, family):
    mean = circle_mean(theta, family)
    return torch.distributions.Normal(mean, OBSERVATION_SIGMA).log_prob(x).sum(dim=1)


def sample_prior_theta(spec, n, mock=True):
    theta, _ = spec["generator"](n, mock=mock)
    return theta


def compute_true_marginal_probs(test_x, k=BASELINE_TRUE_MARGINAL_K):
    log(f"[baseline] computing true marginal probabilities with K={k}")
    log_marginals = torch.zeros(test_x.shape[0], len(MODEL_SPECS))
    for j, spec in enumerate(MODEL_SPECS):
        theta_mc = sample_prior_theta(spec, k, mock=True)
        for i in range(test_x.shape[0]):
            x_i = test_x[i:i + 1].expand(k, -1)
            ll = model_log_likelihood(x_i, theta_mc, spec["family"])
            log_marginals[i, j] = torch.logsumexp(ll, 0) - np.log(k)
    return torch.softmax(log_marginals, dim=1)


def train_classifier():
    log("[baseline] training amortized classifier baseline")
    n = BASELINE_CLASSIFIER_N
    x_parts = []
    y_parts = []
    for label, spec in enumerate(MODEL_SPECS):
        _, x_part = spec["generator"](n, mock=True)
        x_parts.append(x_part)
        y_parts.append(torch.full((n,), label, dtype=torch.long))
    x_train = torch.cat(x_parts)
    y_train = torch.cat(y_parts)
    perm = torch.randperm(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    classifier = nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, len(MODEL_SPECS)),
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, BASELINE_CLASSIFIER_EPOCHS + 1):
        epoch_loss = 0.0
        for i in range(0, len(x_train), 1024):
            logits = classifier(x_train[i:i + 1024])
            loss = criterion(logits, y_train[i:i + 1024])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        log(f"[baseline] classifier epoch {epoch}/{BASELINE_CLASSIFIER_EPOCHS} loss={epoch_loss:.3f}")
    classifier.eval()
    return classifier


def classifier_probs(classifier, test_x):
    with torch.no_grad():
        return torch.softmax(classifier(test_x), dim=1)


def model_parameter_counts(mtf, hyp_names):
    dummy_x = torch.zeros(1, EXPECTED_X_DIM)
    counts = []
    for name in hyp_names:
        counts.append(float(mtf._model_parameter_count(mtf.models_dict[name], dummy_x)))
    return torch.tensor(counts, dtype=torch.float)


def compass_per_observation_ic_probs(mtf, hyp_names):
    no_penalty_probs = torch.stack(
        [torch.as_tensor(mtf.stats[name]["obs_probs"], dtype=torch.float) for name in hyp_names],
        dim=1,
    )
    log_scores = torch.log(no_penalty_probs + 1e-30)
    k = model_parameter_counts(mtf, hyp_names).to(log_scores.device)
    n_single_observation = 1.0
    return {
        "No penalty": torch.softmax(log_scores, dim=1),
        "AIC": torch.softmax(log_scores - k, dim=1),
        "BIC": torch.softmax(log_scores - 0.5 * k * np.log(n_single_observation), dim=1),
    }


def compass_probs(mtf, test_x, hyp_names, data_mean, data_std, device, timesteps=200, order=1, return_ic_probs=False):
    test_x_norm = normalize_x(test_x, data_mean, data_std)
    mtf.compare(
        x=test_x_norm,
        device=device,
        timesteps=timesteps,
        method="dpm",
        order=order,
        multi_obs_inference=False,
    )
    ic_probs = compass_per_observation_ic_probs(mtf, hyp_names)
    if return_ic_probs:
        return ic_probs["No penalty"], ic_probs
    return ic_probs["No penalty"]


def collect_baseline_results(mtf, hyp_names, data_mean, data_std, device):
    log("[baseline] generating calibration test sets from all hypotheses")
    test_sets = {}
    results = {
        method: {"probs": [], "true_labels": []}
        for method in ["True Marginal", "COMPASS", "Classifier"]
    }
    compass_ic_results = {
        method: {"probs": [], "true_labels": []}
        for method in ["No penalty", "AIC", "BIC"]
    }
    classifier = train_classifier()

    for i, spec in enumerate(MODEL_SPECS):
        theta, test_x = spec["generator"](BASELINE_N_TEST, mock=True)
        test_sets[spec["name"]] = {"x": test_x, "theta": theta, "true_label": i}

        log(f"[baseline] evaluating true marginal for {spec['name']} test set")
        true_p = compute_true_marginal_probs(test_x)
        results["True Marginal"]["probs"].append(true_p)
        results["True Marginal"]["true_labels"].extend([i] * BASELINE_N_TEST)

        log(f"[baseline] evaluating classifier for {spec['name']} test set")
        clf_p = classifier_probs(classifier, test_x)
        results["Classifier"]["probs"].append(clf_p)
        results["Classifier"]["true_labels"].extend([i] * BASELINE_N_TEST)

        log(f"[baseline] evaluating COMPASS for {spec['name']} test set")
        compass_p, compass_ic_probs = compass_probs(
            mtf,
            test_x,
            hyp_names,
            data_mean,
            data_std,
            device,
            return_ic_probs=True,
        )
        results["COMPASS"]["probs"].append(compass_p)
        results["COMPASS"]["true_labels"].extend([i] * BASELINE_N_TEST)
        for method, probs in compass_ic_probs.items():
            compass_ic_results[method]["probs"].append(probs)
            compass_ic_results[method]["true_labels"].extend([i] * BASELINE_N_TEST)

    for method in results:
        results[method]["probs"] = torch.cat(results[method]["probs"], dim=0)
        results[method]["true_labels"] = torch.tensor(results[method]["true_labels"])
    for method in compass_ic_results:
        compass_ic_results[method]["probs"] = torch.cat(compass_ic_results[method]["probs"], dim=0)
        compass_ic_results[method]["true_labels"] = torch.tensor(compass_ic_results[method]["true_labels"])
    return results, test_sets, compass_ic_results


def plot_calibration_panels(results, methods, filename, log_label):
    log(f"[baseline] plotting {log_label}")
    fig, axes = plt.subplots(1, len(methods), figsize=(5.35 * len(methods), 4.5))
    if len(methods) == 1:
        axes = [axes]
    n_bins = 10

    for ax, method in zip(axes, methods):
        probs = results[method]["probs"]
        true_labels = results[method]["true_labels"]
        all_pred_probs = []
        all_correct = []
        for cls in range(probs.shape[1]):
            all_pred_probs.append(probs[:, cls])
            all_correct.append((true_labels == cls).float())
        all_pred_probs = torch.cat(all_pred_probs)
        all_correct = torch.cat(all_correct)

        bin_edges = torch.linspace(0, 1, n_bins + 1)
        bin_centers, bin_accs = [], []
        for i in range(n_bins):
            mask = (all_pred_probs >= bin_edges[i]) & (all_pred_probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]).item() / 2)
                bin_accs.append(all_correct[mask].mean().item())

        ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color="steelblue", edgecolor="k")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        ax.set_title(method, fontsize=14)
        ax.set_xlabel("Predicted Probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if method == methods[0]:
            ax.set_ylabel("Observed Frequency")
        ax.legend(loc="lower right")

    fig.tight_layout()
    save_figure(fig, filename)


def plot_baseline_calibration(results):
    plot_calibration_panels(
        results,
        ["True Marginal", "COMPASS", "Classifier"],
        "baseline_calibration.png",
        "calibration comparison",
    )


def plot_compass_ic_calibration(compass_ic_results):
    plot_calibration_panels(
        compass_ic_results,
        ["No penalty", "AIC", "BIC"],
        "baseline_compass_ic_calibration.png",
        "COMPASS per-observation IC calibration comparison",
    )


def cumulative_true_model_curve(probs, permutations, true_model_idx=0):
    log_p = torch.log(probs + 1e-30)
    runs = []
    n_obs = probs.shape[0]
    for perm in permutations:
        cum_probs = []
        for i in range(n_obs + 1):
            if i == 0:
                summed = torch.zeros(log_p.shape[1])
            else:
                summed = log_p[perm[:i], :].sum(0)
            cum_probs.append(torch.nn.functional.softmax(summed, dim=0))
        runs.append(torch.stack(cum_probs))
    runs = torch.stack(runs)
    mean = runs.mean(0)
    stderr = runs.std(0) / torch.sqrt(torch.tensor(float(runs.shape[0])))
    return mean[:, true_model_idx], stderr[:, true_model_idx]


def summarize_nan_runs(runs):
    valid = torch.isfinite(runs)
    counts = valid.sum(dim=0).clamp_min(1)
    safe_runs = torch.where(valid, runs, torch.zeros_like(runs))
    mean = safe_runs.sum(dim=0) / counts
    centered = torch.where(valid, runs - mean, torch.zeros_like(runs))
    denom = (counts - 1).clamp_min(1)
    std = torch.sqrt((centered ** 2).sum(dim=0) / denom)
    stderr = std / torch.sqrt(counts.float())
    mean = torch.where(valid.any(dim=0), mean, torch.full_like(mean, float("nan")))
    stderr = torch.where(valid.any(dim=0), stderr, torch.full_like(stderr, float("nan")))
    return mean, stderr


def cumulative_compass_ic_curve(probs, permutations, k, criterion, true_model_idx=0):
    log_p = torch.log(probs + 1e-30)
    runs = []
    n_obs = probs.shape[0]
    max_k = int(torch.ceil(k.max()).item())
    for perm in permutations:
        true_model_probs = []
        for i in range(n_obs + 1):
            if i == 0:
                scores = torch.zeros(log_p.shape[1])
            else:
                scores = log_p[perm[:i], :].sum(0)
                if criterion == "AIC":
                    scores = scores - k
                elif criterion == "AICc":
                    if i <= max_k + 1:
                        true_model_probs.append(torch.tensor(float("nan")))
                        continue
                    scores = scores - k - (k * (k + 1.0)) / (i - k - 1.0)
                elif criterion == "BIC":
                    scores = scores - 0.5 * k * np.log(i)
                elif criterion != "No penalty":
                    raise ValueError(f"Unknown COMPASS information criterion: {criterion}")
            true_model_probs.append(torch.nn.functional.softmax(scores, dim=0)[true_model_idx])
        runs.append(torch.stack(true_model_probs))
    return summarize_nan_runs(torch.stack(runs))


def plot_compass_ic_convergence(results, mtf, hyp_names):
    log("[baseline] plotting COMPASS information-criterion convergence comparison")
    n_obs = BASELINE_N_TEST
    probs = results["COMPASS"]["probs"][:n_obs]
    permutations = [torch.randperm(n_obs) for _ in range(BASELINE_CONVERGENCE_RUNS)]
    k = model_parameter_counts(mtf, hyp_names).to(probs.device)
    x_axis = torch.arange(0, n_obs + 1)
    styles = {
        "No penalty": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "AIC": {"color": "#9467bd", "linestyle": "--", "marker": "s"},
        "AICc": {"color": "#2ca02c", "linestyle": "-.", "marker": "D"},
        "BIC": {"color": "#d62728", "linestyle": ":", "marker": "^"},
    }

    fig, ax = plt.subplots(figsize=(12, 6), dpi=500)
    for criterion, style in styles.items():
        mean_curve, se_curve = cumulative_compass_ic_curve(
            probs,
            permutations,
            k,
            criterion,
            true_model_idx=0,
        )
        ax.errorbar(
            x_axis,
            mean_curve,
            yerr=se_curve,
            label=f"COMPASS {criterion}",
            marker=style["marker"],
            markevery=2,
            markersize=5,
            linewidth=3,
            elinewidth=1,
            capsize=2,
            color=style["color"],
            ls=style["linestyle"],
        )

    ax.axhline(y=0.95, color="gray", ls=":", label="95% threshold", linewidth=2)
    ax.set_xlabel("# Observations", fontsize=20)
    ax.set_ylabel(r"$P(\mathcal{M}_1 \mid x_1, \ldots, x_n)$", fontsize=20)
    ax.set_title("Convergence: COMPASS Information-Criterion Penalties", fontsize=24)
    ax.legend(fontsize=15, frameon=True)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 20)
    sns.despine()
    fig.tight_layout()
    save_figure(fig, "baseline_compass_ic_convergence.png")


def plot_baseline_convergence(results):
    log("[baseline] plotting convergence comparison")
    n_obs = BASELINE_N_TEST
    permutations = [torch.randperm(n_obs) for _ in range(BASELINE_CONVERGENCE_RUNS)]
    colors_method = {"True Marginal": "black", "COMPASS": "#1f77b4", "Classifier": "#ff7f0e"}
    markers_method = {"True Marginal": "s", "COMPASS": "o", "Classifier": "^"}
    linestyles_method = {"True Marginal": "--", "COMPASS": "-", "Classifier": "-."}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=500)
    for method in ["Classifier", "True Marginal", "COMPASS"]:
        probs = results[method]["probs"][:n_obs]
        mean_curve, se_curve = cumulative_true_model_curve(probs, permutations, true_model_idx=0)
        ax.errorbar(
            torch.arange(0, n_obs + 1),
            mean_curve,
            yerr=se_curve,
            label=method,
            marker=markers_method[method],
            markersize=6,
            linewidth=3,
            elinewidth=1,
            capsize=2,
            color=colors_method[method],
            ls=linestyles_method[method],
        )
    ax.axhline(y=0.95, color="gray", ls=":", label="95% threshold", linewidth=2)
    ax.set_xlabel("# Observations", fontsize=20)
    ax.set_ylabel(r"$P(\mathcal{M}_1 \mid x_1, \ldots, x_n)$", fontsize=20)
    ax.set_title("Convergence: Cumulative Evidence for True Model", fontsize=24)
    ax.legend(fontsize=15, frameon=True)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 20)
    sns.despine()
    fig.tight_layout()
    save_figure(fig, "baseline_convergence.png")


def plot_compass_hyperparameter_sweep(mtf, test_sets, results, hyp_names, data_mean, data_std, device):
    log("[baseline] running COMPASS DPM hyperparameter sweep")
    reference_model = MODEL_SPECS[0]["name"]
    test_x_h1 = test_sets[reference_model]["x"]
    test_x_h1_norm = normalize_x(test_x_h1, data_mean, data_std)
    n_obs = test_x_h1.shape[0]
    permutations = [torch.randperm(n_obs) for _ in range(BASELINE_CONVERGENCE_RUNS)]

    compass_cfg_probs = {}
    for timesteps in BASELINE_SWEEP_TIMESTEPS:
        for order in BASELINE_SWEEP_ORDERS:
            log(f"[baseline] COMPASS sweep timesteps={timesteps}, order={order}")
            mtf.compare(x=test_x_h1_norm, device=device, timesteps=timesteps, method="dpm", order=order)
            probs_cfg = [torch.tensor(mtf.stats[name]["obs_probs"]) for name in hyp_names]
            compass_cfg_probs[(timesteps, order)] = torch.stack(probs_cfg, dim=1)

    x_axis = torch.arange(0, n_obs + 1)
    color_map = sns.color_palette("viridis", n_colors=len(BASELINE_SWEEP_TIMESTEPS))
    line_styles = {1: "-", 2: "--", 3: ":"}
    k = model_parameter_counts(mtf, hyp_names)

    baseline_curves = {}
    for method in ["Classifier", "True Marginal"]:
        baseline_curves[method] = cumulative_true_model_curve(
            results[method]["probs"][:n_obs],
            permutations,
            true_model_idx=0,
        )

    baseline_styles = {
        "True Marginal": {"color": "black", "linestyle": "--"},
        "Classifier": {"color": "#ff7f0e", "linestyle": "-."},
    }
    sweep_variants = {
        "No penalty": "baseline_compass_hyperparameter_sweep.png",
        "AIC": "baseline_compass_hyperparameter_sweep_aic.png",
        "AICc": "baseline_compass_hyperparameter_sweep_aicc.png",
        "BIC": "baseline_compass_hyperparameter_sweep_bic.png",
    }

    for criterion, filename in sweep_variants.items():
        log(f"[baseline] plotting COMPASS DPM hyperparameter sweep ({criterion})")
        fig, ax = plt.subplots(figsize=(13, 7), dpi=500)

        for c_idx, timesteps in enumerate(BASELINE_SWEEP_TIMESTEPS):
            for order in BASELINE_SWEEP_ORDERS:
                probs = compass_cfg_probs[(timesteps, order)][:n_obs]
                if criterion == "No penalty":
                    mean_curve, se_curve = cumulative_true_model_curve(probs, permutations, true_model_idx=0)
                else:
                    mean_curve, se_curve = cumulative_compass_ic_curve(
                        probs,
                        permutations,
                        k.to(probs.device),
                        criterion,
                        true_model_idx=0,
                    )
                ax.plot(
                    x_axis,
                    mean_curve,
                    color=color_map[c_idx],
                    linestyle=line_styles.get(order, "-"),
                    linewidth=2.2,
                    alpha=0.9,
                    label=f"COMPASS t={timesteps}",
                )
                ax.fill_between(
                    x_axis.numpy(),
                    (mean_curve - se_curve).numpy(),
                    (mean_curve + se_curve).numpy(),
                    color=color_map[c_idx],
                    alpha=0.1,
                )

        for method in ["Classifier", "True Marginal"]:
            mean_curve, se_curve = baseline_curves[method]
            style = baseline_styles[method]
            ax.plot(
                x_axis,
                mean_curve,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=3,
                label=method,
                zorder=6,
            )
            ax.fill_between(
                x_axis.numpy(),
                (mean_curve - se_curve).numpy(),
                (mean_curve + se_curve).numpy(),
                color=style["color"],
                alpha=0.08,
                zorder=5,
            )

        ax.axhline(y=0.95, color="gray", ls=":", linewidth=2, label="95% threshold")
        ax.set_xlabel("# Observations", fontsize=18)
        ax.set_ylabel(r"$P(\mathcal{M}_1 \mid x_1, \ldots, x_n)$", fontsize=18)
        ax.set_title(f"Convergence Sensitivity: DPM Settings vs Baselines ({criterion})", fontsize=20)
        ax.set_ylim(0.25, 1.05)
        ax.set_xlim(0, 40)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.legend(fontsize=14, frameon=True, ncol=2, loc="lower right")
        sns.despine()
        fig.tight_layout()
        save_figure(fig, filename)


def save_baseline_probabilities(results):
    for method, payload in results.items():
        probs = to_numpy(payload["probs"])
        labels = to_numpy(payload["true_labels"])
        rows = []
        for i in range(probs.shape[0]):
            row = {
                "index": i,
                "true_label": int(labels[i]),
            }
            for model_idx, spec in enumerate(MODEL_SPECS):
                row[f"probability_{spec['name']}"] = probs[i, model_idx]
            rows.append(row)
        df = pd.DataFrame(rows)
        path = OUTPUT_DIR / f"baseline_{method.lower().replace(' ', '_')}_probabilities.csv"
        df.to_csv(path, index=False)
        log(f"[baseline] saved {path}")


def run_baseline_tests_and_plots(
    mtf,
    hyp_names,
    data_mean,
    data_std,
    device,
    comparison_stats=None,
    comparison_stats_by_true_model=None,
):
    comparison_plot_dir = OUTPUT_DIR / "mtf_plot_comparison"
    comparison_plot_dir.mkdir(parents=True, exist_ok=True)
    if comparison_stats is not None:
        mtf.plot_comparison(stats_dict=comparison_stats, path=str(comparison_plot_dir), show=False)
    if comparison_stats_by_true_model is not None:
        for true_model_name, run_stats in comparison_stats_by_true_model.items():
            if len(run_stats) <= 1:
                continue
            filename_model_name = true_model_name.lower().replace(" ", "_")
            mtf.plot_comparison_across_runs(
                run_stats,
                path=str(comparison_plot_dir),
                show=False,
                filename=(
                    "model_probs_cumulative_comparison_across_confusion_runs_"
                    f"{filename_model_name}_true.png"
                ),
            )
    if comparison_stats is not None or comparison_stats_by_true_model is not None:
        log(f"[baseline] saved ModelTransfuser comparison plots to {comparison_plot_dir}")
    else:
        log("[baseline] skipped ModelTransfuser comparison plots; comparison run is disabled")

    results, test_sets, compass_ic_results = collect_baseline_results(mtf, hyp_names, data_mean, data_std, device)
    save_baseline_probabilities(results)
    plot_baseline_calibration(results)
    plot_compass_ic_calibration(compass_ic_results)
    plot_baseline_convergence(results)
    plot_compass_ic_convergence(results, mtf, hyp_names)
    plot_compass_hyperparameter_sweep(mtf, test_sets, results, hyp_names, data_mean, data_std, device)

def main():
    log("[setup] selecting CUDA devices")
    autocvd(num_gpus=1, interval=1)
    log(f"[setup] output directory: {OUTPUT_DIR}")
    log(f"[setup] model directory: {MODEL_DIR}")
    model_names = [spec["name"] for spec in MODEL_SPECS]
    log("[data] generating raw training and validation data from nested circle simulators")
    raw_train_data_by_model = {}
    raw_val_data_by_model = {}
    raw_mock_data_by_model = {}
    mock_sets_by_model = {}
    for spec in MODEL_SPECS:
        theta, x = spec["generator"](TRAIN_N)
        val_theta, val_x = spec["generator"](VAL_N)
        mock_sets = generate_confusion_mock_sets(spec)
        mock_theta, mock_x = concatenate_datasets(mock_sets)
        raw_train_data_by_model[spec["name"]] = (theta, x)
        raw_val_data_by_model[spec["name"]] = (val_theta, val_x)
        raw_mock_data_by_model[spec["name"]] = (mock_theta, mock_x)
        mock_sets_by_model[spec["name"]] = mock_sets

    inferred_theta_dims = {
        model_name: theta.shape[1]
        for model_name, (theta, _) in raw_train_data_by_model.items()
    }
    data_mean, data_std = compute_normalization(*raw_train_data_by_model.values())
    save_experiment_config(data_mean, data_std, inferred_theta_dims)
    log(f"[data] x normalization mean={to_numpy(data_mean).round(4).tolist()}")
    log(f"[data] x normalization std={to_numpy(data_std).round(4).tolist()}")

    norm_train_data_by_model = {
        name: normalize(theta, x, data_mean, data_std)
        for name, (theta, x) in raw_train_data_by_model.items()
    }
    norm_val_data_by_model = {
        name: normalize(theta, x, data_mean, data_std)
        for name, (theta, x) in raw_val_data_by_model.items()
    }

    log(
        "[data] generated "
        f"train={sum(len(theta) for theta, _ in raw_train_data_by_model.values())} "
        f"validation={sum(len(theta) for theta, _ in raw_val_data_by_model.values())} "
        f"mock={sum(len(theta) for theta, _ in raw_mock_data_by_model.values())} samples "
        f"({CONFUSION_N} sets/model, {CONFUSION_N_OBSERVATIONS} observations/set)"
    )

    #plot_raw_simulator_samples(raw_train_data_by_model)
    #make_pairplot(raw_val_data_by_model)
    #plot_mock_data_overview(raw_mock_data_by_model)

    log(f"[model] setting up ModelTransfuser at {MODEL_DIR}")
    mtf = MTf(path=str(MODEL_DIR))

    log("[model] adding normalized data to transfuser")
    for model_name in model_names:
        theta_n, x_n = norm_train_data_by_model[model_name]
        val_theta_n, val_x_n = norm_val_data_by_model[model_name]
        mtf.add_data(model_name, theta_n, x_n, val_theta_n, val_x_n)

    log("[model] initializing model shells")
    mtf.init_models(**MODEL_KWARGS)
    plot_information_criterion_parameter_counts(mtf, model_names)

    models_by_name = load_or_train_models(mtf, model_names, device="cuda")

    log("[model] adding loaded models to transfuser")
    for model_name, model in progress(models_by_name.items(), "[model] adding models"):
        mtf.add_model(model_name, model)
    #plot_fixed_theta_true_vs_learned(mtf, models_by_name, data_mean, data_std, device="cuda")
    #plot_likelihood_predictive_checks(models_by_name, norm_val_data_by_model, device="cuda")
    #plot_posterior_predictive_checks(models_by_name, norm_val_data_by_model, device="cuda")

    '''
    confusion_rows_ic = []
    confusion_rows_bic = []
    confusion_rows_no_penalty = []
    diagnostic_runs_saved = 0
    comparison_stats_by_true_model = {name: [] for name in model_names}
    log("[compare] creating nested circle confusion matrices with normalized mock observations")
    for true_model_name in progress(model_names, "[compare] true models", leave=True):
        true_model_probs_ic = []
        true_model_probs_bic = []
        true_model_probs_no_penalty = []
        for run_index, (theta_raw, observations_raw) in enumerate(
            progress(
                mock_sets_by_model[true_model_name],
                f"[compare] runs for {true_model_name}",
            ),
            start=1,
        ):
            observations = normalize_x(observations_raw, data_mean, data_std)
            log(
                f"[compare] true={true_model_name} "
                f"run={run_index}/{CONFUSION_N} "
                f"observations={CONFUSION_N_OBSERVATIONS} "
                f"timesteps={CONFUSION_TIMESTEPS} "
                f"samples={CONFUSION_NUM_SAMPLES} "
                f"theta_mode=independent_per_observation"
            )
            mtf.compare(
                x=observations,
                device="cuda",
                timesteps=CONFUSION_TIMESTEPS,
                num_samples=CONFUSION_NUM_SAMPLES,
                method="dpm",
                order=1,
                verbose=TRAIN_VERBOSE,
                multi_obs_inference=False,
            )
            comparison_stats_by_true_model[true_model_name].append(deepcopy(mtf.stats))
            probs_ic = torch.tensor([
                mtf.stats[model_name].get("model_prob_ic", mtf.stats[model_name]["model_prob"])
                for model_name in model_names
            ])
            probs_bic = torch.tensor([
                mtf.stats[model_name]["model_prob_bic"]
                for model_name in model_names
            ])
            probs_no_penalty = torch.tensor([
                mtf.stats[model_name]["model_prob_no_penalty"]
                for model_name in model_names
            ])
            true_model_probs_ic.append(probs_ic)
            true_model_probs_bic.append(probs_bic)
            true_model_probs_no_penalty.append(probs_no_penalty)
            predicted_model_ic = model_names[probs_ic.argmax().item()]
            predicted_model_bic = model_names[probs_bic.argmax().item()]
            predicted_model_no_penalty = model_names[probs_no_penalty.argmax().item()]
            log(
                f"[compare] predicted_ic={predicted_model_ic} "
                f"predicted_bic={predicted_model_bic} "
                f"predicted_no_penalty={predicted_model_no_penalty} "
                f"true_model_prob_ic={probs_ic[model_names.index(true_model_name)]:.3f} "
                f"true_model_prob_bic={probs_bic[model_names.index(true_model_name)]:.3f} "
                f"true_model_prob_no_penalty={probs_no_penalty[model_names.index(true_model_name)]:.3f} "
                f"probs_ic={[round(p.item(), 3) for p in probs_ic]} "
                f"probs_bic={[round(p.item(), 3) for p in probs_bic]} "
                f"probs_no_penalty={[round(p.item(), 3) for p in probs_no_penalty]}"
            )
            if (
                diagnostic_runs_saved < DIAGNOSTIC_MAX_RUNS
                and should_save_compare_diagnostic(mtf, model_names, true_model_name)
            ):
                plot_compare_diagnostic(
                    mtf=mtf,
                    model_names=model_names,
                    true_model_name=true_model_name,
                    observations=observations,
                    run_index=run_index,
                    device="cuda",
                )
                plot_wrong_prediction_map_summary(
                    mtf=mtf,
                    model_names=model_names,
                    true_model_name=true_model_name,
                    observations=observations,
                    data_mean=data_mean,
                    data_std=data_std,
                    run_index=run_index,
                )
                diagnostic_runs_saved += 1

        mean_probs_ic = torch.stack(true_model_probs_ic).mean(dim=0)
        mean_probs_bic = torch.stack(true_model_probs_bic).mean(dim=0)
        mean_probs_no_penalty = torch.stack(true_model_probs_no_penalty).mean(dim=0)
        log(
            f"[compare] finished true={true_model_name}; "
            f"mean_probs_ic={[round(p.item(), 3) for p in mean_probs_ic]} "
            f"mean_probs_bic={[round(p.item(), 3) for p in mean_probs_bic]} "
            f"mean_probs_no_penalty={[round(p.item(), 3) for p in mean_probs_no_penalty]}"
        )
        confusion_rows_ic.append(mean_probs_ic)
        confusion_rows_bic.append(mean_probs_bic)
        confusion_rows_no_penalty.append(mean_probs_no_penalty)

    confusion_probs_ic = torch.stack(confusion_rows_ic)
    confusion_probs_bic = torch.stack(confusion_rows_bic)
    confusion_probs_no_penalty = torch.stack(confusion_rows_no_penalty)

    def plot_confusion_matrix(confusion_probs, filename, title_suffix):
        log(f"[plot] creating {title_suffix} confusion matrix heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            confusion_probs.numpy(),
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=model_names,
            yticklabels=model_names,
            ax=ax,
        )
        ax.set_xlabel("Compared model")
        ax.set_ylabel("True model")
        ax.set_title(
            f"Mean model probability ({title_suffix}) over {CONFUSION_N} runs "
            f"with {CONFUSION_N_OBSERVATIONS} observations each"
        )
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        save_figure(fig, filename)

    plot_confusion_matrix(
        confusion_probs_ic,
        "gaussian_confusion_matrix.png",
        "AICc/AIC penalty",
    )
    plot_confusion_matrix(
        confusion_probs_bic,
        "gaussian_confusion_matrix_bic.png",
        "BIC penalty",
    )
    plot_confusion_matrix(
        confusion_probs_no_penalty,
        "gaussian_confusion_matrix_no_penalty.png",
        "no parameter penalty",
    )
    '''

    log("[baseline] running short Gaussian baseline notebook tests and plots")
    run_baseline_tests_and_plots(
        mtf,
        model_names,
        data_mean,
        data_std,
        device="cuda",
    )
    log(f"[done] saved nested circle diagnostics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
