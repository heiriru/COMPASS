from autocvd import autocvd
from pathlib import Path
from shutil import copyfile

import torch
import torch.nn as nn
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from compass import ScoreBasedInferenceModel as SBIm
from compass import ModelTransfuser as MTf


TUTORIAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TUTORIAL_DIR / "output" / "Gaussians_baseline_blobs"
MODEL_DIR = TUTORIAL_DIR / "data" / "gaussians_baseline_blobs"
CONFUSION_N = 10
CONFUSION_N_OBSERVATIONS = 80
CONFUSION_TIMESTEPS = 125
CONFUSION_NUM_SAMPLES = 800
PPC_N_CASES = 2
PPC_NUM_SAMPLES = 800
PPC_POSTERIOR_CASES = 2
PPC_POSTERIOR_THETA_SAMPLES = 128
PPC_X_SAMPLES_PER_THETA = 2
DIAGNOSTIC_MAX_RUNS = 2
LOW_CONFIDENCE_THRESHOLD = 0.65
TRAIN_BATCH_SIZE = 256
TRAIN_MAX_EPOCHS = 500
TRAIN_LR = 1e-3
TRAIN_EARLY_STOPPING_PATIENCE = 20
TRAIN_VERBOSE = True
EXPECTED_THETA_DIM = 1
EXPECTED_X_DIM = 2
EXPECTED_NODES_SIZE = EXPECTED_THETA_DIM + EXPECTED_X_DIM
BASELINE_N_TEST = 100
BASELINE_TRUE_MARGINAL_K = 50_000
BASELINE_CLASSIFIER_N = 100_000
BASELINE_CLASSIFIER_EPOCHS = 20
BASELINE_CONVERGENCE_RUNS = 50
BASELINE_SWEEP_TIMESTEPS = [50, 100, 200, 300, 400, 500, 1000]
BASELINE_SWEEP_ORDERS = [1]


def gen_data_hyp1(n, theta=None):
    if theta is None:
        theta = 3 * torch.randn(n)
    else:
        theta = theta.reshape(n)
    x1 = 2 * torch.sin(theta) + torch.randn(n) * 0.5
    x2 = 0.1 * theta**2 + 0.5 * torch.abs(x1) * torch.randn(n)
    return theta.unsqueeze(1), torch.stack([x1, x2], dim=1)


def gen_data_hyp2(n, theta=None):
    if theta is None:
        theta = 3 * torch.randn(n)
    else:
        theta = theta.reshape(n)
    x1 = 0.1 * theta**2 + 0.5 * torch.randn(n)
    x2 = 2 * torch.cos(theta) + torch.randn(n) * 0.5
    return theta.unsqueeze(1), torch.stack([x1, x2], dim=1)


def gen_data_hyp3(n, theta=None):
    if theta is None:
        theta = 3 * torch.randn(n)
    else:
        theta = theta.reshape(n)
    x1 = torch.randn(n)
    x2 = torch.abs(torch.randn(n)) * 2
    return theta.unsqueeze(1), torch.stack([x1, x2], dim=1)


def compute_normalization(*datasets):
    joint_data = [torch.cat([theta, x], dim=1) for theta, x in datasets]
    all_data = torch.cat(joint_data, dim=0)
    data_mean = all_data.mean(0)
    data_std = all_data.std(0)
    data_std = torch.where(data_std == 0, torch.ones_like(data_std), data_std)
    return data_mean, data_std


def normalize(theta, x, data_mean, data_std):
    joint = torch.cat([theta, x], dim=1)
    joint_norm = (joint - data_mean) / data_std
    return joint_norm[:, :EXPECTED_THETA_DIM], joint_norm[:, EXPECTED_THETA_DIM:]


def normalize_x(x, data_mean, data_std):
    return (x - data_mean[EXPECTED_THETA_DIM:]) / data_std[EXPECTED_THETA_DIM:]


def unnormalize_x(x_norm, data_mean, data_std):
    return x_norm * data_std[EXPECTED_THETA_DIM:] + data_mean[EXPECTED_THETA_DIM:]


def log(message):
    print(message, flush=True)


def progress(iterable, description, leave=False):
    return tqdm(iterable, desc=description, leave=leave, dynamic_ncols=True)


def save_figure(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"[plot] saved {path}")


def save_experiment_config(data_mean=None, data_std=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "role_model": "Gaussian_models_baselines.ipynb",
        "model_dir": str(MODEL_DIR),
        "output_dir": str(OUTPUT_DIR),
        "theta_dim": EXPECTED_THETA_DIM,
        "x_dim": EXPECTED_X_DIM,
        "theta_prior": "Normal(0, 3)",
        "confusion_n": CONFUSION_N,
        "confusion_n_observations": CONFUSION_N_OBSERVATIONS,
        "confusion_timesteps": CONFUSION_TIMESTEPS,
        "confusion_num_samples": CONFUSION_NUM_SAMPLES,
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


def make_dataframe(theta, x, model_name):
    theta_np = to_numpy(theta)
    x_np = to_numpy(x)
    data = {
        "x_1": x_np[:, 0],
        "x_2": x_np[:, 1],
        "Hypothesis": model_name,
    }
    for dim in range(theta_np.shape[1]):
        data[f"theta_{dim + 1}"] = theta_np[:, dim]
    return pd.DataFrame(data)


def make_pairplot(val_data_by_model):
    log("[plot] creating validation pairplot")
    frames = []
    for model_name, (theta, x) in progress(val_data_by_model.items(), "[plot] pairplot data"):
        frames.append(make_dataframe(theta, x, model_name))

    combined_df = pd.concat(frames, axis=0, ignore_index=True)
    pairplot = sns.pairplot(
        combined_df,
        diag_kind="kde",
        hue="Hypothesis",
        plot_kws=dict(alpha=0.5, s=3),
    )
    save_figure(pairplot.fig, "gaussian_hypotheses_pairplot.png")


def plot_mock_data_overview(data_by_model):
    log("[plot] creating mock-data overview")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
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
    fig.suptitle("Mock data generated from the three Gaussian hypotheses")
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

        replicated_x_np = to_numpy(replicated_x).reshape(PPC_POSTERIOR_CASES, -1, 2)
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


def promote_best_checkpoint(model_name):
    final_path = MODEL_DIR / f"{model_name}.pt"
    checkpoint_path = MODEL_DIR / f"{model_name}_checkpoint.pt"
    if final_path.exists():
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
        "Training missing Gaussian models now."
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
        promote_best_checkpoint(model_name)
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
        train_missing_models(mtf, missing_models, device)
    else:
        log(f"[model] found all checkpoints in {MODEL_DIR}; using presaved models")

    models_by_name = {}
    for model_name, checkpoint_path in progress(
        checkpoint_paths.items(), "[model] loading checkpoints"
    ):
        checkpoint_path = promote_best_checkpoint(model_name)
        log(f"[model] loading {checkpoint_path}")
        model = SBIm.load(str(checkpoint_path), device=device)
        if model.nodes_size != EXPECTED_NODES_SIZE:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has nodes_size={model.nodes_size}, "
                f"but this 1D-theta experiment expects nodes_size={EXPECTED_NODES_SIZE}. "
                "Delete the checkpoint or use the gaussians_baseline_blobs model directory."
            )
        models_by_name[model_name] = model

    return models_by_name



def true_log_likelihood_hyp1(x, theta):
    x1, x2 = x[:, 0], x[:, 1]
    ll_x1 = torch.distributions.Normal(2 * torch.sin(theta), 0.5).log_prob(x1)
    ll_x2 = torch.distributions.Normal(0.1 * theta**2, 0.5 * torch.abs(x1) + 1e-8).log_prob(x2)
    return ll_x1 + ll_x2


def true_log_likelihood_hyp2(x, theta):
    x1, x2 = x[:, 0], x[:, 1]
    ll_x1 = torch.distributions.Normal(0.1 * theta**2, 0.5).log_prob(x1)
    ll_x2 = torch.distributions.Normal(2 * torch.cos(theta), 0.5).log_prob(x2)
    return ll_x1 + ll_x2


def true_log_likelihood_hyp3(x, theta):
    x1, x2 = x[:, 0], x[:, 1]
    ll_x1 = torch.distributions.Normal(0.0, 1.0).log_prob(x1)
    dist_x2 = torch.distributions.HalfNormal(2.0, validate_args=False)
    ll_x2 = torch.where(
        x2 >= 0,
        dist_x2.log_prob(x2),
        torch.full_like(x2, float("-inf")),
    )
    return ll_x1 + ll_x2


def compute_true_marginal_probs(test_x, k=BASELINE_TRUE_MARGINAL_K):
    log(f"[baseline] computing true marginal probabilities with K={k}")
    theta_mc = 3 * torch.randn(k)
    ll_fns = [true_log_likelihood_hyp1, true_log_likelihood_hyp2, true_log_likelihood_hyp3]
    log_marginals = torch.zeros(test_x.shape[0], 3)
    for j, ll_fn in enumerate(ll_fns):
        for i in range(test_x.shape[0]):
            x_i = test_x[i:i + 1].expand(k, -1)
            ll = ll_fn(x_i, theta_mc)
            log_marginals[i, j] = torch.logsumexp(ll, 0) - np.log(k)
    return torch.softmax(log_marginals, dim=1)


def train_classifier():
    log("[baseline] training amortized classifier baseline")
    n = BASELINE_CLASSIFIER_N
    _, x1 = gen_data_hyp1(n)
    _, x2 = gen_data_hyp2(n)
    _, x3 = gen_data_hyp3(n)
    x_train = torch.cat([x1, x2, x3])
    y_train = torch.cat([torch.zeros(n), torch.ones(n), 2 * torch.ones(n)]).long()
    perm = torch.randperm(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    classifier = nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 3),
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


def compass_probs(mtf, test_x, hyp_names, data_mean, data_std, device, timesteps=500, order=1):
    test_x_norm = normalize_x(test_x, data_mean, data_std)
    mtf.compare(x=test_x_norm, device=device, timesteps=timesteps, method="dpm", order=order)
    probs = [torch.tensor(mtf.stats[name]["obs_probs"]) for name in hyp_names]
    return torch.stack(probs, dim=1)


def collect_baseline_results(mtf, hyp_names, data_mean, data_std, device):
    log("[baseline] generating test sets from all hypotheses")
    test_sets = {}
    for i, (name, gen_fn) in enumerate([
        ("Hypothesis 1", gen_data_hyp1),
        ("Hypothesis 2", gen_data_hyp2),
        ("Hypothesis 3", gen_data_hyp3),
    ]):
        _, test_x = gen_fn(BASELINE_N_TEST)
        test_sets[name] = {"x": test_x, "true_label": i}

    classifier = train_classifier()
    results = {
        method: {"probs": [], "true_labels": []}
        for method in ["True Marginal", "COMPASS", "Classifier"]
    }

    for name, data in test_sets.items():
        test_x = data["x"]
        true_label = data["true_label"]
        log(f"[baseline] evaluating {name} test set")

        true_p = compute_true_marginal_probs(test_x)
        results["True Marginal"]["probs"].append(true_p)
        results["True Marginal"]["true_labels"].extend([true_label] * BASELINE_N_TEST)

        clf_p = classifier_probs(classifier, test_x)
        results["Classifier"]["probs"].append(clf_p)
        results["Classifier"]["true_labels"].extend([true_label] * BASELINE_N_TEST)

        compass_p = compass_probs(mtf, test_x, hyp_names, data_mean, data_std, device)
        results["COMPASS"]["probs"].append(compass_p)
        results["COMPASS"]["true_labels"].extend([true_label] * BASELINE_N_TEST)

    for method in results:
        results[method]["probs"] = torch.cat(results[method]["probs"], dim=0)
        results[method]["true_labels"] = torch.tensor(results[method]["true_labels"])

    return results, test_sets


def plot_baseline_calibration(results):
    log("[baseline] plotting calibration comparison")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    n_bins = 10
    for ax, method in zip(axes, ["True Marginal", "COMPASS", "Classifier"]):
        probs = results[method]["probs"]
        true_labels = results[method]["true_labels"]
        all_pred_probs = []
        all_correct = []
        for cls in range(3):
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
        if method == "True Marginal":
            ax.set_ylabel("Observed Frequency")
        ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "baseline_calibration.png")


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
    test_x_h1 = test_sets["Hypothesis 1"]["x"]
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

    fig, ax = plt.subplots(figsize=(13, 7), dpi=500)
    x_axis = torch.arange(0, n_obs + 1)
    color_map = sns.color_palette("viridis", n_colors=len(BASELINE_SWEEP_TIMESTEPS))
    line_styles = {1: "-", 2: "--", 3: ":"}

    for c_idx, timesteps in enumerate(BASELINE_SWEEP_TIMESTEPS):
        for order in BASELINE_SWEEP_ORDERS:
            probs = compass_cfg_probs[(timesteps, order)][:n_obs]
            mean_curve, se_curve = cumulative_true_model_curve(probs, permutations, true_model_idx=0)
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

    baseline_styles = {
        "True Marginal": {"color": "black", "linestyle": "--"},
        "Classifier": {"color": "#ff7f0e", "linestyle": "-."},
    }
    for method in ["Classifier", "True Marginal"]:
        mean_curve, se_curve = cumulative_true_model_curve(
            results[method]["probs"][:n_obs],
            permutations,
            true_model_idx=0,
        )
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
    ax.set_title("Convergence Sensitivity: DPM Settings vs Baselines", fontsize=20)
    ax.set_ylim(0.25, 1.05)
    ax.set_xlim(0, 40)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=14, frameon=True, ncol=2, loc="lower right")
    sns.despine()
    fig.tight_layout()
    save_figure(fig, "baseline_compass_hyperparameter_sweep.png")


def save_baseline_probabilities(results):
    for method, payload in results.items():
        probs = to_numpy(payload["probs"])
        labels = to_numpy(payload["true_labels"])
        rows = []
        for i in range(probs.shape[0]):
            rows.append({
                "index": i,
                "true_label": int(labels[i]),
                "probability_Hypothesis 1": probs[i, 0],
                "probability_Hypothesis 2": probs[i, 1],
                "probability_Hypothesis 3": probs[i, 2],
            })
        df = pd.DataFrame(rows)
        path = OUTPUT_DIR / f"baseline_{method.lower().replace(' ', '_')}_probabilities.csv"
        df.to_csv(path, index=False)
        log(f"[baseline] saved {path}")


def run_baseline_tests_and_plots(mtf, hyp_names, data_mean, data_std, device):
    comparison_plot_dir = OUTPUT_DIR / "mtf_plot_comparison"
    comparison_plot_dir.mkdir(parents=True, exist_ok=True)
    mtf.plot_comparison(path=str(comparison_plot_dir), show=False)
    log(f"[baseline] saved ModelTransfuser comparison plots to {comparison_plot_dir}")

    results, test_sets = collect_baseline_results(mtf, hyp_names, data_mean, data_std, device)
    save_baseline_probabilities(results)
    plot_baseline_calibration(results)
    plot_baseline_convergence(results)
    plot_compass_hyperparameter_sweep(mtf, test_sets, results, hyp_names, data_mean, data_std, device)

def main():
    log("[setup] selecting CUDA devices")
    autocvd(num_gpus=1, interval=1)
    log(f"[setup] output directory: {OUTPUT_DIR}")
    log(f"[setup] model directory: {MODEL_DIR}")
    log(f"[setup] expected nodes_size={EXPECTED_NODES_SIZE} (theta={EXPECTED_THETA_DIM}, x={EXPECTED_X_DIM})")

    log("[data] generating raw training and validation data from baseline blob simulators")
    theta1, x1 = gen_data_hyp1(100_000)
    val_theta1, val_x1 = gen_data_hyp1(1_000)

    theta2, x2 = gen_data_hyp2(100_000)
    val_theta2, val_x2 = gen_data_hyp2(1_000)

    theta3, x3 = gen_data_hyp3(100_000)
    val_theta3, val_x3 = gen_data_hyp3(1_000)

    data_mean, data_std = compute_normalization((theta1, x1), (theta2, x2), (theta3, x3))
    save_experiment_config(data_mean, data_std)
    log(f"[data] normalization mean={to_numpy(data_mean).round(4).tolist()}")
    log(f"[data] normalization std={to_numpy(data_std).round(4).tolist()}")

    theta1_n, x1_n = normalize(theta1, x1, data_mean, data_std)
    theta2_n, x2_n = normalize(theta2, x2, data_mean, data_std)
    theta3_n, x3_n = normalize(theta3, x3, data_mean, data_std)
    val_theta1_n, val_x1_n = normalize(val_theta1, val_x1, data_mean, data_std)
    val_theta2_n, val_x2_n = normalize(val_theta2, val_x2, data_mean, data_std)
    val_theta3_n, val_x3_n = normalize(val_theta3, val_x3, data_mean, data_std)

    log(
        "[data] generated "
        f"train={len(theta1) + len(theta2) + len(theta3)} "
        f"validation={len(val_theta1) + len(val_theta2) + len(val_theta3)} samples"
    )

    model_names = ["Hypothesis 1", "Hypothesis 2", "Hypothesis 3"]
    raw_val_data_by_model = {
        "Hypothesis 1": (val_theta1, val_x1),
        "Hypothesis 2": (val_theta2, val_x2),
        "Hypothesis 3": (val_theta3, val_x3),
    }
    norm_val_data_by_model = {
        "Hypothesis 1": (val_theta1_n, val_x1_n),
        "Hypothesis 2": (val_theta2_n, val_x2_n),
        "Hypothesis 3": (val_theta3_n, val_x3_n),
    }

    make_pairplot(raw_val_data_by_model)
    plot_mock_data_overview(raw_val_data_by_model)

    log(f"[model] setting up ModelTransfuser at {MODEL_DIR}")
    mtf = MTf(path=str(MODEL_DIR))

    log("[model] adding normalized data to transfuser")
    mtf.add_data("Hypothesis 1", theta1_n, x1_n, val_theta1_n, val_x1_n)
    mtf.add_data("Hypothesis 2", theta2_n, x2_n, val_theta2_n, val_x2_n)
    mtf.add_data("Hypothesis 3", theta3_n, x3_n, val_theta3_n, val_x3_n)

    log("[model] initializing model shells")
    mtf.init_models(
        sde_type="vesde",
        sigma=3,
        hidden_size=20,
        depth=4,
        num_heads=5,
        mlp_ratio=4,
    )

    models_by_name = load_or_train_models(mtf, model_names, device="cuda")

    log("[model] adding loaded models to transfuser")
    for model_name, model in progress(models_by_name.items(), "[model] adding models"):
        mtf.add_model(model_name, model)
    plot_likelihood_predictive_checks(models_by_name, norm_val_data_by_model, device="cuda")
    plot_posterior_predictive_checks(models_by_name, norm_val_data_by_model, device="cuda")

    model_generators = {
        "Hypothesis 1": gen_data_hyp1,
        "Hypothesis 2": gen_data_hyp2,
        "Hypothesis 3": gen_data_hyp3,
    }
    confusion_rows = []
    diagnostic_runs_saved = 0
    log("[compare] creating Gaussian confusion matrix with normalized mock observations")
    for true_model_name in progress(model_names, "[compare] true models", leave=True):
        true_model_probs = []
        for run_index in progress(
            range(1, CONFUSION_N + 1),
            f"[compare] runs for {true_model_name}",
        ):
            _, observations_raw = model_generators[true_model_name](CONFUSION_N_OBSERVATIONS)
            observations = normalize_x(observations_raw, data_mean, data_std)
            log(
                f"[compare] true={true_model_name} "
                f"run={run_index}/{CONFUSION_N} "
                f"observations={CONFUSION_N_OBSERVATIONS} "
                f"timesteps={CONFUSION_TIMESTEPS} "
                f"samples={CONFUSION_NUM_SAMPLES}"
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
            probs = torch.tensor([
                mtf.stats[model_name]["model_prob"]
                for model_name in model_names
            ])
            true_model_probs.append(probs)
            predicted_model = model_names[probs.argmax().item()]
            log(
                f"[compare] predicted={predicted_model} "
                f"true_model_prob={probs[model_names.index(true_model_name)]:.3f} "
                f"probs={[round(p.item(), 3) for p in probs]}"
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
                diagnostic_runs_saved += 1

        mean_probs = torch.stack(true_model_probs).mean(dim=0)
        log(
            f"[compare] finished true={true_model_name}; "
            f"mean_probs={[round(p.item(), 3) for p in mean_probs]}"
        )
        confusion_rows.append(mean_probs)

    confusion_probs = torch.stack(confusion_rows)
    log("[plot] creating confusion matrix heatmap")

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
        f"Mean model probability over {CONFUSION_N} runs "
        f"with {CONFUSION_N_OBSERVATIONS} observations each"
    )
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_figure(fig, "gaussian_confusion_matrix.png")

    log("[baseline] running Gaussian baseline notebook tests and plots")
    run_baseline_tests_and_plots(mtf, model_names, data_mean, data_std, device="cuda")
    log(f"[done] saved Gaussian diagnostics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()