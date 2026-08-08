#!/usr/bin/env python3
"""Divergence-head validation for line/parabola and banana tutorials."""

from __future__ import annotations
import os

CPU_THREAD_LIMIT = 3
CPU_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def configure_cpu_usage_limit(max_threads=CPU_THREAD_LIMIT):
    logical = os.cpu_count() or 1
    limit = min(int(max_threads), logical)
    if limit < 1:
        raise RuntimeError(f"Cannot enforce a {max_threads}-thread CPU limit on {logical} CPUs.")
    selected = tuple(sorted(os.sched_getaffinity(0))[:limit])
    if not selected:
        raise RuntimeError("The process has no CPUs available.")
    os.sched_setaffinity(0, selected)
    for name in CPU_THREAD_ENV_VARS:
        os.environ[name] = str(len(selected))
    return logical, selected


CPU_LIMIT_INFO = configure_cpu_usage_limit() if __name__ == "__main__" else None

import argparse
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path

from autocvd import autocvd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import seaborn as sns
import torch

import Divergence_Head as dh
from compass import ScoreBasedInferenceModel as SBIm


TUTORIAL_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = TUTORIAL_DIR / "output" / "divergence_head"
DATA_ROOT = TUTORIAL_DIR / "data" / "divergence_head"
LINE_PARABOLA_ROOT = OUTPUT_ROOT / "line_parabola"
BANANA_OUTPUT = OUTPUT_ROOT / "banana"
LINE_SOURCE_DATA = TUTORIAL_DIR / "data" / "PF_ODE_validation"
COLORS = dh.COLORS
TIMESTEP_SWEEP = (10, 50, 100, 200, 500)
LIKELIHOOD_CONVERGENCE_STEPS = (100, 200, 500)

BASE_CONFIG = copy.deepcopy(dh.DEFAULT_CONFIG)
BASE_CONFIG.update({
    "n_train": 100_000, "n_validation": 1_000, "n_test": 2_000,
    "stage1_max_epochs": 300, "stage2_max_epochs": 200,
    "early_stopping_patience": 75, "drift_time_levels": 18,
    "drift_samples": 96, "log_prob_samples": 192,
    "log_prob_timesteps": 100, "density_grid_size": 35,
    "density_timestep_sweep": TIMESTEP_SWEEP,
    "likelihood_convergence_steps": LIKELIHOOD_CONVERGENCE_STEPS,
    "likelihood_convergence_samples": 512,
    "pairplot_samples": 2_000, "runtime_samples": 96,
    "runtime_repetitions": 3,
})


def log(message):
    print(message, flush=True)

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)



def normal_logpdf(residual, variance):
    residual = np.asarray(residual, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    return -0.5 * np.sum(np.log(2 * np.pi * variance) + residual**2 / variance, axis=-1)


class SimulatorReference:
    """Analytic likelihood and quadrature posterior in model coordinates."""

    def __init__(self, family, theta_mean, theta_std, x_mean, x_std, noise_std, quadrature_points):
        self.family = family
        self.theta_mean = np.asarray(theta_mean, dtype=np.float64)
        self.theta_std = np.asarray(theta_std, dtype=np.float64)
        self.x_mean = np.asarray(x_mean, dtype=np.float64)
        self.x_std = np.asarray(x_std, dtype=np.float64)
        self.noise_std = float(noise_std)
        self.theta_dim = len(self.theta_mean)
        self.x_dim = len(self.x_mean)
        self.nodes_size = self.theta_dim + self.x_dim
        self.posterior_mask = np.concatenate([np.zeros(self.theta_dim), np.ones(self.x_dim)]).astype(np.float32)
        self.likelihood_mask = 1.0 - self.posterior_mask
        self.masks = {"posterior": self.posterior_mask, "likelihood": self.likelihood_mask}
        self.labels = [f"θ{i + 1}" for i in range(self.theta_dim)] + [f"x{i + 1}" for i in range(self.x_dim)]
        self.joint_mean = np.concatenate([self.theta_mean, self.x_mean])
        self.joint_std = np.concatenate([self.theta_std, self.x_std])
        self._build_quadrature(int(quadrature_points))

    @classmethod
    def curve(cls, family, x_mean, x_std):
        return cls(family, np.zeros(1), np.ones(1), x_mean, x_std, 0.1, 2501)

    @classmethod
    def banana(cls, theta_mean, theta_std, x_mean, x_std):
        return cls("banana", theta_mean, theta_std, x_mean, x_std, 0.3, 221)

    def _build_quadrature(self, points):
        if self.family in ("line", "parabola"):
            axis = np.linspace(-2.0, 2.0, points)
            weights = np.full(points, 4.0 / (points - 1))
            weights[[0, -1]] *= 0.5
            self.theta_quad_raw = axis[:, None]
            self.log_prior_weights = np.log(0.25 * weights)
        else:
            axis = np.linspace(-4.5, 4.5, points)
            t1, t2 = np.meshgrid(axis, axis, indexing="xy")
            self.theta_quad_raw = np.column_stack([t1.ravel(), t2.ravel()])
            step = axis[1] - axis[0]
            trap = np.ones((points, points))
            trap[[0, -1], :] *= 0.5
            trap[:, [0, -1]] *= 0.5
            self.log_prior_weights = normal_logpdf(self.theta_quad_raw, np.ones(self.theta_dim)) + 2 * np.log(step) + np.log(trap.ravel())
        self.theta_quad_model = (self.theta_quad_raw - self.theta_mean) / self.theta_std
        self.quad_simulator_mean = self.simulator_mean(self.theta_quad_raw)

    def simulator_mean(self, theta_raw):
        theta_raw = np.atleast_2d(np.asarray(theta_raw, dtype=np.float64))
        if self.family == "line":
            t = theta_raw[:, 0]
            return np.column_stack([t, 0.5 * t])
        if self.family == "parabola":
            t = theta_raw[:, 0]
            return np.column_stack([t, t**2])
        t1, t2 = theta_raw[:, 0], theta_raw[:, 1]
        return np.column_stack([t1, t1**2 + t2])

    def to_model(self, theta_raw, x_raw):
        return ((np.asarray(theta_raw) - self.theta_mean) / self.theta_std,
                (np.asarray(x_raw) - self.x_mean) / self.x_std)

    def to_raw_joint(self, rows):
        rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
        return rows * self.joint_std + self.joint_mean

    def sample_raw(self, n, seed):
        rng = np.random.default_rng(seed)
        theta = (rng.uniform(-2.0, 2.0, size=(n, 1))
                 if self.family in ("line", "parabola")
                 else rng.standard_normal((n, 2)))
        x = self.simulator_mean(theta) + self.noise_std * rng.standard_normal((n, self.x_dim))
        return theta.astype(np.float32), x.astype(np.float32)

    def sample(self, n, seed):
        theta, x = self.sample_raw(n, seed)
        theta, x = self.to_model(theta, x)
        return theta.astype(np.float32), x.astype(np.float32)

    def _is_posterior(self, mask):
        return np.allclose(np.asarray(mask), self.posterior_mask)

    def _posterior_quantities(self, rows, sigma, need_score):
        rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
        raw = self.to_raw_joint(rows)
        z = rows[:, :self.theta_dim]
        x_raw = raw[:, self.theta_dim:]
        sigma2 = float(sigma)**2
        if sigma2 <= 0:
            raise ValueError("Posterior diffusion scale must be positive.")
        log_norm = -0.5 * self.theta_dim * np.log(2 * np.pi * sigma2)
        log_q = np.empty(len(rows))
        scores = np.empty((len(rows), self.theta_dim)) if need_score else None
        divergences = np.empty(len(rows)) if need_score else None
        for index, (z_row, x_row) in enumerate(zip(z, x_raw)):
            log_joint = self.log_prior_weights + normal_logpdf(
                x_row - self.quad_simulator_mean, self.noise_std**2
            )
            log_evidence = logsumexp(log_joint)
            delta = self.theta_quad_model - z_row
            log_terms = log_joint + log_norm - 0.5 * np.sum(delta**2, axis=1) / sigma2
            log_mixture = logsumexp(log_terms)
            log_q[index] = log_mixture - log_evidence
            if need_score:
                responsibility = np.exp(log_terms - log_mixture)
                mean_component = responsibility @ self.theta_quad_model
                scores[index] = (mean_component - z_row) / sigma2
                second = responsibility @ np.sum(self.theta_quad_model**2, axis=1)
                divergences[index] = -self.theta_dim / sigma2 + (
                    second - np.sum(mean_component**2)
                ) / sigma2**2
        return log_q, scores, divergences

    def _likelihood_quantities(self, rows, sigma):
        rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
        raw = self.to_raw_joint(rows)
        theta_raw = raw[:, :self.theta_dim]
        x_model = rows[:, self.theta_dim:]
        mean_model = (self.simulator_mean(theta_raw) - self.x_mean) / self.x_std
        variance = (self.noise_std / self.x_std)**2 + float(sigma)**2
        residual = x_model - mean_model
        return (
            normal_logpdf(residual, variance),
            -residual / variance,
            np.full(len(rows), -np.sum(1.0 / variance)),
        )

    def conditional_log_prob(self, rows, mask, smoothing_sigma=0.0):
        if self._is_posterior(mask):
            return self._posterior_quantities(rows, max(float(smoothing_sigma), 1e-7), False)[0]
        return self._likelihood_quantities(rows, smoothing_sigma)[0]

    def diffused_score(self, rows, mask, sigma_t):
        rows = np.atleast_2d(rows)
        result = np.zeros_like(rows, dtype=np.float64)
        if self._is_posterior(mask):
            result[:, :self.theta_dim] = self._posterior_quantities(rows, sigma_t, True)[1]
        else:
            result[:, self.theta_dim:] = self._likelihood_quantities(rows, sigma_t)[1]
        return result

    def instantaneous_drift(self, rows, mask, sigma_t):
        divergence = (self._posterior_quantities(rows, sigma_t, True)[2]
                      if self._is_posterior(mask)
                      else self._likelihood_quantities(rows, sigma_t)[2])
        return float(sigma_t) * divergence

    def raw_log_density_shift(self, mask):
        latent = np.flatnonzero(1.0 - np.asarray(mask))
        return -float(np.log(self.joint_std[latent]).sum())

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_curve_reference(family):
    with (LINE_SOURCE_DATA / "normalization.json").open() as handle:
        norm = json.load(handle)
    return SimulatorReference.curve(family, norm["data_mean"], norm["data_std"])


def load_curve_splits(family, config):
    """Generate every curve split and normalize it with training-split statistics."""
    provisional = SimulatorReference(
        family, np.zeros(1), np.ones(1), np.zeros(2), np.ones(2), 0.1, 2501
    )
    raw = {
        "train": provisional.sample_raw(config["n_train"], config["seed"]),
        "validation": provisional.sample_raw(config["n_validation"], config["seed"] + 1),
        "test": provisional.sample_raw(config["n_test"], config["seed"] + 2),
    }
    theta_mean = raw["train"][0].mean(axis=0)
    theta_std = raw["train"][0].std(axis=0)
    x_mean = raw["train"][1].mean(axis=0)
    x_std = raw["train"][1].std(axis=0)
    if np.any(theta_std <= 0) or np.any(x_std <= 0):
        raise ValueError("Training data must have non-zero variance for normalization.")
    reference = SimulatorReference(
        family, theta_mean, theta_std, x_mean, x_std, 0.1, 2501
    )
    splits = {}
    for split, (theta_raw, x_raw) in raw.items():
        theta, x = reference.to_model(theta_raw, x_raw)
        splits[split] = (theta.astype(np.float32), x.astype(np.float32))
    return reference, splits


def banana_splits(config, cache_dir):
    data_path = cache_dir / "splits.npz"
    norm_path = cache_dir / "normalization.json"
    if data_path.exists() and norm_path.exists():
        payload = np.load(data_path)
        if (
            len(payload["train_theta"]) == config["n_train"]
            and len(payload["validation_theta"]) == config["n_validation"]
            and len(payload["test_theta"]) == config["n_test"]
        ):
            with norm_path.open() as handle:
                norm = json.load(handle)
            reference = SimulatorReference.banana(
                norm["theta_mean"], norm["theta_std"], norm["x_mean"], norm["x_std"]
            )
            splits = {
                split: (
                    payload[f"{split}_theta"].astype(np.float32),
                    payload[f"{split}_x"].astype(np.float32),
                )
                for split in ("train", "validation", "test")
            }
            return reference, splits

    cache_dir.mkdir(parents=True, exist_ok=True)
    provisional = SimulatorReference.banana(np.zeros(2), np.ones(2), np.zeros(2), np.ones(2))
    raw = {
        "train": provisional.sample_raw(config["n_train"], config["seed"]),
        "validation": provisional.sample_raw(config["n_validation"], config["seed"] + 1),
        "test": provisional.sample_raw(config["n_test"], config["seed"] + 2),
    }
    theta_mean = raw["train"][0].mean(axis=0)
    theta_std = raw["train"][0].std(axis=0)
    x_mean = raw["train"][1].mean(axis=0)
    x_std = raw["train"][1].std(axis=0)
    reference = SimulatorReference.banana(theta_mean, theta_std, x_mean, x_std)
    splits = {}
    payload = {}
    for split, (theta_raw, x_raw) in raw.items():
        theta, x = reference.to_model(theta_raw, x_raw)
        splits[split] = (theta.astype(np.float32), x.astype(np.float32))
        payload[f"{split}_theta"] = splits[split][0]
        payload[f"{split}_x"] = splits[split][1]
    np.savez(data_path, **payload)
    with norm_path.open("w") as handle:
        json.dump({
            "theta_mean": theta_mean.tolist(), "theta_std": theta_std.tolist(),
            "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        }, handle, indent=2)
    return reference, splits


def train_stage(model, splits, config, cache_dir, name, device, divergence):
    log(f"[train] {cache_dir.name}: {name}")
    model.train(
        theta=torch.from_numpy(splits["train"][0]),
        x=torch.from_numpy(splits["train"][1]),
        theta_val=torch.from_numpy(splits["validation"][0]),
        x_val=torch.from_numpy(splits["validation"][1]),
        batch_size=config["batch_size"],
        max_epochs=config["stage2_max_epochs"] if divergence else config["stage1_max_epochs"],
        lr=config["learning_rate"],
        device=device,
        verbose=False,
        path=str(cache_dir),
        name=name,
        early_stopping_patience=config["early_stopping_patience"],
        time_sampling=config["time_sampling"],
        train_divergence=divergence,
        divergence_loss_weight=config["divergence_loss_weight"],
        divergence_target=config["divergence_target"],
        hutchinson_samples=config["hutchinson_samples"],
        divergence_warmup_epochs=config["divergence_warmup_epochs"],
    )
    checkpoint = cache_dir / f"{name}_checkpoint.pt"
    final = cache_dir / f"{name}.pt"
    if checkpoint.exists():
        shutil.copy2(checkpoint, final)
    elif not final.exists():
        model.save(path=str(cache_dir), name=name)
    return dh.history_rows(model.trainer, name)


def load_history(path):
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checked_checkpoint(path, device, expect_head, role):
    model = SBIm.load(str(path), device=device)
    if bool(model.divergence_head_trained) != bool(expect_head):
        raise RuntimeError(
            f"{role} checkpoint metadata mismatch at {path}: expected "
            f"divergence_head_trained={bool(expect_head)}, found "
            f"{bool(model.divergence_head_trained)}."
        )
    return model


def promote_cached_checkpoint(final_path, checkpoint_path, device, expect_head, role):
    if final_path.exists() or not checkpoint_path.exists():
        return
    load_checked_checkpoint(checkpoint_path, device, expect_head, role)
    shutil.copy2(checkpoint_path, final_path)
    log(f"[cache] promoted {checkpoint_path.name} to {final_path.name}")


def preserve_no_head_checkpoint(stage1_path, baseline_path, device):
    source_digest = file_sha256(stage1_path)
    if not baseline_path.exists() or file_sha256(baseline_path) != source_digest:
        shutil.copy2(stage1_path, baseline_path)
        log(f"[cache] preserved untouched score-only model as {baseline_path.name}")
    if file_sha256(baseline_path) != source_digest:
        raise RuntimeError("The no-head checkpoint does not match the Stage-1 checkpoint.")
    return load_checked_checkpoint(
        baseline_path, device, expect_head=False, role="Exact no-head baseline"
    )


def load_or_train_models(problem, family, reference, splits, config, device, force_retrain):
    cache_dir = DATA_ROOT / problem / family
    cache_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = cache_dir / "stage1_score.pt"
    baseline_path = cache_dir / "stage1_exact_no_head.pt"
    stage2_path = cache_dir / "stage2_joint.pt"
    stage1_checkpoint = cache_dir / "stage1_score_checkpoint.pt"
    stage2_checkpoint = cache_dir / "stage2_joint_checkpoint.pt"
    history_path = cache_dir / "training_history.csv"
    history = []
    if force_retrain:
        for path in (
            stage1_path,
            baseline_path,
            stage2_path,
            stage1_checkpoint,
            stage2_checkpoint,
        ):
            path.unlink(missing_ok=True)

    promote_cached_checkpoint(
        stage1_path, stage1_checkpoint, device, expect_head=False, role="Stage-1"
    )
    if stage1_path.exists():
        stage1 = load_checked_checkpoint(
            stage1_path, device, expect_head=False, role="Stage-1"
        )
    else:
        seed_all(config["seed"])
        stage1 = SBIm(
            nodes_size=reference.nodes_size, sde_type="vesde", sigma=25.0,
            hidden_size=64, depth=3, num_heads=4, mlp_ratio=2, device=device,
        )
        history.extend(train_stage(
            stage1, splits, config, cache_dir, "stage1_score", device, False
        ))
        stage1 = load_checked_checkpoint(
            stage1_path, device, expect_head=False, role="Stage-1"
        )

    no_head_model = preserve_no_head_checkpoint(stage1_path, baseline_path, device)

    promote_cached_checkpoint(
        stage2_path, stage2_checkpoint, device, expect_head=True, role="Stage-2"
    )
    if stage2_path.exists():
        stage2 = load_checked_checkpoint(
            stage2_path, device, expect_head=True, role="Stage-2"
        )
    else:
        stage2 = SBIm.load(str(stage1_path), device=device)
        history.extend(train_stage(
            stage2, splits, config, cache_dir, "stage2_joint", device, True
        ))
        stage2 = load_checked_checkpoint(
            stage2_path, device, expect_head=True, role="Stage-2"
        )

    if not history:
        history = load_history(history_path)
    else:
        write_csv(history_path, history)
    return no_head_model, stage2, history, cache_dir, baseline_path


def joint_rows(splits):
    return np.concatenate(splits["test"], axis=1).astype(np.float32)


def plot_pairplot(reference, splits, output_dir, title):
    rows = joint_rows(splits)
    rows = rows[:min(len(rows), BASE_CONFIG["pairplot_samples"])]
    raw = reference.to_raw_joint(rows)
    frame = pd.DataFrame(raw, columns=reference.labels)
    grid = sns.PairGrid(frame, diag_sharey=False, height=2.35)
    grid.map_lower(sns.scatterplot, color=COLORS["stage1"], s=9, alpha=0.20, linewidth=0)
    grid.map_diag(sns.histplot, bins=30, color=COLORS["stage1"], edgecolor="white")
    grid.map_upper(sns.scatterplot, color=COLORS["stage1"], s=7, alpha=0.12, linewidth=0)
    grid.fig.suptitle(title, y=0.99)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.fig.savefig(output_dir / "joint_data_pairplot.png", dpi=220, bbox_inches="tight")
    plt.close(grid.fig)


def plot_combined_curve_pairplot():
    frames = []
    for family in ("line", "parabola"):
        reference = load_curve_reference(family)
        theta, x = reference.sample_raw(1200, BASE_CONFIG["seed"] + (family == "parabola"))
        frame = pd.DataFrame(np.concatenate([theta, x], axis=1), columns=["θ", "x1", "x2"])
        frame["model"] = family.capitalize()
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    grid = sns.pairplot(data, vars=["θ", "x1", "x2"], hue="model", diag_kind="hist",
                        plot_kws={"alpha": 0.25, "s": 7})
    grid.fig.suptitle("Line versus parabola simulator", y=1.02)
    LINE_PARABOLA_ROOT.mkdir(parents=True, exist_ok=True)
    grid.fig.savefig(LINE_PARABOLA_ROOT / "joint_data_pairplot.png", dpi=220, bbox_inches="tight")
    plt.close(grid.fig)

def fixed_raw_joint(reference):
    if reference.family == "banana":
        theta = np.array([[0.0, 1.8]])
        x = np.array([[0.0, 2.0]])
    else:
        theta = np.array([[0.75]])
        x = reference.simulator_mean(theta)
    theta_model, x_model = reference.to_model(theta, x)
    return np.concatenate([theta_model, x_model], axis=1)[0]


def density_grid(reference, mask, grid_size):
    fixed = fixed_raw_joint(reference)
    fixed_raw = reference.to_raw_joint(fixed)[0]
    latent = np.flatnonzero(1.0 - np.asarray(mask))
    if len(latent) == 1:
        axis = np.linspace(-2.0, 2.0, grid_size * 5 + 1)
        raw = np.repeat(fixed_raw[None, :], len(axis), axis=0)
        raw[:, latent[0]] = axis
        model = (raw - reference.joint_mean) / reference.joint_std
        return {"dimension": 1, "axis0": axis, "joint": model.astype(np.float32),
                "shape": axis.shape, "latent": latent}

    if reference._is_posterior(mask):
        if reference.family == "banana":
            axis0 = np.linspace(-2.0, 2.0, grid_size)
            axis1 = np.linspace(-1.5, 3.5, grid_size)
        else:
            axis0 = np.linspace(-2.0, 2.0, grid_size)
            axis1 = np.linspace(-2.0, 2.0, grid_size)
    else:
        theta_raw = fixed_raw[:reference.theta_dim][None, :]
        mean = reference.simulator_mean(theta_raw)[0]
        raw_sigma = np.sqrt(reference.noise_std**2 + (0.04 * reference.x_std)**2)
        axis0 = np.linspace(mean[0] - 4 * raw_sigma[0], mean[0] + 4 * raw_sigma[0], grid_size)
        axis1 = np.linspace(mean[1] - 4 * raw_sigma[1], mean[1] + 4 * raw_sigma[1], grid_size)

    grid0, grid1 = np.meshgrid(axis0, axis1, indexing="xy")
    raw = np.repeat(fixed_raw[None, :], grid_size * grid_size, axis=0)
    raw[:, latent[0]] = grid0.ravel()
    raw[:, latent[1]] = grid1.ravel()
    model = (raw - reference.joint_mean) / reference.joint_std
    return {"dimension": 2, "axis0": grid0, "axis1": grid1,
            "joint": model.astype(np.float32), "shape": grid0.shape,
            "latent": latent}


def model_log_prob(model, joint, mask, timesteps, divergence, config, device):
    return dh.to_numpy(model.log_prob(
        torch.from_numpy(joint), torch.tensor(mask, dtype=torch.float32),
        timesteps=int(timesteps), eps=config["eps"], divergence=divergence,
        device=device, batch_size=config["log_prob_batch_size"], verbose=False,
    ))


def evaluate_likelihood_convergence(reference, splits, stage2, config, device):
    steps = tuple(config["likelihood_convergence_steps"])
    if steps != LIKELIHOOD_CONVERGENCE_STEPS:
        raise ValueError("Likelihood convergence must use 100, 200, and 500 integration steps.")
    joint = joint_rows(splits)[:config["likelihood_convergence_samples"]]
    mask = reference.masks["likelihood"]
    values = {}
    for step in steps:
        log(f"[likelihood convergence] {reference.family}: {step} integration steps")
        values[int(step)] = model_log_prob(
            stage2, joint, mask, step, "exact", config, device
        )
    return values


def plot_likelihood_convergence(values):
    steps = sorted(values)
    reference = values[steps[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for step, color in zip(steps, (COLORS["stage1"], COLORS["stage2"], COLORS["head_exact"])):
        axes[0].scatter(reference, values[step], s=8, alpha=0.35, color=color, label=f"{step} steps")
    lower = min(float(values[step].min()) for step in steps)
    upper = max(float(values[step].max()) for step in steps)
    axes[0].plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axes[0].set(xlabel="Exact log likelihood at 500 integration steps", ylabel="Exact log likelihood", title="Exact likelihood convergence")
    axes[0].legend()
    errors = [values[step] - reference for step in steps]
    axes[1].boxplot(errors, tick_labels=[str(step) for step in steps], showfliers=False)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Integration steps", ylabel="Difference from 500-step log likelihood", title="Per-sample convergence error")
    dh.save_figure(fig, "exact_likelihood_integration_convergence.png")

def evaluate_density(reference, no_head_model, stage2, config, device):
    smoothing = dh.analytic_epsilon_sigma(stage2, config["eps"])
    results = {}
    for mask_name, mask in reference.masks.items():
        log(f"[density] {reference.family}: {mask_name}")
        grid = density_grid(reference, mask, config["density_grid_size"])
        shift = reference.raw_log_density_shift(mask)
        analytic = reference.conditional_log_prob(grid["joint"], mask, smoothing) + shift
        no_head_exact = model_log_prob(
            no_head_model, grid["joint"], mask, config["density_timesteps"],
            "exact", config, device,
        ) + shift
        trained_exact = model_log_prob(
            stage2, grid["joint"], mask, config["density_timesteps"],
            "exact", config, device,
        ) + shift
        trained_learned = model_log_prob(
            stage2, grid["joint"], mask, config["density_timesteps"],
            "learned", config, device,
        ) + shift
        for name, values in (
            ("analytic", analytic),
            ("no-head exact", no_head_exact),
            ("trained exact", trained_exact),
            ("trained learned", trained_learned),
        ):
            dh.finite_or_fail(f"{reference.family} {mask_name} {name} density", values)
        result = dict(grid)
        result.update({
            "analytic_log_prob": analytic.reshape(grid["shape"]),
            "no_head_exact_log_prob": no_head_exact.reshape(grid["shape"]),
            "trained_exact_log_prob": trained_exact.reshape(grid["shape"]),
            "trained_learned_log_prob": trained_learned.reshape(grid["shape"]),
        })
        results[mask_name] = result
    return results


def evaluate_density_sweep(reference, no_head_model, stage2, config, device):
    smoothing = dh.analytic_epsilon_sigma(stage2, config["eps"])
    results = {}
    for mask_name, mask in reference.masks.items():
        grid = density_grid(reference, mask, config["density_grid_size"])
        shift = reference.raw_log_density_shift(mask)
        analytic = reference.conditional_log_prob(grid["joint"], mask, smoothing) + shift
        values = {}
        for timesteps in config["density_timestep_sweep"]:
            log(f"[density sweep] {reference.family}: {mask_name}, {timesteps} integration steps")
            no_head_exact = model_log_prob(
                no_head_model, grid["joint"], mask, timesteps,
                "exact", config, device,
            ) + shift
            trained_exact = model_log_prob(
                stage2, grid["joint"], mask, timesteps,
                "exact", config, device,
            ) + shift
            trained_learned = model_log_prob(
                stage2, grid["joint"], mask, timesteps,
                "learned", config, device,
            ) + shift
            for name, method_values in (
                ("no-head exact", no_head_exact),
                ("trained exact", trained_exact),
                ("trained learned", trained_learned),
            ):
                dh.finite_or_fail(
                    f"{reference.family} {mask_name} {timesteps}-step {name} density",
                    method_values,
                )
            values[int(timesteps)] = {
                "no_head_exact_log_prob": no_head_exact.reshape(grid["shape"]),
                "trained_exact_log_prob": trained_exact.reshape(grid["shape"]),
                "trained_learned_log_prob": trained_learned.reshape(grid["shape"]),
            }
        result = dict(grid)
        result.update({
            "analytic_log_prob": analytic.reshape(grid["shape"]),
            "timestep_results": values,
        })
        results[mask_name] = result
    return results


def plot_density_result(reference, mask_name, result):
    analytic = result["analytic_log_prob"]
    no_head_exact = result["no_head_exact_log_prob"]
    trained_exact = result["trained_exact_log_prob"]
    trained_learned = result["trained_learned_log_prob"]
    if result["dimension"] == 1:
        axis = result["axis0"]
        fig, density_axis = plt.subplots(figsize=(9, 4.8))
        density_axis.plot(axis, np.exp(analytic), "k--", label="Analytic smoothed")
        density_axis.plot(
            axis, np.exp(no_head_exact), color=COLORS["stage1"],
            label="Exact (no head trained)",
        )
        density_axis.plot(
            axis, np.exp(trained_exact), color=COLORS["stage2"],
            label="Exact (head trained)",
        )
        density_axis.plot(
            axis, np.exp(trained_learned), color=COLORS["head_exact"],
            label="Learned divergence head",
        )
        density_axis.set(
            xlabel=reference.labels[result["latent"][0]],
            ylabel="Density",
        )
        density_axis.legend()
        fig.suptitle(f"{reference.family.capitalize()} {mask_name} density validation")
    else:
        grid0, grid1 = result["axis0"], result["axis1"]
        densities = [
            np.exp(np.clip(values, -80, 50))
            for values in (analytic, no_head_exact, trained_exact, trained_learned)
        ]
        density_upper = max(max(float(values.max()) for values in densities), 1e-12)
        density_levels = np.linspace(0.0, density_upper, 19)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharex=True, sharey=True)
        titles = (
            "Analytic smoothed density",
            "Exact (no head trained)",
            "Exact (head trained)",
            "Learned divergence head",
        )
        for axis, values, title in zip(axes, densities, titles):
            filled = axis.contourf(
                grid0, grid1, values, levels=density_levels, cmap="viridis"
            )
            fig.colorbar(filled, ax=axis, shrink=0.82)
            axis.set_title(title)
            axis.set_xlabel(reference.labels[result["latent"][0]])
        axes[0].set_ylabel(reference.labels[result["latent"][1]])
        fig.suptitle(f"{reference.family.capitalize()} {mask_name} density validation")
    dh.save_figure(fig, f"{mask_name}_density_contours.png")


def plot_density_sweep(reference, mask_name, result):
    timesteps = list(result["timestep_results"])
    analytic = result["analytic_log_prob"]
    if result["dimension"] == 1:
        fig, axes = plt.subplots(2, len(timesteps), figsize=(22, 7), sharex=True)
        x = result["axis0"]
        for column, count in enumerate(timesteps):
            values = result["timestep_results"][count]
            no_head_exact = values["no_head_exact_log_prob"]
            trained_exact = values["trained_exact_log_prob"]
            trained_learned = values["trained_learned_log_prob"]
            axes[0, column].plot(x, np.exp(analytic), "k--", label="Analytic")
            axes[0, column].plot(
                x, np.exp(no_head_exact), color=COLORS["stage1"],
                label="Exact (no head trained)",
            )
            axes[0, column].plot(
                x, np.exp(trained_exact), color=COLORS["stage2"],
                label="Exact (head trained)",
            )
            axes[0, column].plot(
                x, np.exp(trained_learned), color=COLORS["head_exact"],
                label="Learned head",
            )
            axes[0, column].set_title(f"{count} integration steps")
            axes[1, column].plot(
                x, no_head_exact - analytic, color=COLORS["stage1"],
                label="No-head exact - analytic",
            )
            axes[1, column].plot(
                x, trained_exact - analytic, color=COLORS["stage2"],
                label="Trained exact - analytic",
            )
            axes[1, column].plot(
                x, trained_learned - analytic, color=COLORS["head_exact"],
                label="Learned head - analytic",
            )
            axes[1, column].axhline(0, color="black", lw=0.7)
        axes[0, 0].set_ylabel("Density")
        axes[1, 0].set_ylabel("Log-density error")
        axes[0, 0].legend(fontsize=7)
        axes[1, 0].legend(fontsize=7)
        for axis in axes[1]:
            axis.set_xlabel(reference.labels[result["latent"][0]])
    else:
        grid0, grid1 = result["axis0"], result["axis1"]
        no_head_densities = []
        trained_exact_densities = []
        trained_learned_densities = []
        no_head_errors = []
        trained_exact_errors = []
        trained_learned_errors = []
        for count in timesteps:
            values = result["timestep_results"][count]
            no_head_exact = values["no_head_exact_log_prob"]
            trained_exact = values["trained_exact_log_prob"]
            trained_learned = values["trained_learned_log_prob"]
            no_head_densities.append(np.exp(np.clip(no_head_exact, -80, 50)))
            trained_exact_densities.append(np.exp(np.clip(trained_exact, -80, 50)))
            trained_learned_densities.append(np.exp(np.clip(trained_learned, -80, 50)))
            no_head_errors.append(no_head_exact - analytic)
            trained_exact_errors.append(trained_exact - analytic)
            trained_learned_errors.append(trained_learned - analytic)

        density_rows = (
            no_head_densities,
            trained_exact_densities,
            trained_learned_densities,
        )
        error_rows = (
            no_head_errors,
            trained_exact_errors,
            trained_learned_errors,
        )
        density_upper = max(
            max(float(item.max()) for row in density_rows for item in row), 1e-12
        )
        error_limit = max(
            max(float(np.abs(item).max()) for row in error_rows for item in row), 1e-6
        )
        density_levels = np.linspace(0.0, density_upper, 19)
        error_levels = np.linspace(-error_limit, error_limit, 19)
        rows = density_rows + error_rows
        labels = (
            "Exact (no head trained) density",
            "Exact (head trained) density",
            "Learned-head density",
            "No-head exact - analytic log density",
            "Trained exact - analytic log density",
            "Learned head - analytic log density",
        )
        fig, all_axes = plt.subplots(
            6,
            len(timesteps) + 1,
            figsize=(25, 21),
            gridspec_kw={"width_ratios": [1] * len(timesteps) + [0.055]},
        )
        axes = all_axes[:, :len(timesteps)]
        color_axes = all_axes[:, -1]
        row_mappables = [None] * len(rows)
        for column, count in enumerate(timesteps):
            axes[0, column].set_title(f"{count} integration steps")
            for row, row_values in enumerate(rows):
                levels = density_levels if row < 3 else error_levels
                cmap = "viridis" if row < 3 else "coolwarm"
                row_mappables[row] = axes[row, column].contourf(
                    grid0, grid1, row_values[column], levels=levels, cmap=cmap
                )
        for row, (label, mappable) in enumerate(zip(labels, row_mappables)):
            axes[row, 0].set_ylabel(
                label + "\n" + reference.labels[result["latent"][1]]
            )
            fig.colorbar(mappable, cax=color_axes[row])
        for axis in axes[-1]:
            axis.set_xlabel(reference.labels[result["latent"][0]])
    fig.suptitle(
        f"{reference.family.capitalize()} {mask_name} density versus integration steps",
        y=1.02,
    )
    dh.save_figure(fig, f"{mask_name}_density_timestep_sweep.png")


def benchmark_summary(raw_records, log_prob_arrays, runtime_summaries):
    rows, warnings, details = dh.build_summary(raw_records, log_prob_arrays, runtime_summaries)
    return rows, warnings, details


def run_benchmark(problem, family, reference, splits, config, output_dir,
                  device, selected_gpu, force_retrain):
    dh.MASKS = reference.masks
    dh.OUTPUT_DIR = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    no_head_model, stage2, history, cache_dir, baseline_path = load_or_train_models(
        problem, family, reference, splits, config, device, force_retrain
    )
    log(f"[evaluate] {family}: instantaneous score and drift")
    drift_rows, score_rows, raw_records = dh.evaluate_instantaneous(
        reference, splits, no_head_model, stage2, config, device
    )
    log(f"[evaluate] {family}: held-out log probabilities")
    log_prob_rows, log_prob_arrays = dh.evaluate_log_probabilities(
        reference, splits, no_head_model, stage2, config, device
    )
    density_results = evaluate_density(
        reference, no_head_model, stage2, config, device
    )
    density_sweep = evaluate_density_sweep(
        reference, no_head_model, stage2, config, device
    )
    likelihood_convergence = evaluate_likelihood_convergence(
        reference, splits, stage2, config, device
    )
    log(f"[benchmark] {family}: exact trace versus learned head")
    runtime_rows, runtime_summaries = dh.benchmark_runtime(stage2, splits, config, device)
    summary_rows, warnings, details = benchmark_summary(
        raw_records, log_prob_arrays, runtime_summaries
    )
    plot_pairplot(reference, splits, output_dir, f"{family.capitalize()} joint simulator data")
    dh.plot_training_curves(history)
    dh.plot_drift_calibration(raw_records)
    dh.plot_error_vs_noise(drift_rows, score_rows)
    dh.plot_log_prob_calibration(log_prob_arrays)
    for mask_name, result in density_results.items():
        plot_density_result(reference, mask_name, result)
    for mask_name, result in density_sweep.items():
        plot_density_sweep(reference, mask_name, result)
    plot_likelihood_convergence(likelihood_convergence)
    dh.plot_runtime(runtime_summaries)

    output_config = copy.deepcopy(config)
    output_config.update({
        "problem": problem, "family": family,
        "model": {
            "nodes_size": stage2.nodes_size, "sde_type": stage2.sde_type,
            "sigma": stage2.sigma, "hidden_size": stage2.hidden_size,
            "depth": stage2.depth, "num_heads": stage2.num_heads,
            "mlp_ratio": stage2.mlp_ratio,
        },
        "normalization": {
            "theta_mean": reference.theta_mean.tolist(),
            "theta_std": reference.theta_std.tolist(),
            "x_mean": reference.x_mean.tolist(),
            "x_std": reference.x_std.tolist(),
        },
        "source_cache": str(cache_dir),
        "checkpoints": {
            "exact_no_head": {
                "path": str(baseline_path),
                "divergence": "exact",
                "divergence_head_trained": False,
            },
            "exact_head_trained": {
                "path": str(cache_dir / "stage2_joint.pt"),
                "divergence": "exact",
                "divergence_head_trained": True,
            },
            "learned_head": {
                "path": str(cache_dir / "stage2_joint.pt"),
                "divergence": "learned",
                "divergence_head_trained": True,
            },
        },
    })
    fingerprint = hashlib.sha256(
        json.dumps(output_config, sort_keys=True, default=list).encode()
    ).hexdigest()
    dh.write_outputs(
        output_config, fingerprint, device, selected_gpu, history,
        drift_rows, score_rows, log_prob_rows, runtime_rows,
        runtime_summaries, summary_rows, details,
    )
    dh.print_summary(summary_rows, warnings)
    return {"family": family, "output": str(output_dir),
            "warnings": warnings, "summary": details}


def write_line_parabola_manifest(results):
    LINE_PARABOLA_ROOT.mkdir(parents=True, exist_ok=True)
    with (LINE_PARABOLA_ROOT / "experiment_config.json").open("w") as handle:
        json.dump({"problem": "line_parabola", "models": results}, handle,
                  indent=2, allow_nan=True)
    readme = (
        "# Line/parabola divergence-head validation\n\n"
        "The full Gaussian-equivalent diagnostic suite is stored separately "
        "under line/ and parabola/; the top-level pairplot compares the two "
        "source simulators. Checkpoints are isolated from "
        "tutorials/data/PF_ODE_validation.\n"
    )
    (LINE_PARABOLA_ROOT / "README.md").write_text(readme)


def write_banana_readme():
    text = (
        "# Banana divergence-head validation\n\n"
        "This suite uses the simulator and normalization workflow from "
        "tutorials/Banana_posterior.ipynb. The reference posterior is "
        "evaluated by deterministic prior quadrature, while the likelihood "
        "reference is analytic Gaussian.\n"
    )
    (BANANA_OUTPUT / "README.md").write_text(text)


def validate_cpu_cap():
    logical, expected = CPU_LIMIT_INFO
    active = tuple(sorted(os.sched_getaffinity(0)))
    if active != expected or len(active) > CPU_THREAD_LIMIT:
        raise RuntimeError("CPU affinity no longer satisfies the 3-thread hard cap.")
    for variable in CPU_THREAD_ENV_VARS:
        if os.environ.get(variable) != str(len(active)):
            raise RuntimeError(f"{variable} no longer matches the CPU cap.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("all", "line_parabola", "banana"), default="all")
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_cpu_cap()
    logical, active = CPU_LIMIT_INFO
    log(f"[setup] CPU limited to {len(active)} of {logical} logical CPUs")
    selected_gpu = autocvd(num_gpus=1, interval=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[setup] autocvd selected {selected_gpu}; device={device}")
    dh.CPU_LIMIT_INFO = CPU_LIMIT_INFO
    dh.configure_plot_style()
    seed_all(BASE_CONFIG["seed"])
    all_results = []

    if args.problem in ("all", "line_parabola"):
        plot_combined_curve_pairplot()
        curve_results = []
        for family in ("line", "parabola"):
            config = copy.deepcopy(BASE_CONFIG)
            reference, splits = load_curve_splits(family, config)
            result = run_benchmark(
                "line_parabola", family, reference, splits, config,
                LINE_PARABOLA_ROOT / family, device, selected_gpu,
                args.force_retrain,
            )
            curve_results.append(result)
            all_results.append(result)
            torch.cuda.empty_cache()
        write_line_parabola_manifest(curve_results)

    if args.problem in ("all", "banana"):
        config = copy.deepcopy(BASE_CONFIG)
        cache_dir = DATA_ROOT / "banana" / "banana"
        reference, splits = banana_splits(config, cache_dir)
        result = run_benchmark(
            "banana", "banana", reference, splits, config,
            BANANA_OUTPUT, device, selected_gpu, args.force_retrain,
        )
        all_results.append(result)
        write_banana_readme()

    log("[done] generated divergence-head suites:")
    for result in all_results:
        log(f"  - {result['family']}: {result['output']}")


if __name__ == "__main__":
    main()
