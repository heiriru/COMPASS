#!/usr/bin/env python3
"""Compare legacy and joint hierarchical annealed-score MAP estimates.

The benchmark uses an exact, known diffused score for the conjugate hierarchy

    g ~ N(0, sigma_g^2)
    l_i ~ N(0, sigma_l^2)
    x_i = g + l_i + epsilon_i,  epsilon_i ~ N(0, sigma_x^2).

Posterior particles are produced by COMPASS DPM-Solver-2 with the Gaussian
composition correction. The same particles initialize:

* ``old_score``: freeze the sampled global mean and refine only local values;
* ``joint_score``: optimize a KDE over the synchronized shared samples, then
  freeze that marginal shared MAP and refine only the local values.

The exact Gaussian posterior MAP is the reference. No neural-network training is
needed because ``ExactSharedLocalScore`` supplies the learned score analytically.
"""

from __future__ import annotations

import os


def configure_cpu_limit(fraction: float = 0.06) -> None:
    logical = os.cpu_count() or 1
    limit = max(1, int(logical * fraction))
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    selected = allowed[:min(limit, len(allowed))]
    os.sched_setaffinity(0, selected)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(len(selected))


if __name__ == "__main__":
    configure_cpu_limit()

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from compass.ModelTransfuser import ModelTransfuser
from compass.MultiObsSampler import MultiObsSampler
from compass.PFODE import PFODE
from compass.SDE import VESDE


OUTPUT_ROOT = ROOT / "tutorials" / "output" / "annealed_score_ascent"
MASK = torch.tensor([0.0, 0.0, 1.0])
HIERARCHY = [0]
SIGMA_GLOBAL = 0.8
SIGMA_LOCAL = 0.6
SIGMA_OBSERVATION = 0.25

NAVY = "#17324D"
BLUE = "#3C78A8"
TEAL = "#1F9D8A"
CORAL = "#E76F51"
GOLD = "#E9C46A"
PALE_BLUE = "#DCEAF5"
PALE_TEAL = "#D8F0EB"
GRID = "#D9E2EA"
METHOD_STYLE = {
    "old_score": {"label": "Legacy: frozen global", "color": CORAL, "marker": "o"},
    "joint_score": {"label": "Shared KDE MAP + local ascent", "color": TEAL, "marker": "D"},
    "analytic": {"label": "Analytic MAP", "color": NAVY, "marker": "_"},
}


class ExactSharedLocalScore(torch.nn.Module):
    """Exact single-observation diffused posterior score over nodes [g, l, x]."""

    def __init__(self, sde: VESDE):
        super().__init__()
        self.sde = sde
        precision = torch.tensor([
            [1 / SIGMA_GLOBAL**2 + 1 / SIGMA_OBSERVATION**2,
             1 / SIGMA_OBSERVATION**2],
            [1 / SIGMA_OBSERVATION**2,
             1 / SIGMA_LOCAL**2 + 1 / SIGMA_OBSERVATION**2],
        ], dtype=torch.float32)
        self.register_buffer("posterior_covariance", torch.linalg.inv(precision))

    def forward(self, x, t, c, return_attn_weights=False):
        noise_std = self.sde.marginal_prob_std(t).to(x.device)
        noise_std = noise_std.reshape(-1, 1)
        if noise_std.shape[0] == 1 and x.shape[0] != 1:
            noise_std = noise_std.expand(x.shape[0], 1)
        if noise_std.shape[0] != x.shape[0]:
            raise ValueError(
                f"Expected one diffusion time per row, got {noise_std.shape[0]} "
                f"times for a batch of {x.shape[0]}."
            )
        theta = x[:, :2]
        observed = x[:, 2]
        rhs = torch.stack([
            observed / SIGMA_OBSERVATION**2,
            observed / SIGMA_OBSERVATION**2,
        ], dim=1)
        posterior_mean = rhs @ self.posterior_covariance.T
        diffused_covariance = (
            self.posterior_covariance.unsqueeze(0)
            + noise_std.square().unsqueeze(-1)
            * torch.eye(2, device=x.device).unsqueeze(0)
        )
        score = torch.linalg.solve(
            diffused_covariance,
            (posterior_mean - theta).unsqueeze(-1),
        ).squeeze(-1)
        output = torch.zeros_like(x)
        output[:, :2] = noise_std * score
        if return_attn_weights:
            return output, torch.zeros(1, device=x.device)
        return output


class ExactScoreInference:
    """Minimal COMPASS-compatible model backed by the exact analytic score."""

    def __init__(self, device: str):
        self.nodes_size = 3
        self.sde_type = "vesde"
        self.sde = VESDE(sigma=25.0)
        self.sde.sigma = self.sde.sigma.to(device)
        self.model = ExactSharedLocalScore(self.sde).to(device)
        self.multi_obs_sampler = MultiObsSampler(self)
        self.pfode = PFODE(self)

    def output_scale_function(self, t, value):
        return value / self.sde.marginal_prob_std(t).to(value.device)

    def log_prob(self, data, condition_mask, **kwargs):
        return self.pfode.log_prob(data, condition_mask, **kwargs)

    def map_estimate(self, data, condition_mask, **kwargs):
        return self.pfode.map_estimate(data, condition_mask, **kwargs)

    def hierarchical_map_estimate(self, data, condition_mask, **kwargs):
        return self.multi_obs_sampler.map_estimate(data, condition_mask, **kwargs)


class AnalyticReference:
    @staticmethod
    def posterior(observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = observations.float().flatten()
        n = len(observations)
        precision = torch.zeros(n + 1, n + 1)
        precision[0, 0] = 1 / SIGMA_GLOBAL**2 + n / SIGMA_OBSERVATION**2
        precision[1:, 1:] = torch.eye(n) * (
            1 / SIGMA_LOCAL**2 + 1 / SIGMA_OBSERVATION**2
        )
        precision[0, 1:] = 1 / SIGMA_OBSERVATION**2
        precision[1:, 0] = 1 / SIGMA_OBSERVATION**2
        rhs = torch.cat([
            (observations.sum() / SIGMA_OBSERVATION**2).reshape(1),
            observations / SIGMA_OBSERVATION**2,
        ])
        covariance = torch.linalg.inv(precision)
        mean = covariance @ rhs
        return mean, covariance, precision

    @staticmethod
    def single_global_precision() -> float:
        precision = torch.tensor([
            [1 / SIGMA_GLOBAL**2 + 1 / SIGMA_OBSERVATION**2,
             1 / SIGMA_OBSERVATION**2],
            [1 / SIGMA_OBSERVATION**2,
             1 / SIGMA_LOCAL**2 + 1 / SIGMA_OBSERVATION**2],
        ])
        covariance = torch.linalg.inv(precision)
        return float(1 / covariance[0, 0])


def parse_n_values(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("observation counts must be positive integers")
    return values


def reserve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        from autocvd import autocvd
    except ImportError as exc:
        raise RuntimeError(
            "CUDA execution requires autocvd in this environment; use --device cpu otherwise."
        ) from exc
    autocvd(num_gpus=1, interval=1)
    if not torch.cuda.is_available():
        raise RuntimeError("autocvd completed, but CUDA is not available.")
    return "cuda"


def configure_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFCFE",
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "semibold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.65,
        "grid.linewidth": 0.75,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def make_output_tree(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "analytic": root / "analytic",
        "old_score": root / "old_score",
        "joint_score": root / "joint_score",
        "comparison": root / "comparison",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"saved {path}")


def joint_from_theta(observations: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    joint = torch.zeros(len(observations), 3)
    joint[:, :2] = theta
    joint[:, 2] = observations
    return joint


def vector_from_rows(rows: torch.Tensor) -> torch.Tensor:
    return torch.cat([rows[:1, 0].flatten(), rows[:, 1]])


def error_metrics(estimate: torch.Tensor, analytic: torch.Tensor,
                  covariance: torch.Tensor, precision: torch.Tensor) -> tuple[float, float, float]:
    """Measure estimator error relative to the observation-conditioned analytic MAP."""
    delta = estimate - analytic
    marginal_std = covariance.diag().sqrt()
    global_error = float(delta[0].abs() / marginal_std[0])
    local_rmse = float(torch.sqrt(torch.mean((delta[1:] / marginal_std[1:])**2)))
    mahalanobis = float(torch.sqrt(delta @ precision @ delta / len(delta)))
    return global_error, local_rmse, mahalanobis


def infer_one(model: ExactScoreInference, observations: torch.Tensor, args,
              device: str, seed: int) -> dict:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    analytic_map, covariance, precision = AnalyticReference.posterior(observations)
    n = len(observations)
    posterior_precision = torch.full(
        (n, 1), AnalyticReference.single_global_precision()
    )

    started = time.perf_counter()
    posterior_joint = model.multi_obs_sampler.sample(
        world_size=1,
        data=observations.reshape(-1, 1),
        condition_mask=MASK,
        timesteps=args.sampling_timesteps,
        eps=args.eps,
        num_samples=args.num_samples,
        hierarchy=HIERARCHY,
        prior=([0.0], [SIGMA_GLOBAL]),
        correction="gauss",
        posterior_precision=posterior_precision,
        order=2,
        corrector_steps=0,
        final_corrector_steps=0,
        method="dpm",
        device=device,
        verbose=False,
    ).cpu()
    if device == "cuda":
        torch.cuda.synchronize()
    sampling_runtime = time.perf_counter() - started
    posterior_theta = posterior_joint[:, :, :2]
    posterior_mean = posterior_theta.mean(dim=1)
    posterior_std = posterior_theta.std(dim=1, unbiased=False)

    legacy_init = joint_from_theta(observations, posterior_mean)
    legacy_mask = torch.tensor([1.0, 0.0, 1.0])
    started = time.perf_counter()
    legacy_rows = model.map_estimate(
        legacy_init,
        legacy_mask,
        sigma_start=max(2.0 * float(posterior_std[:, 1].max()), 1e-3),
        timesteps=args.map_timesteps,
        eps=args.eps,
        iterations_per_level=args.map_iterations,
        device=device,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    legacy_runtime = time.perf_counter() - started

    started = time.perf_counter()
    joint_rows, shared_result = ModelTransfuser._shared_then_local_map(
        model=model,
        posterior_samples=posterior_theta,
        x=observations.reshape(-1, 1),
        condition_mask=MASK,
        hierarchy=HIERARCHY,
        num_starts=args.map_num_starts,
        timesteps=args.map_timesteps,
        eps=args.eps,
        iterations_per_level=args.map_iterations,
        device=device,
    )
    selected_index = int(shared_result["selected_start"])
    candidate_scores = shared_result["candidate_log_densities"]
    if device == "cuda":
        torch.cuda.synchronize()
    joint_runtime = time.perf_counter() - started

    return {
        "analytic": analytic_map,
        "covariance": covariance,
        "precision": precision,
        "legacy_rows": legacy_rows,
        "joint_rows": joint_rows,
        "legacy": vector_from_rows(legacy_rows),
        "joint": vector_from_rows(joint_rows),
        "posterior_theta": posterior_theta,
        "posterior_mean": posterior_mean,
        "sampling_runtime": sampling_runtime,
        "legacy_runtime": legacy_runtime,
        "joint_runtime": joint_runtime,
        "selected_start": selected_index,
        "candidate_scores": candidate_scores,
    }


def benchmark(args, device: str) -> tuple[list[dict], list[dict], list[dict], dict]:
    model = ExactScoreInference(device)
    metric_rows: list[dict] = []
    estimate_rows: list[dict] = []
    sampling_rows: list[dict] = []
    example = None
    total = len(args.n_observations) * args.repeats
    completed = 0

    for n in args.n_observations:
        for repeat in range(args.repeats):
            run_seed = args.seed + 10_000 * n + repeat
            generator = torch.Generator().manual_seed(run_seed)
            true_global = SIGMA_GLOBAL * torch.randn((), generator=generator)
            true_local = SIGMA_LOCAL * torch.randn(n, generator=generator)
            observations = (
                true_global + true_local
                + SIGMA_OBSERVATION * torch.randn(n, generator=generator)
            )
            result = infer_one(model, observations, args, device, run_seed)
            truth = torch.cat([true_global.reshape(1), true_local])

            for method, estimate, runtime in (
                ("analytic", result["analytic"], 0.0),
                ("old_score", result["legacy"], result["legacy_runtime"]),
                ("joint_score", result["joint"], result["joint_runtime"]),
            ):
                global_error, local_rmse, mahalanobis = error_metrics(
                    estimate, result["analytic"], result["covariance"], result["precision"]
                )
                metric_rows.append({
                    "n_observations": n,
                    "repeat": repeat,
                    "method": method,
                    "global_error_analytic_sigma": global_error,
                    "local_rmse_analytic_sigma": local_rmse,
                    "joint_mahalanobis_per_dimension": mahalanobis,
                    "map_runtime_seconds": runtime,
                    "sampling_runtime_seconds": result["sampling_runtime"],
                })
                for parameter_index in range(n + 1):
                    estimate_rows.append({
                        "n_observations": n,
                        "repeat": repeat,
                        "method": method,
                        "parameter": "global" if parameter_index == 0 else "local",
                        "parameter_index": parameter_index,
                        "truth": float(truth[parameter_index]),
                        "analytic_map": float(result["analytic"][parameter_index]),
                        "analytic_std": float(result["covariance"].diag().sqrt()[parameter_index]),
                        "estimate": float(estimate[parameter_index]),
                    })
            sampling_rows.append({
                "n_observations": n,
                "repeat": repeat,
                "sampling_runtime_seconds": result["sampling_runtime"],
                "selected_joint_start": result["selected_start"],
            })
            if n == max(args.n_observations) and repeat == 0:
                example = {
                    **result,
                    "observations": observations,
                    "truth": truth,
                    "n": n,
                }
            completed += 1
            print(f"[{completed:>3}/{total}] N={n:>3}, repeat={repeat:>2}")

    if example is None:
        raise RuntimeError("No example run was collected.")
    return metric_rows, estimate_rows, sampling_rows, example


def aggregate_metric(rows: list[dict], method: str, field: str):
    xs, center, lower, upper = [], [], [], []
    for n in sorted({int(row["n_observations"]) for row in rows}):
        values = np.asarray([
            float(row[field]) for row in rows
            if row["method"] == method and int(row["n_observations"]) == n
        ])
        xs.append(n)
        center.append(np.mean(values))
        lower.append(np.quantile(values, 0.16))
        upper.append(np.quantile(values, 0.84))
    return np.asarray(xs), np.asarray(center), np.asarray(lower), np.asarray(upper)


def plot_accuracy(metrics: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    fields = (
        (
            "global_error_analytic_sigma",
            "Shared MAP offset from analytic MAP",
            r"$|\hat g-g_{\mathrm{MAP}}^{\mathrm{analytic}}(x_{1:N})|"
            r"\,/\,\sigma_g^{\mathrm{analytic}}$",
        ),
        ("local_rmse_analytic_sigma", "Local MAP error", "RMSE / analytic σ"),
        ("map_runtime_seconds", "Refinement cost", "runtime (s)"),
    )
    for axis, (field, title, ylabel) in zip(axes, fields):
        for method in ("old_score", "joint_score"):
            x, mean, low, high = aggregate_metric(metrics, method, field)
            style = METHOD_STYLE[method]
            axis.fill_between(x, low, high, color=style["color"], alpha=0.14)
            axis.plot(
                x, mean, color=style["color"], marker=style["marker"],
                linewidth=2.1, markersize=6, label=style["label"],
            )
        if field != "map_runtime_seconds":
            axis.axhline(
                0, color=NAVY, linestyle="--", linewidth=1.2,
                label="analytic MAP (zero offset)",
            )
        axis.set(xlabel="number of observations", ylabel=ylabel, title=title, xscale="log")
        axis.set_xticks(sorted({int(row["n_observations"]) for row in metrics}))
        axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axes[0].legend(loc="upper right")
    fig.suptitle("Shared marginal mode removes frozen-global initialization error",
                 fontsize=14, fontweight="semibold", color=NAVY)
    save_figure(fig, output)


def plot_parity(estimates: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), constrained_layout=True)
    for axis, parameter, title in zip(axes, ("global", "local"), ("Shared parameter", "Local parameters")):
        subset = [row for row in estimates if row["parameter"] == parameter and row["method"] != "analytic"]
        values = np.asarray([float(row["analytic_map"]) for row in subset])
        inferred = np.asarray([float(row["estimate"]) for row in subset])
        lo, hi = min(values.min(), inferred.min()), max(values.max(), inferred.max())
        pad = 0.05 * max(hi - lo, 1e-3)
        axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=NAVY, linestyle="--", linewidth=1.4)
        for method in ("old_score", "joint_score"):
            rows = [row for row in subset if row["method"] == method]
            style = METHOD_STYLE[method]
            axis.scatter(
                [row["analytic_map"] for row in rows], [row["estimate"] for row in rows],
                s=30 if parameter == "global" else 13, alpha=0.72,
                color=style["color"], marker=style["marker"],
                edgecolor="white", linewidth=0.35, label=style["label"],
            )
        axis.set(xlabel="analytic MAP", ylabel="COMPASS MAP", title=title)
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(loc="upper left")
    fig.suptitle("MAP parity against the exact Gaussian posterior", fontsize=14,
                 fontweight="semibold", color=NAVY)
    save_figure(fig, output)


def plot_example(example: dict, output: Path) -> None:
    n = example["n"]
    indices = np.arange(n)
    analytic = example["analytic"].numpy()
    legacy = example["legacy"].numpy()
    joint = example["joint"].numpy()
    truth = example["truth"].numpy()
    posterior = example["posterior_theta"].numpy()

    fig = plt.figure(figsize=(13.5, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(0.85, 2.15))
    ax_global = fig.add_subplot(grid[0, 0])
    ax_local = fig.add_subplot(grid[0, 1])

    ax_global.hist(
        posterior[0, :, 0], bins=32, density=True, color=PALE_BLUE,
        edgecolor="white", linewidth=0.5, label="DPM2 + Gaussian samples",
    )
    for method, value, linestyle in (
        ("analytic", analytic[0], "--"),
        ("old_score", legacy[0], "-"),
        ("joint_score", joint[0], "-"),
    ):
        style = METHOD_STYLE[method]
        ax_global.axvline(value, color=style["color"], linewidth=2.2,
                          linestyle=linestyle, label=style["label"])
    ax_global.axvline(truth[0], color=GOLD, linewidth=1.5, linestyle=":", label="simulation truth")
    ax_global.set(xlabel="shared parameter g", ylabel="posterior density", title="Shared MAP")
    ax_global.legend(fontsize=8)

    ax_local.scatter(indices, analytic[1:], s=120, color=NAVY, marker="_",
                     linewidth=1.7, zorder=5, label="analytic MAP")
    ax_local.scatter(indices, legacy[1:], s=42, color=CORAL, marker="o",
                     edgecolor="white", linewidth=0.6, label="legacy")
    ax_local.scatter(indices, joint[1:], s=42, color=TEAL, marker="D",
                     edgecolor="white", linewidth=0.6, label="joint")
    ax_local.scatter(indices, truth[1:], s=24, facecolor="none", edgecolor=GOLD,
                     linewidth=1.2, label="simulation truth")
    ax_local.set(xlabel="observation index", ylabel="local parameter lᵢ",
                 title=f"Local MAP reconstruction for N={n}")
    ax_local.legend(ncol=4, loc="upper center")
    fig.suptitle("One hierarchical posterior: shared pooling and local reconstruction",
                 fontsize=14, fontweight="semibold", color=NAVY)
    save_figure(fig, output)


def add_covariance_ellipse(axis, mean, covariance, color, level, label=None):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ellipse = Ellipse(
        mean, width=2 * level * np.sqrt(eigenvalues[0]),
        height=2 * level * np.sqrt(eigenvalues[1]), angle=angle,
        facecolor="none", edgecolor=color, linewidth=1.7,
        linestyle="--" if level > 1.5 else "-", label=label,
    )
    axis.add_patch(ellipse)


def plot_posterior_slice(example: dict, output: Path) -> None:
    samples = example["posterior_theta"].numpy()
    analytic = example["analytic"].numpy()
    legacy = example["legacy"].numpy()
    joint = example["joint"].numpy()
    covariance = example["covariance"].numpy()[np.ix_([0, 1], [0, 1])]

    fig, axis = plt.subplots(figsize=(6.5, 5.6), constrained_layout=True)
    axis.scatter(samples[0, :, 0], samples[0, :, 1], s=9, color=BLUE,
                 alpha=0.18, edgecolor="none", label="DPM2 + Gaussian particles")
    add_covariance_ellipse(axis, analytic[[0, 1]], covariance, NAVY, 1.0, "analytic 1σ")
    add_covariance_ellipse(axis, analytic[[0, 1]], covariance, NAVY, 2.0, "analytic 2σ")
    axis.scatter(legacy[0], legacy[1], s=80, color=CORAL, marker="o",
                 edgecolor="white", linewidth=0.8, label="legacy MAP")
    axis.scatter(joint[0], joint[1], s=85, color=TEAL, marker="D",
                 edgecolor="white", linewidth=0.8, label="joint MAP")
    axis.scatter(analytic[0], analytic[1], s=110, color=NAVY, marker="*",
                 edgecolor="white", linewidth=0.6, label="analytic MAP")
    axis.set(xlabel="shared parameter g", ylabel="first local parameter l₀",
             title="Analytic posterior slice and inferred modes")
    axis.legend(fontsize=8, loc="best")
    save_figure(fig, output)


def save_results(paths: dict[str, Path], metrics: list[dict], estimates: list[dict],
                 sampling: list[dict], args, device: str) -> None:
    for method in ("analytic", "old_score", "joint_score"):
        write_csv(paths[method] / "metrics.csv", [row for row in metrics if row["method"] == method])
        write_csv(paths[method] / "estimates.csv", [row for row in estimates if row["method"] == method])
    write_csv(paths["comparison"] / "all_metrics.csv", metrics)
    write_csv(paths["comparison"] / "all_estimates.csv", estimates)
    write_csv(paths["comparison"] / "sampling_runtime.csv", sampling)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "n_observations": list(args.n_observations),
        "resolved_device": device,
        "sampler": "DPM-Solver-2",
        "composition_correction": "gauss",
        "score": "exact analytic diffused single-observation posterior score",
        "sigma_global": SIGMA_GLOBAL,
        "sigma_local": SIGMA_LOCAL,
        "sigma_observation": SIGMA_OBSERVATION,
    }
    with (paths["root"] / "run_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--n-observations", type=parse_n_values, default=(2, 5, 10, 25, 50))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--sampling-timesteps", type=int, default=100)
    parser.add_argument("--map-timesteps", type=int, default=100)
    parser.add_argument("--map-iterations", type=int, default=3)
    parser.add_argument("--map-num-starts", type=int, default=8)
    parser.add_argument("--log-prob-timesteps", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--quick", action="store_true",
                        help="small smoke-sized benchmark; still writes the complete plot set")
    return parser


def validate_args(args) -> None:
    positive = {
        "repeats": args.repeats,
        "num_samples": args.num_samples,
        "sampling_timesteps": args.sampling_timesteps,
        "map_timesteps": args.map_timesteps,
        "map_iterations": args.map_iterations,
        "map_num_starts": args.map_num_starts,
        "log_prob_timesteps": args.log_prob_timesteps,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if not 0 < args.eps < 1:
        raise ValueError("eps must lie strictly between 0 and 1.")


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.n_observations = (2, 10)
        args.repeats = 3
        args.num_samples = 256
        args.sampling_timesteps = 40
        args.map_timesteps = 40
        args.map_num_starts = 4
        args.log_prob_timesteps = 40
    validate_args(args)
    device = reserve_device(args.device)
    configure_style()
    paths = make_output_tree(args.output_dir)
    metrics, estimates, sampling, example = benchmark(args, device)
    save_results(paths, metrics, estimates, sampling, args, device)
    plot_accuracy(metrics, paths["comparison"] / "01_accuracy_vs_observations.png")
    plot_parity(estimates, paths["comparison"] / "02_analytic_map_parity.png")
    plot_example(example, paths["comparison"] / "03_example_shared_local_map.png")
    plot_posterior_slice(example, paths["comparison"] / "04_analytic_posterior_slice.png")
    print(f"\nAll results written below {paths['root']}")


if __name__ == "__main__":
    main()
