"""Accuracy/runtime benchmark for analytic posterior-score inference.

The toy model is theta ~ N(MU0, PRIOR_STD**2), x | theta ~
N(theta, OBS_STD**2).  The score is the exact score of p_t(theta_t | x), so
the benchmark isolates integration and density-estimation error from network
approximation error.

It compares log-density accuracy at the analytic diffused-posterior mode, for both
VESDE and VPSDE and for uniform-time/log-noise grids:
  * DPM-2 samples evaluated with a KDE,
  * Heun PF-ODE exact-divergence likelihood integration, and
  * Euler PF-ODE samples evaluated with the same KDE.

Run explicitly (this script deliberately does not run on import):
    python tutorials/PF_ODE_1.py
"""
import os


CPU_THREAD_LIMIT = 3
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_cpu_usage_limit(max_threads=CPU_THREAD_LIMIT):
    """Apply the repository-wide hard CPU limit before numerical imports."""
    logical_cpus = os.cpu_count() or 1
    cpu_limit = min(int(max_threads), logical_cpus)
    if cpu_limit < 1:
        raise RuntimeError(
            f"Cannot enforce a {max_threads}-thread CPU limit on a {logical_cpus}-CPU host."
        )
    selected_cpus = tuple(sorted(os.sched_getaffinity(0))[:cpu_limit])
    if not selected_cpus:
        raise RuntimeError("The process has no CPUs available in its affinity mask.")
    os.sched_setaffinity(0, selected_cpus)
    for variable in CPU_THREAD_ENV_VARS:
        os.environ[variable] = str(len(selected_cpus))
    return logical_cpus, selected_cpus


CPU_LIMIT_INFO = None
if __name__ == "__main__":
    CPU_LIMIT_INFO = configure_cpu_usage_limit()


import csv
import sys
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from autocvd import autocvd
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compass.SDE import VESDE, VPSDE


OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "PF_ODE_new"
SEED = 3
EPS = 1e-3
N_THETAS = 10
N_SAMPLES = 4_000
RUNTIME_REPEATS = 5
TIMESTEPS = (10, 50, 100, 200)
# Match tests/test_pfode_vpsde_analytic.py.
MU0, PRIOR_STD, OBS_STD = -2.3, 0.3, 0.5


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    color: str
    marker: str


METHODS = (
    Method("dpm2_kde", "DPM-2 + KDE", "#386CB0", "o"),
    Method("heun_exact", "Heun + exact PF-ODE likelihood", "#E66101", "s"),
    Method("euler_pfode", "Euler PF-ODE + KDE", "#1B9E77", "^"),
)
SDE_FACTORIES = {"VESDE": lambda: VESDE(sigma=25.0), "VPSDE": VPSDE}
GRID_LABELS = {"uniform_t": "uniform t", "log_sigma": "log noise"}


def posterior_moments(x_obs):
    precision = PRIOR_STD ** -2 + OBS_STD ** -2
    variance = 1.0 / precision
    mean = variance * (MU0 / PRIOR_STD**2 + x_obs / OBS_STD**2)
    return mean, variance


def make_time_grid(sde, steps, grid_name, device):
    """Descending integration times with identical endpoints for both grids."""
    if grid_name == "uniform_t":
        return torch.linspace(1.0, EPS, steps, device=device)
    if grid_name == "log_sigma":
        endpoints = torch.tensor([EPS, 1.0], device=device)
        lam_min, lam_max = sde.lambda_t(endpoints)
        lams = torch.logspace(torch.log10(lam_max), torch.log10(lam_min), steps,
                              device=device)
        return sde.time_of_lambda(lams)
    raise ValueError(f"Unknown time grid: {grid_name}")


def score_y(y, t, posterior_mean, posterior_variance, sde):
    """Exact score in y=x/alpha coordinates of the diffused posterior."""
    lam = sde.lambda_t(t).to(y.device)
    return -(y - posterior_mean) / (posterior_variance + lam.square())


def initial_noise(n_samples, sde, generator):
    """The zero-centred terminal Gaussian used by Sampler._initial_sample."""
    lam_max = sde.lambda_t(torch.ones(1))
    return lam_max * torch.randn(n_samples, generator=generator)


def integrate_samples(method, times, posterior_mean, posterior_variance, sde, generator):
    """Mirror Sampler DPM-2/PF-ODE updates with the exact analytic score."""
    y = initial_noise(N_SAMPLES, sde, generator)
    if method == "euler_pfode":
        lam_max = sde.lambda_t(times[0])
        endpoint_mean = y + lam_max.square() * score_y(
            y, times[0], posterior_mean, posterior_variance, sde)
        y = y + endpoint_mean

    for index, (t_now, t_next) in enumerate(zip(times[:-1], times[1:])):
        lam_now, lam_next = sde.lambda_t(t_now), sde.lambda_t(t_next)
        h = lam_next - lam_now
        drift_now = -lam_now * score_y(y, t_now, posterior_mean, posterior_variance, sde)
        y_predict = y + h * drift_now
        if method == "euler_pfode":
            y = y_predict
            continue

        drift_next = -lam_next * score_y(
            y_predict, t_next, posterior_mean, posterior_variance, sde)
        y = y + 0.5 * h * (drift_now + drift_next)

        # Defaults from Sampler._dpm_sampler: five Langevin corrector steps
        # every fifth predictor, doubled during the final three time levels.
        final_levels = len(times) - 3
        if index % 5 == 0 or index >= final_levels:
            corrector_steps = 10 if index >= final_levels else 5
            for _ in range(corrector_steps):
                score = score_y(y, t_next, posterior_mean, posterior_variance, sde)
                noise_scale = torch.sqrt(0.2 * lam_next.square())
                y = y + 0.1 * lam_next.square() * score
                y = y + noise_scale * torch.randn_like(y)

    alpha_eps = sde.alpha_t(times[-1])
    return (alpha_eps * y).cpu().numpy()

def analytic_log_prob(theta, posterior_mean, posterior_variance, sde):
    """Analytic log p_eps(theta | x) used by the PF-ODE test."""
    eps = torch.tensor(EPS)
    alpha_eps = sde.alpha_t(eps).item()
    sigma_eps = sde.sigma_t(eps).item()
    variance_eps = alpha_eps**2 * posterior_variance + sigma_eps**2
    return -0.5 * (np.log(2 * np.pi * variance_eps)
                   + (theta - alpha_eps * posterior_mean)**2 / variance_eps)


def kde_log_prob(samples, theta):
    """Default SciPy KDE log density at one fixed evaluation point."""
    density = gaussian_kde(samples)(np.atleast_1d(theta)).item()
    return float(np.log(density))

def pfode_log_prob(theta, times, posterior_mean, posterior_variance, sde):
    """Exact-divergence PF-ODE log density, integrated with Heun on ``times``."""
    alpha_eps = sde.alpha_t(times[-1]).item()
    y = float(theta) / alpha_eps
    integral = 0.0
    forward_times = torch.flip(times, dims=(0,))
    for t_now, t_next in zip(forward_times[:-1], forward_times[1:]):
        # Forward (epsilon -> terminal) PF-ODE: dy/dlambda = -lambda score_y.
        lam_now, lam_next = sde.lambda_t(t_now).item(), sde.lambda_t(t_next).item()
        h = lam_next - lam_now
        var_now = posterior_variance + lam_now**2
        drift_now = -lam_now * (-(y - posterior_mean) / var_now)
        divergence_now = -1.0 / var_now
        y_predict = y + h * drift_now
        var_next = posterior_variance + lam_next**2
        drift_next = -lam_next * (-(y_predict - posterior_mean) / var_next)
        divergence_next = -1.0 / var_next
        y += 0.5 * h * (drift_now + drift_next)
        integral += 0.5 * h * (lam_now * divergence_now + lam_next * divergence_next)
    lam_terminal = sde.lambda_t(times[0]).item()
    # Match PFODE._log_prob_batch: use its Tweedie terminal-prior estimate.
    score_terminal = -(y - posterior_mean) / (posterior_variance + lam_terminal**2)
    log_terminal = (-0.5 * np.log(2 * np.pi * lam_terminal**2)
                    - 0.5 * lam_terminal**2 * score_terminal**2)
    return log_terminal - integral - np.log(alpha_eps)


def method_log_prob(method, times, posterior_mean, posterior_variance, sde, generator):
    """Return the method density at the analytic diffused-posterior mode."""
    alpha_eps = sde.alpha_t(times[-1]).item()
    theta_mode = alpha_eps * posterior_mean
    if method.key == "heun_exact":
        return pfode_log_prob(theta_mode, times, posterior_mean, posterior_variance, sde)
    samples = integrate_samples(method.key, times, posterior_mean, posterior_variance,
                                sde, generator)
    return kde_log_prob(samples, theta_mode)

def run_benchmark():
    """Benchmark log-density error and robust end-to-end runtime per setting."""
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    theta_true = (MU0 + PRIOR_STD * torch.randn(N_THETAS, generator=generator)).numpy()
    observation_noise = OBS_STD * torch.randn(N_THETAS, generator=generator).numpy()
    posterior_targets = [
        (theta, *posterior_moments(theta + noise))
        for theta, noise in zip(theta_true, observation_noise)
    ]
    rows = []
    for sde_name, make_sde in SDE_FACTORIES.items():
        for grid_name in GRID_LABELS:
            for steps in TIMESTEPS:
                for method in METHODS:
                    sde = make_sde()
                    times = make_time_grid(sde, steps, grid_name, device="cpu")
                    for _, posterior_mean, posterior_variance in posterior_targets:
                        method_log_prob(method, times, posterior_mean, posterior_variance, sde, generator)
                    elapsed_runs = []
                    method_log_probs = None
                    for _ in range(RUNTIME_REPEATS):
                        start = perf_counter()
                        method_log_probs = [
                            method_log_prob(method, times, posterior_mean, posterior_variance, sde, generator)
                            for _, posterior_mean, posterior_variance in posterior_targets
                        ]
                        elapsed_runs.append(perf_counter() - start)
                    median_runtime_per_theta = float(np.median(elapsed_runs) / N_THETAS)
                    for index, ((theta, posterior_mean, posterior_variance), method_log_prob_value) in enumerate(
                        zip(posterior_targets, method_log_probs)
                    ):
                        alpha_eps = sde.alpha_t(times[-1]).item()
                        theta_mode = alpha_eps * posterior_mean
                        analytic_log_prob_value = analytic_log_prob(
                            theta_mode, posterior_mean, posterior_variance, sde)
                        rows.append({
                            "method": method.label, "sde": sde_name,
                            "time_grid": grid_name, "timesteps": steps,
                            "theta_index": index, "theta_true": theta,
                            "theta_mode": theta_mode,
                            "analytic_log_prob": analytic_log_prob_value,
                            "method_log_prob": method_log_prob_value,
                            "absolute_log_prob_error": abs(method_log_prob_value - analytic_log_prob_value),
                            "median_runtime_seconds": median_runtime_per_theta,
                        })
    return rows


def summarise(rows):
    grouped = {}
    for row in rows:
        key = (row["method"], row["sde"], row["time_grid"], row["timesteps"])
        grouped.setdefault(key, []).append(row)
    summary = [
        {
            "method": key[0], "sde": key[1], "time_grid": key[2], "timesteps": key[3],
            "mean_absolute_log_prob_error": float(np.mean([item["absolute_log_prob_error"] for item in group])),
            "median_runtime_seconds": float(np.median([item["median_runtime_seconds"] for item in group])),
        }
        for key, group in grouped.items()
    ]
    for _, group in groupby(
        sorted(summary, key=lambda row: (row["method"], row["sde"], row["time_grid"], row["timesteps"])),
        key=lambda row: (row["method"], row["sde"], row["time_grid"]),
    ):
        runtime_floor = 0.0
        for row in group:
            runtime_floor = max(runtime_floor, row["median_runtime_seconds"])
            row["plot_runtime_seconds"] = runtime_floor
    return summary

def write_csv(rows, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_log_density_error_vs_runtime(summary):
    """Create one SDE panel; colour is method and line style is time grid."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, sde_name in zip(axes, SDE_FACTORIES):
        for method in METHODS:
            for grid_name, linestyle in (("uniform_t", "--"), ("log_sigma", "-")):
                group = sorted(
                    (row for row in summary if row["sde"] == sde_name
                     and row["method"] == method.label and row["time_grid"] == grid_name),
                    key=lambda row: row["timesteps"],
                )
                x = [row["plot_runtime_seconds"] for row in group]
                y = [row["mean_absolute_log_prob_error"] for row in group]
                ax.plot(x, y, color=method.color, marker=method.marker,
                        linestyle=linestyle, linewidth=2.1, markersize=6,
                        alpha=0.92)
                for row in group:
                    ax.annotate(str(row["timesteps"]),
                                (row["plot_runtime_seconds"], row["mean_absolute_log_prob_error"]),
                                xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_title(sde_name, weight="bold")
        ax.set_xlabel("monotone median end-to-end runtime per theta (seconds)")
        ax.grid(True, which="both", color="#DDE3EA", linewidth=0.8)
    axes[0].set_ylabel("mean |log p_method(theta*) - log p_analytic(theta*)|")
    method_handles = [plt.Line2D([], [], color=m.color, marker=m.marker, lw=2, label=m.label)
                      for m in METHODS]
    grid_handles = [plt.Line2D([], [], color="#4A5568", lw=2, ls=ls, label=GRID_LABELS[name])
                    for name, ls in (("uniform_t", "--"), ("log_sigma", "-"))]
    fig.legend(handles=method_handles + grid_handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Exact diffused-posterior score: log-density error versus runtime",
                 fontsize=15, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    fig.savefig(OUTPUT_DIR / "log_density_error_vs_runtime.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    logical_cpus, selected_cpus = CPU_LIMIT_INFO
    print(f"CPU cap: {len(selected_cpus)}/{logical_cpus} logical CPUs "
          f"({len(selected_cpus) / logical_cpus:.1%})")
    autocvd(num_gpus=1, interval=1)
    torch.manual_seed(SEED)
    rows = run_benchmark()
    summary = summarise(rows)
    write_csv(rows, "per_theta_results.csv")
    write_csv(summary, "summary.csv")
    plot_log_density_error_vs_runtime(summary)
    print(f"Saved CSV files and plot to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
