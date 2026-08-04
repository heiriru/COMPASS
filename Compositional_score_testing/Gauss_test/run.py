"""Recreate paper Figures 1--4 with diagonal GAUSS, full Gaussian, and Langevin.

See README.md for the staged execution commands. Importing this module performs
no simulation, training, sampling, or plotting.
"""
from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from compass.MultiObsSampler import MultiObsSampler

from .analytic import AnalyticSBIm, GaussianMixtureToy, GaussianToy
from .common import (ARTIFACTS, atomic_torch_save, load_compass_model,
                     mala_reference, paper_langevin, read_rows, resolve_device,
                     scaled_mmd_to_dirac, seed_everything, sliced_wasserstein,
                     standardize_x, train_compass_model, write_rows)
from .tasks import JRNMMTask, get_task


METHODS = ("gauss", "full_gaussian", "langevin")
MOMENT_METHOD = "gauss_global_local_moment"
FIGURE1_METHODS = (*METHODS, MOMENT_METHOD)
FIGURE1_ROOT = ARTIFACTS / "figure1_gauss_global_local_moment"
STYLE = {
    "gauss": {"label": "COMPASS GAUSS (diagonal)", "color": "#0072B2", "marker": "o"},
    "full_gaussian": {"label": "COMPASS full Gaussian", "color": "#009E73", "marker": "^"},
    "langevin": {"label": "F-NPSE Langevin", "color": "#D55E00", "marker": "s"},
    MOMENT_METHOD: {
        "label": "COMPASS global/local Gaussian + moments",
        "color": "#CC79A7", "marker": "D",
    },
}
N_TOY = (2, 4, 8, 16, 32, 64, 90)
N_TALL = (1, 8, 14, 22, 30)
N_TRAIN = (1000, 3000, 10000, 30000)
BENCHMARKS = ("slcp", "lotka_volterra", "sir")
NOISE_LEVELS = (0.0, 1e-3, 1e-2, 1e-1)


def reserve_gpu(device: str) -> None:
    if device.startswith("cuda"):
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=1)


def simulate_batches(task, z: torch.Tensor, size: int = 1024) -> torch.Tensor:
    return torch.cat([task.simulate(z[start:start + size])
                      for start in range(0, len(z), size)])


@torch.no_grad()
def gauss(model, context: torch.Tensor, theta_dim: int, device: torch.device,
          samples: int, timesteps: int = 1000,
          correction: str = "gauss",
          precision: torch.Tensor | None = None,
          covariance: torch.Tensor | None = None,
          posterior_mean: torch.Tensor | None = None,
          clamp: float | None = 5.0) -> torch.Tensor:
    """Invoke COMPASS DPM with diagonal or full-covariance Gaussian correction."""
    if correction not in ("gauss", "full_gaussian", "Gauss_global_local"):
        raise ValueError(f"Unsupported Gaussian correction: {correction}")
    context = context.to(device)
    mask = torch.cat((torch.zeros(theta_dim, device=device),
                      torch.ones(context.shape[1], device=device)))
    sampler = getattr(model, "multi_obs_sampler", None) or MultiObsSampler(model)
    draws = sampler.sample(
        world_size=1, data=context, condition_mask=mask,
        hierarchy=list(range(theta_dim)),
        prior=(torch.zeros(theta_dim), torch.ones(theta_dim)),
        correction=correction, posterior_precision=precision,
        posterior_covariance=covariance, posterior_mean=posterior_mean,
        precision_est_samples=1000, precision_est_timesteps=100,
        precision_est_batch_size=128, denoise_clamp=clamp,
        timesteps=timesteps, num_samples=samples, method="dpm", order=2,
        corrector_steps=0, final_corrector_steps=0,
        device=str(device), verbose=True,
    )
    return draws[0, :, :theta_dim].detach().cpu()


def _contour(ax, samples: torch.Tensor, color: str, label: str) -> None:
    values = samples[:, :2].T.numpy()
    kde = gaussian_kde(values)
    gx = np.linspace(np.percentile(values[0], .5), np.percentile(values[0], 99.5), 90)
    gy = np.linspace(np.percentile(values[1], .5), np.percentile(values[1], 99.5), 90)
    xx, yy = np.meshgrid(gx, gy)
    zz = kde(np.vstack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=5, colors=color, linewidths=1.4)
    ax.plot([], [], color=color, label=label)


def fig1_sample(args, device) -> None:
    root, toy = FIGURE1_ROOT, GaussianToy(2)
    _, observations = toy.sample_problem(64, args.seed)
    model = AnalyticSBIm(toy, 0.0, device)
    result = {"observations": observations}
    for n in (2, 16, 64):
        context = observations[:n]
        result[f"reference_{n}"] = toy.sample_reference(context, args.num_samples, args.seed + n)
        result[f"gauss_{n}"] = gauss(model, context, 2, device, args.num_samples,
                                      correction="gauss",
                                      precision=toy.single_precision_diagonal(context), clamp=None)
        result[f"full_gaussian_{n}"] = gauss(
            model, context, 2, device, args.num_samples,
            correction="full_gaussian",
            covariance=toy.single_covariance_estimate(context), clamp=None,
        )
        result[f"{MOMENT_METHOD}_{n}"] = gauss(
            model, context, 2, device, args.num_samples,
            correction="Gauss_global_local",
            covariance=toy.single_covariance_estimate(context),
            posterior_mean=toy.single_mean_estimate(context), clamp=None,
        )
        result[f"langevin_{n}"] = paper_langevin(
            model, context, 2, device, num_samples=args.num_samples,
            steps=400, langevin_steps=5, tau=.5)
    atomic_torch_save(result, root / "samples.pt")


def fig1_plot() -> None:
    root = FIGURE1_ROOT
    data = torch.load(root / "samples.pt", map_location="cpu", weights_only=True)
    observation_counts = (2, 16, 64)
    fig, axes = plt.subplots(len(FIGURE1_METHODS), len(observation_counts),
                             figsize=(12, 13.2))
    for row, method in enumerate(FIGURE1_METHODS):
        style = STYLE[method]
        for column, n in enumerate(observation_counts):
            ax = axes[row, column]
            _contour(ax, data[f"reference_{n}"], "black", "analytic")
            _contour(ax, data[f"{method}_{n}"], style["color"], style["label"])
            ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$")
            if row == 0:
                ax.set_title(rf"$n={n}$")
        axes[row, 0].annotate(
            style["label"], xy=(-0.27, 0.5), xycoords="axes fraction",
            ha="center", va="center", rotation=90, color=style["color"],
            fontsize=10, fontweight="bold",
        )
        axes[row, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / "figure1.png", dpi=250); fig.savefig(root / "figure1.pdf"); plt.close(fig)


def fig2_sample(args, device) -> None:
    root, rows = ARTIFACTS / "figure2", []
    for task_name in ("gaussian", "gmm"):
        for epsilon in NOISE_LEVELS:
            for repeat in range(args.repeats):
                seed = args.seed + repeat
                if task_name == "gaussian":
                    gen = torch.Generator().manual_seed(seed)
                    toy = GaussianToy(10, 20 * torch.rand(10, generator=gen) - 10,
                                      25 * torch.rand(10, generator=gen) + .1)
                else:
                    toy = GaussianMixtureToy(10)
                _, observations = toy.sample_problem(100, seed)
                model = AnalyticSBIm(toy, epsilon, device)
                for n in N_TOY:
                    context = observations[:n]
                    reference = (toy.sample_reference(context, args.num_samples, seed + n + 10_000)
                                 if task_name == "gaussian" else
                                 mala_reference(lambda x: toy.log_posterior(x, context.to(device)), 10,
                                                num_samples=args.num_samples, seed=seed + n,
                                                step_size=.08, device=device))
                    for method in args.methods:
                        if method == "gauss":
                            draws = gauss(
                                model, context, 10, device, args.num_samples,
                                correction=method,
                                precision=toy.single_precision_diagonal(context),
                                clamp=None,
                            )
                        elif method == "full_gaussian":
                            draws = gauss(
                                model, context, 10, device, args.num_samples,
                                correction=method,
                                covariance=toy.single_covariance_estimate(context),
                                clamp=None,
                            )
                        else:
                            draws = paper_langevin(
                                model, context, 10, device,
                                num_samples=args.num_samples, steps=400,
                                langevin_steps=5, tau=.5,
                                prior_mean=toy.prior_mean,
                                prior_std=toy.prior_std,
                            )
                        rows.append({"task": task_name, "epsilon": epsilon, "repeat": repeat,
                                     "n": n, "method": method,
                                     "sliced_wasserstein": sliced_wasserstein(reference, draws, seed=seed)})
                        atomic_torch_save(draws, root / "samples" / task_name /
                                          f"eps_{epsilon:g}" / f"seed_{repeat}" /
                                          f"n_{n}_{method}.pt")
    write_rows(rows, root / "metrics.csv")


def _aggregate(rows, filters, x_key, y_key):
    chosen = [r for r in rows if all(r[k] == v for k, v in filters.items())]
    xs = sorted({float(r[x_key]) for r in chosen})
    groups = [np.asarray([float(r[y_key]) for r in chosen if float(r[x_key]) == x]) for x in xs]
    return np.asarray(xs), np.asarray([g.mean() for g in groups]), np.asarray([g.std() for g in groups])


def fig2_plot() -> None:
    root, rows = ARTIFACTS / "figure2", read_rows(ARTIFACTS / "figure2" / "metrics.csv")
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), sharex=True)
    for i, task in enumerate(("gaussian", "gmm")):
        for j, epsilon in enumerate(NOISE_LEVELS):
            ax = axes[i, j]
            for method in METHODS:
                x, mean, std = _aggregate(rows, {"task": task, "epsilon": str(epsilon),
                                                 "method": method}, "n", "sliced_wasserstein")
                st = STYLE[method]; ax.plot(x, mean, color=st["color"], marker=st["marker"], label=st["label"])
                ax.fill_between(x, mean - std, mean + std, color=st["color"], alpha=.15)
            ax.set_xscale("log"); ax.set_title(rf"$\epsilon={epsilon:g}$")
            if j == 0: ax.set_ylabel(f"{task.upper()}\nsliced Wasserstein")
            if i == 1: ax.set_xlabel("observations n")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(root / "figure2.png", dpi=250); fig.savefig(root / "figure2.pdf"); plt.close(fig)


def fig3_prepare(args, device) -> None:
    root = ARTIFACTS / "figure3" / "data"
    for task_name in args.tasks:
        task = get_task(task_name); seed_everything(args.seed)
        train_z = torch.randn(max(args.n_train), task.theta_dim)
        train_x = simulate_batches(task, train_z)
        cases = []
        for case in range(args.cases):
            seed_everything(args.seed + 1000 + case)
            truth_z = torch.randn(1, task.theta_dim)
            observations = simulate_batches(task, truth_z.repeat(30, 1))
            references = {}
            for n in N_TALL:
                context = observations[:n].to(device)
                z_ref = mala_reference(
                    lambda z: -.5 * z.square().sum(-1) + task.log_likelihood(z, context),
                    task.theta_dim, num_samples=args.num_samples,
                    seed=args.seed + 10_000 * case + n,
                    step_size={"slcp": .06, "sir": .04, "lotka_volterra": .025}[task_name],
                    device=device)
                references[n] = task.theta_from_z(z_ref)
            cases.append({"truth_z": truth_z, "truth_theta": task.theta_from_z(truth_z),
                          "observations": observations, "references": references})
        atomic_torch_save({"train_z": train_z, "train_x": train_x, "cases": cases},
                          root / f"{task_name}.pt")


def fig3_train(args, device) -> None:
    root = ARTIFACTS / "figure3"
    for task_name in args.tasks:
        data = torch.load(root / "data" / f"{task_name}.pt", weights_only=False)
        for n_train in args.n_train:
            train_compass_model(data["train_z"][:n_train], data["train_x"][:n_train],
                                root / "models" / task_name / f"n_{n_train}", device,
                                epochs=args.epochs, batch_size=args.batch_size,
                                learning_rate=args.learning_rate, seed=args.seed)


def fig3_sample(args, device) -> None:
    root, rows = ARTIFACTS / "figure3", []
    for task_name in args.tasks:
        task = get_task(task_name)
        data = torch.load(root / "data" / f"{task_name}.pt", weights_only=False)
        for n_train in args.n_train:
            model, norm = load_compass_model(root / "models" / task_name / f"n_{n_train}", device)
            for case, item in enumerate(data["cases"][:args.cases]):
                for n in N_TALL:
                    context, _, _ = standardize_x(item["observations"][:n], norm["x_mean"], norm["x_std"])
                    for method in args.methods:
                        z_draws = (
                            gauss(
                                model, context, task.theta_dim, device,
                                args.num_samples, correction=method,
                            )
                            if method in ("gauss", "full_gaussian") else
                            paper_langevin(
                                model, context, task.theta_dim, device,
                                num_samples=args.num_samples, steps=400,
                                langevin_steps=5, tau=.5,
                                clip=3.0 if args.clip else None,
                            )
                        )
                        draws = task.theta_from_z(z_draws)
                        folder = root / "samples" / task_name / f"ntrain_{n_train}"
                        atomic_torch_save(draws, folder / f"case_{case:02d}_n_{n}_{method}.pt")
                        rows.append({"task": task_name, "n_train": n_train, "case": case,
                                     "n": n, "method": method,
                                     "sliced_wasserstein": sliced_wasserstein(item["references"][n], draws,
                                                                                seed=args.seed + case)})
    write_rows(rows, root / "metrics.csv")


def fig3_plot() -> None:
    root, rows = ARTIFACTS / "figure3", read_rows(ARTIFACTS / "figure3" / "metrics.csv")
    fig, axes = plt.subplots(3, len(METHODS), figsize=(14, 10), sharex=True)
    colors = plt.get_cmap("viridis")(np.linspace(.1, .9, len(N_TALL)))
    for i, task in enumerate(BENCHMARKS):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            for n, color in zip(N_TALL, colors):
                x, mean, std = _aggregate(rows, {"task": task, "method": method, "n": str(n)},
                                           "n_train", "sliced_wasserstein")
                ax.plot(x, mean, "o-", color=color, label=rf"$n={n}$")
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=.12)
            ax.set_xscale("log"); ax.set_title(STYLE[method]["label"])
            if j == 0: ax.set_ylabel(f"{task.replace('_', ' ').title()}\nsliced Wasserstein")
            if i == 2: ax.set_xlabel(r"training simulations $N_{train}$")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(root / "figure3.png", dpi=250); fig.savefig(root / "figure3.pdf"); plt.close(fig)


def fig4_prepare(args) -> None:
    root = ARTIFACTS / "figure4"
    task = JRNMMTask(integration_batch_size=args.simulation_batch_size)
    seed_everything(args.seed); train_z = torch.randn(50_000, 3)
    if args.jrnnm_backend == "exact":
        from .jrnnm_reference import simulate_exact_jrnnm
        simulator = lambda z: simulate_exact_jrnnm(task, z)
    else:
        simulator = lambda z: simulate_batches(task, z, args.simulation_batch_size)
    train_x = simulator(train_z)
    truth = torch.tensor([[135., 220., 2000.]])
    u = ((truth - task.low) / (task.high - task.low)).clamp(1e-7, 1 - 1e-7)
    truth_z = math.sqrt(2.) * torch.erfinv(2 * u - 1)
    observations = simulator(truth_z.repeat(30, 1))
    atomic_torch_save({"train_z": train_z, "train_x": train_x, "truth_z": truth_z,
                       "truth_theta": truth, "observations": observations}, root / "data.pt")


def fig4_train(args, device) -> None:
    root = ARTIFACTS / "figure4"; data = torch.load(root / "data.pt", weights_only=True)
    train_compass_model(data["train_z"], data["train_x"], root / "model", device,
                        epochs=args.epochs, batch_size=args.batch_size,
                        learning_rate=args.learning_rate, seed=args.seed)


def fig4_sample(args, device) -> None:
    root, task = ARTIFACTS / "figure4", JRNMMTask()
    data = torch.load(root / "data.pt", weights_only=True)
    model, norm = load_compass_model(root / "model", device)
    def draw(context, method):
        context, _, _ = standardize_x(context, norm["x_mean"], norm["x_std"])
        z = (
            gauss(
                model, context, 3, device, args.figure4_samples,
                correction=method,
            )
            if method in ("gauss", "full_gaussian") else
            paper_langevin(
                model, context, 3, device,
                num_samples=args.figure4_samples, steps=400,
                langevin_steps=5, tau=.5,
                clip=3.0 if args.clip else None,
            )
        )
        return task.theta_from_z(z)
    for n in N_TALL:
        for method in args.methods:
            atomic_torch_save(draw(data["observations"][:n], method),
                              root / "samples" / f"n_{n}_{method}.pt")
    for index in range(30):
        for method in args.methods:
            atomic_torch_save(draw(data["observations"][index:index + 1], method),
                              root / "samples" / f"single_{index:02d}_{method}.pt")


def fig4_plot() -> None:
    root = ARTIFACTS / "figure4"; data = torch.load(root / "data.pt", weights_only=True)
    truth = data["truth_theta"][0]
    fig = plt.figure(figsize=(14, 10.5)); grid = fig.add_gridspec(
        len(METHODS), 4, width_ratios=(1.2, 1, 1, 1)
    )
    metric_ax = fig.add_subplot(grid[:, 0])
    for method in METHODS:
        metric = [float(scaled_mmd_to_dirac(
            torch.load(root / "samples" / f"n_{n}_{method}.pt", weights_only=True), truth).mean())
                  for n in N_TALL]
        st = STYLE[method]; metric_ax.plot(N_TALL, metric, color=st["color"], marker=st["marker"], label=st["label"])
    metric_ax.set(xlabel="observations n", ylabel="mean marginal MMD to truth", xticks=N_TALL)
    metric_ax.legend(frameon=False, fontsize=8)
    colors = plt.get_cmap("viridis")(np.linspace(.15, .9, len(N_TALL)))
    for row, method in enumerate(METHODS):
        for coordinate, label in enumerate((r"$C$", r"$\mu$", r"$\sigma$")):
            ax = fig.add_subplot(grid[row, coordinate + 1])
            for index in range(30):
                draws = torch.load(root / "samples" / f"single_{index:02d}_{method}.pt", weights_only=True)
                ax.hist(draws[:, coordinate], bins=40, density=True, histtype="step",
                        color=colors[0], alpha=.08)
            for n, color in zip(N_TALL[1:], colors[1:]):
                draws = torch.load(root / "samples" / f"n_{n}_{method}.pt", weights_only=True)
                ax.hist(draws[:, coordinate], bins=45, density=True, histtype="step",
                        linewidth=1.5, color=color, label=rf"$n={n}$")
            ax.axvline(float(truth[coordinate]), color="black", linestyle="--"); ax.set_xlabel(label); ax.set_yticks([])
            if coordinate == 0: ax.set_ylabel(STYLE[method]["label"])
    fig.axes[-1].legend(frameon=False, fontsize=7)
    fig.savefig(root / "figure4.png", dpi=250, bbox_inches="tight"); fig.savefig(root / "figure4.pdf", bbox_inches="tight"); plt.close(fig)


def dispatch(args, device) -> None:
    stages = ("prepare", "train", "sample", "plot") if args.stage == "all" else (args.stage,)
    functions = {
        1: {"sample": lambda: fig1_sample(args, device), "plot": fig1_plot},
        2: {"sample": lambda: fig2_sample(args, device), "plot": fig2_plot},
        3: {"prepare": lambda: fig3_prepare(args, device), "train": lambda: fig3_train(args, device),
            "sample": lambda: fig3_sample(args, device), "plot": fig3_plot},
        4: {"prepare": lambda: fig4_prepare(args), "train": lambda: fig4_train(args, device),
            "sample": lambda: fig4_sample(args, device), "plot": fig4_plot},
    }
    for stage in stages:
        if stage in functions[args.figure]: functions[args.figure][stage]()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--stage", choices=("prepare", "train", "sample", "plot", "all"), default="all")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--figure4-samples", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5); parser.add_argument("--cases", type=int, default=25)
    parser.add_argument("--n-train", type=int, nargs="+", default=list(N_TRAIN))
    parser.add_argument("--tasks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=5000); parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--simulation-batch-size", type=int, default=256)
    parser.add_argument("--jrnnm-backend", choices=("exact", "torch"), default="exact")
    parser.add_argument("--clip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage != "plot": reserve_gpu(args.device)
    dispatch(args, resolve_device(args.device if args.stage != "plot" else "cpu"))


if __name__ == "__main__":
    main()
