#!/usr/bin/env python3
"""Experiment 1 of Arruda et al. (ICLR 2026, arXiv:2505.14429), run with COMPASS's rules.

The benchmark (their Appendix A.6)
----------------------------------
    eta ~ N(0, sigma^2 I_10),   Y_j | eta ~ N(eta, sigma^2 I_10),   sigma = 0.1

with J conditionally independent observations and the analytic posterior
N(sum_j Y_j / (J+1), sigma^2 / (J+1) I). There are no local latents: every
latent coordinate is shared, so this grades the composition rule alone.

Units. Everything here is divided by sigma, i.e. eta ~ N(0, I), Y | eta ~ N(eta, I).
That is an affine change of variables, so their KL divergence, contraction and
calibration error are unchanged; RMSE is reported in prior-sd units (= their
units divided by sigma), next to the same statistic for *exact* posterior draws.

Why the diffusion floor is patched. ``Trainer`` hard-codes eps=1e-3, which puts
the smallest trained noise scale at lambda ~ 0.032. The J=1000 posterior sd is
0.032 and J=10,000 is 0.010, so with the stock floor neither can be resolved --
the terminal noise would be as wide as the posterior. Training and sampling both
use EPS=1e-5 (lambda ~ 0.0032) here, by wrapping ``Trainer._sample_timesteps``
for the duration of training; nothing in ``compass`` is modified.

The arms (all with the same learned network)
--------------------------------------------
    gauss               DPM2 + diagonal GAUSS (Linhart et al.), churn default
    full_gaussian       DPM2 + full-covariance GAUSS (their Algorithm 2)
    gauss_hierarchical  DPM2 + pilot-covariance hierarchical GAUSS
    gauss_jacobian      DPM2 + pilot-free Jacobian GAUSS
    langevin_fnpe       annealed Langevin + F-NPSE (Geffner et al.)
    (Arruda et al.'s own method is run through BayesFlow by
    bayesflow_arruda.py and reported here as ``bayesflow_arruda``.)
    uncorrected         DPM2 + sum of scores with the diffused prior (control)

Metrics follow their Appendix A.4, plus the mean error and width ratio in
posterior-sd units, and every metric is also computed on *exact* posterior
draws of the same size so the Monte-Carlo floor is visible.

Usage:
    python gaussian_toy.py --train
    python gaussian_toy.py --observations 10 100 1000 --methods gauss gauss_jacobian
    python gaussian_toy.py --report
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "gaussian_toy"

DIM = 10
NODES = 2 * DIM
HIERARCHY = list(range(DIM))
EPS = 1e-5
SDE_KWARGS = {"sde_type": "vesde", "sigma": 8.0}
NETWORK = {"hidden_size": 64, "depth": 4, "num_heads": 4, "mlp_ratio": 2}
TRAIN_SAMPLES = int(os.environ.get("TOY_TRAIN", 200_000))
TIMESTEPS = 100
# Network rows (observations x draws) per sampler call; draws are split into
# independent chunks above this. The Jacobian rule runs D forward-mode passes.
MAX_ROWS = 50_000
JACOBIAN_MAX_ROWS = 6_000
# Row cap for the no-grad FD Jacobian: 96k rows peaked at 5.9 GB (toy J=1000)
# and 2.5 GB (AR(1) 64x64); time per draw is flat above ~12k rows.
FD_MAX_ROWS = int(os.environ.get("FD_MAX_ROWS", 96_000))

METHODS = {
    "gauss": {"method": "dpm", "order": 2, "correction": "gauss",
              "precision_est_samples": 500},
    "full_gaussian": {"method": "dpm", "order": 2, "correction": "full_gaussian",
                      "precision_est_samples": 500},
    "gauss_hierarchical": {"method": "dpm", "order": 2,
                           "correction": "gauss_hierarchical",
                           "precision_est_samples": 500},
    "gauss_jacobian": {"method": "dpm", "order": 2, "correction": "gauss_jacobian",
                       "jacobian_refresh": 5},
    # Finite-difference Jacobian, refreshed every step at high noise and every
    # 10 steps below lambda 0.3 (the Partial_Pooling ladder setting).
    "gauss_jacobian_FD": {"method": "dpm", "order": 2, "correction": "gauss_jacobian_FD",
                          "jacobian_refresh": 1, "jacobian_refresh_low_noise": 10,
                          "jacobian_refresh_lambda": 0.3},
    # Same refresh schedule as "gauss_jacobian", to separate FD cost from refresh.
    "gauss_jacobian_FD_r5": {"method": "dpm", "order": 2, "correction": "gauss_jacobian_FD",
                             "jacobian_refresh": 5},
    # Refresh 9 (odd: DPM2 makes two calls per step), at the JVP-era row cap and
    # at the larger cap the no-grad FD Jacobian fits ("max_rows", see run_method).
    "gauss_jacobian_FD_r9": {"method": "dpm", "order": 2, "correction": "gauss_jacobian_FD",
                             "jacobian_refresh": 9},
    "gauss_jacobian_FD_r9_big": {"method": "dpm", "order": 2,
                                 "correction": "gauss_jacobian_FD",
                                 "jacobian_refresh": 9, "max_rows": FD_MAX_ROWS},
    "langevin_fnpe": {"method": "langevin", "correction": "fnpe",
                      "corrector_steps": 10, "snr": 0.2},
    "uncorrected": {"method": "dpm", "order": 2, "correction": "uncorrected"},
}


def condition_mask():
    import torch
    return torch.cat([torch.zeros(DIM), torch.ones(DIM)])


def simulate(count, generator):
    import torch
    eta = torch.randn(count, DIM, generator=generator)
    return eta, eta + torch.randn(count, DIM, generator=generator)


def dataset(observations, seed):
    """One benchmark dataset: a true eta from the prior and J observations."""
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal(DIM)
    y = eta + rng.standard_normal((observations, DIM))
    return eta, y


def exact_posterior(y):
    """Mean and (isotropic) sd of p(eta | Y_1..J) in standardized units."""
    return y.sum(axis=0) / (len(y) + 1), 1.0 / math.sqrt(len(y) + 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def model_directory():
    config = f"h{NETWORK['hidden_size']}d{NETWORK['depth']}"
    return OUT / "models" / f"{config}_n{TRAIN_SAMPLES}_eps{EPS:g}_precond"


def network(device, checkpoint=None):
    import common
    # sigma_data: prior sd 1 for eta, sqrt(2) for the marginal of Y.
    return common.build(NODES, SDE_KWARGS, NETWORK, device, EPS,
                        prior_std=[1.0] * DIM, latent_indices=HIERARCHY,
                        sigma_data=[1.0] * DIM + [2.0 ** 0.5] * DIM,
                        checkpoint=checkpoint)


def train(device, epochs, patience):
    import torch
    import common
    from compass import ConditionalMenu

    generator = torch.Generator().manual_seed(7)
    torch.manual_seed(7)
    theta, x = simulate(TRAIN_SAMPLES, generator)
    theta_val, x_val = simulate(20_000, generator)
    model = network(device)
    # Weight the one conditional the benchmark evaluates, p(eta | Y). Under the
    # stock Bernoulli(0.33) draw on 20 nodes it has probability ~1e-6.
    menu = ConditionalMenu(
        model.schema, {"eta": range(DIM), "y": range(DIM, NODES)},
        [("posterior", "y")], bernoulli_fraction=0.1,
    )
    directory = model_directory()
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with common.floor_training_eps(EPS):
        model.train(theta=theta, x=x, theta_val=theta_val, x_val=x_val,
                    batch_size=1024, max_epochs=epochs,
                    early_stopping_patience=patience, lr=5e-4,
                    time_sampling="log_sigma", mask_sampler=menu,
                    device=device, verbose=True, path=str(directory))
    # The validation loss is dominated by denoising noise, so the Trainer's
    # best-validation checkpoint is typically an early, lucky epoch. Keep the
    # final weights (schedule-free eval weights after the last validation).
    model.save(path=str(directory), name="Model_final")
    print(f"trained in {time.perf_counter() - started:.0f}s -> {directory}")


def load(device):
    import common
    return common.freeze(network(device, model_directory() / "Model_final.pt"))


# ---------------------------------------------------------------------------
# Metrics (Arruda et al. Appendix A.4)
# ---------------------------------------------------------------------------

ALPHAS = np.linspace(0.005, 0.995, 100)


def gaussian_kl(samples, mean_p, sd_p):
    """KL(q || p) with q the moment-matched Gaussian of the samples."""
    mean_q = samples.mean(axis=0)
    cov_q = np.cov(samples, rowvar=False)
    var_p = sd_p**2
    sign, logdet_q = np.linalg.slogdet(cov_q)
    if sign <= 0:
        return float("inf")
    return float(0.5 * (DIM * math.log(var_p) - logdet_q - DIM
                        + np.trace(cov_q) / var_p
                        + ((mean_q - mean_p) ** 2).sum() / var_p))


def dataset_metrics(samples, eta, y):
    mean_p, sd_p = exact_posterior(y)
    mean_q = samples.mean(axis=0)
    sd_q = samples.std(axis=0, ddof=1)
    lower = np.quantile(samples, 0.5 - ALPHAS / 2, axis=0)      # (alphas, dims)
    upper = np.quantile(samples, 0.5 + ALPHAS / 2, axis=0)
    return {
        "kl": gaussian_kl(samples, mean_p, sd_p),
        "rmse": float(np.sqrt(((samples - eta) ** 2).mean(axis=0)).mean()),
        "contraction": float(np.clip(1.0 - sd_q**2, 0.0, 1.0).mean()),
        "mean_error_sd": float((np.abs(mean_q - mean_p) / sd_p).mean()),
        "width_ratio": float((sd_q / sd_p).mean()),
        # (alphas, dims) coverage indicators; pooled across datasets later.
        "covered": ((lower <= eta) & (eta <= upper)).astype(float).tolist(),
        "nonfinite": int((~np.isfinite(samples)).any(axis=1).sum()),
    }


def summarize(rows):
    finite = [row for row in rows
              if math.isfinite(row["kl"]) and row.get("covered") is not None]
    summary = {"datasets": len(rows), "failed": len(rows) - len(finite)}
    for key in ("kl", "rmse", "contraction", "mean_error_sd", "width_ratio",
                "seconds"):
        values = np.array([row[key] for row in finite if key in row])
        if len(values):
            median = float(np.median(values))
            summary[key] = median
            summary[key + "_mad"] = float(np.median(np.abs(values - median)))
    if finite:
        # (alphas, dims); older result files stored (alphas, 1, dims).
        coverage = np.mean([np.reshape(row["covered"], (len(ALPHAS), DIM))
                            for row in finite], axis=0)
        summary["calibration_error"] = float(
            np.median(np.abs(coverage - ALPHAS[:, None]), axis=0).mean()
        )
    return summary


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def run_method(model, name, y, num_samples, device, seed):
    import torch
    import common
    from compass import MultiObsSampler

    observations = len(y)
    kwargs = dict(METHODS[name])
    max_rows = kwargs.pop("max_rows", JACOBIAN_MAX_ROWS
                          if name.startswith("gauss_jacobian") else MAX_ROWS)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sampler = MultiObsSampler(model)
    started = time.perf_counter()
    samples = common.sample_chunked(
        sampler.sample, observations, num_samples,
        max_rows=max_rows,
        world_size=1, data=torch.as_tensor(y, dtype=torch.float32),
        condition_mask=condition_mask(), timesteps=TIMESTEPS, eps=EPS,
        hierarchy=HIERARCHY,
        prior=(torch.zeros(DIM), torch.ones(DIM)) if name != "full_gaussian"
        else (torch.zeros(DIM), torch.eye(DIM)),
        device=device, verbose=False, **kwargs,
    )
    seconds = time.perf_counter() - started
    shared = samples.numpy()[0, :, :DIM].astype(np.float64)
    return shared, seconds


def result_path(name, observations):
    return OUT / "results" / f"{name}_J{observations}.json"


def evaluate(model, name, observations, datasets, num_samples, device):
    path = result_path(name, observations)
    if path.exists():
        print(f"skip {path.name} (exists)")
        return
    rows = []
    for index in range(datasets):
        eta, y = dataset(observations, seed=1000 * observations + index)
        try:
            samples, seconds = run_method(model, name, y, num_samples, device,
                                          seed=index)
            finite = np.isfinite(samples).all(axis=1)
            row = dataset_metrics(samples[finite], eta, y) if finite.sum() > DIM \
                else {"kl": float("inf"), "covered": None}
            row["nonfinite"] = int((~finite).sum())
            row["seconds"] = seconds
        except Exception as error:  # a diverging rule is a result, not a crash
            print(f"  {name} J={observations} dataset {index}: {error!r}")
            row = {"kl": float("inf"), "error": repr(error), "covered": None}
        rows.append(row)
        print(f"  {name} J={observations} [{index}] kl={row['kl']:.3g} "
              f"mean_err={row.get('mean_error_sd', float('nan')):.3g}sd "
              f"width={row.get('width_ratio', float('nan')):.3g} "
              f"t={row.get('seconds', float('nan')):.1f}s", flush=True)
    summary = summarize(rows)
    summary.update(method=name, observations=observations,
                   num_samples=num_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "rows": rows}))
    print(json.dumps({k: v for k, v in summary.items()
                      if not k.endswith("_mad")}), flush=True)


def exact_floor(observations, datasets, num_samples):
    """Every metric on exact posterior draws: the Monte-Carlo floor."""
    path = result_path("exact", observations)
    rows = []
    for index in range(datasets):
        eta, y = dataset(observations, seed=1000 * observations + index)
        mean_p, sd_p = exact_posterior(y)
        draws = mean_p + sd_p * np.random.default_rng(index).standard_normal(
            (num_samples, DIM))
        row = dataset_metrics(draws, eta, y)
        row["seconds"] = 0.0
        rows.append(row)
    summary = summarize(rows)
    summary.update(method="exact", observations=observations,
                   num_samples=num_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "rows": rows}))


def report():
    summaries = []
    for path in sorted((OUT / "results").glob("*.json")):
        stored = json.loads(path.read_text())
        # Recompute from the per-dataset rows so metric fixes apply retroactively.
        summary = summarize(stored["rows"])
        summary.update({key: stored["summary"][key]
                        for key in ("method", "observations", "num_samples")})
        summaries.append(summary)
    order = ["exact"] + list(METHODS) + ["bayesflow_arruda"]
    summaries.sort(key=lambda s: (s["observations"],
                                  order.index(s["method"])
                                  if s["method"] in order else 99))
    header = ("| J | method | KL | RMSE | contraction | cal. err | "
              "|mean err|/sd | width ratio | failed | s/dataset |")
    lines = [header, "|" + "---|" * 10]
    for s in summaries:
        def cell(key, fmt="{:.3g}"):
            return fmt.format(s[key]) if key in s else "-"
        lines.append(
            f"| {s['observations']} | {s['method']} | {cell('kl')} | "
            f"{cell('rmse')} | {cell('contraction', '{:.3f}')} | "
            f"{cell('calibration_error', '{:.3f}')} | {cell('mean_error_sd')} | "
            f"{cell('width_ratio', '{:.3f}')} | {s['failed']}/{s['datasets']} | "
            f"{cell('seconds', '{:.1f}')} |")
    table = "\n".join(lines)
    (OUT / "TABLE.md").write_text(table + "\n")
    print(table)


def probe(device, states=512, seed=0):
    """Single-observation score vs the exact p_lam(eta | y) = N(y/2, 0.5 + lam^2).

    No composition anywhere: grades the network at the noise levels the
    composed sampler queries. A per-observation mean bias b shifts the composed
    mean by ~b/2 (absolute), i.e. by ~b/2 * sqrt(J+1) posterior sd at J.
    """
    import torch
    model = load(device)
    rng = np.random.default_rng(seed)
    mask = condition_mask().to(device).unsqueeze(0).repeat(states, 1)
    rows = []
    for lam in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        y = rng.standard_normal((states, DIM)) * math.sqrt(2.0)
        variance = 0.5 + lam**2
        eta = y / 2 + math.sqrt(variance) * rng.standard_normal((states, DIM))
        exact = -(eta - y / 2) / variance
        state = torch.tensor(np.concatenate([eta, y], 1), dtype=torch.float32,
                             device=device)
        t = model.sde.time_of_lambda(torch.tensor(float(lam))).reshape(1, 1).to(device)
        with torch.no_grad():
            raw = model.model(x=state, t=t.repeat(states, 1), c=mask)
            predicted = model.output_scale_function(t, raw)[:, :DIM]
        predicted = predicted.cpu().numpy().astype(np.float64)
        row = {"lambda": lam,
               "relative_error": float(np.sqrt(((predicted - exact) ** 2).sum()
                                               / (exact**2).sum())),
               "slope": float((predicted * exact).sum() / (exact**2).sum()),
               "mean_bias": float((predicted - exact).mean())}
        rows.append(row)
        print(json.dumps(row))
    (OUT / "score_probe.json").write_text(json.dumps(rows, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--observations", type=int, nargs="+",
                        default=[10, 100, 1000])
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        choices=list(METHODS))
    parser.add_argument("--datasets", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--probe", action="store_true")
    arguments = parser.parse_args()

    if arguments.report:
        report()
        return
    if arguments.device == "cuda":
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=1)
    if arguments.train:
        train(arguments.device, arguments.epochs, arguments.patience)
        return
    if arguments.probe:
        probe(arguments.device)
        return
    model = load(arguments.device)
    for observations in arguments.observations:
        exact_floor(observations, arguments.datasets, arguments.num_samples)
        for name in arguments.methods:
            evaluate(model, name, observations, arguments.datasets,
                     arguments.num_samples, arguments.device)
    report()


if __name__ == "__main__":
    main()
