#!/usr/bin/env python3
"""Experiment 2 of Arruda et al. (ICLR 2026, arXiv:2505.14429): hierarchical AR(1) grid.

The model (their Appendix A.7, checked against
github.com/bayesflow-org/hierarchical-abi experiments/problems/ar1/ar1_grid.py)
----------------------------------------------------------------------------
    alpha, beta, log_sigma ~ N(0, 1)                      global
    theta_raw_j ~ N(0, exp(log_sigma))   (a standard deviation)   local
    theta_j = 2 sigmoid(beta + theta_raw_j) - 1
    y_j0 ~ N(0, 0.1^2),  y_jt ~ N(alpha + theta_j y_j,t-1, 0.1^2),  t = 1..4

J = grid size = 4x4, 16x16, 64x64, 128x128 groups. The network is trained on
single-group simulations (their budget: 10,000) and composed over the J groups.

The reference, without Stan
---------------------------
y_j0 does not depend on any parameter, and the rest of the likelihood is
Gaussian in (alpha, theta_j) with five sufficient statistics per group. Each
theta_j is one-dimensional, so the group marginal likelihood
p(y_j | alpha, beta, sigma) is a 1-D integral, done on a fine grid in theta
with the change-of-variables prior density. The 3-D global posterior is then
sampled by self-normalized importance sampling from a Student-t Laplace
proposal (the ESS is recorded per dataset), and theta_j | globals, y_j is drawn
from the same grid. This is exact up to grid and Monte-Carlo error.

Metrics: their Appendix A.4 (RMSE normalized by the range of the true values,
contraction), plus mean error and width ratio against the reference.

Usage:
    python ar1_grid.py --train
    python ar1_grid.py --grids 4 16 --methods gauss_hierarchical gauss_jacobian
    python ar1_grid.py --report
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
OUT = ROOT / "artifacts" / "ar1_grid"

T = 5
NOISE = 0.1
GLOBALS = 3
LOCAL_INDEX = 3
NODES = GLOBALS + 1 + T
HIERARCHY = [0, 1, 2]
EPS = 1e-5
SDE_KWARGS = {"sde_type": "vesde", "sigma": 8.0}
NETWORK = {"hidden_size": 64, "depth": 4, "num_heads": 4, "mlp_ratio": 2}
TIMESTEPS = 100
# Network rows (groups x draws) per sampler call; see common.sample_chunked.
MAX_ROWS = 200_000
JACOBIAN_MAX_ROWS = 6_000
# Row cap for the no-grad FD Jacobian: 96k rows peaked at 5.9 GB (toy J=1000)
# and 2.5 GB (AR(1) 64x64); time per draw is flat above ~12k rows.
FD_MAX_ROWS = int(os.environ.get("FD_MAX_ROWS", 96_000))
GLOBAL_NAMES = ("alpha", "beta", "log_sigma")
# Spread of the standardized theta_raw (robustly scaled to ~1, heavy tails).
LOCAL_SIGMA_DATA = 1.5

METHODS = {
    "gauss": {"method": "dpm", "order": 2, "correction": "gauss",
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
    # benchmarks/paper: the library's standard method as it ships (FD Jacobian, refresh 9,
    # float32 eigh), under its own name so the archived r9 results keep theirs
    "paper_final": {"method": "dpm", "order": 2, "correction": "gauss_jacobian_FD",
                    "jacobian_refresh": 9, "max_rows": FD_MAX_ROWS},
    "uncorrected": {"method": "dpm", "order": 2, "correction": "uncorrected"},
}


# ---------------------------------------------------------------------------
# Simulator and standardization
# ---------------------------------------------------------------------------

def simulate_groups(rng, alpha, theta):
    """y for one group per entry of theta (alpha broadcast), shape (len, T)."""
    theta = np.asarray(theta, dtype=np.float64)
    noise = rng.normal(0.0, NOISE, size=(theta.size, T))
    y = np.zeros((theta.size, T))
    y[:, 0] = noise[:, 0]
    for t in range(1, T):
        y[:, t] = alpha + theta * y[:, t - 1] + noise[:, t]
    return y


def simulate_single(count, seed):
    """(globals, theta_raw, y) for `count` independent one-group simulations."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((count, GLOBALS))
    raw = rng.standard_normal(count) * np.exp(g[:, 2])
    theta = 2.0 / (1.0 + np.exp(-(g[:, 1] + raw))) - 1.0
    y = np.zeros((count, T))
    noise = rng.normal(0.0, NOISE, size=(count, T))
    y[:, 0] = noise[:, 0]
    for t in range(1, T):
        y[:, t] = g[:, 0] + theta * y[:, t - 1] + noise[:, t]
    return g, raw, y


def normalization():
    """Fixed standardization constants from a large prior-predictive draw."""
    _, raw, y = simulate_single(200_000, seed=12345)
    # theta_raw has a scale-mixture marginal with very heavy tails, so use a
    # robust scale (MAD-based) rather than the std it would otherwise blow up.
    raw_scale = 1.4826 * float(np.median(np.abs(raw - np.median(raw))))
    return {"y_mean": y.mean(axis=0), "y_std": y.std(axis=0),
            "raw_scale": raw_scale}


NORM = None


def norm():
    global NORM
    if NORM is None:
        NORM = normalization()
    return NORM


def to_network(g, raw, y):
    n = norm()
    return np.concatenate([g, (raw / n["raw_scale"])[:, None],
                           (y - n["y_mean"]) / n["y_std"]], axis=1)


def dataset(grid, seed):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(GLOBALS)
    raw = rng.standard_normal(grid * grid) * math.exp(g[2])
    theta = 2.0 / (1.0 + np.exp(-(g[1] + raw))) - 1.0
    return g, raw, theta, simulate_groups(rng, g[0], theta)


# ---------------------------------------------------------------------------
# Reference posterior
# ---------------------------------------------------------------------------

THETA_POINTS = 4000


def sufficient_statistics(y):
    prev, curr = y[:, :-1], y[:, 1:]
    return np.stack([(curr**2).sum(1), curr.sum(1), prev.sum(1),
                     (curr * prev).sum(1), (prev**2).sum(1)], axis=1)


class Reference:
    """Exact-by-quadrature global marginal posterior and local conditionals."""

    def __init__(self, y, device):
        import torch
        self.torch = torch
        self.device = device
        self.stats = torch.as_tensor(sufficient_statistics(y), device=device,
                                     dtype=torch.float64)
        # Midpoint grid on (-1, 1) for theta, dense near the boundaries by
        # building it in z = logit((theta+1)/2) on a wide, fine range.
        z = torch.linspace(-16.0, 16.0, THETA_POINTS, device=device,
                           dtype=torch.float64)
        self.z = z
        self.dz = float(z[1] - z[0])
        self.theta = 2.0 * torch.sigmoid(z) - 1.0

    def _log_lik(self, alpha):
        """log p(y_j,1:4 | alpha, theta) on the grid: (A, J, G) for alphas (A,)."""
        s = self.stats
        a = alpha[:, None, None]
        th = self.theta[None, None, :]
        syy, s1, sx, sxy, sxx = (s[:, k][None, :, None] for k in range(5))
        quad = (syy - 2 * a * s1 - 2 * th * sxy + (T - 1) * a**2
                + 2 * a * th * sx + th**2 * sxx)
        return -0.5 * quad / NOISE**2

    def log_marginal(self, globals_, chunk=None):
        """sum_j log p(y_j | alpha, beta, sigma) + log prior, for (A, 3)."""
        torch = self.torch
        g = torch.as_tensor(globals_, device=self.device, dtype=torch.float64)
        groups = self.stats.shape[0]
        chunk = chunk or max(1, int(2e7 // (groups * THETA_POINTS)))
        out = []
        for start in range(0, g.shape[0], chunk):
            block = g[start:start + chunk]
            alpha, beta, log_sigma = block[:, 0], block[:, 1], block[:, 2]
            sigma = torch.exp(log_sigma)[:, None]
            log_prior_z = (-0.5 * ((self.z[None, :] - beta[:, None]) / sigma) ** 2
                           - torch.log(sigma) - 0.5 * math.log(2 * math.pi))
            total = self._log_lik(alpha) + log_prior_z[:, None, :]
            marginal = torch.logsumexp(total, dim=2) + math.log(self.dz)
            prior = -0.5 * (block**2).sum(1)
            out.append(marginal.sum(1) + prior)
        return torch.cat(out)

    def sample(self, draws, seed, proposals=4000):
        """Self-normalized IS from a Student-t Laplace fit; returns (globals, ESS)."""
        torch = self.torch
        start = torch.zeros(1, 3, device=self.device, dtype=torch.float64,
                            requires_grad=True)
        optimizer = torch.optim.LBFGS([start], max_iter=200,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = -self.log_marginal(start, chunk=1)[0]
            loss.backward()
            return loss
        optimizer.step(closure)
        mode = start.detach()[0]
        hessian = torch.autograd.functional.hessian(
            lambda v: -self.log_marginal(v[None], chunk=1)[0], mode)
        covariance = torch.linalg.inv(hessian + 1e-9 * torch.eye(3, device=self.device,
                                                                 dtype=torch.float64))
        covariance = 0.5 * (covariance + covariance.T)
        # Inflate so the proposal is wider than the target in every direction.
        covariance = 1.5 * covariance
        generator = torch.Generator(device=self.device).manual_seed(seed)
        df = 5.0
        chol = torch.linalg.cholesky(covariance)
        normal = torch.randn(proposals, 3, generator=generator,
                             device=self.device, dtype=torch.float64)
        chi2 = torch.distributions.Chi2(torch.tensor(df, device=self.device,
                                                     dtype=torch.float64))
        scale = torch.sqrt(df / chi2.sample((proposals,)))
        points = mode + scale[:, None] * (normal @ chol.T)
        centred = torch.linalg.solve_triangular(chol, (points - mode).T, upper=False).T
        log_q = -0.5 * (df + 3) * torch.log1p((centred**2).sum(1) / df)
        log_w = self.log_marginal(points) - log_q
        weights = torch.softmax(log_w, dim=0)
        ess = float(1.0 / (weights**2).sum())
        index = torch.multinomial(weights, draws, replacement=True,
                                  generator=generator)
        return points[index].cpu().numpy(), ess

    def sample_locals(self, globals_, seed):
        """theta_j | globals, y_j on the grid, one draw per global draw: (S, J)."""
        torch = self.torch
        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = []
        for row in torch.as_tensor(globals_, device=self.device, dtype=torch.float64):
            sigma = torch.exp(row[2])
            log_prior = -0.5 * ((self.z - row[1]) / sigma) ** 2
            logits = self._log_lik(row[:1])[0] + log_prior[None, :]
            index = torch.multinomial(torch.softmax(logits, dim=1), 1,
                                      generator=generator)[:, 0]
            out.append(self.theta[index])
        return torch.stack(out).cpu().numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def model_directory(train_samples):
    config = f"h{NETWORK['hidden_size']}d{NETWORK['depth']}"
    return OUT / "models" / f"{config}_n{train_samples}_eps{EPS:g}_precond"


def network(device, checkpoint=None):
    import common
    # Baseline only on the globals: the standardized theta_raw marginal is a
    # heavy-tailed scale mixture with no Gaussian closed form.
    return common.build(NODES, SDE_KWARGS, NETWORK, device, EPS,
                        prior_std=[1.0] * GLOBALS, latent_indices=HIERARCHY,
                        sigma_data=[1.0] * GLOBALS + [LOCAL_SIGMA_DATA] + [1.0] * T,
                        checkpoint=checkpoint)


def train(device, train_samples, epochs, patience):
    import torch
    import common

    torch.manual_seed(7)
    data = to_network(*simulate_single(train_samples, seed=7))
    validation = to_network(*simulate_single(max(2000, train_samples // 10), seed=8))
    data = torch.as_tensor(data, dtype=torch.float32)
    validation = torch.as_tensor(validation, dtype=torch.float32)
    split = GLOBALS + 1
    model = network(device)
    directory = model_directory(train_samples)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with common.floor_training_eps(EPS):
        # six_weighted menu: the measured best mask recipe for hierarchical
        # composition (posterior-side conditionals weighted 7:1).
        model.train(theta=data[:, :split], x=data[:, split:],
                    theta_val=validation[:, :split], x_val=validation[:, split:],
                    batch_size=256, max_epochs=epochs,
                    early_stopping_patience=patience, lr=5e-4,
                    time_sampling="log_sigma", mask_menu="six",
                    mask_roles=(GLOBALS, 1), mask_weights=[7, 7, 7, 1, 1, 1],
                    device=device, verbose=True, path=str(directory))
    # The validation loss is dominated by denoising noise, so the Trainer's
    # best-validation checkpoint is typically an early, lucky epoch. Keep the
    # final weights (schedule-free eval weights after the last validation).
    model.save(path=str(directory), name="Model_final")
    print(f"trained in {time.perf_counter() - started:.0f}s -> {directory}")


def load(device, train_samples):
    import common
    return common.freeze(network(device, model_directory(train_samples) / "Model_final.pt"))


# ---------------------------------------------------------------------------
# Sampling and metrics
# ---------------------------------------------------------------------------

def run_method(model, name, y, num_samples, device, seed):
    import torch
    import common
    from compass import MultiObsSampler

    n = norm()
    groups = len(y)
    kwargs = dict(METHODS[name])
    max_rows = kwargs.pop("max_rows", JACOBIAN_MAX_ROWS
                          if name.startswith("gauss_jacobian") else MAX_ROWS)
    data = torch.as_tensor((y - n["y_mean"]) / n["y_std"], dtype=torch.float32)
    mask = torch.cat([torch.zeros(GLOBALS + 1), torch.ones(T)])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    samples = common.sample_chunked(
        MultiObsSampler(model).sample, groups, num_samples,
        max_rows=max_rows,
        world_size=1, data=data, condition_mask=mask, timesteps=TIMESTEPS,
        eps=EPS, hierarchy=HIERARCHY,
        prior=(torch.zeros(GLOBALS), torch.ones(GLOBALS)),
        local_prior=(torch.zeros(1), torch.ones(1) * 3.0),
        device=device, verbose=False, **kwargs,
    ).numpy().astype(np.float64)
    seconds = time.perf_counter() - started
    globals_ = samples[0, :, :GLOBALS]
    raw = samples[:, :, LOCAL_INDEX].T * n["raw_scale"]          # (S, J)
    theta = 2.0 / (1.0 + np.exp(-(globals_[:, 1:2] + raw))) - 1.0
    return globals_, theta, seconds


def compare(draws_g, draws_l, reference_g, reference_l, true_g, true_theta):
    ref_mean, ref_sd = reference_g.mean(0), reference_g.std(0, ddof=1)
    return {
        "global_mean": draws_g.mean(0).tolist(),
        "global_sd": draws_g.std(0, ddof=1).tolist(),
        "global_rmse": np.sqrt(((draws_g - true_g) ** 2).mean(0)).tolist(),
        "global_mean_error_sd": (np.abs(draws_g.mean(0) - ref_mean) / ref_sd).tolist(),
        "global_width_ratio": (draws_g.std(0, ddof=1) / ref_sd).tolist(),
        "global_contraction": np.clip(1.0 - draws_g.var(0, ddof=1), 0, 1).tolist(),
        "local_rmse": float(np.sqrt(((draws_l - true_theta) ** 2).mean(0)).mean()),
        "local_mean_error_sd": float(np.median(
            np.abs(draws_l.mean(0) - reference_l.mean(0))
            / np.maximum(reference_l.std(0, ddof=1), 1e-6))),
        "local_width_ratio": float(np.median(
            draws_l.std(0, ddof=1) / np.maximum(reference_l.std(0, ddof=1), 1e-6))),
    }


def reference_path(grid, index):
    return OUT / "reference" / f"J{grid}x{grid}_{index}.npz"


def reference_for(grid, index, device, draws=1000):
    path = reference_path(grid, index)
    if path.exists():
        stored = np.load(path)
        return stored["globals"], stored["locals"], float(stored["ess"])
    _, _, _, y = dataset(grid, seed=100 * grid + index)
    reference = Reference(y, device)
    started = time.perf_counter()
    globals_, ess = reference.sample(draws, seed=index)
    locals_ = reference.sample_locals(globals_[:200], seed=index)
    print(f"  reference J={grid}x{grid} [{index}] ESS={ess:.0f} "
          f"({time.perf_counter() - started:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, globals=globals_, locals=locals_, ess=ess)
    return globals_, locals_, ess


def result_path(name, grid, train_samples):
    return OUT / "results" / f"n{train_samples}" / f"{name}_J{grid}x{grid}.json"


def summarize(rows, name, grid):
    ok = [row for row in rows if "global_rmse" in row]
    true = np.array([row["true_global"] for row in rows])
    span = np.maximum(true.max(0) - true.min(0), 1e-12)
    summary = {"method": name, "grid": grid, "datasets": len(rows),
               "failed": len(rows) - len(ok)}
    if not ok:
        return summary
    stack = {key: np.array([row[key] for row in ok]) for key in ok[0]
             if key not in ("true_global", "error")}
    summary["global_rmse_norm"] = float((np.median(stack["global_rmse"], 0) / span).mean())
    summary["global_rmse_per_param"] = np.median(stack["global_rmse"], 0).tolist()
    for key in ("global_contraction", "global_mean_error_sd", "global_width_ratio"):
        summary[key] = float(np.median(stack[key], 0).mean())
        summary[key + "_per_param"] = np.median(stack[key], 0).tolist()
    for key in ("local_rmse", "local_mean_error_sd", "local_width_ratio", "seconds"):
        summary[key] = float(np.median(stack[key]))
    return summary


def evaluate(model, name, grid, datasets, num_samples, device, train_samples):
    path = result_path(name, grid, train_samples)
    if path.exists():
        print(f"skip {path.name} (exists)")
        return
    rows = []
    for index in range(datasets):
        true_g, _, true_theta, y = dataset(grid, seed=100 * grid + index)
        ref_g, ref_l, ess = reference_for(grid, index, device)
        row = {"true_global": true_g.tolist(), "reference_ess": ess}
        if name == "reference":
            draws_g, draws_l, seconds = ref_g[:num_samples], ref_l, 0.0
        else:
            try:
                draws_g, draws_l, seconds = run_method(model, name, y, num_samples,
                                                       device, seed=index)
            except Exception as error:
                print(f"  {name} J={grid}x{grid} [{index}] failed: {error!r}")
                row["error"] = repr(error)
                rows.append(row)
                continue
        if not (np.isfinite(draws_g).all() and np.isfinite(draws_l).all()):
            row["error"] = "non-finite draws"
            rows.append(row)
            continue
        # The reference is compared against itself on disjoint halves only when
        # it is the arm; otherwise against the full reference.
        if name == "reference":
            row.update(compare(ref_g[:500], ref_l[:100], ref_g[500:], ref_l[100:],
                               true_g, true_theta))
        else:
            row.update(compare(draws_g, draws_l, ref_g, ref_l, true_g, true_theta))
        row["seconds"] = seconds
        rows.append(row)
        print(f"  {name} J={grid}x{grid} [{index}] "
              f"mean_err={np.round(row['global_mean_error_sd'], 2)}sd "
              f"width={np.round(row['global_width_ratio'], 2)} "
              f"local_rmse={row['local_rmse']:.3f} t={seconds:.1f}s", flush=True)
    summary = summarize(rows, name, grid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "rows": rows}))
    print(json.dumps({k: v for k, v in summary.items() if "per_param" not in k}),
          flush=True)


def report(train_samples):
    folder = OUT / "results" / f"n{train_samples}"
    summaries = [json.loads(p.read_text())["summary"] for p in folder.glob("*.json")]
    order = ["reference"] + list(METHODS) + ["bayesflow_arruda"]
    summaries.sort(key=lambda s: (s["grid"], order.index(s["method"])
                                  if s["method"] in order else 99))
    lines = ["| grid | method | global RMSE (norm) | contraction | "
             "|mean err|/ref sd (α, β, logσ) | width ratio (α, β, logσ) | "
             "local RMSE θ | local |mean err|/sd | failed | s/dataset |",
             "|" + "---|" * 10]

    def triple(values):
        return ", ".join(f"{v:.2f}" for v in values)
    for s in summaries:
        if "global_rmse_norm" not in s:
            lines.append(f"| {s['grid']}x{s['grid']} | {s['method']} | - | - | - | - "
                         f"| - | - | {s['failed']}/{s['datasets']} | - |")
            continue
        lines.append(
            f"| {s['grid']}x{s['grid']} | {s['method']} | {s['global_rmse_norm']:.3f} | "
            f"{s['global_contraction']:.3f} | "
            f"{triple(s['global_mean_error_sd_per_param'])} | "
            f"{triple(s['global_width_ratio_per_param'])} | {s['local_rmse']:.3f} | "
            f"{s['local_mean_error_sd']:.2f} | {s['failed']}/{s['datasets']} | "
            f"{s['seconds']:.1f} |")
    table = "\n".join(lines)
    (folder / "TABLE.md").write_text(table + "\n")
    print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-samples", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--grids", type=int, nargs="+", default=[4, 16, 64, 128])
    parser.add_argument("--methods", nargs="+", default=["reference", *METHODS],
                        choices=["reference", *METHODS])
    parser.add_argument("--datasets", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", action="store_true")
    arguments = parser.parse_args()

    if arguments.report:
        report(arguments.train_samples)
        return
    if arguments.device == "cuda":
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=1)
    if arguments.train:
        train(arguments.device, arguments.train_samples, arguments.epochs,
              arguments.patience)
        return
    model = None
    if any(name != "reference" for name in arguments.methods):
        model = load(arguments.device, arguments.train_samples)
    for grid in arguments.grids:
        for name in arguments.methods:
            evaluate(model, name, grid, arguments.datasets, arguments.num_samples,
                     arguments.device, arguments.train_samples)
    report(arguments.train_samples)


if __name__ == "__main__":
    main()
