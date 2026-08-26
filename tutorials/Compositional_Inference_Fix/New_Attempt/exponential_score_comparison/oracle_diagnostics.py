#!/usr/bin/env python3
"""Oracle and trajectory diagnostics for the exponential-score comparison.

This deliberately keeps the original ``compare.py`` results immutable.  It
answers five separate questions which the headline plot cannot distinguish:

1. Can the numerical sampler transport an *exact* finite-noise score?
2. How much error remains when the single-observation scores are analytical,
   but the finite-noise composition rule is still approximate?
3. At which noise level does each learned trajectory leave its intended path?
4. Do conclusions survive timestep / MCMC / endpoint convergence checks?
5. Are signed error, wall proximity, Jacobian error, or curl hiding behind a
   comparatively benign scalar RMSE?

The expensive full-joint oracle uses ``hierarchy.diffused_score`` directly.
The convergence sweep additionally uses a one-dimensional exact shared-
marginal oracle, which makes 50/100/200/400-step sweeps practical while still
testing the same DPM2 update and the hard-wall marginal shown in the paper.

Usage (one seed is the default and intended use):

    python oracle_diagnostics.py --device cuda --seed 0
    python oracle_diagnostics.py --replot

All rows are cached under ``artifacts/oracle_diagnostics`` and reused on a
restart.  ``--quick`` is a small smoke configuration, not a publishable run.
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import fftconvolve  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

from compass.MultiObsSampler import MultiObsSampler  # noqa: E402

import compare  # noqa: E402
import hierarchy  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "oracle_diagnostics"
METHOD_NAMES = ("gauss_hierarchical", "gauss_jacobian", "langevin_fnpe")
COLOURS = {
    "gauss_hierarchical": compare.CORAL,
    "gauss_jacobian": compare.TEAL,
    "langevin_fnpe": compare.GOLD,
    "exact_diffusion": compare.NAVY,
    "oracle_dpm": compare.BLUE,
    "oracle_langevin": "#8338EC",
    "analytic_gauss_hierarchical": "#C43B62",
    "analytic_gauss_jacobian": "#177E72",
    "analytic_fnpe": "#E69F00",
}


def configure_style():
    compare.configure_style()
    mpl.rcParams.update({"figure.figsize": (15.5, 4.7)})


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def scalar_w1(draws, reference):
    return float(wasserstein_distance(
        np.asarray(draws, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    ))


def sliced_w1(draws, reference, seed=0, projections=64):
    """Sliced W1 after coordinate-wise reference standardisation."""
    draws = np.asarray(draws, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = reference.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-10, scale, 1.0)
    draws = (draws - reference.mean(axis=0)) / scale
    reference = (reference - reference.mean(axis=0)) / scale
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, draws.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(np.mean([
        wasserstein_distance(draws @ direction, reference @ direction)
        for direction in directions
    ]))


def state_matrix(state):
    """Shared-once joint representation: (samples, 1 + observations locals)."""
    state = torch.as_tensor(state).detach().cpu().numpy()
    return np.concatenate([
        state[0, :, hierarchy.GLOBAL_INDEX, None],
        state[:, :, hierarchy.LOCAL_INDEX].T,
    ], axis=1)


def make_state(shared, local, x, device):
    shared = np.asarray(shared, dtype=np.float64)
    local = np.asarray(local, dtype=np.float64)
    n, samples = len(x), len(shared)
    state = torch.zeros(n, samples, hierarchy.NODES,
                        dtype=torch.float64, device=device)
    state[:, :, hierarchy.GLOBAL_INDEX] = torch.as_tensor(
        shared, dtype=torch.float64, device=device
    ).reshape(1, -1)
    state[:, :, hierarchy.LOCAL_INDEX] = torch.as_tensor(
        local.T.copy(), dtype=torch.float64, device=device
    )
    state[:, :, hierarchy.OBSERVED_INDEX] = torch.as_tensor(
        x, dtype=torch.float64, device=device
    ).reshape(-1, 1)
    return state


def exact_diffused_state(x, lam, count, seed, device):
    shared, local, _, _ = hierarchy.sample_diffused(x, lam, count, seed)
    return make_state(shared, local, x, device)


def clean_reference_state(x, count, seed, device="cpu"):
    shared, local, _, _ = hierarchy.sample_diffused(x, 0.0, count, seed)
    return make_state(shared, local, x, device)


def state_metrics(state, reference, clean_std, seed=0):
    state_np, reference_np = state_matrix(state), state_matrix(reference)
    local = state_np[:, 1:]
    reference_local = reference_np[:, 1:]
    return {
        "shared_w1_over_clean_sigma": scalar_w1(
            state_np[:, 0], reference_np[:, 0]
        ) / clean_std,
        "shared_mean_error_over_clean_sigma": abs(
            state_np[:, 0].mean() - reference_np[:, 0].mean()
        ) / clean_std,
        "shared_width_ratio": state_np[:, 0].std(ddof=1)
        / reference_np[:, 0].std(ddof=1),
        "local_w1_over_reference_sigma": float(np.mean([
            scalar_w1(local[:, j], reference_local[:, j])
            / max(reference_local[:, j].std(ddof=1), 1e-12)
            for j in range(local.shape[1])
        ])),
        "joint_sliced_w1": sliced_w1(state_np, reference_np, seed=seed),
    }


class ExactJointFields:
    """Exact tall and exact-single-observation finite-noise score fields."""

    def __init__(self, x, device, grid_points=2001, span=8.0):
        self.x = np.asarray(x, dtype=np.float64)
        self.device = torch.device(device)
        self.tall_grid = hierarchy.quadrature_grid(
            self.x, points=grid_points, span=span, device=self.device
        )
        self.row_grids = [
            hierarchy.quadrature_grid(
                [value], points=grid_points, span=span, device=self.device
            ) for value in self.x
        ]
        self.pilot_covariance = self._clean_single_covariances()

    def _clean_single_covariances(self):
        rows = []
        for value in self.x:
            grid, weights, _, _ = hierarchy.single_observation_reference(
                value, points=20001
            )
            eg = float(np.sum(weights * grid))
            eg2 = float(np.sum(weights * grid**2))
            el_given_g = 0.5 * (grid + value)
            el2_given_g = (grid**2 + grid * value + value**2) / 3.0
            egl_given_g = grid * el_given_g
            el = float(np.sum(weights * el_given_g))
            el2 = float(np.sum(weights * el2_given_g))
            egl = float(np.sum(weights * egl_given_g))
            rows.append([[eg2 - eg**2, egl - eg * el],
                         [egl - eg * el, el2 - el**2]])
        return torch.as_tensor(rows, dtype=torch.float64, device=self.device)

    def tall(self, state, lam):
        shared, local = hierarchy.diffused_score(
            state[0, :, hierarchy.GLOBAL_INDEX],
            state[:, :, hierarchy.LOCAL_INDEX].mT,
            self.x, float(lam), self.tall_grid,
        )
        score = torch.zeros_like(state)
        score[:, :, hierarchy.GLOBAL_INDEX] = shared.reshape(1, -1)
        score[:, :, hierarchy.LOCAL_INDEX] = local.mT
        return score

    def rows(self, state, lam):
        """Exact uncomposed (g,l_j) score, shape (N,S,2)."""
        result = torch.empty(
            len(self.x), state.shape[1], 2,
            dtype=torch.float64, device=self.device,
        )
        shared = state[0, :, hierarchy.GLOBAL_INDEX]
        for j, value in enumerate(self.x):
            score_g, score_l = hierarchy.diffused_score(
                shared, state[j, :, hierarchy.LOCAL_INDEX, None],
                [value], float(lam), self.row_grids[j],
            )
            result[j, :, 0] = score_g
            result[j, :, 1] = score_l[:, 0]
        return result

    def row_jacobian(self, state, lam, relative_step=2e-3):
        """Central-difference Jacobian of the exact row scores."""
        delta = max(float(relative_step * lam), 2e-5)
        columns = []
        for feature in (hierarchy.GLOBAL_INDEX, hierarchy.LOCAL_INDEX):
            tangent = torch.zeros_like(state)
            tangent[:, :, feature] = delta
            plus = self.rows(state + tangent, lam)
            minus = self.rows(state - tangent, lam)
            columns.append((plus - minus) / (2.0 * delta))
        jacobian = torch.stack(columns, dim=-1)
        return 0.5 * (jacobian + jacobian.mT)

    def _compose_from_precision(self, state, lam, row_scores, precision):
        """The same Schur/precision algebra used by covariance GAUSS rules."""
        variance = float(lam) ** 2
        n = len(self.x)
        row_scores = row_scores.clone()
        shared_state = state[0, :, hierarchy.GLOBAL_INDEX]
        local_state = state[:, :, hierarchy.LOCAL_INDEX]
        row_scores[..., 0] = (torch.clamp(
            shared_state.reshape(1, -1) + variance * row_scores[..., 0],
            hierarchy.MU_G - 5.0 * hierarchy.SIGMA_G,
            hierarchy.MU_G + 5.0 * hierarchy.SIGMA_G,
        ) - shared_state.reshape(1, -1)) / variance
        local_low = hierarchy.LOCAL_PRIOR_MEAN - 5.0 * hierarchy.LOCAL_PRIOR_STD
        local_high = hierarchy.LOCAL_PRIOR_MEAN + 5.0 * hierarchy.LOCAL_PRIOR_STD
        row_scores[..., 1] = (torch.clamp(
            local_state + variance * row_scores[..., 1], local_low, local_high
        ) - local_state) / variance
        block_gg = precision[..., :1, :1]
        block_gl = precision[..., :1, 1:]
        block_ll = precision[..., 1:, 1:]
        effective = block_gg - block_gl @ torch.linalg.solve(
            block_ll, precision[..., 1:, :1]
        )
        covariance = torch.linalg.inv(precision)
        cross = covariance[..., 1:, :1] * effective

        # Constant pilot precisions are (N,2,2); Jacobian precisions are
        # (N,S,2,2). From here down both use explicit (N,S) scalar factors.
        effective = effective[..., 0, 0]
        cross = cross[..., 0, 0]
        if effective.dim() == 1:
            effective = effective[:, None]
            cross = cross[:, None]

        inverse_variance = 1.0 / variance
        prior_precision = 1.0 + inverse_variance
        # Match MultiObsSampler's admissible-information projection in 1-D.
        effective = torch.maximum(
            effective, torch.as_tensor(
                prior_precision, dtype=torch.float64, device=self.device
            )
        )
        global_scores = row_scores[..., 0]
        denominator = effective.sum(dim=0) + (1 - n) * prior_precision
        prior_score = -state[0, :, hierarchy.GLOBAL_INDEX] / (1.0 + variance)
        numerator = (effective * global_scores).sum(dim=0)
        numerator = numerator + (1 - n) * prior_precision * prior_score
        composed = numerator / denominator
        composed = (torch.clamp(
            shared_state + variance * composed,
            hierarchy.MU_G - 5.0 * hierarchy.SIGMA_G,
            hierarchy.MU_G + 5.0 * hierarchy.SIGMA_G,
        ) - shared_state) / variance

        local = row_scores[..., 1] + cross * (composed - global_scores)
        local = (torch.clamp(
            local_state + variance * local, local_low, local_high
        ) - local_state) / variance
        result = torch.zeros_like(state)
        result[:, :, hierarchy.GLOBAL_INDEX] = composed.reshape(1, -1)
        result[:, :, hierarchy.LOCAL_INDEX] = local
        return result

    def gauss_hierarchical(self, state, lam):
        row_scores = self.rows(state, lam)
        identity = torch.eye(2, dtype=torch.float64, device=self.device)
        precision = torch.linalg.inv(
            self.pilot_covariance + float(lam) ** 2 * identity
        )
        return self._compose_from_precision(state, lam, row_scores, precision)

    def gauss_jacobian(self, state, lam):
        row_scores = self.rows(state, lam)
        jacobian = self.row_jacobian(state, lam)
        variance = float(lam) ** 2
        curvature = -jacobian
        eigenvalues, eigenvectors = torch.linalg.eigh(curvature)
        eigenvalues = eigenvalues.clamp(
            min=0.0, max=(1.0 - 1e-3) / variance
        )
        backward_precision = 1.0 / (
            variance * (1.0 - variance * eigenvalues)
        )
        precision = (
            eigenvectors * backward_precision.unsqueeze(-2)
        ) @ eigenvectors.mT
        return self._compose_from_precision(state, lam, row_scores, precision)

    def fnpe(self, state, lam):
        row_scores = self.rows(state, lam)
        t = math.log(1.0 + 2.0 * math.log(25.0) * float(lam) ** 2) \
            / (2.0 * math.log(25.0))
        prior_score = -state[0, :, hierarchy.GLOBAL_INDEX] / (1.0 + float(lam) ** 2)
        composed = row_scores[..., 0].sum(dim=0) \
            + (1 - len(self.x)) * (1 - t) * prior_score
        result = torch.zeros_like(state)
        result[:, :, hierarchy.GLOBAL_INDEX] = composed.reshape(1, -1)
        result[:, :, hierarchy.LOCAL_INDEX] = row_scores[..., 1]
        return result


def sigma_schedule(steps, eps=1e-3, device="cpu"):
    sde = hierarchy_sde()
    one = torch.ones(1, dtype=torch.float64, device=device)
    maximum = float(sde.lambda_t(one))
    minimum = float(sde.lambda_t(float(eps) * one))
    return torch.logspace(
        math.log10(maximum), math.log10(minimum), int(steps),
        dtype=torch.float64, device=device,
    )


def hierarchy_sde():
    from compass.SDE import VESDE
    return VESDE()


def production_initial_state(x, lam, count, seed, device, fnpe=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = len(x)
    shared_scale = 1.0 / math.sqrt(n) if fnpe else 1.0
    shared = float(lam) * shared_scale * torch.randn(
        count, generator=generator, dtype=torch.float64
    )
    local = float(lam) * torch.randn(
        n, count, generator=generator, dtype=torch.float64
    )
    state = torch.zeros(n, count, hierarchy.NODES, dtype=torch.float64)
    state[:, :, hierarchy.GLOBAL_INDEX] = shared.reshape(1, -1)
    state[:, :, hierarchy.LOCAL_INDEX] = local
    state[:, :, hierarchy.OBSERVED_INDEX] = torch.as_tensor(
        x, dtype=torch.float64
    ).reshape(-1, 1)
    return state.to(device)


def shared_noise_like(state, generator=None):
    noise = torch.randn(
        state.shape, dtype=state.dtype, device="cpu", generator=generator
    ).to(state.device)
    shared = torch.randn(
        (1, state.shape[1]), dtype=state.dtype, device="cpu", generator=generator
    ).to(state.device)
    noise[:, :, hierarchy.GLOBAL_INDEX] = shared
    return noise


def dpm2(field, initial, sigmas, snapshots=False):
    """The production Heun update, with only the score source replaced."""
    state = initial.clone()
    path = [state.detach().cpu()] if snapshots else None
    for index in range(len(sigmas) - 1):
        now, nxt = float(sigmas[index]), float(sigmas[index + 1])
        h = now - nxt
        score_now = field(state, now)
        proposal = state + h * now * score_now
        score_next = field(proposal, nxt)
        state = state + 0.5 * h * (now * score_now + nxt * score_next)
        state[:, :, hierarchy.OBSERVED_INDEX] = initial[:, :, hierarchy.OBSERVED_INDEX]
        state[:, :, hierarchy.GLOBAL_INDEX] = state[:1, :, hierarchy.GLOBAL_INDEX]
        if snapshots:
            path.append(state.detach().cpu())
    return state, path


def annealed_langevin(field, initial, sigmas, steps_per_level=10, snr=0.2,
                       fnpe_scale=False, seed=0, snapshots=False):
    state = initial.clone()
    path = [] if snapshots else None
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = state.shape[0]
    scale = torch.ones(hierarchy.NODES, dtype=state.dtype, device=state.device)
    if fnpe_scale:
        scale[hierarchy.GLOBAL_INDEX] = 1.0 / n
    for lam_tensor in sigmas:
        lam = float(lam_tensor)
        step = float(snr) * lam**2 * scale
        for _ in range(int(steps_per_level)):
            score = field(state, lam)
            noise = shared_noise_like(state, generator)
            state = state + step * score \
                + torch.sqrt(2.0 * step) * noise
            state[:, :, hierarchy.OBSERVED_INDEX] = initial[:, :, hierarchy.OBSERVED_INDEX]
            state[:, :, hierarchy.GLOBAL_INDEX] = state[:1, :, hierarchy.GLOBAL_INDEX]
        if snapshots:
            path.append(state.detach().cpu())
    return state, path


@dataclass
class MarginalOracle:
    """FFT tabulation of the exact diffused shared marginal and score."""

    x: np.ndarray
    eps: float = 1e-3
    dx: float = 0.0025

    def __post_init__(self):
        clean_grid, _, clean_weights, mean, std = hierarchy.shared_reference(self.x)
        self.clean_grid = clean_grid
        self.clean_weights = clean_weights
        self.mean, self.std = mean, std
        maximum = float(hierarchy_sde().lambda_t(torch.ones(1)))
        lower = clean_grid[0] - 6.5 * maximum
        upper = float(np.min(self.x)) + 6.5 * maximum
        points = int(math.ceil((upper - lower) / self.dx)) + 1
        self.grid = np.linspace(lower, upper, points)
        self.dx = float(self.grid[1] - self.grid[0])
        clean_density = np.interp(
            self.grid, clean_grid,
            clean_weights / (clean_grid[1] - clean_grid[0]), left=0.0, right=0.0,
        )
        clean_density /= np.trapezoid(clean_density, self.grid)
        self.clean_density = clean_density
        self._cache = {}

    def density_score(self, lam):
        key = round(float(lam), 12)
        if key in self._cache:
            return self._cache[key]
        radius = min(len(self.grid) // 2 - 1, int(math.ceil(7.0 * lam / self.dx)))
        offsets = np.arange(-radius, radius + 1) * self.dx
        kernel = np.exp(-0.5 * (offsets / lam) ** 2) / (math.sqrt(2 * math.pi) * lam)
        derivative = -offsets / lam**2 * kernel
        density = fftconvolve(self.clean_density, kernel, mode="same") * self.dx
        numerator = fftconvolve(self.clean_density, derivative, mode="same") * self.dx
        density = np.maximum(density, np.finfo(float).tiny)
        score = numerator / density
        self._cache[key] = density, score
        return density, score

    def score(self, values, lam):
        _, score = self.density_score(float(lam))
        return np.interp(
            np.asarray(values), self.grid, score,
            left=score[0], right=score[-1],
        )

    def sample(self, count, lam, seed):
        rng = np.random.default_rng(seed)
        clean = np.interp(
            rng.random(count), np.cumsum(self.clean_weights), self.clean_grid
        )
        return clean + float(lam) * rng.standard_normal(count)

    def clean_sample(self, count, seed):
        rng = np.random.default_rng(seed)
        return np.interp(
            rng.random(count), np.cumsum(self.clean_weights), self.clean_grid
        )


def marginal_dpm(oracle, steps, samples, eps, seed, exact_start=True):
    sigmas = sigma_schedule(steps, eps=eps).cpu().numpy()
    if exact_start:
        state = oracle.sample(samples, sigmas[0], seed)
    else:
        state = sigmas[0] * np.random.default_rng(seed).standard_normal(samples)
    for now, nxt in zip(sigmas[:-1], sigmas[1:]):
        h = now - nxt
        score_now = oracle.score(state, now)
        proposal = state + h * now * score_now
        score_next = oracle.score(proposal, nxt)
        state = state + 0.5 * h * (now * score_now + nxt * score_next)
    return state


def marginal_langevin(oracle, levels, mcmc_steps, samples, eps, seed, snr=0.2):
    sigmas = sigma_schedule(levels, eps=eps).cpu().numpy()
    rng = np.random.default_rng(seed)
    state = oracle.sample(samples, sigmas[0], seed)
    for lam in sigmas:
        step = snr * lam**2
        for _ in range(int(mcmc_steps)):
            state += step * oracle.score(state, lam) \
                + math.sqrt(2 * step) * rng.standard_normal(samples)
    return state


def marginal_terminal_correct(oracle, state, eps, steps, seed, snr=0.2):
    """Langevin corrections at the fixed numerical endpoint."""
    state = np.asarray(state, dtype=np.float64).copy()
    lam = float(sigma_schedule(2, eps=eps)[-1])
    step_size = float(snr) * lam**2
    rng = np.random.default_rng(seed)
    for _ in range(int(steps)):
        state += step_size * oracle.score(state, lam) \
            + math.sqrt(2.0 * step_size) * rng.standard_normal(state.shape)
    return state


def exact_bridge_reference(x, lam, count, seed, grid_points=12001):
    """Grid-exact F-NPSE shared bridge plus conditional local sampling.

    The bridge factorizes in locals conditional on the shared noisy coordinate:

      b(g_t,l_1t,...,l_Nt) propto p_prior,lam(g_t)^a prod_j p_lam(g_t,l_jt|x_j)

    so its shared marginal is one-dimensional.  Once ``g_t`` is drawn, each
    local is sampled independently by drawing ``g_0 | g_t,x_j``, then
    ``l_0 | g_0,x_j = Uniform(g_0,x_j)``, then adding local diffusion noise.
    """
    x = np.asarray(x, dtype=np.float64)
    sde = hierarchy_sde()
    t = float(sde.time_of_lambda(torch.tensor(float(lam))))
    a = (1 - len(x)) * (1 - t)
    # The clean bridge cannot cross the hard wall. At finite noise only the
    # Gaussian tail beyond that wall is relevant. A much wider grid makes the
    # row convolutions underflow; the negative prior exponent then turns an
    # arbitrary floating-point floor into false bridge mass.
    tall_grid, _, tall_weights, _, _ = hierarchy.shared_reference(x)
    support = np.flatnonzero(tall_weights > 1e-14)
    clean_lower = float(tall_grid[support[0]])
    lower = clean_lower - 2.0 - 7.0 * lam
    upper = float(np.min(x)) + 7.0 * lam
    grid = np.linspace(lower, upper, int(grid_points))
    dx = float(grid[1] - grid[0])
    log_bridge = a * (-0.5 * grid**2 / (1.0 + lam**2)
                      - 0.5 * math.log(2 * math.pi * (1.0 + lam**2)))
    clean_rows = []
    for value in x:
        clean_grid, clean_weights, _, _ = hierarchy.single_observation_reference(
            value, points=4001
        )
        clean_density = np.interp(
            grid, clean_grid,
            clean_weights / (clean_grid[1] - clean_grid[0]), left=0.0, right=0.0,
        )
        radius = min(len(grid) // 2 - 1, int(math.ceil(7.0 * lam / dx)))
        offsets = np.arange(-radius, radius + 1) * dx
        kernel = np.exp(-0.5 * (offsets / lam) ** 2) / (math.sqrt(2 * math.pi) * lam)
        diffused = fftconvolve(clean_density, kernel, mode="same") * dx
        log_bridge += np.log(np.maximum(diffused, np.finfo(float).tiny))
        clean_rows.append((clean_grid, clean_weights))
    weights = np.exp(log_bridge - np.max(log_bridge))
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    shared = np.interp(rng.random(count), np.cumsum(weights), grid)

    local = np.empty((count, len(x)), dtype=np.float64)
    for j, (value, clean) in enumerate(zip(x, clean_rows)):
        clean_grid, clean_weights = clean
        # Chunk to avoid count x 10k temporary arrays becoming material.
        g0 = np.empty(count, dtype=np.float64)
        for start in range(0, count, 256):
            stop = min(start + 256, count)
            log_weight = np.log(np.maximum(clean_weights, 1e-300))[None, :] \
                - 0.5 * ((shared[start:stop, None] - clean_grid[None, :]) / lam) ** 2
            log_weight -= log_weight.max(axis=1, keepdims=True)
            probability = np.exp(log_weight)
            probability /= probability.sum(axis=1, keepdims=True)
            cumulative = np.cumsum(probability, axis=1)
            uniforms = rng.random(stop - start)
            indices = np.array([
                np.searchsorted(cumulative[row], uniforms[row], side="right")
                for row in range(stop - start)
            ])
            g0[start:stop] = clean_grid[np.minimum(indices, len(clean_grid) - 1)]
        l0 = g0 + rng.random(count) * (value - g0)
        local[:, j] = l0 + lam * rng.standard_normal(count)
    return shared, local


def analytic_composition_probe(fields, x, lambdas, states, seed, output):
    methods = {
        "analytic_gauss_hierarchical": fields.gauss_hierarchical,
        "analytic_gauss_jacobian": fields.gauss_jacobian,
        "analytic_fnpe": fields.fnpe,
    }
    rows = []
    for index, lam in enumerate(lambdas):
        state = exact_diffused_state(x, lam, states, seed + index, fields.device)
        exact = fields.tall(state, lam)
        exact_shared = exact[0, :, hierarchy.GLOBAL_INDEX]
        exact_local = exact[:, :, hierarchy.LOCAL_INDEX]
        _, _, clean_g, _ = hierarchy.sample_diffused(
            x, lam, states, seed + index
        )
        wall_distance = float(np.min(x)) - clean_g
        quantiles = np.quantile(wall_distance, [0.0, 0.25, 0.5, 0.75, 1.0])
        for name, method in methods.items():
            estimate = method(state, lam)
            error = estimate[0, :, hierarchy.GLOBAL_INDEX] - exact_shared
            shared_rms = float(torch.sqrt(torch.mean(exact_shared**2)))
            rows.append({
                "kind": "overall", "method": name, "lambda": lam,
                "shared_rel_rmse": float(torch.sqrt(torch.mean(error**2))) / shared_rms,
                "signed_bias": float(error.mean()),
                "signed_bias_over_exact_rms": float(error.mean()) / shared_rms,
                "local_rel_rmse": float(torch.sqrt(torch.mean(
                    (estimate[:, :, hierarchy.LOCAL_INDEX] - exact_local) ** 2
                )) / torch.sqrt(torch.mean(exact_local**2))),
            })
            error_np = error.detach().cpu().numpy()
            for wall_bin in range(4):
                include = ((wall_distance >= quantiles[wall_bin])
                           & (wall_distance <= quantiles[wall_bin + 1]))
                rows.append({
                    "kind": "wall_bin", "method": name, "lambda": lam,
                    "wall_bin": wall_bin,
                    "wall_distance_low": quantiles[wall_bin],
                    "wall_distance_high": quantiles[wall_bin + 1],
                    "shared_rel_rmse": float(np.sqrt(np.mean(error_np[include] ** 2)))
                    / shared_rms,
                    "signed_bias": float(np.mean(error_np[include])),
                    "signed_bias_over_exact_rms": float(np.mean(error_np[include]))
                    / shared_rms,
                })
    write_rows(output / "analytic_composition.csv", rows)
    return rows


def generic_curl(field, state, lam, local_rows=3, relative_step=1e-3):
    """Finite-difference curl for an arbitrary exact or composed field."""
    probes = [(hierarchy.GLOBAL_INDEX, None)] + [
        (hierarchy.LOCAL_INDEX, row)
        for row in range(min(local_rows, state.shape[0]))
    ]
    delta = max(relative_step * float(lam), 2e-5)
    columns = []
    for feature, row in probes:
        tangent = torch.zeros_like(state)
        if row is None:
            tangent[:, :, feature] = delta
        else:
            tangent[row, :, feature] = delta
        derivative = (field(state + tangent, lam) - field(state - tangent, lam)) \
            / (2.0 * delta)
        columns.append(torch.stack([
            derivative[0 if probe_row is None else probe_row, :, probe_feature]
            for probe_feature, probe_row in probes
        ], dim=-1))
    jacobian = torch.stack(columns, dim=-1)
    antisymmetric = jacobian - jacobian.mT
    norm = torch.linalg.matrix_norm(jacobian).clamp_min(1e-15)
    asymmetry = torch.linalg.matrix_norm(antisymmetric) / norm
    return float(asymmetry.mean()), float(asymmetry.max()), float(antisymmetric.abs().max())


def oracle_and_analytic_runs(fields, problem, arguments, output):
    x = problem["x"]
    sigmas = sigma_schedule(arguments.oracle_steps, arguments.eps, fields.device)
    maximum = float(sigmas[0])
    exact_initial = exact_diffused_state(
        x, maximum, arguments.oracle_samples, arguments.seed + 700, fields.device
    )
    production_initial = production_initial_state(
        x, maximum, arguments.oracle_samples, arguments.seed + 700, fields.device
    )
    clean_reference = clean_reference_state(
        x, max(arguments.reference_samples, arguments.oracle_samples),
        arguments.seed + 1700,
    )
    rows, archives = [], {}

    runs = {
        "oracle_dpm_exact_start": lambda: dpm2(
            fields.tall, exact_initial, sigmas, snapshots=True
        ),
        "oracle_dpm_production_start": lambda: dpm2(
            fields.tall, production_initial, sigmas, snapshots=True
        ),
        "oracle_langevin_exact_diffusion": lambda: annealed_langevin(
            fields.tall, exact_initial, sigmas,
            steps_per_level=arguments.oracle_langevin_steps,
            seed=arguments.seed + 701, snapshots=True,
        ),
    }
    if arguments.run_analytic_sampling:
        analytic_steps = sigma_schedule(
            arguments.analytic_steps, arguments.eps, fields.device
        )
        analytic_initial = exact_diffused_state(
            x, float(analytic_steps[0]), arguments.analytic_samples,
            arguments.seed + 702, fields.device,
        )
        fnpe_initial = production_initial_state(
            x, float(analytic_steps[0]), arguments.analytic_samples,
            arguments.seed + 702, fields.device, fnpe=True,
        )
        runs.update({
            "analytic_gauss_hierarchical_dpm": lambda: dpm2(
                fields.gauss_hierarchical, analytic_initial, analytic_steps,
                snapshots=False,
            ),
            "analytic_gauss_jacobian_dpm": lambda: dpm2(
                fields.gauss_jacobian, analytic_initial, analytic_steps,
                snapshots=False,
            ),
            "analytic_fnpe_langevin": lambda: annealed_langevin(
                fields.fnpe, fnpe_initial, analytic_steps,
                steps_per_level=arguments.analytic_langevin_steps,
                fnpe_scale=True, seed=arguments.seed + 703, snapshots=False,
            ),
        })

    for name, run in runs.items():
        print(f"oracle/analytic run: {name}")
        started = time.perf_counter()
        final, path = run()
        seconds = time.perf_counter() - started
        metrics = state_metrics(
            final, clean_reference, problem["global_std"], seed=arguments.seed
        )
        rows.append({"run": name, "seconds": seconds, **metrics})
        archives[name] = final.detach().cpu().numpy()
        if path is not None:
            archives[f"{name}_path"] = np.stack([value.numpy() for value in path])
    np.savez_compressed(output / "oracle_samples.npz", **archives)
    write_rows(output / "oracle_endpoint.csv", rows)
    return rows, archives


def learned_sample(problem, method, samples, timesteps, eps, seed,
                   corrector_steps=None, terminal_steps=0, save_trajectory=False):
    kwargs = dict(compare.METHODS[method]["sample_kwargs"])
    if corrector_steps is not None:
        kwargs["corrector_steps"] = int(corrector_steps)
    sampler = MultiObsSampler(problem["model"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    result = sampler.sample(
        world_size=1, data=problem["data"],
        condition_mask=hierarchy.CONDITION_MASK,
        timesteps=int(timesteps), eps=float(eps), num_samples=int(samples),
        hierarchy=list(hierarchy.HIERARCHY),
        prior=(torch.tensor([hierarchy.MU_G]), torch.tensor([hierarchy.SIGMA_G])),
        local_prior=(torch.tensor([hierarchy.LOCAL_PRIOR_MEAN]),
                     torch.tensor([hierarchy.LOCAL_PRIOR_STD])),
        denoise_clamp=5.0, device=problem["device"], verbose=False,
        terminal_corrector_steps=int(terminal_steps),
        save_trajectory=save_trajectory, **kwargs,
    ).detach().cpu()
    return result, sampler, time.perf_counter() - started


def convergence_sweeps(problem, arguments, output):
    existing = read_rows(output / "convergence.csv")
    keys = {
        (row["family"], row["method"], row["setting"], row["value"])
        for row in existing
    }
    rows = list(existing)
    clean = clean_reference_state(
        problem["x"], arguments.reference_samples,
        arguments.seed + 2200,
    )

    def append(family, method, setting, value, state, seconds):
        metric = state_metrics(
            state, clean, problem["global_std"], seed=arguments.seed
        )
        rows.append({"family": family, "method": method, "setting": setting,
                     "value": value, "seconds": seconds, **metric})
        write_rows(output / "convergence.csv", rows)

    for steps in arguments.dpm_steps:
        key = ("learned", "gauss_jacobian", "timesteps", str(steps))
        if key not in keys:
            state, _, seconds = learned_sample(
                problem, "gauss_jacobian", arguments.convergence_samples,
                steps, arguments.eps, arguments.seed,
            )
            append(*key, state, seconds)

    for steps in arguments.dpm_steps:
        key = ("learned_predictor_only", "gauss_jacobian",
               "timesteps", str(steps))
        if key not in keys:
            state, _, seconds = learned_sample(
                problem, "gauss_jacobian", arguments.convergence_samples,
                steps, arguments.eps, arguments.seed, corrector_steps=0,
            )
            append(*key, state, seconds)

    for mcmc in arguments.mcmc_steps:
        key = ("learned", "langevin_fnpe", "steps_per_level", str(mcmc))
        if key not in keys:
            state, _, seconds = learned_sample(
                problem, "langevin_fnpe", arguments.convergence_samples,
                100, arguments.eps, arguments.seed, corrector_steps=mcmc,
            )
            append(*key, state, seconds)

    for terminal in arguments.terminal_steps:
        key = ("learned", "gauss_jacobian", "terminal_corrector_steps", str(terminal))
        if key not in keys:
            state, _, seconds = learned_sample(
                problem, "gauss_jacobian", arguments.terminal_samples,
                100, arguments.eps, arguments.seed, terminal_steps=terminal,
            )
            append(*key, state, seconds)

    for eps in arguments.eps_values:
        key = ("learned", "gauss_jacobian", "eps", str(eps))
        if key not in keys:
            state, _, seconds = learned_sample(
                problem, "gauss_jacobian", arguments.terminal_samples,
                100, eps, arguments.seed,
            )
            append(*key, state, seconds)

    oracle = MarginalOracle(problem["x"], eps=arguments.eps)
    clean_shared = oracle.clean_sample(arguments.marginal_samples, arguments.seed + 3300)
    for steps in arguments.dpm_steps:
        key = ("oracle_marginal", "dpm2", "timesteps", str(steps))
        if key not in keys:
            started = time.perf_counter()
            draws = marginal_dpm(
                oracle, steps, arguments.marginal_samples,
                arguments.eps, arguments.seed, exact_start=True,
            )
            seconds = time.perf_counter() - started
            rows.append({"family": key[0], "method": key[1], "setting": key[2],
                         "value": key[3], "seconds": seconds,
                         "shared_w1_over_clean_sigma": scalar_w1(draws, clean_shared)
                         / problem["global_std"],
                         "shared_mean_error_over_clean_sigma": abs(
                             draws.mean() - clean_shared.mean()
                         ) / problem["global_std"],
                         "shared_width_ratio": draws.std(ddof=1) / clean_shared.std(ddof=1)})
            write_rows(output / "convergence.csv", rows)
        production_key = ("oracle_marginal", "dpm2_production_start",
                          "timesteps", str(steps))
        if production_key not in keys:
            started = time.perf_counter()
            draws = marginal_dpm(
                oracle, steps, arguments.marginal_samples,
                arguments.eps, arguments.seed, exact_start=False,
            )
            seconds = time.perf_counter() - started
            rows.append({"family": production_key[0],
                         "method": production_key[1],
                         "setting": production_key[2],
                         "value": production_key[3], "seconds": seconds,
                         "shared_w1_over_clean_sigma": scalar_w1(draws, clean_shared)
                         / problem["global_std"],
                         "shared_mean_error_over_clean_sigma": abs(
                             draws.mean() - clean_shared.mean()
                         ) / problem["global_std"],
                         "shared_width_ratio": draws.std(ddof=1) / clean_shared.std(ddof=1)})
            write_rows(output / "convergence.csv", rows)
    for mcmc in arguments.mcmc_steps:
        key = ("oracle_marginal", "annealed_langevin", "steps_per_level", str(mcmc))
        if key not in keys:
            started = time.perf_counter()
            draws = marginal_langevin(
                oracle, 100, mcmc, arguments.marginal_samples,
                arguments.eps, arguments.seed,
            )
            seconds = time.perf_counter() - started
            rows.append({"family": key[0], "method": key[1], "setting": key[2],
                         "value": key[3], "seconds": seconds,
                         "shared_w1_over_clean_sigma": scalar_w1(draws, clean_shared)
                         / problem["global_std"],
                         "shared_mean_error_over_clean_sigma": abs(
                             draws.mean() - clean_shared.mean()
                         ) / problem["global_std"],
                         "shared_width_ratio": draws.std(ddof=1) / clean_shared.std(ddof=1)})
            write_rows(output / "convergence.csv", rows)
    oracle_base = marginal_dpm(
        oracle, 100, arguments.marginal_samples,
        arguments.eps, arguments.seed, exact_start=True,
    )
    for terminal in arguments.terminal_steps:
        key = ("oracle_marginal", "dpm2", "terminal_corrector_steps", str(terminal))
        if key not in keys:
            started = time.perf_counter()
            draws = marginal_terminal_correct(
                oracle, oracle_base, arguments.eps, terminal,
                arguments.seed + 1,
            )
            seconds = time.perf_counter() - started
            rows.append({"family": key[0], "method": key[1], "setting": key[2],
                         "value": key[3], "seconds": seconds,
                         "shared_w1_over_clean_sigma": scalar_w1(draws, clean_shared)
                         / problem["global_std"],
                         "shared_mean_error_over_clean_sigma": abs(
                             draws.mean() - clean_shared.mean()
                         ) / problem["global_std"],
                         "shared_width_ratio": draws.std(ddof=1) / clean_shared.std(ddof=1)})
            write_rows(output / "convergence.csv", rows)
    for eps in arguments.eps_values:
        key = ("oracle_marginal", "dpm2", "eps", str(eps))
        if key not in keys:
            started = time.perf_counter()
            draws = marginal_dpm(
                oracle, 100, arguments.marginal_samples,
                eps, arguments.seed, exact_start=True,
            )
            seconds = time.perf_counter() - started
            rows.append({"family": key[0], "method": key[1], "setting": key[2],
                         "value": key[3], "seconds": seconds,
                         "shared_w1_over_clean_sigma": scalar_w1(draws, clean_shared)
                         / problem["global_std"],
                         "shared_mean_error_over_clean_sigma": abs(
                             draws.mean() - clean_shared.mean()
                         ) / problem["global_std"],
                         "shared_width_ratio": draws.std(ddof=1) / clean_shared.std(ddof=1)})
            write_rows(output / "convergence.csv", rows)
    return rows


def intermediate_checks(problem, fields, arguments, output):
    rows, archives = [], {}
    selected = np.unique(np.linspace(0, 99, arguments.trajectory_levels).round().astype(int))
    for method in METHOD_NAMES:
        print(f"trajectory: {method}")
        samples, sampler, seconds = learned_sample(
            problem, method, arguments.trajectory_samples, 100,
            arguments.eps, arguments.seed, save_trajectory=True,
        )
        trajectory = sampler.data_t.detach().cpu()
        schedule = sampler.timesteps_list.detach().cpu()
        archives[method] = trajectory.numpy()
        archives[f"{method}_times"] = schedule.numpy()
        for pick in selected:
            # Langevin stores the state after lambda[pick-1] in slot pick.
            schedule_index = max(0, pick - 1) if method == "langevin_fnpe" else pick
            lam = float(sampler.sde.lambda_t(schedule[schedule_index]))
            state = trajectory[:, pick]
            if method == "langevin_fnpe":
                bridge_g, bridge_l = exact_bridge_reference(
                    problem["x"], lam, arguments.reference_samples,
                    arguments.seed + 4000 + pick,
                    grid_points=4001 if arguments.quick else 8001,
                )
                reference = make_state(
                    bridge_g, bridge_l, problem["x"], "cpu"
                )
                target = "fnpe_bridge"
            else:
                reference = exact_diffused_state(
                    problem["x"], lam, arguments.reference_samples,
                    arguments.seed + 4000 + pick, "cpu",
                )
                target = "exact_diffusion"
            metrics = state_metrics(
                state, reference, problem["global_std"], seed=arguments.seed
            )
            subset = min(arguments.visited_score_states, state.shape[1])
            visit = state[:, :subset].to(fields.device)
            exact = fields.tall(visit, lam)
            if method == "langevin_fnpe":
                intended = fields.fnpe(visit, lam)
                target_score = intended
            else:
                target_score = exact
            # The configured learned sampler is the exact object that made this path.
            mask = hierarchy.CONDITION_MASK.to(fields.device).reshape(1, 1, -1)
            mask = mask.repeat(len(problem["x"]), subset, 1)
            indices = torch.arange(len(problem["x"]), device=fields.device)
            t = sampler.sde.time_of_lambda(torch.tensor(lam)).reshape(1, 1).to(fields.device)
            sampler._reset_jacobian_state()
            learned = sampler._get_score(
                visit.to(dtype=torch.float32), t, mask, indices
            ).to(torch.float64)
            score_error = learned[0, :, 0] - target_score[0, :, 0]
            score_rms = torch.sqrt(torch.mean(target_score[0, :, 0] ** 2)).clamp_min(1e-15)
            rows.append({
                "method": method, "target": target, "trajectory_index": pick,
                "lambda": lam, "seconds_total": seconds,
                "visited_shared_score_rel_rmse": float(torch.sqrt(
                    torch.mean(score_error**2)) / score_rms),
                "visited_shared_score_bias": float(score_error.mean()),
                **metrics,
            })
        curl_data = torch.zeros(
            len(problem["x"]), hierarchy.NODES, device=problem["device"]
        )
        curl_data[:, hierarchy.OBSERVED_INDEX] = torch.as_tensor(
            problem["x"], dtype=torch.float32, device=problem["device"]
        )
        for diagnostic_index, diagnostic_lambda in enumerate(arguments.lambdas):
            diagnostic_state = exact_diffused_state(
                problem["x"], diagnostic_lambda, arguments.curl_states,
                arguments.seed + 4500 + diagnostic_index, problem["device"],
            )
            diagnostic_time = sampler.sde.time_of_lambda(
                torch.tensor([diagnostic_lambda], device=problem["device"])
            )
            curl = sampler.composition_curl(
                diagnostic_state.to(dtype=torch.float32), curl_data,
                hierarchy.CONDITION_MASK.to(problem["device"]),
                times=diagnostic_time, observations=3, device=problem["device"],
            )
            rows.append({
                "method": method, "target": "curl", "lambda": diagnostic_lambda,
                "curl_asymmetry_mean": curl["asymmetry_mean"][0],
                "curl_asymmetry_max": curl["asymmetry_max"][0],
                "max_abs_curl": curl["max_abs_curl"][0],
            })
            if method == "gauss_jacobian":
                diagnostic_mask = hierarchy.CONDITION_MASK.to(
                    problem["device"]
                ).reshape(1, 1, -1).repeat(
                    len(problem["x"]), arguments.curl_states, 1
                )
                t = diagnostic_time.reshape(1, 1)
                with torch.enable_grad():
                    learned_rows = sampler._raw_row_scores(
                        diagnostic_state.to(dtype=torch.float32), t, diagnostic_mask
                    ).to(torch.float64)[..., [hierarchy.GLOBAL_INDEX,
                                              hierarchy.LOCAL_INDEX]]
                    learned_jacobian = sampler._row_score_jacobian(
                        diagnostic_state.to(dtype=torch.float32), t, diagnostic_mask,
                        [hierarchy.GLOBAL_INDEX, hierarchy.LOCAL_INDEX],
                    )
                exact_rows = fields.rows(diagnostic_state, diagnostic_lambda)
                exact_jacobian = fields.row_jacobian(
                    diagnostic_state, diagnostic_lambda
                )
                rows.append({
                    "method": method, "target": "row_jacobian",
                    "lambda": diagnostic_lambda,
                    "single_row_score_rel_rmse": float(
                        torch.sqrt(torch.mean((learned_rows - exact_rows) ** 2))
                        / torch.sqrt(torch.mean(exact_rows ** 2)).clamp_min(1e-15)
                    ),
                    "row_jacobian_rel_rmse": float(
                        torch.sqrt(torch.mean(
                            (learned_jacobian - exact_jacobian) ** 2
                        )) / torch.sqrt(torch.mean(exact_jacobian ** 2)).clamp_min(1e-15)
                    ),
                })
    np.savez_compressed(output / "learned_trajectories.npz", **archives)
    write_rows(output / "intermediate.csv", rows)
    return rows


def exact_curl_checks(fields, problem, arguments, output):
    rows = []
    methods = {
        "exact_diffusion": fields.tall,
        "analytic_gauss_hierarchical": fields.gauss_hierarchical,
        "analytic_gauss_jacobian": fields.gauss_jacobian,
        "analytic_fnpe": fields.fnpe,
    }
    for index, lam in enumerate(arguments.lambdas):
        state = exact_diffused_state(
            problem["x"], lam, arguments.exact_curl_states,
            arguments.seed + 5000 + index, fields.device,
        )
        for name, field in methods.items():
            print(f"curl: {name} lambda={lam:g}")
            mean, maximum, absolute = generic_curl(field, state, lam)
            rows.append({"method": name, "lambda": lam,
                         "curl_asymmetry_mean": mean,
                         "curl_asymmetry_max": maximum,
                         "max_abs_curl": absolute})
    write_rows(output / "exact_curl.csv", rows)
    return rows


def plot_all(problem, output):
    configure_style()
    exact_grid = problem["global_grid"]
    exact_density = problem["global_density"]

    archive_path = output / "oracle_samples.npz"
    if archive_path.exists():
        archive = np.load(archive_path)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
        axes[0].plot(exact_grid, exact_density, "--", color=compare.NAVY,
                     lw=2.2, label="exact clean posterior")
        for key, label in (
            ("oracle_dpm_exact_start", "oracle DPM2, exact start"),
            ("oracle_dpm_production_start", "oracle DPM2, production start"),
            ("oracle_langevin_exact_diffusion", "oracle ALD, exact diffusion path"),
        ):
            if key in archive:
                axes[0].hist(archive[key][0, :, 0], bins=35, density=True,
                             histtype="step", lw=1.8, label=label)
        axes[0].set(title="Exact joint-score numerical controls",
                    xlabel="global g", ylabel="density")
        axes[0].legend(fontsize=8)
        for key, label in (
            ("analytic_gauss_hierarchical_dpm", "exact rows + pilot GAUSS + DPM2"),
            ("analytic_gauss_jacobian_dpm", "exact rows + Jacobian GAUSS + DPM2"),
            ("analytic_fnpe_langevin", "exact rows + F-NPSE + ALD"),
        ):
            if key in archive:
                axes[1].hist(archive[key][0, :, 0], bins=30, density=True,
                             histtype="step", lw=1.8, label=label)
        axes[1].plot(exact_grid, exact_density, "--", color=compare.NAVY,
                     lw=2.2, label="exact clean posterior")
        axes[1].set(title="Analytical single-row score ablation",
                    xlabel="global g", ylabel="density")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "00_oracle_endpoint.png", bbox_inches="tight")
        plt.close(fig)

    analytic = [row for row in read_rows(output / "analytic_composition.csv")
                if row.get("kind") == "overall"]
    if analytic:
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5))
        for name in sorted({row["method"] for row in analytic}):
            selected = sorted(
                [row for row in analytic if row["method"] == name],
                key=lambda row: float(row["lambda"]),
            )
            axes[0].plot([float(row["lambda"]) for row in selected],
                         [float(row["shared_rel_rmse"]) for row in selected],
                         "o-", label=name.replace("analytic_", ""),
                         color=COLOURS.get(name))
            axes[1].plot([float(row["lambda"]) for row in selected],
                         [float(row["signed_bias_over_exact_rms"]) for row in selected],
                         "o-", label=name.replace("analytic_", ""),
                         color=COLOURS.get(name))
        axes[0].set(xscale="log", yscale="log", title="Exact rows: rule error",
                    xlabel="noise lambda", ylabel="shared relative RMSE")
        axes[1].axhline(0, color=compare.NAVY, ls="--", lw=1)
        axes[1].set(xscale="log", title="Exact rows: signed error",
                    xlabel="noise lambda", ylabel="bias / exact score RMS")
        wall = [row for row in read_rows(output / "analytic_composition.csv")
                if row.get("kind") == "wall_bin"
                and abs(float(row["lambda"]) - min(float(x["lambda"]) for x in analytic)) < 1e-12]
        for name in sorted({row["method"] for row in wall}):
            selected = sorted([row for row in wall if row["method"] == name],
                              key=lambda row: int(row["wall_bin"]))
            axes[2].plot(range(1, 5), [float(row["signed_bias_over_exact_rms"])
                                      for row in selected], "o-",
                         label=name.replace("analytic_", ""), color=COLOURS.get(name))
        axes[2].axhline(0, color=compare.NAVY, ls="--", lw=1)
        axes[2].set(title="Smallest lambda: bias vs wall distance",
                    xlabel="wall-distance quartile (1 = closest)",
                    ylabel="bias / exact score RMS", xticks=range(1, 5))
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "01_analytic_composition.png", bbox_inches="tight")
        plt.close(fig)

    intermediate = [row for row in read_rows(output / "intermediate.csv")
                    if row.get("target") in ("exact_diffusion", "fnpe_bridge")]
    if intermediate:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
        keys = ["shared_w1_over_clean_sigma", "local_w1_over_reference_sigma",
                "joint_sliced_w1", "visited_shared_score_rel_rmse"]
        titles = ["Shared trajectory W1", "Local trajectory W1",
                  "Joint sliced W1", "Score error on visited states"]
        for method in METHOD_NAMES:
            selected = sorted([row for row in intermediate if row["method"] == method],
                              key=lambda row: float(row["lambda"]), reverse=True)
            for axis, key, title in zip(axes, keys, titles):
                axis.plot([float(row["lambda"]) for row in selected],
                          [float(row[key]) for row in selected], "o-",
                          color=COLOURS[method], label=method)
                axis.set(xscale="log", title=title, xlabel="noise lambda",
                         ylabel=key.replace("_", " "))
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "02_intermediate_density.png", bbox_inches="tight")
        plt.close(fig)

    convergence = read_rows(output / "convergence.csv")
    if convergence:
        fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.5))
        panels = [("timesteps", axes[0]), ("steps_per_level", axes[1]),
                  ("terminal_corrector_steps", axes[2]), ("eps", axes[3])]
        for setting, axis in panels:
            for family, method in sorted({(row["family"], row["method"])
                                          for row in convergence
                                          if row["setting"] == setting}):
                selected = sorted([row for row in convergence
                                   if row["setting"] == setting
                                   and row["family"] == family
                                   and row["method"] == method],
                                  key=lambda row: float(row["value"]))
                axis.plot([float(row["value"]) for row in selected],
                          [float(row["shared_w1_over_clean_sigma"]) for row in selected],
                          "o-", label=f"{family}: {method}")
            axis.set(title=setting.replace("_", " "), xlabel=setting,
                     ylabel="endpoint shared W1 / clean sigma")
            if setting == "eps":
                axis.set_xscale("log")
            axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "03_convergence.png", bbox_inches="tight")
        plt.close(fig)

    curls = read_rows(output / "exact_curl.csv")
    learned_curl = [row for row in read_rows(output / "intermediate.csv")
                    if row.get("target") == "curl"]
    if curls or learned_curl:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
        for rows, axis, title in ((curls, axes[0], "Analytical-field curl"),
                                  (learned_curl, axes[1], "Learned-field curl")):
            for method in sorted({row["method"] for row in rows}):
                selected = sorted([row for row in rows if row["method"] == method],
                                  key=lambda row: float(row["lambda"]))
                axis.plot([float(row["lambda"]) for row in selected],
                          [float(row["curl_asymmetry_mean"]) for row in selected],
                          "o-", label=method, color=COLOURS.get(method))
            axis.set(xscale="log", yscale="log", title=title,
                     xlabel="noise lambda", ylabel="relative antisymmetry")
            axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "04_curl.png", bbox_inches="tight")
        plt.close(fig)

    jacobian_rows = [row for row in read_rows(output / "intermediate.csv")
                     if row.get("target") == "row_jacobian"]
    if jacobian_rows:
        selected = sorted(jacobian_rows, key=lambda row: float(row["lambda"]))
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        axes[0].plot([float(row["lambda"]) for row in selected],
                     [float(row["single_row_score_rel_rmse"]) for row in selected],
                     "o-", color=compare.BLUE)
        axes[1].plot([float(row["lambda"]) for row in selected],
                     [float(row["row_jacobian_rel_rmse"]) for row in selected],
                     "o-", color=compare.TEAL)
        axes[0].set(xscale="log", yscale="log",
                    title="Single-row analytical score check",
                    xlabel="noise lambda", ylabel="relative RMSE")
        axes[1].set(xscale="log", yscale="log",
                    title="Learned score-Jacobian check",
                    xlabel="noise lambda", ylabel="relative RMSE")
        fig.tight_layout()
        fig.savefig(output / "05_learned_jacobian.png", bbox_inches="tight")
        plt.close(fig)
    for name in ("00_oracle_endpoint.png", "01_analytic_composition.png",
                 "02_intermediate_density.png", "03_convergence.png", "04_curl.png", "05_learned_jacobian.png"):
        if (output / name).exists():
            print(f"Wrote {output / name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="h16d2")
    parser.add_argument("--train-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--eps-values", nargs="+", type=float,
                        default=[1e-3, 3e-4, 1e-4])
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    parser.add_argument("--dpm-steps", nargs="+", type=int,
                        default=[50, 100, 200, 400])
    parser.add_argument("--mcmc-steps", nargs="+", type=int,
                        default=[5, 10, 20, 40])
    parser.add_argument("--terminal-steps", nargs="+", type=int,
                        default=[0, 10, 50])
    parser.add_argument("--oracle-steps", type=int, default=100)
    parser.add_argument("--oracle-samples", type=int, default=96)
    parser.add_argument("--oracle-langevin-steps", type=int, default=3)
    parser.add_argument("--analytic-steps", type=int, default=50)
    parser.add_argument("--analytic-samples", type=int, default=64)
    parser.add_argument("--analytic-langevin-steps", type=int, default=3)
    parser.add_argument("--score-states", type=int, default=96)
    parser.add_argument("--trajectory-samples", type=int, default=384)
    parser.add_argument("--trajectory-levels", type=int, default=8)
    parser.add_argument("--visited-score-states", type=int, default=48)
    parser.add_argument("--curl-states", type=int, default=12)
    parser.add_argument("--exact-curl-states", type=int, default=8)
    parser.add_argument("--convergence-samples", type=int, default=384)
    parser.add_argument("--terminal-samples", type=int, default=192)
    parser.add_argument("--marginal-samples", type=int, default=5000)
    parser.add_argument("--reference-samples", type=int, default=5000)
    parser.add_argument("--quadrature-points", type=int, default=2001)
    parser.add_argument("--run-analytic-sampling", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-trajectories", action="store_true")
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--skip-curl", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--replot", action="store_true")
    arguments = parser.parse_args()
    if arguments.device is None:
        arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
    if arguments.quick:
        arguments.lambdas = [0.1, 0.5]
        arguments.dpm_steps = [8, 12]
        arguments.mcmc_steps = [1, 2]
        arguments.terminal_steps = [0]
        arguments.eps_values = [1e-3]
        arguments.oracle_steps = 8
        arguments.oracle_samples = 8
        arguments.oracle_langevin_steps = 1
        arguments.analytic_steps = 6
        arguments.analytic_samples = 4
        arguments.score_states = 6
        arguments.trajectory_samples = 12
        arguments.trajectory_levels = 2
        arguments.visited_score_states = 3
        arguments.curl_states = 2
        arguments.exact_curl_states = 2
        arguments.convergence_samples = 8
        arguments.terminal_samples = 8
        arguments.marginal_samples = 100
        arguments.reference_samples = 100
        arguments.quadrature_points = 301

    arguments.output.mkdir(parents=True, exist_ok=True)
    if not arguments.replot:
        (arguments.output / "run_config.json").write_text(
            json.dumps({key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(arguments).items()}, indent=2) + "\n"
        )
    problem = compare.build_problem(
        compare.recipe.model_directory(
            compare.ARTIFACTS, arguments.config, arguments.train_samples
        ), arguments.config, arguments.train_samples, compare.OBSERVATIONS,
        arguments.seed, arguments.device,
    )
    if arguments.replot:
        plot_all(problem, arguments.output)
        return

    fields = ExactJointFields(
        problem["x"], arguments.device,
        grid_points=arguments.quadrature_points,
    )
    analytic_composition_probe(
        fields, problem["x"], arguments.lambdas,
        arguments.score_states, arguments.seed + 6000, arguments.output,
    )
    if not arguments.skip_oracle:
        oracle_and_analytic_runs(fields, problem, arguments, arguments.output)
    if not arguments.skip_trajectories:
        intermediate_checks(problem, fields, arguments, arguments.output)
    if not arguments.skip_convergence:
        convergence_sweeps(problem, arguments, arguments.output)
    if not arguments.skip_curl:
        exact_curl_checks(fields, problem, arguments, arguments.output)
    plot_all(problem, arguments.output)


if __name__ == "__main__":
    main()
