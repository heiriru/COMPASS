#!/usr/bin/env python3
"""Global-mode-first composition: DPM2 + gauss_jacobian -> KDE MAP -> local ascent.

The pipeline this script implements, in four stages:

1. **Joint draws.** DPM-2 with ``correction="gauss_jacobian"`` samples the full
   joint ``p(g, l_1..l_n | x_1..x_n)``. Dense Langevin correctors are on
   (``corrector_steps_interval=1``, ``corrector_steps=10``,
   ``final_corrector_steps=3``, ``snr=0.2``), matching the settings behind
   ``02f_dpm2_gauss_global_local.png``.
2. **Marginalize.** ``p(g | x)`` is the shared column of those draws. The
   projection is exact Monte Carlo; what the corrector sweeps add is the MCMC
   refinement of the draws being projected -- each sweep is a Langevin kernel on
   the composed bridging density, so the marginal is refined rather than merely
   read off a pure ODE trajectory.
3. **Mode.** A Gaussian KDE over the shared draws, evaluated on a fine grid and
   sharpened by a parabolic fit through the peak, gives ``g_hat``.
4. **Locals at fixed g.** Clamping ``g = g_hat`` makes the local problems
   *independent*: ``p(l_j | g_hat, x_j)`` involves observation ``j`` only, so no
   composition, no correction and no shared-precision weighting enter at all.
   Each local is then an ordinary annealed score ascent on its own row, with the
   shared coordinate conditioned rather than latent.

Stage 4 is the reason the pipeline is worth measuring: every compositional error
mode is confined to stage 1-3 and collapses onto a *single scalar*, ``g_hat``.
If the shared mode is right, the locals cost n independent single-observation
ascents and inherit no composition error. The corresponding risk is equally
sharp: the locals are conditioned on a point estimate, so they carry no
shared-parameter uncertainty, and any bias in ``g_hat`` propagates coherently
into all n of them.

Three problems, all producing the four-panel layout of
``02f_dpm2_gauss_global_local.png`` plus a second dashed vertical line for the
KDE MAP:

``gaussian``     the null-test problem of ``experiments.py``: exactly Gaussian
                 per-observation joints, analytic score. Every reference here
                 (shared posterior, local posterior means, conditional modes) is
                 closed-form.
``nongaussian``  the discriminating problem of ``experiments.py``: two-component
                 mixtures with opposite-sign g/l correlation. References come
                 from grid quadrature of the exact tall-data target. It has no
                 data-generating truth, so the "true global parameter" line is
                 absent by construction and the exact posterior mode is drawn
                 instead.
``trained``      the learned-network problem behind
                 ``local_std_1_obs_noise_0.02_extremely_small/``: the 20K-parameter
                 checkpoint, its 30 observations and its analytic Gaussian truth,
                 reused unchanged from ``reference.npz``.

Usage:
    python new_method.py                      # all three
    python new_method.py --problem gaussian
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)


NETWORK_PROBLEMS = ("nongaussian", "trained", "all")


def _wants_network_problem() -> bool:
    """Whether this invocation will touch a problem backed by a trained network.

    Read off argv before torch is imported, because reserving a GPU with
    ``autocvd`` only has an effect while CUDA_VISIBLE_DEVICES is still unread.
    """
    arguments = sys.argv[1:]
    if "--trained-device" in arguments:
        index = arguments.index("--trained-device")
        if index + 1 < len(arguments) and arguments[index + 1] == "cpu":
            return False
    selected = []
    for index, value in enumerate(arguments):
        if value == "--problem":
            for candidate in arguments[index + 1:]:
                if candidate.startswith("-"):
                    break
                selected.append(candidate)
    if not selected:
        return True                              # default is every problem
    return any(value in NETWORK_PROBLEMS for value in selected)


if _wants_network_problem() and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    # An explicit CUDA_VISIBLE_DEVICES is a deliberate choice (the capacity
    # sweep pins one GPU per process); autocvd would only re-pick inside it.
    try:
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=1)
    except Exception as error:                   # noqa: BLE001 - advisory only
        print(f"autocvd unavailable ({error}); falling back to whatever CUDA exposes")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from compass.MultiObsSampler import MultiObsSampler  # noqa: E402
from compass.SDE import VESDE  # noqa: E402

import experiments as base  # noqa: E402
import exponential_hierarchy as expo  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
TRAINED_RUN = ROOT.parent / "local_std_1_obs_noise_0.02_extremely_small"

NAVY, BLUE, TEAL, CORAL, GOLD = "#17223B", "#3A86FF", "#2A9D8F", "#EF476F", "#FFB703"

# The generative model of the Gaussian problems: x_j = g + l_j + N(0, SXH^2).
MU_G, S0G, S0L = 0.0, 1.0, 1.0
GAUSSIAN_SXH = 0.5                       # experiments.py's null test
TRAINED_SXH = 0.02                       # the extremely_small run's obs noise

SAMPLE_KWARGS = {
    "method": "dpm", "order": 2, "correction": "gauss_jacobian",
    "corrector_steps_interval": 1, "corrector_steps": 10,
    "final_corrector_steps": 3, "snr": 0.2,
}


# ---------------------------------------------------------------------------
# Stage 1-2: joint draws, and their shared marginal
# ---------------------------------------------------------------------------

def draw_joint(sbim, data, condition_mask, hierarchy, num_samples, timesteps,
               denoise_clamp, device, seed, verbose=False,
               prior=(MU_G, S0G), local_prior=(0.0, S0L), **overrides):
    """DPM-2 + gauss_jacobian draws of p(g, l_1..l_n | x), correctors included."""
    sampler = MultiObsSampler(sbim)
    arguments = dict(
        world_size=1, data=data, condition_mask=condition_mask,
        timesteps=timesteps, num_samples=num_samples, hierarchy=hierarchy,
        prior=(torch.tensor([float(prior[0])]), torch.tensor([float(prior[1])])),
        local_prior=(torch.tensor([float(local_prior[0])]),
                     torch.tensor([float(local_prior[1])])),
        denoise_clamp=denoise_clamp, device=device, verbose=verbose,
        **SAMPLE_KWARGS,
    )
    arguments.update(overrides)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    started = time.perf_counter()
    samples = sampler.sample(**arguments)
    runtime = time.perf_counter() - started
    samples = samples.detach().cpu()
    synchronization = float(
        (samples[:, :, hierarchy] - samples[:1, :, hierarchy]).abs().max()
    )
    if synchronization > 1e-5:
        raise AssertionError(
            f"shared draws are not synchronized across observations: {synchronization:.3e}"
        )
    return samples, sampler, runtime


# ---------------------------------------------------------------------------
# Stage 3: the KDE mode of the shared marginal
# ---------------------------------------------------------------------------

def kde_mode(draws, bandwidth=None, grid_size=4001, pad=0.15):
    """Peak of a Gaussian KDE over 1-D draws, refined by a parabolic fit.

    The grid alone resolves the mode only to its spacing; the three-point
    parabola through the peak recovers the continuous maximizer of the KDE, so
    the reported mode is a property of the density estimate rather than of the
    grid it was rendered on.
    """
    draws = np.asarray(draws, dtype=np.float64).reshape(-1)
    kde = gaussian_kde(draws, bw_method=bandwidth)
    span = draws.max() - draws.min()
    grid = np.linspace(draws.min() - pad * span, draws.max() + pad * span, grid_size)
    density = kde(grid)
    peak = int(np.argmax(density))
    mode = float(grid[peak])
    if 0 < peak < len(grid) - 1:
        left, centre, right = density[peak - 1], density[peak], density[peak + 1]
        curvature = left - 2 * centre + right
        if curvature < 0:
            step = 0.5 * (left - right) / curvature
            mode = float(grid[peak] + step * (grid[1] - grid[0]))
    return mode, grid, density, float(kde.factor * draws.std(ddof=1))


# ---------------------------------------------------------------------------
# Stage 4: locals by annealed score ascent with the shared coordinate clamped
# ---------------------------------------------------------------------------

@torch.no_grad()
def conditional_local_ascent(sbim, rows, condition_mask, timesteps=200,
                             iterations=3, eps=1e-3, sigma_start=1.0,
                             device="cpu"):
    """Annealed Tweedie ascent on the latent columns, conditioned columns fixed.

    ``z <- z + lambda^2 * s(z, lambda)`` converges to a mode of the
    lambda-smoothed density; annealing lambda down tracks it to the unsmoothed
    MAP. This is the same ascent as :meth:`PFODE.map_estimate`, written against
    ``output_scale_function`` instead of a hardcoded ``1 / sigma`` so it accepts
    both a trained network and the analytic score modules of ``experiments.py``.

    Every row is independent here -- with ``g`` conditioned there is nothing to
    compose -- so this is n single-observation problems solved in one batch.
    """
    sde = sbim.sde
    z = torch.as_tensor(rows, dtype=torch.float32).clone().to(device)
    mask = torch.as_tensor(condition_mask, dtype=torch.float32).to(device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).repeat(z.shape[0], 1)
    latent = 1.0 - mask

    one = torch.ones(1, device=device)
    lam_min = sde.lambda_t(eps * one).item()
    lam_max = sde.lambda_t(one).item()
    lam_hi = min(max(float(sigma_start), 2 * lam_min), lam_max)
    lams = torch.logspace(
        math.log10(lam_hi), math.log10(lam_min), int(timesteps), device=device
    )
    times = sde.time_of_lambda(lams)

    for index in range(int(timesteps)):
        t = times[index].reshape(1, 1)
        alpha = sde.alpha_t(t).to(device)
        lam_squared = float(lams[index].item()) ** 2
        for _ in range(int(iterations)):
            state = z * (alpha * latent + mask)
            score = sbim.output_scale_function(
                t, sbim.model(x=state, t=t, c=mask)
            )
            z = z + lam_squared * (alpha * score) * latent
    alpha_final = sde.alpha_t(times[-1]).to(device)
    return (z * (alpha_final * latent + mask)).cpu()


# ---------------------------------------------------------------------------
# Exact references
# ---------------------------------------------------------------------------

def gaussian_joint_posterior(x, sxh, s0g=S0G, s0l=S0L):
    """Exact p(g, l_1..l_n | x_1..x_n) for x_j = g + l_j + N(0, sxh^2)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    precision = np.zeros((n + 1, n + 1))
    precision[0, 0] = 1.0 / s0g**2 + n / sxh**2
    precision[0, 1:] = 1.0 / sxh**2
    precision[1:, 0] = 1.0 / sxh**2
    precision[1:, 1:] = np.eye(n) * (1.0 / s0l**2 + 1.0 / sxh**2)
    natural = np.concatenate([[x.sum() / sxh**2], x / sxh**2])
    covariance = np.linalg.inv(precision)
    return covariance @ natural, covariance


def gaussian_conditional_local(x, g, sxh, s0l=S0L):
    """Mode (= mean) and std of p(l_j | g, x_j) in the Gaussian model."""
    precision = 1.0 / s0l**2 + 1.0 / sxh**2
    mode = (np.asarray(x, dtype=np.float64).reshape(-1) - g) / sxh**2 / precision
    return mode, np.full(len(mode), precision**-0.5)


def grid_mode(grid, density):
    """Continuous argmax of a density tabulated on a grid (parabolic refine)."""
    peak = int(np.argmax(density))
    if 0 < peak < len(grid) - 1:
        left, centre, right = density[peak - 1], density[peak], density[peak + 1]
        curvature = left - 2 * centre + right
        if curvature < 0:
            step = 0.5 * (left - right) / curvature
            return float(grid[peak] + step * (grid[1] - grid[0]))
    return float(grid[peak])


def mixture_log_marginals(centres, grid_g):
    """log p(g | x_j) on a grid, one row per observation (mixture 1-D marginal)."""
    variance = base.COMPONENT_COVARIANCE[:, 0, 0]
    mean = centres[:, :, 0]
    exponent = -0.5 * (grid_g[None, None, :] - mean[:, :, None]) ** 2 \
        / variance[None, :, None]
    normalizer = -0.5 * torch.log(2 * math.pi * variance)[None, :, None]
    return torch.logsumexp(
        torch.log(base.WEIGHTS)[None, :, None] + normalizer + exponent, dim=1
    )


def mixture_log_joint(centres, grid_g, grid_l, observation):
    """log p(g, l_j | x_j) on a 2-D grid for one observation's mixture."""
    covariance = base.COMPONENT_COVARIANCE
    precision = torch.linalg.inv(covariance)
    _, logdet = torch.linalg.slogdet(covariance)
    points = torch.stack(torch.meshgrid(grid_g, grid_l, indexing="ij"), dim=-1)
    delta = points[:, :, None, :] - centres[observation][None, None, :, :]
    quadratic = torch.einsum("ghki,kij,ghkj->ghk", delta, precision, delta)
    return torch.logsumexp(
        torch.log(base.WEIGHTS)[None, None, :] - 0.5 * quadratic
        - 0.5 * logdet[None, None, :] - math.log(2 * math.pi),
        dim=-1,
    )


def mixture_local_moments(centres, observations, grid_g, grid_l):
    """Exact E[l_j | x_1..x_n] and its std, by quadrature over the 2-D grid.

    p(g, l_j | x_1..n) propto p(g)^(1-n) [prod_{k!=j} p(g | x_k)] p(g, l_j | x_j):
    every factor but observation j's own joint depends on g alone, so the tall
    weight is a 1-D function multiplied onto j's 2-D mixture.
    """
    log_prior = -0.5 * grid_g**2 - 0.5 * math.log(2 * math.pi)
    log_marginals = mixture_log_marginals(centres, grid_g)
    total = log_marginals.sum(dim=0)
    means, stds = [], []
    for observation in range(observations):
        weight = (1 - observations) * log_prior + (
            total - log_marginals[observation]
        )
        log_joint = mixture_log_joint(centres, grid_g, grid_l, observation)
        log_density = log_joint + weight[:, None]
        density = torch.softmax(log_density.reshape(-1), dim=0).reshape(
            log_density.shape
        )
        local = density.sum(dim=0)
        mean = float((local * grid_l).sum())
        means.append(mean)
        stds.append(float(((local * grid_l**2).sum() - mean**2).clamp_min(0.0) ** 0.5))
    return np.asarray(means), np.asarray(stds)


def mixture_conditional_local(centres, observations, g, grid_l):
    """Mode and std of p(l_j | g, x_j) for the mixture problem."""
    grid_g = torch.tensor([g], dtype=torch.float64)
    modes, stds = [], []
    for observation in range(observations):
        log_joint = mixture_log_joint(
            centres, grid_g, grid_l, observation
        ).reshape(-1)
        density = torch.softmax(log_joint, dim=0)
        modes.append(grid_mode(grid_l.numpy(), density.numpy()))
        mean = float((density * grid_l).sum())
        stds.append(float(((density * grid_l**2).sum() - mean**2).clamp_min(0.0) ** 0.5))
    return np.asarray(modes), np.asarray(stds)


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

def build_gaussian_problem(observations, seed, device):
    """experiments.py's null test, with a data-generating truth kept on record."""
    generator = torch.Generator().manual_seed(seed)
    global_truth = 0.7
    local_truth = torch.randn(observations, generator=generator)
    noise = torch.randn(observations, generator=generator)
    x = (global_truth + local_truth + GAUSSIAN_SXH * noise).to(torch.float64)

    means = (4.0 * x.to(torch.float32) / 9.0).unsqueeze(-1).repeat(1, 2)
    model = base.GaussianScore(means, VESDE(sigma=25.0))
    sbim = SimpleNamespace(
        sde=model.sde, model=model,
        output_scale_function=lambda t, value: value,
    )

    mean, covariance = gaussian_joint_posterior(x.numpy(), GAUSSIAN_SXH)
    global_std = float(covariance[0, 0] ** 0.5)
    grid = np.linspace(mean[0] - 5 * global_std, mean[0] + 5 * global_std, 2001)
    density = np.exp(-0.5 * ((grid - mean[0]) / global_std) ** 2) / (
        global_std * math.sqrt(2 * math.pi)
    )
    return {
        "name": "gaussian",
        "title": "DPM2 + gauss_jacobian -> KDE global MAP -> conditional local ascent"
                 "\n(Gaussian null test, analytic score)",
        "sbim": sbim,
        "data": torch.zeros(observations, 1),
        "condition_mask": base.CONDITION_MASK,
        "hierarchy": list(base.HIERARCHY),
        "nodes": base.NODES,
        "observations": observations,
        "device": device,
        "observation_values": x.numpy(),
        "observation_label": "observed x",
        "global_truth": global_truth,
        "local_truth": local_truth.numpy(),
        "global_grid": grid,
        "global_density": density,
        "global_mean": float(mean[0]),
        "global_std": global_std,
        "global_mode": float(mean[0]),
        "local_mean": mean[1:],
        "local_std": np.sqrt(np.diag(covariance)[1:]),
        "conditional_local": lambda g: gaussian_conditional_local(
            x.numpy(), g, GAUSSIAN_SXH
        ),
    }


def build_exponential_problem(observations, seed, device, args):
    """The generative g -> l_j -> x_j exponential hierarchy, learned score.

    Unlike the mixture problem this is simulated forward from one true g, so the
    figure carries a true-parameter line and the composed posterior is judged on
    recovery. See exponential_hierarchy.py for the closed-form references.
    """
    model = expo.train_or_load(
        args.model_dir, device, seed=args.train_seed,
        force=args.force_retrain, quick=args.quick_train, verbose=args.verbose,
    )
    global_truth, local_truth, x = expo.observations(observations, seed)
    grid, density, weights, mean, std = expo.shared_reference(x)
    local_mean, local_std = expo.local_reference(x, grid, weights)
    return {
        "name": "nongaussian",
        "title": "DPM2 + gauss_jacobian -> KDE global MAP -> conditional local ascent"
                 "\n(non-Gaussian exponential hierarchy g → ℓⱼ → xⱼ, learned score)",
        "sbim": model,
        "data": torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1),
        "condition_mask": torch.tensor([0.0, 0.0, 1.0]),
        "hierarchy": [0],
        "nodes": 3,
        "observations": observations,
        "device": device,
        "observation_values": np.asarray(x, dtype=np.float64),
        "observation_label": "observed x",
        "global_truth": global_truth,
        "local_truth": np.asarray(local_truth, dtype=np.float64),
        "global_grid": grid,
        "global_density": density,
        "global_mean": mean,
        "global_std": std,
        "global_mode": grid_mode(grid, density),
        "local_mean": local_mean,
        "local_std": local_std,
        "local_prior": (expo.LOCAL_PRIOR_MEAN, expo.LOCAL_PRIOR_STD),
        "conditional_local": lambda g: expo.conditional_local(x, g),
    }


def build_mixture_problem(observations, seed, device):
    """experiments.py's discriminating test: per-observation Gaussian mixtures."""
    centres = base.mixture_centres(observations, seed)
    model = base.MixtureScore(centres, VESDE(sigma=25.0))
    sbim = SimpleNamespace(
        sde=model.sde, model=model,
        output_scale_function=lambda t, value: value,
    )

    grid_g = torch.linspace(-4.0, 4.0, 2001, dtype=torch.float64)
    grid_l = torch.linspace(-4.0, 4.0, 2001, dtype=torch.float64)
    log_shared = base.exact_shared_marginal(centres, grid_g, observations)
    shared = torch.softmax(log_shared, dim=0)
    spacing = float(grid_g[1] - grid_g[0])
    density = (shared / spacing).numpy()
    grid = grid_g.numpy()
    mean = float((shared * grid_g).sum())
    std = float(((shared * grid_g**2).sum() - mean**2) ** 0.5)
    local_mean, local_std = mixture_local_moments(
        centres, observations, grid_g, grid_l
    )
    # Each observation's own posterior mean for the shared coordinate: the
    # closest thing this problem has to an observation in x-space, since the
    # mixtures are specified as posteriors and no x_j is ever materialized.
    weights = base.WEIGHTS.numpy()
    summary = (centres.numpy()[:, :, 0] * weights[None, :]).sum(axis=1)
    return {
        "name": "mixture",
        "title": "DPM2 + gauss_jacobian -> KDE global MAP -> conditional local ascent"
                 "\n(non-Gaussian mixture test, analytic score)",
        "sbim": sbim,
        "data": torch.zeros(observations, 1),
        "condition_mask": base.CONDITION_MASK,
        "hierarchy": list(base.HIERARCHY),
        "nodes": base.NODES,
        "observations": observations,
        "device": device,
        "observation_values": summary,
        "observation_label": "per-observation posterior mean E[g | x$_j$]",
        "global_truth": None,
        "local_truth": None,
        "global_grid": grid,
        "global_density": density,
        "global_mean": mean,
        "global_std": std,
        "global_mode": grid_mode(grid, density),
        "local_mean": local_mean,
        "local_std": local_std,
        "conditional_local": lambda g: mixture_conditional_local(
            centres, observations, g, grid_l
        ),
    }


def build_trained_problem(device, run_directory=TRAINED_RUN):
    """The learned-network problem behind 02f_dpm2_gauss_global_local.png."""
    from compass import ScoreBasedInferenceModel as SBIm

    reference_path = run_directory / "reference.npz"
    config_path = run_directory / "run_config.json"
    if not reference_path.exists():
        raise FileNotFoundError(f"missing {reference_path}")
    config = json.loads(config_path.read_text())
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint {checkpoint}")
    with np.load(reference_path) as archive:
        reference = {key: archive[key] for key in archive.files}

    model = SBIm.load(str(checkpoint), device=device)
    x = np.asarray(reference["x_observed"], dtype=np.float64).reshape(-1)
    observations = len(x)
    mean = np.asarray(reference["exact_joint_mean"], dtype=np.float64)
    covariance = np.asarray(reference["exact_joint_covariance"], dtype=np.float64)
    global_std = float(covariance[0, 0] ** 0.5)
    grid = np.linspace(mean[0] - 5 * global_std, mean[0] + 5 * global_std, 2001)
    density = np.exp(-0.5 * ((grid - mean[0]) / global_std) ** 2) / (
        global_std * math.sqrt(2 * math.pi)
    )
    obs_noise = float(config.get("obs_noise", TRAINED_SXH))
    return {
        "name": "trained",
        "title": "DPM2 + gauss_jacobian -> KDE global MAP -> conditional local ascent"
                 f"\n(trained {config['model_size']} network, S0L={config['local_std']:g},"
                 f" obs. noise={obs_noise:g})",
        "sbim": model,
        "data": torch.as_tensor(reference["x_observed"], dtype=torch.float32),
        "condition_mask": torch.tensor([0.0, 0.0, 1.0]),
        "hierarchy": [0],
        "nodes": 3,
        "observations": observations,
        "device": device,
        "observation_values": x,
        "observation_label": "observed x",
        "global_truth": float(reference["global_truth"]),
        "local_truth": np.asarray(reference["local_truth"], dtype=np.float64).reshape(-1),
        "global_grid": grid,
        "global_density": density,
        "global_mean": float(mean[0]),
        "global_std": global_std,
        "global_mode": float(mean[0]),
        "local_mean": mean[1:],
        "local_std": np.sqrt(np.diag(covariance)[1:]),
        "conditional_local": lambda g: gaussian_conditional_local(x, g, obs_noise),
    }


BUILDERS = {
    "gaussian": lambda args: build_gaussian_problem(
        args.observations, args.seed, args.device
    ),
    "nongaussian": lambda args: build_exponential_problem(
        args.observations, args.seed, args.trained_device, args
    ),
    "mixture": lambda args: build_mixture_problem(
        args.observations, args.seed, args.device
    ),
    "trained": lambda args: build_trained_problem(args.trained_device),
}


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def run_problem(problem, args):
    device = problem["device"]
    hierarchy = problem["hierarchy"]
    local_index = [
        index for index in range(problem["nodes"])
        if index not in hierarchy
        and float(torch.as_tensor(problem["condition_mask"])[index]) == 0.0
    ]
    if len(local_index) != 1:
        raise ValueError(f"expected exactly one local coordinate, got {local_index}")
    local_index = local_index[0]

    print(f"[{problem['name']}] stage 1: DPM2 + gauss_jacobian joint draws "
          f"({problem['observations']} observations, {args.num_samples} draws, "
          f"{args.timesteps} steps, correctors on) on {device}")
    samples, sampler, sample_runtime = draw_joint(
        problem["sbim"], problem["data"], problem["condition_mask"], hierarchy,
        args.num_samples, args.timesteps, args.denoise_clamp, device, args.seed,
        verbose=args.verbose,
        prior=problem.get("prior", (MU_G, S0G)),
        local_prior=problem.get("local_prior", (0.0, S0L)),
    )
    shared_draws = samples[0, :, hierarchy[0]].to(torch.float64).numpy()
    local_draws = samples[:, :, local_index].to(torch.float64).numpy()

    # Guard the KDE against a solver excursion: one draw many orders of
    # magnitude out would put the whole kernel bandwidth (Scott's rule scales
    # with the sample std) beyond the posterior width and flatten the peak.
    limit = args.excursion_sigma * problem["global_std"]
    keep = np.isfinite(shared_draws) & (
        np.abs(shared_draws - problem["global_mean"]) <= limit
    )
    excluded = int((~keep).sum())
    if excluded:
        print(f"[{problem['name']}] excluded {excluded} shared draw(s) beyond "
              f"{args.excursion_sigma:g} sigma before the KDE")
    kept_shared, kept_local = shared_draws[keep], local_draws[:, keep]

    print(f"[{problem['name']}] stage 3: KDE mode of the shared marginal")
    global_map, kde_grid, kde_density, bandwidth = kde_mode(
        kept_shared, args.kde_bandwidth
    )

    print(f"[{problem['name']}] stage 4: local ascent at g = {global_map:.5f}")
    rows = torch.zeros(problem["observations"], problem["nodes"])
    mask = torch.as_tensor(problem["condition_mask"], dtype=torch.float32)
    mask = mask.unsqueeze(0).repeat(problem["observations"], 1).clone()
    data = torch.as_tensor(problem["data"], dtype=torch.float32)
    observed = torch.where(mask[0] == 1)[0].tolist()
    rows[:, observed] = data.reshape(problem["observations"], -1)
    rows[:, hierarchy[0]] = float(global_map)
    rows[:, local_index] = torch.as_tensor(
        kept_local.mean(axis=1), dtype=torch.float32
    )
    mask[:, hierarchy[0]] = 1.0                  # the shared coordinate is fixed
    started = time.perf_counter()
    ascended = conditional_local_ascent(
        problem["sbim"], rows, mask, timesteps=args.map_timesteps,
        iterations=args.map_iterations, eps=args.map_eps,
        sigma_start=max(float(kept_local.std(axis=1).max()), 1e-3),
        device=device,
    )
    ascent_runtime = time.perf_counter() - started
    local_map = ascended[:, local_index].to(torch.float64).numpy()

    conditional_mode, conditional_std = problem["conditional_local"](global_map)
    metrics = {
        "problem": problem["name"],
        "observations": problem["observations"],
        "num_samples": args.num_samples,
        "timesteps": args.timesteps,
        "excluded_draws": excluded,
        "kde_bandwidth": bandwidth,
        "global_map": global_map,
        "global_map_error_sigma": abs(global_map - problem["global_mode"])
        / problem["global_std"],
        "global_sample_mean_error_sigma": abs(
            kept_shared.mean() - problem["global_mean"]
        ) / problem["global_std"],
        "global_width_ratio": float(kept_shared.std(ddof=1)) / problem["global_std"],
        "local_map_error_vs_posterior_mean_sigma": float(np.mean(
            np.abs(local_map - problem["local_mean"]) / problem["local_std"]
        )),
        "local_map_error_vs_conditional_mode_sigma": float(np.mean(
            np.abs(local_map - conditional_mode) / conditional_std
        )),
        "local_draw_mean_error_sigma": float(np.mean(
            np.abs(kept_local.mean(axis=1) - problem["local_mean"])
            / problem["local_std"]
        )),
        "sample_runtime_seconds": sample_runtime,
        "ascent_runtime_seconds": ascent_runtime,
        "sampler_network_calls": int(sampler.score_network_calls),
        "ascent_network_calls": int(args.map_timesteps * args.map_iterations),
    }
    if problem["global_truth"] is not None:
        metrics["global_map_error_from_truth_sigma"] = abs(
            global_map - problem["global_truth"]
        ) / problem["global_std"]
    result = {
        "shared_draws": shared_draws, "local_draws": local_draws,
        "kept_shared": kept_shared, "kept_local": kept_local,
        "kde_grid": kde_grid, "kde_density": kde_density,
        "global_map": global_map, "local_map": local_map,
        "conditional_mode": conditional_mode, "conditional_std": conditional_std,
        "excluded": excluded,
    }
    return result, metrics


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

def configure_style():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 240, "font.size": 10,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def plot_problem(problem, result, metrics, output):
    observation_values = np.asarray(problem["observation_values"], dtype=float).reshape(-1)
    n = problem["observations"]
    indices = np.arange(n)
    local_draw_mean = result["kept_local"].mean(axis=1)
    local_draw_std = result["kept_local"].std(axis=1, ddof=1)

    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.5))
    fig.suptitle(problem["title"], fontsize=14, fontweight="bold", color=NAVY)

    # Panel 1 -- the observations, stacked as a dot plot in observational space.
    bins = min(15, n)
    counts, edges = np.histogram(observation_values, bins=bins)
    bin_index = np.clip(np.digitize(observation_values, edges[1:-1]), 0, bins - 1)
    height = np.zeros(bins, dtype=int)
    position = np.empty(n, dtype=int)
    for i in np.argsort(observation_values):
        position[i] = height[bin_index[i]]
        height[bin_index[i]] += 1
    centres = (edges[:-1] + edges[1:]) / 2
    axes[0].scatter(centres[bin_index], position + 0.5, s=80, color=BLUE,
                    alpha=0.85, edgecolor="white", linewidth=0.6)
    axes[0].axvline(observation_values.mean(), color="red", linestyle=":",
                    linewidth=2, label="mean observation")
    axes[0].set(xlabel=problem["observation_label"],
                ylabel="observations stacked per bin",
                title="Observations (observational space)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    del counts

    # Panel 2 -- the shared posterior, its exact reference, and the two modes.
    axes[1].hist(result["kept_shared"], bins=46, density=True, color=BLUE,
                 alpha=0.72, label="COMPASS draws")
    axes[1].plot(problem["global_grid"], problem["global_density"], color=NAVY,
                 ls="--", lw=2, label="exact")
    axes[1].plot(result["kde_grid"], result["kde_density"], color=GOLD, lw=1.8,
                 label="KDE of draws")
    if problem["global_truth"] is not None:
        axes[1].axvline(problem["global_truth"], color="red", linestyle=":",
                        linewidth=2, label="true global parameter")
    else:
        axes[1].axvline(problem["global_mode"], color="red", linestyle=":",
                        linewidth=2, label="exact posterior mode")
    axes[1].axvline(result["global_map"], color=TEAL, linestyle="--", linewidth=2,
                    label="KDE MAP (this method)")
    lo = problem["global_mean"] - 4 * problem["global_std"]
    hi = problem["global_mean"] + 4 * problem["global_std"]
    axes[1].set_xlim(lo, hi)
    axes[1].set(xlabel="global parameter g", ylabel="density",
                title="Shared posterior")
    axes[1].legend(fontsize=8.5)
    note = f"KDE MAP off exact mode by {metrics['global_map_error_sigma']:.2f} σ"
    if result["excluded"]:
        note += f"; {result['excluded']} solver excursion(s) excluded"
    # Below the axes, not inside them: the legend already occupies the only
    # corner that is reliably empty across all three problems.
    axes[1].text(0.5, -0.26, note, transform=axes[1].transAxes, ha="center",
                 va="top", color=NAVY, fontsize=9)

    # Panel 3 -- the locals. The joint draws give a spread; the pipeline's own
    # answer is a point estimate conditioned on g_hat, so it is drawn as such.
    axes[2].errorbar(indices, local_draw_mean, yerr=local_draw_std, fmt="o", ms=3,
                     color=CORAL, ecolor="#F4A3B4", alpha=0.75,
                     label="joint draws, mean ± σ")
    axes[2].plot(indices, result["local_map"], "d", ms=5, color=TEAL,
                 label="ascent MAP at g = ĝ")
    axes[2].plot(indices, problem["local_mean"], "_", ms=9, color=NAVY,
                 label="exact mean")
    axes[2].set(xlabel="observation", ylabel="local parameter ℓᵢ",
                title=f"{n} local posteriors")
    axes[2].legend(fontsize=8.5)

    # Panel 4 -- recovery of the locals against their exact posterior means.
    axes[3].scatter(problem["local_mean"], result["local_map"], c=indices,
                    cmap="viridis", s=38, edgecolor="white", linewidth=0.5)
    lo = min(problem["local_mean"].min(), result["local_map"].min())
    hi = max(problem["local_mean"].max(), result["local_map"].max())
    axes[3].plot([lo, hi], [lo, hi], color=NAVY, ls="--", lw=1.6)
    axes[3].text(
        0.04, 0.94,
        "mean |error| = "
        f"{metrics['local_map_error_vs_posterior_mean_sigma']:.2f} analytic σ\n"
        "vs exact p(ℓ|ĝ,x) mode: "
        f"{metrics['local_map_error_vs_conditional_mode_sigma']:.2f} σ",
        transform=axes[3].transAxes, va="top", color=NAVY, fontsize=9,
    )
    axes[3].set(xlabel="exact local posterior mean",
                ylabel="ascent MAP at g = ĝ", title="Local recovery")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

FIELDS = [
    "problem", "observations", "num_samples", "timesteps", "excluded_draws",
    "kde_bandwidth", "global_map", "global_map_error_sigma",
    "global_map_error_from_truth_sigma", "global_sample_mean_error_sigma",
    "global_width_ratio", "local_map_error_vs_posterior_mean_sigma",
    "local_map_error_vs_conditional_mode_sigma", "local_draw_mean_error_sigma",
    "sample_runtime_seconds", "ascent_runtime_seconds", "sampler_network_calls",
    "ascent_network_calls",
]


def write_metrics(path, rows):
    """Merge this run's rows into the table, keyed by problem.

    Rewriting the file with only the problems just run would silently drop the
    others, and a `--problem gaussian` run is the normal way to iterate -- so
    the row is replaced, not the table.
    """
    merged = {}
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                merged[row["problem"]] = row
    for row in rows:
        merged[row["problem"]] = {
            field: row.get(field, "") for field in FIELDS
        }
    order = ["gaussian", "nongaussian", "mixture", "trained"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for name in sorted(merged, key=lambda item: (order + [item]).index(item)):
            writer.writerow({field: merged[name].get(field, "") for field in FIELDS})
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--problem", nargs="+", default=["all"],
                        choices=["gaussian", "nongaussian", "mixture",
                                 "trained", "all"],
                        help="'all' runs gaussian, nongaussian and trained; "
                             "'mixture' is the older analytic-score mixture "
                             "problem, kept but not part of 'all'.")
    parser.add_argument("--observations", type=int, default=30,
                        help="Observation count for the two analytic problems; "
                             "the trained problem uses its own 30.")
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--excursion-sigma", type=float, default=15.0)
    parser.add_argument("--kde-bandwidth", default=None,
                        help="scipy bw_method: 'scott' (default), 'silverman' "
                             "or a float scaling.")
    parser.add_argument("--map-timesteps", type=int, default=200)
    parser.add_argument("--map-iterations", type=int, default=3)
    parser.add_argument("--map-eps", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu",
                        help="Device for the analytic problems.")
    parser.add_argument("--trained-device", default=None,
                        help="Device for the network-backed problems "
                             "(defaults to cuda when available).")
    parser.add_argument("--model-dir", type=Path,
                        default=ARTIFACTS / "models" / "exponential_hierarchy",
                        help="Checkpoint directory for the exponential "
                             "hierarchy's score network.")
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--quick-train", action="store_true",
                        help="Tiny training run, for a pipeline smoke test only.")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--replot", action="store_true",
                        help="Redraw the figures from the saved .npz/.csv "
                             "without resampling. The problem definitions are "
                             "deterministic given --seed, so the references are "
                             "rebuilt rather than stored.")
    arguments = parser.parse_args()

    if arguments.kde_bandwidth is not None:
        try:
            arguments.kde_bandwidth = float(arguments.kde_bandwidth)
        except ValueError:
            pass
    if arguments.trained_device is None:
        arguments.trained_device = "cuda" if torch.cuda.is_available() else "cpu"

    names = ["gaussian", "nongaussian", "trained"] if "all" in arguments.problem \
        else list(dict.fromkeys(arguments.problem))
    configure_style()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    if arguments.replot:
        with (arguments.output_dir / "new_method_metrics.csv").open(newline="") as handle:
            stored = {row["problem"]: row for row in csv.DictReader(handle)}
        for name in names:
            problem = BUILDERS[name](arguments)
            with np.load(arguments.output_dir / f"new_method_{name}.npz") as archive:
                result = {key: archive[key] for key in archive.files}
            result["global_map"] = float(result["global_map"])
            result["excluded"] = int(result["excluded"])
            metrics = {
                key: (float(value) if value not in ("", None) else float("nan"))
                for key, value in stored[name].items() if key != "problem"
            }
            plot_problem(
                problem, result, metrics,
                arguments.output_dir / f"03_new_method_{name}.png",
            )
        return

    rows = []
    for name in names:
        problem = BUILDERS[name](arguments)
        result, metrics = run_problem(problem, arguments)
        rows.append(metrics)
        np.savez_compressed(
            arguments.output_dir / f"new_method_{name}.npz",
            observation_values=problem["observation_values"],
            global_grid=problem["global_grid"],
            global_density=problem["global_density"],
            exact_local_mean=problem["local_mean"],
            exact_local_std=problem["local_std"],
            global_truth=(np.nan if problem["global_truth"] is None
                          else problem["global_truth"]),
            **{key: value for key, value in result.items()},
        )
        plot_problem(
            problem, result, metrics,
            arguments.output_dir / f"03_new_method_{name}.png",
        )
        print(json.dumps(metrics, indent=2, default=float))
    write_metrics(arguments.output_dir / "new_method_metrics.csv", rows)


if __name__ == "__main__":
    main()
