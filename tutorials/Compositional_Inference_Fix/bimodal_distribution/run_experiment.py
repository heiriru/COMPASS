#!/usr/bin/env python3
"""Compositional inference on the two bimodal benchmarks, learned and exact.

For each benchmark (``sign_model``, ``control_model``) this runs every sampler
arm once per score source: two trained networks of different size, and the
*closed-form* row score of ``row_scores.ExactRowScoreNetwork``. Every arm uses
the identical observations, seed, sample count, step count, composition rule and
integrator, so the difference between the runs is row-score error and nothing
else -- and the analytic run's residual error is the composition rule, the
initial law and the integrator, with no network in the loop at all.

The arms
--------
``dpm2_churn8_gauss_hierarchical``  DPM-2 with Karras churn ``eta = 8``,
``dpm2_churn8_gauss_jacobian``      corrector-free, on each composition rule.
                                    ``eta = 8`` is the strongest churn the
                                    predictor can express: the per-step noise-up
                                    ratio is capped at ``sqrt(2) - 1``, so past
                                    ``eta ~ 4`` on this schedule every step is
                                    already at the cap.
``langevin_fnpe``                   Annealed Langevin with F-NPSE bridging
                                    densities -- Geffner et al.'s method, its
                                    own native sampler.
``dpm2_gauss_hierarchical``         The stock dense-corrector DPM-2 arms
``dpm2_gauss_jacobian``             (10 correctors per level, snr 0.2), as the
                                    no-churn baseline the churned arms are read
                                    against.

The score sources
-----------------
``learned``        the small h32d3 network (552K parameters)
``learned_large``  the stock h128d6 backbone (5.2M), to separate "the small
                   network could not represent the row score" from "the
                   composition rule is wrong"
``analytic``       the closed form -- no network at all

What is measured
----------------
Two layers, both graded against exact references:

1. *The composed score itself*, at the noise levels the sampler queries, on
   states drawn from the exact diffused posterior. This is the rule, isolated
   from the sampler entirely.
2. *The draws*: W1 against the exact marginals, whether both modes survive,
   how many landed in the gap between them, how the mass splits, how wide each
   mode is, and how tightly the locals still track the shared coordinate.

Usage:
    python run_experiment.py                        # everything
    python run_experiment.py --models sign
    python run_experiment.py --methods langevin_fnpe --scores analytic
    python run_experiment.py --force                # ignore cached .npz files
    python run_experiment.py --recompute-metrics    # redo the metrics only
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
import contextlib  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import churn  # noqa: E402
import control_model  # noqa: E402
import mixture_reference as reference  # noqa: E402
import row_scores  # noqa: E402
import sign_model  # noqa: E402
import train as recipe  # noqa: E402

ARTIFACTS = ROOT / "artifacts"

MODELS = {"sign": sign_model, "control": control_model}

OBSERVATIONS = 8
# One seed per benchmark. The sign model's is arbitrary -- every dataset it
# produces is bimodal, because the ambiguity lives in the likelihood. The
# control's is *chosen*: with a prior gap this well separated most datasets let
# the data resolve the lobe outright and leave a unimodal posterior, so a seed
# whose exact posterior does retain both lobes (weights 0.36 / 0.64) is used,
# otherwise the arms would be graded on a target with nothing to preserve. See
# control_model.py.
DATA_SEEDS = {"sign": 20260903, "control": 20260911}
INFERENCE_SEED = 1208
NUM_SAMPLES = 3_000
TIMESTEPS = 100
PRECISION_EST_SAMPLES = 4096
SCORE_LAMBDAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
SCORE_STATES = 512
FIDELITY_LAMBDAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)

CHURN_ETA = 8.0

# The library default, and the measured optimum once the score is *learned*: on
# the exponential hierarchy the corrector-free optimum drops from eta = 4 with an
# exact row score to eta = 2 with a learned one. CHURN_ETA = 8 is the strongest
# churn the predictor can express (past eta ~ 4 every step is already at the
# sqrt(2)-1 cap), so the two arms bracket the useful range.
CHURN_ETA_DEFAULT = 2.0

# Retained for reproducing the pre-existing corrector arms with
# `--methods dpm2_gauss_hierarchical_correctors ...`; not part of the default sweep.
CORRECTOR = {"corrector_steps_interval": 1, "corrector_steps": 10,
             "final_corrector_steps": 3, "snr": 0.2, "churn_eta": 0.0}
CORRECTOR_FREE = {"corrector_steps": 0, "final_corrector_steps": 0,
                  "terminal_corrector_steps": 0}

NAVY, BLUE, TEAL, CORAL, GOLD = (
    "#17223B", "#3A86FF", "#2A9D8F", "#EF476F", "#FFB703"
)

# Pure churn, by decision rather than by default.
#
# The ten-corrector DPM-2 arms were the two best in RESULTS.md *with the exact
# row score* (W1/sigma 0.013-0.020 against 0.043-0.075 for the churned ones), so
# dropping them is not free and should not be reported as if it were. The reason
# to drop them anyway is that the ordering reverses under a learned score, which
# is the only score that ever ships: with the 552K and 5.2M networks the
# corrector arms go to 0.185/0.190 and 0.093/0.127 -- no better with nine times
# the capacity, and for gauss_jacobian worse -- while the corrector-free churned
# arms track the row score down (0.112 -> 0.081 -> 0.075 and 0.090 -> 0.056 ->
# 0.043). A long Langevin chain equilibrates to the stationary law of a slightly
# over-confident field; a predictor integrates that error once.
#
# `langevin_fnpe` stays. It is not a Langevin corrector bolted onto our
# predictor -- it is Geffner et al.'s algorithm, the external baseline the rules
# are measured against, and removing it would remove the comparison rather than
# a corrector.
METHODS = {
    "dpm2_churn2_gauss_hierarchical": {
        "label": f"DPM2 + churn {CHURN_ETA_DEFAULT:g} + gauss_hierarchical",
        "short": "H, churn 2",
        "colour": CORAL, "churn": CHURN_ETA_DEFAULT, "linestyle": "--",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
            "precision_est_samples": PRECISION_EST_SAMPLES,
            "precision_est_timesteps": TIMESTEPS, **CORRECTOR_FREE,
        },
    },
    "dpm2_churn2_gauss_jacobian": {
        "label": f"DPM2 + churn {CHURN_ETA_DEFAULT:g} + gauss_jacobian",
        "short": "J, churn 2",
        "colour": TEAL, "churn": CHURN_ETA_DEFAULT, "linestyle": "--",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_jacobian",
            **CORRECTOR_FREE,
        },
    },
    "dpm2_churn8_gauss_hierarchical": {
        "label": f"DPM2 + churn {CHURN_ETA:g} + gauss_hierarchical",
        "short": "H, churn 8",
        "colour": CORAL, "churn": CHURN_ETA, "linestyle": "-",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
            "precision_est_samples": PRECISION_EST_SAMPLES,
            "precision_est_timesteps": TIMESTEPS, **CORRECTOR_FREE,
        },
    },
    "dpm2_churn8_gauss_jacobian": {
        "label": f"DPM2 + churn {CHURN_ETA:g} + gauss_jacobian",
        "short": "J, churn 8",
        "colour": TEAL, "churn": CHURN_ETA, "linestyle": "-",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_jacobian",
            **CORRECTOR_FREE,
        },
    },
    "langevin_fnpe": {
        "label": "Langevin + F-NPSE",
        "short": "Langevin F-NPSE",
        "colour": GOLD, "churn": None, "linestyle": "-",
        "sample_kwargs": {
            "method": "langevin", "correction": "fnpe",
            "corrector_steps": 10, "snr": 0.2,
        },
    },
    # --- retained, not in the default sweep: the corrector arms RESULTS.md ran ---
    "dpm2_gauss_hierarchical_correctors": {
        "label": "DPM2 + gauss_hierarchical (10 correctors)",
        "short": "H, correctors",
        "colour": CORAL, "churn": None, "linestyle": ":",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
            "precision_est_samples": PRECISION_EST_SAMPLES,
            "precision_est_timesteps": TIMESTEPS, **CORRECTOR,
        },
    },
    "dpm2_gauss_jacobian_correctors": {
        "label": "DPM2 + gauss_jacobian (10 correctors)",
        "short": "J, correctors",
        "colour": TEAL, "churn": None, "linestyle": ":",
        "sample_kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_jacobian",
            **CORRECTOR,
        },
    },
}

# The corrector arms are reachable by name but excluded from the default sweep,
# so `run_experiment.py` with no --methods is pure churn plus the F-NPSE baseline.
DEFAULT_METHODS = tuple(
    name for name in METHODS if not name.endswith("_correctors")
)

# Three score sources, not two. The small network is the one the brief asks for
# -- "efficiently small to learn it" -- but its row score is only accurate to
# ~17% relative RMS, and at the smallest probed noise level to 44%, so a
# learned-arm failure could be capacity rather than composition. The stock
# 128-wide backbone reaches ~6% on the same grade, and the closed form is exact.
# Running all three turns "is this the network or the rule?" from an argument
# into a row of the table.
SCORES = ("learned", "learned_large", "analytic")
SCORE_CONFIG = {"learned": recipe.DEFAULT_CONFIG, "learned_large": "h128d6"}


# ---------------------------------------------------------------------------
# The problem: observations and every exact reference the metrics need
# ---------------------------------------------------------------------------

def build_problem(module, observations, seed, device):
    truth, local_truth, discrete, y = module.observations(observations, seed)
    grid, density, mass, mean, std = reference.shared_reference(module, y)
    local = [reference.local_reference(module, y, index)
             for index in range(observations)]
    weights, means, variance = reference.marginal(module, y, 0)
    exact_shared, exact_local = reference.joint_draws(
        module, y, 200_000, seed=seed + 1
    )
    # Resolve structure at the scale of the exact posterior's own modes, not at
    # the scale of the gap between them; see kde_density.
    #
    # The exact posterior seen through that same kernel needs no KDE at all: a
    # Gaussian mixture convolved with a Gaussian of width b is the same mixture
    # with b^2 added to its variance. So the reference the sampled mode counts
    # and valley depths are graded against is closed form, and is exactly what a
    # perfect sampler's KDE would converge to.
    bandwidth = 0.45 * float(np.sqrt(variance))
    smoothed = reference.component_density(
        grid, weights, means, variance + bandwidth**2
    )
    return {
        "module": module, "device": device, "observations": observations,
        "y": np.asarray(y, dtype=np.float64),
        "data": torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1),
        "global_truth": float(truth),
        "local_truth": np.asarray(local_truth, dtype=np.float64),
        "discrete_truth": np.asarray(discrete, dtype=np.float64),
        "global_grid": grid, "global_density": density, "global_mass": mass,
        "global_mean": mean, "global_std": std,
        "global_modes": reference.mode_locations(module, y),
        "local_grids": np.stack([entry[0] for entry in local]),
        "local_densities": np.stack([entry[1] for entry in local]),
        "local_masses": np.stack([entry[2] for entry in local]),
        "local_means": np.array([entry[3] for entry in local]),
        "local_stds": np.array([entry[4] for entry in local]),
        "aligned_mass": reference.aligned_mass(module, y),
        "valley_depth": reference.valley_depth(module, y),
        "kde_bandwidth": bandwidth,
        # The reference every sampled arm is graded against for mode count and
        # valley depth: the exact posterior seen through the same kernel, built
        # from exact draws at the same resolution.
        "smoothed_valley_depth": float(
            np.log(max(smoothed.max(), 1e-300))
            - np.log(max(float(np.interp(0.0, grid, smoothed)), 1e-300))
        ),
        "smoothed_density": smoothed,
        "smoothed_modes": prominent_maxima(grid, smoothed),
        "positive_mass": float(weights[means > 0].sum()),
        "exact_shared": exact_shared, "exact_local": exact_local,
    }


def prior_for(module):
    """The Gaussian prior handed to the sampler's ``(1 - n)`` prior-score term.

    Exact for ``sign_model``; a moment-matched stand-in for ``control_model``,
    whose true prior over ``g`` is itself bimodal (see that module's docstring).
    """
    std = getattr(module, "PRIOR_STD", getattr(module, "SIGMA_G", None))
    return (torch.tensor([module.MU_G]), torch.tensor([float(std)]))


# ---------------------------------------------------------------------------
# Running one arm
# ---------------------------------------------------------------------------

def churn_context(eta):
    """Swap in the corrector-free churned DPM-2 predictor, or do nothing."""
    if eta is None:
        return contextlib.nullcontext()
    return churn.patched_dpm_sampler(churn.churned_dpm_sampler(float(eta)))


def _draw_once(model, problem, method, num_samples, timesteps, seed, verbose):
    """One call into ``MultiObsSampler.sample``, with this arm's configuration."""
    from compass.MultiObsSampler import MultiObsSampler

    module = problem["module"]
    entry = METHODS[method]
    sampler = MultiObsSampler(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with churn_context(entry["churn"]):
        samples = sampler.sample(
            world_size=1, data=problem["data"],
            condition_mask=module.CONDITION_MASK, timesteps=timesteps,
            num_samples=num_samples, hierarchy=list(module.HIERARCHY),
            prior=prior_for(module),
            local_prior=(torch.tensor([module.LOCAL_PRIOR_MEAN]),
                         torch.tensor([module.LOCAL_PRIOR_STD])),
            device=problem["device"], verbose=verbose, **entry["sample_kwargs"],
        )
    return samples.detach().cpu(), sampler


def draw(model, problem, method, num_samples, timesteps, seed, verbose=False,
         chunk=None):
    """Run one arm and return its draws, the sampler that produced them, runtime.

    ``MultiObsSampler`` holds every observation row and every posterior draw on
    the device at once, and ``correction="gauss_jacobian"`` additionally carries
    forward-mode duals through the network, so the widest backbone here runs out
    of memory at 3,000 draws on an 11 GB card. The draws are i.i.d. -- nothing in
    the sampler couples one posterior sample to another -- so they are taken in
    chunks and concatenated, which is statistically identical to one call. The
    chunk size is halved automatically on an out-of-memory error rather than
    hard-coded, so a smaller network still pays for one pass and the numbers stay
    comparable across arms.

    The one thing chunking does change is cost: ``gauss_hierarchical`` re-runs
    its pilot covariance estimate per chunk, so its reported wall clock is an
    over-estimate whenever more than one chunk was needed. The sampler returned
    is the last chunk's, which is what ``probe_composed_score`` needs (its
    configuration, and its pilot covariance, are the same in every chunk).
    """
    size = int(chunk or num_samples)
    while True:
        try:
            blocks, sampler, calls = [], None, 0
            started = time.perf_counter()
            taken = 0
            while taken < num_samples:
                count = min(size, num_samples - taken)
                block, sampler = _draw_once(
                    model, problem, method, count, timesteps,
                    seed + 7919 * len(blocks), verbose,
                )
                calls += int(getattr(sampler, "score_network_calls", 0))
                blocks.append(block)
                taken += count
            runtime = time.perf_counter() - started
            break
        except torch.OutOfMemoryError:
            if size <= 125:
                raise
            size //= 2
            # Drop the failed attempt's tensors *before* emptying the cache;
            # otherwise the sampler and the finished blocks still reference
            # device memory and the retry starts from the same ceiling.
            blocks, sampler = [], None
            torch.cuda.empty_cache()
            print(f"    out of memory; retrying in chunks of {size} draws",
                  flush=True)

    samples = torch.cat(blocks, dim=1)
    sampler.score_network_calls = calls
    module = problem["module"]
    synchronization = float(
        (samples[:, :, module.HIERARCHY] - samples[:1, :, module.HIERARCHY])
        .abs().max()
    )
    if synchronization > 1e-5:
        raise AssertionError(
            f"{method}: shared draws are not synchronized across observations "
            f"({synchronization:.3e})"
        )
    return samples, sampler, runtime


@torch.no_grad()
def probe_composed_score(sampler, problem, lambdas, states, seed):
    """Relative RMS error of the rule's composed score, per noise level.

    ``sampler`` must already have run :meth:`sample`, so it carries the
    configuration -- including ``gauss_hierarchical``'s pilot covariance -- that
    actually steered the draws. States come from the *exact* diffused posterior,
    so the error is weighted by where the sampler really is at that level rather
    than over an arbitrary box.
    """
    module = problem["module"]
    device = problem["device"]
    y = problem["y"]
    n = problem["observations"]
    mask = module.CONDITION_MASK.to(device).reshape(1, 1, -1).repeat(n, states, 1)
    indices = torch.arange(n, device=device)

    rows = []
    for lam in lambdas:
        shared, local, _, _ = reference.sample_diffused(module, y, lam, states, seed)
        exact_shared, exact_local = reference.diffused_score(
            module, shared, local, y, lam
        )
        state = torch.zeros(n, states, module.NODES, dtype=torch.float32,
                            device=device)
        state[:, :, module.GLOBAL_INDEX] = torch.as_tensor(
            shared, dtype=torch.float32, device=device
        )
        state[:, :, module.LOCAL_INDEX] = torch.as_tensor(
            local.T.copy(), dtype=torch.float32, device=device
        )
        state[:, :, module.OBSERVED_INDEX] = torch.as_tensor(
            y, dtype=torch.float32, device=device
        ).reshape(-1, 1)

        t = sampler.sde.time_of_lambda(
            torch.tensor(float(lam))
        ).reshape(1, 1).to(device)
        # Curvature and covariance caches are keyed by refresh counter / time;
        # clear them so a probe never reuses state left by the sampling run.
        sampler._reset_jacobian_state()
        sampler._covariance_time_cache = {}
        composed = sampler._get_score(state, t, mask, indices).to(torch.float64)

        composed_shared = composed[0, :, module.GLOBAL_INDEX].cpu().numpy()
        composed_local = composed[:, :, module.LOCAL_INDEX].cpu().numpy().T

        def rms_relative(estimate, exact):
            return float(np.sqrt(((estimate - exact) ** 2).mean())
                         / max(np.sqrt((exact**2).mean()), 1e-30))

        rows.append({
            "lambda": float(lam),
            "shared_rel_rmse": rms_relative(composed_shared, exact_shared),
            "local_rel_rmse": rms_relative(composed_local, exact_local),
            "shared_bias": float((composed_shared - exact_shared).mean()),
            "composed_shared": composed_shared,
            "exact_shared": exact_shared,
            "state_shared": shared,
        })
    return rows


# ---------------------------------------------------------------------------
# Metrics on the draws
# ---------------------------------------------------------------------------

def prominent_maxima(grid, density, prominence=0.05):
    """Interior local maxima of a tabulated density, keeping only prominent ones.

    ``prominence`` is a fraction of the tallest peak: a shoulder carrying 5% of
    the peak height is noise in 3,000 draws, not a mode. The same rule grades the
    exact reference and the sampled draws, so the two counts are comparable.
    """
    interior = (density[1:-1] > density[:-2]) & (density[1:-1] >= density[2:])
    positions, heights = grid[1:-1][interior], density[1:-1][interior]
    if len(heights) == 0:
        return positions
    return np.sort(positions[heights >= prominence * heights.max()])


def kde_density(draws, grid, bandwidth):
    """Gaussian KDE of 1-D draws at an *absolute* bandwidth, on ``grid``.

    The bandwidth must be given, not left to Scott's rule. Scott's rule scales
    with the sample standard deviation, and on a two-moded posterior that number
    is dominated by the *gap between the modes* rather than by their width -- so
    the default smooths each mode out to roughly the width of the whole
    posterior, flattening the peaks by nearly half and filling in the valley
    that is the entire object of this experiment. Every KDE in this directory is
    therefore given a bandwidth tied to the exact posterior's own component
    width (see ``build_problem``), and the same one is used for the exact
    reference wherever the two are compared.
    """
    from scipy.stats import gaussian_kde

    draws = np.asarray(draws, dtype=np.float64).reshape(-1)
    factor = float(bandwidth) / max(float(draws.std(ddof=1)), 1e-12)
    return gaussian_kde(draws, bw_method=factor)(grid)


def recovered_modes(draws, grid, bandwidth, prominence=0.05):
    """The draws' KDE and its prominent interior maxima, at that bandwidth."""
    density = kde_density(draws, grid, bandwidth)
    return prominent_maxima(grid, density, prominence), density


def gap_window(problem):
    """The interval that counts as "between the modes", and its exact mass.

    The middle fifth of the gap between the two exact modes. A *log density
    ratio* at ``g = 0`` was the obvious first choice and is the wrong one: on
    the control benchmark every arm produces modes narrower than the exact ones,
    the estimated density at the midpoint falls into the kernel's Gaussian tail,
    and the ratio blows up to 50 nats against an exact 9 -- a number that says
    more about tail arithmetic than about the sampler. A mass in a fixed window
    is bounded, is estimated by a plain count, and is exactly the quantity that
    matters: how much probability leaked across the valley.
    """
    modes = np.asarray(problem["smoothed_modes"], dtype=np.float64)
    if len(modes) < 2:
        return None, float("nan")
    low, high = float(modes.min()), float(modes.max())
    centre, half = 0.5 * (low + high), 0.1 * (high - low)
    window = (centre - half, centre + half)
    exact = float(np.mean((problem["exact_shared"] > window[0])
                          & (problem["exact_shared"] < window[1])))
    return window, exact


def metrics(problem, shared, local, runtime, score_calls):  # noqa: C901
    """Every number the comparison table reports for one arm.

    Takes the draws rather than the sampler so it can be recomputed from a saved
    ``.npz`` without re-running anything (``--recompute-metrics``).
    """
    shared = np.asarray(shared, dtype=np.float64)
    local = np.asarray(local, dtype=np.float64)
    draws = len(shared)

    shared_w1 = reference.wasserstein_1d(
        shared, problem["global_grid"], problem["global_mass"]
    ) / problem["global_std"]
    local_w1 = float(np.mean([
        reference.wasserstein_1d(local[index], problem["local_grids"][index],
                                 problem["local_masses"][index])
        / problem["local_stds"][index]
        for index in range(problem["observations"])
    ]))

    grid = problem["global_grid"]
    bandwidth = problem["kde_bandwidth"]
    modes, density = recovered_modes(shared, grid, bandwidth)
    exact_modes = problem["smoothed_modes"]
    if len(modes) and len(exact_modes):
        mode_error = float(np.mean([
            np.abs(modes - value).min() for value in exact_modes
        ])) / problem["global_std"]
    else:
        mode_error = float("nan")

    window, exact_gap = gap_window(problem)
    if window is None:
        gap_draws, gap_draws_exact = float("nan"), float("nan")
    else:
        gap_draws = float(((shared > window[0]) & (shared < window[1])).sum())
        gap_draws_exact = float(exact_gap * draws)

    def mode_width(values):
        """Pooled within-mode standard deviation, draws assigned to the nearest
        exact mode.

        The overall standard deviation cannot say this: on a two-moded posterior
        it is dominated by the gap and by how the mass splits, so an arm with
        modes half the right width but a 50/50 split can report a *larger*
        spread than the truth. This isolates the width of a mode from where the
        mass sits, and it is what the control benchmark's arms get wrong.
        """
        if len(exact_modes) < 1:
            return float("nan")
        assignment = np.abs(np.asarray(values)[:, None]
                            - np.asarray(exact_modes)[None, :]).argmin(axis=1)
        variances = [float(np.var(values[assignment == index], ddof=1))
                     for index in range(len(exact_modes))
                     if int((assignment == index).sum()) > 1]
        return float(np.sqrt(np.mean(variances))) if variances else float("nan")

    # How tightly the shared coordinate and each local move together. In the
    # sign model this is the coupling that makes the signs align at all; in the
    # control, where pooling is weak, it is smaller but still the quantity that
    # says whether the joint structure survived. Unlike a sign-agreement count it
    # is meaningful on both benchmarks -- there the control's locals follow their
    # own observations rather than the shared lobe, and agreement is zero for the
    # exact posterior too.
    def coupling(shared_values, local_values):
        return float(np.mean([
            np.corrcoef(shared_values, local_values[index])[0, 1]
            for index in range(problem["observations"])
        ]))

    # Sign agreement: kept because it is the sharpest statement of the sign
    # model's ferromagnetic structure, even though it is degenerate (exactly
    # zero for the exact posterior as well) on the control.
    coherent = float(np.mean(np.all(
        np.sign(local) == np.sign(shared)[None, :], axis=0
    )))
    exact_coherent = float(np.mean(np.all(
        np.sign(problem["exact_local"].T) == np.sign(problem["exact_shared"])[None, :],
        axis=0,
    )))

    # The valley depth is kept in the table as a diagnostic, with the caveat in
    # gap_window: it is a log ratio and saturates.
    peak = float(density.max())
    centre = float(np.interp(0.0, grid, density))
    recovered_valley = float(np.log(max(peak, 1e-300)) - np.log(max(centre, 1e-300)))

    return {
        "shared_w1_over_sigma": shared_w1,
        "local_w1_over_sigma": local_w1,
        "modes_found": int(len(modes)),
        "modes_exact": int(len(exact_modes)),
        "kde_bandwidth": float(bandwidth),
        "mode_position_error_over_sigma": mode_error,
        "positive_mass": float((shared > 0).mean()),
        "positive_mass_exact": problem["positive_mass"],
        "gap_draws": gap_draws,
        "gap_draws_exact": gap_draws_exact,
        "posterior_draws": float(draws),
        "mode_width": mode_width(shared),
        "mode_width_exact": mode_width(problem["exact_shared"]),
        "shared_local_correlation": coupling(shared, local),
        "shared_local_correlation_exact": coupling(
            problem["exact_shared"], problem["exact_local"].T
        ),
        "valley_depth": recovered_valley,
        "valley_depth_exact": problem["smoothed_valley_depth"],
        "sign_coherence": coherent,
        "sign_coherence_exact": exact_coherent,
        "shared_mean": float(shared.mean()),
        "shared_std": float(shared.std(ddof=1)),
        "shared_mean_exact": problem["global_mean"],
        "shared_std_exact": problem["global_std"],
        "runtime_seconds": runtime,
        "score_network_calls": int(score_calls),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def result_path(model_name, score, method):
    return ARTIFACTS / model_name / score / f"{method}.npz"


SUMMARY_KEYS = (
    "shared_w1_over_sigma", "local_w1_over_sigma",
    "modes_found", "modes_exact", "mode_position_error_over_sigma",
    "positive_mass", "positive_mass_exact",
    "gap_draws", "gap_draws_exact", "posterior_draws",
    "mode_width", "mode_width_exact",
    "shared_local_correlation", "shared_local_correlation_exact",
    "valley_depth", "valley_depth_exact",
    "sign_coherence", "sign_coherence_exact",
    "shared_mean", "shared_std", "shared_mean_exact", "shared_std_exact",
    "runtime_seconds", "score_network_calls",
)


def summary_row(model_name, score, method, payload):
    """One row of ``method_metrics.csv``, from an arm's payload."""
    return {
        "model": model_name, "score": score, "method": method,
        "label": METHODS[method]["label"],
        **{key: float(payload[key]) for key in SUMMARY_KEYS},
        **{f"score_rel_rmse_lam_{lam:g}": float(value) for lam, value
           in zip(payload["score_lambdas"], payload["score_shared_rel_rmse"])},
    }


def recompute_metrics(model_name, score, method, problem):
    """Recompute an arm's metrics from its saved draws, without re-sampling.

    The draws are the expensive part and they are already on disk; changing how
    a metric is *defined* should not cost a GPU run, and should not tempt anyone
    to leave a stale definition in place because rerunning is inconvenient.
    """
    path = result_path(model_name, score, method)
    if not path.exists():
        return None
    payload = dict(np.load(path, allow_pickle=True))
    row = metrics(
        problem, payload["shared_draws"], payload["local_draws"],
        float(payload["runtime_seconds"]), int(payload["score_network_calls"]),
    )
    payload.update(row)
    payload["smoothed_density"] = problem["smoothed_density"]
    payload["kde_bandwidth"] = problem["kde_bandwidth"]
    np.savez_compressed(path, **payload)
    print(f"  recomputed {path.relative_to(ARTIFACTS)}")
    return payload


def run_one(model, problem, model_name, score, method, arguments):
    path = result_path(model_name, score, method)
    if path.exists() and not arguments.force:
        print(f"  reusing {path.relative_to(ARTIFACTS)}")
        return dict(np.load(path, allow_pickle=True))

    module = problem["module"]
    print(f"  running {method} [{score}] ...", flush=True)
    samples, sampler, runtime = draw(
        model, problem, method, arguments.num_samples, arguments.timesteps,
        INFERENCE_SEED, verbose=arguments.verbose,
    )
    shared_draws = samples[0, :, module.GLOBAL_INDEX].to(torch.float64).numpy()
    local_draws = samples[:, :, module.LOCAL_INDEX].to(torch.float64).numpy()
    row = metrics(problem, shared_draws, local_draws, runtime,
                  getattr(sampler, "score_network_calls", 0))
    probe = probe_composed_score(
        sampler, problem, SCORE_LAMBDAS, SCORE_STATES, seed=4242
    )
    payload = {
        "model": model_name, "score": score, "method": method,
        "y": problem["y"], "global_truth": problem["global_truth"],
        "local_truth": problem["local_truth"],
        "shared_draws": samples[0, :, module.GLOBAL_INDEX].numpy(),
        "local_draws": samples[:, :, module.LOCAL_INDEX].numpy(),
        "global_grid": problem["global_grid"],
        "global_density": problem["global_density"],
        "smoothed_density": problem["smoothed_density"],
        "kde_bandwidth": problem["kde_bandwidth"],
        "local_grids": problem["local_grids"],
        "local_densities": problem["local_densities"],
        "score_lambdas": np.array([entry["lambda"] for entry in probe]),
        "score_shared_rel_rmse": np.array(
            [entry["shared_rel_rmse"] for entry in probe]
        ),
        "score_local_rel_rmse": np.array(
            [entry["local_rel_rmse"] for entry in probe]
        ),
        "score_shared_bias": np.array([entry["shared_bias"] for entry in probe]),
        "probe_state_shared": np.stack([entry["state_shared"] for entry in probe]),
        "probe_composed_shared": np.stack(
            [entry["composed_shared"] for entry in probe]
        ),
        "probe_exact_shared": np.stack([entry["exact_shared"] for entry in probe]),
        **{key: value for key, value in row.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"    W1(g)/sigma {row['shared_w1_over_sigma']:.4f}   "
          f"modes {row['modes_found']}/{row['modes_exact']}   "
          f"P(g>0) {row['positive_mass']:.3f}   "
          f"{runtime:.1f}s")
    return payload


def write_table(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        choices=list(MODELS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                        choices=list(METHODS),
                        help="Default sweep is corrector-free (pure churn) "
                             "plus the Langevin+F-NPSE baseline; the "
                             "*_correctors arms reproduce RESULTS.md.")
    parser.add_argument("--scores", nargs="+", default=list(SCORES),
                        choices=list(SCORES))
    parser.add_argument("--observations", type=int, default=OBSERVATIONS)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--train-samples", type=int, default=recipe.TRAIN_SAMPLES)
    parser.add_argument("--time-sampling", default=recipe.DEFAULT_TIME_SAMPLING,
                        choices=("mixture", "log_sigma", "uniform"),
                        help="diffusion-time recipe; the tables in RESULTS.md "
                             "were produced with 'mixture'")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true",
                        help="ignore cached .npz results and rerun")
    parser.add_argument("--recompute-metrics", action="store_true",
                        help="recompute metrics from saved draws; run nothing")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()

    if arguments.device is None:
        try:
            from autocvd import autocvd

            autocvd(num_gpus=1, interval=1)
        except ImportError:
            pass
        arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {arguments.device}")

    from compass.SDE import VESDE

    summary = []
    fidelity_rows = []
    for model_name in arguments.models:
        module = MODELS[model_name]
        print(f"\n########  {module.TITLE}  ########")
        problem = build_problem(
            module, arguments.observations, DATA_SEEDS[model_name],
            arguments.device,
        )
        print(f"  observations y = "
              f"{np.array2string(problem['y'], precision=3)}")
        print(f"  true g = {problem['global_truth']:.3f}; exact modes "
              f"{np.array2string(problem['global_modes'], precision=3)}; "
              f"valley depth {problem['valley_depth']:.2f} nats; "
              f"aligned mass {problem['aligned_mass']:.4f}")

        models = {}
        for score in arguments.scores:
            if score not in SCORE_CONFIG:
                continue
            config = SCORE_CONFIG[score]
            checkpoint = (recipe.model_directory(ROOT, module, config)
                          / "Model_checkpoint.pt")
            if arguments.recompute_metrics and not checkpoint.exists():
                # Nothing to grade and nothing to load; a recompute pass must
                # never start a training run as a side effect.
                continue
            network, seconds = recipe.train_or_load(
                ROOT, module, config, device=arguments.device,
                force=arguments.retrain, verbose=arguments.verbose,
                train_samples=arguments.train_samples,
                time_sampling=arguments.time_sampling,
            )
            fidelity = recipe.score_fidelity(
                network, module, problem["y"], FIDELITY_LAMBDAS,
                device=arguments.device,
            )
            print(f"  {score} ({config}) row score, relative RMS vs exact: "
                  f"mean {fidelity['mean']:.4f}  "
                  + "  ".join(f"lam={key:g}:{value:.3f}"
                              for key, value in fidelity.items()
                              if key != "mean"))
            fidelity_rows.append({
                "model": model_name, "score": score, "config": config,
                "time_sampling": arguments.time_sampling,
                "parameters": recipe.parameter_count(module, config),
                "train_samples": arguments.train_samples,
                "train_seconds": seconds,
                **{f"rel_rmse_lam_{key}": value
                   for key, value in fidelity.items()},
            })
            models[score] = network
        if "analytic" in arguments.scores:
            models["analytic"] = row_scores.AnalyticModel(
                VESDE(sigma=recipe.SDE_KWARGS["sigma"]), module,
                device=arguments.device, seed=INFERENCE_SEED,
            )

        for score in arguments.scores:
            for method in arguments.methods:
                if arguments.recompute_metrics:
                    payload = recompute_metrics(model_name, score, method,
                                                problem)
                else:
                    payload = run_one(models[score], problem, model_name, score,
                                      method, arguments)
                if payload is not None:
                    summary.append(summary_row(model_name, score, method,
                                               payload))

        (ARTIFACTS / model_name).mkdir(parents=True, exist_ok=True)
        with (ARTIFACTS / model_name / "problem.json").open("w") as handle:
            json.dump({
                "title": module.TITLE,
                "observations": arguments.observations,
                "data_seed": DATA_SEEDS[model_name],
                "y": problem["y"].tolist(),
                "global_truth": problem["global_truth"],
                "local_truth": problem["local_truth"].tolist(),
                "discrete_truth": problem["discrete_truth"].tolist(),
                "global_modes": problem["global_modes"].tolist(),
                "global_mean": problem["global_mean"],
                "global_std": problem["global_std"],
                "aligned_mass": problem["aligned_mass"],
                "valley_depth": problem["valley_depth"],
                "kde_bandwidth": problem["kde_bandwidth"],
                "positive_mass": problem["positive_mass"],
            }, handle, indent=2)

    write_table(summary, ARTIFACTS / "method_metrics.csv")
    write_table(fidelity_rows, ARTIFACTS / "network_fidelity.csv")


if __name__ == "__main__":
    main()
