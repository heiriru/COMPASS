#!/usr/bin/env python3
"""Can a *corrector-free* DPM2 reach the exact shared posterior?

The target is ``artifacts/01_gauss_jacobian_analytic_correctors2.png``:
W1 = 0.057 posterior sigma on the shared marginal, width ratio 1.06, with the
exact single-observation score and ``correction="gauss_jacobian"``. Removing the
two Langevin correctors from that arm costs a factor of *forty-five*
(``gauss_jacobian_analytic_predictor_only``: W1 = 2.598, width 1.53).

What the existing artifacts already establish
---------------------------------------------
``artifacts/03_rule_versus_sampler.png`` panel 1 grades the composed field alone
against the exact quadrature: ``gauss_jacobian`` is at 0.10 relative RMS at
lambda = 0.05, peaks at 0.65 near lambda = 0.5, and falls back to 0.13 at
lambda = 2. So the field is good at both ends of the schedule and wrong in the
middle.

``artifacts/initialization_experiments.csv`` splits the predictor-only failure:

    production start   W1 2.598      (every latent at 0 + lambda_max noise)
    A0 exact start     W1 0.387      (draws from the exact diffused joint)
    VP SDE             W1 0.511      (terminal law genuinely forgets the data)

So ~85% of the 2.6 sigma is the initial law, and the remaining ~0.4 sigma is the
mid-lambda rule error, transported to t = 0 by a deterministic map that has no
mechanism to shed it. Neither term is timestep error: 50 and 100 steps agree to
0.3%.

Why a corrector fixes it, and what else could
---------------------------------------------
A Langevin corrector at noise level lambda relaxes the ensemble onto the
*composed* density at that level. Because the composed field is nearly exact for
small lambda, the late correctors re-equilibrate onto very nearly the right
density and erase both the initial-law error and the accumulated mid-lambda
transport error. The probability-flow ODE cannot: it returns the pushforward of
its initial law, error and all.

The mechanism that matters is therefore *stochastic relaxation while the field is
accurate*, not "corrector steps" as such. The reverse-time SDE has exactly that
mechanism built into its predictor, and Karras et al. (2022, Algorithm 2) give
the standard way to run it with a second-order deterministic solver: at each
level, first take a short step **up** in noise (inject noise), then run the
deterministic DPM2 step from the raised level down to the next one. Combined,
noise-up + a longer ODE-down *is* the reverse SDE. It costs **no extra score
evaluations** -- 594 calls, exactly the predictor-only budget, versus 1,236 for
the two-corrector arm.

Parameterization, and why ``eta`` is the honest knob
----------------------------------------------------
In noise-scale coordinates the VE probability-flow ODE is
``dx = -sigma s(x, sigma) dsigma`` and the reverse SDE is
``dx = -2 sigma s dsigma + sqrt(2 sigma |dsigma|) dw``. Injecting variance
``sigma_hat^2 - sigma^2 ~= 2 sigma delta`` and then integrating the ODE from
``sigma + delta`` down to ``sigma_next`` reproduces the SDE exactly when
``delta = sigma - sigma_next``. So define

    sigma_hat = sigma_i + eta * (sigma_i - sigma_{i+1})

with ``eta = 0`` the untouched probability-flow ODE (the current predictor-only
arm), ``eta = 1`` the exact reverse SDE, and ``eta > 1`` the generalized family
of Karras eq. (6) -- a stronger Langevin term, which leaves the marginals
invariant for an exact score and only trades discretization error for mixing.

The schedule here is geometric with ratio ``r = 0.9526`` over 100 steps, so
``eta = 1`` injects ``2 sigma^2 (1 - r) = 0.095 sigma^2`` of variance per level
while the two-corrector arm's Langevin steps inject ``2 * 2 * snr * sigma^2 =
0.8 sigma^2``. Matching the corrector arm's *noise budget* at zero extra score
cost therefore needs ``eta ~ 8``; Karras' stability cap ``gamma <= sqrt(2) - 1``
allows ``eta <= 8.7`` on this schedule.

Nothing in ``compass``, ``compare.py`` or ``analytic_compare.py`` is modified.
``MultiObsSampler._dpm_sampler`` is swapped for the churned variant for the
duration of one run and restored afterwards, exactly as
``initialization_experiments.py`` swaps ``_initial_sample``.

Usage:
    python stochastic_predictor_experiments.py --device cuda
    python stochastic_predictor_experiments.py --variants C_eta4 --device cuda
    python stochastic_predictor_experiments.py --replot
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
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compass.MultiObsSampler import MultiObsSampler  # noqa: E402

import analytic_compare as analytic  # noqa: E402
import compare  # noqa: E402
import hierarchy  # noqa: E402
import initialization_experiments as initialization  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

# Corrector-free throughout: the whole point is what the predictor alone can do.
SAMPLE_KWARGS = {
    "method": "dpm", "order": 2, "correction": "gauss_jacobian",
    "corrector_steps": 0, "final_corrector_steps": 0,
    "terminal_corrector_steps": 0,
}

# Karras et al. (2022) cap the per-step noise-up ratio for stability.
GAMMA_CAP = math.sqrt(2.0) - 1.0

NAVY, BLUE, TEAL, CORAL, GOLD = (
    compare.NAVY, compare.BLUE, compare.TEAL, compare.CORAL, compare.GOLD
)
MUTED = "#8A8F98"


# ---------------------------------------------------------------------------
# The churned DPM2 predictor
# ---------------------------------------------------------------------------

def churned_dpm_sampler(eta, s_noise=1.0, sigma_min=0.0, sigma_max=float("inf")):
    """Build a ``_dpm_sampler`` replacement: noise-up by ``eta``, then DPM2 down.

    ``eta`` scales the noise-up step in units of the schedule's own step length,
    so ``eta = 0`` is the stock probability-flow predictor and ``eta = 1`` is the
    exact reverse SDE (see the module docstring). ``sigma_min`` / ``sigma_max``
    restrict churn to a band of the schedule, Karras' ``S_tmin`` / ``S_tmax``.

    Correctors are refused rather than ignored: this sampler exists to answer a
    question about a predictor, and silently dropping a corrector request would
    make its answers unreadable.
    """
    def sampler(self, data, condition_mask, idx, order=2, snr=0.1,
                corrector_steps_interval=5, corrector_steps=5,
                final_corrector_steps=3, terminal_corrector_steps=0):
        if corrector_steps or final_corrector_steps or terminal_corrector_steps:
            raise ValueError(
                "churned_dpm_sampler is a corrector-free predictor; got "
                f"corrector_steps={corrector_steps}, "
                f"final_corrector_steps={final_corrector_steps}, "
                f"terminal_corrector_steps={terminal_corrector_steps}."
            )
        step = {1: self._dpm_solver_1_step, 2: self._dpm_solver_2_step,
                3: self._dpm_solver_3_step}[order]
        latent = 1 - condition_mask
        # The schedule's largest noise level: churn must never step above it,
        # because t > 1 is outside the SDE's (and the network's) domain.
        ceiling = float(self.sde.lambda_t(torch.ones(1, device=data.device)))

        if self.save_trajectory:
            self.data_t = torch.zeros(data.shape[0], self.timesteps,
                                      data.shape[1], data.shape[2])
            self.data_t[:, 0, :, :] = data

        for i in range(self.timesteps - 1):
            t_now = self.timesteps_list[i].reshape(-1, 1)
            t_next = self.timesteps_list[i + 1].reshape(-1, 1)
            sigma_now = float(self.sde.lambda_t(t_now).reshape(-1)[0])
            sigma_next = float(self.sde.lambda_t(t_next).reshape(-1)[0])

            t_start = t_now
            if eta > 0.0 and sigma_min <= sigma_now <= sigma_max:
                gamma = min(eta * (sigma_now - sigma_next) / sigma_now, GAMMA_CAP)
                sigma_hat = min(sigma_now * (1.0 + gamma), ceiling)
                spread = math.sqrt(max(sigma_hat**2 - sigma_now**2, 0.0))
                if spread > 0.0:
                    # _shared_noise keeps the hierarchy coordinates identical
                    # across observation rows, as everywhere else in the sampler.
                    data = data + s_noise * spread * self._shared_noise(data) * latent
                    t_start = self.sde.time_of_lambda(
                        torch.as_tensor(sigma_hat, dtype=t_now.dtype,
                                        device=t_now.device)
                    ).reshape(1, 1)

            data = step(data, t_start, t_next, condition_mask, idx)

            if self.save_trajectory:
                self.data_t[:, i + 1] = data

        return data.detach()

    return sampler


@contextlib.contextmanager
def patched_dpm_sampler(builder):
    """Swap ``MultiObsSampler._dpm_sampler`` for one run, then restore it."""
    original = MultiObsSampler._dpm_sampler
    MultiObsSampler._dpm_sampler = builder
    try:
        yield
    finally:
        MultiObsSampler._dpm_sampler = original


# ---------------------------------------------------------------------------
# The experiment table
# ---------------------------------------------------------------------------

def variants(x, seed):
    """Every arm is predictor-only; they differ in churn strength and start."""
    single_observation = initialization.single_observation_start(x)
    return {
        "P0_baseline": {
            "label": "eta 0: probability-flow ODE (stock predictor-only)",
            "note": "no churn, production start -- the arm being repaired",
            "eta": 0.0, "start": None, "colour": MUTED,
        },
        "C_eta1": {
            "label": "eta 1: exact reverse SDE",
            "note": "noise-up = one schedule step; production start",
            "eta": 1.0, "start": None, "colour": BLUE,
        },
        "C_eta2": {
            "label": "eta 2: reverse SDE, 2x Langevin term",
            "note": "production start",
            "eta": 2.0, "start": None, "colour": TEAL,
        },
        "C_eta4": {
            "label": "eta 4: reverse SDE, 4x Langevin term",
            "note": "production start",
            "eta": 4.0, "start": None, "colour": CORAL,
        },
        "C_eta8": {
            "label": "eta 8: matches the 2-corrector noise budget",
            "note": "production start; gamma still under the sqrt(2)-1 cap",
            "eta": 8.0, "start": None, "colour": GOLD,
        },
        # If churn really does erase the initial law, the A1b start -- the best
        # initializer found in initialization_experiments.py -- must stop
        # mattering. That is the falsifiable half of the claim.
        "C_eta4_A1b": {
            "label": "eta 4 + single-observation start",
            "note": "churn plus the best initializer; tests whether both are needed",
            "eta": 4.0, "start": single_observation, "colour": "#7F4FC9",
        },
        # The same churn on the *other* composition rule. This is the arm the
        # curl diagnostic says to be suspicious of: exact_curl.csv puts
        # gauss_hierarchical's relative Jacobian antisymmetry at ~0.92 at every
        # lambda, against ~0.09 for gauss_jacobian. A Langevin-type term has no
        # guaranteed invariant measure on a field with that much curl, so churn
        # may fail to help -- or help and still land on the wrong density.
        "H_eta0": {
            "label": "gauss_hierarchical, eta 0 (probability-flow ODE)",
            "note": "4096-draw pilot covariance; no correctors, no churn",
            "eta": 0.0, "start": None, "colour": "#B0453A",
            "sample_kwargs": {
                "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
                "precision_est_samples": compare.PRECISION_EST_SAMPLES,
                "precision_est_timesteps": compare.TIMESTEPS,
                "corrector_steps": 0, "final_corrector_steps": 0,
                "terminal_corrector_steps": 0,
            },
        },
        "H_eta2": {
            "label": "gauss_hierarchical, eta 2",
            "note": "4096-draw pilot covariance; the eta the learned score "
                    "prefers for gauss_jacobian",
            "eta": 2.0, "start": None, "colour": "#E07A5F",
            "sample_kwargs": {
                "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
                "precision_est_samples": compare.PRECISION_EST_SAMPLES,
                "precision_est_timesteps": compare.TIMESTEPS,
                "corrector_steps": 0, "final_corrector_steps": 0,
                "terminal_corrector_steps": 0,
            },
        },
        "H_eta4": {
            "label": "gauss_hierarchical, eta 4",
            "note": "4096-draw pilot covariance; churn on a curl-carrying field",
            "eta": 4.0, "start": None, "colour": CORAL,
            "sample_kwargs": {
                "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
                "precision_est_samples": compare.PRECISION_EST_SAMPLES,
                "precision_est_timesteps": compare.TIMESTEPS,
                "corrector_steps": 0, "final_corrector_steps": 0,
                "terminal_corrector_steps": 0,
            },
        },
        # Corrector references, run in the same session on the same device.
        # Cross-session wall clocks are not comparable -- methods.csv's numbers
        # come from a different compare.py run under different GPU load, and
        # differ from this harness by 3x on identical work -- so the corrector
        # arms have to be re-timed here to be quoted against churn at all.
        # These carry correctors, so run_variant leaves _dpm_sampler unpatched.
        "R_correctors2_J": {
            "label": "gauss_jacobian + 2 correctors (reference, no churn)",
            "note": "the published corrector arm, re-timed in this session",
            "eta": 0.0, "start": None, "colour": "#1F6F8B",
            "sample_kwargs": {
                "method": "dpm", "order": 2, "correction": "gauss_jacobian",
                "corrector_steps_interval": 1, "corrector_steps": 2,
                "final_corrector_steps": 3, "snr": 0.2,
            },
        },
        "R_correctors2_H": {
            "label": "gauss_hierarchical + 2 correctors (reference, no churn)",
            "note": "the published corrector arm, re-timed in this session",
            "eta": 0.0, "start": None, "colour": "#9B5DE5",
            "sample_kwargs": {
                "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
                "precision_est_samples": compare.PRECISION_EST_SAMPLES,
                "precision_est_timesteps": compare.TIMESTEPS,
                "corrector_steps_interval": 1, "corrector_steps": 2,
                "final_corrector_steps": 3, "snr": 0.2,
            },
        },
        # Annealed Langevin never enters _dpm_sampler, so the patch is inert.
        "L_fnpe": {
            "label": "Langevin + F-NPSE (timing reference)",
            "note": "corrector-only sampler; bridging densities, not the "
                    "diffusion marginals",
            "eta": 0.0, "start": None, "colour": GOLD,
            "sample_kwargs": {
                "method": "langevin", "correction": "fnpe",
                "corrector_steps": 10, "snr": 0.2,
            },
        },
    }


def run_variant(name, entry, problem, model, arguments):
    kwargs = entry.get("sample_kwargs", SAMPLE_KWARGS)
    # Reference arms that keep their correctors run on the stock sampler: the
    # churned predictor refuses a corrector request by design, and at eta = 0 it
    # would in any case be the stock predictor with extra bookkeeping.
    corrector_free = not any(
        kwargs.get(key) for key in
        ("corrector_steps", "final_corrector_steps", "terminal_corrector_steps")
    )
    stack = contextlib.ExitStack()
    with stack:
        if corrector_free:
            stack.enter_context(patched_dpm_sampler(churned_dpm_sampler(
                entry["eta"], s_noise=arguments.s_noise,
                sigma_min=arguments.churn_sigma_min,
                sigma_max=arguments.churn_sigma_max,
            )))
        elif entry["eta"]:
            raise ValueError(
                f"variant {name!r} asks for churn (eta={entry['eta']}) *and* "
                "corrector steps; this script exists to separate them."
            )
        if entry["start"] is not None:
            stack.enter_context(
                initialization.patched_initial_sample(entry["start"])
            )
        samples, sampler, runtime = compare.draw(
            model, problem, dict(entry.get("sample_kwargs", SAMPLE_KWARGS)),
            arguments.num_samples,
            arguments.timesteps, arguments.seed, arguments.denoise_clamp,
        )
    result = compare.kde_map_pipeline(model, problem, samples, arguments)
    metrics = compare.evaluate(problem, result, sampler, runtime, [])
    metrics["method"] = name
    return result, metrics


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

TARGET_W1 = 0.0571  # gauss_jacobian_analytic_correctors2, methods_analytic.csv


def plot_summary(problem, results, metrics, table, output):
    order = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fig.suptitle(
        "Corrector-free DPM2: churn turns the probability-flow ODE into the "
        "reverse SDE\n(exact single-observation score, gauss_jacobian, 100 "
        "steps, 3,000 draws, no corrector steps and no extra score calls)",
        fontsize=13.5, fontweight="bold", color=NAVY,
    )

    axes[0].plot(problem["global_grid"], problem["global_density"], color="black",
                 lw=2.2, label="exact", zorder=6)
    for name in order:
        axes[0].hist(results[name]["kept_shared"], bins=70, density=True,
                     histtype="step", lw=1.7, color=table[name]["colour"],
                     label=table[name]["label"])
    axes[0].axvline(problem["x"].min(), color=NAVY, ls="-.", lw=1.3)
    axes[0].set_xlim(problem["global_mean"] - 8 * problem["global_std"],
                     problem["global_mean"] + 3 * problem["global_std"])
    axes[0].set(xlabel="global parameter g", ylabel="density",
                title="Shared posterior")
    axes[0].legend(fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    positions = np.arange(len(order))
    values = [metrics[name]["shared_w1_over_sigma"] for name in order]
    axes[1].barh(positions, values, color=[table[name]["colour"] for name in order],
                 edgecolor="white", linewidth=1.2)
    for position, value in zip(positions, values):
        axes[1].text(value * 1.08, position, f"{value:.3f}", va="center",
                     fontsize=8.5, color=NAVY)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels([table[name]["label"] for name in order], fontsize=7.8)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.02, 12.0)
    axes[1].axvline(TARGET_W1, color=NAVY, ls=":", lw=1.4)
    axes[1].text(TARGET_W1, len(order) - 0.4, " 2-corrector target", fontsize=7.5,
                 color=NAVY, va="center")
    axes[1].set(xlabel="W1 of the shared marginal / posterior σ", title="Accuracy")
    axes[1].grid(axis="y", visible=False)
    axes[1].invert_yaxis()

    width = [metrics[name]["shared_width_ratio"] for name in order]
    axes[2].barh(positions, width, color=[table[name]["colour"] for name in order],
                 edgecolor="white", linewidth=1.2)
    for position, value in zip(positions, width):
        axes[2].text(value + 0.03, position, f"{value:.2f}", va="center",
                     fontsize=8.5, color=NAVY)
    axes[2].axvline(1.0, color=NAVY, ls=":", lw=1.4)
    axes[2].set_yticks(positions)
    axes[2].set_yticklabels([])
    axes[2].set(xlabel="width ratio (1 is correct)", title="Dispersion")
    axes[2].grid(axis="y", visible=False)
    axes[2].invert_yaxis()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


FIELDS = ["method", "label", "eta", "seconds", "calls", "shared_w1_over_sigma",
          "shared_mean_error_sigma", "shared_width_ratio", "local_w1_over_sigma",
          "marginal_score_rel_rmse", "global_map", "global_map_error_sigma",
          "local_map_error_vs_posterior_mean_sigma"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=compare.OBSERVATIONS)
    parser.add_argument("--num-samples", type=int, default=compare.NUM_SAMPLES)
    parser.add_argument("--timesteps", type=int, default=compare.TIMESTEPS)
    parser.add_argument("--quadrature-nodes", type=int, default=385)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--excursion-sigma", type=float, default=15.0)
    parser.add_argument("--kde-bandwidth", default=None)
    parser.add_argument("--map-timesteps", type=int, default=200)
    parser.add_argument("--map-iterations", type=int, default=3)
    parser.add_argument("--map-eps", type=float, default=1e-3)
    parser.add_argument("--s-noise", type=float, default=1.0,
                        help="Karras S_noise: scale on the injected noise only.")
    parser.add_argument("--churn-sigma-min", type=float, default=0.0)
    parser.add_argument("--churn-sigma-max", type=float, default=float("inf"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score", default="exact", choices=["exact", "learned"])
    parser.add_argument("--config", default="h16d2")
    parser.add_argument("--train-samples", type=int, default=200_000)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--replot", action="store_true")
    arguments = parser.parse_args()

    from compass.SDE import VESDE

    compare.configure_style()
    _, _, x = hierarchy.observations(arguments.observations, arguments.seed)
    table = variants(x, arguments.seed)
    selected = arguments.variants or list(table)

    if arguments.score == "exact":
        model = analytic.AnalyticModel(
            VESDE(sigma=compare.recipe.SDE_KWARGS["sigma"]), device=arguments.device,
            nodes=arguments.quadrature_nodes, seed=arguments.seed,
        )
        compare.recipe.train_or_load = lambda *a, **k: (model, 0.0)
        problem = compare.build_problem(
            None, "analytic", 0, arguments.observations, arguments.seed,
            arguments.device,
        )
    else:
        problem = compare.build_problem(
            compare.recipe.model_directory(
                ARTIFACTS, arguments.config, arguments.train_samples
            ),
            arguments.config, arguments.train_samples, arguments.observations,
            arguments.seed, arguments.device,
        )
        model = problem["model"]
    tag = "" if arguments.score == "exact" else "_learned"
    print(f"true g = {problem['global_truth']:.5f}   exact posterior mean "
          f"{problem['global_mean']:.5f}, sd {problem['global_std']:.5f}, "
          f"wall {problem['x'].min():.5f}")

    results, metrics, rows = {}, {}, []
    for name in selected:
        entry = table[name]
        folder = arguments.artifacts / f"stochastic_predictor_{name}{tag}"
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / f"{name}.npz"

        if arguments.replot:
            with np.load(archive, allow_pickle=True) as stored:
                loaded = {key: stored[key] for key in stored.files}
            loaded["global_map"] = float(loaded["global_map"])
            loaded["excluded"] = int(loaded["excluded"])
            results[name] = loaded
            metrics[name] = json.loads(str(loaded["metrics_json"]))
        else:
            print(f"\n=== {entry['label']} ===")
            result, metric = run_variant(name, entry, problem, model, arguments)
            print(f"  {metric['sample_seconds']:.1f}s  "
                  f"{metric['sampler_network_calls']} calls  "
                  f"W1 {metric['shared_w1_over_sigma']:.4f}  "
                  f"width {metric['shared_width_ratio']:.3f}  "
                  f"local W1 {metric['local_w1_over_sigma']:.4f}")
            scalar = {key: value for key, value in metric.items()
                      if np.isscalar(value) or isinstance(value, str)}
            np.savez_compressed(
                archive,
                x=problem["x"], global_grid=problem["global_grid"],
                global_density=problem["global_density"],
                global_truth=problem["global_truth"],
                exact_local_mean=problem["local_mean"],
                exact_local_std=problem["local_std"],
                metrics_json=json.dumps(scalar, default=float),
                score_rows_json="[]",
                score_grid=metric["score_grid"],
                implied_score=metric["implied_score"],
                exact_score=metric["exact_score"],
                **result,
            )
            results[name], metrics[name] = result, metric

        rows.append({
            "method": name, "label": entry["label"], "eta": entry["eta"],
            "seconds": metrics[name]["sample_seconds"],
            "calls": metrics[name]["sampler_network_calls"],
            **{key: metrics[name][key] for key in FIELDS[5:]},
        })

        compare.METHODS[name] = {
            "label": entry["label"], "colour": entry["colour"],
            "note": entry["note"],
            "score_note": "exact single-observation score, no correctors",
            "sample_kwargs": dict(entry.get("sample_kwargs", SAMPLE_KWARGS)),
        }
        compare.plot_method(problem, name, results[name], metrics[name],
                            folder / f"01_{name}.png")

    # Merge by variant rather than overwrite, the same contract compare.py uses.
    path = arguments.artifacts / f"stochastic_predictor_experiments{tag}.csv"
    merged = {}
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                merged[row["method"]] = row
    for row in rows:
        merged[row["method"]] = row
    if merged:
        order = list(table)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for key in sorted(merged, key=lambda item: (order + [item]).index(item)):
                writer.writerow({f: merged[key].get(f, "") for f in FIELDS})
        print(f"Wrote {path}")

    plot_summary(problem, results, metrics, table,
                 arguments.artifacts / f"06_stochastic_predictor{tag}.png")


if __name__ == "__main__":
    main()
