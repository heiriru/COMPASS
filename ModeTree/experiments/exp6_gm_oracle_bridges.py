#!/usr/bin/env python
"""The oracle bridge experiment on the Gaussian-mixture benchmark.

No network is trained or loaded.  Every single-observation score, every score
Jacobian, the prior score and Hessian, and the tall posterior and its diffusion
are the exact closed forms of
``Gaussian_mixtures_compositional_score_modelling`` (see ``gm_oracle``).  The
question the experiment answers is therefore not "how good is the network" but

    with perfect single-observation information, which inference construction
    recovers the correct multimodal GM posterior?

Stages (``--stage``), each writing into ``artifacts/gm_oracle/``:

``reference``   the exact clean targets and the Monte-Carlo floor for W1/sd.
``control``     TRUE-SCORE-DPM2 and its step/solver ladder (section 2).
``geffner``     properness of the literal F-NPSE bridge and the Langevin
                convergence ladder (sections 3-4).
``arruda``      the damped bridge, its ``d_1`` ablation and the adaptive-solver
                tolerance ladder (sections 5-7).
``linhart``     oracle GAUSS and oracle JAC with the DPM2 step ladder
                (sections 8-9).
``naive``       the deliberate negative control (section 10).
``geffner_dpm2``the F-NPSE bridge fed to DPM2 (section 11).
``bridges``     intermediate bridge densities at matched noise levels
                (section 13).
``table``       assemble the required output table (section 16).

Run:  python ModeTree/experiments/exp6_gm_oracle_bridges.py --stage all
"""

import sys
from pathlib import Path

ROOT = Path("/export/home/rheinric/COMPASS")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Gaussian_mixtures_compositional_score_modelling"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402
import math      # noqa: E402
import time      # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

import datasets   # noqa: E402
import model as M  # noqa: E402

from gm_oracle import GMOracle                              # noqa: E402
from gm_bridges import (ComposedField, adaptive_heun, annealed_langevin,  # noqa: E402
                        dpm2, euler_maruyama)
from gm_metrics import monte_carlo_floor, summarise          # noqa: E402

torch.set_default_dtype(torch.float64)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "gm_oracle"
N_LADDER = (8, 32, 128)
REGIME = "asymmetric"
BRANCH_MASS = 0.75
SIGMA = 15.0
COUNT = 2000
# The benchmark's own terminal time, so the oracle arms stop where a trained
# network would have to.  `--lam-min` overrides it; the control stage runs the
# override explicitly to separate solver error from terminal-noise error.
EPS = 1e-3


# ``--mu0`` replaces the benchmark's mu_0 = 2.5.  It exists because the branch
# separation of the standard setting is 2 mu_0 / (tau_g^2 sqrt(1/tau_g^2 + n/v))
# within-branch sds, which falls as 1/sqrt(n): at n = 128 the exact posterior is
# only 2.5 sds wide and its dip ratio is 0.999, i.e. it is bimodal in name only.
# A wider mu_0 restores a genuine valley at n = 128 without changing anything
# else about the model, so the multimodal question can still be asked there.
SETTING = {"mixture": M.DEFAULT, "tag": ""}


def oracle_for(n, seed=0):
    mixture = SETTING["mixture"]
    rng = np.random.default_rng(seed)
    dataset = datasets.make_dataset(REGIME, n, rng, mixture,
                                    branch_mass=BRANCH_MASS)
    return GMOracle(mixture, dataset["x"], SIGMA), dataset


def lam_bounds(oracle, eps=EPS):
    return (float(oracle.schedule.lam(1.0)), float(oracle.schedule.lam(eps)))


def store(name, payload):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{name}{SETTING['tag']}.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2, default=float))
    print(f"wrote {path}")


def record(oracle, samples, stats, extra=None):
    row = summarise(samples[:, 0], oracle.tall)
    row.update({"seconds": stats.get("seconds"), "nfe": stats.get("nfe"),
                "field_calls": stats.get("field_calls"),
                "row_score_evaluations": oracle.row_score_evaluations,
                "row_jacobian_evaluations": oracle.row_jacobian_evaluations})
    # The sampler reports divergence it detected while running; the metrics
    # report samples that ended up meaningless.  Either one disqualifies the arm.
    row["diverged"] = bool(row.get("diverged")) or bool(stats.get("diverged"))
    for key in ("accepted_steps", "rejected_steps", "reached_end"):
        if key in stats:
            row[key] = stats[key]
    if extra:
        row.update(extra)
    return row


def run(field, sampler, oracle, seed=0, **kwargs):
    oracle.reset_counters()
    generator = torch.Generator().manual_seed(seed)
    samples, stats = sampler(field, COUNT, generator=generator, **kwargs)
    return record(oracle, samples, stats,
                  {"field_diagnostics": dict(field.diagnostics)})


# ---------------------------------------------------------------------------
# Stage: reference
# ---------------------------------------------------------------------------

def stage_reference():
    out = {}
    for n in N_LADDER:
        oracle, dataset = oracle_for(n)
        tall = oracle.tall
        mean, sd = tall.global_moments()
        floor_mean, floor_sd = monte_carlo_floor(tall, COUNT)
        within = math.sqrt(tall.global_variance)
        midpoint = float(tall.global_means.mean())
        exact = summarise(tall.sample_global(COUNT, torch.Generator().manual_seed(1)),
                          tall)
        out[str(n)] = {
            "x_total": float(np.sum(dataset["x"])),
            "branch_weights": tall.weights.tolist(),
            "branch_means": tall.global_means.tolist(),
            "within_branch_sd": within,
            "posterior_mean": mean, "posterior_sd": sd,
            "mode_separation_in_within_sd": float(
                abs(tall.global_means[1] - tall.global_means[0]) / within),
            "midpoint": midpoint,
            "lambda_max": float(oracle.schedule.lam(1.0)),
            "lambda_min_at_eps": float(oracle.schedule.lam(EPS)),
            "w1_floor_mean": floor_mean, "w1_floor_sd": floor_sd,
            "exact_draw_metrics": exact,
        }
        print(f"n={n:4d} weights={tall.weights.numpy().round(4)} "
              f"means={tall.global_means.numpy().round(4)} "
              f"sd_within={within:.4f} W1 floor={floor_mean:.4f}")
    store("reference", out)


# ---------------------------------------------------------------------------
# Stage: sampler control
# ---------------------------------------------------------------------------

def stage_control():
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        rows = {}
        field = ComposedField(oracle, "true", lam_max=hi, lam_min=lo)
        for steps in (20, 50, 100, 200, 500):
            rows[f"dpm2/steps={steps}"] = run(
                field, dpm2, oracle, timesteps=steps, lam_max=hi, lam_min=lo)
            print(f"  n={n} TRUE dpm2 steps={steps:4d} "
                  f"W1/sd={rows[f'dpm2/steps={steps}']['w1_over_sd']:.4f}")
        for steps in (200, 1000):
            rows[f"euler/steps={steps}"] = run(
                field, euler_maruyama, oracle, timesteps=steps,
                lam_max=hi, lam_min=lo)
        # terminal noise: the eps floor leaves lambda_min of residual width
        for eps in (1e-4, 1e-5):
            low = float(oracle.schedule.lam(eps))
            rows[f"dpm2/steps=200,eps={eps:g}"] = run(
                field, dpm2, oracle, timesteps=200, lam_max=hi, lam_min=low)
        rows["dpm2/steps=200,denoise"] = run(
            field, dpm2, oracle, timesteps=200, lam_max=hi, lam_min=lo,
            denoise=True)
        rows["dpm2/steps=200,churn=0.5"] = run(
            field, dpm2, oracle, timesteps=200, lam_max=hi, lam_min=lo,
            churn=0.5)
        out[str(n)] = rows
        for key, value in rows.items():
            print(f"  n={n:4d} TRUE {key:32s} W1/sd={value['w1_over_sd']:.4f} "
                  f"TV={value['weight_tv']:.4f} width={value['width_ratio']:.3f}")
    store("control", out)


def stage_control_churn():
    """Pick the stochasticity the compositional DPM2 arms must run with.

    The deterministic ``churn = 0`` predictor integrates the probability-flow
    ODE, which is a *deterministic map*: it cannot repair the difference
    between the start ``N(0, lambda_max^2 I)`` every method uses and the true
    ``p_{lambda_max}(theta | x)``, and under a bimodal target that difference
    is branch mass, which no later step can move.  COMPASS ships DPM with
    Karras churn on for exactly this reason.  This stage sweeps churn on the
    *exact* score so the setting used for every other DPM2 arm is chosen where
    the right answer is known.
    """
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        field = ComposedField(oracle, "true", lam_max=hi, lam_min=lo)
        rows = {}
        for churn in (0.0, 0.1, 0.2, 0.5, 1.0, 2.0):
            for steps in (200, 500):
                key = f"dpm2/churn={churn},steps={steps}"
                rows[key] = run(field, dpm2, oracle, timesteps=steps,
                                lam_max=hi, lam_min=lo, churn=churn)
                print(f"  n={n:4d} TRUE {key:30s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                      f"TV={rows[key]['weight_tv']:.4f} "
                      f"width={rows[key]['width_ratio']:.3f}")
        out[str(n)] = rows
    store("control_churn", out)


def stage_control_init():
    """Does the control's residual error come from the solver or from the start?

    Every arm starts at ``N(0, lambda_max^2 I)``, which is what COMPASS does.
    The exact diffused tall posterior at ``lambda_max`` differs from it by the
    posterior mean, an offset of ``|mean| / lambda_max`` -- small, but it is the
    *only* thing that decides branch mass under a deterministic ODE, because no
    mass crosses between basins after the split forms.  This stage repeats the
    control from the exact start so that difference is attributable.
    """
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        field = ComposedField(oracle, "true", lam_max=hi, lam_min=lo)
        rows = {}
        for steps in (200, 500):
            rows[f"dpm2/steps={steps},exact-init"] = run(
                field, dpm2, oracle, timesteps=steps, lam_max=hi, lam_min=lo,
                init="exact")
        rows["euler/steps=1000,exact-init"] = run(
            field, euler_maruyama, oracle, timesteps=1000, lam_max=hi,
            lam_min=lo, init="exact")
        # Raising lambda_max above lambda(1) is outside COMPASS's own schedule,
        # but an oracle score is defined everywhere, so it is the direct test of
        # whether N(0, lambda_max^2 I) is a good enough reference at the top.
        for lam_top in (4.0, 16.0):
            high = lam_top * hi
            rows[f"dpm2/steps=200,lam_max={high:.4g}"] = run(
                field, dpm2, oracle, timesteps=200, lam_max=high, lam_min=lo)
            rows[f"dpm2/steps=200,lam_max={high:.4g},exact-init"] = run(
                field, dpm2, oracle, timesteps=200, lam_max=high, lam_min=lo,
                init="exact")
        out[str(n)] = rows
        for key, value in rows.items():
            print(f"  n={n:4d} TRUE {key:34s} W1/sd={value['w1_over_sd']:.4f} "
                  f"TV={value['weight_tv']:.4f} width={value['width_ratio']:.3f}")
    store("control_init", out)


def stage_fielderror():
    """Score-field error of every rule against the exact diffused tall score.

    Section 14: this is the meaningful quantity for GAUSS/JAC, which claim to
    approximate the *true* diffused tall score, and the meaningless one for the
    F-NPSE and damped bridges, which are scores of deliberately different
    intermediate distributions.  Both are reported, with that distinction
    stated, and never called an "F-NPSE score error".

    States are drawn from the true diffused tall posterior at each level, so
    the error is measured where the true reverse diffusion actually travels.
    """
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        generator = torch.Generator().manual_seed(5)
        rows = {}
        for tau in (1.0, 0.75, 0.5, 0.25, 0.1, 0.0):
            lam = float(np.exp(math.log(lo) + tau * (math.log(hi) - math.log(lo))))
            t = float(oracle.schedule.time_of_lam(lam))
            tall = oracle.tall
            pick = torch.multinomial(tall.weights, 512, replacement=True,
                                     generator=generator)
            cov = tall.cov + lam ** 2 * torch.eye(oracle.dim, dtype=torch.float64)
            factor = torch.linalg.cholesky(cov)
            state = tall.means[pick] + torch.randn(
                512, oracle.dim, dtype=torch.float64, generator=generator) @ factor.mT
            truth = tall.score(state, lam)
            scale = float(truth[:, 0].pow(2).mean().sqrt())
            entry = {"tau": tau, "lambda": lam, "true_global_score_rms": scale}
            for name, field in (
                    ("naive", ComposedField(oracle, "naive", lam_max=hi, lam_min=lo)),
                    ("fnpe-diffused", ComposedField(oracle, "fnpe", lam_max=hi, lam_min=lo)),
                    ("arruda-d1=0.2", ComposedField(oracle, "damped", damping_d1=0.2,
                                                    lam_max=hi, lam_min=lo)),
                    ("gauss", ComposedField(oracle, "gauss", lam_max=hi, lam_min=lo)),
                    ("jac", ComposedField(oracle, "jac", lam_max=hi, lam_min=lo))):
                field_score = field(state, t, lam)
                gap = field_score[:, 0] - truth[:, 0]
                entry[name] = {
                    "global_rms_error": float(gap.pow(2).mean().sqrt()),
                    "relative": float(gap.pow(2).mean().sqrt()) / max(scale, 1e-300),
                    "bias": float(gap.mean()),
                }
            rows[f"tau={tau}"] = entry
            print(f"  n={n:4d} tau={tau:4.2f} lam={lam:7.4f} "
                  + " ".join(f"{k}={entry[k]['relative']:.3f}" for k in
                             ("naive", "fnpe-diffused", "arruda-d1=0.2",
                              "gauss", "jac")))
        out[str(n)] = rows
    store("fielderror", out)


# ---------------------------------------------------------------------------
# Stage: Geffner
# ---------------------------------------------------------------------------

def bridge_potential_1d(oracle, grid, t, a, prior_kind):
    """``log q_t^F(g)`` for the *marginal* F-NPSE bridge, up to a constant.

    ``(1-n) a log p(g) + sum_i log p_lambda(g | x_i)`` -- the exact
    one-dimensional projection of the bridge, which is all that is needed to
    ask whether it is normalisable and how many modes it has.
    """
    lam = float(oracle.schedule.lam(t))
    prior = oracle.prior.log_prob(grid, 0.0 if prior_kind == "clean" else lam)
    rows = oracle.rows.global_marginal_log_prob(grid, lam).sum(-1)
    return (1 - oracle.n) * a * prior + rows


def stage_geffner():
    out = {}
    times = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        grid = torch.linspace(-60.0, 60.0, 24001, dtype=torch.float64)
        properness = {}
        for prior_kind in ("clean", "diffused"):
            entries = []
            for t in times:
                potential = bridge_potential_1d(oracle, grid, t, 1.0 - t, prior_kind)
                tail = float(potential[-1] - potential[potential.numel() // 2])
                entries.append({
                    "t": t, "lambda": float(oracle.schedule.lam(t)),
                    "a": 1.0 - t,
                    "tail_log_density_change": tail,
                    "normalisable": bool(tail < -20.0),
                })
            properness[prior_kind] = entries
        out.setdefault(str(n), {})["properness"] = properness
        for prior_kind, entries in properness.items():
            bad = [e["t"] for e in entries if not e["normalisable"]]
            print(f"  n={n:4d} fnpe/{prior_kind:8s} improper at t={bad}")

        rows = {}
        for prior_kind in ("clean", "diffused"):
            field = ComposedField(oracle, "fnpe", fnpe_prior=prior_kind,
                                  lam_max=hi, lam_min=lo)
            for levels, steps, snr in ((100, 20, 0.1), (100, 5, 0.1),
                                       (100, 100, 0.1), (200, 100, 0.1),
                                       (50, 20, 0.1), (100, 500, 0.1),
                                       (100, 100, 0.02), (100, 100, 0.5)):
                key = f"{prior_kind}/T={levels},L={steps},snr={snr}"
                started = time.perf_counter()
                rows[key] = run(field, annealed_langevin, oracle, levels=levels,
                                steps_per_level=steps, snr=snr,
                                lam_max=hi, lam_min=lo)
                print(f"  n={n:4d} {key:34s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                      f"TV={rows[key]['weight_tv']:.4f} "
                      f"({time.perf_counter()-started:.0f}s)")
        out[str(n)]["langevin"] = rows
    store("geffner", out)


def stage_geffner_ladder():
    """A clean plateau demonstration for the F-NPSE bridge, plus its control.

    Two things the first Langevin sweep could not separate:

    * ``true`` -- annealed Langevin run against the *exact* diffused tall
      score.  At every level that chain targets ``p_lambda(theta | x)`` itself,
      so if this arm is exact the Langevin machinery is sound and anything the
      F-NPSE arm gets wrong belongs to its bridge; if it is not, the budget is
      simply too small and no statement about the bridge can be made.
    * a single-variable ladder in ``L`` at fixed ``T`` and step size, which is
      what "the metric has plateaued with increasing Langevin effort" actually
      requires -- the first sweep moved ``T``, ``L`` and ``snr`` together.
    """
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        rows = {}
        steps_ladder = (10, 20, 50, 100, 200, 500) if n < 128 else (10, 50, 200)
        for rule, prior in (("true", None), ("fnpe", "diffused")):
            field = (ComposedField(oracle, "true", lam_max=hi, lam_min=lo)
                     if rule == "true" else
                     ComposedField(oracle, "fnpe", fnpe_prior=prior,
                                   lam_max=hi, lam_min=lo))
            # The bridge contracts by 1/n on the shared coordinate; the true
            # diffused posterior does not, so its step must not be shrunk.
            scale = None if rule == "fnpe" else 1.0
            for steps in steps_ladder:
                key = f"{rule}/T=100,L={steps},snr=0.05"
                rows[key] = run(field, annealed_langevin, oracle, levels=100,
                                steps_per_level=steps, snr=0.05, lam_max=hi,
                                lam_min=lo, hierarchy_scale=scale)
                print(f"  n={n:4d} {key:30s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                      f"TV={rows[key]['weight_tv']:.4f} "
                      f"width={rows[key]['width_ratio']:.3f}")
            for levels in (50, 200):
                key = f"{rule}/T={levels},L=100,snr=0.05"
                rows[key] = run(field, annealed_langevin, oracle, levels=levels,
                                steps_per_level=100, snr=0.05, lam_max=hi,
                                lam_min=lo, hierarchy_scale=scale)
                print(f"  n={n:4d} {key:30s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                      f"TV={rows[key]['weight_tv']:.4f}")
            for snr in (0.01, 0.2):
                key = f"{rule}/T=100,L=100,snr={snr}"
                rows[key] = run(field, annealed_langevin, oracle, levels=100,
                                steps_per_level=100, snr=snr, lam_max=hi,
                                lam_min=lo, hierarchy_scale=scale)
                print(f"  n={n:4d} {key:30s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                      f"TV={rows[key]['weight_tv']:.4f}")
        out[str(n)] = rows
    store("geffner_ladder", out)


def stage_geffner_step():
    """Is the F-NPSE arm at large `n` wrong, or merely unable to move?

    Geffner's prescription shrinks the Langevin step on the shared coordinate
    by ``1/n`` because the *bridge* contracts by ``1/n`` there.  That is right
    at the noise end, where the bridge's shared width is ``lambda/sqrt(n)``, and
    wrong at the data end, where its width is the posterior's own ``sqrt(V)``
    and does not depend on ``n`` at all.  At n = 128 and ``lambda_min``, the
    prescribed step is ``snr lambda^2 / n ~ 4e-7`` against a target width of
    0.09, so the final levels need ~1e4 steps each to relax and the measured
    width ratio is 0.03-0.31: the chain has not moved, rather than moved to the
    wrong place.

    This stage repeats the arm with the shared step *unscaled*, which separates
    "the bridge is wrong" from "the prescribed step size cannot converge".
    """
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        field = ComposedField(oracle, "fnpe", fnpe_prior="diffused",
                              lam_max=hi, lam_min=lo)
        rows = {}
        for scale, label in ((None, "1/n"), (1.0, "unscaled")):
            for levels, steps, snr in ((100, 200, 0.05), (100, 200, 0.2),
                                       (100, 500, 0.2)):
                key = f"step={label}/T={levels},L={steps},snr={snr}"
                rows[key] = run(field, annealed_langevin, oracle, levels=levels,
                                steps_per_level=steps, snr=snr, lam_max=hi,
                                lam_min=lo, hierarchy_scale=scale)
                print(f"  n={n:4d} {key:34s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                      f"TV={rows[key]['weight_tv']:.4f} "
                      f"width={rows[key]['width_ratio']:.3f}")
        out[str(n)] = rows
    store("geffner_step", out)


# ---------------------------------------------------------------------------
# Stage: Arruda
# ---------------------------------------------------------------------------

def stage_arruda():
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        rows = {}
        for d1 in (1.0, 0.5, 0.2, 0.1, float(n) ** -0.5):
            field = ComposedField(oracle, "damped", damping_d1=d1,
                                  lam_max=hi, lam_min=lo)
            for rtol in (1e-1, 1e-2, 1e-3):
                key = f"d1={d1:.4f}/heun,rtol={rtol:g}"
                rows[key] = run(field, adaptive_heun, oracle, lam_max=hi,
                                lam_min=lo, e_rel=rtol,
                                e_abs=0.02576 * rtol / 0.1)
                print(f"  n={n:4d} {key:30s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                      f"TV={rows[key]['weight_tv']:.4f} nfe={rows[key]['nfe']}")
            key = f"d1={d1:.4f}/dpm2,steps=200"
            rows[key] = run(field, dpm2, oracle, timesteps=200,
                            lam_max=hi, lam_min=lo)
            print(f"  n={n:4d} {key:30s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                  f"TV={rows[key]['weight_tv']:.4f}")
        out[str(n)] = rows
    store("arruda", out)


# ---------------------------------------------------------------------------
# Stage: Linhart
# ---------------------------------------------------------------------------

def stage_linhart():
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        rows = {}
        for rule in ("gauss", "jac"):
            field = ComposedField(oracle, rule, lam_max=hi, lam_min=lo)
            for steps in (20, 50, 100, 200, 500):
                key = f"{rule}/dpm2,steps={steps}"
                rows[key] = run(field, dpm2, oracle, timesteps=steps,
                                lam_max=hi, lam_min=lo)
                print(f"  n={n:4d} {key:26s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                      f"TV={rows[key]['weight_tv']:.4f} "
                      f"({rows[key]['seconds']:.0f}s)")
        out[str(n)] = rows
    store("linhart", out)


# ---------------------------------------------------------------------------
# Stage: naive and Geffner-DPM2
# ---------------------------------------------------------------------------

def stage_naive():
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        rows = {}
        field = ComposedField(oracle, "naive", lam_max=hi, lam_min=lo)
        for steps in (100, 200, 500):
            rows[f"naive/dpm2,steps={steps}"] = run(
                field, dpm2, oracle, timesteps=steps, lam_max=hi, lam_min=lo)
        rows["naive/euler,steps=500"] = run(
            field, euler_maruyama, oracle, timesteps=500, lam_max=hi, lam_min=lo)
        for prior_kind in ("clean", "diffused"):
            field = ComposedField(oracle, "fnpe", fnpe_prior=prior_kind,
                                  lam_max=hi, lam_min=lo)
            rows[f"fnpe-{prior_kind}/dpm2,steps=200"] = run(
                field, dpm2, oracle, timesteps=200, lam_max=hi, lam_min=lo)
        out[str(n)] = rows
        for key, value in rows.items():
            print(f"  n={n:4d} {key:32s} W1/sd={value['w1_over_sd']:.4f} "
                  f"TV={value['weight_tv']:.4f} width={value['width_ratio']:.3f}")
    store("naive", out)


CHURN = 0.5


def stage_linhart_churn():
    """Oracle GAUSS and JAC under the *corrected* sampler setting.

    ``stage_control`` showed the deterministic predictor cannot reach the exact
    answer even from the exact score, so every DPM2 arm is repeated with the
    Karras churn COMPASS ships with.  Both settings are reported; an arm that
    only fails with churn = 0 was failing for the sampler's reason, not its own.
    """
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        rows = {}
        for rule in ("gauss", "jac", "naive", "fnpe"):
            field = ComposedField(oracle, rule, lam_max=hi, lam_min=lo)
            for steps in (200, 500):
                key = f"{rule}/dpm2,churn={CHURN},steps={steps}"
                rows[key] = run(field, dpm2, oracle, timesteps=steps,
                                lam_max=hi, lam_min=lo, churn=CHURN)
                print(f"  n={n:4d} {key:34s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                      f"TV={rows[key]['weight_tv']:.4f} "
                      f"width={rows[key]['width_ratio']:.3f}")
            key = f"{rule}/euler,steps=1000"
            rows[key] = run(field, euler_maruyama, oracle, timesteps=1000,
                            lam_max=hi, lam_min=lo)
            print(f"  n={n:4d} {key:34s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                  f"TV={rows[key]['weight_tv']:.4f} "
                  f"width={rows[key]['width_ratio']:.3f}")
        out[str(n)] = rows
    store("linhart_churn", out)


def stage_arruda_churn():
    """The damping ablation under the corrected sampler setting."""
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        rows = {}
        for d1 in (1.0, 0.5, 0.2, 0.1, float(n) ** -0.5):
            field = ComposedField(oracle, "damped", damping_d1=d1,
                                  lam_max=hi, lam_min=lo)
            for label, sampler, kwargs in (
                    (f"dpm2,churn={CHURN},steps=500", dpm2,
                     {"timesteps": 500, "churn": CHURN}),
                    ("euler,steps=1000", euler_maruyama, {"timesteps": 1000})):
                key = f"d1={d1:.4f}/{label}"
                rows[key] = run(field, sampler, oracle, lam_max=hi, lam_min=lo,
                                **kwargs)
                print(f"  n={n:4d} {key:34s} W1/sd={rows[key]['w1_over_sd']:.4f} "
                      f"TV={rows[key]['weight_tv']:.4f} "
                      f"width={rows[key]['width_ratio']:.3f}")
        out[str(n)] = rows
    store("arruda_churn", out)


# ---------------------------------------------------------------------------
# Stage: bridge inspection
# ---------------------------------------------------------------------------

def slice_potential(oracle, field, grid, t, lam):
    """The bridge's log density along the profiled slice, up to a constant.

    The slice pins each local at its exact *clean* conditional mode
    ``l_i(g) = a g + b x_i`` (branch-independent for this model), so the curve
    is a genuine one-dimensional cut of the composed field.  Its slope is the
    directional derivative ``S_g + a sum_i S_{l_i}`` along that slice, which is
    integrated by the trapezoid rule.  For the ``true`` rule the result is
    compared against the closed-form diffused marginal, which bounds the error
    the slice construction itself introduces.
    """
    mixture = oracle.mixture
    a, b = mixture.a, mixture.b
    x = torch.as_tensor(oracle.x, dtype=torch.float64)
    locals_ = a * grid.unsqueeze(-1) + b * x
    state = torch.cat([grid.unsqueeze(-1), locals_], dim=-1)
    score = field(state, t, lam)
    slope = score[:, 0] + a * score[:, 1:].sum(-1)
    step = grid[1] - grid[0]
    potential = torch.cat([
        torch.zeros(1, dtype=torch.float64),
        torch.cumsum(0.5 * (slope[1:] + slope[:-1]) * step, dim=0),
    ])
    return potential - potential.max()


def describe_curve(grid, potential, weight_floor=1e-4):
    density = torch.exp(potential - potential.max())
    step = float(grid[1] - grid[0])
    density = density / (density.sum() * step)
    interior = potential[1:-1]
    peaks = ((interior > potential[:-2]) & (interior > potential[2:]))
    indices = torch.nonzero(peaks).reshape(-1) + 1
    order = torch.argsort(potential[indices], descending=True)
    indices = indices[order][:4]
    indices, _ = torch.sort(indices)
    modes = grid[indices].tolist()
    masses, valley = [], None
    if len(indices) >= 2:
        edges = [0] + [int((indices[k] + indices[k + 1]) // 2)
                       for k in range(len(indices) - 1)] + [len(grid)]
        masses = [float(density[edges[k]:edges[k + 1]].sum() * step)
                  for k in range(len(indices))]
        low = int(torch.argmin(potential[indices[0]:indices[1]]) + indices[0])
        valley = float(min(potential[indices[0]], potential[indices[1]])
                       - potential[low])
    elif len(indices) == 1:
        masses = [1.0]
    return {"modes": modes, "masses": masses, "valley_log_depth": valley,
            "num_modes": len(indices)}


def stage_bridges():
    out = {}
    taus = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        grid = torch.linspace(-8.0, 8.0, 6401, dtype=torch.float64)
        rows = {}
        arms = {
            "true": ComposedField(oracle, "true", lam_max=hi, lam_min=lo),
            "naive": ComposedField(oracle, "naive", lam_max=hi, lam_min=lo),
            "fnpe-diffused": ComposedField(oracle, "fnpe", fnpe_prior="diffused",
                                           lam_max=hi, lam_min=lo),
            "fnpe-clean": ComposedField(oracle, "fnpe", fnpe_prior="clean",
                                        lam_max=hi, lam_min=lo),
            "arruda-d1=0.2": ComposedField(oracle, "damped", damping_d1=0.2,
                                           lam_max=hi, lam_min=lo),
            "gauss": ComposedField(oracle, "gauss", lam_max=hi, lam_min=lo),
            "jac": ComposedField(oracle, "jac", lam_max=hi, lam_min=lo),
        }
        for tau in taus:
            lam = float(np.exp(math.log(lo) + tau * (math.log(hi) - math.log(lo))))
            t = float(oracle.schedule.time_of_lam(lam))
            entry = {"tau": tau, "lambda": lam, "t": t}
            exact = oracle.tall.global_log_prob(grid, lam)
            entry["exact_marginal"] = describe_curve(grid, exact - exact.max())
            for name, field in arms.items():
                potential = slice_potential(oracle, field, grid, t, lam)
                entry[name] = describe_curve(grid, potential)
                entry[name]["curve"] = potential[::16].tolist()
            entry["grid"] = grid[::16].tolist()
            entry["exact_curve"] = (exact - exact.max())[::16].tolist()
            rows[f"tau={tau}"] = entry
            print(f"  n={n:4d} tau={tau:4.2f} lam={lam:7.4f} "
                  + " ".join(f"{k}:{entry[k]['num_modes']}" for k in
                             ("exact_marginal", "true", "naive",
                              "fnpe-diffused", "arruda-d1=0.2", "gauss", "jac")))
        out[str(n)] = rows
    store("bridges", out)


# ---------------------------------------------------------------------------

STAGES = {
    "reference": stage_reference, "control": stage_control,
    "control_init": stage_control_init, "fielderror": stage_fielderror,
    "control_churn": stage_control_churn,
    "linhart_churn": stage_linhart_churn, "arruda_churn": stage_arruda_churn,
    "geffner": stage_geffner, "arruda": stage_arruda,
    "geffner_ladder": stage_geffner_ladder, "geffner_step": stage_geffner_step,
    "linhart": stage_linhart, "naive": stage_naive, "bridges": stage_bridges,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs="+", default=["reference"],
                        choices=sorted(STAGES) + ["all"])
    parser.add_argument("--mu0", type=float, default=None,
                        help="override the mixture's mu_0 (see SETTING)")
    parser.add_argument("--n", type=int, nargs="+", default=None,
                        help="observation counts to run (default 8 32 128)")
    args = parser.parse_args(argv)
    if args.mu0 is not None:
        import dataclasses
        SETTING["mixture"] = dataclasses.replace(M.DEFAULT, mu_0=args.mu0)
        SETTING["tag"] = f"_mu{args.mu0:g}"
    if args.n:
        global N_LADDER
        N_LADDER = tuple(args.n)
    names = sorted(STAGES) if "all" in args.stage else args.stage
    for name in names:
        print(f"=== {name} ===")
        started = time.perf_counter()
        STAGES[name]()
        print(f"--- {name} done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
