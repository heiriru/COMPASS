#!/usr/bin/env python
"""Bridge topology, bifurcation-aligned JAC, and an adaptive F-NPSE step.

Follow-up to ``exp6_gm_oracle_bridges``.  Nothing is trained and no learned
score is used anywhere: every quantity is the exact closed form of the
Gaussian-mixture benchmark, exactly as in ``exp6``, and the datasets are built
by ``exp6.oracle_for`` so the baselines carry over unchanged.

The two questions:

1. can JAC's excellent local geometry be preserved while correcting its global
   bifurcation topology, using only a scalar ``alpha(tau, n)`` that interpolates
   between naive composition and JAC?
2. can F-NPSE mix at large ``n`` with a noise-level-dependent Langevin step?

Stages (``--stage``), each writing into ``artifacts/gm_topology/``:

``truth``       the *true* bridge topology -- ``tau_split``, ``m_pm(tau)``,
                ``Delta E_barrier(tau)``, ``w_+(tau)`` -- plus the validation of
                the one-dimensional slice against the exact diffused marginal
                (sections 2-3).
``topoerr``     the same quantities for JAC, GAUSS, F-NPSE and naive, and the
                signed errors against truth (section 4).
``alpha_const`` the constant-``alpha`` sweep: topology at every level and the
                endpoint under the validated churned sampler (sections 5-6).
``alpha_topo``  the per-level oracle ``alpha*`` chosen from bridge geometry
                alone, and smooth low-dimensional fits to it (section 7).
``topojac``     ORACLE-TOPO-JAC: sampling the fitted schedules (section 8).
``alpha_simple``how simple ``alpha`` can be, and what it correlates with
                (section 9).
``fnpe_step``   level-dependent Langevin steps, with stability and mixing
                reported separately (sections 10-12).
``fnpe_ladder`` the convergence ladder for the promising schedules, whose real
                test is n = 128 at mu_0 = 7.5 (section 13).
``cross``       the final cross-comparison (section 14).

The ``mu_0 = 7.5`` setting is mandatory for every multimodal claim at n = 128:
the branch separation of the standard benchmark falls like ``1/sqrt(n)``, so
the default n = 128 posterior has a dip ratio of 0.999 and is bimodal in name
only.  Run those stages as ``--mu0 7.5 --n 128``.

Run:  python ModeTree/experiments/exp7_gm_topology.py --stage truth
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

import argparse    # noqa: E402
import dataclasses  # noqa: E402
import json        # noqa: E402
import math        # noqa: E402
import time        # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

import model as M   # noqa: E402

import exp6_gm_oracle_bridges as exp6           # noqa: E402
import gm_topology as TP                        # noqa: E402
from gm_bridges import (ComposedField, annealed_langevin, dpm2,  # noqa: E402
                        power_schedule, sigmoid_schedule)

torch.set_default_dtype(torch.float64)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "gm_topology"
N_LADDER = (8, 32, 128)

# Everything the previous experiment validated is reused verbatim.
CHURN = exp6.CHURN            # 0.5 -- the setting that fixed the sampler control
COUNT = exp6.COUNT            # 2000 walkers
EPS = exp6.EPS                # 1e-3 terminal time

# Levels the topology is measured on.  41 points, uniform in log lambda, from
# the noise end to the data end; fine enough that ``tau_split`` is resolved to
# 0.025 and coarse enough that a full alpha sweep at n = 128 is minutes.
LEVELS = tuple(float(v) for v in np.linspace(1.0, 0.0, 41))
# A coarse subset for the expensive per-arm profiles and the report's tables.
COARSE = (1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15,
          0.1, 0.05, 0.0)
ALPHA_GRID = tuple(float(v) for v in np.linspace(0.0, 1.0, 41))
SLICE_POINTS = 8001
# The F-NPSE schedule *screen* runs at this many Langevin steps per level;
# convergence is established afterwards by ``fnpe_ladder``, which is what the
# plateau requirement actually needs.  Screening at 500 would cost five times
# as much and still not be a convergence demonstration.
SCREEN_STEPS = 100


def oracle_for(n, seed=0):
    return exp6.oracle_for(n, seed)


def lam_bounds(oracle):
    return exp6.lam_bounds(oracle, EPS)


def tag():
    return exp6.SETTING["tag"]


def store(name, payload):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"{name}{tag()}.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2, default=float))
    print(f"wrote {path}")


def load(name):
    path = ARTIFACTS / f"{name}{tag()}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run that stage first")
    return json.loads(path.read_text())


def run(field, sampler, oracle, seed=0, drop_trace=True, **kwargs):
    """One sampler run, recorded with ``exp6``'s metric set."""
    oracle.reset_counters()
    generator = torch.Generator().manual_seed(seed)
    samples, stats = sampler(field, COUNT, generator=generator, **kwargs)
    trace = stats.pop("level_trace", None)
    row = exp6.record(oracle, samples, stats,
                      {"field_diagnostics": dict(field.diagnostics)})
    # ``exp6.record`` copies a fixed key set; the stability diagnostics this
    # experiment added are carried through explicitly.
    for key in ("max_abs_state", "max_abs_score"):
        if key in stats:
            row[key] = stats[key]
    if trace and not drop_trace:
        # Mixing is judged from how the ensemble width moves across levels,
        # but the full trace is large; keep a decimated copy.
        row["level_trace"] = trace[::max(1, len(trace) // 21)]
    return row


def show(n, key, row, width=40):
    print(f"  n={n:4d} {key:{width}s} W1/sd={row['w1_over_sd']:.4f} "
          f"TV={row['weight_tv']:.4f} width={row['width_ratio']:.3f} "
          f"{'DIVERGED' if row.get('diverged') else ''}")


# ---------------------------------------------------------------------------
# Linear-in-alpha slice machinery
# ---------------------------------------------------------------------------

def alpha_slice_endpoints(oracle, lam_max, lam_min, tau, points=SLICE_POINTS):
    """The two slice slopes the whole ``alpha`` family is spanned by.

    ``S^(alpha)`` is affine in ``alpha`` by construction, and the profiled
    slice is a fixed linear functional of the field, so the slice slope is
    affine in ``alpha`` too:

        slope_alpha(g) = slope_0(g) + alpha [ slope_1(g) - slope_0(g) ].

    Evaluating the two endpoints once per level therefore gives every ``alpha``
    on the grid for free and *exactly* -- no interpolation error enters the
    topology objective.
    """
    lam = TP.lam_of_tau(tau, lam_max, lam_min)
    t = float(oracle.schedule.time_of_lam(lam))
    grid = TP.slice_grid(oracle, lam, points)
    naive = ComposedField(oracle, "naive", lam_max=lam_max, lam_min=lam_min)
    jac = ComposedField(oracle, "jac", lam_max=lam_max, lam_min=lam_min)
    slope0 = TP.slice_slope(oracle, naive, grid, t, lam)
    slope1 = TP.slice_slope(oracle, jac, grid, t, lam)

    def continuous(value, weight):
        point = torch.tensor([float(value)], dtype=torch.float64)
        a = float(TP.slice_slope(oracle, naive, point, t, lam)[0])
        b = float(TP.slice_slope(oracle, jac, point, t, lam)[0])
        return a + weight * (b - a)

    return {"grid": grid, "slope0": slope0, "slope1": slope1,
            "lambda": lam, "t": t, "continuous": continuous}


def topology_at_alpha(endpoints, weight, mass_floor=TP.MASS_FLOOR):
    slope = endpoints["slope0"] + weight * (endpoints["slope1"]
                                            - endpoints["slope0"])
    entry = TP.describe(
        endpoints["grid"], slope,
        lambda v, _w=weight: endpoints["continuous"](v, _w), mass_floor)
    entry["lambda"] = endpoints["lambda"]
    return entry


def trim(entry):
    """The stored form of one topology row: no curves, no tensors."""
    keep = ("tau", "lambda", "num_modes", "modes", "valleys", "split_point",
            "mode_minus", "mode_plus", "barrier", "barrier_minus",
            "barrier_plus", "w_minus", "w_plus", "mode_width", "curve_mean",
            "curve_sd", "normalisable")
    return {k: entry[k] for k in keep if k in entry}


# ---------------------------------------------------------------------------
# Stage: truth (sections 2-3)
# ---------------------------------------------------------------------------

def stage_truth():
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        within = math.sqrt(oracle.tall.global_variance)
        field = ComposedField(oracle, "true", lam_max=hi, lam_min=lo)
        rows = TP.profile(oracle, field, LEVELS, hi, lo, points=SLICE_POINTS)
        split = TP.split_level(rows)
        out[str(n)] = {
            "within_branch_sd": within,
            "lambda_max": hi, "lambda_min": lo,
            "tau_split": split,
            "levels": {k: trim(v) for k, v in rows.items()},
            "slice_validation": TP.slice_validation(oracle, COARSE, hi, lo,
                                                    points=SLICE_POINTS),
            "exact_branch_means": oracle.tall.global_means.tolist(),
            "exact_branch_weights": oracle.tall.weights.tolist(),
        }
        print(f"n={n:4d} tau_split={split} within_sd={within:.4f}")
        for tau in COARSE:
            row = rows[f"tau={tau:.4f}"]
            print(f"    tau={tau:4.2f} lam={row['lambda']:7.4f} "
                  f"modes={row['num_modes']} "
                  f"m=({row.get('mode_minus')}, {row.get('mode_plus')}) "
                  f"barrier={row.get('barrier', 0.0):7.3f} "
                  f"w+={row.get('w_plus')}")
    store("truth", out)


# ---------------------------------------------------------------------------
# Stage: topoerr (section 4)
# ---------------------------------------------------------------------------

ARMS = {
    "jac": dict(rule="jac"),
    "gauss": dict(rule="gauss"),
    "fnpe": dict(rule="fnpe", fnpe_prior="diffused"),
    "naive": dict(rule="naive"),
    "arruda-d1=0.2": dict(rule="damped", damping_d1=0.2),
}


def stage_topoerr():
    truth = load("truth")
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        within = truth[str(n)]["within_branch_sd"]
        true_levels = truth[str(n)]["levels"]
        rows = {}
        for name, spec in ARMS.items():
            field = ComposedField(oracle, lam_max=hi, lam_min=lo, **spec)
            profile = TP.profile(oracle, field, LEVELS, hi, lo,
                                 points=SLICE_POINTS)
            split = TP.split_level(profile)
            errors = {}
            for key, entry in profile.items():
                reference = true_levels[key]
                errors[key] = {
                    "num_modes": entry["num_modes"],
                    "true_num_modes": reference["num_modes"],
                    "mode_minus_error_sd": _gap(entry.get("mode_minus"),
                                                reference.get("mode_minus"),
                                                within),
                    "mode_plus_error_sd": _gap(entry.get("mode_plus"),
                                               reference.get("mode_plus"),
                                               within),
                    "barrier": entry.get("barrier"),
                    "true_barrier": reference.get("barrier"),
                    "barrier_error": _gap(entry.get("barrier"),
                                          reference.get("barrier"), 1.0,
                                          signed=True),
                    "w_plus": entry.get("w_plus"),
                    "true_w_plus": reference.get("w_plus"),
                    "w_plus_error": _gap(entry.get("w_plus"),
                                         reference.get("w_plus"), 1.0,
                                         signed=True),
                }
            rows[name] = {
                "tau_split": split,
                "true_tau_split": truth[str(n)]["tau_split"],
                "delta_tau_split": (None if split is None
                                    or truth[str(n)]["tau_split"] is None
                                    else split - truth[str(n)]["tau_split"]),
                "levels": {k: trim(v) for k, v in profile.items()},
                "errors": errors,
            }
            print(f"  n={n:4d} {name:14s} tau_split={split} "
                  f"(true {truth[str(n)]['tau_split']}) "
                  f"dtau={rows[name]['delta_tau_split']}")
            for tau in (0.75, 0.5, 0.25, 0.1, 0.0):
                key = f"tau={tau:.4f}"
                if key not in errors:
                    continue
                e = errors[key]
                print(f"      tau={tau:4.2f} modes={e['num_modes']}"
                      f"/{e['true_num_modes']} "
                      f"dm+={_fmt(e['mode_plus_error_sd'])} "
                      f"barrier={_fmt(e['barrier'])}/{_fmt(e['true_barrier'])} "
                      f"w+={_fmt(e['w_plus'])}/{_fmt(e['true_w_plus'])}")
        out[str(n)] = rows
    store("topoerr", out)


def _gap(value, reference, scale, signed=False):
    if value is None or reference is None:
        return None
    delta = (value - reference) / scale
    return float(delta if signed else abs(delta))


def _fmt(value, places=3):
    return "----" if value is None else f"{value:.{places}f}"


# ---------------------------------------------------------------------------
# Stage: alpha_const (sections 5-6)
# ---------------------------------------------------------------------------

CONST_ALPHAS = tuple(round(0.1 * k, 1) for k in range(11))


def stage_alpha_const():
    truth = load("truth")
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        within = truth[str(n)]["within_branch_sd"]
        true_levels = truth[str(n)]["levels"]
        endpoints = {tau: alpha_slice_endpoints(oracle, hi, lo, tau)
                     for tau in COARSE}
        rows = {}
        for weight in CONST_ALPHAS:
            topology = {}
            for tau in COARSE:
                entry = topology_at_alpha(endpoints[tau], weight)
                entry["tau"] = tau
                topology[f"tau={tau:.4f}"] = trim(entry)
            split = TP.split_level(topology)
            field = ComposedField(oracle, "alpha", alpha=weight,
                                  lam_max=hi, lam_min=lo)
            # 200 steps, not 500: ``exp6``'s step ladder showed every
            # precision-weighted rule converged by 50-100 steps under this
            # sampler, so the extra 2.5x here would buy nothing but wall
            # clock.  ``topojac`` runs both 200 and 500 on the schedules that
            # matter, which re-checks that on the corrected field.
            sampled = run(field, dpm2, oracle, timesteps=200, lam_max=hi,
                          lam_min=lo, churn=CHURN)
            losses = []
            for tau in COARSE:
                key = f"tau={tau:.4f}"
                losses.append(TP.topology_loss(topology[key], true_levels[key],
                                               within))
            rows[f"alpha={weight:.1f}"] = {
                "alpha": weight, "tau_split": split,
                "true_tau_split": truth[str(n)]["tau_split"],
                "topology": topology,
                "mean_topology_loss": float(np.mean(losses)),
                "sampled": sampled,
            }
            show(n, f"alpha={weight:.1f} split={split} "
                    f"loss={np.mean(losses):.3f}", sampled, width=44)
        out[str(n)] = rows
    store("alpha_const", out)


# ---------------------------------------------------------------------------
# Stage: alpha_topo (section 7)
# ---------------------------------------------------------------------------

def stage_alpha_topo():
    truth = load("truth")
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        within = truth[str(n)]["within_branch_sd"]
        true_levels = truth[str(n)]["levels"]
        best, curves = {}, {}
        for tau in LEVELS:
            key = f"tau={tau:.4f}"
            endpoints = alpha_slice_endpoints(oracle, hi, lo, tau)
            reference = true_levels[key]
            scores, topologies = [], []
            for weight in ALPHA_GRID:
                entry = topology_at_alpha(endpoints, weight)
                topologies.append(entry)
                scores.append(TP.topology_loss(entry, reference, within))
            index = int(np.argmin(scores))
            # The objective is flat in alpha wherever the mode count is the
            # only thing it can see, so record the whole plateau rather than
            # its left edge: the schedule is fitted to the plateau midpoint.
            floor = scores[index]
            plateau = [ALPHA_GRID[k] for k, s in enumerate(scores)
                       if s <= floor + 1e-9]
            best[key] = {
                "tau": tau, "lambda": endpoints["lambda"],
                "alpha_star": ALPHA_GRID[index],
                "alpha_plateau": [min(plateau), max(plateau)],
                "alpha_mid": 0.5 * (min(plateau) + max(plateau)),
                "loss": floor,
                "loss_at_1": scores[-1], "loss_at_0": scores[0],
                "topology": trim(topologies[index]),
                "true_topology": reference,
            }
            curves[key] = {"alpha": list(ALPHA_GRID), "loss": scores,
                           "num_modes": [t["num_modes"] for t in topologies],
                           "w_plus": [t.get("w_plus") for t in topologies],
                           "barrier": [t.get("barrier") for t in topologies],
                           "mode_plus": [t.get("mode_plus") for t in topologies]}
            print(f"  n={n:4d} tau={tau:5.3f} lam={endpoints['lambda']:8.4f} "
                  f"alpha*={ALPHA_GRID[index]:.3f} "
                  f"plateau=[{min(plateau):.2f},{max(plateau):.2f}] "
                  f"loss={floor:.4f} (alpha=1: {scores[-1]:.4f})")
        fits = fit_schedules(best)
        out[str(n)] = {"per_level": best, "loss_curves": curves, "fits": fits}
        for name, fit in fits.items():
            print(f"  n={n:4d} fit {name:18s} rmse={fit['rmse']:.4f} "
                  f"{fit['description']}")
    store("alpha_topo", out)


def fit_schedules(best):
    """Smooth low-dimensional fits to the per-level oracle ``alpha``."""
    taus = np.array([row["tau"] for row in best.values()])
    target = np.array([row["alpha_mid"] for row in best.values()])
    order = np.argsort(taus)
    taus, target = taus[order], target[order]
    fits = {}

    def record(name, values, description, parameters):
        values = np.clip(values, 0.0, 1.0)
        fits[name] = {
            "rmse": float(np.sqrt(np.mean((values - target) ** 2))),
            "description": description, "parameters": parameters,
            "tau": taus.tolist(), "alpha": values.tolist(),
            "target": target.tolist(),
        }

    record("constant", np.full_like(taus, target.mean()),
           f"alpha = {target.mean():.3f}", {"alpha": float(target.mean())})

    # Monotone piecewise-linear in tau with 3 and 5 knots, fitted by least
    # squares on the knot values (the design matrix is the hat basis).
    for knots in (3, 5):
        nodes = np.linspace(0.0, 1.0, knots)
        design = _hat_basis(taus, nodes)
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        record(f"piecewise{knots}", design @ coefficients,
               "knots " + ", ".join(f"{v:.2f}@{u:.2f}"
                                    for u, v in zip(nodes, coefficients)),
               {"nodes": nodes.tolist(), "values": coefficients.tolist()})

    # Sigmoid in tau, fitted on the logit of the (clipped) target.
    clipped = np.clip(target, 0.02, 0.98)
    logit = np.log(clipped / (1.0 - clipped))
    slope, intercept = np.polyfit(taus, logit, 1)
    record("sigmoid_tau", 1.0 / (1.0 + np.exp(-(slope * taus + intercept))),
           f"sigma({slope:.3f} tau + {intercept:.3f})",
           {"slope": float(slope), "intercept": float(intercept)})

    # Low-order polynomial in log lambda, the coordinate the sampler steps in.
    lams = np.array([row["lambda"] for row in best.values()])[order]
    log_lam = np.log(lams)
    for degree in (1, 2):
        coefficients = np.polyfit(log_lam, target, degree)
        record(f"poly_loglambda{degree}", np.polyval(coefficients, log_lam),
               f"degree {degree} in log lambda",
               {"coefficients": coefficients.tolist()})
    return fits


def _hat_basis(points, nodes):
    design = np.zeros((len(points), len(nodes)))
    for index, node in enumerate(nodes):
        left = nodes[index - 1] if index > 0 else node - 1.0
        right = nodes[index + 1] if index < len(nodes) - 1 else node + 1.0
        rising = (points - left) / (node - left)
        falling = (right - points) / (right - node)
        design[:, index] = np.clip(np.minimum(rising, falling), 0.0, 1.0)
    return design


def schedule_from_fit(fit, kind, lam_max, lam_min):
    """Turn a stored fit into an ``alpha(tau)`` callable."""
    if kind == "interp":
        nodes = np.array(fit["tau"])
        values = np.array(fit["alpha"])
        return lambda tau: float(np.clip(np.interp(tau, nodes, values), 0.0, 1.0))
    parameters = fit["parameters"]
    if kind == "constant":
        return lambda tau, _a=parameters["alpha"]: _a
    if kind.startswith("piecewise"):
        nodes = np.array(parameters["nodes"])
        values = np.array(parameters["values"])
        return lambda tau: float(np.clip(np.interp(tau, nodes, values), 0.0, 1.0))
    if kind == "sigmoid_tau":
        slope, intercept = parameters["slope"], parameters["intercept"]
        return lambda tau: 1.0 / (1.0 + math.exp(
            -max(min(slope * tau + intercept, 60.0), -60.0)))
    if kind.startswith("poly_loglambda"):
        coefficients = parameters["coefficients"]
        lo, hi = math.log(lam_min), math.log(lam_max)
        return lambda tau: float(np.clip(
            np.polyval(coefficients, lo + tau * (hi - lo)), 0.0, 1.0))
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Stage: topojac (section 8)
# ---------------------------------------------------------------------------

def stage_topojac():
    topo = load("alpha_topo")
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        fits = topo[str(n)]["fits"]
        rows = {}

        # The unsmoothed per-level alpha, as an upper bound on what any fit of
        # it can achieve.
        levels = topo[str(n)]["per_level"]
        raw = {"tau": [row["tau"] for row in levels.values()],
               "alpha": [row["alpha_mid"] for row in levels.values()]}
        order = np.argsort(raw["tau"])
        raw = {"tau": list(np.array(raw["tau"])[order]),
               "alpha": list(np.array(raw["alpha"])[order])}
        candidates = {"per_level": schedule_from_fit(raw, "interp", hi, lo)}
        for name in ("constant", "piecewise3", "piecewise5", "sigmoid_tau",
                     "poly_loglambda2"):
            candidates[name] = schedule_from_fit(fits[name], name, hi, lo)

        for name, schedule in candidates.items():
            field = ComposedField(oracle, "alpha", alpha=schedule,
                                  lam_max=hi, lam_min=lo)
            for steps in (200, 500):
                key = f"{name}/dpm2,churn={CHURN},steps={steps}"
                rows[key] = run(field, dpm2, oracle, timesteps=steps,
                                lam_max=hi, lam_min=lo, churn=CHURN)
                rows[key]["alpha_profile"] = [
                    [round(tau, 3), round(schedule(tau), 4)]
                    for tau in (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)
                ]
                show(n, key, rows[key], width=44)
        out[str(n)] = rows
    store("topojac", out)


# ---------------------------------------------------------------------------
# Stage: topoprofile (figures 1-3 and 7)
# ---------------------------------------------------------------------------

def stage_topoprofile():
    """Topology *and* curves for true / JAC / TOPO-JAC / naive / F-NPSE.

    Cheap, because the whole ``alpha`` family shares one pair of slice slopes
    per level: ``naive``, ``jac`` and the fitted ``alpha(tau)`` schedule all
    come out of the same two evaluations, so only ``true`` and ``fnpe`` cost
    anything extra.
    """
    topo = load("alpha_topo")
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        best = _best_alpha_schedule(topo[str(n)], hi, lo)
        schedule = best["schedule"]
        true_field = ComposedField(oracle, "true", lam_max=hi, lam_min=lo)
        fnpe_field = ComposedField(oracle, "fnpe", fnpe_prior="diffused",
                                   lam_max=hi, lam_min=lo)
        rows = {}
        for tau in LEVELS:
            key = f"tau={tau:.4f}"
            endpoints = alpha_slice_endpoints(oracle, hi, lo, tau)
            grid, lam, t = endpoints["grid"], endpoints["lambda"], endpoints["t"]
            entry = {"tau": tau, "lambda": lam,
                     "alpha": float(schedule(tau))}
            curves = {}
            for name, weight in (("naive", 0.0), ("jac", 1.0),
                                 ("topojac", float(schedule(tau)))):
                topology = topology_at_alpha(endpoints, weight)
                topology["tau"] = tau
                entry[name] = trim(topology)
                slope = (endpoints["slope0"]
                         + weight * (endpoints["slope1"] - endpoints["slope0"]))
                curves[name] = TP.potential_from_slope(grid, slope)
            for name, field in (("true", true_field), ("fnpe", fnpe_field)):
                slope = TP.slice_slope(oracle, field, grid, t, lam)
                topology = TP.describe(grid, slope)
                topology["tau"] = tau
                entry[name] = trim(topology)
                curves[name] = TP.potential_from_slope(grid, slope)
            stride = max(1, len(grid) // 400)
            entry["grid"] = grid[::stride].tolist()
            for name, curve in curves.items():
                entry[f"{name}_curve"] = curve[::stride].tolist()
            rows[key] = entry
            print(f"  n={n:4d} tau={tau:5.3f} alpha={entry['alpha']:.3f} "
                  + " ".join(f"{k}:{entry[k]['num_modes']}"
                             for k in ("true", "jac", "topojac", "fnpe")))
        out[str(n)] = {"schedule": best["name"], "levels": rows}
    store("topoprofile", out)


# ---------------------------------------------------------------------------
# Stage: alpha_simple (section 9)
# ---------------------------------------------------------------------------

def stage_alpha_simple():
    """How simple can ``alpha`` be, and does it track anything JAC knows?"""
    topo = load("alpha_topo")
    truth = load("truth")
    table, diagnostics = {}, {}
    for n in N_LADDER:
        # Sorted ascending in tau: the stored levels run from the noise end
        # down, and ``np.interp`` silently returns nonsense on a descending
        # abscissa -- which made the "tau-only" and "separable" schedules
        # something other than what they claimed to be.
        levels = sorted(topo[str(n)]["per_level"].values(),
                        key=lambda row: row["tau"])
        table[str(n)] = {"tau": [r["tau"] for r in levels],
                         "alpha": [r["alpha_mid"] for r in levels],
                         "lambda": [r["lambda"] for r in levels]}
        diagnostics[str(n)] = bridge_diagnostics(n, table[str(n)]["lambda"])

    forms = separable_and_parametric(table)
    correlations = {}
    for n in N_LADDER:
        alpha = np.array(table[str(n)]["alpha"])
        correlations[str(n)] = {
            name: _correlation(alpha, np.array(values))
            for name, values in diagnostics[str(n)].items()
        }
        print(f"  n={n:4d} correlations with alpha*: "
              + ", ".join(f"{k}={_fmt(v)}"
                          for k, v in correlations[str(n)].items()))

    # Sample the two simplest forms that survive, at every n.
    sampled = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        rows = {}
        for name in ("n_only", "tau_only", "separable", "parametric"):
            schedule = simple_schedule(forms, name, n, hi, lo)
            field = ComposedField(oracle, "alpha", alpha=schedule,
                                  lam_max=hi, lam_min=lo)
            key = f"{name}/dpm2,churn={CHURN},steps=500"
            rows[key] = run(field, dpm2, oracle, timesteps=500, lam_max=hi,
                            lam_min=lo, churn=CHURN)
            rows[key]["alpha_profile"] = [
                [round(tau, 3), round(schedule(tau), 4)]
                for tau in (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)]
            show(n, key, rows[key], width=44)
        sampled[str(n)] = rows

    store("alpha_simple", {"table": table, "forms": forms,
                           "diagnostics": diagnostics,
                           "correlations": correlations, "sampled": sampled})


def bridge_diagnostics(n, lams):
    """Quantities the JAC rule already computes, at each level.

    Nothing here needs the answer: every entry is available to a sampler that
    has the row posteriors and their curvature, which is exactly what JAC uses.
    If ``alpha*`` tracks one of them, the correction has an oracle-free proxy.
    """
    oracle, _ = oracle_for(n)
    hi, lo = lam_bounds(oracle)
    field = ComposedField(oracle, "jac", lam_max=hi, lam_min=lo)
    rows = oracle.rows
    # Evaluate the curvature diagnostics at the *bridge's own* dominant mode,
    # approximated by the exact posterior mean -- a sampler has an ensemble
    # there, so this is a stand-in for an ensemble average, not a cheat: it
    # uses no branch information.
    centre = float((oracle.tall.weights * oracle.tall.global_means).sum())
    out = {"row_variance": [], "between_mode_variance": [],
           "curvature_magnitude": [], "precision_ratio": [],
           "condition_number": [], "log_lambda": []}
    a, b = oracle.mixture.a, oracle.mixture.b
    x = torch.as_tensor(oracle.x, dtype=torch.float64)
    point = torch.cat([torch.tensor([centre], dtype=torch.float64),
                       a * centre + b * x]).unsqueeze(0)
    rows_state = oracle.rows_state(point)
    for lam in lams:
        lam = float(lam)
        cov = rows.cov + (lam ** 2) * torch.eye(2, dtype=torch.float64)
        hess = rows.hessians(rows_state, lam)                    # (1, n, 2, 2)
        eigenvalues = torch.linalg.eigvalsh(hess)
        out["row_variance"].append(float(cov[0, 0]))
        spread = float(rows.means[:, :, 0].std(unbiased=False).mean())
        out["between_mode_variance"].append(spread ** 2)
        out["curvature_magnitude"].append(float(hess.abs().mean()))
        prior_hess = float(oracle.prior.hessian(
            torch.tensor(centre, dtype=torch.float64), lam))
        out["precision_ratio"].append(abs(prior_hess) / max(
            float(hess[..., 0, 0].abs().mean()), 1e-12))
        out["condition_number"].append(float(
            (eigenvalues.abs().max(-1).values
             / eigenvalues.abs().min(-1).values.clamp_min(1e-12)).mean()))
        out["log_lambda"].append(math.log(lam))
    del field
    return out


def separable_and_parametric(table):
    """``alpha(n)``, ``alpha(tau)``, ``a(tau) b(n)`` and the logit-linear form."""
    ns = [int(k) for k in table]
    matrix = np.array([table[str(n)]["alpha"] for n in ns])       # (n, levels)
    taus = np.array(table[str(ns[0])]["tau"])
    lams = np.array([table[str(n)]["lambda"] for n in ns])

    n_only = {str(n): float(matrix[index].mean()) for index, n in enumerate(ns)}
    tau_only = matrix.mean(axis=0)

    # Rank-one fit by one alternating least-squares sweep from the marginals,
    # which is exact for a rank-one matrix and a sensible projection otherwise.
    a_tau = tau_only / max(tau_only.mean(), 1e-12)
    b_n = matrix @ a_tau / max(float(a_tau @ a_tau), 1e-12)
    separable_error = float(np.sqrt(np.mean(
        (matrix - np.outer(b_n, a_tau)) ** 2)))

    # sigma(a0 + a1 log n + a2 log lambda + a3 log n log lambda)
    clipped = np.clip(matrix, 0.02, 0.98)
    logit = np.log(clipped / (1.0 - clipped))
    log_n = np.log(np.array(ns, dtype=float))[:, None] * np.ones_like(matrix)
    log_lam = np.log(lams)
    design = np.stack([np.ones_like(logit), log_n, log_lam, log_n * log_lam],
                      axis=-1).reshape(-1, 4)
    coefficients, *_ = np.linalg.lstsq(design, logit.reshape(-1), rcond=None)
    predicted = 1.0 / (1.0 + np.exp(-(design @ coefficients)))
    parametric_error = float(np.sqrt(np.mean(
        (predicted.reshape(matrix.shape) - matrix) ** 2)))

    return {
        "n_only": {"alpha": n_only,
                   "rmse": float(np.sqrt(np.mean(
                       (matrix - np.array([n_only[str(n)] for n in ns])[:, None])
                       ** 2)))},
        "tau_only": {"tau": taus.tolist(), "alpha": tau_only.tolist(),
                     "rmse": float(np.sqrt(np.mean(
                         (matrix - tau_only[None, :]) ** 2)))},
        "separable": {"a_tau": a_tau.tolist(),
                      "b_n": {str(n): float(b_n[i]) for i, n in enumerate(ns)},
                      "tau": taus.tolist(), "rmse": separable_error},
        "parametric": {"coefficients": coefficients.tolist(),
                       "rmse": parametric_error,
                       "form": "sigma(a0 + a1 log n + a2 log lam "
                               "+ a3 log n log lam)"},
        "ns": ns,
    }


def simple_schedule(forms, name, n, lam_max, lam_min):
    if name == "n_only":
        value = forms["n_only"]["alpha"][str(n)]
        return lambda tau, _v=value: _v
    if name == "tau_only":
        taus = np.array(forms["tau_only"]["tau"])
        values = np.array(forms["tau_only"]["alpha"])
        return lambda tau: float(np.clip(np.interp(tau, taus, values), 0.0, 1.0))
    if name == "separable":
        taus = np.array(forms["separable"]["tau"])
        a_tau = np.array(forms["separable"]["a_tau"])
        scale = forms["separable"]["b_n"][str(n)]
        return lambda tau: float(np.clip(
            scale * np.interp(tau, taus, a_tau), 0.0, 1.0))
    if name == "parametric":
        c = forms["parametric"]["coefficients"]
        lo, hi = math.log(lam_min), math.log(lam_max)
        log_n = math.log(n)

        def schedule(tau):
            log_lam = lo + tau * (hi - lo)
            z = c[0] + c[1] * log_n + c[2] * log_lam + c[3] * log_n * log_lam
            return 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))
        return schedule
    raise ValueError(name)


def _correlation(left, right):
    if not np.all(np.isfinite(right)) or np.std(right) < 1e-12 \
            or np.std(left) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


# ---------------------------------------------------------------------------
# Stage: fnpe_step (sections 10-12)
# ---------------------------------------------------------------------------

def fnpe_schedules():
    """The step-multiplier schedules of section 11, as ``w(tau)``."""
    out = {"fixed_1/n": None, "unscaled": "unscaled"}
    for p in (0.5, 1, 2, 4):
        out[f"power_p={p}"] = power_schedule(p)
    for k in (5, 10, 20):
        for tau0 in (0.25, 0.5, 0.75):
            out[f"sigmoid_k={k},tau0={tau0}"] = sigmoid_schedule(k, tau0)
    return out


def stage_fnpe_step():
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        field = ComposedField(oracle, "fnpe", fnpe_prior="diffused",
                              lam_max=hi, lam_min=lo)
        rows = {}
        for name, schedule in fnpe_schedules().items():
            kwargs = {"levels": 100, "steps_per_level": SCREEN_STEPS,
                      "snr": 0.2, "lam_max": hi, "lam_min": lo, "trace": True}
            if schedule is None:
                kwargs["hierarchy_scale"] = None
            elif schedule == "unscaled":
                kwargs["hierarchy_scale"] = 1.0
            else:
                kwargs["shared_schedule"] = schedule
            key = f"{name}/T=100,L={SCREEN_STEPS},snr=0.2"
            rows[key] = run(field, annealed_langevin, oracle,
                            drop_trace=False, **kwargs)
            _report_stability(n, key, rows[key])
        # The geometry-aware step, which reads its own width off the ensemble
        # rather than off a hand-designed clock.
        for variance_snr in (0.05, 0.2):
            key = f"variance,snr_v={variance_snr}/T=100,L={SCREEN_STEPS},snr=0.2"
            rows[key] = run(field, annealed_langevin, oracle, drop_trace=False,
                            levels=100, steps_per_level=SCREEN_STEPS, snr=0.2,
                            lam_max=hi, lam_min=lo, variance_step=True,
                            variance_snr=variance_snr, trace=True)
            _report_stability(n, key, rows[key])
        out[str(n)] = rows
    store("fnpe_step", out)


def _report_stability(n, key, row):
    trace = row.get("level_trace") or []
    movement = (max(abs(a["g_mean"] - b["g_mean"])
                    for a, b in zip(trace, trace[1:])) if len(trace) > 1
                else float("nan"))
    row["max_level_mean_move"] = movement
    print(f"  n={n:4d} {key:42s} W1/sd={row['w1_over_sd']:.4f} "
          f"TV={row['weight_tv']:.4f} width={row['width_ratio']:.3f} "
          f"max|x|={row.get('max_abs_state', float('nan')):.3g} "
          f"move={movement:.3g} {'DIVERGED' if row.get('diverged') else ''}")


# ---------------------------------------------------------------------------
# Stage: fnpe_ladder (section 13)
# ---------------------------------------------------------------------------

def stage_fnpe_ladder():
    """Convergence ladder for the schedules that were stable in ``fnpe_step``."""
    step = load("fnpe_step")
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        field = ComposedField(oracle, "fnpe", fnpe_prior="diffused",
                              lam_max=hi, lam_min=lo)
        available = step[str(n)]
        # Rank the stable schedules by width ratio distance from 1: the ladder
        # exists to test convergence, and a chain that never moved has nothing
        # to converge.  W1 is deliberately not the ranking key -- an
        # under-dispersed chain can score well on W1 by accident.
        stable = [(abs(row["width_ratio"] - 1.0), key)
                  for key, row in available.items()
                  if not row.get("diverged")
                  and math.isfinite(row.get("width_ratio", float("nan")))]
        stable.sort()
        # Split on the budget suffix, not on "/": the schedule named
        # "fixed_1/n" contains a slash of its own.
        chosen = [key.split("/T=")[0] for _, key in stable[:3]]
        print(f"  n={n:4d} ladder on {chosen}")
        rows = {}
        schedules = fnpe_schedules()
        for name in chosen:
            for steps in (50, 100, 200, 500):
                kwargs = {"levels": 100, "steps_per_level": steps, "snr": 0.2,
                          "lam_max": hi, "lam_min": lo, "trace": True}
                if name == "fixed_1/n":
                    kwargs["hierarchy_scale"] = None
                elif name == "unscaled":
                    kwargs["hierarchy_scale"] = 1.0
                elif name.startswith("variance"):
                    kwargs["variance_step"] = True
                    kwargs["variance_snr"] = float(name.split("=")[1])
                else:
                    kwargs["shared_schedule"] = schedules[name]
                key = f"{name}/T=100,L={steps},snr=0.2"
                rows[key] = run(field, annealed_langevin, oracle,
                                drop_trace=False, **kwargs)
                _report_stability(n, key, rows[key])
        # The fixed 1/n arm at the same budget, as the thing to beat.
        for steps in (50, 100, 200, 500):
            key = f"fixed_1/n/T=100,L={steps},snr=0.2"
            if key in rows:
                continue
            rows[key] = run(field, annealed_langevin, oracle, drop_trace=False,
                            levels=100, steps_per_level=steps, snr=0.2,
                            lam_max=hi, lam_min=lo, hierarchy_scale=None,
                            trace=True)
            _report_stability(n, key, rows[key])
        out[str(n)] = rows
    store("fnpe_ladder", out)


# ---------------------------------------------------------------------------
# Stage: cross (section 14)
# ---------------------------------------------------------------------------

def stage_cross():
    """The five-way comparison that separates bridge error from mixing error."""
    topo = load("alpha_topo")
    out = {}
    for n in N_LADDER:
        oracle, _ = oracle_for(n)
        hi, lo = lam_bounds(oracle)
        rows = {}

        for name, spec in (("TRUE-SCORE", dict(rule="true")),
                           ("ORACLE-JAC", dict(rule="jac")),
                           ("ORACLE-GAUSS", dict(rule="gauss"))):
            field = ComposedField(oracle, lam_max=hi, lam_min=lo, **spec)
            rows[name] = run(field, dpm2, oracle, timesteps=500, lam_max=hi,
                             lam_min=lo, churn=CHURN)
            show(n, name, rows[name])

        best = _best_alpha_schedule(topo[str(n)], hi, lo)
        field = ComposedField(oracle, "alpha", alpha=best["schedule"],
                              lam_max=hi, lam_min=lo)
        rows["ORACLE-TOPO-JAC"] = run(field, dpm2, oracle, timesteps=500,
                                      lam_max=hi, lam_min=lo, churn=CHURN)
        rows["ORACLE-TOPO-JAC"]["schedule"] = best["name"]
        show(n, f"ORACLE-TOPO-JAC ({best['name']})", rows["ORACLE-TOPO-JAC"])

        fnpe = ComposedField(oracle, "fnpe", fnpe_prior="diffused",
                             lam_max=hi, lam_min=lo)
        rows["FNPE-FIXED"] = run(fnpe, annealed_langevin, oracle,
                                 levels=100, steps_per_level=500, snr=0.2,
                                 lam_max=hi, lam_min=lo, hierarchy_scale=None)
        show(n, "FNPE-FIXED (1/n, L=500)", rows["FNPE-FIXED"])

        adaptive = _best_fnpe_schedule(n)
        kwargs = {"levels": 100, "steps_per_level": 500, "snr": 0.2,
                  "lam_max": hi, "lam_min": lo}
        kwargs.update(adaptive["kwargs"])
        rows["FNPE-ADAPTIVE"] = run(fnpe, annealed_langevin, oracle, **kwargs)
        rows["FNPE-ADAPTIVE"]["schedule"] = adaptive["name"]
        show(n, f"FNPE-ADAPTIVE ({adaptive['name']})", rows["FNPE-ADAPTIVE"])

        out[str(n)] = rows
    store("cross", out)


def _best_alpha_schedule(entry, lam_max, lam_min):
    """The fit with the smallest RMSE against the per-level oracle alpha.

    Chosen on *fit quality to the topology-selected alpha*, never on endpoint
    W1 -- selecting the schedule on the endpoint would smuggle the answer into
    the method.
    """
    fits = entry["fits"]
    name = min(("piecewise3", "piecewise5", "sigmoid_tau", "poly_loglambda2",
                "constant"), key=lambda key: fits[key]["rmse"])
    return {"name": name,
            "schedule": schedule_from_fit(fits[name], name, lam_max, lam_min)}


def _best_fnpe_schedule(n):
    """The stable schedule whose width ratio came closest to 1 in the ladder."""
    try:
        ladder = load("fnpe_ladder")[str(n)]
    except SystemExit:
        ladder = load("fnpe_step")[str(n)]
    candidates = [(abs(row["width_ratio"] - 1.0), key)
                  for key, row in ladder.items()
                  if not row.get("diverged") and "fixed_1/n" not in key
                  and math.isfinite(row.get("width_ratio", float("nan")))]
    if not candidates:
        return {"name": "fixed_1/n", "kwargs": {"hierarchy_scale": None}}
    candidates.sort()
    name = candidates[0][1].split("/T=")[0]
    schedules = fnpe_schedules()
    if name == "unscaled":
        return {"name": name, "kwargs": {"hierarchy_scale": 1.0}}
    if name.startswith("variance"):
        return {"name": name,
                "kwargs": {"variance_step": True,
                           "variance_snr": float(name.split("=")[1])}}
    return {"name": name, "kwargs": {"shared_schedule": schedules[name]}}


# ---------------------------------------------------------------------------

STAGES = {
    "truth": stage_truth, "topoerr": stage_topoerr,
    "alpha_const": stage_alpha_const, "alpha_topo": stage_alpha_topo,
    "topojac": stage_topojac, "alpha_simple": stage_alpha_simple,
    "topoprofile": stage_topoprofile,
    "fnpe_step": stage_fnpe_step, "fnpe_ladder": stage_fnpe_ladder,
    "cross": stage_cross,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs="+", default=["truth"],
                        choices=sorted(STAGES) + ["all"])
    parser.add_argument("--mu0", type=float, default=None,
                        help="override the mixture's mu_0 (7.5 for the "
                             "genuinely bimodal n = 128 test)")
    parser.add_argument("--n", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)
    if args.mu0 is not None:
        exp6.SETTING["mixture"] = dataclasses.replace(M.DEFAULT, mu_0=args.mu0)
        exp6.SETTING["tag"] = f"_mu{args.mu0:g}"
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
