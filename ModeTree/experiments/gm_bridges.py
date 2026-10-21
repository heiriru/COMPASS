"""Composition rules, samplers and metrics for the oracle bridge experiment.

Every rule below is written on the *tall* state ``theta = (g, l_1..l_n)`` and
composes on the shared coordinate only, which is exactly what
``MultiObsSampler._compose_scores`` does: the local coordinates keep their own
row score (plus, for the Jacobian rule, the cross-regression term the Schur
elimination implies).

Rules
-----
``true``
    The exact diffused tall score ``grad log p_lambda(theta | x_{1:n})``.  Not a
    composition rule -- the reference every other arm is measured against, and
    the field the ``TRUE-SCORE-DPM2`` sampler control integrates.

``naive``
    ``(1-n) s_prior,lambda + sum_i s_i,lambda``.  The finite-time
    non-commutation control: the clean factorisation is exact at
    ``lambda = 0`` and is applied at ``lambda > 0`` as if it were still exact.

``fnpe``
    The F-NPSE bridge of Geffner et al. (2023):
    ``(1-n) a(t) s_prior + sum_i s_i,lambda`` with ``a`` running from 1 at the
    data end to 0 at the noise end.  ``prior="clean"`` is the paper's literal
    ``p(theta)^{(1-n)a(t)}``; ``prior="diffused"`` is what COMPASS ships.  The
    two are *different bridges*, and section 3 of the experiment measures the
    difference rather than assuming one is a bug.

``damped``
    The error-damped bridge of Arruda et al., as implemented by BayesFlow's
    ``compositional_bridge``: ``d(t) [ (1-n)(1-t) s_prior,lambda + sum_i s_i ]``
    with ``d(t) = d_1^t`` (so ``d(0) = 1``).  BayesFlow multiplies the *whole*
    compositional score by ``d(t)``; COMPASS multiplies only the shared block.
    ``damp_locals`` selects which, defaulting to the BayesFlow convention.

``gauss``
    The GAUSS rule with oracle second moments: each row is weighted by the
    backward precision ``Sigma_i^-1 + lambda^-2 I`` built from the row
    posterior's *exact total covariance*, Schur-reduced onto the shared block.

``jac``
    The Linhart/JAC rule with oracle curvature: the per-row backward precision
    comes from the exact row Hessian through Tweedie's second-order identity,
    ``Sigma_{t,i} = lambda^2 (I + lambda^2 J_i)``, with COMPASS's own spectral
    clamp and information projection applied unchanged.  The prior precision is
    built the same way from the exact diffused *mixture* prior Hessian -- there
    is no moment-matched prior curvature anywhere in this arm.

``alpha``
    The bifurcation-aligned diagnostic field
    ``S^(alpha) = S^naive + alpha(tau) [ S^jac - S^naive ]``, with ``alpha = 0``
    recovering ``naive`` and ``alpha = 1`` recovering ``jac`` exactly.  Both
    endpoints are computed from a *single* pass of row scores and row Hessians,
    so the blend costs what ``jac`` costs.  ``alpha`` is a function of the
    log-lambda bridge coordinate only: it is a schedule, not a learned object,
    and it is chosen in the driver from bridge geometry alone.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modetree.diffusion import row_jacobian_factor  # noqa: E402


RULES = ("true", "naive", "fnpe", "damped", "gauss", "jac", "alpha")


# ---------------------------------------------------------------------------
# Bridge coordinate
# ---------------------------------------------------------------------------

def bridge_coordinate(oracle, t, kind, lam_max, lam_min):
    """``tau in [0, 1]``: 0 at the data end, 1 at the noise end.

    ``kind="time"`` uses COMPASS's own diffusion time, which already runs on
    ``[eps, 1]`` with 1 at maximum noise -- so ``a(t) = 1 - t`` is literally
    Geffner's ``a = (T - t)/T`` with ``T = 1``.  ``kind="loglambda"`` instead
    spreads ``tau`` uniformly in ``log lambda``, which is where the sampler's
    own schedule is uniform; the two differ a lot because ``lambda(t)`` is
    exponential in ``t``.
    """
    if kind == "time":
        return float(t)
    lam = float(oracle.schedule.lam(t))
    lo, hi = math.log(lam_min), math.log(lam_max)
    return float(min(max((math.log(lam) - lo) / (hi - lo), 0.0), 1.0))


# ---------------------------------------------------------------------------
# The composed field
# ---------------------------------------------------------------------------

class ComposedField:
    """``S_t(theta)`` for one rule, all rows and samples batched."""

    def __init__(self, oracle, rule, *, n_obs=None, fnpe_prior="diffused",
                 damping_d1=1.0, damp_locals=True, tau_kind="time",
                 lam_max=None, lam_min=None, curvature_floor=1e-3, project=True,
                 alpha=1.0):
        assert rule in RULES, rule
        self.oracle = oracle
        self.rule = rule
        self.n = oracle.n if n_obs is None else int(n_obs)
        self.fnpe_prior = fnpe_prior
        self.damping_d1 = float(damping_d1)
        self.damp_locals = bool(damp_locals)
        self.tau_kind = tau_kind
        self.lam_max = lam_max
        self.lam_min = lam_min
        self.curvature_floor = float(curvature_floor)
        self.project = bool(project)
        self.diagnostics = {"information_projections": 0, "eigenvalue_clips": 0}
        self._pilot = None
        # ``alpha`` may be a constant or a callable of the *log-lambda* bridge
        # coordinate.  It is stored as a callable either way so the field never
        # has to branch on its type inside the hot loop.
        self.alpha = alpha if callable(alpha) else (lambda _tau, _a=float(alpha): _a)

    def alpha_at(self, t, lam):
        """``alpha(tau)`` at the log-lambda coordinate, clamped to ``[0, 1]``."""
        assert self.lam_max is not None and self.lam_min is not None, \
            "the alpha rule needs explicit lam_max / lam_min"
        lo, hi = math.log(self.lam_min), math.log(self.lam_max)
        tau = min(max((math.log(max(lam, 1e-12)) - lo) / (hi - lo), 0.0), 1.0)
        return float(min(max(self.alpha(tau), 0.0), 1.0))

    # -- schedule-dependent coefficients ------------------------------------
    def tau(self, t):
        return bridge_coordinate(self.oracle, t, self.tau_kind,
                                 self.lam_max, self.lam_min)

    def damping(self, t):
        """``d(t) = d_0 exp(log(d_1/d_0) tau)`` with ``d_0 = 1``."""
        if self.damping_d1 >= 1.0:
            return 1.0
        return float(math.exp(math.log(self.damping_d1) * self.tau(t)))

    # -- pilot moments for GAUSS -------------------------------------------
    def _pilot_covariance(self):
        """Exact total covariance of each row posterior ``p(g, l_i | x_i)``."""
        if self._pilot is None:
            rows = self.oracle.rows
            weights = rows.log_weights.exp()                       # (n, 2)
            mean = (weights.unsqueeze(-1) * rows.means).sum(-2)     # (n, 2)
            delta = rows.means - mean.unsqueeze(-2)                 # (n, 2, 2)
            spread = (weights[..., None, None]
                      * delta.unsqueeze(-1) * delta.unsqueeze(-2)).sum(-3)
            self._pilot = rows.cov.unsqueeze(0) + spread            # (n, 2, 2)
        return self._pilot

    # -- the field ----------------------------------------------------------
    def __call__(self, state, t, lam=None):
        lam = float(self.oracle.schedule.lam(t)) if lam is None else float(lam)
        state = torch.as_tensor(state, dtype=torch.float64)
        samples = int(np.prod(state.shape[:-1])) if state.dim() > 1 else 1
        self.oracle.field_calls += 1

        if self.rule == "true":
            return self.oracle.tall.score(state, lam)

        g = state[..., 0]
        rows_state = self.oracle.rows_state(state)
        s_rows = self.oracle.row_scores(rows_state, lam, samples)
        s_global = s_rows[..., 0]                                   # (..., n)
        s_local = s_rows[..., 1]                                    # (..., n)
        n = self.n

        if self.rule in ("naive", "fnpe", "damped"):
            if self.rule == "naive":
                coefficient = 1.0 - n
                prior = self.oracle.prior.score(g, lam)
            elif self.rule == "fnpe":
                coefficient = (1.0 - n) * (1.0 - self.tau(t))
                prior = self.oracle.prior.score(
                    g, 0.0 if self.fnpe_prior == "clean" else lam
                )
            else:
                coefficient = (1.0 - n) * (1.0 - self.tau(t))
                prior = self.oracle.prior.score(g, lam)
            composed = coefficient * prior + s_global.sum(-1)
            local = s_local
            if self.rule == "damped":
                factor = self.damping(t)
                composed = factor * composed
                if self.damp_locals:
                    local = factor * local
            return self.oracle.assemble(composed, local)

        # ---- precision-weighted rules ------------------------------------
        inverse_variance = 1.0 / lam ** 2
        if self.rule == "gauss":
            pilot = self._pilot_covariance()                        # (n, 2, 2)
            joint = (torch.linalg.inv(pilot)
                     + inverse_variance * torch.eye(2, dtype=torch.float64))
            precision, cross = _schur(joint)                        # (n,), (n,)
            prior_precision = torch.full_like(
                g, 1.0 / self.oracle.mixture.global_prior_std ** 2 + inverse_variance
            )
        else:
            jac = self.oracle.row_hessians(rows_state, lam, samples)
            joint, cross_matrix, info = row_jacobian_factor(
                jac, lam, 1, self.curvature_floor
            )
            self.diagnostics["eigenvalue_clips"] += int(info["eigenvalue_clips"])
            precision = joint.reshape(s_global.shape)
            cross = cross_matrix.reshape(s_global.shape)
            prior_hessian = self.oracle.prior.hessian(g, lam)
            prior_joint, _, _ = row_jacobian_factor(
                prior_hessian.reshape(*g.shape, 1, 1), lam, 1, self.curvature_floor
            )
            prior_precision = prior_joint.reshape(g.shape)

        if self.project:
            # COMPASS projects ``Lambda_j - Lambda_prior(t)`` onto the PSD cone,
            # which is what makes ``sum_j Lambda_j + (1-n) Lambda_prior`` positive
            # by construction rather than by repair.
            deficit = (prior_precision.unsqueeze(-1) - precision).clamp_min(0.0)
            if bool((deficit > 0).any()):
                self.diagnostics["information_projections"] += int((deficit > 0).sum())
                precision = precision + deficit

        prior_score = self.oracle.prior.score(g, lam)
        total = precision.sum(-1) + (1 - n) * prior_precision
        numerator = ((precision * s_global).sum(-1)
                     + (1 - n) * prior_precision * prior_score)
        if bool((total <= 0).any()):
            self.diagnostics["nonpositive_precision"] = \
                self.diagnostics.get("nonpositive_precision", 0) + int((total <= 0).sum())
            total = total.clamp_min(torch.finfo(torch.float64).eps)
        composed = numerator / total
        local = s_local + cross * (composed.unsqueeze(-1) - s_global)
        if self.rule == "alpha":
            # Both endpoints out of one pass: ``naive`` reuses the row scores
            # and prior score already computed for ``jac``, so the blend costs
            # a handful of flops on top of the Jacobian rule rather than a
            # second field evaluation.
            weight = self.alpha_at(t, lam)
            naive_composed = (1 - n) * prior_score + s_global.sum(-1)
            composed = naive_composed + weight * (composed - naive_composed)
            local = s_local + weight * (local - s_local)
        return self.oracle.assemble(composed, local)


def _schur(joint):
    """Global precision and cross-regression block of a ``(..., 2, 2)`` precision."""
    covariance = torch.linalg.inv(joint)
    global_covariance = covariance[..., :1, :1]
    precision = 1.0 / global_covariance
    cross = covariance[..., 1:, :1] * precision
    return precision.reshape(joint.shape[:-2]), cross.reshape(joint.shape[:-2])


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def _schedule(oracle, timesteps, lam_max, lam_min):
    lams = np.logspace(np.log10(lam_max), np.log10(lam_min), int(timesteps))
    return [(float(oracle.schedule.time_of_lam(l)), float(l)) for l in lams]


def _initial(oracle, count, lam_max, generator, hierarchy_scale=1.0, init=None):
    """``N(0, lambda_max^2 I)``, which is what ``MultiObsSampler`` starts from.

    ``init="exact"`` instead draws from the exact diffused tall posterior
    ``p_{lambda_max}(theta | x)``.  Every method sees the same uninformed start
    by default; the exact start exists only so the sampler control can separate
    "the solver is wrong" from "the top of the schedule is not high enough for
    ``N(0, lambda_max^2)`` to be the right reference".
    """
    if init == "exact":
        tall = oracle.tall
        pick = torch.multinomial(tall.weights, count, replacement=True,
                                 generator=generator)
        cov = tall.cov + (lam_max ** 2) * torch.eye(oracle.dim, dtype=torch.float64)
        factor = torch.linalg.cholesky(cov)
        noise = torch.randn(count, oracle.dim, dtype=torch.float64,
                            generator=generator)
        return tall.means[pick] + noise @ factor.mT
    state = torch.randn(count, oracle.dim, dtype=torch.float64,
                        generator=generator) * lam_max
    state[:, 0] *= math.sqrt(hierarchy_scale)
    return state


def dpm2(field, count, timesteps=200, lam_max=None, lam_min=0.03167,
         generator=None, churn=0.0, denoise=False, init=None):
    """COMPASS's second-order DPM predictor, stepping in ``d lambda``.

    Mirrors ``MultiObsSampler._dpm_solver_2_step`` exactly: with
    ``h = lambda_k - lambda_{k+1}``,

        half = y + h lambda_k S(y, t_k)
        y   <- y + h/2 [ lambda_k S(y, t_k) + lambda_{k+1} S(half, t_{k+1}) ]

    ``churn > 0`` re-injects Karras noise before each step, which is COMPASS's
    shipped default; the control runs with ``churn = 0`` so the arm integrates
    the probability-flow ODE and nothing but solver error separates it from the
    exact answer.
    """
    oracle = field.oracle
    lam_max = lam_max or float(oracle.schedule.lam(1.0))
    grid = _schedule(oracle, timesteps, lam_max, lam_min)
    state = _initial(oracle, count, lam_max, generator, init=init)
    started = time.perf_counter()
    calls = 0
    for k in range(len(grid) - 1):
        t, lam = grid[k]
        t_next, lam_next = grid[k + 1]
        if churn > 0.0:
            inflated = lam * math.sqrt(1.0 + churn ** 2)
            state = state + math.sqrt(max(inflated ** 2 - lam ** 2, 0.0)) * _shared_noise(
                oracle, state.shape, generator
            )
            lam = inflated
            t = float(oracle.schedule.time_of_lam(lam))
        h = lam - lam_next
        score = field(state, t, lam)
        half = state + h * lam * score
        score_next = field(half, t_next, lam_next)
        state = state + 0.5 * h * (lam * score + lam_next * score_next)
        calls += 2
    if denoise:
        t, lam = grid[-1]
        state = state + (lam ** 2) * field(state, t, lam)
        calls += 1
    return state, {"seconds": time.perf_counter() - started, "field_calls": calls,
                   "nfe": calls}


def _shared_noise(oracle, shape, generator):
    noise = torch.randn(shape, dtype=torch.float64, generator=generator)
    return noise


def euler_maruyama(field, count, timesteps=200, lam_max=None, lam_min=0.03167,
                   generator=None, init=None):
    """The VE reverse SDE, one Euler-Maruyama step per level."""
    oracle = field.oracle
    lam_max = lam_max or float(oracle.schedule.lam(1.0))
    grid = _schedule(oracle, timesteps, lam_max, lam_min)
    state = _initial(oracle, count, lam_max, generator, init=init)
    started = time.perf_counter()
    calls = 0
    for k in range(len(grid) - 1):
        t, lam = grid[k]
        _, lam_next = grid[k + 1]
        gap = lam ** 2 - lam_next ** 2
        state = (state + gap * field(state, t, lam)
                 + math.sqrt(gap) * torch.randn(state.shape, dtype=torch.float64,
                                                generator=generator))
        calls += 1
    return state, {"seconds": time.perf_counter() - started, "field_calls": calls,
                   "nfe": calls}


def annealed_langevin(field, count, levels=100, steps_per_level=20, snr=0.1,
                      lam_max=None, lam_min=0.03167, generator=None,
                      hierarchy_scale=None, shared_schedule=None,
                      variance_step=False, variance_snr=0.1, trace=False):
    """Annealed Langevin against a bridging score (Geffner et al., Algorithm 1).

    Step size ``delta = snr lambda^2 s`` with ``s = 1/n`` on the shared
    coordinate, matching ``MultiObsSampler._corrector_step`` and
    ``_hierarchy_scale``: the F-NPSE bridge contracts by ``1/n`` there, so a
    step sized for the local coordinates is ``n`` times too large for the
    shared one.  The update is ``y <- y + delta S + sqrt(2 delta) eps``.

    Level-dependent shared step
    ---------------------------
    The ``1/n`` prescription is right where the bridge's shared width really is
    ``lambda/sqrt(n)`` -- the noise end -- and wrong at the data end, where the
    width is the posterior's own ``sqrt(V)`` and does not depend on ``n``.  Two
    ways to interpolate are offered, and both keep the *local* coordinates at
    the unscaled ``snr lambda^2``:

    ``shared_schedule``
        a callable of the log-lambda coordinate ``tau`` (1 at the noise end,
        0 at the data end) returning ``w(tau) in [0, 1]``; the shared step is
        scaled by ``(1 - w)/n + w``, so ``w = 0`` reproduces ``1/n`` and
        ``w = 1`` removes the scaling entirely.

    ``variance_step``
        geometry-aware instead of hand-designed: at the *start* of each level
        the shared step is set to ``variance_snr * Var(g)`` measured on the
        ensemble entering that level, clipped into
        ``[snr lambda^2 / n, snr lambda^2]`` so it can never be more
        aggressive than the unscaled step nor more timid than Geffner's.
        The estimate is frozen for the level's ``L`` inner steps rather than
        updated per step, which keeps the transition kernel fixed within a
        level; it is still an ensemble-adaptive scheme and is reported as a
        diagnostic, not as an exact sampler.

    ``trace`` records per-level ensemble width and mean on the shared
    coordinate, which is what separates "stable but frozen" from "mixed".
    """
    oracle = field.oracle
    lam_max = lam_max or float(oracle.schedule.lam(1.0))
    grid = _schedule(oracle, levels, lam_max, lam_min)
    base_scale = (1.0 / field.n) if hierarchy_scale is None else float(hierarchy_scale)
    # The initial ensemble's shared coordinate is drawn at the width the
    # *schedule itself* claims at the noise end, not at a fixed 1/n: otherwise
    # a schedule would be compared against the unscaled arm while starting from
    # a different ensemble, and the difference in the result would be partly a
    # difference of initial condition.  Every schedule tested has w(1) ~ 0, so
    # in practice this reproduces Geffner's lambda/sqrt(n) start.
    init_scale = base_scale
    if shared_schedule is not None and not variance_step:
        w_top = float(min(max(shared_schedule(1.0), 0.0), 1.0))
        init_scale = (1.0 - w_top) / field.n + w_top
    state = _initial(oracle, count, lam_max, generator,
                     hierarchy_scale=init_scale)
    scale_vector = torch.ones(oracle.dim, dtype=torch.float64)
    log_lo, log_hi = math.log(lam_min), math.log(lam_max)
    started = time.perf_counter()
    calls = 0
    max_state = 0.0
    max_score = 0.0
    history = []
    for t, lam in grid:
        tau = min(max((math.log(lam) - log_lo) / (log_hi - log_lo), 0.0), 1.0)
        if variance_step:
            spread = float(state[:, 0].var(unbiased=True))
            target = variance_snr * spread
            low, high = snr * lam ** 2 / field.n, snr * lam ** 2
            shared = float(min(max(target, low), high)) / (snr * lam ** 2)
        elif shared_schedule is not None:
            w = float(min(max(shared_schedule(tau), 0.0), 1.0))
            shared = (1.0 - w) / field.n + w
        else:
            shared = base_scale
        scale_vector[0] = shared
        step = snr * lam ** 2 * scale_vector
        if trace:
            history.append({"tau": tau, "lambda": lam, "shared_scale": shared,
                            "g_sd": float(state[:, 0].std(unbiased=True)),
                            "g_mean": float(state[:, 0].mean())})
        for _ in range(steps_per_level):
            score = field(state, t, lam)
            max_score = max(max_score, float(score.abs().max()))
            noise = torch.randn(state.shape, dtype=torch.float64, generator=generator)
            state = state + step * score + torch.sqrt(2.0 * step) * noise
            calls += 1
            max_state = max(max_state, float(state.abs().max()))
            # An improper bridge does not produce NaNs -- it produces finite
            # numbers that grow without bound.  Stop on magnitude, not on
            # finiteness, so an improper target is diagnosed in seconds instead
            # of being integrated for an hour.
            if (not torch.isfinite(state).all()
                    or float(state.abs().max()) > 1e6 * max(lam_max, 1.0)):
                return state, {"seconds": time.perf_counter() - started,
                               "field_calls": calls, "nfe": calls,
                               "max_abs_state": max_state,
                               "max_abs_score": max_score,
                               "level_trace": history,
                               "diverged": True}
    return state, {"seconds": time.perf_counter() - started, "field_calls": calls,
                   "nfe": calls, "max_abs_state": max_state,
                   "max_abs_score": max_score, "level_trace": history,
                   "diverged": False}


def adaptive_heun(field, count, lam_max=None, lam_min=0.03167, generator=None,
                  e_abs=0.02576, e_rel=0.1, safety=0.9, exponent=0.9,
                  max_evaluations=20_000):
    """Jolicoeur-Martineau adaptive stochastic Heun, as BayesFlow ships it.

    This is ``two_step_adaptive_step`` of ``bayesflow.utils.integrate`` -- the
    solver the Arruda construction runs on -- reproduced on the oracle field:
    an Euler-Maruyama predictor and a second Euler step from the predicted
    state *reusing the same noise*, averaged; the local error is the gap
    between the two, normalised by ``max(e_abs, e_rel max(|euler|, |last|))``.
    COMPASS's ``_adaptive_sampler`` is the same method with ``e_abs`` ten times
    tighter (0.002576); the BayesFlow default is used here and the tolerance
    ladder is run explicitly.
    """
    oracle = field.oracle
    lam_max = lam_max or float(oracle.schedule.lam(1.0))
    start_time = float(oracle.schedule.time_of_lam(lam_max))
    end_time = float(oracle.schedule.time_of_lam(lam_min))
    span = start_time - end_time
    state = _initial(oracle, count, lam_max, generator)
    last = state.clone()
    step = span / 50.0
    min_step = max(span * 1e-12, float(torch.finfo(torch.float64).eps))
    current = start_time
    accepted = rejected = evaluations = 0
    started = time.perf_counter()
    while current > end_time and evaluations + 2 <= max_evaluations:
        step = min(step, current - end_time)
        nxt = current - step
        lam, lam_next = (float(oracle.schedule.lam(current)),
                         float(oracle.schedule.lam(nxt)))
        drop = max(lam ** 2 - lam_next ** 2, 0.0)
        noise = torch.randn(state.shape, dtype=torch.float64,
                            generator=generator) * math.sqrt(drop)
        euler = state + drop * field(state, current, lam) + noise
        second = state + drop * field(euler, nxt, lam_next) + noise
        heun = 0.5 * (euler + second)
        evaluations += 2
        delta = torch.maximum(
            torch.as_tensor(e_abs, dtype=torch.float64),
            e_rel * torch.maximum(euler.abs(), last.abs()),
        )
        error = float(((heun - euler).abs() / (delta + 1e-10)).max())
        if not torch.isfinite(heun).all():
            raise RuntimeError("adaptive Heun produced a non-finite proposal")
        if error <= 1.0 or step <= min_step:
            state, last, current = heun, euler, nxt
            accepted += 1
        else:
            rejected += 1
        step = max(min_step, step * safety * max(error, 1e-10) ** (-exponent))
    return state, {"seconds": time.perf_counter() - started,
                   "field_calls": evaluations, "nfe": evaluations,
                   "accepted_steps": accepted, "rejected_steps": rejected,
                   "reached_end": current <= end_time + 1e-9}


# ---------------------------------------------------------------------------
# Shared-step schedules for the F-NPSE Langevin arm
# ---------------------------------------------------------------------------

def power_schedule(p):
    """``w(tau) = (1 - tau)^p``: 0 at the noise end, 1 at the data end."""
    return lambda tau, _p=float(p): (1.0 - tau) ** _p


def sigmoid_schedule(k, tau0):
    """``w(tau) = sigma(k (tau_0 - tau))`` -- a soft switch at ``tau_0``."""
    def schedule(tau, _k=float(k), _t0=float(tau0)):
        z = _k * (_t0 - tau)
        return 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))
    return schedule
