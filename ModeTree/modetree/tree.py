"""Milestones H-K and N -- mode continuation, branch spawning, branch management.

The mode tree follows the roots of the *composed* score field down the diffusion
schedule.  A mode trajectory satisfies ``S_t(m_b(t)) = 0``; differentiating,

    dm_b/dt = -H_b(t)^-1 d_t S_t

which is implemented as predictor-corrector continuation.  The predictor is
either the trivial one (carry the mode over) or the tangent above with
``d_t S_t`` from a central difference in ``lambda``; which one is worth its cost
is measured, not assumed (``ModeTreeConfig.predictor``).

New branches appear two ways:

* **curvature spawning** (I): when the softest eigenvalue of a branch's global
  precision collapses toward zero the mode is about to bifurcate, and probes are
  launched along that eigenvector at a scale set by the local posterior standard
  deviation ``sqrt(v^T P^-1 v)``;
* **sentinel probes** (J): at a few checkpoints, probes along the top few
  *uncertainty* directions plus a small deterministic Sobol set over the prior
  region.  Curvature alone cannot find a mode that never was part of this
  branch, which is the ``remote_small`` benchmark.

Below ``switch_to_clean_t`` the tree stops using the diffusion-time composition
and refines on the exact clean composition (N), which is where the final MAP,
curvature and energies come from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import torch

from .composition import (
    ArrowBlocks, CleanComposer, _rows_state, arrow_blocks_batched,
)
from .diffusion import admissible_row_hessian
from .refine import (
    mahalanobis, mahalanobis_residual, refine_batch, refine_mode,
)


# ----------------------------------------------------------------------------
# A composer view of the composed field at one diffusion time
# ----------------------------------------------------------------------------

class TimeComposer(CleanComposer):
    """``CleanComposer`` interface for ``S_t`` at a fixed diffusion time.

    The curvature used for the Newton step is the composed *row-Jacobian*
    arrowhead, exactly as ``MultiObsSampler.newton_map_estimate(curvature=
    "jacobian")`` builds it: it is the exact Jacobian of the ``uncorrected``
    composed field and an SPD-projectable quasi-Newton matrix for the
    precision-weighted ones, and the two coincide as ``lambda -> 0``.
    """

    def __init__(self, field, t, curvature_projection=False, lower_floor=0.0):
        self.field = field
        self.t = float(t)
        self.lam = field.oracle.schedule.lam(t)
        self.d_g, self.d_l, self.n = field.d_g, field.d_l, field.n
        self.source = None
        self.score_evaluations = 0
        self.hessian_evaluations = 0
        # Off by default: with an exact score the row Jacobian *is* a Hessian
        # and projecting it is a no-op, so the default keeps the composition
        # exactly as derived.  Switched on, the arrowhead blocks are built from
        # the Tweedie-admissible projection of the row Jacobian -- the same band
        # ``row_jacobian_factor`` already clamps into for the precision
        # weighting, now applied to the curvature the Newton step and the
        # Laplace determinant consume.
        self.curvature_projection = bool(curvature_projection)
        self.lower_floor = float(lower_floor)
        self.projection_diagnostics = {}

    def _project(self, H):
        if not self.curvature_projection:
            return H
        H, diagnostics = admissible_row_hessian(
            H, self.lam, self.field.curvature_floor, self.lower_floor
        )
        self.projection_diagnostics = diagnostics
        return H

    def score(self, state):
        self.score_evaluations += self.n
        return self.field.score(state, self.t, self.lam).score

    def global_curvature(self, states):
        """Batched global curvature block for all-global states."""
        assert not self.d_l
        states = torch.as_tensor(states, dtype=torch.float64)
        H = self._project(
            self.field.oracle.row_jacobians(_rows_state(self, states), self.lam))
        self.hessian_evaluations += self.n
        H_prior = self.field.oracle.prior_hessian(states, self.lam)
        A = H_prior + (H - H_prior.unsqueeze(-3)).sum(-3)
        return 0.5 * (A + A.mT)

    def blocks_batched(self, states):
        """Arrowhead blocks for a batch of full states ``(..., dim)``."""
        states = torch.as_tensor(states, dtype=torch.float64)
        H = self._project(
            self.field.oracle.row_jacobians(_rows_state(self, states), self.lam))
        self.hessian_evaluations += self.n
        H_prior = self.field.oracle.prior_hessian(
            states[..., : self.d_g], self.lam
        )
        return arrow_blocks_batched(H, H_prior, self.d_g)

    def blocks(self, state):
        state = torch.as_tensor(state, dtype=torch.float64).reshape(-1)
        rows_state = _rows_state(self, state)
        H = self._project(self.field.oracle.row_jacobians(rows_state, self.lam))
        self.hessian_evaluations += self.n
        H_prior = self.field.oracle.prior_hessian(state[: self.d_g], self.lam)
        A = H_prior + (H[:, : self.d_g, : self.d_g] - H_prior).sum(0)
        A = 0.5 * (A + A.mT)
        if not self.d_l:
            return ArrowBlocks(A=A, B=None, C=None)
        B = H[:, : self.d_g, self.d_g:]
        C = H[:, self.d_g:, self.d_g:]
        return ArrowBlocks(A=A, B=B, C=0.5 * (C + C.mT))


# ----------------------------------------------------------------------------
# Branch bookkeeping (Milestone K)
# ----------------------------------------------------------------------------

@dataclass
class ModeBranch:
    branch_id: int
    parent_id: int | None
    birth_time: float
    global_state: torch.Tensor
    local_states: torch.Tensor | None = None
    score_norm: float = float("nan")
    global_precision: torch.Tensor | None = None
    global_covariance: torch.Tensor | None = None
    log_height: float | None = None
    log_mass: float | None = None
    weight: float | None = None
    active: bool = True
    origin: str = "root"
    diagnostics: dict = field(default_factory=dict)

    @property
    def state(self):
        if self.local_states is None:
            return self.global_state
        return torch.cat([self.global_state, self.local_states.reshape(-1)])


@dataclass
class ModeTreeConfig:
    enabled: bool = True
    max_branches: int = 16
    min_mode_weight: float = 1e-5
    curvature_spawn: bool = True
    curvature_threshold: float = 0.25      # relative softest eigenvalue
    spawn_scale: float = 1.0
    spawn_cooldown: int = 3
    sentinel_enabled: bool = True
    sentinel_checkpoints: int = 3
    sentinel_top_eigendirs: int = 3
    sentinel_sobol: int = 8
    sentinel_scale: tuple = (1.0, 2.5)
    # One last sentinel sweep against the *clean* composed field, after the
    # switch.  A mode of width w only appears in the lambda-smoothed field below
    # lambda ~ w, so a narrow mode can be invisible at every diffusion-time
    # checkpoint and still be a genuine maximum of the target: measured, the
    # 2-D narrow_broad benchmark loses its MAP branch (36% of the mass) at
    # n >= 64 with diffusion-time sentinels alone.  Probing the clean field
    # directly costs one batched refinement and has no such blind spot.
    final_clean_sentinels: bool = True
    switch_to_clean_t: float = 0.02
    levels: int = 32
    lambda_max: float | None = None
    corrector_iterations: int = 12
    probe_iterations: int = 60
    # Budget for the final refinement on the clean composition.  With an
    # oracle score this can be generous; with a network the field is not
    # exactly conservative and the iteration stalls long before it, so a
    # large value buys nothing but wall clock.
    clean_refine_iterations: int = 200
    # Probe acceptance, in local posterior standard deviations.  The probe
    # stage only has to identify a basin: whatever it accepts is polished to
    # ||S|| < 1e-11 by the clean-composition stage (Milestone N).  Demanding
    # solver-grade accuracy here instead throws away genuine modes.
    probe_residual_tolerance: float = 0.05
    max_candidates_per_level: int = 64
    predictor: str = "carry"               # "carry" | "tangent"
    # Mahalanobis separation below which two roots are the same mode.  At
    # n = 100 the local standard deviation is ~0.04, so 0.75 sigma calls two
    # points 0.03 apart different modes; separated modes are tens of sigma
    # apart, so a generous threshold costs nothing and prevents a basin from
    # being split into a crowd of near-duplicates.
    merge_tolerance: float = 3.0
    energy_quadrature_points: int = 16
    energy_adaptive: bool = True
    energy_multiple_paths: bool = True
    cycle_consistency: bool = True
    laplace_plus: bool = False
    laplace_plus_samples: int = 32
    evidence_auditor: bool = True
    auditor_sigma_factor: float = 3.0
    jacobian_stride: int = 1
    # Build the arrowhead blocks from the Tweedie-admissible projection of the
    # row Jacobian (see ``diffusion.admissible_row_hessian``).  A no-op for an
    # exact score, so the default is off and the derived composition stands;
    # switched on, it is the only thing that lets a learned Jacobian whose
    # spectrum leaves the admissible band be used as a curvature at all.
    curvature_projection: bool = False
    curvature_lower_floor: float = 0.0
    return_diagnostics: bool = False
    verbose: bool = False


# ----------------------------------------------------------------------------


class ModeTree:
    """Continuation of the composed-score roots down the diffusion schedule."""

    def __init__(self, field, clean_composer, config=None, prior_std=None):
        self.field = field
        self.clean = clean_composer
        self.config = config or ModeTreeConfig()
        self.d_g, self.d_l, self.n = field.d_g, field.d_l, field.n
        self.prior_std = (
            torch.ones(self.d_g, dtype=torch.float64) if prior_std is None
            else torch.as_tensor(prior_std, dtype=torch.float64).reshape(-1)
        )
        self._next_id = 0
        self.events = []
        self.counters = {
            "score_evaluations": 0, "jacobian_evaluations": 0,
            "newton_iterations": 0, "spawn_probes": 0, "sentinel_probes": 0,
            "merges": 0, "prunes": 0,
        }
        self.trace = []

    # -- schedule -------------------------------------------------------------
    def schedule(self):
        cfg = self.config
        schedule = self.field.oracle.schedule
        lam_max = cfg.lambda_max or float(self.prior_std.max()) * 4.0
        lam_switch = schedule.lam(cfg.switch_to_clean_t)
        lam_switch = min(lam_switch, lam_max * 0.5)
        lams = torch.logspace(
            math.log10(lam_max), math.log10(lam_switch), cfg.levels,
            dtype=torch.float64,
        )
        return [(schedule.time_of_lambda(float(l)), float(l)) for l in lams]

    # -- branch helpers -------------------------------------------------------
    def _new_branch(self, state, t, parent, origin):
        branch = ModeBranch(
            branch_id=self._next_id, parent_id=parent, birth_time=t,
            global_state=state[: self.d_g].clone(),
            local_states=(None if not self.d_l
                          else state[self.d_g:].reshape(self.n, self.d_l).clone()),
            origin=origin,
        )
        self._next_id += 1
        self.events.append(
            {"event": "birth", "branch": branch.branch_id, "t": t,
             "parent": parent, "origin": origin,
             "position": branch.global_state.tolist()}
        )
        return branch

    def _refine(self, composer, state, max_iter, tol=1e-9, radius=None):
        result = refine_mode(
            composer, state, tol=tol, max_iter=max_iter, require_mode=False,
            radius=radius,
        )
        self.counters["newton_iterations"] += result.iterations
        return result

    @staticmethod
    def _is_mode(precision):
        return precision is not None and bool(
            (torch.linalg.eigvalsh(0.5 * (precision + precision.mT)) > 0).all()
        )

    def _dedupe(self, branches, composer):
        kept = []
        for branch in branches:
            duplicate = None
            for other in kept:
                P = other.global_precision
                if P is None:
                    continue
                # Never merge into a saddle.  A saddle has no basin, so
                # "duplicate of the saddle" is not a meaningful statement, and
                # with a generous Mahalanobis tolerance it swallows the very
                # branches the saddle just spawned: at n = 1 the two modes sit
                # only 1.8 local standard deviations from the origin, so the
                # discovery returned zero branches.
                if not self._is_mode(P):
                    continue
                if mahalanobis(branch.global_state, other.global_state, P) \
                        < self.config.merge_tolerance:
                    duplicate = other
                    break
            if duplicate is None:
                kept.append(branch)
            else:
                self.counters["merges"] += 1
                self.events.append({
                    "event": "merge", "branch": branch.branch_id,
                    "into": duplicate.branch_id,
                    "t": branch.birth_time,
                })
        return kept

    # -- Milestone I ----------------------------------------------------------
    def _curvature_probes(self, branch, level):
        """Probe along the softest curvature direction of ``P_b``.

        ``P_b`` is the *negative* Hessian, so a mode has ``P_b > 0`` and the
        bifurcation signal is its smallest eigenvalue falling toward -- and then
        through -- zero.  A branch sitting exactly on a saddle has a negative
        eigenvalue there, and that is the case that most needs probing, so the
        trigger is ``lambda_min <= threshold * max|lambda|`` and explicitly
        includes negative values.  (Requiring ``lambda_min > 0`` silently
        disables the whole mechanism: measured, it leaves the symmetric
        benchmark with one branch pinned on the saddle at the origin.)

        The probe scale is the local posterior standard deviation along the
        soft direction where that is defined, capped at four prior standard
        deviations so a nearly-flat or inverted direction cannot throw a probe
        out of the plausible region.
        """
        P = branch.global_precision
        if P is None:
            return []
        last = branch.diagnostics.get("last_spawn_level")
        if last is not None and level - last < self.config.spawn_cooldown:
            return []
        eigenvalues, eigenvectors = torch.linalg.eigh(P)
        softest = float(eigenvalues[0])
        scale_reference = float(eigenvalues.abs().max())
        branch.diagnostics.setdefault("curvature_spectrum", []).append(
            (level, eigenvalues.tolist())
        )
        if softest > self.config.curvature_threshold * max(scale_reference, 1e-300):
            return []
        v = eigenvectors[:, 0]
        branch.diagnostics["last_spawn_level"] = level
        branch.diagnostics["is_saddle"] = softest <= 0.0
        probes = []
        for scale in self._probe_scales(branch, v):
            step = self.config.spawn_scale * scale * v
            probes.append(branch.global_state + step)
            probes.append(branch.global_state - step)
        self.counters["spawn_probes"] += len(probes)
        return probes

    def _probe_scales(self, branch, direction):
        """Distances to probe along ``direction``.

        The obvious choice -- the local posterior standard deviation
        ``1/sqrt(|lambda_min|)`` -- is *wrong* at a bifurcation and measurably
        so: once the branch sits on a saddle, ``|lambda_min|`` grows as the
        split deepens, so the probe distance *shrinks* exactly when the new
        modes are moving further away, every probe falls back into the saddle
        and no branch is ever created (measured: zero branches on the symmetric
        benchmark at n = 100).  The distance at which a pitchfork's new roots
        appear is set by the *pre-bifurcation* width of the parent mode, so the
        reference scale is the last width the branch had while it was still a
        genuine maximum, and a prior-scale ladder is carried alongside it as
        insurance for saddle-node births that no local curvature announces.
        """
        prior_scale = float((self.prior_std * direction.abs()).sum())
        reference = branch.diagnostics.get("last_positive_sigma")
        scales = set()
        if reference:
            scales.update(c * reference for c in (0.5, 1.0, 2.0, 4.0))
        scales.update(c * prior_scale for c in (0.25, 0.5, 1.0))
        cap = 4.0 * prior_scale
        return sorted(x for x in scales if 0 < x <= cap)

    # -- Milestone J ----------------------------------------------------------
    def _sentinel_probes(self, branch, sobol=True):
        probes = []
        P = branch.global_precision
        if P is not None:
            eigenvalues, eigenvectors = torch.linalg.eigh(P)
            order = torch.argsort(eigenvalues)           # softest = widest
            for j in order[: self.config.sentinel_top_eigendirs]:
                v = eigenvectors[:, j]
                for scale in self._probe_scales(branch, v):
                    probes.append(branch.global_state + scale * v)
                    probes.append(branch.global_state - scale * v)
        if sobol and self.config.sentinel_sobol:
            engine = torch.quasirandom.SobolEngine(
                dimension=self.d_g, scramble=True, seed=17
            )
            u = engine.draw(self.config.sentinel_sobol).to(torch.float64)
            span = 4.0 * self.prior_std
            probes.extend(list((u * 2.0 - 1.0) * span))
        self.counters["sentinel_probes"] += len(probes)
        return probes

    def _full_state(self, branch, template):
        state = template.clone()
        state[: self.d_g] = branch.global_state
        if self.d_l and branch.local_states is not None:
            state[self.d_g:] = branch.local_states.reshape(-1)
        return state

    # -- the continuation -----------------------------------------------------
    def run(self):
        cfg = self.config
        levels = self.schedule()
        template = torch.zeros(self.d_g + self.n * self.d_l, dtype=torch.float64)
        started = time.perf_counter()

        # root branch: the mode of the heavily smoothed field is unique and sits
        # essentially at the prior mean
        composer = TimeComposer(self.field, levels[0][0],
                                cfg.curvature_projection,
                                cfg.curvature_lower_floor)
        root_state = template.clone()
        result = self._refine(composer, root_state, cfg.corrector_iterations * 3)
        root = self._new_branch(result.theta, levels[0][0], None, "root")
        root.global_precision = result.blocks.global_precision()
        root.score_norm = result.score_norm
        branches = [root]

        checkpoints = set()
        if cfg.sentinel_enabled and cfg.sentinel_checkpoints:
            step = max(1, len(levels) // (cfg.sentinel_checkpoints + 1))
            checkpoints = {min(len(levels) - 1, (k + 1) * step)
                           for k in range(cfg.sentinel_checkpoints)}

        for index, (t, lam) in enumerate(levels):
            composer = TimeComposer(self.field, t,
                                    cfg.curvature_projection,
                                    cfg.curvature_lower_floor)
            previous = {b.branch_id: b.global_state.clone() for b in branches}
            # ---- predictor + corrector
            for branch in branches:
                state = self._full_state(branch, template)
                if cfg.predictor == "tangent" and index > 0:
                    state = self._tangent_predictor(branch, composer, state, levels, index)
                result = self._refine(composer, state, cfg.corrector_iterations)
                branch.global_state = result.theta[: self.d_g]
                if self.d_l:
                    branch.local_states = result.theta[self.d_g:].reshape(
                        self.n, self.d_l
                    )
                branch.score_norm = result.score_norm
                try:
                    branch.global_precision = result.blocks.global_precision()
                    eigenvalues = torch.linalg.eigvalsh(branch.global_precision)
                    if float(eigenvalues.min()) > 0:
                        branch.diagnostics["last_positive_sigma"] = float(
                            1.0 / eigenvalues.min().sqrt()
                        )
                except RuntimeError:
                    branch.global_precision = None

            # ---- spawning
            candidates = []
            if cfg.curvature_spawn:
                for branch in list(branches):
                    for probe in self._curvature_probes(branch, index):
                        candidates.append((probe, branch.branch_id, "curvature"))
            if cfg.sentinel_enabled and index in checkpoints:
                for branch in list(branches):
                    sobol = branch is branches[0]   # one global Sobol set only
                    for probe in self._sentinel_probes(branch, sobol=sobol):
                        candidates.append((probe, branch.branch_id, "sentinel"))
            candidates = candidates[: cfg.max_candidates_per_level]

            spawned_from = set()
            if candidates:
                accepted = self._refine_candidates(composer, candidates, t)
                for probe_result, parent, origin in accepted:
                    if len(branches) >= cfg.max_branches:
                        break
                    marker = len(self.events)
                    candidate = self._new_branch(probe_result[0], t, parent, origin)
                    candidate.global_precision = probe_result[1]
                    candidate.score_norm = probe_result[2]
                    merged = self._dedupe(branches + [candidate], composer)
                    if len(merged) > len(branches):
                        branches = merged
                        spawned_from.add(parent)
                    else:
                        # roll the rejected candidate back completely: _dedupe
                        # may have appended merge events after the birth, so
                        # truncating to the marker is the only correct undo
                        # (popping one event left orphaned births in the log).
                        self._next_id -= 1
                        del self.events[marker:]

            # a branch that sat on a saddle and has now produced a genuine
            # child is no longer a mode candidate; drop it rather than carry a
            # saddle all the way to the clean stage.
            survivors = []
            for branch in branches:
                if (branch.branch_id in spawned_from
                        and branch.diagnostics.get("is_saddle")):
                    self.counters["prunes"] += 1
                    self.events.append({
                        "event": "prune", "branch": branch.branch_id, "t": t,
                        "reason": "saddle that has bifurcated",
                    })
                    continue
                survivors.append(branch)
            branches = survivors or branches

            self.trace.append({
                "t": t, "lambda": lam,
                "branches": [
                    {"id": b.branch_id, "g": b.global_state.tolist(),
                     "score_norm": b.score_norm}
                    for b in branches
                ],
            })
            if cfg.verbose:
                print(f"  t={t:.4f} lam={lam:8.3f}  B={len(branches)}  "
                      + " ".join(f"{float(b.global_state[0]):+.3f}" for b in branches))

        # ---- Milestone N: finish on the exact clean composition
        final = []
        for branch in branches:
            state = self._full_state(branch, template)
            result = refine_mode(self.clean, state, tol=1e-11,
                                 max_iter=cfg.clean_refine_iterations)
            self.counters["newton_iterations"] += result.iterations
            if not result.diagnostics["is_local_maximum"]:
                self.counters["prunes"] += 1
                self.events.append({
                    "event": "prune", "branch": branch.branch_id,
                    "reason": "not a local maximum of the clean field",
                })
                continue
            branch.global_state = result.theta[: self.d_g]
            if self.d_l:
                branch.local_states = result.theta[self.d_g:].reshape(self.n, self.d_l)
            branch.global_precision = result.blocks.global_precision()
            branch.global_covariance = torch.cholesky_inverse(
                torch.linalg.cholesky(branch.global_precision)
            )
            branch.score_norm = result.score_norm
            branch.diagnostics["clean_result"] = result
            eigenvalues = torch.linalg.eigvalsh(branch.global_precision)
            branch.diagnostics["last_positive_sigma"] = float(
                1.0 / eigenvalues.min().clamp_min(1e-300).sqrt()
            )
            final.append(branch)
        final = self._dedupe(final, self.clean)
        if cfg.sentinel_enabled and cfg.final_clean_sentinels and final:
            final = self._final_clean_sweep(final, template)

        self.counters["score_evaluations"] = self.field.oracle.score_calls
        self.counters["jacobian_evaluations"] = self.field.oracle.jacobian_calls
        self.counters["wall_clock"] = time.perf_counter() - started
        return final

    def _refine_candidates(self, composer, candidates, t):
        """Refine every probe of this level, batched where the state allows it.

        All-global states go through :func:`refine_batch` in a single pass, which
        is what keeps mode discovery from costing more than the sampling it is
        meant to replace; hierarchical states fall back to the per-probe loop.
        """
        cfg = self.config
        out = []
        if not self.d_l:
            seeds = torch.stack([p.reshape(-1) for p, _, _ in candidates])
            radius = max(1.0, float(seeds.abs().max()) / 4.0)
            theta, norms, precision, residual = refine_batch(
                composer, seeds, tol=1e-10, max_iter=cfg.probe_iterations,
                radius=radius, step_tol=cfg.probe_residual_tolerance,
            )
            self.counters["newton_iterations"] += cfg.probe_iterations
            for k, (_, parent, origin) in enumerate(candidates):
                if float(residual[k]) > cfg.probe_residual_tolerance:
                    continue
                # ``refine_batch`` already returns the Schur-complemented
                # *precision* P_g, positive definite exactly at a mode -- it is
                # not the curvature block, and negating it here inverted the
                # acceptance test: every genuine mode was rejected and every
                # saddle admitted.  The hierarchical path below reads
                # ``blocks.global_precision()`` directly, with the same sign.
                P = precision[k]
                P = 0.5 * (P + P.mT)
                if not bool((torch.linalg.eigvalsh(P) > 0).all()):
                    continue
                out.append(((theta[k], P, float(norms[k])), parent, origin))
            return self._collapse_candidates(out)
        template = torch.zeros(self.d_g + self.n * self.d_l, dtype=torch.float64)
        for probe, parent, origin in candidates:
            state = template.clone()
            state[: self.d_g] = probe
            result = self._refine(
                composer, state, cfg.probe_iterations,
                radius=max(1.0, float(probe.abs().max()) / 4.0),
            )
            try:
                P = result.blocks.global_precision()
            except RuntimeError:
                continue
            residual = mahalanobis_residual(
                composer.score(result.theta)[: self.d_g], -P
            )
            if float(residual) > cfg.probe_residual_tolerance:
                continue
            if not bool((torch.linalg.eigvalsh(P) > 0).all()):
                continue
            out.append(((result.theta, P, result.score_norm), parent, origin))
        return self._collapse_candidates(out)

    def _final_clean_sweep(self, branches, template):
        """Sentinel probes against the clean field, after the switch (N + J).

        See ``ModeTreeConfig.final_clean_sentinels``: the diffusion-time
        checkpoints are blind to any mode narrower than the noise scale they sit
        at, so the last word belongs to the target itself.
        """
        probes = []
        for branch in list(branches):
            for probe in self._sentinel_probes(
                    branch, sobol=branch is branches[0]):
                probes.append((probe, branch.branch_id, "sentinel_clean"))
        if not probes:
            return branches
        probes = probes[: self.config.max_candidates_per_level]
        accepted = self._refine_candidates(self.clean, probes, 0.0)
        for probe_result, parent, origin in accepted:
            if len(branches) >= self.config.max_branches:
                break
            theta, P, norm = probe_result
            marker = len(self.events)
            candidate = self._new_branch(theta, 0.0, parent, origin)
            candidate.global_precision = P
            candidate.global_covariance = torch.cholesky_inverse(
                torch.linalg.cholesky(P)
            )
            candidate.score_norm = norm
            candidate.diagnostics["clean_result"] = refine_mode(
                self.clean, theta, tol=1e-11,
                max_iter=self.config.clean_refine_iterations,
            )
            if not candidate.diagnostics["clean_result"].diagnostics[
                    "is_local_maximum"]:
                self._next_id -= 1
                del self.events[marker:]
                continue
            merged = self._dedupe(branches + [candidate], self.clean)
            if len(merged) > len(branches):
                branches = merged
            else:
                self._next_id -= 1
                del self.events[marker:]
        return branches

    def _collapse_candidates(self, accepted):
        """Collapse probe results that landed in the same basin.

        Dozens of probes are launched per level and most of them converge to the
        *same* root.  Letting them reach the branch list one at a time fills
        ``max_branches`` with near-duplicates of the branch that already exists,
        and the genuinely new mode then never gets a slot -- which is exactly how
        the dominant mode of ``map_vs_mass`` was lost at n = 100 while its probe
        was converging correctly.  Collapsing here first is also what keeps the
        probe count from growing with the branch count.
        """
        kept = []
        for result in accepted:
            theta, P, _ = result[0]
            duplicate = False
            for other in kept:
                other_theta, other_P, _ = other[0]
                if not (self._is_mode(P) and self._is_mode(other_P)):
                    continue
                if min(mahalanobis(theta[: self.d_g], other_theta[: self.d_g], P),
                       mahalanobis(theta[: self.d_g], other_theta[: self.d_g],
                                   other_P)) < self.config.merge_tolerance:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(result)
        return kept

    # -- tangent predictor (benchmarked against "carry") ----------------------
    def _tangent_predictor(self, branch, composer, state, levels, index):
        """``m += dt * (-H^-1 d_t S_t)`` with ``d_t S`` by a central difference."""
        t_now = levels[index][0]
        t_prev = levels[index - 1][0]
        h = 0.25 * (t_prev - t_now)
        if h <= 0:
            return state
        plus = self.field.score(state, t_now + h).score
        minus = self.field.score(state, max(t_now - h, 1e-6)).score
        dS = (plus - minus) / (2.0 * h)
        blocks = composer.blocks(state)
        try:
            if self.d_l:
                # only the global block is continued; locals follow in the corrector
                P = blocks.global_precision()
                delta = torch.linalg.solve(P, dS[: self.d_g].unsqueeze(-1)).squeeze(-1)
                state = state.clone()
                state[: self.d_g] = state[: self.d_g] + (t_now - t_prev) * delta
            else:
                P = blocks.global_precision()
                delta = torch.linalg.solve(P, dS.unsqueeze(-1)).squeeze(-1)
                state = state + (t_now - t_prev) * delta
        except RuntimeError:
            pass
        return state
