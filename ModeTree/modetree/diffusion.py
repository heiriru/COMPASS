"""Diffusion-time composed score fields, oracle version.

ModeTree needs a *composed* score at diffusion time ``t``, not the exact
diffused tall posterior: what the sampler actually integrates is whatever the
composition rule produces from the per-row scores.  This module builds exactly
that, with the per-row scores supplied by an oracle instead of a network, so the
mode-tree machinery can be tested without any network error in the way (Oracle B
and Oracle C of the error ladder).

Coordinates follow COMPASS: the state is ``y = x / alpha(t)`` and the noise
scale is ``lambda(t)``, so every SDE is variance-exploding,
``y_t = y_0 + lambda_t eps``.  The default schedule is the COMPASS VESDE with
``sigma = 25``.

Because every row posterior of :mod:`modetree.benchmarks` is an exact Gaussian
mixture, its ``lambda``-smoothed version is the same mixture with
``S_k -> S_k + lambda^2 I``.  Row scores and row Jacobians at time ``t`` are
therefore exact and analytic, and so is the diffused Gaussian prior score.

Composition rules implemented (names match ``MultiObsSampler.correction``):

``uncorrected``    ``(1-n) s_prior,t + sum_j s_j,t``
``fnpe``           ``(1-n)(1-t) s_prior,t + sum_j s_j,t``
``gauss``          diagonal precision weighting with the pilot covariance
``gauss_jacobian`` precision weighting with ``Lambda_j(t)`` from Tweedie's
                   second-order identity, ``Sigma_t,j = lambda^2 (I + lambda^2 J_j)``,
                   with the same PSD information projection COMPASS applies
                   (``Lambda(t) = Lambda_prior + sum_j I_j``), and the same
                   global/local Schur elimination for hierarchical models.
``modetree_jac``   ``gauss_jacobian`` evaluated *per branch* -- the branch state
                   is what changes, not the algebra, so the row contribution is
                   the single reusable function :func:`row_jacobian_factor`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .composition import (
    ArrowBlocks, BatchedGaussianMixtures, _batched_row_posteriors, _rows_state,
)


class VESchedule:
    """The COMPASS VESDE schedule, ``lambda(t) = sqrt((s^{2t}-1)/(2 log s))``."""

    def __init__(self, sigma=25.0):
        self.sigma = float(sigma)

    def lam(self, t):
        s = self.sigma
        return math.sqrt((s ** (2.0 * t) - 1.0) / (2.0 * math.log(s)))

    def time_of_lambda(self, lam):
        s = self.sigma
        return math.log(1.0 + 2.0 * math.log(s) * lam ** 2) / (2.0 * math.log(s))


# ----------------------------------------------------------------------------


class DiffusedOracle:
    """Exact ``lambda``-smoothed row posteriors, prior and their Jacobians."""

    def __init__(self, model, schedule=None):
        self.model = model
        self.schedule = schedule or VESchedule()
        self.d_g, self.d_l, self.n = model.d_g, model.d_l, model.n
        self.dim_row = self.d_g + self.d_l
        self._rows = _batched_row_posteriors(model)
        self.score_calls = 0
        self.jacobian_calls = 0

    # -- rows -----------------------------------------------------------------
    def rows_at(self, lam):
        eye = torch.eye(self.dim_row, dtype=torch.float64)
        return BatchedGaussianMixtures(
            self._rows.weights, self._rows.means,
            self._rows.covs + (lam ** 2) * eye,
        )

    def row_scores(self, rows_state, lam):
        self.score_calls += 1
        return self.rows_at(lam).score(rows_state)

    def row_jacobians(self, rows_state, lam):
        self.jacobian_calls += 1
        return self.rows_at(lam).hessian(rows_state)

    # -- prior ----------------------------------------------------------------
    def prior_covariance_t(self, lam):
        return self.model.prior_cov + (lam ** 2) * torch.eye(
            self.d_g, dtype=torch.float64
        )

    def prior_score(self, g, lam):
        cov = self.prior_covariance_t(lam)
        delta = (g - self.model.prior_mean).unsqueeze(-1)
        return -torch.linalg.solve(cov, delta).squeeze(-1)

    def prior_hessian(self, g, lam):
        """``grad^2 log p_t(g)``; constant for this module's Gaussian prior."""
        g = torch.as_tensor(g, dtype=torch.float64)
        precision = torch.cholesky_inverse(
            torch.linalg.cholesky(self.prior_covariance_t(lam))
        )
        return -precision.expand(g.shape[:-1] + precision.shape)

    def prior_backward_precision(self, lam):
        """``Lambda_prior(t) = Sigma_prior^-1 + lambda^-2 I`` (COMPASS's form)."""
        eye = torch.eye(self.d_g, dtype=torch.float64)
        prior_precision = torch.cholesky_inverse(
            torch.linalg.cholesky(self.model.prior_cov)
        )
        return prior_precision + eye / lam ** 2


# ----------------------------------------------------------------------------
# The reusable per-row JAC contribution (task section L)
# ----------------------------------------------------------------------------

def row_jacobian_factor(jacobian, lam, d_g, curvature_floor=1e-3):
    """Per-row backward precision from the score Jacobian, plus its Schur blocks.

    This is the single place the ``Sigma_t,j = lambda^2 (I + lambda^2 J_j)``
    algebra lives, mirroring ``MultiObsSampler._jacobian_joint_precision`` /
    ``_schur_factors`` step for step (spectral clamp into
    ``[0, (1-floor)/lambda^2]`` included, so an indefinite learned Jacobian is
    projected exactly as COMPASS projects it).

    Returns ``(global_precision, cross_factor, diagnostics)`` where
    ``global_precision`` is the marginal (Schur-complemented) global block and
    ``cross_factor`` is ``Cov_lg P_g``, the local regression coefficient used to
    propagate the composed global score into each row's local score.  Shapes
    carry any leading batch, so a whole ``(n, B)`` grid of row-by-branch
    evaluations goes through in one call.
    """
    variance = lam ** 2
    curvature = -jacobian
    curvature = 0.5 * (curvature + curvature.mT)
    antisymmetric = float(
        (torch.linalg.matrix_norm(jacobian - jacobian.mT)
         / torch.linalg.matrix_norm(jacobian).clamp_min(1e-300)).max()
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(curvature)
    raw_min = float(eigenvalues.min())
    clipped = int(((eigenvalues < 0.0)
                   | (eigenvalues > (1.0 - curvature_floor) / variance)).sum())
    eigenvalues = eigenvalues.clamp(
        min=0.0, max=(1.0 - curvature_floor) / variance
    )
    precision = 1.0 / (variance * (1.0 - variance * eigenvalues))
    joint = (eigenvectors * precision.unsqueeze(-2)) @ eigenvectors.mT

    if joint.shape[-1] == d_g:
        return joint, None, {
            "antisymmetric_norm": antisymmetric,
            "eigenvalue_clips": clipped, "min_raw_eigenvalue": raw_min,
        }
    factor = torch.linalg.cholesky(joint)
    eye = torch.eye(joint.shape[-1], dtype=torch.float64).expand_as(joint)
    covariance = torch.cholesky_solve(eye, factor)
    global_covariance = covariance[..., :d_g, :d_g]
    global_precision = torch.cholesky_inverse(
        torch.linalg.cholesky(global_covariance)
    )
    global_precision = 0.5 * (global_precision + global_precision.mT)
    cross = covariance[..., d_g:, :d_g] @ global_precision
    return global_precision, cross, {
        "antisymmetric_norm": antisymmetric,
        "eigenvalue_clips": clipped, "min_raw_eigenvalue": raw_min,
    }


def project_information(effective, prior_precision_t):
    """COMPASS's PSD projection of ``I_j = Lambda_j - Lambda_prior``.

    Returns the per-row delta to add (or ``None`` when already admissible) and
    the diagnostics the task requires -- never a silent clip.
    """
    information = effective - prior_precision_t
    information = 0.5 * (information + information.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(information)
    negative = int((eigenvalues < 0).sum())
    diagnostics = {
        "negative_information_eigenvalues": negative,
        "min_information_eigenvalue": float(eigenvalues.min()),
    }
    if negative == 0:
        return None, diagnostics
    deficits = (-eigenvalues).clamp_min(0.0)
    delta = (eigenvectors * deficits.unsqueeze(-2)) @ eigenvectors.mT
    diagnostics["adaptation_relative"] = float(
        (torch.linalg.matrix_norm(delta)
         / torch.linalg.matrix_norm(information).clamp_min(1e-300)).max()
    )
    return delta, diagnostics


# ----------------------------------------------------------------------------
# The composed field
# ----------------------------------------------------------------------------

@dataclass
class ComposedState:
    score: torch.Tensor
    global_precision: torch.Tensor | None = None
    diagnostics: dict | None = None


class ComposedField:
    """``S_t(theta)`` for one composition rule, all rows batched."""

    RULES = ("uncorrected", "fnpe", "gauss_jacobian")

    def __init__(self, oracle, rule="gauss_jacobian", curvature_floor=1e-3,
                 project=True, jacobian_refresh=1):
        assert rule in self.RULES, rule
        self.oracle = oracle
        self.rule = rule
        self.curvature_floor = curvature_floor
        self.project = project
        self.d_g, self.d_l, self.n = oracle.d_g, oracle.d_l, oracle.n
        self.last_diagnostics = {}
        # Milestone M2: reuse the Jacobian for `jacobian_refresh` consecutive
        # evaluations.  What is cached is the *lambda-free* information
        # Lambda_j - lambda^-2 I, exactly as MultiObsSampler does: Lambda_j
        # carries an exact lambda^-2 I term and the composition subtracts
        # (n-1) copies of Lambda_prior(t) built at the current lambda, so
        # caching Lambda_j itself leaves the two lambda^-2 terms mismatched and
        # the composed precision goes indefinite for n > 1.
        self.jacobian_refresh = int(jacobian_refresh)
        self._jacobian_index = 0
        self._cached_information = None
        self._cached_cross = None
        self._cached_shape = None

    @property
    def dim(self):
        return self.d_g + self.n * self.d_l

    def _rows_state(self, state):
        return _rows_state(self, state)

    def score(self, state, t, lam=None):
        state = torch.as_tensor(state, dtype=torch.float64)
        lam = self.oracle.schedule.lam(t) if lam is None else lam
        g = state[..., : self.d_g]
        rows_state = self._rows_state(state)
        s_rows = self.oracle.row_scores(rows_state, lam)
        s_prior = self.oracle.prior_score(g, lam)
        n = self.n

        if self.rule in ("uncorrected", "fnpe"):
            coefficient = (1 - n) if self.rule == "uncorrected" else (1 - n) * (1 - t)
            global_score = coefficient * s_prior + s_rows[..., : self.d_g].sum(-2)
            precision = None
            diagnostics = {}
        else:
            inverse_variance = 1.0 / lam ** 2
            identity = torch.eye(self.d_g, dtype=torch.float64)
            reuse = (
                self.jacobian_refresh > 1
                and self._cached_information is not None
                and self._cached_shape == tuple(rows_state.shape)
                and self._jacobian_index % self.jacobian_refresh != 0
            )
            if reuse:
                effective = self._cached_information + inverse_variance * identity
                cross = self._cached_cross
                diagnostics = {"jacobian_reused": True}
            else:
                jac = self.oracle.row_jacobians(rows_state, lam)
                effective, cross, diagnostics = row_jacobian_factor(
                    jac, lam, self.d_g, self.curvature_floor
                )
                self._cached_information = effective - inverse_variance * identity
                self._cached_cross = cross
                self._cached_shape = tuple(rows_state.shape)
                diagnostics["jacobian_reused"] = False
            self._jacobian_index += 1
            prior_t = self.oracle.prior_backward_precision(lam)
            if self.project:
                delta, extra = project_information(effective, prior_t)
                diagnostics.update(extra)
                if delta is not None:
                    effective = effective + delta
            weighted = torch.einsum(
                "...nij,...nj->...ni", effective, s_rows[..., : self.d_g]
            ).sum(-2)
            precision = effective.sum(-3) + (1 - n) * prior_t
            numerator = weighted + (1 - n) * torch.einsum(
                "ij,...j->...i", prior_t, s_prior
            )
            global_score = torch.linalg.solve(
                precision, numerator.unsqueeze(-1)
            ).squeeze(-1)
            diagnostics["cross_factor"] = cross

        if not self.d_l:
            self.last_diagnostics = diagnostics
            return ComposedState(global_score, precision, diagnostics)

        local = s_rows[..., self.d_g:]
        if self.rule == "gauss_jacobian" and diagnostics.get("cross_factor") is not None:
            cross = diagnostics["cross_factor"]
            delta_g = global_score.unsqueeze(-2) - s_rows[..., : self.d_g]
            local = local + torch.einsum("...nlg,...ng->...nl", cross, delta_g)
        flat = local.reshape(state.shape[:-1] + (self.n * self.d_l,))
        self.last_diagnostics = diagnostics
        return ComposedState(
            torch.cat([global_score, flat], dim=-1), precision, diagnostics
        )

    def __call__(self, state, t):
        return self.score(state, t).score


def admissible_row_hessian(jacobian, lam, curvature_floor=1e-3,
                           lower_floor=0.0):
    """Project a row Jacobian onto the Hessians Tweedie's identity permits.

    ``Sigma_t,j = lambda^2 (I + lambda^2 J_j)`` is a conditional covariance, so
    ``eig(-J_j)`` must lie in ``[0, (1-floor)/lambda^2]`` -- the *same* band
    ``row_jacobian_factor`` (and ``MultiObsSampler._jacobian_joint_precision``
    before it) already clamps into for the precision weighting. This applies it
    to the full row Hessian, which is what the arrowhead blocks are built from.

    For an exact score the projection is the identity and nothing changes; the
    clamp only bites where the network has produced something that is not the
    Hessian of any density. Measured on the IK checkpoint, 30% of the spectrum
    is outside the band at ``lambda = 2`` and 35% at ``lambda = 0.001``.

    Returns ``(hessian, diagnostics)``; ``diagnostics`` carries the clip count
    and the raw extremes, so the projection is never silent.
    """
    curvature = -0.5 * (jacobian + jacobian.mT)
    ceiling = (1.0 - curvature_floor) / lam ** 2
    # ``lower_floor > 0`` additionally asks for a *strict* maximum, i.e.
    # ``Sigma_t <= (1 - lower_floor) lambda^2``.  Zero -- COMPASS's own choice --
    # admits a perfectly flat direction, which is the honest answer for a
    # non-identifiable row but leaves ``-C`` singular, so the strict
    # ``is_local_maximum`` test can never pass on a problem like the arm's
    # 3-parameter local block against 2 observables.
    floor = lower_floor / lam ** 2
    eigenvalues, eigenvectors = torch.linalg.eigh(curvature)
    clipped = int(((eigenvalues < floor) | (eigenvalues > ceiling)).sum())
    diagnostics = {
        "eigenvalue_clips": clipped,
        "eigenvalue_total": int(eigenvalues.numel()),
        "min_raw_eigenvalue": float(eigenvalues.min()),
        "max_raw_eigenvalue": float(eigenvalues.max()),
        "antisymmetric_norm": float(
            (torch.linalg.matrix_norm(jacobian - jacobian.mT)
             / torch.linalg.matrix_norm(jacobian).clamp_min(1e-300)).max()
        ),
    }
    eigenvalues = eigenvalues.clamp(min=floor, max=ceiling)
    projected = (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.mT
    return -projected, diagnostics
