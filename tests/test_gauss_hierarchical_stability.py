"""Numerical-validity guards for the covariance-aware GAUSS corrections.

GAUSS composes the backward kernels as p(theta_0|theta_t)^(1-n) prod_j
p(theta_0|theta_t,x_j), whose precision is

    Lambda(t) = sum_j Lambda_j(t) + (1-n) Lambda_prior(t),

and Lemma 3.1 of Linhart et al. 2024 holds only while that is positive
definite -- their Eq. 21, which requires every single-observation posterior to
be narrower than the prior. Hierarchical models break this routinely: one
observation is nearly uninformative about a population-scale parameter, so
Lambda_j ~= Lambda_prior and the composition sits on the boundary of
properness.

The fix is structural rather than numerical. Writing Lambda_j(t) =
Lambda_prior(t) + I_j(t) turns the composition into the identity

    Lambda(t) = Lambda_prior(t) + sum_j I_j(t),

so positive definiteness follows from I_j(t) >= 0 -- a property every real
posterior has, since conditioning on an observation cannot leave you less
certain about the shared parameters than the prior alone. Projecting the
*estimated* I_j onto that cone corrects an inadmissible estimate; it does not
constrain the inference.

These tests pin:

1. Lambda(t) >= Lambda_prior(t) >= 0 by construction, with no floor, clamp or
   repair involved -- including with every clamp switched off;
2. the projection is an exact no-op on admissible estimates;
3. the composition still contracts with n, i.e. validity was not bought by
   discarding the observations' information;
4. the remaining clamps (defence in depth) are wired correctly.
"""
import math
from types import SimpleNamespace

import pytest
import torch

from compass.MultiObsSampler import MultiObsSampler
from compass.SDE import VESDE


def wide_posterior_sampler(correction, n=20, scale=1.15, coupling=0.35,
                           hierarchy_size=2, local_size=1, local_prior=None):
    """A hierarchy whose single-observation posteriors are *wider* than the prior.

    This is the regime Linhart et al. Eq. 21 excludes and that partial-pooling
    models land in: with n=20 the prior is subtracted 19 times, so posteriors
    only ~15% wider than the prior already drive Lambda(t) indefinite. The
    global/local coupling is nonzero so the cross-correction is not vacuous.
    """
    # "full_gaussian" carries a globals-only covariance; the Schur modes carry
    # the joint global/local one.
    if correction in MultiObsSampler.FULL_GAUSSIAN_CORRECTIONS:
        local_size = 0
    width = hierarchy_size + local_size
    covariance = scale * torch.eye(width, dtype=torch.float64)
    covariance[:hierarchy_size, hierarchy_size:] = coupling
    covariance[hierarchy_size:, :hierarchy_size] = coupling
    assert bool(torch.linalg.eigvalsh(covariance).min() > 0)
    covariance = covariance.expand(n, -1, -1).contiguous()

    sampler = MultiObsSampler(SimpleNamespace(sde=VESDE(sigma=25.0)))
    sampler.hierarchy = list(range(hierarchy_size))
    sampler.local_latent_indices = list(range(hierarchy_size, width))
    sampler.full_gaussian_features = (
        sampler.hierarchy + sampler.local_latent_indices
    )
    sampler.covariance_features = (
        sampler.full_gaussian_features
        if correction in MultiObsSampler.SCHUR_GAUSSIAN_CORRECTIONS
        else sampler.hierarchy
    )
    sampler.correction = correction
    sampler.denoise_clamp = 5.0
    sampler.prior_mean = torch.zeros(hierarchy_size, dtype=torch.float64)
    sampler.prior_std = torch.ones(hierarchy_size, dtype=torch.float64)
    sampler.prior_covariance = torch.eye(hierarchy_size, dtype=torch.float64)
    sampler.prior_precision_matrix = torch.eye(hierarchy_size, dtype=torch.float64)
    sampler.posterior_covariance = covariance
    sampler.posterior_precision_matrix = sampler._precision_from_covariance(covariance)
    sampler.posterior_precision = None
    sampler.posterior_mean = None
    sampler.global_posterior_mean = None
    sampler.global_posterior_covariance = None
    sampler.global_posterior_precision_matrix = None
    sampler.local_prior_mean, sampler.local_prior_std = (
        sampler._resolve_local_prior(local_prior)
    )
    sampler.world_size = 1
    sampler.num_observations = n
    sampler._covariance_time_cache = {}
    sampler._reset_covariance_diagnostics()
    sampler._configure_damping(n, 1.0, 0.5, None)
    return sampler


def composed_precision(sampler, var_t, n):
    """Lambda(t) as `_compositional_score` builds it, from the returned factors."""
    effective, _ = sampler._effective_global_factors(var_t, n)
    h = len(sampler.hierarchy)
    prior_t = sampler.prior_precision_matrix.to(torch.float64) + (
        1.0 / float(var_t)
    ) * torch.eye(h, dtype=torch.float64)
    summed = (
        n * effective[0] if effective.shape[0] == 1 else effective.sum(dim=0)
    )
    return summed + (1 - n) * prior_t


@pytest.mark.parametrize("correction", ["gauss_hierarchical", "full_gaussian"])
def test_composed_precision_dominates_the_prior_at_every_diffusion_time(
    correction,
):
    """Lambda(t) >= Lambda_prior(t) > 0 for the whole schedule, by construction.

    This is the property that makes the method valid: it bounds the solve by
    the prior covariance, so no clamp or eigenvalue floor is load-bearing.
    """
    sampler = wide_posterior_sampler(correction)
    sampler.denoise_clamp = None                    # nothing may depend on it
    n, h = sampler.num_observations, len(sampler.hierarchy)
    sigmas = torch.logspace(
        math.log10(float(sampler.sde.lambda_t(torch.tensor(1.0)))),
        math.log10(float(sampler.sde.lambda_t(torch.tensor(1e-3)))),
        25, dtype=torch.float64,
    )
    for sigma in sigmas.tolist():
        var_t = sigma * sigma
        effective, _ = sampler._effective_global_factors(var_t, n)
        prior_t = sampler._prior_precision_at(1.0 / var_t, h, effective)
        composed = effective.sum(dim=0) + (1 - n) * prior_t
        assert torch.linalg.eigvalsh(composed - prior_t).min() >= -1e-9
        assert torch.linalg.eigvalsh(0.5 * (composed + composed.mT)).min() > 0
    assert sampler.covariance_diagnostics["repair_count"] == 0
    assert sampler.covariance_diagnostics["negative_information_fraction"] > 0


@pytest.mark.parametrize("correction", ["gauss_hierarchical", "full_gaussian"])
def test_composed_precision_is_positive_definite_when_posteriors_exceed_prior(
    correction,
):
    """The regime Eq. 21 excludes must still yield a usable precision."""
    sampler = wide_posterior_sampler(correction)
    var_t = float(sampler.sde.lambda_t(torch.tensor(1.0)).square())

    # Without adaptation this composition is indefinite: check the raw form.
    raw = sampler.posterior_precision_matrix.to(torch.float64) + (
        1.0 / var_t
    ) * torch.eye(
        sampler.posterior_precision_matrix.shape[-1], dtype=torch.float64
    )
    h = len(sampler.hierarchy)
    unadapted = MultiObsSampler._schur_global_precision(raw, h).sum(dim=0) + (
        1 - sampler.num_observations
    ) * (
        sampler.prior_precision_matrix.to(torch.float64)
        + (1.0 / var_t) * torch.eye(h, dtype=torch.float64)
    )
    assert torch.linalg.eigvalsh(unadapted).min() < 0, (
        "fixture no longer reproduces the indefinite regime"
    )

    adapted = composed_precision(sampler, var_t, sampler.num_observations)
    assert torch.linalg.eigvalsh(adapted).min() > 0
    assert sampler.covariance_diagnostics["adaptation_count"] == 1
    assert sampler.covariance_diagnostics["adaptation_fraction"] == 1.0


def test_composition_still_contracts_with_more_observations():
    """Validity must not come from throwing the observations' information away.

    Each subject carries genuine information here, so the composed posterior
    has to keep tightening as subjects are added; a projection that simply
    collapsed onto the prior would show a flat curve.
    """
    sampler = wide_posterior_sampler(
        "gauss_hierarchical", scale=0.45, coupling=0.15,
    )
    sampler.denoise_clamp = None
    h = len(sampler.hierarchy)
    var_t = float(sampler.sde.lambda_t(torch.tensor(1e-3)).square())

    widths = []
    for n in (1, 2, 5, 10, 20):
        sampler._covariance_time_cache = {}
        effective, _ = sampler._effective_global_factors(var_t, n)
        prior_t = sampler._prior_precision_at(1.0 / var_t, h, effective)
        composed = n * effective[0] + (1 - n) * prior_t
        clean = composed - (1.0 / var_t) * torch.eye(h, dtype=torch.float64)
        widths.append(float(torch.linalg.inv(clean).diagonal().max().sqrt()))

    assert all(a > b for a, b in zip(widths, widths[1:])), widths
    prior_width = float(sampler.prior_std.max())
    assert widths[0] < prior_width          # one observation already informs
    assert widths[-1] < 0.5 * widths[0]     # and 20 tighten it substantially


def test_adaptation_is_a_no_op_when_posteriors_are_narrower_than_the_prior():
    """Valid compositions must be left bit-for-bit alone."""
    sampler = wide_posterior_sampler(
        "gauss_hierarchical", n=4, scale=0.2, coupling=0.05,
    )
    var_t = float(sampler.sde.lambda_t(torch.tensor(0.5)).square())
    effective, cross = sampler._effective_global_factors(var_t, 4)

    h = len(sampler.hierarchy)
    raw = sampler.posterior_precision_matrix.to(torch.float64) + (
        1.0 / var_t
    ) * torch.eye(
        sampler.posterior_precision_matrix.shape[-1], dtype=torch.float64
    )
    torch.testing.assert_close(
        effective, MultiObsSampler._schur_global_precision(raw, h)
    )
    diagnostics = sampler.covariance_diagnostics
    assert diagnostics["adaptation_count"] == 0
    assert diagnostics["negative_information_fraction"] == 0.0
    assert diagnostics["minimum_information_eigenvalue"] > 0


def test_local_cross_correction_reads_the_clamped_composed_score():
    """The local kick must inherit the shared coordinates' denoise clamp."""
    sampler = wide_posterior_sampler("gauss_hierarchical")
    n = sampler.num_observations
    t = torch.tensor([[1.0]], dtype=torch.float64)
    var_t = float(sampler.sde.lambda_t(t).square())

    # A state far outside the prior box, where the clamp binds hard.
    x = torch.full((n, 1, 3), 40.0, dtype=torch.float64)
    x[:, :, sampler.hierarchy] = x[:1, :, sampler.hierarchy]
    scores = torch.full((n, 1, 3), 3.0, dtype=torch.float64)
    scores[0, 0, 0] = -900.0                    # one badly disagreeing subject

    result = sampler._compositional_score(scores.clone(), x, t)

    # Shared coordinates land inside the clamp box, by construction.
    x0_shared = x[:1, :, sampler.hierarchy] + var_t * result[:1, :, sampler.hierarchy]
    assert bool((x0_shared.abs() <= 5.0 + 1e-9).all())
    # The local coordinates must be bounded by the same clamp acting through
    # the cross-correction, not by nothing at all.
    local = result[:, :, sampler.local_latent_indices]
    shared = result[:1, :, sampler.hierarchy]
    bound = 4.0 * float(shared.abs().max() + scores.abs().max())
    assert float(local.abs().max()) < bound
    assert torch.isfinite(result).all()


def test_local_denoise_clamp_bounds_the_local_denoised_prediction():
    sampler = wide_posterior_sampler(
        "gauss_hierarchical", local_prior=([0.0], [1.0]),
    )
    n = sampler.num_observations
    t = torch.tensor([[1.0]], dtype=torch.float64)
    var_t = float(sampler.sde.lambda_t(t).square())

    x = torch.zeros((n, 1, 3), dtype=torch.float64)
    scores = torch.zeros((n, 1, 3), dtype=torch.float64)
    scores[:, :, sampler.local_latent_indices] = 1e6

    result = sampler._compositional_score(scores.clone(), x, t)
    local_indices = sampler.local_latent_indices
    x0_local = (
        x[:, :, local_indices] + var_t * result[:, :, local_indices]
    )
    assert float(x0_local.abs().max()) <= 5.0 + 1e-9


def test_local_clamp_is_inert_without_a_local_prior():
    """Corrections that never saw a local prior keep their previous behavior."""
    sampler = wide_posterior_sampler("gauss_hierarchical")
    assert sampler.local_prior_mean is None
    x = torch.zeros((sampler.num_observations, 1, 3), dtype=torch.float64)
    scores = torch.arange(
        sampler.num_observations * 3, dtype=torch.float64
    ).reshape(sampler.num_observations, 1, 3)
    clamped = sampler._clamp_local_scores(scores.clone(), x, 1.0)
    torch.testing.assert_close(clamped, scores)


def test_after_the_fact_repair_no_longer_amplifies_by_a_billion():
    """The fallback repair must stay bounded, not divide by pd_epsilon."""
    sampler = wide_posterior_sampler("full_gaussian", n=2)
    precision = torch.diag(torch.tensor([1.0, -0.25], dtype=torch.float64))
    subject_scores = torch.tensor(
        [[[1.0, 2.0]], [[3.0, -1.0]]], dtype=torch.float64
    )
    # A numerator whose exact solution deviates strongly from the score mean,
    # which is what the repair amplifies.
    numerator = torch.tensor([[[0.3, -0.4]]], dtype=torch.float64)

    solved, repaired, _, _ = sampler._solve_composed_global(
        precision, numerator, subject_scores
    )
    assert torch.linalg.eigvalsh(repaired).min() > 0
    # pd_epsilon=1e-8 would have produced |solved| ~ 1e7 here.
    assert float(solved.abs().max()) < 1e4
    assert sampler.covariance_diagnostics["repair_fraction"] == 1.0


def test_local_prior_validation_rejects_bad_shapes_and_scales():
    sampler = wide_posterior_sampler("gauss_hierarchical")
    with pytest.raises(ValueError, match="must have length 1"):
        sampler._resolve_local_prior(([0.0, 0.0], [1.0, 1.0]))
    with pytest.raises(ValueError, match="finite and positive"):
        sampler._resolve_local_prior(([0.0], [0.0]))
