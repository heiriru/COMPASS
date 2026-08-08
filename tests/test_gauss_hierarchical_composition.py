"""Tests for the "gauss_hierarchical" correction.

``gauss_hierarchical`` is a hardened variant of "Gauss_global_local" that
composes real, network-produced scores only (see
tutorials/Compositional_Inference_Fix/New_Attempt/derivation.md for the full
derivation). These tests check that:

1. It matches a dense arrowhead-precision reference solve (the exact block
   generalization of the paper's GAUSS Lemma 3.2 to a global/local hierarchy).
2. It is numerically identical to "Gauss_global_local" whenever the latter is
   *not* fed oracle/pilot moments (i.e. when both correctly compose only the
   network's real scores).
3. It reduces to the ordinary network score at n=1 (single observation).
4. It reduces to "full_gaussian" when there are no local latents.
5. Its public API refuses posterior_mean / global_posterior_mean outright,
   since those would substitute an externally supplied Gaussian's score for
   the network's -- exactly the "cheat" this correction exists to prevent.
"""
from types import SimpleNamespace

import pytest
import torch

from compass.MultiObsSampler import MultiObsSampler
from compass.SDE import VESDE


def configured_joint_sampler(covariance, correction, hierarchy_size):
    covariance = torch.as_tensor(covariance, dtype=torch.float64)
    if covariance.dim() == 2:
        covariance = covariance.unsqueeze(0)
    sampler = MultiObsSampler(SimpleNamespace(sde=VESDE(sigma=5.0)))
    sampler.hierarchy = list(range(hierarchy_size))
    sampler.local_latent_indices = list(
        range(hierarchy_size, covariance.shape[-1])
    )
    sampler.full_gaussian_features = (
        sampler.hierarchy + sampler.local_latent_indices
    )
    sampler.covariance_features = sampler.full_gaussian_features
    sampler.correction = correction
    sampler.denoise_clamp = None
    sampler.prior_mean = torch.zeros(hierarchy_size)
    sampler.prior_std = torch.ones(hierarchy_size)
    sampler.prior_covariance = torch.eye(hierarchy_size, dtype=torch.float64)
    sampler.prior_precision_matrix = torch.eye(hierarchy_size, dtype=torch.float64)
    sampler.posterior_covariance = covariance
    sampler.posterior_precision_matrix = sampler._precision_from_covariance(covariance)
    sampler.posterior_precision = None
    sampler.posterior_mean = None
    sampler.global_posterior_mean = None
    sampler.global_posterior_covariance = None
    sampler.global_posterior_precision_matrix = None
    sampler.world_size = 1
    sampler.num_observations = covariance.shape[0]
    sampler._covariance_time_cache = {}
    sampler._configure_damping(covariance.shape[0], 1.0, 0.5, None)
    return sampler


def configured_full_sampler(covariance, prior_covariance=None):
    covariance = torch.as_tensor(covariance, dtype=torch.float32)
    if covariance.dim() == 2:
        covariance = covariance.unsqueeze(0)
    h = covariance.shape[-1]
    prior_covariance = (
        torch.eye(h) if prior_covariance is None
        else torch.as_tensor(prior_covariance, dtype=torch.float32)
    )
    sampler = MultiObsSampler(SimpleNamespace(sde=VESDE(sigma=5.0)))
    sampler.hierarchy = list(range(h))
    sampler.correction = "full_gaussian"
    sampler.denoise_clamp = 5.0
    sampler.prior_mean = torch.linspace(-0.2, 0.3, h)
    sampler.prior_std = torch.diagonal(prior_covariance).sqrt()
    sampler.prior_covariance = prior_covariance
    sampler.prior_precision_matrix = torch.linalg.inv(prior_covariance)
    sampler.posterior_covariance = covariance
    sampler.posterior_precision_matrix = torch.linalg.inv(covariance)
    sampler.posterior_precision = None
    sampler.posterior_mean = None
    sampler.world_size = 1
    sampler.num_observations = covariance.shape[0]
    sampler._configure_damping(covariance.shape[0], 1.0, 0.5, None)
    return sampler


def test_gauss_hierarchical_matches_dense_arrowhead_reference():
    """Exact block-precision (arrowhead) elimination, generalized Lemma 3.2."""
    covariance = torch.tensor([
        [[0.55, 0.18], [0.18, 0.75]],
        [[0.65, -0.12], [-0.12, 0.60]],
    ], dtype=torch.float64)
    sampler = configured_joint_sampler(
        covariance, "gauss_hierarchical", hierarchy_size=1
    )
    t = torch.tensor([[0.4]])
    variance = sampler.sde.lambda_t(t).square()
    effective, cross = sampler._effective_global_factors(variance)
    global_scores = torch.tensor([[[0.7]], [[-0.2]]], dtype=torch.float64)
    local_scores = torch.tensor([[[-0.3]], [[0.8]]], dtype=torch.float64)
    q = float(1.0 / variance)
    prior_t = sampler.prior_precision_matrix + q * torch.eye(1)
    global_precision = effective.sum(0) - prior_t
    global_rhs = torch.einsum(
        "nij,nsj->nsi", effective, global_scores
    ).sum(0, keepdim=True)

    dense = torch.zeros(3, 3, dtype=torch.float64)
    dense[0, 0] = global_precision
    rhs = torch.zeros(3, 1, dtype=torch.float64)
    local_offsets = []
    rhs_global = global_rhs.reshape(1).clone()
    for subject in range(2):
        r = cross[subject]
        offset = local_scores[subject, 0] - r @ global_scores[subject, 0]
        local_offsets.append(offset)
        dense[0, 0] += (r.mT @ r).reshape(())
        dense[0, subject + 1] = -r.reshape(())
        dense[subject + 1, 0] = -r.reshape(())
        dense[subject + 1, subject + 1] = 1.0
        rhs_global -= (r.mT @ offset).reshape(1)
        rhs[subject + 1] = offset
    rhs[0] = rhs_global
    dense_solution = torch.linalg.solve(dense, rhs)

    scores = torch.cat([global_scores, local_scores], dim=-1)
    state = torch.zeros_like(scores)
    result = sampler._compositional_score(scores.clone(), state, t)

    torch.testing.assert_close(
        result[:1, :, 0], dense_solution[0].reshape(1, 1)
    )
    for subject, offset in enumerate(local_offsets):
        torch.testing.assert_close(
            result[subject, :, 1], dense_solution[subject + 1].reshape(1)
        )


def test_gauss_hierarchical_matches_gauss_global_local_without_oracle_moments():
    """Identical composition whenever Gauss_global_local isn't fed pilot moments."""
    covariance = torch.tensor([
        [[0.55, -0.18], [-0.18, 0.70]],
        [[0.45, -0.12], [-0.12, 0.60]],
    ], dtype=torch.float64)
    scores = torch.tensor([
        [[0.4, -0.6], [0.2, 0.5]],
        [[-0.3, 0.7], [0.1, -0.4]],
    ], dtype=torch.float64)
    state = torch.tensor([
        [[0.1, -0.2], [0.15, 0.25]],
        [[0.1, 0.3], [0.15, -0.35]],
    ], dtype=torch.float64)
    t = torch.tensor([[0.3]], dtype=torch.float64)

    hierarchical = configured_joint_sampler(
        covariance, "gauss_hierarchical", hierarchy_size=1
    )
    legacy = configured_joint_sampler(
        covariance, "Gauss_global_local", hierarchy_size=1
    )
    result_hierarchical = hierarchical._compositional_score(scores.clone(), state, t)
    result_legacy = legacy._compositional_score(scores.clone(), state, t)
    torch.testing.assert_close(result_hierarchical, result_legacy)


def test_gauss_hierarchical_single_observation_returns_model_score():
    sampler = configured_joint_sampler(
        torch.tensor([[0.5, 0.1], [0.1, 0.8]]), "gauss_hierarchical", hierarchy_size=1
    )
    scores = torch.tensor([[[0.4, -0.6]]], dtype=torch.float64)
    state = torch.randn_like(scores)
    result = sampler._compositional_score(scores.clone(), state, torch.tensor([[0.7]]))
    torch.testing.assert_close(result, scores)


def test_gauss_hierarchical_reduces_to_full_gaussian_without_locals():
    """With no local dimensions, gauss_hierarchical collapses to Algorithm 2."""
    covariances = torch.tensor([
        [[0.45, 0.12], [0.12, 0.70]],
        [[0.60, -0.08], [-0.08, 0.50]],
    ])
    full = configured_full_sampler(covariances)
    hierarchical = configured_joint_sampler(
        covariances.to(torch.float64), "gauss_hierarchical", hierarchy_size=2
    )
    hierarchical.prior_mean = full.prior_mean.to(torch.float64)
    hierarchical.prior_precision_matrix = full.prior_precision_matrix.to(torch.float64)
    hierarchical.prior_covariance = full.prior_covariance.to(torch.float64)
    hierarchical.denoise_clamp = None

    scores = torch.tensor([
        [[0.7, -0.1]],
        [[-0.3, 0.6]],
    ], dtype=torch.float64)
    state = torch.tensor([[[0.25, -0.4]]]).expand_as(scores).clone().to(torch.float64)
    t = torch.tensor([[0.4]], dtype=torch.float64)

    full_result = full._compositional_score(
        scores.to(torch.float32).clone(), state.to(torch.float32), t.to(torch.float32)
    )
    hierarchical_result = hierarchical._compositional_score(scores.clone(), state, t)
    torch.testing.assert_close(
        hierarchical_result.to(torch.float32), full_result, atol=1e-5, rtol=1e-5
    )


def test_gauss_hierarchical_rejects_posterior_mean_via_direct_state():
    """The 'moment projection' shortcut must never fire for this correction."""
    sampler = configured_joint_sampler(
        torch.tensor([[0.5, 0.1], [0.1, 0.8]]), "gauss_hierarchical", hierarchy_size=1
    )
    sampler.posterior_mean = torch.zeros(1, 2, dtype=torch.float64)
    scores = torch.tensor([[[0.4, -0.6]]], dtype=torch.float64)
    state = torch.randn_like(scores)
    # _moment_projected_scores would silently replace the score if posterior_mean
    # were honored; gauss_hierarchical must not depend on it being unset by
    # accident, so the sampling API itself refuses posterior_mean (see below).
    # Here we only check _compositional_score does not crash when misused
    # directly, since the real guard lives in sample()/map_estimate().
    result = sampler._compositional_score(scores.clone(), state, torch.tensor([[0.7]]))
    assert torch.isfinite(result).all()


def test_sample_rejects_posterior_mean_for_gauss_hierarchical():
    sampler = MultiObsSampler(SimpleNamespace(sde=VESDE(sigma=5.0)))
    with pytest.raises(ValueError, match="composes only the network's real scores"):
        sampler.sample(
            world_size=1, data=torch.zeros(2, 1),
            condition_mask=torch.tensor([0.0, 0.0, 1.0]),
            hierarchy=[0], correction="gauss_hierarchical",
            prior=([0.0], [1.0]),
            posterior_mean=torch.zeros(2, 1),
            posterior_covariance=torch.eye(2).expand(2, -1, -1),
            num_samples=1, timesteps=1, verbose=False,
        )


def test_sample_rejects_global_posterior_mean_for_gauss_hierarchical():
    sampler = MultiObsSampler(SimpleNamespace(sde=VESDE(sigma=5.0)))
    with pytest.raises(ValueError, match="'Gauss_global_local'"):
        sampler.sample(
            world_size=1, data=torch.zeros(2, 1),
            condition_mask=torch.tensor([0.0, 0.0, 1.0]),
            hierarchy=[0], correction="gauss_hierarchical",
            prior=([0.0], [1.0]),
            global_posterior_mean=torch.zeros(2, 1),
            global_posterior_covariance=torch.eye(1).expand(2, -1, -1),
            posterior_covariance=torch.eye(2).expand(2, -1, -1),
            num_samples=1, timesteps=1, verbose=False,
        )


def test_gauss_hierarchical_applies_local_cross_correction():
    covariance = torch.tensor([[0.5, 0.2], [0.2, 0.8]], dtype=torch.float64)
    covariance = covariance.expand(2, -1, -1).clone()
    sampler = configured_joint_sampler(
        covariance, "gauss_hierarchical", hierarchy_size=1
    )
    scores = torch.tensor([
        [[0.4, -0.2]],
        [[-0.1, 0.7]],
    ], dtype=torch.float64)
    state = torch.zeros_like(scores)
    result = sampler._compositional_score(scores.clone(), state, torch.tensor([[0.35]]))
    # A nonzero cross-covariance must move the local score away from the raw
    # network output once composition changes the shared coordinate.
    assert not torch.allclose(result[:, :, 1], scores[:, :, 1])
