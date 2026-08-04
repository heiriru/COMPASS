from types import SimpleNamespace

import pytest
import torch

from compass.MultiObsSampler import MultiObsSampler
from compass.SDE import VESDE


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
    sampler.world_size = 1
    sampler.num_observations = covariance.shape[0]
    sampler._configure_damping(covariance.shape[0], 1.0, 0.5, None)
    return sampler
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
    sampler.prior_covariance = torch.eye(
        hierarchy_size, dtype=torch.float64
    )
    sampler.prior_precision_matrix = torch.eye(
        hierarchy_size, dtype=torch.float64
    )
    sampler.posterior_covariance = covariance
    sampler.posterior_precision_matrix = sampler._precision_from_covariance(
        covariance
    )
    sampler.posterior_precision = None
    sampler.world_size = 1
    sampler.num_observations = covariance.shape[0]
    sampler._covariance_time_cache = {}
    sampler._configure_damping(covariance.shape[0], 1.0, 0.5, None)
    return sampler



def test_full_gaussian_matches_algorithm_2_matrix_formula():
    covariances = torch.tensor([
        [[0.45, 0.12], [0.12, 0.70]],
        [[0.60, -0.08], [-0.08, 0.50]],
        [[0.52, 0.16], [0.16, 0.65]],
    ])
    prior_covariance = torch.tensor([[1.20, 0.25], [0.25, 0.90]])
    sampler = configured_full_sampler(covariances, prior_covariance)
    scores = torch.tensor([
        [[0.7, -0.1], [0.2, 0.4]],
        [[-0.3, 0.6], [0.8, -0.5]],
        [[0.1, 0.2], [-0.4, 0.9]],
    ])
    state = torch.tensor([
        [[0.25, -0.4], [-0.2, 0.5]],
    ]).expand_as(scores).clone()
    t = torch.tensor([[0.4]])

    result = sampler._compositional_score(scores.clone(), state, t)

    variance = sampler.sde.lambda_t(t).square().reshape(())
    identity = torch.eye(2)
    prior_precision_t = torch.linalg.inv(prior_covariance) + identity / variance
    observation_precision_t = torch.linalg.inv(covariances) + identity / variance
    theta = state[:1, :, :2]
    prior_score = -torch.linalg.solve(
        prior_covariance + variance * identity,
        (theta - sampler.prior_mean).squeeze(0).mT,
    ).mT.unsqueeze(0)
    precision = (
        (1 - len(covariances)) * prior_precision_t
        + observation_precision_t.sum(0)
    )
    numerator = (
        (1 - len(covariances))
        * torch.einsum("ij,bsj->bsi", prior_precision_t, prior_score)
        + torch.einsum(
            "nij,nsj->nsi", observation_precision_t, scores
        ).sum(0, keepdim=True)
    )
    expected = torch.linalg.solve(
        precision, numerator.squeeze(0).mT
    ).mT.unsqueeze(0)
    torch.testing.assert_close(result[0:1], expected)
    torch.testing.assert_close(result, expected.expand_as(result))


def test_gaussian_moment_projection_replaces_biased_joint_scores():
    covariance = torch.tensor([
        [[0.55, -0.18], [-0.18, 0.70]],
        [[0.45, -0.12], [-0.12, 0.60]],
    ], dtype=torch.float64)
    means = torch.tensor([[0.4, -0.2], [-0.3, 0.5]], dtype=torch.float64)
    projected = configured_joint_sampler(
        covariance, "Gauss_global_local", hierarchy_size=1
    )
    projected.posterior_mean = means
    reference = configured_joint_sampler(
        covariance, "Gauss_global_local", hierarchy_size=1
    )
    reference.posterior_mean = None
    state = torch.tensor([
        [[0.2, -0.4], [0.7, 0.1]],
        [[0.2, 0.3], [0.7, -0.6]],
    ], dtype=torch.float64)
    time = torch.tensor([[0.25]], dtype=torch.float64)
    variance = projected.sde.lambda_t(time).square().reshape(())
    covariance_t = covariance + variance * torch.eye(2, dtype=torch.float64)
    exact_scores = -torch.linalg.solve(
        covariance_t[:, None], (state - means[:, None]).unsqueeze(-1)
    ).squeeze(-1)
    biased_scores = exact_scores + torch.tensor([0.8, -0.6])

    result = projected._compositional_score(
        biased_scores, state, time
    )
    expected = reference._compositional_score(
        exact_scores, state, time
    )

    torch.testing.assert_close(result, expected)


def test_posterior_mean_validation_rejects_bad_shape_and_nonfinite_values():
    sampler = configured_joint_sampler(
        torch.eye(2), "Gauss_global_local", hierarchy_size=1
    )
    with pytest.raises(ValueError, match="posterior_mean must have shape"):
        sampler._validate_posterior_mean(torch.zeros(3), 2, 2)
    with pytest.raises(ValueError, match="finite"):
        sampler._validate_posterior_mean(
            torch.tensor([[0.0, float("nan")]]), 2, 2
        )


def test_full_gaussian_single_observation_returns_model_score():
    sampler = configured_full_sampler(torch.tensor([[0.5, 0.1], [0.1, 0.8]]))
    scores = torch.tensor([[[0.4, -0.6], [0.9, 0.2]]])
    state = torch.randn_like(scores)
    result = sampler._compositional_score(
        scores.clone(), state, torch.tensor([[0.7]])
    )
    torch.testing.assert_close(result, scores)


def test_full_gaussian_clamps_overflowing_tail_scores_before_composition():
    covariances = torch.tensor([
        [[0.45, 0.12], [0.12, 0.70]],
        [[0.60, -0.08], [-0.08, 0.50]],
    ])
    sampler = configured_full_sampler(covariances)
    scores = torch.tensor([
        [[float("inf"), -float("inf")]],
        [[float("inf"), -float("inf")]],
    ])
    state = torch.zeros_like(scores)
    t = torch.tensor([[0.4]])

    result = sampler._compositional_score(scores, state, t)

    assert torch.isfinite(result).all()
    variance = sampler.sde.lambda_t(t).square()
    denoised = state[:1, :, :2] + variance * result[:1, :, :2]
    lower = sampler.prior_mean - sampler.denoise_clamp * sampler.prior_std
    upper = sampler.prior_mean + sampler.denoise_clamp * sampler.prior_std
    assert torch.all(denoised >= lower)
    assert torch.all(denoised <= upper)


def test_full_gaussian_reduces_to_diagonal_gauss():
    covariances = torch.diag_embed(torch.tensor([[0.5, 0.8], [0.4, 0.6]]))
    prior_covariance = torch.diag(torch.tensor([1.0, 1.5]))
    full = configured_full_sampler(covariances, prior_covariance)
    scores = torch.tensor([
        [[0.4, -0.1], [0.7, 0.2]],
        [[-0.2, 0.8], [0.1, -0.5]],
    ])
    state = torch.tensor([[[0.2, -0.3], [0.5, 0.4]]]).expand_as(scores).clone()
    t = torch.tensor([[0.35]])
    full_result = full._compositional_score(scores.clone(), state, t)

    diagonal = configured_full_sampler(covariances, prior_covariance)
    diagonal.correction = "gauss"
    diagonal.posterior_precision = 1.0 / torch.diagonal(
        covariances, dim1=-2, dim2=-1
    )
    diagonal_result = diagonal._compositional_score(scores.clone(), state, t)
    torch.testing.assert_close(full_result, diagonal_result)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (torch.ones(2, 3), "shape"),
        (torch.tensor([[1.0, 0.4], [0.1, 1.0]]), "symmetric"),
        (torch.tensor([[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
        (torch.tensor([[1.0, float("nan")], [float("nan"), 1.0]]), "finite"),
    ],
)
def test_covariance_validation_errors(covariance, message):
    sampler = MultiObsSampler(SimpleNamespace(sde=VESDE(sigma=5.0)))
    sampler.hierarchy = [0, 1]
    with pytest.raises(ValueError, match=message):
        sampler._validate_covariance(covariance, 3, "posterior_covariance")


def test_correlated_prior_is_resolved_as_covariance():
    sampler = MultiObsSampler(SimpleNamespace(sde=VESDE(sigma=5.0)))
    sampler.hierarchy = [0, 1]
    covariance = torch.tensor([[1.0, 0.35], [0.35, 2.0]])
    mean, std, resolved = sampler._resolve_prior(([0.1, -0.2], covariance))
    torch.testing.assert_close(mean, torch.tensor([0.1, -0.2]))
    torch.testing.assert_close(std, torch.tensor([1.0, 2.0**0.5]))
    torch.testing.assert_close(resolved, covariance)


class CovarianceDrawSampler:
    def __init__(self):
        self.offsets = {}
        self.calls = []
        self.draws = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [0.0, 2.0],
            [2.0, 1.0], [-1.0, 1.5],
        ])

    def sample(self, data, num_samples, capture_attention, **kwargs):
        subject = int(torch.as_tensor(data)[0, 0])
        start = self.offsets.get(subject, 0)
        self.offsets[subject] = start + num_samples
        self.calls.append((subject, num_samples, capture_attention))
        result = torch.zeros(1, num_samples, 3)
        result[0, :, :2] = self.draws[start:start + num_samples] + subject
        return result


def test_automatic_posterior_moments_return_matching_mean_and_covariance():
    draw_sampler = CovarianceDrawSampler()
    sampler = MultiObsSampler(
        SimpleNamespace(sde=VESDE(sigma=5.0), sampler=draw_sampler)
    )
    sampler.hierarchy = [0, 1]
    sampler.verbose = False

    mean, covariance = sampler.estimate_posterior_moments(
        data=torch.tensor([[0.0], [1.0]]),
        condition_mask=torch.tensor([0.0, 0.0, 1.0]),
        num_samples=5, timesteps=3, eps=1e-3,
        batch_size=2, device="cpu", feature_indices=[0, 1],
    )

    expected_mean = draw_sampler.draws.to(torch.float64).mean(dim=0)
    expected_covariance = sampler._regularize_covariance(
        torch.cov(draw_sampler.draws.to(torch.float64).mT)
    )
    torch.testing.assert_close(mean, torch.stack((
        expected_mean, expected_mean + 1,
    )))
    torch.testing.assert_close(
        covariance, expected_covariance.expand(2, -1, -1)
    )


def test_automatic_covariance_estimation_is_full_and_memory_bounded():
    draw_sampler = CovarianceDrawSampler()
    sampler = MultiObsSampler(
        SimpleNamespace(sde=VESDE(sigma=5.0), sampler=draw_sampler)
    )
    sampler.hierarchy = [0, 1]
    sampler.verbose = False
    covariance = sampler._estimate_posterior_covariance(
        data=torch.tensor([[0.0], [1.0]]),
        condition_mask=torch.tensor([0.0, 0.0, 1.0]),
        num_samples=5, timesteps=3, eps=1e-3,
        batch_size=2, device="cpu",
    )
    expected = sampler._regularize_covariance(
        torch.cov(draw_sampler.draws.to(torch.float64).mT)
    )
    torch.testing.assert_close(covariance, expected.expand(2, -1, -1))
    assert [count for subject, count, _ in draw_sampler.calls if subject == 0] == [2, 2, 1]
    assert all(not capture for _, _, capture in draw_sampler.calls)


def test_composed_precision_repair_updates_matrix_and_numerator():
    sampler = configured_full_sampler(torch.eye(2))
    precision = torch.tensor([[1.0, 0.0], [0.0, -0.25]], dtype=torch.float64)
    numerator = torch.tensor([[[0.3, -0.4]]], dtype=torch.float64)
    scores = torch.tensor([
        [[1.0, 2.0]],
        [[3.0, -1.0]],
    ], dtype=torch.float64)

    _, repaired, repaired_numerator, adjustment = (
        sampler._solve_composed_global(precision, numerator, scores)
    )

    assert torch.linalg.eigvalsh(repaired).min() > 0
    score_mean = scores.mean(dim=0, keepdim=True)
    torch.testing.assert_close(
        repaired_numerator,
        numerator + torch.einsum("ij,bsj->bsi", adjustment, score_mean),
    )
    assert sampler.covariance_diagnostics["repair_fraction"] == 1.0
    assert sampler.covariance_diagnostics["maximum_relative_repair"] > 0


def test_positive_composed_precision_is_not_repaired():
    sampler = configured_full_sampler(torch.eye(2))
    precision = torch.tensor([[1.2, 0.1], [0.1, 0.8]], dtype=torch.float64)
    numerator = torch.tensor([[[0.3, -0.4]]], dtype=torch.float64)
    scores = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64)

    _, repaired, repaired_numerator, adjustment = (
        sampler._solve_composed_global(precision, numerator, scores)
    )

    torch.testing.assert_close(repaired, precision)
    torch.testing.assert_close(repaired_numerator, numerator)
    torch.testing.assert_close(adjustment, torch.zeros_like(adjustment))
    assert sampler.covariance_diagnostics["repair_fraction"] == 0.0


def test_one_dimensional_full_gaussian_matches_diagonal_gauss():
    covariance = torch.tensor([[[0.45]], [[0.60]], [[0.52]]])
    full = configured_full_sampler(covariance)
    diagonal = configured_full_sampler(covariance)
    diagonal.correction = "gauss"
    diagonal.posterior_precision = covariance.squeeze(-1).reciprocal()
    scores = torch.tensor([[[0.7], [-0.2]], [[-0.3], [0.8]], [[0.1], [0.4]]])
    state = torch.full_like(scores, 0.15)

    full_score = full._compositional_score(
        scores.clone(), state, torch.tensor([[0.4]])
    )
    diagonal_score = diagonal._compositional_score(
        scores.clone(), state, torch.tensor([[0.4]])
    )
    torch.testing.assert_close(full_score, diagonal_score)


def test_zero_cross_covariance_schur_matches_full_gaussian():
    joint_covariance = torch.tensor([
        [[0.45, 0.0], [0.0, 0.80]],
        [[0.60, 0.0], [0.0, 0.70]],
    ])
    schur = configured_joint_sampler(
        joint_covariance, "Gauss_schur_global", hierarchy_size=1
    )
    full = configured_full_sampler(joint_covariance[:, :1, :1])
    full.prior_mean.zero_()
    full.denoise_clamp = None
    scores = torch.tensor([[[0.4, -0.2]], [[-0.1, 0.7]]])
    state = torch.tensor([[[0.15, 0.3]], [[0.15, -0.4]]])

    schur_score = schur._compositional_score(
        scores.clone(), state, torch.tensor([[0.35]])
    )
    full_score = full._compositional_score(
        scores.clone(), state, torch.tensor([[0.35]])
    )
    torch.testing.assert_close(schur_score[:, :, 0], full_score[:, :, 0])
    torch.testing.assert_close(schur_score[:, :, 1], scores[:, :, 1])


def test_global_local_correction_limits():
    block_diagonal = torch.tensor([
        [[0.5, 0.0], [0.0, 0.8]],
        [[0.6, 0.0], [0.0, 0.7]],
    ])
    zero_cross = configured_joint_sampler(
        block_diagonal, "Gauss_global_local", hierarchy_size=1
    )
    scores = torch.tensor([[[0.4, -0.2]], [[-0.1, 0.7]]])
    state = torch.tensor([[[0.15, 0.3]], [[0.15, -0.4]]])
    corrected = zero_cross._compositional_score(
        scores.clone(), state, torch.tensor([[0.35]])
    )
    torch.testing.assert_close(corrected[:, :, 1], scores[:, :, 1])

    correlated = configured_joint_sampler(
        torch.tensor([[0.5, 0.2], [0.2, 0.8]]),
        "Gauss_global_local", hierarchy_size=1,
    )
    one_score = torch.tensor([[[0.4, -0.2], [0.1, 0.7]]])
    one_state = torch.zeros_like(one_score)
    corrected = correlated._compositional_score(
        one_score.clone(), one_state, torch.tensor([[0.35]])
    )
    torch.testing.assert_close(corrected[:, :, 1], one_score[:, :, 1])


def test_schur_solution_matches_dense_arrowhead_reference():
    covariance = torch.tensor([
        [[0.55, 0.18], [0.18, 0.75]],
        [[0.65, -0.12], [-0.12, 0.60]],
    ], dtype=torch.float64)
    sampler = configured_joint_sampler(
        covariance, "Gauss_global_local", hierarchy_size=1
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
        offset = (
            local_scores[subject, 0] - r @ global_scores[subject, 0]
        )
        local_offsets.append(offset)
        dense[0, 0] += (r.mT @ r).reshape(())
        dense[0, subject + 1] = -r.reshape(())
        dense[subject + 1, 0] = -r.reshape(())
        dense[subject + 1, subject + 1] = 1.0
        rhs_global -= (r.mT @ offset).reshape(1)
        rhs[subject + 1] = offset
    rhs[0] = rhs_global
    dense_solution = torch.linalg.solve(dense, rhs)

    schur_global = torch.linalg.solve(
        global_precision, global_rhs.reshape(1, 1)
    ).reshape(())
    torch.testing.assert_close(dense_solution[0, 0], schur_global)
    for subject, offset in enumerate(local_offsets):
        expected_local = (
            local_scores[subject, 0]
            + cross[subject] @ (
                schur_global.reshape(1) - global_scores[subject, 0]
            )
        )
        torch.testing.assert_close(
            dense_solution[subject + 1], expected_local
        )


def test_covariance_modes_share_complete_eps_schedule():
    covariance = torch.tensor([
        [[0.5, 0.1], [0.1, 0.8]],
        [[0.6, -0.1], [-0.1, 0.7]],
    ])
    samplers = [
        configured_full_sampler(covariance[:, :1, :1]),
        configured_joint_sampler(
            covariance, "Gauss_schur_global", hierarchy_size=1
        ),
        configured_joint_sampler(
            covariance, "Gauss_global_local", hierarchy_size=1
        ),
    ]
    schedules = []
    for sampler in samplers:
        sampler.timesteps = 17
        sampler.eps = 1e-3
        schedules.append(sampler._make_sampling_schedule("cpu"))
    torch.testing.assert_close(schedules[0], schedules[1])
    torch.testing.assert_close(schedules[0], schedules[2])
    endpoint_sigma = samplers[0].sde.lambda_t(schedules[0][-1:])
    expected_sigma = samplers[0].sde.lambda_t(torch.tensor([1e-3]))
    torch.testing.assert_close(endpoint_sigma, expected_sigma)
    assert not torch.isclose(endpoint_sigma, torch.tensor([0.2])).all()


def test_joint_covariance_storage_and_factors_scale_linearly():
    observations, global_dim, local_dim = 5, 2, 1
    base = torch.tensor([
        [0.7, 0.1, 0.08],
        [0.1, 0.6, -0.05],
        [0.08, -0.05, 0.9],
    ], dtype=torch.float64)
    covariance = base.expand(observations, -1, -1).clone()
    sampler = configured_joint_sampler(
        covariance, "Gauss_global_local", hierarchy_size=global_dim
    )
    effective, cross = sampler._effective_global_factors(torch.tensor(0.4))
    assert sampler.posterior_covariance.shape == (
        observations, global_dim + local_dim, global_dim + local_dim
    )
    assert effective.shape == (observations, global_dim, global_dim)
    assert cross.shape == (observations, local_dim, global_dim)
    assert sampler.posterior_covariance.numel() == (
        observations * (global_dim + local_dim) ** 2
    )


def test_sample_rejects_precision_and_covariance_together():
    sampler = MultiObsSampler(SimpleNamespace(sde=VESDE(sigma=5.0)))
    with pytest.raises(ValueError, match="mutually exclusive"):
        sampler.sample(
            world_size=1, data=torch.zeros(2, 1),
            condition_mask=torch.tensor([0.0, 0.0, 1.0]),
            hierarchy=[0, 1], correction="full_gaussian",
            prior=([0.0, 0.0], [1.0, 1.0]),
            posterior_precision=torch.ones(2),
            posterior_covariance=torch.eye(2),
            num_samples=1, timesteps=1, verbose=False,
        )


def test_global_local_uses_marginal_global_pilot_score_and_conjugate_reference():
    """The shared composition must ignore current local states entirely."""
    observations = torch.tensor([-1.0, 0.25, 1.5], dtype=torch.float64)
    single_variance = torch.tensor(1.0 / (1.0 + 1.0 / 1.25), dtype=torch.float64)
    single_means = (observations / 1.25 * single_variance).unsqueeze(1)
    joint_covariance = torch.tensor(
        [[5.0 / 9.0, -4.0 / 9.0], [-4.0 / 9.0, 5.0 / 9.0]],
        dtype=torch.float64,
    ).expand(len(observations), -1, -1).clone()
    sampler = configured_joint_sampler(
        joint_covariance, "Gauss_global_local", hierarchy_size=1
    )
    sampler.global_posterior_mean = single_means
    sampler.global_posterior_covariance = single_variance.reshape(1, 1, 1).expand(
        len(observations), -1, -1
    ).clone()
    sampler.global_posterior_precision_matrix = sampler._precision_from_covariance(
        sampler.global_posterior_covariance
    )

    state = torch.tensor([
        [[0.2, -100.0], [-0.4, 50.0]],
        [[0.2, 7.0], [-0.4, -3.0]],
        [[0.2, 12.0], [-0.4, 9.0]],
    ], dtype=torch.float64)
    arbitrary_scores = torch.tensor([
        [[80.0, -2.0], [-30.0, 5.0]],
        [[-60.0, 4.0], [90.0, -7.0]],
        [[15.0, 8.0], [10.0, -6.0]],
    ], dtype=torch.float64)
    time = torch.tensor([[0.35]], dtype=torch.float64)
    result = sampler._compositional_score(arbitrary_scores, state, time)

    posterior_precision = 1.0 + len(observations) / 1.25
    posterior_mean = (observations.sum() / 1.25) / posterior_precision
    variance_t = sampler.sde.lambda_t(time).square().reshape(())
    expected_global_score = -(state[0, :, 0] - posterior_mean) / (
        1.0 / posterior_precision + variance_t
    )
    torch.testing.assert_close(result[:, :, 0], expected_global_score.expand(
        len(observations), -1
    ))
    # At t=0 the pilot factors reduce exactly to the stated conjugate update.
    assert torch.allclose(
        torch.as_tensor(1.0 / (1.0 + len(observations) / 1.25)),
        torch.as_tensor(1.0 / posterior_precision),
    )
