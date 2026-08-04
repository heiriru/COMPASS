"""Focused tests for joint hierarchical score-ascent MAP refinement."""

import math

import pytest
import torch

from compass.ModelTransfuser import ModelTransfuser
from compass.MultiObsSampler import MultiObsSampler
from compass.SDE import VESDE, VPSDE


class MockSBIm:
    def __init__(self, model, nodes_size, sde=None):
        self.sde = sde or VESDE(sigma=25.0)
        self.model = model(self.sde)
        self.nodes_size = nodes_size

    def output_scale_function(self, t, value):
        return value / self.sde.marginal_prob_std(t).to(value.device)


class SharedLocalGaussianScore(torch.nn.Module):
    """Exact diffused score for g,l ~ Normal and x = g + l + noise."""

    sigma_global = 0.8
    sigma_local = 0.6
    sigma_x = 0.25

    def __init__(self, sde):
        super().__init__()
        self.sde = sde
        precision = torch.tensor([
            [1 / self.sigma_global**2 + 1 / self.sigma_x**2, 1 / self.sigma_x**2],
            [1 / self.sigma_x**2, 1 / self.sigma_local**2 + 1 / self.sigma_x**2],
        ])
        self.register_buffer("posterior_covariance", torch.linalg.inv(precision))

    def forward(self, x, t, c, return_attn_weights=False):
        noise_std = self.sde.marginal_prob_std(t).to(x.device)
        alpha = self.sde.alpha_t(t).to(x.device)
        theta = x[:, :2]
        observed = x[:, 2]
        rhs = torch.stack([
            observed / self.sigma_x**2,
            observed / self.sigma_x**2,
        ], dim=1)
        mean = rhs @ self.posterior_covariance.T
        covariance = alpha**2 * self.posterior_covariance + noise_std**2 * torch.eye(2, device=x.device)
        mean = alpha * mean
        score = torch.linalg.solve(covariance, (mean - theta).unsqueeze(-1)).squeeze(-1)
        result = torch.zeros_like(x)
        result[:, :2] = noise_std * score
        return (result, torch.zeros(1)) if return_attn_weights else result


def analytic_shared_local_map(observations):
    n = len(observations)
    sg = SharedLocalGaussianScore.sigma_global
    sl = SharedLocalGaussianScore.sigma_local
    sx = SharedLocalGaussianScore.sigma_x
    precision = torch.zeros(n + 1, n + 1)
    precision[0, 0] = 1 / sg**2 + n / sx**2
    precision[1:, 1:] = torch.eye(n) * (1 / sl**2 + 1 / sx**2)
    precision[0, 1:] = 1 / sx**2
    precision[1:, 0] = 1 / sx**2
    rhs = torch.cat([(observations.sum() / sx**2).reshape(1), observations / sx**2])
    covariance = torch.linalg.inv(precision)
    return covariance @ rhs, covariance.diag().sqrt()


def make_joint(observations, global_value, local_values):
    result = torch.zeros(len(observations), 3)
    result[:, 0] = global_value
    result[:, 1] = torch.as_tensor(local_values)
    result[:, 2] = observations
    return result


@pytest.mark.parametrize("sde", [VESDE(sigma=25.0), VPSDE()])
def test_joint_map_recovers_exact_shared_and_local_gaussian_mode(sde):
    observations = torch.tensor([-0.7, 0.1, 0.8, 1.2])
    expected, posterior_std = analytic_shared_local_map(observations)
    initial = make_joint(observations, 1.5, torch.full((len(observations),), -1.0))
    sampler = MultiObsSampler(MockSBIm(SharedLocalGaussianScore, 3, sde=sde))
    result = sampler.map_estimate(
        data=initial, condition_mask=torch.tensor([0.0, 0.0, 1.0]), init=initial,
        hierarchy=[0], prior=([0.0], [SharedLocalGaussianScore.sigma_global]),
        correction="uncorrected", sigma_start=0.35, timesteps=100,
        iterations_per_level=4, eps=1e-3, device="cpu",
    )[:, 0]

    inferred = torch.cat([result[:1, 0].flatten(), result[:, 1]])
    error_in_std = (inferred - expected).abs() / posterior_std
    assert float(error_in_std.max()) < 0.05
    assert float((result[:, 0] - result[0, 0]).abs().max()) <= 1e-6
    assert torch.equal(result[:, 2], observations)


def test_damped_joint_map_recovers_exact_shared_and_local_gaussian_mode():
    observations = torch.tensor([-0.7, 0.1, 0.8, 1.2])
    expected, posterior_std = analytic_shared_local_map(observations)
    initial = make_joint(
        observations, 1.5, torch.full((len(observations),), -1.0)
    )
    sampler = MultiObsSampler(MockSBIm(SharedLocalGaussianScore, 3))
    result = sampler.map_estimate(
        data=initial, condition_mask=torch.tensor([0.0, 0.0, 1.0]),
        init=initial, hierarchy=[0],
        prior=([0.0], [SharedLocalGaussianScore.sigma_global]),
        correction="damping", damping_at_noise=0.5,
        sigma_start=0.35, timesteps=100, iterations_per_level=4,
        eps=1e-3, device="cpu",
    )[:, 0]

    inferred = torch.cat([result[:1, 0].flatten(), result[:, 1]])
    error_in_std = (inferred - expected).abs() / posterior_std
    assert float(error_in_std.max()) < 0.05
    assert float((result[:, 0] - result[0, 0]).abs().max()) <= 1e-6
    assert torch.equal(result[:, 2], observations)


def test_gaussian_corrected_joint_map_does_not_shrink_at_high_observation_count():
    observations = 0.7 + torch.linspace(-0.9, 0.9, 50)
    expected, posterior_std = analytic_shared_local_map(observations)
    initial = make_joint(observations, expected[0], expected[1:])
    single_global_precision = (
        1 / SharedLocalGaussianScore.sigma_global**2
        + 1 / (
            SharedLocalGaussianScore.sigma_local**2
            + SharedLocalGaussianScore.sigma_x**2
        )
    )
    sampler = MultiObsSampler(MockSBIm(SharedLocalGaussianScore, 3))
    result = sampler.map_estimate(
        data=initial,
        condition_mask=torch.tensor([0.0, 0.0, 1.0]),
        init=initial,
        hierarchy=[0],
        prior=([0.0], [SharedLocalGaussianScore.sigma_global]),
        correction="gauss",
        posterior_precision=torch.full((len(observations), 1), single_global_precision),
        sigma_start=0.4,
        timesteps=100,
        iterations_per_level=3,
        eps=1e-3,
        device="cpu",
    )[:, 0]

    inferred = torch.cat([result[:1, 0].flatten(), result[:, 1]])
    error_in_std = (inferred - expected).abs() / posterior_std
    assert float(error_in_std.max()) < 0.05
    assert float((result[:, 0] - result[0, 0]).abs().max()) <= 1e-6
    assert torch.equal(result[:, 2], observations)


def test_all_global_and_single_observation_are_supported():
    observations = torch.tensor([-0.4, 0.2, 0.9])
    sg = SharedLocalGaussianScore.sigma_global
    sl = SharedLocalGaussianScore.sigma_local
    sx = SharedLocalGaussianScore.sigma_x
    precision = torch.tensor([
        [1 / sg**2 + len(observations) / sx**2, len(observations) / sx**2],
        [len(observations) / sx**2, 1 / sl**2 + len(observations) / sx**2],
    ])
    expected = torch.linalg.solve(
        precision,
        torch.full((2,), observations.sum() / sx**2),
    )
    initial = make_joint(observations, 1.2, torch.full((len(observations),), 1.2))
    sampler = MultiObsSampler(MockSBIm(SharedLocalGaussianScore, 3))
    result = sampler.map_estimate(
        initial, torch.tensor([0.0, 0.0, 1.0]), init=initial,
        hierarchy=[0, 1], prior=([0.0, 0.0], [sg, sl]),
        correction="uncorrected", sigma_start=0.35, timesteps=100,
        iterations_per_level=4, device="cpu",
    )[:, 0]
    assert torch.allclose(result[0, :2], expected, atol=8e-3)
    assert float((result[:, :2] - result[:1, :2]).abs().max()) <= 1e-6

    one_observation = observations[:1]
    one_initial = make_joint(one_observation, -1.3, torch.tensor([1.4]))
    one_result = sampler.map_estimate(
        one_initial, torch.tensor([0.0, 0.0, 1.0]), init=one_initial,
        hierarchy=[0], prior=([0.0], [sg]), correction="uncorrected",
        sigma_start=0.35, timesteps=100, iterations_per_level=4, device="cpu",
    )
    assert torch.isfinite(one_result).all()


class SymmetricMixtureScore(torch.nn.Module):
    def __init__(self, sde):
        super().__init__()
        self.sde = sde

    def forward(self, x, t, c, return_attn_weights=False):
        noise_std = self.sde.marginal_prob_std(t).to(x.device)
        variance = 0.05**2 + noise_std**2
        theta = x[:, :1]
        means = torch.tensor([-2.0, 2.0], device=x.device).reshape(1, 2)
        log_weights = -0.5 * (theta - means)**2 / variance
        weights = torch.softmax(log_weights, dim=1)
        score = (weights * (means - theta)).sum(dim=1, keepdim=True) / variance
        result = torch.zeros_like(x)
        result[:, :1] = noise_std * score
        return (result, torch.zeros(1)) if return_attn_weights else result


class AnalyticMixtureDensity:
    def log_prob(self, data, condition_mask, **kwargs):
        theta = torch.as_tensor(data)[:, 0]
        terms = torch.stack([
            -0.5 * ((theta + 2.0) / 0.05)**2,
            -0.5 * ((theta - 2.0) / 0.05)**2,
        ], dim=1)
        return torch.logsumexp(terms, dim=1) - math.log(2.0)


def test_multistart_avoids_low_density_posterior_mean():
    negative = -2.0 + 0.04 * torch.randn(40, generator=torch.Generator().manual_seed(2))
    positive = 2.0 + 0.04 * torch.randn(40, generator=torch.Generator().manual_seed(3))
    posterior = torch.cat([negative, positive]).reshape(1, 80, 1)
    starts, _ = ModelTransfuser._joint_map_initializations(
        posterior, torch.zeros(1, 1), torch.tensor([0.0, 1.0]), [0], 5
    )
    assert abs(float(starts[0, 0, 0])) < 0.1
    assert float(starts[0, :, 0].min()) < -1.8
    assert float(starts[0, :, 0].max()) > 1.8

    sampler = MultiObsSampler(MockSBIm(SymmetricMixtureScore, 2))
    refined = sampler.map_estimate(
        starts[:, 0, :], torch.tensor([0.0, 1.0]), init=starts,
        hierarchy=[0], prior=([0.0], [1.0]), correction="uncorrected",
        sigma_start=0.12, timesteps=80, iterations_per_level=3, device="cpu",
    )
    scores = ModelTransfuser._hierarchical_candidate_scores(
        AnalyticMixtureDensity(), refined, torch.tensor([0.0, 1.0]), [0],
        (torch.tensor([0.0]), torch.tensor([1.0])), 10, 1e-3, "cpu", False,
    )
    selected = refined[0, int(torch.argmax(scores)), 0]
    assert abs(float(selected)) > 1.8


def test_shared_marginal_kde_optimizes_correlated_block_jointly():
    generator = torch.Generator().manual_seed(18)
    dominant_count = 700
    secondary_count = 300
    dominant_base = torch.randn(dominant_count, 2, generator=generator)
    secondary_base = torch.randn(secondary_count, 2, generator=generator)
    transform = torch.tensor([[0.16, 0.0], [0.12, 0.08]])
    dominant = torch.tensor([1.1, -0.7]) + dominant_base @ transform.T
    secondary = torch.tensor([-1.8, 1.6]) + 0.25 * secondary_base
    shared = torch.cat([dominant, secondary], dim=0)

    n_obs = 4
    posterior = torch.empty(n_obs, len(shared), 3)
    posterior[:, :, :2] = shared.unsqueeze(0)
    for observation in range(n_obs):
        posterior[observation, :, 2] = (
            0.3 * observation - shared[:, 0] + 0.4 * shared[:, 1]
        )

    result = ModelTransfuser._shared_marginal_kde_map(
        posterior,
        condition_mask=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        hierarchy=[0, 1],
        num_starts=6,
    )
    assert torch.allclose(
        result["shared_map"], torch.tensor([1.1, -0.7]), atol=0.12
    )
    assert result["shared_map"].shape == (2,)
    assert result["candidate_modes"].shape[1] == 2
    assert torch.isclose(result["sample_weights"].sum(), torch.tensor(1.0))
    assert result["effective_sample_size"] > 1.0


def test_hierarchical_map_validation():
    observations = torch.tensor([0.1, 0.2])
    initial = make_joint(observations, 0.0, torch.zeros(2))
    sampler = MultiObsSampler(MockSBIm(SharedLocalGaussianScore, 3))
    with pytest.raises(ValueError, match="fnpe"):
        sampler.map_estimate(
            initial, torch.tensor([0.0, 0.0, 1.0]), hierarchy=[0],
            prior=([0.0], [1.0]), correction="fnpe",
        )
    with pytest.raises(ValueError, match="posterior_precision"):
        sampler.map_estimate(
            initial, torch.tensor([0.0, 0.0, 1.0]), hierarchy=[0],
            prior=([0.0], [1.0]), correction="gauss",
        )
    with pytest.raises(ValueError, match="at least one"):
        sampler.map_estimate(
            initial, torch.tensor([0.0, 0.0, 1.0]), hierarchy=[],
            prior=([], []), correction="uncorrected",
        )
    unsynchronized = initial.unsqueeze(1).clone()
    unsynchronized[1, 0, 0] = 0.2
    with pytest.raises(ValueError, match="synchronized"):
        sampler.map_estimate(
            initial, torch.tensor([0.0, 0.0, 1.0]), init=unsynchronized,
            hierarchy=[0], prior=([0.0], [1.0]), correction="uncorrected",
        )

class JointScoreCompareMock:
    nodes_size = 3

    def __init__(self):
        from types import SimpleNamespace
        self.sampler = SimpleNamespace(all_attn_weights=None)
        self.multi_obs_sampler = SimpleNamespace(
            prior_mean=torch.tensor([0.0]), prior_std=torch.tensor([1.0]),
            posterior_precision=None, denoise_clamp=5.0,
        )
        self.map_calls = 0
        self.map_masks = []

    def sample(self, x=None, num_samples=20, **kwargs):
        x = torch.as_tensor(x, dtype=torch.float32).flatten()
        global_samples = torch.linspace(0.5, 1.5, num_samples)
        result = torch.empty(len(x), num_samples, 2)
        result[:, :, 0] = global_samples.unsqueeze(0)
        result[:, :, 1] = x[:, None] - global_samples[None, :]
        return result

    def map_estimate(self, data, condition_mask, **kwargs):
        self.map_calls += 1
        self.map_masks.append(torch.as_tensor(condition_mask).clone())
        result = torch.as_tensor(data, dtype=torch.float32).clone()
        result[:, 1] = result[:, 2] - result[:, 0]
        return result

    def log_prob(self, data, condition_mask, **kwargs):
        data = torch.as_tensor(data, dtype=torch.float32)
        mask = torch.as_tensor(condition_mask, dtype=torch.float32)
        if mask[0] == 0:
            return -0.5 * ((data[:, 0] - 1.0)**2 + (data[:, 1] - (data[:, 2] - 1.0))**2)
        return -0.5 * (data[:, 2] - data[:, 0] - data[:, 1])**2


def test_model_transfuser_joint_score_dispatches_without_changing_legacy_modes():
    observations = torch.tensor([[0.2], [1.1], [2.0], [-0.4]])
    model = JointScoreCompareMock()
    transfuser = ModelTransfuser()
    transfuser.add_model("mock", model)
    transfuser.compare(
        observations, condition_mask=torch.tensor([0.0, 0.0, 1.0]),
        multi_obs_inference=True, hierarchy=[0], correction="uncorrected",
        map_method="joint_score", map_num_starts=4, map_timesteps=5,
        map_iterations_per_level=1, likelihood_method="pfode",
        criterion="bic", num_samples=20, device="cpu", verbose=False,
    )
    inferred = torch.as_tensor(transfuser.stats["mock"]["MAP"][:, 0])
    shared_map = transfuser.stats["mock"]["joint_map_shared_map"]
    assert model.map_calls == 1
    assert torch.equal(model.map_masks[0], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.allclose(inferred[:, 0], shared_map.expand(len(observations)))
    assert torch.allclose(
        inferred[:, 1], observations.flatten() - shared_map,
    )
    assert "joint_map_candidate_scores" in transfuser.stats["mock"]
    assert transfuser.stats["mock"]["joint_map_strategy"] == (
        "shared_marginal_kde_then_fixed_shared_local_score"
    )
