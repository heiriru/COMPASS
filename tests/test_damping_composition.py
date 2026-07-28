"""Focused CPU tests for compositional error damping and adaptive sampling."""

import math

import pytest
import torch

from compass.MultiObsSampler import MultiObsSampler
from compass.Sampler import Sampler
from compass.SDE import VESDE


class ExactGaussianScore(torch.nn.Module):
    """Exact diffused score for theta~N(0,1), x|theta~N(theta,1)."""

    def __init__(self, sde):
        super().__init__()
        self.sde = sde

    def forward(self, x, t, c, return_attn_weights=False):
        noise_std = self.sde.marginal_prob_std(t).to(x.device)
        theta, observation = x[:, :1], x[:, 1:]
        posterior_mean = 0.5 * observation
        posterior_variance = 0.5
        score = -(theta - posterior_mean) / (posterior_variance + noise_std**2)
        result = torch.zeros_like(x)
        result[:, :1] = noise_std * score
        return (result, torch.zeros(1)) if return_attn_weights else result


class MockSBIm:
    def __init__(self):
        self.sde = VESDE(sigma=5.0)
        self.model = ExactGaussianScore(self.sde)
        self.sampler = Sampler(self)
        self.nodes_size = 2

    def output_scale_function(self, t, value):
        return value / self.sde.marginal_prob_std(t).to(value.device)


def configured_sampler(correction, n=2, at_noise=0.25):
    sampler = MultiObsSampler(MockSBIm())
    sampler.hierarchy = [0]
    sampler.correction = correction
    sampler.denoise_clamp = None
    sampler.prior_mean = torch.tensor([0.0])
    sampler.prior_std = torch.tensor([1.0])
    sampler.num_observations = n
    sampler.world_size = 1
    sampler._configure_damping(n, 1.0, at_noise, None)
    return sampler


def test_damping_endpoint_names_and_default():
    sampler = configured_sampler("damping", n=4, at_noise=None)
    assert sampler.damping_at_data == 1.0
    assert sampler.damping_at_noise == pytest.approx(0.5)
    assert float(sampler._damping_factor(torch.tensor(0.0))) == pytest.approx(1.0)
    assert float(sampler._damping_factor(torch.tensor(1.0))) == pytest.approx(0.5)
    assert float(sampler._damping_factor(torch.tensor(0.5))) == pytest.approx(math.sqrt(0.5))

    with pytest.raises(ValueError, match="damping_at_noise"):
        sampler._configure_damping(4, 0.5, 0.8, None)
    with pytest.raises(ValueError, match="composition_batch_size"):
        sampler._configure_damping(4, 1.0, 0.5, 5)


def test_plain_damping_formula_and_local_scores():
    sampler = configured_sampler("damping")
    scores = torch.tensor([[[1.0, 7.0]], [[2.0, 9.0]]])
    state = torch.zeros_like(scores)
    result = sampler._compositional_score(scores.clone(), state, torch.tensor([[0.5]]))
    # d(0.5)=0.5, the prior score is zero, and sum_j score_j=3.
    assert torch.allclose(result[:, :, 0], torch.full((2, 1), 1.5))
    assert torch.equal(result[:, :, 1], scores[:, :, 1])


def test_minibatch_damping_is_unbiased_over_all_singletons(monkeypatch):
    sampler = configured_sampler("damping", n=3, at_noise=1.0)
    sampler.composition_batch_size = 1
    scores = torch.tensor([[[1.0]], [[2.0]], [[6.0]]])
    state = torch.zeros_like(scores)
    outputs = []
    for selected in range(3):
        permutation = torch.tensor([selected] + [i for i in range(3) if i != selected])
        monkeypatch.setattr(torch, "randperm", lambda n, device=None, p=permutation: p.to(device))
        outputs.append(
            sampler._compositional_score(
                scores.clone(), state, torch.tensor([[0.0]])
            )[0, 0, 0]
        )
    assert float(torch.stack(outputs).mean()) == pytest.approx(9.0)


@pytest.mark.parametrize("correction", ["gauss_damping", "hybrid_damping"])
def test_gaussian_damping_formulas_match_manual_diagonal_result(correction):
    sampler = configured_sampler(correction)
    sampler.posterior_precision = torch.tensor([[2.0], [4.0]])
    scores = torch.tensor([[[0.7]], [[-0.2]]])
    state = torch.full_like(scores, 0.4)
    t = torch.tensor([[0.5]])
    variance = sampler.sde.marginal_prob_std(t) ** 2
    prior_score = -0.4 / (1.0 + variance)
    prior_precision = 1.0 + 1.0 / variance
    observation_precision = sampler.posterior_precision + 1.0 / variance
    weighted = observation_precision[0] * 0.7 + observation_precision[1] * -0.2
    if correction == "hybrid_damping":
        a_t = (1 - 2) * (1 - t)
        denominator = observation_precision.sum() + a_t * prior_precision
        prior_numerator = a_t * prior_precision * prior_score
    else:
        denominator = observation_precision.sum() + (1 - 2) * prior_precision
        prior_numerator = (1 - 2) * prior_precision * prior_score
    expected = 0.5 * (
        weighted + prior_numerator
    ) / denominator
    result = sampler._compositional_score(scores.clone(), state, t)
    assert float(result[0, 0, 0]) == pytest.approx(float(expected), rel=1e-5)


def test_damping_uses_standard_reference_scale():
    sampler = configured_sampler("damping", n=20)
    scale = sampler._hierarchy_scale(torch.zeros(20, 4, 2))
    assert torch.equal(scale, torch.ones(2))


def test_damped_score_ascent_recovers_global_gaussian_mode():
    observations = torch.tensor([[-0.8], [0.2], [0.7], [1.1]])
    expected = observations.sum() / (1.0 + len(observations))
    initial = torch.cat([
        torch.full((len(observations), 1), -1.0), observations
    ], dim=1)
    for correction in ("damping", "gauss_damping", "hybrid_damping"):
        sampler = MultiObsSampler(MockSBIm())
        precision = (
            torch.full((len(observations), 1), 2.0)
            if correction != "damping" else None
        )
        result = sampler.map_estimate(
            initial, torch.tensor([0.0, 1.0]), init=initial,
            hierarchy=[0], prior=([0.0], [1.0]), correction=correction,
            posterior_precision=precision, damping_at_noise=0.3,
            sigma_start=0.3, timesteps=50, iterations_per_level=4,
            device="cpu",
        )
        assert abs(float(result[0, 0, 0] - expected)) < 0.03
        assert float((result[:, 0, 0] - result[0, 0, 0]).abs().max()) <= 1e-6


def test_adaptive_reverse_sde_is_finite_and_synchronized():
    observations = torch.tensor([[-0.4], [0.6], [1.0]])
    sampler = MultiObsSampler(MockSBIm())
    torch.manual_seed(11)
    result = sampler.sample(
        world_size=1, data=observations,
        condition_mask=torch.tensor([0.0, 1.0]), hierarchy=[0],
        prior=([0.0], [1.0]), correction="damping",
        damping_at_noise=0.5, method="adaptive", num_samples=32,
        timesteps=20, adaptive_abs_tol=0.05, adaptive_rel_tol=0.5,
        adaptive_max_evals=400, device="cpu", verbose=False,
    )
    assert torch.isfinite(result).all()
    assert float((result[:, :, 0] - result[:1, :, 0]).abs().max()) <= 1e-6
    assert torch.allclose(
        result[:, :, 1], observations.expand(-1, result.shape[1])
    )
    assert sampler.solver_stats["accepted_steps"] > 0
    assert sampler.solver_stats["score_evaluations"] <= 400


def test_adaptive_tighter_tolerance_uses_at_least_as_many_evaluations():
    observations = torch.tensor([[-0.4], [0.6], [1.0]])
    evaluations = []
    for absolute, relative in ((0.1, 0.8), (0.01, 0.1)):
        sampler = MultiObsSampler(MockSBIm())
        torch.manual_seed(4)
        sampler.sample(
            world_size=1, data=observations,
            condition_mask=torch.tensor([0.0, 1.0]), hierarchy=[0],
            prior=([0.0], [1.0]), correction="damping",
            method="adaptive", num_samples=8, timesteps=20,
            adaptive_abs_tol=absolute, adaptive_rel_tol=relative,
            adaptive_max_evals=1000, device="cpu", verbose=False,
        )
        evaluations.append(sampler.solver_stats["score_evaluations"])
    assert evaluations[1] >= evaluations[0]


def test_adaptive_evaluation_limit_fails_clearly():
    sampler = MultiObsSampler(MockSBIm())
    with pytest.raises(RuntimeError, match="exhausted adaptive_max_evals"):
        sampler.sample(
            world_size=1, data=torch.tensor([[-1.0], [1.0]]),
            condition_mask=torch.tensor([0.0, 1.0]), hierarchy=[0],
            prior=([0.0], [1.0]), correction="damping",
            method="adaptive", num_samples=4, timesteps=10,
            adaptive_abs_tol=1e-12, adaptive_rel_tol=1e-12,
            adaptive_initial_step=0.5, adaptive_max_evals=2,
            device="cpu", verbose=False,
        )


def test_gaussian_minibatch_and_adaptive_validation():
    observations = torch.zeros(3, 1)
    sampler = MultiObsSampler(MockSBIm())
    with pytest.raises(ValueError, match="composition_batch_size"):
        sampler.sample(
            world_size=1, data=observations,
            condition_mask=torch.tensor([0.0, 1.0]), hierarchy=[0],
            prior=([0.0], [1.0]), correction="gauss_damping",
            posterior_precision=torch.full((3, 1), 2.0),
            composition_batch_size=1, num_samples=2, verbose=False,
        )
    with pytest.raises(ValueError, match="adaptive_max_evals"):
        sampler.sample(
            world_size=1, data=observations,
            condition_mask=torch.tensor([0.0, 1.0]), hierarchy=[0],
            prior=([0.0], [1.0]), correction="damping",
            method="adaptive", adaptive_max_evals=1,
            num_samples=2, verbose=False,
        )


@pytest.mark.parametrize("correction", ["gauss_damping", "hybrid_damping"])
def test_damped_gaussian_modes_estimate_precision_and_final_denoise(
    correction, monkeypatch,
):
    observations = torch.tensor([[-0.2], [0.4]])
    sampler = MultiObsSampler(MockSBIm())
    calls = {"precision": 0, "denoise": 0}

    def estimate(*args, **kwargs):
        calls["precision"] += 1
        return torch.full((len(observations), 1), 2.0)

    original_denoise = sampler._final_denoise

    def denoise(*args, **kwargs):
        calls["denoise"] += 1
        return original_denoise(*args, **kwargs)

    monkeypatch.setattr(sampler, "_estimate_posterior_precision", estimate)
    monkeypatch.setattr(sampler, "_final_denoise", denoise)
    result = sampler.sample(
        world_size=1, data=observations,
        condition_mask=torch.tensor([0.0, 1.0]), hierarchy=[0],
        prior=([0.0], [1.0]), correction=correction,
        posterior_precision=None, method="euler", timesteps=5,
        num_samples=8, device="cpu", verbose=False,
    )
    assert torch.isfinite(result).all()
    assert calls == {"precision": 1, "denoise": 1}
