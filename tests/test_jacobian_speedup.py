"""Tests for the gauss_jacobian(_FD) speed-ups (docs/jacobian_speedup_plan.md).

1. Refresh bands: the fixed high-noise edge and the adaptive band return the
   advertised periods.
2. Secant updates: for a Gaussian row posterior the pair (u, v) = (s2 - s1,
   D1 - D2) satisfies K v = u at *any* two noise levels, so an exact K is a fixed
   point, a wrong K converges to the exact one, and noisy pairs never break
   positive definiteness.
3. Warm start: the truncated schedule keeps exactly the cold schedule's levels
   below start_lambda, and the arguments are validated.
4. Laplace initial state: exact on a linear-Gaussian hierarchy, and sampling
   from it (alone or with the band schedule and secant updates) recovers the
   exact tall posterior.
"""
import pytest
import torch

from compass.MultiObsSampler import MultiObsSampler
from compass.experimental.oracles import LinearGaussianHierarchy, oracle_model


# ---------------------------------------------------------------------------
# 1. Refresh bands
# ---------------------------------------------------------------------------

def schedule_sampler():
    sampler = MultiObsSampler(oracle_model(gaussian_hierarchy()))
    sampler.jacobian_refresh = 1
    return sampler


def test_fixed_high_noise_edge():
    sampler = schedule_sampler()
    sampler._configure_jacobian_schedule(1e-2, 10, 0.3, refresh_high_noise=20,
                                         refresh_high_lambda=3.0)
    assert sampler._jacobian_refresh_at(1.0 / 5.0 ** 2) == 20
    assert sampler._jacobian_refresh_at(1.0 / 1.0 ** 2) == 1
    assert sampler._jacobian_refresh_at(1.0 / 0.1 ** 2) == 10


def test_adaptive_band_uses_the_widest_direction():
    sampler = schedule_sampler()
    sampler._configure_jacobian_schedule(1e-2, None, 0.3, refresh_band=(1.0, 15.0))
    # Before the first refresh nothing is known: refresh.
    assert sampler._jacobian_refresh_at(1.0) == 1
    sampler._jacobian_info_scale = (4.0, 400.0)       # k_min, k_max
    # r = lambda^2 * k_min
    assert sampler._jacobian_refresh_at(1.0 / 1.0) == 1          # r = 4: in band
    assert sampler._jacobian_refresh_at(1.0 / 10.0) == sampler.NEVER_REFRESH  # r = 40
    assert sampler._jacobian_refresh_at(1.0 / 0.1) == sampler.NEVER_REFRESH   # r = 0.4
    sampler._configure_jacobian_schedule(1e-2, None, 0.3, refresh_band=(1.0, 15.0),
                                         refresh_outside_band=7)
    sampler._jacobian_info_scale = (4.0, 400.0)
    assert sampler._jacobian_refresh_at(1.0 / 10.0) == 7


@pytest.mark.parametrize("kwargs", [
    dict(refresh_high_noise=5),                       # without its lambda
    dict(refresh_high_noise=0, refresh_high_lambda=1.0),
    dict(refresh_band=(2.0, 1.0)),
    dict(refresh_band=(0.0, 1.0), refresh_outside_band=0),
])
def test_schedule_arguments_are_validated(kwargs):
    with pytest.raises(ValueError):
        schedule_sampler()._configure_jacobian_schedule(1e-2, None, 0.3, **kwargs)


# ---------------------------------------------------------------------------
# 2. Secant updates
# ---------------------------------------------------------------------------

def gaussian_rows(rows=4, samples=6, dimension=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    factor = torch.randn(rows, dimension, dimension, generator=generator, dtype=torch.float64)
    covariance = factor @ factor.mT / dimension + 0.2 * torch.eye(dimension, dtype=torch.float64)
    mean = torch.randn(rows, 1, dimension, generator=generator, dtype=torch.float64)
    return covariance, mean, generator


def gaussian_score(y, covariance, mean, variance):
    eye = torch.eye(covariance.shape[-1], dtype=torch.float64)
    precision = torch.linalg.inv(covariance + variance * eye).unsqueeze(1)
    return -(precision @ (y - mean).unsqueeze(-1)).squeeze(-1)


def secant_sampler(dimension):
    sampler = schedule_sampler()
    sampler.full_gaussian_features = list(range(dimension))
    sampler._reset_jacobian_state()
    return sampler


def feed(sampler, information, covariance, mean, generator, pairs, samples=6, noise=0.0):
    rows, dimension = covariance.shape[0], covariance.shape[-1]
    for step in range(pairs + 1):
        variance = float(torch.exp(torch.empty(()).uniform_(-4, 1, generator=generator)))
        y = mean + torch.randn(rows, samples, dimension, generator=generator, dtype=torch.float64)
        score = gaussian_score(y, covariance, mean, variance)
        score = score + noise * torch.randn(score.shape, generator=generator, dtype=torch.float64)
        if step:
            information = sampler._secant_update(information, y, score, variance)
        sampler._secant_previous = (y, score, variance)
    return information


def test_exact_information_is_a_fixed_point_across_noise_levels():
    covariance, mean, generator = gaussian_rows()
    exact = torch.linalg.inv(covariance).unsqueeze(1).expand(-1, 6, -1, -1).clone()
    sampler = secant_sampler(3)
    updated = feed(sampler, exact.clone(), covariance, mean, generator, pairs=5)
    assert torch.allclose(updated, exact, rtol=1e-8, atol=1e-8)
    assert sampler._secant_updates > 0


def test_wrong_information_converges_to_the_exact_one():
    covariance, mean, generator = gaussian_rows()
    exact = torch.linalg.inv(covariance).unsqueeze(1).expand(-1, 6, -1, -1)
    start = torch.eye(3, dtype=torch.float64).expand(4, 6, 3, 3).clone()
    sampler = secant_sampler(3)
    updated = feed(sampler, start, covariance, mean, generator, pairs=40)
    before = torch.linalg.matrix_norm(start - exact) / torch.linalg.matrix_norm(exact)
    after = torch.linalg.matrix_norm(updated - exact) / torch.linalg.matrix_norm(exact)
    assert float(after.max()) < 0.05 < float(before.min())


def test_noisy_pairs_keep_the_information_positive_definite():
    covariance, mean, generator = gaussian_rows(seed=3)
    start = torch.linalg.inv(covariance).unsqueeze(1).expand(-1, 6, -1, -1).clone()
    sampler = secant_sampler(3)
    updated = feed(sampler, start, covariance, mean, generator, pairs=30, noise=0.5)
    assert torch.allclose(updated, updated.mT)
    assert float(torch.linalg.eigvalsh(updated).min()) > 0.0


# ---------------------------------------------------------------------------
# 3. Warm-start schedule
# ---------------------------------------------------------------------------

def test_warm_schedule_keeps_the_cold_levels_below_start():
    sampler = schedule_sampler()
    sampler.sde = sampler.SBIm.sde
    sampler.eps = 1e-3
    sampler.correction = "gauss_jacobian_FD"
    sampler.timesteps = 40
    sampler.start_lambda = None
    cold = sampler.sde.lambda_t(sampler._make_sampling_schedule("cpu")).reshape(-1)
    sampler.start_lambda = 0.5
    warm = sampler.sde.lambda_t(sampler._make_sampling_schedule("cpu")).reshape(-1)
    assert float(warm[0]) == pytest.approx(0.5, rel=1e-5)
    below = cold[cold < 0.5]
    assert torch.allclose(warm[1:], below, rtol=1e-5)
    assert sampler.timesteps == len(below) + 1


# ---------------------------------------------------------------------------
# 4. Laplace initial state on an exact linear-Gaussian hierarchy
# ---------------------------------------------------------------------------

def gaussian_hierarchy():
    # H = 2 correlated globals, L = 1 local, X = 2 observations per subject.
    return LinearGaussianHierarchy([[1.0, 0.5]], [[1.0], [0.6]], [[0.0, 0.8], [0.4, 0.0]],
                                   s_g=1.0, tau=0.6, nu=0.4)


def oracle_problem(observations=6, seed=0):
    hier = gaussian_hierarchy()
    torch.manual_seed(seed)
    obs = hier.simulate(observations)[:, hier.observed_indices]
    mask = torch.zeros(hier.nodes)
    mask[hier.observed_indices] = 1.0
    local_sd = (hier.A @ hier.A.T * hier.s_g ** 2
                + hier.tau ** 2).diagonal().sqrt()
    common = dict(hierarchy=list(range(hier.H)),
                  prior=(torch.zeros(hier.H), hier.s_g * torch.ones(hier.H)),
                  local_prior=(torch.zeros(hier.L), local_sd.float()))
    return hier, obs.float(), mask, common


def sample(model, obs, mask, common, draws, **kwargs):
    sampler = MultiObsSampler(model)
    out = sampler.sample(world_size=1, data=obs, condition_mask=mask, num_samples=draws,
                         timesteps=40, correction="gauss_jacobian_FD", denoise_clamp=None,
                         churn_eta=0.0, device="cpu", verbose=False, **common, **kwargs)
    return out, sampler


def test_laplace_initial_state_is_exact_on_a_gaussian_hierarchy():
    hier, obs, mask, common = oracle_problem()
    model = oracle_model(hier)
    sampler = MultiObsSampler(model)
    lam, draws = 0.5, 6000
    generator = torch.Generator().manual_seed(1)
    state = sampler.laplace_initial_state(obs, mask, draws, lam, generator=generator,
                                          denoise_clamp=None, **common)
    mean_g, cov_g = hier.global_posterior(obs.double())
    g = state[0, :, :hier.H].double()
    target_cov = cov_g + lam ** 2 * torch.eye(hier.H, dtype=torch.float64)
    standard_error = target_cov.diagonal().sqrt() / draws ** 0.5
    assert torch.all((g.mean(0) - mean_g).abs() < 4 * standard_error)
    assert torch.allclose(torch.cov(g.T), target_cov, rtol=0.1, atol=0.01)
    local_means, local_covs = hier.local_posterior(obs.double())
    locals_ = state[:, :, hier.H:hier.H + hier.L].double()
    local_target = local_covs[:, 0, 0] + lam ** 2
    assert torch.all((locals_.mean(1)[:, 0] - local_means[:, 0]).abs()
                     < 4 * local_target.sqrt() / draws ** 0.5)
    assert torch.allclose(locals_.var(1)[:, 0], local_target, rtol=0.1)
    assert sampler.laplace_diagnostics["row_samples"] > 0


@pytest.mark.parametrize("variant", [
    {},
    {"jacobian_refresh": 5, "jacobian_secant": True},
    {"jacobian_refresh_band": (1.0, 15.0)},
])
def test_warm_started_sampling_recovers_the_exact_posterior(variant):
    hier, obs, mask, common = oracle_problem()
    model = oracle_model(hier)
    lam, draws = 0.5, 4000
    init = MultiObsSampler(model).laplace_initial_state(
        obs, mask, draws, lam, generator=torch.Generator().manual_seed(2),
        denoise_clamp=None, **common)
    torch.manual_seed(3)
    out, sampler = sample(model, obs, mask, common, draws, start_lambda=lam,
                          init_state=init, **variant)
    mean_g, cov_g = hier.global_posterior(obs.double())
    g = out[0, :, :hier.H].double()
    standard_error = cov_g.diagonal().sqrt() / draws ** 0.5
    assert torch.all((g.mean(0) - mean_g).abs() < 5 * standard_error)
    assert torch.allclose(torch.cov(g.T), cov_g, rtol=0.15, atol=0.003)
    assert sampler.timesteps < 40


def test_warm_start_arguments_are_validated():
    hier, obs, mask, common = oracle_problem()
    model = oracle_model(hier)
    with pytest.raises(ValueError):
        sample(model, obs, mask, common, 10, start_lambda=0.5)
    with pytest.raises(ValueError):
        sample(model, obs, mask, common, 10, start_lambda=0.5,
               init_state=torch.zeros(obs.shape[0], 11, hier.nodes))
