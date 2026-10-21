"""Tests for compass.experimental.marginal_schur, against exact oracles."""

import math

import pytest
import torch

from compass.experimental.marginal_schur import MarginalSchurSampler, install
from compass.experimental.oracles import LinearGaussianHierarchy, oracle_model


def hierarchy():
    # H = 2 correlated globals, L = 1 local, X = 2 observations.
    A = [[1.0, 0.5]]
    B = [[1.0], [0.6]]
    C = [[0.0, 0.8], [0.4, 0.0]]
    return LinearGaussianHierarchy(A, B, C, s_g=1.0, tau=0.6, nu=0.4)


def configured(model, correction, estimator, H, L):
    sampler = MarginalSchurSampler(model, estimator=estimator)
    sampler.hierarchy = list(range(H))
    sampler.local_latent_indices = list(range(H, H + L))
    sampler.full_gaussian_features = sampler.hierarchy + sampler.local_latent_indices
    sampler.correction = correction
    sampler._configure_jacobian_schedule(1e-3, None, 0.3)
    sampler.jacobian_refresh = 1
    sampler.jacobian_curvature_floor = 1e-6
    sampler._reset_marginal_state()
    sampler.score_network_calls = 0
    sampler.evaluated_subject_rows = 0
    return sampler


def exact_marginal_score(hier, g, x_obs, sigma):
    """nabla_g log p_t(g_t | x) for the Gaussian oracle."""
    mask = torch.zeros(hier.nodes)
    mask[hier.observed_indices] = 1
    U, gain, cov = hier.conditional(mask)
    H = hier.H
    mean = (gain @ x_obs)[:H]
    cov_g = cov[:H, :H] + sigma ** 2 * torch.eye(H, dtype=torch.float64)
    return -torch.linalg.solve(cov_g, (g - mean).T).T


@pytest.mark.parametrize("estimator", ["schur", "laplace"])
def test_marginal_is_exact_for_any_local_state_on_a_gaussian(estimator):
    hier = hierarchy()
    model = oracle_model(hier)
    sampler = configured(model, "arruda_schur", estimator, hier.H, hier.L)
    torch.manual_seed(0)
    x_obs = hier.simulate(1)[0, hier.observed_indices]
    S = 64
    t = torch.tensor([0.35])
    sigma = float(model.sde.marginal_prob_std(t))
    state = torch.zeros(1, S, hier.nodes, dtype=torch.float64)
    state[:, :, :hier.H] = torch.randn(1, S, hier.H, dtype=torch.float64)
    state[:, :, hier.H:hier.H + hier.L] = 3.0 * torch.randn(1, S, hier.L, dtype=torch.float64)
    state[:, :, hier.observed_indices] = x_obs
    mask = torch.zeros(1, S, hier.nodes, dtype=torch.float64)
    mask[..., hier.observed_indices] = 1
    scores = sampler._raw_row_scores(state, t, mask)
    marginal, _ = sampler._marginal_scores(scores.clone(), state, t, mask, sigma ** 2)
    exact = exact_marginal_score(hier, state[0, :, :hier.H], x_obs, sigma)
    assert torch.allclose(marginal[0], exact, rtol=1e-4, atol=1e-5)
    # The joint g-score itself is not the marginal (it depends on l).
    assert (scores[0, :, :hier.H].double() - exact).abs().max() > 1e-2


class ScaleModel(torch.nn.Module):
    """Clean joint score of l ~ N(mu, e^{2s}), x ~ N(l, v) with a flat prior, nodes [mu, s, l, x].

    The conditional precision of l depends on s, so the l-volume (log-det) term
    of the marginal is non-zero -- the partial-pooling situation in one line.
    """

    def __init__(self, sde, v=0.3):
        super().__init__()
        self.sde, self.v = sde, v

    def forward(self, x, t, c, return_attn_weights=False):
        mu, s, l, obs = x.double().unbind(-1)
        prec = torch.exp(-2 * s)
        d_mu = (l - mu) * prec
        d_s = (l - mu) ** 2 * prec - 1.0
        d_l = -(l - mu) * prec + (obs - l) / self.v
        score = torch.stack([d_mu, d_s, d_l, torch.zeros_like(l)], -1)
        sigma = self.sde.marginal_prob_std(t).double().reshape(-1, 1)
        return (score * sigma).to(x.dtype)


def scale_marginal_score(mu, s, obs, v):
    total = torch.exp(2 * s) + v
    d_mu = (obs - mu) / total
    d_s = ((obs - mu) ** 2 / total ** 2 - 1.0 / total) * torch.exp(2 * s)
    return torch.stack([d_mu, d_s], -1)


def scale_setup(estimator):
    from compass import ScoreBasedInferenceModel as SBIm
    model = SBIm(nodes_size=4, sde_type="vesde", sigma=25.0, hidden_size=8, depth=1,
                 num_heads=1, mlp_ratio=1)
    model.model = ScaleModel(model.sde)
    sampler = configured(model, "arruda_schur", estimator, 2, 1)
    return model, sampler


def scale_state(S, l_values):
    state = torch.zeros(1, S, 4, dtype=torch.float64)
    state[:, :, 0] = 0.2
    state[:, :, 1] = math.log(0.7)
    state[:, :, 2] = l_values
    state[:, :, 3] = 1.1
    return state


def scale_mask(S):
    mask = torch.zeros(1, S, 4, dtype=torch.float64)
    mask[..., 3] = 1
    return mask


def test_laplace_log_det_term_recovers_the_exact_marginal_for_a_scale_parameter():
    model, sampler = scale_setup("laplace")
    t = torch.tensor([1e-3])
    sigma = float(model.sde.marginal_prob_std(t))
    state = scale_state(5, torch.tensor([-2.0, -0.5, 0.3, 1.0, 2.5], dtype=torch.float64))
    mask = scale_mask(5)
    scores = sampler._raw_row_scores(state, t, mask)
    marginal, _ = sampler._marginal_scores(scores.clone(), state, t, mask, sigma ** 2)
    exact = scale_marginal_score(torch.tensor(0.2), torch.tensor(math.log(0.7)),
                                 torch.tensor(1.1), 0.3)
    assert torch.allclose(marginal[0], exact.expand(5, 2).double(), atol=2e-3)


def test_schur_without_log_det_is_biased_but_the_fisher_average_is_not():
    model, sampler = scale_setup("direct")
    t = torch.tensor([1e-3])
    sigma = float(model.sde.marginal_prob_std(t))
    mu, s, obs, v = 0.2, math.log(0.7), 1.1, 0.3
    prec = math.exp(-2 * s) + 1 / v
    mode = (mu * math.exp(-2 * s) + obs / v) / prec
    torch.manual_seed(0)
    S = 40000
    draws = mode + torch.randn(S, dtype=torch.float64) / math.sqrt(prec)
    exact = scale_marginal_score(torch.tensor(mu), torch.tensor(s), torch.tensor(obs), v)

    state = scale_state(S, draws)
    mask = scale_mask(S)
    scores = sampler._raw_row_scores(state, t, mask)
    marginal, _ = sampler._marginal_scores(scores.clone(), state, t, mask, sigma ** 2)
    assert torch.allclose(marginal[0].mean(0), exact.double(), atol=0.02)

    sampler.configure(estimator="schur")
    state = scale_state(1, torch.tensor([mode], dtype=torch.float64))
    mask = scale_mask(1)
    scores = sampler._raw_row_scores(state, t, mask)
    at_mode, _ = sampler._marginal_scores(scores.clone(), state, t, mask, sigma ** 2)
    # The dropped term -1/2 d/ds log(e^{-2s} + 1/v) = +e^{-2s}/(e^{-2s} + 1/v)
    # = v/(sigma^2 + v): without it the log-sigma score is too low, i.e. every
    # subject pushes sigma down by that much.
    missing = math.exp(-2 * s) / prec
    assert exact[1].item() - at_mode[0, 0, 1].item() == pytest.approx(missing, abs=2e-3)


def composed_moments(hier, correction, estimator, observations, samples=1500,
                     mean_field=False, **kwargs):
    model = oracle_model(hier, mean_field=mean_field)
    install(model, estimator=estimator)
    mask = torch.zeros(hier.nodes)
    mask[hier.observed_indices] = 1
    torch.manual_seed(1)
    draws = model.sample(x=observations.float(), condition_mask=mask, multi_obs_inference=True,
                         hierarchy=list(range(hier.H)), correction=correction,
                         num_samples=samples, timesteps=kwargs.pop("timesteps", 40),
                         device="cpu", verbose=False, **kwargs)
    g = draws[0, :, :hier.H].double()
    return g.mean(0), torch.cov(g.T), draws


@pytest.mark.parametrize("correction,estimator", [
    ("arruda_schur", "laplace"),
    ("arruda_schur", "schur"),
    ("arruda_schur_gauss", "schur"),
    ("gauss_jacobian_FD", "laplace"),
])
def test_composed_global_posterior_matches_the_exact_one(correction, estimator):
    hier = hierarchy()
    torch.manual_seed(3)
    observations = hier.simulate(12)[:, hier.observed_indices]
    mean, cov = composed_moments(hier, correction, estimator, observations)[:2]
    exact_mean, exact_cov = hier.global_posterior(observations)
    sd = exact_cov.diagonal().sqrt()
    assert ((mean - exact_mean).abs() / sd).max() < 0.15
    if correction == "arruda_schur":
        # The plain sum follows the bridge p_t(g)^(1-J) prod_j p_t(g|x_j), which is
        # not the diffused posterior; a reverse-SDE solver on it is not exact in
        # width (measured ~0.83x here). Only the mean is a property of the rule.
        return
    assert ((cov.diagonal().sqrt() / sd) - 1).abs().max() < 0.15


def test_ancestral_locals_match_the_exact_local_posterior():
    hier = hierarchy()
    torch.manual_seed(4)
    observations = hier.simulate(6)[:, hier.observed_indices]
    _, _, draws = composed_moments(hier, "arruda_schur_gauss", "laplace", observations)
    local = draws[:, :, hier.local_indices].double()
    means, covs = hier.local_posterior(observations)
    sd = covs[:, 0, 0].sqrt()
    assert ((local.mean(1)[:, 0] - means[:, 0]).abs() / sd).max() < 0.15
    assert ((local.std(1)[:, 0] / sd) - 1).abs().max() < 0.15


def test_mean_field_factors_compose_to_the_wrong_posterior():
    """The production trunk's per-coordinate factors: same rule, wrong answer."""
    hier = hierarchy()
    torch.manual_seed(3)
    observations = hier.simulate(12)[:, hier.observed_indices]
    exact_mean, exact_cov = hier.global_posterior(observations)
    sd = exact_cov.diagonal().sqrt()

    def error(mean_field, estimator):
        mean, cov = composed_moments(hier, "arruda_schur_gauss", estimator, observations,
                                     mean_field=mean_field)[:2]
        return max(((mean - exact_mean).abs() / sd).max().item(),
                   ((cov.diagonal().sqrt() / sd) - 1).abs().max().item())

    mean_field, joint = error(True, "direct"), error(False, "laplace")
    assert mean_field > 0.15 and mean_field > 2 * joint, (mean_field, joint)


class CurvedScaleModel(ScaleModel):
    """ScaleModel plus a quartic local term: H_ll = e^{-2s} + 1/v + 3 c l^2 depends on l."""

    def __init__(self, sde, v=0.3, c=0.5):
        super().__init__(sde, v)
        self.c = c

    def forward(self, x, t, c, return_attn_weights=False):
        out = super().forward(x, t, c).double()
        sigma = self.sde.marginal_prob_std(t).double().reshape(-1, 1)
        l = x.double()[..., 2]
        out[..., 2] = out[..., 2] - self.c * l ** 3 * sigma[..., 0]
        return out.to(x.dtype)


def test_log_det_gradient_is_taken_at_one_local_state():
    """Regression: the baseline curvature must be evaluated at the same l as the shifts."""
    from compass import ScoreBasedInferenceModel as SBIm
    model = SBIm(nodes_size=4, sde_type="vesde", sigma=25.0, hidden_size=8, depth=1,
                 num_heads=1, mlp_ratio=1)
    model.model = CurvedScaleModel(model.sde)
    sampler = configured(model, "arruda_schur", "laplace", 2, 1)
    t = torch.tensor([1e-3])
    variance = float(model.sde.marginal_prob_std(t)) ** 2
    ls = torch.tensor([-1.0, 0.3, 1.2], dtype=torch.float64)
    state = scale_state(3, ls)
    mask = scale_mask(3)
    gradient = sampler._logdet_gradient(state, t, mask, variance, 1e-6)
    s = math.log(0.7)
    H = math.exp(-2 * s) + 1 / 0.3 + 3 * 0.5 * ls ** 2
    assert torch.allclose(gradient[0, :, 0], torch.zeros(3, dtype=torch.float64), atol=1e-2)
    assert torch.allclose(gradient[0, :, 1], -2 * math.exp(-2 * s) / H, atol=1e-2)
