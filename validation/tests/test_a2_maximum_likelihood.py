"""A2: differentiable PF-ODE likelihood and the MLE on exact-score oracles.

    .COMPASS/bin/python -m pytest -q validation/tests
"""

import validation.common.runtime  # noqa: F401

import math

import pytest
import torch

from compass.likelihood_maximization import maximize_likelihood
from validation.common.gaussian_oracles import GaussianJoint, nested_offset_pair, oracle_sbim

NU = 0.5
B = torch.tensor([[1.0, 0.0], [0.5, 1.0], [-0.3, 0.8]], dtype=torch.float64)


def regression_model(s=1.0):
    M = torch.zeros(5, 5, dtype=torch.float64)
    M[:2, :2] = s * torch.eye(2)
    M[2:, :2] = s * B
    M[2:, 2:] = NU * torch.eye(3)
    return oracle_sbim(GaussianJoint(M, 2))


def rows(n=4, seed=0, s=1.0):
    g = torch.Generator().manual_seed(seed)
    theta = s * torch.randn(n, 2, generator=g, dtype=torch.float64)
    x = theta @ B.T + NU * torch.randn(n, 3, generator=g, dtype=torch.float64)
    return theta, x


MASK = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])


def test_differentiable_log_prob_matches_plain():
    model = regression_model()
    theta, x = rows()
    data = torch.cat([theta, x], 1).float()
    plain = model.log_prob(data, MASK, timesteps=32)
    graph = model.log_prob(data.clone().requires_grad_(True), MASK, timesteps=32,
                           differentiable=True)
    assert graph.requires_grad
    assert torch.allclose(plain, graph.detach().cpu(), atol=1e-5)


def test_gradient_matches_finite_differences():
    model = regression_model()
    theta, x = rows(n=2)
    data = torch.cat([theta, x], 1).float()
    leaf = data.clone().requires_grad_(True)
    grad, = torch.autograd.grad(model.log_prob(leaf, MASK, timesteps=32,
                                               differentiable=True).sum(), leaf)
    h = 1e-2
    for d in (0, 1):
        up, down = data.clone(), data.clone()
        up[:, d] += h
        down[:, d] -= h
        fd = (model.log_prob(up, MASK, timesteps=32) - model.log_prob(down, MASK, timesteps=32)) / (2 * h)
        assert torch.allclose(grad[:, d], fd, atol=2e-2, rtol=2e-2)


def test_mle_recovers_closed_form_and_never_loses_to_start():
    model = regression_model(s=0.3)    # strong prior: MAP far from MLE
    _, x = rows(n=4, s=0.3)
    mle = torch.linalg.solve(B.T @ B, B.T @ x.T).T
    start = torch.cat([torch.zeros(4, 2, dtype=torch.float64), x], 1).float()
    res = maximize_likelihood(model, start, MASK, max_iterations=50, timesteps=32)
    assert torch.allclose(res.theta[:, :2].double(), mle, atol=5e-3)
    assert torch.all(res.log_likelihood >= res.log_likelihood_start - 1e-6)
    assert not res.at_bound.any()


def test_shared_dims_are_shared_and_bounds_are_reported():
    joint1, _ = nested_offset_pair(s_g=1.0)
    model = oracle_sbim(joint1)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(5, 2, generator=g) + torch.tensor([0.0, 1.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    data = torch.cat([torch.zeros(5, 2), x], 1)
    res = maximize_likelihood(model, data, mask, shared_dims=[0], max_iterations=50,
                              timesteps=32)
    assert torch.allclose(res.theta[:, 0], res.theta[0, 0].expand(5))
    assert res.theta[0, 0].item() == pytest.approx(x[:, 1].mean().item(), abs=5e-3)
    assert torch.allclose(res.theta[:, 1], x[:, 0], atol=5e-3)

    upper = torch.full((4,), math.inf)
    upper[0] = 0.2
    res = maximize_likelihood(model, data, mask, shared_dims=[0], upper=upper,
                              max_iterations=50, timesteps=32)
    assert res.theta[0, 0].item() == pytest.approx(0.2, abs=1e-5)
    assert res.at_bound[:, 0].all() and not res.at_bound[:, 1].any()


def test_screening_keeps_the_best_basin():
    """Bimodal likelihood: a wrong-basin start plus screened draws finds the global max."""
    from validation.a2_maximum_likelihood.nonconcave import BimodalLikelihood
    model = BimodalLikelihood()
    mask = torch.tensor([1.0, 0.0, 0.0])
    x1, x2 = torch.tensor([1.5, 1.5]), torch.tensor([0.6, -0.6])   # global max at +, -
    wrong = -torch.sign(x2) * x1.sqrt()
    data = torch.stack([wrong, x1, x2], 1)
    candidates = torch.linspace(-2, 2, 16)
    starts = torch.stack([torch.stack([c.expand(2), x1, x2], 1) for c in candidates])
    alone = maximize_likelihood(model, data, mask, max_iterations=100)
    screened = maximize_likelihood(model, data, mask, starts=starts, keep_best_starts=2,
                                   max_iterations=100)
    assert torch.all(torch.sign(alone.theta[:, 0]) == -torch.sign(x2))       # stuck
    assert torch.all(torch.sign(screened.theta[:, 0]) == torch.sign(x2))     # found
    assert torch.all(screened.log_likelihood > alone.log_likelihood)
