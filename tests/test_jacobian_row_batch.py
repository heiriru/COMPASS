"""Observation mini-batching for the Jacobian corrections (``jacobian_row_batch``).

Each call refreshes only ``jacobian_row_batch`` rows and rebuilds the others from their
cached lambda-free natural parameters ``(K_j, eta_j)``. For Gaussian single-observation
posteriors those are constants, so mini-batched draws must equal the full rule's up to
rounding, however few rows are refreshed. The oracle network is the exact conditional
score of a linear-Gaussian two-level model.
"""
import pytest
import torch

from compass.MultiObsSampler import MultiObsSampler
from compass.experimental.oracles import LinearGaussianHierarchy, oracle_model


def oracle():
    generator = torch.Generator().manual_seed(0)
    A = torch.randn(1, 2, generator=generator, dtype=torch.float64) * 0.7
    B = torch.randn(2, 1, generator=generator, dtype=torch.float64)
    C = torch.randn(2, 2, generator=generator, dtype=torch.float64) * 0.5
    return oracle_model(LinearGaussianHierarchy(A, B, C, tau=0.6, nu=0.5), sigma=10.0)


def draw(model, n, **kw):
    torch.manual_seed(3)
    data = torch.randn(n, 2) * 0.8
    mask = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32)
    sampler = MultiObsSampler(model)
    torch.manual_seed(4)
    out = sampler.sample(world_size=1, data=data, condition_mask=mask, timesteps=20,
                         num_samples=32, hierarchy=[0, 1], correction="gauss_jacobian_FD",
                         prior=(torch.zeros(2), torch.ones(2)), verbose=False, **kw)
    return out, sampler.solver_stats


def test_full_batch_reproduces_the_default_rule():
    model = oracle()
    reference, stats_ref = draw(model, 6)
    batched, stats = draw(model, 6, jacobian_row_batch=6)
    torch.testing.assert_close(batched, reference, rtol=1e-4, atol=1e-4)


def test_small_batches_are_exact_for_gaussian_rows():
    model = oracle()
    reference, stats_ref = draw(model, 12)
    batched, stats = draw(model, 12, jacobian_row_batch=2)
    torch.testing.assert_close(batched, reference, rtol=1e-3, atol=1e-3)
    # the per-call network rows shrink: ~2 refreshed rows x (D+2) copies vs 12 x (D+2)
    assert stats["evaluated_row_samples"] < 0.45 * stats_ref["evaluated_row_samples"]


def test_row_batch_validation():
    model = oracle()
    with pytest.raises(ValueError):
        draw(model, 4, jacobian_row_batch=0)
    with pytest.raises(ValueError):
        draw(model, 4, jacobian_row_batch=2, jacobian_refresh=3)
    sampler = MultiObsSampler(model)
    with pytest.raises(ValueError):
        sampler.sample(world_size=1, data=torch.zeros(3, 2),
                       condition_mask=torch.tensor([0., 0, 0, 1, 1]), timesteps=4,
                       num_samples=4, hierarchy=[0, 1], correction="gauss_hierarchical",
                       jacobian_row_batch=2, verbose=False)
