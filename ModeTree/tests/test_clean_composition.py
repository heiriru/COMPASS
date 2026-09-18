"""Milestone B gate: the composed clean score must equal grad log p exactly."""

import math

import pytest
import torch

from modetree import benchmarks as B
from modetree.composition import CleanComposer, OracleRowSource, MarginalGlobalComposer

TOL = 1e-6
N_LADDER = (1, 2, 8, 32, 100)


def _relative(a, b):
    return float(torch.linalg.vector_norm(a - b) / (1.0 + torch.linalg.vector_norm(b)))


@pytest.mark.parametrize("n", N_LADDER)
@pytest.mark.parametrize("regime", ["symmetric", "narrow_broad", "separated"])
def test_all_global_1d(n, regime):
    model = B.bimodal_1d(n, regime)
    composer = CleanComposer(OracleRowSource(model))
    torch.manual_seed(0)
    points = torch.linspace(-6.0, 6.0, 13, dtype=torch.float64).unsqueeze(-1)
    for g in points:
        exact = model.tall_score(g)
        got = composer.score(g)
        assert _relative(got, exact) < TOL, (n, regime, float(g))


@pytest.mark.parametrize("n", N_LADDER)
def test_all_global_2d(n):
    model = B.four_mode_2d(n, "unequal_covariance")
    composer = CleanComposer(OracleRowSource(model))
    torch.manual_seed(1)
    points = torch.randn(9, 2, dtype=torch.float64) * 2.0
    for g in points:
        assert _relative(composer.score(g), model.tall_score(g)) < TOL


@pytest.mark.parametrize("n", N_LADDER)
@pytest.mark.parametrize("d_g,d_l", [(1, 1), (2, 1), (7, 3)])
def test_hierarchical_factorisation(n, d_g, d_l):
    """Both the global and every local block of the composed score."""
    model = B.hierarchical(n, d_g=d_g, d_l=d_l, modes=2)
    composer = CleanComposer(OracleRowSource(model))
    torch.manual_seed(2)
    state = torch.randn(composer.dim, dtype=torch.float64)
    exact = model.tall_score(state)
    got = composer.score(state)
    assert _relative(got, exact) < TOL
    # and the local blocks individually, so a global-only agreement cannot hide
    # a compensating local error
    for i in range(n):
        a = d_g + i * d_l
        assert _relative(got[a:a + d_l], exact[a:a + d_l]) < TOL


@pytest.mark.parametrize("n", (1, 8, 32))
def test_arrowhead_hessian_matches_autograd(n):
    model = B.hierarchical(n, d_g=2, d_l=1, modes=2)
    composer = CleanComposer(OracleRowSource(model))
    torch.manual_seed(3)
    state = torch.randn(composer.dim, dtype=torch.float64) * 0.5
    dense = composer.dense_hessian(state)
    exact = model.tall_hessian(state)
    assert _relative(dense, exact) < 1e-6


@pytest.mark.parametrize("n", (1, 8, 100))
def test_schur_global_precision(n):
    """``P_g`` from the arrowhead elimination == inverse of the global block of
    the full inverse negative Hessian."""
    model = B.hierarchical(n, d_g=2, d_l=1, modes=2)
    composer = CleanComposer(OracleRowSource(model))
    torch.manual_seed(4)
    state = torch.randn(composer.dim, dtype=torch.float64) * 0.3
    blocks = composer.blocks(state)
    P = blocks.global_precision()
    dense = -composer.dense_hessian(state)
    cov = torch.linalg.inv(dense)
    reference = torch.linalg.inv(cov[:2, :2])
    assert _relative(P, reference) < 1e-8


@pytest.mark.parametrize("n", (1, 8, 100))
def test_marginal_composer_matches_grid(n):
    """Locals-integrated composition == grad of the exact global marginal."""
    model = B.hierarchical(n, d_g=1, d_l=1, modes=2)
    composer = MarginalGlobalComposer(model)
    for g in torch.linspace(-4, 4, 9, dtype=torch.float64).unsqueeze(-1):
        assert _relative(composer.score(g), model.global_score(g)) < TOL


def test_numerically_stable_form_is_used():
    """The increment form must not be silently replaced by the naive sum."""
    model = B.bimodal_1d(100, "symmetric")
    composer = CleanComposer(OracleRowSource(model))
    g = torch.tensor([3.5], dtype=torch.float64)
    naive = sum(model.row_score(i, g) for i in range(100)) - 99 * model.prior_score(g)
    exact = model.tall_score(g)
    stable = composer.score(g)
    assert _relative(stable, exact) <= _relative(naive, exact) + 1e-15
