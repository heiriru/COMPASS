"""Milestones H-K, L, M, N plus the prior-interface refactor (section 3)."""

import math
import os
import sys

import pytest
import torch

from modetree import benchmarks as B
from modetree.composition import make_composer
from modetree.diffusion import (
    ComposedField, DiffusedOracle, VESchedule, project_information,
    row_jacobian_factor,
)
from modetree.laplace import assemble
from modetree.tree import ModeTree, ModeTreeConfig, TimeComposer

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
))


# --- the composed field -----------------------------------------------------

@pytest.mark.parametrize("rule", ["uncorrected", "gauss_jacobian"])
@pytest.mark.parametrize("n", [1, 8, 32])
def test_composed_field_converges_to_the_clean_composition(rule, n):
    """Every rule must agree with the exact clean composition as lambda -> 0."""
    model = B.bimodal_1d(n, "symmetric")
    clean = make_composer(model)
    field = ComposedField(DiffusedOracle(model), rule)
    state = torch.tensor([0.7], dtype=torch.float64)
    errors = []
    for t in (1e-3, 1e-4, 1e-5):
        errors.append(float(
            (field.score(state, t).score - clean.score(state)).abs().max()
        ))
    # the gap is O(lambda^2), and lambda^2 is proportional to t for small t, so
    # the claim to test is the *rate*: two decades of t must buy about four
    # decades of agreement.
    # the absolute gap also scales with n (the composition sums n rows), so the
    # bound is per row
    assert errors[-1] < 1e-3 * max(n, 1) / 8.0 + 1e-4
    assert errors[0] / max(errors[-1], 1e-300) > 30.0


@pytest.mark.parametrize("n", [1, 8])
def test_hierarchical_field_matches_clean_composition(n):
    model = B.hierarchical(n, d_g=2, d_l=1, modes=2)
    clean = make_composer(model)
    field = ComposedField(DiffusedOracle(model), "uncorrected")
    torch.manual_seed(0)
    state = torch.randn(clean.dim, dtype=torch.float64) * 0.4
    errors = [float((field.score(state, t).score - clean.score(state)).abs().max())
              for t in (1e-3, 1e-5)]
    assert errors[-1] < 1e-3
    assert errors[0] / max(errors[-1], 1e-300) > 30.0


def test_row_jacobian_factor_matches_the_gauss_form_for_a_gaussian_row():
    """For a Gaussian row posterior, Sigma_t = lambda^2 (I + lambda^2 J) must
    reduce to the pilot form Lambda = Sigma_0^-1 + lambda^-2 I."""
    lam = 0.7
    sigma0 = torch.tensor([[0.5, 0.1], [0.1, 0.9]], dtype=torch.float64)
    covariance_t = sigma0 + (lam ** 2) * torch.eye(2, dtype=torch.float64)
    jacobian = -torch.linalg.inv(covariance_t).unsqueeze(0)
    precision, cross, diagnostics = row_jacobian_factor(jacobian, lam, 2)
    expected = torch.linalg.inv(sigma0) + torch.eye(2, dtype=torch.float64) / lam ** 2
    assert float((precision[0] - expected).abs().max()) < 1e-9
    assert cross is None
    assert diagnostics["eigenvalue_clips"] == 0


def test_information_projection_is_a_noop_when_admissible():
    prior = torch.eye(2, dtype=torch.float64) * 2.0
    effective = prior + torch.eye(2, dtype=torch.float64)
    delta, diagnostics = project_information(effective, prior)
    assert delta is None
    assert diagnostics["negative_information_eigenvalues"] == 0


def test_information_projection_reports_rather_than_hides():
    prior = torch.eye(2, dtype=torch.float64) * 2.0
    effective = prior - torch.diag(torch.tensor([0.5, 0.0], dtype=torch.float64))
    delta, diagnostics = project_information(effective, prior)
    assert delta is not None
    assert diagnostics["negative_information_eigenvalues"] == 1
    assert diagnostics["min_information_eigenvalue"] < 0
    assert "adaptation_relative" in diagnostics


def test_jacobian_stride_reuses_and_refreshes():
    model = B.bimodal_1d(8, "symmetric")
    oracle = DiffusedOracle(model)
    field = ComposedField(oracle, "gauss_jacobian", jacobian_refresh=4)
    state = torch.tensor([[1.0]], dtype=torch.float64)
    for _ in range(4):
        field.score(state, 0.4)
    assert oracle.jacobian_calls == 1
    field.score(state, 0.4)
    assert oracle.jacobian_calls == 2


# --- mode discovery ---------------------------------------------------------

@pytest.mark.parametrize("n", [4, 32, 100])
def test_discovery_finds_both_modes_symmetric(n):
    model = B.bimodal_1d(n, "symmetric")
    ref = B.reference_1d(model)
    tree = ModeTree(
        ComposedField(DiffusedOracle(model), "gauss_jacobian"),
        make_composer(model), ModeTreeConfig(),
        prior_std=model.prior_cov.diagonal().sqrt(),
    )
    branches = tree.run()
    found = sorted(float(b.global_state) for b in branches)
    assert len(found) == 2
    truth = sorted(float(m) for m in ref.modes.reshape(-1))
    assert max(abs(a - b) for a, b in zip(found, truth)) < 1e-3


def test_discovery_finds_the_mode_it_never_passes_through():
    """map_vs_mass: the continuation slides into the broad mode, so the narrow
    higher one can only come from a sentinel probe."""
    model = B.map_vs_mass_1d(100)
    tree = ModeTree(
        ComposedField(DiffusedOracle(model), "gauss_jacobian"),
        make_composer(model), ModeTreeConfig(),
        prior_std=model.prior_cov.diagonal().sqrt(),
    )
    branches = tree.run()
    positions = sorted(float(b.global_state) for b in branches)
    assert len(positions) == 2
    assert positions[0] < 0 < positions[1]


def test_empty_valley_places_no_mode_in_the_valley():
    """Section 14: no artificial central mode between two separated ones."""
    model = B.empty_valley_1d(32)
    tree = ModeTree(
        ComposedField(DiffusedOracle(model), "gauss_jacobian"),
        make_composer(model), ModeTreeConfig(),
        prior_std=model.prior_cov.diagonal().sqrt(),
    )
    branches = tree.run()
    for branch in branches:
        assert abs(float(branch.global_state)) > 4.0


def test_map_and_mass_branch_are_distinct_after_discovery():
    model = B.map_vs_mass_1d(32)
    clean = make_composer(model)
    tree = ModeTree(
        ComposedField(DiffusedOracle(model), "gauss_jacobian"), clean,
        ModeTreeConfig(), prior_std=model.prior_cov.diagonal().sqrt(),
    )
    branches = tree.run()
    posterior = assemble(
        clean, [b.diagnostics["clean_result"] for b in branches],
        laplace_plus_samples=0,
    )
    assert posterior.map_index != posterior.mass_index


def test_branch_ancestry_is_recorded():
    model = B.bimodal_1d(8, "symmetric")
    tree = ModeTree(
        ComposedField(DiffusedOracle(model), "gauss_jacobian"),
        make_composer(model), ModeTreeConfig(),
        prior_std=model.prior_cov.diagonal().sqrt(),
    )
    branches = tree.run()
    assert all(b.parent_id is not None for b in branches)
    assert any(event["event"] == "birth" for event in tree.events)
    assert tree.trace and "lambda" in tree.trace[0]


# --- prior interface (section 3) --------------------------------------------

def test_gaussian_mixture_prior_score_is_exact_at_every_lambda():
    from compass.priors import GaussianMixturePrior
    prior = GaussianMixturePrior(
        [0.3, 0.7], torch.tensor([[-2.0], [2.0]], dtype=torch.float64),
        torch.tensor([[[1.0]], [[0.5]]], dtype=torch.float64),
    )
    points = torch.linspace(-6, 6, 25, dtype=torch.float64).unsqueeze(-1)
    for lam in (0.0, 0.3, 1.0, 5.0):
        query = points.clone().requires_grad_(True)
        value = prior.log_prob(query, lam).sum()
        gradient, = torch.autograd.grad(value, query)
        analytic = prior.score(points, 0.0, lam)
        assert float((gradient - analytic).abs().max()) < 1e-10


def test_tuple_prior_still_takes_the_historical_path():
    from compass.priors import resolve_prior_object
    assert resolve_prior_object((0.0, 1.0), 1) is None
    assert resolve_prior_object([0.0, 1.0], 1) is None
    assert resolve_prior_object(None, 1) is None


def test_prior_object_is_validated():
    from compass.priors import GaussianMixturePrior, resolve_prior_object
    prior = GaussianMixturePrior(
        [0.5, 0.5], torch.tensor([[-1.0], [1.0]], dtype=torch.float64),
        torch.tensor([[[1.0]], [[1.0]]], dtype=torch.float64),
    )
    assert resolve_prior_object(prior, 1) is prior
    with pytest.raises(ValueError):
        resolve_prior_object(prior, 3)
    with pytest.raises(ValueError):
        resolve_prior_object("not a prior", 1)


def test_gaussian_prior_object_matches_the_tuple_formula():
    from compass.priors import GaussianPrior
    prior = GaussianPrior(torch.tensor([0.5]), torch.tensor([[2.0]]))
    y = torch.tensor([[1.5]], dtype=torch.float64)
    lam = 0.8
    expected = -(y - 0.5) / (2.0 + lam ** 2)
    assert float((prior.score(y, 0.0, lam) - expected).abs().max()) < 1e-12


@pytest.mark.parametrize("n", [1, 2])
def test_discovery_at_small_n_does_not_merge_into_the_saddle(n):
    """Regression: at n = 1 the two modes sit ~1.8 local standard deviations
    from the origin, so a generous Mahalanobis merge tolerance swallowed the
    branches the saddle had just spawned and discovery returned *nothing*."""
    model = B.bimodal_1d(n, "symmetric")
    ref = B.reference_1d(model)
    tree = ModeTree(
        ComposedField(DiffusedOracle(model), "gauss_jacobian"),
        make_composer(model), ModeTreeConfig(),
        prior_std=model.prior_cov.diagonal().sqrt(),
    )
    branches = tree.run()
    found = sorted(float(b.global_state) for b in branches)
    truth = sorted(float(m) for m in ref.modes.reshape(-1))
    assert len(found) == 2
    assert max(abs(a - b) for a, b in zip(found, truth)) < 1e-3


def test_saddles_are_never_merge_targets():
    """A saddle has no basin, so nothing may be called a duplicate of it."""
    indefinite = torch.tensor([[-2.0]], dtype=torch.float64)
    definite = torch.tensor([[2.0]], dtype=torch.float64)
    assert not ModeTree._is_mode(indefinite)
    assert ModeTree._is_mode(definite)
    assert not ModeTree._is_mode(None)
