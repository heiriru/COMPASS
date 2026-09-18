"""Milestones C, D, E, F, G against exact references."""

import math

import pytest
import torch

from modetree import benchmarks as B
from modetree.composition import CleanComposer, OracleRowSource, make_composer
from modetree.energy import (
    integrate_segment, mode_energies, reconcile, gauss_legendre,
)
from modetree.laplace import assemble, branch_laplace, laplace_plus
from modetree.refine import refine_mode, deduplicate, mahalanobis


# --- Milestone C ------------------------------------------------------------

@pytest.mark.parametrize("n", (1, 8, 100))
@pytest.mark.parametrize("offset", (0.0, 0.4, -0.9))
def test_refine_converges_from_nearby_seeds(n, offset):
    model = B.bimodal_1d(n, "symmetric")
    ref = B.reference_1d(model)
    composer = CleanComposer(OracleRowSource(model))
    seed = ref.modes[0] + offset
    result = refine_mode(composer, seed, tol=1e-11, max_iter=200)
    assert result.converged
    assert result.score_norm < 1e-7
    assert result.diagnostics["is_local_maximum"]
    # the reference grid mode is itself only parabola-accurate, so compare to
    # the score root the grid is approximating, not to the grid point
    assert float((result.theta - ref.modes[0]).abs()) < 1e-4


@pytest.mark.parametrize("n", (1, 8, 32))
def test_refine_hierarchical_arrowhead(n):
    model = B.hierarchical(n, d_g=2, d_l=1, modes=2)
    composer = CleanComposer(OracleRowSource(model))
    torch.manual_seed(0)
    seed = torch.zeros(composer.dim, dtype=torch.float64)
    seed[:2] = torch.tensor([2.0, 2.0], dtype=torch.float64)
    result = refine_mode(composer, seed, tol=1e-10, max_iter=300)
    assert result.score_norm < 1e-7
    P = result.blocks.global_precision()
    assert bool((torch.linalg.eigvalsh(P) > 0).all())
    # Schur precision == global block of the inverse full negative Hessian
    dense = -composer.dense_hessian(result.theta)
    reference = torch.linalg.inv(torch.linalg.inv(dense)[:2, :2])
    assert float((P - reference).abs().max()) < 1e-6 * float(P.abs().max())


def test_deduplicate_needs_mahalanobis_not_euclidean():
    model = B.bimodal_1d(4, "symmetric")
    composer = CleanComposer(OracleRowSource(model))
    ref = B.reference_1d(model)
    a = refine_mode(composer, ref.modes[0] + 0.2, tol=1e-11)
    b = refine_mode(composer, ref.modes[0] - 0.2, tol=1e-11)
    c = refine_mode(composer, ref.modes[1], tol=1e-11)
    unique = deduplicate([a, b, c], lambda r: r.blocks.global_precision(), tol=0.5)
    assert len(unique) == 2


# --- Milestone D ------------------------------------------------------------

@pytest.mark.parametrize("n", (1, 8, 100))
def test_line_integral_matches_exact_log_density(n):
    model = B.bimodal_1d(n, "symmetric")
    composer = CleanComposer(OracleRowSource(model))
    a = torch.tensor([-2.0], dtype=torch.float64)
    b = torch.tensor([1.3], dtype=torch.float64)
    exact = float(model.global_log_prob(b) - model.global_log_prob(a))
    got = integrate_segment(composer.score, a, b, adaptive=True, tol=1e-12)
    assert abs(got.value - exact) < 1e-8


def test_quadrature_converges_with_refinement():
    model = B.four_mode_2d(16, "unequal_covariance")
    composer = CleanComposer(OracleRowSource(model))
    a = torch.tensor([2.0, 2.0], dtype=torch.float64)
    b = torch.tensor([-2.0, -2.0], dtype=torch.float64)
    exact = float(model.global_log_prob(b) - model.global_log_prob(a))
    errors = []
    for panels in (1, 2, 4, 8, 16):
        got = integrate_segment(composer.score, a, b, q=4, adaptive=False,
                                panel_ladder=(panels,))
        errors.append(abs(got.value - exact))
    assert errors[-1] < 1e-7
    assert errors[-1] < errors[0]


def test_empty_valley_symmetric_energies_close():
    """The regime that defeats order-only refinement: a huge integrand and a
    zero integral.  Panel refinement must still return a tie at n = 100."""
    model = B.empty_valley_1d(100)
    composer = make_composer(model)
    ref = B.reference_1d(model, -20, 20)
    results = [refine_mode(composer, m, tol=1e-11) for m in ref.modes]
    graph = mode_energies(composer.score, [r.theta for r in results])
    assert abs(float(graph.energies[1] - graph.energies[0])) < 1e-3


# --- Milestone E ------------------------------------------------------------

def test_cycle_residuals_vanish_for_a_conservative_field():
    model = B.four_mode_2d(8, "unequal_weights")
    composer = CleanComposer(OracleRowSource(model))
    ref = B.reference_2d(model, (-7, -7), (7, 7), points=401)
    modes = [m for m in ref.modes[:4]]
    graph = mode_energies(composer.score, modes, tol=1e-12)
    assert graph.cycle_residuals
    assert max(abs(v) for v in graph.cycle_residuals.values()) < 1e-6


def test_energy_differences_match_exact_heights_2d():
    model = B.four_mode_2d(8, "unequal_weights")
    composer = CleanComposer(OracleRowSource(model))
    ref = B.reference_2d(model, (-7, -7), (7, 7), points=601)
    seeds = ref.modes[:4]
    results = [refine_mode(composer, s, tol=1e-11, max_iter=300) for s in seeds]
    modes = [r.theta for r in results]
    graph = mode_energies(composer.score, modes, tol=1e-12)
    exact = torch.stack([model.global_log_prob(m) for m in modes]).reshape(-1)
    exact = exact - exact[0]
    got = graph.energies - graph.energies[0]
    assert float((got - exact).abs().max()) < 1e-3


def test_reconcile_recovers_exact_potentials():
    truth = torch.tensor([0.0, 1.5, -0.7, 2.2], dtype=torch.float64)
    means = {(a, b): float(truth[b] - truth[a])
             for a in range(4) for b in range(a + 1, 4)}
    sigma = {k: 1e-3 for k in means}
    energies, residual = reconcile(means, sigma, 4)
    assert float((energies - truth).abs().max()) < 1e-6
    assert residual < 1e-6


# --- Milestone F / G --------------------------------------------------------

@pytest.mark.parametrize("n", (1, 8, 32, 100))
def test_laplace_masses_1d(n):
    """Structured Laplace weights against exact grid basin masses."""
    model = B.tilted_two_mode_1d(n, delta=0.5)
    ref = B.reference_1d(model, -12, 12)
    composer = CleanComposer(OracleRowSource(model))
    results = [refine_mode(composer, m, tol=1e-11, max_iter=200) for m in ref.modes]
    posterior = assemble(composer, results, laplace_plus_samples=0)
    truth = ref.mode_masses / ref.mode_masses.sum()
    tv = 0.5 * float((posterior.weights - truth).abs().sum())
    # Laplace is an approximation; it must be good, not exact.  n = 1 is the
    # hardest because the mode is widest relative to its curvature.
    assert tv < (0.05 if n == 1 else 0.01)


def test_map_and_mass_branches_differ():
    """Section 13: the higher peak and the heavier mode must not be conflated."""
    model = B.map_vs_mass_1d(32)
    ref = B.reference_1d(model, -12, 12)
    composer = CleanComposer(OracleRowSource(model))
    results = [refine_mode(composer, m, tol=1e-11, max_iter=200) for m in ref.modes]
    posterior = assemble(composer, results, laplace_plus_samples=0)
    assert posterior.map_index != posterior.mass_index
    assert float(posterior.map_global) > 0     # the narrow high peak is at +mu
    heavier = posterior.branches[posterior.mass_index].global_state
    assert float(heavier) < 0                  # the broad mode carries the mass


def test_laplace_plus_is_neutral_on_a_gaussian_mode():
    model = B.bimodal_1d(64, "separated")
    ref = B.reference_1d(model, -16, 16)
    composer = CleanComposer(OracleRowSource(model))
    results = [refine_mode(composer, m, tol=1e-11) for m in ref.modes]
    posterior = assemble(composer, results, laplace_plus_samples=32)
    for branch in posterior.branches:
        assert abs(branch.log_correction) < 1e-2
        assert branch.diagnostics["laplace_plus"]["ess_fraction"] > 0.9


def test_laplace_plus_helps_a_skewed_mode():
    """A deliberately non-Gaussian mode: Laplace+ must reduce the mass error."""
    n = 2
    model = B.tilted_two_mode_1d(n, delta=0.5, s1=0.5, s2=1.6, mu=1.6, tau=2.0)
    ref = B.reference_1d(model, -10, 10)
    composer = CleanComposer(OracleRowSource(model))
    results = [refine_mode(composer, m, tol=1e-11) for m in ref.modes]
    truth = ref.mode_masses / ref.mode_masses.sum()
    posterior = assemble(composer, results, laplace_plus_samples=64)
    tv = 0.5 * float((posterior.weights - truth).abs().sum())
    tv_plus = 0.5 * float((posterior.weights_plus - truth).abs().sum())
    assert tv_plus < tv


def test_log_determinant_uses_block_structure():
    model = B.hierarchical(8, d_g=2, d_l=1, modes=2)
    composer = CleanComposer(OracleRowSource(model))
    seed = torch.zeros(composer.dim, dtype=torch.float64)
    seed[:2] = 2.0
    result = refine_mode(composer, seed, tol=1e-10, max_iter=300)
    blocks = result.blocks
    dense = -composer.dense_hessian(result.theta)
    reference = float(torch.linalg.slogdet(dense).logabsdet)
    assert abs(float(blocks.log_det_negative_hessian()) - reference) < 1e-8
