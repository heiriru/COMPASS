"""The closed forms, checked against brute force.

Everything downstream is graded against :mod:`model`, so if these expressions
are wrong the whole benchmark reports a confident, self-consistent lie. Each
test here re-derives the quantity a second, independent way -- numerical
integration of the unnormalised joint, or a large Monte Carlo draw from the
generative model -- rather than restating the algebra.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import datasets  # noqa: E402
from model import Mixture  # noqa: E402

MIXTURE = Mixture()
UNBALANCED = Mixture(weights=(0.3, 0.7))


def brute_force_global(mixture, x, grid):
    """``p(g | x)`` by integrating the locals out numerically, on a grid.

    ``p(g | x) ∝ p(g) prod_i int p(l|g) p(x_i|l) dl``, and the inner integral is
    the one closed form this benchmark is allowed to assume (a convolution of
    two Gaussians), because it is a textbook identity rather than part of what
    is being tested.
    """
    grid = np.asarray(grid, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).ravel()
    log_likelihood = np.zeros_like(grid)
    for value in x:
        log_likelihood += -0.5 * (value - grid) ** 2 / mixture.v
    log_posterior = log_likelihood + np.log(mixture.prior_density(grid))
    log_posterior -= log_posterior.max()
    density = np.exp(log_posterior)
    return density / np.trapezoid(density, grid)


@pytest.mark.parametrize("mixture", [MIXTURE, UNBALANCED])
@pytest.mark.parametrize("n", [1, 2, 5, 16])
def test_global_density_matches_numerical_integration(mixture, n):
    rng = np.random.default_rng(4)
    x = rng.normal(0.5, 1.2, n)
    grid = np.linspace(-6.0, 6.0, 40_001)
    assert np.allclose(
        mixture.global_density(grid, x), brute_force_global(mixture, x, grid),
        atol=1e-8, rtol=1e-6,
    )


@pytest.mark.parametrize("mixture", [MIXTURE, UNBALANCED])
def test_branch_log_odds_matches_the_evidence_integral(mixture):
    """The shortcut :mod:`datasets` inverts must equal the full computation."""
    rng = np.random.default_rng(11)
    for n in (1, 3, 8, 32):
        x = rng.normal(-0.3, 1.5, n)
        weights = mixture.branch_weights(x)
        assert mixture.branch_log_odds(x) == pytest.approx(
            float(np.log(weights[1] / weights[0])), abs=1e-9
        )


def test_prescribed_branch_mass_is_hit_at_every_n():
    rng = np.random.default_rng(5)
    for n in (1, 2, 4, 8, 16, 32):
        for mass in (0.5, 0.6, 0.75, 0.9):
            data = datasets.make_dataset("asymmetric", n, rng,
                                         branch_mass=mass)
            weights = MIXTURE.branch_weights(data["x"])
            assert weights[1] == pytest.approx(mass, abs=1e-9)


def test_symmetric_and_antisymmetric_targets_are_exactly_symmetric():
    rng = np.random.default_rng(6)
    grid = np.linspace(-6.0, 6.0, 4001)
    for regime in ("symmetric", "antisymmetric"):
        for n in (1, 2, 5, 8):
            data = datasets.make_dataset(regime, n, rng)
            assert data["x"].sum() == pytest.approx(0.0, abs=1e-12)
            density = MIXTURE.global_density(grid, data["x"])
            assert np.allclose(density, density[::-1], atol=1e-12)


def test_antisymmetric_observations_are_individually_one_sided():
    """The point of the regime: each term is lopsided, only the product is not."""
    rng = np.random.default_rng(7)
    data = datasets.make_dataset("antisymmetric", 8, rng)
    per_observation = np.array([
        MIXTURE.branch_weights([value])[1] for value in data["x"]
    ])
    assert per_observation.min() < 0.02
    assert per_observation.max() > 0.98


def test_joint_moments_match_a_monte_carlo_draw_from_the_posterior():
    """Mean vector and covariance of the two-branch mixture, sampled."""
    rng = np.random.default_rng(3)
    x = np.array([0.4, -1.1, 2.0])
    weights, _, _ = MIXTURE.global_posterior(x)
    covariance = MIXTURE.branch_covariance(len(x))
    count = 400_000
    branch = rng.random(count) < weights[1]
    draws = np.empty((count, len(x) + 1))
    for index in (0, 1):
        rows = branch == bool(index)
        draws[rows] = rng.multivariate_normal(
            MIXTURE.branch_mean_vector(x, index), covariance, int(rows.sum())
        )
    mean, sd = MIXTURE.global_moments(x)
    assert draws[:, 0].mean() == pytest.approx(mean, abs=0.01)
    assert draws[:, 0].std() == pytest.approx(sd, abs=0.01)
    for index in range(len(x)):
        local_mean, local_sd = MIXTURE.local_moments(x, index)
        assert draws[:, index + 1].mean() == pytest.approx(local_mean, abs=0.01)
        assert draws[:, index + 1].std() == pytest.approx(local_sd, abs=0.01)


def test_simulator_matches_the_declared_marginals():
    """``x | g ~ N(g, v)`` and the prior sd used for every Gaussian API."""
    rng = np.random.default_rng(9)
    rows = MIXTURE.simulate(400_000, rng)
    assert rows[:, 0].std() == pytest.approx(MIXTURE.global_prior_std, abs=0.01)
    assert rows[:, 1].std() == pytest.approx(MIXTURE.local_prior_std, abs=0.01)
    residual = rows[:, 2] - rows[:, 0]
    assert residual.std() ** 2 == pytest.approx(MIXTURE.v, abs=0.02)


def test_diffused_prior_score_matches_a_numerical_derivative():
    grid = np.linspace(-6.0, 6.0, 200_001)
    for variance in (0.0, 0.09, 4.0):
        total = MIXTURE.tau_g ** 2 + variance
        density = np.zeros_like(grid)
        for weight, centre in zip(MIXTURE.weights, MIXTURE.components):
            density += weight * np.exp(-0.5 * (grid - centre) ** 2 / total)
        numerical = np.gradient(np.log(density), grid)
        analytic = MIXTURE.diffused_prior_score(grid, variance)
        interior = slice(10, -10)
        assert np.allclose(numerical[interior], analytic[interior], atol=1e-4)


def test_joint_log_density_integrates_to_one_at_n_equals_one():
    x = np.array([1.3])
    axis = np.linspace(-8.0, 8.0, 1201)
    mesh = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
    density = np.exp(MIXTURE.joint_log_density(mesh.reshape(-1, 2), x))
    density = density.reshape(len(axis), len(axis))
    mass = np.trapezoid(np.trapezoid(density, axis, axis=1), axis)
    assert mass == pytest.approx(1.0, abs=1e-6)
    # Its ``g`` marginal must be the closed-form shared posterior.
    marginal = np.trapezoid(density, axis, axis=1)
    assert np.allclose(marginal, MIXTURE.global_density(axis, x), atol=1e-6)
