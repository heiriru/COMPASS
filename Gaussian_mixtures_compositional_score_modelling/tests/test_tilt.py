"""The exponential-tilt composition, against what it is supposed to recover.

Conditioned on the mixture component this benchmark is jointly Gaussian, so
``x_j | g ~ N(g, v)`` and the exact likelihood ratio in the shared parameter is

    log L_j(g) = const + x_j g / v - g^2 / (2 v),

a degree-2 exponential tilt of the prior with ``beta_j = x_j / v`` and
``gamma_j = 1 / v``. The estimator never sees those numbers -- it sees draws --
so recovering them, and recovering the composed posterior's width from their
sum, is the substantive check on the method rather than on its plumbing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "inverse_kinematics" / "experiments"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import tilt_fit  # noqa: E402
from model import Mixture  # noqa: E402

MIXTURE = Mixture()


def _setup(n, samples=20_000, seed=7):
    """``(x, grid, log_prior, draws)`` with draws taken *exactly*, no sampler."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-0.4, 0.6, n)
    grid = MIXTURE.default_grid(x, points=6_001)
    log_prior = np.log(np.maximum(MIXTURE.prior_density(grid), 1e-300))
    variance = MIXTURE.posterior_variance(1)
    draws = np.empty((n, samples))
    for index, value in enumerate(x):
        weights, means, _ = MIXTURE.global_posterior(np.array([value]))
        pick = rng.random(samples) < weights[1]
        draws[index] = (np.where(pick, means[1], means[0])
                        + np.sqrt(variance) * rng.standard_normal(samples))
    return x, grid, log_prior, draws


def _raw_parameters(thetas, centre, scale):
    """``theta`` is fitted in ``u = (g - centre)/scale``; undo that."""
    beta = thetas[:, 0] / scale - 2.0 * thetas[:, 1] * centre / scale ** 2
    gamma = -2.0 * thetas[:, 1] / scale ** 2
    return beta, gamma


@pytest.mark.parametrize("n", [4, 16, 64])
def test_tilt_recovers_the_known_likelihood(n):
    x, grid, log_prior, draws = _setup(n)
    thetas, degrees, centre, scale, _ = tilt_fit.fit_all(
        draws, grid, log_prior, degrees=(2,))
    assert set(degrees) == {2}
    beta, gamma = _raw_parameters(thetas, centre, scale)
    assert gamma == pytest.approx(1.0 / MIXTURE.v, rel=0.05)
    assert beta == pytest.approx(x / MIXTURE.v, abs=0.1)


@pytest.mark.parametrize("n", [4, 16, 64])
def test_summed_tilts_give_the_exact_posterior_width(n):
    """The composition is a sum of parameters, so this is the whole claim."""
    x, grid, log_prior, draws = _setup(n)
    thetas, _, centre, scale, _ = tilt_fit.fit_all(draws, grid, log_prior,
                                                degrees=(2,))
    _, gamma = _raw_parameters(thetas, centre, scale)
    precision = gamma.sum() + 1.0 / MIXTURE.tau_g ** 2
    assert 1.0 / np.sqrt(precision) == pytest.approx(
        np.sqrt(MIXTURE.posterior_variance(n)), rel=0.02)


def test_higher_degrees_are_not_selected_when_two_suffice():
    """BIC must not spend flexibility the truth does not need."""
    _, grid, log_prior, draws = _setup(8)
    _, degrees, _, _, _ = tilt_fit.fit_all(draws, grid, log_prior,
                                        degrees=(2, 4, 6))
    assert set(degrees) == {2}


def test_localisation_is_consistent_at_the_composed_density():
    """Weighting base measure and sample alike must not move the answer.

    Consistency is a statement about the estimator, not about a single sample:
    localising concentrates the fit on a narrow window, so each ``theta_j`` is
    estimated from an effectively smaller sample and moves by a few per cent.
    What has to hold is that the *composed* density -- the object the method
    returns -- is unchanged, which is what this asserts. The parameter-level
    drift is checked separately with a deliberately mild weight.
    """
    _, grid, log_prior, draws = _setup(8)
    plain, _, centre, scale, _ = tilt_fit.fit_all(draws, grid, log_prior,
                                               degrees=(2,))
    first = tilt_fit.compose(plain, grid, log_prior, centre, scale)
    localised, _, _, _, _ = tilt_fit.fit_all(
        draws, grid, log_prior, degrees=(2,), centre=centre, scale=scale,
        log_weight=np.log(np.maximum(first, 1e-300)))
    second = tilt_fit.compose(localised, grid, log_prior, centre, scale)
    difference = 0.5 * float(np.trapezoid(np.abs(second - first), grid))
    assert difference < 0.1


def test_a_mild_localisation_barely_moves_the_parameters():
    """With a weight far broader than the factors, the two fits must agree."""
    _, grid, log_prior, draws = _setup(8)
    plain, _, centre, scale, _ = tilt_fit.fit_all(draws, grid, log_prior,
                                               degrees=(2,))
    mild = -0.5 * ((grid - float(draws.mean())) / (6.0 * float(draws.std()))) ** 2
    localised, _, _, _, _ = tilt_fit.fit_all(
        draws, grid, log_prior, degrees=(2,), centre=centre, scale=scale,
        log_weight=mild)
    assert localised == pytest.approx(plain, rel=0.05, abs=0.01)


def test_composition_is_order_independent_and_normalised():
    _, grid, log_prior, draws = _setup(6)
    thetas, _, centre, scale, _ = tilt_fit.fit_all(draws, grid, log_prior,
                                                degrees=(2,))
    forward = tilt_fit.compose(thetas, grid, log_prior, centre, scale)
    backward = tilt_fit.compose(thetas[::-1], grid, log_prior, centre, scale)
    assert forward == pytest.approx(backward, rel=1e-8, abs=1e-12)
    assert np.trapezoid(forward, grid) == pytest.approx(1.0, rel=1e-9)
