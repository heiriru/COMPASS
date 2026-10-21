"""The parts of the composition rules that do not need a trained network.

The sampler-driven arms are graded by the sweep itself; what is checked here is
the machinery around them, where a silent error would be attributed to a
composition rule instead of to the code holding it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import methods  # noqa: E402
import metrics  # noqa: E402
from model import Mixture  # noqa: E402

MIXTURE = Mixture()


def test_shared_marginal_reproduces_an_analytic_product():
    """Two Gaussian 'posteriors' whose product with a prior is closed form."""
    grid = np.linspace(-8.0, 8.0, 4001)
    rng = np.random.default_rng(0)
    prior_variance, prior_mean = 4.0, 0.0
    prior = np.exp(-0.5 * (grid - prior_mean) ** 2 / prior_variance)
    prior /= np.trapezoid(prior, grid)
    means, variance = np.array([0.6, -0.4]), 0.5
    draws = np.stack([
        mean + np.sqrt(variance) * rng.standard_normal(200_000) for mean in means
    ])
    density, _ = methods.shared_marginal(draws, grid, prior, bandwidth=0.15)

    # prod_i N(g; m_i, s^2) / N(g; 0, p^2) is Gaussian with precision
    # 2/s^2 - 1/p^2 and mean (sum m_i / s^2) / that precision.
    precision = len(means) / variance - 1.0 / prior_variance
    expected_mean = means.sum() / variance / precision
    got_mean = float(np.trapezoid(grid * density, grid))
    got_variance = float(np.trapezoid((grid - got_mean) ** 2 * density, grid))
    assert got_mean == pytest.approx(expected_mean, abs=0.02)
    # The KDE bandwidth adds to the variance of every factor, so the product is
    # wider than the exact one by a known, small amount; 10% is the tolerance
    # that separates "smoothing" from "wrong".
    assert got_variance == pytest.approx(1.0 / precision, rel=0.10)


def test_inverse_cdf_sample_recovers_the_density_it_was_given():
    grid = np.linspace(-6.0, 6.0, 2001)
    density = MIXTURE.global_density(grid, np.array([0.3, -0.2]))
    draws = methods.inverse_cdf_sample(grid, density, 200_000,
                                       np.random.default_rng(1))
    mean, sd = MIXTURE.global_moments(np.array([0.3, -0.2]))
    assert draws.mean() == pytest.approx(mean, abs=0.02)
    assert draws.std() == pytest.approx(sd, abs=0.02)


def test_mixture_prior_patch_is_exact_and_is_removed_again():
    """The patched score must equal :meth:`Mixture.diffused_prior_score`.

    The sampler's own signature is exercised through a stand-in that carries the
    two attributes the patched function reads -- the hierarchy and the SDE --
    so the shape contract is checked without building a sampler.
    """
    from compass.MultiObsSampler import MultiObsSampler
    from compass.SDE import VESDE

    original = MultiObsSampler._diffused_gaussian_prior_score

    class Stand_in:
        hierarchy = [0]
        sde = VESDE(sigma=15.0)

    theta = torch.linspace(-5.0, 5.0, 101).reshape(1, -1, 1)
    with methods.mixture_prior_score(MIXTURE) as enabled:
        assert enabled
        assert MultiObsSampler._diffused_gaussian_prior_score is not original
        for time in (0.02, 0.3, 0.9):
            t = torch.tensor([[time]])
            got = MultiObsSampler._diffused_gaussian_prior_score(
                Stand_in(), theta, t
            )
            variance = float(Stand_in.sde.lambda_t(t).square())
            want = MIXTURE.diffused_prior_score(
                theta.numpy().reshape(-1), variance
            )
            assert got.shape == theta.shape
            assert np.allclose(got.numpy().reshape(-1), want, atol=1e-5)
    assert MultiObsSampler._diffused_gaussian_prior_score is original

    with methods.mixture_prior_score(MIXTURE, enabled=False) as enabled:
        assert not enabled
        assert MultiObsSampler._diffused_gaussian_prior_score is original


def test_evaluate_scores_exact_draws_near_zero():
    """The grader must report ~0 error on draws from the exact posterior.

    This is the calibration of the measurement itself: if ``evaluate`` reported
    a systematic error here, every method's number would inherit it.
    """
    import model as M

    x = np.array([0.7, -0.2, 1.1, 0.4])
    rng = np.random.default_rng(2)
    exact = metrics.sample_exact(MIXTURE, x, 40_000, rng)
    samples = np.zeros((len(x), 40_000, M.NODES))
    samples[:, :, M.HIERARCHY[0]] = exact[:, 0][None, :]
    samples[:, :, M.LOCAL_INDICES[0]] = exact[:, 1:].T
    samples[:, :, M.OBSERVED_INDICES[0]] = x[:, None]
    scored = metrics.evaluate(samples, MIXTURE, x)
    assert scored["w1_over_sigma"] < 0.02
    assert abs(scored["branch_mass_error"]) < 0.02
    assert scored["width_ratio"] == pytest.approx(1.0, abs=0.02)
    assert scored["local_w1_over_sigma"] < 0.02
