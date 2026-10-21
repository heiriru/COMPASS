"""The oracle score, checked three ways.

If the oracle were wrong, every "perfect score" number in this directory would
be wrong in the same direction and look self-consistent. So it is checked
against (1) a numerical derivative of the exact diffused density, (2) the
independent NumPy conditioning in :func:`oracle.exact_conditional`, and (3) the
closed-form single-observation posterior of :mod:`model`, which was itself
checked against brute force.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import model as M  # noqa: E402
import oracle  # noqa: E402
from model import Mixture  # noqa: E402

MIXTURE = Mixture()
MASKS = {
    "posterior": [0.0, 0.0, 1.0],      # p(g, l | x)
    "local": [1.0, 0.0, 1.0],          # p(l | g, x)
    "shared": [0.0, 1.0, 1.0],         # p(g | l, x)
    "joint": [0.0, 0.0, 0.0],          # p(g, l, x)
}


def mixture_log_density(values, weights, means, covariances):
    terms = []
    for weight, mean, covariance in zip(weights, means, covariances):
        delta = values - mean
        sign, log_determinant = np.linalg.slogdet(covariance)
        quadratic = np.einsum("ij,jk,ik->i", delta,
                              np.linalg.inv(covariance), delta)
        terms.append(np.log(weight) - 0.5 * (
            len(mean) * np.log(2.0 * np.pi) + log_determinant + quadratic))
    stacked = np.stack(terms)
    peak = stacked.max(axis=0)
    return peak + np.log(np.exp(stacked - peak).sum(axis=0))


@pytest.mark.parametrize("name", sorted(MASKS))
@pytest.mark.parametrize("time", [0.02, 0.2, 0.7])
def test_oracle_score_matches_a_numerical_derivative(name, time):
    """The returned score must be d/dy log p_t(y_latent | y_observed)."""
    network = oracle.OracleModel(MIXTURE)
    mask = torch.tensor(MASKS[name])
    latent = [i for i, value in enumerate(MASKS[name]) if value == 0.0]
    observed = [i for i, value in enumerate(MASKS[name]) if value == 1.0]
    variance = float(network.sde.lambda_t(torch.tensor([[time]])).square())

    rng = np.random.default_rng(3)
    states = rng.normal(0.0, 2.0, size=(24, M.NODES))
    weights, means, covariances = oracle.exact_conditional(
        MIXTURE, states[0], observed, latent, variance
    )

    x = torch.as_tensor(states, dtype=torch.float64)
    t = torch.full((len(states), 1), time, dtype=torch.float64)
    got = network.model(x=x, t=t, c=mask.expand(len(states), M.NODES))
    got = network.output_scale_function(t, got).numpy()

    step = 1e-5
    for row, state in enumerate(states):
        # Rows differ in their observed values, so the conditional is re-derived
        # per row -- which is the point of the check.
        weights, means, covariances = oracle.exact_conditional(
            MIXTURE, state, observed, latent, variance
        )
        for index, coordinate in enumerate(latent):
            plus, minus = state[latent].copy(), state[latent].copy()
            plus[index] += step
            minus[index] -= step
            derivative = (
                mixture_log_density(plus[None, :], weights, means, covariances)
                - mixture_log_density(minus[None, :], weights, means, covariances)
            )[0] / (2.0 * step)
            assert got[row, coordinate] == pytest.approx(derivative, abs=2e-4)
        for coordinate in observed:
            assert got[row, coordinate] == 0.0


def test_oracle_posterior_matches_the_closed_form_at_n_equals_one():
    """``p(g, l | x)`` from the oracle is the closed-form single-obs posterior."""
    x = 1.4
    weights, means, covariances = oracle.exact_conditional(
        MIXTURE, [0.0, 0.0, x], observed=[2], latent=[0, 1], variance=0.0
    )
    assert np.allclose(weights, MIXTURE.branch_weights([x]))
    for index in (0, 1):
        assert np.allclose(means[index],
                           MIXTURE.branch_mean_vector([x], index))
        assert np.allclose(covariances[index], MIXTURE.branch_covariance(1))


def test_oracle_sampler_reproduces_the_exact_posterior():
    """End to end: the ordinary sampler, on the oracle, at n = 1.

    This is the check that the whole interface -- shapes, the ``raw / sigma``
    convention, the zeros in the observed columns -- is right, because a slip in
    any of them moves this sample off the closed form.
    """
    import metrics

    network = oracle.OracleModel(MIXTURE)
    x = torch.tensor([[1.4]], dtype=torch.float32)
    draws = network.sample(
        x=x, num_samples=8_000, timesteps=100, method="dpm", order=2,
        churn_eta=2.0, corrector_steps=0, final_corrector_steps=0,
        terminal_denoise=True, device="cpu", verbose=False,
    ).detach().cpu().numpy()
    shared = draws[0, :, 0]
    reference = np.array([1.4])
    mean, sd = MIXTURE.global_moments(reference)
    assert shared.mean() == pytest.approx(mean, abs=0.06)
    assert metrics.wasserstein(shared, MIXTURE, reference) / sd < 0.06
