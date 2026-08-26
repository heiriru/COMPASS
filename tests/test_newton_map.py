"""Tests for `MultiObsSampler.newton_map_estimate` (arrow-Newton MAP ascent).

These tests deliberately start every optimizer from the **prior mean**, not from
the analytic answer. The accuracy tests in `test_hierarchical_map.py` initialize
at `expected`, so they verify only that the iteration does not walk away from the
mode -- which is why they pass at n=50 while
`tutorials/plot_local_vs_global_joint_map_validation.py` records 0.73 sigma at
n=50 and 43.5 sigma at n=200 on the same estimator.
"""

import pytest
import torch

from compass.MultiObsSampler import MultiObsSampler
from compass.SDE import VESDE, VPSDE

from test_hierarchical_map import (
    MockSBIm, SharedLocalGaussianScore, analytic_shared_local_map, make_joint,
)

# Widths used by tutorials/plot_local_vs_global_joint_map_validation.py, where
# the Tweedie ascent degrades with the observation count.
NARROW = dict(sigma_global=0.3, sigma_local=0.4, sigma_x=0.2)


def narrow_score_class():
    return type("NarrowSharedLocalScore", (SharedLocalGaussianScore,), dict(NARROW))


def simulate(n, score_cls, seed=0):
    generator = torch.Generator().manual_seed(seed)
    true_global = score_cls.sigma_global * torch.randn(1, generator=generator)
    true_local = score_cls.sigma_local * torch.randn(n, generator=generator)
    noise = score_cls.sigma_x * torch.randn(n, generator=generator)
    return true_global + true_local + noise


def analytic_map(observations, score_cls):
    """Closed-form arrowhead mode; `analytic_shared_local_map` with custom widths."""
    n = len(observations)
    sg, sl, sx = score_cls.sigma_global, score_cls.sigma_local, score_cls.sigma_x
    precision = torch.zeros(n + 1, n + 1, dtype=torch.float64)
    precision[0, 0] = 1 / sg**2 + n / sx**2
    precision[1:, 1:] = torch.eye(n, dtype=torch.float64) * (1 / sl**2 + 1 / sx**2)
    precision[0, 1:] = 1 / sx**2
    precision[1:, 0] = 1 / sx**2
    values = observations.to(torch.float64)
    rhs = torch.cat([(values.sum() / sx**2).reshape(1), values / sx**2])
    covariance = torch.linalg.inv(precision)
    return (covariance @ rhs).float(), covariance.diag().sqrt().float()


def run(method, n, score_cls, sde=None, denoise_clamp=None, timesteps=100, **kwargs):
    observations = simulate(n, score_cls)
    expected, posterior_std = analytic_map(observations, score_cls)
    initial = make_joint(observations, 0.0, torch.zeros(n))
    sampler = MultiObsSampler(MockSBIm(score_cls, 3, sde=sde))
    shared = dict(
        data=initial, condition_mask=torch.tensor([0.0, 0.0, 1.0]), init=initial,
        hierarchy=[0], prior=([0.0], [score_cls.sigma_global]),
        correction="uncorrected", sigma_start=2.0 * score_cls.sigma_global,
        timesteps=timesteps, eps=1e-3, denoise_clamp=denoise_clamp, device="cpu",
    )
    shared.update(kwargs)
    if method == "tweedie":
        result = sampler.map_estimate(iterations_per_level=3, **shared)[:, 0]
    else:
        result = sampler.newton_map_estimate(**shared)[:, 0]
    inferred = torch.cat([result[:1, 0].flatten(), result[:, 1]])
    deviation = (inferred - expected) / posterior_std
    return sampler, result, deviation


# --------------------------------------------------------------------------
# Accuracy: the property the existing suite cannot see, because it starts at
# the answer.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 10, 50, 200])
def test_newton_map_recovers_analytic_mode_from_prior_mean(n):
    """Error must stay flat in n -- growth with n is the defect being fixed."""
    _, result, deviation = run("newton", n, narrow_score_class())
    assert float(deviation.abs().max()) < 0.05
    assert float((result[:, 0] - result[0, 0]).abs().max()) <= 1e-6


@pytest.mark.parametrize("n", [2, 10, 50])
def test_newton_map_is_not_worse_than_tweedie_ascent(n):
    """Like-for-like A/B: identical setup, identical fixed point, different step.

    Both arms keep `denoise_clamp=5.0`, the historical `map_estimate` default,
    because the Tweedie ascent relies on it to stay finite here (see
    `test_tweedie_ascent_depends_on_the_denoise_clamp`).
    """
    score_cls = narrow_score_class()
    _, _, newton = run("newton", n, score_cls, denoise_clamp=5.0)
    _, _, tweedie = run("tweedie", n, score_cls, denoise_clamp=5.0)
    assert float(newton.abs().max()) <= float(tweedie.abs().max()) + 1e-6


def test_newton_map_uses_far_fewer_score_evaluations():
    """A correctly preconditioned step converges instead of exhausting its budget."""
    score_cls = narrow_score_class()
    newton_sampler, _, _ = run("newton", 25, score_cls, denoise_clamp=5.0)
    tweedie_sampler, _, _ = run("tweedie", 25, score_cls, denoise_clamp=5.0)
    assert newton_sampler.score_network_calls * 5 < tweedie_sampler.score_network_calls


def test_tweedie_ascent_depends_on_the_denoise_clamp_but_newton_does_not():
    """The clamp is load-bearing for the old step and merely optional for the new.

    Without it the Tweedie ascent either diverges to a non-finite score or lands
    far from the mode; the arrow-Newton step converges either way, because the
    trust region and the positive-definite curvature do that job instead.
    """
    score_cls = narrow_score_class()
    _, _, newton = run("newton", 25, score_cls, denoise_clamp=None)
    assert float(newton.abs().max()) < 0.05

    try:
        _, _, tweedie = run("tweedie", 25, score_cls, denoise_clamp=None)
    except RuntimeError as error:
        assert "non-finite" in str(error)
    else:
        assert float(tweedie.abs().max()) > 0.05


@pytest.mark.parametrize("sde", [VESDE(sigma=25.0), VPSDE()])
def test_newton_map_supports_both_sdes(sde):
    _, _, deviation = run("newton", 10, narrow_score_class(), sde=sde)
    assert float(deviation.abs().max()) < 0.05


def test_newton_map_matches_wide_prior_regime():
    """The widths the existing suite uses must not regress."""
    _, _, deviation = run("newton", 50, SharedLocalGaussianScore)
    assert float(deviation.abs().max()) < 0.05


# --------------------------------------------------------------------------
# The clamp-pinning failure mode.
# --------------------------------------------------------------------------

def test_newton_map_does_not_pin_at_the_denoise_clamp():
    """The trust region must keep the iterate off the clamp box edge."""
    score_cls = narrow_score_class()
    _, result, deviation = run("newton", 200, score_cls, denoise_clamp=5.0)
    edge = -5.0 * score_cls.sigma_global
    assert abs(float(result[0, 0]) - edge) > 1e-3
    assert float(deviation.abs().max()) < 0.05


# --------------------------------------------------------------------------
# Structural guarantees and reductions.
# --------------------------------------------------------------------------

def test_all_global_hierarchy_is_supported():
    """With no local latents the arrow system collapses to a plain Newton solve."""
    score_cls = narrow_score_class()
    observations = simulate(3, score_cls)
    sg, sl, sx = score_cls.sigma_global, score_cls.sigma_local, score_cls.sigma_x
    precision = torch.tensor([
        [1 / sg**2 + len(observations) / sx**2, len(observations) / sx**2],
        [len(observations) / sx**2, 1 / sl**2 + len(observations) / sx**2],
    ])
    expected = torch.linalg.solve(
        precision, torch.full((2,), observations.sum() / sx**2)
    )
    initial = make_joint(observations, 0.0, torch.zeros(len(observations)))
    sampler = MultiObsSampler(MockSBIm(score_cls, 3))
    result = sampler.newton_map_estimate(
        initial, torch.tensor([0.0, 0.0, 1.0]), init=initial,
        hierarchy=[0, 1], prior=([0.0, 0.0], [sg, sl]),
        correction="uncorrected", sigma_start=0.6, timesteps=100, device="cpu",
    )[:, 0]
    assert torch.allclose(result[0, :2], expected, atol=8e-3)
    assert float((result[:, :2] - result[:1, :2]).abs().max()) <= 1e-6


def test_single_observation_is_finite_and_synchronized():
    score_cls = narrow_score_class()
    observations = simulate(1, score_cls)
    initial = make_joint(observations, 0.0, torch.zeros(1))
    sampler = MultiObsSampler(MockSBIm(score_cls, 3))
    result = sampler.newton_map_estimate(
        initial, torch.tensor([0.0, 0.0, 1.0]), init=initial, hierarchy=[0],
        prior=([0.0], [score_cls.sigma_global]), correction="uncorrected",
        sigma_start=0.6, timesteps=50, device="cpu",
    )
    assert torch.isfinite(result).all()
    assert torch.equal(result[:, 0, 2], observations)


def test_observed_columns_are_never_modified():
    score_cls = narrow_score_class()
    observations = simulate(8, score_cls)
    _, result, _ = run("newton", 8, score_cls)
    assert torch.equal(result[:, 2], observations)


def test_jacobian_curvature_matches_a_finite_difference_reference():
    """The JVP block must equal a central difference of the raw row scores."""
    score_cls = narrow_score_class()
    observations = simulate(4, score_cls)
    sampler = MultiObsSampler(MockSBIm(score_cls, 3))
    sampler.hierarchy = [0]
    sampler.local_latent_indices = [1]
    mask = torch.tensor([0.0, 0.0, 1.0]).repeat(4, 1).unsqueeze(1)
    z = make_joint(observations, 0.3, torch.full((4,), -0.2)).unsqueeze(1)
    t = torch.tensor([[0.4]])

    jacobian = sampler._row_score_jacobian(z, t, mask, [0, 1])
    # The model evaluates in float32, so the central difference carries ~1e-3
    # relative truncation noise; the JVP is the exact side of this comparison.
    step = 1e-3
    for column, feature in enumerate([0, 1]):
        shift = torch.zeros_like(z)
        shift[:, :, feature] = step
        plus = sampler._raw_row_scores(z + shift, t, mask)[:, :, [0, 1]]
        minus = sampler._raw_row_scores(z - shift, t, mask)[:, :, [0, 1]]
        reference = ((plus - minus) / (2 * step)).to(torch.float64)
        assert torch.allclose(
            jacobian[..., column], reference, rtol=2e-2, atol=1e-3
        )


def test_richardson_extrapolation_is_reported_and_small():
    """The extrapolation is a bias correction, not a rescue -- it must stay small."""
    sampler, _, _ = run("newton", 10, narrow_score_class())
    shift = sampler.map_diagnostics["richardson_shift"]
    assert 0.0 <= shift < 0.05


def test_diagnostics_report_convergence():
    sampler, _, _ = run("newton", 10, narrow_score_class(), timesteps=50)
    diagnostics = sampler.map_diagnostics
    assert diagnostics["levels"] == 50
    # Every level should reach its fixed point, not exhaust its iteration budget.
    assert diagnostics["converged_levels"] == 50
    # Steps that fail the merit safeguard should be rare; they are float noise
    # near the fixed point, not divergence.
    assert diagnostics["unimproved_steps"] <= 0.1 * diagnostics["iterations"]


# --------------------------------------------------------------------------
# Argument validation.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({"curvature": "nonsense"}, "curvature"),
    ({"trust_radius": 0.0}, "trust_radius"),
    ({"max_backtracks": -1}, "max_backtracks"),
    ({"curvature_refresh": 0}, "curvature_refresh"),
])
def test_newton_map_validation(kwargs, message):
    score_cls = narrow_score_class()
    observations = simulate(2, score_cls)
    initial = make_joint(observations, 0.0, torch.zeros(2))
    sampler = MultiObsSampler(MockSBIm(score_cls, 3))
    with pytest.raises(ValueError, match=message):
        sampler.newton_map_estimate(
            initial, torch.tensor([0.0, 0.0, 1.0]), init=initial, hierarchy=[0],
            prior=([0.0], [score_cls.sigma_global]), correction="uncorrected",
            timesteps=5, device="cpu", **kwargs,
        )


def test_gaussian_curvature_requires_a_matching_covariance():
    """'gaussian' needs a covariance spanning shared *and* local coordinates."""
    score_cls = narrow_score_class()
    observations = simulate(2, score_cls)
    initial = make_joint(observations, 0.0, torch.zeros(2))
    sampler = MultiObsSampler(MockSBIm(score_cls, 3))
    with pytest.raises(ValueError, match="posterior_covariance"):
        sampler.newton_map_estimate(
            initial, torch.tensor([0.0, 0.0, 1.0]), init=initial, hierarchy=[0],
            prior=([0.0], [score_cls.sigma_global]), correction="uncorrected",
            curvature="gaussian", timesteps=5, device="cpu",
        )
