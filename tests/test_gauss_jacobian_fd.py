"""Tests for correction="gauss_jacobian_FD" and the low-noise refresh schedule.

``gauss_jacobian_FD`` is ``gauss_jacobian`` with the score Jacobian taken by
one batched forward-difference network call instead of D forward-mode JVPs.
Everything downstream is shared, so the claims tested here are narrow:

1. The finite-difference Jacobian equals the forward-mode one (to float32
   rounding on a linear score, to O(step) on a real network), including when
   the stacked copies are split over several calls.
2. It therefore reproduces the pilot precision on a Gaussian score and gives
   the same composed score and the same draws as ``gauss_jacobian``.
3. It uses one network call per Jacobian instead of D.
4. ``jacobian_refresh_low_noise`` switches the refresh period below
   ``jacobian_refresh_lambda`` and is validated.
5. The registry and the pilot-free MAP path treat it as a Jacobian correction.
"""
import pytest
import torch

from compass import ScoreBasedInferenceModel
from compass.MultiObsSampler import MultiObsSampler

from test_gauss_jacobian_composition import (
    TIME, VARIANCE, build_state, configure, gaussian_setup, make_sbim,
)

NODES = 4
OBSERVED = 2
HIERARCHY = [0]


def fd_and_ad_samplers(setup):
    samplers = {}
    for correction in ("gauss_jacobian", "gauss_jacobian_FD"):
        sampler = MultiObsSampler(make_sbim(setup["model"]))
        configure(sampler, correction, setup["hierarchy"], setup["locals"],
                  setup["observations"])
        samplers[correction] = sampler
    return samplers


def tiny_model():
    torch.manual_seed(0)
    return ScoreBasedInferenceModel(
        nodes_size=NODES, sigma=5.0, hidden_size=8, depth=1, num_heads=2,
        mlp_ratio=2,
    )


def observations(count=3):
    torch.manual_seed(1)
    return torch.randn(count, OBSERVED) * 0.5


def draw(model, data, **kwargs):
    arguments = dict(
        x=data, multi_obs_inference=True, hierarchy=HIERARCHY,
        correction="gauss_jacobian_FD", timesteps=6, num_samples=5,
        verbose=False,
    )
    arguments.update(kwargs)
    torch.manual_seed(2)
    return model.sample(**arguments)


# ---------------------------------------------------------------------------
# 1. The finite-difference Jacobian is the forward-mode Jacobian
# ---------------------------------------------------------------------------

def test_fd_jacobian_matches_forward_mode_on_a_linear_score():
    setup = gaussian_setup()
    samplers = fd_and_ad_samplers(setup)
    state, mask = build_state(setup)
    order = setup["features"]

    with torch.enable_grad():
        exact = samplers["gauss_jacobian"]._row_score_jacobian(
            state, TIME, mask, order)
    approx = samplers["gauss_jacobian_FD"]._row_score_jacobian(
        state, TIME, mask, order)

    assert approx.shape == exact.shape
    assert approx.dtype == torch.float64
    torch.testing.assert_close(approx, exact, rtol=2e-3, atol=2e-3)


def test_fd_jacobian_converges_to_forward_mode_on_a_network():
    """First-order convergence to the JVP Jacobian on a nonlinear network.

    The randomized network is far rougher than a trained, lambda-smoothed
    score (which is smooth on the scale lambda), so the default step is not
    what is checked here -- the linear-score test covers that. What is checked
    is that the construction itself is right: no batch coupling between the
    stacked copies, the right columns in the right order, and an error that
    vanishes linearly with the step.
    """
    model = tiny_model()
    # The untrained network is zero-initialized at its output, so its Jacobian
    # is exactly zero; randomize it to get a genuinely nonlinear score.
    generator = torch.Generator().manual_seed(6)
    with torch.no_grad():
        for parameter in model.model.parameters():
            parameter.add_(0.3 * torch.randn(
                parameter.shape, generator=generator))
    sampler = model.multi_obs_sampler
    sampler.hierarchy = HIERARCHY
    generator = torch.Generator().manual_seed(4)
    rows, samples = 3, 6
    state = torch.randn(rows, samples, NODES, generator=generator)
    state[:, :, HIERARCHY] = state[:1, :, HIERARCHY]
    mask = torch.zeros(rows, samples, NODES)
    mask[:, :, OBSERVED:] = 1.0
    order = [0, 1]
    t = model.sde.time_of_lambda(torch.tensor(2.0)).reshape(1, 1)
    sampler.correction = "gauss_jacobian"
    with torch.enable_grad():
        exact = sampler._row_score_jacobian(state, t, mask, order)
    assert float(exact.norm()) > 0.0
    sampler.correction = "gauss_jacobian_FD"
    errors = []
    for step in (1e-3, 1e-4):
        sampler.jacobian_fd_step = step
        approx = sampler._row_score_jacobian(state, t, mask, order)
        errors.append(float((approx - exact).norm() / exact.norm()))
    assert errors[1] < 1e-2, errors
    assert errors[1] < 0.3 * errors[0], errors


def test_fd_jacobian_is_unchanged_by_chunking():
    setup = gaussian_setup()
    sampler = fd_and_ad_samplers(setup)["gauss_jacobian_FD"]
    state, mask = build_state(setup)
    order = setup["features"]

    whole = sampler._row_score_jacobian(state, TIME, mask, order)
    rows, samples = state.shape[:2]
    # One copy per network call.
    sampler.FD_JACOBIAN_MAX_ROWS = rows * samples
    before = sampler.score_network_calls
    split = sampler._row_score_jacobian(state, TIME, mask, order)

    assert sampler.score_network_calls - before == len(order) + 1
    torch.testing.assert_close(split, whole)


# ---------------------------------------------------------------------------
# 2. Same precision, same composition, same draws
# ---------------------------------------------------------------------------

def test_fd_precision_equals_pilot_precision_on_gaussian_score():
    setup = gaussian_setup()
    sampler = fd_and_ad_samplers(setup)["gauss_jacobian_FD"]
    state, mask = build_state(setup)

    precision = sampler._jacobian_joint_precision(state, TIME, mask, VARIANCE)

    identity = torch.eye(len(setup["features"]), dtype=torch.float64)
    expected = torch.linalg.inv(setup["covariance"]) + identity / VARIANCE
    torch.testing.assert_close(
        precision, expected[:, None].expand_as(precision),
        rtol=5e-3, atol=5e-3,
    )


def test_fd_composed_score_matches_gauss_jacobian():
    setup = gaussian_setup()
    samplers = fd_and_ad_samplers(setup)
    state, mask = build_state(setup)
    features = setup["features"]
    scores = torch.zeros_like(state)
    scores[:, :, features] = torch.randn(
        state.shape[0], state.shape[1], len(features),
        generator=torch.Generator().manual_seed(5),
    )

    composed = {
        name: sampler._compositional_score(
            scores, state, TIME, num_observations=setup["observations"],
            condition_mask=mask,
        )
        for name, sampler in samplers.items()
    }
    torch.testing.assert_close(
        composed["gauss_jacobian_FD"], composed["gauss_jacobian"],
        rtol=5e-3, atol=5e-3,
    )


def test_fd_draws_match_gauss_jacobian_draws():
    data = observations()
    ad_model, fd_model = tiny_model(), tiny_model()
    exact = draw(ad_model, data, correction="gauss_jacobian")
    approx = draw(fd_model, data, correction="gauss_jacobian_FD")

    assert torch.all(torch.isfinite(approx))
    shared = approx[:, :, HIERARCHY]
    assert float((shared - shared[:1]).abs().max()) == 0.0
    assert fd_model.multi_obs_sampler.posterior_covariance is None
    torch.testing.assert_close(approx, exact, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 3. One network call per Jacobian
# ---------------------------------------------------------------------------

def test_fd_uses_one_network_call_per_jacobian():
    data = observations()
    ad_model, fd_model = tiny_model(), tiny_model()
    draw(ad_model, data, correction="gauss_jacobian")
    draw(fd_model, data, correction="gauss_jacobian_FD")
    ad, fd = ad_model.multi_obs_sampler, fd_model.multi_obs_sampler

    assert ad._jacobian_evaluations == fd._jacobian_evaluations > 0
    latent = NODES - OBSERVED
    ad_extra = ad.score_network_calls - fd.score_network_calls
    # D JVP calls per Jacobian for AD against one stacked call for FD.
    assert ad_extra == (latent - 1) * ad._jacobian_evaluations


# ---------------------------------------------------------------------------
# 4. The low-noise refresh schedule
# ---------------------------------------------------------------------------

def test_refresh_period_switches_below_the_threshold():
    sampler = MultiObsSampler(make_sbim(gaussian_setup()["model"]))
    sampler.jacobian_refresh = 2
    sampler._configure_jacobian_schedule(1e-2, 20, 0.3)

    assert sampler._jacobian_refresh_at(1.0 / 1.0**2) == 2
    assert sampler._jacobian_refresh_at(1.0 / 0.3**2) == 2
    assert sampler._jacobian_refresh_at(1.0 / 0.1**2) == 20
    sampler._configure_jacobian_schedule(1e-2, None, 0.3)
    assert sampler._jacobian_refresh_at(1.0 / 0.1**2) == 2


@pytest.mark.parametrize("correction", ["gauss_jacobian", "gauss_jacobian_FD"])
def test_low_noise_refresh_reduces_jacobian_evaluations(correction):
    data = observations()
    dense, sparse = tiny_model(), tiny_model()
    draw(dense, data, correction=correction, timesteps=12)
    draw(sparse, data, correction=correction, timesteps=12,
         jacobian_refresh_low_noise=50, jacobian_refresh_lambda=1.0)

    dense_count = dense.multi_obs_sampler._jacobian_evaluations
    sparse_count = sparse.multi_obs_sampler._jacobian_evaluations
    assert 0 < sparse_count < dense_count


@pytest.mark.parametrize("kwargs", [
    dict(jacobian_fd_step=0.0),
    dict(jacobian_fd_step=float("nan")),
    dict(jacobian_refresh_low_noise=0),
    dict(jacobian_refresh_lambda=-1.0),
])
def test_schedule_arguments_are_validated(kwargs):
    with pytest.raises(ValueError):
        draw(tiny_model(), observations(), **kwargs)


# ---------------------------------------------------------------------------
# 5. Registry and the pilot-free MAP path
# ---------------------------------------------------------------------------

def test_registry_places_fd_with_the_jacobian_corrections():
    name = "gauss_jacobian_FD"
    assert name in MultiObsSampler.VALID_CORRECTIONS
    assert name in MultiObsSampler.JACOBIAN_GAUSSIAN_CORRECTIONS
    assert name in MultiObsSampler.SCHUR_GAUSSIAN_CORRECTIONS
    assert name in MultiObsSampler.LOCAL_CROSS_CORRECTIONS
    assert name in MultiObsSampler.NO_MOMENT_SUBSTITUTION_CORRECTIONS
    assert name in MultiObsSampler.TEMPERABLE_CORRECTIONS
    assert "gauss_jacobian" not in MultiObsSampler.FINITE_DIFFERENCE_JACOBIAN_CORRECTIONS


def test_pilot_moments_are_rejected():
    with pytest.raises(ValueError):
        draw(tiny_model(), observations(),
             posterior_covariance=torch.eye(NODES - OBSERVED))


def test_newton_map_runs_on_the_fd_jacobian():
    model = tiny_model()
    data = observations()
    joint = torch.zeros(data.shape[0], NODES)
    joint[:, OBSERVED:] = data
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0])

    estimate = model.multi_obs_sampler.newton_map_estimate(
        data=joint, condition_mask=mask, hierarchy=HIERARCHY,
        correction="gauss_jacobian_FD", curvature="jacobian",
        timesteps=3, iterations_per_level=1, max_iterations_per_level=1,
        device="cpu",
    )
    assert estimate.shape == (data.shape[0], 1, NODES)
    assert torch.all(torch.isfinite(estimate))
    shared = estimate[:, :, HIERARCHY]
    assert float((shared - shared[:1]).abs().max()) == 0.0
