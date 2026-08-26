"""Tests for the "gauss_jacobian" correction and the composition diagnostics.

``gauss_jacobian`` composes exactly like ``gauss_hierarchical`` (arrow/Schur
elimination of the joint backward precision) but takes every single-observation
backward covariance from the network's own Jacobian via Tweedie's second-order
identity

    Sigma_t,j(y) = lambda^2 ( I + lambda^2 grad s_j(y) )

instead of from a pilot covariance estimate. The claim being tested is that this
is a *strict generalization*: identical to the pilot form whenever the
single-observation posterior really is Gaussian, and defined without that
assumption otherwise.

1. The Jacobian precision reproduces Sigma_0,j^-1 + lambda^-2 I exactly on an
   analytically Gaussian score.
2. The full composition therefore matches ``gauss_hierarchical`` given the
   matching pilot covariance -- with no pilot run.
3. n=1 reduces to the network's own score.
4. ``jacobian_refresh`` reuses curvature on the schedule it advertises.
5. The API refuses pilot moments and mean substitution, and needs neither.
6. ``composition_curl`` reports ~0 antisymmetry on a conservative field and
   detects a deliberately non-conservative one.
7. ``certify_composition`` assembles the exact tall-data target, its effective
   sample size and its reweighting correctly.
"""
import math
from types import SimpleNamespace

import pytest
import torch

from compass.MultiObsSampler import MultiObsSampler
from compass.SDE import VESDE


SDE = VESDE(sigma=5.0)
TIME = torch.tensor([[0.4]])
VARIANCE = float(SDE.lambda_t(TIME).square().reshape(()))


class LinearGaussianScore(torch.nn.Module):
    """Exact score of a Gaussian N(mean_j, Sigma_0,j) smoothed by lambda(t)^2.

    The precision is rebuilt from the *passed* ``t`` as
    ``(Sigma_0,j + lambda_t^2 I)^-1``, so this is a consistent score family
    across the whole diffusion schedule rather than one frozen matrix. That
    matters for anything that varies ``t``: a ``t``-frozen stand-in is not the
    score of any diffusion, and tests built on one measure its inconsistency
    instead of the code's.
    """

    def __init__(self, features, means, covariance, observations, sde):
        super().__init__()
        self.features = list(features)
        self.register_buffer("means", means.to(torch.float32))
        self.register_buffer("covariance", covariance.to(torch.float64))
        self.observations = int(observations)
        self.sde = sde

    def forward(self, x, t, c):
        variance = self.sde.lambda_t(t).square().reshape(()).to(torch.float64)
        identity = torch.eye(self.covariance.shape[-1], dtype=torch.float64)
        precision = torch.linalg.inv(
            self.covariance + variance * identity
        ).to(torch.float32)
        rows = x.shape[0] // self.observations
        block = x.reshape(self.observations, rows, -1)[:, :, self.features]
        delta = block - self.means[:, None, :]
        score = -torch.einsum("nij,nsj->nsi", precision, delta)
        out = torch.zeros(
            self.observations, rows, x.shape[-1], dtype=x.dtype, device=x.device
        )
        out[:, :, self.features] = score
        return out.reshape(x.shape)


class RotationScore(torch.nn.Module):
    """A deliberately non-conservative field: a rotation has no potential."""

    def __init__(self, features, observations, strength=0.6):
        super().__init__()
        self.features = list(features)
        self.observations = int(observations)
        self.strength = float(strength)

    def forward(self, x, t, c):
        rows = x.shape[0] // self.observations
        block = x.reshape(self.observations, rows, -1)[:, :, self.features]
        rotated = torch.stack(
            [-self.strength * block[..., 1], self.strength * block[..., 0]],
            dim=-1,
        )
        out = torch.zeros(
            self.observations, rows, x.shape[-1], dtype=x.dtype, device=x.device
        )
        out[:, :, self.features] = block * -1.0 + rotated
        return out.reshape(x.shape)


def make_sbim(model):
    return SimpleNamespace(
        sde=SDE, model=model,
        output_scale_function=lambda t, value: value,
    )


def configure(sampler, correction, hierarchy, locals_, observations,
              covariance=None):
    sampler.hierarchy = list(hierarchy)
    sampler.local_latent_indices = list(locals_)
    sampler.full_gaussian_features = list(hierarchy) + list(locals_)
    sampler.covariance_features = sampler.full_gaussian_features
    sampler.correction = correction
    sampler.denoise_clamp = None
    width = len(hierarchy)
    sampler.prior_mean = torch.zeros(width)
    sampler.prior_std = torch.ones(width)
    sampler.prior_covariance = torch.eye(width, dtype=torch.float64)
    sampler.prior_precision_matrix = torch.eye(width, dtype=torch.float64)
    sampler.posterior_precision = None
    sampler.posterior_mean = None
    sampler.global_posterior_mean = None
    sampler.global_posterior_covariance = None
    sampler.global_posterior_precision_matrix = None
    sampler.world_size = 1
    sampler.num_observations = observations
    sampler.cfg_alpha = None
    sampler.device = "cpu"
    sampler._covariance_time_cache = {}
    sampler._reset_jacobian_state()
    if covariance is None:
        sampler.posterior_covariance = None
        sampler.posterior_precision_matrix = None
    else:
        sampler.posterior_covariance = covariance
        sampler.posterior_precision_matrix = sampler._precision_from_covariance(
            covariance
        )
    sampler._configure_damping(observations, 1.0, 0.5, None)
    return sampler


def gaussian_setup(observations=3, hierarchy=(0,), locals_=(1,), nodes=3):
    """A per-observation Gaussian joint (g, l_j) with known Sigma_0,j."""
    features = list(hierarchy) + list(locals_)
    dimension = len(features)
    generator = torch.Generator().manual_seed(11)
    covariance = []
    for _ in range(observations):
        root = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
        covariance.append(root @ root.mT / dimension + 0.4 * torch.eye(dimension, dtype=torch.float64))
    covariance = torch.stack(covariance)
    means = torch.randn(observations, dimension, generator=generator, dtype=torch.float64) * 0.3
    model = LinearGaussianScore(features, means, covariance, observations, SDE)
    return {
        "features": features, "covariance": covariance, "means": means,
        "model": model, "nodes": nodes,
        "observations": observations, "hierarchy": list(hierarchy),
        "locals": list(locals_),
    }


def build_state(setup, num_samples=4, seed=3):
    """Synchronized shared coordinates, independent local ones."""
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(
        setup["observations"], num_samples, setup["nodes"], generator=generator
    ) * 0.5
    shared = state[:1, :, setup["hierarchy"]]
    state[:, :, setup["hierarchy"]] = shared.expand(setup["observations"], -1, -1)
    mask = torch.zeros(setup["observations"], num_samples, setup["nodes"])
    mask[:, :, setup["nodes"] - 1:] = 1.0
    return state, mask


# ---------------------------------------------------------------------------
# 1. The Tweedie construction reproduces the pilot form exactly
# ---------------------------------------------------------------------------

def test_jacobian_precision_equals_pilot_precision_on_gaussian_score():
    """Sigma_t,j^-1 from the Jacobian == Sigma_0,j^-1 + lambda^-2 I."""
    setup = gaussian_setup()
    sampler = MultiObsSampler(make_sbim(setup["model"]))
    configure(sampler, "gauss_jacobian", setup["hierarchy"], setup["locals"],
              setup["observations"])
    state, mask = build_state(setup)

    precision = sampler._jacobian_joint_precision(state, TIME, mask, VARIANCE)

    identity = torch.eye(len(setup["features"]), dtype=torch.float64)
    expected = torch.linalg.inv(setup["covariance"]) + identity / VARIANCE
    # (observations, samples, D, D) -- constant across samples for a Gaussian.
    assert precision.shape == (setup["observations"], state.shape[1],
                               len(setup["features"]), len(setup["features"]))
    torch.testing.assert_close(
        precision, expected[:, None].expand_as(precision),
        rtol=1e-3, atol=1e-4,
    )


def test_jacobian_precision_is_state_independent_only_for_gaussians():
    """The whole point: the weighting tracks the state when it should."""
    setup = gaussian_setup()
    sampler = MultiObsSampler(make_sbim(setup["model"]))
    configure(sampler, "gauss_jacobian", setup["hierarchy"], setup["locals"],
              setup["observations"])
    state, mask = build_state(setup)
    gaussian = sampler._jacobian_joint_precision(state, TIME, mask, VARIANCE)
    spread = (gaussian - gaussian.mean(dim=1, keepdim=True)).abs().max()
    assert float(spread) < 1e-6

    nonlinear = MultiObsSampler(make_sbim(RotationScore([0, 1], setup["observations"])))
    configure(nonlinear, "gauss_jacobian", [0], [1], setup["observations"])
    rotated = nonlinear._jacobian_joint_precision(state, TIME, mask, VARIANCE)
    assert torch.all(torch.isfinite(rotated))


# ---------------------------------------------------------------------------
# 2. Full composition equals the pilot-covariance rule it generalizes
# ---------------------------------------------------------------------------

def test_gauss_jacobian_matches_gauss_hierarchical_given_matching_covariance():
    """The strict-generalization claim, end to end through _compositional_score."""
    setup = gaussian_setup()
    state, mask = build_state(setup)

    jacobian_sampler = MultiObsSampler(make_sbim(setup["model"]))
    configure(jacobian_sampler, "gauss_jacobian", setup["hierarchy"],
              setup["locals"], setup["observations"])

    pilot_sampler = MultiObsSampler(make_sbim(setup["model"]))
    configure(pilot_sampler, "gauss_hierarchical", setup["hierarchy"],
              setup["locals"], setup["observations"],
              covariance=setup["covariance"])

    scores = setup["model"](
        state.reshape(-1, setup["nodes"]), TIME, mask.reshape(-1, setup["nodes"])
    ).reshape(state.shape)

    from_jacobian = jacobian_sampler._compositional_score(
        scores.clone(), state, TIME, condition_mask=mask,
    )
    from_pilot = pilot_sampler._compositional_score(
        scores.clone(), state, TIME,
    )
    torch.testing.assert_close(from_jacobian, from_pilot, rtol=2e-3, atol=2e-4)


def test_gauss_jacobian_single_observation_returns_the_network_score():
    setup = gaussian_setup(observations=1)
    sampler = MultiObsSampler(make_sbim(setup["model"]))
    configure(sampler, "gauss_jacobian", setup["hierarchy"], setup["locals"], 1)
    state, mask = build_state(setup)
    scores = setup["model"](
        state.reshape(-1, setup["nodes"]), TIME, mask.reshape(-1, setup["nodes"])
    ).reshape(state.shape)

    composed = sampler._compositional_score(
        scores.clone(), state, TIME, condition_mask=mask,
    )
    torch.testing.assert_close(composed, scores, rtol=1e-4, atol=1e-5)


def test_gauss_jacobian_without_local_latents_matches_pilot_reduction():
    setup = gaussian_setup(hierarchy=(0, 1), locals_=(), nodes=3)
    state, mask = build_state(setup)
    scores = setup["model"](
        state.reshape(-1, setup["nodes"]), TIME, mask.reshape(-1, setup["nodes"])
    ).reshape(state.shape)

    jacobian_sampler = configure(
        MultiObsSampler(make_sbim(setup["model"])), "gauss_jacobian",
        setup["hierarchy"], [], setup["observations"],
    )
    pilot_sampler = configure(
        MultiObsSampler(make_sbim(setup["model"])), "gauss_hierarchical",
        setup["hierarchy"], [], setup["observations"],
        covariance=setup["covariance"],
    )
    torch.testing.assert_close(
        jacobian_sampler._compositional_score(
            scores.clone(), state, TIME, condition_mask=mask
        ),
        pilot_sampler._compositional_score(scores.clone(), state, TIME),
        rtol=2e-3, atol=2e-4,
    )


# ---------------------------------------------------------------------------
# 3. Refresh schedule
# ---------------------------------------------------------------------------

def test_jacobian_refresh_reuses_curvature_on_schedule():
    setup = gaussian_setup()
    sampler = MultiObsSampler(make_sbim(setup["model"]))
    configure(sampler, "gauss_jacobian", setup["hierarchy"], setup["locals"],
              setup["observations"])
    sampler.jacobian_refresh = 3
    state, mask = build_state(setup)

    for _ in range(7):
        sampler._effective_global_factors(
            VARIANCE, setup["observations"], state=state, t=TIME,
            condition_mask=mask,
        )
    # Evaluations at calls 0, 3 and 6.
    assert sampler._jacobian_evaluations == 3


def test_jacobian_refresh_stays_finite_across_a_full_schedule():
    """A lagged refresh must not carry a stale lambda into Lambda_j(t).

    Lambda_j(t) contains an exact lambda^-2 I term, and the composition
    subtracts (n-1) copies of Lambda_prior(t) built at the *current* time. If a
    cached Lambda_j is reused at a different t the two lambda^-2 terms no
    longer cancel, and because they diverge as lambda shrinks the composed
    precision goes indefinite for n > 1 -- which showed up as a non-finite
    score partway down a real sampling schedule, not at any single time.
    """
    setup = gaussian_setup(observations=8)
    state, mask = build_state(setup)
    scores = setup["model"](
        state.reshape(-1, setup["nodes"]), TIME, mask.reshape(-1, setup["nodes"])
    ).reshape(state.shape)

    schedule = SDE.time_of_lambda(torch.logspace(math.log10(3.0), math.log10(0.02), 40))
    for refresh in (1, 10):
        sampler = MultiObsSampler(make_sbim(setup["model"]))
        configure(sampler, "gauss_jacobian", setup["hierarchy"], setup["locals"],
                  setup["observations"])
        sampler.jacobian_refresh = refresh
        for time_value in schedule:
            composed = sampler._compositional_score(
                scores.clone(), state, time_value.reshape(1, 1),
                condition_mask=mask,
            )
            assert torch.all(torch.isfinite(composed)), (
                f"refresh={refresh} produced a non-finite score at t={time_value}"
            )


def test_jacobian_refresh_tracks_the_unlagged_composition():
    """What is cached is Sigma_0,j, so lagging it costs nothing on a Gaussian.

    For a genuinely Gaussian single-observation posterior the cached quantity
    ``Lambda_j(t) - lambda_t^-2 I`` equals ``Sigma_0,j^-1`` at *every* t, so a
    lagged refresh must reproduce the unlagged composition to float precision.
    Caching ``Lambda_j(t)`` itself instead would carry a stale ``lambda^-2 I``
    and fail here by orders of magnitude.
    """
    setup = gaussian_setup(observations=4)
    state, mask = build_state(setup)
    schedule = SDE.time_of_lambda(torch.logspace(math.log10(2.0), math.log10(0.05), 12))

    results = {}
    for refresh in (1, 6):
        sampler = MultiObsSampler(make_sbim(setup["model"]))
        configure(sampler, "gauss_jacobian", setup["hierarchy"], setup["locals"],
                  setup["observations"])
        sampler.jacobian_refresh = refresh
        composed = []
        for time_value in schedule:
            time_value = time_value.reshape(1, 1)
            # The scores must come from the same t as the composition, or the
            # comparison measures the stand-in's inconsistency, not the code's.
            scores = setup["model"](
                state.reshape(-1, setup["nodes"]), time_value,
                mask.reshape(-1, setup["nodes"]),
            ).reshape(state.shape)
            composed.append(sampler._compositional_score(
                scores, state, time_value, condition_mask=mask,
            )[0, :, setup["hierarchy"]])
        results[refresh] = torch.stack(composed)

    difference = (results[1] - results[6]).abs().max()
    scale = results[1].abs().max().clamp_min(1e-8)
    assert float(difference / scale) < 1e-5


def test_jacobian_correction_requires_the_current_state():
    setup = gaussian_setup()
    sampler = MultiObsSampler(make_sbim(setup["model"]))
    configure(sampler, "gauss_jacobian", setup["hierarchy"], setup["locals"],
              setup["observations"])
    with pytest.raises(ValueError, match="needs the current state"):
        sampler._effective_global_factors(VARIANCE, setup["observations"])


# ---------------------------------------------------------------------------
# 4. Registry and API guards
# ---------------------------------------------------------------------------

def test_registry_places_gauss_jacobian_in_the_schur_family():
    assert "gauss_jacobian" in MultiObsSampler.VALID_CORRECTIONS
    assert "gauss_jacobian" in MultiObsSampler.SCHUR_GAUSSIAN_CORRECTIONS
    assert "gauss_jacobian" in MultiObsSampler.LOCAL_CROSS_CORRECTIONS
    assert "gauss_jacobian" in MultiObsSampler.GAUSSIAN_CORRECTIONS
    # Mean substitution is structurally impossible, as for gauss_hierarchical.
    assert "gauss_jacobian" in MultiObsSampler.NO_MOMENT_SUBSTITUTION_CORRECTIONS
    # The pilot-covariance rules must be untouched.
    assert "gauss_hierarchical" in MultiObsSampler.SCHUR_GAUSSIAN_CORRECTIONS
    assert "gauss_jacobian" not in MultiObsSampler.FULL_GAUSSIAN_CORRECTIONS


def test_map_configuration_rejects_pilot_moments_and_needs_none():
    setup = gaussian_setup(observations=2)
    sampler = MultiObsSampler(make_sbim(setup["model"]))
    data = torch.zeros(2, setup["nodes"])
    mask = torch.zeros(setup["nodes"])
    mask[-1] = 1.0

    arguments = dict(
        data=data, condition_mask=mask, init=None, hierarchy=[0],
        prior=None, local_prior=None, correction="gauss_jacobian",
        posterior_precision=None, posterior_covariance=None,
        posterior_mean=None, global_posterior_mean=None,
        global_posterior_covariance=None, denoise_clamp=None, cfg_alpha=None,
        damping_at_data=1.0, damping_at_noise=None, timesteps=2,
        iterations_per_level=1, max_iterations_per_level=1,
        convergence_tol=1e-6, device="cpu",
    )
    state = sampler._configure_map_estimate(**arguments)
    assert sampler.posterior_covariance is None
    assert sampler.posterior_precision is None
    assert state["hierarchy"] == [0]

    with pytest.raises(ValueError, match="takes no pilot estimate"):
        sampler._configure_map_estimate(
            **{**arguments, "posterior_covariance": torch.eye(2)}
        )
    with pytest.raises(ValueError, match="composes only the network's real"):
        sampler._configure_map_estimate(
            **{**arguments, "posterior_mean": torch.zeros(1, 2)}
        )


# ---------------------------------------------------------------------------
# 5. Curl diagnostic
# ---------------------------------------------------------------------------

def curl_ready_sampler(model, observations, hierarchy, locals_, nodes):
    sampler = MultiObsSampler(make_sbim(model))
    configure(sampler, "gauss_jacobian", hierarchy, locals_, observations)
    sampler.timesteps_list = SDE.time_of_lambda(
        torch.logspace(math.log10(0.5), math.log10(0.2), 3)
    )
    return sampler


def test_curl_is_zero_for_a_conservative_single_observation_field():
    """One observation composes to the network's own score, which is a gradient."""
    setup = gaussian_setup(observations=1, hierarchy=(0,), locals_=(1,))
    sampler = curl_ready_sampler(setup["model"], 1, [0], [1], setup["nodes"])
    samples = torch.randn(1, 6, setup["nodes"]) * 0.4
    data = torch.zeros(1, 1)
    mask = torch.zeros(setup["nodes"])
    mask[-1] = 1.0

    report = sampler.composition_curl(
        samples=samples, data=data, condition_mask=mask,
        times=SDE.time_of_lambda(torch.tensor([0.6])),
    )
    assert max(report["asymmetry_mean"]) < 1e-3


def test_curl_detects_a_non_conservative_field():
    observations, nodes = 1, 3
    model = RotationScore([0, 1], observations)
    sampler = curl_ready_sampler(model, observations, [0], [1], nodes)
    samples = torch.randn(observations, 6, nodes) * 0.4
    data = torch.zeros(observations, 1)
    mask = torch.zeros(nodes)
    mask[-1] = 1.0

    report = sampler.composition_curl(
        samples=samples, data=data, condition_mask=mask,
        times=SDE.time_of_lambda(torch.tensor([0.6])),
    )
    assert min(report["asymmetry_mean"]) > 0.1


def test_curl_requires_a_configured_sampler():
    sampler = MultiObsSampler(SimpleNamespace(sde=SDE))
    with pytest.raises(RuntimeError, match="needs a configured sampler"):
        sampler.composition_curl(
            samples=torch.zeros(1, 2, 3), data=torch.zeros(1, 1),
            condition_mask=torch.tensor([0.0, 0.0, 1.0]),
        )


# ---------------------------------------------------------------------------
# 6. PF-ODE certification
# ---------------------------------------------------------------------------

class RecordingPFODE:
    """Returns a prescribed log-density per point and records the call."""

    def __init__(self, values):
        self.values = values
        self.calls = []

    def log_prob(self, data, condition_mask, **kwargs):
        self.calls.append((data.shape, condition_mask.shape))
        return self.values.reshape(-1).clone()


def certification_sampler(observations, num_samples, nodes, values):
    sbim = SimpleNamespace(sde=SDE, pfode=RecordingPFODE(values))
    sampler = MultiObsSampler(sbim)
    sampler.covariance_shrinkage = 0.05
    sampler.covariance_nugget = 1e-6
    return sampler


def test_certification_assembles_the_exact_tall_data_target():
    observations, num_samples, nodes = 3, 8, 3
    terms = torch.arange(
        observations * num_samples, dtype=torch.float64
    ).reshape(observations, num_samples) * 0.1
    sampler = certification_sampler(observations, num_samples, nodes, terms)

    generator = torch.Generator().manual_seed(5)
    shared = torch.randn(1, num_samples, 1, generator=generator)
    samples = torch.randn(observations, num_samples, nodes, generator=generator)
    samples[:, :, :1] = shared.expand(observations, -1, -1)
    samples[:, :, 2] = 0.0
    data = torch.randn(observations, 1, generator=generator)
    mask = torch.tensor([0.0, 0.0, 1.0])

    report = sampler.certify_composition(
        samples=samples, data=data, condition_mask=mask, hierarchy=[0],
        proposal=None,
    )

    expected_prior = (1 - observations) * (
        -0.5 * shared[0, :, 0].to(torch.float64) ** 2
        - 0.5 * math.log(2 * math.pi)
    )
    torch.testing.assert_close(report["log_prior_term"], expected_prior)
    torch.testing.assert_close(
        report["log_target"], expected_prior + terms.sum(dim=0)
    )
    assert report["num_observations"] == observations
    # All observation/sample points went through in a single batched call.
    assert len(sampler.SBIm.pfode.calls) == 1
    assert sampler.SBIm.pfode.calls[0][0] == (observations * num_samples, nodes)


def test_certification_weights_resampling_and_ess():
    observations, num_samples, nodes = 2, 64, 3
    terms = torch.zeros(observations, num_samples, dtype=torch.float64)
    sampler = certification_sampler(observations, num_samples, nodes, terms)

    generator = torch.Generator().manual_seed(7)
    shared = torch.randn(1, num_samples, 1, generator=generator)
    samples = torch.randn(observations, num_samples, nodes, generator=generator)
    samples[:, :, :1] = shared.expand(observations, -1, -1)
    samples[:, :, 2] = 0.0
    data = torch.randn(observations, 1, generator=generator)
    mask = torch.tensor([0.0, 0.0, 1.0])

    report = sampler.certify_composition(
        samples=samples, data=data, condition_mask=mask, hierarchy=[0],
        proposal="gaussian", resample=True,
    )
    assert 0.0 < report["ess_fraction"] <= 1.0
    assert report["ess"] == pytest.approx(
        float(1.0 / report["weights"].square().sum())
    )
    torch.testing.assert_close(
        report["weights"].sum(), torch.tensor(1.0, dtype=torch.float64)
    )
    assert report["resampled_samples"].shape == (observations, num_samples, nodes)
    # Resampling must preserve shared-coordinate synchronization.
    resampled = report["resampled_samples"][:, :, 0]
    assert float((resampled - resampled[:1]).abs().max()) == 0.0

    factorized = sampler.certify_composition(
        samples=samples, data=data, condition_mask=mask, hierarchy=[0],
        proposal="factorized",
    )
    assert torch.all(torch.isfinite(factorized["log_weights"]))


def test_certification_rejects_desynchronized_shared_coordinates():
    observations, num_samples, nodes = 2, 4, 3
    terms = torch.zeros(observations, num_samples, dtype=torch.float64)
    sampler = certification_sampler(observations, num_samples, nodes, terms)
    samples = torch.randn(observations, num_samples, nodes)
    with pytest.raises(ValueError, match="must be synchronized"):
        sampler.certify_composition(
            samples=samples, data=torch.zeros(observations, 1),
            condition_mask=torch.tensor([0.0, 0.0, 1.0]), hierarchy=[0],
            proposal=None,
        )
