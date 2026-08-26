"""End-to-end plumbing for the pilot-free hierarchical pipeline.

These run a real (untrained) ConditionTransformer, so they assert shapes,
finiteness and API contracts rather than statistical accuracy. What they cover
is that `correction="gauss_jacobian"` survives the whole path -- schedule,
DPM predictor, Langevin corrector, local cross-correction, MAP refinement --
and that the two diagnostics accept exactly what `sample()` returns.
"""
import torch

from compass import ScoreBasedInferenceModel

NODES = 4
OBSERVED = 2
HIERARCHY = [0]


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
        correction="gauss_jacobian", timesteps=4, num_samples=5,
        corrector_steps=1, corrector_steps_interval=2, final_corrector_steps=0,
        verbose=False,
    )
    arguments.update(kwargs)
    return model.sample(**arguments)


def test_sampling_runs_without_any_pilot_estimate():
    model = tiny_model()
    data = observations()
    samples = draw(model, data)

    assert samples.shape == (data.shape[0], 5, NODES - OBSERVED)
    assert torch.all(torch.isfinite(samples))
    # No pilot covariance was estimated or stored.
    assert model.multi_obs_sampler.posterior_covariance is None
    assert model.multi_obs_sampler.posterior_precision is None
    assert model.multi_obs_sampler._jacobian_evaluations > 0
    # The shared coordinate stays synchronized across observation rows.
    shared = samples[:, :, HIERARCHY]
    assert float((shared - shared[:1]).abs().max()) == 0.0


def test_jacobian_refresh_reduces_jacobian_evaluations():
    data = observations()
    dense = tiny_model()
    draw(dense, data, jacobian_refresh=1)
    sparse = tiny_model()
    draw(sparse, data, jacobian_refresh=4)
    assert (sparse.multi_obs_sampler._jacobian_evaluations
            < dense.multi_obs_sampler._jacobian_evaluations)
    assert sparse.multi_obs_sampler._jacobian_evaluations > 0


def test_pilot_moments_are_rejected():
    model = tiny_model()
    data = observations()
    for argument in ("posterior_covariance", "posterior_precision"):
        try:
            draw(model, data, **{argument: torch.eye(1)})
        except ValueError as error:
            assert "takes no pilot estimate" in str(error)
        else:                                        # pragma: no cover
            raise AssertionError(f"{argument} should have been rejected")


def test_certification_runs_on_what_sample_returns():
    model = tiny_model()
    data = observations()
    samples = draw(model, data)

    report = model.certify_composition(
        samples=samples, x=data, hierarchy=HIERARCHY, timesteps=5,
        proposal="gaussian", resample=True,
    )
    assert report["log_target"].shape == (samples.shape[1],)
    assert report["log_observation_terms"].shape == (data.shape[0], samples.shape[1])
    assert torch.all(torch.isfinite(report["log_target"]))
    assert 0.0 < report["ess_fraction"] <= 1.0
    assert report["resampled_samples"].shape == (
        data.shape[0], samples.shape[1], NODES
    )


def test_curl_runs_on_the_configured_sampler():
    model = tiny_model()
    data = observations()
    samples = draw(model, data)

    report = model.composition_curl(
        samples=samples, x=data, observations=2,
    )
    assert len(report["asymmetry_mean"]) == len(report["times"])
    assert all(value >= 0.0 for value in report["asymmetry_mean"])
    assert all(
        low <= high for low, high
        in zip(report["asymmetry_mean"], report["asymmetry_max"])
    )
    # The Jacobian is measured over the shared block plus two rows of locals.
    assert report["width"] == len(HIERARCHY) + 2 * (NODES - OBSERVED - len(HIERARCHY))
    # A refresh setting must not leak out of the diagnostic.
    assert model.multi_obs_sampler.jacobian_refresh == 1


def test_newton_map_is_pilot_free_with_matching_curvature():
    """correction and Newton curvature both from the same network Jacobian."""
    model = tiny_model()
    data = observations()
    joint = torch.zeros(data.shape[0], NODES)
    joint[:, OBSERVED:] = data
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0])

    estimate = model.multi_obs_sampler.newton_map_estimate(
        data=joint, condition_mask=mask, hierarchy=HIERARCHY,
        correction="gauss_jacobian", curvature="jacobian",
        timesteps=3, iterations_per_level=1, max_iterations_per_level=1,
        device="cpu",
    )
    assert estimate.shape == (data.shape[0], 1, NODES)
    assert torch.all(torch.isfinite(estimate))
    assert model.multi_obs_sampler.posterior_covariance is None
    shared = estimate[:, :, HIERARCHY]
    assert float((shared - shared[:1]).abs().max()) == 0.0
