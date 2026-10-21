"""The tokenized network must be a drop-in for every consumer of the score.

The diffusion state stays a flat raw vector, so ``Sampler``, ``PFODE`` and the
~3.9k-line ``MultiObsSampler`` should need no changes at all. This file is the
verification of that claim rather than the assumption of it: every entry point
into the network is exercised on a grouped (trial-token) model, including the
two paths the partial-pooling benchmark actually runs -- ``gauss_jacobian``
composition and ``newton_map_estimate(curvature="jacobian")``, both of which
push forward-mode AD through the network.

It also pins the property the trial tokenization exists to buy: permuting whole
trials permutes their scores exactly, and leaves the theta scores untouched.
"""

import pytest
import torch

torch.set_num_threads(1)

from compass import IID, ScoreBasedInferenceModel

GLOBALS = 7
LOCALS = 3
THETA = GLOBALS + LOCALS
TRIALS = 5
ITEM = 3
NODES = THETA + TRIALS * ITEM


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    built = ScoreBasedInferenceModel(
        theta_size=THETA, x_spec=[IID("trials", item_dim=ITEM, count=TRIALS)],
        hidden_size=8, depth=2, num_heads=2, mlp_ratio=2, device="cpu",
    )
    built.model.eval()
    return built


@pytest.fixture(scope="module")
def observations():
    generator = torch.Generator().manual_seed(1)
    return torch.randn(4, TRIALS * ITEM, generator=generator)


def test_shapes_and_token_count(model):
    assert model.nodes_size == NODES
    assert model.token_count == THETA + TRIALS
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(3, NODES, generator=generator)
    t = torch.rand(3, 1, generator=generator)
    c = (torch.rand(3, NODES, generator=generator) < 0.5).float()
    with torch.no_grad():
        out = model.model(x=x, t=t, c=c)
    assert out.shape == (3, NODES)
    assert torch.isfinite(out).all()


def test_all_latent_input_stays_finite(model):
    """No conditioned token means no attention key; add_bias_kv is the escape."""
    x = torch.randn(2, NODES)
    t = torch.full((2, 1), 0.5)
    with torch.no_grad():
        out = model.model(x=x, t=t, c=torch.zeros(2, NODES))
    assert torch.isfinite(out).all()


def test_permuting_trials_permutes_their_scores(model):
    """The whole point of one token per trial: exchangeability, exactly."""
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(1, NODES, generator=generator)
    t = torch.full((1, 1), 0.4)
    c = torch.cat([torch.zeros(1, THETA), torch.ones(1, TRIALS * ITEM)], dim=1)

    order = torch.tensor([3, 0, 4, 1, 2])
    def permute(flat):
        head, tail = flat[:, :THETA], flat[:, THETA:].reshape(1, TRIALS, ITEM)
        return torch.cat([head, tail[:, order].reshape(1, -1)], dim=1)

    with torch.no_grad():
        base = model.model(x=x, t=t, c=c)
        permuted = model.model(x=permute(x), t=t, c=permute(c))

    assert torch.allclose(base[:, :THETA], permuted[:, :THETA], atol=1e-6)
    assert torch.allclose(permute(base)[:, THETA:], permuted[:, THETA:], atol=1e-6)


def test_partially_observed_trial_is_latent_as_a_whole(model):
    """A trial with one masked feature must not act as an attention key."""
    schema = model.schema
    full = torch.cat([torch.zeros(1, THETA), torch.ones(1, TRIALS * ITEM)], dim=1)
    partial = full.clone()
    partial[0, THETA + 1] = 0.0                      # one feature of trial 0
    assert schema.token_mask(full)[0, THETA] == 1.0
    assert schema.token_mask(partial)[0, THETA] == 0.0


def test_single_observation_sampler(model, observations):
    torch.manual_seed(4)
    draws = model.sample(
        x=observations, timesteps=8, num_samples=6, method="dpm",
        device="cpu", verbose=False, capture_attention=False,
    )
    assert draws.shape == (4, 6, THETA)
    assert torch.isfinite(draws).all()


def test_attention_capture(model, observations):
    torch.manual_seed(4)
    model.sample(
        x=observations[:1], timesteps=8, num_samples=4, method="dpm",
        device="cpu", verbose=False, capture_attention=True,
    )
    weights = model.sampler.all_attn_weights
    assert len(weights) > 0, "no attention weights captured"
    tokens = model.token_count
    # tokens + 1 keys: add_bias_kv appends an always-attendable key, which is
    # what keeps an all-latent example from taking a softmax over nothing.
    assert weights[0].shape[-2:] == (tokens, tokens + 1)


def test_pfode_log_prob(model, observations):
    condition_mask = torch.cat([torch.zeros(THETA), torch.ones(TRIALS * ITEM)])
    joint = torch.cat([torch.zeros(4, THETA), observations], dim=1)
    values = model.log_prob(
        data=joint, condition_mask=condition_mask, timesteps=8,
        divergence="exact", device="cpu", verbose=False,
    )
    assert values.shape == (4,)
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("correction, method", [
    ("gauss_jacobian", "dpm"),
    ("gauss_hierarchical", "dpm"),
    ("fnpe", "langevin"),
])
def test_multi_observation_composition(model, observations, correction, method):
    torch.manual_seed(5)
    draws = model.sample(
        x=observations, timesteps=8, num_samples=4, method=method, order=2,
        multi_obs_inference=True, hierarchy=list(range(GLOBALS)),
        correction=correction, jacobian_refresh=20,
        precision_est_samples=32, precision_est_timesteps=8,
        corrector_steps=0, final_corrector_steps=0,
        device="cpu", verbose=False,
    )
    assert draws.shape[-1] == THETA
    assert torch.isfinite(draws).all()


def test_arrow_newton_map_with_jacobian_curvature(model, observations):
    """Forward-mode AD through the schema embedder and the shared trunk."""
    condition_mask = torch.cat([torch.zeros(THETA), torch.ones(TRIALS * ITEM)])
    start = torch.cat([torch.zeros(4, THETA), observations], dim=1)
    refined = model.multi_obs_sampler.newton_map_estimate(
        data=start, condition_mask=condition_mask, init=start,
        hierarchy=list(range(GLOBALS)),
        prior=([0.0] * GLOBALS, [1.0] * GLOBALS),
        local_prior=([0.0] * LOCALS, [1.0] * LOCALS),
        correction="gauss_jacobian", curvature="jacobian",
        timesteps=8, device="cpu",
    )
    refined = refined.reshape(start.shape)
    assert torch.isfinite(refined).all()
    # The observed trial features are conditioned and must come back untouched.
    assert torch.allclose(refined[:, THETA:], observations, atol=1e-6)


def test_training_runs_and_round_trips(tmp_path, model):
    """One short training run: grouped masks, checkpointing, reload."""
    torch.manual_seed(6)
    trained = ScoreBasedInferenceModel(
        theta_size=THETA, x_spec=[IID("trials", item_dim=ITEM, count=TRIALS)],
        hidden_size=8, depth=1, num_heads=2, mlp_ratio=2, device="cpu",
    )
    theta = torch.randn(64, THETA)
    x = torch.randn(64, TRIALS * ITEM)
    trained.train(theta, x, theta, x, batch_size=16, max_epochs=2, lr=1e-3,
                  device="cpu", verbose=False, path=str(tmp_path), name="arm")
    restored = ScoreBasedInferenceModel.load(
        str(tmp_path / "arm_checkpoint.pt"), device="cpu", arch="tokenized",
    )
    assert restored.schema == trained.schema
    assert restored.token_count == THETA + TRIALS
