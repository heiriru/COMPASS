"""Correctness checks that run against the *trained* ladder checkpoints.

Skipped wholesale when an arm has not been trained yet, so the file is safe on a
clean checkout. What it verifies, on the real benchmark data rather than on
random tensors:

* the checkpoint round-trips to the architecture it claims, and a legacy
  checkpoint still refuses to load as a tokenized one;
* the trial-token model is exactly exchangeable in the trials of a real
  observation, and its theta scores do not move when trials are permuted;
* the score is finite in the two masks inference actually uses (posterior and
  likelihood) and under the all-latent mask the samplers start from;
* the parameter count is what the ladder claims it is.
"""

from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)

from compass import ScoreBasedInferenceModel
from Partial_Pooling.artifact_names import checkpoint_path
from Partial_Pooling.config import get_config
from Partial_Pooling.model_pipeline import load_normalizers
from Partial_Pooling.paths import BenchmarkPaths

SHAPE = dict(hidden_size=16, depth=2, num_heads=2, mlp_ratio=4)
THETA = 10
TRIALS = 30
ITEM = 3
NODES = THETA + TRIALS * ITEM

# Every arm of the ladder that has been trained. Keys are the checkpoint
# name fragment; values are the `get_config` overrides that reproduce them.
def _shape(hidden, depth, heads):
    return dict(hidden_size=hidden, depth=depth, num_heads=heads, mlp_ratio=4)


HEADS = {16: 2, 32: 4, 48: 4, 64: 8}

ARMS = {
    "legacy_h16d2": dict(train_subset=100_000, **_shape(16, 2, 2)),
    "tok_trial_h16d2_offset": dict(
        train_subset=100_000, arch="tokenized", tokenization="trial",
        token_offset=True, **_shape(16, 2, 2),
    ),
}
for _hidden in (16, 32, 48, 64):
    for _depth in (2, 3, 4):
        ARMS[f"tok_trial_h{_hidden}d{_depth}"] = dict(
            train_subset=100_000, arch="tokenized", tokenization="trial",
            **_shape(_hidden, _depth, HEADS[_hidden]),
        )
ARMS["tok_scalar_h16d2"] = dict(
    train_subset=100_000, arch="tokenized", tokenization="scalar",
    **_shape(16, 2, 2),
)
for _hidden in (32, 48, 64):
    ARMS[f"tok_scalar_h{_hidden}d4"] = dict(
        train_subset=100_000, arch="tokenized", tokenization="scalar",
        **_shape(_hidden, 4, HEADS[_hidden]),
    )

# Independently computed from the trunk's shapes, so a silent change to the
# architecture is caught as a parameter-count change rather than only as a
# performance change.
EXPECTED_PARAMETERS = {
    "legacy_h16d2": 5_957_984,
    "tok_scalar_h16d2": 100_369,
    "tok_scalar_h32d4": 336_737,
    "tok_scalar_h48d4": 541_841,
    "tok_scalar_h64d4": 771_521,
    "tok_trial_h16d2": 97_604,
    "tok_trial_h16d3": 125_524,
    "tok_trial_h16d4": 153_444,
    "tok_trial_h32d2": 207_236,
    "tok_trial_h32d3": 269_220,
    "tok_trial_h32d4": 331_204,
    "tok_trial_h48d2": 329_156,
    "tok_trial_h48d3": 431_348,
    "tok_trial_h48d4": 533_540,
    "tok_trial_h64d2": 463_364,
    "tok_trial_h64d3": 611_908,
    "tok_trial_h64d4": 760_452,
    "tok_trial_h16d2_offset": 105_284,
}
EXPECTED_TOKENS = {
    name: (100 if "trial" not in name else 40) for name in ARMS
}
# The arm that meets the ladder's success criterion; called out by name so a
# retrain or a refactor that changes it fails loudly rather than quietly.
WINNER = "tok_trial_h64d2"
PARAMETER_BUDGET = 1_000_000


def arm_config(name):
    return get_config("small", **ARMS[name])


def load_arm(name):
    config = arm_config(name)
    paths = BenchmarkPaths.default()
    path = checkpoint_path(config, "sde_joint", paths)
    if not path.exists():
        pytest.skip(f"{name} is not trained yet ({path.name}).")
    model = ScoreBasedInferenceModel.load(
        str(path), device="cpu", arch=config.arch,
    )
    model.model.eval()
    return config, model


@pytest.fixture(scope="module")
def observation():
    """One real test observation, normalized exactly as inference normalizes it."""
    config = get_config("small")
    paths = BenchmarkPaths.default()
    blob = paths.data / (
        f"partial_pooling-train-{config.train_size}"
        f"-test-{config.test_datasets}.pt"
    )
    if not blob.exists():
        pytest.skip(f"Missing test data {blob}.")
    raw = torch.load(blob, map_location="cpu")["payload"]["sde"]["observations"][0, :3]
    return load_normalizers(config, paths)["sde_observations"].transform(raw)


@pytest.mark.parametrize("name", sorted(ARMS))
def test_checkpoint_identity_and_size(name):
    config, model = load_arm(name)
    assert model.arch == config.arch
    assert model.nodes_size == NODES
    assert model.token_count == EXPECTED_TOKENS[name]
    parameters = sum(p.numel() for p in model.model.parameters())
    assert parameters == EXPECTED_PARAMETERS[name], (
        f"{name}: {parameters:,} parameters, expected "
        f"{EXPECTED_PARAMETERS[name]:,}."
    )


@pytest.mark.parametrize("name", sorted(ARMS))
def test_scores_are_finite_in_every_inference_mask(name, observation):
    _, model = load_arm(name)
    batch = observation.shape[0]
    x = torch.cat([torch.randn(batch, THETA), observation], dim=1)
    t = torch.full((batch, 1), 0.5)
    posterior = torch.cat([torch.zeros(batch, THETA),
                           torch.ones(batch, TRIALS * ITEM)], dim=1)
    likelihood = 1.0 - posterior
    for label, mask in (("posterior", posterior), ("likelihood", likelihood),
                        ("all-latent", torch.zeros_like(posterior))):
        with torch.no_grad():
            out = model.model(x=x, t=t, c=mask)
        assert torch.isfinite(out).all(), f"{name}: non-finite score under {label}"


def _permuted_pair(model, observation, seed=0):
    batch = observation.shape[0]
    x = torch.cat([torch.randn(batch, THETA), observation], dim=1)
    t = torch.full((batch, 1), 0.3)
    c = torch.cat([torch.zeros(batch, THETA),
                   torch.ones(batch, TRIALS * ITEM)], dim=1)
    order = torch.randperm(TRIALS, generator=torch.Generator().manual_seed(seed))

    def permute(flat):
        head, tail = flat[:, :THETA], flat[:, THETA:].reshape(batch, TRIALS, ITEM)
        return torch.cat([head, tail[:, order].reshape(batch, -1)], dim=1)

    with torch.no_grad():
        base = model.model(x=x, t=t, c=c)
        permuted = model.model(x=permute(x), t=t, c=permute(c))
    return base, permuted, permute


@pytest.mark.parametrize(
    "name", [n for n in sorted(ARMS)
             if n.startswith("tok_trial") and not n.endswith("_offset")],
)
def test_trial_tokens_are_exchangeable(name, observation):
    """The property the trial tokenization exists to provide, on real data."""
    _, model = load_arm(name)
    base, permuted, permute = _permuted_pair(model, observation)
    assert torch.allclose(base[:, :THETA], permuted[:, :THETA], atol=1e-5)
    assert torch.allclose(permute(base)[:, THETA:], permuted[:, THETA:], atol=1e-5)


def test_token_offset_ablation_destroys_exchangeability(observation):
    """Why the Stage-2 modulation offset is not adopted, beyond its cost.

    A per-token offset on the shared adaLN modulation gives every trial token
    its own parameters, so permuting trials changes the answer. That trades away
    exactly what the trial tokenization was introduced to buy -- and it buys
    only 0.65 of held-out denoising loss at H=16/depth=2, against 3.9 for
    simply widening to H=32 at a comparable budget.
    """
    _, model = load_arm("tok_trial_h16d2_offset")
    base, permuted, _ = _permuted_pair(model, observation)
    assert not torch.allclose(base[:, :THETA], permuted[:, :THETA], atol=1e-5)


def test_scalar_tokenization_is_not_exchangeable(observation):
    """The control: without trial tokens, permuting trials changes the answer."""
    _, model = load_arm("tok_scalar_h16d2")
    batch = observation.shape[0]
    x = torch.cat([torch.randn(batch, THETA), observation], dim=1)
    t = torch.full((batch, 1), 0.3)
    c = torch.cat([torch.zeros(batch, THETA),
                   torch.ones(batch, TRIALS * ITEM)], dim=1)
    order = torch.randperm(TRIALS, generator=torch.Generator().manual_seed(0))

    def permute(flat):
        head, tail = flat[:, :THETA], flat[:, THETA:].reshape(batch, TRIALS, ITEM)
        return torch.cat([head, tail[:, order].reshape(batch, -1)], dim=1)

    with torch.no_grad():
        base = model.model(x=x, t=t, c=c)
        permuted = model.model(x=permute(x), t=t, c=permute(c))
    assert not torch.allclose(base[:, :THETA], permuted[:, :THETA], atol=1e-4)


@pytest.mark.parametrize(
    "name", [n for n in sorted(ARMS) if n.startswith("tok_")],
)
def test_tokenized_checkpoint_refuses_to_load_as_legacy(name):
    config = arm_config(name)
    path = checkpoint_path(config, "sde_joint", BenchmarkPaths.default())
    if not path.exists():
        pytest.skip(f"{name} is not trained yet.")
    with pytest.raises(RuntimeError, match="not convertible"):
        ScoreBasedInferenceModel.load(str(path), device="cpu", arch="legacy")


def test_legacy_checkpoint_refuses_to_load_as_tokenized():
    path = checkpoint_path(arm_config("legacy_h16d2"), "sde_joint",
                           BenchmarkPaths.default())
    if not path.exists():
        pytest.skip("legacy_h16d2 is not trained yet.")
    with pytest.raises(RuntimeError, match="predates arch_version"):
        ScoreBasedInferenceModel.load(str(path), device="cpu", arch="tokenized")


def test_winning_arm_is_within_the_parameter_budget():
    """The ladder's headline claim, as an assertion rather than a table entry."""
    _, model = load_arm(WINNER)
    parameters = sum(p.numel() for p in model.model.parameters())
    assert parameters == EXPECTED_PARAMETERS[WINNER]
    assert parameters < PARAMETER_BUDGET
    assert parameters * 12 < 5_957_984, "the point is an order of magnitude"
    assert model.token_count == 40
    assert model.schema.blocks[1].representation == "trial_tokens"
