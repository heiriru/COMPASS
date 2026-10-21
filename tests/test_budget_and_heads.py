"""Correctness tests for the two architecture knobs added after Stage 4.

``time_embedding_size`` (task 3) and ``per_position_head`` (task 4) are both
meant to be *cheap*: neither may change the network's contract -- flat vector in,
flat vector out, trials exchangeable, checkpoints round-tripping -- and neither
may silently alter an arm trained before it existed.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compass import (  # noqa: E402
    IID, NodeSchema, ScoreBasedInferenceModel, parameter_breakdown,
)
from compass.TokenizedTransformer import SchemaHead, TokenizedTransformer  # noqa: E402

GLOBALS, LOCALS, TRIALS, FEATURES = 7, 3, 30, 3
THETA = GLOBALS + LOCALS
NODES = THETA + TRIALS * FEATURES
HIDDEN, DEPTH, HEADS, RATIO = 64, 2, 8, 4


def build(**kwargs):
    return ScoreBasedInferenceModel(
        theta_size=THETA, x_spec=[IID("trials", item_dim=FEATURES, count=TRIALS)],
        arch="tokenized", hidden_size=HIDDEN, depth=DEPTH, num_heads=HEADS,
        mlp_ratio=RATIO, **kwargs,
    )


def count(model):
    return sum(p.numel() for p in model.model.parameters())


def inputs(batch=4):
    torch.manual_seed(0)
    x = torch.randn(batch, NODES)
    t = torch.rand(batch, 1) * 0.9 + 0.05
    c = torch.zeros(batch, NODES)
    c[:, THETA:] = 1.0          # the deployed posterior mask
    return x, t, c


# ----- task 3: the diffusion-time embedding ------------------------------


def test_default_time_embedding_size_is_unchanged():
    """The pre-existing trial-h64/d2 arm must still be 463,364 parameters."""
    model = build()
    assert model.time_embedding_size == 256
    assert count(model) == 463_364


def test_time_embedding_size_shrinks_the_adaln_input_not_just_the_clock():
    """It sets the *input width* of every adaLN modulation, so it compounds."""
    wide = parameter_breakdown(build(time_embedding_size=256).model)
    narrow = parameter_breakdown(build(time_embedding_size=64).model)
    assert wide["time_embedding"] == 131_584
    # Every consumer of the time embedding shrinks with it, not only t_embedder.
    for key in ("time_embedding", "block_adaLN", "final_layer"):
        assert narrow[key] < wide[key]
    # The data pathway (attention + MLP + token embedding) is untouched.
    for key in ("attention_and_mlp", "input_embedding"):
        assert narrow[key] == wide[key]
    assert narrow["total"] == 192_836


@pytest.mark.parametrize("size", [32, 64, 128, 256])
def test_forward_pass_is_finite_at_every_time_embedding_size(size):
    model = build(time_embedding_size=size)
    x, t, c = inputs()
    out = model.model(x=x, t=t, c=c)
    assert out.shape == (4, NODES)
    assert torch.isfinite(out).all()


def test_time_embedding_size_is_rejected_on_the_position_trunk():
    """trunk='position' exists to be bit-for-bit legacy; it is pinned at 256."""
    schema = NodeSchema.legacy(NODES)
    with pytest.raises(ValueError, match="fixed at time_embedding_size=256"):
        TokenizedTransformer(schema, hidden_size=8, depth=1, num_heads=2,
                             trunk="position", time_embedding_size=64)


def test_legacy_arch_ignores_the_option():
    model = ScoreBasedInferenceModel(nodes_size=NODES, hidden_size=8, depth=1,
                                     num_heads=2, mlp_ratio=2,
                                     time_embedding_size=64)
    assert model.time_embedding_size == 256


# ----- task 4: per-position output heads ---------------------------------


def test_per_position_head_costs_exactly_one_readout_per_atomic_scalar():
    """A shared Linear(H, 1) becomes a (dim, H) weight plus a (dim,) bias."""
    shared, per_position = build(), build(per_position_head=True)
    expected = (THETA * HIDDEN + THETA) - (HIDDEN + 1)
    assert count(per_position) - count(shared) == expected == 585


def test_per_position_head_gives_each_theta_its_own_readout():
    """The defect it fixes: one shared direction for every parameter."""
    model = build(per_position_head=True)
    head = model.model.final_layer.head
    assert head.per_position
    weights = head.weight_for(0)
    assert weights.shape == (THETA, HIDDEN)

    x, t, c = inputs()
    baseline = model.model(x=x, t=t, c=c)
    with torch.no_grad():
        weights[2] += 5.0          # perturb the readout of theta dimension 2
    perturbed = model.model(x=x, t=t, c=c)
    changed = (~torch.isclose(baseline, perturbed, atol=1e-6)).any(dim=0)
    assert changed[2]
    assert not changed[torch.arange(NODES) != 2].any()


def test_shared_head_couples_every_theta_dimension():
    """Contrast: with one head, a readout change moves all ten at once."""
    model = build()
    head = model.model.final_layer.head
    assert not head.per_position
    x, t, c = inputs()
    baseline = model.model(x=x, t=t, c=c)
    with torch.no_grad():
        head.heads[0].weight += 5.0
    perturbed = model.model(x=x, t=t, c=c)
    changed = (~torch.isclose(baseline, perturbed, atol=1e-6)).any(dim=0)
    assert changed[:THETA].all()


def test_per_position_head_keeps_trials_exchangeable():
    """IID blocks stay shared on purpose; permuting trials permutes the output."""
    model = build(per_position_head=True)
    x, t, c = inputs(batch=2)
    out = model.model(x=x, t=t, c=c)

    permutation = torch.randperm(TRIALS)
    def shuffle(flat):
        head, tail = flat[:, :THETA], flat[:, THETA:]
        tail = tail.reshape(-1, TRIALS, FEATURES)[:, permutation].reshape(
            -1, TRIALS * FEATURES)
        return torch.cat((head, tail), dim=1)

    permuted = model.model(x=shuffle(x), t=t, c=shuffle(c))
    assert torch.allclose(shuffle(out), permuted, atol=1e-5)


def test_per_position_head_is_not_zero_at_initialization():
    """Zero-init would make the whole theta score identically zero at step 0."""
    model = build(per_position_head=True)
    weights = model.model.final_layer.head.weight_for(0)
    assert weights.abs().max() > 0
    bound = (6.0 / (HIDDEN + 1)) ** 0.5
    assert weights.abs().max() <= bound + 1e-6


def test_per_position_head_is_rejected_on_the_position_trunk():
    schema = NodeSchema.legacy(NODES)
    with pytest.raises(ValueError, match="per_position_head is a trunk='shared'"):
        TokenizedTransformer(schema, hidden_size=8, depth=1, num_heads=2,
                             trunk="position", per_position_head=True)


def test_schema_head_shapes_agree_between_modes():
    schema = NodeSchema.from_spec(
        THETA, [IID("trials", item_dim=FEATURES, count=TRIALS)])
    tokens = torch.randn(3, schema.token_count, HIDDEN)
    for flag in (False, True):
        assert SchemaHead(schema, HIDDEN, per_position=flag)(tokens).shape == (3, NODES)


# ----- checkpoints -------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {}, {"time_embedding_size": 64}, {"per_position_head": True},
    {"time_embedding_size": 64, "per_position_head": True},
])
def test_checkpoint_round_trip(tmp_path, kwargs):
    model = build(**kwargs)
    model.save(str(tmp_path), "m")
    back = ScoreBasedInferenceModel.load(str(tmp_path / "m.pt"), device="cpu")
    assert back.time_embedding_size == kwargs.get("time_embedding_size", 256)
    assert back.per_position_head == kwargs.get("per_position_head", False)
    x, t, c = inputs()
    assert torch.equal(model.model(x=x, t=t, c=c), back.model(x=x, t=t, c=c))


def test_defaults_are_not_written_into_the_payload():
    """An arm trained before these options existed must still round-trip."""
    payload = build().architecture_payload()
    assert "time_embedding_size" not in payload
    assert "per_position_head" not in payload
    assert payload["arch_version"] == 1


def test_a_checkpoint_without_the_new_keys_loads_at_the_old_defaults(tmp_path):
    model = build()
    model.save(str(tmp_path), "m")
    blob = torch.load(str(tmp_path / "m.pt"), map_location="cpu")
    assert "time_embedding_size" not in blob and "per_position_head" not in blob
    back = ScoreBasedInferenceModel.load(str(tmp_path / "m.pt"), device="cpu")
    assert (back.time_embedding_size, back.per_position_head) == (256, False)


def test_every_checkpoint_on_disk_still_loads():
    """Regression: a new nn.Parameter must not appear in the state dict.

    Registering placeholder parameters for the *unused* head mode is invisible
    to a save/load round trip -- both sides gain the same keys -- but it makes
    every checkpoint trained before the option unloadable. This test is the one
    that catches that, because it reads checkpoints written months earlier.
    """
    root = Path(__file__).resolve().parents[1] / "Partial_Pooling" / "artifacts"
    checkpoints = sorted(root.glob("checkpoints/*/*/*-checkpoint.pt"))
    if not checkpoints:
        pytest.skip("no trained checkpoints in this working tree")
    for path in checkpoints:
        ScoreBasedInferenceModel.load(str(path), device="cpu")
