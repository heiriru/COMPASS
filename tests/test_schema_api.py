"""Stage 1: the schema API, and the guarantees that keep legacy models separate.

Three things are under test:

1. The schema itself -- layout, token grouping, the key-padding convention and
   the grouped training mask.
2. Constructor dispatch: ``nodes_size=`` still means legacy, ``theta_size=``/
   ``x_spec=`` mean the schema network, and the illegal combinations raise.
3. That ``arch="tokenized"`` with ``trunk="position"`` and a single atomic block
   is *the same model* as legacy -- proved by loading legacy weights into it and
   comparing outputs, not by assertion.
"""

import tempfile
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)

from compass import Atomic, IID, NodeSchema, ScoreBasedInferenceModel
from compass.schema import GroupedBernoulli
from compass.ConditionTransformer import ConditionTransformer
from compass.TokenizedTransformer import TokenizedTransformer


#############################################
# ----- Schema layout -----
#############################################

def test_legacy_schema_is_one_token_per_scalar():
    schema = NodeSchema.legacy(100)
    assert schema.nodes_size == 100
    assert schema.token_count == 100
    assert schema.is_atomic


def test_partial_pooling_schema_layout():
    schema = NodeSchema.from_spec(10, [IID("trials", item_dim=3, count=30)])
    assert schema.nodes_size == 100          # 10 theta + 30 trials x 3 features
    assert schema.token_count == 40          # 10 theta tokens + 30 trial tokens
    assert not schema.is_atomic
    assert schema.node_slices == [(0, 10), (10, 100)]
    assert schema.token_slices == [(0, 10), (10, 40)]


def test_block_view_is_item_major():
    schema = NodeSchema.from_spec(2, [IID("trials", item_dim=3, count=4)])
    x = torch.arange(14.0).reshape(1, 14)
    theta = schema.block_view(x, 0)
    trials = schema.block_view(x, 1)
    assert theta.shape == (1, 2, 1)
    assert trials.shape == (1, 4, 3)
    # Trial 0 is the first three observation scalars, i.e. item-major -- the
    # layout Partial_Pooling/schema.py:transformed_observations already writes.
    assert torch.equal(trials[0, 0], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(trials[0, 3], torch.tensor([11.0, 12.0, 13.0]))


def test_token_mask_requires_every_scalar_of_an_item():
    schema = NodeSchema.from_spec(2, [IID("trials", item_dim=3, count=3)])
    c = torch.tensor([[1.0, 0.0,  1, 1, 1,  1, 1, 0,  0, 0, 0]])
    mask = schema.token_mask(c)
    assert mask.shape == (1, 5)
    # theta: passes through. trials: fully observed, partially observed, latent.
    assert torch.equal(mask[0], torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]))


def test_atomic_token_mask_is_the_identity():
    schema = NodeSchema.legacy(7)
    c = (torch.rand(4, 7) < 0.5).float()
    assert torch.equal(schema.token_mask(c), c)


#############################################
# ----- Grouped training mask -----
#############################################

def test_grouped_bernoulli_matches_legacy_stream_for_atomic_schemas():
    """An all-atomic schema must draw the *identical* mask, not merely a
    same-distribution one, so legacy training stays reproducible."""
    schema = NodeSchema.legacy(100)
    torch.manual_seed(7)
    grouped = GroupedBernoulli(schema, 0.33).sample(torch.Size([100]))
    torch.manual_seed(7)
    legacy = torch.distributions.bernoulli.Bernoulli(0.33).sample(torch.Size([100]))
    assert torch.equal(grouped, legacy)


def test_grouped_bernoulli_keeps_items_whole():
    schema = NodeSchema.from_spec(10, [IID("trials", item_dim=3, count=30)])
    torch.manual_seed(0)
    mask = GroupedBernoulli(schema, 0.33).sample(torch.Size([4096, 100]))
    trials = mask[:, 10:].reshape(4096, 30, 3)
    # every scalar of a trial shares the trial's fate
    assert torch.equal(trials.min(dim=-1).values, trials.max(dim=-1).values)
    # and the marginal rate of observed scalars is unchanged, which is what
    # makes the grouped model comparable with the legacy one
    assert abs(mask.mean().item() - 0.33) < 0.01


def test_grouped_bernoulli_rejects_the_wrong_width():
    schema = NodeSchema.legacy(10)
    with pytest.raises(ValueError, match="trailing dimension"):
        GroupedBernoulli(schema).sample(torch.Size([4, 9]))


#############################################
# ----- Constructor dispatch -----
#############################################

def test_nodes_size_signature_still_means_legacy():
    model = ScoreBasedInferenceModel(nodes_size=12, hidden_size=4, depth=1,
                                     num_heads=1, mlp_ratio=2)
    assert model.arch == "legacy"
    assert isinstance(model.model, ConditionTransformer)
    assert model.nodes_size == 12


def test_schema_signature_means_tokenized():
    model = ScoreBasedInferenceModel(
        theta_size=10, x_spec=[IID("trials", item_dim=3, count=30)],
        hidden_size=16, depth=2, num_heads=2, mlp_ratio=4,
    )
    assert model.arch == "tokenized"
    assert isinstance(model.model, TokenizedTransformer)
    assert model.nodes_size == 100
    assert model.token_count == 40


def test_tokenized_raw_needs_only_nodes_size():
    """The Stage-4 'tokenized-raw' arm: 100 scalar tokens, efficient trunk."""
    model = ScoreBasedInferenceModel(nodes_size=100, arch="tokenized",
                                     hidden_size=16, depth=2, num_heads=2)
    assert model.arch == "tokenized"
    assert model.token_count == 100


@pytest.mark.parametrize("kwargs, message", [
    (dict(theta_size=10), "both theta_size"),
    (dict(x_spec=[Atomic("x", 3)]), "both theta_size"),
    (dict(nodes_size=10, theta_size=5, x_spec=[Atomic("x", 5)]), "not both"),
    (dict(theta_size=5, x_spec=[Atomic("x", 5)], arch="legacy"), "not available"),
    (dict(), "needs either"),
    (dict(nodes_size=10, arch="chimera"), "Unknown arch"),
])
def test_illegal_signatures_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ScoreBasedInferenceModel(hidden_size=4, depth=1, num_heads=1,
                                 mlp_ratio=2, **kwargs)


def test_legacy_arch_rejects_a_grouped_schema():
    schema = NodeSchema.from_spec(2, [IID("trials", item_dim=3, count=3)])
    with pytest.raises(ValueError, match="one token per scalar"):
        ScoreBasedInferenceModel(schema=schema, arch="legacy", hidden_size=4,
                                 depth=1, num_heads=1, mlp_ratio=2)


#############################################
# ----- Stage 1: tokenized/position == legacy -----
#############################################

@pytest.mark.parametrize("nodes_size, hidden, depth, heads, ratio", [
    (12, 4, 2, 1, 2),
    (100, 16, 2, 2, 4),
])
def test_position_trunk_is_the_legacy_model(nodes_size, hidden, depth, heads, ratio):
    """Same parameters, same shapes, same outputs -- a pure re-parameterization."""
    torch.manual_seed(11)
    legacy = ScoreBasedInferenceModel(nodes_size=nodes_size, hidden_size=hidden,
                                      depth=depth, num_heads=heads, mlp_ratio=ratio)
    tokenized = ScoreBasedInferenceModel(
        nodes_size=nodes_size, arch="tokenized", trunk="position",
        hidden_size=hidden, depth=depth, num_heads=heads, mlp_ratio=ratio,
    )
    legacy.model.eval()
    tokenized.model.eval()

    assert (sum(p.numel() for p in legacy.model.parameters())
            == sum(p.numel() for p in tokenized.model.parameters()))
    # Key-compatible state dicts are the point: the legacy weights load in.
    missing, unexpected = tokenized.model.load_state_dict(
        legacy.model.state_dict(), strict=True,
    )
    assert not missing and not unexpected

    generator = torch.Generator().manual_seed(3)
    x = torch.randn(5, nodes_size, generator=generator)
    t = torch.rand(5, 1, generator=generator) * 0.9 + 0.05
    c = (torch.rand(5, nodes_size, generator=generator) < 0.4).float()
    c[0] = 0.0
    c[1] = 1.0
    with torch.no_grad():
        assert torch.equal(legacy.model(x=x, t=t, c=c), tokenized.model(x=x, t=t, c=c))


def test_position_trunk_refuses_a_grouped_schema():
    schema = NodeSchema.from_spec(10, [IID("trials", item_dim=3, count=30)])
    with pytest.raises(ValueError, match="only defined for an all-atomic schema"):
        TokenizedTransformer(schema, hidden_size=8, depth=1, num_heads=1, trunk="position")


#############################################
# ----- Checkpoint dispatch -----
#############################################

def _roundtrip(model, directory, name):
    model.save(str(directory), name=name)
    return Path(directory) / f"{name}.pt"


def test_legacy_checkpoint_has_no_arch_version_and_loads_as_legacy():
    model = ScoreBasedInferenceModel(nodes_size=12, hidden_size=4, depth=1,
                                     num_heads=1, mlp_ratio=2)
    with tempfile.TemporaryDirectory() as directory:
        path = _roundtrip(model, directory, "legacy")
        blob = torch.load(path, map_location="cpu")
        assert "arch_version" not in blob
        assert "schema" not in blob
        restored = ScoreBasedInferenceModel.load(str(path), device="cpu")
        assert restored.arch == "legacy"
        ScoreBasedInferenceModel.load(str(path), device="cpu", arch="legacy")
        with pytest.raises(RuntimeError, match="predates arch_version"):
            ScoreBasedInferenceModel.load(str(path), device="cpu", arch="tokenized")


def test_tokenized_checkpoint_records_and_restores_its_schema():
    model = ScoreBasedInferenceModel(
        theta_size=10, x_spec=[IID("trials", item_dim=3, count=30)],
        hidden_size=16, depth=2, num_heads=2, mlp_ratio=4,
    )
    model.model.eval()
    with tempfile.TemporaryDirectory() as directory:
        path = _roundtrip(model, directory, "tokenized")
        blob = torch.load(path, map_location="cpu")
        assert blob["arch_version"] == 1
        assert blob["arch"] == "tokenized"
        assert blob["schema"]["blocks"][1]["type"] == "iid"

        restored = ScoreBasedInferenceModel.load(str(path), device="cpu")
        restored.model.eval()
        assert restored.arch == "tokenized"
        assert restored.schema == model.schema
        assert restored.trunk == "shared"

        generator = torch.Generator().manual_seed(5)
        x = torch.randn(3, 100, generator=generator)
        t = torch.rand(3, 1, generator=generator)
        c = (torch.rand(3, 100, generator=generator) < 0.5).float()
        with torch.no_grad():
            assert torch.equal(model.model(x=x, t=t, c=c), restored.model(x=x, t=t, c=c))

        with pytest.raises(RuntimeError, match="arch='legacy' was requested"):
            ScoreBasedInferenceModel.load(str(path), device="cpu", arch="legacy")


def test_future_arch_version_is_refused():
    model = ScoreBasedInferenceModel(nodes_size=12, arch="tokenized", hidden_size=4,
                                     depth=1, num_heads=1, mlp_ratio=2)
    with tempfile.TemporaryDirectory() as directory:
        path = _roundtrip(model, directory, "future")
        blob = torch.load(path, map_location="cpu")
        blob["arch_version"] = 99
        torch.save(blob, path)
        with pytest.raises(RuntimeError, match="arch_version 99"):
            ScoreBasedInferenceModel.load(str(path), device="cpu")
