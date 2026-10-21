"""Contract tests for the out-of-src time-embedding rewiring.

Run: .COMPASS/bin/python -m pytest inverse_kinematics/experiments/test_te_rewire.py -q
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import pytest
import torch

import te_rewire as TE
from compass import ScoreBasedInferenceModel as SBIm
from compass.ConditionTransformer import ConditionTransformer

NODES, HIDDEN, DEPTH, HEADS, MLP = 6, 16, 2, 2, 2
SDE = {"sde_type": "vesde", "sigma": 8.0}


def build(te=None, device="cpu"):
    model = SBIm(nodes_size=NODES, arch="legacy", device=device, hidden_size=HIDDEN,
                 depth=DEPTH, num_heads=HEADS, mlp_ratio=MLP, **SDE)
    if te is not None:
        TE.rewire(model, te)
    return model


def _wake(trunk, std=0.02):
    """Break the library's two zero-initializations so behaviour is observable."""
    for block in trunk.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[1].weight, std=std)
    torch.nn.init.normal_(trunk.final_layer.adaLN_modulation[1].weight, std=std)
    torch.nn.init.normal_(trunk.final_layer.embedding_params, std=std)
    return trunk


def batch(n=4, device="cpu"):
    return (torch.randn(n, NODES, device=device),
            torch.rand(n, 1, device=device),
            torch.ones(n, NODES, device=device))


# --- the two defects this module exists to work around ----------------------

def test_sbim_discards_time_embedding_size_for_legacy():
    """SBIm pins legacy to 256 regardless of the argument (src line 136)."""
    model = SBIm(nodes_size=NODES, arch="legacy", hidden_size=HIDDEN, depth=DEPTH,
                 num_heads=HEADS, mlp_ratio=MLP, time_embedding_size=64, **SDE)
    assert model.time_embedding_size == 256
    assert model.model.blocks[0].adaLN_modulation[1].in_features == 256


def test_condition_transformer_miswires_a_nondefault_width():
    """Reaching past SBIm builds a model that cannot run a forward pass."""
    trunk = ConditionTransformer(nodes_size=NODES, hidden_size=HIDDEN, depth=DEPTH,
                                 num_heads=HEADS, mlp_ratio=MLP,
                                 time_embedding_size=64)
    assert trunk.t_embedder.frequency_embedding_size == 64
    assert trunk.blocks[0].adaLN_modulation[1].in_features == 256   # not forwarded
    x, t, c = batch()
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        trunk(x, t, c)


# --- the rewiring itself ----------------------------------------------------

@pytest.mark.parametrize("te", [256, 128, 64, 32])
def test_rewired_model_forwards_and_shapes_are_consistent(te):
    model = build(te)
    trunk = model.model
    assert trunk.time_embedding_size == te
    assert trunk.t_embedder.frequency_embedding_size == te
    assert trunk.blocks[0].adaLN_modulation[1].in_features == te
    assert trunk.final_layer.adaLN_modulation[1].in_features == te
    x, t, c = batch()
    assert trunk(x, t, c).shape == (4, NODES)


def test_te256_rewiring_is_the_same_model_as_stock():
    """A te=256 arm must be a genuine control, not a differently-initialized one."""
    stock = build(None)
    rewired = build(256)
    assert (sum(p.numel() for p in stock.model.parameters())
            == sum(p.numel() for p in rewired.model.parameters()))
    assert (set(stock.model.state_dict()) == set(rewired.model.state_dict()))
    rewired.model.load_state_dict(stock.model.state_dict())
    x, t, c = batch()
    with torch.no_grad():
        assert torch.equal(stock.model(x, t, c), rewired.model(x, t, c))


def test_parameter_budget_matches_the_closed_form():
    """Guards against a rewiring that silently leaves a 256-wide layer behind."""
    def predicted(n, h, d, mlp, te):
        t_emb = te * (mlp * h) + mlp * h + (mlp * h) * te + te
        block = (te * (6 * n * h) + 6 * n * h) + (4 * h * h + 4 * h + 2 * h) \
                + (h * (mlp * h) + mlp * h + (mlp * h) * h + h)
        return n * h + t_emb + d * block + te * (2 * n * h) + 2 * n * h + (n * h) * n
    for te in (256, 128, 64, 32):
        got = sum(p.numel() for p in build(te).model.parameters())
        assert got == predicted(NODES, HIDDEN, DEPTH, MLP, te), te


def test_adaln_is_zero_initialized_like_the_library():
    """adaLN-Zero: a fresh block must be the identity, as ConditionTransformer builds it."""
    trunk = build(64).model
    assert torch.all(trunk.blocks[0].adaLN_modulation[1].weight == 0)
    assert torch.all(trunk.final_layer.adaLN_modulation[1].weight == 0)


def test_a_training_step_updates_every_rewired_parameter():
    model = build(64)
    x, t, c = batch(8)
    loss = model.model(x, t, c).square().mean()
    loss.backward()
    for name, p in model.model.named_parameters():
        assert p.grad is not None, name
        if "adaLN" in name:                 # zero-init gates: grad reaches them anyway
            continue
    # the modulation must be *reachable*: wake it and check the output moves.
    # Both zero-inits have to be woken -- adaLN-Zero on every block AND the
    # final layer's zero ``embedding_params`` -- or a fresh legacy model emits
    # exactly zero and any behavioural assertion passes vacuously.
    _wake(model.model)
    with torch.no_grad():
        a = model.model(x, t, c)
        b = model.model(x, t + 0.3, c)
    assert not torch.allclose(a, b), "output is independent of t after rewiring"


# --- checkpoints ------------------------------------------------------------

def test_sbim_load_cannot_read_a_rewired_checkpoint(tmp_path):
    """Recorded so the failure is a raise, never a silently-wrong reconstruction."""
    model = build(64)
    model.save(str(tmp_path))                 # SBIm.save takes a DIRECTORY
    path = tmp_path / "Model.pt"
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    # A third defect, and a deliberate one: architecture_payload() returns {}
    # for a legacy model so legacy checkpoints stay byte-identical, which means
    # the width is never recorded however it is set. It has to be read off the
    # weights instead.
    assert "time_embedding_size" not in payload
    assert TE.width_of(payload["model_state_dict"]) == 64
    with pytest.raises(RuntimeError):
        SBIm.load(str(tmp_path) + "/Model.pt")        # but load rebuilds at 256


def test_load_rewired_round_trips_bit_for_bit(tmp_path):
    model = build(64)
    _wake(model.model)                               # else every output is zero
    model.save(str(tmp_path))
    path = tmp_path / "Model.pt"
    reloaded, te = TE.load_rewired(path, lambda w, d: build(w, d))
    assert te == 64
    x, t, c = batch()
    with torch.no_grad():
        assert torch.equal(model.model(x, t, c), reloaded.model(x, t, c))


def test_parameter_report_accounting_adds_up():
    report = TE.parameter_report(build(64).model)
    assert report["time_only"] + report["data_dependent"] == report["total"]
    assert 0.0 < report["data_dependent_share"] < 1.0
