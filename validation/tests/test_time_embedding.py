"""Opt-in log-lambda time map: default unchanged, remaps t, round-trips through save/load."""

import validation.common.runtime  # noqa: F401

import torch

from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm


def _model(te):
    torch.manual_seed(0)
    m = SBIm(nodes_size=5, hidden_size=16, depth=1, num_heads=2, mlp_ratio=2,
             time_embedding=te)
    # adaLN-Zero makes a fresh trunk output exactly zero; wake every weight so that a
    # change in the time input can show up in the output at all.
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in m.model.parameters():
            p.copy_(0.2 * torch.randn(p.shape, generator=g))
    return m


def _out(model, t):
    x, c = torch.randn(3, 5, generator=torch.Generator().manual_seed(1)), torch.zeros(3, 5)
    with torch.no_grad():
        return model.model(x=x, t=t, c=c)


def test_default_is_raw_and_checkpoint_unchanged(tmp_path):
    m = _model("raw")
    assert m.time_payload() == {}
    m.save(str(tmp_path), "m")
    assert "time_embedding" not in torch.load(tmp_path / "m.pt", weights_only=False)


def test_log_lambda_feeds_tau_to_the_network():
    raw, ll = _model("raw"), _model("log_lambda")
    ll.model.load_state_dict(raw.model.state_dict())
    t = torch.full((3, 1), 0.5)
    lam = raw.sde.lambda_t(t)
    one = torch.ones(1)
    lo, hi = torch.log(raw.sde.lambda_t(1e-3 * one)), torch.log(raw.sde.lambda_t(one))
    tau = 1000 * (torch.log(lam) - lo) / (hi - lo)
    assert torch.allclose(_out(ll, t), _out(raw, tau), atol=1e-6)
    assert not torch.allclose(_out(ll, t), _out(raw, t), atol=1e-4)
    t1 = torch.ones(3, 1)            # t = 1 maps to tau = 1000
    assert torch.allclose(_out(ll, t1), _out(raw, torch.full((3, 1), 1000.0)), atol=1e-5)


def test_round_trip(tmp_path):
    m = _model("log_lambda")
    m.save(str(tmp_path), "m")
    back = SBIm.load(str(tmp_path / "m.pt"), device="cpu")
    assert back.time_embedding == "log_lambda"
    t = torch.full((3, 1), 0.3)
    assert torch.allclose(_out(m, t), _out(back, t), atol=1e-6)
