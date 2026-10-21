"""Tests for compass.experimental.latent_attention."""

import pytest
import torch

from compass import ScoreBasedInferenceModel as SBIm, IID, hierarchical_roles
from compass.experimental.latent_attention import (
    enable_latent_attention, load_latent_attention, save_latent_attention,
)

G, L, TRIALS, ITEM = 2, 2, 4, 3


def build(seed=0, iid=True):
    torch.manual_seed(seed)
    if iid:
        model = SBIm(theta_size=G + L, x_spec=[IID("trials", item_dim=ITEM, count=TRIALS)],
                     arch="tokenized", hidden_size=16, depth=2, num_heads=2, mlp_ratio=2)
    else:
        model = SBIm(nodes_size=G + L + 3, arch="tokenized", hidden_size=16, depth=2,
                     num_heads=2, mlp_ratio=2)
    # adaLN-Zero makes a fresh trunk the identity; wake it.
    for module in model.model.modules():
        if hasattr(module, "adaLN_modulation"):
            torch.nn.init.normal_(module.adaLN_modulation[-1].weight, std=0.3)
            torch.nn.init.normal_(module.adaLN_modulation[-1].bias, std=0.3)
    return model


def roles(model):
    return hierarchical_roles(model.schema, G, L)


def jacobian(net, mask, seed=1):
    torch.manual_seed(seed)
    N = mask.numel()
    x = torch.randn(1, N)
    t = torch.tensor([0.4])
    c = mask.reshape(1, N)
    return torch.autograd.functional.jacobian(lambda z: net(x=z, t=t, c=c)[0], x)[:, 0, :]


def block(J, rows, cols):
    return J[rows][:, cols].abs().max().item()


@pytest.mark.parametrize("iid", [True, False])
def test_mode_none_reproduces_the_production_trunk(iid):
    model = build(iid=iid)
    N = model.nodes_size
    x, t = torch.randn(5, N), torch.rand(5, 1)
    masks = torch.zeros(5, N)
    masks[:, G + L:] = 1
    masks[1, :G] = 1
    masks[2, :] = 0
    masks[2, G:G + L] = 1
    before = model.model(x=x, t=t, c=masks)
    enable_latent_attention(model, "none", roles(model))
    after = model.model(x=x, t=t, c=masks)
    assert torch.allclose(before, after, atol=1e-5)


def test_parameters_and_state_dict_are_unchanged():
    model = build()
    keys = set(model.model.state_dict())
    enable_latent_attention(model, "full", roles(model))
    assert set(model.model.state_dict()) == keys


@pytest.mark.parametrize("mode", ["none", "global", "hierarchical", "full"])
def test_posterior_mask_coupling_structure(mode):
    model = build()
    enable_latent_attention(model, mode, roles(model))
    N = model.nodes_size
    mask = torch.zeros(N)
    mask[G + L:] = 1                                   # p(g, l | x)
    J = jacobian(model.model, mask)
    g, l = list(range(G)), list(range(G, G + L))
    off = lambda idx: J[idx][:, idx] - torch.diag(torch.diag(J[idx][:, idx]))
    gg, ll = off(g).abs().max().item(), off(l).abs().max().item()
    gl, lg = block(J, g, l), block(J, l, g)
    tiny = 1e-7
    expected = {
        "none": dict(gg=False, ll=False, gl=False, lg=False),
        "global": dict(gg=True, ll=False, gl=False, lg=False),
        "hierarchical": dict(gg=True, ll=True, gl=False, lg=True),
        "full": dict(gg=True, ll=True, gl=True, lg=True),
    }[mode]
    got = dict(gg=gg > tiny, ll=ll > tiny, gl=gl > tiny, lg=lg > tiny)
    assert got == expected, (mode, dict(gg=gg, ll=ll, gl=gl, lg=lg))


@pytest.mark.parametrize("mode", ["none", "global", "hierarchical", "full"])
def test_latent_trials_stay_conditionally_independent(mode):
    model = build()
    enable_latent_attention(model, mode, roles(model))
    N = model.nodes_size
    mask = torch.zeros(N)
    mask[:G + L] = 1                                   # p(x | g, l)
    J = jacobian(model.model, mask)
    for a in range(TRIALS):
        for b in range(TRIALS):
            if a == b:
                continue
            ra = list(range(G + L + a * ITEM, G + L + (a + 1) * ITEM))
            rb = list(range(G + L + b * ITEM, G + L + (b + 1) * ITEM))
            assert block(J, ra, rb) == 0.0


def test_sidecar_round_trip(tmp_path):
    model = build()
    enable_latent_attention(model, "hierarchical", roles(model))
    model.save(str(tmp_path), name="Model")
    save_latent_attention(model, tmp_path)
    restored = SBIm.load(str(tmp_path / "Model.pt"), device="cpu")
    load_latent_attention(restored, tmp_path)
    assert restored.model.latent_attention == "hierarchical"
    N = model.nodes_size
    x, t = torch.randn(3, N), torch.rand(3, 1)
    c = torch.zeros(3, N)
    c[:, G + L:] = 1
    restored.model.to(x.device)
    assert torch.allclose(model.model(x=x, t=t, c=c), restored.model(x=x, t=t, c=c), atol=1e-6)


def test_training_step_runs_and_attention_weights_are_available():
    model = build()
    enable_latent_attention(model, "full", roles(model))
    N = model.nodes_size
    x, t = torch.randn(4, N), torch.rand(4, 1)
    c = torch.zeros(4, N)
    c[:, G + L:] = 1
    out = model.model(x=x, t=t, c=c)
    out.square().mean().backward()
    assert all(p.grad is not None for p in model.model.blocks[0].attn.parameters() if p.requires_grad)
    _, weights = model.model(x=x, t=t, c=c, return_attn_weights=True)
    assert weights.shape[0] == len(model.model.blocks)
