"""Single-device training writes the final-epoch model as {name}.pt (C9.2 finding)."""

import validation.common.runtime  # noqa: F401

import os

import torch

from compass.ModelTransfuser import ModelTransfuser
from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm


def _data(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(n, 1, generator=g)
    return theta, theta + 0.3 * torch.randn(n, 1, generator=g)


def test_final_model_saved_and_equal_to_memory(tmp_path):
    theta, x = _data()
    torch.manual_seed(0)
    m = SBIm(nodes_size=2, hidden_size=16, depth=1, num_heads=2, mlp_ratio=2)
    m.train(theta, x, batch_size=64, max_epochs=3, device="cpu", verbose=False,
            path=str(tmp_path), name="M", early_stopping_patience=100)
    assert os.path.exists(tmp_path / "M.pt")
    back = SBIm.load(str(tmp_path / "M.pt"), device="cpu")
    for a, b in zip(m.model.state_dict().values(), back.model.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu())


def test_train_models_round_trip(tmp_path):
    mtf = ModelTransfuser(path=str(tmp_path))
    theta, x = _data()
    mtf.add_model("A", SBIm(nodes_size=2, hidden_size=16, depth=1, num_heads=2, mlp_ratio=2))
    mtf.add_data("A", theta, x)
    mtf.train_models(batch_size=64, max_epochs=2, device="cpu", verbose=False,
                     path=str(tmp_path))
    assert mtf.trained_models and os.path.exists(tmp_path / "A.pt")


def test_edm_time_weight_trains_and_default_unchanged(tmp_path):
    theta, x = _data()
    for tlw in (None, "edm"):
        torch.manual_seed(0)
        m = SBIm(nodes_size=2, hidden_size=16, depth=1, num_heads=2, mlp_ratio=2)
        m.train(theta, x, batch_size=64, max_epochs=2, device="cpu", verbose=False,
                time_loss_weight=tlw)
        assert m.trainer.time_loss_weight == tlw
        assert all(torch.isfinite(torch.tensor(m.trainer.train_loss)))


def test_inference_mask_menu_api():
    from compass.schema import ConditionalMenu
    m = SBIm(nodes_size=5, hidden_size=16, depth=1, num_heads=2, mlp_ratio=2)
    sampler = m._resolve_mask_sampler(None, "inference", 2, None, 0.0, 0.33)
    assert isinstance(sampler, ConditionalMenu)
    torch.manual_seed(0)
    masks = sampler.sample((30000, 5))
    post = (masks == torch.tensor([0., 0., 1., 1., 1.])).all(1).float().mean()
    like = (masks == torch.tensor([1., 1., 0., 0., 0.])).all(1).float().mean()
    # 1/3 from the menu each, plus their small Bernoulli share
    assert 0.33 < post < 0.37 and 0.33 < like < 0.39
