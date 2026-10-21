"""The optional mask reweighting must not change Bernoulli mask sampling."""
from types import SimpleNamespace

import torch

from compass.SDE import VESDE
from compass.Trainer import Trainer


def make_trainer(nodes=3):
    trainer = Trainer(SimpleNamespace(sde=VESDE(sigma=5.0), nodes_size=nodes))
    trainer.mask_loss_weights = []
    return trainer


def test_unit_mask_weights_reproduce_legacy_loss():
    torch.manual_seed(4)
    trainer = make_trainer()
    score = torch.randn(11, 3)
    timestep = torch.rand(11, 1)
    noise = torch.randn(11, 3)
    mask = torch.randint(0, 2, (11, 3)).float()

    sigma = trainer.sde.marginal_prob_std(timestep).unsqueeze(1)
    legacy = torch.mean(
        sigma**2 * torch.sum(
            (1 - mask.unsqueeze(2))
            * (noise.unsqueeze(2) + sigma * score.unsqueeze(2))**2
        )
    )
    actual = trainer.loss_fn(score, timestep, noise, mask)
    torch.testing.assert_close(actual, legacy)


def test_mask_weight_changes_relative_not_overall_loss_scale():
    trainer = make_trainer(nodes=2)
    score = torch.zeros(2, 2)
    timestep = torch.full((2, 1), 0.4)
    noise = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    trainer.mask_loss_weights = [(torch.tensor([0.0, 1.0]), 3.0)]
    actual = trainer.loss_fn(score, timestep, noise, mask)

    sigma = trainer.sde.marginal_prob_std(timestep).square().mean()
    # Mean-one normalized weights are [1.5, 0.5]. Only the first row has
    # residual 1; the second row's only latent coordinate has residual zero.
    expected = sigma * 1.5
    torch.testing.assert_close(actual, expected)
