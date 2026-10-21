"""Validation must score the model on the same task every epoch.

The training condition mask is redrawn per visit, which is what makes it an
augmentation. Doing the same on the validation split turns validation into a
moving target. For a ConditionalMenu the six conditionals differ by roughly a
factor of fifty in loss magnitude, so which masks a validation epoch happens to
draw moves the reported number far more than a real improvement does -- and both
early stopping and best-checkpoint selection then fire on a lucky draw.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compass import (  # noqa: E402
    IID, ScoreBasedInferenceModel, six_conditional_menu,
)
from compass.schema import GroupedBernoulli  # noqa: E402

GLOBALS, LOCALS, TRIALS, FEATURES = 7, 3, 30, 3
NODES = GLOBALS + LOCALS + TRIALS * FEATURES


def model_and_trainer(menu=False):
    model = ScoreBasedInferenceModel(
        theta_size=GLOBALS + LOCALS,
        x_spec=[IID("trials", item_dim=FEATURES, count=TRIALS)],
        arch="tokenized", hidden_size=8, depth=1, num_heads=2, mlp_ratio=2,
    )
    trainer = model.trainer
    trainer.world_size = 1
    trainer.validation_mask_seed = 20250718
    trainer.mask_sampler = (
        six_conditional_menu(model.schema, GLOBALS, LOCALS) if menu else None
    )
    return model, trainer


@pytest.mark.parametrize("menu", [False, True])
def test_fixed_validation_masks_are_reproducible(menu):
    _, trainer = model_and_trainer(menu)
    data = torch.randn(64, NODES)
    first = trainer._fixed_validation_masks(data)
    second = trainer._fixed_validation_masks(data)
    assert torch.equal(first, second)
    assert first.shape == (64, NODES)


@pytest.mark.parametrize("menu", [False, True])
def test_drawing_validation_masks_does_not_disturb_the_global_rng(menu):
    _, trainer = model_and_trainer(menu)
    data = torch.randn(64, NODES)
    torch.manual_seed(0)
    expected = torch.randn(8)
    torch.manual_seed(0)
    trainer._fixed_validation_masks(data)
    assert torch.equal(torch.randn(8), expected)


@pytest.mark.parametrize("menu", [False, True])
def test_validation_loader_repeats_the_same_mask_for_a_row(menu):
    _, trainer = model_and_trainer(menu)
    data = torch.randn(32, NODES)
    loader = trainer._prepare_data(
        data, batch_size=8, rank=0,
        fixed_masks=trainer._fixed_validation_masks(data),
    )
    first = torch.stack([loader.dataset[i][1] for i in range(32)])
    second = torch.stack([loader.dataset[i][1] for i in range(32)])
    assert torch.equal(first, second)


def test_the_training_loader_still_redraws_every_visit():
    """Training masks are an augmentation and must not be frozen."""
    _, trainer = model_and_trainer(menu=False)
    data = torch.randn(256, NODES)
    loader = trainer._prepare_data(data, batch_size=8, rank=0)
    assert isinstance(loader.dataset.tensor2, GroupedBernoulli)
    first = torch.stack([loader.dataset[i][1] for i in range(256)])
    second = torch.stack([loader.dataset[i][1] for i in range(256)])
    assert not torch.equal(first, second)


def test_a_menu_validation_set_covers_every_conditional():
    """A fixed draw must still exercise all six, not collapse onto one."""
    model, trainer = model_and_trainer(menu=True)
    data = torch.randn(4096, NODES)
    masks = trainer._fixed_validation_masks(data)
    menu = trainer.mask_sampler
    seen = [int((masks == row).all(dim=1).sum()) for row in menu.masks]
    assert all(count > 0 for count in seen), seen
