import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import simulator as sim
from train_bimodal_bernoulli import mixed_rows


def test_mixed_rows_accepts_zero_focus_fraction():
    theta, x = mixed_rows(17, 0.0, 0.2, sim.generator("zero-focus-test", 4))

    assert theta.shape == (17, 4)
    assert x.shape == (17, 2)


def test_mixed_rows_accepts_all_focused_fraction():
    theta, x = mixed_rows(17, 1.0, 0.2, sim.generator("all-focus-test", 4))

    assert theta.shape == (17, 4)
    assert x.shape == (17, 2)
    assert torch.all(x[:, 0].abs() < 0.2)
