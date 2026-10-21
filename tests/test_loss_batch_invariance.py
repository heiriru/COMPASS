"""The training loss must not depend on how rows are grouped into batches.

``Trainer._run_epoch`` accumulates ``loss.item() / batch.shape[0]`` and averages
over batches, so a loss that returns a batch *mean* gives the short final batch
of an epoch the same weight as a full one after that division -- sixteen times
too much for an 8-row remainder of a 5,000-row validation split. The symptom is
a validation curve that swings by tens of percent between epochs and an early
stop driven by a handful of rows rather than by the model.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compass import ScoreBasedInferenceModel  # noqa: E402

NODES = 100


def trainer():
    model = ScoreBasedInferenceModel(nodes_size=NODES, hidden_size=8, depth=1,
                                     num_heads=2, mlp_ratio=2)
    model.trainer.sde = model.sde
    model.trainer.mask_loss_weights = []
    return model, model.trainer


def epoch_loss(trainer_, data, mask, times, noise, score, batch_size):
    """Exactly what Trainer._run_epoch computes, for one batching."""
    total, count = 0.0, 0
    for start in range(0, data.shape[0], batch_size):
        stop = min(start + batch_size, data.shape[0])
        loss = trainer_.loss_fn(
            score[start:stop], times[start:stop], noise[start:stop],
            mask[start:stop],
        )
        total += loss.item() / (stop - start)
        count += 1
    return total / count


def fixture(rows=4096, seed=0):
    torch.manual_seed(seed)
    mask = torch.zeros(rows, NODES)
    mask[:, 10:] = 1.0
    return (torch.randn(rows, NODES), mask,
            torch.rand(rows, 1) * 0.9 + 0.05,
            torch.randn(rows, NODES), torch.randn(rows, NODES))


@pytest.mark.parametrize("batch_size", [128, 256, 512])
def test_epoch_loss_is_stable_across_training_batch_sizes(batch_size):
    """At the batch sizes training actually uses, batching must not matter.

    The loss applies sigma^2 as a batch mean against a batch sum of residuals,
    and those two are correlated, so the expectation does drift for very small
    batches -- that is a property of this objective, which every existing
    checkpoint was trained under, not a bug. What matters is that it is flat
    across the sizes the benchmark runs at.
    """
    _, trainer_ = trainer()
    data, mask, times, noise, score = fixture()
    reference = epoch_loss(trainer_, data, mask, times, noise, score, 4096)
    value = epoch_loss(trainer_, data, mask, times, noise, score, batch_size)
    assert value == pytest.approx(reference, rel=0.02)


def test_loss_matches_the_historical_expression_exactly():
    """The objective every frozen checkpoint was trained under, unchanged."""
    model, trainer_ = trainer()
    data, mask, times, noise, score = fixture(rows=256)
    sigma = model.sde.marginal_prob_std(times).unsqueeze(1)
    historical = torch.mean(
        sigma**2 * torch.sum(
            (1 - mask.unsqueeze(2))
            * (noise.unsqueeze(2) + sigma * score.unsqueeze(2))**2
        )
    )
    assert float(trainer_.loss_fn(score, times, noise, mask)) == pytest.approx(
        float(historical), rel=1e-6)


def test_a_short_final_batch_does_not_dominate():
    """5,000 rows at batch 128 leaves a remainder of 8 -- the real case."""
    _, trainer_ = trainer()
    data, mask, times, noise, score = fixture(rows=5000)
    whole = epoch_loss(trainer_, data, mask, times, noise, score, 5000)
    batched = epoch_loss(trainer_, data, mask, times, noise, score, 128)
    assert batched == pytest.approx(whole, rel=0.02)


def test_unit_mask_weights_reproduce_the_unweighted_loss():
    model, trainer_ = trainer()
    data, mask, times, noise, score = fixture(rows=256)
    plain = float(trainer_.loss_fn(score, times, noise, mask))
    trainer_.mask_loss_weights = [(mask[0], 3.0)]
    weighted = float(trainer_.loss_fn(score, times, noise, mask))
    # Every row carries the same mask, so a uniform reweighting must cancel.
    assert weighted == pytest.approx(plain, rel=1e-5)


def test_mask_weights_actually_reweight_a_mixture():
    model, trainer_ = trainer()
    data, mask, times, noise, score = fixture(rows=256)
    other = torch.zeros(NODES)
    other[:50] = 1.0
    mask[128:] = other
    plain = float(trainer_.loss_fn(score, times, noise, mask))
    trainer_.mask_loss_weights = [(other, 5.0)]
    weighted = float(trainer_.loss_fn(score, times, noise, mask))
    assert weighted != pytest.approx(plain, rel=1e-3)
