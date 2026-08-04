from types import SimpleNamespace

import torch

from compass.ConditionTransformer import TransformerBlock
from compass.MultiObsSampler import MultiObsSampler
from compass.Sampler import Sampler
from compass.SDE import VESDE
from compass.Trainer import Trainer


class DeterministicSampler:
    def __init__(self):
        self.calls = []
        self.offsets = {}

    def sample(self, data, num_samples, capture_attention, **kwargs):
        subject = int(torch.as_tensor(data)[0, 0].item())
        start = self.offsets.get(subject, 0)
        self.offsets[subject] = start + num_samples
        self.calls.append((subject, num_samples, capture_attention))
        draw = torch.arange(start, start + num_samples, dtype=torch.float32)
        result = torch.zeros(1, num_samples, 4)
        result[0, :, 0] = draw + subject
        result[0, :, 1] = 2 * draw - subject
        return result


def estimate_precision(subjects, samples, batch_size):
    single_sampler = DeterministicSampler()
    sampler = MultiObsSampler(SimpleNamespace(sde=object(), sampler=single_sampler))
    sampler.hierarchy = [0, 1]
    sampler.verbose = False
    data = torch.arange(subjects, dtype=torch.float32).unsqueeze(1)
    precision = sampler._estimate_posterior_precision(
        data=data,
        condition_mask=torch.tensor([0, 0, 1, 1]),
        num_samples=samples,
        timesteps=3,
        eps=1e-3,
        batch_size=batch_size,
        device="cpu",
    )
    return precision, single_sampler.calls


def test_precision_estimation_preserves_256_draws_with_bounded_calls():
    precision, calls = estimate_precision(subjects=20, samples=256, batch_size=128)
    assert precision.shape == (20, 2)
    for subject in range(20):
        subject_calls = [count for index, count, _ in calls if index == subject]
        assert subject_calls == [128, 128]
        assert sum(subject_calls) == 256
    assert max(count for _, count, _ in calls) <= 128
    assert all(not capture for _, _, capture in calls)


def test_precision_estimation_handles_non_divisible_draw_count():
    _, calls = estimate_precision(subjects=2, samples=257, batch_size=128)
    for subject in range(2):
        assert [count for index, count, _ in calls if index == subject] == [
            128, 128, 1,
        ]


def test_batched_and_unbatched_precision_match_deterministic_sampler():
    batched, _ = estimate_precision(subjects=3, samples=257, batch_size=128)
    unbatched, _ = estimate_precision(subjects=3, samples=257, batch_size=257)
    torch.testing.assert_close(batched, unbatched)


class AttentionRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.capture_calls = []

    def forward(self, x, t, c, return_attn_weights=False):
        self.capture_calls.append(return_attn_weights)
        score = torch.zeros_like(x)
        if return_attn_weights:
            return score, torch.ones(1, x.shape[-1], x.shape[-1])
        return score


def run_single_observation_sampler(capture_attention=True):
    model = AttentionRecordingModel()
    sbim = SimpleNamespace(
        model=model,
        sde=VESDE(sigma=2.0),
        output_scale_function=lambda _t, score: score,
    )
    sampler = Sampler(sbim)
    sampler.sample(
        world_size=1,
        data=torch.ones(1, 1),
        condition_mask=torch.tensor([0.0, 1.0]),
        timesteps=3,
        num_samples=2,
        method="euler",
        verbose=False,
        capture_attention=capture_attention,
    )
    return sampler, model


def test_attention_capture_can_be_disabled_or_retained():
    disabled, disabled_model = run_single_observation_sampler(False)
    assert disabled.all_attn_weights is None
    assert not any(disabled_model.capture_calls)

    enabled, enabled_model = run_single_observation_sampler()
    assert enabled.all_attn_weights is not None
    assert enabled.all_attn_weights.numel() > 0
    assert sum(enabled_model.capture_calls) == 1


def test_key_padding_mask_matches_intended_expanded_attention_mask():
    torch.manual_seed(4)
    block = TransformerBlock(hidden_size=8, num_heads=2, nodes_size=5)
    q = torch.randn(3, 5, 8)
    condition_mask = torch.tensor([
        [0, 0, 1, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ])
    expanded = (
        (1 - condition_mask).bool()[:, None, None, :]
        .expand(3, 2, 5, 5)
        .reshape(6, 5, 5)
    )
    compact = (1 - condition_mask).bool()
    with torch.no_grad():
        expected = block.attn(
            q, q, q, need_weights=False, attn_mask=expanded
        )[0]
        actual = block.attn(
            q, q, q, need_weights=False, key_padding_mask=compact
        )[0]
    torch.testing.assert_close(actual, expected)


def test_validation_uses_configured_training_batch_size():
    model = torch.nn.Linear(1, 1)
    trainer = Trainer(SimpleNamespace(model=model, sde=object()))
    trainer.world_size = 1
    trainer.batch_size = 17
    trainer.max_epochs = 0
    trainer.early_stopping_patience = 1
    trainer.lr = 1e-3
    trainer.verbose = False
    trainer.device = "cpu"
    seen = []

    def prepare(data, batch_size, rank):
        seen.append(batch_size)
        return []

    trainer._prepare_data = prepare
    trainer._train_loop(0, torch.zeros(2, 1), torch.zeros(2, 1))
    assert seen == [17, 17]
