from types import SimpleNamespace

import torch

from compass.ConditionTransformer import TransformerBlock
from compass.Trainer import Trainer


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
