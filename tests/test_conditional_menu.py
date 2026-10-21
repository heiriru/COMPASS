"""Correctness tests for the structured condition-mask menu.

The training-time condition mask decides *which conditional* each example
teaches. COMPASS has always drawn it as an independent Bernoulli per token,
which spreads the signal over all ``2**tokens`` conditionals; these tests pin
down the alternative, a menu of named conditionals over a (global, local,
observation) partition of the node vector.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compass.schema import (  # noqa: E402
    Atomic, ConditionalMenu, GroupedBernoulli, IID, NodeSchema,
    SIX_CONDITIONALS, hierarchical_roles, six_conditional_menu,
)

GLOBALS, LOCALS, TRIALS, FEATURES = 7, 3, 30, 3
NODES = GLOBALS + LOCALS + TRIALS * FEATURES


def trial_schema():
    return NodeSchema.from_spec(
        GLOBALS + LOCALS, [IID("trials", item_dim=FEATURES, count=TRIALS)],
    )


def scalar_schema():
    return NodeSchema.legacy(NODES)


# ----- the menu itself ---------------------------------------------------


def test_six_conditionals_are_the_six_non_degenerate_ones():
    """All 2**3 role subsets, minus the two that carry no training signal."""
    conditioned = {roles for _, roles in SIX_CONDITIONALS}
    every = {"".join(sorted(subset))
             for subset in ("", "g", "l", "x", "gl", "gx", "lx", "glx")}
    assert {"".join(sorted(c)) for c in conditioned} == every - {"", "glx"}
    assert len(SIX_CONDITIONALS) == 6
    # Names and role sets agree: everything left of "|" is latent.
    for name, roles in SIX_CONDITIONALS:
        latent, given = name[2:-1].split("|") if "|" in name else (name, "")
        assert set(given.replace(",", "")) == set(roles)
        assert set(latent.replace(",", "")) == {"g", "l", "x"} - set(roles)


@pytest.mark.parametrize("schema_fn", [trial_schema, scalar_schema])
def test_each_pattern_conditions_exactly_its_roles(schema_fn):
    schema = schema_fn()
    menu = six_conditional_menu(schema, GLOBALS, LOCALS)
    roles = hierarchical_roles(schema, GLOBALS, LOCALS)
    assert menu.names == [name for name, _ in SIX_CONDITIONALS]
    for row, (_, conditioned) in zip(menu.masks, SIX_CONDITIONALS):
        for role, index in roles.items():
            expected = 1.0 if role in conditioned else 0.0
            assert torch.all(row[index] == expected)


def test_deployed_posterior_mask_is_actually_drawn():
    """The point of the menu: p(g,l|x) is unreachable under Bernoulli(0.33)."""
    schema = trial_schema()
    posterior = torch.zeros(NODES)
    posterior[GLOBALS + LOCALS:] = 1.0

    menu = six_conditional_menu(schema, GLOBALS, LOCALS)
    torch.manual_seed(0)
    drawn = menu.sample((20000, NODES))
    frequency = (drawn == posterior).all(dim=1).float().mean().item()
    assert frequency == pytest.approx(1 / 6, abs=0.02)

    # The Bernoulli sampler's probability for the same mask, exactly.
    bernoulli = 0.67 ** (GLOBALS + LOCALS) * 0.33 ** TRIALS
    assert bernoulli < 1e-15


def test_sample_draws_only_menu_masks_and_is_uniform():
    menu = six_conditional_menu(trial_schema(), GLOBALS, LOCALS)
    torch.manual_seed(1)
    drawn = menu.sample((30000, NODES))
    distinct = torch.unique(drawn, dim=0)
    assert distinct.shape[0] == 6
    for row in menu.masks:
        share = (drawn == row).all(dim=1).float().mean().item()
        assert share == pytest.approx(1 / 6, abs=0.02)


def test_weights_are_honoured():
    schema = trial_schema()
    weights = [10.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    menu = six_conditional_menu(schema, GLOBALS, LOCALS, weights=weights)
    torch.manual_seed(2)
    drawn = menu.sample((30000, NODES))
    share = (drawn == menu.masks[0]).all(dim=1).float().mean().item()
    assert share == pytest.approx(10 / 15, abs=0.02)


def test_bernoulli_fraction_mixes_in_unstructured_masks():
    schema = trial_schema()
    menu = six_conditional_menu(
        schema, GLOBALS, LOCALS, bernoulli_fraction=0.5,
    )
    torch.manual_seed(3)
    drawn = menu.sample((20000, NODES))
    on_menu = torch.zeros(drawn.shape[0], dtype=torch.bool)
    for row in menu.masks:
        on_menu |= (drawn == row).all(dim=1)
    # Half the draws are Bernoulli; those essentially never land on the menu.
    assert on_menu.float().mean().item() == pytest.approx(0.5, abs=0.03)
    # The Bernoulli half still respects the schema's item grouping.
    tokens = schema.token_mask(drawn)
    assert torch.equal(schema.expand_token_values(tokens), drawn)


def test_bernoulli_fraction_one_matches_the_plain_bernoulli_marginal():
    schema = trial_schema()
    menu = six_conditional_menu(
        schema, GLOBALS, LOCALS, bernoulli_fraction=1.0,
        bernoulli_probability=0.33,
    )
    torch.manual_seed(4)
    drawn = menu.sample((20000, NODES))
    assert drawn.mean().item() == pytest.approx(0.33, abs=0.02)


# ----- shapes ------------------------------------------------------------


def test_sample_shapes_match_grouped_bernoulli():
    """The Trainer draws one mask per example with shape ``(nodes_size,)``."""
    schema = trial_schema()
    menu = six_conditional_menu(schema, GLOBALS, LOCALS)
    reference = GroupedBernoulli(schema, 0.33)
    for shape in [(NODES,), (8, NODES), (2, 3, NODES)]:
        assert menu.sample(shape).shape == reference.sample(shape).shape


def test_sample_rejects_a_wrong_trailing_dimension():
    menu = six_conditional_menu(trial_schema(), GLOBALS, LOCALS)
    with pytest.raises(ValueError, match="trailing dimension"):
        menu.sample((8, NODES + 1))


# ----- validation --------------------------------------------------------


def test_roles_must_partition_the_node_vector():
    schema = trial_schema()
    roles = hierarchical_roles(schema, GLOBALS, LOCALS)
    roles["x"] = roles["x"][:-1]
    with pytest.raises(ValueError, match="partition"):
        ConditionalMenu(schema, roles, SIX_CONDITIONALS)


def test_degenerate_patterns_are_rejected():
    schema = trial_schema()
    roles = hierarchical_roles(schema, GLOBALS, LOCALS)
    with pytest.raises(ValueError, match="conditions nothing"):
        ConditionalMenu(schema, roles, [("joint", "")])
    with pytest.raises(ValueError, match="conditions every node"):
        ConditionalMenu(schema, roles, [("all", "glx")])


def test_unknown_role_is_rejected():
    schema = trial_schema()
    roles = hierarchical_roles(schema, GLOBALS, LOCALS)
    with pytest.raises(ValueError, match="unknown roles"):
        ConditionalMenu(schema, roles, [("bad", "gz")])


def test_a_pattern_may_not_split_a_grouped_token():
    """An IID item must be conditioned or latent as a whole."""
    schema = trial_schema()
    first, rest = NODES - TRIALS * FEATURES, None
    roles = {
        "g": list(range(GLOBALS)),
        "l": list(range(GLOBALS, GLOBALS + LOCALS)),
        # Split one trial down the middle: feature 0 in one role, 1-2 in another.
        "x_head": [first],
        "x_tail": list(range(first + 1, NODES)),
    }
    with pytest.raises(ValueError, match="splits a grouped token"):
        ConditionalMenu(schema, roles, [("bad", "g,l,x_head".split(",")) ])


def test_hierarchical_roles_rejects_an_empty_observation_block():
    schema = NodeSchema.legacy(10)
    with pytest.raises(ValueError, match="no observation nodes"):
        hierarchical_roles(schema, 7, 3)


# ----- the Trainer wiring ------------------------------------------------


def test_trainer_defaults_to_the_historical_bernoulli():
    """No mask_sampler must leave the legacy draw bit-for-bit unchanged."""
    from compass import ScoreBasedInferenceModel

    model = ScoreBasedInferenceModel(nodes_size=NODES, hidden_size=8, depth=1,
                                     num_heads=2, mlp_ratio=2)
    trainer = model.trainer
    trainer.mask_sampler = None
    trainer.world_size = 1
    loader = trainer._prepare_data(torch.randn(16, NODES), batch_size=4, rank=0)
    sampler = loader.dataset.tensor2
    assert isinstance(sampler, GroupedBernoulli)
    assert sampler.probability == 0.33


def test_trainer_uses_an_explicit_mask_sampler():
    from compass import ScoreBasedInferenceModel

    model = ScoreBasedInferenceModel(
        theta_size=GLOBALS + LOCALS,
        x_spec=[IID("trials", item_dim=FEATURES, count=TRIALS)],
        arch="tokenized", hidden_size=8, depth=1, num_heads=2, mlp_ratio=2,
    )
    menu = six_conditional_menu(model.schema, GLOBALS, LOCALS)
    trainer = model.trainer
    trainer.mask_sampler = menu
    trainer.world_size = 1
    loader = trainer._prepare_data(torch.randn(16, NODES), batch_size=4, rank=0)
    assert loader.dataset.tensor2 is menu
    # And the masks that come out of the loader are menu masks.
    _, mask, _ = loader.dataset[0]
    assert any(torch.equal(mask, row) for row in menu.masks)


def test_menu_masks_survive_the_loss_as_nonzero_signal():
    """Every menu mask must leave at least one latent node to score."""
    from compass import ScoreBasedInferenceModel

    model = ScoreBasedInferenceModel(
        theta_size=GLOBALS + LOCALS,
        x_spec=[IID("trials", item_dim=FEATURES, count=TRIALS)],
        arch="tokenized", hidden_size=8, depth=1, num_heads=2, mlp_ratio=2,
    )
    trainer = model.trainer
    trainer.sde = model.sde
    trainer.mask_loss_weights = []
    menu = six_conditional_menu(model.schema, GLOBALS, LOCALS)
    torch.manual_seed(5)
    for row in menu.masks:
        mask = row.expand(4, NODES)
        assert (1 - mask).sum() > 0
        t = torch.rand(4, 1) * 0.9 + 0.05
        score = torch.randn(4, NODES)
        loss = trainer.loss_fn(score, t, torch.randn(4, NODES), mask)
        assert torch.isfinite(loss) and loss > 0
