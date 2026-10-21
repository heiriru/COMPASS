"""The ``Group`` block: layout, masking, roles, and the symmetry it does *not* claim.

``Group`` is structural, not exchangeable. Read that as a contract with two
halves, both tested here:

* trials permuted **within** a repeat permute exactly that repeat's tokens --
  the inner ``IID`` block's guarantee, inherited unchanged;
* repeats permuted among themselves promise **nothing**. The schema layer cannot
  test the second half (it has no weights), so what is tested here is that the
  layout keeps each repeat's locals attached to its own trials, which is the
  property the network then relies on. The score-level version lives in
  ``tests/test_group_transformer.py``.

Everything else is the same contract ``tests/test_schema_api.py`` pins for
``Atomic`` and ``IID``, re-asserted through one more level of nesting.
"""

import pytest
import torch

torch.set_num_threads(1)

from compass import Atomic, Group, IID, NodeSchema
from compass.schema import (
    ConditionalMenu, GroupedBernoulli, SIX_CONDITIONALS,
    hierarchical_roles, nested_hierarchical_roles, six_conditional_menu,
)

GLOBALS, LOCALS, TRIALS, ITEM = 7, 3, 30, 3
INNER_WIDTH = LOCALS + TRIALS * ITEM      # 93
INNER_TOKENS = LOCALS + TRIALS            # 33


def subject_group(count, **kwargs):
    return Group("subjects", count=count, blocks=[
        Atomic("local", LOCALS), IID("trials", ITEM, TRIALS),
    ], **kwargs)


def schema(count=4, **kwargs):
    return NodeSchema([Atomic("theta", GLOBALS), subject_group(count, **kwargs)])


#############################################
# ----- Layout -----
#############################################

@pytest.mark.parametrize("count", [1, 2, 5, 20])
def test_layout_is_repeat_major(count):
    s = schema(count)
    assert s.nodes_size == GLOBALS + count * INNER_WIDTH
    assert s.token_count == GLOBALS + count * INNER_TOKENS
    assert s.node_slices == [(0, GLOBALS), (GLOBALS, s.nodes_size)]
    assert s.token_slices == [(0, GLOBALS), (GLOBALS, s.token_count)]


def test_group_is_not_atomic_and_is_flagged_as_nested():
    s = schema(3)
    assert not s.is_atomic
    assert s.has_group
    assert not NodeSchema([Atomic("theta", 10), IID("trials", 3, 30)]).has_group


def test_a_group_of_atomics_is_still_not_reported_atomic():
    """One token per scalar, but the repeat structure is real.

    ``trunk="position"`` claims bit-for-bit legacy equivalence and keys off
    ``is_atomic``; it must never accept a nested schema even when the token and
    scalar counts happen to coincide.
    """
    s = NodeSchema([Atomic("theta", 4), Group("subjects", 3, [Atomic("local", 2)])])
    assert s.nodes_size == s.token_count == 4 + 6
    assert not s.is_atomic


def test_repeat_view_separates_the_repeats():
    s = schema(3)
    x = torch.arange(float(s.nodes_size)).reshape(1, -1)
    view = s.repeat_view(x, 1)
    assert view.shape == (1, 3, INNER_WIDTH)
    # Repeat r starts at GLOBALS + r * INNER_WIDTH and is contiguous.
    for repeat in range(3):
        base = GLOBALS + repeat * INNER_WIDTH
        assert torch.equal(view[0, repeat], torch.arange(float(base), base + INNER_WIDTH))


def test_block_view_refuses_a_group_and_says_what_to_use():
    s = schema(2)
    with pytest.raises(TypeError, match="repeat_view"):
        s.block_view(torch.zeros(1, s.nodes_size), 1)


def test_repeat_view_refuses_a_non_group():
    s = schema(2)
    with pytest.raises(TypeError, match="only defined for Group"):
        s.repeat_view(torch.zeros(1, s.nodes_size), 0)


def test_nesting_a_group_inside_a_group_is_refused():
    with pytest.raises(ValueError, match="two-level"):
        Group("outer", 2, [subject_group(2)])


@pytest.mark.parametrize("kwargs, message", [
    (dict(pool="median"), "pool"),
    (dict(slot="random"), "slot"),
])
def test_unimplemented_options_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        subject_group(2, **kwargs)


def test_group_has_no_item_dim():
    """Its repeats hold tokens of different widths, so there is no single one."""
    with pytest.raises(TypeError, match="no single item_dim"):
        subject_group(2).item_dim


#############################################
# ----- Masking -----
#############################################

def test_token_mask_and_expand_are_inverse():
    s = schema(4)
    torch.manual_seed(0)
    tokens = (torch.rand(8, s.token_count) < 0.5).float()
    assert torch.equal(s.token_mask(s.expand_token_values(tokens)), tokens)


def test_a_partially_observed_trial_is_latent_as_a_whole():
    s = schema(2)
    mask = torch.ones(1, s.nodes_size)
    # Flip one scalar of repeat 1's third trial.
    node = GLOBALS + INNER_WIDTH + LOCALS + 2 * ITEM + 1
    mask[0, node] = 0.0
    token = s.token_mask(mask)
    trial_token = GLOBALS + INNER_TOKENS + LOCALS + 2
    assert token[0, trial_token] == 0.0
    assert token.sum() == s.token_count - 1


def test_masking_one_repeat_leaves_the_others_untouched():
    """The nesting is what this block exists for: repeats do not bleed."""
    s = schema(3)
    mask = torch.ones(1, s.nodes_size)
    start = GLOBALS + INNER_WIDTH
    mask[0, start:start + INNER_WIDTH] = 0.0        # repeat 1, entirely latent
    token = s.token_mask(mask)
    lo, hi = GLOBALS + INNER_TOKENS, GLOBALS + 2 * INNER_TOKENS
    assert token[0, lo:hi].sum() == 0.0
    assert token[0, :lo].sum() == lo
    assert token[0, hi:].sum() == token.shape[-1] - hi


def test_grouped_bernoulli_keeps_items_whole_and_holds_the_rate():
    s = schema(4)
    torch.manual_seed(1)
    sample = GroupedBernoulli(s, 0.33).sample((2048, s.nodes_size))
    assert torch.equal(s.expand_token_values(s.token_mask(sample)), sample)
    assert abs(sample.mean().item() - 0.33) < 0.02


#############################################
# ----- Roles and menus -----
#############################################

@pytest.mark.parametrize("count", [1, 2, 5, 20])
def test_nested_roles_partition_the_node_vector(count):
    s = schema(count)
    roles = nested_hierarchical_roles(s, GLOBALS, LOCALS)
    combined = sorted(roles["g"] + roles["l"] + roles["x"])
    assert combined == list(range(s.nodes_size))
    assert len(roles["g"]) == GLOBALS
    assert len(roles["l"]) == count * LOCALS
    assert len(roles["x"]) == count * TRIALS * ITEM


def test_locals_are_collected_from_every_repeat():
    s = schema(3)
    roles = nested_hierarchical_roles(s, GLOBALS, LOCALS)
    expected = []
    for repeat in range(3):
        base = GLOBALS + repeat * INNER_WIDTH
        expected.extend(range(base, base + LOCALS))
    assert roles["l"] == expected


def test_hierarchical_roles_delegates_for_a_nested_schema():
    """A caller that forgot to switch must not get the flat answer."""
    s = schema(3)
    assert hierarchical_roles(s, GLOBALS, LOCALS) == \
        nested_hierarchical_roles(s, GLOBALS, LOCALS)
    # The flat answer would have put 3 contiguous nodes in "l"; the nested one
    # has 3 per repeat, scattered.
    assert len(hierarchical_roles(s, GLOBALS, LOCALS)["l"]) == 9


def test_flat_schema_roles_are_unchanged():
    flat = NodeSchema([Atomic("theta", 10), IID("trials", ITEM, TRIALS)])
    roles = hierarchical_roles(flat, GLOBALS, LOCALS)
    assert roles["g"] == list(range(7))
    assert roles["l"] == [7, 8, 9]
    assert roles["x"] == list(range(10, 100))


def test_a_local_size_off_an_inner_boundary_is_refused():
    s = schema(2)
    with pytest.raises(ValueError, match="inner block boundary"):
        nested_hierarchical_roles(s, GLOBALS, 2)


def test_roles_require_the_globals_to_precede_the_group():
    s = NodeSchema([subject_group(2), Atomic("theta", GLOBALS)])
    with pytest.raises(ValueError, match="before the Group block"):
        nested_hierarchical_roles(s, GLOBALS, LOCALS)


def test_six_conditionals_survive_the_nesting():
    """Every menu pattern must be representable without splitting a token."""
    s = schema(4)
    menu = six_conditional_menu(s, GLOBALS, LOCALS)
    assert menu.names == [name for name, _ in SIX_CONDITIONALS]
    for row in menu.masks:
        assert torch.equal(s.expand_token_values(s.token_mask(row)), row)
    draws = menu.sample((256, s.nodes_size))
    assert draws.shape == (256, s.nodes_size)
    assert torch.equal(s.expand_token_values(s.token_mask(draws)), draws)


def test_the_deployed_posterior_mask_is_in_the_menu():
    """p(g,l|x): every theta node latent, every observation conditioned."""
    s = schema(4)
    menu = six_conditional_menu(s, GLOBALS, LOCALS)
    roles = nested_hierarchical_roles(s, GLOBALS, LOCALS)
    expected = torch.zeros(s.nodes_size)
    expected[roles["x"]] = 1.0
    assert any(torch.equal(row, expected) for row in menu.masks)


def test_a_pattern_that_splits_a_repeat_is_rejected():
    s = schema(2)
    roles = nested_hierarchical_roles(s, GLOBALS, LOCALS)
    # Condition one scalar of one trial: a role name is one character, since
    # ConditionalMenu reads a pattern's conditioned roles character by character.
    roles["h"] = [GLOBALS + LOCALS]
    roles["x"] = [i for i in roles["x"] if i != GLOBALS + LOCALS]
    with pytest.raises(ValueError, match="splits a grouped token"):
        ConditionalMenu(s, roles, [("bad", "h")])


#############################################
# ----- Serialization -----
#############################################

@pytest.mark.parametrize("pool, slot", [
    ("mean", "learned"), ("mean", "none"), ("pma", "sinusoidal"),
])
def test_schema_round_trips_through_a_dict(pool, slot):
    s = schema(5, pool=pool, slot=slot)
    restored = NodeSchema.from_dict(s.to_dict())
    assert restored == s
    assert restored.nodes_size == s.nodes_size
    assert restored.token_count == s.token_count
    block = restored.blocks[1]
    assert (block.pool, block.slot, block.count) == (pool, slot, 5)
    assert block.inner.blocks == [Atomic("local", LOCALS), IID("trials", ITEM, TRIALS)]


def test_slot_and_pool_defaults_are_the_documented_ones():
    """slot defaults to "learned": the block promises no subject symmetry."""
    block = subject_group(3)
    assert block.slot == "learned"
    assert block.pool == "mean"


def test_from_spec_accepts_a_bare_group():
    s = NodeSchema.from_spec(GLOBALS, subject_group(3))
    assert s.nodes_size == GLOBALS + 3 * INNER_WIDTH
    assert s.blocks[0] == Atomic("theta", GLOBALS)
