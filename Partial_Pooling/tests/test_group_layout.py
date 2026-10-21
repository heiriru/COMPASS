"""The 100-subjects <-> 100/K-groups mapping must be exactly invertible.

Everything downstream of inference -- ``_parameter_rows``, the recovery CSVs,
every figure -- is per-subject code that must not know grouping happened. That
holds only if un-grouping restores the test file's own subject order for *any*
grouping, which is what these tests pin, including the random re-assignments the
partition-sensitivity study varies.
"""

import pytest
import torch

from Partial_Pooling.group_layout import (
    GLOBAL_DIM, LOCAL_DIM, build_rows, condition_mask, extract_locals,
    group_observations, grouping, inverse_transform, map_condition_mask,
    split_latents, subjects_per_group, tile_local_prior, ungroup_locals,
)

SUBJECTS, OBS = 100, 90


def test_contiguous_grouping_is_the_default():
    assignment = grouping(SUBJECTS, 20)
    assert assignment.shape == (5, 20)
    assert torch.equal(assignment[0], torch.arange(20))
    assert torch.equal(assignment.reshape(-1), torch.arange(SUBJECTS))


def test_a_seed_permutes_without_losing_a_subject():
    assignment = grouping(SUBJECTS, 20, seed=3)
    assert sorted(assignment.reshape(-1).tolist()) == list(range(SUBJECTS))
    assert not torch.equal(assignment, grouping(SUBJECTS, 20))
    assert torch.equal(assignment, grouping(SUBJECTS, 20, seed=3))
    assert not torch.equal(assignment, grouping(SUBJECTS, 20, seed=4))


@pytest.mark.parametrize("per_group", [3, 7, 30])
def test_a_grouping_that_does_not_divide_is_refused(per_group):
    with pytest.raises(ValueError, match="must divide"):
        grouping(SUBJECTS, per_group)


@pytest.mark.parametrize("per_group", [1, 5, 10, 20, 50, 100])
@pytest.mark.parametrize("seed", [None, 11])
def test_grouping_and_ungrouping_round_trip(per_group, seed):
    """The property everything downstream depends on."""
    assignment = grouping(SUBJECTS, per_group, seed=seed)
    generator = torch.Generator().manual_seed(0)
    truth = torch.randn(SUBJECTS, LOCAL_DIM, generator=generator)
    rows = SUBJECTS // per_group
    # Lay the per-subject values out the way the sampler would return them.
    grouped = truth[assignment].reshape(rows, 1, per_group, LOCAL_DIM)
    restored = ungroup_locals(grouped, assignment)
    assert restored.shape == (1, SUBJECTS, LOCAL_DIM)
    assert torch.equal(restored[0], truth)


@pytest.mark.parametrize("per_group", [1, 5, 20])
def test_observations_are_gathered_in_slot_order(per_group):
    assignment = grouping(SUBJECTS, per_group, seed=5)
    generator = torch.Generator().manual_seed(1)
    observations = torch.randn(SUBJECTS, OBS, generator=generator)
    grouped = group_observations(observations, assignment)
    assert grouped.shape == (SUBJECTS // per_group, per_group * OBS)
    for row in range(grouped.shape[0]):
        for slot in range(per_group):
            chunk = grouped[row, slot * OBS:(slot + 1) * OBS]
            assert torch.equal(chunk, observations[assignment[row, slot]])


@pytest.mark.parametrize("per_group", [1, 5, 20])
def test_condition_mask_conditions_exactly_the_trials(per_group):
    mask = condition_mask(per_group, OBS)
    assert mask.numel() == GLOBAL_DIM + per_group * (LOCAL_DIM + OBS)
    assert mask.sum() == per_group * OBS
    assert mask[:GLOBAL_DIM].sum() == 0
    for slot in range(per_group):
        base = GLOBAL_DIM + slot * (LOCAL_DIM + OBS)
        assert mask[base:base + LOCAL_DIM].sum() == 0        # locals latent
        assert mask[base + LOCAL_DIM:base + LOCAL_DIM + OBS].sum() == OBS


def test_the_mask_selects_latents_in_node_order():
    """Which is what makes split_latents a slice rather than a gather."""
    per_group = 4
    mask = condition_mask(per_group, OBS)
    nodes = torch.arange(float(mask.numel()))
    latent = nodes[(1 - mask).bool()]
    shared, locals_ = split_latents(latent.reshape(1, 1, -1), per_group)
    assert torch.equal(shared[0, 0], torch.arange(float(GLOBAL_DIM)))
    for slot in range(per_group):
        base = GLOBAL_DIM + slot * (LOCAL_DIM + OBS)
        assert torch.equal(
            locals_[0, 0, slot], torch.arange(float(base), base + LOCAL_DIM))


def test_split_latents_rejects_the_wrong_width():
    with pytest.raises(ValueError, match="Expected"):
        split_latents(torch.zeros(2, 3, 12), per_group=4)


def test_ungroup_rejects_a_mismatched_assignment():
    assignment = grouping(SUBJECTS, 20)
    with pytest.raises(ValueError, match="do not match"):
        ungroup_locals(torch.zeros(5, 2, 10, LOCAL_DIM), assignment)


def test_local_prior_is_tiled_per_slot():
    mean, std = tile_local_prior(([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]), 3)
    assert mean.tolist() == [0.0, 1.0, 2.0] * 3
    assert std.tolist() == [1.0, 2.0, 3.0] * 3


def test_inverse_transform_uses_the_right_constants_per_block():
    class Normalizer:
        mean = torch.arange(10.0)
        scale = torch.arange(1.0, 11.0)

    shared = torch.zeros(1, 1, GLOBAL_DIM)
    locals_ = torch.zeros(1, 1, 2, LOCAL_DIM)
    shared, locals_ = inverse_transform(shared, locals_, Normalizer())
    assert torch.equal(shared[0, 0], torch.arange(7.0))
    for slot in range(2):
        assert torch.equal(locals_[0, 0, slot], torch.arange(7.0, 10.0))


#############################################
# ----- The MAP stage's own layout -----
#############################################

@pytest.mark.parametrize("per_group", [1, 5, 20])
def test_map_mask_conditions_the_globals_and_the_trials(per_group):
    """Stage 2 fixes the globals at their KDE mode; only the locals stay latent."""
    mask = map_condition_mask(per_group, OBS)
    assert mask.numel() == GLOBAL_DIM + per_group * (LOCAL_DIM + OBS)
    assert mask[:GLOBAL_DIM].sum() == GLOBAL_DIM
    assert mask.sum() == GLOBAL_DIM + per_group * OBS
    latent = int((1 - mask).sum())
    assert latent == per_group * LOCAL_DIM


@pytest.mark.parametrize("per_group", [1, 5, 20])
@pytest.mark.parametrize("seed", [None, 2])
def test_build_rows_and_extract_locals_round_trip(per_group, seed):
    """What the MAP stage relies on: locals go out and come back per subject."""
    assignment = grouping(SUBJECTS, per_group, seed=seed)
    generator = torch.Generator().manual_seed(4)
    shared = torch.randn(GLOBAL_DIM, generator=generator)
    locals_ = torch.randn(SUBJECTS, LOCAL_DIM, generator=generator)
    observations = torch.randn(SUBJECTS, OBS, generator=generator)

    rows = build_rows(shared, locals_, observations, assignment)
    assert rows.shape == (
        SUBJECTS // per_group, GLOBAL_DIM + per_group * (LOCAL_DIM + OBS),
    )
    assert torch.equal(rows[:, :GLOBAL_DIM], shared.expand(rows.shape[0], -1))
    assert torch.equal(extract_locals(rows, per_group, OBS, assignment), locals_)


def test_build_rows_places_each_subject_in_its_own_slot():
    per_group = 4
    assignment = grouping(SUBJECTS, per_group, seed=8)
    generator = torch.Generator().manual_seed(5)
    shared = torch.zeros(GLOBAL_DIM)
    locals_ = torch.randn(SUBJECTS, LOCAL_DIM, generator=generator)
    observations = torch.randn(SUBJECTS, OBS, generator=generator)
    rows = build_rows(shared, locals_, observations, assignment)
    width = LOCAL_DIM + OBS
    for row in range(rows.shape[0]):
        for slot in range(per_group):
            base = GLOBAL_DIM + slot * width
            subject = assignment[row, slot]
            assert torch.equal(rows[row, base:base + LOCAL_DIM], locals_[subject])
            assert torch.equal(
                rows[row, base + LOCAL_DIM:base + width], observations[subject])


#############################################
# ----- Flat arms take the old path -----
#############################################

def test_a_flat_model_reports_one_subject_per_group():
    """Which is what routes every published arm down the original code path."""
    class Flat:
        arch = "tokenized"

    class Legacy:
        arch = "legacy"

    assert subjects_per_group(Flat()) == 1
    assert subjects_per_group(Legacy()) == 1
    assert subjects_per_group(object()) == 1


def test_a_grouped_model_reports_its_count():
    class Block:
        kind = "group"
        count = 17

    class Schema:
        blocks = [type("A", (), {"kind": "atomic"})(), Block()]

    class Grouped:
        arch = "grouped"
        schema = Schema()

    assert subjects_per_group(Grouped()) == 17
