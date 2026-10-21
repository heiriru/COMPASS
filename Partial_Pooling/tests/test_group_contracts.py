"""Benchmark-side contracts for the grouped arm.

The grouped arm is the first one that is a **data** axis as well as a model one:
its training split is different simulations. That makes two invariants worth
pinning explicitly, because breaking either would silently make the new numbers
incomparable with the published ones:

* every existing signature, tag and filename is byte-identical to what it was;
* a grouped arm generates its own split but **borrows** the preset's
  normalization constants and test datasets, so its inputs are on the same
  scale as the flat arms it is measured against.
"""

import pytest
import torch

from Partial_Pooling.artifact_names import (
    base_training_tag, checkpoint_path, checkpoint_tag, normalization_path,
    test_data_path, training_tag,
)
from Partial_Pooling.config import get_config
from Partial_Pooling.model_pipeline import (
    build_mask_sampler, build_model, group_subjects, model_specs,
)
from Partial_Pooling.paths import BenchmarkPaths

GLOBALS, LOCALS, TRIALS, ITEM = 7, 3, 30, 3


def grouped(subjects=20, **overrides):
    return get_config(
        "small", arch="tokenized", tokenization="group", train_subjects=subjects,
        hidden_size=64, depth=2, num_heads=8, mlp_ratio=4, **overrides,
    )


#############################################
# ----- Nothing that existed may move -----
#############################################

def test_existing_signatures_and_names_are_untouched():
    """Four new config fields must invalidate no checkpoint and no split."""
    assert get_config("full").model_signature == "aacc35ff55f5604e"
    small = get_config("small")
    assert training_tag(small) == "partial_pooling-train-400000"
    assert checkpoint_tag(small, "sde_joint") == "sde_joint-train-400000"
    assert checkpoint_tag(get_config("large"), "sde_joint") == "sde_joint-train-100000"


def test_the_published_mask6mix_tag_is_unchanged():
    """The arm every table in docs/followup_results.md reports on."""
    config = get_config(
        "small", train_subset=100000, arch="tokenized", tokenization="trial",
        hidden_size=64, depth=2, num_heads=8, mlp_ratio=4,
        mask_menu="six_mixed", mask_bernoulli_fraction=0.25,
    )
    assert checkpoint_tag(config, "sde_joint") == (
        "sde_joint-tok-trial-h64d2x8m4-mask6b0p25-sub100000-train-400000"
    )
    assert config.data_signature == get_config("small").data_signature


#############################################
# ----- The grouped arm's own contracts -----
#############################################

def test_a_grouped_arm_gets_its_own_split():
    """Different simulations, so a different signature and a different name."""
    base = get_config("small")
    config = grouped()
    assert config.data_signature != base.data_signature
    assert training_tag(config) == "partial_pooling-train-400000-subj20"


def test_a_grouped_arm_borrows_normalization_and_test_data():
    """Refitting or regenerating either would make the arm incomparable."""
    paths = BenchmarkPaths.default()
    base, config = get_config("small"), grouped()
    assert base_training_tag(config) == training_tag(base)
    assert normalization_path(config, paths) == normalization_path(base, paths)
    assert test_data_path(config, paths) == test_data_path(base, paths)
    # ...and the borrowed files validate, which is what actually lets them load.
    assert base.data_signature in config.compatible_data_signatures


def test_the_split_size_is_independent_of_the_borrowed_constants():
    """A 5,000-group split still normalizes against the preset's 400,000-row fit."""
    paths = BenchmarkPaths.default()
    config = grouped(train_size=5000, validation_size=250)
    assert training_tag(config) == "partial_pooling-train-5000-subj20"
    assert normalization_path(config, paths).name == (
        "partial_pooling-train-400000-normalization.pt"
    )
    assert get_config("small").data_signature in config.compatible_data_signatures


@pytest.mark.parametrize("subjects", [5, 10, 20])
def test_grouped_arms_get_distinct_names(subjects):
    config = grouped(subjects, mask_menu="six_mixed", mask_bernoulli_fraction=0.25)
    assert config.architecture_tag == (
        f"tok-group-h64d2x8m4-gK{subjects}-mask6b0p25"
    )


def test_ablation_tags_are_distinct(tmp_path):
    paths = BenchmarkPaths(tmp_path)
    names = {
        checkpoint_path(config, "sde_joint", paths).name
        for config in (
            grouped(), grouped(depth_readout=1), grouped(group_pool="pma"),
            grouped(group_slot="none"), grouped(5), grouped(10),
        )
    }
    assert len(names) == 6


#############################################
# ----- Coherence -----
#############################################

@pytest.mark.parametrize("kwargs, message", [
    (dict(tokenization="trial", train_subjects=4), "tokenization='group'"),
    (dict(tokenization="group", train_subjects=3), "must divide"),
    (dict(tokenization="trial", depth_readout=1), "tokenization='group'"),
    (dict(tokenization="trial", group_slot="none"), "tokenization='group'"),
    (dict(tokenization="group", train_subjects=4, group_pool="median"), "group_pool"),
    (dict(tokenization="group", train_subjects=4, group_slot="fixed"), "group_slot"),
])
def test_incoherent_configurations_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        get_config("small", arch="tokenized", **kwargs)


def test_train_subjects_must_divide_the_test_subject_count():
    """100 test subjects have to regroup evenly into composed units."""
    for subjects in (5, 10, 20, 25, 50, 100):
        assert get_config("small").subjects % subjects == 0
        grouped(subjects)
    with pytest.raises(ValueError, match="must divide"):
        grouped(30)


#############################################
# ----- The model the config builds -----
#############################################

@pytest.mark.parametrize("subjects", [5, 10, 20])
def test_build_model_matches_the_configured_representation(subjects):
    config = grouped(subjects)
    torch.manual_seed(0)
    model = build_model(config, model_specs(config)["sde_joint"], device="cpu")
    assert model.arch == "grouped"
    assert group_subjects(config) == subjects
    assert model.nodes_size == GLOBALS + subjects * (LOCALS + TRIALS * ITEM)
    assert model.token_count == GLOBALS + subjects * (LOCALS + TRIALS)
    assert (model.depth_within, model.depth_across, model.depth_readout) == (2, 2, 0)


@pytest.mark.parametrize("subjects", [5, 10, 20, 50, 100])
def test_every_grouped_arm_is_under_the_parameter_budget(subjects):
    """The cap is 1,000,000 and it binds: see docs/group_block_progress.md."""
    config = grouped(subjects)
    torch.manual_seed(0)
    model = build_model(config, model_specs(config)["sde_joint"], device="cpu")
    parameters = sum(p.numel() for p in model.model.parameters())
    assert parameters < 1_000_000
    # Only the K x hidden_size slot table varies with K, so a K sweep measures
    # the representation rather than the capacity.
    assert 760_000 < parameters < 767_000


def test_one_readout_block_is_the_whole_headroom():
    config = grouped(20, depth_readout=1)
    torch.manual_seed(0)
    model = build_model(config, model_specs(config)["sde_joint"], device="cpu")
    parameters = sum(p.numel() for p in model.model.parameters())
    assert parameters == 910_341
    assert parameters < 1_000_000


def test_the_mask_menu_follows_the_nested_layout():
    """The six conditionals must mean the same six under a Group."""
    config = grouped(4)
    torch.manual_seed(0)
    model = build_model(config, model_specs(config)["sde_joint"], device="cpu")
    menu = build_mask_sampler(
        grouped(4, mask_menu="six_mixed", mask_bernoulli_fraction=0.25), model,
    )
    assert menu is not None
    conditioned = {name: int(row.sum()) for name, row in zip(menu.names, menu.masks)}
    subjects, trials = 4, TRIALS * ITEM
    assert conditioned["p(g,l|x)"] == subjects * trials
    assert conditioned["p(g|l,x)"] == subjects * (trials + LOCALS)
    assert conditioned["p(l|g,x)"] == subjects * trials + GLOBALS
    assert conditioned["p(x|g,l)"] == GLOBALS + subjects * LOCALS
    # No pattern may split a grouped token.
    for row in menu.masks:
        assert torch.equal(
            model.schema.expand_token_values(model.schema.token_mask(row)), row)
