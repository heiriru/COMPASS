"""Benchmark-side contracts for the IID-embedding ladder.

The ladder arms share a preset with the legacy models they are compared against,
so the invariants that keep them apart -- and keep legacy artifacts valid -- are
worth pinning explicitly.
"""

import pytest
import torch

from Partial_Pooling.artifact_names import checkpoint_path, checkpoint_tag
from Partial_Pooling.config import get_config
from Partial_Pooling.model_pipeline import build_model, model_specs
from Partial_Pooling.paths import BenchmarkPaths


def ladder(**overrides):
    return get_config("large", arch="tokenized", **overrides)


def test_architecture_overrides_do_not_touch_the_data_signature():
    """Every ladder arm must reuse the same simulations and normalization."""
    legacy = get_config("large")
    for config in (
        ladder(),
        ladder(tokenization="trial"),
        ladder(tokenization="trial", hidden_size=32, depth=4),
        ladder(token_offset=True),
    ):
        assert config.data_signature == legacy.data_signature
        assert config.model_signature != legacy.model_signature


def test_legacy_signatures_and_names_are_unchanged():
    """Adding architecture fields must not invalidate any existing checkpoint."""
    full = get_config("full")
    assert full.model_signature == "aacc35ff55f5604e"
    assert full.architecture_tag == ""
    assert checkpoint_tag(get_config("small"), "sde_joint") == "sde_joint-train-400000"
    assert checkpoint_tag(get_config("large"), "sde_joint") == "sde_joint-train-100000"


def test_ladder_arms_get_distinct_checkpoint_names(tmp_path):
    paths = BenchmarkPaths(tmp_path)
    names = {
        checkpoint_path(config, "sde_joint", paths).name
        for config in (
            get_config("large"),
            ladder(hidden_size=16, depth=2, num_heads=2, mlp_ratio=4),
            ladder(tokenization="trial", hidden_size=16, depth=2, num_heads=2, mlp_ratio=4),
            ladder(tokenization="trial", hidden_size=32, depth=4, num_heads=4, mlp_ratio=4),
            ladder(tokenization="trial", hidden_size=16, depth=2, num_heads=2,
                   mlp_ratio=4, token_offset=True),
        )
    }
    assert len(names) == 5
    assert "sde_joint-train-100000-checkpoint.pt" in names
    assert (
        "sde_joint-tok-trial-h16d2x2m4-train-100000-checkpoint.pt" in names
    )


@pytest.mark.parametrize("kwargs, message", [
    (dict(arch="legacy", tokenization="trial"), "arch='legacy' is the per-node"),
    (dict(arch="legacy", trunk="position"), "arch='legacy' is the per-node"),
    (dict(arch="tokenized", tokenization="summary"), "Unknown tokenization"),
    (dict(arch="quantum"), "Unknown arch"),
])
def test_incoherent_configurations_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        get_config("large", **kwargs)


def test_build_model_matches_the_configured_representation():
    spec = model_specs(get_config("large"))["sde_joint"]
    assert (spec.theta_dim, spec.x_dim) == (10, 90)

    legacy = build_model(get_config("large", hidden_size=16, depth=2,
                                    num_heads=2, mlp_ratio=4), spec)
    assert legacy.arch == "legacy"
    assert legacy.token_count == 100
    assert sum(p.numel() for p in legacy.model.parameters()) == 5_957_984

    scalar = build_model(ladder(hidden_size=16, depth=2, num_heads=2,
                                mlp_ratio=4), spec)
    assert scalar.token_count == 100

    trial = build_model(ladder(tokenization="trial", hidden_size=16, depth=2,
                               num_heads=2, mlp_ratio=4), spec)
    assert trial.token_count == 40           # 10 theta + 30 trial tokens
    assert trial.nodes_size == 100           # the diffusion state is unchanged
    assert trial.schema.blocks[1].item_dim == 3

    # The trunk is token-count agnostic: the two tokenized arms differ only in
    # the embedder/head, never in the modulation cost.
    scalar_parameters = sum(p.numel() for p in scalar.model.parameters())
    trial_parameters = sum(p.numel() for p in trial.model.parameters())
    assert scalar_parameters < 1_000_000
    assert trial_parameters < 1_000_000
    assert abs(scalar_parameters - trial_parameters) < 5_000


def test_token_offset_is_the_only_token_count_dependent_option():
    spec = model_specs(get_config("large"))["sde_joint"]
    plain = build_model(ladder(hidden_size=16, depth=2, num_heads=2, mlp_ratio=4), spec)
    offset = build_model(ladder(hidden_size=16, depth=2, num_heads=2,
                                mlp_ratio=4, token_offset=True), spec)
    added = (sum(p.numel() for p in offset.model.parameters())
             - sum(p.numel() for p in plain.model.parameters()))
    assert added == 2 * 100 * 6 * 16          # depth x tokens x 6H


def test_training_subset_is_a_model_axis_not_a_data_axis(tmp_path):
    """100,000 rows of the 400,000-row split: same simulations, same test sets."""
    full = get_config("small")
    subset = get_config("small", train_subset=100_000)
    assert subset.data_signature == full.data_signature
    assert subset.model_signature != full.model_signature
    assert (subset.effective_train_size, subset.effective_validation_size) == (
        100_000, 5_000,
    )
    assert (full.effective_train_size, full.effective_validation_size) == (
        400_000, 20_000,
    )
    paths = BenchmarkPaths(tmp_path)
    assert checkpoint_path(full, "sde_joint", paths).name == (
        "sde_joint-train-400000-checkpoint.pt"
    )
    assert checkpoint_path(subset, "sde_joint", paths).name == (
        "sde_joint-sub100000-train-400000-checkpoint.pt"
    )
    assert checkpoint_path(
        get_config("small", train_subset=100_000, arch="tokenized",
                  tokenization="trial", hidden_size=16, depth=2,
                  num_heads=2, mlp_ratio=4),
        "sde_joint", paths,
    ).name == (
        "sde_joint-tok-trial-h16d2x2m4-sub100000-train-400000-checkpoint.pt"
    )


def test_training_subset_bounds_are_enforced():
    with pytest.raises(ValueError, match="train_subset must be in"):
        get_config("small", train_subset=500_000)
