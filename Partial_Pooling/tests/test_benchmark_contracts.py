import torch

from Partial_Pooling.artifact_names import (
    checkpoint_manifest_path,
    checkpoint_path,
    inference_dataset_filename,
    inference_report_filename,
    normalization_path,
    recovery_run_name,
    test_data_path,
    training_index_path,
    training_shard_directory,
    training_shard_filename,
)
from Partial_Pooling.config import get_config
from Partial_Pooling.infer_partial_pooling import (
    DEFAULT_DATASETS,
    DEFAULT_DRAWS,
    DEFAULT_SUBJECTS,
    DEFAULT_TIMESTEPS,
    density_panel_counts,
    draw_batch_size,
    estimated_runtime_seconds,
    estimated_sweep_runtime_seconds,
    inference_plan,
    observation_counts,
    parser,
)
from Partial_Pooling.normalization import Normalizer
from Partial_Pooling.paths import BenchmarkPaths
from Partial_Pooling.plot_partial_pooling_comparison import (
    method_metadata,
    parser as comparison_parser,
)
from Partial_Pooling.schema import GLOBAL_NAMES, LOCAL_NAMES, JOINT_NAMES


def test_presets_and_train_only_normalization_provenance():
    smoke = get_config("smoke")
    full = get_config("full")
    large = get_config("large")
    assert (smoke.train_size, smoke.validation_size, smoke.test_datasets) == (2048, 256, 10)
    assert (full.train_size, full.validation_size, full.subjects) == (32768, 4096, 100)
    assert (large.train_size, large.validation_size, large.subjects) == (100000, 5000, 100)
    norm = Normalizer.fit(torch.tensor([[1.0, 2.0], [1.0, 4.0]]))
    assert norm.fitted_split == "train"
    assert norm.degenerate.tolist() == [True, False]
    assert norm.scale[0] == 1
    assert GLOBAL_NAMES == (
        "mu_nu", "mu_log_alpha", "mu_log_t0", "log_sigma_nu",
        "log_sigma_log_alpha", "log_sigma_log_t0", "beta_raw",
    )
    assert LOCAL_NAMES == ("nu", "log_alpha", "log_t0")
    assert JOINT_NAMES == GLOBAL_NAMES + LOCAL_NAMES


def test_inference_draw_batches_bound_flattened_transformer_rows():
    smoke = get_config("smoke")
    full = get_config("full")
    assert draw_batch_size(128, smoke.batch_size, subjects=20) == 6
    assert draw_batch_size(1000, full.batch_size, subjects=100) == 1
    assert draw_batch_size(128, smoke.batch_size, subjects=1) == 128


def test_focused_defaults_are_small_and_estimated_from_measured_work():
    seconds = estimated_runtime_seconds(
        DEFAULT_DATASETS, DEFAULT_SUBJECTS, DEFAULT_DRAWS, DEFAULT_TIMESTEPS,
    )
    assert (DEFAULT_DATASETS, DEFAULT_SUBJECTS) == (5, 20)
    assert (DEFAULT_DRAWS, DEFAULT_TIMESTEPS) == (256, 50)
    args = parser().parse_args([])
    assert args.inference_method == "dpm2_gaussian"
    primary = inference_plan(args.inference_method)
    reference = inference_plan("langevin_fnpse")
    assert (primary["sampler"], primary["correction"], primary["order"]) == (
        "dpm", "gauss", 2,
    )
    assert (reference["sampler"], reference["correction"]) == (
        "langevin", "fnpe",
    )
    assert not args.skip_observation_sweep
    assert observation_counts(20) == (1, 2, 4, 8, 16, 20)
    assert density_panel_counts(observation_counts(20)) == (1, 4, 8, 20)
    sweep_seconds = estimated_sweep_runtime_seconds(
        DEFAULT_DATASETS, DEFAULT_SUBJECTS, DEFAULT_DRAWS, DEFAULT_TIMESTEPS,
    )
    assert 300 < seconds < 600
    assert 2.5 * seconds < sweep_seconds < 2.6 * seconds


def test_plot_comparison_recognizes_legacy_and_current_methods():
    legacy = method_metadata({
        "correction": "damped_sum",
        "joint_map": {"settings": {}},
    })
    gaussian = method_metadata({
        "inference_method": "dpm2_gaussian",
        "sampler": "dpm",
        "correction": "gauss",
        "order": 2,
        "joint_map": {"settings": {"estimator": "hierarchical_score_map"}},
    })
    langevin = method_metadata({
        "inference_method": "langevin_fnpse",
        "sampler": "langevin",
        "correction": "fnpe",
        "joint_map": {
            "settings": {"estimator": "posterior_global_kde_mode"},
        },
    })
    assert legacy["key"] == "dpm2_damped_sum"
    assert gaussian["key"] == "dpm2_gaussian"
    assert gaussian["point_estimator"] == "hierarchical_score_map"
    assert langevin["key"] == "langevin_fnpse"
    assert langevin["point_estimator"] == "posterior_global_kde_mode"

    args = comparison_parser().parse_args([
        "--preset", "full",
        "--run-signature", "legacy",
        "--run-signature", "gaussian",
        "--output-signature", "comparison",
    ])
    assert args.run_signatures == ["legacy", "gaussian"]
    assert args.output_signature == "comparison"


def test_artifact_names_include_training_size_and_inference_method(tmp_path):
    config = get_config("full")
    paths = BenchmarkPaths(tmp_path)
    assert training_index_path(config, paths).name == (
        "partial_pooling-train-32768.json"
    )
    assert training_shard_directory(config, paths).name == (
        "partial_pooling-train-32768"
    )
    assert training_shard_filename(config, "train", 0, 2048) == (
        "partial_pooling-train-32768-train-shard-000000-002048.pt"
    )
    assert normalization_path(config, paths).name == (
        "partial_pooling-train-32768-normalization.pt"
    )
    assert test_data_path(config, paths).name == (
        "partial_pooling-train-32768-test-100.pt"
    )
    assert checkpoint_path(config, "sde_joint", paths).name == (
        "sde_joint-train-32768-checkpoint.pt"
    )
    assert checkpoint_manifest_path(config, "sde_joint", paths).name == (
        "sde_joint-train-32768-manifest.json"
    )
    assert recovery_run_name("dpm2_gaussian", "abc123") == (
        "dpm2_gaussian-abc123"
    )
    assert inference_dataset_filename("dpm2_gaussian", 7) == (
        "dpm2_gaussian-dataset-0007.pt"
    )
    assert inference_report_filename(
        "langevin_fnpse", "summary", "json",
    ) == "langevin_fnpse-summary.json"
