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
    _global_kde_state,
    _hierarchy_validation_state,
    _is_numerical_map_failure,
    _joint_candidate_diagnostics,
    _robust_density_limits,
    _safe_1d_kde,
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


def test_marginal_kde_treats_identical_float32_draws_as_degenerate():
    values = torch.full((256,), -4.2529325, dtype=torch.float32)
    grid = torch.linspace(-4.5, -4.0, 50).numpy()
    assert torch.all(values == values[0])
    assert _safe_1d_kde(values.numpy(), grid) is None


def test_marginal_kde_returns_finite_density_for_varying_draws():
    values = torch.linspace(-1.0, 1.0, 64).numpy()
    grid = torch.linspace(-1.5, 1.5, 50).numpy()
    density = _safe_1d_kde(values, grid)
    assert density is not None
    assert density.shape == grid.shape
    assert torch.isfinite(torch.from_numpy(density)).all()


def test_robust_density_limits_ignore_a_minority_of_solver_excursions():
    values = torch.cat((
        torch.linspace(-1.0, 1.0, 95),
        torch.full((5,), 1e18),
    )).numpy()
    lower, upper = _robust_density_limits([values], anchors=(0.25,))
    assert lower < -1.0
    assert upper > 1.0
    assert upper < 10.0


def test_marginal_kde_preserves_central_density_amid_solver_excursions():
    values = torch.cat((
        torch.linspace(-1.0, 1.0, 95),
        torch.full((5,), 1e18),
    )).numpy()
    grid = torch.linspace(-2.0, 2.0, 101).numpy()
    density = _safe_1d_kde(values, grid)
    assert density is not None
    assert density[50] > density[0]
    assert density[50] > density[-1]


def test_focused_defaults_are_small_and_estimated_from_measured_work():
    seconds = estimated_runtime_seconds(
        DEFAULT_DATASETS, DEFAULT_SUBJECTS, DEFAULT_DRAWS, DEFAULT_TIMESTEPS,
    )
    assert (DEFAULT_DATASETS, DEFAULT_SUBJECTS) == (5, 20)
    assert (DEFAULT_DRAWS, DEFAULT_TIMESTEPS) == (256, 50)
    args = parser().parse_args([])
    assert args.inference_method == "dpm2_gaussian"
    assert args.gaussian_precision_batch_size == 128
    primary = inference_plan(args.inference_method)
    full_gaussian = inference_plan("dpm2_full_gaussian")
    moment = inference_plan("dpm2_gauss_global_local_moment")
    reference = inference_plan("langevin_fnpse")
    assert (primary["sampler"], primary["correction"], primary["order"]) == (
        "dpm", "gauss", 2,
    )
    assert (
        full_gaussian["sampler"], full_gaussian["correction"],
        full_gaussian["order"],
    ) == ("dpm", "full_gaussian", 2)
    assert (
        moment["sampler"], moment["correction"], moment["order"],
        moment["moment_projection"],
    ) == ("dpm", "Gauss_global_local", 2, True)
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
    full_gaussian = method_metadata({
        "inference_method": "dpm2_full_gaussian",
        "sampler": "dpm",
        "correction": "full_gaussian",
        "order": 2,
        "joint_map": {"settings": {"estimator": "hierarchical_score_map"}},
    })
    moment = method_metadata({
        "inference_method": "dpm2_gauss_global_local_moment",
        "sampler": "dpm",
        "correction": "Gauss_global_local",
        "order": 2,
        "joint_map": {
            "settings": {"estimator": "hierarchical_score_map"},
        },
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
    assert full_gaussian["key"] == "dpm2_full_gaussian"
    assert moment["key"] == "dpm2_gauss_global_local_moment"
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
    assert inference_dataset_filename(
        "dpm2_gauss_global_local_moment", 7,
    ) == "dpm2_gauss_global_local_moment-dataset-0007.pt"
    assert inference_report_filename(
        "langevin_fnpse", "summary", "json",
    ) == "langevin_fnpse-summary.json"


def test_vpsde_configuration_reuses_data_and_isolates_model_artifacts(tmp_path):
    ve = get_config("full")
    vp = get_config(
        "full", sde_type="vpsde",
        beta_min=0.2, beta_max=12.0,
    )
    assert ve.model_signature == "aacc35ff55f5604e"
    assert ve.data_signature == vp.data_signature
    assert ve.model_signature != vp.model_signature
    assert ve.legacy_data_signature in vp.compatible_data_signatures
    assert checkpoint_path(
        vp, "sde_joint", BenchmarkPaths(tmp_path),
    ).name == (
        "sde_joint-vpsde-b0p2-12-train-32768-checkpoint.pt"
    )

    from Partial_Pooling.cli import parser as stage_parser
    stage_args = stage_parser("train_models").parse_args([
        "--preset", "full", "--sde-type", "vpsde",
        "--beta-min", "0.2", "--beta-max", "12",
    ])
    assert (stage_args.sde_type, stage_args.beta_min, stage_args.beta_max) == (
        "vpsde", 0.2, 12.0,
    )

    inference_args = parser().parse_args([
        "--sde-type", "vpsde", "--beta-min", "0.2", "--beta-max", "12",
    ])
    assert inference_args.sde_type == "vpsde"



def test_joint_map_validation_rejects_hierarchy_collapse_and_clamp_boundary():
    torch.manual_seed(7)
    draws, subjects = 64, 4
    prior_mean = torch.tensor([0.5, 0.0, -1.0, -1.0, -3.0, -1.0, 0.0])
    globals_ = prior_mean + 0.05 * torch.randn(draws, 7)
    scales = globals_[:, None, 3:6].exp()
    locals_ = globals_[:, None, :3] + 0.5 * scales * torch.randn(
        draws, subjects, 3,
    )
    posterior = {"globals": globals_, "locals": locals_}
    density = _global_kde_state(globals_, torch)
    validation = _hierarchy_validation_state(posterior, density, torch)

    good = torch.cat((
        globals_[0].expand(subjects, -1), locals_[0],
    ), dim=1)
    good_diagnostics = _joint_candidate_diagnostics(
        good, density, validation, torch,
    )
    assert good_diagnostics["coherent"]
    assert not good_diagnostics["at_denoise_boundary"]

    bad = good.clone()
    bad[:, 3] = -6.0
    bad[:, 7] = 0.5
    bad_diagnostics = _joint_candidate_diagnostics(
        bad, density, validation, torch,
    )
    assert not bad_diagnostics["coherent"]
    assert bad_diagnostics["at_denoise_boundary"]


def test_joint_map_only_recovers_from_numerical_runtime_failures():
    assert _is_numerical_map_failure(RuntimeError(
        "The stabilized full-Gaussian score became non-finite."
    ))
    assert _is_numerical_map_failure(RuntimeError(
        "The composed precision became non-positive definite."
    ))
    assert not _is_numerical_map_failure(RuntimeError("CUDA out of memory"))
