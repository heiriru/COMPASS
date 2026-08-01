"""Focused tests for damping-factor plot statistics."""

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tutorials" / "damping_factor_comparison.py"
SPEC = importlib.util.spec_from_file_location("damping_factor_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_calibration_curve_is_exact_for_analytic_quantiles():
    nominal = np.linspace(0.0, 0.9, 10)
    probabilities = (np.arange(200_000) + 0.5) / 200_000
    # Invert the standard-normal CDF with a deterministic bisection.  Keeping
    # this dependency-free makes the tutorial test portable without SciPy.
    lo = np.full_like(probabilities, -8.0)
    hi = np.full_like(probabilities, 8.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        cdf = MODULE.normal_cdf(mid, 0.0, 1.0)
        lo = np.where(cdf < probabilities, mid, lo)
        hi = np.where(cdf >= probabilities, mid, hi)
    samples = 0.5 * (lo + hi)
    curve = MODULE.calibration_curve(samples, 0.0, 1.0, nominal)
    assert np.allclose(curve, nominal, atol=2e-4)


def test_calibration_detects_under_and_over_dispersion():
    rng = np.random.default_rng(9)
    exact = rng.normal(size=200_000)
    nominal = np.asarray([0.5, 0.8, 0.9])
    under = MODULE.calibration_curve(0.5 * exact, 0.0, 1.0, nominal)
    over = MODULE.calibration_curve(2.0 * exact, 0.0, 1.0, nominal)
    assert np.all(under < nominal)
    assert np.all(over > nominal)


def test_cli_defaults_exclude_n200_and_support_hybrid_variants_only():
    defaults = MODULE.build_parser().parse_args([])
    assert defaults.n_values == (1, 2, 5, 10, 25, 50, 100)
    selected = MODULE.build_parser().parse_args([
        "--only-method", "hybrid_damping", "--force-new",
    ])
    assert selected.only_method == "hybrid_damping"
    assert selected.force_new
    undamped = MODULE.build_parser().parse_args([
        "--only-method", "hybrid",
    ])
    assert undamped.only_method == "hybrid"
    assert any(
        variant["key"] == "dpm2_hybrid"
        and variant["correction"] == "hybrid"
        for variant in MODULE.SOLVER_VARIANTS
    )
    assert any(
        filename == "02g_dpm2_hybrid.png" and correction == "hybrid"
        for _, _, filename, correction in MODULE.LOCAL_VARIANTS
    )


def test_50_step_local_profile_is_tagged_and_can_include_gaussian_baseline(tmp_path):
    args = MODULE.build_parser().parse_args([
        "--output-dir", str(tmp_path / "integration_steps_50"),
        "--timesteps", "50", "--stages", "local",
        "--include-gaussian-baseline",
    ])
    variants = MODULE.selected_local_variants(args)
    assert [item[3] for item in variants] == [
        "gauss", "damping", "gauss_damping", "hybrid_damping", "hybrid",
    ]
    assert MODULE.local_figure_path(
        args, "02a_dpm2_gaussian.png",
    ).name == "02a_dpm2_gaussian_integration_steps_50.png"


def test_old_joint_map_cache_migrates_to_score_map_without_resampling(tmp_path):
    path = tmp_path / "cell.npz"
    shared = {
        "checkpoint": {"path": "checkpoint", "size": 1, "mtime_ns": 2},
        "observation_pool": {"path": "pool", "size": 3, "mtime_ns": 4},
        "variant": {"key": "dpm2_damping_c0"},
        "repeat": 0, "n_observations": 2, "posterior_samples": 3,
        "timesteps": 100, "damping_at_data": 1.0,
        "damping_at_noise": 2 ** -0.5, "composition_batch_size": 2,
        "adaptive_abs_tol": 0.002576, "adaptive_rel_tol": 0.1,
        "adaptive_safety": 0.9, "adaptive_exponent": 0.9,
        "adaptive_max_evals": 10_000, "seed": 7,
    }
    old = {"schema": 1, **shared, "map_timesteps": 100, "map_iterations": 3}
    new = {"schema": 2, **shared, "map_method": "score"}
    samples = np.asarray([1.0, 2.0, 4.0])
    MODULE.save_cell(
        path, old, samples=samples, map_value=-99.0, runtime_seconds=12.0,
        accepted_steps=0, rejected_steps=0, score_evaluations=0,
    )
    migrated = MODULE.load_learned_cell(path, new, force_new=False)
    assert float(migrated["map_value"]) == samples.mean()
    assert np.array_equal(migrated["samples"], samples)
    assert float(migrated["runtime_seconds"]) == 12.0


def test_stale_hybrid_formula_cache_is_detected_before_runtime_selection(tmp_path):
    path = tmp_path / "learned_solver" / "dpm2_hybrid_damping" / "cell.npz"
    old_config = {
        "variant": {
            "key": "dpm2_hybrid_damping",
            "correction": "hybrid_damping",
        }
    }
    MODULE.save_cell(path, old_config, samples=np.asarray([0.0]))
    assert MODULE.selected_hybrid_cache_needs_refresh([path])

    current_config = {
        "variant": {
            "key": "dpm2_hybrid_damping",
            "correction": "hybrid_damping",
            "normalizer": "adaptive_a_t_v2",
        }
    }
    MODULE.save_cell(path, current_config, samples=np.asarray([0.0]))
    assert not MODULE.selected_hybrid_cache_needs_refresh([path])


def test_cache_mismatch_has_a_distinct_recoverable_exception(tmp_path):
    path = tmp_path / "cell.npz"
    MODULE.save_cell(path, {"schema": 1}, samples=np.asarray([0.0]))
    try:
        MODULE.load_cell(path, {"schema": 2}, force_new=False)
    except MODULE.CacheSettingsMismatch:
        pass
    else:
        raise AssertionError("cache mismatch was not reported as recoverable")
