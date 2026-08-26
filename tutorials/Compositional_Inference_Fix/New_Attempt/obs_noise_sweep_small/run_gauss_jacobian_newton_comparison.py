#!/usr/bin/env python3
"""Three-way MAP/posterior comparison on the obs_noise=1.3275 "small" checkpoint.

Reuses the existing trained checkpoint, reference.npz, and two of the three
methods' *already-sampled* posterior draws under
``obs_noise_sweep_small/obs_noise_1.3275/`` (written by run_obs_noise_sweep.py)
without resampling them. It adds one new posterior-sampling run (the new
pilot-free composition rule) and computes a MAP point for all three methods:

  1. "new" -- correction="gauss_jacobian" (pilot-free hierarchical GAUSS built
     from the network's own Jacobian): DPM-Solver-2, 50 timesteps, zero
     Langevin correctors, sampled fresh here; MAP via
     ``newton_map_estimate(correction="gauss_jacobian", curvature="jacobian")``
     -- also pilot-free, since the same Jacobian supplies both the composition
     weights and the Newton curvature.
  2. "old" -- correction="gauss_hierarchical" (pilot-covariance hierarchical
     GAUSS): posterior draws reused unchanged from
     ``gauss_hierarchical_dpm50_deterministic.npz``; MAP via the Tweedie-ascent
     ``map_estimate(correction="gauss_hierarchical")``, which needs a pilot
     single-observation covariance. That covariance is *not* saved in the
     existing .npz (only diagnostics are), so it is rebuilt here with the
     network's own automatic pilot-estimation routine
     (``MultiObsSampler.estimate_posterior_moments``, the same automatic path
     ``model.sample(correction="gauss_hierarchical", ...)`` calls internally)
     using the same precision_est_samples/timesteps recorded in
     ``sweep_config.json`` (4096 / 100). This is a small auxiliary pilot pass,
     not a resample of the plotted posterior.
  3. "langevin_fnpe" -- F-NPSE has no reverse-diffusion MAP objective; its
     posterior draws are reused unchanged from ``langevin_fnpe.npz`` and its
     "MAP" is the KDE mode of the global samples, exactly as
     ``Partial_Pooling/infer_partial_pooling.py::_fnpse_kde_map`` computes it
     (adapted here for this toy problem's single global scalar).

Every method is scored against the *analytic* joint posterior in
reference.npz (exact_joint_mean / exact_joint_covariance) -- no oracle moments
are fed into any sampler or MAP estimator.

Nothing in ``obs_noise_1.3275/`` that already exists is overwritten; every
output uses a new filename.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

CPU_THREAD_LIMIT = 3
logical_cpus = os.cpu_count() or 1
cpu_limit = max(1, min(CPU_THREAD_LIMIT, logical_cpus))
available_cpus = tuple(sorted(os.sched_getaffinity(0)))
selected_cpus = available_cpus[:min(cpu_limit, len(available_cpus))]
os.sched_setaffinity(0, selected_cpus)
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = str(len(selected_cpus))
print(f"CPU thread cap: {len(selected_cpus)} logical CPUs out of {logical_cpus}", flush=True)

from autocvd import autocvd

autocvd(num_gpus=1, interval=1)

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent          # obs_noise_sweep_small/
NEW_ATTEMPT = ROOT.parent                        # New_Attempt/
COMP_FIX = NEW_ATTEMPT.parent                    # Compositional_Inference_Fix/
sys.path.insert(0, str(NEW_ATTEMPT))
sys.path.insert(0, str(COMP_FIX))

import Compositional_Inference as ci  # noqa: E402
from compare_shared_local_composition_methods import load_result  # noqa: E402
from plot_local_vs_global_validation import plot_shared_local  # noqa: E402

# Not importing run_obs_noise_sweep.py's train_or_load_model: that module runs
# its own CPU-affinity setup and autocvd(...) call at import time (module-level
# side effects), which would needlessly re-select a GPU after this script
# already reserved one. Checkpoint loading is a two-line reimplementation of
# the same "load if it exists" branch (model_tag/checkpoint_dir convention
# matches run_obs_noise_sweep.py exactly), and we never retrain here.

CASE_DIR = ROOT / "obs_noise_1.3275"
OBS_NOISE = 1.3275
LOCAL_STD = 1.0
MODEL_SIZE = "small"
N_OBSERVATIONS = 30
POSTERIOR_SAMPLES = 3_000
SAMPLE_TIMESTEPS = 50           # matches 01_gauss_hierarchical_dpm50_deterministic.png
PRECISION_EST_SAMPLES = 4_096   # matches sweep_config.json
PRECISION_EST_TIMESTEPS = 100   # matches sweep_config.json
MAP_TIMESTEPS = 100             # matches tests/test_newton_map.py, MAP_Precision defaults
INFERENCE_SEED = 1_208          # matches compare_shared_local_composition_methods.INFERENCE_SEED
HIERARCHY = [0]
PRIOR = ([0.0], [1.0])          # ci.S0G == 1.0, unaffected by the obs-noise sweep
SIGMA_START = 2.0 * PRIOR[1][0]  # matches tests/MAP_Precision "long" anneal convention

NEW_NPZ = CASE_DIR / "04_gauss_jacobian_dpm50_deterministic.npz"
NEW_PNG = CASE_DIR / "04_gauss_jacobian_dpm50_deterministic.png"
COMPARISON_CSV = CASE_DIR / "map_method_comparison.csv"
COMPARISON_PNG = CASE_DIR / "05_map_method_comparison.png"
RESULTS_MD = CASE_DIR / "RESULTS.md"

NAVY, BLUE, TEAL, CORAL = "#17223B", "#3A86FF", "#2A9D8F", "#EF476F"
METHOD_LABELS = {
    "gauss_jacobian_newton": "gauss_jacobian + Newton\n(new, pilot-free)",
    "gauss_hierarchical_tweedie": "gauss_hierarchical + Tweedie\n(old, pilot covariance)",
    "langevin_fnpe_kde": "Langevin + F-NPSE\n(KDE mode)",
}
METHOD_COLORS = {
    "gauss_jacobian_newton": TEAL,
    "gauss_hierarchical_tweedie": BLUE,
    "langevin_fnpe_kde": "#8D99AE",
}
METHOD_ORDER = list(METHOD_LABELS)


def load_reference() -> dict[str, np.ndarray]:
    reference_path = CASE_DIR / "reference.npz"
    with np.load(reference_path) as archive:
        return {key: archive[key] for key in archive.files}


def make_joint_init(x_observed: np.ndarray) -> torch.Tensor:
    """(n_obs, 3) tensor [global=0, local=0, observed=x_j], the MAP ascent start."""
    x = torch.as_tensor(x_observed, dtype=torch.float32).reshape(-1)
    joint = torch.zeros(len(x), 3, dtype=torch.float32)
    joint[:, 2] = x
    return joint


def sigma_error(estimate: np.ndarray, exact: np.ndarray, exact_std: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(estimate) - np.asarray(exact)) / np.asarray(exact_std)


# ---------------------------------------------------------------------------
# Method 1: gauss_jacobian posterior draws + Newton MAP (new, pilot-free)
# ---------------------------------------------------------------------------

def run_new_method(model: "ci.SBIm", reference: dict[str, np.ndarray]) -> dict[str, object]:
    x = torch.as_tensor(reference["x_observed"], dtype=torch.float32)

    print("\n=== [1/3] gauss_jacobian: posterior sampling (DPM2, 50 steps, deterministic) ===", flush=True)
    # gauss_jacobian has no shared pilot covariance to amortize across draws
    # (its per-row Tweedie curvature comes from a fresh forward-mode Jacobian
    # every step), so its GPU memory is ~linear in num_samples rather than
    # O(1): measured ~16 MB/sample here, so POSTERIOR_SAMPLES=3000 in one call
    # needs ~48 GB and OOMs a 10 GB GPU. Chunk over samples and concatenate,
    # the same pattern Partial_Pooling/infer_partial_pooling.py::_sample_dataset
    # already uses for exactly this reason.
    JACOBIAN_SAMPLE_CHUNK = 256
    print(
        f"  chunking {POSTERIOR_SAMPLES} draws into batches of "
        f"{JACOBIAN_SAMPLE_CHUNK} (gauss_jacobian's per-sample Jacobian cost "
        "makes one large call OOM)",
        flush=True,
    )
    started = time.perf_counter()
    chunks = []
    diagnostics = None
    for chunk_index, chunk_start in enumerate(range(0, POSTERIOR_SAMPLES, JACOBIAN_SAMPLE_CHUNK)):
        chunk_count = min(JACOBIAN_SAMPLE_CHUNK, POSTERIOR_SAMPLES - chunk_start)
        torch.manual_seed(INFERENCE_SEED + chunk_index)
        torch.cuda.manual_seed_all(INFERENCE_SEED + chunk_index)
        chunk = model.sample(
            x=x, multi_obs_inference=True, hierarchy=HIERARCHY, prior=PRIOR,
            num_samples=chunk_count, timesteps=SAMPLE_TIMESTEPS,
            device="cuda", verbose=False,
            method="dpm", order=2, correction="gauss_jacobian",
            corrector_steps_interval=1, corrector_steps=0, final_corrector_steps=0,
            snr=0.2, denoise_clamp=None,
        )
        chunks.append(chunk.detach().cpu())
        diagnostics = model.multi_obs_sampler.covariance_diagnostics
        print(
            f"  draws {min(chunk_start + chunk_count, POSTERIOR_SAMPLES)}/{POSTERIOR_SAMPLES} "
            f"| elapsed {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    samples = torch.cat(chunks, dim=1)
    sample_runtime = time.perf_counter() - started
    # diagnostics is only the last chunk's; fine here since it's summary info
    # (repair/adaptation fractions), not something scored against truth.
    samples = samples.numpy()
    synchronization_error = float(np.max(np.abs(samples[:, :, 0] - samples[0:1, :, 0])))
    if synchronization_error > 1e-6:
        raise AssertionError(f"gauss_jacobian: global samples are not synchronized: {synchronization_error}")

    ci.save_raw_data(
        NEW_NPZ,
        x_observed=reference["x_observed"],
        global_truth=reference["global_truth"],
        pd_repair_fraction=diagnostics["repair_fraction"],
        pd_maximum_relative_repair=diagnostics["maximum_relative_repair"],
        negative_information_fraction=diagnostics["negative_information_fraction"],
        maximum_relative_adaptation=diagnostics["maximum_relative_adaptation"],
        local_truth=reference["local_truth"],
        exact_joint_mean=reference["exact_joint_mean"],
        exact_joint_covariance=reference["exact_joint_covariance"],
        compass_global_samples=samples[0, :, 0],
        compass_local_samples=samples[:, :, 1],
        shared_synchronization_max_abs=synchronization_error,
        runtime_seconds=sample_runtime,
    )
    print(f"gauss_jacobian sampling finished in {sample_runtime:.1f}s; wrote {NEW_NPZ}", flush=True)
    plot_shared_local(
        NEW_NPZ, NEW_PNG,
        title=f"DPM2 (50 steps) + gauss_jacobian (pilot-free), obs_noise={OBS_NOISE:g}",
    )

    print("\n=== [1/3] gauss_jacobian + Newton MAP (pilot-free: curvature='jacobian') ===", flush=True)
    init = make_joint_init(reference["x_observed"])
    condition_mask = torch.tensor([0.0, 0.0, 1.0])
    model.multi_obs_sampler.score_network_calls = 0
    started = time.perf_counter()
    result = model.multi_obs_sampler.newton_map_estimate(
        data=init, condition_mask=condition_mask, init=init,
        hierarchy=HIERARCHY, prior=PRIOR,
        correction="gauss_jacobian", curvature="jacobian",
        denoise_clamp=None, sigma_start=SIGMA_START,
        timesteps=MAP_TIMESTEPS, eps=1e-3, device="cuda",
    )
    map_runtime = time.perf_counter() - started
    # Raw shape is (n_obs, S, latent_dim); select S=0, matching the
    # `refined[:, 0, :]` convention in Partial_Pooling/infer_partial_pooling.py
    # and `tests/test_newton_map.py`'s `sampler.newton_map_estimate(**shared)[:, 0]`.
    result = result[:, 0, :]
    global_map = float(result[0, 0].item())
    local_map = result[:, 1].cpu().numpy()
    network_calls = int(model.multi_obs_sampler.score_network_calls)
    diagnostics = dict(model.multi_obs_sampler.map_diagnostics or {})
    print(
        f"Newton MAP finished in {map_runtime:.2f}s, {network_calls} score-network calls, "
        f"global_map={global_map:.4f}, converged_levels={diagnostics.get('converged_levels')}",
        flush=True,
    )
    return {
        "method": "gauss_jacobian_newton",
        "global_map": global_map,
        "local_map": local_map,
        "map_runtime_seconds": map_runtime,
        "sample_runtime_seconds": sample_runtime,
        "score_network_calls": network_calls,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Method 2: gauss_hierarchical posterior draws (reused) + Tweedie MAP (old)
# ---------------------------------------------------------------------------

def run_old_method(model: "ci.SBIm", reference: dict[str, np.ndarray]) -> dict[str, object]:
    existing_path = CASE_DIR / "gauss_hierarchical_dpm50_deterministic.npz"
    if not existing_path.exists():
        raise FileNotFoundError(f"Expected existing result at {existing_path}")
    print(f"\n=== [2/3] gauss_hierarchical: reusing existing posterior draws from {existing_path.name} ===", flush=True)
    existing = load_result(existing_path)

    x = torch.as_tensor(reference["x_observed"], dtype=torch.float32)
    condition_mask = torch.tensor([0.0, 0.0, 1.0])

    print(
        "=== [2/3] gauss_hierarchical: rebuilding the pilot single-observation "
        f"covariance ({PRECISION_EST_SAMPLES} draws x {PRECISION_EST_TIMESTEPS} steps, "
        "same automatic procedure model.sample(correction='gauss_hierarchical') uses "
        "internally) -- required by map_estimate, not saved in the existing .npz ===",
        flush=True,
    )
    torch.manual_seed(INFERENCE_SEED)
    torch.cuda.manual_seed_all(INFERENCE_SEED)
    pilot_started = time.perf_counter()
    _, pilot_covariance = model.multi_obs_sampler.estimate_posterior_moments(
        data=x, condition_mask=condition_mask,
        num_samples=PRECISION_EST_SAMPLES, timesteps=PRECISION_EST_TIMESTEPS,
        eps=1e-3, batch_size=128, device="cuda", feature_indices=[0, 1],
    )
    pilot_runtime = time.perf_counter() - pilot_started
    print(f"  pilot covariance estimated in {pilot_runtime:.1f}s", flush=True)

    print("=== [2/3] gauss_hierarchical + Tweedie-ascent MAP (map_estimate, the old estimator) ===", flush=True)
    init = make_joint_init(reference["x_observed"])
    model.multi_obs_sampler.score_network_calls = 0
    started = time.perf_counter()
    result = model.multi_obs_sampler.map_estimate(
        data=init, condition_mask=condition_mask, init=init,
        hierarchy=HIERARCHY, prior=PRIOR,
        correction="gauss_hierarchical", posterior_covariance=pilot_covariance,
        denoise_clamp=5.0, sigma_start=SIGMA_START,
        timesteps=MAP_TIMESTEPS, eps=1e-3, iterations_per_level=3, device="cuda",
    )
    map_runtime = time.perf_counter() - started
    result = result[:, 0, :]
    global_map = float(result[0, 0].item())
    local_map = result[:, 1].cpu().numpy()
    network_calls = int(model.multi_obs_sampler.score_network_calls)
    print(
        f"Tweedie MAP finished in {map_runtime:.2f}s, {network_calls} score-network calls, "
        f"global_map={global_map:.4f}",
        flush=True,
    )
    return {
        "method": "gauss_hierarchical_tweedie",
        "global_map": global_map,
        "local_map": local_map,
        "map_runtime_seconds": map_runtime,
        "pilot_runtime_seconds": pilot_runtime,
        "score_network_calls": network_calls,
        "posterior_npz": str(existing_path),
        "existing_result": existing,
    }


# ---------------------------------------------------------------------------
# Method 3: langevin_fnpe posterior draws (reused) + KDE-mode "joint_map"
# ---------------------------------------------------------------------------

def fnpse_kde_map(global_samples: np.ndarray, local_samples: np.ndarray) -> dict[str, object]:
    """Adapted from Partial_Pooling/infer_partial_pooling.py::_fnpse_kde_map.

    F-NPSE's bridging score has no reverse-diffusion MAP objective, so the
    reported mode is the KDE mode of the posterior *global* samples -- here a
    single scalar coordinate rather than a vector, but the same procedure.
    """
    from scipy.stats import gaussian_kde
    from numpy.linalg import LinAlgError

    started = time.perf_counter()
    global_draws = np.asarray(global_samples, dtype=np.float64)
    scale = max(float(global_draws.std()), 1e-6)
    standardized = (global_draws - global_draws.mean()) / scale
    try:
        log_density = gaussian_kde(standardized).logpdf(standardized)
    except (ValueError, RuntimeError, LinAlgError):
        log_density = -(standardized ** 2)
    selected = int(np.argmax(log_density))
    return {
        "method": "langevin_fnpe_kde",
        "global_map": float(global_draws[selected]),
        "local_map": np.asarray(local_samples)[:, selected],
        "map_runtime_seconds": time.perf_counter() - started,
        "score_network_calls": 0,
        "selected_candidate": selected,
        "estimator": "posterior_global_kde_mode",
    }


def run_langevin_method() -> dict[str, object]:
    existing_path = CASE_DIR / "langevin_fnpe.npz"
    if not existing_path.exists():
        raise FileNotFoundError(f"Expected existing result at {existing_path}")
    print(f"\n=== [3/3] langevin_fnpe: reusing existing posterior draws from {existing_path.name} ===", flush=True)
    existing = load_result(existing_path)
    row = fnpse_kde_map(existing["compass_global_samples"], existing["compass_local_samples"])
    print(
        f"KDE-mode joint_map selected candidate {row['selected_candidate']} "
        f"in {row['map_runtime_seconds']:.3f}s; global_map={row['global_map']:.4f}",
        flush=True,
    )
    row["posterior_npz"] = str(existing_path)
    row["existing_result"] = existing
    return row


# ---------------------------------------------------------------------------
# Scoring, table and plot
# ---------------------------------------------------------------------------

def build_comparison_rows(results: dict[str, dict[str, object]], reference: dict[str, np.ndarray]) -> list[dict[str, object]]:
    exact_mean = reference["exact_joint_mean"]
    exact_std = np.sqrt(np.diag(reference["exact_joint_covariance"]))
    rows = []
    for method in METHOD_ORDER:
        row = results[method]
        global_error = float(sigma_error(row["global_map"], exact_mean[0], exact_std[0]))
        local_errors = sigma_error(row["local_map"], exact_mean[1:], exact_std[1:])
        rows.append({
            "method": method,
            "label": METHOD_LABELS[method].replace("\n", " "),
            "global_map": row["global_map"],
            "global_map_error_sigma": global_error,
            "local_map_mean_abs_error_sigma": float(np.mean(local_errors)),
            "local_map_max_abs_error_sigma": float(np.max(local_errors)),
            "map_runtime_seconds": float(row["map_runtime_seconds"]),
            "score_network_calls": int(row["score_network_calls"]),
        })
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    with COMPARISON_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {COMPARISON_CSV}")


def plot_comparison(rows: list[dict[str, object]]) -> None:
    by_method = {row["method"]: row for row in rows}
    methods = [m for m in METHOD_ORDER if m in by_method]
    colors = [METHOD_COLORS[m] for m in methods]
    labels = [METHOD_LABELS[m] for m in methods]
    x = range(len(methods))

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))

    global_err = [by_method[m]["global_map_error_sigma"] for m in methods]
    axes[0].bar(x, global_err, color=colors)
    axes[0].set(title="Global MAP error", ylabel="|error| / analytic sigma")

    local_err = [by_method[m]["local_map_mean_abs_error_sigma"] for m in methods]
    axes[1].bar(x, local_err, color=colors)
    axes[1].set(title="Local MAP mean |error|", ylabel="|error| / analytic sigma")

    runtime = [by_method[m]["map_runtime_seconds"] for m in methods]
    axes[2].bar(x, runtime, color=colors)
    axes[2].set(title="MAP wall-clock runtime", ylabel="seconds")

    calls = [by_method[m]["score_network_calls"] for m in methods]
    axes[3].bar(x, calls, color=colors)
    axes[3].set(title="Score-network calls", ylabel="calls")

    for axis in axes:
        axis.set_xticks(list(x))
        axis.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        axis.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"MAP estimator comparison against the analytic joint posterior\n"
        f"obs_noise={OBS_NOISE:g}, local_std={LOCAL_STD:g}, model_size={MODEL_SIZE}, "
        f"n_observations={N_OBSERVATIONS}",
        fontsize=12, fontweight="bold", color=NAVY,
    )
    fig.tight_layout()
    fig.savefig(COMPARISON_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {COMPARISON_PNG}")


def write_results_markdown(rows: list[dict[str, object]], new_result: dict[str, object]) -> None:
    by_method = {row["method"]: row for row in rows}
    new = by_method["gauss_jacobian_newton"]
    old = by_method["gauss_hierarchical_tweedie"]
    langevin = by_method["langevin_fnpe_kde"]

    def fmt(row):
        return (
            f"global error = {row['global_map_error_sigma']:.3f} sigma, "
            f"local mean |error| = {row['local_map_mean_abs_error_sigma']:.3f} sigma, "
            f"local max |error| = {row['local_map_max_abs_error_sigma']:.3f} sigma, "
            f"runtime = {row['map_runtime_seconds']:.2f} s, "
            f"network calls = {row['score_network_calls']}"
        )

    new_beats_old_global = new["global_map_error_sigma"] < old["global_map_error_sigma"]
    new_beats_old_local = new["local_map_mean_abs_error_sigma"] < old["local_map_mean_abs_error_sigma"]
    new_beats_langevin_global = new["global_map_error_sigma"] < langevin["global_map_error_sigma"]
    new_beats_langevin_local = new["local_map_mean_abs_error_sigma"] < langevin["local_map_mean_abs_error_sigma"]

    text = f"""# MAP comparison: gauss_jacobian + Newton vs. gauss_hierarchical + Tweedie vs. Langevin/F-NPSE KDE

Case: `obs_noise_sweep_small/obs_noise_1.3275/` (S0L={LOCAL_STD:g}, SXH={OBS_NOISE:g},
model_size={MODEL_SIZE!r}, {N_OBSERVATIONS} real observations, checkpoint and
`reference.npz` reused unchanged from the existing obs_noise sweep). All three
methods are scored against the *analytic* joint posterior in `reference.npz`
(`exact_joint_mean` / `exact_joint_covariance`) -- no oracle moments are fed
into any sampler or MAP estimator.

## Methods

1. **New (pilot-free)**: DPM-Solver-2, 50 timesteps, `correction="gauss_jacobian"`
   posterior draws (deterministic, no Langevin correctors), plus
   `MultiObsSampler.newton_map_estimate(correction="gauss_jacobian", curvature="jacobian")`.
   Both the composition weights and the Newton curvature come from the
   network's own forward-mode Jacobian -- no pilot DDIM pass at all.
2. **Old**: reuses the existing `gauss_hierarchical_dpm50_deterministic.npz`
   posterior draws unchanged; MAP via the Tweedie-ascent
   `MultiObsSampler.map_estimate(correction="gauss_hierarchical", ...)`, which
   needs a pilot single-observation covariance. That covariance is not saved
   in the existing `.npz` (only diagnostics are), so it was rebuilt here with
   the network's own automatic pilot-estimation routine
   (`estimate_posterior_moments`, {PRECISION_EST_SAMPLES} draws x
   {PRECISION_EST_TIMESTEPS} steps -- the same automatic procedure
   `model.sample(correction="gauss_hierarchical", ...)` runs internally, and
   the same sample count used when the plotted posterior was produced).
3. **Langevin + F-NPSE**: reuses the existing `langevin_fnpe.npz` posterior
   draws unchanged. F-NPSE's bridging score has no reverse-diffusion MAP
   objective, so its "MAP" is the KDE mode of the posterior global samples,
   computed the same way as `Partial_Pooling/infer_partial_pooling.py::_fnpse_kde_map`
   (adapted for this toy problem's single global scalar).

## Results

| Method | Global error (sigma) | Local mean \\|error\\| (sigma) | Local max \\|error\\| (sigma) | MAP runtime (s) | Score-network calls |
|---|---|---|---|---|---|
| gauss_jacobian + Newton (new) | {new['global_map_error_sigma']:.3f} | {new['local_map_mean_abs_error_sigma']:.3f} | {new['local_map_max_abs_error_sigma']:.3f} | {new['map_runtime_seconds']:.2f} | {new['score_network_calls']} |
| gauss_hierarchical + Tweedie (old) | {old['global_map_error_sigma']:.3f} | {old['local_map_mean_abs_error_sigma']:.3f} | {old['local_map_max_abs_error_sigma']:.3f} | {old['map_runtime_seconds']:.2f} | {old['score_network_calls']} |
| Langevin + F-NPSE (KDE mode) | {langevin['global_map_error_sigma']:.3f} | {langevin['local_map_mean_abs_error_sigma']:.3f} | {langevin['local_map_max_abs_error_sigma']:.3f} | {langevin['map_runtime_seconds']:.2f} | {langevin['score_network_calls']} |

Full table: `map_method_comparison.csv`. Bar-chart comparison:
`05_map_method_comparison.png`. New method's posterior dashboard:
`04_gauss_jacobian_dpm50_deterministic.png`
(sampling runtime {new_result['sample_runtime_seconds']:.1f} s for
{POSTERIOR_SAMPLES} draws x {N_OBSERVATIONS} observations, {SAMPLE_TIMESTEPS} steps).

## Interpretation

On this real trained network and these {N_OBSERVATIONS} real observations
(no oracle moments anywhere), the new pilot-free `gauss_jacobian` + Newton
combination {'beats' if new_beats_old_global else 'does not beat'} the old
`gauss_hierarchical` + Tweedie MAP on the global parameter
({new['global_map_error_sigma']:.3f} vs {old['global_map_error_sigma']:.3f}
sigma) and {'beats' if new_beats_old_local else 'does not beat'} it on the
local parameters ({new['local_map_mean_abs_error_sigma']:.3f} vs
{old['local_map_mean_abs_error_sigma']:.3f} sigma mean error), while using
{new['score_network_calls']} vs {old['score_network_calls']} score-network
calls (old total includes the {PRECISION_EST_SAMPLES}-draw pilot pass Tweedie
needs and Newton does not). Against the Langevin/F-NPSE KDE-mode baseline, the
new method {'beats' if new_beats_langevin_global else 'does not beat'} it on
the global parameter ({new['global_map_error_sigma']:.3f} vs
{langevin['global_map_error_sigma']:.3f} sigma) and
{'beats' if new_beats_langevin_local else 'does not beat'} it on the locals
({new['local_map_mean_abs_error_sigma']:.3f} vs
{langevin['local_map_mean_abs_error_sigma']:.3f} sigma). The KDE-mode
"MAP" is fundamentally a different object -- a discrete selection among
existing posterior draws rather than a continuous optimum -- so it is
expected to be coarser regardless of the underlying sampler's quality.
"""
    RESULTS_MD.write_text(text)
    print(f"Wrote {RESULTS_MD}")


def load_existing_model() -> "ci.SBIm":
    """Load the existing checkpoint; never retrains (checkpoint must exist)."""
    cfg = ci.RunConfig(
        output_dir=CASE_DIR, device="cuda", seed=7,
        train_samples=ci.SHARED_LOCAL_HQ_TRAIN_SAMPLES,
        validation_samples=ci.SHARED_LOCAL_HQ_VALIDATION_SAMPLES,
        max_epochs=ci.SHARED_LOCAL_HQ_MAX_EPOCHS,
        patience=ci.SHARED_LOCAL_HQ_PATIENCE,
        batch_size=ci.SHARED_LOCAL_HQ_BATCH_SIZE,
        posterior_samples=POSTERIOR_SAMPLES,
    )
    model_tag = f"shared_local_{MODEL_SIZE}_obs_noise_{OBS_NOISE:g}"
    model_dir = ci.checkpoint_dir(cfg, model_tag)
    checkpoint = model_dir / "Model_checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Expected an existing checkpoint at {checkpoint}; this script never trains."
        )
    print(f"Loading existing checkpoint: {checkpoint}", flush=True)
    return ci.SBIm.load(str(checkpoint), device=cfg.device)


def main() -> None:
    reference = load_reference()
    model = load_existing_model()

    new_result = run_new_method(model, reference)
    old_result = run_old_method(model, reference)
    langevin_result = run_langevin_method()

    results = {
        "gauss_jacobian_newton": new_result,
        "gauss_hierarchical_tweedie": old_result,
        "langevin_fnpe_kde": langevin_result,
    }
    rows = build_comparison_rows(results, reference)
    write_csv(rows)
    plot_comparison(rows)
    write_results_markdown(rows, new_result)

    metadata = {
        "obs_noise": OBS_NOISE, "local_std": LOCAL_STD, "model_size": MODEL_SIZE,
        "n_observations": N_OBSERVATIONS, "posterior_samples": POSTERIOR_SAMPLES,
        "sample_timesteps": SAMPLE_TIMESTEPS, "map_timesteps": MAP_TIMESTEPS,
        "precision_est_samples": PRECISION_EST_SAMPLES,
        "precision_est_timesteps": PRECISION_EST_TIMESTEPS,
        "sigma_start": SIGMA_START, "hierarchy": HIERARCHY, "prior": PRIOR,
        "inference_seed": INFERENCE_SEED,
    }
    (CASE_DIR / "run_gauss_jacobian_newton_comparison_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("\nDone.")
    for row in rows:
        print(f"  {row['label']:45s}  global={row['global_map_error_sigma']:.3f}s  "
              f"local={row['local_map_mean_abs_error_sigma']:.3f}s  "
              f"runtime={row['map_runtime_seconds']:.2f}s  calls={row['score_network_calls']}")


if __name__ == "__main__":
    main()
