#!/usr/bin/env python3
"""Incremental error-damping benchmarks for compositional score inference.

The default run reuses the learned observation pools and checkpoints produced by
``Compositional_Score/Compositional_Inference.py``.  Every new posterior is cached
independently, so an interrupted run resumes at the first missing method/N/repeat
cell.  Historical source artifacts are copied into ``output/Dumping_factor`` but
are never modified or recomputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable


CPU_THREAD_LIMIT = 3
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_cpu_limit() -> tuple[int, tuple[int, ...]]:
    """Enforce the required process-wide 3-thread CPU cap before numeric imports."""
    logical = os.cpu_count() or 1
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    host_limit = min(CPU_THREAD_LIMIT, logical)
    if host_limit < 1:
        raise RuntimeError(
            f"A {CPU_THREAD_LIMIT}-thread CPU cap permits fewer than one CPU on this {logical}-CPU host."
        )
    count = min(len(allowed), host_limit)
    if count < 1:
        raise RuntimeError("No CPU is available inside the current affinity mask.")
    selected = allowed[:count]
    os.sched_setaffinity(0, selected)
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(count)
    return logical, selected


CPU_LIMIT = configure_cpu_limit()
print(
    f"CPU limited to {len(CPU_LIMIT[1])}/{CPU_LIMIT[0]} logical CPUs "
    f"({len(CPU_LIMIT[1]) / CPU_LIMIT[0]:.2%})."
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
BASE_LEARNED = ROOT / "output" / "compositional_inference"
BASE_ANALYTIC = ROOT / "output" / "compositional_score_analytic"
BASE_LOCAL = ROOT / "output" / "compositional_inference_local_vs_global" / "method_comparison"
BASE_MAP = ROOT / "output" / "annealed_score_ascent" / "method_comparison"
DEFAULT_OUTPUT = ROOT / "output" / "Dumping_factor"
DEFAULT_SHARED_CHECKPOINT = BASE_LEARNED / "models" / "shared_mixture_full" / "Model_checkpoint.pt"
DEFAULT_LOCAL_CHECKPOINT = BASE_LEARNED / "models" / "shared_local_mixture_full" / "Model_checkpoint.pt"
DEFAULT_SCALING_RAW = BASE_LEARNED / "03_observation_scaling" / "raw_plot_data.npz"
DEFAULT_LOCAL_REFERENCE = BASE_LEARNED / "06b_shared_local" / "raw_plot_data.npz"
DEFAULT_N_VALUES = (1, 2, 5, 10, 25, 50, 100)
NOMINAL_COVERAGE = np.linspace(0.0, 1.0, 11)

torch = None
SBIm = None
MultiObsSampler = None
Sampler = None
VESDE = None
ModelTransfuser = None


SOLVER_VARIANTS = (
    {
        "key": "dpm2_damping_c0", "label": "DPM-2 + damping (0 correctors)",
        "correction": "damping", "method": "dpm",
        "order": 2, "corrector_steps": 0, "corrector_steps_interval": 5,
        "final_corrector_steps": 0, "snr": 0.1,
    },
    {
        "key": "dpm2_damping_c5", "label": "DPM-2 + damping (5 correctors)",
        "correction": "damping", "method": "dpm",
        "order": 2, "corrector_steps": 5, "corrector_steps_interval": 5,
        "final_corrector_steps": 3, "snr": 0.1,
    },
    {
        "key": "euler_damping", "label": "Euler–Maruyama + damping",
        "correction": "damping", "method": "euler", "order": 2,
        "corrector_steps": 0, "corrector_steps_interval": 5,
        "final_corrector_steps": 0, "snr": 0.1,
    },
    {
        "key": "langevin_damping", "label": "Annealed Langevin + damping",
        "correction": "damping", "method": "langevin", "order": 2,
        "corrector_steps": 5, "corrector_steps_interval": 5,
        "final_corrector_steps": 0, "snr": 0.1,
    },
    {
        "key": "adaptive_damping", "label": "Adaptive reverse-SDE + damping",
        "correction": "damping", "method": "adaptive", "order": 2,
        "corrector_steps": 0, "corrector_steps_interval": 5,
        "final_corrector_steps": 0, "snr": 0.1,
    },
    {
        "key": "dpm2_gauss_damping", "label": "DPM-2 + Gaussian + damping",
        "correction": "gauss_damping", "method": "dpm", "order": 2,
        "corrector_steps": 5, "corrector_steps_interval": 5,
        "final_corrector_steps": 3, "snr": 0.1,
    },
    {
        "key": "dpm2_hybrid", "label": "DPM-2 + hybrid (no damping)",
        "correction": "hybrid", "method": "dpm", "order": 2,
        "corrector_steps": 5, "corrector_steps_interval": 5,
        "final_corrector_steps": 3, "snr": 0.1,
        "normalizer": "adaptive_a_t_v2",
    },
    {
        "key": "dpm2_hybrid_damping", "label": "DPM-2 + hybrid damping",
        "correction": "hybrid_damping", "method": "dpm", "order": 2,
        "corrector_steps": 5, "corrector_steps_interval": 5,
        "final_corrector_steps": 3, "snr": 0.1,
        "normalizer": "adaptive_a_t_v2",
    },
)

SWEEP_VARIANTS = (
    ("d1_1", "d₁ = 1", lambda n: 1.0),
    ("d1_0p3", "d₁ = 0.3", lambda n: 0.3),
    ("d1_0p1", "d₁ = 0.1", lambda n: 0.1),
    ("d1_inverse_sqrt_n", "d₁ = 1/√N", lambda n: n ** -0.5),
    ("d1_inverse_n", "d₁ = 1/N", lambda n: 1.0 / n),
)

NEW_SCALING_KEYS = (
    "dpm2_damping_c5", "dpm2_gauss_damping", "dpm2_hybrid",
    "dpm2_hybrid_damping",
)

LOCAL_VARIANTS = (
    ("dpm2_error_damping", "DPM2 + error damping", "02d_dpm2_error_damping.png", "damping"),
    ("dpm2_gaussian_damping", "DPM2 + Gaussian + damping", "02e_dpm2_gaussian_damping.png", "gauss_damping"),
    ("dpm2_hybrid_damping", "DPM2 + hybrid damping", "02f_dpm2_hybrid_damping.png", "hybrid_damping"),
    ("dpm2_hybrid", "DPM2 + hybrid (no damping)", "02g_dpm2_hybrid.png", "hybrid"),
)

LOCAL_GAUSSIAN_BASELINE = (
    "dpm2_gaussian", "DPM2 + Gaussian composition",
    "02a_dpm2_gaussian.png", "gauss",
)


COMPARISON_LINE_STYLES = (
    {"linestyle": "-", "marker": "o"},
    {"linestyle": ":", "marker": "s"},
    {"linestyle": "--", "marker": "^"},
    {"linestyle": "-.", "marker": "D"},
    {"linestyle": (0, (1, 1)), "marker": "v"},
    {"linestyle": (0, (5, 2)), "marker": "P"},
    {"linestyle": (0, (3, 1, 1, 1)), "marker": "X"},
)


def configure_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 125, "savefig.dpi": 240, "font.size": 9.5,
        "axes.titlesize": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.2, "legend.fontsize": 7.4,
    })


def parse_n_values(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("observation counts must be positive integers")
    return values


def jsonable_config(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def save_cell(path: Path, config: dict[str, Any], **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, config_json=np.asarray(jsonable_config(config)),
        **{key: np.asarray(value) for key, value in arrays.items()},
    )
    os.replace(temporary, path)


class CacheSettingsMismatch(RuntimeError):
    """A valid resumable cache belongs to a different experiment config."""


def load_cell(path: Path, config: dict[str, Any], force_new: bool) -> dict[str, np.ndarray] | None:
    if not path.exists() or force_new:
        return None
    result = load_npz(path)
    cached = str(result.get("config_json", np.asarray("")).item())
    expected = jsonable_config(config)
    if cached != expected:
        raise CacheSettingsMismatch(
            f"Cache settings differ for {path}. Use --force-new to replace only "
            "this new damping result. Historical source results remain untouched."
        )
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finish_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def normal_cdf(value: np.ndarray | float, mean: float, std: float) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    standardized = (values - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(standardized))


def calibration_curve(
    samples: np.ndarray, analytic_mean: float, analytic_std: float,
    nominal: np.ndarray = NOMINAL_COVERAGE,
) -> np.ndarray:
    """Exact posterior mass inside empirical central quantile intervals."""
    values = np.asarray(samples, dtype=float).reshape(-1)
    result = []
    for coverage in np.asarray(nominal, dtype=float):
        tail = (1.0 - coverage) / 2.0
        lower, upper = np.quantile(values, (tail, 1.0 - tail))
        result.append(
            float(normal_cdf(upper, analytic_mean, analytic_std)
                  - normal_cdf(lower, analytic_mean, analytic_std))
        )
    return np.asarray(result)


def copy_historical_outputs(output: Path) -> None:
    """Synchronize requested source artifacts without modifying their sources."""
    copies = (
        (BASE_ANALYTIC / "samplers_vs_observation_count_prior0p3_likelihood0p5.png",
         output / "samplers_vs_observation_count_prior0p3_likelihood0p5.png"),
        (BASE_ANALYTIC / "samplers_vs_observation_count_prior1p0_likelihood1p0.png",
         output / "samplers_vs_observation_count_prior1p0_likelihood1p0.png"),
        (BASE_LEARNED / "03_observation_scaling" / "samplers_vs_observation_count.png",
         output / "samplers_vs_observation_count.png"),
        (BASE_MAP / "02a_dpm2_gaussian.png", output / "02a_dpm2_gaussian.png"),
        (BASE_MAP / "02b_pfode_gaussian.png", output / "02b_pfode_gaussian.png"),
        (BASE_MAP / "02c_langevin_fnpe.png", output / "02c_langevin_fnpe.png"),
        (DEFAULT_SCALING_RAW, output / "cache" / "source_results" / "learned_scaling_raw.npz"),
        (BASE_LEARNED / "03_observation_scaling" / "observation_scaling_metrics.csv",
         output / "cache" / "source_results" / "learned_scaling_metrics.csv"),
    )
    for source, destination in copies:
        if not source.exists():
            raise FileNotFoundError(f"Required historical result is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"Synchronized {source} -> {destination}")


def ensure_runtime(device: str, needs_compute: bool) -> str:
    """Reserve a GPU only for missing work, then import Torch/COMPASS."""
    global torch, SBIm, MultiObsSampler, Sampler, VESDE, ModelTransfuser
    if needs_compute and device == "cuda":
        try:
            from autocvd import autocvd
        except ImportError as exc:
            raise RuntimeError("autocvd is required for the GPU tutorial run") from exc
        autocvd(num_gpus=1, interval=1)
    import torch as torch_module
    from compass import ScoreBasedInferenceModel as sbim_class
    from compass.MultiObsSampler import MultiObsSampler as multi_class
    from compass.Sampler import Sampler as sampler_class
    from compass.SDE import VESDE as vesde_class
    from compass.ModelTransfuser import ModelTransfuser as transfuser_class
    torch, SBIm = torch_module, sbim_class
    MultiObsSampler, Sampler, VESDE = multi_class, sampler_class, vesde_class
    ModelTransfuser = transfuser_class
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable after autocvd reservation")
    return device


def load_score_checkpoint(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = SBIm(
        nodes_size=checkpoint["nodes_size"], sde_type=checkpoint["sde_type"],
        sigma=checkpoint["sigma"], beta_min=checkpoint["beta_min"],
        beta_max=checkpoint["beta_max"], hidden_size=checkpoint["hidden_size"],
        depth=checkpoint["depth"], num_heads=checkpoint["num_heads"],
        mlp_ratio=checkpoint["mlp_ratio"], device=device,
    )
    state = {
        key: value for key, value in checkpoint["model_state_dict"].items()
        if not key.startswith("divergence_head.")
    }
    model.model.load_state_dict(state, strict=True)
    model.model.eval()
    return model


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def learned_truth(observations: np.ndarray) -> tuple[float, float]:
    values = np.asarray(observations, dtype=float).reshape(-1)
    precision = 1.0 + len(values)
    return float(values.sum() / precision), float(precision ** -0.5)


def learned_cell_path(output: Path, family: str, key: str, repeat: int, n: int) -> Path:
    return output / "cache" / family / key / f"repeat_{repeat}_n_{n}.npz"


def learned_config(
    args: argparse.Namespace, variant: dict[str, Any], repeat: int, n: int,
    damping_at_noise: float,
) -> dict[str, Any]:
    return {
        "schema": 2, "checkpoint": source_signature(args.shared_checkpoint),
        "observation_pool": source_signature(args.scaling_raw),
        "variant": variant, "repeat": repeat, "n_observations": n,
        "posterior_samples": args.posterior_samples, "timesteps": args.timesteps,
        "damping_at_data": 1.0, "damping_at_noise": damping_at_noise,
        "composition_batch_size": n, "map_method": "score",
        "adaptive_abs_tol": args.adaptive_abs_tol,
        "adaptive_rel_tol": args.adaptive_rel_tol,
        "adaptive_safety": args.adaptive_safety,
        "adaptive_exponent": args.adaptive_exponent,
        "adaptive_max_evals": args.adaptive_max_evals,
        "seed": args.seed + 10_000 * repeat + 100 * n + sum(map(ord, variant["key"])),
    }


def load_learned_cell(
    path: Path, config: dict[str, Any], force_new: bool,
) -> dict[str, np.ndarray] | None:
    """Load a learned cell and migrate old joint-score MAP metadata cheaply.

    Schema-1 cells contain valid posterior samples and solver diagnostics, but
    their MAP was obtained with hierarchical joint score ascent.  The requested
    ``map_method='score'`` holds shared coordinates at the posterior sample mean,
    so migration requires no score-network evaluations and never resamples.
    """
    if not path.exists() or force_new:
        return None
    result = load_npz(path)
    cached_json = str(result.get("config_json", np.asarray("")).item())
    try:
        cached_config = json.loads(cached_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid cache metadata in {path}") from exc
    if cached_config == config:
        return result

    def sampling_only(value: dict[str, Any]) -> dict[str, Any]:
        ignored = {"schema", "map_method", "map_timesteps", "map_iterations"}
        return {key: item for key, item in value.items() if key not in ignored}

    if sampling_only(cached_config) != sampling_only(config):
        raise CacheSettingsMismatch(
            f"Cache sampling settings differ for {path}. Use --force-new with "
            "the matching --only-method selector to replace that damping method."
        )
    samples = np.asarray(result["samples"], dtype=float)
    save_cell(
        path, config, samples=samples, map_value=float(samples.mean()),
        runtime_seconds=float(result["runtime_seconds"]),
        accepted_steps=int(result.get("accepted_steps", np.asarray(0))),
        rejected_steps=int(result.get("rejected_steps", np.asarray(0))),
        score_evaluations=int(result.get("score_evaluations", np.asarray(0))),
    )
    print(f"Migrated cached MAP to map_method='score' without resampling: {path}")
    return load_npz(path)


def run_learned_cell(
    model, observations: np.ndarray, variant: dict[str, Any], repeat: int, n: int,
    damping_at_noise: float, path: Path, args: argparse.Namespace, device: str,
    allow_compute: bool = True,
) -> dict[str, np.ndarray] | None:
    config = learned_config(args, variant, repeat, n, damping_at_noise)
    try:
        cached = load_learned_cell(
            path, config, force_new=(args.force_new and allow_compute),
        )
    except CacheSettingsMismatch:
        if not allow_compute:
            return None
        if args.plot_only:
            raise
        print(f"Recalculating selected cache with updated settings: {path}")
        cached = None
    if cached is not None:
        return cached
    if not allow_compute:
        return None
    if args.plot_only:
        raise FileNotFoundError(f"Missing new damping cache in --plot-only mode: {path}")

    x = torch.as_tensor(observations, dtype=torch.float32, device=device)
    precision = None
    if variant["correction"] in ("gauss_damping", "hybrid", "hybrid_damping"):
        precision = torch.full((n, 1), 2.0)
    seed_all(config["seed"])
    synchronize(device)
    started = time.perf_counter()
    samples = model.sample(
        x=x, multi_obs_inference=True, hierarchy=[0], prior=([0.0], [1.0]),
        correction=variant["correction"], posterior_precision=precision,
        damping_at_data=1.0, damping_at_noise=damping_at_noise,
        composition_batch_size=(n if variant["correction"] == "damping" else None),
        num_samples=args.posterior_samples,
        timesteps=args.timesteps, method=variant["method"], order=variant["order"],
        corrector_steps=variant["corrector_steps"],
        corrector_steps_interval=variant["corrector_steps_interval"],
        final_corrector_steps=variant["final_corrector_steps"], snr=variant["snr"],
        adaptive_abs_tol=args.adaptive_abs_tol,
        adaptive_rel_tol=args.adaptive_rel_tol,
        adaptive_safety=args.adaptive_safety,
        adaptive_exponent=args.adaptive_exponent,
        adaptive_max_evals=args.adaptive_max_evals,
        device=device, verbose=args.verbose,
    )
    synchronize(device)
    runtime = time.perf_counter() - started
    values = samples[0, :, 0].detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{variant['label']} produced non-finite learned-score samples "
            f"for repeat={repeat}, N={n}."
        )

    # Match ModelTransfuser(map_method="score") for compositional inference:
    # shared coordinates are held at their synchronized posterior sample mean;
    # score ascent is applied only to local coordinates (there are none here).
    map_value = float(values.mean())
    if not math.isfinite(map_value):
        raise RuntimeError(
            f"{variant['label']} produced a non-finite score-ascent MAP "
            f"for repeat={repeat}, N={n}."
        )
    stats = model.multi_obs_sampler.solver_stats or {}
    save_cell(
        path, config, samples=values, map_value=map_value,
        runtime_seconds=runtime,
        accepted_steps=int(stats.get("accepted_steps", 0)),
        rejected_steps=int(stats.get("rejected_steps", 0)),
        score_evaluations=int(stats.get("score_evaluations", 0)),
    )
    print(f"Finished {variant['label']}, repeat={repeat}, N={n} in {runtime:.1f}s")
    if device == "cuda":
        torch.cuda.empty_cache()
    return load_npz(path)


def learned_metrics(
    result: dict[str, np.ndarray], mean: float, std: float,
    key: str, label: str, repeat: int, n: int,
) -> dict[str, Any]:
    samples = np.asarray(result["samples"], dtype=float)
    return {
        "variant": key, "label": label, "repeat": repeat,
        "n_observations": n,
        "map_error_sigma": abs(float(result["map_value"]) - mean) / std,
        "sample_mean": float(samples.mean()), "sample_std": float(samples.std(ddof=1)),
        "std_ratio": float(samples.std(ddof=1) / std),
        "runtime_seconds": float(result["runtime_seconds"]),
        "truth_mean": mean, "truth_std": std,
        "calibration": calibration_curve(samples, mean, std),
        "accepted_steps": int(result["accepted_steps"]),
        "rejected_steps": int(result["rejected_steps"]),
        "score_evaluations": int(result["score_evaluations"]),
    }


def aggregate_by_n(rows: list[dict[str, Any]], key: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(sorted({int(row["n_observations"]) for row in rows if row["variant"] == key}))
    groups = [np.asarray([float(row[metric]) for row in rows
                         if row["variant"] == key and int(row["n_observations"]) == n])
              for n in xs]
    return xs, np.asarray([group.mean() for group in groups]), np.asarray([
        group.std(ddof=1) if len(group) > 1 else 0.0 for group in groups
    ])


def plot_three_panel(
    rows: list[dict[str, Any]], variants: Iterable[tuple[str, str]], path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.45))
    for index, (key, label) in enumerate(variants):
        line_style = COMPARISON_LINE_STYLES[index % len(COMPARISON_LINE_STYLES)]
        selected = [row for row in rows if row["variant"] == key]
        for axis, metric in zip(axes[:2], ("map_error_sigma", "std_ratio")):
            xs, values, spreads = aggregate_by_n(rows, key, metric)
            line, = axis.plot(
                xs, values, lw=1.8, ms=4.5, label=label, **line_style,
            )
            axis.fill_between(xs, values - spreads, values + spreads,
                              color=line.get_color(), alpha=0.12)
        curves = np.stack([row["calibration"] for row in selected])
        centre = curves.mean(axis=0)
        lower, upper = curves.min(axis=0), curves.max(axis=0)
        line, = axes[2].plot(
            NOMINAL_COVERAGE, centre, lw=1.8, ms=4.0, label=label,
            **line_style,
        )
        axes[2].fill_between(NOMINAL_COVERAGE, lower, upper,
                             color=line.get_color(), alpha=0.10)
    axes[0].set(xscale="log", xlabel="observations N",
                ylabel="|score MAP − analytic MAP| / analytic σ",
                title="MAP offset")
    axes[1].set(xscale="log", xlabel="observations N",
                ylabel="sample posterior σ / analytic σ", title="Posterior width")
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[2].plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="ideal")
    axes[2].set(xlabel="nominal central coverage",
                ylabel="analytic mass in empirical interval", title="Calibration",
                xlim=(0, 1), ylim=(0, 1))
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.23)
    finish_figure(fig, path)


def run_learned_benchmarks(model, pools: dict[str, np.ndarray], args, device: str) -> tuple[list[dict], list[dict]]:
    solver_rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        pool = pools[f"x_pool_r{repeat}"]
        for n in args.n_values:
            observations = pool[:n]
            mean, std = learned_truth(observations)
            for variant in SOLVER_VARIANTS:
                selected = (
                    args.only_method == "all"
                    or variant["correction"] == args.only_method
                )
                path = learned_cell_path(args.output_dir, "learned_solver", variant["key"], repeat, n)
                result = run_learned_cell(
                    model, observations, variant, repeat, n, n ** -0.5,
                    path, args, device, allow_compute=selected,
                )
                if result is None:
                    continue
                solver_rows.append(learned_metrics(
                    result, mean, std, variant["key"], variant["label"], repeat, n,
                ))
    write_rows(args.output_dir / "learned_score_damping_solver_metrics.csv", [
        {key: value for key, value in row.items() if key != "calibration"}
        for row in solver_rows
    ])
    available_solver_variants = [
        (item["key"], item["label"]) for item in SOLVER_VARIANTS
        if any(row["variant"] == item["key"] for row in solver_rows)
    ]
    if available_solver_variants:
        plot_three_panel(
            solver_rows, available_solver_variants,
            args.output_dir / "learned_score_damping_solver_comparison.png",
            "Learned-score damping: sampler and correction comparison",
        )

    adaptive = next(item for item in SOLVER_VARIANTS if item["key"] == "adaptive_damping")
    sweep_rows: list[dict[str, Any]] = []
    if args.only_method in ("all", "damping"):
        for repeat in range(args.repeats):
            pool = pools[f"x_pool_r{repeat}"]
            for n in args.n_values:
                observations = pool[:n]
                mean, std = learned_truth(observations)
                for key, label, endpoint in SWEEP_VARIANTS:
                    d1 = float(endpoint(n))
                    if key == "d1_inverse_sqrt_n":
                        path = learned_cell_path(args.output_dir, "learned_solver", "adaptive_damping", repeat, n)
                        variant = adaptive
                    else:
                        path = learned_cell_path(args.output_dir, "d1_sweep", key, repeat, n)
                        variant = {**adaptive, "key": key, "label": label}
                    result = run_learned_cell(
                        model, observations, variant, repeat, n, d1, path, args,
                        device, allow_compute=True,
                    )
                    sweep_rows.append(learned_metrics(result, mean, std, key, label, repeat, n))
        write_rows(args.output_dir / "learned_score_damping_d1_metrics.csv", [
            {key: value for key, value in row.items() if key != "calibration"}
            for row in sweep_rows
        ])
        plot_three_panel(
            sweep_rows, [(key, label) for key, label, _ in SWEEP_VARIANTS],
            args.output_dir / "learned_score_damping_d1_sweep.png",
            "Adaptive reverse-SDE: terminal damping sweep",
        )
    return solver_rows, sweep_rows


def update_learned_scaling(solver_rows: list[dict[str, Any]], args) -> None:
    source = BASE_LEARNED / "03_observation_scaling" / "observation_scaling_metrics.csv"
    rows = [
        row for row in read_rows(source)
        if int(float(row["n_observations"])) in args.n_values
    ]
    by_key = {item["key"]: item for item in SOLVER_VARIANTS}
    for row in solver_rows:
        if row["variant"] not in NEW_SCALING_KEYS:
            continue
        variant = by_key[row["variant"]]
        rows.append({
            "variant": row["variant"], "label": row["label"],
            "method": variant["method"], "correction": variant["correction"],
            "repeat": row["repeat"], "n_observations": row["n_observations"],
            "mean_error_sigma": abs(row["sample_mean"] - row["truth_mean"]) / row["truth_std"],
            "std_ratio": row["std_ratio"], "runtime_seconds": row["runtime_seconds"],
            "sample_mean": row["sample_mean"], "sample_std": row["sample_std"],
            "truth_mean": row["truth_mean"], "truth_std": row["truth_std"],
        })
    write_rows(args.output_dir / "learned_observation_scaling_metrics.csv", rows)
    variants = []
    for row in rows:
        key, label = str(row["variant"]), str(row["label"])
        if (key, label) not in variants:
            variants.append((key, label))
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0))
    axes = axes.ravel()
    for key, label in variants:
        chosen = [row for row in rows if str(row["variant"]) == key]
        if not chosen:
            continue
        for axis, metric in zip(axes[:3], ("mean_error_sigma", "std_ratio", "runtime_seconds")):
            xs = np.asarray(sorted({int(float(row["n_observations"])) for row in chosen}))
            groups = [np.asarray([float(row[metric]) for row in chosen
                                 if int(float(row["n_observations"])) == n]) for n in xs]
            values = np.asarray([group.mean() for group in groups])
            spreads = np.asarray([group.std(ddof=1) if len(group) > 1 else 0.0 for group in groups])
            line, = axis.plot(xs, values, "o-", ms=3.5, label=label)
            axis.fill_between(xs, values - spreads, values + spreads,
                              color=line.get_color(), alpha=0.09)
        xs = np.asarray(sorted({int(float(row["n_observations"])) for row in chosen}))
        runtimes = [np.mean([float(row["runtime_seconds"]) for row in chosen
                             if int(float(row["n_observations"])) == n]) for n in xs]
        errors = [np.mean([float(row["mean_error_sigma"]) for row in chosen
                           if int(float(row["n_observations"])) == n]) for n in xs]
        axes[3].plot(runtimes, errors, "o-", ms=3.5, label=label)
    axes[0].set(xscale="log", xlabel="observations N", ylabel="mean error / analytic σ")
    axes[1].set(xscale="log", xlabel="observations N", ylabel="sample σ / analytic σ")
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[2].set(xscale="log", yscale="log", xlabel="observations N", ylabel="runtime (s)")
    axes[3].set(xscale="log", yscale="log", xlabel="runtime (s)", ylabel="mean error / analytic σ")
    axes[0].legend(ncol=2)
    finish_figure(fig, args.output_dir / "samplers_vs_observation_count.png")


def analytic_variant(key: str, label: str, correction: str) -> dict[str, Any]:
    result = {
        "key": key, "label": label, "correction": correction, "method": "dpm",
        "order": 2, "corrector_steps": 5, "corrector_steps_interval": 5,
        "final_corrector_steps": 3, "snr": 0.1,
    }
    if correction in ("hybrid", "hybrid_damping"):
        result["normalizer"] = "adaptive_a_t_v2"
    return result


ANALYTIC_VARIANTS = (
    analytic_variant("dpm2_damping", "DPM-2 + damping", "damping"),
    analytic_variant("dpm2_gauss_damping", "DPM-2 + Gaussian + damping", "gauss_damping"),
    analytic_variant("dpm2_hybrid", "DPM-2 + hybrid (no damping)", "hybrid"),
    analytic_variant("dpm2_hybrid_damping", "DPM-2 + hybrid damping", "hybrid_damping"),
)


def exact_inference(device: str, mu0, sig0, sigx):
    class LinearGaussianScore(torch.nn.Module):
        def __init__(self, sde):
            super().__init__()
            self.sde = sde

        def forward(self, x, t, c, return_attn_weights=False):
            std_t = self.sde.marginal_prob_std(t).to(x.device)
            theta, observed = x[:, :2], x[:, 2:]
            precision = 1 / sig0**2 + 1 / sigx**2
            mean = (mu0 / sig0**2 + observed / sigx**2) / precision
            score = -(theta - mean) / (1 / precision + std_t**2)
            result = torch.zeros_like(x)
            result[:, :2] = std_t * score
            return (result, torch.zeros(1, device=x.device)) if return_attn_weights else result

    class ExactWrapper:
        def __init__(self):
            self.sde = VESDE(sigma=25.0)
            self.sde.sigma = self.sde.sigma.to(device)
            self.model = LinearGaussianScore(self.sde).to(device)
            self.sampler = Sampler(self)
            self.nodes_size = 4

        def output_scale_function(self, t, value):
            return value / self.sde.marginal_prob_std(t).to(value.device)

    return ExactWrapper()


def analytic_problem(n: int, repeat: int, device: str, mu0, sig0, sigx):
    seed_all(1000 + 101 * n + repeat)
    theta = mu0 + sig0 * torch.randn(2, device=device)
    observations = theta + sigx * torch.randn(n, 2, device=device)
    precision = 1 / sig0**2 + n / sigx**2
    mean = (mu0 / sig0**2 + observations.sum(0) / sigx**2) / precision
    return observations, mean, torch.sqrt(1 / precision)


def run_analytic(args, device: str) -> None:
    configurations = (
        ("prior1p0_likelihood1p0", 1.0, 1.0),
        ("prior0p3_likelihood0p5", 0.3, 0.5),
    )
    for config_name, prior_std, likelihood_std in configurations:
        mu0 = torch.tensor([-2.3, -2.89], device=device)
        sig0 = torch.full((2,), prior_std, device=device)
        sigx = torch.full((2,), likelihood_std, device=device)
        model = exact_inference(device, mu0, sig0, sigx)
        new_rows = []
        for repeat in range(args.repeats):
            for n in args.n_values:
                observations, truth_mean, truth_std = analytic_problem(
                    n, repeat, device, mu0, sig0, sigx,
                )
                for variant in ANALYTIC_VARIANTS:
                    selected = (
                        args.only_method == "all"
                        or variant["correction"] == args.only_method
                    )
                    path = learned_cell_path(args.output_dir, f"analytic_{config_name}", variant["key"], repeat, n)
                    config = {
                        "schema": 1, "problem": config_name, "variant": variant,
                        "repeat": repeat, "n_observations": n,
                        "posterior_samples": args.posterior_samples,
                        "timesteps": args.timesteps, "damping_at_data": 1.0,
                        "damping_at_noise": n ** -0.5,
                    }
                    try:
                        result = load_cell(
                            path, config,
                            force_new=(args.force_new and selected),
                        )
                    except CacheSettingsMismatch:
                        if not selected:
                            # A stale unselected cache must neither block nor be
                            # silently recomputed by a method-specific rerun.
                            continue
                        if args.plot_only:
                            raise
                        print(f"Recalculating selected cache with updated settings: {path}")
                        result = None
                    if result is None:
                        if not selected:
                            continue
                        if args.plot_only:
                            raise FileNotFoundError(path)
                        precision = None
                        if variant["correction"] != "damping":
                            precision = (1 / sig0**2 + 1 / sigx**2).repeat(n, 1).cpu()
                        seed_all(20_000 + 10_000 * repeat + 100 * n + sum(map(ord, variant["key"])))
                        synchronize(device)
                        started = time.perf_counter()
                        samples = MultiObsSampler(model).sample(
                            world_size=1, data=observations,
                            condition_mask=torch.tensor([0.0, 0.0, 1.0, 1.0], device=device),
                            hierarchy=[0, 1], prior=(mu0.cpu(), sig0.cpu()),
                            correction=variant["correction"], posterior_precision=precision,
                            damping_at_noise=n ** -0.5,
                            composition_batch_size=(n if variant["correction"] == "damping" else None),
                            num_samples=args.posterior_samples, timesteps=args.timesteps,
                            method="dpm", order=2, corrector_steps=5,
                            corrector_steps_interval=5, final_corrector_steps=3, snr=0.1,
                            device=device, verbose=args.verbose,
                        )[0, :, :2]
                        synchronize(device)
                        runtime = time.perf_counter() - started
                        sample_values = samples.cpu().numpy()
                        if not np.isfinite(sample_values).all():
                            raise RuntimeError(
                                f"{variant['label']} produced non-finite analytic-score "
                                f"samples for repeat={repeat}, N={n}."
                            )
                        save_cell(path, config, samples=sample_values, runtime_seconds=runtime)
                        result = load_npz(path)
                    values = np.asarray(result["samples"])
                    normalized = (values.mean(0) - truth_mean.cpu().numpy()) / truth_std.cpu().numpy()
                    new_rows.append({
                        "variant": variant["label"], "n_observations": n,
                        "repeat": repeat,
                        "mean_error_sigma": float(np.linalg.norm(normalized)),
                        "std_ratio": float(np.mean(values.std(axis=0, ddof=1) / truth_std.cpu().numpy())),
                        "runtime_seconds": float(result["runtime_seconds"]),
                        "prior_std": prior_std, "likelihood_std": likelihood_std,
                    })
        source_csv = BASE_ANALYTIC / f"sampler_metrics_{config_name}.csv"
        combined = [
            row for row in read_rows(source_csv)
            if int(float(row["n_observations"])) in args.n_values
        ] + new_rows
        write_rows(args.output_dir / f"analytic_sampler_metrics_{config_name}.csv", combined)
        plot_analytic_scaling(
            combined, args.output_dir / f"samplers_vs_observation_count_{config_name}.png",
            f"Prior std={prior_std:g}, likelihood std={likelihood_std:g}",
        )


def plot_analytic_scaling(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    variants = []
    for row in rows:
        label = str(row["variant"])
        if label not in variants:
            variants.append(label)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.25))
    for label in variants:
        selected = [row for row in rows if str(row["variant"]) == label]
        xs = np.asarray(sorted({int(float(row["n_observations"])) for row in selected}))
        for axis, metric in zip(axes, ("mean_error_sigma", "std_ratio")):
            groups = [np.asarray([float(row[metric]) for row in selected
                                 if int(float(row["n_observations"])) == n]) for n in xs]
            values = np.asarray([group.mean() for group in groups])
            spreads = np.asarray([group.std(ddof=1) if len(group) > 1 else 0 for group in groups])
            line, = axis.plot(xs, values, "o-", ms=3.8, label=label)
            axis.fill_between(xs, values - spreads, values + spreads,
                              color=line.get_color(), alpha=0.10)
    axes[0].set(xscale="log", xlabel="observations N",
                ylabel="Euclidean normalized mean error", title="Posterior-centre accuracy")
    axes[1].set(xscale="log", xlabel="observations N",
                ylabel="sample σ / analytic σ", title="Posterior-width accuracy")
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[0].legend(ncol=2)
    fig.suptitle(title)
    finish_figure(fig, path)


def plot_shared_local_score_map(
    raw: dict[str, np.ndarray], joint_rows: np.ndarray,
    score_rows: np.ndarray, output: Path, title: str,
) -> None:
    """Compare shared-marginal joint MAP and fixed-shared score MAP."""
    exact = np.asarray(raw["exact_joint_mean"])
    covariance = np.asarray(raw["exact_joint_covariance"])
    global_samples = np.asarray(raw["compass_global_samples"])
    local_samples = np.asarray(raw["compass_local_samples"])
    local_mean = local_samples.mean(axis=1)
    local_std = local_samples.std(axis=1, ddof=1)
    exact_local_std = np.sqrt(np.diag(covariance)[1:])
    joint_local = np.asarray(joint_rows)[:, 1]
    score_local = np.asarray(score_rows)[:, 1]
    indices = np.arange(len(local_mean))

    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.45))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    axes[0].hist(global_samples, bins=46, density=True, alpha=0.68,
                 color="tab:blue", label="COMPASS posterior")
    grid = np.linspace(
        exact[0] - 4 * math.sqrt(covariance[0, 0]),
        exact[0] + 4 * math.sqrt(covariance[0, 0]), 400,
    )
    density = np.exp(-0.5 * (grid - exact[0]) ** 2 / covariance[0, 0])
    density /= math.sqrt(2 * math.pi * covariance[0, 0])
    axes[0].plot(grid, density, "k--", lw=2, label="exact posterior")
    axes[0].axvline(float(raw["global_truth"]), color="tab:red", ls=":", lw=2,
                    label="true global")
    axes[0].axvline(float(joint_rows[0, 0]), color="tab:green",
                    ls=(0, (1.2, 2.0)), lw=2.5, label="COMPASS joint MAP")
    axes[0].axvline(float(score_rows[0, 0]), color="tab:purple", ls="-.", lw=2.2,
                    label="COMPASS score MAP")
    axes[0].set(xlabel="global parameter g", ylabel="density", title="Shared posterior")
    axes[0].legend(fontsize=8)

    axes[1].errorbar(indices, local_mean, yerr=local_std, fmt="o", ms=3.8,
                     alpha=0.8, label="posterior mean ± σ")
    axes[1].plot(indices, exact[1:], "_", ms=9, color="black", label="exact MAP")
    axes[1].scatter(indices, joint_local, s=34, color="tab:green", marker="D",
                    edgecolor="white", linewidth=0.5, label="COMPASS joint MAP")
    axes[1].scatter(indices, score_local, s=32, color="tab:purple",
                    edgecolor="white", linewidth=0.5, label="COMPASS score MAP")
    axes[1].set(xlabel="observation", ylabel="local parameter ℓᵢ",
                title=f"{len(indices)} local posteriors")
    axes[1].legend(fontsize=8)

    axes[2].scatter(exact[1:], joint_local, color="tab:green", marker="D", s=43,
                    edgecolor="white", linewidth=0.5, label="COMPASS joint MAP")
    axes[2].scatter(exact[1:], score_local, color="tab:purple", s=40,
                    edgecolor="white", linewidth=0.5, label="COMPASS score MAP")
    lo = min(float(exact[1:].min()), float(joint_local.min()), float(score_local.min()))
    hi = max(float(exact[1:].max()), float(joint_local.max()), float(score_local.max()))
    pad = max(0.04 * (hi - lo), 1e-3)
    axes[2].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.5)
    joint_mae = np.mean(np.abs(joint_local - exact[1:]) / exact_local_std)
    score_mae = np.mean(np.abs(score_local - exact[1:]) / exact_local_std)
    axes[2].text(0.04, 0.94, f"joint mean |error| = {joint_mae:.2f} analytic σ",
                 transform=axes[2].transAxes, va="top", color="tab:green")
    axes[2].text(0.04, 0.87, f"score mean |error| = {score_mae:.2f} analytic σ",
                 transform=axes[2].transAxes, va="top", color="tab:purple")
    axes[2].set(xlabel="exact local MAP", ylabel="estimated local MAP",
                title="Local MAP recovery")
    axes[2].legend(fontsize=8, loc="lower right")
    finish_figure(fig, output)


def selected_local_variants(args) -> tuple[tuple[str, str, str, str], ...]:
    variants = LOCAL_VARIANTS
    if args.include_gaussian_baseline:
        variants = (LOCAL_GAUSSIAN_BASELINE, *variants)
    return tuple(
        item for item in variants
        if args.only_method == "all" or item[3] == args.only_method
    )


def local_figure_path(args, filename: str) -> Path:
    """Tag non-default integration grids in filenames, not inside figures."""
    path = args.output_dir / filename
    if args.timesteps == 100:
        return path
    return path.with_name(
        f"{path.stem}_integration_steps_{args.timesteps}{path.suffix}"
    )


def run_local_panels(model, reference: dict[str, np.ndarray], args, device: str) -> None:
    observations = np.asarray(reference["x_observed"], dtype=np.float32)
    n = len(observations)
    precision = torch.full((n, 1), 1.0 + 1.0 / (1.0 + 0.5**2))
    diagnostic_rows = []
    for key, title, filename, correction in selected_local_variants(args):
        variant = {
            "key": key, "label": title, "correction": correction,
            "method": "dpm", "order": 2, "corrector_steps": 5,
            "corrector_steps_interval": 5, "final_corrector_steps": 3, "snr": 0.1,
        }
        if correction == "gauss":
            # Preserve the original 02a settings; only the integration grid changes.
            variant.update(
                corrector_steps=10, corrector_steps_interval=1,
                final_corrector_steps=3, snr=0.2,
            )
        if correction in ("hybrid", "hybrid_damping"):
            variant["normalizer"] = "adaptive_a_t_v2"
        inference_seed = (
            1_208 if correction == "gauss"
            else args.seed + sum(map(ord, key))
        )
        path = args.output_dir / "cache" / "shared_local" / f"{key}.npz"
        config = {
            "schema": 2, "checkpoint": source_signature(args.local_checkpoint),
            "reference": source_signature(args.local_reference), "variant": variant,
            "posterior_samples": args.posterior_samples, "timesteps": args.timesteps,
            "seed": inference_seed, "map_method": "score",
        }
        try:
            result = load_cell(path, config, args.force_new)
        except CacheSettingsMismatch:
            if args.plot_only:
                raise
            print(f"Recalculating selected cache with updated settings: {path}")
            result = None
        if result is None:
            if args.plot_only:
                raise FileNotFoundError(path)
            seed_all(config["seed"])
            started = time.perf_counter()
            sample_precision = precision if correction != "damping" else None
            samples = model.sample(
                x=torch.as_tensor(observations, device=device),
                multi_obs_inference=True, hierarchy=[0], prior=([0.0], [1.0]),
                correction=correction, posterior_precision=sample_precision,
                damping_at_noise=n ** -0.5,
                composition_batch_size=(n if correction == "damping" else None),
                num_samples=args.posterior_samples, timesteps=args.timesteps,
                method=variant["method"], order=variant["order"],
                corrector_steps=variant["corrector_steps"],
                corrector_steps_interval=variant["corrector_steps_interval"],
                final_corrector_steps=variant["final_corrector_steps"],
                snr=variant["snr"], device=device, verbose=args.verbose,
            ).detach().cpu()
            runtime = time.perf_counter() - started
            if not torch.isfinite(samples).all():
                raise RuntimeError(
                    f"{title} produced non-finite shared/local samples. "
                    "Increase --timesteps or inspect the checkpoint scores."
                )
            synchronization = float((samples[:, :, 0] - samples[:1, :, 0]).abs().max())
            save_cell(
                path, config, x_observed=reference["x_observed"],
                global_truth=reference["global_truth"], local_truth=reference["local_truth"],
                exact_joint_mean=reference["exact_joint_mean"],
                exact_joint_covariance=reference["exact_joint_covariance"],
                compass_global_samples=samples[0, :, 0].numpy(),
                compass_local_samples=samples[:, :, 1].numpy(),
                runtime_seconds=runtime,
                shared_synchronization_max_abs=synchronization,
            )
            result = load_npz(path)
        score_path = args.output_dir / "cache" / "shared_local" / f"{key}_score_map.npz"
        score_config = {
            **config, "map_kind": "fixed_shared_per_observation_score_ascent",
            "map_timesteps": args.map_timesteps,
            "map_iterations": args.map_iterations, "eps": 1e-3,
        }
        try:
            score_result = load_cell(score_path, score_config, args.force_new)
        except CacheSettingsMismatch:
            if args.plot_only:
                raise
            print(f"Recalculating selected cache with updated settings: {score_path}")
            score_result = None
        if score_result is None:
            if args.plot_only:
                raise FileNotFoundError(score_path)
            global_samples = torch.as_tensor(result["compass_global_samples"])
            local_samples = torch.as_tensor(result["compass_local_samples"])
            initial_rows = torch.stack([
                global_samples.mean().repeat(n), local_samples.mean(dim=1),
                torch.as_tensor(observations).flatten(),
            ], dim=1)
            posterior_scale = max(
                float(global_samples.std(unbiased=False)),
                float(local_samples.std(dim=1, unbiased=False).max()),
            )
            score_rows = model.map_estimate(
                data=initial_rows, condition_mask=torch.tensor([1.0, 0.0, 1.0]),
                init=initial_rows, sigma_start=max(2.0 * posterior_scale, 1e-3),
                timesteps=args.map_timesteps, eps=1e-3,
                iterations_per_level=args.map_iterations, device=device,
            ).numpy()
            save_cell(score_path, score_config, score_map_rows=score_rows)
        else:
            score_rows = np.asarray(score_result["score_map_rows"])

        joint_path = args.output_dir / "cache" / "shared_local" / f"{key}_joint_map.npz"
        joint_config = {
            **config, "map_kind": "shared_marginal_kde_then_fixed_shared_local_score",
            "map_num_starts": args.map_num_starts,
            "map_timesteps": args.map_timesteps,
            "map_iterations": args.map_iterations, "eps": 1e-3,
        }
        try:
            joint_result = load_cell(joint_path, joint_config, args.force_new)
        except CacheSettingsMismatch:
            if args.plot_only:
                raise
            print(f"Recalculating selected cache with updated settings: {joint_path}")
            joint_result = None
        if joint_result is None:
            if args.plot_only:
                raise FileNotFoundError(joint_path)
            global_samples = torch.as_tensor(result["compass_global_samples"])
            local_samples = torch.as_tensor(result["compass_local_samples"])
            posterior_samples = torch.empty(n, len(global_samples), 2)
            posterior_samples[:, :, 0] = global_samples.unsqueeze(0)
            posterior_samples[:, :, 1] = local_samples
            joint_rows_tensor, shared_result = ModelTransfuser._shared_then_local_map(
                model=model, posterior_samples=posterior_samples,
                x=torch.as_tensor(observations),
                condition_mask=torch.tensor([0.0, 0.0, 1.0]), hierarchy=[0],
                num_starts=args.map_num_starts, timesteps=args.map_timesteps,
                eps=1e-3, iterations_per_level=args.map_iterations, device=device,
            )
            joint_rows = joint_rows_tensor.numpy()
            save_cell(
                joint_path, joint_config, joint_map_rows=joint_rows,
                shared_map=shared_result["shared_map"].numpy(),
                candidate_modes=shared_result["candidate_modes"].numpy(),
                candidate_scores=shared_result["candidate_log_densities"].numpy(),
                selected_candidate=shared_result["selected_start"],
                conditional_effective_samples=shared_result["effective_sample_size"],
            )
        else:
            joint_rows = np.asarray(joint_result["joint_map_rows"])

        global_values = np.asarray(result["compass_global_samples"], dtype=float)
        analytic_mean = float(result["exact_joint_mean"][0])
        analytic_std = float(np.sqrt(result["exact_joint_covariance"][0, 0]))
        empirical_std = float(global_values.std(ddof=1))
        standardized = (global_values - global_values.mean()) / empirical_std
        exact_local_std = np.sqrt(np.diag(result["exact_joint_covariance"])[1:])
        diagnostic_rows.append({
            "variant": key, "label": title,
            "shared_sample_mean": float(global_values.mean()),
            "shared_sample_std": empirical_std,
            "analytic_mean": analytic_mean, "analytic_std": analytic_std,
            "mean_error_analytic_sigma": float(
                (global_values.mean() - analytic_mean) / analytic_std
            ),
            "std_ratio": empirical_std / analytic_std,
            "skewness": float(np.mean(standardized ** 3)),
            "excess_kurtosis": float(np.mean(standardized ** 4) - 3.0),
            "calibration_gap": float(np.mean(np.abs(calibration_curve(
                global_values, analytic_mean, analytic_std,
            ) - NOMINAL_COVERAGE))),
            "joint_shared_map": float(joint_rows[0, 0]),
            "score_shared_map": float(score_rows[0, 0]),
            "joint_local_mean_abs_error_sigma": float(np.mean(
                np.abs(joint_rows[:, 1] - result["exact_joint_mean"][1:])
                / exact_local_std
            )),
            "score_local_mean_abs_error_sigma": float(np.mean(
                np.abs(score_rows[:, 1] - result["exact_joint_mean"][1:])
                / exact_local_std
            )),
        })
        plot_shared_local_score_map(
            result, joint_rows, score_rows, local_figure_path(args, filename), title,
        )
    if diagnostic_rows:
        diagnostics_path = args.output_dir / "shared_local_posterior_diagnostics.csv"
        updated_variants = {row["variant"] for row in diagnostic_rows}
        retained_rows = [
            row for row in read_rows(diagnostics_path)
            if row.get("variant") not in updated_variants
        ]
        write_rows(diagnostics_path, retained_rows + diagnostic_rows)


def expected_compute_paths(args, stages: set[str]) -> list[Path]:
    paths = []
    if "learned" in stages:
        for repeat in range(args.repeats):
            for n in args.n_values:
                paths.extend(
                    learned_cell_path(args.output_dir, "learned_solver", item["key"], repeat, n)
                    for item in SOLVER_VARIANTS
                    if args.only_method == "all" or item["correction"] == args.only_method
                )
                if args.only_method in ("all", "damping"):
                    paths.extend(
                        learned_cell_path(args.output_dir, "d1_sweep", key, repeat, n)
                        for key, _, _ in SWEEP_VARIANTS
                        if key != "d1_inverse_sqrt_n"
                    )
    if "analytic" in stages:
        for name in ("prior1p0_likelihood1p0", "prior0p3_likelihood0p5"):
            for repeat in range(args.repeats):
                for n in args.n_values:
                    paths.extend(
                        learned_cell_path(args.output_dir, f"analytic_{name}", item["key"], repeat, n)
                        for item in ANALYTIC_VARIANTS
                        if args.only_method == "all" or item["correction"] == args.only_method
                    )
    if "local" in stages:
        for key, _, _, correction in selected_local_variants(args):
            paths.append(args.output_dir / "cache" / "shared_local" / f"{key}.npz")
            paths.append(args.output_dir / "cache" / "shared_local" / f"{key}_score_map.npz")
            paths.append(args.output_dir / "cache" / "shared_local" / f"{key}_joint_map.npz")
    return paths


def selected_hybrid_cache_needs_refresh(paths: list[Path]) -> bool:
    """Detect the one formula-version change before choosing CPU versus GPU."""
    for path in paths:
        if "hybrid_damping" not in str(path) or not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                cached_json = str(archive["config_json"].item())
            cached_config = json.loads(cached_json)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            # Malformed caches are reported by the strict loader; do not silently
            # classify them as the known, safely replaceable formula update.
            continue
        variant = cached_config.get("variant", {})
        if variant.get("normalizer") != "adaptive_a_t_v2":
            return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shared-checkpoint", type=Path, default=DEFAULT_SHARED_CHECKPOINT)
    parser.add_argument("--local-checkpoint", type=Path, default=DEFAULT_LOCAL_CHECKPOINT)
    parser.add_argument("--scaling-raw", type=Path, default=DEFAULT_SCALING_RAW)
    parser.add_argument("--local-reference", type=Path, default=DEFAULT_LOCAL_REFERENCE)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--posterior-samples", type=int, default=3_000)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-values", type=parse_n_values,
                        default=DEFAULT_N_VALUES)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--map-timesteps", type=int, default=100)
    parser.add_argument("--map-iterations", type=int, default=3)
    parser.add_argument("--map-num-starts", type=int, default=8)
    parser.add_argument(
        "--include-gaussian-baseline", action="store_true",
        help=(
            "include the historical DPM2 + Gaussian correction in local-panel "
            "runs while preserving its dense-corrector settings"
        ),
    )
    parser.add_argument("--adaptive-abs-tol", type=float, default=0.002576)
    parser.add_argument("--adaptive-rel-tol", type=float, default=0.1)
    parser.add_argument("--adaptive-safety", type=float, default=0.9)
    parser.add_argument("--adaptive-exponent", type=float, default=0.9)
    parser.add_argument("--adaptive-max-evals", type=int, default=10_000)
    parser.add_argument("--stages", default="copy,learned,analytic,local",
                        help="comma-separated subset of copy,learned,analytic,local")
    parser.add_argument(
        "--only-method",
        choices=("all", "gauss", "damping", "gauss_damping", "hybrid", "hybrid_damping"),
        default="all",
        help=("calculate only one correction family; hybrid is Gaussian + "
              "(1-t) prior correction without damping"),
    )
    parser.add_argument("--force-new", action="store_true",
                        help=("recalculate only cells selected by --only-method; "
                              "never alters historical source results"))
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stages = {item.strip() for item in args.stages.split(",") if item.strip()}
    unknown = stages - {"copy", "learned", "analytic", "local"}
    if unknown:
        raise ValueError("Unknown stages: " + ", ".join(sorted(unknown)))
    if args.posterior_samples < 2 or args.repeats < 1 or args.timesteps < 2:
        raise ValueError("posterior samples/repeats/timesteps must be at least 2/1/2")
    if args.output_dir.resolve() == DEFAULT_OUTPUT.resolve() and (
        args.timesteps != 100 or args.include_gaussian_baseline
    ):
        raise ValueError(
            "Non-default local integration profiles must use a separate "
            "--output-dir so the existing 100-step figures are not overwritten."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    if "copy" in stages:
        copy_historical_outputs(args.output_dir)

    selected_paths = expected_compute_paths(args, stages)
    missing = [path for path in selected_paths if not path.exists()]
    stale_hybrid = selected_hybrid_cache_needs_refresh(selected_paths)
    needs_compute = bool(missing) or stale_hybrid or args.force_new
    if args.plot_only and missing:
        raise FileNotFoundError(f"--plot-only requested but {len(missing)} damping cache cells are missing")
    if args.plot_only and stale_hybrid:
        raise CacheSettingsMismatch(
            "--plot-only requested but selected hybrid-damping caches use the "
            "obsolete precision normalizer. Rerun without --plot-only."
        )
    if stages & {"learned", "analytic", "local"}:
        runtime_device = args.device if needs_compute else "cpu"
        ensure_runtime(runtime_device, needs_compute)
    else:
        runtime_device = args.device

    solver_rows = []
    if "learned" in stages:
        pools = load_npz(args.scaling_raw)
        shared_model = load_score_checkpoint(args.shared_checkpoint, runtime_device)
        solver_rows, _ = run_learned_benchmarks(shared_model, pools, args, runtime_device)
        update_learned_scaling(solver_rows, args)
        del shared_model
        if runtime_device == "cuda":
            torch.cuda.empty_cache()
    if "analytic" in stages:
        run_analytic(args, runtime_device)
    if "local" in stages:
        local_reference = load_npz(args.local_reference)
        local_model = load_score_checkpoint(args.local_checkpoint, runtime_device)
        run_local_panels(local_model, local_reference, args, runtime_device)

    metadata = {
        "schema": 2, "cpu_limit": {"logical": CPU_LIMIT[0], "active": list(CPU_LIMIT[1])},
        "device": args.device, "posterior_samples": args.posterior_samples,
        "timesteps": args.timesteps, "repeats": args.repeats,
        "integration_grid_points": args.timesteps,
        "dpm_predictor_intervals": args.timesteps - 1,
        "integration_profile": f"integration_steps_{args.timesteps}",
        "integration_profile_is_in_figure_text": False,
        "n_values": list(args.n_values), "damping_at_data": 1.0,
        "default_damping_at_noise": "1/sqrt(N)",
        "composition_batch_size": "full observation set",
        "map_methods": ["score", "shared_marginal_kde_then_fixed_shared_local_score"],
        "map_num_starts": args.map_num_starts,
        "include_gaussian_baseline": args.include_gaussian_baseline,
        "only_method": args.only_method,
        "stages": sorted(stages), "solver_variants": SOLVER_VARIANTS,
        "sweep": [{"key": key, "label": label} for key, label, _ in SWEEP_VARIANTS],
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Damping-factor outputs are available under {args.output_dir}")


if __name__ == "__main__":
    main()
