"""Focused global/local recovery for the hierarchical SDE partial-pooling model."""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path


DEFAULT_DATASETS = 5
DEFAULT_SUBJECTS = 20
DEFAULT_DRAWS = 256
DEFAULT_TIMESTEPS = 50
DEFAULT_MAP_STARTS = 4
DEFAULT_MAP_TIMESTEPS = 20
DEFAULT_MAP_ITERATIONS = 3
DEFAULT_MAP_LOGPROB_TIMESTEPS = 20
FIGURE_DPI = 300
INFERENCE_METHODS = ("dpm2_gaussian", "langevin_fnpse")


def inference_plan(name):
    """Return the exact compositional sampler requested by a CLI profile."""
    if name == "dpm2_gaussian":
        return {
            "name": name,
            "sampler": "dpm",
            "correction": "gauss",
            "order": 2,
            "label": "DPM-Solver-2 + Gaussian",
        }
    if name == "langevin_fnpse":
        return {
            "name": name,
            "sampler": "langevin",
            "correction": "fnpe",
            "order": None,
            "label": "Langevin + F-NPSE",
        }
    raise ValueError(
        f"Unknown inference method {name!r}; choose one of "
        f"{', '.join(INFERENCE_METHODS)}."
    )



# Measured on this host: 2,935 seconds for one native-joint dataset with
# 100 subjects, 1,000 posterior draws, and 100 DPM steps.
REFERENCE_SECONDS = 2935.0
REFERENCE_WORK = 100 * 1000 * 100


def parser():
    result = argparse.ArgumentParser(
        description=(
            "Infer shared global and subject-local parameters for the "
            "hierarchical SDE partial-pooling model."
        )
    )
    result.add_argument(
        "--preset", choices=("smoke", "full", "large"), default="full",
    )
    result.add_argument("--root", type=Path)
    result.add_argument("--seed", type=int)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    result.add_argument("--datasets", type=int, default=DEFAULT_DATASETS)
    result.add_argument("--dataset-start", type=int, default=0)
    result.add_argument("--subjects", type=int, default=DEFAULT_SUBJECTS)
    result.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    result.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    result.add_argument(
        "--inference-method", choices=INFERENCE_METHODS,
        default="dpm2_gaussian",
        help=(
            "Primary DPM-Solver-2 + Gaussian composition, or the "
            "Langevin + F-NPSE reference."
        ),
    )
    result.add_argument(
        "--gaussian-precision-samples", type=int, default=256,
        help="Single-subject draws used once per dataset for Gaussian precision.",
    )
    result.add_argument(
        "--gaussian-precision-timesteps", type=int, default=50,
        help="Diffusion steps for the reusable Gaussian precision estimate.",
    )
    result.add_argument(
        "--langevin-steps-per-level", type=int, default=10,
        help="Annealed Langevin updates at every F-NPSE noise level.",
    )
    result.add_argument(
        "--langevin-snr", type=float, default=0.1,
        help="Langevin signal-to-noise ratio for the F-NPSE reference.",
    )
    result.add_argument("--map-starts", type=int, default=DEFAULT_MAP_STARTS)
    result.add_argument("--map-timesteps", type=int, default=DEFAULT_MAP_TIMESTEPS)
    result.add_argument("--map-iterations", type=int, default=DEFAULT_MAP_ITERATIONS)
    result.add_argument(
        "--map-logprob-timesteps", type=int,
        default=DEFAULT_MAP_LOGPROB_TIMESTEPS,
    )
    result.add_argument(
        "--progress-every", type=int, default=5,
        help="Print progress after this many posterior batches.",
    )
    result.add_argument("--force", action="store_true")
    result.add_argument(
        "--skip-observation-sweep", action="store_true",
        help="Skip the 1, 2, 4, ... subject recovery experiment.",
    )
    result.add_argument(
        "--estimate-only", action="store_true",
        help="Print the workload estimate without loading PyTorch or a checkpoint.",
    )
    return result


def positive(name, value):
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}.")
    return value


def observation_counts(maximum):
    """Powers of two followed by the exact maximum subject count."""
    positive("maximum observations", maximum)
    counts = []
    value = 1
    while value < maximum:
        counts.append(value)
        value *= 2
    if not counts or counts[-1] != maximum:
        counts.append(maximum)
    return tuple(counts)


def density_panel_counts(counts, columns=4):
    """Select evenly spaced sweep entries, retaining both endpoints."""
    counts = tuple(sorted(set(int(count) for count in counts)))
    positive("density columns", columns)
    if len(counts) <= columns:
        return counts
    indices = [
        round(index * (len(counts) - 1) / (columns - 1))
        for index in range(columns)
    ]
    return tuple(counts[index] for index in indices)


def draw_batch_size(draws, training_batch_size, subjects):
    """Keep subject rows per transformer call within the training batch size."""
    return max(1, min(draws, training_batch_size // max(1, subjects)))


def estimated_runtime_seconds(datasets, subjects, draws, timesteps):
    """Linear estimate based on the measured native-joint full-preset run."""
    work = datasets * subjects * draws * timesteps
    return REFERENCE_SECONDS * work / REFERENCE_WORK


def estimated_sweep_runtime_seconds(datasets, subjects, draws, timesteps):
    """Sampling estimate for 1, 2, 4, ... through the maximum subjects."""
    return estimated_runtime_seconds(
        datasets, sum(observation_counts(subjects)), draws, timesteps,
    )


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 120:
        return f"{seconds:.0f} seconds"
    if seconds < 7200:
        return f"{seconds / 60:.1f} minutes"
    return f"{seconds / 3600:.1f} hours"


def inference_signature(config_signature, args, plan):
    payload = {
        "config_signature": config_signature,
        "dataset_start": args.dataset_start,
        "datasets": args.datasets,
        "subjects": args.subjects,
        "draws": args.draws,
        "timesteps": args.timesteps,
        "inference_method": plan["name"],
        "sampler": plan["sampler"],
        "correction": plan["correction"],
        "order": plan["order"],
        "gaussian_precision_samples": (
            args.gaussian_precision_samples
            if plan["correction"] == "gauss" else None
        ),
        "gaussian_precision_timesteps": (
            args.gaussian_precision_timesteps
            if plan["correction"] == "gauss" else None
        ),
        "langevin_steps_per_level": (
            args.langevin_steps_per_level
            if plan["correction"] == "fnpe" else None
        ),
        "langevin_snr": (
            args.langevin_snr if plan["correction"] == "fnpe" else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _normalized_global_prior(normalizer, torch):
    from Partial_Pooling.schema import GLOBAL_INDICES
    from Partial_Pooling.simulators.priors import GLOBAL_PRIOR_MEAN, GLOBAL_PRIOR_STD

    indices = torch.as_tensor(GLOBAL_INDICES, dtype=torch.long)
    mean = torch.as_tensor(GLOBAL_PRIOR_MEAN, dtype=torch.float32)
    std = torch.as_tensor(GLOBAL_PRIOR_STD, dtype=torch.float32)
    return (
        (mean - normalizer.mean[indices]) / normalizer.scale[indices],
        std / normalizer.scale[indices],
    )


def _sample_dataset(model, observations, normalizers, args, config, device, torch):
    from Partial_Pooling.schema import GLOBAL_INDICES

    plan = inference_plan(args.inference_method)
    normalized_observations = normalizers["observations"].transform(observations)
    theta_normalizer = normalizers["theta"]
    condition_mask = torch.cat((
        torch.zeros(model.nodes_size - normalized_observations.shape[-1]),
        torch.ones(normalized_observations.shape[-1]),
    )).to(device)
    prior = _normalized_global_prior(theta_normalizer, torch)
    batch_size = draw_batch_size(args.draws, config.batch_size, args.subjects)
    batch_count = math.ceil(args.draws / batch_size)
    chunks = []
    posterior_precision = None
    totals = {"score_network_calls": 0, "evaluated_subject_rows": 0}
    started = time.perf_counter()

    print(
        f"  {plan['label']}: sampling {args.draws} draws in {batch_count} batches "
        f"(draw batch={batch_size}, subjects={args.subjects})",
        flush=True,
    )
    for batch_index, start in enumerate(range(0, args.draws, batch_size), start=1):
        count = min(batch_size, args.draws - start)
        chunk = model.sample(
            x=normalized_observations.to(device),
            condition_mask=condition_mask,
            timesteps=args.timesteps,
            num_samples=count,
            multi_obs_inference=True,
            hierarchy=list(GLOBAL_INDICES),
            prior=prior,
            correction=plan["correction"],
            posterior_precision=posterior_precision,
            precision_est_samples=args.gaussian_precision_samples,
            precision_est_timesteps=args.gaussian_precision_timesteps,
            order=plan["order"] or 2,
            snr=args.langevin_snr,
            corrector_steps_interval=1,
            corrector_steps=(
                args.langevin_steps_per_level
                if plan["sampler"] == "langevin" else 0
            ),
            final_corrector_steps=0,
            device=device,
            verbose=False,
            method=plan["sampler"],
        ).cpu()
        if plan["correction"] == "gauss" and posterior_precision is None:
            posterior_precision = (
                model.multi_obs_sampler.posterior_precision.detach().cpu()
            )
        chunks.append(chunk)
        stats = dict(model.multi_obs_sampler.solver_stats or {})
        for key in totals:
            totals[key] += int(stats.get(key, 0))

        if (
            batch_index == 1
            or batch_index == batch_count
            or batch_index % args.progress_every == 0
        ):
            completed = min(start + count, args.draws)
            elapsed = time.perf_counter() - started
            eta = elapsed * (args.draws - completed) / max(completed, 1)
            print(
                f"  draws {completed:4d}/{args.draws} "
                f"| elapsed {format_duration(elapsed)} "
                f"| ETA {format_duration(eta)}",
                flush=True,
            )

    normalized = torch.cat(chunks, dim=1)
    shared = normalized[:, :, :len(GLOBAL_INDICES)]
    if not torch.allclose(shared, shared[:1].expand_as(shared), atol=1e-5):
        raise RuntimeError("Partial-pooling inference returned unsynchronized globals.")
    raw = theta_normalizer.inverse(
        normalized.reshape(-1, normalized.shape[-1])
    ).reshape_as(normalized)
    posterior = {
        "globals": raw[0, :, :len(GLOBAL_INDICES)],
        "locals": raw[:, :, len(GLOBAL_INDICES):].permute(1, 0, 2),
        "posterior_precision": posterior_precision,
    }
    totals["score_evaluations"] = totals["score_network_calls"]
    return posterior, totals, time.perf_counter() - started


def _map_settings(args):
    plan = inference_plan(args.inference_method)
    return {
        "version": 2,
        "starts": int(args.map_starts),
        "timesteps": int(args.map_timesteps),
        "iterations_per_level": int(args.map_iterations),
        "logprob_timesteps": int(args.map_logprob_timesteps),
        "sampling_correction": plan["correction"],
        "optimizer_correction": (
            "gauss" if plan["correction"] == "gauss" else None
        ),
        "estimator": (
            "hierarchical_score_map"
            if plan["correction"] == "gauss"
            else "posterior_global_kde_mode"
        ),
    }


def _fnpse_kde_map(posterior, args, torch):
    """Select a coherent draw at the joint KDE mode of F-NPSE globals."""
    from scipy.stats import gaussian_kde
    from numpy.linalg import LinAlgError

    started = time.perf_counter()
    global_draws = posterior["globals"].double()
    scale = global_draws.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardized = (
        (global_draws - global_draws.mean(dim=0)) / scale
    ).numpy()
    try:
        log_density = gaussian_kde(standardized.T).logpdf(standardized.T)
        scores = torch.from_numpy(log_density).float()
    except (ValueError, RuntimeError, LinAlgError):
        scores = -torch.from_numpy((standardized ** 2).sum(axis=1)).float()
    selected = int(torch.argmax(scores))
    return {
        "globals": posterior["globals"][selected],
        "locals": posterior["locals"][selected],
        "selected_candidate": selected,
        "candidate_scores": scores,
        "settings": _map_settings(args),
        "runtime_seconds": time.perf_counter() - started,
        "shared_synchronization_max_abs": 0.0,
        "estimator": "posterior_global_kde_mode",
    }


def _joint_map_dataset(
    model, observations, posterior, normalizers, args, config, device, torch,
):
    """Estimate a compatible global/local mode for the selected sampler."""
    from compass.ModelTransfuser import ModelTransfuser
    from Partial_Pooling.schema import GLOBAL_INDICES

    plan = inference_plan(args.inference_method)
    if plan["correction"] == "fnpe":
        return _fnpse_kde_map(posterior, args, torch)

    posterior_precision = posterior.get("posterior_precision")
    if posterior_precision is None:
        raise RuntimeError(
            "DPM2 + Gaussian MAP requires the precision estimate saved during "
            "posterior sampling."
        )
    started = time.perf_counter()
    normalized_observations = normalizers["observations"].transform(observations)
    theta_normalizer = normalizers["theta"]
    global_draws = posterior["globals"]
    local_draws = posterior["locals"]
    subjects = int(observations.shape[0])
    draws = int(global_draws.shape[0])
    latent_raw = torch.cat((
        global_draws.unsqueeze(0).expand(subjects, -1, -1),
        local_draws.permute(1, 0, 2),
    ), dim=2)
    latent_width = int(latent_raw.shape[-1])
    latent_normalized = theta_normalizer.transform(
        latent_raw.reshape(-1, latent_width)
    ).reshape(subjects, draws, latent_width)
    condition_mask = torch.cat((
        torch.zeros(latent_width),
        torch.ones(normalized_observations.shape[-1]),
    ))
    hierarchy = list(GLOBAL_INDICES)
    prior = _normalized_global_prior(theta_normalizer, torch)
    starts, annealing_scales = ModelTransfuser._joint_map_initializations(
        latent_normalized, normalized_observations, condition_mask,
        hierarchy, args.map_starts,
    )

    print(
        f"  refining Gaussian joint MAP from {starts.shape[1]} starts "
        f"({args.map_timesteps} levels)",
        flush=True,
    )
    candidates = []
    for start_index, sigma_start in enumerate(annealing_scales):
        start = starts[:, start_index, :]
        refined = model.hierarchical_map_estimate(
            data=start,
            condition_mask=condition_mask,
            init=start,
            hierarchy=hierarchy,
            prior=prior,
            correction="gauss",
            posterior_precision=posterior_precision,
            sigma_start=sigma_start,
            timesteps=args.map_timesteps,
            iterations_per_level=args.map_iterations,
            max_iterations_per_level=args.map_iterations,
            device=device,
        )
        candidates.append(refined[:, 0, :])
    candidate_tensor = torch.stack(candidates, dim=1)
    scores = ModelTransfuser._hierarchical_candidate_scores(
        model, candidate_tensor, condition_mask, hierarchy, prior,
        args.map_logprob_timesteps, 1e-3, device, False,
    ).cpu()
    selected = int(torch.argmax(scores))
    normalized_map = candidate_tensor[:, selected, :latent_width].cpu()
    raw_map = theta_normalizer.inverse(normalized_map)
    sync_error = float(
        (raw_map[:, :len(hierarchy)] - raw_map[:1, :len(hierarchy)])
        .abs().max()
    )
    if sync_error > 1e-4:
        raise RuntimeError(
            f"Joint MAP globals are not synchronized (error {sync_error:.3e})."
        )
    return {
        "globals": raw_map[0, :len(hierarchy)],
        "locals": raw_map[:, len(hierarchy):],
        "selected_candidate": selected,
        "candidate_scores": scores,
        "settings": _map_settings(args),
        "runtime_seconds": time.perf_counter() - started,
        "shared_synchronization_max_abs": sync_error,
        "estimator": "hierarchical_score_map",
    }




def _parameter_rows(dataset_id, scope, names, draws, truth, joint_map):
    rows = []
    if scope == "global":
        draws = draws[:, None, :]
        truth = truth[None, :]
        joint_map = joint_map[None, :]
    for unit in range(truth.shape[0]):
        for parameter, name in enumerate(names):
            values = draws[:, unit, parameter]
            target = float(truth[unit, parameter])
            estimate = float(values.mean())
            map_estimate = float(joint_map[unit, parameter])
            std = float(values.std(unbiased=False))
            lower = float(values.quantile(0.025))
            median = float(values.quantile(0.5))
            upper = float(values.quantile(0.975))
            map_error = map_estimate - target
            mean_error = estimate - target
            rows.append({
                "dataset": dataset_id,
                "scope": scope,
                "unit": "shared" if scope == "global" else unit,
                "parameter": name,
                "truth": target,
                "joint_map": map_estimate,
                "posterior_mean": estimate,
                "posterior_median": median,
                "posterior_std": std,
                "lower_95": lower,
                "upper_95": upper,
                "error": map_error,
                "absolute_error": abs(map_error),
                "joint_map_error": map_error,
                "joint_map_absolute_error": abs(map_error),
                "posterior_mean_error": mean_error,
                "posterior_z_error": map_error / max(std, 1e-12),
                "covered_95": int(lower <= target <= upper),
            })
    return rows
def _complete_artifact_joint_map(
    artifact, model, source, normalizers, args, config, device, torch,
):
    """Add or refresh the joint MAP and all MAP-based recovery rows."""
    from Partial_Pooling.schema import GLOBAL_NAMES, LOCAL_NAMES

    dataset_id = int(artifact["dataset_id"])
    subjects = int(artifact["subjects"])
    observations = source["observations"][dataset_id, :subjects]
    current = artifact.get("joint_map")
    if current is None or current.get("settings") != _map_settings(args):
        current = _joint_map_dataset(
            model, observations, artifact["posterior"], normalizers,
            args, config, device, torch,
        )
        artifact["joint_map"] = current
        artifact["map_runtime_seconds"] = float(current["runtime_seconds"])
    artifact["global_rows"] = _parameter_rows(
        dataset_id, "global", GLOBAL_NAMES,
        artifact["posterior"]["globals"], artifact["truth_globals"],
        current["globals"],
    )
    artifact["local_rows"] = _parameter_rows(
        dataset_id, "local", LOCAL_NAMES,
        artifact["posterior"]["locals"], artifact["truth_locals"],
        current["locals"],
    )
    return artifact


def _run_observation_sweep(
    base_artifacts, run_directory, signature, model, source, normalizers,
    args, config, device, torch,
):
    """Infer nested subject counts and retain resumable per-count artifacts."""
    from Partial_Pooling.artifact_names import (
        checkpoint_tag, inference_dataset_filename, inference_report_filename,
    )
    from Partial_Pooling.io_utils import atomic_json, atomic_torch
    from Partial_Pooling.rng import derive_seed

    plan = inference_plan(args.inference_method)
    counts = observation_counts(args.subjects)
    by_count = {args.subjects: sorted(
        base_artifacts, key=lambda artifact: int(artifact["dataset_id"])
    )}
    sweep_directory = run_directory / "observation_sweep"
    sweep_directory.mkdir(parents=True, exist_ok=True)

    for count in counts:
        if count == args.subjects:
            continue
        count_args = argparse.Namespace(**vars(args))
        count_args.subjects = count
        count_directory = sweep_directory / f"subjects-{count:04d}"
        count_directory.mkdir(parents=True, exist_ok=True)
        artifacts = []
        print(
            f"Observation sweep: {count} subjects "
            f"({count * config.trials} trials per dataset)",
            flush=True,
        )
        for dataset_id in range(
            args.dataset_start, args.dataset_start + args.datasets
        ):
            output = count_directory / inference_dataset_filename(
                plan["name"], dataset_id,
            )
            artifact = None
            if output.exists() and not args.force:
                existing = torch.load(output, map_location="cpu")
                if (
                    existing.get("inference_signature") == signature
                    and int(existing.get("subjects", -1)) == count
                ):
                    artifact = existing
            observations = source["observations"][dataset_id, :count]
            if artifact is None:
                seed = derive_seed(
                    config.root_seed, "partial_pooling_observation_sweep",
                    dataset_id, count,
                )
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                posterior, solver_stats, runtime_seconds = _sample_dataset(
                    model, observations, normalizers, count_args,
                    config, device, torch,
                )
                artifact = {
                    "inference_signature": signature,
                    "config_signature": config.signature,
                    "dataset_id": dataset_id,
                    "subject_ids": torch.arange(count),
                    "seed": seed,
                    "subjects": count,
                    "trials_per_subject": config.trials,
                    "draws": args.draws,
                    "timesteps": args.timesteps,
                    "inference_method": plan["name"],
                    "training_model": "sde_joint",
                    "training_samples": config.train_size,
                    "checkpoint_tag": checkpoint_tag(config, "sde_joint"),
                    "sampler": plan["sampler"],
                    "correction": plan["correction"],
                    "order": plan["order"],
                    "truth_globals": source["globals"][dataset_id],
                    "truth_locals": source["locals"][dataset_id, :count],
                    "posterior": posterior,
                    "solver_stats": solver_stats,
                    "runtime_seconds": runtime_seconds,
                }
            artifact = _complete_artifact_joint_map(
                artifact, model, source, normalizers, count_args,
                config, device, torch,
            )
            atomic_torch(output, artifact)
            artifacts.append(artifact)
            print(
                f"  dataset {dataset_id}: saved {output.relative_to(run_directory)}",
                flush=True,
            )
        by_count[count] = artifacts

    ordered = {count: by_count[count] for count in counts}
    atomic_json(sweep_directory / inference_report_filename(
        plan["name"], "manifest", "json",
    ), {
        "inference_signature": signature,
        "observation_unit": "subject",
        "trials_per_subject": config.trials,
        "counts": list(counts),
        "datasets": args.datasets,
        "inference_method": plan["name"],
        "training_model": "sde_joint",
        "training_samples": config.train_size,
        "checkpoint_tag": checkpoint_tag(config, "sde_joint"),
        "sampler": plan["sampler"],
        "correction": plan["correction"],
        "point_estimator": _map_settings(args)["estimator"],
        "map_settings": _map_settings(args),
    })
    return ordered




def _summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["scope"], row["parameter"]), []).append(row)
    result = {"global": {}, "local": {}}
    for (scope, parameter), values in sorted(groups.items()):
        errors = [float(row["error"]) for row in values]
        result[scope][parameter] = {
            "count": len(values),
            "mae": sum(abs(value) for value in errors) / len(errors),
            "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
            "coverage_95": sum(int(row["covered_95"]) for row in values) / len(values),
            "mean_posterior_z_error": (
                sum(float(row["posterior_z_error"]) for row in values) / len(values)
            ),
        }
    return result


def _load_completed(run_directory, signature, inference_method, torch):
    from Partial_Pooling.artifact_names import inference_dataset_glob

    artifacts = []
    for path in sorted(run_directory.glob(inference_dataset_glob(inference_method))):
        payload = torch.load(path, map_location="cpu")
        if payload.get("inference_signature") == signature:
            artifacts.append(payload)
    return artifacts


def _write_reports(run_directory, artifacts, inference_method):
    from Partial_Pooling.artifact_names import inference_report_filename
    from Partial_Pooling.io_utils import atomic_csv, atomic_json

    rows = [
        row
        for artifact in artifacts
        for row in [*artifact["global_rows"], *artifact["local_rows"]]
    ]
    global_rows = [row for row in rows if row["scope"] == "global"]
    local_rows = [row for row in rows if row["scope"] == "local"]
    atomic_csv(run_directory / inference_report_filename(
        inference_method, "global_recovery", "csv",
    ), global_rows)
    atomic_csv(run_directory / inference_report_filename(
        inference_method, "local_recovery", "csv",
    ), local_rows)
    summary = _summary(rows)
    sampling_runtime = sum(
        float(artifact["runtime_seconds"]) for artifact in artifacts
    )
    map_runtime = sum(
        float(artifact.get("map_runtime_seconds", 0.0))
        for artifact in artifacts
    )
    atomic_json(run_directory / inference_report_filename(
        inference_method, "summary", "json",
    ), {
        "inference_method": inference_method,
        "completed_datasets": len(artifacts),
        "summary": summary,
        "total_runtime_seconds": sampling_runtime + map_runtime,
        "total_sampling_runtime_seconds": sampling_runtime,
        "total_map_runtime_seconds": map_runtime,
    })
    return summary


def _plot_limits(xs, ys):
    lower = min([*xs, *ys])
    upper = max([*xs, *ys])
    span = upper - lower
    padding = 0.08 * span if span > 0 else max(abs(lower) * 0.08, 0.1)
    return lower - padding, upper + padding


def _finish_figure(fig, output, plt):
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output


def _dataset_legend(fig, dataset_ids, cmap):
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor=cmap(index % 10), markeredgecolor="none",
            label=f"dataset {dataset}",
        )
        for index, dataset in enumerate(dataset_ids)
    ]
    fig.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5),
        ncol=1, frameon=True,
    )


def _parameter_panels(
    rows, names, output, title, ylabel, value, reference, plt,
):
    columns = 4 if len(names) > 4 else len(names)
    panel_rows = math.ceil(len(names) / columns)
    fig, axes = plt.subplots(
        panel_rows, columns, figsize=(4.1 * columns, 3.7 * panel_rows),
        squeeze=False, constrained_layout=True,
    )
    dataset_ids = sorted({int(row["dataset"]) for row in rows})
    color_index = {dataset: index for index, dataset in enumerate(dataset_ids)}
    cmap = plt.get_cmap("tab10")
    for axis, name in zip(axes.flat, names):
        selected = [row for row in rows if row["parameter"] == name]
        xs = [float(row["truth"]) for row in selected]
        ys = [float(value(row)) for row in selected]
        colors = [cmap(color_index[int(row["dataset"])] % 10) for row in selected]
        axis.scatter(xs, ys, c=colors, s=24, alpha=0.65, edgecolors="none")
        reference(axis, xs, ys)
        errors = [
            float(row["joint_map"]) - float(row["truth"])
            for row in selected
        ]
        rmse = math.sqrt(sum(error ** 2 for error in errors) / len(errors))
        axis.set(
            title=f"{name}\nRMSE={rmse:.3g}",
            xlabel="simulator truth", ylabel=ylabel,
        )
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(names):]:
        axis.set_visible(False)
    _dataset_legend(fig, dataset_ids, cmap)
    fig.suptitle(title, fontsize=14)
    return _finish_figure(fig, output, plt)


def _parity_reference(axis, xs, ys):
    lower, upper = _plot_limits(xs, ys)
    axis.plot([lower, upper], [lower, upper], color="black", lw=1, ls="--")
    axis.set(xlim=(lower, upper), ylim=(lower, upper))


def _residual_reference(axis, xs, residuals):
    axis.axhline(0.0, color="black", lw=1, ls="--")
    if xs:
        lower, upper = min(xs), max(xs)
        padding = 0.05 * (upper - lower) if upper > lower else 0.1
        axis.set_xlim(lower - padding, upper + padding)


def _plot_parity(
    global_rows, local_rows, global_names, local_names, directory, plt,
    filename_suffix="", method_label=None,
):
    prefix = f"{method_label}: " if method_label else ""
    outputs = []
    outputs.append(_parameter_panels(
        global_rows, global_names,
        directory / f"global_parity{filename_suffix}.png",
        f"{prefix}Global parameter recovery", "joint MAP",
        lambda row: row["joint_map"], _parity_reference, plt,
    ))
    outputs.append(_parameter_panels(
        local_rows, local_names,
        directory / f"local_parity{filename_suffix}.png",
        f"{prefix}Local parameter recovery", "joint MAP",
        lambda row: row["joint_map"], _parity_reference, plt,
    ))
    return outputs


def _plot_residuals(
    global_rows, local_rows, global_names, local_names, directory, plt,
    filename_suffix="", method_label=None,
):
    prefix = f"{method_label}: " if method_label else ""
    value = lambda row: float(row["joint_map"]) - float(row["truth"])
    outputs = []
    outputs.append(_parameter_panels(
        global_rows, global_names,
        directory / f"global_residuals{filename_suffix}.png",
        f"{prefix}Global recovery residuals", "joint MAP - truth",
        value, _residual_reference, plt,
    ))
    outputs.append(_parameter_panels(
        local_rows, local_names,
        directory / f"local_residuals{filename_suffix}.png",
        f"{prefix}Local recovery residuals", "joint MAP - truth",
        value, _residual_reference, plt,
    ))
    return outputs


def _plot_global_density_grid(
    sweep_artifacts, global_names, output, plt, method_label=None,
):
    """Posterior marginal densities for dataset 0 across four subject counts."""
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.stats import gaussian_kde

    available = tuple(sorted(sweep_artifacts))
    counts = density_panel_counts(available, columns=4)
    dataset_id = min(
        int(artifact["dataset_id"])
        for artifact in sweep_artifacts[counts[0]]
    )
    artifacts = {
        count: next(
            artifact for artifact in sweep_artifacts[count]
            if int(artifact["dataset_id"]) == dataset_id
        )
        for count in counts
    }
    trials_per_subject = int(
        artifacts[counts[0]].get("trials_per_subject", 30)
    )
    fig, axes = plt.subplots(
        len(global_names), len(counts),
        figsize=(4.1 * len(counts), 2.55 * len(global_names)),
        squeeze=False, constrained_layout=True,
    )
    for parameter, name in enumerate(global_names):
        truth = float(artifacts[counts[0]]["truth_globals"][parameter])
        all_values = [
            float(value)
            for count in counts
            for value in artifacts[count]["posterior"]["globals"][:, parameter]
        ]
        all_maps = [
            float(artifacts[count]["joint_map"]["globals"][parameter])
            for count in counts
        ]
        lower = min([*all_values, *all_maps, truth])
        upper = max([*all_values, *all_maps, truth])
        span = upper - lower
        padding = 0.06 * span if span > 0 else 0.1
        grid = np.linspace(lower - padding, upper + padding, 300)
        for column, count in enumerate(counts):
            axis = axes[parameter, column]
            values = artifacts[count]["posterior"]["globals"][:, parameter].numpy()
            if float(np.std(values)) > 1e-10:
                density = gaussian_kde(values)(grid)
                axis.fill_between(grid, density, color="#56B4E9", alpha=0.30)
                axis.plot(grid, density, color="#0072B2", lw=1.5)
            else:
                axis.axvline(float(values[0]), color="#0072B2", lw=1.5)
            axis.axvline(truth, color="#D55E00", lw=1.4, ls=":")
            axis.axvline(
                float(artifacts[count]["joint_map"]["globals"][parameter]),
                color="#009E73", lw=1.3, ls="-.",
            )
            if parameter == 0:
                axis.set_title(
                    f"{count} subject{'s' if count != 1 else ''}\n"
                    f"({count * trials_per_subject} trials)"
                )
            if column == 0:
                axis.set_ylabel(f"{name}\ndensity")
            if parameter == len(global_names) - 1:
                axis.set_xlabel("parameter value")
            axis.set_xlim(grid[0], grid[-1])
            axis.set_yticks([])
            axis.grid(axis="x", alpha=0.16)
    fig.legend(
        handles=[
            Patch(facecolor="#56B4E9", alpha=0.30, label="posterior sample density"),
            Line2D([0], [0], color="#0072B2", lw=1.5, label="KDE"),
            Line2D([0], [0], color="#D55E00", lw=1.4, ls=":", label="simulator truth"),
            Line2D([0], [0], color="#009E73", lw=1.3, ls="-.", label="joint MAP"),
        ],
        loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=True,
    )
    method = f"{method_label}: " if method_label else ""
    fig.suptitle(
        f"{method}Global posterior densities as observations increase "
        f"(dataset {dataset_id})",
        fontsize=14,
    )
    return _finish_figure(fig, output, plt)


def _plot_local_shrinkage(
    global_rows, local_rows, local_names, output, plt, method_label=None,
):
    global_mean_names = {
        "nu": "mu_nu", "log_alpha": "mu_log_alpha", "log_t0": "mu_log_t0",
    }
    datasets = sorted({int(row["dataset"]) for row in local_rows})
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        1, len(local_names), figsize=(5.0 * len(local_names), 4.5),
        constrained_layout=True,
    )
    for axis, local_name in zip(axes, local_names):
        selected = [row for row in local_rows if row["parameter"] == local_name]
        for dataset_index, dataset_id in enumerate(datasets):
            color = cmap(dataset_index % 10)
            dataset_rows = [row for row in selected if int(row["dataset"]) == dataset_id]
            truths = [float(row["truth"]) for row in dataset_rows]
            estimates = [float(row["joint_map"]) for row in dataset_rows]
            for truth, estimate in zip(truths, estimates):
                axis.plot([truth, truth], [truth, estimate], color=color, alpha=0.12, lw=0.8)
            axis.scatter(
                truths, estimates, color=color, s=20, alpha=0.55,
                edgecolors="none", label=f"dataset {dataset_id}",
            )
        mean_name = global_mean_names[local_name]
        mean_rows = [row for row in global_rows if row["parameter"] == mean_name]
        axis.scatter(
            [float(row["truth"]) for row in mean_rows],
            [float(row["joint_map"]) for row in mean_rows],
            marker="*", s=130, color="black", label="population means", zorder=4,
        )
        truths = [float(row["truth"]) for row in selected]
        estimates = [float(row["joint_map"]) for row in selected]
        lower, upper = _plot_limits(truths, estimates)
        axis.plot([lower, upper], [lower, upper], color="black", lw=1, ls="--")
        axis.set(
            title=local_name, xlabel="simulator truth", ylabel="joint MAP",
            xlim=(lower, upper), ylim=(lower, upper),
        )
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1.005, 0.5),
        ncol=1, frameon=True,
    )
    method = f"{method_label}: " if method_label else ""
    fig.suptitle(
        f"{method}Hierarchical shrinkage: subject estimates and population means",
        fontsize=14,
    )
    return _finish_figure(fig, output, plt)


def _plot_error_distributions(
    global_rows, local_rows, global_names, local_names, output, plt,
    method_label=None,
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for axis, rows, names, title in (
        (axes[0], global_rows, global_names, "Global joint MAP errors"),
        (axes[1], local_rows, local_names, "Local joint MAP errors"),
    ):
        values = [
            [
                float(row["joint_map"]) - float(row["truth"])
                for row in rows if row["parameter"] == name
            ]
            for name in names
        ]
        boxes = axis.boxplot(
            values, labels=names, patch_artist=True, showmeans=True,
            meanprops={
                "marker": "D", "markerfacecolor": "#009E73",
                "markeredgecolor": "black",
            },
        )
        for box in boxes["boxes"]:
            box.set(facecolor="#56B4E9", alpha=0.55)
        for parameter_index, parameter_values in enumerate(values, start=1):
            offsets = [((index % 9) - 4) * 0.012 for index in range(len(parameter_values))]
            axis.scatter(
                [parameter_index + offset for offset in offsets], parameter_values,
                s=12, color="#0072B2", alpha=0.35, edgecolors="none",
            )
        axis.axhline(0.0, color="black", lw=1, ls="--")
        axis.set(title=title, ylabel="joint MAP - truth")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    fig.legend(
        handles=[
            Patch(
                facecolor="#56B4E9", alpha=0.55,
                label="joint MAP error distribution",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none", color="#0072B2",
                alpha=0.5, label="individual joint MAP error",
            ),
            Line2D(
                [0], [0], marker="D", linestyle="none", color="#009E73",
                markeredgecolor="black", label="mean joint MAP error",
            ),
            Line2D([0], [0], color="black", linestyle="--", label="zero error"),
        ],
        loc="center left", bbox_to_anchor=(1.005, 0.5), ncol=1, frameon=True,
    )
    method = f"{method_label}: " if method_label else ""
    fig.suptitle(f"{method}Recovery error distributions", fontsize=14)
    return _finish_figure(fig, output, plt)


def _plot_error_by_observation_count(
    sweep_artifacts, theta_scale, output, plt, trials_per_subject,
):
    """Mean joint-MAP error with between-dataset variability."""
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    counts = tuple(sorted(sweep_artifacts))
    scale = theta_scale.detach().cpu()
    global_scale = scale[:7]
    local_scale = scale[7:]
    global_values = []
    local_values = []
    for count in counts:
        global_errors = []
        local_errors = []
        for artifact in sweep_artifacts[count]:
            global_error = (
                artifact["joint_map"]["globals"] - artifact["truth_globals"]
            ) / global_scale
            local_error = (
                artifact["joint_map"]["locals"] - artifact["truth_locals"]
            ) / local_scale
            global_errors.append(float(global_error.square().mean().sqrt()))
            local_errors.append(float(local_error.square().mean().sqrt()))
        global_values.append(global_errors)
        local_values.append(local_errors)

    global_mean = np.asarray([np.mean(values) for values in global_values])
    global_std = np.asarray([np.std(values) for values in global_values])
    local_mean = np.asarray([np.mean(values) for values in local_values])
    local_std = np.asarray([np.std(values) for values in local_values])
    x = np.asarray(counts)

    fig, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    axis.plot(x, global_mean, marker="o", lw=2, color="#0072B2")
    axis.fill_between(
        x, np.maximum(0.0, global_mean - global_std),
        global_mean + global_std, color="#0072B2", alpha=0.18,
    )
    axis.plot(x, local_mean, marker="s", lw=2, color="#D55E00")
    axis.fill_between(
        x, np.maximum(0.0, local_mean - local_std),
        local_mean + local_std, color="#D55E00", alpha=0.18,
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, [str(count) for count in counts])
    axis.set(
        xlabel=f"subjects used ({trials_per_subject} trials per subject)",
        ylabel="normalized joint MAP RMSE",
        title="Recovery error versus number of subject observations",
    )
    axis.grid(alpha=0.22)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#0072B2", lw=2,
                   label="global mean RMSE"),
            Patch(facecolor="#0072B2", alpha=0.18,
                  label="global +/- 1 SD across datasets"),
            Line2D([0], [0], marker="s", color="#D55E00", lw=2,
                   label="local mean RMSE"),
            Patch(facecolor="#D55E00", alpha=0.18,
                  label="local +/- 1 SD across datasets"),
        ],
        loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=True,
    )
    return _finish_figure(fig, output, plt)


def _plot_mock_data(source, dataset_id, subjects, global_names, local_names, output, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    raw = source["observations_raw"][dataset_id, :subjects].cpu()
    truth_globals = source["globals"][dataset_id].cpu()
    truth_locals = source["locals"][dataset_id, :subjects].cpu()
    subject_ids = list(range(subjects))
    choice_rate = raw[..., 0].mean(1).tolist()
    mean_rt = raw[..., 1].mean(1).tolist()
    std_rt = raw[..., 1].std(1, unbiased=False).tolist()
    censor_rate = raw[..., 2].mean(1).tolist()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    axes[0, 0].plot(subject_ids, choice_rate, marker="o", ms=3, color="#0072B2")
    axes[0, 0].set(title="Choice rate by subject", xlabel="subject", ylabel="P(choice=1)", ylim=(-0.03, 1.03))

    axes[0, 1].errorbar(
        subject_ids, mean_rt, yerr=std_rt, fmt="o", ms=3,
        color="#0072B2", ecolor="#56B4E9", alpha=0.85, capsize=2,
    )
    axes[0, 1].set(title="Reaction time by subject", xlabel="subject", ylabel="RT: mean +/- SD")

    axes[0, 2].plot(subject_ids, censor_rate, marker="o", ms=3, color="#0072B2")
    axes[0, 2].set(title="Censoring rate by subject", xlabel="subject", ylabel="censored fraction", ylim=(-0.03, 1.03))

    choices = raw[..., 0].reshape(-1)
    reaction_times = raw[..., 1].reshape(-1)
    axes[0, 3].hist(
        [
            reaction_times[choices == 0].tolist(),
            reaction_times[choices == 1].tolist(),
        ],
        bins=24, histtype="bar", color=["#56B4E9", "#D55E00"],
        label=["choice 0", "choice 1"], alpha=0.8,
    )
    axes[0, 3].set(title="Trial-level RT distribution", xlabel="reaction time", ylabel="trials")

    for parameter, (local_name, global_index) in enumerate(zip(local_names, range(3))):
        axis = axes[1, parameter]
        values = truth_locals[:, parameter].tolist()
        population_mean = float(truth_globals[global_index])
        axis.plot(subject_ids, values, marker="o", ms=3, color="#009E73")
        axis.axhline(population_mean, color="black", ls="--", lw=1.2)
        axis.set(
            title=f"True local {local_name}", xlabel="subject",
            ylabel="simulator value",
        )

    global_axis = axes[1, 3]
    global_values = truth_globals.tolist()
    positions = list(range(len(global_names)))
    global_axis.barh(positions, global_values, color="#CC79A7", alpha=0.75)
    global_axis.axvline(0.0, color="black", lw=0.8)
    global_axis.set(
        title="True shared global parameters", xlabel="simulator value",
        yticks=positions, yticklabels=global_names,
    )

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#0072B2", label="observed subject summary"),
            Line2D([0], [0], color="#56B4E9", lw=3, label="within-subject RT variability"),
            Line2D([0], [0], marker="o", color="#009E73", label="true local parameter"),
            Line2D([0], [0], color="black", ls="--", label="true population mean"),
            Patch(facecolor="#CC79A7", alpha=0.75, label="true global parameter"),
            Patch(facecolor="#56B4E9", alpha=0.6, label="choice 0 RT"),
            Patch(facecolor="#D55E00", alpha=0.6, label="choice 1 RT"),
        ],
        loc="center left", bbox_to_anchor=(1.005, 0.5), ncol=1, frameon=True,
    )
    fig.suptitle(
        f"Mock hierarchical SDE data and recovery targets (dataset {dataset_id})",
        fontsize=14,
    )
    return _finish_figure(fig, output, plt)


def _make_plots(
    artifacts, figure_directory, global_names, local_names, source=None,
    sweep_artifacts=None, theta_scale=None, trials_per_subject=30,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from Partial_Pooling.io_utils import atomic_json

    if not artifacts:
        raise RuntimeError("Cannot create recovery plots without completed datasets.")
    figure_directory.mkdir(parents=True, exist_ok=True)
    global_rows = [row for artifact in artifacts for row in artifact["global_rows"]]
    local_rows = [row for artifact in artifacts for row in artifact["local_rows"]]
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    })
    outputs = []
    outputs.extend(_plot_parity(
        global_rows, local_rows, global_names, local_names, figure_directory, plt,
    ))
    outputs.extend(_plot_residuals(
        global_rows, local_rows, global_names, local_names, figure_directory, plt,
    ))
    if sweep_artifacts:
        outputs.append(_plot_global_density_grid(
            sweep_artifacts, global_names,
            figure_directory / "global_forest.png", plt,
        ))
    outputs.append(_plot_local_shrinkage(
        global_rows, local_rows, local_names,
        figure_directory / "local_shrinkage.png", plt,
    ))
    outputs.append(_plot_error_distributions(
        global_rows, local_rows, global_names, local_names,
        figure_directory / "error_distributions.png", plt,
    ))
    if sweep_artifacts and theta_scale is not None:
        outputs.append(_plot_error_by_observation_count(
            sweep_artifacts, theta_scale,
            figure_directory / "error_vs_observations.png", plt,
            trials_per_subject,
        ))
    if source is not None:
        dataset_id = int(artifacts[0]["dataset_id"])
        outputs.append(_plot_mock_data(
            source, dataset_id, int(artifacts[0]["subjects"]),
            global_names, local_names, figure_directory / "mock_data_overview.png", plt,
        ))
    estimator = artifacts[0]["joint_map"]["settings"]["estimator"]
    atomic_json(figure_directory / "manifest.json", {
        "inference_method": artifacts[0]["inference_method"],
        "sampler": artifacts[0]["sampler"],
        "correction": artifacts[0]["correction"],
        "point_estimator": estimator,
        "completed_datasets": len(artifacts),
        "observation_unit": "subject",
        "trials_per_subject": trials_per_subject,
        "observation_counts": (
            list(sorted(sweep_artifacts)) if sweep_artifacts else []
        ),
        "figures": [path.name for path in outputs],
    })
    return outputs


def run(args, cpu_limit):
    import torch

    from Partial_Pooling.artifact_names import (
        checkpoint_tag,
        inference_dataset_filename,
        inference_report_filename,
        recovery_run_directory,
        recovery_run_name,
        test_data_path,
    )
    from Partial_Pooling.config import get_config
    from Partial_Pooling.io_utils import atomic_json, atomic_torch
    from Partial_Pooling.model_pipeline import load_model, load_normalizers
    from Partial_Pooling.paths import BenchmarkPaths
    from Partial_Pooling.rng import derive_seed
    from Partial_Pooling.schema import GLOBAL_NAMES, LOCAL_NAMES

    config = get_config(args.preset, args.seed)
    for name in (
        "datasets", "subjects", "draws", "timesteps", "progress_every",
        "gaussian_precision_samples", "gaussian_precision_timesteps",
        "langevin_steps_per_level", "map_starts", "map_timesteps",
        "map_iterations", "map_logprob_timesteps",
    ):
        positive(name, getattr(args, name))
    if args.langevin_snr <= 0:
        raise ValueError("--langevin-snr must be positive.")
    if args.dataset_start < 0:
        raise ValueError("--dataset-start cannot be negative.")
    if args.dataset_start + args.datasets > config.test_datasets:
        raise ValueError(
            f"Requested datasets [{args.dataset_start}, "
            f"{args.dataset_start + args.datasets}), but the {args.preset} "
            f"preset contains {config.test_datasets}."
        )
    if args.subjects > config.subjects:
        raise ValueError(
            f"Requested {args.subjects} subjects, but the {args.preset} "
            f"preset contains {config.subjects}."
        )

    paths = BenchmarkPaths(args.root) if args.root else BenchmarkPaths.default()
    paths.ensure()
    test_path = test_data_path(config, paths)
    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing {test_path}; run create_test_data.py first."
        )
    test = torch.load(test_path, map_location="cpu")
    if test.get("config_signature") != config.signature:
        raise RuntimeError("Test-data configuration does not match the selected preset.")

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device = "cuda:0"

    plan = inference_plan(args.inference_method)
    signature = inference_signature(config.signature, args, plan)
    run_directory = recovery_run_directory(
        paths, config.preset, plan["name"], signature,
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    estimate = (
        estimated_runtime_seconds(
            args.datasets, args.subjects, args.draws, args.timesteps,
        )
        if args.skip_observation_sweep else
        estimated_sweep_runtime_seconds(
            args.datasets, args.subjects, args.draws, args.timesteps,
        )
    )
    run_config_path = run_directory / inference_report_filename(
        plan["name"], "run_config", "json",
    )
    atomic_json(run_config_path, {
        "inference_signature": signature,
        "config_signature": config.signature,
        "preset": config.preset,
        "device": device,
        "dataset_start": args.dataset_start,
        "datasets": args.datasets,
        "subjects": args.subjects,
        "draws": args.draws,
        "timesteps": args.timesteps,
        "inference_method": plan["name"],
        "training_model": "sde_joint",
        "training_samples": config.train_size,
        "checkpoint_tag": checkpoint_tag(config, "sde_joint"),
        "sampler": plan["sampler"],
        "correction": plan["correction"],
        "order": plan["order"],
        "gaussian_precision_samples": args.gaussian_precision_samples,
        "gaussian_precision_timesteps": args.gaussian_precision_timesteps,
        "langevin_steps_per_level": args.langevin_steps_per_level,
        "langevin_snr": args.langevin_snr,
        "estimated_runtime_seconds": estimate,
        "observation_sweep": not args.skip_observation_sweep,
        "observation_counts": (
            [] if args.skip_observation_sweep
            else list(observation_counts(args.subjects))
        ),
        "map_settings": _map_settings(args),
        "cpu_limit": cpu_limit,
    })

    print(
        "Focused partial-pooling recovery\n"
        f"  datasets: {args.datasets} starting at {args.dataset_start}\n"
        f"  subjects/draws/steps: {args.subjects}/{args.draws}/{args.timesteps}\n"
        f"  inference: {plan['label']}\n"
        f"  sampler/correction: {plan['sampler']}/{plan['correction']}\n"
        f"  estimated posterior-sampling time: {format_duration(estimate)}\n"
        f"  output: {run_directory}",
        flush=True,
    )

    normalizers = load_normalizers(config, paths)
    selected_normalizers = {
        "theta": normalizers["sde_joint_theta"],
        "observations": normalizers["sde_observations"],
    }
    model = load_model(config, "sde_joint", paths, device)
    source = test["payload"]["sde"]

    for offset, dataset_id in enumerate(
        range(args.dataset_start, args.dataset_start + args.datasets), start=1
    ):
        output = run_directory / inference_dataset_filename(
            plan["name"], dataset_id,
        )
        artifact = None
        if output.exists() and not args.force:
            existing = torch.load(output, map_location="cpu")
            if existing.get("inference_signature") != signature:
                raise RuntimeError(f"Incompatible existing output: {output}")
            artifact = existing
            print(
                f"[{offset}/{args.datasets}] dataset {dataset_id}: reused posterior",
                flush=True,
            )
        if artifact is None:
            print(f"[{offset}/{args.datasets}] dataset {dataset_id}", flush=True)
            seed = derive_seed(
                config.root_seed, "focused_partial_pooling", dataset_id
            )
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            observations = source["observations"][dataset_id, :args.subjects]
            posterior, solver_stats, runtime_seconds = _sample_dataset(
                model, observations, selected_normalizers,
                args, config, device, torch,
            )
            artifact = {
                "inference_signature": signature,
                "config_signature": config.signature,
                "dataset_id": dataset_id,
                "subject_ids": test["subject_ids"][dataset_id, :args.subjects],
                "seed": seed,
                "subjects": args.subjects,
                "trials_per_subject": config.trials,
                "draws": args.draws,
                "timesteps": args.timesteps,
                "inference_method": plan["name"],
                "training_model": "sde_joint",
                "training_samples": config.train_size,
                "checkpoint_tag": checkpoint_tag(config, "sde_joint"),
                "sampler": plan["sampler"],
                "correction": plan["correction"],
                "order": plan["order"],
                "truth_globals": source["globals"][dataset_id],
                "truth_locals": source["locals"][dataset_id, :args.subjects],
                "posterior": posterior,
                "solver_stats": solver_stats,
                "runtime_seconds": runtime_seconds,
            }
        artifact["trials_per_subject"] = config.trials
        artifact = _complete_artifact_joint_map(
            artifact, model, source, selected_normalizers,
            args, config, device, torch,
        )
        atomic_torch(output, artifact)
        print(
            f"  saved {output.name}; joint MAP candidate "
            f"{artifact['joint_map']['selected_candidate']}",
            flush=True,
        )
        _write_reports(
            run_directory,
            _load_completed(run_directory, signature, plan["name"], torch),
            plan["name"],
        )

    artifacts = _load_completed(
        run_directory, signature, plan["name"], torch,
    )
    sweep_artifacts = None
    if not args.skip_observation_sweep:
        sweep_artifacts = _run_observation_sweep(
            artifacts, run_directory, signature, model, source,
            selected_normalizers, args, config, device, torch,
        )
    summary = _write_reports(run_directory, artifacts, plan["name"])
    figure_directory = (
        paths.figures / "partial_pooling" / config.preset
        / recovery_run_name(plan["name"], signature)
    )
    figures = _make_plots(
        artifacts, figure_directory, GLOBAL_NAMES, LOCAL_NAMES, source=source,
        sweep_artifacts=sweep_artifacts,
        theta_scale=selected_normalizers["theta"].scale,
        trials_per_subject=config.trials,
    )
    print(
        f"Completed {len(artifacts)}/{args.datasets} datasets. "
        f"Summary: {run_directory / inference_report_filename(plan['name'], 'summary', 'json')}\n"
        f"Created {len(figures)} figures in {figure_directory}",
        flush=True,
    )
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    for name in (
        "datasets", "subjects", "draws", "timesteps",
        "gaussian_precision_samples", "gaussian_precision_timesteps",
        "langevin_steps_per_level", "map_starts", "map_timesteps",
        "map_iterations", "map_logprob_timesteps",
    ):
        positive(name, getattr(args, name))
    if args.langevin_snr <= 0:
        raise ValueError("--langevin-snr must be positive.")
    estimate = (
        estimated_runtime_seconds(
            args.datasets, args.subjects, args.draws, args.timesteps,
        )
        if args.skip_observation_sweep else
        estimated_sweep_runtime_seconds(
            args.datasets, args.subjects, args.draws, args.timesteps,
        )
    )
    if args.estimate_only:
        print(
            f"Estimated sampling time: {format_duration(estimate)} "
            f"({estimate:.1f} seconds)"
        )
        return 0

    # This must happen before importing NumPy, SciPy, PyTorch, or COMPASS.
    from runtime import configure_runtime
    cpu_limit = configure_runtime()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run(args, cpu_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
