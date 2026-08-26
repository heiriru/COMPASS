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
DEFAULT_DRAWS = 4096
DEFAULT_TIMESTEPS = 50
DEFAULT_MAP_STARTS = 4
DEFAULT_MAP_TIMESTEPS = 20
DEFAULT_MAP_ITERATIONS = 3
DEFAULT_MAP_LOGPROB_TIMESTEPS = 20
MAP_SETTINGS_VERSION = 5
MAP_ESTIMATORS = ("method_default", "kde_global_then_local_ascent")
DEFAULT_MAP_ESTIMATOR = "method_default"
MAP_RETRY_SIGMA_START = 0.1
MAP_RETRY_TIMESTEPS = 10
MAP_COHERENCE_QUANTILE = 0.95
MAP_MIN_COHERENCE_LIMIT = 8.0
# Bumped whenever the composition semantics change, so cached artifacts from an
# earlier rule are never silently reused. Version 2: per-observation
# positive-definiteness adaptation before composing, the local cross-correction
# reading the clamped composed score, and a local-latent denoise clamp.
COMPOSITION_SEMANTICS_VERSION = 2
MAP_SUPPORT_QUANTILE = 0.01
FIGURE_DPI = 300
GAUSSIAN_CORRECTIONS = frozenset({
    "gauss", "full_gaussian", "Gauss_global_local", "gauss_hierarchical",
})
INFERENCE_METHODS = (
    "dpm2_gaussian", "dpm2_full_gaussian",
    "dpm2_gauss_global_local", "dpm2_gauss_global_local_moment",
    "dpm2_gauss_hierarchical",
    "dpm2_gauss_jacobian_newton",
    "langevin_fnpse",
)


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
    if name == "dpm2_full_gaussian":
        return {
            "name": name,
            "sampler": "dpm",
            "correction": "full_gaussian",
            "order": 2,
            "label": "DPM-Solver-2 + full Gaussian",
        }
    if name == "dpm2_gauss_global_local":
        return {
            "name": name,
            "sampler": "dpm",
            "correction": "Gauss_global_local",
            "order": 2,
            "moment_projection": False,
            "label": "DPM-Solver-2 + global/local Gaussian",
        }
    if name == "dpm2_gauss_global_local_moment":
        return {
            "name": name,
            "sampler": "dpm",
            "correction": "Gauss_global_local",
            "order": 2,
            "moment_projection": True,
            "label": "DPM-Solver-2 + global/local Gaussian moments",
        }
    if name == "dpm2_gauss_hierarchical":
        return {
            "name": name,
            "sampler": "dpm",
            "correction": "gauss_hierarchical",
            "order": 2,
            "label": "DPM-Solver-2 + hierarchical Gaussian (GAUSS)",
        }
    if name == "dpm2_gauss_jacobian_newton":
        return {
            "name": name,
            "sampler": "dpm",
            "correction": "gauss_jacobian",
            "order": 2,
            "map_method": "newton",
            "label": (
                "DPM-Solver-2 + Jacobian-Gaussian (pilot-free) "
                "+ arrow-Newton MAP"
            ),
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


def diffusion_metadata(config):
    return {
        "model_signature": config.model_signature,
        "data_signature": config.data_signature,
        "sde_type": config.sde_type,
        "diffusion": config.diffusion_tag,
        "sigma": config.sigma if config.sde_type == "vesde" else None,
        "beta_min": config.beta_min if config.sde_type == "vpsde" else None,
        "beta_max": config.beta_max if config.sde_type == "vpsde" else None,
    }



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
        "--preset", choices=("smoke", "full", "large", "compact", "small"), default="full",
    )
    result.add_argument("--sde-type", choices=("vesde", "vpsde"), default="vesde")
    result.add_argument("--beta-min", type=float, default=0.1)
    result.add_argument("--beta-max", type=float, default=20.0)
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
            "DPM-Solver-2 with diagonal, full-covariance, or global/local "
            "Gaussian moment composition, or the Langevin + F-NPSE reference."
        ),
    )
    result.add_argument(
        "--gaussian-precision-samples", type=int, default=1024,
        help=(
            "Single-subject draws used once per dataset for Gaussian "
            "precision. This estimate is a covariance in dim(theta) "
            "dimensions, so its eigenvalues spread by roughly "
            "sqrt(dim/samples) (Marchenko-Pastur). At 256 draws and 10 "
            "latent dimensions that is ~20%, enough to make a subject's "
            "estimated information about the shared parameters come out "
            "negative -- outside the model class, and the direct cause of "
            "an indefinite composed precision. Raising it is the cheapest "
            "way to shrink the information projection towards a no-op and "
            "recover unmodified GAUSS."
        ),
    )
    result.add_argument(
        "--gaussian-precision-timesteps", type=int, default=50,
        help="Diffusion steps for the reusable Gaussian precision estimate.",
    )
    result.add_argument(
        "--gaussian-precision-batch-size", type=int, default=128,
        help=(
            "Maximum flattened transformer rows per Gaussian precision batch; "
            "does not change the number of precision samples."
        ),
    )
    result.add_argument(
        "--langevin-steps-per-level", type=int, default=10,
        help="Annealed Langevin updates at every F-NPSE noise level.",
    )
    result.add_argument(
        "--dpm-corrector-steps", type=int, default=0,
        help=(
            "Langevin MCMC corrector steps applied after every DPM-Solver "
            "predictor step (predictor-corrector sampling). 0 disables the "
            "corrector."
        ),
    )
    result.add_argument(
        "--dpm-corrector-interval", type=int, default=1,
        help="Apply the DPM corrector every this many predictor steps.",
    )
    result.add_argument(
        "--dpm-corrector-snr", type=float, default=0.1,
        help="Langevin signal-to-noise ratio for the DPM corrector steps.",
    )
    result.add_argument(
        "--langevin-snr", type=float, default=0.1,
        help="Langevin signal-to-noise ratio for the F-NPSE reference.",
    )
    result.add_argument(
        "--map-estimator", choices=MAP_ESTIMATORS,
        default=DEFAULT_MAP_ESTIMATOR,
        help=(
            "method_default: each method's own mode finder (compositional "
            "hierarchical score ascent, or the global KDE mode for F-NPSE). "
            "kde_global_then_local_ascent: fix the globals at their posterior "
            "KDE mode, then refine only the locals by conditioned score "
            "ascent, identically for every method."
        ),
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
        "composition_semantics": COMPOSITION_SEMANTICS_VERSION,
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
        "moment_projection": bool(plan.get("moment_projection", False)),
        "gaussian_precision_samples": (
            args.gaussian_precision_samples
            if plan["correction"] in GAUSSIAN_CORRECTIONS else None
        ),
        "gaussian_precision_timesteps": (
            args.gaussian_precision_timesteps
            if plan["correction"] in GAUSSIAN_CORRECTIONS else None
        ),
        "langevin_steps_per_level": (
            args.langevin_steps_per_level
            if plan["correction"] == "fnpe" else None
        ),
        "langevin_snr": (
            args.langevin_snr if plan["correction"] == "fnpe" else None
        ),
        "dpm_corrector_steps": (
            args.dpm_corrector_steps if plan["sampler"] == "dpm" else None
        ),
        "dpm_corrector_interval": (
            args.dpm_corrector_interval
            if plan["sampler"] == "dpm" and args.dpm_corrector_steps > 0
            else None
        ),
        "dpm_corrector_snr": (
            args.dpm_corrector_snr
            if plan["sampler"] == "dpm" and args.dpm_corrector_steps > 0
            else None
        ),
    }
    if args.map_estimator != DEFAULT_MAP_ESTIMATOR:
        # Inserted only when a non-default estimator is selected, so that every
        # signature published before this option existed still hashes to the
        # same value and its completed artifacts stay reusable.
        payload["map_estimator"] = args.map_estimator
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


def _normalized_local_prior(normalizer, torch):
    """Marginal local-latent prior in normalized coordinates.

    Bounds the denoised prediction of the subject-level latents exactly as the
    global prior bounds the shared ones. Without it the local coordinates are
    the only unbounded latents in the sampler.
    """
    from Partial_Pooling.schema import GLOBAL_INDICES, LOCAL_NAMES
    from Partial_Pooling.simulators.priors import (
        LOCAL_PRIOR_MEAN, LOCAL_PRIOR_STD,
    )

    start = len(GLOBAL_INDICES)
    indices = torch.arange(start, start + len(LOCAL_NAMES), dtype=torch.long)
    mean = torch.as_tensor(LOCAL_PRIOR_MEAN, dtype=torch.float32)
    std = torch.as_tensor(LOCAL_PRIOR_STD, dtype=torch.float32)
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
    local_prior = _normalized_local_prior(theta_normalizer, torch)
    batch_size = draw_batch_size(args.draws, config.batch_size, args.subjects)
    batch_count = math.ceil(args.draws / batch_size)
    chunks = []
    posterior_precision = None
    posterior_covariance = None
    posterior_mean = None
    global_posterior_mean = None
    global_posterior_covariance = None
    totals = {"score_network_calls": 0, "evaluated_subject_rows": 0}
    started = time.perf_counter()

    if plan["correction"] == "Gauss_global_local":
        latent_width = model.nodes_size - normalized_observations.shape[-1]
        estimator = model.multi_obs_sampler
        estimator.verbose = False
        print(
            "  estimating reusable single-subject posterior means and "
            "covariances ...",
            flush=True,
        )
        posterior_mean, posterior_covariance = (
            estimator.estimate_posterior_moments(
                data=normalized_observations.cpu(),
                condition_mask=condition_mask.cpu(),
                num_samples=args.gaussian_precision_samples,
                timesteps=args.gaussian_precision_timesteps, eps=1e-3,
                batch_size=args.gaussian_precision_batch_size, device=device,
                feature_indices=list(range(latent_width)),
            )
        )

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
            local_prior=local_prior,
            correction=plan["correction"],
            posterior_precision=posterior_precision,
            posterior_covariance=posterior_covariance,
            posterior_mean=posterior_mean,
            precision_est_samples=args.gaussian_precision_samples,
            precision_est_timesteps=args.gaussian_precision_timesteps,
            precision_est_batch_size=args.gaussian_precision_batch_size,
            order=plan["order"] or 2,
            snr=(
                args.langevin_snr if plan["sampler"] == "langevin"
                else args.dpm_corrector_snr
            ),
            corrector_steps_interval=(
                1 if plan["sampler"] == "langevin"
                else args.dpm_corrector_interval
            ),
            corrector_steps=(
                args.langevin_steps_per_level if plan["sampler"] == "langevin"
                else args.dpm_corrector_steps
            ),
            final_corrector_steps=0,
            device=device,
            verbose=False,
            method=plan["sampler"],
            capture_attention=False,
        ).cpu()
        if plan["correction"] == "gauss" and posterior_precision is None:
            posterior_precision = (
                model.multi_obs_sampler.posterior_precision.detach().cpu()
            )
        if (
            plan["correction"]
            in {"full_gaussian", "Gauss_global_local", "gauss_hierarchical"}
            and posterior_covariance is None
        ):
            posterior_covariance = (
                model.multi_obs_sampler.posterior_covariance.detach().cpu()
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
    if not torch.all(torch.isfinite(normalized)):
        raise RuntimeError(
            "Partial-pooling inference returned non-finite posterior draws."
        )
    sync_error = float((shared - shared[:1]).abs().max())
    if sync_error > 1e-5:
        raise RuntimeError(
            "Partial-pooling inference returned unsynchronized globals "
            f"(maximum absolute difference {sync_error:.3e})."
        )
    raw = theta_normalizer.inverse(
        normalized.reshape(-1, normalized.shape[-1])
    ).reshape_as(normalized)
    posterior = {
        "globals": raw[0, :, :len(GLOBAL_INDICES)],
        "locals": raw[:, :, len(GLOBAL_INDICES):].permute(1, 0, 2),
        "posterior_precision": posterior_precision,
        "posterior_covariance": posterior_covariance,
        "posterior_mean": posterior_mean,
    }
    totals["score_evaluations"] = totals["score_network_calls"]
    return posterior, totals, time.perf_counter() - started


def _map_settings(args):
    plan = inference_plan(args.inference_method)
    from Partial_Pooling.schema import LOCAL_NAMES

    if args.map_estimator == "kde_global_then_local_ascent":
        # Deliberately method-independent apart from the draws it consumes:
        # no composition, no correction, no candidate log-probability ranking.
        return {
            "version": MAP_SETTINGS_VERSION,
            "map_estimator": args.map_estimator,
            "estimator": "kde_global_then_local_score_ascent",
            "coherence_limit": _prior_coherence_limit(
                args.subjects, len(LOCAL_NAMES),
            ),
            "global_stage": "standardized_gaussian_kde_mode_of_draws",
            "local_stage": "conditioned_annealed_tweedie_score_ascent",
            "timesteps": int(args.map_timesteps),
            "iterations_per_level": int(args.map_iterations),
            "sampling_correction": plan["correction"],
            "optimizer_correction": None,
            "map_method": "tweedie",
        }
    return {
        "version": MAP_SETTINGS_VERSION,
        "coherence_limit": _prior_coherence_limit(
            args.subjects, len(LOCAL_NAMES),
        ),
        "starts": int(args.map_starts),
        "timesteps": int(args.map_timesteps),
        "iterations_per_level": int(args.map_iterations),
        "logprob_timesteps": int(args.map_logprob_timesteps),
        "sampling_correction": plan["correction"],
        "optimizer_correction": (
            plan["correction"]
            if plan["correction"] != "fnpe" else None
        ),
        "map_method": plan.get("map_method", "tweedie"),
        "estimator": (
            "validated_hierarchical_score_map"
            if plan["correction"] != "fnpe"
            else "posterior_global_kde_mode"
        ),
        "retry_sigma_start": (
            MAP_RETRY_SIGMA_START
            if plan["correction"] != "fnpe" else None
        ),
        "retry_timesteps": (
            min(MAP_RETRY_TIMESTEPS, int(args.map_timesteps))
            if plan["correction"] != "fnpe" else None
        ),
    }


def _global_kde_state(global_draws, torch):
    """Fit the standardized global KDE used for starts and support checks."""
    from numpy.linalg import LinAlgError
    from scipy.stats import gaussian_kde

    values = torch.as_tensor(global_draws, dtype=torch.float64)
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardized = ((values - mean) / scale).numpy()
    try:
        kde = gaussian_kde(standardized.T)
        scores = torch.from_numpy(kde.logpdf(standardized.T)).float()
    except (ValueError, RuntimeError, LinAlgError):
        kde = None
        scores = -torch.from_numpy((standardized ** 2).sum(axis=1)).float()
    return {"mean": mean, "scale": scale, "kde": kde, "scores": scores}


def _global_kde_log_density(state, values, torch):
    """Evaluate raw global coordinates under a fitted standardized KDE."""
    points = torch.as_tensor(values, dtype=torch.float64)
    if points.dim() == 1:
        points = points.unsqueeze(0)
    standardized = ((points - state["mean"]) / state["scale"]).numpy()
    if state["kde"] is None:
        result = -(standardized ** 2).sum(axis=1)
    else:
        result = state["kde"].logpdf(standardized.T)
    return torch.from_numpy(result).float()


def _prior_coherence_limit(subjects, parameters):
    """Prior-implied bound on max |z| across every local coordinate.

    Under the model l_ik = mu_k + exp(log_sigma_k) * z_ik with z_ik ~ N(0, 1),
    so the coherence statistic is the maximum of `subjects * parameters`
    standard normals and its MAP_COHERENCE_QUANTILE point follows from a Sidak
    correction. The limit therefore depends only on the model.

    The previous limit was a quantile of the very draws it validated, which
    made it vacuous exactly when it mattered: a posterior that ran away to
    |z| ~ 1e11 raised its own threshold to 1e11 and was certified coherent.
    """
    from scipy.stats import norm

    count = max(int(subjects) * int(parameters), 1)
    tail = 1.0 - MAP_COHERENCE_QUANTILE ** (1.0 / count)
    return max(
        MAP_MIN_COHERENCE_LIMIT, float(norm.ppf(1.0 - 0.5 * tail)),
    )


def _hierarchy_validation_state(posterior, density, torch):
    """Derive the prior-implied local/global coherence limit."""
    global_draws = posterior["globals"].float()
    local_draws = posterior["locals"].float()
    scales = global_draws[:, None, 3:6].exp().clamp_min(1e-8)
    standardized = (
        (local_draws - global_draws[:, None, :3]) / scales
    ).abs()
    max_abs_z = standardized.amax(dim=(1, 2))
    coherence_limit = _prior_coherence_limit(
        local_draws.shape[1], local_draws.shape[2],
    )
    return {
        "coherence_limit": coherence_limit,
        "sample_max_abs_z": max_abs_z,
        "support_limit": float(torch.quantile(
            density["scores"], MAP_SUPPORT_QUANTILE,
        )),
    }


def _joint_candidate_diagnostics(
    raw_candidate, density, validation, torch,
):
    """Measure posterior support and exact hierarchy coherence for one candidate."""
    from Partial_Pooling.simulators.priors import (
        GLOBAL_PRIOR_MEAN, GLOBAL_PRIOR_STD,
    )

    raw_candidate = torch.as_tensor(raw_candidate, dtype=torch.float32)
    globals_ = raw_candidate[0, :7]
    locals_ = raw_candidate[:, 7:10]
    scales = globals_[3:6].exp().clamp_min(1e-8)
    max_abs_z = float(
        ((locals_ - globals_[:3]) / scales).abs().max()
    )
    global_log_density = float(
        _global_kde_log_density(density, globals_, torch)[0]
    )
    prior_mean = torch.as_tensor(GLOBAL_PRIOR_MEAN, dtype=torch.float32)
    prior_std = torch.as_tensor(GLOBAL_PRIOR_STD, dtype=torch.float32)
    lower = prior_mean - 5.0 * prior_std
    upper = prior_mean + 5.0 * prior_std
    boundary_distance = torch.minimum(
        (globals_ - lower).abs(), (globals_ - upper).abs(),
    )
    return {
        "max_abs_z": max_abs_z,
        "coherence_limit": float(validation["coherence_limit"]),
        "coherent": max_abs_z <= float(validation["coherence_limit"]),
        "global_log_density": global_log_density,
        "global_support_limit": float(validation["support_limit"]),
        "global_supported": global_log_density >= float(validation["support_limit"]),
        "at_denoise_boundary": bool(torch.any(boundary_distance <= 1e-4)),
    }


def _is_numerical_map_failure(error):
    """Return whether one MAP candidate failed for a recoverable numeric reason."""
    message = str(error).lower()
    return any(fragment in message for fragment in (
        "non-finite", "non-positive", "singular",
    ))


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


def _kde_global_then_local_ascent_map(
    model, observations, posterior, normalizers, args, config, device, torch,
):
    """Fix the globals at their posterior KDE mode, then ascend the locals.

    A two-stage estimator that is identical for every inference method, so a
    comparison isolates the posterior sampler rather than the mode finder:

    1. The seven shared coordinates are set to the joint KDE mode of the global
       posterior draws, exactly as ``_fnpse_kde_map`` already selects F-NPSE's
       global mode (standardized 7-dimensional Gaussian KDE, evaluated at the
       draws, arg-max retained).
    2. Those globals are then *conditioned* -- moved from the latent block into
       the condition mask -- and only the three local coordinates per subject
       are refined by annealed Tweedie score ascent.

    Stage 2 needs no score composition and therefore no correction argument at
    all. Once the shared coordinates are fixed, the subjects are conditionally
    independent: subject r's locals are the mode of p(l_r | x_r, globals), a
    single-observation problem that the network scores directly. That is what
    makes this estimator applicable to F-NPSE, whose bridging scores do not
    define a compositional MAP objective and which
    ``MultiObsSampler.map_estimate`` rejects outright.

    The hierarchical ascent path cannot express this: it requires every
    hierarchy coordinate to be latent, so conditioning the globals is only
    possible through the per-row ``PFODE.map_estimate``. One consequence is
    recorded rather than hidden -- the per-row ascent applies no local prior
    clamp, so this function reports prior-box violations in its diagnostics
    instead of bounding the denoised prediction the way
    ``hierarchical_map_estimate`` does.
    """
    from Partial_Pooling.schema import GLOBAL_INDICES

    started = time.perf_counter()
    global_count = len(GLOBAL_INDICES)
    global_draws = posterior["globals"].float()
    local_draws = posterior["locals"].float()
    subjects = int(observations.shape[0])
    theta_normalizer = normalizers["theta"]
    normalized_observations = normalizers["observations"].transform(observations)

    # Stage 1: joint KDE mode of the global draws.
    density = _global_kde_state(global_draws, torch)
    selected = int(torch.argmax(density["scores"]))
    globals_hat = global_draws[selected]

    # Stage 2: condition the globals, ascend the locals per subject.
    local_init = local_draws.median(dim=0).values
    raw_theta = torch.cat((
        globals_hat.unsqueeze(0).expand(subjects, -1), local_init,
    ), dim=1)
    latent_width = int(raw_theta.shape[1])
    normalized_theta = theta_normalizer.transform(raw_theta)
    rows = torch.cat((normalized_theta, normalized_observations), dim=1)
    condition_mask = torch.cat((
        torch.ones(global_count),
        torch.zeros(latent_width - global_count),
        torch.ones(normalized_observations.shape[-1]),
    ))

    # Anneal from the posterior spread of the local draws, in the normalized
    # coordinates the ascent actually moves in.
    normalized_draws = theta_normalizer.transform(torch.cat((
        global_draws[:, None, :].expand(-1, subjects, -1),
        local_draws,
    ), dim=2).reshape(-1, latent_width)).reshape(-1, subjects, latent_width)
    sigma_start = float(
        normalized_draws[:, :, global_count:]
        .std(dim=0, unbiased=False).max().clamp_min(1e-3)
    )

    print(
        f"  KDE global mode (draw {selected}) fixed; ascending "
        f"{latent_width - global_count} local coordinates for {subjects} "
        f"subjects ({args.map_timesteps} levels, sigma_start={sigma_start:.3g})",
        flush=True,
    )
    refined = model.map_estimate(
        data=rows, condition_mask=condition_mask, init=rows,
        sigma_start=sigma_start, timesteps=args.map_timesteps,
        iterations_per_level=args.map_iterations, device=device,
    )
    raw_map = theta_normalizer.inverse(refined[:, :latent_width].cpu())

    # The globals were conditioned, so they must come back bit-for-bit.
    clamp_error = float((raw_map[:, :global_count] - globals_hat).abs().max())
    if clamp_error > 1e-3:
        raise RuntimeError(
            "Conditioned globals drifted during local ascent "
            f"(error {clamp_error:.3e})."
        )
    if not bool(torch.isfinite(raw_map).all()):
        raise RuntimeError("Local score ascent produced non-finite estimates.")

    validation = _hierarchy_validation_state(posterior, density, torch)
    diagnostics = _joint_candidate_diagnostics(
        torch.cat((
            globals_hat.unsqueeze(0).expand(subjects, -1),
            raw_map[:, global_count:],
        ), dim=1),
        density, validation, torch,
    )
    return {
        "globals": globals_hat,
        "locals": raw_map[:, global_count:],
        "selected_candidate": selected,
        "candidate_scores": density["scores"],
        "settings": _map_settings(args),
        "runtime_seconds": time.perf_counter() - started,
        "shared_synchronization_max_abs": 0.0,
        "estimator": "kde_global_then_local_score_ascent",
        "selection_phase": "kde_global_mode_then_local_ascent",
        "global_clamp_max_abs": clamp_error,
        "local_ascent_sigma_start": sigma_start,
        "validation": diagnostics,
    }


def _joint_map_dataset(
    model, observations, posterior, normalizers, args, config, device, torch,
):
    """Estimate and validate a compatible global/local mode."""
    from compass.ModelTransfuser import ModelTransfuser
    from Partial_Pooling.schema import GLOBAL_INDICES

    plan = inference_plan(args.inference_method)
    if args.map_estimator == "kde_global_then_local_ascent":
        return _kde_global_then_local_ascent_map(
            model, observations, posterior, normalizers, args, config,
            device, torch,
        )
    if plan["correction"] == "fnpe":
        return _fnpse_kde_map(posterior, args, torch)

    gaussian_state = {}
    if plan["correction"] == "gauss":
        gaussian_state["posterior_precision"] = posterior.get(
            "posterior_precision"
        )
    elif plan["correction"] in {
        "full_gaussian", "Gauss_global_local", "gauss_hierarchical",
    }:
        gaussian_state["posterior_covariance"] = posterior.get(
            "posterior_covariance"
        )
        if plan.get("moment_projection", False):
            gaussian_state["posterior_mean"] = posterior.get("posterior_mean")
    # "gauss_jacobian" is pilot-free by construction (its covariance comes from
    # the network's own Jacobian at each step, not a saved pilot estimate), so
    # it never populates -- and must never require -- gaussian_state.
    if plan["correction"] != "gauss_jacobian" and (
        not gaussian_state
        or any(value is None for value in gaussian_state.values())
    ):
        raise RuntimeError(
            "Gaussian MAP requires the matching estimate saved during "
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
    local_prior = _normalized_local_prior(theta_normalizer, torch)
    density = _global_kde_state(global_draws, torch)
    validation = _hierarchy_validation_state(posterior, density, torch)

    def posterior_fallback(phase):
        coherent_draws = torch.where(
            validation["sample_max_abs_z"]
            <= float(validation["coherence_limit"])
        )[0]
        if coherent_draws.numel() == 0:
            coherent_draws = torch.arange(draws)
        selected_draw = int(coherent_draws[
            torch.argmax(density["scores"][coherent_draws])
        ])
        fallback_map = latent_raw[:, selected_draw, :]
        fallback_diagnostics = _joint_candidate_diagnostics(
            fallback_map, density, validation, torch,
        )
        return (
            fallback_map, fallback_diagnostics, density["scores"],
            selected_draw, phase,
        )

    def refine_candidate(start, sigma_start, timesteps, label):
        try:
            if plan.get("map_method") == "newton":
                # Pilot-free arrow-Newton ascent: curvature="jacobian" builds
                # H_j = -grad(s_j) by forward-mode AD directly from the
                # network, matching correction="gauss_jacobian"'s own
                # pilot-free covariance -- both come from the same Jacobian,
                # so this needs no gaussian_state at all (see
                # MultiObsSampler.newton_map_estimate's docstring).
                refined = model.multi_obs_sampler.newton_map_estimate(
                    data=start,
                    condition_mask=condition_mask,
                    init=start,
                    hierarchy=hierarchy,
                    prior=prior,
                    local_prior=local_prior,
                    correction=plan["correction"],
                    curvature="jacobian",
                    sigma_start=sigma_start,
                    timesteps=timesteps,
                    device=device,
                )
            else:
                refined = model.hierarchical_map_estimate(
                    data=start,
                    condition_mask=condition_mask,
                    init=start,
                    hierarchy=hierarchy,
                    prior=prior,
                    local_prior=local_prior,
                    correction=plan["correction"],
                    **gaussian_state,
                    sigma_start=sigma_start,
                    timesteps=timesteps,
                    iterations_per_level=args.map_iterations,
                    max_iterations_per_level=args.map_iterations,
                    device=device,
                )
        except RuntimeError as error:
            if not _is_numerical_map_failure(error):
                raise
            print(f"  {label} skipped after numerical failure: {error}", flush=True)
            return None
        return refined[:, 0, :]

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
        candidate = refine_candidate(
            start, sigma_start, args.map_timesteps,
            f"MAP start {start_index + 1}",
        )
        if candidate is not None:
            candidates.append(candidate)
    if candidates:
        candidate_tensor = torch.stack(candidates, dim=1)
        scores = ModelTransfuser._hierarchical_candidate_scores(
            model, candidate_tensor, condition_mask, hierarchy, prior,
            args.map_logprob_timesteps, 1e-3, device, False,
        ).cpu()
        raw_candidates = theta_normalizer.inverse(
            candidate_tensor[:, :, :latent_width].reshape(-1, latent_width).cpu()
        ).reshape(subjects, candidate_tensor.shape[1], latent_width)
        selected = int(torch.argmax(scores))
        raw_map = raw_candidates[:, selected]
        diagnostics = _joint_candidate_diagnostics(
            raw_map, density, validation, torch,
        )
        phase = "initial_score_ascent"
    else:
        raw_map, diagnostics, scores, selected, phase = posterior_fallback(
            "posterior_draw_fallback_after_numerical_failure",
        )

    if candidates and (
        not diagnostics["coherent"] or diagnostics["at_denoise_boundary"]
    ):
        count = min(int(args.map_starts), draws)
        source_indices = torch.argsort(
            density["scores"], descending=True,
        )[:count]
        retry_latent = latent_normalized[:, source_indices, :]
        retry_starts = torch.cat((
            retry_latent,
            normalized_observations[:, None, :].expand(-1, count, -1),
        ), dim=2)
        print(
            "  selected MAP failed hierarchy/boundary validation; "
            f"retrying {count} posterior-supported starts at "
            f"sigma={MAP_RETRY_SIGMA_START:g}",
            flush=True,
        )
        retry_candidates = []
        for retry_index in range(count):
            retry_start = retry_starts[:, retry_index, :]
            candidate = refine_candidate(
                retry_start, MAP_RETRY_SIGMA_START,
                min(MAP_RETRY_TIMESTEPS, int(args.map_timesteps)),
                f"low-noise MAP retry {retry_index + 1}",
            )
            if candidate is not None:
                retry_candidates.append(candidate)
        valid = []
        if retry_candidates:
            candidate_tensor = torch.stack(retry_candidates, dim=1)
            scores = ModelTransfuser._hierarchical_candidate_scores(
                model, candidate_tensor, condition_mask, hierarchy, prior,
                args.map_logprob_timesteps, 1e-3, device, False,
            ).cpu()
            raw_candidates = theta_normalizer.inverse(
                candidate_tensor[:, :, :latent_width]
                .reshape(-1, latent_width).cpu()
            ).reshape(subjects, len(retry_candidates), latent_width)
            retry_diagnostics = [
                _joint_candidate_diagnostics(
                    raw_candidates[:, index], density, validation, torch,
                )
                for index in range(len(retry_candidates))
            ]
            valid = [
                index for index, item in enumerate(retry_diagnostics)
                if item["coherent"] and item["global_supported"]
                and not item["at_denoise_boundary"]
            ]
        if valid:
            selected = max(valid, key=lambda index: float(scores[index]))
            raw_map = raw_candidates[:, selected]
            diagnostics = retry_diagnostics[selected]
            phase = "posterior_supported_low_noise_retry"
        else:
            raw_map, diagnostics, scores, selected, phase = posterior_fallback(
                "posterior_draw_fallback",
            )

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
        "estimator": "validated_hierarchical_score_map",
        "selection_phase": phase,
        "validation": diagnostics,
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
                    **diffusion_metadata(config),
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
        **diffusion_metadata(config),
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


def _safe_1d_kde(values, grid):
    """Evaluate a tail-robust marginal KDE, or return ``None`` if degenerate.

    ``gaussian_kde`` bases its bandwidth on the sample standard deviation. A
    handful of finite solver excursions can therefore flatten the scientifically
    relevant central density just as thoroughly as using the excursions to set
    the plotting limits. Retain every draw in the KDE, but base its bandwidth on
    the smaller of the standard deviation and the normal-equivalent IQR.
    """
    import numpy as np
    from numpy.linalg import LinAlgError
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    grid = np.asarray(grid, dtype=np.float64)
    if values.size < 2 or not np.all(np.isfinite(values)):
        return None
    lower = float(np.min(values))
    upper = float(np.max(values))
    if upper <= lower:
        return None
    standard_scale = float(np.std(values))
    if not np.isfinite(standard_scale) or standard_scale <= 0:
        return None
    center = float(np.median(values))
    q25, q75 = np.quantile(values, (0.25, 0.75))
    robust_scale = float((q75 - q25) / 1.3489795003921634)
    scale = (
        robust_scale
        if np.isfinite(robust_scale) and robust_scale > 0
        else standard_scale
    )
    standardized = (values - center) / scale
    standardized_scale = float(np.std(standardized))
    bandwidth_scale = min(1.0, 1.0 / standardized_scale)
    try:
        reference = gaussian_kde(standardized)
        bandwidth = max(
            reference.scotts_factor() * bandwidth_scale,
            np.finfo(np.float64).tiny ** 0.25,
        )
        density = gaussian_kde(
            standardized, bw_method=bandwidth,
        )((grid - center) / scale) / scale
    except (ValueError, RuntimeError, LinAlgError):
        return None
    if not np.all(np.isfinite(density)):
        return None
    return density


def _robust_density_limits(value_sets, anchors=(), width=4.0):
    """Return shared density limits resistant to a minority of solver outliers."""
    import numpy as np

    bounds = [float(value) for value in anchors if np.isfinite(value)]
    for values in value_sets:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        median = float(np.median(values))
        mad_scale = float(
            1.482602218505602 * np.median(np.abs(values - median))
        )
        q25, q75 = np.quantile(values, (0.25, 0.75))
        iqr_scale = float((q75 - q25) / 1.3489795003921634)
        scale = max(mad_scale, iqr_scale)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.std(values))
        if not np.isfinite(scale) or scale <= 0:
            bounds.append(median)
        else:
            bounds.extend((median - width * scale, median + width * scale))
    if not bounds:
        return -0.1, 0.1
    lower, upper = min(bounds), max(bounds)
    span = upper - lower
    padding = 0.06 * span if span > 0 else max(abs(lower) * 0.06, 0.1)
    return lower - padding, upper + padding


def _plot_global_density_grid(
    sweep_artifacts, global_names, output, plt, method_label=None,
):
    """Posterior marginal densities for dataset 0 across four subject counts."""
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

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
        value_sets = [
            artifacts[count]["posterior"]["globals"][:, parameter].numpy()
            for count in counts
        ]
        all_maps = [
            float(artifacts[count]["joint_map"]["globals"][parameter])
            for count in counts
        ]
        all_medians = [float(np.median(values)) for values in value_sets]
        lower, upper = _robust_density_limits(
            value_sets, anchors=(*all_maps, *all_medians, truth),
        )
        grid = np.linspace(lower, upper, 300)
        for column, count in enumerate(counts):
            axis = axes[parameter, column]
            values = artifacts[count]["posterior"]["globals"][:, parameter].numpy()
            density = _safe_1d_kde(values, grid)
            if density is not None:
                axis.fill_between(grid, density, color="#56B4E9", alpha=0.30)
                axis.plot(grid, density, color="#0072B2", lw=1.5)
            else:
                axis.axvline(float(values[0]), color="#0072B2", lw=1.5)
            axis.axvline(truth, color="#D55E00", lw=1.4, ls=":")
            axis.axvline(
                float(artifacts[count]["joint_map"]["globals"][parameter]),
                color="#009E73", lw=1.3, ls="-.",
            )
            axis.axvline(
                float(np.median(values)), color="#CC79A7", lw=1.3, ls="--",
            )
            outside = int(np.count_nonzero((values < lower) | (values > upper)))
            if outside:
                axis.text(
                    0.98, 0.94, f"{outside}/{values.size} draws outside view",
                    transform=axis.transAxes, ha="right", va="top", fontsize=7,
                    color="#666666",
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
            Line2D([0], [0], color="#CC79A7", lw=1.3, ls="--",
                   label="posterior median"),
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

    config = get_config(
        args.preset, args.seed, args.sde_type,
        args.beta_min, args.beta_max,
    )
    for name in (
        "datasets", "subjects", "draws", "timesteps", "progress_every",
        "gaussian_precision_samples", "gaussian_precision_timesteps",
        "gaussian_precision_batch_size",
        "langevin_steps_per_level", "map_starts", "map_timesteps",
        "map_iterations", "map_logprob_timesteps", "dpm_corrector_interval",
    ):
        positive(name, getattr(args, name))
    if args.langevin_snr <= 0:
        raise ValueError("--langevin-snr must be positive.")
    if args.dpm_corrector_steps < 0:
        raise ValueError("--dpm-corrector-steps cannot be negative.")
    if args.dpm_corrector_snr <= 0:
        raise ValueError("--dpm-corrector-snr must be positive.")
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
    test_signature = test.get("data_signature", test.get("config_signature"))
    if test_signature not in config.compatible_data_signatures:
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
        **diffusion_metadata(config),
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
        "moment_projection": bool(plan.get("moment_projection", False)),
        "gaussian_precision_samples": args.gaussian_precision_samples,
        "gaussian_precision_timesteps": args.gaussian_precision_timesteps,
        "gaussian_precision_batch_size": args.gaussian_precision_batch_size,
        "langevin_steps_per_level": args.langevin_steps_per_level,
        "langevin_snr": args.langevin_snr,
        "dpm_corrector_steps": args.dpm_corrector_steps,
        "dpm_corrector_interval": args.dpm_corrector_interval,
        "dpm_corrector_snr": args.dpm_corrector_snr,
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
        f"  diffusion: {config.diffusion_tag}\n"
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
                **diffusion_metadata(config),
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
            # Posterior sampling is the expensive phase.  Persist it before MAP
            # refinement so a numerical MAP failure can resume without resampling.
            atomic_torch(output, artifact)
            print(f"  checkpointed posterior in {output.name}", flush=True)
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
        "gaussian_precision_batch_size",
        "langevin_steps_per_level", "map_starts", "map_timesteps",
        "map_iterations", "map_logprob_timesteps", "dpm_corrector_interval",
    ):
        positive(name, getattr(args, name))
    if args.langevin_snr <= 0:
        raise ValueError("--langevin-snr must be positive.")
    if args.dpm_corrector_steps < 0:
        raise ValueError("--dpm-corrector-steps cannot be negative.")
    if args.dpm_corrector_snr <= 0:
        raise ValueError("--dpm-corrector-snr must be positive.")
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
