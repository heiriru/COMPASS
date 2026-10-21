#!/usr/bin/env python3
"""Compositional inference for the shared shoulder angle, three rules.

One trained single-observation network, one tall dataset, three ways of
composing the per-observation scores into a shared posterior:

    langevin_fnpse       annealed Langevin on Geffner et al.'s bridging
                         densities; the cheapest rule, and the only one whose
                         target is deliberately *not* the diffusion marginals
    gauss_hierarchical   GAUSS (Linhart et al.) generalised to a global/local
                         hierarchy by arrow elimination, weighted by a pilot
                         covariance estimated from the network's own draws
    gauss_jacobian       the same composition with the per-observation curvature
                         taken from the network's Jacobian by Tweedie's
                         second-order identity -- no pilot run at all

Both Gaussian rules run with the churned predictor (``--churn-eta``, default 2),
which on every benchmark in this repository is worth more than the Langevin
correctors it replaces and costs no extra score evaluations.

Everything is graded against :func:`hierarchical.reference_global`, an ABC
reference computed by Monte Carlo through the forward model with no network in
the loop -- so it can referee the rules rather than restate one of them.

Usage:
    python inverse_kinematics/run_inference.py
    python inverse_kinematics/run_inference.py --observations 16 --draws 4000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

CPU_INFO = configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import churn  # noqa: E402
import hierarchical as H  # noqa: E402
import simulator as sim  # noqa: E402
import train_model as trainer  # noqa: E402

ROOT = Path(__file__).resolve().parent

# Shared by every rule, so a comparison isolates the composition.
COMMON = {"timesteps": 100}

METHODS = {
    "langevin_fnpse": {
        "label": "Langevin + F-NPSE",
        "colour": "#2A9D8F",
        "churn": False,
        "note": "bridging densities; 10 Langevin updates per level",
        "kwargs": {
            "method": "langevin", "correction": "fnpe",
            "corrector_steps": 10, "snr": 0.1,
        },
    },
    "langevin_fnpse_arruda_damped": {
        "label": "Langevin + F-NPSE + Arruda damping",
        "colour": "#D28A00",
        "churn": False,
        "note": (
            "Arruda error-damped F-NPSE; d(0)=1 and d(1)=1/N; "
            "all observations are evaluated"
        ),
        "kwargs": {
            "method": "langevin", "correction": "damping",
            "corrector_steps": 10, "snr": 0.1,
        },
    },
    "gauss_hierarchical": {
        "label": "DPM2 + gauss_hierarchical + churn",
        "colour": "#E76F51",
        "churn": True,
        "note": "pilot covariance from the network's own draws",
        "kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_hierarchical",
            "corrector_steps": 0, "final_corrector_steps": 0,
            "terminal_corrector_steps": 0,
        },
    },
    "gauss_jacobian": {
        "label": "DPM2 + gauss_jacobian + churn",
        # Indigo, not the near-black #264653 it used to be: the reference curve
        # is drawn in INK (#22262B) and the two were indistinguishable wherever
        # they overlapped, which is exactly where the comparison matters.
        "colour": "#5E60CE",
        "churn": True,
        "note": "pilot-free; curvature from the network Jacobian",
        "kwargs": {
            "method": "dpm", "order": 2, "correction": "gauss_jacobian",
            "corrector_steps": 0, "final_corrector_steps": 0,
            "terminal_corrector_steps": 0,
        },
    },
}


def draw(model, data, method, arguments, device):
    """Run one composition rule and return its draws plus the sampler."""
    from compass.MultiObsSampler import MultiObsSampler

    entry = METHODS[method]
    kwargs = dict(entry["kwargs"])
    if method == "langevin_fnpse_arruda_damped":
        # Arruda et al. leaves the data endpoint exact (d(0)=1) and damps
        # accumulated score error at the noise endpoint by 1/N.  Do not
        # mini-batch: row-specific local latents require every score.
        n_observations = int(torch.as_tensor(data).shape[0])
        endpoint = getattr(arguments, "arruda_damping_at_noise", None)
        kwargs.update(
            damping_at_data=1.0,
            damping_at_noise=(1.0 / n_observations if endpoint is None else endpoint),
            damping_schedule=getattr(arguments, "arruda_damping_schedule", "exponential"),
        )
    if entry["kwargs"]["correction"] == "gauss_hierarchical":
        kwargs.update(
            precision_est_samples=arguments.pilot_draws,
            precision_est_timesteps=COMMON["timesteps"],
        )

    sampler = MultiObsSampler(model)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)

    eta = arguments.churn_eta if entry["churn"] else 0.0
    refresh = arguments.jacobian_refresh if method == "gauss_jacobian" else None

    started = time.perf_counter()
    with churn.patched_dpm_sampler(eta), churn.forced_jacobian_refresh(refresh):
        samples = sampler.sample(
            world_size=1, data=data,
            condition_mask=H.CONDITION_MASK,
            num_samples=arguments.draws,
            hierarchy=list(H.HIERARCHY),
            prior=(torch.as_tensor(H.GLOBAL_PRIOR_MEAN, dtype=torch.float32),
                   torch.as_tensor(H.GLOBAL_PRIOR_STD, dtype=torch.float32)),
            local_prior=(torch.as_tensor(H.LOCAL_PRIOR_MEAN, dtype=torch.float32),
                         torch.as_tensor(H.LOCAL_PRIOR_STD, dtype=torch.float32)),
            denoise_clamp=arguments.denoise_clamp,
            composition_temperature=getattr(
                arguments, "composition_temperature", 1.0
            ),
            device=device, verbose=False, **COMMON, **kwargs,
        )
    runtime = time.perf_counter() - started

    samples = samples.detach().cpu().numpy()
    # The shared coordinate must be identical across observation rows; if it is
    # not, the composition never actually shared it and every number below is
    # about something else.
    spread = float(np.abs(samples[:, :, H.HIERARCHY]
                          - samples[:1, :, H.HIERARCHY]).max())
    if spread > 1e-4:
        raise AssertionError(
            f"{method}: shared draws are not synchronized across observations "
            f"({spread:.3e})."
        )
    return samples, sampler, runtime


def wasserstein_to_grid(draws, grid, density):
    """W1 between 1-D draws and a density tabulated on ``grid``."""
    draws = np.sort(np.asarray(draws, dtype=np.float64))
    cumulative = np.cumsum(density)
    cumulative /= cumulative[-1]
    quantiles = (np.arange(len(draws)) + 0.5) / len(draws)
    positions = np.clip(np.searchsorted(cumulative, quantiles), 0, len(grid) - 1)
    return float(np.abs(draws - grid[positions]).mean())


def evaluate(shared_draws, grid, density, truth, sampler, runtime):
    reference_mean, reference_std, reference_mode = H.grid_moments(grid, density)
    draws = np.asarray(shared_draws, dtype=np.float64)
    return {
        "w1_over_sigma": wasserstein_to_grid(draws, grid, density) / reference_std,
        "mean_error_sigma": (draws.mean() - reference_mean) / reference_std,
        "width_ratio": draws.std() / reference_std,
        "truth_error_sigma": (draws.mean() - truth) / reference_std,
        "posterior_mean": float(draws.mean()),
        "posterior_std": float(draws.std()),
        "reference_mean": reference_mean,
        "reference_std": reference_std,
        "reference_mode": reference_mode,
        "seconds": runtime,
        "network_calls": int(getattr(sampler, "network_calls", 0) or 0),
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", choices=sorted(trainer.CONFIGS), default="h16d2")
    result.add_argument("--arch", choices=("tokenized", "legacy"), default="tokenized")
    result.add_argument("--simulations", type=int, default=200_000)
    result.add_argument("--observations", type=int, default=8,
                        help="Observations sharing one shoulder angle.")
    result.add_argument("--draws", type=int, default=2_000)
    result.add_argument("--churn-eta", type=float, default=churn.DEFAULT_ETA)
    result.add_argument("--composition-temperature", type=float, default=1.0)
    result.add_argument("--jacobian-refresh", type=int, default=5)
    result.add_argument("--pilot-draws", type=int, default=2_000)
    result.add_argument("--denoise-clamp", type=float, default=5.0)
    result.add_argument("--reference-draws", type=int, default=200_000)
    result.add_argument("--reference-bandwidth", type=float, default=0.05)
    result.add_argument("--seed", type=int, default=3)
    result.add_argument("--methods", nargs="+", default=list(METHODS))
    result.add_argument("--menu", action="store_true",
                        help="Load the checkpoint trained with the "
                             "posterior-concentrated ConditionalMenu.")
    result.add_argument("--device", default="cuda")
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    print(
        f"CPU affinity: {CPU_INFO['active_cpus']}/{CPU_INFO['host_cpus']} CPUs "
        f"({CPU_INFO['percent']:.3f}%)"
    )

    directory = trainer.model_directory(
        arguments.artifacts, arguments.config, arguments.arch,
        arguments.simulations, menu=arguments.menu,
    )
    checkpoint = directory / "Model_checkpoint.pt"
    if not checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint}. Train one first:\n"
            f"  python inverse_kinematics/train_model.py --config "
            f"{arguments.config} --arch {arguments.arch}"
        )
    model = trainer.load_to(checkpoint, arguments.device)
    parameters = sum(p.numel() for p in model.model.parameters())
    print(f"Loaded {arguments.arch}/{arguments.config} ({parameters:,} parameters)")

    # --- the tall dataset ----------------------------------------------
    dataset_rng = sim.generator("dataset", arguments.seed)
    truth, local_truth, x = H.simulate_dataset(arguments.observations, dataset_rng)
    data = torch.as_tensor(x, dtype=torch.float32)
    print(f"{arguments.observations} observations share angle_1 = {truth:.4f}")

    # --- the reference --------------------------------------------------
    started = time.perf_counter()
    grid, density = H.reference_global(
        x, draws=arguments.reference_draws,
        bandwidth=arguments.reference_bandwidth, seed=arguments.seed,
    )
    reference_mean, reference_std, reference_mode = H.grid_moments(grid, density)
    print(
        f"Reference posterior: mean {reference_mean:.4f} sd {reference_std:.4f} "
        f"mode {reference_mode:.4f}  (prior sd {H.GLOBAL_PRIOR_STD[0]:.2f}, "
        f"{time.perf_counter() - started:.0f}s)"
    )

    # --- the rules ------------------------------------------------------
    results, metrics = {}, {}
    for method in arguments.methods:
        entry = METHODS[method]
        print(f"\n=== {entry['label']} ===")
        samples, sampler, runtime = draw(
            model, data, method, arguments, arguments.device
        )
        shared = samples[0, :, H.HIERARCHY[0]]
        row = evaluate(shared, grid, density, truth, sampler, runtime)
        row.update(label=entry["label"], colour=entry["colour"],
                   note=entry["note"],
                   churn_eta=arguments.churn_eta if entry["churn"] else 0.0)
        print(
            f"  {runtime:.1f}s   W1/sigma {row['w1_over_sigma']:.4f}   "
            f"width {row['width_ratio']:.3f}   "
            f"mean err {row['mean_error_sigma']:+.4f} sigma"
        )
        results[method] = samples
        metrics[method] = row

    # --- persist --------------------------------------------------------
    output = arguments.artifacts / "inference"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "posteriors.npz",
        grid=grid, density=density, truth=truth, local_truth=local_truth,
        observations=x, **{f"samples_{k}": v for k, v in results.items()},
    )
    (output / "metrics.json").write_text(json.dumps({
        "arch": arguments.arch,
        "config": arguments.config,
        "parameters": int(parameters),
        "simulations": int(arguments.simulations),
        "observations": int(arguments.observations),
        "draws": int(arguments.draws),
        "timesteps": COMMON["timesteps"],
        "churn_eta": float(arguments.churn_eta),
        "jacobian_refresh": int(arguments.jacobian_refresh),
        "pilot_draws": int(arguments.pilot_draws),
        "seed": int(arguments.seed),
        "truth": truth,
        "reference": {
            "draws": int(arguments.reference_draws),
            "bandwidth": float(arguments.reference_bandwidth),
            "mean": reference_mean, "std": reference_std, "mode": reference_mode,
        },
        "methods": metrics,
    }, indent=2, sort_keys=True))
    print(f"\nWrote {output / 'posteriors.npz'}")
    print(f"Wrote {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
