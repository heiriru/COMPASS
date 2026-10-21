#!/usr/bin/env python3
"""Run the inverse-kinematics simulations and write the pre-training figures.

Reproduces everything the BayesFlow "Diffusion Models" example does *before* it
builds or trains a network: draw from the prior, push the draws through the
forward model, and plot. No model is constructed here.

Usage:
    python inverse_kinematics/run_simulations.py
    python inverse_kinematics/run_simulations.py --simulations 100000 --seed 7
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

CPU_INFO = configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import simulator as sim  # noqa: E402
import plots  # noqa: E402

ROOT = Path(__file__).resolve().parent
REFERENCE_URL = "https://bayesflow.org/v2.0.13/_examples/Diffusion_Models.html"
BENCHMARK_REFERENCE = "https://arxiv.org/pdf/2101.10763.pdf"

# The notebook's own count. Kept as the default so the reproduced histogram has
# the same sampling noise as the published one.
DEFAULT_SIMULATIONS = 10_000


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    result.add_argument("--seed", type=int, default=sim.ROOT_SEED)
    result.add_argument("--example-arms", type=int, default=12,
                        help="Prior draws shown in the left illustration panel.")
    result.add_argument("--consistent-arms", type=int, default=120,
                        help="Configurations reaching one end position, right panel.")
    result.add_argument("--tolerance", type=float, default=0.01,
                        help="Acceptance radius for those configurations.")
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    print(
        f"CPU affinity: {CPU_INFO['active_cpus']}/{CPU_INFO['host_cpus']} CPUs "
        f"({CPU_INFO['percent']:.3f}%)"
    )

    figures = arguments.artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # --- simulate -------------------------------------------------------
    training_rng = sim.generator("training", arguments.seed)
    parameters, observables = sim.simulate(arguments.simulations, training_rng)
    print(f"Generated {arguments.simulations} simulations")
    print(f"Observables shape (robot arm endpoints): {observables.shape}")
    print(f"Parameters shape (heights and angles):   {parameters.shape}")

    # The forward model is deterministic, so this is a real check on it rather
    # than a restatement: the joint chain must terminate exactly where the
    # closed form says, in the reversed coordinate order the plots use.
    endpoints = sim.joint_positions(parameters[:256])[:, -1, :]
    drift = float(np.abs(endpoints - observables[:256, ::-1]).max())
    if drift > 1e-12:
        raise RuntimeError(f"Joint chain disagrees with forward(): {drift:.3e}")

    # --- illustration draws --------------------------------------------
    example_rng = sim.generator("illustration", arguments.seed)
    example_parameters = sim.sample_prior(arguments.example_arms, example_rng)

    # A target the prior actually reaches often, so rejection sampling is cheap:
    # the median end position of the training draws.
    target = np.median(observables, axis=0)
    consistent_rng = sim.generator("rejection", arguments.seed)
    consistent, drawn = sim.sample_consistent_with(
        target, arguments.consistent_arms, consistent_rng,
        tolerance=arguments.tolerance,
    )
    acceptance = len(consistent) / drawn
    print(
        f"Target end position (x1, x2): ({target[0]:.4f}, {target[1]:.4f})   "
        f"{len(consistent)} configurations within {arguments.tolerance} "
        f"from {drawn:,} draws (acceptance {acceptance:.2e})"
    )

    # --- figures --------------------------------------------------------
    written = [
        plots.plot_example_simulations(
            example_parameters, consistent, target,
            figures / "01_example_simulations.png", plt,
        ),
        plots.plot_prior_parameters(
            parameters, figures / "02_prior_parameters.png", plt,
        ),
        plots.plot_observation_space(
            observables, figures / "03_observation_space.png", plt,
        ),
    ]

    # --- persist --------------------------------------------------------
    draws = arguments.artifacts / f"simulations-{arguments.simulations}.npz"
    np.savez_compressed(draws, parameters=parameters, observables=observables)

    manifest = {
        "reference_url": REFERENCE_URL,
        "benchmark_reference": BENCHMARK_REFERENCE,
        "stage": "simulation only; no model is constructed or trained",
        "simulations": int(arguments.simulations),
        "root_seed": int(arguments.seed),
        "prior_scales": sim.PRIOR_SCALES.tolist(),
        "segment_lengths": [sim.L1, sim.L2, sim.L3],
        "parameter_names": sim.PARAMETER_NAMES,
        "parameter_mean": parameters.mean(axis=0).tolist(),
        "parameter_std": parameters.std(axis=0).tolist(),
        "observable_mean": observables.mean(axis=0).tolist(),
        "observable_std": observables.std(axis=0).tolist(),
        "illustration": {
            "example_arms": int(arguments.example_arms),
            "consistent_arms": int(len(consistent)),
            "target_observable": target.tolist(),
            "tolerance": float(arguments.tolerance),
            "rejection_draws": int(drawn),
            "acceptance_rate": acceptance,
        },
        "draws_file": draws.name,
        "figures": [path.name for path in written],
        "cpu": CPU_INFO,
    }
    manifest_path = arguments.artifacts / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    for path in written:
        print(f"Wrote {path}")
    print(f"Wrote {draws}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
