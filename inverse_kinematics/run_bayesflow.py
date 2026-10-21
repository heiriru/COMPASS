#!/usr/bin/env python3
"""The BayesFlow reference method on the same arm, with its own diagnostics.

Reproduces the inference half of
https://bayesflow.org/v2.0.13/_examples/Diffusion_Models.html -- their adapter,
their ``bf.networks.DiffusionModel``, their ``BasicWorkflow``, their training
budget, and their diagnostic plots -- so that "their method" and the COMPASS
compositional runs can be compared on identical data rather than on two papers'
worth of differing conventions.

Runs in a **separate virtual environment** (``COMPASS/.bayesflow``): BayesFlow
pulls Keras and JAX, and installing those beside the torch build that the rest of
this repository depends on is a risk with no upside. Invoke it as::

    CUDA_VISIBLE_DEVICES=<free gpu> \\
      COMPASS/.bayesflow/bin/python inverse_kinematics/run_bayesflow.py

Two things are deliberately shared with the COMPASS side rather than re-typed:
the **forward model** (imported from :mod:`simulator`, so a discrepancy between
the two methods' simulators is impossible) and the **endpoint-reconstruction
metric**, which is what lets the two be put in one table.

Scope note: this is their *single-observation* problem -- four parameters from
one 2-D end position, no shared coordinate and no composition. That is the setup
their notebook poses, and it is the right control for the COMPASS runs: it shows
what a single observation can identify before any tall-data machinery is added.
"""

import os
import sys
from pathlib import Path

# Keras must be told its backend before it is imported.
os.environ.setdefault("KERAS_BACKEND", "jax")
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "3"
if hasattr(os, "sched_setaffinity"):
    _available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, _available[:3])

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bayesflow as bf  # noqa: E402

import simulator as sim  # noqa: E402
from plots import (ACCENT, INK, _draw_arm, _draw_rail, _style_axis,  # noqa: E402
                   plot_params_kinematic)

ROOT = Path(__file__).resolve().parent
VARIABLE_NAMES = sim.PARAMETER_NAMES
VARIABLE_NAMES_NICE = sim.PARAMETER_LABELS


def prior():
    """The notebook's prior, drawing through our own constants."""
    return dict(parameters=np.random.normal(loc=0.0, scale=sim.PRIOR_SCALES))


def observation_model(parameters):
    """The notebook's forward model -- delegated, so the two methods cannot differ."""
    return dict(observables=sim.forward(np.asarray(parameters, dtype=np.float64)))


def build_workflow(simulator):
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .rename("parameters", "inference_variables")
        .rename("observables", "inference_conditions")
    )
    return bf.BasicWorkflow(
        adapter=adapter,
        simulator=simulator,
        inference_network=bf.networks.DiffusionModel(),
        standardize="inference_conditions",
    )


def endpoint_error(parameter_samples, observed):
    """Mean distance from each posterior draw's arm tip to the observed point.

    The same quantity the COMPASS runs report, so the two methods land in one
    table. It needs no reference posterior: the forward map is deterministic, so
    a correct posterior draw reproduces its own observation exactly.
    """
    errors = []
    for index in range(parameter_samples.shape[0]):
        predicted = sim.forward(parameter_samples[index])
        errors.append(np.linalg.norm(predicted - observed[index], axis=1))
    return np.concatenate(errors)


def plot_arm_posterior(parameter_samples, observed, truth, output,
                       count=4, ensemble=150):
    """Their ``plot_arm_posterior``, redrawn with our arm renderer.

    Theirs calls ``misc.diffusion.kinematics_helper.InverseKinematicsModel``,
    which ships with the documentation build rather than with BayesFlow and so
    cannot be imported here -- the same substitution documented in ``plots.py``.
    """
    count = min(count, parameter_samples.shape[0])
    figure, axes = plt.subplots(1, count, figsize=(2.5 * count, 3.1),
                                layout="constrained", sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    stride = max(1, parameter_samples.shape[1] // ensemble)
    for index, axis in enumerate(axes):
        for chain in sim.joint_positions(parameter_samples[index, ::stride]):
            _draw_arm(axis, chain, ACCENT, 1.0, 0.10)
        _draw_arm(axis, sim.joint_positions(truth[index])[0], INK, 1.8, 0.9,
                  zorder=4, joint_size=2.8)
        axis.plot(observed[index, 1], observed[index, 0], marker="X",
                  markersize=9, color=INK, markeredgecolor="white",
                  markeredgewidth=0.9, zorder=6)
        _draw_rail(axis, span=1.6)
        _style_axis(axis)
        axis.set_aspect("equal")
        axis.set_xlim(-1.6, 2.3)
        axis.set_ylim(-1.6, 1.6)
        axis.set_title(f"Test set {index}", fontsize=9, color=INK)
        axis.set_xlabel("$x_2$", color=INK)
    axes[0].set_ylabel("$x_1$", color=INK)
    figure.suptitle(
        "BayesFlow DiffusionModel — posterior arm configurations\n"
        "faint: posterior draws · solid: truth · cross: observed end position",
        fontsize=10.5, color=INK,
    )
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--simulations", type=int, default=10_000,
                        help="The notebook's training budget.")
    result.add_argument("--epochs", type=int, default=200)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--test-datasets", type=int, default=300)
    result.add_argument("--draws", type=int, default=1_000)
    result.add_argument("--sample-chunk", type=int, default=25,
                        help="Test datasets sampled per network call.")
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    np.random.seed(arguments.seed)
    keras_simulator = bf.make_simulator([prior, observation_model])

    output = arguments.artifacts / "bayesflow"
    output.mkdir(parents=True, exist_ok=True)
    figures = arguments.artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    training_data = keras_simulator.sample(arguments.simulations)
    print(f"Generated {arguments.simulations} simulations")
    print(f"  observables {training_data['observables'].shape}   "
          f"parameters {training_data['parameters'].shape}")

    workflow = build_workflow(keras_simulator)
    started = time.perf_counter()
    history = workflow.fit_offline(
        training_data, epochs=arguments.epochs, batch_size=arguments.batch_size,
    )
    train_seconds = time.perf_counter() - started
    parameters = int(sum(
        np.prod(w.shape) for w in workflow.approximator.trainable_weights
    ))
    print(f"Trained in {train_seconds / 60:.1f} min; {parameters:,} parameters")

    test_data = keras_simulator.sample(arguments.test_datasets)
    # Chunked over test datasets. One call for all of them asks the network to
    # hold `test_datasets * draws` states at once -- 300,000 here, which OOMs an
    # 11 GB card at 4.5 GiB for a single reduction. The posteriors are
    # independent per dataset, so chunking changes nothing statistically.
    blocks = []
    for start in range(0, arguments.test_datasets, arguments.sample_chunk):
        stop = min(start + arguments.sample_chunk, arguments.test_datasets)
        block = {key: value[start:stop] for key, value in test_data.items()}
        blocks.append(np.asarray(
            workflow.approximator.sample(
                conditions=block, num_samples=arguments.draws,
            )["parameters"], dtype=np.float64,
        ))
    samples = np.concatenate(blocks, axis=0)
    estimates = {"parameters": samples}
    observed = np.asarray(test_data["observables"], dtype=np.float64)
    truth = np.asarray(test_data["parameters"], dtype=np.float64)

    errors = endpoint_error(samples, observed)
    print(f"endpoint |err|  mean {errors.mean():.4f}  median "
          f"{np.median(errors):.4f}")

    written = []
    # Their calibration diagnostic.
    figure = bf.diagnostics.plots.calibration_ecdf(
        estimates=estimates, targets=test_data,
        variable_names=VARIABLE_NAMES_NICE,
    )
    path = figures / "08_bayesflow_calibration_ecdf.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    figure = bf.diagnostics.plots.recovery(
        estimates=estimates, targets=test_data,
        variable_names=VARIABLE_NAMES_NICE,
    )
    path = figures / "09_bayesflow_recovery.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    written.append(plot_arm_posterior(
        samples, observed, truth, figures / "10_bayesflow_arm_posterior.png",
    ))

    # --- their single-observation figure, reproduced exactly ------------
    # The notebook fixes obs = [[0, 1.5]] and calls this "a nice bimodal
    # posterior". That observation is not arbitrary: x1 = 0 is precisely where
    # the arm's reflection (a -> -a, h -> 2*x1 - h) costs no prior mass, since
    # the penalty is 2*x1*(h - x1)/sigma_h^2. See the symmetry section of
    # README.md -- their figure is that involution, at the one end position
    # where it is free.
    single = {"observables": np.array([[0.0, 1.5]], dtype=np.float64)}
    single_samples = np.asarray(
        workflow.approximator.sample(conditions=single, num_samples=1000)
        ["parameters"], dtype=np.float64,
    )
    path = figures / "11_bayesflow_posterior_single.png"
    plot_params_kinematic(
        single_samples[0], np.asarray(training_data["parameters"]),
        output=path, plt_module=plt,
        title="BayesFlow posterior at obs = [0, 1.5]  "
              "(magenta: posterior, black: prior)",
    )
    written.append(path)

    single_truth = np.zeros((1, 4))          # no ground truth for a fixed obs
    path = figures / "12_bayesflow_arm_single.png"
    figure, axis = plt.subplots(figsize=(3.6, 3.6), layout="constrained")
    for chain in sim.joint_positions(single_samples[0, ::max(1, 1000 // 200)]):
        _draw_arm(axis, chain, ACCENT, 1.0, 0.12)
    axis.plot(single["observables"][0, 1], single["observables"][0, 0],
              marker="X", markersize=10, color=INK, markeredgecolor="white",
              markeredgewidth=0.9, zorder=6)
    _draw_rail(axis, span=1.6)
    _style_axis(axis)
    axis.set_aspect("equal")
    axis.set_xlim(-1.8, 2.3)
    axis.set_ylim(-1.8, 1.8)
    axis.set_xlabel("$x_2$", color=INK)
    axis.set_ylabel("$x_1$", color=INK)
    axis.set_title("Arm configurations at obs = [0, 1.5]", fontsize=10, color=INK)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    # Quantify the claim rather than eyeballing it: the reflection predicts the
    # two branches are mirror images in the angles.
    angles = single_samples[0][:, 1:]
    print("single-observation posterior at [0, 1.5]:")
    for index, name in enumerate(VARIABLE_NAMES):
        column = single_samples[0][:, index]
        print(f"  {name:12} mean {column.mean():+.3f}  sd {column.std():.3f}  "
              f"frac>0 {np.mean(column > 0):.3f}")
    print(f"  sign(angle_1)==sign(angle_3) in "
          f"{np.mean(np.sign(angles[:, 0]) == np.sign(angles[:, 2])):.3f} of draws")

    metrics = bf.diagnostics.metrics.calibration_error(
        estimates=estimates, targets=test_data, variable_names=VARIABLE_NAMES_NICE,
    )
    summary = {
        "method": "bayesflow DiffusionModel (single observation)",
        "reference_url":
            "https://bayesflow.org/v2.0.13/_examples/Diffusion_Models.html",
        "bayesflow": bf.__version__,
        "simulations": int(arguments.simulations),
        "epochs": int(arguments.epochs),
        "batch_size": int(arguments.batch_size),
        "parameters": parameters,
        "train_seconds": train_seconds,
        "test_datasets": int(arguments.test_datasets),
        "draws": int(arguments.draws),
        "endpoint_error_mean": float(errors.mean()),
        "endpoint_error_median": float(np.median(errors)),
        "calibration_error": {
            str(name): float(value) for name, value
            in zip(metrics["variable_names"], np.asarray(metrics["values"]).ravel())
        },
        "final_loss": float(np.asarray(history.history["loss"])[-1]),
        "single_observation": {
            "observables": [0.0, 1.5],
            "note": "x1 = 0 is where the arm's reflection costs no prior mass",
            "mean": single_samples[0].mean(axis=0).tolist(),
            "std": single_samples[0].std(axis=0).tolist(),
            "fraction_positive": (single_samples[0] > 0).mean(axis=0).tolist(),
        },
        "figures": [path.name for path in written],
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    np.savez_compressed(
        output / "posteriors.npz",
        samples=samples, observed=observed, truth=truth,
    )
    for path in written:
        print(f"Wrote {path}")
    print(f"Wrote {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
