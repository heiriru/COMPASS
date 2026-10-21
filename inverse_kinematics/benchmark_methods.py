#!/usr/bin/env python3
"""COMPASS against the BayesFlow reference: accuracy, size, and inference time.

The two methods do not solve the same problem, so a single table needs a stated
basis of comparison. The one used here is the **single-observation posterior**
``p(theta | x)``, because it is the only task both perform natively: BayesFlow's
``DiffusionModel`` is a dedicated amortized posterior and does nothing else,
while COMPASS reaches the same conditional through its joint model with the
observables conditioned. Everything COMPASS can additionally do -- arbitrary
conditioning, tall-data composition -- is outside this table and is the thing it
is paying for.

Metrics, all on identical test data drawn from the shared simulator:

* **Endpoint error** -- the mean distance from a posterior draw's arm tip to the
  observation it was conditioned on. Reference-free: the forward map is
  deterministic, so a correct draw reproduces its own observation exactly. This
  is the accuracy number.
* **Parameters** -- trainable weights.
* **Training time** and **inference time** -- wall clock, on the same GPU, for
  the same number of posterior draws.

Inference time is reported per 1,000 draws for one observation, since the two
frameworks batch differently and a raw total would mostly measure batching.

Usage:
    # COMPASS side (this venv)
    python inverse_kinematics/benchmark_methods.py --side compass
    # BayesFlow side (its own venv)
    COMPASS/.bayesflow/bin/python inverse_kinematics/benchmark_methods.py --side bayesflow
    # merge whatever has been written
    python inverse_kinematics/benchmark_methods.py --side report
"""

import os
import sys
from pathlib import Path

if "--side" in sys.argv and "bayesflow" in sys.argv:
    os.environ.setdefault("KERAS_BACKEND", "jax")
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "3"
if hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import simulator as sim  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "benchmark"
TEST_SEED = 4242


def test_set(count):
    """The same observations for both sides, from one named stream."""
    rng = sim.generator("benchmark-test", TEST_SEED)
    theta = sim.sample_prior(count, rng)
    return theta, sim.forward(theta)


def endpoint_error(samples, observed):
    return float(np.mean(np.concatenate([
        np.linalg.norm(sim.forward(samples[i]) - observed[i], axis=1)
        for i in range(samples.shape[0])
    ])))


def run_compass(arguments):
    import torch
    import hierarchical as H
    import churn
    import train_model as trainer

    directory = trainer.model_directory(
        ROOT / "artifacts", arguments.config, arguments.arch,
        arguments.simulations, menu=arguments.mask_menu,
        time_sampling=arguments.time_sampling,
    )
    if arguments.model_suffix:
        directory = directory.with_name(directory.name + arguments.model_suffix)
    model = trainer.load_to(directory / "Model_checkpoint.pt", arguments.device)
    parameters = sum(p.numel() for p in model.model.parameters())
    training = json.loads((directory / "training.json").read_text())

    theta, x = test_set(arguments.test_observations)
    data = torch.as_tensor(x, dtype=torch.float32)

    # Pure churn: the corrector arm is no longer part of this repository's
    # sampler recipe, so timing it would time something nothing runs.
    sample_kwargs = dict(timesteps=100, method="dpm", order=2,
                         churn_eta=churn.DEFAULT_ETA,
                         corrector_steps=0, final_corrector_steps=0,
                         terminal_denoise=arguments.terminal_denoise,
                         device=arguments.device, verbose=False)
    torch.manual_seed(0)
    model.sample(x=data[:1], num_samples=64, **sample_kwargs)   # warm up kernels
    torch.cuda.synchronize() if arguments.device.startswith("cuda") else None

    torch.manual_seed(0)
    started = time.perf_counter()
    draws = model.sample(x=data, num_samples=arguments.draws, **sample_kwargs)
    if arguments.device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    draws = draws.detach().cpu().numpy()

    # Node order -> simulator order for the shared metric.
    samples = np.stack([H.to_simulator_order(draws[i, :, :4])
                        for i in range(draws.shape[0])])
    return {
        "method": f"COMPASS {arguments.arch}/{arguments.config}"
                  f" +{trainer.menu_name(arguments.mask_menu)}"
                  f" {arguments.time_sampling}"
                  f"{' +denoise' if arguments.terminal_denoise else ''}",
        "model_directory": directory.name,
        "mask_menu": trainer.menu_name(arguments.mask_menu),
        "time_sampling": arguments.time_sampling,
        "terminal_denoise": bool(arguments.terminal_denoise),
        "parameters": int(parameters),
        "train_seconds": float(training.get("train_seconds", float("nan"))),
        "train_simulations": int(training.get("simulations", 0)),
        "endpoint_error": endpoint_error(samples, x),
        "inference_seconds_total": seconds,
        "inference_seconds_per_1k_draws": seconds / (
            arguments.test_observations * arguments.draws / 1000.0),
        "test_observations": int(arguments.test_observations),
        "draws": int(arguments.draws),
        "note": "joint p(theta,x) with masking; supports composition",
    }


def run_bayesflow(arguments):
    import bayesflow as bf
    from run_bayesflow import build_workflow, prior, observation_model

    keras_simulator = bf.make_simulator([prior, observation_model])
    np.random.seed(7)
    training_data = keras_simulator.sample(arguments.bf_simulations)
    workflow = build_workflow(keras_simulator)
    started = time.perf_counter()
    workflow.fit_offline(training_data, epochs=arguments.bf_epochs,
                         batch_size=128, verbose=0)
    train_seconds = time.perf_counter() - started
    parameters = int(sum(np.prod(w.shape)
                         for w in workflow.approximator.trainable_weights))

    theta, x = test_set(arguments.test_observations)
    conditions = {"observables": x}
    workflow.approximator.sample(
        conditions={"observables": x[:1]}, num_samples=64)   # warm up

    started = time.perf_counter()
    blocks = []
    for start in range(0, len(x), arguments.bf_chunk):
        block = {"observables": x[start:start + arguments.bf_chunk]}
        blocks.append(np.asarray(workflow.approximator.sample(
            conditions=block, num_samples=arguments.draws)["parameters"],
            dtype=np.float64))
    seconds = time.perf_counter() - started
    samples = np.concatenate(blocks, axis=0)

    return {
        "method": "BayesFlow DiffusionModel",
        "parameters": parameters,
        "train_seconds": train_seconds,
        "train_simulations": int(arguments.bf_simulations),
        "endpoint_error": endpoint_error(samples, x),
        "inference_seconds_total": seconds,
        "inference_seconds_per_1k_draws": seconds / (
            arguments.test_observations * arguments.draws / 1000.0),
        "test_observations": int(arguments.test_observations),
        "draws": int(arguments.draws),
        "note": "dedicated amortized posterior p(theta|x)",
    }


def report():
    rows = []
    for path in sorted(OUTPUT.glob("*.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        raise SystemExit(f"No results in {OUTPUT}; run --side compass/bayesflow first.")
    rows.sort(key=lambda r: r["endpoint_error"])
    best = rows[0]["endpoint_error"]
    print(f"{'method':40}{'params':>11}{'train s':>10}{'endpoint':>11}"
          f"{'vs best':>9}{'infer s/1k':>12}")
    for row in rows:
        print(f"{row['method']:40}{row['parameters']:>11,}"
              f"{row['train_seconds']:>10.0f}{row['endpoint_error']:>11.4f}"
              f"{row['endpoint_error'] / best:>8.1f}x"
              f"{row['inference_seconds_per_1k_draws']:>12.3f}")
    print("\nBasis: single-observation posterior p(theta | x), identical test "
          "draws.\nEndpoint error is reference-free (deterministic forward map).")
    return rows


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--side", choices=("compass", "bayesflow", "report"),
                        required=True)
    result.add_argument("--config", default="h16d2")
    result.add_argument("--arch", default="legacy")
    result.add_argument("--mask-menu", default="six_weighted",
                        help="Training condition-mask recipe of the checkpoint "
                             "to load; see train_model.MASK_MENUS.")
    result.add_argument("--menu", dest="mask_menu", action="store_const",
                        const="posterior",
                        help="Deprecated alias for --mask-menu posterior.")
    result.add_argument("--no-menu", dest="mask_menu", action="store_const",
                        const="bernoulli",
                        help="Deprecated alias for --mask-menu bernoulli.")
    result.add_argument("--time-sampling", default="log_sigma",
                        choices=("log_sigma", "mixture", "uniform"))
    result.add_argument("--model-suffix", default="",
                        help="Appended to the resolved model directory name; "
                             "matches train_model.py --output-suffix.")
    # Off by default would understate COMPASS by ~6x: on the identical wide
    # checkpoint the terminal Tweedie step takes median endpoint error from
    # 0.0691 to 0.0112. It is part of the sampler, not a tuning trick, so the
    # benchmark runs it and --no-terminal-denoise is the control.
    result.add_argument("--terminal-denoise", action="store_true", default=True)
    result.add_argument("--no-terminal-denoise", dest="terminal_denoise",
                        action="store_false")
    result.add_argument("--simulations", type=int, default=200_000)
    result.add_argument("--test-observations", type=int, default=50)
    result.add_argument("--draws", type=int, default=1_000)
    result.add_argument("--bf-simulations", type=int, default=10_000)
    result.add_argument("--bf-epochs", type=int, default=200)
    result.add_argument("--bf-chunk", type=int, default=25)
    result.add_argument("--device", default="cuda")
    # Without this every run wrote benchmark/compass.json, so a second arm
    # silently destroyed the first one's result. The 10k rows in this
    # directory had to be renamed by hand after the fact.
    result.add_argument("--tag", default=None,
                        help="Suffix for the output file: writes "
                             "benchmark/<side>_<tag>.json instead of "
                             "benchmark/<side>.json.")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.side == "report":
        report()
        return 0
    row = run_compass(arguments) if arguments.side == "compass" else run_bayesflow(arguments)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = arguments.side if not arguments.tag else f"{arguments.side}_{arguments.tag}"
    path = OUTPUT / f"{stem}.json"
    path.write_text(json.dumps(row, indent=2, sort_keys=True))
    print(json.dumps(row, indent=2, sort_keys=True))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
