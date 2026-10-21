#!/usr/bin/env python3
"""Arruda et al.'s own method, run through BayesFlow, on the two paper benchmarks.

This is the comparison arm for ``gaussian_toy.py`` and ``ar1_grid.py``: the
error-damped, mini-batched compositional score of arXiv:2505.14429 as shipped in
BayesFlow 2.0.13 (``CompositionalApproximator.compositional_sample``), with
the paper's own training and sampling settings.

Faithfulness notes
------------------
Network  -- the paper's configuration (their Appendix A.5 and the ABI-DM
            baseline they call "the same size as ours"): MLP 5 x 256, dropout
            0.1, cosine VP schedule, velocity prediction, likelihood weighting,
            AdamW with a cosine learning-rate decay from 5e-4, 1000 epochs,
            batch 128. Toy: 10,000 fresh simulations per epoch (online, as in
            the paper). AR(1): a fixed budget of 10,000 simulations.
Sampling -- BayesFlow's ``two_step_adaptive`` solver (their "adaptive
            second-order sampler"), capped at 10,000 steps; mini-batch = 10% of
            the observations; bridge d(t) = d0 (d1/d0)^t with d0 = 1 at the
            data end; cosine schedule shift s at inference. The prior score is
            passed *without* a time argument, so BayesFlow applies the paper's
            (1-t) bridging weight (their Eq. 9).
Tuning   -- the paper tunes d1 and s per task by Bayesian optimization on
            "RMSE + calibration error" (their "inference-time hyperparameter
            optimization"). Here: a grid over d1 and s on 10 *tuning* datasets
            (seeds disjoint from the test datasets), same criterion, then the
            chosen configuration is scored on the test datasets.
AR(1)    -- BayesFlow composes only the globals, so the paper's two-stage
            sampler (their Eq. 12) is built from two approximators: globals
            composed across the J groups, then theta_raw_j ~ q(theta_raw | eta,
            y_j) with ordinary (non-compositional) sampling.

Results are written next to the COMPASS arms, as method ``bayesflow_arruda``,
so ``gaussian_toy.py --report`` / ``ar1_grid.py --report`` show them together.

Usage (from COMPASS/):
    bash Arruda/paper_benchmarks/run_bayesflow.sh toy --train
    bash Arruda/paper_benchmarks/run_bayesflow.sh toy --observations 10 100 1000
    bash Arruda/paper_benchmarks/run_bayesflow.sh ar1 --train
    bash Arruda/paper_benchmarks/run_bayesflow.sh ar1 --grids 4 16 64
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gaussian_toy as G  # noqa: E402  (applies the 3-thread CPU cap)
import ar1_grid as A  # noqa: E402

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

METHOD = "bayesflow_arruda"
OUT = HERE / "artifacts" / "bayesflow"
EPOCHS = 1000
BATCH = 128
LR = 5e-4
MAX_STEPS = 10_000
TOY_SCALE = 0.1                   # the paper's sigma; G works in units of it
D1_GRID = (1.0, 1e-1, 1e-2, 1e-3, 1e-5)
SHIFT_GRID = (0.0, 2.0, 5.0, 10.0)
TUNE_DATASETS = 10
TUNE_SEED_OFFSET = 777_000        # tuning seeds never coincide with test seeds


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

def make_approximator(variables, conditions, steps_per_epoch):
    import keras
    import bayesflow as bf

    adapter = bf.CompositionalApproximator.build_adapter(
        inference_variables=variables, inference_conditions=conditions)
    network = bf.networks.DiffusionModel(
        subnet_kwargs={"widths": (256,) * 5, "dropout": 0.1},
        noise_schedule="cosine",
        schedule_kwargs={"weighting": "likelihood_weighting"},
        prediction_type="velocity",
    )
    approximator = bf.CompositionalApproximator(
        inference_network=network, adapter=adapter,
        standardize=["inference_variables", "inference_conditions"])
    schedule = keras.optimizers.schedules.CosineDecay(
        LR, decay_steps=EPOCHS * steps_per_epoch)
    approximator.compile(optimizer=keras.optimizers.AdamW(learning_rate=schedule))
    return approximator


def fit_or_load(approximator, dataset, weights, train):
    weights = Path(weights)
    if train or not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        approximator.fit(dataset=dataset, epochs=EPOCHS, verbose=2)
        approximator.save_weights(weights)
        print(f"trained {weights.name} in {time.perf_counter() - started:.0f}s",
              flush=True)
    else:
        # zero-epoch fit builds the layers and the standardizer, then load
        approximator.fit(dataset=dataset, epochs=0, verbose=0)
        approximator.load_weights(weights)
    return approximator


class ToySimulator:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_shape, **_):
        count = int(np.prod(batch_shape))
        eta = TOY_SCALE * self.rng.standard_normal((count, G.DIM))
        y = eta + TOY_SCALE * self.rng.standard_normal((count, G.DIM))
        return {"eta": eta.astype("float32"), "y": y.astype("float32")}


def toy_approximator(train):
    import bayesflow as bf
    steps = 10_000 // BATCH
    approximator = make_approximator("eta", "y", steps)
    dataset = bf.datasets.OnlineDataset(ToySimulator(7), batch_size=BATCH,
                                        num_batches=steps,
                                        adapter=approximator.adapter)
    return fit_or_load(approximator, dataset, OUT / "toy" / "global.weights.h5", train)


def ar1_data(seed):
    g, raw, y = A.simulate_single(10_000, seed=seed)
    return {"g": g.astype("float32"), "raw": raw[:, None].astype("float32"),
            "y": y.astype("float32")}


def ar1_approximators(train):
    import bayesflow as bf
    data = ar1_data(seed=7)
    steps = math.ceil(10_000 / BATCH)
    global_ = make_approximator("g", "y", steps)
    local = make_approximator("raw", ["g", "y"], steps)
    global_ = fit_or_load(
        global_, bf.datasets.OfflineDataset(data=data, batch_size=BATCH,
                                            adapter=global_.adapter),
        OUT / "ar1" / "global.weights.h5", train)
    local = fit_or_load(
        local, bf.datasets.OfflineDataset(data=data, batch_size=BATCH,
                                          adapter=local.adapter),
        OUT / "ar1" / "local.weights.h5", train)
    return global_, local


# ---------------------------------------------------------------------------
# The paper's compositional sampler
# ---------------------------------------------------------------------------

def standard_normal_prior_score(key, scale=1.0):
    """grad log N(0, scale^2 I), time-free: BayesFlow applies the (1-t) bridge."""
    def score(values):
        return {key: -np.asarray(values[key]) / scale**2}
    return score


def compose(approximator, key, conditions, prior_score, num_samples, d1, shift,
            seed):
    network = approximator.inference_network
    schedule = network.noise_schedule
    observations = next(iter(conditions.values())).shape[1]
    original = schedule._shift
    schedule._shift = float(shift)
    try:
        samples = approximator.compositional_sample(
            num_samples=num_samples, conditions=conditions,
            compute_prior_score=prior_score,
            method="two_step_adaptive", steps="adaptive", max_steps=MAX_STEPS,
            mini_batch_size=max(1, observations // 10),
            compositional_bridge_d0=1.0, compositional_bridge_d1=float(d1),
            seed=seed,
        )
    finally:
        schedule._shift = original
    return np.asarray(samples[key], dtype=np.float64)[0]


# ---------------------------------------------------------------------------
# Toy
# ---------------------------------------------------------------------------

def toy_draws(approximator, y_std, num_samples, d1, shift, seed):
    """Draws in G's standardized units for a dataset given in those units."""
    samples = compose(approximator, "eta", {"y": (TOY_SCALE * y_std)[None].astype("float32")},
                      standard_normal_prior_score("eta", TOY_SCALE), num_samples,
                      d1, shift, seed)
    return samples / TOY_SCALE


def toy_score(rows):
    """The paper's tuning criterion: normalized RMSE + calibration error."""
    summary = G.summarize(rows)
    if "rmse" not in summary or "calibration_error" not in summary:
        return float("inf")
    return summary["rmse"] + summary["calibration_error"]


def toy_rows(approximator, observations, seeds, num_samples, d1, shift):
    rows = []
    for index, seed in enumerate(seeds):
        eta, y = G.dataset(observations, seed=seed)
        started = time.perf_counter()
        try:
            draws = toy_draws(approximator, y, num_samples, d1, shift, index)
            finite = np.isfinite(draws).all(axis=1)
            row = G.dataset_metrics(draws[finite], eta, y) if finite.sum() > G.DIM \
                else {"kl": float("inf"), "covered": None}
            row["nonfinite"] = int((~finite).sum())
        except Exception as error:
            row = {"kl": float("inf"), "covered": None, "error": repr(error)}
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
    return rows


def run_toy(arguments):
    approximator = toy_approximator(arguments.train)
    if arguments.train:
        return
    for observations in arguments.observations:
        path = G.result_path(METHOD, observations)
        if path.exists():
            print(f"skip {path.name}")
            continue
        tuning = [TUNE_SEED_OFFSET + 1000 * observations + i
                  for i in range(TUNE_DATASETS)]
        grid = []
        for d1, shift in itertools.product(D1_GRID, SHIFT_GRID):
            rows = toy_rows(approximator, observations, tuning,
                            arguments.tune_samples, d1, shift)
            grid.append({"d1": d1, "shift": shift, "criterion": toy_score(rows),
                         "kl": G.summarize(rows).get("kl")})
            print(f"  tune J={observations} d1={d1:g} s={shift:g} "
                  f"crit={grid[-1]['criterion']:.4f} kl={grid[-1]['kl']}", flush=True)
        best = min(grid, key=lambda row: row["criterion"])
        test = [1000 * observations + i for i in range(arguments.datasets)]
        rows = toy_rows(approximator, observations, test, arguments.num_samples,
                        best["d1"], best["shift"])
        summary = G.summarize(rows)
        summary.update(method=METHOD, observations=observations,
                       num_samples=arguments.num_samples,
                       d1=best["d1"], shift=best["shift"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": summary, "rows": rows,
                                    "tuning": grid}))
        print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_mad")}),
              flush=True)


# ---------------------------------------------------------------------------
# AR(1)
# ---------------------------------------------------------------------------

def ar1_globals(global_, y, num_samples, d1, shift, seed):
    return compose(global_, "g", {"y": y[None].astype("float32")},
                   standard_normal_prior_score("g"), num_samples, d1, shift, seed)


def ar1_locals(local, draws_g, y, seed, batch=10_000):
    """theta_j | eta, y_j for every global draw and group: (S, J) in theta units."""
    samples, groups = len(draws_g), len(y)
    g = np.repeat(draws_g, groups, axis=0).astype("float32")
    yy = np.tile(y, (samples, 1)).astype("float32")
    raw = np.asarray(local.sample(num_samples=1, conditions={"g": g, "y": yy},
                                  batch_size=batch)["raw"], dtype=np.float64)
    raw = raw.reshape(samples, groups)
    return 2.0 / (1.0 + np.exp(-(draws_g[:, 1:2] + raw))) - 1.0


def ar1_tune_score(draws, truths):
    """Normalized global RMSE + calibration error over the tuning datasets."""
    truths = np.asarray(truths)
    span = np.maximum(truths.max(0) - truths.min(0), 1e-12)
    rmse = np.median([np.sqrt(((d - t) ** 2).mean(0)) for d, t in zip(draws, truths)],
                     axis=0) / span
    alphas = G.ALPHAS
    covered = []
    for d, t in zip(draws, truths):
        lower = np.quantile(d, 0.5 - alphas / 2, axis=0)
        upper = np.quantile(d, 0.5 + alphas / 2, axis=0)
        covered.append((lower <= t) & (t <= upper))
    calibration = np.median(np.abs(np.mean(covered, axis=0) - alphas[:, None]), axis=0)
    return float(rmse.mean() + calibration.mean())


def run_ar1(arguments):
    global_, local = ar1_approximators(arguments.train)
    if arguments.train:
        return
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for grid_size in arguments.grids:
        path = A.result_path(METHOD, grid_size, 10_000)
        if path.exists():
            print(f"skip {path.name}")
            continue
        tuning = [A.dataset(grid_size, seed=TUNE_SEED_OFFSET + 100 * grid_size + i)
                  for i in range(TUNE_DATASETS)]
        grid = []
        for d1, shift in itertools.product(D1_GRID, SHIFT_GRID):
            draws = []
            try:
                for index, (_, _, _, y) in enumerate(tuning):
                    draws.append(ar1_globals(global_, y, arguments.tune_samples,
                                             d1, shift, index))
                ok = all(np.isfinite(d).all() for d in draws)
                criterion = ar1_tune_score(draws, [t[0] for t in tuning]) if ok \
                    else float("inf")
            except Exception as error:
                print(f"  tune failed d1={d1:g} s={shift:g}: {error!r}")
                criterion = float("inf")
            grid.append({"d1": d1, "shift": shift, "criterion": criterion})
            print(f"  tune {grid_size}x{grid_size} d1={d1:g} s={shift:g} "
                  f"crit={criterion:.4f}", flush=True)
        best = min(grid, key=lambda row: row["criterion"])
        rows = []
        for index in range(arguments.datasets):
            true_g, _, true_theta, y = A.dataset(grid_size, seed=100 * grid_size + index)
            ref_g, ref_l, ess = A.reference_for(grid_size, index, device)
            row = {"true_global": true_g.tolist(), "reference_ess": ess}
            started = time.perf_counter()
            try:
                draws_g = ar1_globals(global_, y, arguments.num_samples,
                                      best["d1"], best["shift"], index)
                draws_l = ar1_locals(local, draws_g, y, index)
            except Exception as error:
                row["error"] = repr(error)
                rows.append(row)
                continue
            if not (np.isfinite(draws_g).all() and np.isfinite(draws_l).all()):
                row["error"] = "non-finite draws"
                rows.append(row)
                continue
            row.update(A.compare(draws_g, draws_l, ref_g, ref_l, true_g, true_theta))
            row["seconds"] = time.perf_counter() - started
            rows.append(row)
            print(f"  {METHOD} {grid_size}x{grid_size} [{index}] "
                  f"mean_err={np.round(row['global_mean_error_sd'], 2)}sd "
                  f"width={np.round(row['global_width_ratio'], 2)} "
                  f"local_rmse={row['local_rmse']:.3f} t={row['seconds']:.1f}s",
                  flush=True)
        summary = A.summarize(rows, METHOD, grid_size)
        summary.update(d1=best["d1"], shift=best["shift"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": summary, "rows": rows, "tuning": grid}))
        print(json.dumps({k: v for k, v in summary.items() if "per_param" not in k}),
              flush=True)


def main():
    global EPOCHS, OUT, D1_GRID, SHIFT_GRID, TUNE_DATASETS, METHOD
    # GPU selection before anything imports torch (bayesflow does, via keras).
    from autocvd import autocvd
    autocvd(num_gpus=1, interval=1)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("benchmark", choices=["toy", "ar1"])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--observations", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--grids", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--datasets", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--tune-samples", type=int, default=100)
    parser.add_argument("--tune-datasets", type=int, default=TUNE_DATASETS)
    parser.add_argument("--d1", type=float, nargs="+", default=list(D1_GRID))
    parser.add_argument("--shifts", type=float, nargs="+", default=list(SHIFT_GRID))
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="anything but 1000 writes to a separate *_smoke folder")
    arguments = parser.parse_args()
    EPOCHS = arguments.epochs
    D1_GRID, SHIFT_GRID = tuple(arguments.d1), tuple(arguments.shifts)
    TUNE_DATASETS = arguments.tune_datasets
    if EPOCHS != 1000:
        OUT = OUT.parent / "bayesflow_smoke"
        METHOD = "bayesflow_arruda_smoke"
    if arguments.benchmark == "toy":
        arguments.datasets = arguments.datasets or 10
        arguments.num_samples = arguments.num_samples or 500
        run_toy(arguments)
    else:
        arguments.datasets = arguments.datasets or 10
        arguments.num_samples = arguments.num_samples or 100
        run_ar1(arguments)


if __name__ == "__main__":
    main()
