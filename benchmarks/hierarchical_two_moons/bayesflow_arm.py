"""Arruda et al.'s method (arXiv:2505.14429) as shipped in BayesFlow 2.0.13.

Same faithfulness choices as ``Arruda/paper_benchmarks/bayesflow_arruda.py``:

* network: MLP 5 x 256, dropout 0.1, cosine VP schedule, velocity prediction,
  likelihood weighting, AdamW 5e-4 with cosine decay, batch 128;
* two approximators (the paper's Eq. 12 two-stage sampler): globals g | x_i
  composed over the N groups by ``compositional_sample``, then
  theta_i ~ q(theta_i | g, x_i) by ordinary sampling for every global draw;
* sampling: ``two_step_adaptive`` (<= 10k steps), mini-batch 10% of N, bridge
  d0 = 1, d1 and the schedule shift s tuned per N on separate tuning datasets by
  "global RMSE + calibration error" (the paper's criterion);
* globals in the probit coordinates, so the prior is exactly N(0, I).

Two arms share the networks:

``bayesflow_arruda``            prior score passed without time: BayesFlow
                                applies the paper's (1 - t) weighting.
``bayesflow_arruda_diffprior``  the exactly diffused N(0, I) prior score passed
                                with time (the fix found on the GM benchmark).

Training budget: the same single-group simulations as the COMPASS networks.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

import task as T

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts" / "bayesflow"
BATCH = 128
LR = 5e-4
MAX_STEPS = 10_000
D1_GRID = (1.0, 1e-1, 1e-2, 1e-3)
SHIFT_GRID = (0.0, 2.0, 5.0)
TUNE_DATASETS = 5
TUNE_SAMPLES = 100
METHODS = ("bayesflow_arruda", "bayesflow_arruda_diffprior")


def _approximator(variables, conditions, steps_total):
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
    schedule = keras.optimizers.schedules.CosineDecay(LR, decay_steps=steps_total)
    approximator.compile(optimizer=keras.optimizers.AdamW(learning_rate=schedule))
    return approximator


def _data(train_samples, seed=7):
    g, locals_, x = T.simulate(1, train_samples, seed)
    return {"g": T.globals_to_z(g).astype("float32"),
            "l": locals_[:, 0].astype("float32"),
            "x": x[:, 0].astype("float32")}


_CACHE = {}


def approximators(train_samples, epochs, train=False):
    """(global, local) approximators; trained and saved if missing."""
    import bayesflow as bf

    key = (train_samples, epochs)
    if key in _CACHE:
        return _CACHE[key]
    folder = OUT / f"n{train_samples}_e{epochs}"
    data = _data(train_samples)
    steps = math.ceil(train_samples / BATCH)
    out = []
    for name, variables, conditions in (("global", "g", "x"), ("local", "l", ["g", "x"])):
        approximator = _approximator(variables, conditions, epochs * steps)
        dataset = bf.datasets.OfflineDataset(data=data, batch_size=BATCH,
                                             adapter=approximator.adapter)
        weights = folder / f"{name}.weights.h5"
        if not weights.exists():
            if not train:
                raise FileNotFoundError(f"{weights} missing; run `train --methods bayesflow_arruda`")
            folder.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            approximator.fit(dataset=dataset, epochs=epochs, verbose=2)
            approximator.save_weights(weights)
            (folder / f"{name}.json").write_text(json.dumps(
                {"seconds": time.perf_counter() - started, "epochs": epochs,
                 "train_samples": train_samples}))
            print(f"trained BayesFlow {name} in {time.perf_counter() - started:.0f}s", flush=True)
        else:
            approximator.fit(dataset=dataset, epochs=0, verbose=0)
            approximator.load_weights(weights)
        out.append(approximator)
    _CACHE[key] = tuple(out)
    return _CACHE[key]


def _prior_score(approximator, diffused):
    """Score of the N(0, I) prior on g, in BayesFlow's calling convention."""
    if not diffused:
        def score(values):
            return {"g": -np.asarray(values["g"])}
        return score

    from keras import ops
    network = approximator.inference_network
    layer = approximator.standardizer.standardize_layers["inference_variables"]
    m = ops.convert_to_numpy(layer.moving_mean[0]).reshape(1, -1)
    s = ops.convert_to_numpy(layer.moving_std(0)).reshape(1, -1)

    def score(values, time):
        # BayesFlow passes the un-standardized noisy state and multiplies the
        # returned score by s. In standardized units z0 ~ N(-m/s, 1/s^2), so the
        # diffused marginal is N(-alpha m/s, alpha^2/s^2 + sigma^2).
        log_snr = network.noise_schedule.get_log_snr(
            t=ops.convert_to_tensor(time, "float32"), training=False)
        alpha, sigma = network.noise_schedule.get_alpha_sigma(log_snr_t=log_snr)
        alpha = ops.convert_to_numpy(alpha).reshape(-1, 1)
        sigma = ops.convert_to_numpy(sigma).reshape(-1, 1)
        z = (np.asarray(ops.convert_to_numpy(values["g"])) - m) / s
        var = alpha ** 2 / s ** 2 + sigma ** 2
        score_z = -(z + alpha * m / s) / var
        return {"g": (score_z / s).astype("float32")}
    return score


def _globals(global_, x, num_samples, d1, shift, seed, diffused):
    network = global_.inference_network
    n = len(x)
    if n == 1:
        # composition needs >= 2 observations; one group is plain sampling
        out = global_.sample(num_samples=num_samples,
                             conditions={"x": x[None].astype("float32")[:, 0]})
        return np.asarray(out["g"], dtype=np.float64)[0]
    schedule = network.noise_schedule
    original = schedule._shift
    schedule._shift = float(shift)
    try:
        out = global_.compositional_sample(
            num_samples=num_samples, conditions={"x": x[None].astype("float32")},
            compute_prior_score=_prior_score(global_, diffused),
            method="two_step_adaptive", steps="adaptive", max_steps=MAX_STEPS,
            mini_batch_size=max(1, n // 10),
            compositional_bridge_d0=1.0, compositional_bridge_d1=float(d1), seed=seed)
    finally:
        schedule._shift = original
    return np.asarray(out["g"], dtype=np.float64)[0]


def _locals(local, z, x, batch=20_000):
    """theta_i | g, x_i for every global draw and group: (S, N, 2)."""
    s, n = len(z), len(x)
    g = np.repeat(z, n, axis=0).astype("float32")
    xx = np.tile(x, (s, 1)).astype("float32")
    out = local.sample(num_samples=1, conditions={"g": g, "x": xx}, batch_size=batch)
    return np.asarray(out["l"], dtype=np.float64).reshape(s, n, 2)


def _tune_criterion(draws, truths):
    """Normalized global RMSE + calibration error (the paper's tuning objective), z units."""
    truths = np.asarray(truths)
    span = np.maximum(truths.max(0) - truths.min(0), 1e-12)
    rmse = np.median([np.sqrt(((d - t) ** 2).mean(0)) for d, t in zip(draws, truths)], 0) / span
    alphas = np.linspace(0.05, 0.95, 19)
    covered = []
    for d, t in zip(draws, truths):
        lo = np.quantile(d, 0.5 - alphas / 2, axis=0)
        hi = np.quantile(d, 0.5 + alphas / 2, axis=0)
        covered.append((lo <= t) & (t <= hi))
    calibration = np.median(np.abs(np.mean(covered, 0) - alphas[:, None]), 0)
    return float(rmse.mean() + calibration.mean())


_TUNED = {}


def tune(method, n, train_samples, epochs, log=print):
    """Pick (d1, shift) for this N on tuning datasets disjoint from the test ones."""
    key = (method, n, train_samples, epochs, T.SIGMA_MAX)
    path = OUT / f"n{train_samples}_e{epochs}" / f"tuning{T.regime_tag()}" / f"{method}_N{n}.json"
    if key in _TUNED:
        return _TUNED[key]
    if path.exists():
        _TUNED[key] = json.loads(path.read_text())["best"]
        return _TUNED[key]
    if n == 1:
        best = {"d1": 1.0, "shift": 0.0, "criterion": None}
        _TUNED[key] = best
        return best
    global_, _ = approximators(train_samples, epochs)
    if T.SIGMA_MAX is None:
        datasets = [T.simulate(n, 1, seed=777_000 + 1000 * n + i) for i in range(TUNE_DATASETS)]
    else:
        datasets = [tuple(a[None] for a in T.conditioned_draw(n, 777_000 + 1000 * n + i))
                    for i in range(TUNE_DATASETS)]
    truths = [T.globals_to_z(d[0][0]) for d in datasets]
    grid = []
    for d1, shift in itertools.product(D1_GRID, SHIFT_GRID):
        try:
            draws = [_globals(global_, d[2][0], TUNE_SAMPLES, d1, shift, i,
                              method.endswith("diffprior")) for i, d in enumerate(datasets)]
            ok = all(np.isfinite(d).all() for d in draws)
            criterion = _tune_criterion(draws, truths) if ok else float("inf")
        except Exception as error:      # a diverging configuration is just a bad one
            log(f"  tune {method} N={n} d1={d1:g} s={shift:g} failed: {error!r}")
            criterion = float("inf")
        grid.append({"d1": d1, "shift": shift, "criterion": criterion})
        log(f"  tune {method} N={n} d1={d1:g} s={shift:g} crit={criterion:.4f}")
    best = min(grid, key=lambda row: row["criterion"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"best": best, "grid": grid}))
    _TUNED[key] = best
    return best


def sample(method, x, num_samples, seed, train_samples, epochs):
    """Joint draws: (globals physical (S, 4), locals (S, N, 2), seconds)."""
    best = tune(method, len(x), train_samples, epochs)
    global_, local = approximators(train_samples, epochs)
    started = time.perf_counter()
    z = _globals(global_, np.asarray(x), num_samples, best["d1"], best["shift"], seed,
                 method.endswith("diffprior"))
    locals_ = _locals(local, z, np.asarray(x))
    return T.z_to_globals(z), locals_, time.perf_counter() - started
