"""Why does Arruda + diffused prior miss the n = 128 hierarchical GM posterior, and what fixes it?

Data: the `hier_g1.2` datasets (one true g = 1.2, nested sequence). Network: `sym_inprior`.
All arms use the exact diffused prior score. Each arm changes one thing:

    baseline      learned local scores, 10-item mini-batch, two_step_adaptive, d = 1   (the plotted method)
    full_batch    every observation at every step                                     -> mini-batch noise?
    corrector     + 3 Langevin corrector steps per step                                -> sampler / mixing?
    paper_damping d0 = 1, d1 = 0.1 (Arruda et al.'s schedule)                          -> damping?
    exact_local   exact single-observation scores instead of the network               -> rule vs network?
    stop_t*       stop the reverse diffusion at noise level t and rescale by 1/alpha: never uses the
                  low-noise scores, where the network is badly wrong (score_error_vs_t.py) -> low-t error?
    grid_*        1-D global: integrate the composed score at noise level t over a grid of g and
                  sample that density directly -- no reverse-diffusion path at all     -> path error?
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "Arruda")]
import argparse as _argparse
_cli = _argparse.ArgumentParser()
_cli.add_argument("--weights-tag", default="sym_inprior", help="trained GM network under Arruda/artifacts/gm/")
_cli.add_argument("--ns", type=int, nargs="+", default=[32, 64, 128])
_cli.add_argument("--skip", nargs="*", default=[], help="arm names to skip")
CLI = _cli.parse_args()
sys.argv = [sys.argv[0]]
import run_arruda as R  # applies the 3-thread CPU cap on import
import torch
from keras import ops

MU0, SIGMA_X, G_TRUE, SEED, DRAWS = 1.5, 6.0, 1.2, 20260923, 4_000
mixture = R.gm_model.Mixture(mu_0=MU0, sigma_x=SIGMA_X, weights=(0.5, 0.5))
args = R.argparse.Namespace(benchmark="gm", weights_dir=str(ROOT / "Arruda/artifacts/gm" / CLI.weights_tag), tag="hier_g1.2",
                            retrain=False, train_samples=16_384, batch_size=64, epochs=0, seed=SEED, num_samples=DRAWS,
                            gm_mu0=MU0, gm_sigma_x=SIGMA_X)
approximator = R.train_or_load(args, mixture)
network = approximator.inference_network
learned_score = network.score
layer = approximator.standardizer.standardize_layers["inference_variables"]
m = float(ops.convert_to_numpy(layer.moving_mean[0]).ravel()[0])
s = float(ops.convert_to_numpy(layer.moving_std(0)).ravel()[0])
prior_fn = R.diffused_prior_score(approximator, mixture.components, mixture.tau_g ** 2, mixture.weights)
tau2, v, means = mixture.tau_g ** 2, mixture.v, mixture.components
q = 1.0 / (1.0 / tau2 + 1.0 / v)


def alpha_sigma(time=None, log_snr_t=None):
    if log_snr_t is None:
        log_snr_t = network.noise_schedule.get_log_snr(t=time, training=False)
    return network.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)


def exact_local_score(xz, time=None, log_snr_t=None, conditions=None, training=False, **kwargs):
    alpha, sigma = alpha_sigma(time, log_snr_t)
    alpha = ops.reshape(alpha, (-1, 1)); sigma = ops.reshape(sigma, (-1, 1))
    x = ops.reshape(ops.cast(conditions, "float32"), (-1, 1))
    post = q * (ops.convert_to_tensor(means[None, :] / tau2, "float32") + x / v)
    log_w = -0.5 * (x - ops.convert_to_tensor(means[None, :], "float32")) ** 2 / (tau2 + v)
    var = alpha ** 2 * (q / s ** 2) + sigma ** 2
    diff = alpha * (post - m) / s - ops.reshape(xz, (-1, 1))
    resp = ops.softmax(log_w - 0.5 * diff ** 2 / var - 0.5 * ops.log(var), axis=-1)
    return ops.sum(resp * diff / var, axis=-1, keepdims=True)


def diffusion(x, *, mini_batch, d1=1.0, corrector=0, chunks=1, stop=0.0):
    draws = []
    extra = {"stop_time": stop} if stop > 0 else {}
    for c in range(chunks):
        out = approximator.compositional_sample(
            num_samples=DRAWS // chunks, conditions={"x": x[None]}, compute_prior_score=prior_fn,
            method="two_step_adaptive", steps="adaptive", mini_batch_size=mini_batch, corrector_steps=corrector,
            compositional_bridge_d0=1.0, compositional_bridge_d1=d1, seed=SEED + c, **extra)
        draws.append(np.asarray(out["g"])[0, :, 0])
        torch.cuda.empty_cache()
    values = np.concatenate(draws)
    if stop > 0:
        # BayesFlow un-standardises the noisy state z_t; undo alpha so it lives on the g scale.
        alpha = float(np.asarray(ops.convert_to_numpy(alpha_sigma(time=ops.convert_to_tensor([[stop]], "float32"))[0])).ravel()[0])
        values = ((values - m) / s) / alpha * s + m
    return values


def grid_density(x, t, score_fn):
    """log p(g | X) up to a constant, from the composed score at noise level t integrated over g.

    Evaluated on clean-g points inside the prior's support (|g| <= mu_0 + 4 tau_g), mapped to the
    noisy coordinate z = alpha (g - m) / s, so the network is never queried far out in the tails.
    """
    alpha, sigma = (float(np.asarray(ops.convert_to_numpy(u)).ravel()[0]) for u in alpha_sigma(time=ops.convert_to_tensor([[t]], "float32")))
    span = MU0 + 4 * mixture.tau_g
    zg = (alpha * (np.linspace(-span, span, 801) - m) / s).astype("float32")
    n = len(x)
    zz = np.repeat(zg, n)[:, None]; xx = np.tile(x, (len(zg), 1)); tt = ops.full((len(zz), 1), t)
    local = np.concatenate([ops.convert_to_numpy(score_fn(ops.convert_to_tensor(zz[i:i + 65_536]), time=tt[i:i + 65_536],
                                                            conditions=ops.convert_to_tensor(xx[i:i + 65_536])))
                            for i in range(0, len(zz), 65_536)]).reshape(len(zg), n)
    prior = prior_fn({"g": (zg * s + m)[:, None]}, np.full((len(zg), 1), t, "float32"))["g"].ravel() * s
    composed = local.sum(1) - (n - 1) * prior                                    # score in z at noisy level t
    # The composed field at t is (approximately) the score of the posterior diffused to t; undo the
    # alpha scaling so the density lives on g: g = (z / alpha) * s + m.
    logp = np.concatenate([[0.0], np.cumsum(0.5 * (composed[1:] + composed[:-1]) * np.diff(zg))])
    g = (zg / alpha) * s + m
    p = np.exp(logp - logp.max()); p /= np.trapezoid(p, g)
    return g, p, sigma * s / alpha                                                # smoothing width in g units


def sample_grid(g, p, rng):
    cdf = np.cumsum(p); cdf /= cdf[-1]
    return np.interp(rng.random(DRAWS), cdf, g)


rows, panels = [], {}
for n in CLI.ns:
    x, grid, exact = R.gm_tall_problem(n, SEED, mixture, "hierarchical", G_TRUE)
    ref = np.random.default_rng(SEED + 10 * n).choice(grid, size=50_000, p=exact / exact.sum())
    p_exact = float(np.trapezoid(np.where(grid > 0, exact, 0), grid))
    arms = {}
    chunks = max(1, n // 8)                      # full batches: DRAWS/chunks samples x n observations per call
    plan = {
        "baseline": lambda: diffusion(x, mini_batch=10),
        "full_batch": lambda: diffusion(x, mini_batch=n, chunks=chunks),
        "stop_t0.25": lambda: diffusion(x, mini_batch=10, stop=0.25),
        "stop_t0.3": lambda: diffusion(x, mini_batch=10, stop=0.3),
        **{f"grid_learned_t{t}": (lambda t=t: sample_grid(*grid_density(x, t, learned_score)[:2], np.random.default_rng(SEED)))
           for t in (0.25, 0.3, 0.35)},
    }
    exact_plan = {
        "exact_local": lambda: diffusion(x, mini_batch=10),
        **{f"grid_exact_t{t}": (lambda t=t: sample_grid(*grid_density(x, t, exact_local_score)[:2], np.random.default_rng(SEED)))
           for t in (0.1, 0.25, 0.3)},
    }
    network.score = learned_score
    for name, run in plan.items():
        if name not in CLI.skip:
            arms[name] = run()
    network.score = exact_local_score
    for name, run in exact_plan.items():
        if name not in CLI.skip:
            arms[name] = run()
    network.score = learned_score
    for name, values in arms.items():
        row = dict(n=n, arm=name, P_pos=float(np.mean(values > 0)), exact_P_pos=p_exact,
                   w1_over_sd=float(R.wasserstein_distance(values, ref) / ref.std()),
                   off=float(np.mean(np.abs(values) > grid[-1])))
        rows.append(row)
        print(json.dumps({k: (round(w, 3) if isinstance(w, float) else w) for k, w in row.items()}), flush=True)
    panels[n] = (grid, exact, arms)
OUT = f"hier_improve_{CLI.weights_tag}"
(HERE / f"{OUT}.json").write_text(json.dumps(rows, indent=2) + "\n")
np.savez(HERE / f"{OUT}_samples.npz", **{f"n{n}_{a}": v for n, (_, _, arms) in panels.items() for a, v in arms.items()})
print("saved", flush=True)
