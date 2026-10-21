"""Where does BayesFlow's Arruda sampler lose the GM posterior: the learned score or the rule?

2x2 inside BayesFlow's own `compositional_sample` (two_step_adaptive, d0 = d1 = 1), on the
`sym_inprior` model (mu_0 = 1.5, tau_g = 0.4, sigma_x = 6, symmetric data):

    local score : learned network  | exact diffused single-observation posterior score
    prior score : BayesFlow default ((1 - t) * clean prior score at the noisy point)
                  | exact diffused prior score

Both exact scores are closed form: in standardised z = (g - m) / s the prior and every
single-observation posterior are 2-component Gaussian mixtures, and the variance-preserving
EDM schedule maps a component N(c, w) to N(alpha c, alpha^2 w + sigma^2).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "Arruda")]
FULL_BATCH = "--full-batch" in sys.argv   # every observation at every step instead of BayesFlow's 10-item mini-batch
sys.argv = [sys.argv[0]]
import run_arruda as R  # applies the 3-thread CPU cap on import
import keras
from keras import ops

MU0, SIGMA_X, SEED = 1.5, 6.0, 20260923
DRAWS = 1_000 if FULL_BATCH else 4_000   # full batches at n=128 need the smaller batch to fit an 11 GB card
mixture = R.gm_model.Mixture(mu_0=MU0, sigma_x=SIGMA_X, weights=(0.5, 0.5))
args = R.argparse.Namespace(benchmark="gm", weights_dir="", tag="sym_inprior", retrain=False, train_samples=16_384,
                            batch_size=64, epochs=0, seed=SEED, num_samples=DRAWS, gm_mu0=MU0, gm_sigma_x=SIGMA_X)
approximator = R.train_or_load(args, mixture)
network = approximator.inference_network
layer = approximator.standardizer.standardize_layers["inference_variables"]
m = float(ops.convert_to_numpy(layer.moving_mean[0]).ravel()[0])
s = float(ops.convert_to_numpy(layer.moving_std(0)).ravel()[0])
print(f"standardiser: mean {m:+.4f} std {s:.4f}", flush=True)
tau2, v = mixture.tau_g ** 2, mixture.v
means = np.array([-MU0, MU0])


def mixture_score(z, centres, variances, log_weights, alpha, sigma):
    """d/dz log sum_k w_k N(z; alpha c_k, alpha^2 var_k + sigma^2), everything broadcastable."""
    var = alpha ** 2 * variances + sigma ** 2
    diff = alpha * centres - z
    logits = log_weights - 0.5 * diff ** 2 / var - 0.5 * ops.log(var)
    resp = ops.softmax(logits, axis=-1)
    return ops.sum(resp * diff / var, axis=-1, keepdims=True)


def alpha_sigma(time=None, log_snr_t=None):
    if log_snr_t is None:
        log_snr_t = network.noise_schedule.get_log_snr(t=time, training=False)
    return network.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)


def exact_local_score(xz, time=None, log_snr_t=None, conditions=None, training=False, **kwargs):
    """Diffused score of p(g | x_i) in z-space; `conditions` holds the raw x_i."""
    alpha, sigma = alpha_sigma(time, log_snr_t)
    alpha = ops.reshape(alpha, (-1, 1)); sigma = ops.reshape(sigma, (-1, 1))
    x = ops.reshape(ops.cast(conditions, "float32"), (-1, 1))
    q = 1.0 / (1.0 / tau2 + 1.0 / v)
    post_means = q * (ops.convert_to_tensor(means[None, :] / tau2, "float32") + x / v)           # (B, 2)
    log_w = -0.5 * (x - ops.convert_to_tensor(means[None, :], "float32")) ** 2 / (tau2 + v)       # equal prior weights
    return mixture_score(ops.reshape(xz, (-1, 1)), (post_means - m) / s, q / s ** 2, log_w, alpha, sigma)


def exact_prior_score(values, time):
    """Diffused prior score. BayesFlow passes un-standardised z_t and multiplies our output by s."""
    alpha, sigma = alpha_sigma(time=ops.convert_to_tensor(time, "float32"))
    alpha = ops.reshape(ops.convert_to_tensor(alpha), (-1, 1)); sigma = ops.reshape(ops.convert_to_tensor(sigma), (-1, 1))
    z = (ops.convert_to_tensor(values["g"], "float32") - m) / s
    centres = ops.convert_to_tensor(((means - m) / s)[None, :], "float32")
    score_z = mixture_score(z, centres, tau2 / s ** 2, ops.zeros((1, 2)), alpha, sigma)
    return {"g": ops.convert_to_numpy(score_z / s)}


learned_score = network.score
# Units check: the learned and exact local scores should roughly agree on in-distribution points.
x_test = np.array([[-6.0], [0.0], [4.0]], dtype="float32")
for t in (0.2, 0.5, 0.8):
    z = np.linspace(-2, 2, 5, dtype="float32")[:, None]
    zz = np.repeat(z, 3, 0); xx = np.tile(x_test, (5, 1)); tt = ops.full((15, 1), t)
    a = ops.convert_to_numpy(learned_score(ops.convert_to_tensor(zz), time=tt, conditions=ops.convert_to_tensor(xx)))
    b = ops.convert_to_numpy(exact_local_score(ops.convert_to_tensor(zz), time=tt, conditions=ops.convert_to_tensor(xx)))
    print(f"t={t}: learned vs exact local score  corr {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.4f}  "
          f"max|diff| {np.abs(a - b).max():.3f}  max|exact| {np.abs(b).max():.3f}", flush=True)

def sample(x, prior_fn):
    if not FULL_BATCH:
        return R.sample_one(approximator, x, prior_fn, args, 1.0)
    out = approximator.compositional_sample(num_samples=DRAWS, conditions={"x": x[None, ...]}, compute_prior_score=prior_fn,
                                            method="two_step_adaptive", steps="adaptive", mini_batch_size=len(x),
                                            compositional_bridge_d0=1.0, compositional_bridge_d1=1.0, seed=SEED)
    return np.asarray(out["g"])[0, :, 0]


SUFFIX = "_fullbatch" if FULL_BATCH else ""
rows, curves = [], {}
for n in ((32,) if FULL_BATCH else (8, 32, 128)):   # full batch at n=128 does not fit in 11 GB
    x, grid, exact = R.gm_tall_problem(n, SEED, mixture, "symmetric")
    ref = np.random.default_rng(SEED + 10 * n).choice(grid, size=50_000, p=exact / exact.sum())
    curves[n] = (grid, exact, {})
    for local in ("learned", "exact"):
        network.score = learned_score if local == "learned" else exact_local_score
        for prior in ("bayesflow_clean", "exact_diffused"):
            prior_fn = R.gm_prior_score(mixture) if prior == "bayesflow_clean" else exact_prior_score
            g = sample(x, prior_fn)
            row = dict(n=n, local=local, prior=prior, P_pos=float(np.mean(g > 0)), valley=float(np.mean(np.abs(g) < 0.3)),
                       exact_valley=float(np.trapezoid(np.where(np.abs(grid) < 0.3, exact, 0), grid)),
                       off=float(np.mean(np.abs(g) > grid[-1])), sd=float(g.std()),
                       w1_over_sd=float(R.wasserstein_distance(g, ref) / ref.std()))
            rows.append(row); curves[n][2][(local, prior)] = g
            print(json.dumps({k: (round(w, 4) if isinstance(w, float) else w) for k, w in row.items()}), flush=True)
network.score = learned_score
(HERE / f"gm_oracle_arruda{SUFFIX}.json").write_text(json.dumps(rows, indent=2) + "\n")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(len(curves), 4, figsize=(16, 3.2 * len(curves)), constrained_layout=True)
for i, (n, (grid, exact, sampled)) in enumerate(curves.items()):
    bins = np.linspace(grid[0], grid[-1], 81)
    for j, key in enumerate([(l, p) for l in ("learned", "exact") for p in ("bayesflow_clean", "exact_diffused")]):
        ax, g = axes[i, j], sampled[key]
        ax.plot(grid, exact, color="black", lw=2, label="exact posterior")
        ax.hist(g, bins=bins, weights=np.full(len(g), 1 / (len(g) * (bins[1] - bins[0]))), histtype="step", lw=1.5,
                color="#e76f51", label=f"Arruda d=1 ({np.mean(np.abs(g) > grid[-1]):.0%} off-plot)")
        for k, c in enumerate((-MU0, MU0)):
            ax.axvline(c, color="0.4", ls=":", lw=1.1, label="prior modes" if k == 0 else None)
        ax.set(title=f"n={n}  local: {key[0]}  prior: {key[1]}", xlabel="g")
        ax.legend(fontsize=7, loc="upper right")
fig.suptitle("GM in-prior: BayesFlow Arruda sampler, learned vs exact scores", y=1.01)
fig.savefig(HERE / f"gm_oracle_arruda{SUFFIX}.png", dpi=150, bbox_inches="tight")
print("saved", HERE / f"gm_oracle_arruda{SUFFIX}.png", flush=True)
