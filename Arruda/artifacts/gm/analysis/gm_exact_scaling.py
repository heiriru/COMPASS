"""Why does Arruda's sampler work to n = 100,000 in the paper but not perfectly at n = 128 here?

Same BayesFlow `compositional_sample` (two_step_adaptive), both scores exact and diffused, so the
only thing that varies is the *target*:

    unimodal : mu_0 = 0 (the mixture prior collapses to N(0, 0.4^2)), data from g_true = 0.3
               -- the paper's controlled benchmark is of this Gaussian kind
    bimodal  : mu_0 = 1.5, symmetric data (T = 0), exact posterior two equal branches

sigma_x = 6 and tau_g = 0.4 in both. Mini-batch = 10% of n (the paper's recommendation), and two
bridges: d0 = d1 = 1 (tutorial default) and d0 = 1, d1 = 0.1 (the paper's damping schedule).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "Arruda")]
CHECK = "--minibatch-check" in sys.argv   # unimodal n=1000 only: 10% mini-batch vs every observation, 3 seeds
sys.argv = [sys.argv[0]]
import run_arruda as R  # applies the 3-thread CPU cap on import
from keras import ops

SEED, DRAWS, SIGMA_X, G_TRUE = 20260923, 2_000, 6.0, 0.3
# The trained sym_inprior network is only loaded to get a built approximator; its score is replaced.
base = R.gm_model.Mixture(mu_0=1.5, sigma_x=SIGMA_X, weights=(0.5, 0.5))
args = R.argparse.Namespace(benchmark="gm", weights_dir="", tag="sym_inprior", retrain=False, train_samples=16_384,
                            batch_size=64, epochs=0, seed=SEED, num_samples=DRAWS, gm_mu0=1.5, gm_sigma_x=SIGMA_X)
approximator = R.train_or_load(args, base)
network = approximator.inference_network
layer = approximator.standardizer.standardize_layers["inference_variables"]
m = float(ops.convert_to_numpy(layer.moving_mean[0]).ravel()[0])
s = float(ops.convert_to_numpy(layer.moving_std(0)).ravel()[0])


def alpha_sigma(time=None, log_snr_t=None):
    if log_snr_t is None:
        log_snr_t = network.noise_schedule.get_log_snr(t=time, training=False)
    return network.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)


def mixture_score(z, centres, variances, log_weights, alpha, sigma):
    var = alpha ** 2 * variances + sigma ** 2
    diff = alpha * centres - z
    resp = ops.softmax(log_weights - 0.5 * diff ** 2 / var - 0.5 * ops.log(var), axis=-1)
    return ops.sum(resp * diff / var, axis=-1, keepdims=True)


def make_scores(mixture):
    means = mixture.components; tau2 = mixture.tau_g ** 2; v = mixture.v
    q = 1.0 / (1.0 / tau2 + 1.0 / v)

    def local(xz, time=None, log_snr_t=None, conditions=None, training=False, **kwargs):
        alpha, sigma = alpha_sigma(time, log_snr_t)
        alpha = ops.reshape(alpha, (-1, 1)); sigma = ops.reshape(sigma, (-1, 1))
        x = ops.reshape(ops.cast(conditions, "float32"), (-1, 1))
        post = q * (ops.convert_to_tensor(means[None, :] / tau2, "float32") + x / v)
        log_w = -0.5 * (x - ops.convert_to_tensor(means[None, :], "float32")) ** 2 / (tau2 + v)
        return mixture_score(ops.reshape(xz, (-1, 1)), (post - m) / s, q / s ** 2, log_w, alpha, sigma)

    def prior(values, time):
        alpha, sigma = alpha_sigma(time=ops.convert_to_tensor(time, "float32"))
        alpha = ops.reshape(ops.convert_to_tensor(alpha), (-1, 1)); sigma = ops.reshape(ops.convert_to_tensor(sigma), (-1, 1))
        z = (ops.convert_to_tensor(values["g"], "float32") - m) / s
        centres = ops.convert_to_tensor(((means - m) / s)[None, :], "float32")
        return {"g": ops.convert_to_numpy(mixture_score(z, centres, tau2 / s ** 2, ops.zeros((1, 2)), alpha, sigma) / s)}
    return local, prior


if CHECK:
    mixture = R.gm_model.Mixture(mu_0=0.0, sigma_x=SIGMA_X, weights=(0.5, 0.5)); local, prior = make_scores(mixture)
    network.score = local; n = 1_000
    x = (G_TRUE + np.sqrt(mixture.v) * np.random.default_rng(SEED + n).standard_normal(n)).reshape(n, 1)
    _, post_means, post_var = mixture.global_posterior(x.ravel())
    for batch, clip in ((n, (-3, 3)), (n, None)):
        for seed in (SEED, 1):
            g = np.asarray(approximator.compositional_sample(num_samples=DRAWS, conditions={"x": x[None].astype("float32")},
                           compute_prior_score=prior, method="two_step_adaptive", steps="adaptive", mini_batch_size=batch,
                           compositional_bridge_d0=1.0, compositional_bridge_d1=1.0, seed=seed, clip=clip)["g"])[0, :, 0]
            print(json.dumps(dict(n=n, mini_batch=batch, clip=clip, seed=seed, mean_err_over_sd=round(float((g.mean() - post_means[0]) / np.sqrt(post_var)), 3),
                                  sd_ratio=round(float(g.std() / np.sqrt(post_var)), 3))), flush=True)
    sys.exit()

rows = []
for target, mu0, ns in (("unimodal", 0.0, (128, 1_000, 10_000, 100_000)), ("bimodal", 1.5, (128, 1_000))):
    mixture = R.gm_model.Mixture(mu_0=mu0, sigma_x=SIGMA_X, weights=(0.5, 0.5))
    local, prior = make_scores(mixture)
    network.score = local
    for n in ns:
        rng = np.random.default_rng(SEED + n)
        if target == "unimodal":
            x = (G_TRUE + np.sqrt(mixture.v) * rng.standard_normal(n)).reshape(n, 1)
        else:
            x = np.asarray(R.gm_datasets.make_dataset("symmetric", n, rng, mixture=mixture)["x"]).reshape(n, 1)
        weights, post_means, post_var = mixture.global_posterior(x.ravel())
        sd_branch = float(np.sqrt(post_var))
        lo = min(post_means) - 8 * sd_branch; hi = max(post_means) + 8 * sd_branch
        grid = np.linspace(lo, hi, 4_001)
        exact = mixture.global_density(grid, x.ravel()); exact /= np.trapezoid(exact, grid)
        ref = np.random.default_rng(1).choice(grid, size=50_000, p=exact / exact.sum())
        for d1 in (1.0, 0.1):
            out = approximator.compositional_sample(num_samples=DRAWS, conditions={"x": x[None].astype("float32")},
                                                    compute_prior_score=prior, method="two_step_adaptive", steps="adaptive",
                                                    mini_batch_size=max(10, n // 10), compositional_bridge_d0=1.0,
                                                    compositional_bridge_d1=d1, seed=SEED)
            g = np.asarray(out["g"])[0, :, 0]
            row = dict(target=target, n=n, d1=d1, exact_sd=float(ref.std()), sd_ratio=float(g.std() / ref.std()),
                       mean_err_over_sd=float((g.mean() - ref.mean()) / ref.std()),
                       w1_over_sd=float(R.wasserstein_distance(g, ref) / ref.std()),
                       P_pos=float(np.mean(g > 0)) if target == "bimodal" else None,
                       valley=float(np.mean(np.abs(g) < 0.5 * max(post_means))) if target == "bimodal" else None,
                       exact_valley=float(np.trapezoid(np.where(np.abs(grid) < 0.5 * max(post_means), exact, 0), grid))
                       if target == "bimodal" else None)
            rows.append(row)
            print(json.dumps({k: (round(w, 4) if isinstance(w, float) else w) for k, w in row.items()}), flush=True)
(HERE / "gm_exact_scaling.json").write_text(json.dumps(rows, indent=2) + "\n")
