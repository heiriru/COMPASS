"""At each noise level: the density Arruda's composed score integrates to vs the true diffused posterior.

Both scores exact (closed form), BayesFlow's EDM variance-preserving schedule, g standardised by s.
composed(z) = sum_i s_i,t(z) - (n-1) s_prior,t(z) is exact only at t = 0.
"""
import os
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:3]))
import json, sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, expit
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "Gaussian_mixtures_compositional_score_modelling")]
from Gaussian_mixtures_compositional_score_modelling import model as M, datasets as D
import matplotlib.pyplot as plt

S, SEED, SIGMA_X = 1.5523, 20260923, 6.0          # s = the trained standardiser's std; any s gives the same picture
smin, smax, rho = 1e-4, 80.0, 7

def alpha_sigma(t):
    lam = -2 * rho * np.log(smax ** (1 / rho) + (1 - t) * (smin ** (1 / rho) - smax ** (1 / rho)))
    return np.sqrt(expit(lam)), np.sqrt(expit(-lam))

def log_mix(z, c, var, logw, a, sg):            # log sum_k w_k N(z; a c_k, a^2 var_k + sg^2); z (G,1), c (...,K)
    v = a ** 2 * var + sg ** 2
    return logsumexp(logw - 0.5 * (z - a * c) ** 2 / v - 0.5 * np.log(2 * np.pi * v), axis=-1)

zg = np.linspace(-3, 3, 3001)[:, None]
fig, axes = plt.subplots(2, 5, figsize=(20, 6.5), constrained_layout=True)
rows = []
for r, (target, mu0) in enumerate((("unimodal", 0.0), ("bimodal", 1.5))):
    mix = M.Mixture(mu_0=mu0, sigma_x=SIGMA_X); n = 128
    rng = np.random.default_rng(SEED + n)
    x = (0.3 + np.sqrt(mix.v) * rng.standard_normal(n)) if target == "unimodal" else D.make_dataset("symmetric", n, rng, mixture=mix)["x"]
    tau2, v = mix.tau_g ** 2, mix.v; q = 1 / (1 / tau2 + 1 / v); mus = mix.components
    w, pm, pv = mix.global_posterior(x)
    for c, t in enumerate((0.9, 0.7, 0.5, 0.3, 0.05)):
        a, sg = alpha_sigma(t)
        exact = log_mix(zg, (np.asarray(pm) / S)[None], pv / S ** 2, np.log(w)[None], a, sg)
        prior = log_mix(zg, (mus / S)[None], tau2 / S ** 2, np.log([.5, .5])[None], a, sg)
        post_c = (q * (mus[None] / tau2 + x[:, None] / v)) / S                          # (n, 2) local branch centres
        post_lw = -0.5 * (x[:, None] - mus[None]) ** 2 / (tau2 + v)
        post_lw -= logsumexp(post_lw, axis=1, keepdims=True)
        local = sum(log_mix(zg, post_c[i][None], q / S ** 2, post_lw[i][None], a, sg) for i in range(n))
        composed = local - (n - 1) * prior                                                 # log density up to a constant
        pe, pc = np.exp(exact - exact.max()), np.exp(composed - composed.max())
        pe /= np.trapezoid(pe, zg[:, 0]); pc /= np.trapezoid(pc, zg[:, 0])
        modes = lambda p: int(np.sum((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] > 1e-3 * p.max())))
        sd = lambda p: float(np.sqrt(np.trapezoid(p * zg[:, 0] ** 2, zg[:, 0]) - np.trapezoid(p * zg[:, 0], zg[:, 0]) ** 2))
        rows.append(dict(target=target, t=t, sigma=round(float(sg), 3), modes_exact=modes(pe), modes_composed=modes(pc),
                         sd_ratio=round(sd(pc) / sd(pe), 3), tv=round(float(0.5 * np.trapezoid(np.abs(pc - pe), zg[:, 0])), 3)))
        print(json.dumps(rows[-1]), flush=True)
        ax = axes[r, c]
        ax.plot(zg * S, pe, "k", lw=2, label="exact diffused posterior"); ax.plot(zg * S, pc, color="#e76f51", lw=1.6, ls="--", label="Arruda composed")
        ax.set(title=f"{target}, t={t} (noise sd {sg:.2f})", xlabel="g-scale coordinate z*s"); ax.legend(fontsize=7)
fig.suptitle("n = 128, both scores exact: density of the composed score vs the true diffused posterior", y=1.02)
fig.savefig(HERE / "composed_vs_exact_field.png", dpi=140, bbox_inches="tight")
(HERE / "composed_vs_exact_field.json").write_text(json.dumps(rows, indent=2) + "\n")
