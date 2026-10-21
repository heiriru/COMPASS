"""Can the COMPASS legacy trunk represent latent-latent interactions?

In ``ConditionTransformer`` every block uses ``key_padding_mask = 1 - c``, so
latent tokens never attend to other latent tokens. Latent-latent information
can only flow through the final flattened linear read-out (which alone would
make the score *additive*, s1 = f(th1) + g(th2)) and through the joint
LayerNorm statistics. This script measures, on a trained Two Moons checkpoint,
how non-additive the learned posterior score actually is, and compares it with
the exact noise-diffused Two Moons posterior at the same noise level.

Interaction fraction of s1 on a grid (two-way ANOVA without replication):
    s1(i, j) = mu + a_i + b_j + r_ij,   frac = sum r^2 / sum (s1 - mu)^2,
weighted by the diffused posterior density so only the region with mass
counts. An additive score has frac = 0.

CPU only; no GPU is touched.
"""

import os
import sys

CPU_THREAD_LIMIT = 3
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_avail = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, set(_avail[(os.getpid() * 3 + i) % len(_avail)] for i in range(CPU_THREAD_LIMIT)))
for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_n] = str(CPU_THREAD_LIMIT)

import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from compass import ScoreBasedInferenceModel as SBIm  # noqa: E402

R_MEAN, R_STD = 0.1, 0.01


def two_moons_loglik(t1, t2, x):
    """Exact sbibm Two Moons likelihood p(x | theta) on a grid."""
    s0 = -np.abs(t1 + t2) / math.sqrt(2.0)
    s1 = (-t1 + t2) / math.sqrt(2.0)
    u = x[0] - s0 - 0.25
    v = x[1] - s1
    r = np.sqrt(u ** 2 + v ** 2)
    ll = -0.5 * ((r - R_MEAN) / R_STD) ** 2 - np.log(R_STD * math.sqrt(2 * math.pi)) \
        - math.log(math.pi) - np.log(np.maximum(r, 1e-12))
    return np.where(u >= 0, ll, -np.inf)


def gaussian_blur_logdensity(logp, h, sigma):
    """log of (p * N(0, sigma^2 I)) on a regular grid with spacing h (FFT convolution)."""
    p = np.exp(logp - logp[np.isfinite(logp)].max())
    p[~np.isfinite(p)] = 0.0
    n = p.shape[0]
    k = np.arange(-n // 2, n // 2) * h
    g = np.exp(-0.5 * (k / sigma) ** 2)
    g /= g.sum()
    from numpy.fft import fft2, ifft2, ifftshift
    G = np.outer(g, g)
    q = np.real(ifft2(fft2(p) * fft2(ifftshift(G))))
    return np.log(np.maximum(q, 1e-300))


def interaction_fraction(s, w):
    """Weighted two-way ANOVA: fraction of s's variance that is non-additive."""
    w = w / w.sum()
    mu = (w * s).sum()
    wi, wj = w.sum(1), w.sum(0)
    a = ((w * s).sum(1) / np.maximum(wi, 1e-300)) - mu
    b = ((w * s).sum(0) / np.maximum(wj, 1e-300)) - mu
    # Backfit a few rounds: with non-uniform weights the one-shot means are not the LS fit.
    for _ in range(50):
        a = ((w * (s - mu - b[None, :])).sum(1) / np.maximum(wi, 1e-300))
        b = ((w * (s - mu - a[:, None])).sum(0) / np.maximum(wj, 1e-300))
    r = s - mu - a[:, None] - b[None, :]
    return float((w * r ** 2).sum() / (w * (s - mu) ** 2).sum())


def main(ckpt_dir, x_obs=(-0.07, 0.0), n=161, t_values=(0.01, 0.03, 0.1)):
    z = np.load(os.path.join(ckpt_dir, "zscore.npz"))
    mean, std = z["mean"], z["std"]
    model = SBIm.load(os.path.join(ckpt_dir, "Model_checkpoint.pt"), device="cpu")
    model.model.eval()

    grid = np.linspace(-1.0, 1.0, n)
    h = grid[1] - grid[0]
    T1, T2 = np.meshgrid(grid, grid, indexing="ij")
    x = np.array(x_obs)
    loglik = two_moons_loglik(T1, T2, x)                   # flat prior on [-1, 1]^2

    print(f"checkpoint: {ckpt_dir}\nx_obs = {x_obs}, grid {n}x{n} on [-1,1]^2")
    print(f"{'t':>6} {'sigma_z':>8} {'sigma_theta':>11} | {'true frac':>9} {'COMPASS frac':>12} | {'rel err s1':>10} {'rel err s2':>10}")
    for t in t_values:
        tt = torch.tensor([t], dtype=torch.float32)
        sigma_z = float(model.sde.marginal_prob_std(tt))
        sig_th = sigma_z * std[:2]                         # per-dimension in theta units
        sig = float(np.mean(sig_th))
        # Exact diffused posterior and its score (finite differences on the fine grid).
        logq = gaussian_blur_logdensity(loglik, h, sig)
        s1_true = np.gradient(logq, h, axis=0)
        w = np.exp(logq - logq.max())
        w[w < 1e-4] = 0.0                                   # region with posterior mass

        # COMPASS score at the same noise level (z-space -> theta units: divide by std).
        th = np.stack([T1.ravel(), T2.ravel()], 1)
        zin = np.concatenate([(th - mean[:2]) / std[:2],
                              np.repeat(((x - mean[2:]) / std[2:])[None], len(th), 0)], 1)
        zin = torch.tensor(zin, dtype=torch.float32)
        c = torch.tensor([0.0, 0.0, 1.0, 1.0]).expand(len(th), 4)
        with torch.no_grad():
            out = []
            for k in range(0, len(zin), 4096):
                o = model.model(x=zin[k:k + 4096], t=tt.expand(len(zin[k:k + 4096]), 1), c=c[k:k + 4096])
                out.append(model.output_scale_function(tt.expand(len(o), 1), o))
            sc = torch.cat(out).numpy()
        s1_model = (sc[:, 0] / std[0]).reshape(n, n)
        s2_model = (sc[:, 1] / std[1]).reshape(n, n)
        s2_true = np.gradient(logq, h, axis=1)
        ww = w / w.sum()
        e1 = float((ww * (s1_model - s1_true) ** 2).sum() / (ww * s1_true ** 2).sum())
        e2 = float((ww * (s2_model - s2_true) ** 2).sum() / (ww * s2_true ** 2).sum())

        f_true = interaction_fraction(s1_true, w)
        f_model = interaction_fraction(s1_model, w)
        print(f"{t:6.4f} {sigma_z:8.4f} {sig:11.4f} | {f_true:9.3f} {f_model:12.3f} | {e1:10.3f} {e2:10.3f}")


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 161
    ts = tuple(float(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0.01, 0.03, 0.1)
    main(sys.argv[1], n=n, t_values=ts)
