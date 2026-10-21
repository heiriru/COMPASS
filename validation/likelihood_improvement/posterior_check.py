"""Does a training-recipe change hurt the posterior? Grade posteriors against exact grids.

    .COMPASS/bin/python -m validation.likelihood_improvement.posterior_check \
        --family gauss --seed 0 [--variant loglam]

For 64 fixed test observations of a B5/B6 family (first 64 rows of its test set), the
exact posterior p(theta | x) is computed on a 2-D grid (prior N(0, I) x exact
likelihood). Two things are graded, the same way for every network:

* posterior *samples* from ``SBIm.sample`` (default sampler: DPM + churn, 50 steps):
  per-dim mean error and sd ratio in exact-posterior sds, and the 1-D Wasserstein
  distance of each marginal in sds;
* the PF-ODE posterior *density* log p(theta* | x) at the true theta (posterior mask),
  against the exact density smoothed by sigma(eps) (2-D Gauss-Hermite in theta).

Output: results/posterior/<family>_s<seed>[_variant].pt
"""

import validation.common.runtime_gpu  # noqa: F401

import argparse
import math
import os

import numpy as np
import torch

from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm
from validation.b5_b6_pfode_accuracy.evaluate import test_set
from validation.b5_b6_pfode_accuracy.families import Family, THETA_DIM, X_DIM
from validation.b5_b6_pfode_accuracy.train import model_dir

OUT = os.path.join(os.path.dirname(__file__), "results", "posterior")
N_OBS, N_DRAWS = 64, 2000
GRID = torch.linspace(-5, 5, 401, dtype=torch.float64)       # spacing 0.025
POST_MASK = torch.tensor([0.0] * THETA_DIM + [1.0] * X_DIM)


def exact_grid_posterior(family, x):
    """log posterior on GRID x GRID for each row of x: (n, G, G), normalised."""
    g1, g2 = torch.meshgrid(GRID, GRID, indexing="ij")
    th = torch.stack([g1.reshape(-1), g2.reshape(-1)], 1)
    dA = float(GRID[1] - GRID[0]) ** 2
    out = []
    for xi in x.double():
        lp = family.log_lik(th, xi.expand(len(th), -1), 0.0) - 0.5 * (th ** 2).sum(1) \
            - math.log(2 * math.pi)
        lp = lp - torch.logsumexp(lp, 0) - math.log(dA)
        out.append(lp.reshape(len(GRID), len(GRID)))
    return torch.stack(out)


def smoothed_density_at(logpost, theta, sigma):
    """log of (posterior * N(0, sigma^2 I))(theta), bilinear on the grid + 2-D GH."""
    xi, w = np.polynomial.hermite.hermgauss(8)
    xi, w = torch.tensor(xi), torch.tensor(w) / math.sqrt(math.pi)
    vals = []
    for lp, th in zip(logpost, theta.double()):
        acc = 0.0
        for a, wa in zip(xi, w):
            for b, wb in zip(xi, w):
                pt = th + math.sqrt(2) * sigma * torch.stack([a, b])
                acc = acc + wa * wb * _bilinear(lp.exp(), pt)
        vals.append(math.log(max(float(acc), 1e-300)))
    return torch.tensor(vals, dtype=torch.float64)


def _bilinear(img, pt):
    h = float(GRID[1] - GRID[0])
    u = (pt - GRID[0]) / h
    i, j = int(u[0].floor()), int(u[1].floor())
    if not (0 <= i < len(GRID) - 1 and 0 <= j < len(GRID) - 1):
        return torch.tensor(0.0, dtype=torch.float64)
    fu, fv = u[0] - i, u[1] - j
    return ((1 - fu) * (1 - fv) * img[i, j] + fu * (1 - fv) * img[i + 1, j]
            + (1 - fu) * fv * img[i, j + 1] + fu * fv * img[i + 1, j + 1])


def exact_grid_draws(logpost, n, generator):
    """Draws from the grid posterior (cell + uniform jitter): the Monte-Carlo floor of
    every sample-based metric at the same number of draws."""
    h = float(GRID[1] - GRID[0])
    G = len(GRID)
    out = []
    for lp in logpost:
        idx = torch.multinomial(torch.softmax(lp.reshape(-1), 0), n, replacement=True,
                                generator=generator)
        i, j = idx // G, idx % G
        jit = (torch.rand(n, 2, generator=generator, dtype=torch.float64) - 0.5) * h
        out.append(torch.stack([GRID[i], GRID[j]], 1) + jit)
    return torch.stack(out).float()


def marginal_stats(logpost):
    p = logpost.exp()
    h = float(GRID[1] - GRID[0])
    m1, m2 = p.sum(2) * h, p.sum(1) * h                  # marginals over theta_1, theta_2
    stats = []
    for m in (m1, m2):
        m = m / (m.sum(1, keepdim=True) * h)
        mean = (m * GRID).sum(1) * h
        sd = ((m * (GRID - mean[:, None]) ** 2).sum(1) * h).sqrt()
        cdf = torch.cumsum(m * h, 1) - 0.5 * m * h      # CDF at the grid points (midpoint rule)
        stats.append((mean, sd, cdf))
    return stats


def w1_from_samples(samples, cdf):
    """W1 between an empirical 1-D sample and the grid CDF (per row)."""
    out = []
    for s, c in zip(samples, cdf):
        emp = torch.searchsorted(torch.sort(s.double()).values, GRID, right=True).double() / len(s)
        out.append(float((emp - c).abs().sum() * (GRID[1] - GRID[0])))
    return torch.tensor(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="")
    parser.add_argument("--budget", type=int, default=65536)
    args = parser.parse_args()
    tag = (f"{args.family}_s{args.seed}" + (f"_{args.variant}" if args.variant else "")
           + ("" if args.budget == 65536 else f"_n{args.budget}"))
    path = os.path.join(OUT, f"{tag}.pt")
    if os.path.exists(path):
        print(f"exists: {path}")
        return
    os.makedirs(OUT, exist_ok=True)

    family = Family(args.family, 0.5)
    theta, x = test_set(family)
    theta, x = theta[:N_OBS], x[:N_OBS]
    model = SBIm.load(os.path.join(model_dir(args.family, 0.5, args.budget, args.seed, args.variant),
                                   "final.pt"), device="cuda")
    model.model.eval()

    logpost = exact_grid_posterior(family, x)
    stats = marginal_stats(logpost)
    torch.manual_seed(1234)
    # Chunks of 16 observations keep the larger architectures within GPU memory.
    draws = torch.cat([model.sample(x=x[i:i + 16], condition_mask=POST_MASK,
                                    num_samples=N_DRAWS, timesteps=50, device="cuda",
                                    verbose=False).cpu() for i in range(0, len(x), 16)])
    exact_draws = exact_grid_draws(logpost, N_DRAWS, torch.Generator().manual_seed(99))
    res = {"theta": theta, "x": x}
    for d, (mean, sd, cdf) in enumerate(stats):
        for name, smp in (("", draws), ("floor_", exact_draws)):
            s = smp[:, :, d]
            res[f"{name}mean_err_sd_{d}"] = (s.double().mean(1) - mean) / sd
            res[f"{name}sd_ratio_{d}"] = s.double().std(1) / sd
            res[f"{name}w1_sd_{d}"] = w1_from_samples(s, cdf) / sd
    sigma = float(model.sde.marginal_prob_std(torch.tensor(1e-3)))
    exact_at = smoothed_density_at(logpost, theta, sigma)
    endpoint = "fixed" if getattr(model, "base_moments", None) is not None else "isotropic"
    pf = model.log_prob(torch.cat([theta, x], 1), POST_MASK, device="cuda", timesteps=100,
                        endpoint=endpoint).double()
    res["density_err"] = pf - exact_at
    torch.save(res, path)
    print(f"{tag}: |mean err| {torch.cat([res['mean_err_sd_0'], res['mean_err_sd_1']]).abs().mean():.3f} sd, "
          f"W1 {torch.cat([res['w1_sd_0'], res['w1_sd_1']]).mean():.3f} sd "
          f"(exact-draw floor {torch.cat([res['floor_w1_sd_0'], res['floor_w1_sd_1']]).mean():.3f}), "
          f"density RMSE {res['density_err'].pow(2).mean().sqrt():.3f}")


if __name__ == "__main__":
    main()
