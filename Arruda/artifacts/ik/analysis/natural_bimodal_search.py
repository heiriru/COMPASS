"""Search *natural* IK datasets (nothing selected or forced) for a bimodal ABC reference at n = 128.

Every dataset is a draw from the model: angle_1 ~ N(0, 0.5^2) shared, (height, angle_2, angle_3) ~ prior
per observation, 128 observations through the forward map. For each, the ABC reference of
`hierarchical.reference_global` (Gaussian kernel h = 0.05, local-prior Monte Carlo, log-mean-exp over
draws, sum over observations, plus the angle_1 prior) is computed on the GPU with fewer draws and a
coarser grid as a *screen*. Candidates are then confirmed with the full-accuracy reference and
several Monte Carlo seeds, because MC noise alone can fake a second mode.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:3]))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "inverse_kinematics")]
import hierarchical as H

L1, L2, L3, BANDWIDTH, N = 0.5, 0.5, 1.0, 0.05, 128


def forward(a1, locals_):
    """Node order: a1 (G,), locals (D, 3) = (height, angle_2, angle_3) -> (G, D, 2)."""
    h, b2, b3 = locals_[:, 0], locals_[:, 1], locals_[:, 2]
    a1 = a1[:, None]
    s1, s2, s3 = a1, a1 + b2[None], a1 + b2[None] + b3[None]
    x1 = L1 * torch.sin(s1) + L2 * torch.sin(s2) + L3 * torch.sin(s3) + h[None]
    x2 = L1 * torch.cos(s1) + L2 * torch.cos(s2) + L3 * torch.cos(s3)
    return torch.stack([x1, x2], dim=-1)


def log_posterior(x, grid, draws, seed, device, chunk=8):
    """Same estimator as hierarchical.reference_global, in float64 on the GPU."""
    rng = np.random.default_rng(seed)
    locals_ = torch.as_tensor(rng.normal(H.LOCAL_PRIOR_MEAN, H.LOCAL_PRIOR_STD, size=(draws, 3)), device=device)
    x = torch.as_tensor(x, device=device)
    out = torch.empty(len(grid), dtype=torch.float64, device=device)
    for start in range(0, len(grid), chunk):
        g = torch.as_tensor(grid[start:start + chunk], device=device)
        pred = forward(g, locals_)                                                   # (c, D, 2)
        sq = ((pred[:, :, None, :] - x[None, None]) ** 2).sum(-1)                   # (c, D, n)
        log_kernel = torch.logsumexp(-sq / (2 * BANDWIDTH ** 2), dim=1) - np.log(draws)
        out[start:start + chunk] = log_kernel.sum(-1)
    out = out - 0.5 * (torch.as_tensor(grid, device=device) / H.GLOBAL_PRIOR_STD[0]) ** 2
    return (out - out.max()).cpu().numpy()


def modes(grid, logp, depth=1.0, window=8.0):
    """Local maxima within `window` nats of the top, separated from it by a dip of >= `depth` nats."""
    peaks = [i for i in range(1, len(grid) - 1) if logp[i] >= logp[i - 1] and logp[i] >= logp[i + 1] and logp[i] > -window]
    top = int(np.argmax(logp))
    kept = [top]
    for i in sorted(peaks, key=lambda k: -logp[k]):
        if i == top:
            continue
        lo, hi = sorted((i, top))
        if min(logp[i], logp[top]) - logp[lo:hi + 1].min() >= depth:
            kept.append(i)
    return sorted(kept)


def summary(grid, logp):
    p = np.exp(logp); p /= np.trapezoid(p, grid)
    peaks = modes(grid, logp)
    second = None
    if len(peaks) > 1:
        top = int(np.argmax(logp))
        other = max((i for i in peaks if i != top), key=lambda i: logp[i])
        cut = min(top, other) + int(np.argmin(logp[min(top, other):max(top, other) + 1]))
        side = grid > grid[cut] if grid[other] > grid[cut] else grid < grid[cut]
        second = dict(location=float(grid[other]), mass=float(np.trapezoid(np.where(side, p, 0), grid)),
                      log_height=float(logp[other]))
    mode = float(grid[np.argmax(logp)])
    # log p(-mode) - log p(mode): how far the mirror image of the mode is from being a second mode.
    mirror = float(np.interp(-mode, grid, logp))
    return dict(peaks=[float(grid[i]) for i in peaks], mode=mode, mirror_log_ratio=mirror, second=second)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--datasets", type=int, default=2_000)
    cli.add_argument("--screen-draws", type=int, default=20_000)
    cli.add_argument("--confirm-draws", type=int, default=200_000)
    cli.add_argument("--min-second-mass", type=float, default=0.05)
    cli.add_argument("--seed", type=int, default=20260929)
    args = cli.parse_args()
    device = "cuda"
    screen_grid = np.linspace(-2.0, 2.0, 321)
    rows, candidates = [], []
    for k in range(args.datasets):
        rng = np.random.default_rng([args.seed, k])
        a1 = float(rng.normal(0.0, H.GLOBAL_PRIOR_STD[0]))
        locals_ = rng.normal(H.LOCAL_PRIOR_MEAN, H.LOCAL_PRIOR_STD, size=(N, 3))
        x = H.forward(np.concatenate([np.full((N, 1), a1), locals_], axis=1))
        s = summary(screen_grid, log_posterior(x, screen_grid, args.screen_draws, seed=k, device=device))
        row = dict(dataset=k, true_angle=a1, **s)
        rows.append(row)
        if s["second"] and s["second"]["mass"] >= args.min_second_mass:
            candidates.append(row)
            print(f"candidate {k}: true {a1:+.3f} peaks {np.round(s['peaks'], 3)} second-mode mass {s['second']['mass']:.3f}", flush=True)
        if (k + 1) % 250 == 0:
            multi = sum(1 for r in rows if len(r["peaks"]) > 1)
            print(f"screened {k + 1}: {multi} with >1 local peak (any mass), {len(candidates)} candidates", flush=True)
    (HERE / "natural_bimodal_screen.json").write_text(json.dumps(rows, indent=2) + "\n")

    confirmed = []
    fine_grid = np.linspace(-2.0, 2.0, 1_401)
    for row in candidates:
        rng = np.random.default_rng([args.seed, row["dataset"]])
        a1 = float(rng.normal(0.0, H.GLOBAL_PRIOR_STD[0]))
        locals_ = rng.normal(H.LOCAL_PRIOR_MEAN, H.LOCAL_PRIOR_STD, size=(N, 3))
        x = H.forward(np.concatenate([np.full((N, 1), a1), locals_], axis=1))
        checks = [summary(fine_grid, log_posterior(x, fine_grid, args.confirm_draws, seed=1000 + j, device=device)) for j in range(3)]
        masses = [c["second"]["mass"] if c["second"] else 0.0 for c in checks]
        ok = all(m >= args.min_second_mass for m in masses)
        print(f"confirm {row['dataset']}: second-mode mass over 3 seeds {np.round(masses, 3)} -> {'BIMODAL' if ok else 'not robust'}", flush=True)
        if ok:
            confirmed.append(dict(dataset=row["dataset"], true_angle=a1, checks=checks, x=x.tolist()))
    (HERE / "natural_bimodal_confirmed.json").write_text(json.dumps(confirmed, indent=2) + "\n")
    print(f"done: {len(rows)} screened, {len(candidates)} candidates, {len(confirmed)} confirmed", flush=True)


if __name__ == "__main__":
    main()
