"""Rule or network? Compose the same per-observation factors two ways, from two sources, against the exact posterior.

Factor sources (per observation j, the single-observation angle_1 posterior p(a1 | x_j)):
  exact    inverse-CDF draws from ik_exact (noise-free, no network): isolates the composition rule
  network  the stock IK checkpoint's own single-observation posterior (the one every COMPASS arm uses)
Rules (inverse_kinematics/experiments/mixture_inference.infer, p(a)^(1-n) prod_j p_j(a)):
  mixture  Gaussian-mixture summary, K chosen per observation by BIC (k_max 5), guarded merge at beam 64
  gauss    the same pipeline with K = 1, i.e. the single-Gaussian summary the COMPASS rules make
  (+ "_bag": mixture with 64 bootstrap replicates, the n >= 16 setting)

    rule_vs_network.py DATA.npy|hier OUT_DIR EXACT_REFERENCE.npz

Saves factor draws (single_{exact,network}.npy), the composed densities (rule_vs_network.npz), a table
(rule_vs_network.json) and, for the plot in compare_compass_ik.py, samples_mixture_n{n}.npy
(network factors, mixture rule, bagged at n >= 16).
"""
import os
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:3]))   # 3-thread CPU cap, before numpy/torch
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "inverse_kinematics" / "recovery"), str(ROOT / "inverse_kinematics" / "experiments"),
                str(ROOT / "inverse_kinematics"), str(ROOT / "Arruda")]
import ik_exact as E  # noqa: E402
from exact_reference import load, log_lik, w1_over_sd  # noqa: E402

NS = (1, 2, 4, 8, 16, 32, 64, 128)
SINGLE = 20_000
SEED = 20260930


def exact_single(x, grid):
    """(n, SINGLE) draws from each exact p(a1 | x_j), by interpolated inverse CDF on a fine grid."""
    rng = np.random.default_rng(SEED)
    out = np.empty((len(x), SINGLE))
    for j in range(len(x)):
        logp = log_lik(grid, x[j, 0], x[j, 1]) + E.log_normal(grid, E.SD_A1)
        p = np.exp(logp - logp.max())
        cdf = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))])
        out[j] = np.interp(rng.random(SINGLE), cdf / cdf[-1], grid)
    return out


def network_single(x):
    """(n, SINGLE) angle_1 draws from the network's single-observation posterior (DPM2, 100 steps, no churn)."""
    import torch
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=1)
    sys.path[:0] = [str(ROOT / "ModeTree" / "experiments"), str(ROOT / "ModeTree"), str(ROOT / "src")]
    import experiment_ik as X
    import hierarchical as H
    model = X.trainer.load_to(os.environ.get("IK_CHECKPOINT") or X.CHECKPOINT, "cuda")
    pieces = []
    for start in range(0, len(x), 16):
        torch.manual_seed(SEED + start); torch.cuda.manual_seed_all(SEED + start)
        s = model.sample(x=torch.as_tensor(x[start:start + 16], dtype=torch.float32), num_samples=SINGLE, timesteps=100,
                         method="dpm", order=2, churn_eta=0.0, corrector_steps=0, final_corrector_steps=0,
                         terminal_denoise=True, device="cuda", verbose=False)
        pieces.append(s.detach().cpu().numpy()[:, :, H.HIERARCHY[0]])
        torch.cuda.empty_cache()
        print(f"  network single-observation draws {start + len(pieces[-1])}/{len(x)}", flush=True)
    return np.concatenate(pieces).astype(np.float64)


def main():
    import mixture_inference as M
    x = load(sys.argv[1])
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    reference = np.load(sys.argv[3])
    grid = reference["grid"]
    fine = np.linspace(-3.0, 3.0, 6001)
    factors = {}
    for source, make in (("exact", lambda: exact_single(x, fine)), ("network", lambda: network_single(x))):
        path = out / f"single_{source}.npy"
        if not path.exists():
            np.save(path, make())
        factors[source] = np.load(path)

    # Per-observation factor quality: how far is each network p(a1|x_j) from the exact one?
    per_obs = [float(np.mean(np.abs(np.sort(factors["network"][j]) - np.sort(factors["exact"][j]))))
               for j in range(len(x))]
    bias = factors["network"].mean(1) - factors["exact"].mean(1)
    table = {"factor_w1": {"mean": float(np.mean(per_obs)), "median": float(np.median(per_obs)), "max": float(np.max(per_obs))},
             "factor_mean_bias": {"mean": float(bias.mean()), "mean_abs": float(np.abs(bias).mean())},
             "factor_p_positive": {s: float((factors[s] > 0).mean()) for s in factors}}
    print(f"per-observation W1(network, exact): mean {np.mean(per_obs):.3f} median {np.median(per_obs):.3f} "
          f"max {np.max(per_obs):.3f}; mean bias {bias.mean():+.3f} (mean |bias| {np.abs(bias).mean():.3f}); "
          f"P(a1>0) exact {table['factor_p_positive']['exact']:.3f} network {table['factor_p_positive']['network']:.3f}",
          flush=True)

    densities, rows = {}, {}
    rng = np.random.default_rng(SEED + 1)
    for source in ("exact", "network"):
        for rule, kwargs in (("gauss", dict(k_max=1)), ("mixture", dict(k_max=5)), ("mixture_bag", dict(k_max=5, bag=64))):
            label = f"{source}/{rule}"
            for n in NS:
                started = time.time()
                try:
                    p, info = M.infer(factors[source][:n], grid, E.SD_A1, device="cuda", **kwargs)
                except RuntimeError as error:        # improper composed product (a near-cancelled component)
                    print(f"  {label} n={n}: {error}", flush=True)
                    rows.setdefault(label, {})[n] = dict(w1_over_sd=float("nan"), p_positive=float("nan"), failed=str(error))
                    continue
                p = p / np.trapezoid(p, grid)
                densities[f"{source}_{rule}_n{n}"] = p
                w1 = w1_over_sd(grid, p, grid, reference[f"n{n}"])
                rows.setdefault(label, {})[n] = dict(
                    w1_over_sd=round(w1, 3), p_positive=round(float(np.trapezoid(np.where(grid > 0, p, 0), grid)), 3),
                    k_median=info["k_median"], outside_support=round(info["outside_support"], 3),
                    seconds=round(time.time() - started, 2))
                if source == "network" and rule == ("mixture_bag" if n >= 16 else "mixture"):
                    cdf = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))])
                    np.save(out / f"samples_mixture_n{n}.npy",
                            np.interp(rng.random(4_000), cdf / cdf[-1], grid).astype(np.float32))
            print(f"{label:22s} W1/sd " + "  ".join(f"n={n}:{r['w1_over_sd']:.2f}" for n, r in rows[label].items()), flush=True)
            print(f"{'':22s} P(a1>0) " + "  ".join(f"n={n}:{r['p_positive']:.2f}" for n, r in rows[label].items()), flush=True)
    ref_pos = {n: round(float(np.trapezoid(np.where(grid > 0, reference[f'n{n}'], 0), grid)), 3) for n in NS}
    print(f"{'exact posterior':22s} P(a1>0) " + "  ".join(f"n={n}:{v:.2f}" for n, v in ref_pos.items()), flush=True)
    table.update(rules=rows, exact_p_positive=ref_pos)
    np.savez(out / "rule_vs_network.npz", grid=grid, **densities)
    (out / "rule_vs_network.json").write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main()
