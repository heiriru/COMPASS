"""Accuracy and recovery of the gauss_jacobian_FD composed posterior (COMPOSED_RECOVERY.md).

    .COMPASS/bin/python -m validation.a4_waic.diag.composed_recovery ref --family population --true LV  # CPU
    .COMPASS/bin/python -m validation.a4_waic.diag.composed_recovery net --family population           # GPU
    .COMPASS/bin/python -m validation.a4_waic.diag.composed_recovery analyze

The true model is fitted to each test dataset (the same datasets as the WAIC runs). For
the global(s) and every row's locals: posterior mean, sd, the posterior CDF at the true
value (PIT, uniform over datasets for a calibrated posterior) and the correlation between
the global and the row's local, for the composed posterior and for the exact one
(grid quadrature for offset/noise, emcee with restarts for population).
"""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "net":
    import validation.common.runtime_gpu  # noqa: F401
else:
    import validation.common.runtime  # noqa: F401

import argparse
import json
import math
import os

import numpy as np
from scipy.special import logsumexp

from validation.a4_waic import common, references
from validation.a4_waic.families import family

SETUP = {   # family: (true models, datasets per model, budget)
    "offset": (("M1",), 50, 65536),
    "noise": (("G", "T"), 50, 65536),
    "population": (("LV", "LP", "SP", "RM"), 30, 131072),
    "population_wide": (("LV", "LP", "SP", "RM"), 30, 131072),
}
DRAWS = 1000
OUT = os.path.join(common.RESULTS, "composed_recovery")


def path(fam, kind, true=None):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, f"{fam}_{kind}" + (f"_{true}" if true else "") + ".npz")


def summarize_draws(g, l, g_true, l_true):
    """g (S, H), l (S, N, L) draws -> summary arrays (corr between g_0 and l_i0)."""
    out = dict(l_mean=l.mean(0), l_sd=l.std(0), l_pit=(l < l_true[None]).mean(0))
    if g.shape[1]:
        out.update(g_mean=g.mean(0), g_sd=g.std(0), g_pit=(g < g_true[None]).mean(0))
        gc = g[:, :1] - g[:, :1].mean(0)
        lc = l[..., 0] - l[..., 0].mean(0)
        out["corr"] = (gc * lc).mean(0) / (gc.std(0) * lc.std(0))
    return out


def grid_summary(model, x, g_true, l_true, n_g=601, n_l=1601, width=8.0):
    """Exact posterior summaries by grid quadrature (offset / noise: <= 1 global, 1 local)."""
    sl = float(model.local_std[0])
    l_grid = np.linspace(model.local_mean[0] - width * sl, model.local_mean[0] + width * sl, n_l)
    if model.n_global:
        sg, mg = float(model.global_std[0]), float(model.global_mean[0])
        coarse = np.linspace(mg - width * sg, mg + width * sg, 2001)

        def log_m(g_grid):
            return np.stack([logsumexp(ll + lp[None], 1) for ll, lp in
                             (references._row_tables(model, g_grid, l_grid, xr) for xr in x)], 1)

        post = model.log_prior_global(coarse[:, None]) + log_m(coarse).sum(1)
        keep = coarse[post > post.max() - 40.0]
        pad = 2 * (coarse[1] - coarse[0])
        g_grid = np.linspace(keep.min() - pad, keep.max() + pad, n_g)
        lm = log_m(g_grid)
        lj = model.log_prior_global(g_grid[:, None]) + lm.sum(1)
        w_g = np.exp(lj - logsumexp(lj))
    else:
        g_grid = np.zeros(1)
        w_g = np.ones(1)
    out = {k: [] for k in ("l_mean", "l_sd", "l_pit", "corr")}
    gm = (w_g * g_grid).sum()
    gs = math.sqrt(max((w_g * g_grid ** 2).sum() - gm ** 2, 0.0))
    for i, xr in enumerate(x):
        ll, lp = references._row_tables(model, g_grid, l_grid, xr)
        a = ll + lp[None]
        w_l = np.exp(a - logsumexp(a, 1, keepdims=True))          # p(l | g, x_i), rows sum to 1
        joint = w_g[:, None] * w_l
        pl = joint.sum(0)
        m = (pl * l_grid).sum()
        s = math.sqrt(max((pl * l_grid ** 2).sum() - m ** 2, 0.0))
        out["l_mean"].append(m); out["l_sd"].append(s)
        out["l_pit"].append(pl[l_grid < l_true[i, 0]].sum())
        if model.n_global:
            cov = (joint * (g_grid[:, None] - gm) * (l_grid[None] - m)).sum()
            out["corr"].append(cov / (gs * s))
    res = dict(l_mean=np.array(out["l_mean"])[:, None], l_sd=np.array(out["l_sd"])[:, None],
               l_pit=np.array(out["l_pit"])[:, None])
    if model.n_global:
        res.update(g_mean=np.array([gm]), g_sd=np.array([gs]),
                   g_pit=np.array([w_g[g_grid < g_true[0]].sum()]), corr=np.array(out["corr"]))
    return res


def stack(dicts):
    return {k: np.stack([d[k] for d in dicts]) for k in dicts[0]}


# ---------------------------------------------------------------------------
# exact references (CPU)
# ---------------------------------------------------------------------------

def run_ref(fam, true):
    _, reps, _ = SETUP[fam]
    model = family(fam)[true]
    out = []
    for rep in range(reps):
        data = common.dataset(fam, true, rep)
        x, g_true, l_true = data["x"], np.atleast_1d(data["g"]), data["l"]
        if fam in ("offset", "noise"):
            out.append(grid_summary(model, x, g_true, l_true))
        else:
            _, chain = references.mcmc(model, x, seed=rep + 17, n_steps=6000, burn=2000, restarts=2)
            g, l = model.unpack(chain, x.shape[0])
            out.append(summarize_draws(g, l, g_true, l_true))
        print(fam, true, "rep", rep, flush=True)
    np.savez(path(fam, "ref", true), **stack(out))


# ---------------------------------------------------------------------------
# composed posterior (GPU)
# ---------------------------------------------------------------------------

def run_net(fam):
    import torch

    from validation.a4_waic.scaling import Scaler
    from validation.a4_waic.train import load

    trues, reps, budget = SETUP[fam]
    for true in trues:
        model = family(fam)[true]
        scaler = Scaler.for_model(fam, model)
        net = load(fam, true, budget, 0, device="cuda")
        net.model.to("cuda")
        h = model.n_global
        mask = torch.tensor(model.condition_mask, dtype=torch.float32)
        out = []
        for rep in range(reps):
            data = common.dataset(fam, true, rep)
            x = data["x"]
            torch.manual_seed(1_000_003 * rep + 7)
            xt = torch.tensor(scaler.x_forward(x), dtype=torch.float32)
            draws = net.sample(x=xt, condition_mask=mask, timesteps=50, num_samples=DRAWS,
                               multi_obs_inference=True, hierarchy=list(range(h)),
                               prior=(torch.zeros(h), torch.ones(h)), correction="gauss_jacobian_FD",
                               jacobian_refresh=9, device="cuda", verbose=False)
            theta = scaler.theta_backward(draws.cpu().double().numpy())      # (N, S, H + L)
            g = theta[0, :, :h]
            l = np.transpose(theta[:, :, h:], (1, 0, 2))                    # (S, N, L)
            out.append(summarize_draws(g, l, np.atleast_1d(data["g"]), data["l"]))
            print(fam, true, "rep", rep, flush=True)
        np.savez(path(fam, "net", true), **stack(out))
        del net
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def metrics(c, e):
    """Composed c vs exact e summaries (dicts of arrays, datasets first)."""
    res = {}
    for p in ("g", "l"):
        if f"{p}_mean" not in c:
            continue
        z = (c[f"{p}_mean"] - e[f"{p}_mean"]) / e[f"{p}_sd"]
        r = c[f"{p}_sd"] / e[f"{p}_sd"]
        pc, pe = c[f"{p}_pit"].ravel(), e[f"{p}_pit"].ravel()
        res[p] = dict(mean_err_rms=float(np.sqrt((z ** 2).mean())), mean_err_median=float(np.median(z)),
                      sd_ratio_median=float(np.median(r)),
                      sd_ratio_q=[float(np.quantile(r, 0.1)), float(np.quantile(r, 0.9))],
                      sd_ratio_per_dim=np.median(r.reshape(-1, r.shape[-1]), 0).tolist(),
                      mean_err_rms_per_dim=np.sqrt((z.reshape(-1, z.shape[-1]) ** 2).mean(0)).tolist(),
                      cover50=float(np.mean(np.abs(pc - 0.5) < 0.25)),
                      cover90=float(np.mean(np.abs(pc - 0.5) < 0.45)),
                      cover50_exact=float(np.mean(np.abs(pe - 0.5) < 0.25)),
                      cover90_exact=float(np.mean(np.abs(pe - 0.5) < 0.45)), n=int(pc.size))
    if "corr" in c:
        res["corr"] = dict(composed_median=float(np.median(c["corr"])),
                           exact_median=float(np.median(e["corr"])),
                           abs_err_median=float(np.median(np.abs(c["corr"] - e["corr"]))))
    return res


def analyze():
    from validation.common import plotting
    from validation.common.plotting import EXACT, INK2, LEGACY, NEW, plt

    summary, pits = {}, {}
    for fam, (trues, _, _) in SETUP.items():
        for true in trues:
            if not (os.path.exists(path(fam, "ref", true)) and os.path.exists(path(fam, "net", true))):
                continue
            c, e = dict(np.load(path(fam, "net", true))), dict(np.load(path(fam, "ref", true)))
            summary[f"{fam}/{true}"] = metrics(c, e)
            pits[f"{fam}/{true}"] = (c, e)
    with open(os.path.join(common.RESULTS, "composed_recovery_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for k, v in summary.items():
        line = [k]
        for p in ("g", "l"):
            if p in v:
                line.append(f"{p}: mean err {v[p]['mean_err_rms']:.2f} sd ratio {v[p]['sd_ratio_median']:.2f}"
                            f" [{v[p]['sd_ratio_q'][0]:.2f},{v[p]['sd_ratio_q'][1]:.2f}]"
                            f" cov90 {v[p]['cover90']:.2f}/{v[p]['cover90_exact']:.2f}")
        if "corr" in v:
            line.append(f"corr(g,l) {v['corr']['composed_median']:+.2f}/{v['corr']['exact_median']:+.2f}")
        print(" | ".join(line))

    plotting.style()
    keys = list(summary)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for j, (p, mk) in enumerate((("g", "o"), ("l", "s"))):
        ks = [k for k in keys if p in summary[k]]
        xs = [keys.index(k) for k in ks]
        axs[0].scatter(np.array(xs) + (j - 0.5) * 0.25, [summary[k][p]["mean_err_rms"] for k in ks],
                       marker=mk, color=(NEW, LEGACY)[j], label=("global", "locals")[j])
        med = [summary[k][p]["sd_ratio_median"] for k in ks]
        lo = [summary[k][p]["sd_ratio_q"][0] for k in ks]
        hi = [summary[k][p]["sd_ratio_q"][1] for k in ks]
        axs[1].errorbar(np.array(xs) + (j - 0.5) * 0.25, med,
                        yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt=mk,
                        color=(NEW, LEGACY)[j], label=("global", "locals")[j])
    axs[0].axhline(0.1, color=INK2, ls="--", lw=0.8, label="0.1 sd")
    axs[0].set_ylabel("RMS (composed − exact mean) / exact sd"); axs[0].set_yscale("log")
    axs[1].axhline(1, color=INK2, lw=0.8)
    axs[1].set_ylabel("composed sd / exact sd (median, 10–90%)")
    for ax in axs[:2]:
        ax.set_xticks(range(len(keys)), keys, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8)
    axs[0].set_title("posterior mean"); axs[1].set_title("posterior width")
    ax = axs[2]
    grid = np.linspace(0, 1, 101)
    for key, (c, e) in pits.items():
        for p, col in (("g", NEW), ("l", LEGACY)):
            if f"{p}_pit" in c:
                v = np.sort(c[f"{p}_pit"].ravel())
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=0.7, alpha=0.6)
                v = np.sort(e[f"{p}_pit"].ravel())
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=EXACT, lw=0.5, alpha=0.4)
    ax.plot(grid, grid, color=INK2, ls="--", lw=0.8)
    ax.plot([], [], color=NEW, label="composed, global"); ax.plot([], [], color=LEGACY, label="composed, locals")
    ax.plot([], [], color=EXACT, label="exact")
    ax.set_xlabel("PIT of the true value"); ax.set_ylabel("empirical CDF"); ax.legend(fontsize=8)
    ax.set_title("recovery calibration (diagonal = calibrated)")
    fig.suptitle("gauss_jacobian_FD composed posterior vs exact (true model fitted)")
    fig.tight_layout()
    print(plotting.save(fig, os.path.join(common.RESULTS, "figures"), "fig_composed_recovery"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("ref", "net", "analyze"))
    parser.add_argument("--family")
    parser.add_argument("--true")
    args = parser.parse_args()
    if args.command == "ref":
        run_ref(args.family, args.true)
    elif args.command == "net":
        run_net(args.family)
    else:
        analyze()


if __name__ == "__main__":
    main()
