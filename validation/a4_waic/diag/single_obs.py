"""Single-observation accuracy of the network posterior and the PF-ODE likelihood
(SINGLE_OBS.md).

    .COMPASS/bin/python -m validation.a4_waic.diag.single_obs mcmc --family offset   # CPU
    .COMPASS/bin/python -m validation.a4_waic.diag.single_obs net --family offset    # GPU
    .COMPASS/bin/python -m validation.a4_waic.diag.single_obs analyze

``mcmc`` draws the rows and the exact single-row posteriors; ``net`` adds the network
draws and the PF-ODE log p at every set of draws; ``analyze`` writes the summary json
and the figures.
"""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "net":
    import validation.common.runtime_gpu  # noqa: F401
else:
    import validation.common.runtime  # noqa: F401

import argparse
import json
import os

import numpy as np

from validation.a4_waic import common, references
from validation.a4_waic.families import family

FAMILIES = ("offset", "noise", "population", "population_wide", "tutorial")
BUDGET = {"offset": 65536, "noise": 65536, "population": 131072, "population_wide": 131072,
          "tutorial": 131072}
ROWS = 20          # single rows per data model
S_NET = 200        # network draws, and MCMC draws given to the PF-ODE
OUT = os.path.join(common.RESULTS, "single_obs")


def path(fam, kind):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, f"{fam}_{kind}.npz")


def rows_for(fam, data_name):
    """``ROWS`` single rows from the data model's prior predictive and their parameters."""
    data_fam = common.DATA_FAMILY.get(fam, fam)
    models = family(data_fam)
    rng = np.random.default_rng([11, sum(map(ord, data_fam)), list(models).index(data_name)])
    joint = models[data_name].training_data(ROWS, rng)
    m = models[data_name]
    p = m.n_global + m.n_local
    return joint[:, p:], joint[:, :p]


def waic_row(ll):
    """ll (..., S) -> elpd_waic of one row, lppd and p_waic."""
    mx = ll.max(-1, keepdims=True)
    lppd = (mx[..., 0] + np.log(np.exp(ll - mx).mean(-1)))
    p = ll.var(-1, ddof=1)
    return lppd - p, lppd, p


# ---------------------------------------------------------------------------
# exact single-row posteriors (CPU)
# ---------------------------------------------------------------------------

def run_mcmc(fam):
    models = family(fam)
    out = {}
    for d in models:
        x, theta = rows_for(fam, d)
        out[f"x_{d}"], out[f"theta_{d}"] = x, theta
        for name, model in models.items():
            draws, ll_all, taus = [], [], []
            for i in range(ROWS):
                res, chain = references.mcmc(model, x[i:i + 1], seed=1000 * i + 7, n_steps=3000,
                                             burn=1000, thin=5, walkers=32, restarts=2)
                ll_all.append(model.log_lik_rows(chain, x[i:i + 1])[:, 0])
                pick = np.random.default_rng(i).choice(len(chain), S_NET, replace=False)
                draws.append(chain[pick])
                taus.append(res["tau_max"])
            out[f"mcmc_{name}_{d}"] = np.stack(draws)                     # (K, S, P)
            out[f"llall_{name}_{d}"] = np.stack(ll_all)                    # (K, 12800)
            out[f"tau_{name}_{d}"] = np.array(taus)
            print(fam, "fit", name, "data", d, "tau max", np.nanmax(taus), flush=True)
    np.savez(path(fam, "mcmc"), **out)


# ---------------------------------------------------------------------------
# network draws and PF-ODE log p (GPU)
# ---------------------------------------------------------------------------

def run_net(fam, timesteps=50, batch_size=2048):
    import torch

    from validation.a4_waic.diagnose_cross import pfode_ll
    from validation.a4_waic.scaling import Scaler
    from validation.a4_waic.train import load

    models = family(fam)
    mc = dict(np.load(path(fam, "mcmc")))
    out = {}
    for name, model in models.items():
        scaler = Scaler.for_model(fam, model)
        net = load(fam, name, BUDGET[fam], 0, device="cuda")
        net.model.to("cuda")
        mask = torch.tensor(model.condition_mask, dtype=torch.float32)
        h = model.n_global
        for d in models:
            x = mc[f"x_{d}"]
            xt = torch.tensor(scaler.x_forward(x), dtype=torch.float32)
            torch.manual_seed(0)
            draws = net.sample(x=xt, condition_mask=mask, timesteps=timesteps, num_samples=S_NET,
                               multi_obs_inference=False, device="cuda", verbose=False)
            net_theta = scaler.theta_backward(draws.cpu().double().numpy())        # (K, S, P)
            sets = {"net": net_theta, "mcmc": mc[f"mcmc_{name}_{d}"]}
            if d == name:
                sets["true"] = mc[f"theta_{d}"][:, None, :]
            for key, theta in sets.items():
                exact = model.loglik(theta[..., :h], theta[..., h:], x[:, None, :])
                pf = pfode_ll_batched(pfode_ll, net, scaler, model, x, theta, batch_size, timesteps)
                out[f"{key}theta_{name}_{d}"] = theta
                out[f"{key}exact_{name}_{d}"] = exact
                out[f"{key}pf_{name}_{d}"] = pf
            print(fam, "fit", name, "data", d, "done", flush=True)
        del net
        torch.cuda.empty_cache()
    np.savez(path(fam, "net"), **out)


def pfode_ll_batched(pfode_ll, net, scaler, model, x, theta, batch_size, timesteps):
    """PF-ODE log p in raw-x units with the exact divergence, rows in chunks."""
    import torch

    n, s, _ = theta.shape
    per = max(1, batch_size // s)
    out = []
    for a in range(0, n, per):
        nodes = np.concatenate([theta[a:a + per],
                                np.broadcast_to(x[a:a + per, None], (min(per, n - a), s, x.shape[1]))], -1)
        z = torch.tensor(scaler.forward(nodes).reshape(-1, nodes.shape[-1]), dtype=torch.float32)
        mask = torch.tensor(model.condition_mask, dtype=torch.float32)
        lp = net.log_prob(z, condition_mask=1 - mask, timesteps=timesteps, divergence="exact",
                          device="cuda", batch_size=batch_size)
        out.append(lp.double().numpy().reshape(-1, s) - np.log(scaler.x_sd).sum())
    return np.concatenate(out, 0)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def posterior_metrics(ref, net):
    """ref, net (S, P) draws of one row -> mean error / sd, sd ratios, narrowest-direction
    sd ratio, mean Mahalanobis d^2 of the network draws under the reference."""
    mu, sd = ref.mean(0), ref.std(0)
    cov = np.cov(ref.T) + 1e-12 * np.eye(ref.shape[1])
    val, vec = np.linalg.eigh(cov)
    proj = (net - mu) @ vec
    d2 = ((proj ** 2) / val).sum(1).mean()
    return dict(mean_z=np.abs((net.mean(0) - mu) / sd), sd_ratio=net.std(0) / sd,
                narrow_ratio=float(proj[:, 0].std() / np.sqrt(val[0])),
                eig_ratio=float(np.sqrt(val[0] / val[-1])), d2=float(d2), dim=ref.shape[1])


def analyze():
    from validation.common import plotting
    from validation.common.plotting import EXACT, FOURTH, INK2, LEGACY, NEW, plt

    summary = {}
    for fam in FAMILIES:
        if not (os.path.exists(path(fam, "mcmc")) and os.path.exists(path(fam, "net"))):
            continue
        mc, nt = dict(np.load(path(fam, "mcmc"))), dict(np.load(path(fam, "net")))
        names = list(family(fam))
        fam_out = {}
        for name in names:
            for d in names:
                key = f"{name}_{d}"
                ref_all = mc[f"llall_{key}"]                                  # exact, 12800 draws
                e_ref, _, p_ref = waic_row(ref_all)
                e_net_ex, _, p_net_ex = waic_row(nt[f"netexact_{key}"])
                e_mc_pf, _, p_mc_pf = waic_row(nt[f"mcmcpf_{key}"])
                e_net_pf, _, p_net_pf = waic_row(nt[f"netpf_{key}"])
                e_mc_ex, _, _ = waic_row(nt[f"mcmcexact_{key}"])           # same 200 MCMC draws
                lik_err = nt[f"mcmcpf_{key}"] - nt[f"mcmcexact_{key}"]       # (K, S)
                post = [posterior_metrics(nt[f"mcmctheta_{key}"][i], nt[f"nettheta_{key}"][i])
                        for i in range(ROWS)]
                entry = dict(
                    in_dist=name == d,
                    lik_bias_mcmc=float(lik_err.mean()), lik_sd_rows=float(lik_err.mean(1).std()),
                    lik_sd_within=float(lik_err.std(1).mean()),
                    lik_err_rows=lik_err.mean(1).tolist(),
                    elpd_err_posterior=float(np.median(e_net_ex - e_ref)),
                    elpd_err_likelihood=float(np.median(e_mc_pf - e_mc_ex)),
                    elpd_err_both=float(np.median(e_net_pf - e_ref)),
                    elpd_err_both_mean=float(np.mean(e_net_pf - e_ref)),
                    elpd_mc_noise_200=float(np.median(np.abs(e_mc_ex - e_ref))),
                    p_waic_ref=float(np.median(p_ref)), p_waic_net_draws=float(np.median(p_net_ex)),
                    p_waic_pf_mcmc=float(np.median(p_mc_pf)), p_waic_compass=float(np.median(p_net_pf)),
                    elpd_rows=dict(ref=e_ref.tolist(), post=(e_net_ex - e_ref).tolist(),
                                   lik=(e_mc_pf - e_mc_ex).tolist(), both=(e_net_pf - e_ref).tolist()),
                    mean_z_max=float(np.median([p["mean_z"].max() for p in post])),
                    sd_ratio_median=float(np.median([np.median(p["sd_ratio"]) for p in post])),
                    sd_ratio_range=[float(np.median([p["sd_ratio"].min() for p in post])),
                                    float(np.median([p["sd_ratio"].max() for p in post]))],
                    narrow_ratio=float(np.median([p["narrow_ratio"] for p in post])),
                    eig_ratio=float(np.median([p["eig_ratio"] for p in post])),
                    d2_over_dim=float(np.median([p["d2"] / p["dim"] for p in post])),
                    tau_max=float(np.nanmax(mc[f"tau_{key}"])))
                if name == d:
                    t_err = nt[f"truepf_{key}"][:, 0] - nt[f"trueexact_{key}"][:, 0]
                    entry.update(lik_bias_true=float(t_err.mean()), lik_sd_true=float(t_err.std()),
                                 lik_err_true_rows=t_err.tolist())
                fam_out[key] = entry
        summary[fam] = fam_out
    with open(os.path.join(common.RESULTS, "single_obs_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # Compact table: in-distribution and misspecified medians per family.
    table = {}
    for fam, fo in summary.items():
        for flag, lab in ((True, "in"), (False, "mis")):
            es = [e for e in fo.values() if e["in_dist"] == flag]
            agg = lambda k: float(np.median([e[k] for e in es]))  # noqa: E731
            table[f"{fam}/{lab}"] = {k: round(agg(k), 3) for k in (
                "lik_bias_mcmc", "lik_sd_rows", "lik_sd_within", "elpd_err_posterior",
                "elpd_err_likelihood", "elpd_err_both", "elpd_mc_noise_200", "p_waic_ref",
                "p_waic_net_draws", "p_waic_compass", "mean_z_max", "sd_ratio_median",
                "narrow_ratio", "eig_ratio", "d2_over_dim")}
            if flag:
                table[f"{fam}/{lab}"]["lik_bias_true"] = round(agg("lik_bias_true"), 3)
                table[f"{fam}/{lab}"]["lik_sd_true"] = round(agg("lik_sd_true"), 3)
    print(json.dumps(table, indent=1))
    for fam, fo in summary.items():
        print(fam, {k: (round(e["elpd_err_posterior"], 2), round(e["elpd_err_likelihood"], 2),
                        round(e["narrow_ratio"], 2)) for k, e in fo.items()})

    plotting.style()
    fams = list(summary)
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5))
    for row, flag, lab in ((0, True, "in-distribution (data model = fitted model)"),
                           (1, False, "misspecified (data from another model)")):
        # (a) per-row elpd error split.
        ax = axs[row, 0]
        for i, fam in enumerate(fams):
            es = [e for e in summary[fam].values() if e["in_dist"] == flag]
            for j, (k, col, nm) in enumerate((("post", NEW, "posterior (net draws, exact lik)"),
                                              ("lik", FOURTH, "likelihood (MCMC draws, PF-ODE)"),
                                              ("both", LEGACY, "both (= COMPASS)"))):
                v = np.concatenate([e["elpd_rows"][k] for e in es])
                q = np.quantile(v, [0.1, 0.5, 0.9])
                ax.errorbar(i + (j - 1) * 0.22, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt="o",
                            color=col, label=nm if i == 0 else None)
        ax.axhspan(-0.1, 0.1, color=EXACT, alpha=0.15, label="target ±0.1")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xticks(range(len(fams)), fams, fontsize=8)
        ax.set_ylabel("elpd_waic error per row (median, 10–90%)")
        ax.set_title(lab, fontsize=10)
        if row == 0:
            ax.legend(fontsize=7)
        # (b) likelihood error at MCMC draws (and true theta).
        ax = axs[row, 1]
        for i, fam in enumerate(fams):
            es = [e for e in summary[fam].values() if e["in_dist"] == flag]
            v = np.concatenate([e["lik_err_rows"] for e in es])
            q = np.quantile(v, [0.1, 0.5, 0.9])
            ax.errorbar(i - 0.12, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt="o", color=NEW,
                        label="at exact-posterior draws" if i == 0 else None)
            if flag:
                v = np.concatenate([e["lik_err_true_rows"] for e in es])
                q = np.quantile(v, [0.1, 0.5, 0.9])
                ax.errorbar(i + 0.12, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt="s", color=EXACT,
                            label="at true θ" if i == 0 else None)
        ax.axhspan(-0.1, 0.1, color=EXACT, alpha=0.15)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xticks(range(len(fams)), fams, fontsize=8)
        ax.set_ylabel("PF-ODE − exact log p(x|θ) per row")
        ax.set_title("likelihood error", fontsize=10)
        ax.legend(fontsize=7)
        # (c) posterior shape.
        ax = axs[row, 2]
        for i, fam in enumerate(fams):
            es = [e for e in summary[fam].values() if e["in_dist"] == flag]
            for j, (k, col, nm) in enumerate((("sd_ratio_median", NEW, "marginal sd ratio"),
                                              ("narrow_ratio", LEGACY, "sd ratio, narrowest direction"),
                                              ("d2_over_dim", FOURTH, "Mahalanobis d²/D"))):
                v = [e[k] for e in es]
                ax.scatter(np.full(len(v), i + (j - 1) * 0.22), v, s=14, color=col,
                           label=nm if i == 0 else None)
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(fams)), fams, fontsize=8)
        ax.set_ylabel("network / exact (1 = correct)")
        ax.set_title("posterior shape (one point per fitted × data model)", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("single observation: network posterior and PF-ODE likelihood vs exact")
    fig.tight_layout()
    print(plotting.save(fig, os.path.join(common.RESULTS, "figures"), "fig_single_obs"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("mcmc", "net", "analyze"))
    parser.add_argument("--family")
    args = parser.parse_args()
    if args.command == "mcmc":
        run_mcmc(args.family)
    elif args.command == "net":
        run_net(args.family)
    else:
        analyze()


if __name__ == "__main__":
    main()
