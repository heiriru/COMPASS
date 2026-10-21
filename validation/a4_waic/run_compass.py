"""WAIC of every candidate model through ModelTransfuser.compare(criterion="waic").

    # exact-score oracle networks (offset family only), CPU
    .COMPASS/bin/python -m validation.a4_waic.run_compass --family offset --level oracle --true M0 --reps 200
    # trained networks, one free GPU via autocvd
    .COMPASS/bin/python -m validation.a4_waic.run_compass --family noise --level trained --true T --reps 200

Everything below the network is production code: the gauss_jacobian_FD composed
posterior (jacobian_refresh 9, churn), the PF-ODE pointwise likelihoods and
compass.information_criteria.waic. WAIC values are converted to raw-x coordinates
(scaling.Scaler.waic_jacobian) so they can be compared with the exact references.
"""

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--family", required=True)
parser.add_argument("--level", choices=("oracle", "trained"), required=True)
parser.add_argument("--true", required=True)
parser.add_argument("--reps", type=int, default=200)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--shards", type=int, default=1)
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--timesteps", type=int, default=50)
parser.add_argument("--waic-timesteps", type=int, default=50)
parser.add_argument("--waic-divergence", default="exact", choices=("exact", "hutchinson"))
parser.add_argument("--waic-batch-size", type=int, default=None,
                    help="PF-ODE batch (default 4096; 1024 for the 44-node tutorial family)")
parser.add_argument("--budget", type=int, default=None)
parser.add_argument("--net-seed", type=int, default=0)
parser.add_argument("--tag", default=None)
args = parser.parse_args()

if args.level == "trained":
    import validation.common.runtime_gpu  # noqa: F401
else:
    import validation.common.runtime  # noqa: F401

import contextlib  # noqa: E402
import io  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from compass.ModelTransfuser import ModelTransfuser  # noqa: E402
from validation.a4_waic import common  # noqa: E402
from validation.a4_waic.families import family  # noqa: E402
from validation.a4_waic.scaling import Scaler  # noqa: E402

DEFAULT_BUDGET = {"offset": 65536, "noise": 65536, "population": 131072,
                  "population_wide": 131072, "tutorial": 131072}
BATCH = {"tutorial": 1024}


def candidates(family_name, level, budget, net_seed, device):
    """``{name: (network, scaler, hierarchy, prior)}`` in network coordinates."""
    models = family(family_name)
    out = {}
    if level == "oracle":
        if family_name != "offset":
            raise ValueError("exact-score oracles exist for the offset family only")
        from validation.common.gaussian_oracles import nested_offset_pair, oracle_sbim
        m1 = models["M1"]
        j1, j0 = nested_offset_pair(s_g=m1.s_g, s_l=m1.s_l, nu=m1.nu)
        identity = Scaler(np.zeros(2), np.ones(2), np.zeros(2), np.ones(2))
        out["M1"] = (oracle_sbim(j1), identity, [0],
                     (torch.zeros(1), torch.full((1,), float(m1.s_g))))
        out["M0"] = (oracle_sbim(j0), Scaler(np.zeros(1), np.ones(1), np.zeros(2), np.ones(2)),
                     [], None)
        return out
    from validation.a4_waic.train import load
    for name, model in models.items():
        net = load(family_name, name, budget, net_seed, device=device)
        h = model.n_global
        prior = (torch.zeros(h), torch.ones(h)) if h else None
        out[name] = (net, Scaler.for_model(family_name, model), list(range(h)), prior)
    return out


def main():
    device = "cuda" if args.level == "trained" else "cpu"
    budget = args.budget or DEFAULT_BUDGET[args.family]
    tag = args.tag or (f"oracle_d{args.draws}" if args.level == "oracle"
                       else f"trained_n{budget}_s{args.net_seed}_d{args.draws}"
                       + ("_hutch" if args.waic_divergence == "hutchinson" else ""))
    cands = candidates(args.family, args.level, budget, args.net_seed, device)
    names = list(cands)
    path = common.result_path("compass", args.family, tag, args.true, args.shard, args.shards)
    done = common.done_reps(path)
    mtf = ModelTransfuser()
    with contextlib.redirect_stdout(io.StringIO()):
        for name in names:
            mtf.add_model(name, cands[name][0])
    mtf.trained_models = True
    x_scaler = cands[names[0]][1]

    for rep in range(args.start, args.start + args.reps):
        if rep % args.shards != args.shard or rep in done:
            continue
        data = common.dataset(args.family, args.true, rep)
        x_raw = data["x"]
        torch.manual_seed(1_000_003 * rep + 7)
        x_net = torch.tensor(x_scaler.x_forward(x_raw), dtype=torch.float32)
        t0 = time.time()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            mtf.compare(x=x_net, num_samples=args.draws, timesteps=args.timesteps, device=device,
                        multi_obs_inference=True, criterion="waic",
                        hierarchy={n: cands[n][2] for n in names},
                        prior={n: cands[n][3] for n in names if cands[n][3] is not None},
                        waic_draws=args.draws, waic_timesteps=args.waic_timesteps,
                        waic_divergence=args.waic_divergence, waic_hutchinson_samples=1,
                        waic_batch_size=args.waic_batch_size or BATCH.get(args.family, 4096),
                        verbose=False)
        seconds = time.time() - t0
        record = dict(rep=rep, family=args.family, true=args.true, level=args.level,
                      tag=tag, seconds=seconds, n_rows=int(x_raw.shape[0]),
                      g_true=np.asarray(data["g"]).tolist(), models={})
        for name in names:
            st = mtf.stats[name]
            scaler = cands[name][1]
            jac = scaler.waic_jacobian(x_raw.shape[0])
            theta = st["posterior_samples"].double().numpy()           # (N, S, latent)
            h = len(cands[name][2])
            g_raw = scaler.theta_backward(theta[0])[:, :h] if h else None
            elpd_jac = -jac / (2 * x_raw.shape[0])
            record["models"][name] = dict(
                waic=float(st["WAIC"]) + jac, waic_net=float(st["WAIC"]),
                lppd=st["lppd"] - jac / 2, p_waic=st["p_waic"], se=st["waic_se"],
                mc_se=st["waic_mc_se"], n_p_high=st["n_p_high"],
                elpd_rows=(st["elpd_rows"].numpy() + elpd_jac).tolist(),
                p_waic_rows=st["p_waic_rows"].numpy().tolist(),
                weight=st["model_prob"], weight_pbma_plus=st["model_prob_pseudo_bma_plus"],
                loo=float(st["LOO"]) + jac, loo_mc_se=st["loo_mc_se"], p_loo=st["p_loo"],
                n_k_high=st["n_k_high"], k_max=float(st["pareto_k"].max()),
                elpd_loo_rows=(st["elpd_loo_rows"].numpy() + elpd_jac).tolist(),
                g_mean=g_raw.mean(0).tolist() if h else None,
                g_sd=g_raw.std(0).tolist() if h else None)
        loo = np.array([record["models"][n]["loo"] for n in names])
        w = np.exp(-0.5 * (loo - loo.min())); w /= w.sum()
        for n, wn in zip(names, w):
            record["models"][n]["weight_loo"] = float(wn)
        common.append(path, record)
        best = max(names, key=lambda n: record["models"][n]["weight"])
        print(f"rep {rep}: {seconds:.1f}s best {best} "
              + " ".join(f"{n} {record['models'][n]['waic']:.2f}" for n in names), flush=True)


if __name__ == "__main__":
    sys.exit(main())
