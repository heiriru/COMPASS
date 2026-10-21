"""Reference WAIC / evidence of every candidate model with the exact likelihood (CPU).

    .COMPASS/bin/python -m validation.a4_waic.run_reference --family noise --true G --reps 200 \
        --refs quad,mcmc,ns --ns-every 4

``quad``: exact (offset and noise families). ``mcmc``: emcee. ``ns``: dynesty, on
every ``--ns-every``-th dataset only (it is the expensive one). The datasets are the
ones run_compass.py sees (common.dataset).
"""

import validation.common.runtime  # noqa: F401

import argparse
import time

import numpy as np

from validation.a4_waic import common, references
from validation.a4_waic.families import family


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--true", required=True)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--refs", default="quad,mcmc,ns")
    parser.add_argument("--ns-every", type=int, default=1)
    parser.add_argument("--mcmc-steps", type=int, default=6000)
    parser.add_argument("--mcmc-burn", type=int, default=2000)
    parser.add_argument("--mcmc-chains", type=int, default=1)
    parser.add_argument("--mcmc-restarts", type=int, default=0)
    parser.add_argument("--nlive", type=int, default=500)
    parser.add_argument("--tag", default="ref")
    args = parser.parse_args()

    refs = set(args.refs.split(","))
    models = family(args.family)
    path = common.result_path("reference", args.family, args.tag, args.true, args.shard, args.shards)
    done = common.done_reps(path)
    for rep in range(args.start, args.start + args.reps):
        if rep % args.shards != args.shard or rep in done:
            continue
        x = common.dataset(args.family, args.true, rep)["x"]
        record = dict(rep=rep, family=args.family, true=args.true, models={})
        t0 = time.time()
        for name, model in models.items():
            entry = {}
            if "quad" in refs:
                entry["quad"] = references.quadrature(model, x)
            if "mcmc" in refs:
                chains = []
                for c in range(args.mcmc_chains):
                    if model.n_global == 0 and x.shape[0] > 1:
                        out, draws = references.mcmc_rows(model, x, seed=rep + 1_000_003 * c)
                    else:
                        out, draws = references.mcmc(model, x, seed=rep + 1_000_003 * c,
                                                     n_steps=args.mcmc_steps, burn=args.mcmc_burn,
                                                     restarts=args.mcmc_restarts)
                    chains.append((out, draws))
                entry["mcmc"] = chains[0][0]
                if args.mcmc_chains > 1:
                    # Pooled draws; the per-chain spread flags an unconverged reference
                    # (e.g. a multimodal posterior of a misspecified model).
                    pooled = np.concatenate([d for _, d in chains])
                    entry["mcmc"] = dict(references.waic_from_loglik(model.log_lik_rows(pooled, x).T),
                                         chain_waic=[o["waic"] for o, _ in chains],
                                         tau_max=max(o["tau_max"] for o, _ in chains),
                                         n_eff=sum(o["n_eff"] for o, _ in chains),
                                         seconds=sum(o["seconds"] for o, _ in chains))
            if "ns" in refs and rep % args.ns_every == 0:
                entry["ns"], _ = references.nested(model, x, seed=rep, nlive=args.nlive)
            record["models"][name] = entry
        record["seconds"] = time.time() - t0
        common.append(path, record)
        summary = " | ".join(
            f"{name} " + " ".join(f"{k} {v['waic']:.2f}" for k, v in entry.items())
            for name, entry in record["models"].items())
        print(f"rep {rep} ({record['seconds']:.0f}s): {summary}", flush=True)


if __name__ == "__main__":
    main()
