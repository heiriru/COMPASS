"""Attribute the COMPASS WAIC error to the composed posterior or to the PF-ODE likelihood.

For each dataset and candidate model, four WAICs from the same machinery:

    sampler \\ likelihood     exact            PF-ODE (network)
    MCMC draws               reference        net_lik_only
    COMPASS draws            sampler_only     compass (full pipeline)

``sampler_only - reference`` is the error of the composed posterior, ``net_lik_only -
reference`` the error of the network likelihood.

    .COMPASS/bin/python -m validation.a4_waic.diagnose_cross --family population --datasets LV:2,RM:0
"""

import validation.common.runtime_gpu  # noqa: F401

import argparse
import json
import os

import numpy as np
import torch

from validation.a4_waic import common, references
from validation.a4_waic.families import family
from validation.a4_waic.scaling import Scaler
from validation.a4_waic.train import load

BUDGET = {"offset": 65536, "noise": 65536, "population": 131072}


def pfode_ll(net, scaler, model, x_raw, theta_raw, divergence, timesteps=50):
    """(N, S) PF-ODE log p(x_i | theta_is) in raw-x units for raw theta (N, S, H+L)."""
    n, s, _ = theta_raw.shape
    nodes = np.concatenate([theta_raw, np.broadcast_to(x_raw[:, None], (n, s, x_raw.shape[1]))], -1)
    z = torch.tensor(scaler.forward(nodes).reshape(n * s, -1), dtype=torch.float32)
    mask = torch.tensor(model.condition_mask, dtype=torch.float32)
    lp = net.log_prob(z, condition_mask=1 - mask, timesteps=timesteps, divergence=divergence,
                      hutchinson_samples=1, device="cuda", batch_size=4096)
    return lp.double().numpy().reshape(n, s) - np.log(scaler.x_sd).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--datasets", required=True, help="TRUE:rep,TRUE:rep,...")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--divergence", default="hutchinson")
    args = parser.parse_args()
    models = family(args.family)
    out_path = os.path.join(common.RESULTS, f"diagnose_cross_{args.family}.jsonl")
    for item in args.datasets.split(","):
        true, rep = item.split(":")
        rep = int(rep)
        x = common.dataset(args.family, true, rep)["x"]
        n = x.shape[0]
        record = dict(true=true, rep=rep, models={})
        for name, model in models.items():
            scaler = Scaler.for_model(args.family, model)
            net = load(args.family, name, BUDGET[args.family], 0, device="cuda")
            net.model.to("cuda")
            h = model.n_global
            xt = torch.tensor(scaler.x_forward(x), dtype=torch.float32)
            torch.manual_seed(rep)
            draws = net.sample(x=xt, condition_mask=torch.tensor(model.condition_mask, dtype=torch.float32),
                               timesteps=50, num_samples=args.draws, multi_obs_inference=True,
                               hierarchy=list(range(h)), prior=(torch.zeros(h), torch.ones(h)),
                               correction="gauss_jacobian_FD", jacobian_refresh=9,
                               device="cuda", verbose=False).cpu().double().numpy()
            compass_theta = scaler.theta_backward(draws)                       # (N, S, H+L)
            _, chain = references.mcmc(model, x, seed=rep, n_steps=10000, burn=4000)
            pick = np.linspace(0, len(chain) - 1, args.draws).round().astype(int)
            g, l = model.unpack(chain[pick], n)                                # (S, H), (S, N, L)
            mcmc_theta = np.concatenate([np.broadcast_to(g[:, None], (len(pick), n, h)), l], -1)
            mcmc_theta = np.transpose(mcmc_theta, (1, 0, 2))                    # (N, S, H+L)

            def exact(theta):
                gg = theta[..., :h]
                ll = theta[..., h:]
                return model.loglik(gg, ll, x[:, None, :])

            entry = {
                "reference": references.waic_from_loglik(exact(mcmc_theta))["waic"],
                "sampler_only": references.waic_from_loglik(exact(compass_theta))["waic"],
                "net_lik_only": references.waic_from_loglik(
                    pfode_ll(net, scaler, model, x, mcmc_theta, args.divergence))["waic"],
                "compass": references.waic_from_loglik(
                    pfode_ll(net, scaler, model, x, compass_theta, args.divergence))["waic"],
                "g_mcmc": [g.mean(0).tolist(), g.std(0).tolist()],
                "g_compass": [compass_theta[0, :, :h].mean(0).tolist(),
                              compass_theta[0, :, :h].std(0).tolist()],
            }
            record["models"][name] = entry
            print(true, rep, name, {k: round(v, 1) for k, v in entry.items() if not k.startswith("g_")},
                  "g mcmc", np.round(entry["g_mcmc"], 3).tolist(),
                  "g compass", np.round(entry["g_compass"], 3).tolist(), flush=True)
        common.append(out_path, record)


if __name__ == "__main__":
    main()
