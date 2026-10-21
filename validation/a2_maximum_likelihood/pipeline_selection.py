"""E-A1.5 / E-A2.3: selection rates through the real ModelTransfuser.compare pipeline.

Level (b) of the validation ladder: exact-score oracle networks, but everything
else is production code -- compositional posterior sampling (gauss_hierarchical),
the score-ascent MAP, the PF-ODE likelihood, the PF-ODE MLE, and the counted
criteria.

Nested pair (validation/common/gaussian_oracles.nested_offset_pair), data from
M0 (g = 0) or with an offset g = delta:
    M1: x = (l + nu e, g + nu e)   global g ~ N(0, s_g^2), local l ~ N(0, s_l^2)
    M0: x = (l + nu e,     nu e)   local l
s_g = nu / sqrt(N) makes the composed posterior of g shrink the MLE by c = 1/2,
so the MAP plug-in has a known, non-negligible bias.

For every replicate dataset the *exact* decision is computed too (closed-form
MLE likelihoods and the same counted criteria), so the pipeline is graded
per dataset (agreement with the exact decision, error in the LR statistic), not
only through the rate.

Predictions (null, delta = 0):
    MLE plug-in: AIC rate 0.157, agreement with the exact decision ~ 1
    MAP plug-in: AIC rate P(chi2_1 > 2 / 0.75) = 0.103 (c = 1/2)
M1 and M0 run through separate compare() calls because compare() takes one
`multi_obs_inference`/`hierarchy` setting for all models (see README); the
criteria are then combined with compass.information_criteria.compare_criteria,
the same function compare() uses internally.

Run from the repository root (shards can run in parallel):
    .COMPASS/bin/python -m validation.a2_maximum_likelihood.pipeline_selection \
        --n 16 --reps 200 --shard 0 --shards 1
"""

import validation.common.runtime  # noqa: F401

import argparse
import contextlib
import io
import json
import math
import os
import time

import numpy as np
import torch

from compass import information_criteria as ic
from compass.ModelTransfuser import ModelTransfuser
from validation.common.gaussian_oracles import nested_offset_pair, oracle_sbim

OUT = os.path.join(os.path.dirname(__file__), "results", "pipeline")
NU, S_L = 1.0, 1.0
CRITERIA = ("aic", "aicc", "bic")


def gauss_logpdf(x, mean, sd):
    return -0.5 * np.log(2 * np.pi * sd ** 2) - 0.5 * (x - mean) ** 2 / sd ** 2


def exact_row_loglik(x, smoothing_sd=0.0):
    """Exact per-row maximum log-likelihoods of (M1, M0) at their MLEs."""
    sd = math.sqrt(NU ** 2 + smoothing_sd ** 2)
    x1, x2 = x[:, 0], x[:, 1]
    ll_local = gauss_logpdf(x1, x1, sd)
    return ll_local + gauss_logpdf(x2, x2.mean(), sd), ll_local + gauss_logpdf(x2, 0.0, sd)


def run_compare(model, x, multi_obs, hierarchy, prior, num_samples, timesteps):
    mtf = ModelTransfuser()
    with contextlib.redirect_stdout(io.StringIO()):
        mtf.add_model("m", model)
        mtf.trained_models = True
        mtf.compare(x=x, num_samples=num_samples, timesteps=timesteps, device="cpu",
                    multi_obs_inference=multi_obs, hierarchy=hierarchy, prior=prior,
                    correction="gauss_hierarchical" if multi_obs else "gauss",
                    criterion="aic", point_estimate="mle", verbose=False)
    return mtf.stats["m"]


def decide(ll1, ll0, criterion, k1, k0):
    results, used = ic.compare_criteria([torch.as_tensor(ll1), torch.as_tensor(ll0)],
                                        criterion, [k1[0], k0[0]], [k1[1], k0[1]], 2)
    return bool(results[0]["total"] < results[1]["total"]), used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    n = args.n
    s_g = NU / math.sqrt(n)
    joint1, joint0 = nested_offset_pair(s_g=s_g, s_l=S_L, nu=NU)
    model1, model0 = oracle_sbim(joint1), oracle_sbim(joint0)
    smoothing = float(model1.sde.marginal_prob_std(torch.tensor(1e-3)))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"n{n}_delta{args.delta:g}_shard{args.shard}of{args.shards}.jsonl")
    done = 0
    if os.path.exists(path):
        with open(path) as f:
            done = sum(1 for _ in f)

    for rep in range(args.reps):
        if rep % args.shards != args.shard:
            continue
        if done > 0:
            done -= 1
            continue
        rng = np.random.default_rng([args.seed, n, rep])
        torch.manual_seed(int(rng.integers(2 ** 31)))
        x = np.stack([rng.normal(0.0, math.sqrt(S_L ** 2 + NU ** 2), n),
                      rng.normal(args.delta, NU, n)], 1)
        x_t = torch.tensor(x, dtype=torch.float32)

        t0 = time.time()
        st1 = run_compare(model1, x_t, True, [0], (torch.zeros(1), torch.full((1,), s_g)),
                          args.num_samples, args.timesteps)
        st0 = run_compare(model0, x_t, False, None, None, args.num_samples, args.timesteps)
        elapsed = time.time() - t0

        ex1, ex0 = exact_row_loglik(x)
        exs1, exs0 = exact_row_loglik(x, smoothing)
        record = {"rep": rep, "n": n, "delta": args.delta, "seconds": elapsed,
                  "lr_exact": float(2 * (ex1.sum() - ex0.sum())),
                  "lr_mle": float(2 * (st1["log_probs"].sum() - st0["log_probs"].sum())),
                  "lr_map": float(2 * (st1["log_probs_at_map"].sum()
                                       - st0["log_probs_at_map"].sum())),
                  "ll_error_m1_mle": float(st1["log_probs"].sum() - exs1.sum()),
                  "ll_error_m0_mle": float(st0["log_probs"].sum() - exs0.sum()),
                  "g_mle": float(st1["MLE"][0, 0]), "g_exact_mle": float(x[:, 1].mean()),
                  "g_map": float(st1["MAP"][0, 0, 0]),
                  "m1_rows_using_map": st1["mle_info"]["rows_using_map"],
                  "m1_at_bound": int(st1["mle_info"]["at_bound"].any(1).sum()),
                  "m0_at_bound": int(st0["mle_info"]["at_bound"].any(1).sum())}
        for criterion in CRITERIA:
            for arm, key in (("mle", "log_probs"), ("map", "log_probs_at_map")):
                pick, used = decide(st1[key].numpy(), st0[key].numpy(), criterion, (1, 1), (0, 1))
                record[f"{criterion}_{arm}"] = pick
            record[f"{criterion}_exact"], record[f"{criterion}_used"] = decide(
                ex1, ex0, criterion, (1, 1), (0, 1))
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"rep {rep}: {elapsed:.1f}s  LR exact {record['lr_exact']:.3f} "
              f"mle {record['lr_mle']:.3f} map {record['lr_map']:.3f}", flush=True)


if __name__ == "__main__":
    main()
