"""E-A2.1 / E-A2.2: is the PF-ODE MLE the true MLE, and how wrong is the MAP plug-in?

Model (exact-score oracle, so every error is a pipeline error):
    theta ~ N(0, s^2 I_2),   x = B theta + nu eps,   x in R^3, B full column rank.
Closed forms:
    MLE        theta_hat = (B^T B)^-1 B^T x
    max logL   sum_j log N(x_j; (B theta_hat)_j, nu^2)
    MAP        posterior mean (B^T B / nu^2 + I / s^2)^-1 B^T x / nu^2

Measured per prior scale s/nu in {0.3, 1, 3, 10}, N rows each:
    * MLE location error |theta_PF - theta_hat| in posterior sds
    * exact log-likelihood deficit at the PF-ODE MLE and at the MAP (nats/row)
    * PF-ODE log-likelihood offset at the optimum (solver + sigma(eps) smoothing)
    * the AIC error of each plug-in: 2 * sum_i deficit_i
    * how often the default box (posterior range +- 3 sd) binds
Predictions (written before running, see README):
    * PF-ODE MLE deficit < 1e-3 nats/row unbounded, for every s
    * MAP deficit = 0.5 |B (theta_hat - theta_MAP)|^2 / nu^2, growing as s shrinks
    * the default box binds for s/nu = 0.3 (the MLE is many posterior sds away)

Run from the repository root:
    .COMPASS/bin/python -m validation.a2_maximum_likelihood.exactness
"""

import validation.common.runtime  # noqa: F401

import json
import math
import os
import time

import torch

from compass.likelihood_maximization import maximize_likelihood
from validation.common.gaussian_oracles import GaussianJoint, oracle_sbim

OUT = os.path.join(os.path.dirname(__file__), "results")
NU = 0.5
B = torch.tensor([[1.0, 0.0], [0.5, 1.0], [-0.3, 0.8]], dtype=torch.float64)
N_ROWS = 32
SCALES = (0.3, 1.0, 3.0, 10.0)


def model_for(s):
    M = torch.zeros(5, 5, dtype=torch.float64)
    M[:2, :2] = s * torch.eye(2)
    M[2:, :2] = s * B
    M[2:, 2:] = NU * torch.eye(3)
    return GaussianJoint(M, 2, ["t1", "t2", "x1", "x2", "x3"])


def exact_loglik(theta, x):
    resid = x - theta @ B.T
    return (-0.5 * math.log(2 * math.pi * NU ** 2) * 3 - 0.5 * (resid ** 2).sum(-1) / NU ** 2)


def run_scale(s, generator):
    joint = model_for(s)
    model = oracle_sbim(joint)
    theta_true = s * torch.randn(N_ROWS, 2, generator=generator, dtype=torch.float64)
    x = theta_true @ B.T + NU * torch.randn(N_ROWS, 3, generator=generator, dtype=torch.float64)

    mle = torch.linalg.solve(B.T @ B, B.T @ x.T).T
    precision = B.T @ B / NU ** 2 + torch.eye(2, dtype=torch.float64) / s ** 2
    post_cov = torch.linalg.inv(precision)
    post_sd = post_cov.diagonal().sqrt()
    map_ = (post_cov @ B.T @ x.T / NU ** 2).T
    ll_max = exact_loglik(mle, x)

    mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    data = torch.cat([map_, x], 1).float()
    scale = torch.cat([post_sd, torch.ones(3, dtype=torch.float64)]).float()
    # Default compare() box: posterior range +- 3 sd, approximated by mean +- (2.6 + 3) sd
    # (the range of ~256 Gaussian draws is about +- 2.6 sd).
    half = (5.6 * post_sd).float()
    lower = torch.cat([data[:, :2] - half, torch.full((N_ROWS, 3), -math.inf)], 1)
    upper = torch.cat([data[:, :2] + half, torch.full((N_ROWS, 3), math.inf)], 1)

    out = {"s_over_nu": s / NU, "rows": N_ROWS,
           "posterior_sd": post_sd.tolist()}
    pf_at_map = model.log_prob(data, mask, timesteps=100).double()
    out["map_exact_deficit_per_row"] = (ll_max - exact_loglik(map_, x)).mean().item()
    out["map_aic_error"] = 2 * (ll_max - exact_loglik(map_, x)).sum().item()
    out["mle_minus_map_in_post_sd"] = ((mle - map_).abs() / post_sd).mean().item()

    for label, lo, hi in (("unbounded", None, None), ("default_box", lower, upper)):
        t0 = time.time()
        res = maximize_likelihood(model, data, mask, scale=scale, lower=lo, upper=hi,
                                  max_iterations=50, timesteps=32)
        elapsed = time.time() - t0
        theta_pf = res.theta[:, :2].double()
        pf_at_opt = model.log_prob(res.theta, mask, timesteps=100).double()
        out[label] = {
            "location_error_post_sd_max": ((theta_pf - mle).abs() / post_sd).max().item(),
            "location_error_post_sd_mean": ((theta_pf - mle).abs() / post_sd).mean().item(),
            "exact_deficit_per_row_mean": (ll_max - exact_loglik(theta_pf, x)).mean().item(),
            "exact_deficit_per_row_max": (ll_max - exact_loglik(theta_pf, x)).max().item(),
            "aic_error": 2 * (ll_max - exact_loglik(theta_pf, x)).sum().item(),
            "pfode_offset_per_row_mean": (pf_at_opt - ll_max).mean().item(),
            "pfode_offset_per_row_sd": (pf_at_opt - ll_max).std().item(),
            "pfode_gain_over_map_per_row": (pf_at_opt - pf_at_map).mean().item(),
            "rows_at_bound": int(res.at_bound.any(1).sum()),
            "evaluations": res.evaluations,
            "seconds": elapsed,
            "messages": sorted(set(res.messages)),
            # Convergence trace: exact-model total log-likelihood gap is not available
            # per evaluation, so store the PF-ODE objective (coarse grid) per evaluation
            # and the exact optimum of the same objective's target for reference.
            "trace_total_loglik": res.history[0],
            "exact_max_total": ll_max.sum().item(),
            "start_total_exact": exact_loglik(map_, x).sum().item(),
        }
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    generator = torch.Generator().manual_seed(20261013)
    results = [run_scale(s * NU, generator) for s in SCALES]
    with open(os.path.join(OUT, "exactness.json"), "w") as f:
        json.dump(results, f, indent=1)

    head = ("| s/nu | MLE-MAP (post sd) | MAP deficit/row | MAP AIC error | box | "
            "PF-MLE loc err max (sd) | PF-MLE deficit/row max | PF-MLE AIC error | "
            "PF offset/row (mean ± sd) | rows at box | evals | s |")
    lines = [head, "|" + "---|" * 12]
    for r in results:
        for label in ("unbounded", "default_box"):
            m = r[label]
            lines.append(
                f"| {r['s_over_nu']:.1f} | {r['mle_minus_map_in_post_sd']:.2f} | "
                f"{r['map_exact_deficit_per_row']:.4f} | {r['map_aic_error']:.2f} | {label} | "
                f"{m['location_error_post_sd_max']:.2e} | {m['exact_deficit_per_row_max']:.2e} | "
                f"{m['aic_error']:.2e} | {m['pfode_offset_per_row_mean']:+.4f} ± "
                f"{m['pfode_offset_per_row_sd']:.4f} | {m['rows_at_bound']} | "
                f"{m['evaluations']} | {m['seconds']:.0f} |")
    table = "\n".join(lines)
    with open(os.path.join(OUT, "exactness.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
