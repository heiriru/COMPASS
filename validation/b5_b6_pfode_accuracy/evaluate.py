"""Evaluate PF-ODE log-likelihood accuracy of one trained network (or the oracle).

    .COMPASS/bin/python -m validation.b5_b6_pfode_accuracy.evaluate --family gauss \
        --budget 16384 --seed 0            # a trained network
    .COMPASS/bin/python -m validation.b5_b6_pfode_accuracy.evaluate --family gauss --oracle

Every setting evaluates log p(x | theta) at the true theta of a fixed test set (512
draws per family, identical across budgets/seeds) and stores the *per-point* values
next to the exact log-likelihood smoothed by sigma(eps) and the unsmoothed one, so
later analyses (tails, model differences) need no re-evaluation.

Settings (reference = 100 steps, exact divergence, eps 1e-3, isotropic endpoint):
    steps        T in {25, 50, 100, 200}
    divergence   exact, hutchinson with {1, 4, 32} probes
    endpoint     isotropic (default), fitted Gaussian
    eps          {3e-4 (below the trained range), 1e-3, 3e-3, 1e-2, 3e-2}
    hybrid       trained network above lambda_c, exact oracle below (trained nets only)
    mle          "too good MLE" check: maximise the PF-ODE likelihood from the true
                 theta for {5, 20, 100} L-BFGS iterations (64 points) and record the
                 PF-ODE value and the exact likelihood at the returned theta
"""

import sys

if "--oracle" in sys.argv:          # oracle runs are CPU-only: no GPU reservation
    import validation.common.runtime  # noqa: F401
else:
    import validation.common.runtime_gpu  # noqa: F401

import argparse
import json
import os
import time

import torch

from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm
from compass.likelihood_maximization import maximize_likelihood
from validation.b5_b6_pfode_accuracy.families import (
    Family, HybridNetwork, LikelihoodOracle, THETA_DIM, X_DIM)
from validation.b5_b6_pfode_accuracy.train import model_dir

RESULT_ROOT = os.path.join(os.path.dirname(__file__), "results", "raw")
MASK = torch.tensor([1.0] * THETA_DIM + [0.0] * X_DIM)
N_TEST, N_MLE = 512, 64
STEPS = (25, 50, 100, 200)
PROBES = (1, 4, 32)
EPS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
LAMBDA_C = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9)   # 0 = all trained, 1e9 = all oracle
MLE_ITERS = (5, 20, 100)
# MLE box: +-4 prior sds on theta (theta ~ N(0, I)); x dims unconstrained (not optimised).
SCALE = torch.tensor([0.2] * THETA_DIM + [1.0] * X_DIM)
LOWER = torch.tensor([-4.0] * THETA_DIM + [-float("inf")] * X_DIM)
UPPER = torch.tensor([4.0] * THETA_DIM + [float("inf")] * X_DIM)


def test_set(family):
    gen = torch.Generator().manual_seed(424242)
    return family.simulate(N_TEST, gen)


class ExactLikelihood:
    """``log_prob`` interface over the exact smoothed likelihood (for the exact MLE)."""

    def __init__(self, family, smooth):
        self.family, self.smooth = family, smooth

    def log_prob(self, data, condition_mask, **kwargs):
        data = torch.as_tensor(data)
        theta, x = data[:, :THETA_DIM], data[:, THETA_DIM:]
        return _family_log_lik_torch(self.family, theta, x, self.smooth)


def _family_log_lik_torch(family, theta, x, smooth):
    """Differentiable exact smoothed log-likelihood (gauss / bimodal / student via grid)."""
    import math
    from validation.b5_b6_pfode_accuracy import families as F
    r = (x - theta @ F.B.to(theta.dtype).T).double()
    if family.name == "gauss":
        sd = math.sqrt(family.nu ** 2 + smooth ** 2)
        return F._log_normal(r, sd).sum(-1)
    if family.name == "bimodal":
        sd = math.sqrt(family.nu ** 2 + smooth ** 2)
        return (torch.logaddexp(F._log_normal(r - F.BIMODAL_SHIFT, sd),
                                F._log_normal(r + F.BIMODAL_SHIFT, sd)) - math.log(2.0)).sum(-1)
    return F._log_student(r, family.nu).sum(-1)   # smoothing ~1e-3 nats, see oracle runs


def cross_eval(model, device):
    """This network's log p on every family's (reproducible) test set, both endpoints."""
    cross = {}
    for other in ("gauss", "student", "bimodal"):
        th_o, x_o = test_set(Family(other, 0.5))
        d_o = torch.cat([th_o, x_o], 1)
        cross[other] = {"x": x_o, "theta": th_o}
        eps_list = ["gaussian", "isotropic"]
        if getattr(model, "base_moments", None) is not None:
            eps_list.append("fixed")
        for ep in eps_list:
            cross[other][ep] = model.log_prob(d_o, MASK, device=device, batch_size=512,
                                              timesteps=100, eps=1e-3, endpoint=ep).double()
    return cross


def evaluate(model, family, device, trained=True):
    theta, x = test_set(family)
    data = torch.cat([theta, x], 1)
    out = {"theta": theta, "x": x, "settings": {}}
    sde = model.sde
    smooth = {e: float(sde.marginal_prob_std(torch.tensor(e))) for e in EPS}
    out["exact"] = {e: family.log_lik(theta, x, s) for e, s in smooth.items()}
    out["exact_unsmoothed"] = family.log_lik(theta, x, 0.0)

    def run(key, **kw):
        t0 = time.time()
        # Networks fine-tuned through the PF-ODE carry base moments and are evaluated
        # with the fixed base density they were trained against.
        default_ep = "fixed" if getattr(model, "base_moments", None) is not None else "isotropic"
        args = dict(timesteps=100, eps=1e-3, divergence="exact", endpoint=default_ep)
        args.update(kw)
        lp = model.log_prob(data, MASK, device=device, batch_size=512, **args).double()
        out["settings"][key] = {"logp": lp, "eps": args["eps"], "seconds": time.time() - t0}

    for T in STEPS:
        run(f"steps_{T}", timesteps=T)
    for k in PROBES:
        run(f"hutch_{k}", divergence="hutchinson", hutchinson_samples=k)
    run("endpoint_gaussian", endpoint="gaussian")
    for e in EPS:
        run(f"eps_{e:g}", eps=e)

    if trained:
        oracle = LikelihoodOracle(family, sde).to(device)
        net = model.model
        for lc in LAMBDA_C:
            model.model = HybridNetwork(net, oracle, sde, lc).to(device)
            run(f"hybrid_{lc:g}")
        lam_max = float(sde.lambda_t(torch.ones(1)))
        model.model = HybridNetwork(net, oracle, sde, 0.0, lam_max * (1 - 1e-4)).to(device)
        run("oracle_endpoint_gaussian")
        run("oracle_endpoint_isotropic", endpoint="isotropic")
        model.model = net

        # B6.3: this network's log p on every family's test set (same theta law), so a
        # per-row model difference log p_i(x) - log p_j(x) can be graded against the
        # exact one. Only for the nu = 0.5 networks that form the model pairs.
        if family.nu == 0.5:
            out["cross"] = cross_eval(model, device)

        idx = torch.arange(N_MLE)
        start = data[idx].clone()
        exact_model = ExactLikelihood(family, smooth[1e-3])
        exact_opt = maximize_likelihood(exact_model, start, MASK, max_iterations=200,
                                        timesteps=1, scale=SCALE, lower=LOWER, upper=UPPER)
        mle = {"exact_max": exact_opt.log_likelihood.double(),
               "exact_theta": exact_opt.theta[:, :THETA_DIM]}
        for iters in MLE_ITERS:
            res = maximize_likelihood(model, start, MASK, max_iterations=iters,
                                      timesteps=32, device=device, scale=SCALE,
                                      lower=LOWER, upper=UPPER)
            pf_fine = model.log_prob(res.theta, MASK, device=device, timesteps=100).double()
            exact_at = family.log_lik(res.theta[:, :THETA_DIM], res.theta[:, THETA_DIM:],
                                      smooth[1e-3])
            mle[f"iters_{iters}"] = {"pf_at_mle": pf_fine, "exact_at_mle": exact_at,
                                     "theta": res.theta[:, :THETA_DIM],
                                     "evaluations": res.evaluations}
        out["mle"] = mle
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--variant", default="", help='training-recipe tag, e.g. "loglam"')
    parser.add_argument("--best-checkpoint", action="store_true",
                        help="with --cross-only: evaluate Model_checkpoint.pt (best validation "
                             "epoch) into out['cross_best'] (C9.2)")
    parser.add_argument("--cross-only", action="store_true",
                        help="recompute only the cross-family block of an existing result")
    args = parser.parse_args()

    family = Family(args.family, args.nu)
    device = "cuda" if torch.cuda.is_available() and not args.oracle else "cpu"
    name = "oracle" if args.oracle else (
        f"n{args.budget}_s{args.seed}" + (f"_{args.variant}" if args.variant else ""))
    path = os.path.join(RESULT_ROOT, family.tag, f"{name}.pt")
    if args.cross_only:
        out = torch.load(path, weights_only=False)
        key = "cross_best" if args.best_checkpoint else "cross"
        if key in out and "x" in out[key]["student"]:
            print(f"{key} up to date: {path}")
            return
        ckpt = "Model_checkpoint.pt" if args.best_checkpoint else "final.pt"
        model = SBIm.load(os.path.join(model_dir(args.family, args.nu, args.budget, args.seed, args.variant),
                                       ckpt), device=device)
        model.model.eval()
        out[key] = cross_eval(model, device)
        torch.save(out, path)
        print(f"cross recomputed: {path}")
        return
    if os.path.exists(path):
        print(f"exists: {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if args.oracle:
        model = SBIm(nodes_size=5, sde_type="vesde", sigma=25.0, hidden_size=8, depth=1,
                     num_heads=1, mlp_ratio=1)
        model.model = LikelihoodOracle(family, model.sde)
    else:
        model = SBIm.load(os.path.join(model_dir(args.family, args.nu, args.budget, args.seed, args.variant),
                                       "final.pt"), device=device)
        model.model.eval()
    t0 = time.time()
    out = evaluate(model, family, device, trained=not args.oracle)
    out["meta"] = dict(family=args.family, nu=args.nu, budget=args.budget, seed=args.seed,
                       oracle=args.oracle, seconds=time.time() - t0)
    torch.save(out, path)
    ref = out["settings"]["steps_100"]["logp"] - out["exact"][1e-3]
    print(f"{path}: ref error mean {ref.mean():+.4f} sd {ref.std():.4f} "
          f"({out['meta']['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
