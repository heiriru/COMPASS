#!/usr/bin/env python3
"""Roadmap 2.9 on hierarchical two moons: target score matching on the LOCALS too.

    python two_moons.py train                 # attn_hier_tsm_gl (100k sims, 200 epochs, as attn_hier_tsm)
    python two_moons.py score  10 25 100      # checks/score_check.py on it (per-group global score)
    python two_moons.py run --sigma-max 0.6 --Ns 1 10 25 100 --methods attn_hier_tsm_gl:gauss_jacobian_FD

``attn_hier_tsm_gl`` is ``attn_hier_tsm`` (same data, seed, network, recipe and the same
global TSM target with w = 1 / (1 + lambda^2)) plus the local target

    T_l = grad_theta [ log TN(theta; mu, sigma^2, [-1, 1]^2) + log p(x | theta) ]

in network units (theta / local_scale), blended by inverse variance with the measured
Var T_l ~ 1550 (analysis/two_moons_local_tsm_variance.py): w_l = 1 / (1 + 1550 lambda^2),
1/2 at lambda 0.025, 0.98 at the training floor lambda 0.003. The likelihood is the
hsbibm crescent: p = x - (-|z_0|, z_1), (u, v) = p - (0.25, 0), r = |(u, v)| ~ N(0.1, 0.01^2),
density N(r) / (pi r). Validation is left unseeded (as the existing arms were trained), so the
only difference from attn_hier_tsm is the local target. Nothing in the benchmark is edited:
the arm is registered in-process. Library: ``COMPASS_SRC``.
"""
import os
import sys
from pathlib import Path

CPU = sorted(os.sched_getaffinity(0))[:3]
os.sched_setaffinity(0, CPU)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "3"
print(f"CPU cap: {CPU}", flush=True)

SRC = os.environ.get("COMPASS_SRC", str(Path(__file__).resolve().parents[1] / "src"))
B = Path("/export/home/rheinric/COMPASS/benchmarks/hierarchical_two_moons")
C = B.parents[1]
sys.path[:0] = [SRC, str(B / ".vendor"), str(B / "external" / "hsbibm"), str(B),
                str(C / "Arruda" / "paper_benchmarks"), str(C / "ModeTree" / "experiments")]
import compass  # noqa: E402  (before compass_arms, which would put the main tree's src first)
assert Path(compass.__file__).resolve().is_relative_to(Path(SRC).resolve()), compass.__file__

import math  # noqa: E402
import torch  # noqa: E402
import task as T  # noqa: E402
import compass_arms as CA  # noqa: E402

NAME = "attn_hier_tsm_gl"
LOCAL_VARIANCE = 1550.0
CA.NETWORKS[NAME] = {"arch": "tokenized", "attention": "hierarchical"}
CA.METHODS[f"{NAME}:gauss_jacobian_FD"] = (NAME, None, {**CA.DPM, "correction": "gauss_jacobian_FD",
                                                        **CA.FD_SCHEDULE})
CA.METHODS[f"{NAME}:arruda_direct"] = (NAME, {"estimator": "direct"},
                                       {**CA.DPM, "correction": "arruda_schur"})


def local_target(nodes):
    """grad of log TN(theta | g) + log p(x | theta) w.r.t. theta / local_scale, (B, 2)."""
    from reference import _log_diff_ndtr
    norm = T.normalization()
    scale = norm["local_scale"]
    with torch.enable_grad():
        nodes = torch.as_tensor(nodes).detach().double()
        z = nodes[:, :T.GLOBALS]
        theta = (nodes[:, T.GLOBALS:T.GLOBALS + T.LOCAL] * scale).requires_grad_(True)
        x = nodes[:, T.GLOBALS + T.LOCAL:] * torch.as_tensor(norm["x_std"]).to(nodes) \
            + torch.as_tensor(norm["x_mean"]).to(nodes)
        u_ = torch.special.ndtr(z)
        mu = T.MU_LOW + (T.MU_HIGH - T.MU_LOW) * u_[:, :2]
        sigma = T.SIGMA_LOW + (T.SIGMA_HIGH - T.SIGMA_LOW) * u_[:, 2:]
        log_tn = (-0.5 * ((theta - mu) / sigma) ** 2).sum(-1)   # normaliser is theta-free
        c, s = math.cos(T.ROTATION), math.sin(T.ROTATION)
        z0 = c * theta[:, 0] - s * theta[:, 1]
        z1 = s * theta[:, 0] + c * theta[:, 1]
        u = x[:, 0] + z0.abs() - T.BASE_OFFSET
        v = x[:, 1] - z1
        r = torch.sqrt(u ** 2 + v ** 2).clamp_min(1e-6)
        log_lik = -0.5 * ((r - T.R_LOC) / T.R_SCALE) ** 2 - torch.log(r)
        (grad,) = torch.autograd.grad((log_tn + log_lik).sum(), theta)
    return grad * scale


def target(nodes):
    return torch.cat([T.global_score_target(nodes, prior_scale=1.0).double(),
                      local_target(nodes)], 1).to(torch.as_tensor(nodes).dtype)


def variance(noisy, lam):
    v = torch.tensor([1.0] * T.GLOBALS + [LOCAL_VARIANCE] * T.LOCAL, dtype=noisy.dtype,
                     device=noisy.device)
    return v.expand_as(noisy)


def install_tsm():
    from compass import ScoreBasedInferenceModel
    original = ScoreBasedInferenceModel.train

    def train(self, *args, **kwargs):
        kwargs.update(global_target_fn=target, global_target_indices=list(range(T.GLOBALS + T.LOCAL)),
                      tsm_variance_fn=variance, validation_noise_seed=None)
        print("[two_moons] TSM on globals + locals", flush=True)
        return original(self, *args, **kwargs)
    ScoreBasedInferenceModel.train = train


def main():
    action, rest = sys.argv[1], sys.argv[2:]
    if action == "train":
        install_tsm()
        CA.train(NAME, "cuda", 100_000, 200, seed=7)
    elif action == "score":
        import runpy
        sys.argv = [str(B / "checks" / "score_check.py"), NAME] + rest
        runpy.run_path(sys.argv[0], run_name="__main__")
    elif action == "run":
        import benchmark
        sys.argv = [str(B / "benchmark.py"), "run"] + rest
        benchmark.main()
    else:
        raise SystemExit(f"unknown action {action!r}")


if __name__ == "__main__":
    main()
