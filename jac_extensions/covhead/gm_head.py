#!/usr/bin/env python3
"""Covariance head on the GM benchmark: train it, and grade it against the exact
Tweedie covariance next to the FD Jacobian it replaces.

    python gm_head.py train --mode hierarchical
    python gm_head.py accuracy --mode hierarchical
Composition is then run unchanged through experiments_arruda_schur/gm.py with
COMPASS_COVARIANCE_HEAD=artifacts/gm_<mode>/head.pt (see run_gm.sh).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = Path("/export/home/rheinric/COMPASS")
sys.path.insert(0, str(HERE.parent))
import gpu  # noqa: E402

if __name__ == "__main__":
    gpu.claim(int(os.environ.get("NEED_MB", "1500")))
sys.path.insert(0, str(MAIN / "experiments_arruda_schur"))
sys.path.insert(0, str(MAIN / "Gaussian_mixtures_compositional_score_modelling"))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import compass  # noqa: E402  worktree copy first: experiments_arruda_schur/common.py
import compass.MultiObsSampler  # noqa: E402,F401  prepends the main checkout's src
import compass.experimental.covariance_head  # noqa: E402,F401
import compass.experimental.multilevel  # noqa: E402,F401
assert "COMPASS-jacext" in compass.__file__, compass.__file__

torch.set_num_threads(3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDAS = [0.032, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


def load_network(mode):
    import common
    import gm
    import train
    recipe = gm.DEFAULT_RECIPE
    if mode == "none":
        gm.link_production_checkpoint(recipe)
    train.load_to = common.load_with_latent_attention(train.load_to)
    directory = train.model_directory(gm.artifacts(mode), recipe["config"], recipe["arch"],
                                      recipe["simulations"], recipe["mask_menu"],
                                      recipe["time_sampling"])
    network, _, _ = train.train_or_load(directory, recipe["config"], recipe["arch"],
                                        recipe["simulations"], DEVICE,
                                        mask_menu=recipe["mask_menu"],
                                        time_sampling=recipe["time_sampling"], verbose=False)
    for p in network.model.parameters():
        p.requires_grad_(False)
    return network


def simulate(count, generator):
    import model as M
    rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator)))
    rows = torch.as_tensor(M.DEFAULT.simulate(count, rng), dtype=torch.float32)
    return rows[:, :2], rows[:, 2:]


def out_dir(mode):
    path = HERE / "artifacts" / f"gm_{mode}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_head(mode, steps):
    from compass.experimental.covariance_head import train_covariance_head
    network = load_network(mode)
    lam = lambda t: float(network.sde.lambda_t(torch.tensor([t])))   # noqa: E731
    head = train_covariance_head(network, simulate, [0, 1], [2], lam(1e-3), lam(1.0),
                                 steps=steps, batch=2048, lr=1e-3, device=DEVICE)
    torch.save(head.cpu(), out_dir(mode) / "head.pt")
    json.dump(head.history, open(out_dir(mode) / "head_history.json", "w"))


def score_fn(model, sde):
    """y -> score (y coordinates, VE), latents [0, 1], observed [2]."""
    def f(y, x, lam):
        t = sde.time_of_lambda(torch.as_tensor(lam, dtype=torch.float32, device=y.device)).reshape(1, 1)
        state = torch.cat([y, x], -1).float()
        c = torch.tensor([0.0, 0.0, 1.0], device=y.device).expand(y.shape[0], -1)
        s = model(x=state, t=t.expand(y.shape[0], 1), c=c)
        s = s / sde.marginal_prob_std(t)
        return s[:, :2].double()
    return f


def accuracy(mode, count):
    import model as M
    from oracle import OracleScore
    from compass.experimental.multilevel import finite_difference_jacobian, jacobian_precision
    network = load_network(mode)
    sde = network.sde
    head = torch.load(out_dir(mode) / "head.pt", weights_only=False).to(DEVICE)
    oracle = OracleScore(M.DEFAULT, sde).to(DEVICE)
    net_score = score_fn(network.model, sde)
    exact_score = score_fn(oracle, sde)
    generator = torch.Generator().manual_seed(123)
    theta, x = simulate(count, generator)
    theta, x = theta.to(DEVICE), x.to(DEVICE)
    eye = torch.eye(2, dtype=torch.float64, device=DEVICE)
    rows = []
    for lam in LAMBDAS:
        y = theta + lam * torch.randn(theta.shape, generator=generator).to(DEVICE)
        with torch.no_grad():
            s_net = net_score(y, x, lam)
            s_ex = exact_score(y, x, lam)
            J_fd = finite_difference_jacobian(lambda p, l: net_score(p.float(), x.repeat(3, 1), l),
                                              y.double(), lam)
            J_ex = finite_difference_jacobian(lambda p, l: exact_score(p.float(), x.repeat(3, 1), l),
                                              y.double(), lam, step=1e-3)
            K_head = head(y, x, torch.full((count,), lam, device=DEVICE)).double()

        def info(precision):
            return precision - eye / lam ** 2
        K_ex = info(jacobian_precision(J_ex, lam, floor=1e-6, lower_clip=False))
        K_fd = info(jacobian_precision(J_fd, lam, floor=1e-3))
        K_ex_clip = info(jacobian_precision(J_ex, lam, floor=1e-3))

        def rel(a, b):
            return ((a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1).clamp_min(1e-12))

        def cov(K):
            return torch.linalg.inv(K + eye / lam ** 2)
        antisym = ((J_fd - J_fd.mT).flatten(1).norm(dim=1) / J_fd.flatten(1).norm(dim=1))
        row = {
            "lam": lam,
            "score_rel_err": float(rel(s_net, s_ex).median()),
            "jac_rel_err_fd": float(rel(J_fd, J_ex).median()),
            "jac_antisym_fd": float(antisym.median()),
            "info_rel_err_fd": float(rel(K_fd, K_ex_clip).median()),
            "info_rel_err_head": float(rel(K_head, K_ex_clip).median()),
            "cov_rel_err_fd": float(rel(cov(K_fd), cov(K_ex)).median()),
            "cov_rel_err_head": float(rel(cov(K_head), cov(K_ex)).median()),
            "neg_info_frac_exact": float((torch.linalg.eigvalsh(0.5 * (K_ex + K_ex.mT))[:, 0] < -1e-8).double().mean()),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    json.dump(rows, open(out_dir(mode) / "accuracy.json", "w"), indent=1)
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(f"{r[k]:.3g}" for k in keys) + " |" for r in rows]
    (out_dir(mode) / "accuracy.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "accuracy"])
    parser.add_argument("--mode", default="hierarchical")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--count", type=int, default=4096)
    args = parser.parse_args()
    if args.command == "train":
        train_head(args.mode, args.steps)
    else:
        accuracy(args.mode, args.count)
