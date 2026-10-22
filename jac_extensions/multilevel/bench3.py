#!/usr/bin/env python3
"""Three-level benchmark for compass.experimental.multilevel with a trained network.

Model (linear-Gaussian, so the joint posterior is exact):
    g ~ N(0, I_2),  h_k = A_h g + 0.7 eps (dim 2),  l_kj = A_l h_k + 0.7 eps (dim 1),
    x_kj = B (g, h_k, l_kj) + 0.5 eps (dim 2).
One network is trained on single paths (g, h, l, x). Arms, all scored against the exact
posterior over every latent (g, all h_k, all l_kj):
    oracle_jac   exact path scores + exact Jacobian  (floor of the rule + sampler)
    jac_fd       network, FD Jacobian, refresh 1
    jac_fd_r9    network, FD Jacobian, refresh 9
    gh_pilot     network, pilot covariance from 500 single-observation draws per leaf
    jac_noprior  network, FD, mid-level prior factors dropped (ablation; wrong by design)

    python bench3.py train
    python bench3.py run [--configs 4x4 4x16 4x64 16x16] [--datasets 10]
    python bench3.py report
"""
import os
import sys
from pathlib import Path

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-CPU_THREAD_LIMIT:])
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(CPU_THREAD_LIMIT)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "Arruda" / "paper_benchmarks"))
import gpu  # noqa: E402

if __name__ == "__main__":
    gpu.claim(int(os.environ.get("NEED_MB", "1500")))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from compass.experimental.multilevel import (  # noqa: E402
    LinearGaussianTree, MultiLevelSampler, NestedTree, SBIMPathModel,
    single_observation_draws,
)

OUT = HERE / "artifacts" / os.environ.get("BENCH_TAG", "bench3")
DIMS = [2, 2, 1]
X = 2
D = sum(DIMS)
NODES = D + X
EPS = 1e-5
SDE_KWARGS = {"sde_type": "vesde", "sigma": 8.0}
NETWORK = {"hidden_size": 64, "depth": 4, "num_heads": 4, "mlp_ratio": 2}
TRAIN = int(os.environ.get("TRAIN", 200_000))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_model(K, n):
    g = torch.Generator().manual_seed(0)
    A = [0.8 * torch.randn(DIMS[m], DIMS[m - 1], dtype=torch.float64, generator=g) for m in (1, 2)]
    B = torch.randn(X, D, dtype=torch.float64, generator=g)
    return LinearGaussianTree(NestedTree.balanced(DIMS, [K, n]), A, [0.7, 0.7], B, nu=0.5)


def network(checkpoint=None):
    import common
    model = make_model(1, 1)
    path_sd = model.path_cov[-1].diagonal().sqrt().tolist()
    x_sd = (model.B @ model.path_cov[-1] @ model.B.T).diagonal().add(0.25).sqrt().tolist()
    return common.build(NODES, SDE_KWARGS, NETWORK, DEVICE, EPS, prior_std=path_sd,
                        latent_indices=list(range(D)), sigma_data=path_sd + x_sd,
                        checkpoint=checkpoint)


def train(epochs):
    import common
    from compass import ConditionalMenu
    model = make_model(1, 1)
    g = torch.Generator().manual_seed(7)
    theta, x = model.simulate_paths(TRAIN, g)
    theta_val, x_val = model.simulate_paths(20_000, g)
    torch.manual_seed(7)
    net = network()
    menu = ConditionalMenu(net.schema, {"theta": range(D), "x": range(D, NODES)},
                           [("posterior", ["x"])], bernoulli_fraction=0.1)
    directory = OUT / "model"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with common.floor_training_eps(EPS):
        net.train(theta=theta.float(), x=x.float(), theta_val=theta_val.float(),
                  x_val=x_val.float(), batch_size=1024, max_epochs=epochs,
                  early_stopping_patience=epochs, lr=5e-4, time_sampling="log_sigma",
                  mask_sampler=menu, device=DEVICE, verbose=True, path=str(directory))
    net.save(path=str(directory), name="Model_final")
    print(f"trained in {time.time() - started:.0f}s", flush=True)


def load():
    import common
    return common.freeze(network(OUT / "model" / "Model_final.pt"))


def metrics(flat, mean, cov, tree):
    sd = cov.diagonal().sqrt()
    z = ((flat.mean(0) - mean) / sd).numpy()
    ratio = (flat.std(0) / sd).numpy()
    out = {}
    start = 0
    for name, n, d in zip(("g", "h", "l"), tree.sizes, tree.dims):
        sl = slice(start, start + n * d)
        out[f"{name}_mean_rmse"] = float(np.sqrt(np.mean(z[sl] ** 2)))
        out[f"{name}_logsd"] = float(np.sqrt(np.mean(np.log(ratio[sl]) ** 2)))
        start += n * d
    corr = cov / sd[:, None] / sd[None, :]
    emp = torch.corrcoef(flat.T)
    cross = slice(0, DIMS[0] + tree.sizes[1] * DIMS[1])            # g and h block
    out["gh_corr_err"] = float((emp[cross, cross] - corr[cross, cross]).abs().max())
    return out


def run(configs, datasets, samples, steps):
    net = load()
    rows = []
    path = OUT / "results.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["config"], r["dataset"], r["arm"]))
    for config in configs:
        K, n = (int(v) for v in config.split("x"))
        model = make_model(K, n)
        tree = model.tree
        for ds in range(datasets):
            g = torch.Generator().manual_seed(1000 + ds)
            _, x = model.simulate(generator=g)
            mean, cov = model.posterior(x)
            exact_score, exact_jac = model.path_score(x)
            net_score = SBIMPathModel(net, x, list(range(D)), list(range(D, NODES)), device=DEVICE)
            lam_max = float(net.sde.lambda_t(torch.tensor([1.0])))

            def pilot():
                draws = single_observation_draws(net_score, tree.sizes[-1], D, 500, lam_max,
                                                 lam_min=float(net.sde.lambda_t(torch.tensor([EPS]))),
                                                 steps=steps, generator=torch.Generator().manual_seed(ds),
                                                 device=DEVICE)
                c = draws - draws.mean(1, keepdim=True)
                return (c.mT @ c / (draws.shape[1] - 1)).cpu()

            arms = {
                "oracle_jac": dict(score_fn=exact_score, precision="jacobian", jacobian_fn=exact_jac),
                "jac_fd": dict(score_fn=net_score, precision="jacobian_fd"),
                "jac_fd_r9": dict(score_fn=net_score, precision="jacobian_fd", refresh=9),
                "gh_pilot": dict(score_fn=net_score, precision="pilot"),
                "jac_noprior": dict(score_fn=net_score, precision="jacobian_fd",
                                    drop_internal_priors=True),
            }
            for arm, kw in arms.items():
                if (config, ds, arm) in done:
                    continue
                t0 = time.time()
                if arm == "gh_pilot":
                    kw["pilot_covariance"] = pilot()
                sampler = MultiLevelSampler(tree, prior=model.prior(), **kw)
                lam_min = float(net.sde.lambda_t(torch.tensor([EPS])))
                out = sampler.sample(samples, lam_max=lam_max, lam_min=lam_min, steps=steps,
                                     generator=torch.Generator().manual_seed(10_000 + ds),
                                     device="cpu")
                flat = tree.flatten([o.cpu() for o in out])
                rec = {"config": config, "dataset": ds, "arm": arm, "seconds": time.time() - t0,
                       "projected": {str(k): v for k, v in sampler.diagnostics.items()},
                       **metrics(flat, mean, cov, tree)}
                print(json.dumps(rec), flush=True)
                with path.open("a") as handle:
                    handle.write(json.dumps(rec) + "\n")


def report():
    import collections
    rows = [json.loads(line) for line in (OUT / "results.jsonl").read_text().splitlines()]
    keys = ["g_mean_rmse", "h_mean_rmse", "l_mean_rmse", "g_logsd", "h_logsd", "l_logsd",
            "gh_corr_err", "seconds"]
    table = collections.defaultdict(list)
    for r in rows:
        table[(r["config"], r["arm"])].append(r)
    lines = ["| config | arm | n | " + " | ".join(keys) + " |",
             "|---|---|---|" + "---|" * len(keys)]
    for (config, arm), rs in sorted(table.items(), key=lambda kv: (int(kv[0][0].split('x')[0]) * int(kv[0][0].split('x')[1]), kv[0][1])):
        vals = [np.mean([r[k] for r in rs]) for k in keys]
        lines.append(f"| {config} | {arm} | {len(rs)} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    text = "\n".join(lines)
    (OUT / "report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "run", "report"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--configs", nargs="+", default=["4x4", "4x16", "4x64", "16x16"])
    parser.add_argument("--datasets", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        train(args.epochs)
    elif args.command == "run":
        run(args.configs, args.datasets, args.samples, args.steps)
    else:
        report()
