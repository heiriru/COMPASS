#!/usr/bin/env python3
"""Covariance output vs Jacobian vs pilot on Arruda's Gaussian toy (exact posterior).

The toy's recipe (h64d4, 200k sims, PreconditionedScore, eps 1e-5, posterior menu, final
weights) plus a covariance output on the trunk, trained jointly (``cov``) or readout-only
(``cov_det``, the score training unchanged). On each network:

    gh       gauss_hierarchical (pilot covariance, 500 draws per row)
    jac_r9   gauss_jacobian_FD, refresh 9 (the library default)
    netcov   gauss_jacobian_FD with the network's covariance output in place of the Jacobian

    python toy.py train --arm cov
    python toy.py run --arm cov --observations 10 100 1000
    python toy.py report
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "jac_extensions"))
import gpu  # noqa: E402

if __name__ == "__main__" and "report" not in sys.argv:
    gpu.claim(int(os.environ.get("NEED_MB", "3000")))
sys.path.insert(0, str(ROOT / "Arruda" / "paper_benchmarks"))
import gaussian_toy as toy  # noqa: E402  (caps CPU threads)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402
import compass  # noqa: E402
from compass import ConditionalMenu, MultiObsSampler  # noqa: E402
from compass.covariance_output import (  # noqa: E402
    CovarianceObjective, NetworkCovarianceHead, enable_covariance_output,
)

OUT = Path(os.environ.get("COVOUT_TOY", "/export/data/rheinric/covout/toy"))
RANK = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METHODS = {
    "gh": {"method": "dpm", "order": 2, "correction": "gauss_hierarchical",
           "precision_est_samples": 500},
    "jac_r9": {"method": "dpm", "order": 2, "correction": "gauss_jacobian_FD",
               "jacobian_refresh": 9},
    "netcov": {"method": "dpm", "order": 2, "correction": "gauss_jacobian_FD",
               "jacobian_refresh": 1},
    # the network output on the library's default schedule (refresh 9)
    "netcov_r9": {"method": "dpm", "order": 2, "correction": "gauss_jacobian_FD",
                  "jacobian_refresh": 9},
}
MAX_ROWS = {"gh": 20_000, "jac_r9": 48_000, "netcov": 20_000, "netcov_r9": 20_000}   # shared 11 GB cards


def directory(arm):
    return OUT / arm


def network(arm, checkpoint=None):
    model = toy.network(DEVICE)
    enable_covariance_output(model, rank=RANK,
                             node_scale=[1.0] * toy.DIM + [2.0 ** 0.5] * toy.DIM)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        model.model.load_state_dict(state.get("model_state_dict", state))
    return model


def train(arm, epochs):
    generator = torch.Generator().manual_seed(7)
    torch.manual_seed(7)
    theta, x = toy.simulate(toy.TRAIN_SAMPLES, generator)
    theta_val, x_val = toy.simulate(20_000, generator)
    model = network(arm)
    menu = ConditionalMenu(model.schema, {"eta": range(toy.DIM), "y": range(toy.DIM, toy.NODES)},
                           [("posterior", "y")], bernoulli_fraction=0.1)
    out = directory(arm)
    out.mkdir(parents=True, exist_ok=True)
    objective = CovarianceObjective(model.covariance_output, weight=1.0,
                                    detach_trunk=(arm == "cov_det"))
    started = time.time()
    with common.floor_training_eps(toy.EPS):
        model.train(theta=theta, x=x, theta_val=theta_val, x_val=x_val, batch_size=1024,
                    max_epochs=epochs, early_stopping_patience=epochs, lr=5e-4,
                    time_sampling="log_sigma", mask_sampler=menu, device=DEVICE,
                    verbose=False, path=str(out), covariance_objective=objective)
    model.save(path=str(out), name="Model_final")
    (out / "training.json").write_text(json.dumps({
        "minutes": (time.time() - started) / 60, "epochs": epochs,
        "val_loss": model.trainer.val_loss}))
    print(f"trained {arm} in {(time.time() - started) / 60:.1f} min", flush=True)


def run(arm, observations, datasets, num_samples, methods):
    model = common.freeze(network(arm, directory(arm) / "Model_final.pt"))
    path = directory(arm) / "results.jsonl"
    done = set()
    if path.exists():
        done = {(r["J"], r["method"], r["dataset"]) for r in map(json.loads, path.read_text().splitlines())}
    for J in observations:
        for name in methods:
            for index in range(datasets if J < 1000 else min(datasets, 3)):
                if (J, name, index) in done:
                    continue
                eta, y = toy.dataset(J, seed=1000 * J + index)
                torch.manual_seed(index)
                sampler = MultiObsSampler(model)
                head = None
                if name.startswith("netcov"):
                    head = NetworkCovarianceHead(model, toy.HIERARCHY, list(range(toy.DIM, toy.NODES)))
                    sampler.set_covariance_head(head)
                started = time.perf_counter()
                samples = common.sample_chunked(
                    sampler.sample, J, num_samples, max_rows=MAX_ROWS[name], world_size=1,
                    data=torch.as_tensor(y, dtype=torch.float32), condition_mask=toy.condition_mask(),
                    timesteps=toy.TIMESTEPS, eps=toy.EPS, hierarchy=toy.HIERARCHY,
                    prior=(torch.zeros(toy.DIM), torch.ones(toy.DIM)), device=DEVICE,
                    verbose=False, **METHODS[name])
                seconds = time.perf_counter() - started
                shared = samples.numpy()[0, :, :toy.DIM].astype(np.float64)
                row = toy.dataset_metrics(shared, eta, y)
                row.pop("covered")
                row.update(arm=arm, method=name, J=J, dataset=index, seconds=seconds)
                print(json.dumps(row), flush=True)
                with path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")


def report():
    lines = ["| net | J | method | n | KL | mean err/sd | width ratio | s/dataset |",
             "|---|---|---|---|---|---|---|---|"]
    for arm_dir in sorted(OUT.glob("*")):
        path = arm_dir / "results.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        for J in sorted({r["J"] for r in rows}):
            for name in METHODS:
                rs = [r for r in rows if r["J"] == J and r["method"] == name]
                if not rs:
                    continue
                med = lambda k: np.median([r[k] for r in rs])  # noqa: E731
                lines.append(f"| {arm_dir.name} | {J} | {name} | {len(rs)} | {med('kl'):.3g} | "
                             f"{med('mean_error_sd'):.3f} | {med('width_ratio'):.3f} | {med('seconds'):.0f} |")
    text = "\n".join(lines)
    (OUT / "TABLE.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "run", "report"])
    parser.add_argument("--arm", default="cov", choices=["cov", "cov_det"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--observations", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--datasets", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        train(args.arm, args.epochs)
    elif args.command == "run":
        run(args.arm, args.observations, args.datasets, args.num_samples, args.methods)
    else:
        report()
