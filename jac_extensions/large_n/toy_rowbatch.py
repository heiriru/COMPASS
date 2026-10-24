#!/usr/bin/env python3
"""Mini-batched JAC (``jacobian_row_batch``) on Arruda's Gaussian toy at J = 1,000-10,000.

Reuses the stored toy network and metrics from Arruda/paper_benchmarks/gaussian_toy.py
(main checkout, read-only). The full-row JAC arm (FD, refresh 9) is the reference
for accuracy at J = 1000; the stored TABLE.md has GH/JAC numbers and times for J <= 1000.

    python toy_rowbatch.py --observations 1000 10000 --arms full_r9 rb10 rb50
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOY = Path("/export/home/rheinric/COMPASS/Arruda/paper_benchmarks")
sys.path.insert(0, str(HERE.parent))
import gpu  # noqa: E402

if __name__ == "__main__":
    gpu.claim(int(os.environ.get("NEED_MB", "3500")))
sys.path.insert(0, str(TOY))
import gaussian_toy as toy  # noqa: E402  (caps CPU threads on import)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402
from compass import MultiObsSampler  # noqa: E402

OUT = HERE / "artifacts" / "toy"
CACHE_ROWS = 500_000          # observations x draws per chunk (bounds the K/eta cache)


def arm_kwargs(arm, J):
    base = {"method": "dpm", "order": 2, "correction": "gauss_jacobian_FD"}
    if arm == "full_r9":
        return {**base, "jacobian_refresh": 9}, 96_000
    if arm.startswith("rb"):
        frac = int(arm[2:])                         # refresh J/frac rows per call
        return {**base, "jacobian_row_batch": max(1, J // frac)}, CACHE_ROWS
    raise ValueError(arm)


def run(arm, J, datasets, num_samples, device):
    path = OUT / f"{arm}_J{J}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["dataset"] for l in path.read_text().splitlines()}
    model = toy.load(device)
    for index in range(datasets):
        if index in done:
            continue
        eta, y = toy.dataset(J, seed=1000 * J + index)
        kwargs, max_rows = arm_kwargs(arm, J)
        torch.manual_seed(index)
        sampler = MultiObsSampler(model)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        samples = common.sample_chunked(
            sampler.sample, J, num_samples, max_rows=max_rows, world_size=1,
            data=torch.as_tensor(y, dtype=torch.float32), condition_mask=toy.condition_mask(),
            timesteps=toy.TIMESTEPS, eps=toy.EPS, hierarchy=toy.HIERARCHY,
            prior=(torch.zeros(toy.DIM), torch.ones(toy.DIM)), device=device, verbose=False,
            **kwargs)
        seconds = time.perf_counter() - started
        shared = samples.numpy()[0, :, :toy.DIM].astype(np.float64)
        row = toy.dataset_metrics(shared, eta, y)
        row.pop("covered")
        row.update(dataset=index, arm=arm, J=J, seconds=seconds,
                   rows=int(sampler.solver_stats["evaluated_row_samples"]),
                   peak_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else 0))
        print(json.dumps(row, default=float), flush=True)
        with path.open("a") as handle:
            handle.write(json.dumps(row, default=float) + "\n")


def report():
    lines = ["| J | arm | n | KL | |mean err|/sd | width ratio | s/dataset | network rows | peak MB |",
             "|---|---|---|---|---|---|---|---|---|"]
    for path in sorted(OUT.glob("*.jsonl"), key=lambda p: (int(p.stem.split("_J")[1]), p.stem)):
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        def m(k):
            return np.median([r[k] for r in rows])
        lines.append(f"| {rows[0]['J']} | {rows[0]['arm']} | {len(rows)} | {m('kl'):.3g} | "
                     f"{m('mean_error_sd'):.3g} | {m('width_ratio'):.3f} | {m('seconds'):.0f} | "
                     f"{m('rows'):.3g} | {m('peak_mb'):.0f} |")
    text = "\n".join(lines)
    (OUT / "TABLE.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--arms", nargs="+", default=["full_r9", "rb10", "rb50"])
    parser.add_argument("--datasets", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
        sys.exit()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for J in args.observations:
        for arm in args.arms:
            if arm == "full_r9" and J > 1000:
                continue                                 # ~hours per dataset; see TABLE
            run(arm, J, args.datasets, args.num_samples, device)
    report()
