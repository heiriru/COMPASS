#!/usr/bin/env python3
"""Mini-batched JAC (``jacobian_row_batch``) on the hierarchical AR(1) grid.

Runs Arruda/paper_benchmarks/ar1_grid.py's own evaluate() (same datasets, cached exact
references, same metrics) with extra arms; only the result files are redirected here.
Compare with the stored n10000 results there (gauss_hierarchical, gauss_jacobian_FD_r9_big).

    python ar1_rowbatch.py --grids 16 64 --arms rb20 rb100
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = Path("/export/home/rheinric/COMPASS/Arruda/paper_benchmarks")
sys.path.insert(0, str(HERE.parent))
import gpu  # noqa: E402

if __name__ == "__main__" and "--report" not in sys.argv:
    gpu.claim(int(os.environ.get("NEED_MB", "3000")))
sys.path.insert(0, str(BENCH))
import ar1_grid as ar1  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402

import compass  # noqa: E402
assert "COMPASS-jacext" in compass.__file__, compass.__file__
OUT = HERE / "artifacts" / "ar1"


def add_arms(grid):
    groups = grid * grid
    for frac in (20, 100):
        ar1.METHODS[f"rb{frac}"] = {"method": "dpm", "order": 2, "correction": "gauss_jacobian_FD",
                                    "jacobian_row_batch": max(1, groups // frac),
                                    "max_rows": 4_000_000}


def result_path(name, grid, train_samples):
    return OUT / f"{name}_J{grid}x{grid}.json"


def report():
    stored = BENCH / "artifacts" / "ar1_grid" / "results" / "n10000"
    lines = ["| grid | arm | datasets | global RMSE (norm) | global |mean err|/sd | global width | local mean err/sd | local width | s/dataset |",
             "|---|---|---|---|---|---|---|---|---|"]
    for grid in (16, 64):
        files = [(p.stem.replace(f"_J{grid}x{grid}", ""), p) for p in sorted(OUT.glob(f"*_J{grid}x{grid}.json"))]
        for name in ("gauss_hierarchical", "gauss_jacobian_FD_r9_big", "reference"):
            p = stored / f"{name}_J{grid}x{grid}.json"
            if p.exists():
                files.insert(0, (name + " (stored)", p))
        for name, p in files:
            s = json.load(open(p))["summary"]
            if "global_rmse_norm" not in s:
                continue
            lines.append(f"| {grid} | {name} | {s['datasets'] - s['failed']} | {s['global_rmse_norm']:.3f} | "
                         f"{s['global_mean_error_sd']:.3f} | {s['global_width_ratio']:.3f} | "
                         f"{s['local_mean_error_sd']:.3f} | {s['local_width_ratio']:.3f} | {s['seconds']:.0f} |")
    text = "\n".join(lines)
    (OUT / "TABLE.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--arms", nargs="+", default=["rb20", "rb100"])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
        sys.exit()
    ar1.result_path = result_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ar1.load(device, 10_000)
    for grid in args.grids:
        add_arms(grid)
        datasets = len(list((ar1.OUT / "reference").glob(f"J{grid}x{grid}_*.npz")))
        for arm in args.arms:
            ar1.evaluate(model, arm, grid, datasets, 100, device, 10_000)
    report()
