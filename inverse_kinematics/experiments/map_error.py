#!/usr/bin/env python3
"""Rank every saved inverse-kinematics run by MAP distance to the true angle.

Why this metric is worth promoting: it contains **no reference at all**. The ABC
reference disagrees with itself by W1 0.424 at n = 32 (see
``reference_noise_floor.py``), so every W1 comparison at n >= 32 is inside the
referee's own scatter. The generating value is known exactly -- it is an input to
``bimodality_study.selected_dataset`` -- so MAP-to-truth is measurable without a
referee at any n.

The shared posterior is bimodal by the reflection
``(a1,a2,a3,h) -> (-a1,-a2,-a3,-h)``, and both ``+0.6`` and ``-0.6`` are
equally the truth under it, so the error reported is

    map_error = min_s |argmax p(a) - s * 0.6| = | |MAP| - 0.6 |.

Two companions are reported alongside, because a single number hides the two
distinct ways to fail:

* ``mode_error`` -- the same distance for the *nearest* mode rather than the
  tallest one. A method that finds the right location but puts the taller peak
  on the mirror branch scores badly on ``map_error`` and well here.
* ``endpoint`` -- the mean tip-to-observation residual, also reference-free,
  carried through from the run's own JSON.

Usage:
    python map_error.py                    # rank everything
    python map_error.py --observations 64  # one n
"""
from __future__ import annotations

import os

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

AR = Path(__file__).resolve().parent.parent / "artifacts" / "mixture_inference"
TRUE_ANGLE = 0.6


def statistics(grid, density, truth=TRUE_ANGLE):
    density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        return None
    density = density / area
    peak = float(grid[int(np.argmax(density))])
    interior = np.arange(1, len(grid) - 1)
    hit = interior[(density[interior] >= density[interior - 1])
                   & (density[interior] >= density[interior + 1])
                   & (density[interior] > 0.05 * density.max())]
    modes = [float(grid[i]) for i in hit] or [peak]
    nearest = min(modes, key=lambda v: min(abs(v - truth), abs(v + truth)))
    return {
        "map": peak,
        "map_error": float(abs(abs(peak) - truth)),
        "mode_error": float(min(abs(nearest - truth), abs(nearest + truth))),
        "modes": modes,
        "mass_positive": float(np.trapezoid(density[grid > 0], grid[grid > 0])),
    }


def scan(observations=None):
    rows = []
    seen_reference = set()
    for npz in sorted(AR.glob("*/inverse_kinematics_n*.npz")):
        match = re.search(r"inverse_kinematics_n(\d+)\.npz$", npz.name)
        n = int(match.group(1))
        if observations and n not in observations:
            continue
        setting = npz.parent.name
        table = {}
        candidate = npz.with_suffix(".json")
        if candidate.exists():
            table = json.loads(candidate.read_text())
        with np.load(npz) as store:
            grid = store["grid"]
            if n not in seen_reference and "reference" in store:
                seen_reference.add(n)
                statistic = statistics(grid, store["reference"])
                if statistic:
                    rows.append({"n": n, "setting": "(ABC reference)",
                                 "method": "reference", "endpoint": float("nan"),
                                 "w1": float("nan"), **statistic})
            for key in store.files:
                if not key.startswith("curve_"):
                    continue
                method = key[len("curve_"):]
                statistic = statistics(grid, store[key])
                if statistic is None:
                    continue
                entry = table.get(method, {})
                rows.append({"n": n, "setting": setting, "method": method,
                             "endpoint": float(entry.get("endpoint_mean",
                                                          float("nan"))),
                             "w1": float(entry.get("w1_over_sigma",
                                                    float("nan"))),
                             **statistic})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, nargs="+", default=None)
    parser.add_argument("--top", type=int, default=8)
    arguments = parser.parse_args(argv)

    rows = scan(arguments.observations)
    ns = sorted({r["n"] for r in rows})
    for n in ns:
        block = [r for r in rows if r["n"] == n and r["method"] != "reference"]
        block.sort(key=lambda r: r["map_error"])
        reference = next((r for r in rows
                          if r["n"] == n and r["method"] == "reference"), None)
        print(f"\n### n = {n}   best {arguments.top} by |MAP| - 0.6")
        if reference:
            print(f"    the ABC reference itself: MAP {reference['map']:+.3f}  "
                  f"error {reference['map_error']:.3f}")
        print(f"{'rank':>5} {'setting':<22}{'method':<22}{'MAP':>8}"
              f"{'|MAP|-0.6':>11}{'mode err':>10}{'mass>0':>8}"
              f"{'endpoint':>10}{'W1':>8}")
        for rank, row in enumerate(block[:arguments.top], start=1):
            print(f"{rank:5d} {row['setting'][:21]:<22}{row['method'][:21]:<22}"
                  f"{row['map']:+8.3f}{row['map_error']:11.3f}"
                  f"{row['mode_error']:10.3f}{row['mass_positive']:8.2f}"
                  f"{row['endpoint']:10.4f}{row['w1']:8.3f}")

    print("\n\n### overall, averaged over the n values each setting covers")
    by_setting = {}
    for row in rows:
        if row["method"] == "reference":
            continue
        key = (row["setting"], row["method"])
        by_setting.setdefault(key, []).append(row)
    summary = []
    for (setting, method), group in by_setting.items():
        summary.append({
            "setting": setting, "method": method,
            "covers": sorted(r["n"] for r in group),
            "mean_map_error": float(np.mean([r["map_error"] for r in group])),
            "worst": float(np.max([r["map_error"] for r in group])),
            "endpoint": float(np.nanmean([r["endpoint"] for r in group])),
        })
    summary.sort(key=lambda r: r["mean_map_error"])
    print(f"{'setting':<22}{'method':<22}{'mean':>8}{'worst':>8}"
          f"{'endpoint':>10}  covers")
    for row in summary[:20]:
        print(f"{row['setting'][:21]:<22}{row['method'][:21]:<22}"
              f"{row['mean_map_error']:8.3f}{row['worst']:8.3f}"
              f"{row['endpoint']:10.4f}  {row['covers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
