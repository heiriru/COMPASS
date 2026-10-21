#!/usr/bin/env python3
"""Is the composed posterior too narrow, or is the ABC reference too wide?

The reference convolves the likelihood with a Gaussian of width ``bandwidth`` in
observation space, which *inflates* it: with n observations the induced variance
inflation compounds. So a composed posterior narrower than the reference is not
by itself evidence of overconfidence -- the comparison is only fair in the limit
``bandwidth -> 0``, where the Monte Carlo estimate also gets noisier.

This script measures that trade directly: the reference's own width as the
bandwidth shrinks, at a draw count raised to keep the estimate supported. If the
reference width extrapolates to the composed width, the rules are calibrated and
the earlier gap was the reference's bias. If it plateaus above them, the rules
really are overconfident.

Usage:
    python inverse_kinematics/bandwidth_study.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import hierarchical as H  # noqa: E402

ROOT = Path(__file__).resolve().parent


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bandwidths", type=float, nargs="+",
                        default=[0.20, 0.10, 0.05, 0.035, 0.025])
    result.add_argument("--draws", type=int, default=400_000)
    result.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    store = np.load(arguments.artifacts / "inference" / "posteriors.npz")
    metrics = json.loads(
        (arguments.artifacts / "inference" / "metrics.json").read_text()
    )
    x = np.asarray(store["observations"])

    composed = {
        method: float(store[f"samples_{method}"][0, :, H.HIERARCHY[0]].std())
        for method in metrics["methods"]
    }
    print("Composed posterior widths:")
    for method, width in composed.items():
        print(f"  {method:20} {width:.4f}")
    print(f"\n{len(x)} observations, {arguments.draws:,} local draws per point, "
          f"{len(arguments.seeds)} seeds\n")
    print(f"{'bandwidth':>10}{'mean':>10}{'sd':>10}{'sd spread':>11}{'seconds':>9}")

    rows = []
    for bandwidth in arguments.bandwidths:
        started = time.perf_counter()
        means, stds = [], []
        for seed in arguments.seeds:
            grid, density = H.reference_global(
                x, draws=arguments.draws, bandwidth=bandwidth, seed=seed,
            )
            mean, std, _ = H.grid_moments(grid, density)
            means.append(mean)
            stds.append(std)
        seconds = time.perf_counter() - started
        # Spread across seeds is the Monte Carlo error of the estimate itself;
        # once it is comparable to the trend, the ladder has bottomed out.
        row = {
            "bandwidth": bandwidth,
            "mean": float(np.mean(means)),
            "std": float(np.mean(stds)),
            "std_spread": float(np.std(stds)),
            "seconds": seconds,
        }
        rows.append(row)
        print(f"{bandwidth:>10.3f}{row['mean']:>10.4f}{row['std']:>10.4f}"
              f"{row['std_spread']:>11.4f}{seconds:>9.0f}")

    output = arguments.artifacts / "inference" / "bandwidth_study.json"
    output.write_text(json.dumps(
        {"composed_widths": composed, "observations": len(x),
         "draws": arguments.draws, "seeds": arguments.seeds, "rows": rows},
        indent=2, sort_keys=True,
    ))
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
