#!/usr/bin/env python3
"""Compare mask laws by where they send the gradient. Pure arithmetic, no data.

    python comparisons/simformer/analyze_mask_laws.py

Answers one question: is the Simformer condition-mask law structurally better
balanced than COMPASS's, or does it merely sit on tasks where any
dimension-proportional law is already balanced?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.mask_allocation import (BENCHMARKS, bernoulli_law, mix,
                                    posterior_share, simformer_law,
                                    six_conditional_law)


def laws_for(sizes):
    roles = [r for r, n in sizes.items() if n]
    six_roles = {"g", "l", "x"} & set(roles)
    laws = {
        "COMPASS Bernoulli(0.33)": bernoulli_law(roles),
        "Simformer five-way": simformer_law(roles),
    }
    if six_roles:
        uniform = six_conditional_law()
        weighted = six_conditional_law([7, 7, 7, 1, 1, 1])
        laws["COMPASS six (uniform)"] = uniform
        laws["COMPASS six [7,7,7,1,1,1]"] = weighted
        laws["COMPASS six_mixed"] = mix(weighted, bernoulli_law(roles), "1/4")
    return laws


def main():
    names = list(laws_for({"g": 1, "l": 1, "x": 1}))
    width = max(len(n) for n in names)
    header = "  ".join(f"{n:>{width}}" for n in names)
    print(f"Share of loss-bearing (latent) nodes that are PARAMETERS, "
          f"per training draw\n")
    print(f"{'benchmark':<26}  dim(theta):dim(x)  {header}")
    for label, sizes in BENCHMARKS.items():
        laws = laws_for(sizes)
        theta = sizes["g"] + sizes["l"]
        ratio = f"{theta}:{sizes['x']}"
        row = "  ".join(
            f"{posterior_share(laws[n], sizes):>{width}.3f}" if n in laws
            else f"{'--':>{width}}" for n in names)
        print(f"{label:<26}  {ratio:>16}  {row}")

    print("\nSame table, with the measured partial-pooling difficulty applied:")
    print("observation nodes cost 6x a parameter node to denoise "
          "(docs/followup_results.md)")
    difficulty = {"x": 6}
    for label, sizes in BENCHMARKS.items():
        laws = laws_for(sizes)
        theta = sizes["g"] + sizes["l"]
        ratio = f"{theta}:{sizes['x']}"
        row = "  ".join(
            f"{posterior_share(laws[n], sizes, difficulty):>{width}.3f}"
            if n in laws else f"{'--':>{width}}" for n in names)
        print(f"{label:<26}  {ratio:>16}  {row}")


if __name__ == "__main__":
    main()
