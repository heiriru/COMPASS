#!/usr/bin/env python
"""Merge roots into basins and write the first-order topology report.

Root *counts* proved unstable under grid resolution, profiling iterations and
local initialization (the local profiling problem is itself multimodal, so
``l*(g)`` is not unique). Root *clusters* did not. This merges roots separated
by less than ``--merge`` into one basin and reports the merged answer alongside
the raw one, so the stability claim is visible rather than asserted.

Run:  python ModeTree/experiments/report_first_order_topology.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402

import numpy as np  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
OUT = ARTIFACTS / "first_order_topology"


def merge(values, tolerance):
    """Single-link clustering of 1-D roots; returns cluster centres."""
    if not values:
        return []
    order = sorted(values)
    clusters, current = [], [order[0]]
    for value in order[1:]:
        if value - current[-1] <= tolerance:
            current.append(value)
        else:
            clusters.append(current)
            current = [value]
    clusters.append(current)
    return [float(np.mean(c)) for c in clusters]


def match(found, reference):
    """Greedy nearest-reference assignment -> (errors, recall, false count)."""
    if not found:
        return [], 0.0, 0
    remaining = list(reference)
    errors, matched = [], 0
    for value in sorted(found, key=lambda v: min(
            (abs(v - r) for r in reference), default=np.inf)):
        if not remaining:
            continue
        best = min(remaining, key=lambda r: abs(value - r))
        if abs(value - best) < 0.5:          # half a typical mode separation
            errors.append(abs(value - best))
            remaining.remove(best)
            matched += 1
    return errors, matched / max(len(reference), 1), len(found) - matched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", type=float, default=0.05)
    args = parser.parse_args(argv)

    lines = []
    for problem in ("ik", "gm"):
        path = OUT / f"{problem}_rows.json"
        if not path.exists():
            continue
        rows = json.load(open(path))
        lines.append(f"\n### {problem.upper()} — roots merged at "
                     f"separation {args.merge}\n")
        lines.append("| n | t | source | raw roots | merged basins | "
                     "basin locations | reference | mode error | recall | false |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in sorted(rows, key=lambda r: (r["score_source"], r["n"], -r["t"])):
            reference = row["reference"]["modes"]
            merged = merge(row["modes_found"], args.merge)
            errors, recall, false = match(merged, reference)
            lines.append(
                f"| {row['n']} | {row['t']:g} | {row['score_source']} | "
                f"{len(row['modes_found'])} | **{len(merged)}** | "
                f"{', '.join(f'{m:+.3f}' for m in merged)} | "
                f"{', '.join(f'{m:+.3f}' for m in reference)} | "
                f"{(max(errors) if errors else float('nan')):.3f} | "
                f"{recall:.2f} | {false} |")
    text = "\n".join(lines)
    (OUT / "merged_basins.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
