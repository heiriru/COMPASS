#!/usr/bin/env python3
"""Markdown tables from the sweep, so the README's numbers are never retyped.

Emits, for each metric, a table of medians over replicates: rows are ``n``,
columns are the arms, one block per regime. The ``exact_mc`` column is the
Monte-Carlo floor and the ``gaussian prior`` column, when
``prior_ablation.json`` is present, is the analytic control -- exact
per-observation posteriors composed with the moment-matched Gaussian prior, i.e.
what the rule would return with a perfect network.

Usage:
    python report.py --tag main main_n32
    python report.py --tag diagnostic diagnostic_n32 --metric branch_mass
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402  (before numeric imports)

configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import datasets  # noqa: E402
import plot_scaling as P  # noqa: E402

ROOT = Path(__file__).resolve().parent


def table(rows, regime, observations, names, metric, ablation=None):
    header = ["n"] + list(names) + (["exact scores + gaussian prior"]
                                    if ablation else [])
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for n in observations:
        cells = [str(n)]
        for name in names:
            median, _, _ = P.series(rows, regime, name, metric, [n])
            cells.append("--" if not np.isfinite(median[0])
                         else f"{median[0]:.3f}")
        if ablation:
            value = ablation.get((regime, n))
            cells.append("--" if value is None else f"{value:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tag", nargs="+", default=["main", "main_n32"])
    result.add_argument("--metric", default="w1_over_sigma")
    result.add_argument("--regimes", nargs="+", default=None)
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    result.add_argument("--output", type=Path, default=None)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    rows, _, _ = P.load_all(arguments.artifacts, arguments.tag)
    regimes = arguments.regimes or [regime for regime in datasets.REGIMES
                                    if regime in P.present(rows, "regime")]
    observations = sorted(P.present(rows, "observations"))
    names = ["exact_mc"] + P.method_order(rows)
    ablation = (P.ablation_curve(arguments.artifacts)
                if arguments.metric == "w1_over_sigma" else None)

    blocks = [f"### {arguments.metric}, median over replicates "
              f"({' + '.join(arguments.tag)})\n"]
    for regime in regimes:
        blocks.append(f"**{regime}**\n")
        blocks.append(table(rows, regime, observations, names,
                            arguments.metric, ablation))
        blocks.append("")
    text = "\n".join(blocks)
    print(text)
    output = arguments.output or (
        Path(arguments.artifacts) / "scaling"
        / f"table_{arguments.metric}_{arguments.tag[0]}.md"
    )
    output.write_text(text)
    print(f"-> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
