#!/usr/bin/env python3
"""Render the result CSVs as Markdown tables, so RESULTS.md is never retyped.

Writes ``artifacts/tables.md``. Every number in ``RESULTS.md`` comes from here;
if a run changes, regenerate and paste rather than editing figures by hand.

Usage:
    python make_tables.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"

METHOD_COLUMNS = (
    ("label", "arm", None),
    ("shared_w1_over_sigma", "W1(g)/sigma", "{:.4f}"),
    ("local_w1_over_sigma", "W1(l)/sigma", "{:.4f}"),
    ("modes_found", "modes", "{:.0f}"),
    ("gap_draws", "gap draws", "{:.0f}"),
    ("positive_mass", "P(g>0)", "{:.3f}"),
    ("mode_width", "mode s.d.", "{:.3f}"),
    ("shared_local_correlation", "corr(g,l)", "{:.3f}"),
    ("runtime_seconds", "seconds", "{:.1f}"),
    ("score_network_calls", "score calls", "{:.0f}"),
)

FIDELITY_COLUMNS = (
    ("model", "benchmark", None),
    ("score", "score source", None),
    ("config", "config", None),
    ("parameters", "parameters", "{:,.0f}"),
    ("rel_rmse_lam_0.05", "lam 0.05", "{:.3f}"),
    ("rel_rmse_lam_0.2", "lam 0.2", "{:.3f}"),
    ("rel_rmse_lam_1.0", "lam 1", "{:.3f}"),
    ("rel_rmse_lam_mean", "mean", "{:.3f}"),
)


# Chunking (see run_experiment.draw) splits the 3,000 draws across several
# passes when they do not fit in memory. Total work is unchanged -- each pass
# carries a quarter of the rows -- but the *call count* multiplies, and
# gauss_hierarchical genuinely does re-estimate its pilot covariance per pass,
# so both columns overstate the cost of a chunked arm.
NOTE = ("_Seconds and score calls overstate the cost of any arm whose draws had "
        "to be chunked to fit in memory (the 5.2M-parameter `gauss_jacobian` "
        "arms); the sampled statistics are unaffected._")


def render(rows, columns):
    header = "| " + " | ".join(name for _, name, _ in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, rule]
    for row in rows:
        cells = []
        for key, _, fmt in columns:
            value = row.get(key, "")
            if fmt and value not in ("", None):
                try:
                    value = fmt.format(float(value))
                except (TypeError, ValueError):
                    pass
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    methods = read(ARTIFACTS / "method_metrics.csv")
    fidelity = read(ARTIFACTS / "network_fidelity.csv")

    sections = []
    if fidelity:
        sections.append("### Network fidelity (relative RMS of the learned row "
                        "score against the closed form)\n\n"
                        + render(fidelity, FIDELITY_COLUMNS))
    for model in sorted({row["model"] for row in methods}):
        exact = next(row for row in methods if row["model"] == model)
        sections.append(
            f"### {model}: exact targets\n\n"
            f"modes = {float(exact['modes_exact']):.0f}; "
            f"P(g>0) = {float(exact['positive_mass_exact']):.3f}; "
            f"gap draws = {float(exact['gap_draws_exact']):.2f} of "
            f"{float(exact['posterior_draws']):.0f}; "
            f"mode s.d. = {float(exact['mode_width_exact']):.3f}; "
            f"corr(g,l) = {float(exact['shared_local_correlation_exact']):.3f}; "
            f"sign coherence = {float(exact['sign_coherence_exact']):.3f}"
        )
        for score in ("analytic", "learned_large", "learned"):
            rows = [row for row in methods
                    if row["model"] == model and row["score"] == score]
            if rows:
                sections.append(f"### {model} / {score}\n\n"
                                + render(rows, METHOD_COLUMNS) + "\n\n" + NOTE)

    path = ARTIFACTS / "tables.md"
    path.write_text("\n\n".join(sections) + "\n")
    print(f"Wrote {path}")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
