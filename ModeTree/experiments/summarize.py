#!/usr/bin/env python
"""Turn the experiment JSONs into the markdown tables used by REPORT.md."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import json  # noqa: E402

ARTIFACTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
METHOD_ORDER = ["uncorrected", "gauss_jacobian", "fnpe",
                "modetree_known", "modetree", "modetree_plus"]


def load(name):
    path = os.path.join(ARTIFACTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    if value != value:                       # nan
        return "-"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def comparison_tables(rows, ns=(8, 32, 100)):
    blocks = []
    cases = sorted({r["case"] for r in rows})
    for n in ns:
        body = []
        for case in cases:
            for method in METHOD_ORDER:
                match = [r for r in rows if r["case"] == case and r["n"] == n
                         and r["method"] == method and "error" not in r]
                if not match:
                    continue
                r = match[0]
                body.append([
                    case, method,
                    fmt(r.get("w1_analytic", r.get("w1"))),
                    fmt(r.get("e_map_sigma"), 3)
                    + ("*" if r.get("map_ambiguous") else ""),
                    fmt(r.get("weight_tv")),
                    fmt(r.get("missing_mass")),
                    fmt(r.get("seconds"), 2),
                    fmt(r.get("score_field_calls")),
                ])
        blocks.append(f"#### n = {n}\n\n" + table(
            ["case", "method", "W1", "e_MAP/sd", "weight TV", "missing mass",
             "s", "field calls"], body))
    return "\n\n".join(blocks)


def speed_table(rows):
    body = []
    for n in sorted({r["n"] for r in rows}):
        entry = {}
        for method in ("gauss_jacobian", "fnpe", "modetree_plus"):
            times = [r["seconds"] for r in rows
                     if r["n"] == n and r["method"] == method and "error" not in r]
            entry[method] = sum(times) / len(times) if times else None
        ratio = (entry["modetree_plus"] / entry["fnpe"]
                 if entry["fnpe"] and entry["modetree_plus"] else None)
        body.append([n, fmt(entry["gauss_jacobian"], 2), fmt(entry["fnpe"], 2),
                     fmt(entry["modetree_plus"], 2), fmt(ratio, 2)])
    return table(["n", "gauss_jacobian (s)", "F-NPSE (s)", "ModeTree+ (s)",
                  "ModeTree / F-NPSE"], body)


def ablation_table(rows, ns=(8, 32, 100)):
    blocks = []
    for n in ns:
        body = []
        for row in rows:
            if row["n"] != n or "error" in row:
                continue
            label = row["variant"] + ("+laplace+" if row.get("laplace_plus") else "")
            body.append([
                row["case"], label, fmt(row["mode_recall"], 2),
                fmt(row["missing_mass"]), fmt(row["weight_tv"]),
                fmt(row["e_map_sigma"], 3) + ("*" if row.get("map_ambiguous") else ""),
                fmt(row["n_branches"]), fmt(row["seconds"], 2),
                fmt(row["jacobian_evaluations"]),
            ])
        blocks.append(f"#### n = {n}\n\n" + table(
            ["case", "variant", "recall", "missing mass", "weight TV",
             "e_MAP/sd", "B", "s", "JAC evals"], body))
    return "\n\n".join(blocks)


def two_d_table(rows):
    body = []
    for row in rows:
        tag = ("both" if row["sentinel"] and row["curvature"]
               else "curvature only" if row["curvature"] else "sentinel only")
        body.append([
            row["regime"], row["n"], tag, fmt(row["mode_recall"], 2),
            fmt(row["missing_mass"]), fmt(row["false_modes"]),
            fmt(row["weight_tv"]), fmt(row.get("cycle_residual")),
            fmt(row["seconds"], 2),
        ])
    return table(["regime", "n", "discovery", "recall", "missing mass",
                  "false", "weight TV", "max cycle residual", "s"], body)


def hierarchical_table(rows):
    body = []
    for row in rows:
        if "error" in row:
            body.append([f"{row['d_g']}x{row['d_l']}", row["n"], row["method"],
                         "FAILED", "-", "-", "-"])
            continue
        body.append([
            f"{row['d_g']}x{row['d_l']}", row["n"], row["method"],
            fmt(row.get("w1")), fmt(row.get("mode_recall"), 2),
            f"{row.get('n_branches', '-')}/{row.get('n_true_modes', '-')}",
            fmt(row["seconds"], 2),
        ])
    return table(["d_g x d_l", "n", "method", "W1 (global)", "recall",
                  "B / true", "s"], body)


def score_error_table(rows):
    body = []
    for row in rows:
        ns = row["n"]
        index = ns.index(100)
        body.append([
            row["shape"], row["magnitude"], f"{row['alpha_score']:+.3f}",
            fmt(row["score_error"][0]), fmt(row["score_error"][index]),
            fmt(row["map_error"][index]), fmt(row["weight_tv"][index]),
            row["modes_found"][index],
        ])
    return table(["error shape", "magnitude", "alpha", "|dS| (n=1)",
                  "|dS| (n=100)", "e_MAP (n=100)", "weight TV (n=100)",
                  "modes found"], body)


def experiment1_table(rows, ns=(1, 8, 100)):
    body = []
    for row in rows:
        if row["n"] not in ns:
            continue
        body.append([
            row["case"], row["n"], fmt(row["score_norm_max"]),
            fmt(row["e_map_sigma"]) + ("*" if row.get("map_ambiguous") else ""),
            fmt(row["height_error"]), fmt(row["tv"]), fmt(row["tv_plus"]),
            fmt(row.get("tv_plus_unguarded")),
            fmt(row.get("ess_fraction_min"), 3),
        ])
    return table(["case", "n", "max |S|", "e_MAP/sd", "energy-gap err",
                  "TV (Laplace)", "TV (Laplace+)", "TV (Laplace+, no guard)",
                  "min ESS frac"], body)


def main():
    sections = []
    experiment1 = load("experiment1_known_modes.json")
    if experiment1:
        sections.append("### Experiment 1 - known modes, oracle clean score\n\n"
                        + experiment1_table(experiment1))
    comparison = load("compare_methods.json") or load("compare_methods_quick.json")
    if comparison:
        sections.append("### Method comparison\n\n" + comparison_tables(comparison))
        sections.append("### Runtime vs F-NPSE\n\n" + speed_table(comparison))
    ablations = load("ablations.json")
    if ablations:
        sections.append("### Ablations\n\n" + ablation_table(ablations))
    two_d = load("experiment_2d.json")
    if two_d:
        sections.append("### Two dimensions / remote small mode\n\n"
                        + two_d_table(two_d))
    hierarchical = load("experiment_hierarchical.json")
    if hierarchical:
        sections.append("### Hierarchical\n\n" + hierarchical_table(hierarchical))
    score_error = load("score_error_scaling.json")
    if score_error:
        sections.append("### Score-error accumulation with n\n\n"
                        + score_error_table(score_error))
    calibration = load("calibration.json")
    if calibration:
        body = [[q, fmt(v, 3)] for q, v in sorted(calibration["coverage"].items())]
        sections.append(
            f"### Calibration ({calibration['datasets']} datasets, "
            f"n = {calibration['n']})\n\n"
            + table(["nominal", "empirical"], body))
    text = "\n\n".join(sections) + "\n\n(* = exactly tied mode heights; the MAP "
    text += "identity is ambiguous and the error is taken to the nearest tied mode.)\n"
    path = os.path.join(ARTIFACTS, "TABLES.md")
    with open(path, "w") as handle:
        handle.write(text)
    print(text)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
