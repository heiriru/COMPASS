#!/usr/bin/env python
"""Assemble the section-16 table and the convergence evidence for exp6.

Reads the stage JSON files in ``artifacts/gm_oracle`` and writes
``TABLE.md``.  ``--tag`` selects the ``mu_0`` variant (e.g. ``_mu7.5``).

An arm is reported at its *most converged* configuration -- the top of its own
effort ladder -- not at its best-scoring one, and the ladder itself is printed
underneath so a plateau can be checked rather than asserted.
"""

import argparse
import json
import math
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "gm_oracle"

COLUMNS = [
    ("W1/sd", "w1_over_sd", "{:.4f}"),
    ("W2/sd", "w2_over_sd", "{:.4f}"),
    ("recall", "mode_recall", "{:.2f}"),
    ("wTV", "weight_tv", "{:.4f}"),
    ("+mass", "positive_branch_mass", "{:.3f}"),
    ("valley", "valley_mass", "{:.4f}"),
    ("dip", "dip_ratio", "{:.3f}"),
    ("loc-", "mode_location_error_sd", "{:.3f}"),
    ("width", "width_ratio", "{:.3f}"),
    ("mean", "mean_error_sd", "{:.3f}"),
    ("s", "seconds", "{:.0f}"),
    ("rowevals", "row_score_evaluations", "{:.0f}"),
    ("NFE", "nfe", "{:.0f}"),
]


def load(name, tag):
    path = ARTIFACTS / f"{name}{tag}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def cell(row, key, fmt):
    value = row.get(key)
    if isinstance(value, list):
        value = value[0]
    if value is None:
        return "-"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def render(rows, title):
    head = "| method | " + " | ".join(name for name, _, _ in COLUMNS) + " |"
    rule = "|---|" + "|".join("---:" for _ in COLUMNS) + "|"
    lines = [f"### {title}", "", head, rule]
    for label, row in rows:
        if row is None:
            lines.append(f"| {label} | " + " | ".join("-" for _ in COLUMNS) + " |")
            continue
        lines.append(f"| {label} | "
                     + " | ".join(cell(row, key, fmt) for _, key, fmt in COLUMNS)
                     + " |")
    lines.append("")
    return lines


def best(rows, prefix, key="w1_over_sd"):
    """The lowest-``key`` configuration whose name starts with ``prefix``.

    Diverged arms are skipped rather than compared: their ``W1`` is ``nan`` or
    astronomically large, and ``min`` over ``nan`` silently returns whichever
    one it met first, which would report a blown-up run as the best setting.
    When every setting diverged that is the honest answer, so the first of them
    is returned and its ``diverged`` flag carries the fact.
    """
    candidates = {k: v for k, v in rows.items() if k.startswith(prefix)}
    if not candidates:
        return None, None
    finite = {k: v for k, v in candidates.items()
              if not v.get("diverged")
              and isinstance(v.get(key), (int, float))
              and math.isfinite(v[key])}
    pool = finite or candidates
    name = min(pool, key=lambda k: pool[k].get(key, float("inf")))
    return name, pool[name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    tag = args.tag

    reference = load("reference", tag)
    control = load("control", tag)
    control_churn = load("control_churn", tag)
    control_init = load("control_init", tag)
    geffner = load("geffner", tag)
    ladder = load("geffner_ladder", tag)
    arruda = load("arruda", tag)
    arruda_churn = load("arruda_churn", tag)
    linhart = load("linhart", tag)
    linhart_churn = load("linhart_churn", tag)
    naive = load("naive", tag)
    fielderror = load("fielderror", tag)

    lines = [f"# Oracle bridge comparison{tag or ' (mu_0 = 2.5, the benchmark default)'}",
             ""]
    for n in sorted(reference, key=int):
        meta = reference[n]
        exact = meta["exact_draw_metrics"]
        lines += [
            f"## n = {n}", "",
            f"Exact posterior: branch weights {[round(w, 4) for w in meta['branch_weights']]}, "
            f"branch means {[round(m, 4) for m in meta['branch_means']]}, "
            f"within-branch sd {meta['within_branch_sd']:.4f}, "
            f"mode separation {exact['mode_separation_in_within_sd']:.2f} within-branch sds, "
            f"exact dip ratio {exact['exact_dip_ratio']:.4f} "
            f"({exact['exact_num_modes']} exact modes).",
            "",
            f"Monte-Carlo floor for W1/sd at 2000 exact draws: "
            f"{meta['w1_floor_mean']:.4f} +- {meta['w1_floor_sd']:.4f}.",
            "",
        ]
        rows = [("exact draws", exact)]
        sources = [
            ("TRUE-SCORE-DPM2 (churn=0, as run)", control.get(n, {}),
             "dpm2/steps=500"),
            ("TRUE-SCORE-DPM2 (churn=0.5, corrected)", control_churn.get(n, {}),
             "dpm2/churn=0.5,steps=500"),
            ("TRUE-SCORE-DPM2 (churn=0, exact init)", control_init.get(n, {}),
             "dpm2/steps=500,exact-init"),
            ("TRUE-SCORE-Langevin", ladder.get(n, {}),
             "true/T=100,L=200,snr=0.05"),
            ("ORACLE-FNPSE (Langevin, converged)", ladder.get(n, {}),
             "fnpe/T=100,L=500,snr=0.05" if n != "128"
             else "fnpe/T=100,L=200,snr=0.05"),
            ("ORACLE-NAIVE-DPM2 (churn=0.5)", linhart_churn.get(n, {}),
             "naive/dpm2,churn=0.5,steps=500"),
            ("GEFFNER-DPM2 (churn=0.5)", linhart_churn.get(n, {}),
             "fnpe/dpm2,churn=0.5,steps=500"),
            ("ORACLE-GAUSS (churn=0.5)", linhart_churn.get(n, {}),
             "gauss/dpm2,churn=0.5,steps=500"),
            ("ORACLE-JAC-DPM2 (churn=0.5)", linhart_churn.get(n, {}),
             "jac/dpm2,churn=0.5,steps=500"),
        ]
        for label, table, key in sources:
            rows.append((label, table.get(key)))
        name, row = best(arruda_churn.get(n, {}), "", "w1_over_sd")
        rows.append((f"ORACLE-ARRUDA ({name})" if name else "ORACLE-ARRUDA", row))
        lines += render(rows, f"Required table, n = {n}")

        for title, table in (
                ("TRUE-SCORE sampler ladder", control.get(n, {})),
                ("TRUE-SCORE churn sweep", control_churn.get(n, {})),
                ("TRUE-SCORE start/schedule", control_init.get(n, {})),
                ("Langevin ladder: true score and F-NPSE bridge",
                 ladder.get(n, {})),
                ("F-NPSE Langevin sweep (first pass)",
                 geffner.get(n, {}).get("langevin", {})),
                ("Arruda damping, adaptive-Heun tolerance ladder",
                 arruda.get(n, {})),
                ("Arruda damping under the corrected sampler",
                 arruda_churn.get(n, {})),
                ("Linhart DPM2 step ladder (churn = 0)", linhart.get(n, {})),
                ("Composed rules under the corrected sampler",
                 linhart_churn.get(n, {})),
                ("Naive and Geffner-DPM2 controls (churn = 0)",
                 naive.get(n, {}))):
            if table:
                lines += render(sorted(table.items()), f"{title}, n = {n}")

        errors = fielderror.get(n, {})
        if errors:
            lines += ["#### Score-field error against the exact diffused tall score",
                      "",
                      "RMS error on the shared coordinate, relative to the RMS of the "
                      "exact score, at states drawn from the true diffused tall "
                      "posterior.  For GAUSS and JAC this is the quantity those "
                      "rules are trying to make small.  For the F-NPSE and damped "
                      "bridges it is **not** an error: those are scores of "
                      "deliberately different intermediate distributions, and the "
                      "column is shown only to make the size of that deliberate "
                      "difference visible.", "",
                      "| tau | lambda | naive | fnpe (bridge) | arruda (bridge) | gauss | jac |",
                      "|---:|---:|---:|---:|---:|---:|---:|"]
            for key in sorted(errors, key=lambda k: -float(k.split("=")[1])):
                entry = errors[key]
                lines.append(
                    f"| {entry['tau']:g} | {entry['lambda']:.4f} | "
                    + " | ".join(f"{entry[a]['relative']:.3f}" for a in
                                 ("naive", "fnpe-diffused", "arruda-d1=0.2",
                                  "gauss", "jac")) + " |")
            lines.append("")

        properness = geffner.get(n, {}).get("properness")
        if properness:
            lines += ["#### F-NPSE bridge properness", "",
                      "| prior | t | lambda | a(t) | tail log-density change | normalisable |",
                      "|---|---:|---:|---:|---:|---|"]
            for kind, entries in properness.items():
                for entry in entries:
                    lines.append(
                        f"| {kind} | {entry['t']:g} | {entry['lambda']:.4f} | "
                        f"{entry['a']:.3f} | {entry['tail_log_density_change']:.3g} | "
                        f"{'yes' if entry['normalisable'] else '**no**'} |")
            lines.append("")

    path = ARTIFACTS / f"TABLE{tag}.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
