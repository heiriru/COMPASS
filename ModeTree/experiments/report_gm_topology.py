#!/usr/bin/env python
"""Assemble the tables for the bridge-topology experiment.

Two tables per setting:

``TOPOLOGY``  the bridge-geometry table -- ``tau_split``, mode positions,
              barrier depth and branch mass at matched levels, per arm.  These
              are the experiment's primary quantities; endpoint error is a
              consequence, not the diagnostic.
``ENDPOINT``  the sampled arms, with the full metric set of section 1.

A diverged arm is printed as ``diverged`` and never competes for "best": a
blown-up run scores whatever the metric happens to return, and letting it win
a minimum is how a divergence gets reported as a result.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "gm_topology"
ORACLE = Path(__file__).resolve().parents[1] / "artifacts" / "gm_oracle"

SHOW_TAUS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)


def load(name, tag="", root=ARTIFACTS):
    path = root / f"{name}{tag}.json"
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value, places=4):
    if value is None:
        return "--"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt(v, places) for v in value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{places}f}"


def endpoint_row(name, row):
    if row.get("diverged"):
        return (f"| {name} | diverged | -- | -- | -- | -- | -- | "
                f"{fmt(row.get('seconds'), 1)} | "
                f"{row.get('row_score_evaluations')} | {row.get('nfe')} |")
    return (f"| {name} | {fmt(row['w1_over_sd'])} | {fmt(row['mode_recall'], 2)}"
            f" | {fmt(row['weight_tv'])} | {fmt(row['positive_branch_mass'], 3)}"
            f" | {fmt(row['valley_mass'], 4)} | {fmt(row['width_ratio'], 3)}"
            f" | {fmt(row.get('seconds'), 1)}"
            f" | {row.get('row_score_evaluations')} | {row.get('nfe')} |")


ENDPOINT_HEADER = (
    "| arm | W1/sd | mode recall | weight TV | w_+ | valley mass | "
    "width ratio | seconds | row scores | NFE |\n"
    "|---|---|---|---|---|---|---|---|---|---|")


def topology_table(tag, out):
    truth = load("truth", tag)
    topoerr = load("topoerr", tag)
    if truth is None:
        return
    out.append(f"\n## Bridge topology{tag or ' (standard setting)'}\n")
    for n in sorted(truth, key=int):
        reference = truth[n]
        out.append(f"\n### n = {n}\n")
        out.append(f"within-branch sd {fmt(reference['within_branch_sd'])}; "
                   f"exact branch means {fmt(reference['exact_branch_means'], 3)}; "
                   f"exact branch weights {fmt(reference['exact_branch_weights'], 3)}\n")
        splits = [f"true {reference['tau_split']}"]
        if topoerr and n in topoerr:
            for arm, entry in topoerr[n].items():
                splits.append(f"{arm} {entry['tau_split']} "
                              f"(d={fmt(entry['delta_tau_split'], 3)})")
        out.append("**tau_split**: " + "; ".join(str(s) for s in splits) + "\n")

        out.append("\n| tau | lambda | arm | modes | m_- | m_+ | barrier | w_+ |")
        out.append("|---|---|---|---|---|---|---|---|")
        for tau in SHOW_TAUS:
            key = f"tau={tau:.4f}"
            if key not in reference["levels"]:
                continue
            entries = [("true", reference["levels"][key])]
            if topoerr and n in topoerr:
                entries += [(arm, topoerr[n][arm]["levels"][key])
                            for arm in topoerr[n]]
            for arm, entry in entries:
                out.append(
                    f"| {tau:g} | {entry['lambda']:.4f} | {arm} | "
                    f"{entry['num_modes']} | {fmt(entry.get('mode_minus'), 3)} | "
                    f"{fmt(entry.get('mode_plus'), 3)} | "
                    f"{fmt(entry.get('barrier'), 3)} | "
                    f"{fmt(entry.get('w_plus'), 4)} |")

        validation = reference.get("slice_validation", {})
        if validation:
            out.append("\nSlice-vs-marginal validation (the 1-D reduction's "
                       "own error, for the *true* bridge):\n")
            out.append("| tau | slice modes | marginal modes | slice w_+ | "
                       "marginal w_+ | slice barrier | marginal barrier |")
            out.append("|---|---|---|---|---|---|---|")
            for tau in SHOW_TAUS:
                key = f"tau={tau:.4f}"
                if key not in validation:
                    continue
                v = validation[key]
                out.append(
                    f"| {tau:g} | {v['slice_modes']} | {v['marginal_modes']} | "
                    f"{fmt(v['slice_w_plus'])} | {fmt(v['marginal_w_plus'])} | "
                    f"{fmt(v['slice_barrier'], 3)} | "
                    f"{fmt(v['marginal_barrier'], 3)} |")


def alpha_table(tag, out):
    topo = load("alpha_topo", tag)
    if topo is None:
        return
    out.append(f"\n## Oracle alpha{tag or ' (standard setting)'}\n")
    out.append("`alpha*` is chosen at each level by the topology objective "
               "alone -- mode count first, then mode positions, barrier depth "
               "and branch mass. The endpoint plays no part in the choice.\n")
    out.append("\n| n | tau | lambda | alpha* | plateau | loss(alpha*) | "
               "loss(alpha=1) | loss(alpha=0) | modes at alpha* | true modes |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in sorted(topo, key=int):
        for tau in SHOW_TAUS:
            key = f"tau={tau:.4f}"
            row = topo[n]["per_level"].get(key)
            if row is None:
                continue
            out.append(
                f"| {n} | {tau:g} | {row['lambda']:.4f} | "
                f"{fmt(row['alpha_star'], 3)} | "
                f"[{row['alpha_plateau'][0]:.2f}, {row['alpha_plateau'][1]:.2f}] | "
                f"{fmt(row['loss'], 3)} | {fmt(row['loss_at_1'], 3)} | "
                f"{fmt(row['loss_at_0'], 3)} | "
                f"{row['topology']['num_modes']} | "
                f"{row['true_topology']['num_modes']} |")
    out.append("\n### Smooth fits to alpha*\n")
    out.append("| n | form | RMSE | description |")
    out.append("|---|---|---|---|")
    for n in sorted(topo, key=int):
        for name, fit in sorted(topo[n]["fits"].items(),
                                key=lambda item: item[1]["rmse"]):
            out.append(f"| {n} | {name} | {fmt(fit['rmse'])} | "
                       f"{fit['description']} |")

    simple = load("alpha_simple", tag)
    if simple:
        out.append("\n### How simple can alpha be (section 9)\n")
        out.append("| form | RMSE against alpha* | detail |")
        out.append("|---|---|---|")
        forms = simple["forms"]
        out.append(f"| n-only | {fmt(forms['n_only']['rmse'])} | "
                   + ", ".join(f"n={k}: {v:.3f}"
                               for k, v in forms['n_only']['alpha'].items())
                   + " |")
        out.append(f"| tau-only | {fmt(forms['tau_only']['rmse'])} | "
                   "one schedule shared by every n |")
        out.append(f"| separable a(tau) b(n) | {fmt(forms['separable']['rmse'])} | "
                   + ", ".join(f"b({k})={v:.3f}"
                               for k, v in forms['separable']['b_n'].items())
                   + " |")
        out.append(f"| parametric | {fmt(forms['parametric']['rmse'])} | "
                   f"{forms['parametric']['form']}, coefficients "
                   f"{fmt(forms['parametric']['coefficients'], 3)} |")
        out.append("\nCorrelation of `alpha*` with quantities JAC already "
                   "computes:\n")
        names = sorted(next(iter(simple["correlations"].values())))
        out.append("| n | " + " | ".join(names) + " |")
        out.append("|---" * (len(names) + 1) + "|")
        for n in sorted(simple["correlations"], key=int):
            row = simple["correlations"][n]
            out.append(f"| {n} | " + " | ".join(fmt(row[name], 3)
                                                for name in names) + " |")


def endpoint_table(tag, out):
    out.append(f"\n## Endpoints{tag or ' (standard setting)'}\n")
    for name, title in (("cross", "Final cross-comparison (section 14)"),
                        ("topojac", "ORACLE-TOPO-JAC schedules (section 8)"),
                        ("alpha_const", "Constant-alpha sweep (section 6)"),
                        ("alpha_simple", "Simplified alpha forms (section 9)"),
                        ("fnpe_step", "F-NPSE step schedules (sections 10-12)"),
                        ("fnpe_ladder", "F-NPSE convergence ladder (section 13)")):
        data = load(name, tag)
        if data is None:
            continue
        # ``alpha_simple`` stores its sampled arms under a sub-key; every
        # other stage is keyed by n at the top level.
        if name == "alpha_simple":
            data = data["sampled"]
        out.append(f"\n### {title}\n")
        for n in sorted(data, key=int):
            rows = data[n]
            if name == "alpha_const":
                rows = {f"alpha={entry['alpha']:.1f} "
                        f"(split {entry['tau_split']}, "
                        f"loss {entry['mean_topology_loss']:.2f})":
                        entry["sampled"] for entry in data[n].values()}
            out.append(f"\n**n = {n}**\n")
            out.append(ENDPOINT_HEADER)
            for key, row in rows.items():
                if not isinstance(row, dict) or "w1_over_sd" not in row:
                    continue
                out.append(endpoint_row(key, row))
        out.append("")


def stability_table(tag, out):
    for name in ("fnpe_step", "fnpe_ladder"):
        data = load(name, tag)
        if data is None:
            continue
        out.append(f"\n### {name}: stability separated from mixing\n")
        out.append("| n | schedule | diverged | max abs state | max abs score | "
                   "width ratio | branch TV | max level mean move | W1/sd |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for n in sorted(data, key=int):
            for key, row in data[n].items():
                out.append(
                    f"| {n} | {key} | "
                    f"{'YES' if row.get('diverged') else 'no'} | "
                    f"{fmt(row.get('max_abs_state'), 3)} | "
                    f"{fmt(row.get('max_abs_score'), 3)} | "
                    f"{fmt(row.get('width_ratio'), 3)} | "
                    f"{fmt(row.get('weight_tv'))} | "
                    f"{fmt(row.get('max_level_mean_move'), 4)} | "
                    f"{fmt(row.get('w1_over_sd'))} |")


def floors(out):
    """The Monte-Carlo floor from the previous experiment, for calibration."""
    for tag in ("", "_mu7.5"):
        reference = load("reference", tag, ORACLE)
        if reference is None:
            continue
        out.append(f"\n### W1/sd Monte-Carlo floor{tag or ' (standard)'} "
                   "(2000 exact draws)\n")
        out.append("| n | floor mean | floor sd | mode separation (within sd) | "
                   "exact dip ratio |")
        out.append("|---|---|---|---|---|")
        for n in sorted(reference, key=int):
            entry = reference[n]
            out.append(f"| {n} | {fmt(entry['w1_floor_mean'])} | "
                       f"{fmt(entry['w1_floor_sd'])} | "
                       f"{fmt(entry['mode_separation_in_within_sd'], 3)} | "
                       f"{fmt(entry['exact_draw_metrics']['exact_dip_ratio'], 5)} |")


def main():
    for tag in ("", "_mu7.5"):
        out = [f"# Bridge topology, TOPO-JAC and adaptive F-NPSE"
               f"{tag or ' -- standard setting'}",
               "",
               "Every number below is produced with exact analytic scores, "
               "exact score Jacobians and the closed-form tall posterior. "
               "No network is trained or loaded anywhere in this experiment."]
        if tag == "":
            floors(out)
        topology_table(tag, out)
        alpha_table(tag, out)
        endpoint_table(tag, out)
        stability_table(tag, out)
        path = ARTIFACTS / f"TABLE{tag}.md"
        path.write_text("\n".join(out) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
