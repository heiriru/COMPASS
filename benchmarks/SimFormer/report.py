"""Tables and figures from the harness's summary.csv files.

Called by ``run_benchmark.py`` after every finished run, so the comparison is
visible while the queue is still going. Reads ``results/<phase>/summary.csv``
(written by the official ``scoresbibm`` harness, one row per run and metric,
``value`` = the list of per-observation / per-conditional C2ST scores).

Outputs
-------
results/tables/<phase>.csv   long table, one row per (task, budget, method)
results/tables/<phase>.md    wide table, mean +- s.e.m., COMPASS - Simformer delta
results/figures/<phase>.png  C2ST vs number of simulations, one panel per task
"""

import ast
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Fixed categorical order (dataviz reference palette, light mode): colour
# follows the method, never its rank.
METHOD_STYLE = {
    "score_transformer": ("Simformer (official)", "#2a78d6", "o"),
    "compass_legacy": ("COMPASS (legacy trunk)", "#eb6834", "s"),
    "compass_tokenized": ("COMPASS (tokenized trunk)", "#1baf7a", "D"),
    "compass_legacy_euler500": ("COMPASS (legacy, Euler 500)", "#e87ba4", "v"),
    "compass_legacy_dpm200": ("COMPASS (legacy, DPM-2 200)", "#eda100", "P"),
    "compass_legacy_full": ("COMPASS (legacy, full budget)", "#4a3aa7", "X"),
    "compass_legacy_full_eps5": ("COMPASS (legacy, full, eps 1e-5)", "#008300", "*"),
    "compass_legacy_full_fixeps": ("COMPASS (eps bug fixed)", "#eb6834", "s"),
    "compass_shared_t256": ("COMPASS shared-time (1.0M)", "#1baf7a", "D"),
    "compass_shared_t64": ("COMPASS shared-time, t64 (0.44M)", "#e87ba4", "v"),
    "compass_fx_final": ("COMPASS + final weights, all data", "#1baf7a", "D"),
    "compass_fx_condemb": ("COMPASS + token embeddings", "#e87ba4", "v"),
    "compass_fx_euler500": ("COMPASS + Euler 500 sampler", "#4a3aa7", "X"),
    "compass_fx_final_condemb": ("COMPASS + final + embeddings", "#eda100", "P"),
    "compass_fx_lowrank8": ("COMPASS + low-rank time cond. (r=8)", "#2a78d6", "o"),
    "compass_fx_wd": ("COMPASS + weight decay 0.1", "#008300", "*"),
    "compass_fx_notime_input": ("COMPASS, no time modulation (time at input)", "#eb6834", "s"),
    "compass_fx_notime_add": ("COMPASS, no time modulation (additive time/block)", "#4a3aa7", "X"),
    "compass_fx_notime_none": ("COMPASS, no time input at all", "#52514e", "^"),
    "compass_fx_mask6mix": ("COMPASS, mask6mix training masks", "#008300", "D"),
    "compass_fx_fullattn": ("COMPASS + full attention (latents see latents)", "#eda100", "P"),
    "compass_fx_fullattn_small": ("COMPASS small + full attention", "#2a78d6", "v"),
    "compass_fx_inscale": ("COMPASS + input scaling", "#c83c8c", "h"),
    "compass_fx_inscale_fullattn": ("COMPASS + input scaling + full attention", "#0a8a8a", "d"),
    "compass_fx_simopt": ("COMPASS + Simformer optimizer, final weights", "#7a5c00", "p"),
    "compass_fx_simopt_unif": ("COMPASS + Simformer optimizer + uniform time", "#5c7a00", "H"),
    "npe":("NPE (sbi, NSF)", "#52514e", "^"),
}
# Superseded diagnostic arms: kept in the tables, left out of the figures.
FIGURE_EXCLUDE = {"compass_legacy_full_eps5", "compass_legacy_euler500", "compass_legacy_dpm200"}
# Improvement arms get their own table (improve_table), not the main figures.
IMPROVE = ["compass_fx_final", "compass_fx_condemb", "compass_fx_euler500", "compass_fx_final_condemb",
           "compass_shared_t256", "compass_shared_t64", "compass_fx_lowrank8", "compass_fx_wd",
           "compass_fx_notime_input", "compass_fx_notime_add", "compass_fx_notime_none",
           "compass_fx_mask6mix", "compass_fx_fullattn", "compass_fx_fullattn_small",
           "compass_fx_inscale", "compass_fx_inscale_fullattn",
           "compass_fx_simopt", "compass_fx_simopt_unif"]
FIGURE_EXCLUDE |= set(IMPROVE)
TASK_TITLE = {
    "gaussian_linear": "Gaussian Linear", "gaussian_mixture": "Gaussian Mixture",
    "two_moons": "Two Moons", "slcp": "SLCP",
    "tree_all_cond": "Tree (all cond.)", "marcov_chain_all_cond": "HMM (all cond.)",
    "two_moons_all_cond": "Two Moons (all cond.)", "slcp_all_cond": "SLCP (all cond.)",
}
TASK_ORDER = list(TASK_TITLE)


def load(phase, metric=None):
    """Rows of one metric (default: any c2st*), one per (task, budget, method)."""
    path = RESULTS / phase / "summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if metric is None:
        df = df[df["metric"].astype(str).str.startswith("c2st")]
    else:
        df = df[df["metric"].astype(str) == metric]
    rows = []
    for _, r in df.iterrows():
        try:
            vals = [float(v) for v in ast.literal_eval(r["value"])]
        except Exception:
            continue
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        rows.append({
            "task": r["task"], "budget": int(r["num_simulations"]), "method": r["method"],
            "n": len(vals), "c2st_mean": float(np.mean(vals)),
            "c2st_sem": float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
            "c2st_median": float(np.median(vals)),
            "time_train_s": _num(r.get("time_train")), "time_eval_s": _num(r.get("time_eval")),
            "values": vals,
        })
    if not rows:
        return None
    out = pd.DataFrame(rows)
    # A rerun of the same cell supersedes the earlier row.
    return out.drop_duplicates(["task", "budget", "method"], keep="last")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def paired_delta(df, task, budget, method, ref="score_transformer"):
    """COMPASS - Simformer, paired over the shared observations/conditionals."""
    a = df[(df.task == task) & (df.budget == budget) & (df.method == method)]
    b = df[(df.task == task) & (df.budget == budget) & (df.method == ref)]
    if a.empty or b.empty:
        return None
    va, vb = np.array(a.iloc[0]["values"]), np.array(b.iloc[0]["values"])
    if len(va) != len(vb):
        return float(va.mean() - vb.mean()), float("nan")
    d = va - vb
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")


def table(df, phase, what="C2ST (0.5 = indistinguishable from the reference; lower is better)"):
    out_dir = RESULTS / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["values"]).sort_values(["task", "budget", "method"]).to_csv(
        out_dir / f"{phase}.csv", index=False)
    methods = [m for m in METHOD_STYLE if m in set(df.method)] + \
        sorted(set(df.method) - set(METHOD_STYLE))
    lines = [f"# {phase}: {what}", "",
             "Mean ± s.e.m. over the evaluated observations/conditionals. "
             "Δ = method − Simformer, paired over the same observations; **bold** "
             "where |Δ| > 2 s.e.m.", ""]
    header = "| task | sims | " + " | ".join(METHOD_STYLE.get(m, (m,))[0] for m in methods)
    compass_methods = [m for m in methods if m != "score_transformer"]
    header += " | " + " | ".join(f"Δ {METHOD_STYLE.get(m, (m,))[0]}" for m in compass_methods) + " |"
    lines += [header, "|" + "---|" * (2 + len(methods) + len(compass_methods))]
    tasks = [t for t in TASK_ORDER if t in set(df.task)] + sorted(set(df.task) - set(TASK_ORDER))
    for task in tasks:
        for budget in sorted(set(df[df.task == task].budget)):
            cells = []
            for m in methods:
                r = df[(df.task == task) & (df.budget == budget) & (df.method == m)]
                cells.append("—" if r.empty else f"{r.iloc[0].c2st_mean:.3f} ± {r.iloc[0].c2st_sem:.3f}")
            deltas = []
            for m in compass_methods:
                d = paired_delta(df, task, budget, m)
                if d is None:
                    deltas.append("—")
                else:
                    txt = f"{d[0]:+.3f} ± {d[1]:.3f}"
                    deltas.append(f"**{txt}**" if math.isfinite(d[1]) and abs(d[0]) > 2 * d[1] else txt)
            lines.append(f"| {TASK_TITLE.get(task, task)} | {budget:,} | " + " | ".join(cells)
                         + " | " + " | ".join(deltas) + " |")
    (out_dir / f"{phase}.md").write_text("\n".join(lines) + "\n")
    return out_dir / f"{phase}.md"


def figure(df, phase, title, ylabel="C2ST", ylim=(0.48, 1.0)):
    tasks = [t for t in TASK_ORDER if t in set(df.task)] + sorted(set(df.task) - set(TASK_ORDER))
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.2 * len(tasks), 3.1), squeeze=False)
    handles = {}
    for ax, task in zip(axes[0], tasks):
        sub = df[(df.task == task) & ~df.method.isin(FIGURE_EXCLUDE)]
        for m in [m for m in METHOD_STYLE if m in set(sub.method)] + sorted(set(sub.method) - set(METHOD_STYLE)):
            label, colour, marker = METHOD_STYLE.get(m, (m, "#888888", "x"))
            s = sub[sub.method == m].sort_values("budget")
            h = ax.errorbar(s.budget, s.c2st_mean, yerr=s.c2st_sem, color=colour, marker=marker,
                            ms=6, lw=2, capsize=0, label=label)
            handles[label] = h
        ax.set_xscale("log")
        ax.set_xticks([1e3, 1e4, 1e5])
        ax.set_xlim(6e2, 1.6e5)
        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.axhline(0.5, color="#b5b4ae", lw=1, ls=":")
        ax.set_title(TASK_TITLE.get(task, task), fontsize=10)
        ax.set_xlabel("Number of simulations", fontsize=9)
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0][0].set_ylabel(ylabel, fontsize=9)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles),
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out_dir = RESULTS / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{phase}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_dir / f"{phase}.png"


TITLES = {
    "pilot": "Pilot — posterior C2ST, Two Moons",
    "fig4a": "Fig. 4a — posterior C2ST vs sbibm reference",
    "fig4b": "Fig. 4b — C2ST over arbitrary conditionals",
}


def improve_table(df, phase, base="compass_legacy_full_fixeps"):
    """Each improvement arm vs our current arm (Δ base) and vs Simformer (Δ Sim), paired."""
    arms = [m for m in IMPROVE if m in set(df.method)]
    if not arms:
        return None
    lines = [f"# {phase}: improvement arms, C2ST (lower is better)", "",
             "Δ base = arm − COMPASS (eps fixed); Δ Sim = arm − Simformer; both paired over the same "
             "observations. **bold** where |Δ| > 2 s.e.m.", "",
             "| task | sims | Simformer | COMPASS (eps fixed) | " + " | ".join(
                 f"{METHOD_STYLE[m][0]} (Δ base, Δ Sim)" for m in arms) + " |",
             "|---|---|---|---|" + "---|" * len(arms)]
    fmt = lambda d: "—" if d is None else (f"**{d[0]:+.3f}**" if math.isfinite(d[1]) and abs(d[0]) > 2 * d[1]
                                           else f"{d[0]:+.3f}")
    for t in [t for t in TASK_ORDER if t in set(df.task)]:
        for b in sorted(set(df[df.task == t].budget)):
            sub = df[(df.task == t) & (df.budget == b)]
            if not set(arms) & set(sub.method):
                continue
            val = lambda m: "—" if sub[sub.method == m].empty else f"{sub[sub.method == m].iloc[0].c2st_mean:.3f}"
            cells = ["—" if sub[sub.method == m].empty else
                     f"{val(m)} ({fmt(paired_delta(df, t, b, m, ref=base))}, {fmt(paired_delta(df, t, b, m))})"
                     for m in arms]
            lines.append(f"| {TASK_TITLE.get(t, t)} | {b:,} | {val('score_transformer')} | {val(base)} | "
                         + " | ".join(cells) + " |")
    out = RESULTS / "tables" / f"{phase}_improve.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def load_coverage(phase, metric):
    path = RESULTS / phase / "summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["metric"].astype(str) == metric]
    rows = []
    for _, r in df.iterrows():
        try:
            alphas, covs = ast.literal_eval(r["value"])
        except Exception:
            continue
        a, c = np.array(alphas), np.array(covs)
        rows.append({"task": r["task"], "budget": int(r["num_simulations"]), "method": r["method"],
                     "alphas": a, "covs": c,
                     # mean of (coverage - level): > 0 conservative, < 0 overconfident
                     "signed": float(np.mean(c - a)), "abs_err": float(np.mean(np.abs(c - a)))})
    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates(["task", "budget", "method"], keep="last")


def coverage_report(phase, metric, title):
    cov = load_coverage(phase, metric)
    if cov is None:
        return None
    tasks = [t for t in TASK_ORDER if t in set(cov.task)] + sorted(set(cov.task) - set(TASK_ORDER))
    budgets = sorted(set(cov.budget))
    fig, axes = plt.subplots(len(budgets), len(tasks), figsize=(2.6 * len(tasks), 2.5 * len(budgets)),
                             squeeze=False)
    handles = {}
    for i, b in enumerate(budgets):
        for j, t in enumerate(tasks):
            ax = axes[i][j]
            ax.plot([0, 1], [0, 1], color="#b5b4ae", lw=1, ls=":")
            for m in [m for m in METHOD_STYLE if m in set(cov.method)]:
                r = cov[(cov.task == t) & (cov.budget == b) & (cov.method == m)]
                if r.empty:
                    continue
                label, colour, _ = METHOD_STYLE[m]
                h, = ax.plot(r.iloc[0].alphas, r.iloc[0].covs, color=colour, lw=2, label=label)
                handles[label] = h
            if i == 0:
                ax.set_title(TASK_TITLE.get(t, t), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{b:,} sims\nempirical coverage", fontsize=8.5)
            if i == len(budgets) - 1:
                ax.set_xlabel("credibility level", fontsize=8.5)
            ax.set_aspect("equal")
            ax.grid(True, color="#e6e5e0", lw=0.6)
            ax.tick_params(labelsize=7.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = RESULTS / "figures" / f"{phase}_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    lines = [f"# {phase}: {metric} — mean(coverage − level); > 0 conservative, < 0 overconfident; "
             "|.| = mean absolute calibration error", "",
             "| task | sims | " + " | ".join(f"{METHOD_STYLE.get(m, (m,))[0]} signed / abs"
                                            for m in METHOD_STYLE if m in set(cov.method)) + " |",
             "|---|---|" + "---|" * len([m for m in METHOD_STYLE if m in set(cov.method)])]
    for t in tasks:
        for b in budgets:
            cells = []
            for m in [m for m in METHOD_STYLE if m in set(cov.method)]:
                r = cov[(cov.task == t) & (cov.budget == b) & (cov.method == m)]
                cells.append("—" if r.empty else f"{r.iloc[0].signed:+.3f} / {r.iloc[0].abs_err:.3f}")
            lines.append(f"| {TASK_TITLE.get(t, t)} | {b:,} | " + " | ".join(cells) + " |")
    md = RESULTS / "tables" / f"{phase}_{metric}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n")
    return md


# Evaluation-only phases: one table/figure per metric.
EXTRA = {
    "extra4a": [("nll_posterior", "NLL of the true θ under the posterior (lower is better)", "NLL (posterior)"),
                ("nll_likelihood", "NLL of the true x under the likelihood (lower is better)", "NLL (likelihood)"),
                ("cov_posterior", None, None), ("cov_likelihood", None, None)],
    "extra4b": [("c2st_likelihood", "C2ST of the likelihood p(x | θ) (lower is better)", "C2ST (likelihood)"),
                ("c2st_joint", "C2ST of the joint p(θ, x) (lower is better)", "C2ST (joint)")],
}


def extra_report(phase):
    for metric, what, ylabel in EXTRA[phase]:
        name = f"{phase}_{metric}"
        if metric.startswith("cov"):
            md = coverage_report(phase, metric, f"Expected coverage — {metric.split('_')[1]}")
        else:
            df = load(phase, metric)
            if df is None:
                continue
            md = table(df, name, what)
            figure(df, name, what, ylabel=ylabel, ylim=(0.48, 1.0) if metric.startswith("c2st") else None)
        if md is not None:
            print(f"[report] {name}: {md.relative_to(ROOT)}")
            print(md.read_text())


def main(phases=None):
    phases = phases or sorted(p.name for p in RESULTS.iterdir()
                              if (p / "summary.csv").exists() and p.name != "smoke")
    for phase in phases:
        if phase in EXTRA:
            extra_report(phase)
            continue
        df = load(phase)
        if df is None:
            continue
        t = table(df, phase)
        imp = improve_table(df, phase)
        if imp is not None:
            print(imp.read_text())
        f = figure(df, phase, TITLES.get(phase, phase))
        print(f"[report] {phase}: {t.relative_to(ROOT)} {f.relative_to(ROOT)}")
        print(t.read_text())


if __name__ == "__main__":
    main(sys.argv[1:] or None)
