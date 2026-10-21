#!/usr/bin/env python
"""Figures for the oracle bridge experiment.

Two panels, which together are the argument of section 13:

``bridges``  the intermediate bridge of every rule at matched noise levels,
             on the profiled ``g`` slice, against the exact diffused marginal.
``summary``  W1/sd of every arm against ``n``, with the Monte-Carlo floor.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "gm_oracle"
ARMS = ("true", "naive", "fnpe-diffused", "arruda-d1=0.2", "gauss", "jac")
COLOURS = {"true": "k", "naive": "tab:red", "fnpe-diffused": "tab:blue",
           "arruda-d1=0.2": "tab:green", "gauss": "tab:orange",
           "jac": "tab:purple"}


def bridge_figure(tag=""):
    data = json.loads((ARTIFACTS / f"bridges{tag}.json").read_text())
    ns = sorted(data, key=int)
    taus = ["tau=1.0", "tau=0.5", "tau=0.25", "tau=0.1", "tau=0.0"]
    figure, axes = plt.subplots(len(ns), len(taus),
                                figsize=(3.1 * len(taus), 2.6 * len(ns)),
                                squeeze=False)
    for row, n in enumerate(ns):
        reference = data[n]
        for column, key in enumerate(taus):
            entry = reference[key]
            axis = axes[row][column]
            grid = np.asarray(entry["grid"])
            exact = np.exp(np.asarray(entry["exact_curve"]))
            axis.plot(grid, exact / exact.max(), "k--", lw=1.4,
                      label="exact marginal")
            for arm in ARMS:
                curve = np.exp(np.asarray(entry[arm]["curve"]))
                axis.plot(grid, curve / curve.max(), color=COLOURS[arm], lw=1.0,
                          alpha=0.85, label=arm)
            axis.set_xlim(-4.0, 4.0)
            axis.set_ylim(0.0, 1.08)
            if row == 0:
                axis.set_title(f"tau={entry['tau']:g}, "
                               f"lambda={entry['lambda']:.3f}", fontsize=9)
            if column == 0:
                axis.set_ylabel(f"n = {n}\nnormalised density")
            axis.tick_params(labelsize=8)
    axes[0][0].legend(fontsize=6, loc="upper left")
    figure.suptitle("Intermediate bridges on the profiled g slice"
                    f"{tag or ' (mu_0 = 2.5)'}", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path = ARTIFACTS / f"bridges{tag}.png"
    figure.savefig(path, dpi=150)
    print(f"wrote {path}")


def _single_n_figure(reference, tables, picks, n, tag):
    """One `n` only: a bar chart, because a line through one point is not one.

    Values above the axis ceiling are drawn at the ceiling with their true
    number printed, and diverged arms are hatched, so nothing is silently
    clipped into looking finite.
    """
    ceiling = 8.0
    labels, values, flags = [], [], []
    for label, source, key in picks:
        table = tables.get(source, {}).get(n, {})
        if key is None:
            finite = [v for v in table.values()
                      if not v.get("diverged") and np.isfinite(v["w1_over_sd"])]
            row = min(finite, key=lambda v: v["w1_over_sd"]) if finite else None
        elif isinstance(key, tuple):
            row = next((table[k] for k in key if k in table), None)
        else:
            row = table.get(key)
        labels.append(label)
        if row is None:
            values.append(np.nan); flags.append("missing")
        elif row.get("diverged") or not np.isfinite(row["w1_over_sd"]):
            values.append(ceiling); flags.append("diverged")
        else:
            values.append(row["w1_over_sd"]); flags.append("ok")
    floor = reference[n]["w1_floor_mean"]
    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    y = np.arange(len(labels))[::-1]
    drawn = [min(v, ceiling) if np.isfinite(v) else 0.0 for v in values]
    colours = ["0.75" if f == "diverged" else "tab:blue" for f in flags]
    bars = axis.barh(y, drawn, color=colours, height=0.65,
                     hatch=["///" if f == "diverged" else "" for f in flags])
    for bar, value, flag in zip(bars, values, flags):
        text = ("diverged" if flag == "diverged"
                else "not run" if flag == "missing" else f"{value:.3f}")
        axis.text(bar.get_width() * 1.06, bar.get_y() + bar.get_height() / 2,
                  text, va="center", fontsize=8)
    axis.axvline(floor, color="k", ls=":", lw=2,
                 label=f"Monte-Carlo floor {floor:.4f}")
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xscale("log")
    axis.set_xlim(0.008, ceiling * 3.0)
    axis.set_xlabel("W1 / posterior sd")
    setting = tag.replace("_mu", ", mu_0 = ") if tag else ", mu_0 = 2.5"
    axis.set_title(f"Oracle arms at n = {n}{setting}, corrected sampler",
                   fontsize=11)
    axis.grid(alpha=0.3, axis="x", which="both")
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    path = ARTIFACTS / f"summary{tag}.png"
    figure.savefig(path, dpi=150)
    print(f"wrote {path}")


def summary_figure(tag=""):
    """Every arm at its converged configuration, under the corrected sampler.

    Arms that diverged are drawn at the top of the axis with an open marker
    rather than dropped, so a rule that blows up is visible rather than absent.
    """
    reference = json.loads((ARTIFACTS / f"reference{tag}.json").read_text())
    tables = {}
    for name in ("control", "control_churn", "geffner_ladder", "geffner_step",
                 "arruda_churn", "linhart_churn"):
        path = ARTIFACTS / f"{name}{tag}.json"
        if path.exists():
            tables[name] = json.loads(path.read_text())
    ns = sorted(reference, key=int)
    picks = [
        ("TRUE-SCORE-DPM2 (churn=0)", "control", "dpm2/steps=500"),
        ("TRUE-SCORE-DPM2 (churn=0.5)", "control_churn",
         "dpm2/churn=0.5,steps=500"),
        # The *highest-effort stable* configuration, not the best-scoring one.
        # This arm's ladder is non-monotone: it passes through a flattering
        # minimum at intermediate effort and then settles higher, so "best"
        # would report an unconverged chain (see the report's section 3-4).
        ("ORACLE-FNPSE (Langevin, most effort)", "geffner_step",
         "step=1/n/T=100,L=500,snr=0.2"),
        ("ORACLE-NAIVE-DPM2", "linhart_churn", "naive/dpm2,churn=0.5,steps=500"),
        ("GEFFNER-DPM2", "linhart_churn", "fnpe/dpm2,churn=0.5,steps=500"),
        ("ORACLE-GAUSS", "linhart_churn", "gauss/dpm2,churn=0.5,steps=500"),
        ("ORACLE-JAC-DPM2", "linhart_churn", "jac/dpm2,churn=0.5,steps=500"),
        ("ORACLE-ARRUDA (best d1)", "arruda_churn", None),
    ]
    single = len(ns) == 1
    figure, axis = plt.subplots(figsize=(7.6, 4.4) if single else (7.2, 4.8))
    x = np.arange(len(ns))
    ceiling = 3.0
    if single:
        return _single_n_figure(reference, tables, picks, ns[0], tag)
    for label, source, key in picks:
        values, diverged = [], []
        for n in ns:
            table = tables.get(source, {}).get(n, {})
            if key is None:
                finite = [v for v in table.values()
                          if not v.get("diverged") and np.isfinite(v["w1_over_sd"])]
                row = min(finite, key=lambda v: v["w1_over_sd"]) if finite else None
            elif isinstance(key, tuple):
                row = next((table[k] for k in key if k in table), None)
            else:
                row = table.get(key)
            if row is None:
                values.append(np.nan); diverged.append(False)
            elif row.get("diverged") or not np.isfinite(row["w1_over_sd"]):
                values.append(ceiling); diverged.append(True)
            else:
                values.append(min(row["w1_over_sd"], ceiling))
                diverged.append(False)
        line, = axis.plot(x, values, "o-", label=label, lw=1.4, ms=5)
        for position, value, flag in zip(x, values, diverged):
            if flag:
                axis.plot([position], [value], "^", ms=11, mfc="none",
                          mec=line.get_color(), mew=1.8)
    floor = [reference[n]["w1_floor_mean"] for n in ns]
    axis.plot(x, floor, "k:", lw=2, label="Monte-Carlo floor (2000 draws)")
    axis.set_xticks(x)
    axis.set_xticklabels([f"n = {n}" for n in ns])
    axis.set_yscale("log")
    axis.set_ylim(0.005, 4.0)
    axis.set_ylabel("W1 / posterior sd")
    axis.set_title(f"Oracle arms, corrected sampler{tag or ' (mu_0 = 2.5)'}\n"
                   "open triangles at the ceiling = diverged", fontsize=10)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5),
                frameon=False)
    figure.tight_layout()
    path = ARTIFACTS / f"summary{tag}.png"
    figure.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    bridge_figure(tag)
    summary_figure(tag)
