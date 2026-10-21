"""Large vs small COMPASS vs Simformer across 1k/10k/100k (C2ST, all 8 tasks)."""
import os
CPU_THREAD_LIMIT = 3
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = str(CPU_THREAD_LIMIT)
if hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT])
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report import load, TASK_TITLE, TASK_ORDER, RESULTS

ARMS = [("score_transformer", "Simformer (180k)", "#2a78d6", "o"),
        ("compass_legacy_full_fixeps", "COMPASS large (2.8–12.9M)", "#eb6834", "s"),
        ("compass_fx_notime_add", "COMPASS small, no time mod. (0.45–0.48M)", "#4a3aa7", "X")]
BUDGETS = [1000, 10000, 100000]

df = pd.concat([load("fig4a"), load("fig4b")])
df = df[df.budget.isin(BUDGETS) & df.method.isin([a[0] for a in ARMS])]
tasks = [t for t in TASK_ORDER if t in set(df.task)]

fig, axes = plt.subplots(3, 3, figsize=(12, 10.5))
for ax, t in zip(axes.flat, tasks):
    for m, lab, col, mk in ARMS:
        s = df[(df.task == t) & (df.method == m)].sort_values("budget")
        ax.errorbar(s.budget, s.c2st_mean, yerr=s.c2st_sem, color=col, marker=mk, ms=7, lw=2, capsize=3, label=lab)
    ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels(["1k", "10k", "100k"])
    ax.axhline(0.5, color="#999", lw=1, ls=":")
    ax.set_title(TASK_TITLE.get(t, t)); ax.set_ylabel("C2ST (lower is better)")
    ax.grid(alpha=0.25)

# Mean gap to Simformer, paired per task.
ax = axes.flat[len(tasks)]
ref = df[df.method == "score_transformer"].set_index(["task", "budget"]).c2st_mean
for (m, lab, col, mk), dy in zip(ARMS[1:], (8, -14)):
    s = df[df.method == m].set_index(["task", "budget"]).c2st_mean
    gap = (s - ref.reindex(s.index)).groupby(level="budget").agg(["mean", "sem"])
    ax.errorbar(gap.index, gap["mean"], yerr=gap["sem"], color=col, marker=mk, ms=7, lw=2, capsize=3, label=lab)
    for b, v in gap["mean"].items():
        ax.annotate(f"{v:+.3f}", (b, v), textcoords="offset points", xytext=(8, dy), fontsize=8, color=col)
ax.axhline(0, color="#2a78d6", lw=1.5, ls="--", label="Simformer")
ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels(["1k", "10k", "100k"])
ax.set_title("Mean gap to Simformer (8 tasks)"); ax.set_ylabel("Δ C2ST vs Simformer"); ax.grid(alpha=0.25)

for a in axes.flat[len(tasks) + 1:]:
    a.axis("off")
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=10)
fig.supxlabel("number of simulations")
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = RESULTS / "figures" / "scaling_small_vs_large.png"
fig.savefig(out, dpi=150); print(out)
