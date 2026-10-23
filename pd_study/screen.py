"""CPU screen of the Lambda_S strategies on the exact-score toys (abs, lean; oracle scores).

    python screen.py run      # all variants x toys x n, 6 jobs at a time (3 cores each)
    python screen.py score    # W1 / exact sd and P(g>0) vs the Gibbs reference -> results/screen/

Runs go through abs_hierarchy/run.py with --device cpu and suffix "_scr_<variant>", so they never
collide with the GPU runs. The "default" variant also writes the COMPASS_PD_PROBE jsonl (the probe
does not change the draws; tests/test_pd_strategies.py checks that).
"""
import os

CPU_THREAD_LIMIT = 3
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, str(CPU_THREAD_LIMIT))
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT])

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
COMPASS = HERE.parent
TOYS = COMPASS / "abs_hierarchy"
OUT = HERE / "results" / "screen"
PY = str(COMPASS / ".COMPASS" / "bin" / "python")

VARIANTS = {
    "default": {},
    "consistent": {"COMPASS_PD_STRATEGY": "row_projection_consistent"},
    "trunc": {"COMPASS_PD_STRATEGY": "row_truncation"},
    "blend": {"COMPASS_PD_STRATEGY": "row_projection_blend"},
    "gl0": {"COMPASS_PD_STRATEGY": "gloeckler", "COMPASS_PD_NUGGET": "0"},
    "gl001": {"COMPASS_PD_STRATEGY": "gloeckler", "COMPASS_PD_NUGGET": "0.01"},
    "gl01": {"COMPASS_PD_STRATEGY": "gloeckler", "COMPASS_PD_NUGGET": "0.1"},
    "gl1": {"COMPASS_PD_STRATEGY": "gloeckler", "COMPASS_PD_NUGGET": "1"},
    "glcons": {"COMPASS_PD_STRATEGY": "gloeckler_consistent", "COMPASS_PD_NUGGET": "0.1"},
}
TOY_RUNS = {"abs": "artifacts/runs", "lean": "artifacts/lean/runs"}
MODES = ("oracle_hierarchical", "oracle_joint")
NS = (4, 16, 64, 128)
CORE_SLOTS = [f"{c}-{c + 2}" for c in range(30, 48, 3)]
METHOD = "gauss_jacobian_FD"


def jobs(variants):
    for variant in variants:
        for model in TOY_RUNS:
            for mode in MODES:
                for n in NS:
                    yield variant, model, mode, n


def launch(variant, model, mode, n, cores):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", **VARIANTS[variant],
           "PYTHONPATH": ".:../Gaussian_mixtures_compositional_score_modelling:../experiments_arruda_schur"}
    if variant == "default":
        env["COMPASS_PD_PROBE"] = str(OUT / f"probe_{model}_{mode}_n{n}.jsonl")
    log = open(OUT / "logs" / f"{variant}_{model}_{mode}_n{n}.log", "w")
    cmd = ["taskset", "-c", cores, PY, "-u", "run.py", "--model", model, "--mode", mode,
           "--methods", METHOD, "--suffix", f"_scr_{variant}", "--observations", str(n),
           "--device", "cpu"]
    return subprocess.Popen(cmd, cwd=TOYS, env=env, stdout=log, stderr=subprocess.STDOUT)


def run(variants):
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    queue = list(jobs(variants))
    # largest n first so the long jobs do not trail at the end
    queue.sort(key=lambda j: -j[3])
    free, running = list(CORE_SLOTS), {}
    while queue or running:
        for cores, (job, process) in list(running.items()):
            if process.poll() is not None:
                running.pop(cores)
                free.append(cores)
                print(time.strftime("%H:%M:%S"), "done", *job, "exit", process.returncode, flush=True)
        while queue and free:
            job = queue.pop(0)
            cores = free.pop(0)
            running[cores] = (job, launch(*job, cores))
        time.sleep(5)


def score(variants):
    import numpy as np
    from scipy.stats import wasserstein_distance
    rows = []
    for model, folder in TOY_RUNS.items():
        runs = TOYS / folder
        for mode in MODES:
            for n in NS:
                exact = np.load(runs / f"mcmc_n{n}.npz")["g"]
                sd = float(exact.std())
                row = {"toy": model, "mode": mode, "n": n, "p_exact": float((exact > 0).mean())}
                for variant in variants:
                    path = runs / f"{mode}_scr_{variant}_{METHOD}_n{n}.npz"
                    if not path.exists():
                        continue
                    g = np.load(path)["g"]
                    row[variant] = {"w1": float(wasserstein_distance(g, exact) / sd),
                                    "p": float((g > 0).mean())}
                rows.append(row)
    (OUT / "screen.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = [f"{'toy':5s} {'mode':20s} {'n':>4s} {'p_ex':>5s}  " +
             "  ".join(f"{v:>13s}" for v in variants), " " * 38 + "  ".join(f"{'W1/sd   P>0':>13s}" for _ in variants)]
    for r in rows:
        cells = "  ".join(f"{r[v]['w1']:6.3f} {r[v]['p']:5.3f}" if v in r else f"{'-':>13s}" for v in variants)
        lines.append(f"{r['toy']:5s} {r['mode']:20s} {r['n']:4d} {r['p_exact']:5.3f}  {cells}")
    lines.append("")
    lines.append("geometric-mean W1 ratio to default (below 1 = better):")
    for model in TOY_RUNS:
        cells = []
        for v in variants:
            ratios = [np.log(r[v]["w1"] / r["default"]["w1"]) for r in rows
                      if r["toy"] == model and v in r and "default" in r]
            cells.append(f"{np.exp(np.mean(ratios)):13.2f}" if ratios else f"{'-':>13s}")
        lines.append(f"{model:31s}" + "  ".join(cells))
    text = "\n".join(lines)
    print(text)
    (OUT / "screen.txt").write_text(text + "\n")


if __name__ == "__main__":
    action = sys.argv[1]
    chosen = sys.argv[2:] or list(VARIANTS)
    {"run": run, "score": score}[action](chosen)
