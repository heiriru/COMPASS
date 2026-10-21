#!/usr/bin/env python3
"""Sampler test on GPU 9: Langevin correctors instead of churn.

1. Every hierarchical / hierarchical-bimodal experiment with DPM2 + gauss_jacobian_FD and two
   Langevin corrector steps per level (churn_eta 0; run_noproj.py variant "langevin", via
   COMPASS_SAMPLER_OVERRIDE). The stored runs are the churn arm.
2. The Simformer benchmarks (Fig. 4a + 4b, 8 tasks x 1k/10k/100k), resampling the existing
   compass_legacy_full(_fixeps) checkpoints twice: compass_fx_churn2 (churn eta 2) and
   compass_fx_langevin (the same corrector recipe). No training. Results go to their own phases
   (results/fig4a_sampler, fig4b_sampler), so they never race the running SimFormer queue on
   results/fig4a/summary.csv; Simformer jobs run one at a time for the same reason.

Memory: the combined GPU memory of this scheduler's jobs never exceeds BUDGET_MB. A running job
counts as max(its current use, its expected peak), so a job still growing keeps its full share;
a new job starts only when its expected peak fits, and only if the GPU also has that much free
plus MARGIN_MB (other users share GPU 9).

    python run_langevin.py            # log: artifacts/langevin/scheduler.log
"""
import os

os.sched_setaffinity(0, [68, 69])

import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import run_noproj as rn  # noqa: E402

os.sched_setaffinity(0, [68, 69])   # run_noproj pins itself to 70-71 on import

GPU = "9"
BUDGET_MB = 7500
MARGIN_MB = 1000
SETTLE_S = 75
OUT = rn.EXP / "artifacts" / "langevin"
LOGS, DIAG = OUT / "logs", OUT / "diagnostics"
CORES = ["3-5", "6-8", "45-47", "48-50", "51-53"]
SF = rn.COMPASS / "benchmarks" / "SimFormer"
SF_PY = str(SF / ".simformer" / "bin" / "python")
SF_ARMS = ("compass_fx_churn2", "compass_fx_langevin")
SF_PHASES = {"fig4a": "c2st_posterior", "fig4b": "c2st"}

# measured peaks of the same jobs in the noproj/nocurv/noboth reruns (+10%)
NEED = {"pp": 3300, "toy": 2200, "ik": 1500, "gm": 1500, "sf": 1500}


def hierarchical_jobs():
    out = []
    for job in rn.jobs("langevin"):
        kind = job["name"].split("_")[1]
        job.update(need=NEED.get(kind, 2200), argv=[rn.PY, "-u", *job["cmd"]], simformer=False)
        out.append(job)
    return out


def simformer_jobs():
    sys.path.insert(0, str(SF))
    import run_benchmark as rb
    out = []
    for budget in (1_000, 10_000, 100_000):
        for phase, eval_name in SF_PHASES.items():
            for task in rb.PHASES[phase][0]:
                ck = next((c for c in (rb.find_checkpoint(phase, task, budget, m)
                                       for m in ("compass_legacy_full_fixeps", "compass_legacy_full")) if c), None)
                if ck is None:
                    print(f"no checkpoint for {phase} {task} {budget}", flush=True)
                    continue
                for arm in SF_ARMS:
                    name = f"sf_{phase}_{task}_{budget}_{arm}"
                    out.append(dict(
                        name=name, cwd=SF, need=NEED["sf"], simformer=True,
                        sf_key=(f"{phase}_sampler", task, budget, arm),
                        env={"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
                             "TF_CPP_MIN_LOG_LEVEL": "2", "NUMEXPR_NUM_THREADS": "3"},
                        argv=[SF_PY, str(SF / "bridge" / "run_scoresbi.py"), f"name={phase}_sampler",
                              f"method={arm}", f"task={task}", f"task.num_simulations={budget}",
                              f"eval={eval_name}", "save_model=false", f"++method.compass.load_from={ck}"]))
    return out


def sf_done(key):
    phase, task, budget, method = key
    path = SF / "results" / phase / "summary.csv"
    if not path.exists():
        return False
    import pandas as pd
    df = pd.read_csv(path)
    rows = df[(df.task == task) & (df.num_simulations == budget) & (df.method == method)]
    return not rows[rows.metric.astype(str).str.startswith("c2st")].empty


def log(message):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with open(OUT / "scheduler.log", "a") as handle:
        handle.write(line + "\n")


def job_pids():
    """GPU memory per scheduler job, summed over its processes (a Simformer job is two)."""
    usage, per_job = rn.per_pid_mb(), {}
    for pid, mb in usage.items():
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except OSError:
            continue
        for item in env:
            if item.startswith(b"COMPASS_RUN_LABEL=lgv:"):
                name = item.split(b":", 1)[1].decode()
                per_job[name] = per_job.get(name, 0) + mb
    return per_job


def launch(job, cores):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3",
           "OPENBLAS_NUM_THREADS": "3", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **job["env"],
           "COMPASS_PROJECTION_LOG": str(DIAG / f"{job['name']}.jsonl"), "COMPASS_RUN_LABEL": f"lgv:{job['name']}"}
    path = LOGS / f"{job['name']}.log"
    offset = path.stat().st_size if path.exists() else 0
    handle = open(path, "a")
    process = subprocess.Popen(["taskset", "-c", cores, *job["argv"]], cwd=job["cwd"], env=env,
                               stdout=handle, stderr=subprocess.STDOUT)
    return process, offset


def failed_oom(job, offset):
    with open(LOGS / f"{job['name']}.log", errors="ignore") as handle:
        handle.seek(offset)
        return "out of memory" in handle.read().lower()


def main():
    for d in (LOGS, DIAG):
        d.mkdir(parents=True, exist_ok=True)
    # First fit, in this order: one partial-pooling run (the longest), the short Simformer jobs
    # (one at a time), the second partial-pooling run, then the rest.
    hier = hierarchical_jobs()
    pp = [j for j in hier if j["name"].startswith("langevin_pp_")]
    rest = [j for j in hier if j not in pp]
    queue = pp[:1] + [j for j in simformer_jobs() if not sf_done(j["sf_key"])] + pp[1:] + rest
    running, attempts, last_launch = {}, {}, 0.0   # cores -> (job, process, started, peak, offset)
    log(f"GPU {GPU}, budget {BUDGET_MB} MB: {len(queue)} jobs")
    while queue or running:
        usage = job_pids()
        for cores, (job, process, started, peak, offset) in list(running.items()):
            peak = max(peak, usage.get(job["name"], 0))
            running[cores] = (job, process, started, peak, offset)
            code = process.poll()
            if code is None:
                continue
            running.pop(cores)
            minutes = (time.time() - started) / 60
            if failed_oom(job, offset) and attempts[job["name"]] < 3:
                job["need"] = int(job["need"] * 1.5)
                queue.insert(0, job)
                log(f"OOM   {job['name']} after {minutes:.0f} min (peak {peak} MB); requeued needing {job['need']} MB")
            else:
                log(f"done  {job['name']} exit {code} in {minutes:.0f} min, peak {peak} MB")
        if queue and time.time() - last_launch >= SETTLE_S:
            free_cores = [c for c in CORES if c not in running]
            held = sum(max(job["need"], usage.get(job["name"], 0)) for job, *_ in running.values())
            sf_busy = any(job["simformer"] for job, *_ in running.values())
            free = rn.gpu_free_mb(GPU)
            fits = [j for j in queue if held + j["need"] <= BUDGET_MB and free >= j["need"] + MARGIN_MB
                    and not (j["simformer"] and sf_busy)]
            if free_cores and fits:
                job = fits[0]
                queue.remove(job)
                attempts[job["name"]] = attempts.get(job["name"], 0) + 1
                process, offset = launch(job, free_cores[0])
                running[free_cores[0]] = (job, process, time.time(), 0, offset)
                last_launch = time.time()
                log(f"start {job['name']} cores {free_cores[0]} (held {held} + {job['need']} MB of {BUDGET_MB}, "
                    f"GPU free {free} MB); {len(queue)} queued, {len(running)} running")
        time.sleep(10)
    log("all done")


if __name__ == "__main__":
    main()
