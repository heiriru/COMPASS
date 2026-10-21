#!/usr/bin/env python3
"""Rerun every hierarchical experiment of run_noproj.py under the Lambda_S / clamp variants
(gauss_jacobian_FD arms only):

    default      all defaults (row projection, coordinate clamp) -- must reproduce the stored
                 draws bit for bit, and logs the raw Lambda_S and clamp diagnostics
    lams         no row projection; Lambda_S = Lambda_prior + sum_j I_j checked, and the old
                 composed-level spectral repair where it is not PD
    lamsrow      no row projection; the row projection only on samples whose Lambda_S is not PD
    gloeckler    Gloeckler et al. 2025 (markovsbi CorrectedScoreFn): V max(-A,0) V^T/(n-1)
                 + 0.1 I added to every factor, on every call
    consistent   row projection, with each row's score replaced by the score of the lifted
                 Gaussian (same clean mean)
    clampmetric  default PD rule; conservative (metric) clamp of the denoised predictions

Unlike run_noproj.py this never shares a GPU with another user: a job starts only on a card
whose compute processes are all ours, and waits otherwise (AGENTS.md: autocvd semantics).

    python run_pd.py --variants default lams ...    [--gpus 0,1,...]
"""
import os

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])

import argparse  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import run_noproj as rn  # noqa: E402

PD_VARIANTS = {
    "default": {},
    "lams": {"COMPASS_PD_STRATEGY": "composed_check", "COMPASS_PD_FALLBACK": "repair"},
    "lamsrow": {"COMPASS_PD_STRATEGY": "composed_check", "COMPASS_PD_FALLBACK": "row_projection"},
    "gloeckler": {"COMPASS_PD_STRATEGY": "gloeckler", "COMPASS_PD_NUGGET": "0.1"},
    "consistent": {"COMPASS_PD_STRATEGY": "row_projection_consistent"},
    "clampmetric": {"COMPASS_CLAMP_MODE": "metric"},
}
for name, env in PD_VARIANTS.items():
    rn.VARIANTS[name] = dict(env=env, jac_only=True, controls=False)

ME = os.environ.get("USER", "rheinric")
MAX_JOBS_PER_GPU = 4
MARGIN_MB = 1500
SETTLE_S = 75


def foreign_users(gpu):
    """Users other than us holding a compute process on ``gpu``."""
    uuid = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", gpu],
                          capture_output=True, text=True).stdout.strip()
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    users = set()
    for line in out.strip().splitlines():
        pid, gid = (v.strip() for v in line.split(","))
        if gid != uuid:
            continue
        user = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
        users.add(user or "?")
    return users - {ME}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(PD_VARIANTS), choices=sorted(PD_VARIANTS))
    parser.add_argument("--gpus", default=",".join(str(i) for i in range(10)))
    parser.add_argument("--only", nargs="*", default=None, help="job-name substrings to keep")
    parser.add_argument("--max-jobs-per-gpu", type=int, default=MAX_JOBS_PER_GPU)
    args = parser.parse_args()
    for d in (rn.LOGS, rn.DIAG):
        d.mkdir(parents=True, exist_ok=True)
    queue = [job for v in args.variants for job in rn.jobs(v)]
    if args.only:
        queue = [j for j in queue if any(s in j["name"] for s in args.only)]
    gpus = args.gpus.split(",")
    # three cores per job
    free_cores = [f"{c}-{c + 2}" for c in range(24, 72, 3)]
    running = {}
    last_launch = {g: 0.0 for g in gpus}
    rn.log(f"run_pd on GPUs {gpus}: {len(queue)} jobs ({', '.join(args.variants)})")
    while queue or running:
        for slot, (job, process, started, gpu) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            running.pop(slot)
            free_cores.append(slot)
            minutes = (time.time() - started) / 60
            if rn.failed_oom(job) and job.get("tries", 0) < 3:
                job["need"] = int(job["need"] * 1.6)
                job["tries"] = job.get("tries", 0) + 1
                queue.insert(0, job)
                rn.log(f"OOM   {job['name']} after {minutes:.0f} min; requeued needing {job['need']} MB")
            else:
                rn.log(f"done  {job['name']} exit {code} in {minutes:.0f} min on GPU {gpu}")
        for gpu in gpus:
            if not queue or not free_cores or time.time() - last_launch[gpu] < SETTLE_S:
                continue
            if sum(1 for r in running.values() if r[3] == gpu) >= args.max_jobs_per_gpu:
                continue
            if foreign_users(gpu):
                continue
            free = rn.gpu_free_mb(gpu)
            fits = [j for j in queue if free >= j["need"] + MARGIN_MB]
            if not fits:
                continue
            job = fits[0]
            queue.remove(job)
            slot = free_cores.pop(0)
            running[slot] = (job, rn.launch(job, slot, gpu), time.time(), gpu)
            last_launch[gpu] = time.time()
            rn.log(f"start {job['name']} on GPU {gpu} cores {slot} (free {free} MB); "
                   f"{len(queue)} queued, {len(running)} running")
        time.sleep(15)
    rn.log("run_pd all done")


if __name__ == "__main__":
    main()
