#!/usr/bin/env python3
"""Run the paper benchmark jobs (jobs.py) on ONE GPU, several at a time when memory allows.

    python jobqueue.py --gpu 7                 # run every pending job (resumable)
    python jobqueue.py --gpu 7 --only pp_      # jobs whose name contains a substring
    python jobqueue.py --list                  # job table with status

The GPU is chosen once (it must carry no other user's process when the queue starts) and held:
a job starts only when (a) its dependencies finished, (b) fewer than --max-jobs run, and (c) the
card has the job's expected peak plus a margin free. Jobs run from the benchmark tree (the
detached worktree at the benchmarked commit, PAPER_TREE), with benchmarks/paper/hook on
PYTHONPATH (sitecustomize.py: the overrides, and autocvd pinned to this GPU). Each job writes
logs/<name>.log; state/<name>.ok or .fail records the exit. A job that fails with a CUDA OOM is
retried once with nothing else running. 3 CPU cores per job.
"""
import argparse
import json
import os
import pwd
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jobs as J  # noqa: E402

ME = pwd.getpwuid(os.getuid()).pw_name
STATE, LOGS = J.OUT / "state", J.OUT / "logs"
SLOTS = ["39-41", "42-44", "45-47", "66-68", "69-71"]
MARGIN_MB = 1200
SETTLE_S = 45


def say(message):
    line = f"{time.strftime('%F %T')} {message}"
    print(line, flush=True)
    with open(J.OUT / "queue.log", "a") as handle:
        handle.write(line + "\n")


def gpu_state(gpu):
    used, total = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    pids = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.split()
    users = set()
    for pid in pids:
        users.add(subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip() or "?")
    return int(total) - int(used), users - {ME, "?", ""}


def status(job):
    if (STATE / f"{job['name']}.ok").exists():
        return "ok"
    if (STATE / f"{job['name']}.fail").exists():
        return "fail"
    return "pending"


class Adopted:
    """A job a previous queue process started and that is still running (state/<name>.running):
    poll its pid; the exit code is unknown, so success is read from the log."""

    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except OSError:
            return 0


def launch(job, gpu, slot):
    env = dict(os.environ)
    env.update({
        "PAPER_GPU": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3",
        "NUMEXPR_NUM_THREADS": "3", "VECLIB_MAXIMUM_THREADS": "3",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "COMPASS_RUN_LABEL": job["name"],
    })
    for key, value in job.get("env", {}).items():
        env[key] = value if isinstance(value, str) else json.dumps(value)
    paths = [str(HERE / "hook"), str(J.TREE / "src")]
    if job.get("pythonpath"):
        paths += [str(J.TREE / p) for p in job["pythonpath"]]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(paths)
    log = open(LOGS / f"{job['name']}.log", "a")
    log.write(f"\n===== {time.strftime('%F %T')} {job['name']} on GPU {gpu}, cores {slot}\n"
              f"cwd {job['cwd']}\ncmd {' '.join(job['cmd'])}\nenv {json.dumps(job.get('env', {}))}\n")
    log.flush()
    cmd = ["taskset", "-c", slot] + [J.PY if c == "PY" else c for c in job["cmd"]]
    process = subprocess.Popen(cmd, cwd=J.TREE / job["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT)
    (STATE / f"{job['name']}.running").write_text(f"{process.pid} {slot}\n")
    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--max-jobs", type=int, default=3)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    for d in (STATE, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    table = J.jobs()
    names = [j["name"] for j in table]
    assert len(names) == len(set(names)), "duplicate job names"
    if args.only:
        table = [j for j in table if any(s in j["name"] for s in args.only)]
    if args.list:
        for j in table:
            print(f"{status(j):8s} {j['need']:6d} MB  {j['name']:46s} after {j.get('after', [])}")
        print(f"{len(table)} jobs")
        return
    if args.retry_failed:
        for j in table:
            (STATE / f"{j['name']}.fail").unlink(missing_ok=True)
    free, foreign = gpu_state(args.gpu)
    if foreign:
        raise SystemExit(f"GPU {args.gpu} carries processes of {sorted(foreign)}; pick a free one")
    say(f"queue on GPU {args.gpu}: {len(table)} jobs, {free} MB free")
    known = {j["name"] for j in J.jobs()}
    running = {}            # name -> (process, slot, job, started)
    retried = set()
    by_name = {j["name"]: j for j in J.jobs()}
    for marker in STATE.glob("*.running"):
        name = marker.stem
        pid, slot = marker.read_text().split()
        if name in by_name and Adopted(int(pid)).poll() is None:
            running[name] = (Adopted(int(pid)), slot, by_name[name], marker.stat().st_mtime)
            say(f"adopt {name} (pid {pid}, still running)")
    while True:
        for name, (process, slot, job, started) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            del running[name]
            (STATE / f"{name}.running").unlink(missing_ok=True)
            minutes = (time.time() - started) / 60
            text = (LOGS / f"{name}.log").read_text(errors="ignore")[-20000:].lower()
            if code == 0 and "traceback (most recent call last)" not in text[-4000:]:
                (STATE / f"{name}.ok").write_text(f"{minutes:.1f} min\n")
                say(f"done  {name} ({minutes:.1f} min)")
            elif "out of memory" in text and name not in retried:
                retried.add(name)
                job["need"] = 10**6         # rerun alone
                say(f"OOM   {name}: retry alone")
            else:
                (STATE / f"{name}.fail").write_text(f"exit {code} after {minutes:.1f} min\n")
                say(f"FAIL  {name} (exit {code}, {minutes:.1f} min)")
        pending = [j for j in table if status(j) == "pending" and j["name"] not in running]
        if not pending and not running:
            break
        ready = []
        for job in pending:
            after = [a for a in job.get("after", []) if a in known]
            if any((STATE / f"{a}.fail").exists() for a in after):
                (STATE / f"{job['name']}.fail").write_text("dependency failed\n")
                say(f"SKIP  {job['name']}: a dependency failed")
                continue
            if all((STATE / f"{a}.ok").exists() for a in after):
                ready.append(job)
        launched = False
        if ready and len(running) < args.max_jobs:
            free, foreign = gpu_state(args.gpu)
            alone = [j for j in ready if j["need"] >= 10**6]
            if alone:                      # an OOM retry drains the card first
                job = alone[0] if not running else None
            else:                          # the first ready job (in table order) that fits
                job = next((j for j in ready if free >= j["need"] + MARGIN_MB), None)
                if job is None and not running:
                    job = ready[0]         # larger than the stated margin allows: try it alone
            if job is not None:
                slot = next(s for s in SLOTS if s not in {r[1] for r in running.values()})
                process = launch(job, args.gpu, slot)
                running[job["name"]] = (process, slot, job, time.time())
                say(f"start {job['name']} (need {min(job['need'], 99999)} MB, {free} MB free, "
                    f"{len(running)} running){' foreign: ' + str(sorted(foreign)) if foreign else ''}")
                launched = True
        time.sleep(SETTLE_S if launched else 20)
    say("queue empty")


if __name__ == "__main__":
    main()
