#!/usr/bin/env python3
"""Rerun every hierarchical / hierarchical-bimodal experiment with a safeguard switched off --
the information projection (COMPASS_INFORMATION_PROJECTION=0, variant noproj), the lower clip at 0
of the gauss_jacobian curvature (COMPASS_CURVATURE_CLIP=0, nocurv), or both (noboth) -- on one or
two GPUs, in parallel, guarded against running out of GPU memory.

Only arms whose composition the projection touches are rerun (gauss_jacobian_FD,
gauss_hierarchical, arruda_gauss_direct); the stored runs reproduce bit for bit with it on,
so they are the "with projection" arm. arruda_direct (no projection in its path) rides along
in a few jobs as a control: its draws must equal the stored ones exactly.

Each job writes its draws next to the stored ones under a *_noproj name and one diagnostics
line per sample() call to artifacts/noproj/diagnostics/<job>.jsonl.

    python run_noproj.py                               # GPUS=8,9 (comma list) to spread over GPUs
    python run_noproj.py --skip a b --wait-pids 1 2    # take over from a running scheduler
"""
import os

os.sched_setaffinity(0, [70, 71])

import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

GPUS = os.environ.get("GPUS", os.environ.get("GPU", "8")).split(",")
COMPASS = Path("/export/home/rheinric/COMPASS")
EXP = COMPASS / "experiments_arruda_schur"
TOYS = COMPASS / "abs_hierarchy"
PY = str(COMPASS / ".COMPASS/bin/python")
OUT = EXP / "artifacts" / "noproj"
LOGS = OUT / "logs"
DIAG = OUT / "diagnostics"
# Cores per GPU; cores held by jobs a previous scheduler started are left out.
SLOTS_BY_GPU = {"8": ["24-26", "27-29", "30-32", "33-35"],
                "9": ["3-5", "6-8", "45-47", "48-50"]}
# Free memory to keep on top of a job's expected peak, per GPU (GPU 8: more head room, by request;
# GPU 9: the toys grow ~0.7 GB at n=128 after launch, and the user's SimFormer job shares it).
MARGIN_BY_GPU = {"8": 3000, "9": 2500}
MAX_JOBS_PER_GPU = 4      # counts every scheduler job on the GPU, including ones a previous scheduler started
MARGIN_MB = 1200          # head room kept free on top of a job's expected peak
SETTLE_S = 75             # let a launched job reach its working set before the next check

JAC, GH, AD = "gauss_jacobian_FD", "gauss_hierarchical", "arruda_direct"


GM_ARGS = ["--", "--observations", "8", "16", "32", "64", "--regimes", "asymmetric", "symmetric",
           "--replicates", "1", "--draws", "2000"]

# The two-corrector recipe of compass/churn.py's comparison (exponential_score_comparison/compare.py).
LANGEVIN = {"churn_eta": 0.0, "corrector_steps": 2, "corrector_steps_interval": 1,
            "final_corrector_steps": 3, "snr": 0.2}

# Variants: which safeguard(s) to switch off. The curvature clip only exists in gauss_jacobian,
# so the curvature variants rerun the gauss_jacobian_FD arms only.
VARIANTS = {
    "noproj": dict(env={"COMPASS_INFORMATION_PROJECTION": "0"}, jac_only=False),
    "nocurv": dict(env={"COMPASS_CURVATURE_CLIP": "0"}, jac_only=True),
    "noboth": dict(env={"COMPASS_INFORMATION_PROJECTION": "0", "COMPASS_CURVATURE_CLIP": "0"}, jac_only=True,
                   controls=False),
    # Not a safeguard: DPM2 + gauss_jacobian_FD with two Langevin corrector steps per level (the
    # predictor-corrector SDE sampler) in place of churn; both safeguards on. See run_langevin.py.
    "langevin": dict(env={"COMPASS_SAMPLER_OVERRIDE": json.dumps(LANGEVIN)}, jac_only=True, controls=False),
}


def jobs(variant):
    settings = VARIANTS[variant]
    suffix, tag, env = f"_{variant}", variant, settings["env"]
    keep = (lambda ms: [m for m in ms if m in (JAC, AD)]) if settings["jac_only"] else (lambda ms: list(ms))
    controls = settings.get("controls", True)   # arruda_direct must match the stored draws exactly

    def toy(model, mode, methods, extra=(), toy_suffix=None):
        methods = [m for m in keep(methods) if controls or m != AD]
        sfx = toy_suffix or suffix
        return dict(name=f"{variant}_toy_{model}_{mode}{sfx}", cwd=TOYS, need=1500, methods=methods,
                    env={"PYTHONPATH": ".:../Gaussian_mixtures_compositional_score_modelling:../experiments_arruda_schur",
                         **env},
                    cmd=["run.py", "--model", model, "--mode", mode, "--methods", *methods, "--suffix", sfx, *extra])

    def exp(name, args, need, extra_env=None, methods=None):
        if methods is not None:
            methods = [m for m in keep(methods) if controls or m != AD]
            args = args + ["--methods", *methods]
        return dict(name=f"{variant}_{name}", cwd=EXP, need=need, methods=methods or [JAC],
                    env={"PYTHONPATH": ".", **(extra_env or {}), **env}, cmd=args)

    ik = lambda script, *a: [script, "run", *a, "--suffix", suffix]  # noqa: E731
    gm = lambda mode: ["gm.py", "run", "--mode", mode, "--tag", f"posteriors_{tag}", *GM_ARGS]  # noqa: E731
    out = [
        # partial pooling C2, 30 datasets (the longest; start first)
        exp("pp_hierarchical", ["pp.py", "infer", "--mode", "hierarchical", "--method", JAC,
                                "--datasets", "30", "--tag", tag], 4500),
        exp("pp_hierarchical_Rg", ["pp.py", "infer", "--mode", "hierarchical", "--precondition", "Rg",
                                   "--method", JAC, "--datasets", "30", "--tag", tag], 4500),
        # IK
        exp("ik_absx1", ik("ik_absx1.py"), 1600, methods=[JAC]),
        exp("ik_absx1_R", ik("ik_absx1.py", "--residual"), 1600, methods=[JAC]),
        exp("ik_flat_fixed", ik("ik_flat.py", "--variant", "fixed"), 1600, methods=[JAC]),
        exp("ik_flat_global", ik("ik_flat.py", "--variant", "global"), 1600, {"IK_SD_H": "0.05"}, methods=[JAC]),
        exp("ik_reach_sym", ik("ik_reach.py", "--variant", "sym"), 1600, methods=[JAC, AD]),
        exp("ik_reach_shift", ik("ik_reach.py", "--variant", "shift"), 1600, methods=[JAC]),
        exp("ik_reach_sym_bernoulli", ik("ik_reach.py", "--variant", "sym", "--menu", "bernoulli"), 1600,
            methods=[JAC]),
        exp("ik_reach_shift_bernoulli", ik("ik_reach.py", "--variant", "shift", "--menu", "bernoulli"), 1600,
            methods=[JAC]),
        # Gaussian mixture (symmetric + asymmetric regimes); run_scaling takes --methods last
        exp("gm_hierarchical", gm("hierarchical"), 2000, methods=[JAC, "arruda_gauss_direct", AD]),
        exp("gm_none", gm("none"), 2000, methods=[JAC, "arruda_gauss_direct"]),
        exp("gm_oracle", gm("oracle"), 2000, methods=[JAC, AD]),
        # folded-hierarchy toys
        toy("abs", "hierarchical", [JAC, AD]),
        toy("abs", "none", [GH, JAC]),
        toy("abs", "oracle_joint", [GH, JAC]),
        toy("abs", "oracle_hierarchical", [JAC]),
        toy("square", "hierarchical", [JAC]),
        toy("square", "none", [GH, JAC]),
        toy("lean", "hierarchical", [JAC]),
        toy("lean", "none", [GH, JAC]),
        toy("lean", "oracle_joint", [GH, JAC]),
        toy("lean", "oracle_hierarchical", [JAC]),
        toy("lean", "hierarchical_R", [JAC]),
        toy("lean", "hierarchical_T", [JAC]),
        toy("lean", "hierarchical_TF", [JAC]),
        toy("lean", "hierarchical_TFR", [JAC]),
        toy("lean", "hierarchical_T", [JAC], extra=["--sample-eps", "1e-4"], toy_suffix=f"_e4{suffix}"),
        toy("curved", "hierarchical", [JAC]),
        toy("curved", "hierarchical_R", [JAC]),
    ]
    return [j for j in out if j["methods"]]


def gpu_free_mb(gpu):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits",
                          "-i", gpu], capture_output=True, text=True).stdout.split(",")
    return int(out[1]) - int(out[0])


def per_pid_mb():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    usage = {}
    for line in out.strip().splitlines():
        pid, mb = (v.strip() for v in line.split(","))
        usage[int(pid)] = int(mb)
    return usage


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(OUT / "scheduler.log", "a") as handle:
        handle.write(line + "\n")


def launch(job, slot, gpu):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "3",
           "MKL_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **job["env"],
           "COMPASS_PROJECTION_LOG": str(DIAG / f"{job['name']}.jsonl"),
           "COMPASS_RUN_LABEL": job["name"]}
    handle = open(LOGS / f"{job['name']}.log", "a")
    process = subprocess.Popen(["taskset", "-c", slot, PY, "-u", *job["cmd"]], cwd=job["cwd"], env=env,
                               stdout=handle, stderr=subprocess.STDOUT)
    return process


def failed_oom(job):
    text = (LOGS / f"{job['name']}.log").read_text(errors="ignore")
    return "out of memory" in text.lower()


def ours_on(gpu):
    """Number of processes on ``gpu`` started by any run_noproj scheduler (they carry COMPASS_RUN_LABEL)."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", gpu],
                         capture_output=True, text=True).stdout.split()
    count = 0
    for pid in out:
        try:
            if b"COMPASS_RUN_LABEL=" in Path(f"/proc/{pid}/environ").read_bytes():
                count += 1
        except OSError:
            pass
    return count


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["noproj"], choices=sorted(VARIANTS))
    parser.add_argument("--skip", nargs="*", default=[], help="job names already running elsewhere")
    parser.add_argument("--wait-pids", type=int, nargs="*", default=[], help="finish only after these exit")
    arguments = parser.parse_args()
    for d in (LOGS, DIAG):
        d.mkdir(parents=True, exist_ok=True)
    queue = [job for v in arguments.variants for job in jobs(v) if job["name"] not in arguments.skip]
    running = {}      # (gpu, cores) -> (job, process, started, peak)
    slots = [(gpu, cores) for gpu in GPUS for cores in SLOTS_BY_GPU[gpu]]
    peaks = {}
    attempts = {}
    last_launch = {gpu: 0.0 for gpu in GPUS}
    log(f"scheduler on GPUs {GPUS}: {len(queue)} jobs, skipping {arguments.skip}")
    while queue or running or any(alive(p) for p in arguments.wait_pids):
        usage = per_pid_mb()
        for slot, (job, process, started, peak) in list(running.items()):
            mb = usage.get(process.pid, 0)
            running[slot] = (job, process, started, max(peak, mb))
            code = process.poll()
            if code is None:
                continue
            peak = running.pop(slot)[3]
            peaks[job["name"]] = peak
            minutes = (time.time() - started) / 60
            if failed_oom(job) and attempts.get(job["name"], 0) < 3:
                job["need"] = int(job["need"] * 1.6)
                queue.insert(0, job)
                log(f"OOM   {job['name']} after {minutes:.0f} min (peak {peak} MB); requeued needing {job['need']} MB")
            else:
                log(f"done  {job['name']} exit {code} in {minutes:.0f} min, peak {peak} MB")
        for gpu in GPUS:
            free_slots = [s for s in slots if s[0] == gpu and s not in running]
            if not queue or not free_slots or time.time() - last_launch[gpu] < SETTLE_S:
                continue
            if ours_on(gpu) >= MAX_JOBS_PER_GPU:
                continue
            free = gpu_free_mb(gpu)
            # the first queued job that fits (jobs are ordered longest first)
            fits = [j for j in queue if free >= j["need"] + MARGIN_BY_GPU.get(gpu, MARGIN_MB)]
            if fits:
                job = fits[0]
                queue.remove(job)
                attempts[job["name"]] = attempts.get(job["name"], 0) + 1
                slot = free_slots[0]
                running[slot] = (job, launch(job, slot[1], gpu), time.time(), 0)
                last_launch[gpu] = time.time()
                log(f"start {job['name']} on GPU {gpu} cores {slot[1]} (free {free} MB, needs {job['need']} MB); "
                    f"{len(queue)} queued, {len(running)} running")
        time.sleep(10)
    (OUT / "peaks.json").write_text(json.dumps(peaks, indent=2) + "\n")
    log("all done")


if __name__ == "__main__":
    main()
