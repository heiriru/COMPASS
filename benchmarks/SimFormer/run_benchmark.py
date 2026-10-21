"""Sequential driver: one run at a time, one GPU, report after every run.

Run with the repository env (it needs ``autocvd``, pandas, matplotlib)::

    cd COMPASS/benchmarks/SimFormer
    ../../.COMPASS/bin/python run_benchmark.py pilot
    ../../.COMPASS/bin/python run_benchmark.py fig4a --methods score_transformer compass_legacy

Each job is one official ``scoresbi`` run (train + evaluate) launched in the
``.simformer`` env through ``bridge/run_scoresbi.py``. Before each job
``autocvd(num_gpus=1)`` waits for a free GPU and pins the job to it; JAX
preallocation is disabled so the harness and the COMPASS worker share that one
card politely. Jobs already present in ``results/<phase>/summary.csv`` are
skipped, so the driver can be restarted at any point.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

CPU_THREAD_LIMIT = 3

ROOT = Path(__file__).resolve().parent
SIMFORMER_PY = ROOT / ".simformer" / "bin" / "python"
LOGS = ROOT / "results" / "logs"

PHASES = {
    # phase: (tasks, budgets, default methods, eval override)
    "pilot": (["two_moons"], [10_000],
              ["score_transformer", "compass_legacy", "compass_tokenized"], "c2st_posterior"),
    # Resample a trained checkpoint with other samplers (pass
    # --extra ++method.compass.load_from=<run>/local/compass_model).
    "sampler": (["two_moons"], [10_000],
                ["compass_legacy_euler500", "compass_legacy_dpm200"], "c2st_posterior"),
    "fig4a": (["gaussian_linear", "gaussian_mixture", "two_moons", "slcp"], [1_000, 10_000, 100_000],
              ["score_transformer", "compass_legacy_full"], "c2st_posterior"),
    "fig4b": (["tree_all_cond", "marcov_chain_all_cond", "two_moons_all_cond", "slcp_all_cond"],
              [1_000, 10_000, 100_000], ["score_transformer", "compass_legacy_full"], "c2st"),
    # Paper appendix (Fig. A8, A10): NLL and expected coverage of the posterior
    # and the likelihood, on the Fig. 4a models (evaluation only, no training).
    "extra4a": (["gaussian_linear", "gaussian_mixture", "two_moons", "slcp"], [1_000, 10_000, 100_000],
                ["score_transformer", "compass_legacy_full_fixeps"], "nll_cov"),
    # Likelihood and joint C2ST on the Fig. 4b models (evaluation only).
    "extra4b": (["tree_all_cond", "marcov_chain_all_cond", "two_moons_all_cond", "slcp_all_cond"],
                [1_000, 10_000, 100_000], ["score_transformer", "compass_legacy_full_fixeps"], "lik_joint"),
}
# Evaluation-only phases reuse the models trained in these phases.
SOURCE_PHASE = {"extra4a": "fig4a", "extra4b": "fig4b"}
EVAL_METRICS = {
    "c2st_posterior": ["c2st_posterior"], "c2st": ["c2st"],
    "nll_cov": ["nll_likelihood", "cov_likelihood", "nll_posterior", "cov_posterior"],
    "lik_joint": ["c2st_likelihood", "c2st_joint"],
}


def _cpu_times():
    out = {}
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith("cpu") and line[3].isdigit():
                name, *vals = line.split()
                vals = list(map(int, vals))
                out[int(name[3:])] = (vals[3] + vals[4], sum(vals))   # idle+iowait, total
    return out


def pick_idle_cpus(n, window=2.0):
    """The n least-busy allowed CPUs over a short window.

    Pinning every job to the first three CPUs put the COMPASS worker on the
    same cores as two other 100%-CPU jobs and made it 4x slower per step.
    """
    allowed = sorted(os.sched_getaffinity(0))
    a = _cpu_times()
    time.sleep(window)
    b = _cpu_times()
    def busy(c):
        di, dt = b[c][0] - a[c][0], b[c][1] - a[c][1]
        return 1.0 - di / dt if dt > 0 else 1.0
    return sorted(sorted(allowed, key=busy)[:n])


def done(phase, task, budget, method):
    path = ROOT / "results" / phase / "summary.csv"
    if not path.exists():
        return False
    import pandas as pd
    df = pd.read_csv(path)
    wanted = EVAL_METRICS.get(PHASES[phase][3], None)
    rows = df[(df.task == task) & (df.num_simulations == budget) & (df.method == method)]
    if wanted is None:
        return not rows[rows.metric.astype(str).str.startswith("c2st")].empty
    return set(wanted) <= set(rows.metric.astype(str))


def seed_eval_phase(phase):
    """An evaluation-only phase reloads Simformer models by id: the harness
    looks the id up in <phase>/summary.csv and unpickles <phase>/models/.
    Seed both from the source phase (rows relabelled so they never count as a
    result) and never touch the source phase itself."""
    import pandas as pd
    src, dst = ROOT / "results" / SOURCE_PHASE[phase], ROOT / "results" / phase
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "models").exists():
        (dst / "models").symlink_to(src / "models")
    if not (dst / "summary.csv").exists():
        df = pd.read_csv(src / "summary.csv")
        df = df[(df.method == "score_transformer") & df.metric.astype(str).str.startswith("c2st")].copy()
        df["metric"] = "source_model"
        df.to_csv(dst / "summary.csv", index=False)


def simformer_model_id(phase, task, budget):
    import pandas as pd
    df = pd.read_csv(ROOT / "results" / SOURCE_PHASE[phase] / "summary.csv")
    hit = df[(df.method == "score_transformer") & (df.task == task) & (df.num_simulations == budget)]
    return None if hit.empty else int(hit.model_id.iloc[-1])


def find_checkpoint(phase, task, budget, method):
    """compass_model dir of the latest finished run of (task, budget, method)."""
    import yaml
    hits = []
    for cfg in sorted((ROOT / "results" / phase).glob("*/local/.hydra/config.yaml")):
        c = yaml.safe_load(cfg.read_text())
        if (c["method"]["name"] == method and c["task"]["name"] == task
                and int(c["task"]["num_simulations"]) == budget):
            ck = cfg.parents[1] / "compass_model"
            if (ck / "Model_checkpoint.pt").exists() and (ck / "zscore.npz").exists():
                hits.append(ck)
    return hits[-1] if hits else None


def pick_gpu():
    import contextlib
    import io
    from autocvd import autocvd
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    print(f"[driver] {time.strftime('%H:%M:%S')} waiting for a free GPU (autocvd)", flush=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ids = autocvd(num_gpus=1, interval=1)   # its spinner would flood the log
    if len(ids) != 1:
        raise RuntimeError(f"autocvd returned {ids}")
    return str(ids[0])


def run_job(phase, task, budget, method, eval_name, extra):
    gpu = pick_gpu()
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        **{k: str(CPU_THREAD_LIMIT) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                               "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    })
    save_model = "false" if method.startswith("compass") else "true"
    cmd = [str(SIMFORMER_PY), str(ROOT / "bridge" / "run_scoresbi.py"),
           f"name={phase}", f"method={method}", f"task={task}",
           f"task.num_simulations={budget}", f"eval={eval_name}", f"save_model={save_model}",
           *extra]
    cpus = pick_idle_cpus(CPU_THREAD_LIMIT)
    cmd = ["taskset", "-c", ",".join(map(str, cpus))] + cmd
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{phase}__{task}__{budget}__{method}.log"
    print(f"[driver] {time.strftime('%H:%M:%S')} START {phase} {task} n={budget} {method} "
          f"on physical GPU {gpu}  (log: {log.relative_to(ROOT)})", flush=True)
    started = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    minutes = (time.time() - started) / 60
    print(f"[driver] {time.strftime('%H:%M:%S')} END   {phase} {task} n={budget} {method} "
          f"rc={rc} ({minutes:.1f} min)", flush=True)
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=sorted(PHASES))
    p.add_argument("--methods", nargs="+")
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--budgets", nargs="+", type=int)
    p.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides for every job")
    p.add_argument("--resample-from", nargs="+", help="for compass_* jobs: reuse the checkpoint of the "
                   "first of these methods that has one for the same (task, budget); no training")
    args = p.parse_args()

    tasks, budgets, methods, eval_name = PHASES[args.phase]
    tasks, budgets, methods = args.tasks or tasks, args.budgets or budgets, args.methods or methods
    # Budget-major: the cheap budgets fill the whole table first, and each
    # (task, budget) cell finishes both methods back to back.
    queue = [(t, b, m) for b in budgets for t in tasks for m in methods]
    print(f"[driver] {len(queue)} jobs in phase {args.phase}", flush=True)
    ck_phase = SOURCE_PHASE.get(args.phase, args.phase)
    if args.phase in SOURCE_PHASE:
        seed_eval_phase(args.phase)
    for task, budget, method in queue:
        if done(args.phase, task, budget, method):
            print(f"[driver] skip (done) {task} n={budget} {method}", flush=True)
            continue
        extra = list(args.extra)
        if args.resample_from and method.startswith("compass"):
            ck = next((c for c in (find_checkpoint(ck_phase, task, budget, m) for m in args.resample_from)
                       if c is not None), None)
            if ck is not None:
                extra.append(f"++method.compass.load_from={ck}")
                print(f"[driver] resampling checkpoint {ck.relative_to(ROOT)}", flush=True)
            elif args.phase in SOURCE_PHASE:
                print(f"[driver] skip (no checkpoint) {task} n={budget} {method}", flush=True)
                continue
        if args.phase in SOURCE_PHASE and method == "score_transformer":
            mid = simformer_model_id(args.phase, task, budget)
            if mid is None:
                print(f"[driver] skip (no Simformer model) {task} n={budget}", flush=True)
                continue
            extra.append(f"model_id={mid}")
        rc = run_job(args.phase, task, budget, method, eval_name, extra)
        subprocess.call([sys.executable, str(ROOT / "report.py"), args.phase], cwd=str(ROOT))
        if rc != 0:
            print(f"[driver] job failed (rc={rc}); continuing with the next one", flush=True)


if __name__ == "__main__":
    main()
