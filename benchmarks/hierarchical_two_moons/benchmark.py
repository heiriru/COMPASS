#!/usr/bin/env python3
"""Hierarchical Two Moons (hsbibm): every COMPASS composition rule + BayesFlow-Arruda.

Questions asked of every method, separately (see ``metrics.py``):
  Q1 the local posteriors p(theta_i | x_1:N), incl. their crescent / two-branch shape
  Q2 the population mean mu_g
  Q3 the population spread sigma_g
  Q4 the dependence between globals and locals
  Q5 how all of these change with N

Usage (from this folder; ``run.sh`` sets up the environment):
    ./run.sh all                                   # reference, train, run, report
    ./run.sh reference --Ns 1 5 10 25 50 100
    ./run.sh train --methods gauss attn_hier:arruda_direct attn_full:gauss_hierarchical
    ./run.sh train --methods bayesflow_arruda
    ./run.sh run --methods gauss_hierarchical --Ns 10
    ./run.sh report
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "artifacts"

NS = (1, 5, 10, 25, 50, 100)
DATASETS = 10
NUM_SAMPLES = 500
REFERENCE_DRAWS = 2500          # first 500 = the "reference" arm (MC floor), rest = reference
TRAIN_SAMPLES = 100_000
COMPASS_EPOCHS = 200            # 78k steps at batch 256
BAYESFLOW_EPOCHS = 100          # 78k steps at batch 128


def compass_methods():
    import compass_arms
    return list(compass_arms.METHODS)


def all_methods():
    import bayesflow_arm
    return ["reference", *compass_methods(), *bayesflow_arm.METHODS]


def is_bayesflow(method):
    return method.startswith("bayesflow")


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def reference_path(n, index):
    import task as T
    return OUT / f"reference{T.regime_tag()}" / f"N{n}_{index}.npz"


def reference(n, index, device):
    import numpy as np
    import task as T
    path = reference_path(n, index)
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    from reference import Reference
    g, locals_, x = T.dataset(n, index)
    started = time.perf_counter()
    ref_g, ref_l, diag = Reference(x, device=device).sample(REFERENCE_DRAWS, seed=index)
    if T.SIGMA_MAX is not None:
        # The posterior is exactly mirror-symmetric: mirroring a random half of the
        # draws is exact and removes any imbalance from incomplete mode mixing.
        flip = np.random.default_rng(10_000 + index).random(len(ref_g)) < 0.5
        diag["mode_balance_raw"] = float(np.mean(T.symmetric_globals(ref_g)[:, 0] > 0))
        ref_g[flip], ref_l[flip] = T.mirror(ref_g[flip], ref_l[flip])
    diag["seconds"] = time.perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, true_g=g, true_l=locals_, x=x, ref_g=ref_g, ref_l=ref_l,
             diag=json.dumps(diag))
    log(f"reference N={n} [{index}] {diag}")
    return dict(np.load(path, allow_pickle=True))


def run_reference(arguments):
    for n in arguments.Ns:
        for index in range(arguments.datasets):
            reference(n, index, arguments.device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_train(arguments):
    methods = arguments.methods or default_methods(arguments.family)
    networks = []
    for method in methods:
        if method == "reference":
            continue
        if is_bayesflow(method):
            networks.append("bayesflow")
        else:
            import compass_arms
            networks.append(compass_arms.METHODS[method][0])
    for network in dict.fromkeys(networks):
        if network == "bayesflow":
            import bayesflow_arm
            bayesflow_arm.approximators(arguments.train_samples, arguments.bayesflow_epochs,
                                        train=True)
        else:
            import compass_arms
            compass_arms.train(network, arguments.device, arguments.train_samples,
                               arguments.compass_epochs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

RESULTS = OUT          # --out moves results and draws (not the reference)


def result_path(method, n):
    return RESULTS / "results" / f"{method.replace(':', '__')}_N{n}.json"


def draws_path(method, n, index=0):
    return RESULTS / "draws" / f"{method.replace(':', '__')}_N{n}_dataset{index}.npz"


def evaluate(method, n, arguments):
    import numpy as np
    import metrics

    path = result_path(method, n)
    if path.exists() and not arguments.force:
        log(f"skip {path.name} (exists)")
        return
    rows = []
    for index in range(arguments.datasets):
        data = reference(n, index, arguments.device)
        ref_g, ref_l = data["ref_g"][NUM_SAMPLES:], data["ref_l"][NUM_SAMPLES:]
        row = {"dataset": index}
        try:
            if method == "reference":
                draws_g, draws_l, seconds = data["ref_g"][:NUM_SAMPLES], \
                    data["ref_l"][:NUM_SAMPLES], json.loads(str(data["diag"]))["seconds"]
            elif is_bayesflow(method):
                import bayesflow_arm
                draws_g, draws_l, seconds = bayesflow_arm.sample(
                    method, data["x"], arguments.num_samples, index,
                    arguments.train_samples, arguments.bayesflow_epochs)
            else:
                import compass_arms
                draws_g, draws_l, seconds = compass_arms.sample(
                    method, data["x"], arguments.num_samples, arguments.device, index,
                    arguments.train_samples, arguments.compass_epochs)
        except Exception as error:
            import traceback
            traceback.print_exc()
            row["error"] = repr(error)
            rows.append(row)
            log(f"{method} N={n} [{index}] FAILED {error!r}")
            continue
        if index == 0 or arguments.sigma_max is not None:
            draws_path(method, n, index).parent.mkdir(parents=True, exist_ok=True)
            np.savez(draws_path(method, n, index), g=draws_g, l=draws_l)
        if not np.isfinite(draws_g).all(1).any():
            row["error"] = "no finite draws"
            rows.append(row)
            continue
        row.update(metrics.evaluate(draws_g, draws_l, ref_g, ref_l, data["true_g"],
                                    data["true_l"], data["x"], seed=index))
        row["seconds"] = seconds
        rows.append(row)
        log(f"{method} N={n} [{index}] local_c2st={row['local_c2st']:.3f} "
            f"branch={row['branch_err']:.3f} mu_w1={row['mu_w1']:.2f} "
            f"sigma_w1={row['sigma_w1']:.2f} dep={row['dep_corr_err']:.3f} "
            f"t={seconds:.1f}s")
    summary = metrics.summarize(rows)
    summary.update(method=method, N=n)
    if is_bayesflow(method):
        import bayesflow_arm
        summary["tuned"] = bayesflow_arm.tune(method, n, arguments.train_samples,
                                              arguments.bayesflow_epochs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in summary.items()}))


def default_methods(family):
    import bayesflow_arm
    if family == "compass":
        return ["reference", *compass_methods()]
    if family == "bayesflow":
        return list(bayesflow_arm.METHODS)
    return all_methods()


def run_methods(arguments):
    methods = arguments.methods or default_methods(arguments.family)
    for n in arguments.Ns:
        for method in methods:
            evaluate(method, n, arguments)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("command", choices=["reference", "train", "run", "report", "all"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--Ns", type=int, nargs="+", default=list(NS))
    parser.add_argument("--datasets", type=int, default=DATASETS)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--compass-epochs", type=int, default=COMPASS_EPOCHS)
    parser.add_argument("--bayesflow-epochs", type=int, default=BAYESFLOW_EPOCHS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--family", choices=["compass", "bayesflow", "all"], default="all",
                        help="default method set when --methods is not given")
    parser.add_argument("--out", default=None,
                        help="folder for results/draws (default artifacts/); reference is shared")
    parser.add_argument("--bf-d1", type=float, nargs="+", default=None,
                        help="BayesFlow tuning grid for d1 (default bayesflow_arm.D1_GRID)")
    parser.add_argument("--bf-shifts", type=float, nargs="+", default=None)
    parser.add_argument("--bf-tune-datasets", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="recompute existing results")
    parser.add_argument("--sigma-max", type=float, default=None,
                        help="test/tuning datasets from the prior conditioned on max(sigma) <= this; "
                             "results default to artifacts_smax<v>/")
    arguments = parser.parse_args()
    import task as T
    T.SIGMA_MAX = arguments.sigma_max
    if arguments.bf_d1 or arguments.bf_shifts or arguments.bf_tune_datasets:
        import bayesflow_arm
        bayesflow_arm.D1_GRID = tuple(arguments.bf_d1 or bayesflow_arm.D1_GRID)
        bayesflow_arm.SHIFT_GRID = tuple(arguments.bf_shifts or bayesflow_arm.SHIFT_GRID)
        bayesflow_arm.TUNE_DATASETS = arguments.bf_tune_datasets or bayesflow_arm.TUNE_DATASETS
    global RESULTS
    if arguments.out:
        RESULTS = Path(arguments.out).resolve()
    elif arguments.sigma_max is not None:
        RESULTS = HERE / f"artifacts{T.regime_tag()}"
    import report
    report.REFERENCE = OUT / f"reference{T.regime_tag()}"
    if arguments.num_samples != NUM_SAMPLES:
        parser.error("--num-samples must stay 500: the reference arm is 500 draws")

    if arguments.command == "report":
        import report
        report.main(RESULTS)
        return
    if arguments.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        from autocvd import autocvd
        autocvd(num_gpus=1, interval=1)
    if arguments.family == "bayesflow" or any(is_bayesflow(m) for m in (arguments.methods or [])):
        os.environ.setdefault("KERAS_BACKEND", "torch")
    if arguments.command in ("reference", "all"):
        run_reference(arguments)
    if arguments.command in ("train", "all"):
        run_train(arguments)
    if arguments.command in ("run", "all"):
        run_methods(arguments)
    if arguments.command == "all":
        import report
        report.main(RESULTS)


if __name__ == "__main__":
    main()
