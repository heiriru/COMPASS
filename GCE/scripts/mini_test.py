#!/usr/bin/env python3
"""End-to-end wiring check at toy size, CPU only. Not an experiment: the numbers it
prints say the plumbing works, nothing about accuracy.

    CUDA_VISIBLE_DEVICES=-1 ../.COMPASS/bin/python scripts/mini_test.py

1. vendored Chempy reproduces the old-numpy outputs; the per-step IMF hook with a
   constant slope reproduces the stock run bit for bit;
2. every registered model simulates a few prior draws (failure rate, seconds/run,
   slope range of the variable-IMF schedules);
3. a tiny bank -> a 2-epoch network (SMOKE_RECIPE) -> gauss_jacobian_FD composition
   of 5 TNG-mock stars -> finite, synchronised globals;
4. the observed-data loaders build x in their profiles' layouts.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gce.paths import configure  # noqa: E402

configure(cpus=3)                  # COMPASS/CLAUDE.md: at most 3 logical CPUs, before numpy

import numpy as np  # noqa: E402

from gce import compass_setup, data_sources, generate, models  # noqa: E402
from gce.observation import PROFILES  # noqa: E402
from gce.paths import DATA_ROOT, EXTERNAL  # noqa: E402
from gce.simulator import StarParameters, simulate_star, simulate_star_stock  # noqa: E402


def check_port(rows=5):
    stored = np.load(EXTERNAL["chempy_tng_val"])
    params, abundances = stored["params"], stored["abundances"]
    rng = np.random.default_rng(0)
    worst, worst_hook, compared = 0.0, 0.0, 0
    for i in rng.choice(len(params), rows * 3, replace=False):
        if not np.isfinite(abundances[i]).all() or (abundances[i] == 0).all():
            continue
        stock = simulate_star_stock(params[i])
        star = StarParameters(*params[i].astype(float))
        ours = simulate_star(star)
        hook = simulate_star(star, slope_schedule=lambda log_z, t, a=float(params[i][0]): a)
        if stock is None or ours.abundances is None:
            print(f"  row {i}: failed now but not in the stored data ({ours.failure})")
            continue
        worst = max(worst, float(np.abs(stock - abundances[i]).max()), float(np.abs(ours.abundances - stock).max()))
        worst_hook = max(worst_hook, float(np.abs(hook.abundances - ours.abundances).max()))
        compared += 1
        if compared == rows:
            break
    print(f"1. port: {compared} rows, max |new - old numpy| = {worst:.2e}, "
          f"max |constant hook - stock| = {worst_hook:.2e}")
    assert worst < 1e-10 and worst_hook == 0.0


def check_models(draws=6):
    print("2. models (prior draws at the training scale 1.5):")
    summary = {}
    for name, model in models.MODELS.items():
        rng = np.random.default_rng(1)
        g = model.sample_globals(rng, draws, 1.5)
        l = model.sample_locals(rng, g)
        fails, seconds, slopes = [], [], []
        for gi, li in zip(g, l):
            star, schedule = model.star(gi, li)
            started = time.perf_counter()
            result = simulate_star(star, imf_type=model.imf_type, slope_schedule=schedule,
                                   record_slopes=schedule is not None)
            seconds.append(time.perf_counter() - started)
            fails.append(result.failure)
            slopes += result.slopes
        failed = [f for f in fails if f]
        slope_text = f", SSP slopes {min(slopes):.2f}..{max(slopes):.2f}" if slopes else ""
        print(f"   {name:20s} theta={model.theta_names}: {len(failed)}/{draws} failed "
              f"{sorted(set(f.split(':')[0] for f in failed))}, {np.mean(seconds):.2f}s/run{slope_text}")
        summary[name] = {"failed": len(failed), "draws": draws, "seconds_per_run": float(np.mean(seconds))}
    return summary


def check_pipeline(root, workers, n_train, n_val):
    model_name, profile = "universal", PROFILES["tng_mock"]
    started = time.time()
    train_dir = generate.generate(model_name, n_train, split="smoke", seed=0, shard_size=max(n_train // 2, 1),
                                  workers=workers, root=root, verbose=False)
    val_dir = generate.generate(model_name, n_val, split="smoke", seed=1, shard_size=n_val, workers=workers,
                                root=Path(root) / "val", verbose=False)
    train_bank, val_bank = generate.load_bank(train_dir), generate.load_bank(val_dir)
    print(f"3. pipeline: banks {len(train_bank['globals'])}+{len(val_bank['globals'])} usable runs "
          f"(failure rate {train_bank['failure_rate']:.1%} {train_bank['failure_counts']}) in {time.time() - started:.0f}s")
    layout = compass_setup.Layout(models.get(model_name), profile)
    out = Path(root) / "networks" / "universal_tng_mock_smoke"
    started = time.time()
    compass_setup.train(layout, train_bank, val_bank, out, "smoke", recipe=compass_setup.SMOKE_RECIPE,
                        device="cpu")
    print(f"   trained SMOKE_RECIPE network in {time.time() - started:.0f}s -> {out}")
    model, layout, normalizer, sidecar = compass_setup.load(out, "smoke", final=True, device="cpu")
    mock = data_sources.load_tng_mock().subset(np.arange(5))
    x, notes = mock.x(layout.profile)
    started = time.time()
    composed = compass_setup.compose(model, layout, normalizer, x, sidecar, draws=16, draw_batch=8,
                                     device="cpu", timesteps=10)
    print(f"   composed 5 TNG-mock stars, 16 draws, 10 steps in {time.time() - started:.0f}s: globals "
          f"{composed['globals'].shape}, locals {composed['locals'].shape}, "
          f"median alpha {np.median(composed['globals'][:, 0]):.2f} (untrained net: meaningless); notes {notes}")
    return composed


def check_loaders():
    print("4. loaders:")
    for name, profile in (("tng_mock", "tng_mock"), ("chempy_alt_mock", "tng_mock"), ("nissen", "nissen")):
        sample = data_sources.SOURCES[name]()
        x, notes = sample.x(PROFILES[profile])
        print(f"   {name:16s} -> profile {profile:9s}: x {x.shape}, finite {np.isfinite(x).all()}, "
              f"notes {sample.notes + notes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DATA_ROOT / "smoke"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--skip-pipeline", action="store_true")
    arguments = parser.parse_args()
    check_port()
    summary = check_models()
    check_loaders()
    if not arguments.skip_pipeline:
        check_pipeline(arguments.root, arguments.workers, arguments.n_train, arguments.n_val)
    Path(arguments.root).mkdir(parents=True, exist_ok=True)
    (Path(arguments.root) / "mini_test_models.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("mini test passed")


if __name__ == "__main__":
    main()
