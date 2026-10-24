"""Simulation banks: (globals, locals, clean abundances) for one model and yield set.

Noise is NOT applied here. A bank stores clean Chempy output, and observation.py
adds a profile's noise when a training set is built, so one bank serves every
survey profile and the noise can be redrawn.

Layout (``DATA_ROOT/banks/<model>/<yields tag>/<split>-wide<scale>/``)::

    manifest.json        model, priors, yields, seed, shard size, code fingerprint
    shard-00000.npz      globals [s, G], locals [s, L], abundances [s, 9] (NaN on failure),
                         failure [s] ("" if fine), seconds [s]

Shards are resumable: an existing shard is never recomputed. Failed runs are kept,
because the region where Chempy cannot form the star is an implicit truncation of
the prior, and its size has to be reported (PLAN.md, G0).

Runs are single-threaded; ``--workers`` processes share ``--cpus`` logical CPUs (3 by the
repository rule in COMPASS/CLAUDE.md; ~0.35 s/run, so 200k runs take ~6.5 h on 3 CPUs).
Raise ``--cpus`` only deliberately.

    python -m gce.generate --model universal --n 2000 --split smoke
    python -m gce.generate --model metallicity --n 200000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .paths import DATA_ROOT, GCE_ROOT
from .simulator import TRACED, TNG_YIELDS, YieldSet

SPLIT_SEEDS = {"train": 0, "validation": 1, "test": 2, "smoke": 9}


def bank_dir(model_name, yields: YieldSet = TNG_YIELDS, split="train", scale=1.5, root=DATA_ROOT):
    return Path(root) / "banks" / model_name / yields.tag / f"{split}-wide{scale:g}"


def code_fingerprint():
    """Hash of the files that decide what a bank contains."""
    digest = hashlib.sha256()
    for path in sorted([GCE_ROOT / "gce" / "simulator.py", GCE_ROOT / "gce" / "models.py",
                        *(GCE_ROOT / "vendor" / "Chempy").glob("*.py")]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


_WORKER = {}


def _init_worker(model_name, yields):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    from . import models
    _WORKER["model"] = models.get(model_name)
    _WORKER["yields"] = yields


def _run(row):
    from .simulator import simulate_star
    model, yields = _WORKER["model"], _WORKER["yields"]
    g, l = row
    star, schedule = model.star(g, l)
    started = time.perf_counter()
    result = simulate_star(star, yields, model.imf_type, schedule)
    seconds = time.perf_counter() - started
    values = result.abundances if result.abundances is not None else np.full(len(TRACED), np.nan)
    return values, result.failure, seconds


def generate(model_name, n, yields: YieldSet = TNG_YIELDS, split="train", scale=1.5, seed=0,
             shard_size=2000, workers=4, root=DATA_ROOT, verbose=True):
    from . import models
    model = models.get(model_name)
    out = bank_dir(model_name, yields, split, scale, root)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": model.name, "description": model.description,
        "global_names": model.global_names, "local_names": model.local_names,
        "global_prior_mean": model.global_prior()[0].tolist(), "global_prior_sd": model.global_prior()[1].tolist(),
        "training_scale": scale, "imf_type": model.imf_type, "schedule": model.schedule,
        "yields": yields.__dict__, "traced": TRACED, "split": split, "seed": seed,
        "shard_size": shard_size, "code": code_fingerprint(),
    }
    path = out / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        stale = {k: (old.get(k), v) for k, v in manifest.items() if k != "code" and old.get(k) != json.loads(json.dumps(v))}
        if stale:
            raise RuntimeError(f"{out} holds a different bank: {stale}")
        if old.get("code") != manifest["code"]:
            print(f"WARNING: code fingerprint changed ({old.get('code')} -> {manifest['code']}); "
                  "new shards may differ from old ones", flush=True)
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")

    from multiprocessing import get_context
    shards = int(np.ceil(n / shard_size))
    with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(model_name, yields)) as pool:
        for index in range(shards):
            target = out / f"shard-{index:05d}.npz"
            if target.exists():
                continue
            count = min(shard_size, n - index * shard_size)
            rng = np.random.default_rng([SPLIT_SEEDS[split], seed, index])
            g = model.sample_globals(rng, count, scale)
            l = model.sample_locals(rng, g)
            started = time.time()
            results = pool.map(_run, list(zip(g, l)), chunksize=max(1, count // (4 * workers)))
            abundances = np.stack([r[0] for r in results])
            failure = np.array([r[1] for r in results])
            seconds = np.array([r[2] for r in results])
            np.savez(target.with_suffix(".tmp.npz"), globals=g, locals=l, abundances=abundances,
                     failure=failure, seconds=seconds)
            os.replace(target.with_suffix(".tmp.npz"), target)
            if verbose:
                print(f"[{model_name} {split}] shard {index + 1}/{shards}: {count} runs, "
                      f"{(failure != '').mean():.1%} failed, {time.time() - started:.0f}s wall, "
                      f"{seconds.mean():.2f}s/run", flush=True)
    return out


def galaxy_dir(model_name, n_stars, yields: YieldSet = TNG_YIELDS, root=DATA_ROOT):
    return Path(root) / "galaxies" / model_name / yields.tag / f"stars{n_stars}"


def generate_galaxies(model_name, n_galaxies, n_stars, yields: YieldSet = TNG_YIELDS, seed=0, workers=3,
                      root=DATA_ROOT, verbose=True):
    """Known-answer test sets: per galaxy one draw of the globals from the *target* prior
    (scale 1) and ``n_stars`` stars from p(l | g). ``galaxy-00000.npz`` holds globals [G],
    locals [n, L], abundances [n, 9] (NaN where Chempy failed), failure [n]. Failed stars
    are kept in the file; inference drops them, which is a (reported) selection."""
    from . import models
    model = models.get(model_name)
    out = galaxy_dir(model_name, n_stars, yields, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({
        "model": model.name, "global_names": model.global_names, "local_names": model.local_names,
        "n_stars": n_stars, "yields": yields.__dict__, "seed": seed, "code": code_fingerprint()}, indent=2) + "\n")
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(model_name, yields)) as pool:
        for index in range(n_galaxies):
            target = out / f"galaxy-{index:05d}.npz"
            if target.exists():
                continue
            rng = np.random.default_rng([SPLIT_SEEDS["test"], seed, index])
            g = model.sample_globals(rng, 1, 1.0)
            l = model.sample_locals(rng, np.repeat(g, n_stars, axis=0))
            results = pool.map(_run, [(g[0], li) for li in l], chunksize=max(1, n_stars // (4 * workers)))
            failure = np.array([r[1] for r in results])
            np.savez(target, globals=g[0], locals=l, abundances=np.stack([r[0] for r in results]), failure=failure)
            if verbose:
                print(f"[{model_name}] galaxy {index + 1}/{n_galaxies}: {n_stars} stars, "
                      f"{(failure != '').mean():.1%} failed", flush=True)
    return out


def load_bank(directory, drop_failures=True):
    """Concatenate a bank's shards. Returns a dict and prints the failure breakdown."""
    directory = Path(directory)
    files = sorted(directory.glob("shard-*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    if not files:
        raise FileNotFoundError(f"no shards in {directory}")
    parts = [np.load(f) for f in files]
    bank = {k: np.concatenate([p[k] for p in parts]) for k in ("globals", "locals", "abundances", "failure", "seconds")}
    bank["manifest"] = json.loads((directory / "manifest.json").read_text())
    failed = bank["failure"] != ""
    bank["failure_counts"] = {str(k): int(v) for k, v in zip(*np.unique(
        [f.split(":")[0] for f in bank["failure"][failed]], return_counts=True))} if failed.any() else {}
    bank["failure_rate"] = float(failed.mean())
    if drop_failures:
        for k in ("globals", "locals", "abundances", "seconds"):
            bank[k] = bank[k][~failed]
        bank["failure"] = bank["failure"][~failed]
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, required=True, help="runs (bank) or galaxies (--galaxy-stars)")
    parser.add_argument("--galaxy-stars", type=int, default=0,
                        help="> 0: make --n known-answer galaxies of this many stars instead of a bank")
    parser.add_argument("--split", default="train", choices=tuple(SPLIT_SEEDS))
    parser.add_argument("--scale", type=float, default=1.5, help="training-prior widening of the globals")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=3, help="logical-CPU cap (repository rule: 3)")
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: --cpus)")
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--sn2", default=TNG_YIELDS.sn2)
    parser.add_argument("--agb", default=TNG_YIELDS.agb)
    parser.add_argument("--sn1a", default=TNG_YIELDS.sn1a)
    parser.add_argument("--root", default=str(DATA_ROOT))
    arguments = parser.parse_args()
    from .paths import configure
    configure(arguments.cpus)      # children inherit the affinity mask
    arguments.workers = arguments.workers or arguments.cpus
    yields = YieldSet(sn2=arguments.sn2, agb=arguments.agb, sn1a=arguments.sn1a)
    if arguments.galaxy_stars > 0:
        out = generate_galaxies(arguments.model, arguments.n, arguments.galaxy_stars, yields, arguments.seed,
                                arguments.workers, arguments.root)
        print(f"{out}: {arguments.n} galaxies of {arguments.galaxy_stars} stars")
        return
    out = generate(arguments.model, arguments.n, yields, arguments.split, arguments.scale, arguments.seed,
                   arguments.shard_size, arguments.workers, arguments.root)
    bank = load_bank(out)
    print(f"{out}: {len(bank['globals'])} usable runs, failure rate {bank['failure_rate']:.2%} "
          f"{bank['failure_counts']}")


if __name__ == "__main__":
    main()
