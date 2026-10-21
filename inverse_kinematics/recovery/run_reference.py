#!/usr/bin/env python3
"""Exact MCMC reference posteriors for every recovery dataset, with convergence diagnostics.

Usage (from COMPASS/):
    .COMPASS/bin/python inverse_kinematics/recovery/run_reference.py --kind single --count 200
    .COMPASS/bin/python inverse_kinematics/recovery/run_reference.py --kind hier --layout angle --n 8

Writes ``reference.npz`` next to the dataset: ``theta`` draws ``(count, draws, n, 4)`` in
node order, plus per-dataset R-hat / bulk-ESS of every continuous state coordinate.
CPU only.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import json  # noqa: E402

import numpy as np  # noqa: E402

import datasets  # noqa: E402
import ik_mcmc as M  # noqa: E402
import simulator as sim  # noqa: E402


def sample(x, collapsed, rng, batch, chains, warmup, draws, thin, partial=None, stream=None):
    """MCMC for every dataset in ``x``: ``theta (len(x), chains*draws, n, 4)``, R-hat and ESS lists.

    With ``partial`` (a directory) every finished batch is saved there and a rerun skips it, so
    an interrupted run resumes where it stopped. Each batch then draws from its own named stream
    ``stream + batch start``, which makes a resumed run identical to an uninterrupted one."""
    count = len(x)
    theta = np.empty((count, chains * draws, x.shape[1], 4))
    rhat, ess, acceptance = [], [], []
    started = time.perf_counter()
    if partial is not None:
        partial.mkdir(parents=True, exist_ok=True)
    for start in range(0, count, batch):
        xb = x[start:start + batch]
        saved = None if partial is None else partial / f"batch_{start:04d}.npz"
        if saved is not None and saved.exists():
            stored = np.load(saved, allow_pickle=True)
            theta[start:start + len(xb)] = stored["theta"]
            rhat.extend(stored["rhat"].tolist())
            ess.extend(stored["ess"].tolist())
            acceptance.append(stored["acceptance"].item())
            print(f"  datasets {start}-{start + len(xb) - 1}: resumed from {saved.name}", flush=True)
            continue
        if stream is not None:
            rng = sim.generator(f"{stream}-batch{start}")
        if collapsed:
            out = M.sample_angle_collapsed(xb, rng, chains=chains, warmup=warmup, draws=draws, thin=thin)
        else:
            out = M.sample_angle_height(xb, rng, chains=chains, warmup=warmup, draws=draws, thin=thin)
        acceptance.append(out["acceptance"])
        for b in range(len(xb)):
            coords = {"angle_1": out["a1"][b]}
            h = out["h"][b]
            if h.ndim == 3:                         # (chains, draws, n): one height per observation
                coords.update({f"height_{j}": h[..., j] for j in range(h.shape[-1])})
            else:
                coords["height"] = h
            rhat.append({k: M.rhat(v) for k, v in coords.items()})
            ess.append({k: M.ess_bulk(v) for k, v in coords.items()})
            a1 = out["a1"][b].reshape(-1)
            hb = h.reshape(chains * draws, -1)
            hb = np.broadcast_to(hb, (len(a1), xb.shape[1])) if hb.shape[1] == 1 else hb
            a2, a3 = M.draw_locals(a1, hb, xb[b], rng)
            theta[start + b] = np.stack([np.broadcast_to(a1[:, None], hb.shape), hb, a2, a3], -1)
        if saved is not None:
            tmp = saved.with_suffix(".tmp.npz")
            np.savez(tmp, theta=theta[start:start + len(xb)], rhat=np.array(rhat[-len(xb):], dtype=object),
                     ess=np.array(ess[-len(xb):], dtype=object), acceptance=np.array(out["acceptance"], dtype=object))
            tmp.rename(saved)
        print(f"  datasets {start}-{start + len(xb) - 1}: {time.perf_counter() - started:.0f}s  "
              f"max rhat {max(max(r.values()) for r in rhat[-len(xb):]):.3f}  "
              f"min ess {min(min(e.values()) for e in ess[-len(xb):]):.0f}", flush=True)
    return theta, rhat, ess, acceptance, time.perf_counter() - started


def run(kind, layout, n, count, batch, chains, warmup, draws, thin, seed, repair=0, threshold=1.01,
        resumable=False):
    """``repair > 0``: rerun only datasets whose max R-hat exceeds ``threshold``, with
    chains ``4**round`` times longer (same retained draw count, larger thinning),
    for up to ``repair`` rounds, replacing their draws and diagnostics in place."""
    data = datasets.load_or_make(kind, layout, n, count)
    x = data["x"]
    folder = datasets.folder(kind, layout, n)
    collapsed = kind == "single" or layout == "angle"
    if repair:
        theta = np.load(folder / "reference.npz")["theta"].astype(np.float64)
        summary = json.loads((folder / "reference_diagnostics.json").read_text())
        rhat = [d["rhat"] for d in summary["per_dataset"]]
        ess = [d["ess"] for d in summary["per_dataset"]]
        acceptance, seconds = [summary["acceptance"]], summary["seconds"]
        for round_ in range(1, repair + 1):
            bad = [i for i, r in enumerate(rhat) if max(r.values()) > threshold]
            print(f"repair round {round_}: {len(bad)} datasets above {threshold}", flush=True)
            if not bad:
                break
            rng = sim.generator(f"reference-{kind}-{layout}-n{n}-repair{round_}", seed)
            factor = 4 ** round_
            t2, r2, e2, _, s2 = sample(x[bad], collapsed, rng, batch, chains, warmup * factor, draws, thin * factor)
            seconds += s2
            for k, i in enumerate(bad):
                theta[i], rhat[i], ess[i] = t2[k], r2[k], e2[k]
        thin = f"{thin} (repaired datasets up to x{4 ** repair})"
    else:
        rng = sim.generator(f"reference-{kind}-{layout}-n{n}", seed)
        partial = folder / "reference_partial" if resumable else None
        theta, rhat, ess, acceptance, seconds = sample(
            x, collapsed, rng, batch, chains, warmup, draws, thin, partial=partial,
            stream=f"reference-{kind}-{layout}-n{n}-{seed}" if resumable else None)
    np.savez_compressed(folder / "reference.npz", theta=theta.astype(np.float32))
    summary = {
        "sampler": "collapsed (a1 | heights integrated out)" if collapsed else "2-D (a1, h)",
        "chains": chains, "warmup": warmup, "draws_per_chain": draws, "thin": thin,
        "seconds": seconds,
        "max_rhat": max(max(r.values()) for r in rhat),
        "median_rhat_angle_1": float(np.median([r["angle_1"] for r in rhat])),
        "min_ess": min(min(e.values()) for e in ess),
        "median_ess_angle_1": float(np.median([e["angle_1"] for e in ess])),
        "datasets_rhat_above_1.01": int(sum(max(r.values()) > 1.01 for r in rhat)),
        "acceptance": {k: float(np.mean([a[k] for a in acceptance])) for k in acceptance[0]},
        "per_dataset": [{"rhat": r, "ess": e} for r, e in zip(rhat, ess)],
    }
    (folder / "reference_diagnostics.json").write_text(json.dumps(summary, indent=1))
    print(f"{folder.name}: {seconds:.0f}s, max R-hat {summary['max_rhat']:.4f}, "
          f"min bulk-ESS {summary['min_ess']:.0f}, {summary['datasets_rhat_above_1.01']} datasets above 1.01")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("single", "hier"), required=True)
    parser.add_argument("--layout", choices=("angle", "angle_height"), default=None)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=250, help="retained draws per chain")
    parser.add_argument("--thin", type=int, default=4)
    parser.add_argument("--seed", type=int, default=datasets.SEED)
    parser.add_argument("--resumable", action="store_true",
                        help="save each finished batch under reference_partial/ and skip it on a rerun")
    parser.add_argument("--repair", type=int, default=0,
                        help="rounds of rerunning only the datasets with R-hat > 1.01, 4x longer each round")
    a = parser.parse_args()
    if a.kind == "hier" and a.layout is None:
        parser.error("--kind hier needs --layout")
    run(a.kind, a.layout, a.n, a.count, a.batch, a.chains, a.warmup, a.draws, a.thin, a.seed,
        repair=a.repair, resumable=a.resumable)


if __name__ == "__main__":
    main()
