#!/usr/bin/env python3
"""With vs without a safeguard (run_noproj.py variants: noproj = information projection off,
nocurv = gauss_jacobian curvature lower clip off, noboth = both off), on every rerun experiment.

For each (experiment, arm, n): the global's P(>0) for exact / with / without, the W1 distance
to the exact posterior in exact-posterior sd, and the posterior sd relative to exact; plus the
diagnostics of the no-projection run: share of negative information eigenvalues, share of
evaluations where the projection would have lifted, the mean relative lift ||I+ - I||/||I||,
and the share of evaluations where the composed-level spectral repair fired instead.
Partial pooling is scored as in score_push.py.

    python compare_noproj.py [--variant noproj|nocurv|noboth]
        -> artifacts/noproj/comparison[_<variant>].{txt,json}, figures/16_<variant>.png
"""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "3")
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:3])

import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
TOYS = HERE.parent / "abs_hierarchy" / "artifacts"
OUT = ART / "noproj"
RNG = np.random.default_rng(0)
JAC, GH = "gauss_jacobian_FD", "gauss_hierarchical"
V = "noproj"   # set from --variant in main()


def label_of(name):
    """Diagnostics label: the first (noproj) scheduler used bare names, later ones <variant>_<name>."""
    return name if V == "noproj" else f"{V}_{name}"


def from_grid(grid, density, count=200_000):
    cdf = np.concatenate([[0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    return np.interp(RNG.random(count), cdf / cdf[-1], grid)


def diagnostics():
    """(job, correction, n) -> mean of each diagnostic over that run's sample() calls."""
    table = defaultdict(lambda: defaultdict(list))
    for path in (OUT / "diagnostics").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            d = json.loads(line)
            key = (d["label"], d["correction"], d["observations"])
            evaluations = max(d["score_evaluations"], d["composed_precision_evaluations"], 1)
            table[key]["neg"].append(d["negative_information_fraction"])
            table[key]["active"].append(d["adaptation_fraction"])
            table[key]["lift"].append(d["mean_relative_adaptation"])
            table[key]["repair"].append(d["repair_count"] / evaluations)
            table[key]["curv"].append(d.get("negative_curvature_fraction", 0.0))
    return {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in table.items()}


def rows():
    """(experiment, arm label, job, correction, n, exact, with, without)."""
    out = []
    # folded-hierarchy toys
    for model, root, modes in (
            ("abs", TOYS, [("hierarchical", [JAC]), ("none", [GH, JAC]), ("oracle_joint", [GH, JAC]),
                           ("oracle_hierarchical", [JAC])]),
            ("square", TOYS / "square", [("hierarchical", [JAC]), ("none", [GH, JAC])]),
            ("lean", TOYS / "lean", [("hierarchical", [JAC]), ("none", [GH, JAC]), ("oracle_joint", [GH, JAC]),
                                     ("oracle_hierarchical", [JAC]), ("hierarchical_R", [JAC]),
                                     ("hierarchical_T", [JAC]), ("hierarchical_TF", [JAC]),
                                     ("hierarchical_TFR", [JAC])]),
            ("curved", TOYS / "curved", [("hierarchical", [JAC]), ("hierarchical_R", [JAC])])):
        runs = root / "runs"
        for mode, methods in modes:
            for method in methods:
                for n in (4, 16, 64, 128):
                    on, off = runs / f"{mode}_{method}_n{n}.npz", runs / f"{mode}_{V}_{method}_n{n}.npz"
                    if on.exists() and off.exists():
                        out.append((f"toy {model}", f"{mode} {method}", label_of(f"toy_{model}_{mode}_{V}"), method, n,
                                    np.load(runs / f"mcmc_n{n}.npz")["g"], np.load(on)["g"], np.load(off)["g"]))
    # Gaussian mixture
    for mode in ("hierarchical", "none", "oracle"):
        on_path = ART / "gm" / mode / "scaling" / "posteriors" / "draws.npz"
        off_path = ART / "gm" / mode / "scaling" / f"posteriors_{V}" / "draws.npz"
        if not off_path.exists():
            continue
        on, off = np.load(on_path), np.load(off_path)
        for regime in ("symmetric", "asymmetric"):
            for method, correction in ((JAC, JAC), ("arruda_gauss_direct", "arruda_schur_gauss")):
                for n in (8, 16, 32, 64):
                    key = f"{regime}_n{n}_{method}"
                    if key in on.files and key in off.files:
                        out.append((f"GM {regime}", f"{mode} {method}", label_of(f"gm_{mode}"), correction, n,
                                    on[f"{regime}_n{n}_exact_mc"], on[key], off[key]))
    # IK
    e = np.load(HERE.parent / "Arruda" / "artifacts" / "ik" / "exact" / "absx1_a0.6.npz")
    for folder, job in (("hierarchical", label_of("ik_absx1")), ("hierarchical_R", label_of("ik_absx1_R"))):
        for n in (8, 32, 128):
            on = ART / "ik_absx1" / "runs" / folder / f"{JAC}_n{n}.npz"
            off = ART / "ik_absx1" / "runs" / f"{folder}_{V}" / f"{JAC}_n{n}.npz"
            if off.exists():
                out.append(("IK |x1|", f"{folder} {JAC}", job, JAC, n, from_grid(e["grid"], e[f"n{n}"]),
                            np.load(on)["theta"][:, 0, 0], np.load(off)["theta"][:, 0, 0]))
    e = np.load(ART / "ik_flat" / "exact.npz")
    for variant in ("fixed", "global"):
        for n in (8, 32, 128):
            on = ART / "ik_flat" / "runs" / variant / f"{JAC}_n{n}.npz"
            off = ART / "ik_flat" / "runs" / f"{variant}_{V}" / f"{JAC}_n{n}.npz"
            if off.exists():
                out.append((f"IK flat {variant}", f"hierarchical {JAC}", label_of(f"ik_flat_{variant}"), JAC, n,
                            from_grid(e[f"{variant}_grid_n{n}"], e[f"{variant}_n{n}"]),
                            np.load(on)["theta"][:, 0, 0], np.load(off)["theta"][:, 0, 0]))
    e = np.load(ART / "ik_reach" / "exact.npz")
    for variant in ("sym", "shift"):
        for menu in ("", "_bernoulli"):
            for n in (8, 32, 128):
                on = ART / "ik_reach" / "runs" / f"{variant}{menu}" / f"{JAC}_n{n}.npz"
                off = ART / "ik_reach" / "runs" / f"{variant}{menu}_{V}" / f"{JAC}_n{n}.npz"
                if off.exists():
                    out.append((f"IK reach {variant}{menu}", f"hierarchical {JAC}", label_of(f"ik_reach_{variant}{menu}"),
                                JAC, n, from_grid(e["grid"], e[f"{variant}_n{n}"]),
                                np.load(on)["theta"][:, 0, 0], np.load(off)["theta"][:, 0, 0]))
    return out


def controls():
    """arruda_direct has no projection in its path: the no-projection rerun must match exactly."""
    checks = []
    pairs = [(TOYS / "runs" / f"hierarchical_arruda_direct_n{n}.npz",
              TOYS / "runs" / f"hierarchical_{V}_arruda_direct_n{n}.npz", "g") for n in (4, 16, 64, 128)]
    pairs += [(ART / "ik_reach" / "runs" / "sym" / f"arruda_direct_n{n}.npz",
               ART / "ik_reach" / "runs" / f"sym_{V}" / f"arruda_direct_n{n}.npz", "theta") for n in (8, 32, 128)]
    for a, b, key in pairs:
        if a.exists() and b.exists():
            checks.append((str(b.relative_to(HERE.parent)), float(np.abs(np.load(a)[key] - np.load(b)[key]).max())))
    for mode in ("hierarchical", "oracle"):
        a = ART / "gm" / mode / "scaling" / "posteriors" / "draws.npz"
        b = ART / "gm" / mode / "scaling" / f"posteriors_{V}" / "draws.npz"
        if b.exists():
            za, zb = np.load(a), np.load(b)
            for k in zb.files:
                if k.endswith("arruda_direct") and k in za.files:
                    checks.append((f"gm {mode} {k}", float(np.abs(za[k] - zb[k]).max())))
    return checks


def pp_table():
    import torch
    import score_rerun as sr
    scale = sr.load_normalizers(sr.get_config("small"), sr.BenchmarkPaths.default())["sde_joint_theta"].scale
    arms = []
    for net, folder in (("baseline", "hierarchical"), ("Rg", "hierarchical_Rg")):
        for label, tag in (("with (stored)", "rerun"), (f"without ({V})", V)):
            files = sr.files(ART / "pp" / folder / "recovery" / tag / f"{JAC}-s100-t50")
            if files:
                arms.append((net, label, files))
    if not arms:
        return []
    ids = sorted(set.intersection(*(set(f) for *_, f in arms)))
    table = []
    for net, label, files in arms:
        g, l, r, per, secs = sr.score(files, ids, scale)
        collapsed = 0
        for i in ids:
            o = torch.load(files[i], map_location="cpu", weights_only=False)
            collapsed += float(o["posterior"]["globals"][:, 3].median()) < -4.5 and float(o["truth_globals"][3]) > -3.5
        table.append(dict(net=net, arm=label, datasets=len(ids), global_rmse=g, local_rmse=l, mean_r=r,
                          per_global_r=[float(v) for v in per], collapses=int(collapsed), hours=secs / 3600))
    return table


def main():
    global V
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="noproj", choices=("noproj", "nocurv", "noboth", "langevin"))
    V = parser.parse_args().variant
    diag = diagnostics()
    lines, records = [], []
    head = (f"{'experiment':22s} {'arm':34s} {'n':>4s}  {'P+ exact/with/without':>22s}  {'W1/sd with→without':>18s}  "
            f"{'sd/exact with→without':>21s}  neg-eig neg-curv active  lift  repair")
    lines.append(head)
    for experiment, arm, job, correction, n, ref, on, off in rows():
        w = lambda a: wasserstein_distance(a, ref) / ref.std()  # noqa: E731
        d = diag.get((job, correction, n), {})
        rec = dict(experiment=experiment, arm=arm, n=n, p_exact=float(np.mean(ref > 0)),
                   p_with=float(np.mean(on > 0)), p_without=float(np.mean(off > 0)),
                   w1_with=w(on), w1_without=w(off), sd_with=float(on.std() / ref.std()),
                   sd_without=float(off.std() / ref.std()), **{k: d.get(k) for k in ("neg", "curv", "active", "lift", "repair")})
        records.append(rec)
        fmt = lambda v: f"{v:6.2f}" if v is not None else "     -"  # noqa: E731
        lines.append(f"{experiment:22s} {arm:34s} {n:4d}  {rec['p_exact']:6.2f} {rec['p_with']:6.2f} {rec['p_without']:6.2f}"
                     f"     {rec['w1_with']:6.2f} → {rec['w1_without']:6.2f}     {rec['sd_with']:6.2f} → {rec['sd_without']:6.2f}"
                     f"   {fmt(rec['neg'])} {fmt(rec['curv'])} {fmt(rec['active'])} {fmt(rec['lift'])} {fmt(rec['repair'])}")
    lines.append("")
    lines.append("controls (arruda_direct, no projection in its path; must be 0):")
    for name, diff in controls():
        lines.append(f"  {name}: max |diff| {diff:.3g}")
    pp = pp_table()
    if pp:
        lines.append("")
        lines.append(f"partial pooling C2 ({pp[0]['datasets']} datasets)")
        for t in pp:
            d = diag.get((label_of("pp_" + ("hierarchical" if t["net"] == "baseline" else "hierarchical_Rg")), JAC, 100), {})
            extra = (f"  neg-eig {d['neg']:.2f} neg-curv {d.get('curv', 0):.3f} active {d['active']:.2f} "
                     f"lift {d['lift']:.2f} repair {d['repair']:.2f}" if d and t["arm"].startswith("without") else "")
            lines.append(f"  {t['net']:8s} {t['arm']:16s} global RMSE {t['global_rmse']:.3f}  local {t['local_rmse']:.3f}  "
                         f"mean r {t['mean_r']:.3f}  per-global r {' '.join(f'{v:.2f}' for v in t['per_global_r'])}  "
                         f"collapses {t['collapses']}  {t['hours']:.1f} h{extra}")
    text = "\n".join(lines)
    print(text)
    name = "comparison" if V == "noproj" else f"comparison_{V}"
    (OUT / f"{name}.txt").write_text(text + "\n")
    (OUT / f"{name}.json").write_text(json.dumps({"variant": V, "rows": records, "pp": pp}, indent=2) + "\n")
    figure(records)


def figure(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not records:
        return
    ink, muted, warm, cool = "#1f1f1e", "#6b6a63", "#eda100", "#2a78d6"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    w_on = np.array([r["w1_with"] for r in records])
    w_off = np.array([r["w1_without"] for r in records])
    p_on = np.array([abs(r["p_with"] - r["p_exact"]) for r in records])
    p_off = np.array([abs(r["p_without"] - r["p_exact"]) for r in records])
    groups = sorted({r["experiment"] for r in records})
    markers = dict(zip(groups, "osD^v<>phX*P8"))
    for ax, a, b, title in ((axes[0], w_on, w_off, "W1 to exact / exact sd"),
                            (axes[1], p_on, p_off, "|P(global>0) − exact|")):
        top = max(a.max(), b.max()) * 1.05
        ax.plot([0, top], [0, top], color=muted, lw=0.8, ls=":")
        for g in groups:
            idx = [i for i, r in enumerate(records) if r["experiment"] == g]
            ax.scatter(a[idx], b[idx], s=26, marker=markers[g], color=warm if "toy" in g else cool,
                       edgecolor=ink, linewidth=0.4, label=g)
        ax.set_xlabel("all safeguards on (as published)", color=ink)
        ax.set_ylabel(f"variant {V}", color=ink)
        ax.set_title(f"{title}  (below the diagonal: removing it helps)", fontsize=9, color=ink)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    path = ART / "figures" / ("16_projection.png" if V == "noproj" else f"16_{V}.png")
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    main()
