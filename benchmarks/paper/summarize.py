#!/usr/bin/env python3
"""Tables of the paper benchmark -> benchmarks/paper/RESULTS.md (and results/*.json).

    python summarize.py [--out RESULTS.md]

Every table is paired: the same networks, datasets and seeds as the final arm, one component
changed. Missing jobs show as "—". Metrics:
- exact-reference problems: W1 to the exact posterior of the global in exact sd, |P(g > 0) error|
  (bimodal problems), sd ratio.
- GM (roadmap 2): exact-MC W1/sd, branch-mass error, width, local W1.
- partial pooling: mean r of posterior median vs truth over the 7 globals, global RMSE (normalised),
  90% coverage, z_rms, local RMSE, collapses (log_sigma_nu median < -4.5 with truth > -3.5), s/dataset.
- fair benchmark: upstream's own metrics (fair.py score).
"""
import os

CPU = sorted(os.sched_getaffinity(0))[:3]
os.sched_setaffinity(0, CPU)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "3"

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jobs as J  # noqa: E402

TREE = J.TREE
ART = TREE / "experiments_arruda_schur" / "artifacts"
TOYS = TREE / "abs_hierarchy" / "artifacts"
JAC, GH, AD = J.JAC, J.GH, J.AD
RNG = np.random.default_rng(0)
VARIANTS = list(J.SAMPLER_VARIANTS)


def fmt(x, digits=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def from_grid(grid, density, count=200_000):
    cdf = np.concatenate([[0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    return np.interp(RNG.random(count), cdf / cdf[-1], grid)


# ---------------------------------------------------------------------------- exact-reference problems
def exact_cases():
    """(benchmark, arm, n, reference draws, {(variant, method): draws})."""
    cases = []

    def add(bench, arm, n, ref, draws):
        if draws:
            cases.append((bench, arm, n, ref, draws))

    toy_root = {"abs": TOYS, "lean": TOYS / "lean", "square": TOYS / "square", "curved": TOYS / "curved"}
    for model, mode, methods in J.TOY_CASES:
        runs = toy_root[model] / "runs"
        for n in (4, 16, 64, 128):
            ref = runs / f"mcmc_n{n}.npz"
            if not ref.exists():
                continue
            draws = {}
            for v in VARIANTS:
                for m in methods:
                    p = runs / f"{mode}_paper_{v}_{m}_n{n}.npz"
                    if p.exists():
                        draws[(v, m)] = np.load(p)["g"]
            add(f"toy {model}", mode, n, np.load(ref)["g"], draws)
    for mode in ("hierarchical", "oracle"):
        for v in VARIANTS:
            p = ART / "gm" / mode / "scaling" / f"posteriors_paper_{v}" / "draws.npz"
            if not p.exists():
                continue
            d = np.load(p)
            for regime in ("symmetric", "asymmetric"):
                for n in (8, 16, 32, 64):
                    ref_key = f"{regime}_n{n}_exact_mc"
                    if ref_key not in d.files:
                        continue
                    draws = {(v, m): d[f"{regime}_n{n}_{m}"] for m in (JAC, "arruda_gauss_direct", AD)
                             if f"{regime}_n{n}_{m}" in d.files}
                    key = (f"GM {regime}", mode, n)
                    hit = next((c for c in cases if c[:3] == key), None)
                    if hit:
                        hit[4].update(draws)
                    else:
                        add(*key, d[ref_key], draws)
    ik = []
    e = np.load(ART / "ik_reach" / "exact.npz")
    for name, variant, menu in (("reach_sym", "sym", ""), ("reach_shift", "shift", ""),
                                ("reach_sym_bernoulli", "sym", "_bernoulli")):
        ik.append((f"IK {name}", ART / "ik_reach" / "runs", f"{variant}{menu}",
                   lambda n, variant=variant: from_grid(e["grid"], e[f"{variant}_n{n}"])))
    for v in ("sym", "shift"):
        ik.append((f"IK reach_{v} (final recipe)", ART / "ik_reach" / "runs", v,
                   lambda n, v=v: from_grid(e["grid"], e[f"{v}_n{n}"])))
    f = np.load(ART / "ik_flat" / "exact.npz")
    for v in ("global", "fixed"):
        ik.append((f"IK flat_{v}", ART / "ik_flat" / "runs", v,
                   lambda n, v=v: from_grid(f[f"{v}_grid_n{n}"], f[f"{v}_n{n}"])))
    a = TREE / "Arruda" / "artifacts" / "ik" / "exact" / "absx1_a0.6.npz"
    if a.exists():
        a = np.load(a)
        ik.append(("IK absx1", ART / "ik_absx1" / "runs", "hierarchical", lambda n: from_grid(a["grid"], a[f"n{n}"])))
    for bench, runs, folder, reference in ik:
        recipe = "final recipe" in bench
        for n in (8, 32, 128):
            draws = {}
            labels = ({"final": "recipe_final", "netcov": "recipe_netcov"} if recipe
                      else {v: v for v in VARIANTS})
            for v, label in labels.items():
                for m in (JAC, AD):
                    p = runs / f"{folder}_paper_{label}" / f"{m}_n{n}.npz"
                    if p.exists():
                        draws[(v, m)] = np.load(p)["theta"][:, 0, 0]
            if draws:
                add(bench, "hierarchical attn", n, reference(n), draws)
    return cases


def exact_tables():
    cases = exact_cases()
    records = []
    for bench, arm, n, ref, draws in cases:
        sd = ref.std()
        p_ref = float(np.mean(ref > 0))
        for (v, m), x in draws.items():
            records.append(dict(benchmark=bench, arm=arm, n=n, variant=v, method=m,
                                w1=float(wasserstein_distance(x, ref) / sd), p_err=abs(float(np.mean(x > 0)) - p_ref),
                                sd_ratio=float(x.std() / sd), p_exact=p_ref))
    # table 1: the final method against the baselines (same networks)
    lines = ["### Final method vs baselines (W1 / exact sd; |P(g>0) error| in brackets)", "",
             "| benchmark | arm | n | P exact | final (JAC-FD r9) | gauss_hierarchical | arruda_direct | arruda_gauss_direct | netcov |",
             "|---|---|---|---|---|---|---|---|---|"]
    by = defaultdict(dict)
    for r in records:
        by[(r["benchmark"], r["arm"], r["n"])][(r["variant"], r["method"])] = r
    for key in sorted(by):
        row = by[key]
        def cell(v, m):
            r = row.get((v, m))
            return "—" if r is None else f"{r['w1']:.2f} ({r['p_err']:.2f})"
        p = next(iter(row.values()))["p_exact"]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {p:.2f} | {cell('final', JAC)} | {cell('final', GH)} | "
                     f"{cell('final', AD)} | {cell('final', 'arruda_gauss_direct')} | {cell('netcov', JAC)} |")
    # table 2: sampler variants of the JAC rule, W1 relative to final
    lines += ["", "### Sampler variants (JAC-FD rule): W1 / exact sd", "",
              "| benchmark | arm | n | " + " | ".join(VARIANTS) + " |", "|---|---|---|" + "---|" * len(VARIANTS)]
    ratios = defaultdict(list)
    for key in sorted(by):
        row = by[key]
        base = row.get(("final", JAC))
        cells = []
        for v in VARIANTS:
            r = row.get((v, JAC))
            cells.append("—" if r is None else f"{r['w1']:.2f}")
            if r is not None and base is not None and v != "final":
                ratios[v].append((r["w1"] + 0.02) / (base["w1"] + 0.02))
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(cells) + " |")
    lines += ["", "### Sampler variants, summary over every exact-reference case (JAC-FD rule)", "",
              "Ratio = (W1_variant + 0.02) / (W1_final + 0.02); the 0.02 keeps near-exact cases from dominating. "
              "Better / worse: ratio below 0.87 / above 1.15 (the ±15% band of the earlier safeguard comparisons).", "",
              "| variant | cases | geometric-mean ratio | worst ratio | better | same | worse |", "|---|---|---|---|---|---|---|"]
    for v in VARIANTS[1:]:
        rs = np.array(ratios.get(v, []))
        if len(rs) == 0:
            lines.append(f"| {v} | 0 | — | — | — | — | — |")
            continue
        lines.append(f"| {v} | {len(rs)} | {np.exp(np.log(rs).mean()):.2f} | {rs.max():.2f} | {(rs < 0.87).sum()} | "
                     f"{((rs >= 0.87) & (rs <= 1.15)).sum()} | {(rs > 1.15).sum()} |")
    return "\n".join(lines), records


# ---------------------------------------------------------------------------- GM (roadmap 2, exact MC)
GM_LABEL = {"gauss_jacobian_fd": "JAC-FD r9, Gaussian prior", "gauss_jacobian_mixture_prior_fd": "JAC-FD r9, mixture prior",
            "gauss_jacobian_netcov": "netcov, Gaussian prior", "gauss_jacobian_mixture_prior_netcov": "netcov, mixture prior",
            "gauss_hierarchical": "GH pilot, Gaussian prior", "gauss_hierarchical_mixture_prior": "GH pilot, mixture prior",
            "langevin_fnpse": "F-NPSE (Langevin)"}


def gm_table():
    rows = defaultdict(list)
    for d in sorted((J.OUT / "gm").glob("*_s*")):
        arm = d.name.rsplit("_s", 1)[0]
        for tag in sorted((d / "scaling").glob("*_paper_r*")):
            path = tag / "metrics.csv"
            if not path.exists():
                continue
            import csv
            refresh = tag.name.rsplit("_r", 1)[1]
            for r in csv.DictReader(open(path)):
                if r["method"] == "exact_mc":
                    continue
                method = r["method"] + (f" (r{refresh})" if "netcov" in r["method"] else "")
                if "netcov" not in r["method"] and refresh != "9":
                    continue
                rows[(arm, int(r["observations"]), r["regime"], method)].append(
                    (float(r["w1_over_sigma"]), abs(float(r["branch_mass_error"])), float(r["width_ratio"]),
                     float(r["local_w1_over_sigma"]), float(r["seconds"])))
    if not rows:
        return "_GM: no results yet._", {}
    lines = ["| arm | n | regime | method | W1/sd | branch err | width | local W1 | s | k |", "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for key in sorted(rows):
        v = np.array(rows[key])
        m = v.mean(0)
        base = key[3].split(" (")[0]
        label = GM_LABEL.get(base, base) + (key[3][len(base):] if key[3] != base else "")
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {label} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.2f} | {m[3]:.3f} | {m[4]:.0f} | {len(v)} |")
        out["|".join(map(str, key))] = m.tolist()
    return "\n".join(lines), out


# ---------------------------------------------------------------------------- partial pooling (COMPASS net)
PP = ART / "pp"


def pp_folder(variant, tag, method="gauss_jacobian_FD_r9", subjects=100, timesteps=50, prefix="hierarchical_Rg_wide1.5"):
    return PP / f"{prefix}_{variant}" / "recovery" / tag / f"{method}-s{subjects}-t{timesteps}"


def pp_metrics(folder, datasets):
    import torch
    files = [folder / f"dataset-{i:04d}.pt" for i in datasets]
    if not all(f.exists() for f in files):
        return None
    rs = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    g = [r["posterior"]["globals"].double().numpy() for r in rs]
    med = np.array([np.median(x, 0) for x in g])
    sd = np.array([x.std(0) for x in g])
    lo, hi = np.array([np.quantile(x, 0.05, 0) for x in g]), np.array([np.quantile(x, 0.95, 0) for x in g])
    truth = np.array([r["truth_globals"].double().numpy() for r in rs])
    scale = truth.std(0)
    e = (med - truth) / scale
    lmed = [np.median(r["posterior"]["locals"].double().numpy(), 0) for r in rs]
    ltruth = [r["truth_locals"].double().numpy() for r in rs]
    lscale = np.concatenate(ltruth).std(0)
    return dict(
        r=float(np.mean([np.corrcoef(med[:, k], truth[:, k])[0, 1] for k in range(7)])),
        per_r=[float(np.corrcoef(med[:, k], truth[:, k])[0, 1]) for k in range(7)],
        rmse=float(np.sqrt((e ** 2).mean())), cover=float(np.mean((truth >= lo) & (truth <= hi))),
        z=float(np.mean(np.sqrt((((med - truth) / sd) ** 2).mean(0)))),
        local=float(np.sqrt(np.mean([(((a - b) / lscale) ** 2).mean() for a, b in zip(lmed, ltruth)]))),
        collapses=int(np.sum((med[:, 3] < -4.5) & (truth[:, 3] > -3.5))),
        s=float(np.mean([r["runtime_seconds"] for r in rs])), datasets=len(rs))


def identity_row(variant, net):
    import torch
    diffs, secs, base_secs = [], [], []
    for i in range(5):
        a = pp_folder(net, "paper_final") / f"dataset-{i:04d}.pt"
        b = pp_folder(net, f"paper_{variant}") / f"dataset-{i:04d}.pt"
        if not (a.exists() and b.exists()):
            return f"| {variant} | — | — | — | — |"
        ra, rb = (torch.load(f, map_location="cpu", weights_only=False) for f in (a, b))
        ga, gb = ra["posterior"]["globals"].double().numpy(), rb["posterior"]["globals"].double().numpy()
        diffs.append(np.abs(ga - gb) / ga.std(0))
        secs.append(rb["runtime_seconds"]); base_secs.append(ra["runtime_seconds"])
    d = np.concatenate([x.ravel() for x in diffs])
    return f"| {variant} | {d.max():.2e} | {np.median(d):.2e} | {np.mean(secs):.0f} | {np.mean(base_secs):.0f} |"


def pp_tables():
    lines, out = [], {}
    head = "| arm | seeds | datasets | mean r | global RMSE | cover90 | z_rms | local RMSE | collapses | s/dataset |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"

    def row(label, entries):
        ms = [m for m in entries if m is not None]
        if not ms:
            return f"| {label} | 0 | — | — | — | — | — | — | — | — |"
        f = lambda k: float(np.mean([m[k] for m in ms]))  # noqa: E731
        out[label] = ms
        spread = f" ({min(m['r'] for m in ms):.3f}–{max(m['r'] for m in ms):.3f})" if len(ms) > 1 else ""
        return (f"| {label} | {len(ms)} | {ms[0]['datasets']} | {f('r'):.3f}{spread} | {f('rmse'):.3f} | {f('cover'):.2f} | "
                f"{f('z'):.2f} | {f('local'):.3f} | {sum(m['collapses'] for m in ms)} | {f('s'):.0f} |")

    finals = {7: "covout_cov_s7", 8: "covout_cov_s8", 9: "paper_final_s9"}
    d100, d30 = range(100), range(30)
    lines += ["### Final method, 3 training seeds x 100 datasets", "", head, sep,
              row("final: JAC-FD r9", [pp_metrics(pp_folder(v, "paper_final"), d100) for v in finals.values()]),
              row("final net, netcov r9", [pp_metrics(pp_folder(v, "paper_netcov_r9"), d100) for v in finals.values()])]
    lines += ["", "### Sampler ablations and baselines (seed-7 final net, datasets 0-29, paired)", "", head, sep,
              row("final: JAC-FD r9", [pp_metrics(pp_folder(finals[7], "paper_final"), d30)])]
    for v in VARIANTS[1:]:
        lines.append(row(v, [pp_metrics(pp_folder(finals[7], f"paper_{v}"), d30)]))
    for label, tag, method in (("netcov r9", "paper_netcov_r9", "gauss_jacobian_FD_r9"),
                               ("netcov r1", "paper_netcov_r1", "gauss_jacobian_FD_r1"),
                               ("gauss_hierarchical (pilot)", "paper_gh", GH),
                               ("arruda_direct (plain sum)", "paper_arruda", AD),
                               ("F-NPSE (Langevin)", "paper_fnpse", "langevin_fnpse"),
                               ("no churn", "paper_churn0", "gauss_jacobian_FD_r9"),
                               ("best-validation weights", "paper_bestval", "gauss_jacobian_FD_r9")):
        lines.append(row(label, [pp_metrics(pp_folder(finals[7], tag, method), d30)]))
    for steps in (25, 100):
        lines.append(row(f"{steps} steps", [pp_metrics(pp_folder(finals[7], f"paper_t{steps}", timesteps=steps), d30)]))
    lines += ["", "Same draws? fp64 eigh and the dense / Cholesky route against the final arm, datasets 0-4 "
              "(max |difference| over all global draws, in units of the posterior sd; seconds per dataset):", "",
              "| variant | max diff / sd | median diff / sd | s/dataset | final s/dataset |", "|---|---|---|---|---|"]
    lines += [identity_row(v, finals[7]) for v in ("fp64", "dense")]
    lines += ["", "seed-8 cross-check of the two PD choices closest to the default:", "", head, sep,
              row("final (seed 8)", [pp_metrics(pp_folder(finals[8], "paper_final"), d30)])]
    for v in ("consistent", "gate"):
        lines.append(row(f"{v} (seed 8)", [pp_metrics(pp_folder(finals[8], f"paper_{v}"), d30)]))
    lines += ["", "### Number of subjects (seed-7 final net, datasets 0-29)", "", head, sep]
    for n in (10, 25, 100):
        tag = "paper_final" if n == 100 else f"paper_n{n}"
        lines.append(row(f"n = {n}", [pp_metrics(pp_folder(finals[7], tag, subjects=n), d30)]))
    lines += ["", "### Training ablations (one component of the final recipe changed; datasets 0-29)", "", head, sep,
              row("final recipe", [pp_metrics(pp_folder(finals[s], "paper_final"), d30) for s in (7, 8)])]
    for arm, prefix in (("nocov", "hierarchical_Rg_wide1.5"), ("noedm", "hierarchical_Rg_wide1.5"),
                        ("unseeded", "hierarchical_Rg_wide1.5"), ("tsm", "hierarchical_Rg_wide1.5"),
                        ("norg", "hierarchical_wide1.5"), ("nowide", "hierarchical_Rg"), ("full", "full_Rg_wide1.5")):
        entries = [pp_metrics(pp_folder(f"paper_{arm}_s{s}", "paper_final", prefix=prefix), d30) for s in (7, 8)]
        lines.append(row(arm, entries))
    epochs = []
    for d in sorted(PP.glob("*paper_*_s*")) + [PP / f"hierarchical_Rg_wide1.5_{v}" for v in ("covout_cov_s7", "covout_cov_s8")]:
        for f in (d / "checkpoints").glob("*.json"):
            try:
                j = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if "minutes" in j:
                vl = j["validation_loss"]
                epochs.append(f"| {d.name} | {len(j['train_loss'])} | {int(np.argmin(vl))} | {j['minutes']:.0f} |")
    if epochs:
        lines += ["", "Training length (epochs run, best-validation epoch, minutes):", "",
                  "| network | epochs | best epoch | minutes |", "|---|---|---|---|", *epochs]
    return "\n".join(lines), out


# ---------------------------------------------------------------------------- fair benchmark (BayesFlow nets)
def fair_table():
    env = {**os.environ, "KERAS_BACKEND": "torch", "FAIR_POSTERIORS": str(J.OUT / "fair" / "posteriors"),
           "CUDA_VISIBLE_DEVICES": "-1", "FAIR_DEVICE": "cpu", "PAPER_GPU": "",
           "PYTHONPATH": ":".join([str(TREE / "Arruda" / ".vendor"), str(TREE), str(TREE / "src")])}
    result = subprocess.run([J.PY, "fair.py", "score"], cwd=TREE / "Arruda" / "partial_pooling_fair", env=env,
                            capture_output=True, text=True)
    text = "\n".join(l for l in result.stdout.splitlines() if l.strip() and "warn" not in l.lower())
    if result.returncode != 0:
        text += "\n" + result.stderr[-2000:]
    return "```\n" + text + "\n```"


# ---------------------------------------------------------------------------- Gaussian toy, AR(1)
def gausstoy_table():
    path = J.OUT / "toy" / "cov" / "results.jsonl"
    if not path.exists():
        return "_Gaussian toy: no results yet._"
    rows = defaultdict(list)
    for r in map(json.loads, path.read_text().splitlines()):
        rows[(r["J"], r["method"])].append(r)
    lines = ["| J | method | datasets | KL | mean err / sd | width ratio | s/dataset |", "|---|---|---|---|---|---|---|"]
    for (Jn, m), rs in sorted(rows.items()):
        f = lambda k: float(np.mean([r[k] for r in rs if k in r]))  # noqa: E731
        lines.append(f"| {Jn} | {m} | {len(rs)} | {f('kl'):.3g} | {f('mean_error_sd'):.3f} | {f('width_ratio'):.3f} | {f('seconds'):.0f} |")
    return "\n".join(lines)


def ar1_table():
    path = TREE / "Arruda" / "paper_benchmarks" / "artifacts" / "ar1_grid" / "results" / "n10000" / "TABLE.md"
    if not path.exists():
        return "_AR(1): no results yet._"
    keep = [l for l in path.read_text().splitlines()
            if l.startswith("| grid") or l.startswith("|---") or any(f"| {m} |" in l for m in (
                "reference", "paper_final", "gauss_hierarchical", "gauss_jacobian_FD_r9", "langevin_fnpe", "bayesflow_arruda"))]
    return "\n".join(keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "RESULTS.md"))
    args = parser.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    exact_text, exact_records = exact_tables()
    gm_text, gm_records = gm_table()
    pp_text, pp_records = pp_tables()
    sections = [
        "# Paper benchmark: results", "",
        f"Benchmarked commit: `{(J.OUT / 'COMMIT').read_text().strip() if (J.OUT / 'COMMIT').exists() else '?'}`. "
        "Written by `summarize.py`; design in `README.md`.", "",
        "## 1. Partial pooling on BayesFlow's own network (fair benchmark)", "",
        "`bf_*` = upstream's `compositional_sample` on its global net (+ its local net); `paper-*` = COMPASS on "
        "the joint net with the same architecture and simulations. 100 test datasets.", "", fair_table(), "",
        "## 2. Partial pooling, COMPASS network", "", pp_text, "",
        "## 3. Exact-reference problems (toys, IK, latent-attention GM)", "", exact_text, "",
        "## 4. GM with the covariance-output networks (exact MC; 2 seeds x 2 replicates)", "", gm_text, "",
        "## 5. Gaussian toy (Arruda Exp. 1, analytic posterior; covariance-output net)", "", gausstoy_table(), "",
        "## 6. AR(1) grid (Arruda Exp. 2; `paper_final` = this commit, the rest archived)", "", ar1_table(), "",
    ]
    Path(args.out).write_text("\n".join(sections) + "\n")
    (HERE / "results" / "exact.json").write_text(json.dumps(exact_records))
    (HERE / "results" / "gm.json").write_text(json.dumps(gm_records))
    (HERE / "results" / "pp.json").write_text(json.dumps(pp_records))
    print("\n".join(sections))


if __name__ == "__main__":
    main()
