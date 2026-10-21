"""Tables (one per question) and figures from ``<results>/results/*.json``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "artifacts" / "reference"       # set by benchmark.py for other regimes

# Validated reference categorical palette (dataviz skill), fixed order, + markers.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

QUESTIONS = [
    ("Q1 local posterior p(θ_i | x_1:N)", [
        ("local_c2st", "local C2ST (0.5 best)"), ("local_w1", "local W1/sd"),
        ("branch_err", "branch-mass err"), ("crescent_dev", "crescent |r-0.1|/0.01 (exact 0.67)"),
        ("off_support", "off-support frac"), ("local_mean_err", "local |mean err|/sd"),
        ("local_width", "local width ratio")]),
    ("Q2 population mean μ_g", [
        ("mu_mean_err", "|mean err|/sd"), ("mu_width", "width ratio"), ("mu_w1", "W1/sd"),
        ("mu_cover90", "90% coverage")]),
    ("Q3 population spread σ_g", [
        ("sigma_mean_err", "|mean err|/sd"), ("sigma_width", "width ratio"),
        ("sigma_w1", "W1/sd"), ("sigma_cover90", "90% coverage"),
        ("global_c2st", "joint global C2ST")]),
    ("Q2+Q3 globals in mirror-adapted coordinates", [
        ("mode_balance", "P(u>0) (exact 0.5)"), ("mode_balance_err", "|P(u>0) - ref|"),
        ("sym_u_w1", "u=(μ0+μ1)/√2 W1/sd (bimodal)"), ("sym_w_w1", "w=σ0-σ1 W1/sd (bimodal)"),
        ("sym_v_w1", "v=(μ1-μ0)/√2 W1/sd"), ("sym_s_w1", "s=σ0+σ1 W1/sd"),
        ("sym_inv_w1", "mirror-invariant W1/sd")]),
    ("Q4 global–local dependence", [
        ("dep_corr_err", "|Δ corr(g, θ_i)|"), ("dep_corr_ref", "ref |corr| (signal)"),
        ("joint_c2st", "joint (g, θ_i) C2ST"),
        ("joint_c2st_indep", "same, dependence shuffled out")]),
    ("Cost", [("seconds", "s / dataset"), ("nonfinite", "non-finite frac")]),
]

FAMILIES = [
    ("production rules (legacy net)", ["uncorrected", "gauss", "gauss_hierarchical",
                                       "gauss_jacobian", "gauss_jacobian_FD", "langevin_fnpe"]),
    ("latent attention nets", ["attn_hier:arruda_direct", "attn_hier:arruda_gauss_direct",
                               "attn_hier:gauss_jacobian_FD", "attn_hier:gauss_jacobian_FD_r5_churn4",
                               "attn_full:gauss_hierarchical",
                               "attn_full:gauss_jacobian_FD", "attn_full:arruda_schur",
                               "attn_full:arruda_laplace"]),
    ("BayesFlow (Arruda et al.)", ["bayesflow_arruda", "bayesflow_arruda_diffprior"]),
    ("wider training prior (1.5x) vs same rule, standard prior",
     ["attn_hier_w1.5:gauss_jacobian_FD", "legacy_w1.5:gauss_jacobian_FD", "legacy_w1.5:gauss_hierarchical",
      "attn_hier:gauss_jacobian_FD", "gauss_jacobian_FD", "gauss_hierarchical"]),
]
ORDER = list(dict.fromkeys(["reference"] + [m for _, ms in FAMILIES for m in ms]))


def load(results):
    summaries = {}
    for path in sorted((results / "results").glob("*.json")):
        s = json.loads(path.read_text())["summary"]
        summaries[(s["method"], s["N"])] = s
    return summaries


def tables(summaries):
    methods = sorted({m for m, _ in summaries}, key=lambda m: ORDER.index(m) if m in ORDER else 99)
    ns = sorted({n for _, n in summaries})
    lines = ["# Hierarchical Two Moons — results", "",
             "Medians over datasets (coverage: mean). `reference` = 500 independent "
             "reference draws scored the same way: the Monte-Carlo floor.", ""]
    for title, keys in QUESTIONS:
        lines += [f"## {title}", ""]
        for key, label in keys:
            lines += [f"**{label}** (`{key}`)", "",
                      "| method | " + " | ".join(f"N={n}" for n in ns) + " |",
                      "|---|" + "---|" * len(ns)]
            for m in methods:
                cells = []
                for n in ns:
                    s = summaries.get((m, n))
                    v = None if s is None else s.get(key)
                    cells.append("–" if v is None else f"{v:.3g}")
                if any(c != "–" for c in cells):
                    lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines.append("")
    tuned = [(m, n, s["tuned"]) for (m, n), s in summaries.items() if "tuned" in s]
    if tuned:
        lines += ["## BayesFlow tuned (d1, shift)", "", "| method | N | d1 | shift |", "|---|---|---|---|"]
        for m, n, t in sorted(tuned, key=lambda r: (r[0], r[1])):
            lines.append(f"| {m} | {n} | {t['d1']:g} | {t['shift']:g} |")
    failed = [(m, n, s["failed"]) for (m, n), s in summaries.items() if s.get("failed")]
    if failed:
        lines += ["", "## Failed datasets", ""] + [f"- {m} N={n}: {f}" for m, n, f in failed]
    return "\n".join(lines) + "\n"


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def scaling_figures(summaries, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.ticker  # noqa: F401
    import matplotlib.pyplot as plt

    ns = sorted({n for _, n in summaries})
    for title, keys in QUESTIONS:
        fig, axes = plt.subplots(len(keys), len(FAMILIES), figsize=(4.2 * len(FAMILIES), 2.3 * len(keys)),
                                 squeeze=False, sharex=True)
        for r, (key, label) in enumerate(keys):
            for c, (family, methods) in enumerate(FAMILIES):
                ax = axes[r, c]
                _style(ax)
                ref = [summaries.get(("reference", n), {}).get(key) for n in ns]
                if any(v is not None for v in ref):
                    ax.plot([n for n, v in zip(ns, ref) if v is not None], [v for v in ref if v is not None],
                            color=MUTED, linestyle="--", linewidth=1.5, marker="o", markersize=3,
                            label="reference (MC floor)")
                for k, m in enumerate(methods):
                    pts = [(n, summaries[(m, n)].get(key)) for n in ns
                           if (m, n) in summaries and summaries[(m, n)].get(key) is not None]
                    if pts:
                        ax.plot(*zip(*pts), color=PALETTE[k], marker=MARKERS[k], markersize=4,
                                linewidth=2, label=m.split(":")[-1] if ":" not in m else m)
                ax.set_xscale("log")
                if key in ("seconds",):
                    ax.set_yscale("log")
                if r == 0:
                    ax.set_title(family, fontsize=9, color=INK)
                if c == 0:
                    ax.set_ylabel(label, fontsize=8, color=INK)
                if r == len(keys) - 1:
                    ax.set_xlabel("N (groups)", fontsize=8, color=INK)
                    ax.set_xticks(ns, [str(n) for n in ns])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        for c in range(len(FAMILIES)):
            handles, labels = axes[0, c].get_legend_handles_labels()
            if handles:
                axes[0, c].legend(handles, labels, fontsize=6, frameon=False, loc="best")
        fig.suptitle(title, fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        name = title.split()[0].lower().replace("+", "")
        fig.savefig(folder / f"scaling_{name}.png", dpi=130, facecolor="white")
        plt.close(fig)


GLOBAL_NAMES = [("μ_0", (-1.0, 1.0)), ("μ_1", (-1.0, 1.0)), ("σ_0", (0.1, 3.0)), ("σ_1", (0.1, 3.0))]
FAMILY_KEYS = ["production", "attention", "bayesflow", "wide"]


def _draws(results, method, n):
    path = results / "draws" / f"{method.replace(':', '__')}_N{n}_dataset0.npz"
    return np.load(path) if path.exists() else None


def global_marginal_figures(results, folder, ns=None):
    """Marginal posteriors of mu_0, mu_1, sigma_0, sigma_1: rows = methods, cols = parameters.

    One figure per N and method family (dataset 0). Grey fill = reference (2000 draws),
    coloured line = method (500 draws), dashed = truth.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_dir = REFERENCE
    ns = ns or sorted(int(p.stem.split("_")[0][1:]) for p in ref_dir.glob("N*_0.npz"))
    for n in ns:
        ref_path = ref_dir / f"N{n}_0.npz"
        if not ref_path.exists():
            continue
        ref = np.load(ref_path, allow_pickle=True)
        ref_g, true_g = ref["ref_g"][500:], ref["true_g"]
        for key, (family, members) in zip(FAMILY_KEYS, FAMILIES):
            methods = [m for m in ["reference", *members] if _draws(results, m, n) is not None]
            if len(methods) <= 1:
                continue
            fig, axes = plt.subplots(len(methods), 4, figsize=(12, 1.55 * len(methods) + 0.5),
                                     squeeze=False, sharex="col")
            for r, m in enumerate(methods):
                g = _draws(results, m, n)["g"]
                g = g[np.isfinite(g).all(1)]
                colour = MUTED if m == "reference" else PALETTE[(r - 1) % len(PALETTE)]
                for c, (name, (lo, hi)) in enumerate(GLOBAL_NAMES):
                    ax = axes[r, c]
                    _style(ax)
                    bins = np.linspace(lo, hi, 26)
                    ax.hist(ref_g[:, c], bins=bins, density=True, color="#d6d5d0",
                            edgecolor="none", label="reference")
                    ax.hist(np.clip(g[:, c], lo, hi), bins=bins, density=True, histtype="step",
                            color=colour, linewidth=1.8, label=m)
                    ax.axvline(true_g[c], color=INK, linestyle="--", linewidth=1.2)
                    ax.set_xlim(lo, hi)
                    ax.set_yticks([])
                    ax.grid(False, axis="y")
                    if r == 0:
                        ax.set_title(f"{name}   (truth {true_g[c]:.2f})", fontsize=10, color=INK)
                    if r == len(methods) - 1:
                        ax.set_xlabel(name, fontsize=9, color=INK)
                        ax.tick_params(labelsize=8)
                axes[r, 0].set_ylabel(m.replace(":", ":\n"), fontsize=8, color=INK, rotation=0,
                                      ha="right", va="center", labelpad=8)
            fig.suptitle(f"N={n}, dataset 0 — global marginal posteriors, {family}\n"
                         "grey = reference (2000 draws), line = method (500 draws), dashed = truth; "
                         "draws outside the support are piled at the edge",
                         fontsize=10, color=INK)
            fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
            fig.savefig(folder / f"globals_marginals_N{n}_{key}.png", dpi=130, facecolor="white")
            plt.close(fig)


SYM_NAMES = [("u = (μ_0+μ_1)/√2", -1.0), ("v = (μ_1-μ_0)/√2", 1.0),
             ("w = σ_0-σ_1", -1.0), ("s = σ_0+σ_1", 1.0)]


def _sym(g):
    import task as T
    return T.symmetric_globals(g)


def mirror_marginal_figures(results, folder, ns=None, datasets=(0, 1, 2)):
    """Marginals in mirror-adapted coordinates: u and w flip sign under the exact symmetry
    g -> (-mu_1, -mu_0, sigma_1, sigma_0), so the two global modes appear as two peaks there;
    v and s are invariant. Rows = methods, cols = u, v, w, s; one figure per N, family, dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = folder / "bimodality"
    out.mkdir(exist_ok=True)
    ns = ns or sorted({int(p.stem.split("_")[0][1:]) for p in REFERENCE.glob("N*_0.npz")})
    for n in ns:
        for ds in datasets:
            ref_path = REFERENCE / f"N{n}_{ds}.npz"
            if not ref_path.exists():
                continue
            ref = np.load(ref_path, allow_pickle=True)
            ref_s, true_s = _sym(ref["ref_g"][500:]), _sym(ref["true_g"])
            lo = np.quantile(ref_s, 0.002, axis=0)
            hi = np.quantile(ref_s, 0.998, axis=0)
            pad = 0.15 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            for c in (0, 2):                       # symmetric axes for the bimodal coordinates
                m = max(abs(lo[c]), abs(hi[c]))
                lo[c], hi[c] = -m, m
            for key, (family, members) in zip(FAMILY_KEYS, FAMILIES):
                methods = [m for m in ["reference", *members]
                           if (results / "draws" / f"{m.replace(':', '__')}_N{n}_dataset{ds}.npz").exists()]
                if len(methods) <= 1:
                    continue
                fig, axes = plt.subplots(len(methods), 4, figsize=(12, 1.45 * len(methods) + 0.7),
                                         squeeze=False, sharex="col")
                for r, m in enumerate(methods):
                    g = np.load(results / "draws" / f"{m.replace(':', '__')}_N{n}_dataset{ds}.npz")["g"]
                    g = _sym(g[np.isfinite(g).all(1)])
                    colour = MUTED if m == "reference" else PALETTE[(r - 1) % len(PALETTE)]
                    for c, (name, sign) in enumerate(SYM_NAMES):
                        ax = axes[r, c]
                        _style(ax)
                        bins = np.linspace(lo[c], hi[c], 31)
                        ax.hist(ref_s[:, c], bins=bins, density=True, color="#d6d5d0", edgecolor="none")
                        ax.hist(np.clip(g[:, c], lo[c], hi[c]), bins=bins, density=True,
                                histtype="step", color=colour, linewidth=1.8)
                        ax.axvline(true_s[c], color=INK, linestyle="--", linewidth=1.2)
                        if sign < 0:
                            ax.axvline(-true_s[c], color="#e34948", linestyle=":", linewidth=1.4)
                        ax.set_xlim(lo[c], hi[c])
                        ax.set_yticks([])
                        ax.grid(False, axis="y")
                        if r == 0:
                            ax.set_title(name + ("  (bimodal)" if sign < 0 else "  (invariant)"),
                                         fontsize=10, color=INK)
                        if r == len(methods) - 1:
                            ax.tick_params(labelsize=8)
                    bal = np.mean(g[:, 0] > 0)
                    axes[r, 0].text(0.02, 0.92, f"P(u>0)={bal:.2f}", transform=axes[r, 0].transAxes,
                                    fontsize=7, color=MUTED, va="top")
                    axes[r, 0].set_ylabel(m.replace(":", ":\n"), fontsize=8, color=INK, rotation=0,
                                          ha="right", va="center", labelpad=8)
                t = ref["true_g"]
                fig.suptitle(f"N={n}, dataset {ds}, truth μ=({t[0]:.2f},{t[1]:.2f}) σ=({t[2]:.2f},{t[3]:.2f}) — {family}\n"
                             "grey = reference, line = method, black dashed = truth, red dotted = its mirror image",
                             fontsize=10, color=INK)
                fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
                fig.savefig(out / f"mirror_N{n}_ds{ds}_{key}.png", dpi=120, facecolor="white")
                plt.close(fig)


def joint_global_figures(results, folder, ns=None, datasets=(0, 1, 2)):
    """(mu_0, mu_1) and (sigma_0, sigma_1) scatter per method vs the reference, truth x and
    mirror o; the two global modes are the two blobs either side of the dotted symmetry line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = folder / "bimodality"
    out.mkdir(exist_ok=True)
    ns = ns or sorted({int(p.stem.split("_")[0][1:]) for p in REFERENCE.glob("N*_0.npz")})
    for n in ns:
        for ds in datasets:
            ref_path = REFERENCE / f"N{n}_{ds}.npz"
            if not ref_path.exists():
                continue
            ref = np.load(ref_path, allow_pickle=True)
            rg, t = ref["ref_g"][500:1500], ref["true_g"]
            methods = [m for m in ORDER
                       if (results / "draws" / f"{m.replace(':', '__')}_N{n}_dataset{ds}.npz").exists()]
            if not methods:
                continue
            per_row = 3
            rows = int(np.ceil(len(methods) / per_row))
            fig, axes = plt.subplots(rows, 2 * per_row, figsize=(15, 2.6 * rows + 0.6), squeeze=False)
            for ax in axes.ravel():
                ax.set_visible(False)
            slo = max(0.0, min(rg[:, 2:].min(), t[2:].min()) - 0.1)
            shi = min(3.0, max(np.quantile(rg[:, 2:], 0.995), t[2:].max()) + 0.15)
            for k, m in enumerate(methods):
                g = np.load(results / "draws" / f"{m.replace(':', '__')}_N{n}_dataset{ds}.npz")["g"]
                colour = MUTED if m == "reference" else PALETTE[(ORDER.index(m) - 1) % len(PALETTE)]
                for j, (a, b, lim, mirror_pt, diag) in enumerate((
                        (0, 1, (-1, 1), (-t[1], -t[0]), ([-1, 1], [1, -1])),
                        (2, 3, (slo, shi), (t[3], t[2]), ([slo, shi], [slo, shi])))):
                    ax = axes[k // per_row, 2 * (k % per_row) + j]
                    ax.set_visible(True)
                    _style(ax)
                    ax.plot(*diag, color=MUTED, linestyle=":", linewidth=0.8)
                    ax.scatter(rg[:, a], rg[:, b], s=2, color="#c4c3bd", linewidths=0)
                    ax.scatter(g[:, a], g[:, b], s=2, color=colour, linewidths=0, alpha=0.6)
                    ax.plot(t[a], t[b], "x", color=INK, ms=9, mew=2)
                    ax.plot(*mirror_pt, "o", mfc="none", mec="#e34948", ms=9, mew=1.6)
                    ax.set_xlim(*lim)
                    ax.set_ylim(*lim)
                    ax.set_aspect("equal")
                    ax.set_xlabel("μ_0" if j == 0 else "σ_0", fontsize=8)
                    ax.set_ylabel("μ_1" if j == 0 else "σ_1", fontsize=8)
                    if j == 0:
                        ax.set_title(m, fontsize=8, color=INK, loc="left")
            fig.suptitle(f"N={n}, dataset {ds} — joint globals; grey = reference, × = truth, "
                         "○ = mirror (-μ_1,-μ_0,σ_1,σ_0), dotted = symmetry line", fontsize=10, color=INK)
            fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
            fig.savefig(out / f"joint_N{n}_ds{ds}.png", dpi=110, facecolor="white")
            plt.close(fig)


def reference_overview(folder):
    """u and w marginals of the reference for every dataset and N: where the bimodality is."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted({int(p.stem.split("_")[0][1:]) for p in REFERENCE.glob("N*_0.npz")})
    if not ns:
        return
    nd = 10
    fig, axes = plt.subplots(2 * len(ns), nd, figsize=(1.6 * nd, 1.1 * 2 * len(ns) + 0.6), squeeze=False)
    for r, n in enumerate(ns):
        for ds in range(nd):
            path = REFERENCE / f"N{n}_{ds}.npz"
            for j, c in enumerate((0, 2)):
                ax = axes[2 * r + j, ds]
                _style(ax)
                ax.set_yticks([])
                if not path.exists():
                    continue
                d = np.load(path, allow_pickle=True)
                v, t = _sym(d["ref_g"])[:, c], _sym(d["true_g"])[c]
                m = max(np.abs(v).max(), abs(t)) * 1.05
                ax.hist(v, bins=np.linspace(-m, m, 31), color=PALETTE[j], alpha=0.8)
                ax.axvline(t, color=INK, linestyle="--", linewidth=1)
                ax.axvline(-t, color="#e34948", linestyle=":", linewidth=1)
                ax.tick_params(labelsize=6)
                if ds == 0:
                    ax.set_ylabel(f"N={n}\n{'u' if c == 0 else 'w'}", fontsize=8, rotation=0,
                                  ha="right", va="center")
                if r == 0 and j == 0:
                    ax.set_title(f"dataset {ds}", fontsize=8)
    fig.suptitle("Reference posterior of u=(μ_0+μ_1)/√2 (blue) and w=σ_0-σ_1 (orange); "
                 "black = truth, red = mirror", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    fig.savefig(folder / "bimodality" / "reference_overview.png", dpi=110, facecolor="white")
    plt.close(fig)


def posterior_figures(results, folder, ns=(10, 50)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_dir = REFERENCE
    for n in ns:
        ref_path = ref_dir / f"N{n}_0.npz"
        if not ref_path.exists():
            continue
        ref = np.load(ref_path, allow_pickle=True)
        methods = [m for m in ORDER if _draws(results, m, n) is not None]
        if not methods:
            continue
        cols = 4
        rows = int(np.ceil(len(methods) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for k, m in enumerate(methods):
            d = _draws(results, m, n)
            ax = axes.ravel()[k]
            ax.set_visible(True)
            _style(ax)
            for i in range(min(3, n)):
                ax.scatter(*ref["ref_l"][500:1500, i].T, s=2, color="#b9b8b2", linewidths=0)
                ax.scatter(*d["l"][:, i].T, s=2, color=PALETTE[i], linewidths=0, alpha=0.7)
                ax.scatter(*ref["true_l"][i], s=40, marker="x", color=INK)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("θ_i[0]", fontsize=7)
            ax.set_ylabel("θ_i[1]", fontsize=7)
            ax.set_title(m, fontsize=8, color=INK)
        fig.suptitle(f"N={n}, dataset 0 — locals θ_1..3 (colour) vs reference (grey); × = truth",
                     fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(folder / f"posterior_locals_N{n}.png", dpi=130, facecolor="white")
        plt.close(fig)


def main(results=None):
    results = Path(results or HERE / "artifacts")
    summaries = load(results)
    if not summaries:
        print(f"no results in {results / 'results'}")
        return
    text = tables(summaries)
    (results / "TABLES.md").write_text(text)
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scaling_figures(summaries, figures)
    posterior_figures(results, figures)
    global_marginal_figures(results, figures)
    mirror_marginal_figures(results, figures)
    joint_global_figures(results, figures)
    reference_overview(figures)
    print(f"wrote {results / 'TABLES.md'} and {figures}/")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
