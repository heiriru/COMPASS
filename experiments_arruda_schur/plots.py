"""Figures for everything checked so far (CPU only; reads logs, reruns cheap oracle checks)."""
import glob
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "3"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = HERE / "artifacts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
LOGS = HERE / "logs"
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e2da"
# Validated categorical palette (dataviz reference), fixed order; colour follows the rule.
RULE_COLOURS = {
    "gauss_jacobian_FD": "#2a78d6", "arruda_direct": "#eb6834", "arruda_schur": "#1baf7a",
    "arruda_laplace": "#eda100", "arruda_gauss_direct": "#e87ba4", "arruda_gauss_laplace": "#4a3aa7",
    "gauss_jacobian": "#008300",
}
RULE_ORDER = list(RULE_COLOURS)
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "text.color": INK, "legend.frameon": False, "figure.dpi": 150,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# 1. Latent-attention Jacobian structure -------------------------------------------------
def fig_attention():
    from compass import ScoreBasedInferenceModel as SBIm, hierarchical_roles
    from compass.experimental.latent_attention import enable_latent_attention
    G, L, X = 3, 2, 3
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, mode in zip(axes, ("none", "global", "hierarchical", "full")):
        torch.manual_seed(0)
        m = SBIm(nodes_size=G + L + X, arch="tokenized", hidden_size=16, depth=2, num_heads=2, mlp_ratio=2)
        for mod in m.model.modules():
            if hasattr(mod, "adaLN_modulation"):
                torch.nn.init.normal_(mod.adaLN_modulation[-1].weight, std=0.3)
                torch.nn.init.normal_(mod.adaLN_modulation[-1].bias, std=0.3)
        enable_latent_attention(m, mode, hierarchical_roles(m.schema, G, L))
        c = torch.zeros(1, G + L + X)
        c[0, G + L:] = 1
        x = torch.randn(1, G + L + X)
        J = torch.autograd.functional.jacobian(
            lambda z: m.model(x=z, t=torch.tensor([0.4]), c=c)[0], x)[:, 0, :]
        block = J[:G + L, :G + L].abs().detach().numpy()
        block = block / block.max()
        ax.imshow(block > 1e-7, cmap=matplotlib.colors.ListedColormap(["#f3f2ee", "#2a78d6"]), vmin=0, vmax=1)
        names = [f"g{i+1}" for i in range(G)] + [f"l{i+1}" for i in range(L)]
        ax.set_xticks(range(G + L), names)
        ax.set_yticks(range(G + L), names)
        ax.axhline(G - 0.5, color=MUTED, lw=0.8)
        ax.axvline(G - 0.5, color=MUTED, lw=0.8)
        ax.grid(False)
        ax.set_title(mode, color=INK)
        ax.set_xlabel("input (noisy latent)")
    axes[0].set_ylabel("output score")
    fig.suptitle("Which latent outputs depend on which latent inputs, mask p(g,l|x)  "
                 "(blue = nonzero Jacobian entry)", fontsize=10)
    save(fig, "1_attention_modes_jacobian.png")


# 2. Scale-parameter marginal score: Schur vs Laplace vs Fisher ------------------------------
def fig_scale():
    sys.path.insert(0, str(common.COMPASS / "tests"))
    from test_experimental_marginal_schur import (ScaleModel, configured, scale_marginal_score)
    from compass import ScoreBasedInferenceModel as SBIm
    mu, s, obs, v = 0.2, math.log(0.7), 1.1, 0.3
    ls = np.linspace(-1.5, 2.5, 81)
    rows = {}
    for est in ("direct", "schur", "laplace"):
        model = SBIm(nodes_size=4, sde_type="vesde", sigma=25.0, hidden_size=8, depth=1, num_heads=1, mlp_ratio=1)
        model.model = ScaleModel(model.sde, v)
        sm = configured(model, "arruda_schur", est, 2, 1)
        t = torch.tensor([1e-3])
        var = float(model.sde.marginal_prob_std(t)) ** 2
        state = torch.zeros(1, len(ls), 4, dtype=torch.float64)
        state[..., 0], state[..., 1], state[..., 3] = mu, s, obs
        state[..., 2] = torch.as_tensor(ls)
        mask = torch.zeros(1, len(ls), 4, dtype=torch.float64)
        mask[..., 3] = 1
        scores = sm._raw_row_scores(state, t, mask)
        out, _ = sm._marginal_scores(scores.clone(), state, t, mask, var)
        rows[est] = out[0, :, 1].numpy()
    exact = float(scale_marginal_score(torch.tensor(mu), torch.tensor(s), torch.tensor(obs), v)[1])
    prec = math.exp(-2 * s) + 1 / v
    mode = (mu * math.exp(-2 * s) + obs / v) / prec
    sd = 1 / math.sqrt(prec)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(mode - sd, mode + sd, color="#f3f2ee", zorder=0)
    ax.axhline(exact, color=INK, lw=1.2, ls="--", label="exact marginal score")
    for est, lab in (("direct", "direct: S_g at carried l (Fisher, 1 sample)"),
                     ("schur", "schur: S_g + A H⁻¹ S_l"),
                     ("laplace", "laplace: Newton to mode − ½∇log det H")):
        c = {"direct": RULE_COLOURS["arruda_direct"], "schur": RULE_COLOURS["arruda_schur"],
             "laplace": RULE_COLOURS["arruda_laplace"]}[est]
        ax.plot(ls, rows[est], color=c, lw=2, label=lab)
    ax.annotate(f"missing log-det term\nv/(σ²+v) = {math.exp(-2*s)/prec:.3f}",
                xy=(mode, exact - math.exp(-2 * s) / prec), xytext=(mode + 0.6, exact - 0.9),
                arrowprops=dict(arrowstyle="->", color=MUTED), color=MUTED, fontsize=8)
    ax.set_xlabel("carried local value l  (shaded: conditional mode ± 1 sd)")
    ax.set_ylabel("∂/∂log σ  of  log p(μ, log σ | x)")
    ax.set_title("Scale parameter: only Laplace recovers the marginal score, from any l", fontsize=10)
    ax.set_ylim(exact - 2.5, exact + 2.5)
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "2_scale_parameter_marginal_score.png")


# 3. Exact linear-Gaussian oracle: composed global posteriors --------------------------------
def fig_oracle_linear():
    sys.path.insert(0, str(common.COMPASS / "tests"))
    from test_experimental_marginal_schur import hierarchy, composed_moments
    hier = hierarchy()
    torch.manual_seed(3)
    obs = hier.simulate(12)[:, hier.observed_indices]
    mean, cov = hier.global_posterior(obs)
    arms = [("gauss_jacobian_FD", "laplace", False, "gauss_jacobian_FD (production)"),
            ("arruda_schur", "laplace", False, "arruda sum + laplace"),
            ("arruda_schur_gauss", "laplace", False, "arruda precision-weighted + laplace"),
            ("arruda_schur_gauss", "direct", True, "same rule, mean-field factors")]
    colours = [RULE_COLOURS["gauss_jacobian_FD"], RULE_COLOURS["arruda_laplace"],
               RULE_COLOURS["arruda_gauss_laplace"], RULE_COLOURS["arruda_direct"]]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharex=True, sharey=True)
    L = torch.linalg.cholesky(cov)
    th = torch.linspace(0, 2 * math.pi, 200, dtype=torch.float64)
    circle = torch.stack([th.cos(), th.sin()])
    for ax, (corr, est, mf, label), c in zip(axes, arms, colours):
        _, _, draws = composed_moments(hier, corr, est, obs, samples=1500, mean_field=mf)
        g = draws[0, :, :2].double().numpy()
        ax.scatter(g[:, 0], g[:, 1], s=3, color=c, alpha=0.35, lw=0)
        for k in (1, 2):
            e = (mean[:, None] + k * L @ circle).numpy()
            ax.plot(e[0], e[1], color=INK, lw=1, ls="--" if k == 2 else "-")
        sd = cov.diagonal().sqrt()
        err = max(float(((g.mean(0) - mean.numpy()) / sd.numpy()).__abs__().max()),
                  float(abs(g.std(0) / sd.numpy() - 1).max()))
        ax.set_title(f"{label}\nmax |error| {err:.2f} sd", fontsize=8.5)
        ax.set_xlabel("g₁")
    axes[0].set_ylabel("g₂")
    fig.suptitle("Exact linear-Gaussian oracle, J = 12 subjects: composed p(g | x₁..x₁₂) "
                 "(black: exact 1σ / 2σ)", fontsize=10)
    save(fig, "3_oracle_composed_posteriors.png")


# 4. Gaussian-mixture benchmark ---------------------------------------------------------
PATTERN = re.compile(r"(\w+)\s+n=(\d+)\s+rep=(\d+)\s+(\w+)\s+W1/sd=([\d.]+) mass=([\d.]+) \(exact ([\d.]+)\)")


def read_gm(name):
    rows = []
    path = LOGS / f"{name}.log"
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            m = PATTERN.search(line)
            if m:
                rows.append(dict(regime=m[1], n=int(m[2]), rep=int(m[3]), rule=m[4],
                                 w1=float(m[5]), mass=float(m[6]), exact=float(m[7])))
    return rows


def fig_gm():
    panels = [("gm_oracle", "Exact diffused scores (oracle)"),
              ("gm_net_hierarchical", "Trained network, 'hierarchical' attention")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for col, (log, title) in enumerate(panels):
        rows = [r for r in read_gm(log) if r["regime"] == "asymmetric"]
        ns = sorted({r["n"] for r in rows})
        rules = [r for r in RULE_ORDER if any(x["rule"] == r for x in rows)]
        width = 0.8 / max(1, len(rules))
        for k, rule in enumerate(rules):
            for metric, ax in (("w1", axes[0, col]), ("mass", axes[1, col])):
                xs, ys = [], []
                for i, n in enumerate(ns):
                    vals = [r[metric] for r in rows if r["n"] == n and r["rule"] == rule]
                    if vals:
                        xs.append(i + (k - (len(rules) - 1) / 2) * width)
                        ys.append(np.mean(vals))
                control = log == "gm_net_hierarchical" and "laplace" in rule
                ax.bar(xs, ys, width * 0.9, color=RULE_COLOURS[rule], label=rule + (" (double-count control)" if control else ""),
                       hatch="///" if control else None, edgecolor="white", lw=0.5)
        exact = {r["n"]: r["exact"] for r in rows}
        axes[1, col].hlines([exact[n] for n in ns], [i - 0.45 for i in range(len(ns))],
                            [i + 0.45 for i in range(len(ns))], color=INK, lw=1.5, ls="--")
        for ax in axes[:, col]:
            ax.set_xticks(range(len(ns)), [f"n = {n}" for n in ns])
        axes[0, col].set_yscale("log")
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].set_ylabel("W1 / posterior sd (log, lower is better)")
        axes[1, col].set_ylabel("mass on the + branch (dashed: exact)")
        axes[1, col].set_ylim(0, 1)
        axes[0, col].legend(fontsize=6.8, loc="upper left")
    fig.suptitle("Gaussian mixture, asymmetric regime (exact branch mass 0.75); "
                 "network n=8 averages 2 replicates", fontsize=10)
    save(fig, "4_gaussian_mixture.png")


# 5. Noise twins (exact scores) --------------------------------------------------------
def fig_twins():
    text = (LOGS / "twins.log").read_text(errors="ignore")
    blocks = re.split(r"#{10} (\w+):", text)[1:]
    data = {}
    for name, body in zip(blocks[::2], blocks[1::2]):
        for line in body.splitlines():
            m = re.match(r"\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                data.setdefault(name, {})[m[1]] = (float(m[2]), float(m[3]), float(m[7]))
    problems = [p for p in ("gauss", "laplace", "exponential") if p in data]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    rules = [r for r in RULE_ORDER if any(r in data[p] for p in problems)]
    width = 0.8 / len(rules)
    for k, rule in enumerate(rules):
        xs = [i + (k - (len(rules) - 1) / 2) * width for i, p in enumerate(problems) if rule in data[p]]
        w1 = [data[p][rule][1] for p in problems if rule in data[p]]
        sec = [data[p][rule][0] for p in problems if rule in data[p]]
        axes[0].bar(xs, w1, width * 0.9, color=RULE_COLOURS[rule], label=rule, edgecolor="white", lw=0.5)
        axes[1].bar(xs, sec, width * 0.9, color=RULE_COLOURS[rule], edgecolor="white", lw=0.5)
    for ax in axes:
        ax.set_xticks(range(len(problems)), [f"{p} twin" for p in problems])
    axes[0].set_ylabel("W1 / posterior sd (lower is better)")
    axes[1].set_ylabel("seconds (CPU, log)")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.suptitle("Noise twins, exact joint scores, n = 30 (Laplace / exponential twins still running)", fontsize=10)
    save(fig, "5_noise_twins.png")


# 6. Training losses -------------------------------------------------------------------
def val_curve(path):
    vals = re.findall(r"Epoch:\s*(\d+).*?Validation Loss:\s*([\d.]+)", Path(path).read_text(errors="ignore"))
    return np.array([int(e) for e, _ in vals]), np.array([float(v) for _, v in vals])


def fig_training():
    panels = [
        ("Partial pooling (C2 recipe, 26 nodes)", [
            ("none (production C2)", common.COMPASS / "Partial_Pooling/ladder_encsum/logs/train-C2-h96d6-te64-mask6mix.log"),
            ("full", LOGS / "pp_train_full.log"), ("global", LOGS / "pp_train_global.log")]),
        ("Inverse kinematics, angle+height (h32d3wide)", [
            ("none", LOGS / "ik_train_angle_height_none.log"), ("global", LOGS / "ik_train_angle_height_global.log"),
            ("full", LOGS / "ik_train_angle_height_full.log")]),
        ("Gaussian mixture (h32d3wide)", [
            ("hierarchical", LOGS / "gm_train_hierarchical.log"), ("full", LOGS / "gm_train_full.log")]),
    ]
    mode_colour = {"none": "#6b6a63", "global": "#2a78d6", "hierarchical": "#1baf7a", "full": "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (title, curves) in zip(axes, panels):
        for label, path in curves:
            if not Path(path).exists():
                continue
            e, v = val_curve(path)
            if len(e) == 0:
                continue
            key = label.split()[0]
            ax.plot(e, v, color=mode_colour[key], lw=2, label=f"{label}: {v.min():.3g}")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation DSM loss")
        ax.legend(fontsize=7.5, title="attention mode: best", title_fontsize=7.5)
    fig.suptitle("Validation loss by latent-attention mode (same data, same validation split per panel)",
                 fontsize=10)
    save(fig, "6_training_losses.png")


# 7. Partial-pooling smoke --------------------------------------------------------------
def fig_pp_smoke():
    base = HERE / "artifacts" / "pp" / "none" / "recovery" / "smoke"
    sys.path[:0] = [str(common.COMPASS)]
    rules = ["gauss_jacobian_FD", "arruda_direct", "arruda_gauss_direct"]
    res = {}
    for rule in rules:
        for f in sorted(glob.glob(str(base / f"{rule}-s100-t50" / "dataset-*.pt"))):
            d = torch.load(f, map_location="cpu", weights_only=False)
            p = d["posterior"]
            res.setdefault(rule, []).append((p["globals"].median(0).values, d["truth_globals"],
                                             p["locals"].median(0).values, d["truth_locals"]))
    if not res:
        return
    from Partial_Pooling.config import get_config
    from Partial_Pooling.model_pipeline import load_normalizers
    from Partial_Pooling.paths import BenchmarkPaths
    scale = load_normalizers(get_config("small"), BenchmarkPaths.default())["sde_joint_theta"].scale
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for k, rule in enumerate(rules):
        g = [float((((a - b) / scale[:7]) ** 2).mean().sqrt()) for a, b, _, _ in res[rule]]
        l = [float((((a - b) / scale[7:]) ** 2).mean().sqrt()) for _, _, a, b in res[rule]]
        for ax, vals in zip(axes, (g, l)):
            ax.scatter([k] * len(vals), vals, color=RULE_COLOURS[rule], s=40, zorder=3)
            ax.hlines(np.mean(vals), k - 0.25, k + 0.25, color=RULE_COLOURS[rule], lw=2)
    for ax, t in zip(axes, ("global std-RMSE", "local std-RMSE")):
        ax.set_xticks(range(3), ["gauss_jacobian_FD", "arruda_direct\n(+ ancestral locals)",
                                 "arruda_gauss_direct\n(+ ancestral locals)"], fontsize=7.5)
        ax.set_ylabel(t)
    fig.suptitle("Partial pooling smoke, production C2 network, 2 datasets × 10 draws (indicative only)",
                 fontsize=9.5)
    save(fig, "7_partial_pooling_smoke.png")


# 8. Mean-field diagnostic on saved production posteriors -----------------------------------
def fig_mean_field():
    sys.path[:0] = [str(common.COMPASS)]
    from scipy.stats import rankdata, pearsonr
    from Partial_Pooling.simulators.priors import GLOBAL_PRIOR_MEAN as M, GLOBAL_PRIOR_STD as S
    d = common.COMPASS / "Partial_Pooling/ladder_k1/mask6mix/artifacts_churn_eta2/partial_pooling_recovery/small/dpm2_gauss_hierarchical-a6ee43ef3b4dff69"
    T, Med, Lm = [], [], []
    for f in sorted(glob.glob(str(d / "*.pt"))):
        o = torch.load(f, map_location="cpu", weights_only=False)
        g = o["posterior"]["globals"].numpy()
        g = g[0] if g.ndim == 3 else g
        T.append(o["truth_globals"].numpy())
        Med.append(np.median(g, 0))
        Lm.append(o["truth_locals"].numpy().mean(0))
    T, Med, Lm = map(np.array, (T, Med, Lm))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for k, (ax, name) in enumerate(zip(axes, ("nu", "log_alpha", "log_t0"))):
        s = 3 + k
        err = Med[:, s] - T[:, s]
        z2 = ((Lm[:, k] - np.asarray(M)[k]) / np.asarray(S)[k]) ** 2
        ctrl = [rankdata(T[:, s]), rankdata(T[:, k])]

        def resid(y):
            X = np.column_stack([np.ones(len(y))] + ctrl)
            return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        r = pearsonr(resid(rankdata(err)), resid(rankdata(z2))).statistic
        ax.scatter(z2, err, s=12, color="#2a78d6", alpha=0.7, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("((mean local − prior mean of μ) / prior sd)²")
        ax.set_title(f"log_sigma_{name}: partial ρ = {r:+.2f}", fontsize=9.5)
    axes[0].set_ylabel("posterior median − truth")
    fig.suptitle("Production (mean-field) trunk, 100 saved datasets: log-σ error grows with distance "
                 "from the prior mean, as mean-field composition predicts", fontsize=9.5)
    save(fig, "8_mean_field_diagnostic.png")


if __name__ == "__main__" and "--final" not in sys.argv:
    for f in (fig_attention, fig_scale, fig_gm, fig_twins, fig_training, fig_pp_smoke, fig_mean_field,
              fig_oracle_linear):
        try:
            f()
        except Exception as error:  # noqa: BLE001
            print(f"!! {f.__name__}: {error!r}")


# 10. IK and partial pooling, final ---------------------------------------------------
MODE_ORDER = ("none", "global", "hierarchical", "full")


def fig_ik_final():
    import json as _json
    rows = [r for r in _json.loads((HERE / "artifacts" / "ik" / "grades.json").read_text())
            if r["layout"] == "angle_height" and r["n"] == 8]
    rules = [r for r in RULE_ORDER + ["gauss_hierarchical"] if any(x["method"] == r for x in rows)]
    colours = dict(RULE_COLOURS, gauss_hierarchical="#6b6a63")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    width = 0.8 / len(rules)
    for ax, param in zip(axes, ("angle_1", "height_arm")):
        for k, rule in enumerate(rules):
            xs, ys = [], []
            for i, mode in enumerate(MODE_ORDER):
                hit = [r for r in rows if r["mode"] == mode and r["method"] == rule]
                if hit:
                    xs.append(i + (k - (len(rules) - 1) / 2) * width)
                    ys.append(hit[0][param]["median_w1_over_ref_sd"])
            ax.bar(xs, ys, width * 0.9, color=colours[rule], label=rule, edgecolor="white", lw=0.5)
        ax.set_xticks(range(4), MODE_ORDER)
        ax.set_xlabel("latent-attention mode")
        ax.set_title(f"{param}", fontsize=10)
    axes[0].set_ylabel("W1 / exact-posterior sd (lower is better)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Inverse kinematics, angle+height layout, n = 8, 30 datasets, graded against the exact MCMC reference",
                 fontsize=10)
    save(fig, "10_ik_attention_modes.png")


def fig_pp_final():
    sys.path[:0] = [str(common.COMPASS)]
    from Partial_Pooling.config import get_config
    from Partial_Pooling.model_pipeline import load_normalizers
    from Partial_Pooling.paths import BenchmarkPaths
    sc = load_normalizers(get_config("small"), BenchmarkPaths.default())["sde_joint_theta"].scale
    C2 = common.COMPASS / "Partial_Pooling/artifacts/encoder_summary/recovery/setenc-d16-w64x2s4-h96d6-log_sigma-te64-mask6mix"
    E = HERE / "artifacts" / "pp"
    arms = [("none", "gauss_hierarchical", C2 / "dpm2_gauss_hierarchical-s100-t50"),
            ("none", "gauss_jacobian", C2 / "dpm2_gauss_jacobian_newton-s100-t50"),
            ("global", "gauss_jacobian_FD", E / "global/recovery/main/gauss_jacobian_FD-s100-t50"),
            ("global", "arruda_direct", E / "global/recovery/main/arruda_direct-s100-t50"),
            ("hierarchical", "gauss_jacobian_FD", E / "hierarchical/recovery/main/gauss_jacobian_FD-s100-t50"),
            ("hierarchical", "arruda_direct", E / "hierarchical/recovery/main/arruda_direct-s100-t50"),
            ("full", "gauss_jacobian_FD", E / "full/recovery/main/gauss_jacobian_FD-s100-t50"),
            ("full", "gauss_hierarchical", E / "full/recovery/main/gauss_hierarchical-s100-t50")]
    colours = dict(RULE_COLOURS, gauss_hierarchical="#6b6a63")
    stats = []
    for mode, rule, d in arms:
        G, L, P, T = [], [], [], []
        for i in range(10):
            f = glob.glob(str(d / "**" / f"dataset-{i:04d}.pt"), recursive=True)
            o = torch.load(f[0], map_location="cpu", weights_only=False)
            p = o["posterior"]
            m = p["globals"].median(0).values
            G.append(float((((m - o["truth_globals"]) / sc[:7]) ** 2).mean().sqrt()))
            L.append(float((((p["locals"].median(0).values - o["truth_locals"]) / sc[7:]) ** 2).mean().sqrt()))
            P.append(m.numpy())
            T.append(o["truth_globals"].numpy())
        P, T = np.array(P), np.array(T)
        r = np.mean([np.corrcoef(P[:, k], T[:, k])[0, 1] for k in range(7)])
        stats.append((mode, rule, np.mean(G), np.mean(L), r))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    labels = [f"{m}\n{r}" for m, r, *_ in stats]
    for ax, idx, title in zip(axes, (2, 3, 4), ("global std-RMSE (lower better)", "local std-RMSE (lower better)",
                                                 "global mean r (higher better)")):
        vals = [s[idx] for s in stats]
        ax.barh(range(len(stats)), vals, color=[colours[s[1]] for s in stats], edgecolor="white")
        ax.set_yticks(range(len(stats)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9.5)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=7, color=INK)
    axes[2].set_xlim(0.6, 0.9)
    fig.suptitle("Partial pooling (encoder summary, K = 1), datasets 0–9, 100 subjects × 100 draws; "
                 "rows are attention mode / rule", fontsize=10)
    save(fig, "11_partial_pooling_final.png")


if __name__ == "__main__" and "--final" in sys.argv:
    for f in (fig_ik_final, fig_pp_final, fig_gm, fig_twins, fig_training):
        try:
            f()
        except Exception as error:  # noqa: BLE001
            print(f"!! {f.__name__}: {error!r}")
