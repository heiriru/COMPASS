#!/usr/bin/env python3
"""IK where only the horizontal reach is observed: an exactly bimodal angle_1 posterior.

The arm (lengths 0.5, 0.5, 1.0, relative angles a1, a2, a3, shoulder at the origin) is
observed through the horizontal coordinate of its end point only, with Gaussian noise:

    x2 = L1 cos(a1) + L2 cos(a1 + a2) + L3 cos(a1 + a2 + a3) + N(0, NOISE^2)

cos is even, so the arm and its mirror image (all three angles negated) give the same x2
for every observation: the likelihood of +a1 and -a1 is identical, no data selection or
folding is involved. a1 is global (shared by all observations), a2 and a3 are local (one
pair per observation); all priors N(mu, 0.5^2). Data are plain simulator draws with
a1 = 0.6. Two variants, each with its own network:

``sym``    prior mean of a1 is 0: the exact posterior has P(a1 > 0) = 0.5 at every n.
``shift``  prior mean of a1 is 0.15: the likelihood is still symmetric, so the prior alone
           tilts the two modes to P(a1 > 0) ~ 0.67 -- a known, unequal split.

Exact posterior: 1-D grid over a1, likelihood p(x2 | a1) by quadrature over (a2, a3).
Networks: tokenized h32d3wide with hierarchical attention (recipe of ``ik.py``);
composed with gauss_jacobian_FD and arruda_direct.

    python ik_reach.py data
    python ik_reach.py train --variant sym|shift
    python ik_reach.py run --variant sym|shift
    python ik_reach.py grade
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ik  # noqa: E402  (CPU cap, autocvd unless CUDA_VISIBLE_DEVICES, sys.path)
import common  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

H = ik.H
ROOT = HERE / "artifacts" / "ik_reach"
NS = (8, 32, 128)
L1, L2, L3 = 0.5, 0.5, 1.0
SD, NOISE, A1_TRUE, SEED = 0.5, 0.1, 0.6, 1
PRIOR_MEAN = {"sym": 0.0, "shift": 0.15}
VARIANTS = (("sym", "prior mean 0 (exact split 50/50)"), ("shift", "prior mean 0.15 (exact split ~67/33)"))
METHODS = (("gauss_jacobian_FD", "hier. attn + gauss_jacobian_FD", "#eda100"),
           ("arruda_direct", "hier. attn + Arruda (plain sum)", "#e87ba4"))


def reach(a1, a2, a3):
    return L1 * np.cos(a1) + L2 * np.cos(a1 + a2) + L3 * np.cos(a1 + a2 + a3)


def use_reach_model(variant):
    """Patch the hierarchical IK module to nodes (a1 | a2, a3 | x2)."""
    mean = np.array([PRIOR_MEAN[variant], 0.0, 0.0])
    H.GLOBAL_NAMES, H.LOCAL_NAMES, H.OBSERVED_NAMES = ["angle_1"], ["angle_2", "angle_3"], ["x2"]
    H.NODE_NAMES = H.GLOBAL_NAMES + H.LOCAL_NAMES + H.OBSERVED_NAMES
    H.NODES, H.HIERARCHY, H.LOCAL_INDICES = 4, [0], [1, 2]
    H.OBSERVED_INDICES, H.LATENT_INDICES = [3], [0, 1, 2]
    H.CONDITION_MASK = torch.tensor([0.0, 0.0, 0.0, 1.0])
    H.PRIOR_MEAN, H.PRIOR_STD = mean, np.full(3, SD)

    def simulate(count, rng):
        theta = rng.normal(mean, SD, size=(int(count), 3))
        x = reach(*theta.T) + rng.normal(0.0, NOISE, len(theta))
        return (torch.as_tensor(theta, dtype=torch.float32),
                torch.as_tensor(x[:, None], dtype=torch.float32))

    H.simulate = simulate
    return mean


MENUS = ("six_weighted", "bernoulli")   # bernoulli = the plain Bernoulli(0.33) mask (menu=False)


def tag(variant, menu):
    return variant if menu == "six_weighted" else f"{variant}_{menu}"


def model_dir(variant, menu="six_weighted"):
    return ROOT / "models" / f"tokenized_{ik.CONFIG}_{tag(variant, menu)}_hierarchical"


# -- data and exact reference --------------------------------------------------------------
def make_data(n=128):
    rng = np.random.default_rng(SEED)
    local = rng.normal(0.0, SD, (n, 2))
    return reach(A1_TRUE, local[:, 0], local[:, 1]) + rng.normal(0.0, NOISE, n)


def log_likelihood(grid, x, nodes=321, bins=2001):
    """log p(x_j | a1) on ``grid`` (n_grid, n): (a2, a3) integrated on a +-4 sd grid, the
    noise-free reach histogrammed per a1 and convolved with the noise. Computed for
    a1 >= 0 and mirrored, so the symmetry is exact."""
    u = np.linspace(-4 * SD, 4 * SD, nodes)
    a2, a3 = (v.ravel() for v in np.meshgrid(u, u, indexing="ij"))
    weight = np.exp(-0.5 * (a2 ** 2 + a3 ** 2) / SD ** 2)
    weight /= weight.sum()
    xb = np.linspace(-2.5, 2.5, bins)
    dx = xb[1] - xb[0]
    k = np.arange(-int(6 * NOISE / dx), int(6 * NOISE / dx) + 1) * dx
    kernel = np.exp(-0.5 * (k / NOISE) ** 2)
    kernel /= kernel.sum() * dx
    half = np.abs(grid)
    unique, inverse = np.unique(half, return_inverse=True)
    out = np.empty((len(unique), len(x)))
    for i, a in enumerate(unique):
        h, _ = np.histogram(reach(a, a2, a3), bins=bins, range=(xb[0] - dx / 2, xb[-1] + dx / 2),
                            weights=weight)
        out[i] = np.log(np.maximum(np.interp(x, xb, np.convolve(h, kernel, mode="same")), 1e-300))
    return out[inverse]


def exact(x, mean, grid):
    lp = -0.5 * ((grid - mean) / SD) ** 2 + log_likelihood(grid, x).sum(1)
    d = np.exp(lp - lp.max())
    return d / np.trapezoid(d, grid)


def data(arguments):
    ROOT.mkdir(parents=True, exist_ok=True)
    x = make_data()
    np.save(ROOT / "data.npy", x)
    grid = np.linspace(-1.6, 1.6, 3201)
    ll = log_likelihood(grid, x)
    out = {"grid": grid}
    for variant, _ in VARIANTS:
        for n in NS:
            lp = -0.5 * ((grid - PRIOR_MEAN[variant]) / SD) ** 2 + ll[:, :n].sum(1)
            d = np.exp(lp - lp.max())
            out[f"{variant}_n{n}"] = d / np.trapezoid(d, grid)
            print(f"{variant:5s} n={n:<3} exact P(a1>0) {np.trapezoid(out[f'{variant}_n{n}'] * (grid > 0), grid):.3f}",
                  flush=True)
    np.savez(ROOT / "exact.npz", **out)


# -- train / run ---------------------------------------------------------------------------
def train(arguments):
    use_reach_model(arguments.variant)
    import train_model as trainer
    from compass import hierarchical_roles
    from compass.experimental.latent_attention import enable_latent_attention, save_latent_attention

    directory = model_dir(arguments.variant, arguments.menu)
    directory.mkdir(parents=True, exist_ok=True)
    original = trainer.build

    def build(config, arch, device="cpu"):
        model = original(config, arch, device=device)
        enable_latent_attention(model, "hierarchical", hierarchical_roles(model.schema, 1, 2))
        save_latent_attention(model, directory)
        return model

    trainer.build = build
    trainer.load_to = common.load_with_latent_attention(trainer.load_to)
    started = time.time()
    _, _, parameters = trainer.train_or_load(
        directory, ik.CONFIG, "tokenized", arguments.train_samples, "cuda", seed=7,
        menu=False if arguments.menu == "bernoulli" else arguments.menu, time_sampling="log_sigma")
    (directory / "run.json").write_text(json.dumps({
        "variant": arguments.variant, "nodes": H.NODE_NAMES, "prior_mean": PRIOR_MEAN[arguments.variant],
        "noise": NOISE, "menu": arguments.menu, "parameters": parameters, "minutes": (time.time() - started) / 60}, indent=2))


def composed(model, x, method, draws, seed, mean, pair_budget=32_000):
    from compass.experimental.marginal_schur import MarginalSchurSampler
    settings, kwargs = ik.METHODS[method]
    per_call = max(1, min(draws, pair_budget // len(x)))
    parts = []
    for k, start in enumerate(range(0, draws, per_call)):
        torch.manual_seed(seed + k)
        s = MarginalSchurSampler(model, **settings).sample(
            world_size=1, data=torch.as_tensor(x[:, None], dtype=torch.float32),
            condition_mask=H.CONDITION_MASK, num_samples=min(per_call, draws - start),
            timesteps=ik.TIMESTEPS, eps=1e-3, hierarchy=[0],
            prior=(torch.tensor(mean[:1], dtype=torch.float32), torch.full((1,), SD)),
            local_prior=(torch.tensor(mean[1:], dtype=torch.float32), torch.full((2,), SD)),
            denoise_clamp=5.0, device="cuda", verbose=False, **ik.DPM, **kwargs)
        parts.append(s.detach().cpu().numpy()[:, :, :3].transpose(1, 0, 2))
        torch.cuda.empty_cache()
    return np.concatenate(parts)


def run(arguments):
    mean = use_reach_model(arguments.variant)
    import train_model as trainer
    load = common.load_with_latent_attention(trainer.load_to)
    model = load(model_dir(arguments.variant, arguments.menu) / "Model_checkpoint.pt", "cuda")
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    model.model.eval()
    x = np.load(ROOT / "data.npy")
    folder = ROOT / "runs" / (tag(arguments.variant, arguments.menu) + arguments.suffix)
    folder.mkdir(parents=True, exist_ok=True)
    for n in arguments.ns:
        for method in arguments.methods:
            path = folder / f"{method}_n{n}.npz"
            if path.exists():
                continue
            started = time.perf_counter()
            theta = composed(model, x[:n], method, arguments.draws, 2026 + n, mean)
            np.savez(path, theta=theta.astype(np.float32), seconds=time.perf_counter() - started)
            a1 = theta[:, 0, 0]
            print(f"{tag(arguments.variant, arguments.menu):16s} {method:18s} n={n:<3} P(a1>0) {np.mean(a1 > 0):.3f}  "
                  f"{time.perf_counter() - started:.0f}s", flush=True)


# -- grade ---------------------------------------------------------------------------------
def draws_from(grid, density, count=200_000):
    cdf = np.concatenate([[0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    return np.interp(np.random.default_rng(0).random(count), cdf / cdf[-1], grid)


def stats(a1):
    """Mass of the a1 > 0 mode, and the mean and sd of each mode (a1 > 0 and a1 < 0)."""
    pos, neg = a1[a1 > 0], a1[a1 < 0]
    moment = lambda v, f: float(f(v)) if len(v) > 20 else float("nan")  # noqa: E731
    return {"P+": float(np.mean(a1 > 0)), "mean+": moment(pos, np.mean), "sd+": moment(pos, np.std),
            "mean-": moment(neg, np.mean), "sd-": moment(neg, np.std)}


def grade(arguments):
    from scipy.stats import wasserstein_distance
    exact_ = np.load(ROOT / "exact.npz")
    grid = exact_["grid"]
    rows = {}
    for variant, _ in VARIANTS:
        for n in NS:
            ref = draws_from(grid, exact_[f"{variant}_n{n}"])
            r = stats(ref)
            print(f"{variant:5s} n={n:<3} {'exact':18s} P(a1>0) {r['P+']:.2f}  mode a1>0 {r['mean+']:+.3f}±{r['sd+']:.3f}"
                  f"  mode a1<0 {r['mean-']:+.3f}±{r['sd-']:.3f}")
            rows[f"{variant}/exact/n{n}"] = r
            for menu in MENUS:
              for method, _, _ in METHODS:
                path = ROOT / "runs" / tag(variant, menu) / f"{method}_n{n}.npz"
                if not path.exists():
                    continue
                a1 = np.load(path)["theta"][:, 0, 0]
                s = stats(a1) | {"w1_sd": float(wasserstein_distance(a1, ref) / ref.std())}
                rows[f"{tag(variant, menu)}/{method}/n{n}"] = s
                print(f"{variant:5s} n={n:<3} {menu:12s} {method:18s} P(a1>0) {s['P+']:.2f}  mode a1>0 {s['mean+']:+.3f}±{s['sd+']:.3f}"
                      f"  mode a1<0 {s['mean-']:+.3f}±{s['sd-']:.3f}  W1/sd {s['w1_sd']:.2f}")
    (ROOT / "grades.json").write_text(json.dumps(rows, indent=1))
    for menu in MENUS:
        plot(exact_, rows, menu)
    plot_menus(rows)


MENU_LABEL = {"six_weighted": "six_weighted mask menu", "bernoulli": "plain Bernoulli(0.33) mask"}


def plot_menus(rows):
    """P(a1>0) and W1/sd against n: both rules x both mask recipes, per variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid_ = "#1f1f1e", "#6b6a63", "#e6e5df"
    fig, axes = plt.subplots(2, len(VARIANTS), figsize=(4.6 * len(VARIANTS), 6.2), squeeze=False)
    for j, (variant, vtitle) in enumerate(VARIANTS):
        top, bottom = axes[0, j], axes[1, j]
        top.plot(NS, [rows[f"{variant}/exact/n{n}"]["P+"] for n in NS], "o-", color=ink, lw=2, ms=7,
                 label="exact")
        for method, mlabel, colour in METHODS:
            for menu, style in (("six_weighted", "-"), ("bernoulli", "--")):
                key = lambda n: f"{tag(variant, menu)}/{method}/n{n}"  # noqa: E731
                if not all(key(n) in rows for n in NS):
                    continue
                label = f"{mlabel.replace('hier. attn + ', '')}, {MENU_LABEL[menu]}"
                top.plot(NS, [rows[key(n)]["P+"] for n in NS], "o" + style, color=colour, lw=2, ms=7,
                         label=label, mfc="white" if menu == "bernoulli" else colour)
                bottom.plot(NS, [rows[key(n)]["w1_sd"] for n in NS], "o" + style, color=colour, lw=2, ms=7,
                            mfc="white" if menu == "bernoulli" else colour)
        top.set_ylim(0, 1)
        top.axhline(0.5, color=muted, lw=0.6, ls=":")
        top.set_title(vtitle, fontsize=9, color=ink)
        top.set_ylabel("P(a1 > 0)", color=ink)
        bottom.set_ylabel("W1 / sd(exact)", color=ink)
        bottom.set_ylim(bottom=0)
        for ax in (top, bottom):
            ax.set_xscale("log", base=2)
            ax.set_xticks(NS, [str(n) for n in NS])
            ax.set_xlabel("n observations", color=ink)
            ax.grid(axis="y", color=grid_, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0, 0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.suptitle("IK reach-only: does the mask recipe matter?  solid = six_weighted, dashed = plain Bernoulli",
                 fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(ROOT / "reach_menus.png", dpi=150, bbox_inches="tight")
    print("wrote", ROOT / "reach_menus.png")


def plot(exact_, rows, menu="six_weighted"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, ref = "#1f1f1e", "#6b6a63", "#d9d8d2"
    grid = exact_["grid"]
    arms = [(v, vt, m, ml, c) for v, vt in VARIANTS for m, ml, c in METHODS]
    fig, axes = plt.subplots(len(arms), len(NS) + 1, figsize=(3.3 * (len(NS) + 1), 1.9 * len(arms)),
                             squeeze=False, gridspec_kw={"width_ratios": [1] * len(NS) + [0.8]})
    bins = np.linspace(-1.5, 1.5, 121)
    for i, (variant, vtitle, method, mlabel, colour) in enumerate(arms):
        for j, n in enumerate(NS):
            ax = axes[i, j]
            density = exact_[f"{variant}_n{n}"]
            ax.fill_between(grid, density, color=ref, lw=0)
            ax.plot(grid, density, color=ink, lw=1)
            ax.axvline(0, color=muted, lw=0.6, ls=":")
            path = ROOT / "runs" / tag(variant, menu) / f"{method}_n{n}.npz"
            if path.exists():
                a1 = np.load(path)["theta"][:, 0, 0]
                ax.hist(a1, bins=bins, density=True, histtype="step", color=colour, lw=2)
                exact_p = rows[f"{variant}/exact/n{n}"]["P+"]
                ax.set_title(f"P(a1>0): exact {exact_p:.2f}, sampler {np.mean(a1 > 0):.2f}", fontsize=7.5,
                             color=ink)
            else:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color=muted)
            ax.set_xlim(-1.5, 1.5)
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"{vtitle}\n{mlabel}", fontsize=7, color=ink, rotation=0, ha="right", va="center")
            if i == 0:
                ax.annotate(f"n = {n}", (0.5, 1.3), xycoords="axes fraction", ha="center", fontsize=10,
                            color=ink, weight="bold")
            if i == len(arms) - 1:
                ax.set_xlabel("angle_1", color=ink)
        # last column: mass of the a1 > 0 mode against n
        ax = axes[i, -1]
        exact_p = [rows[f"{variant}/exact/n{n}"]["P+"] for n in NS]
        ax.plot(range(len(NS)), exact_p, "o-", color=ink, lw=1.5, ms=5, label="exact")
        got = [rows.get(f"{tag(variant, menu)}/{method}/n{n}", {}).get("P+", np.nan) for n in NS]
        ax.plot(range(len(NS)), got, "o-", color=colour, lw=2, ms=6, label="sampler")
        ax.set_xticks(range(len(NS)), [str(n) for n in NS], fontsize=7)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=muted, lw=0.6, ls=":")
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if i == 0:
            ax.set_title("P(a1 > 0) against n", fontsize=8, color=ink)
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        if i == len(arms) - 1:
            ax.set_xlabel("n", color=ink)
    fig.suptitle("IK, only the horizontal reach x2 observed (noise sd 0.1), true angle_1 = 0.6\n"
                 f"grey/black = exact posterior of angle_1;  colour = composed sampler;  net trained with the {MENU_LABEL[menu]}",
                 fontsize=10, color=ink)
    fig.tight_layout()
    name = "reach.png" if menu == "six_weighted" else f"reach_{menu}.png"
    fig.savefig(ROOT / name, dpi=150, bbox_inches="tight")
    print("wrote", ROOT / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("data", "train", "run", "grade"))
    parser.add_argument("--variant", choices=("sym", "shift"))
    parser.add_argument("--methods", nargs="+", default=[m for m, _, _ in METHODS])
    parser.add_argument("--ns", type=int, nargs="+", default=list(NS))
    parser.add_argument("--menu", choices=MENUS, default="six_weighted")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--suffix", default="", help="appended to the runs/ folder name (e.g. _noproj)")
    parser.add_argument("--train-samples", type=int, default=200_000)
    arguments = parser.parse_args()
    {"data": data, "train": train, "run": run, "grade": grade}[arguments.action](arguments)


if __name__ == "__main__":
    main()
