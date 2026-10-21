#!/usr/bin/env python3
"""Natural bimodal IK: observations taken at (almost) the shoulder height.

Reflecting the whole arm about the horizontal line through the shoulder negates every
angle and maps x1 - h -> -(x1 - h); all angle priors are zero-mean, so for observations
with x1 = h the likelihood of +a1 and -a1 is identical. Nothing is folded or re-defined:
the data are ordinary simulator rows whose end point happens to sit at the shoulder
height, |x1 - h_true| < delta. Rows whose arm is (nearly) fully stretched or folded are
dropped too: their end point lies on the reach boundary, where the noise-free
likelihood is cut off, and a deviation of delta pushes the mirror arm out of reach
(infinite log-odds from one row). Truth: a1 = 0.6, h = 0.

Two models, the same 128 observations:

``fixed``   height known (= 0), not a parameter: latents (a1 | a2, a3), a1 global.
``global``  height a parameter, global with a1, prior sd 0.05 (IK_SD_H):
            latents (a1, h | a2, a3). The mirror is (a1, h) -> (-a1, -h); with the
            stock sd 0.25 the exact posterior instead moves to (-+0.3, +-0.65).

Each gets a tokenized h32d3wide hierarchical-attention net (recipe of ``ik.py``),
composed with gauss_jacobian_FD and arruda_direct, graded against the exact posterior
(1-D grid for ``fixed``, 2-D (a1, h) grid for ``global``; ``recovery/ik_exact.py``).

    python ik_flat.py data
    IK_SD_H=0.05 python ik_flat.py train --variant global      (the env var must be set)
    python ik_flat.py train --variant fixed
    python ik_flat.py run --variant fixed|global
    python ik_flat.py grade
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VARIANT = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--variant"), None)
if VARIANT == "global" and os.environ.get("IK_SD_H") != "0.05":
    raise SystemExit("the global variant needs IK_SD_H=0.05 in the environment (read at import)")

import ik  # noqa: E402  (CPU cap, autocvd unless CUDA_VISIBLE_DEVICES, sys.path)
import common  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

H = ik.H
import simulator as sim  # noqa: E402
import ik_exact as E  # noqa: E402

ROOT = HERE / "artifacts" / "ik_flat"
NS = (8, 32, 128)
A1_TRUE, H_TRUE, DELTA, EDGE, SEED = 0.6, 0.0, 1e-3, 0.01, 6
SD_H_GLOBAL = 0.05


# -- the "fixed" model: height known, three latents ------------------------------------
def use_fixed_height():
    H.GLOBAL_NAMES, H.LOCAL_NAMES = ["angle_1"], ["angle_2", "angle_3"]
    H.NODE_NAMES = H.GLOBAL_NAMES + H.LOCAL_NAMES + H.OBSERVED_NAMES
    H.NODES, H.HIERARCHY, H.LOCAL_INDICES = 5, [0], [1, 2]
    H.OBSERVED_INDICES, H.LATENT_INDICES = [3, 4], [0, 1, 2]
    H.CONDITION_MASK = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    H.PRIOR_MEAN, H.PRIOR_STD = np.zeros(3), np.full(3, 0.5)

    def forward(theta):           # (a1, a2, a3) with the height pinned at 0
        theta = np.atleast_2d(np.asarray(theta, np.float64))
        return sim.forward(np.column_stack([np.full(len(theta), H_TRUE), theta]))

    def simulate(count, rng):
        theta = rng.normal(0.0, 0.5, size=(int(count), 3))
        return (torch.as_tensor(theta, dtype=torch.float32),
                torch.as_tensor(forward(theta), dtype=torch.float32))

    H.forward, H.simulate = forward, simulate


def setup(variant):
    """``(n_global, n_latent, prior_std)`` and the module state for ``variant``."""
    if variant == "fixed":
        use_fixed_height()
        return 1, 3, np.full(3, 0.5)
    ik.set_layout("angle_height")
    assert abs(sim.PRIOR_SCALES[0] - SD_H_GLOBAL) < 1e-12
    return 2, 4, np.array([0.5, SD_H_GLOBAL, 0.5, 0.5])


def model_dir(variant):
    return ROOT / "models" / f"tokenized_{ik.CONFIG}_{variant}_hierarchical"


# -- data and exact references -----------------------------------------------------------
def make_data(n=128):
    rng = np.random.default_rng(SEED)
    rows = []
    while sum(len(r) for r in rows) < n:
        m = 400_000
        theta = np.column_stack([np.full(m, H_TRUE), np.full(m, A1_TRUE),
                                 rng.normal(0, .5, m), rng.normal(0, .5, m)])
        x = sim.forward(theta)
        reach = np.hypot(x[:, 0] - H_TRUE - E.L1 * np.sin(A1_TRUE), x[:, 1] - E.L1 * np.cos(A1_TRUE))
        keep = (np.abs(x[:, 0] - H_TRUE) < DELTA) & (reach > E.R_MIN + EDGE) & (reach < E.R_MAX - EDGE)
        rows.append(x[keep])
    return np.concatenate(rows)[:n]


def exact_fixed(x, grid=np.linspace(-1.6, 1.6, 32_001)):
    lp = E.log_normal(grid, 0.5)
    for j in range(len(x)):
        lp = lp + E.log_lik_a1h(grid, H_TRUE, x[j, 0], x[j, 1])
    d = np.exp(lp - lp.max())
    return grid, d / np.trapezoid(d, grid)


def exact_global(x):
    E.SD_H = SD_H_GLOBAL
    ga, gh, density = E.grid_posterior_a1h(x, points=1201)
    return ga, density.sum(1) * (gh[1] - gh[0]), gh, density.sum(0) * (ga[1] - ga[0])


def data(arguments):
    ROOT.mkdir(parents=True, exist_ok=True)
    x = make_data()
    np.save(ROOT / "data.npy", x)
    out = {}
    for n in NS:
        g, d = exact_fixed(x[:n])
        out[f"fixed_grid_n{n}"], out[f"fixed_n{n}"] = g, d
        ga, da, gh, dh = exact_global(x[:n])
        out[f"global_grid_n{n}"], out[f"global_n{n}"] = ga, da
        out[f"global_hgrid_n{n}"], out[f"global_h_n{n}"] = gh, dh
        print(f"n={n:<3} exact P(a1>0): fixed {np.trapezoid(d * (g > 0), g):.3f}  "
              f"global {np.trapezoid(da * (ga > 0), ga):.3f}", flush=True)
    np.savez(ROOT / "exact.npz", **out)


# -- train / run ---------------------------------------------------------------------------
def train(arguments):
    n_global, n_latent, _ = setup(arguments.variant)
    import train_model as trainer
    from compass import hierarchical_roles
    from compass.experimental.latent_attention import enable_latent_attention, save_latent_attention

    directory = model_dir(arguments.variant)
    directory.mkdir(parents=True, exist_ok=True)
    original = trainer.build

    def build(config, arch, device="cpu"):
        model = original(config, arch, device=device)
        roles = hierarchical_roles(model.schema, n_global, n_latent - n_global)
        enable_latent_attention(model, "hierarchical", roles)
        save_latent_attention(model, directory)
        return model

    trainer.build = build
    trainer.load_to = common.load_with_latent_attention(trainer.load_to)
    started = time.time()
    _, _, parameters = trainer.train_or_load(
        directory, ik.CONFIG, "tokenized", 200_000, "cuda", seed=7, menu="six_weighted",
        time_sampling="log_sigma")
    (directory / "run.json").write_text(json.dumps({
        "variant": arguments.variant, "nodes": H.NODE_NAMES, "sd_h": float(sim.PRIOR_SCALES[0]),
        "parameters": parameters, "minutes": (time.time() - started) / 60}, indent=2))


def composed(model, x, method, draws, seed, n_global, n_latent, prior_std, pair_budget=32_000):
    from compass.experimental.marginal_schur import MarginalSchurSampler
    settings, kwargs = ik.METHODS[method]
    per_call = max(1, min(draws, pair_budget // len(x)))
    parts = []
    for k, start in enumerate(range(0, draws, per_call)):
        torch.manual_seed(seed + k)
        s = MarginalSchurSampler(model, **settings).sample(
            world_size=1, data=torch.as_tensor(x, dtype=torch.float32), condition_mask=H.CONDITION_MASK,
            num_samples=min(per_call, draws - start), timesteps=ik.TIMESTEPS, eps=1e-3,
            hierarchy=list(range(n_global)),
            prior=(torch.zeros(n_global), torch.as_tensor(prior_std[:n_global], dtype=torch.float32)),
            local_prior=(torch.zeros(n_latent - n_global),
                         torch.as_tensor(prior_std[n_global:], dtype=torch.float32)),
            denoise_clamp=5.0, device="cuda", verbose=False, **ik.DPM, **kwargs)
        parts.append(s.detach().cpu().numpy()[:, :, :n_latent].transpose(1, 0, 2))
        torch.cuda.empty_cache()
    return np.concatenate(parts)


def run(arguments):
    n_global, n_latent, prior_std = setup(arguments.variant)
    import train_model as trainer
    load = common.load_with_latent_attention(trainer.load_to)
    model = load(model_dir(arguments.variant) / "Model_checkpoint.pt", "cuda")
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    model.model.eval()
    x = np.load(ROOT / "data.npy")
    folder = ROOT / "runs" / (arguments.variant + arguments.suffix)
    folder.mkdir(parents=True, exist_ok=True)
    for n in arguments.ns:
        for method in arguments.methods:
            path = folder / f"{method}_n{n}.npz"
            if path.exists():
                continue
            started = time.perf_counter()
            theta = composed(model, x[:n], method, arguments.draws, 2026 + n, n_global, n_latent, prior_std)
            np.savez(path, theta=theta.astype(np.float32), seconds=time.perf_counter() - started)
            a1 = theta[:, 0, 0]
            print(f"{arguments.variant:6s} {method:18s} n={n:<3} P(a1>0) {np.mean(a1 > 0):.3f} "
                  f"E|a1| {np.abs(a1).mean():.3f}  {time.perf_counter() - started:.0f}s", flush=True)


# -- grade ---------------------------------------------------------------------------------
def draws_from(grid, density, count=200_000):
    cdf = np.concatenate([[0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    return np.interp(np.random.default_rng(0).random(count), cdf / cdf[-1], grid)


def stats(a1):
    med = lambda v: float(np.median(v)) if len(v) > 20 else float("nan")  # noqa: E731
    return {"P+": float(np.mean(a1 > 0)), "mode+": med(a1[a1 > 0]), "mode-": med(-a1[a1 < 0]),
            "valley": float(np.mean(np.abs(a1) < 0.2))}


METHODS = (("gauss_jacobian_FD", "hier. attn + gauss_jacobian_FD", "#eda100"),
           ("arruda_direct", "hier. attn + Arruda (plain sum)", "#e87ba4"))


def grade(arguments):
    from scipy.stats import wasserstein_distance
    exact = np.load(ROOT / "exact.npz")
    rows = {}
    for variant in ("fixed", "global"):
        for n in NS:
            ref = draws_from(exact[f"{variant}_grid_n{n}"], exact[f"{variant}_n{n}"])
            r = stats(ref)
            sd_branch = np.std(np.abs(ref[ref > 0])) if np.mean(ref > 0) > 0.01 else np.std(np.abs(ref))
            print(f"{variant:6s} n={n:<3} exact                  P+ {r['P+']:.2f}  |a1| +{r['mode+']:.3f} "
                  f"-{r['mode-']:.3f}  valley {r['valley']:.2f}")
            for method, _, _ in METHODS:
                path = ROOT / "runs" / variant / f"{method}_n{n}.npz"
                if not path.exists():
                    continue
                theta = np.load(path)["theta"]
                a1 = theta[:, 0, 0]
                s = stats(a1)
                s["w1_sd"] = float(wasserstein_distance(a1, ref) / ref.std())
                s["width"] = float(np.std(np.abs(a1[a1 > 0])) / sd_branch) if np.mean(a1 > 0) > 0.01 else float("nan")
                if variant == "global":
                    s["h+"], s["h-"] = (float(theta[a1 > 0, 0, 1].mean()) if np.any(a1 > 0) else float("nan"),
                                        float(theta[a1 < 0, 0, 1].mean()) if np.any(a1 < 0) else float("nan"))
                rows[f"{variant}/{method}/n{n}"] = s
                print(f"{variant:6s} n={n:<3} {method:22s} P+ {s['P+']:.2f}  |a1| +{s['mode+']:.3f} "
                      f"-{s['mode-']:.3f}  valley {s['valley']:.2f}  W1/sd {s['w1_sd']:.2f}  width+ {s['width']:.2f}"
                      + (f"  h|+ {s['h+']:+.3f} h|- {s['h-']:+.3f}" if variant == "global" else ""))
    (ROOT / "grades.json").write_text(json.dumps(rows, indent=1))
    plot(exact)


def plot(exact):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [(v, m, f"{'h known' if v == 'fixed' else 'h global (sd .05)'}\n{label}", c)
            for v in ("fixed", "global") for m, label, c in METHODS]
    fig, axes = plt.subplots(len(arms), len(NS), figsize=(3.4 * len(NS), 2.0 * len(arms)), squeeze=False)
    bins = np.linspace(-1.0, 1.0, 161)
    for j, n in enumerate(NS):
        for i, (variant, method, label, colour) in enumerate(arms):
            ax = axes[i, j]
            grid, density = exact[f"{variant}_grid_n{n}"], exact[f"{variant}_n{n}"]
            ax.fill_between(grid, density, color="#e6e5df", lw=0)
            ax.plot(grid, density, color="#1f1f1e", lw=1)
            path = ROOT / "runs" / variant / f"{method}_n{n}.npz"
            title = f"exact P(a1>0) {np.trapezoid(density * (grid > 0), grid):.2f}"
            if path.exists():
                a1 = np.load(path)["theta"][:, 0, 0]
                ax.hist(a1, bins=bins, density=True, histtype="step", color=colour, lw=2)
                title += f"   arm {np.mean(a1 > 0):.2f}"
            else:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center")
            ax.set_title(title, fontsize=7.5)
            ax.set_yticks([])
            ax.set_xlim(-1.0, 1.0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if j == 0:
                ax.set_ylabel(label, fontsize=7.5, rotation=0, ha="right", va="center")
            if i == 0:
                ax.annotate(f"n = {n}", (0.5, 1.3), xycoords="axes fraction", ha="center", weight="bold")
            if i == len(arms) - 1:
                ax.set_xlabel("angle_1")
    fig.suptitle(f"IK, observations at the shoulder height (|x1 - h| < {DELTA:g}), true a1 = {A1_TRUE}, h = {H_TRUE}\n"
                 "black/grey = exact posterior; colour = composed sampler", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "flat_height.png", dpi=150, bbox_inches="tight")
    print("wrote", ROOT / "flat_height.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("data", "train", "run", "grade"))
    parser.add_argument("--variant", choices=("fixed", "global"))
    parser.add_argument("--methods", nargs="+", default=[m for m, _, _ in METHODS])
    parser.add_argument("--ns", type=int, nargs="+", default=list(NS))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--suffix", default="", help="appended to the runs/ folder name (e.g. _noproj)")
    arguments = parser.parse_args()
    {"data": data, "train": train, "run": run, "grade": grade}[arguments.action](arguments)


if __name__ == "__main__":
    main()
