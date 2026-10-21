#!/usr/bin/env python3
"""IK |x1| benchmark (exactly symmetric angle_1 posterior) with hierarchical attention,
with and without the residual output (``abs_hierarchy/precondition.py``).

Question: can a composed sampler recover the two angle_1 modes (+-0.51/0.62/0.64 at
n = 8/32/128)? The mean-field legacy checkpoint gave one bump at 0 under every
COMPASS rule. The rows are the recovery recipe's angle-layout rows with x1 -> |x1|
(as ``inverse_kinematics/experiments/train_abs_x1.py``); the trunk is the tokenized
h32d3wide of ``ik.py``. ``--residual`` writes the raw output as
``lam * s_prior + backbone`` with the exact Gaussian IK prior on all four latents.
Data and exact reference: ``Arruda/artifacts/ik/absx1/data_a0.6.npy`` and
``Arruda/artifacts/ik/exact/absx1_a0.6.npz`` (one nested 128-long dataset).

    python ik_absx1.py train [--residual]
    python ik_absx1.py run [--residual] --methods gauss_jacobian_FD arruda_direct
    python ik_absx1.py grade
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "abs_hierarchy")]
import ik  # noqa: E402  (CPU cap, autocvd unless CUDA_VISIBLE_DEVICES, layout plumbing)
import common  # noqa: E402

import numpy as np  # noqa: E402

H = ik.H
ARRUDA = common.COMPASS / "Arruda" / "artifacts" / "ik"
DATA = ARRUDA / "absx1" / "data_a0.6.npy"
EXACT = ARRUDA / "exact" / "absx1_a0.6.npz"
ROOT = HERE / "artifacts" / "ik_absx1"
NS = (8, 32, 128)

_forward = H.forward


def forward_abs_x1(theta_nodes):
    x = np.array(_forward(theta_nodes), copy=True)
    x[..., 0] = np.abs(x[..., 0])
    return x


H.forward = forward_abs_x1


def tag(residual):
    return "hierarchical_R" if residual else "hierarchical"


def model_dir(residual):
    return ROOT / "models" / f"tokenized_{ik.CONFIG}_angle_{tag(residual)}"


def prior():
    import layouts
    return [0.0] * 4, list(map(float, layouts.PRIOR_STD))


def wrap_loader(trainer, directory):
    import precondition
    load_attention = common.load_with_latent_attention(trainer.load_to)

    def load_to(checkpoint, device):
        network = load_attention(checkpoint, device)
        precondition.load(network, Path(checkpoint).parent, None)
        return network

    trainer.load_to = load_to


def train(arguments):
    layout = ik.set_layout("angle")
    import train_model as trainer
    import precondition
    from compass import hierarchical_roles
    from compass.experimental.latent_attention import enable_latent_attention, save_latent_attention

    directory = model_dir(arguments.residual)
    directory.mkdir(parents=True, exist_ok=True)
    original = trainer.build

    def build(config, arch, device="cpu"):
        model = original(config, arch, device=device)
        roles = hierarchical_roles(model.schema, layout.n_global, 4 - layout.n_global)
        enable_latent_attention(model, "hierarchical", roles)
        save_latent_attention(model, directory)
        config = precondition.attach(model, None, residual=arguments.residual, prior=prior())
        precondition.save(config, directory)
        return model

    trainer.build = build
    wrap_loader(trainer, directory)
    started = time.time()
    _, _, parameters = trainer.train_or_load(
        directory, ik.CONFIG, "tokenized", 200_000, "cuda", seed=7, menu="six_weighted",
        time_sampling="log_sigma")
    (directory / "run.json").write_text(json.dumps({
        "variant": "abs(x1)", "layout": "angle", "mode": "hierarchical", "residual": arguments.residual,
        "parameters": parameters, "minutes": (time.time() - started) / 60}, indent=2))


def run(arguments):
    layout = ik.set_layout("angle")
    import train_model as trainer
    wrap_loader(trainer, model_dir(arguments.residual))
    model = trainer.load_to(model_dir(arguments.residual) / "Model_checkpoint.pt", "cuda")
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    model.model.eval()
    data = np.load(DATA)
    folder = ROOT / "runs" / (tag(arguments.residual) + arguments.suffix)
    folder.mkdir(parents=True, exist_ok=True)
    for n in arguments.ns:
        for method in arguments.methods:
            path = folder / f"{method}_n{n}.npz"
            if path.exists():
                continue
            started = time.perf_counter()
            theta = ik.composed(model, layout, data[:n], method, arguments.draws, 2026 + n)
            a1 = theta[:, 0, 0]
            np.savez(path, theta=theta.astype(np.float32), seconds=time.perf_counter() - started)
            print(f"{tag(arguments.residual):15s} {method:18s} n={n:<3} P(a1>0) {np.mean(a1 > 0):.3f} "
                  f"E|a1| {np.abs(a1).mean():.3f} sd {a1.std():.3f}  "
                  f"{time.perf_counter() - started:.0f}s", flush=True)


def summary(a1, grid=None, density=None):
    """P(a1>0), median |a1| per branch, and the share of mass in |a1| < 0.2 (the valley)."""
    if density is not None:
        cdf = np.concatenate([[0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
        a1 = np.interp(np.random.default_rng(0).random(200_000), cdf / cdf[-1], grid)
    pos, neg = a1[a1 > 0], -a1[a1 < 0]
    med = lambda v: float(np.median(v)) if len(v) > 20 else float("nan")  # noqa: E731
    return {"P+": float(np.mean(a1 > 0)), "mode+": med(pos), "mode-": med(neg),
            "valley": float(np.mean(np.abs(a1) < 0.2)), "draws": a1}


def grade(arguments):
    from scipy.stats import wasserstein_distance
    exact = np.load(EXACT)
    rows = {}
    for n in NS:
        ref = summary(None, exact["grid"], exact[f"n{n}"])
        sd = ref["draws"].std()
        print(f"n={n:<3} exact                              P+ {ref['P+']:.2f}  |a1| +{ref['mode+']:.2f} "
              f"-{ref['mode-']:.2f}  valley {ref['valley']:.2f}")
        for net in ("hierarchical", "hierarchical_R"):
            for path in sorted((ROOT / "runs" / net).glob(f"*_n{n}.npz")):
                a1 = np.load(path)["theta"][:, 0, 0]
                s = summary(a1)
                w1 = wasserstein_distance(a1, ref["draws"]) / sd
                method = path.stem.rsplit("_n", 1)[0]
                rows[f"{net}/{method}/n{n}"] = {k: v for k, v in s.items() if k != "draws"} | {"w1_sd": float(w1)}
                print(f"n={n:<3} {net:15s} {method:18s} P+ {s['P+']:.2f}  |a1| +{s['mode+']:.2f} "
                      f"-{s['mode-']:.2f}  valley {s['valley']:.2f}  W1/sd {w1:.2f}")
    (ROOT / "grades.json").write_text(json.dumps(rows, indent=1))
    plot(exact)


def plot(exact):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [("hierarchical", "gauss_jacobian_FD", "hier. attn + gauss_jacobian_FD", "#eda100"),
            ("hierarchical", "arruda_direct", "hier. attn + Arruda (plain sum)", "#e87ba4"),
            ("hierarchical_R", "gauss_jacobian_FD", "R + gauss_jacobian_FD", "#2a78d6"),
            ("hierarchical_R", "arruda_direct", "R + Arruda (plain sum)", "#1baf7a")]
    fig, axes = plt.subplots(len(arms), len(NS), figsize=(3.4 * len(NS), 1.9 * len(arms)), squeeze=False,
                             sharex=True)
    grid = exact["grid"]
    bins = np.linspace(-1.5, 1.5, 121)
    for j, n in enumerate(NS):
        for i, (net, method, label, colour) in enumerate(arms):
            ax = axes[i, j]
            ax.fill_between(grid, exact[f"n{n}"], color="#e6e5df", lw=0)
            ax.plot(grid, exact[f"n{n}"], color="#1f1f1e", lw=1)
            path = ROOT / "runs" / net / f"{method}_n{n}.npz"
            if path.exists():
                a1 = np.load(path)["theta"][:, 0, 0]
                ax.hist(a1, bins=bins, density=True, histtype="step", color=colour, lw=2)
                ax.set_title(f"P(a1>0) {np.mean(a1 > 0):.2f}   valley {np.mean(np.abs(a1) < 0.2):.2f}",
                             fontsize=7.5)
            else:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center")
            ax.set_yticks([])
            ax.set_xlim(-1.5, 1.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if j == 0:
                ax.set_ylabel(label, fontsize=7.5, rotation=0, ha="right", va="center")
            if i == 0:
                ax.annotate(f"n = {n}", (0.5, 1.3), xycoords="axes fraction", ha="center", weight="bold")
            if i == len(arms) - 1:
                ax.set_xlabel("angle_1")
    fig.suptitle("IK with |x1| observed (exactly symmetric posterior, true angle_1 = 0.6)\n"
                 "black/grey = exact posterior; colour = composed sampler; valley = mass in |a1| < 0.2",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "absx1_residual.png", dpi=150, bbox_inches="tight")
    print("wrote", ROOT / "absx1_residual.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "run", "grade"))
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["gauss_jacobian_FD", "arruda_direct"])
    parser.add_argument("--ns", type=int, nargs="+", default=list(NS))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--suffix", default="", help="appended to the runs/ folder name (e.g. _noproj)")
    arguments = parser.parse_args()
    {"train": train, "run": run, "grade": grade}[arguments.action](arguments)


if __name__ == "__main__":
    main()
