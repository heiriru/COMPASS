#!/usr/bin/env python
"""Section 21 -- the six diagnostic figures, from the JSON artifacts.

Run after the experiment scripts:
    python ModeTree/experiments/figures.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import json    # noqa: E402
import math    # noqa: E402

import matplotlib                                 # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
import torch                                      # noqa: E402

from modetree import benchmarks as B                          # noqa: E402
from modetree.composition import make_composer                # noqa: E402
from modetree.diffusion import ComposedField, DiffusedOracle  # noqa: E402
from modetree.laplace import assemble                         # noqa: E402
from modetree.samplers import annealed_langevin, reverse_diffusion  # noqa: E402
from modetree.tree import ModeTree, ModeTreeConfig            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")
FIGURES = os.path.join(ARTIFACTS, "figures")
STYLE = {"figure.dpi": 140, "font.size": 9, "axes.grid": True,
         "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False}
COLOURS = {
    "uncorrected": "#9e9e9e", "gauss_jacobian": "#1f77b4", "fnpe": "#d62728",
    "modetree": "#2ca02c", "modetree_plus": "#17a15a",
    "modetree_known": "#8c564b",
}


def load(name):
    path = os.path.join(ARTIFACTS, name)
    if not os.path.exists(path):
        print(f"  (skipping: {name} not found)")
        return None
    with open(path) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------


def figure_1_posterior(case="map_vs_mass", n=32):
    """Exact posterior vs every method, with true and estimated modes marked."""
    model = B.map_vs_mass_1d(n) if case == "map_vs_mass" else B.bimodal_1d(n, case)
    ref = B.reference_1d(model, -8, 8, points=200_001)
    clean = make_composer(model)
    prior_std = model.prior_cov.diagonal().sqrt()
    generator = torch.Generator().manual_seed(3)

    draws = {}
    for rule, kind in (("gauss_jacobian", "diffusion"), ("fnpe", "langevin")):
        field = ComposedField(DiffusedOracle(model), rule)
        if kind == "diffusion":
            samples, _ = reverse_diffusion(field, 20000, levels=200,
                                           prior_std=prior_std, generator=generator)
        else:
            samples, _ = annealed_langevin(field, 20000, levels=100,
                                           steps_per_level=10, prior_std=prior_std,
                                           generator=generator)
        draws[rule] = samples[:, 0]

    tree = ModeTree(ComposedField(DiffusedOracle(model), "gauss_jacobian"),
                    clean, ModeTreeConfig(), prior_std=prior_std)
    branches = tree.run()
    posterior = assemble(clean, [b.diagnostics["clean_result"] for b in branches],
                         laplace_plus_samples=32)

    grid = ref.grid
    density = torch.zeros_like(grid)
    for weight, branch in zip(posterior.weights_plus, posterior.branches):
        mean = float(branch.global_state.reshape(()))
        sigma = float(branch.global_covariance.reshape(()).sqrt())
        density += float(weight) * torch.exp(
            -0.5 * ((grid - mean) / sigma) ** 2
        ) / (sigma * math.sqrt(2 * math.pi))

    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(7.0, 3.6))
        axis.plot(grid, ref.log_density.exp(), color="k", lw=2.0,
                  label="exact tall posterior")
        for rule in ("gauss_jacobian", "fnpe"):
            axis.hist(draws[rule].numpy(), bins=300, range=(-8, 8), density=True,
                      histtype="step", lw=1.2, color=COLOURS[rule], label=rule)
        axis.plot(grid, density, color=COLOURS["modetree"], lw=1.8, ls="--",
                  label="ModeTree-JAC + Laplace+")
        for mode in ref.modes.reshape(-1):
            axis.axvline(float(mode), color="k", ls=":", lw=0.8)
        for branch in posterior.branches:
            axis.axvline(float(branch.global_state), color=COLOURS["modetree"],
                         ls="--", lw=0.8)
        axis.axvline(float(posterior.map_global), color="#ff7f0e", lw=1.4,
                     label="ModeTree MAP")
        axis.set_xlim(-6, 6)
        axis.set_xlabel("global parameter g")
        axis.set_ylabel("density")
        axis.set_title(f"{case}, n = {n}: dotted = true modes, dashed = found")
        axis.legend(fontsize=7.5, loc="upper left")
        figure.tight_layout()
        figure.savefig(os.path.join(FIGURES, "figure1_posterior.png"))
        plt.close(figure)


def figure_2_versus_n(rows):
    """MAP error, mode-weight error, W1 and runtime against n."""
    cases = sorted({row["case"] for row in rows})
    panels = ("e_map_sigma", "weight_tv", "w1_best", "seconds")
    titles = ("MAP error / posterior sd", "mode-weight TV",
              "Wasserstein-1 to the exact posterior", "wall clock (s)")
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(len(panels), len(cases),
                                    figsize=(3.0 * len(cases), 2.3 * len(panels)),
                                    sharex=True, squeeze=False)
        for column, case in enumerate(cases):
            for panel, (key, title) in enumerate(zip(panels, titles)):
                axis = axes[panel][column]
                for method in COLOURS:
                    series = [(r["n"], r.get("w1_analytic", r.get("w1"))
                               if key == "w1_best" else r.get(key))
                              for r in rows
                              if r["case"] == case and r["method"] == method
                              and "error" not in r]
                    series = [(n, v) for n, v in series if v is not None]
                    if not series:
                        continue
                    series.sort()
                    axis.plot([s[0] for s in series],
                              [max(s[1], 1e-12) for s in series],
                              marker="o", ms=2.5, lw=1.1,
                              color=COLOURS[method], label=method)
                axis.set_xscale("log")
                axis.set_yscale("log")
                if panel == 0:
                    axis.set_title(case, fontsize=9)
                if column == 0:
                    axis.set_ylabel(title, fontsize=7.5)
                if panel == len(panels) - 1:
                    axis.set_xlabel("n")
        axes[0][-1].legend(fontsize=6.0, loc="best")
        figure.tight_layout()
        figure.savefig(os.path.join(FIGURES, "figure2_versus_n.png"))
        plt.close(figure)


def figure_3_mode_tree(case="symmetric", n=32):
    """Noise level (vertical) against mode position (horizontal)."""
    model = B.bimodal_1d(n, case)
    clean = make_composer(model)
    field = ComposedField(DiffusedOracle(model), "gauss_jacobian")
    tree = ModeTree(field, clean, ModeTreeConfig(),
                    prior_std=model.prior_cov.diagonal().sqrt())
    tree.run()
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(5.0, 4.0))
        series = {}
        for level in tree.trace:
            for branch in level["branches"]:
                series.setdefault(branch["id"], []).append(
                    (branch["g"][0], level["lambda"])
                )
        for branch_id, points in series.items():
            axis.plot([p[0] for p in points], [p[1] for p in points],
                      marker="o", ms=2.0, lw=1.2, label=f"branch {branch_id}")
        for event in tree.events:
            if event["event"] == "birth" and event["parent"] is not None:
                axis.scatter([event["position"][0]],
                             [field.oracle.schedule.lam(event["t"])],
                             marker="*", s=60, color="k", zorder=5)
        axis.set_yscale("log")
        axis.invert_yaxis()
        axis.set_xlabel("global parameter g")
        axis.set_ylabel("noise scale lambda(t)")
        axis.set_title(f"mode tree, {case}, n = {n} (stars: branch births)")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(os.path.join(FIGURES, "figure3_mode_tree.png"))
        plt.close(figure)


def figure_4_branch_weights(rows):
    """How posterior mass moves between modes as observations accumulate."""
    cases = sorted({row["case"] for row in rows})
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, len(cases), figsize=(2.9 * len(cases), 2.8),
                                    sharey=True, squeeze=False)
        for column, case in enumerate(cases):
            axis = axes[0][column]
            series = [(r["n"], r.get("weights"), r.get("weights_true"))
                      for r in rows
                      if r["case"] == case and r["method"] == "modetree_plus"
                      and r.get("weights")]
            series.sort()
            if not series:
                continue
            width = max(len(s[1]) for s in series)
            for b in range(width):
                axis.plot([s[0] for s in series],
                          [s[1][b] if b < len(s[1]) else 0.0 for s in series],
                          marker="o", ms=3, lw=1.3, label=f"branch {b}")
                axis.plot([s[0] for s in series],
                          [s[2][b] if b < len(s[2]) else 0.0 for s in series],
                          ls=":", lw=1.0, color="k")
            axis.set_xscale("log")
            axis.set_title(case, fontsize=9)
            axis.set_xlabel("n")
        axes[0][0].set_ylabel("mode weight (dotted: exact)")
        axes[0][-1].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(os.path.join(FIGURES, "figure4_branch_weights.png"))
        plt.close(figure)


def figure_5_runtime(rows):
    """Runtime against n for every method, and the ModeTree cost split."""
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
        for method in COLOURS:
            by_n = {}
            for row in rows:
                if row["method"] != method or "error" in row:
                    continue
                by_n.setdefault(row["n"], []).append(row["seconds"])
            if not by_n:
                continue
            ns = sorted(by_n)
            axes[0].plot(ns, [sum(by_n[n]) / len(by_n[n]) for n in ns],
                         marker="o", ms=3, color=COLOURS[method], label=method)
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("n")
        axes[0].set_ylabel("wall clock (s), mean over cases")
        axes[0].legend(fontsize=7)
        axes[0].set_title("inference cost")

        for method in ("gauss_jacobian", "fnpe", "modetree_plus"):
            by_n = {}
            for row in rows:
                if row["method"] != method or "error" in row:
                    continue
                by_n.setdefault(row["n"], []).append(row["score_field_calls"])
            if not by_n:
                continue
            ns = sorted(by_n)
            axes[1].plot(ns, [sum(by_n[n]) / len(by_n[n]) for n in ns],
                         marker="o", ms=3, color=COLOURS[method], label=method)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("n")
        axes[1].set_ylabel("composed-field evaluations")
        axes[1].set_title("network function evaluations")
        axes[1].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(os.path.join(FIGURES, "figure5_runtime.png"))
        plt.close(figure)


def figure_6_calibration(case="map_vs_mass", n=16, repeats=200):
    """SBC ranks and empirical coverage of the ModeTree posterior.

    Each repeat draws a fresh global truth from the prior, simulates ``n``
    factors centred on it, and ranks the truth inside the returned posterior.
    """
    torch.manual_seed(0)
    tau = 2.0
    ranks, coverage = [], {q: 0 for q in (0.5, 0.8, 0.9, 0.95)}
    draws = 200
    for repeat in range(repeats):
        # SBC is only valid when the truth is drawn from the *same* prior the
        # inference uses.  Drawing it from a narrower distribution than the
        # model's N(0, tau^2) makes the ranks non-uniform for a reason that has
        # nothing to do with the method.
        truth = float(torch.randn(1, dtype=torch.float64) * tau)
        factors = []
        for _ in range(n):
            centre = truth + float(torch.randn(1, dtype=torch.float64)) * 0.5
            factors.append(B._mix([0.5, 0.5], [centre, -centre], [0.7, 0.7]))
        model = B.FactorModel(
            torch.zeros(1, dtype=torch.float64),
            torch.tensor([[tau ** 2]], dtype=torch.float64), factors,
        )
        clean = make_composer(model)
        field = ComposedField(DiffusedOracle(model), "gauss_jacobian")
        try:
            tree = ModeTree(field, clean,
                            ModeTreeConfig(levels=20, probe_iterations=40,
                                           sentinel_checkpoints=2),
                            prior_std=torch.tensor([tau], dtype=torch.float64))
            branches = tree.run()
            posterior = assemble(
                clean, [b.diagnostics["clean_result"] for b in branches],
                laplace_plus_samples=16,
            )
            samples = posterior.sample(draws).reshape(-1)
        except Exception:
            continue
        ranks.append(float((samples < truth).to(torch.float64).mean()))
        ordered = samples.sort().values
        for q in coverage:
            lo = ordered[int((0.5 - q / 2) * draws)]
            hi = ordered[min(int((0.5 + q / 2) * draws), draws - 1)]
            coverage[q] += int(lo <= truth <= hi)
    n_used = max(len(ranks), 1)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        axes[0].hist(ranks, bins=20, range=(0, 1), color=COLOURS["modetree"],
                     alpha=0.8)
        axes[0].axhline(n_used / 20, color="k", ls="--", lw=1.0)
        axes[0].set_xlabel("SBC rank")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"SBC ranks ({n_used} datasets, n = {n})")
        nominal = sorted(coverage)
        empirical = [coverage[q] / n_used for q in nominal]
        axes[1].plot([0, 1], [0, 1], "k--", lw=1.0)
        axes[1].plot(nominal, empirical, marker="o", color=COLOURS["modetree"])
        axes[1].set_xlabel("nominal coverage")
        axes[1].set_ylabel("empirical coverage")
        axes[1].set_title("credible-interval coverage")
        figure.tight_layout()
        figure.savefig(os.path.join(FIGURES, "figure6_calibration.png"))
        plt.close(figure)
    return {"ranks": ranks,
            "coverage": {str(q): coverage[q] / n_used for q in coverage},
            "datasets": n_used, "n": n}


def main():
    os.makedirs(FIGURES, exist_ok=True)
    rows = load("compare_methods.json") or load("compare_methods_quick.json")
    print("figure 1 ...")
    figure_1_posterior()
    if rows:
        print("figure 2 ...")
        figure_2_versus_n(rows)
        print("figure 4 ...")
        figure_4_branch_weights(rows)
        print("figure 5 ...")
        figure_5_runtime(rows)
    print("figure 3 ...")
    figure_3_mode_tree()
    print("figure 6 ...")
    calibration = figure_6_calibration()
    with open(os.path.join(ARTIFACTS, "calibration.json"), "w") as handle:
        json.dump(calibration, handle, indent=1)
    print("wrote", FIGURES)


if __name__ == "__main__":
    main()
