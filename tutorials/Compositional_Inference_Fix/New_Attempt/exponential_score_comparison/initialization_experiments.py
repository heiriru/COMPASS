#!/usr/bin/env python3
"""Can predictor-only DPM2 be fixed by fixing where it starts?

``01_gauss_jacobian_analytic_predictor_only.png`` is 2.6 sigma wrong with an
*exact* single-observation score, and it does not move when the timestep count is
halved (2.598 at 100 steps, 2.606 at 50). A converged integrator carrying a fixed
bias is the signature of a wrong initial law: a probability-flow ODE returns the
pushforward of whatever it starts from, and has no mechanism to forget it.

``MultiObsSampler._initial_sample`` starts every latent at **zero** plus
``lambda_max`` noise (``_prepare_data`` builds ``joint_data`` as zeros and fills
only the observed slots). The exact diffused joint at ``lambda_max`` is centred on
the tall posterior instead: ``E[g|x] = 1.39`` for the shared coordinate, and
``(E[g|x] + x_j)/2`` -- between 1.5 and 4.2 -- for the locals. Relative to
``lambda_max = 3.89`` the locals are the badly placed coordinates, and there are
thirty of them feeding the shared one through the composition.

Five starts are compared, all with the exact row score, ``gauss_jacobian``, and no
correctors whatsoever, against the unmodified production start:

``A0_exact_start``      draws from ``hierarchy.sample_diffused(x, lambda_max)``,
                        i.e. the exact diffused joint. Not a method -- it is the
                        measurement that says how much of the 2.6 sigma is the
                        initial law at all, and how much is the rule.
``A1_observation``      locals at ``x_j - 1/rate``, the generative-model guess.
                        Uses nothing but the simulator; available to any problem.
``A1b_single_obs``      locals and shared at their exact *single-observation*
                        posterior means -- what a pilot run estimates, and what
                        the network could supply directly.
``A2_composed_warm``    one Tweedie step of the sampler's own composed field at
                        ``lambda_max``: ``mu = z + lambda^2 s(z)``, then renoise
                        around ``mu``. Self-consistent -- it can only introduce a
                        bias the rule already has -- and costs one evaluation.
``A3_wide_schedule``    VESDE ``sigma=25`` instead of 8, so ``lambda_max`` grows
                        3.89 -> 9.85. The Gaussian estimate of the initialization
                        error scales like ``m/lambda_max``, so this should cut it
                        ~2.5x. Free here only because an exact score needs no
                        retraining; the learned arm would need a new checkpoint.

Nothing in ``compass`` or in ``compare.py`` is modified: the initial-state
override is patched onto ``MultiObsSampler`` for the duration of one run and
restored afterwards, so every other script keeps its stock behaviour.

Usage:
    python initialization_experiments.py --device cuda
    python initialization_experiments.py --replot
"""
from __future__ import annotations

import os
import sys

CPU_THREAD_LIMIT = 3
os.sched_setaffinity(0, set(list(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(CPU_THREAD_LIMIT)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compass.MultiObsSampler import MultiObsSampler  # noqa: E402

import analytic_compare as analytic  # noqa: E402
import compare  # noqa: E402
import hierarchy  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

# Predictor only, throughout: the point is what a deterministic map can do.
SAMPLE_KWARGS = {
    "method": "dpm", "order": 2, "correction": "gauss_jacobian",
    "corrector_steps": 0, "final_corrector_steps": 0,
    "terminal_corrector_steps": 0,
}

NAVY, BLUE, TEAL, CORAL, GOLD = (
    compare.NAVY, compare.BLUE, compare.TEAL, compare.CORAL, compare.GOLD
)
MUTED = "#8A8F98"


# ---------------------------------------------------------------------------
# Initial-state overrides
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def patched_initial_sample(builder):
    """Swap ``MultiObsSampler._initial_sample`` for one run, then restore it.

    ``builder`` receives ``(sampler, data, condition_mask)`` -- ``data`` already
    holds the observed values with zeros in the latent slots -- and returns the
    initial state. Patching the class rather than editing it keeps every other
    caller of the sampler on stock behaviour.
    """
    original = MultiObsSampler._initial_sample
    MultiObsSampler._initial_sample = (
        lambda self, data, condition_mask: builder(self, data, condition_mask)
    )
    try:
        yield
    finally:
        MultiObsSampler._initial_sample = original


def _lambda_max(sampler, data):
    return float(sampler.sde.lambda_t(torch.ones(1, device=data.device)))


def _shared_noise(sampler, data):
    """The stock noise draw: shared coordinates synchronized across rows."""
    return sampler._shared_noise(data)


def production_start(sampler, data, condition_mask):
    """The unmodified sampler behaviour, restated so the baseline is explicit."""
    lam = _lambda_max(sampler, data)
    noise = _shared_noise(sampler, data)
    return data + lam * noise * (1 - condition_mask)


def exact_start(x, seed):
    """A0: draw from the exact diffused joint p_lam(g_t, l_t | x) at lambda_max."""
    def builder(sampler, data, condition_mask):
        lam = _lambda_max(sampler, data)
        samples = data.shape[1]
        shared, local, _, _ = hierarchy.sample_diffused(x, lam, samples, seed)
        state = data.clone()
        state[:, :, hierarchy.GLOBAL_INDEX] = torch.as_tensor(
            shared, dtype=data.dtype, device=data.device
        ).reshape(1, -1)
        state[:, :, hierarchy.LOCAL_INDEX] = torch.as_tensor(
            local.T.copy(), dtype=data.dtype, device=data.device
        )
        return state
    return builder


def observation_start(x):
    """A1: locals at ``x_j - 1/rate``; shared left at the prior mean."""
    centres = np.asarray(x, dtype=np.float64) - 1.0 / hierarchy.RATE

    def builder(sampler, data, condition_mask):
        lam = _lambda_max(sampler, data)
        noise = _shared_noise(sampler, data)
        centre = data.clone()
        centre[:, :, hierarchy.LOCAL_INDEX] = torch.as_tensor(
            centres, dtype=data.dtype, device=data.device
        ).reshape(-1, 1)
        centre[:, :, hierarchy.GLOBAL_INDEX] = hierarchy.MU_G
        return centre + lam * noise * (1 - condition_mask)
    return builder


def single_observation_start(x):
    """A1b: every latent at its exact single-observation posterior mean."""
    shared_means, local_means = [], []
    for value in np.asarray(x, dtype=np.float64):
        grid_g, weights_g, grid_l, weights_l = \
            hierarchy.single_observation_reference(value)
        shared_means.append(float((weights_g * grid_g).sum()))
        local_means.append(float((weights_l * grid_l).sum()))
    # The shared coordinate is one number shared by every row, so the per-row
    # estimates are averaged; this is what a pilot run reports.
    shared_mean = float(np.mean(shared_means))
    local_means = np.asarray(local_means)

    def builder(sampler, data, condition_mask):
        lam = _lambda_max(sampler, data)
        noise = _shared_noise(sampler, data)
        centre = data.clone()
        centre[:, :, hierarchy.GLOBAL_INDEX] = shared_mean
        centre[:, :, hierarchy.LOCAL_INDEX] = torch.as_tensor(
            local_means, dtype=data.dtype, device=data.device
        ).reshape(-1, 1)
        return centre + lam * noise * (1 - condition_mask)
    return builder


def prior_start(sampler, data, condition_mask):
    """B1: draw from the *diffused prior* instead of from N(0, lambda_max^2).

    The exact terminal law of a VE diffusion is N(theta, lambda_max^2)
    marginalized over the prior, i.e. N(prior_mean, prior_var + lambda_max^2).
    The stock initializer uses N(0, lambda_max^2): right variance, and a mean of
    zero that is the prior mean only by coincidence for the shared coordinate and
    plain wrong for the locals, whose prior mean is 1.0 here.

    Everything used here is already an argument of ``sample()`` -- ``prior`` and
    ``local_prior`` -- which the initializer currently ignores. No pilot, no
    reference, no extra evaluation.
    """
    lam = _lambda_max(sampler, data)
    noise = _shared_noise(sampler, data)
    centre = data.clone()
    spread = torch.ones_like(data)

    shared_mean = float(sampler.prior_mean.reshape(-1)[0])
    shared_std = float(sampler.prior_std.reshape(-1)[0])
    centre[:, :, hierarchy.GLOBAL_INDEX] = shared_mean
    spread[:, :, hierarchy.GLOBAL_INDEX] = (shared_std**2 + lam**2) ** 0.5

    if getattr(sampler, "local_prior_mean", None) is not None:
        local_mean = float(sampler.local_prior_mean.reshape(-1)[0])
        local_std = float(sampler.local_prior_std.reshape(-1)[0])
        centre[:, :, hierarchy.LOCAL_INDEX] = local_mean
        spread[:, :, hierarchy.LOCAL_INDEX] = (local_std**2 + lam**2) ** 0.5

    return centre + spread * noise * (1 - condition_mask)


def composed_warm_start(sampler, data, condition_mask):
    """A2: one Tweedie step of the composed field at lambda_max, then renoise.

    For the VESDE the denoised mean is ``E[x_0 | z] = z + lambda^2 s(z, lambda)``,
    so a single evaluation of the *composed* score at the schedule maximum gives
    an estimate of where the tall posterior actually sits. Renoising around that
    centre keeps the initial variance the exact diffused joint has at lambda_max
    (``lambda^2 + Var[posterior] ~= lambda^2``) while removing the mean error,
    which is the term the deterministic map cannot forget.
    """
    lam = _lambda_max(sampler, data)
    time = sampler.timesteps_list[0].reshape(1, 1).to(data.device)
    draw = data + lam * _shared_noise(sampler, data) * (1 - condition_mask)
    indices = torch.arange(data.shape[0], device=data.device)
    score = sampler._get_score(draw, time, condition_mask, indices)
    centre = draw + lam**2 * score * (1 - condition_mask)
    # The shared coordinate must stay identical across rows.
    centre[:, :, hierarchy.GLOBAL_INDEX] = centre[:1, :, hierarchy.GLOBAL_INDEX]
    centre = torch.where(condition_mask > 0.5, data, centre)
    return centre + lam * _shared_noise(sampler, data) * (1 - condition_mask)


# ---------------------------------------------------------------------------
# The experiment table
# ---------------------------------------------------------------------------

def variants(x, seed):
    return {
        "baseline_production_start": {
            "label": "Production start (unmodified)",
            "note": "every latent at zero + lambda_max noise",
            "builder": production_start, "sde": None, "colour": MUTED,
        },
        "A0_exact_start": {
            "label": "A0: exact diffused start",
            "note": "draws from p_lam(g_t, l_t | x) at lambda_max",
            "builder": exact_start(x, seed + 4242), "sde": None, "colour": NAVY,
        },
        "A1_observation_start": {
            "label": "A1: locals at x_j - 1/rate",
            "note": "generative-model guess; no reference used",
            "builder": observation_start(x), "sde": None, "colour": BLUE,
        },
        "A1b_single_observation_start": {
            "label": "A1b: single-observation posterior means",
            "note": "what a pilot run estimates",
            "builder": single_observation_start(x), "sde": None, "colour": TEAL,
        },
        "A2_composed_warm_start": {
            "label": "A2: composed-field Tweedie warm start",
            "note": "one evaluation of the rule's own field at lambda_max",
            "builder": composed_warm_start, "sde": None, "colour": CORAL,
        },
        "B1_prior_start": {
            "label": "B1: diffused-prior start",
            "note": "N(prior_mean, prior_var + lambda_max^2); already passed to sample()",
            "builder": prior_start, "sde": None, "colour": "#7F4FC9",
        },
        "VP_variance_preserving": {
            "label": "VP: variance-preserving SDE",
            "note": "stock start; alpha(1) = 0.0066 so the terminal law forgets the data",
            "builder": production_start, "sde": "vpsde", "colour": "#0F8B8D",
        },
        "A3_wide_schedule": {
            "label": "A3: lambda_max 3.89 -> 9.85 (sigma 25)",
            "note": "production start, wider VESDE",
            "builder": production_start, "sde": "vesde25", "colour": GOLD,
        },
    }


def run_variant(name, entry, problem, base_model, arguments):
    from compass.SDE import VESDE

    model = base_model
    if entry["sde"] is not None:
        if arguments.score != "exact":
            raise SystemExit(
                f"variant {name!r} changes the SDE, which a checkpoint trained "
                "under a different one cannot serve; run it with --score exact."
            )
        from compass.SDE import VPSDE
        sde = VPSDE() if entry["sde"] == "vpsde" else VESDE(sigma=25.0)
        model = analytic.AnalyticModel(
            sde, device=arguments.device,
            nodes=arguments.quadrature_nodes, seed=arguments.seed,
        )
    with patched_initial_sample(entry["builder"]):
        samples, sampler, runtime = compare.draw(
            model, problem, dict(SAMPLE_KWARGS), arguments.num_samples,
            arguments.timesteps, arguments.seed, arguments.denoise_clamp,
        )
    result = compare.kde_map_pipeline(model, problem, samples, arguments)
    metrics = compare.evaluate(problem, result, sampler, runtime, [])
    metrics["method"] = name
    return result, metrics


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_summary(problem, results, metrics, table, output):
    order = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fig.suptitle(
        "Predictor-only DPM2 + gauss_jacobian: how much of the error is the "
        "initial law?\n(exact single-observation score, no correctors anywhere, "
        "100 steps, 3,000 draws)",
        fontsize=13.5, fontweight="bold", color=NAVY,
    )

    axes[0].plot(problem["global_grid"], problem["global_density"], color="black",
                 lw=2.2, label="exact", zorder=6)
    for name in order:
        axes[0].hist(results[name]["kept_shared"], bins=70, density=True,
                     histtype="step", lw=1.7, color=table[name]["colour"],
                     label=table[name]["label"])
    axes[0].axvline(problem["x"].min(), color=NAVY, ls="-.", lw=1.3)
    axes[0].set_xlim(problem["global_mean"] - 8 * problem["global_std"],
                     problem["global_mean"] + 3 * problem["global_std"])
    axes[0].set(xlabel="global parameter g", ylabel="density",
                title="Shared posterior")
    axes[0].legend(fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    positions = np.arange(len(order))
    values = [metrics[name]["shared_w1_over_sigma"] for name in order]
    axes[1].barh(positions, values, color=[table[name]["colour"] for name in order],
                 edgecolor="white", linewidth=1.2)
    for position, value in zip(positions, values):
        axes[1].text(value * 1.08, position, f"{value:.3f}", va="center",
                     fontsize=8.5, color=NAVY)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels([table[name]["label"] for name in order], fontsize=7.8)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.03, 12.0)
    axes[1].axvline(0.106, color=MUTED, ls=":", lw=1.4)
    axes[1].text(0.106, len(order) - 0.4, " integrator floor", fontsize=7.5,
                 color=MUTED, va="center")
    axes[1].set(xlabel="W1 of the shared marginal / posterior σ",
                title="Accuracy")
    axes[1].grid(axis="y", visible=False)
    axes[1].invert_yaxis()

    width = [metrics[name]["shared_width_ratio"] for name in order]
    axes[2].barh(positions, width, color=[table[name]["colour"] for name in order],
                 edgecolor="white", linewidth=1.2)
    for position, value in zip(positions, width):
        axes[2].text(value + 0.03, position, f"{value:.2f}", va="center",
                     fontsize=8.5, color=NAVY)
    axes[2].axvline(1.0, color=NAVY, ls=":", lw=1.4)
    axes[2].set_yticks(positions)
    axes[2].set_yticklabels([])
    axes[2].set(xlabel="width ratio (1 is correct)", title="Dispersion")
    axes[2].grid(axis="y", visible=False)
    axes[2].invert_yaxis()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


FIELDS = ["method", "label", "seconds", "calls", "shared_w1_over_sigma",
          "shared_mean_error_sigma", "shared_width_ratio", "local_w1_over_sigma",
          "global_map", "global_map_error_sigma",
          "local_map_error_vs_posterior_mean_sigma"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=compare.OBSERVATIONS)
    parser.add_argument("--num-samples", type=int, default=compare.NUM_SAMPLES)
    parser.add_argument("--timesteps", type=int, default=compare.TIMESTEPS)
    parser.add_argument("--quadrature-nodes", type=int, default=385)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--denoise-clamp", type=float, default=5.0)
    parser.add_argument("--excursion-sigma", type=float, default=15.0)
    parser.add_argument("--kde-bandwidth", default=None)
    parser.add_argument("--map-timesteps", type=int, default=200)
    parser.add_argument("--map-iterations", type=int, default=3)
    parser.add_argument("--map-eps", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score", default="exact", choices=["exact", "learned"],
                        help="exact row score, or the trained checkpoint.")
    parser.add_argument("--config", default="h16d2")
    parser.add_argument("--train-samples", type=int, default=200_000)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--replot", action="store_true")
    arguments = parser.parse_args()

    from compass.SDE import VESDE

    compare.configure_style()
    _, _, x = hierarchy.observations(arguments.observations, arguments.seed)
    table = variants(x, arguments.seed)
    selected = arguments.variants or list(table)

    if arguments.score == "exact":
        base_model = analytic.AnalyticModel(
            VESDE(sigma=compare.recipe.SDE_KWARGS["sigma"]), device=arguments.device,
            nodes=arguments.quadrature_nodes, seed=arguments.seed,
        )
        compare.recipe.train_or_load = lambda *a, **k: (base_model, 0.0)
        problem = compare.build_problem(
            None, "analytic", 0, arguments.observations, arguments.seed,
            arguments.device,
        )
    else:
        problem = compare.build_problem(
            compare.recipe.model_directory(
                ARTIFACTS, arguments.config, arguments.train_samples
            ),
            arguments.config, arguments.train_samples, arguments.observations,
            arguments.seed, arguments.device,
        )
        base_model = problem["model"]
    tag = "" if arguments.score == "exact" else "_learned"
    print(f"true g = {problem['global_truth']:.5f}   exact posterior mean "
          f"{problem['global_mean']:.5f}, sd {problem['global_std']:.5f}, "
          f"wall {problem['x'].min():.5f}")

    results, metrics, rows = {}, {}, []
    for name in selected:
        entry = table[name]
        folder = arguments.artifacts / f"initialization_{name}{tag}"
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / f"{name}.npz"

        if arguments.replot:
            with np.load(archive, allow_pickle=True) as stored:
                loaded = {key: stored[key] for key in stored.files}
            loaded["global_map"] = float(loaded["global_map"])
            loaded["excluded"] = int(loaded["excluded"])
            results[name] = loaded
            metrics[name] = json.loads(str(loaded["metrics_json"]))
            rows.append({
                "method": name, "label": entry["label"],
                "seconds": metrics[name]["sample_seconds"],
                "calls": metrics[name]["sampler_network_calls"],
                **{key: metrics[name][key] for key in FIELDS[4:]},
            })
        else:
            print(f"\n=== {entry['label']} ===")
            result, metric = run_variant(
                name, entry, problem, base_model, arguments
            )
            print(f"  {metric['sample_seconds']:.1f}s  "
                  f"{metric['sampler_network_calls']} calls  "
                  f"W1 {metric['shared_w1_over_sigma']:.4f}  "
                  f"width {metric['shared_width_ratio']:.3f}")
            scalar = {key: value for key, value in metric.items()
                      if np.isscalar(value) or isinstance(value, str)}
            np.savez_compressed(
                archive,
                x=problem["x"], global_grid=problem["global_grid"],
                global_density=problem["global_density"],
                global_truth=problem["global_truth"],
                exact_local_mean=problem["local_mean"],
                exact_local_std=problem["local_std"],
                metrics_json=json.dumps(scalar, default=float),
                score_rows_json="[]",
                score_grid=metric["score_grid"],
                implied_score=metric["implied_score"],
                exact_score=metric["exact_score"],
                **result,
            )
            results[name], metrics[name] = result, metric
            rows.append({
                "method": name, "label": entry["label"],
                "seconds": metric["sample_seconds"],
                "calls": metric["sampler_network_calls"],
                **{key: metric[key] for key in FIELDS[4:]},
            })

        # Each variant keeps the standard four-panel layout in its own folder.
        compare.METHODS[name] = {
            "label": entry["label"], "colour": entry["colour"],
            "note": entry["note"],
            "score_note": "exact single-observation score, predictor only",
            "sample_kwargs": dict(SAMPLE_KWARGS),
        }
        compare.plot_method(problem, name, results[name], metrics[name],
                            folder / f"01_{name}.png")

    # Merge by variant rather than overwrite, so running one variant does not
    # drop the others -- the same contract compare.write_metrics uses.
    path = arguments.artifacts / f"initialization_experiments{tag}.csv"
    merged = {}
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                merged[row["method"]] = row
    for row in rows:
        merged[row["method"]] = row
    if merged:
        order = list(variants(x, arguments.seed))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for key in sorted(merged, key=lambda item: (order + [item]).index(item)):
                writer.writerow({f: merged[key].get(f, "") for f in FIELDS})
        print(f"Wrote {path}")

    plot_summary(problem, results, metrics, table,
                 arguments.artifacts / f"05_initialization_experiments{tag}.png")


if __name__ == "__main__":
    main()
