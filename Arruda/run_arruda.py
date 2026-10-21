#!/usr/bin/env python3
"""Faithful BayesFlow/Arruda compositional-diffusion runs for COMPASS.

The implementation follows the v2.0.13 Compositional Diffusion tutorial:
one-observation posterior diffusion training, then `compositional_sample` with
the exact prior score and adaptive two-step integration.  It intentionally does
not use a combined analytic posterior score at sampling time: that would test a
different oracle method rather than the Arruda amortized method.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


CPU_THREAD_LIMIT = 3


def configure_cpu_limit() -> None:
    """Cap this executable at CPU_THREAD_LIMIT logical CPUs (see COMPASS/CLAUDE.md)."""
    available = sorted(os.sched_getaffinity(0))
    chosen = set(available[:CPU_THREAD_LIMIT])
    if not chosen:
        raise RuntimeError("No CPU cores available for the CPU thread cap.")
    os.sched_setaffinity(0, chosen)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = str(len(chosen))
    print(f"CPU cap: {len(chosen)}/{os.cpu_count() or len(available)} logical CPUs; affinity={sorted(chosen)}", flush=True)


configure_cpu_limit()
os.environ.setdefault("KERAS_BACKEND", "torch")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

import bayesflow as bf

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "Gaussian_mixtures_compositional_score_modelling"), str(ROOT / "inverse_kinematics")]

from Gaussian_mixtures_compositional_score_modelling import datasets as gm_datasets
from Gaussian_mixtures_compositional_score_modelling import model as gm_model
import hierarchical as ik_model
import bimodality_study as ik_study


@dataclass(frozen=True)
class RunRecord:
    benchmark: str
    n: int
    bridge: float
    mean: float
    std: float
    wasserstein_over_reference_std: float
    samples_path: str
    method: str = ""        # label; empty in runs recorded before --methods existed


def method_label(prior: str, bridge: float) -> str:
    base = f"Arruda adaptive, d={bridge:g}"
    return base if prior == "clean" else f"{base}, diffused prior"


def record_label(row: "RunRecord") -> str:
    return row.method or method_label("clean", row.bridge)


def make_approximator() -> bf.CompositionalApproximator:
    adapter = bf.CompositionalApproximator.build_adapter(inference_variables="g", inference_conditions="x")
    network = bf.networks.DiffusionModel(subnet="time_mlp", noise_schedule="edm")
    approximator = bf.CompositionalApproximator(inference_network=network, adapter=adapter)
    approximator.compile(optimizer="adam")
    return approximator


def gm_single_observation_data(count: int, seed: int, mixture: gm_model.Mixture) -> dict[str, np.ndarray]:
    rows = mixture.simulate(count, np.random.default_rng(seed))
    return {"g": rows[:, :1].astype("float32"), "x": rows[:, 2:3].astype("float32")}


def ik_single_observation_data(count: int, seed: int) -> dict[str, np.ndarray]:
    theta, x = ik_model.simulate(count, np.random.default_rng(seed))
    return {"g": theta[:, :1].detach().cpu().numpy().astype("float32"), "x": x.detach().cpu().numpy().astype("float32")}


def artifact_dir(args: argparse.Namespace) -> Path:
    """One folder per trained model, so a new GM setting never reuses or overwrites old weights."""
    output = ROOT / "Arruda" / "artifacts" / args.benchmark
    return output / args.tag if args.tag else output


def train_or_load(args: argparse.Namespace, mixture: gm_model.Mixture | None) -> bf.CompositionalApproximator:
    output = artifact_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    weights = (Path(args.weights_dir) if args.weights_dir else output) / "single_observation.weights.h5"
    if args.weights_dir and not weights.exists():
        raise FileNotFoundError(f"--weights-dir given but {weights} does not exist; refusing to retrain into it.")
    approximator = make_approximator()
    if args.benchmark == "gm":
        data = gm_single_observation_data(args.train_samples, args.seed, mixture)
    else:
        data = ik_single_observation_data(args.train_samples, args.seed)
    dataset = bf.datasets.OfflineDataset(data=data, batch_size=args.batch_size, adapter=approximator.adapter)
    if weights.exists() and not args.retrain:
        # A zero-epoch fit initializes shapes and adapts the standardizer before loading.
        approximator.fit(dataset=dataset, epochs=0, verbose=0)
        approximator.load_weights(weights)
        print(f"Loaded single-observation posterior from {weights}", flush=True)
    else:
        print(f"Training {args.benchmark} local diffusion posterior: {args.train_samples} simulations, "
              f"batch={args.batch_size}, epochs={args.epochs}", flush=True)
        approximator.fit(dataset=dataset, epochs=args.epochs, verbose=2)
        approximator.save_weights(weights)
        (output / "training_config.json").write_text(json.dumps({
            "train_samples": args.train_samples, "batch_size": args.batch_size,
            "epochs": args.epochs, "seed": args.seed, "backend": "torch",
            "bayesflow": bf.__version__,
            "gm_mu0": args.gm_mu0 if args.benchmark == "gm" else None,
            "gm_sigma_x": args.gm_sigma_x if args.benchmark == "gm" else None,
        }, indent=2) + "\n")
    return approximator


GM_HIERARCHICAL_MAX_N = 128


def gm_tall_problem(n: int, seed: int, mixture: gm_model.Mixture, regime: str,
                    g_true: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if regime == "hierarchical":
        # The generative model itself with one fixed g: l_i ~ N(g, tau_l^2), x_i ~ N(l_i, sigma_x^2).
        # Nested: every n takes the first n of one sequence, so panels show the same data accumulating.
        if n > GM_HIERARCHICAL_MAX_N:
            raise ValueError(f"hierarchical regime draws at most {GM_HIERARCHICAL_MAX_N} observations")
        rng = np.random.default_rng(seed)
        local = g_true + mixture.tau_l * rng.standard_normal(GM_HIERARCHICAL_MAX_N)
        x = (local + mixture.sigma_x * rng.standard_normal(GM_HIERARCHICAL_MAX_N))[:n].reshape(n, 1)
    else:
        dataset = gm_datasets.make_dataset(regime, n, np.random.default_rng(seed + n), mixture=mixture, branch_mass=0.75)
        x = np.asarray(dataset["x"]).reshape(n, -1)
    span = mixture.mu_0 + 6.0 * mixture.tau_g
    grid = np.linspace(-span, span, 2_001)
    exact = mixture.global_density(grid, x)
    exact /= np.trapezoid(exact, grid)
    return x.astype("float32"), grid, exact


IK_TRUE_ANGLE = 0.6
IK_HIERARCHICAL_MAX_N = 128


IK_DATA_FILE: Path | None = None     # set from --ik-data for the "file" regime
IK_REFERENCE_FILE: Path | None = None  # --ik-reference: precomputed (grid, n{n}) reference for that data


def ik_tall_problem(n: int, seed: int, regime: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``selected``: n observations of angle_1 = 0.6 kept where |x1| < 0.15.

    ``mirror_paired``: ceil(n/2) such observations plus their reflections
    ``(x1, x2) -> (-x1, x2)``. The reflection ``(h, a1, a2, a3) -> (-h, -a1, -a2, -a3)``
    maps ``x1 -> -x1`` and leaves every (zero-mean) prior invariant, so
    ``p(x | a) = p(Mx | -a)`` and a mirror-closed dataset has an exactly symmetric
    posterior at every n. An odd n keeps one unpaired observation with x1 set to 0,
    which is its own mirror. Symmetrising the ABC reference is then exact in
    expectation and removes its Monte Carlo asymmetry.
    """
    rng = np.random.default_rng(seed + n)
    if regime == "file":
        # A saved dataset (e.g. one found by searching natural draws); nested: the first n rows.
        x = np.load(IK_DATA_FILE)[:n]
    elif regime == "hierarchical":
        # The generative model itself with one fixed angle_1: every observation draws its own
        # (height, angle_2, angle_3) from the prior; no selection. Nested like the GM regime.
        rng = np.random.default_rng(seed)
        locals_ = rng.normal(loc=ik_model.LOCAL_PRIOR_MEAN, scale=ik_model.LOCAL_PRIOR_STD, size=(IK_HIERARCHICAL_MAX_N, 3))
        theta = np.concatenate([np.full((IK_HIERARCHICAL_MAX_N, 1), IK_TRUE_ANGLE), locals_], axis=1)
        x = ik_model.forward(theta)[:n]
    elif regime == "selected":
        x = ik_study.selected_dataset(n, true_angle=IK_TRUE_ANGLE, epsilon=0.15, rng=rng)
    else:
        half = ik_study.selected_dataset((n + 1) // 2, true_angle=IK_TRUE_ANGLE, epsilon=0.15, rng=rng)
        mirrored = half * np.array([-1.0, 1.0])
        if n % 2:
            half[-1, 0] = 0.0
            mirrored = mirrored[:-1]
        x = np.concatenate([half, mirrored])
    if IK_REFERENCE_FILE is not None:
        stored = np.load(IK_REFERENCE_FILE)
        grid, exact = stored["grid"], stored[f"n{n}"].copy()
    else:
        grid, exact = ik_model.reference_global(x, grid=np.linspace(-2.0, 2.0, 1_401), draws=80_000, seed=seed + n,
                                                symmetrize=regime == "mirror_paired")
    exact /= np.trapezoid(exact, grid)
    return x.astype("float32"), grid, exact


def gm_prior_score(mixture: gm_model.Mixture):
    means = mixture.components.astype("float32")
    weights = np.asarray(mixture.weights, dtype="float32")
    variance = float(mixture.tau_g ** 2)
    def score(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        g = values["g"]
        log_weight = np.log(weights)[None, :] - 0.5 * ((g - means[None, :]) ** 2 / variance)
        responsibilities = np.exp(log_weight - log_weight.max(axis=-1, keepdims=True))
        responsibilities /= responsibilities.sum(axis=-1, keepdims=True)
        return {"g": (responsibilities * (means[None, :] - g) / variance).sum(axis=-1, keepdims=True)}
    return score


def ik_prior_score(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"g": -values["g"] / (0.5 ** 2)}


def diffused_prior_score(approximator: bf.CompositionalApproximator, centres, variance: float, weights):
    """Exact score of the *diffused* Gaussian-mixture prior at the sampler's noise level.

    BayesFlow calls a prior score without a ``time`` argument as (1 - t) * clean score at the
    noisy point; with ``time`` it uses our value directly. BayesFlow hands us the un-standardised
    z_t (= z_t * s + m) and multiplies our return value by s, so we compute the score in
    standardised coordinates and divide by s. The EDM schedule is variance preserving: a prior
    component N(c, w) in standardised units diffuses to N(alpha c, alpha^2 w + sigma^2).
    """
    import keras.ops as ops
    network = approximator.inference_network
    layer = approximator.standardizer.standardize_layers["inference_variables"]
    m = float(ops.convert_to_numpy(layer.moving_mean[0]).ravel()[0])
    s = float(ops.convert_to_numpy(layer.moving_std(0)).ravel()[0])
    c = ((np.asarray(centres, dtype="float64") - m) / s)[None, :]
    w = variance / s ** 2
    log_w = np.log(np.asarray(weights, dtype="float64"))[None, :]

    def score(values: dict[str, np.ndarray], time) -> dict[str, np.ndarray]:
        log_snr = network.noise_schedule.get_log_snr(t=ops.convert_to_tensor(time, "float32"), training=False)
        alpha, sigma = network.noise_schedule.get_alpha_sigma(log_snr_t=log_snr)
        alpha = np.asarray(ops.convert_to_numpy(alpha), dtype="float64").reshape(-1, 1)
        sigma = np.asarray(ops.convert_to_numpy(sigma), dtype="float64").reshape(-1, 1)
        z = (np.asarray(ops.convert_to_numpy(values["g"]), dtype="float64").reshape(-1, 1) - m) / s
        var = alpha ** 2 * w + sigma ** 2
        diff = alpha * c - z
        logits = log_w - 0.5 * diff ** 2 / var - 0.5 * np.log(var)
        resp = np.exp(logits - logits.max(axis=1, keepdims=True)); resp /= resp.sum(axis=1, keepdims=True)
        return {"g": ((resp * diff / var).sum(axis=1, keepdims=True) / s).astype("float32")}
    return score


def sample_one(approximator: bf.CompositionalApproximator, x: np.ndarray, prior_score, args: argparse.Namespace, bridge: float) -> np.ndarray:
    if len(x) == 1:
        # Composition needs >= 2 observations; with one it is the trained posterior itself (no bridge).
        samples = approximator.sample(num_samples=args.num_samples, conditions={"x": x})
        return np.asarray(samples["g"]).reshape(-1)
    samples = approximator.compositional_sample(
        num_samples=args.num_samples,
        conditions={"x": x[None, ...]},
        compute_prior_score=prior_score,
        method="two_step_adaptive",
        steps="adaptive",
        mini_batch_size=10,
        compositional_bridge_d0=bridge,
        compositional_bridge_d1=bridge,
        seed=args.seed,
    )
    return np.asarray(samples["g"])[0, :, 0]


def truth_lines(args: argparse.Namespace, mixture: gm_model.Mixture | None) -> tuple[list[float], str, str]:
    """Where the data came from, for the dashed reference line(s) on each panel."""
    if args.benchmark == "ik":
        if args.ik_regime == "mirror_paired":
            return [-IK_TRUE_ANGLE, IK_TRUE_ANGLE], "true angle_1 (+0.6 half, mirrored -0.6 half)", "--"
        if args.ik_regime == "file":
            return ([args.ik_true_angle], f"true angle_1 = {args.ik_true_angle:.3f}", "--") if args.ik_true_angle is not None else ([], "", "--")
        return [IK_TRUE_ANGLE], f"true angle_1 = {IK_TRUE_ANGLE}", "--"
    if args.gm_regime == "hierarchical":
        return [args.gm_true_g], f"true g = {args.gm_true_g:g}", "--"
    # The constructed GM regimes have no single true g; mark the prior modes instead.
    return [-mixture.mu_0, mixture.mu_0], f"prior modes +-{mixture.mu_0:g} (no true g: data constructed)", ":"


def plot_runs(records: list[RunRecord], curves: dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]], benchmark: str, output: Path,
              truth: tuple[list[float], str, str], title: str | None = None) -> None:
    ns = sorted(curves)
    columns = 4 if len(ns) > 5 else len(ns)
    rows = int(np.ceil(len(ns) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows), sharey=False, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    palette = ["#e76f51", "#2a9d8f", "#3a86ff", "#8338ec"]
    labels = list(dict.fromkeys(label for _, _, sampled in curves.values() for label in sampled))
    colors = {label: palette[k % len(palette)] for k, label in enumerate(labels)}
    for axis, n in zip(axes, ns):
        grid, exact, sampled = curves[n]
        axis.plot(grid, exact, color="black", lw=2.2, label="exact posterior")
        for label, values in sampled.items():
            # Normalise by *all* samples: mass outside the grid is lost, not silently redistributed.
            bins = np.linspace(grid[0], grid[-1], 81)
            outside = float(np.mean((values < grid[0]) | (values > grid[-1])))
            axis.hist(values, bins=bins, weights=np.full(len(values), 1.0 / (len(values) * (bins[1] - bins[0]))),
                      histtype="step", lw=1.6, color=colors[label], label=label)
            if outside > 0.005:
                axis.text(0.02, 0.97 - 0.07 * labels.index(label), f"{outside:.0%} off-plot", transform=axis.transAxes,
                          color=colors[label], fontsize=7, va="top")
        for k, value in enumerate(truth[0]):
            axis.axvline(value, color="0.35", ls=truth[2], lw=1.3, label=truth[1] if k == 0 else None)
        axis.set(title=f"n = {n}", xlabel="global parameter g")
        axis.grid(alpha=.2)
    for axis in axes[len(ns):]:
        axis.set_visible(False)
    for axis in axes[::columns]:
        axis.set_ylabel("density")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(names), fontsize=9, frameon=False)
    fig.suptitle(title or f"{benchmark.upper()}: BayesFlow Arruda compositional diffusion", y=1.08)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_title(args: argparse.Namespace, mixture: gm_model.Mixture | None) -> str | None:
    if args.benchmark == "ik" and args.ik_regime == "hierarchical":
        return ("IK hierarchical model: angle_1 ~ N(0, 0.5^2) shared; (height, angle_2, angle_3) ~ prior per observation; "
                f"data from one true angle_1 = {IK_TRUE_ANGLE:g}; BayesFlow Arruda sampler")
    if args.benchmark == "gm" and args.gm_regime == "hierarchical":
        return (f"GM hierarchical model: g ~ 0.5 N(-{mixture.mu_0:g}, {mixture.tau_g:g}^2) + 0.5 N(+{mixture.mu_0:g}, {mixture.tau_g:g}^2), "
                f"l_i ~ N(g, {mixture.tau_l:g}^2), x_i ~ N(l_i, {mixture.sigma_x:g}^2); data from one true g = {args.gm_true_g:g}; "
                "BayesFlow Arruda sampler")
    return None


def replot(args: argparse.Namespace, mixture: gm_model.Mixture | None, output: Path) -> None:
    records = [RunRecord(**row) for row in json.loads((output / "results.json").read_text())]
    stored = np.load(output / "reference.npz") if (output / "reference.npz").exists() else None
    curves: dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
    for n in sorted({row.n for row in records}):
        if stored is not None and f"n{n}" in stored:
            grid, exact = stored["grid"], stored[f"n{n}"]
        else:
            _, grid, exact = gm_tall_problem(n, args.seed, mixture, args.gm_regime, args.gm_true_g) if args.benchmark == "gm" else ik_tall_problem(n, args.seed, args.ik_regime)
            print(f"recomputed reference n={n}", flush=True)
        curves[n] = (grid, exact, {record_label(row): np.load(row.samples_path) for row in records if row.n == n})
    np.savez(output / "reference.npz", grid=next(iter(curves.values()))[0],
             **{f"n{n}": exact for n, (_, exact, _) in curves.items()})
    plot_runs(records, curves, args.benchmark, output / "posterior_comparison.png", truth_lines(args, mixture),
              plot_title(args, mixture))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("gm", "ik"), required=True)
    parser.add_argument("--stage", choices=("train", "sample", "all", "plot"), default="all",
                        help="plot: redraw from saved samples (recomputes the reference, no network)")
    parser.add_argument("--observations", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--train-samples", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--num-samples", type=int, default=4_000)
    parser.add_argument("--gm-mu0", type=float, default=7.5, help="separated prior mode for a visibly bimodal tall posterior")
    parser.add_argument("--gm-sigma-x", type=float, default=0.8, help="observation noise; large values keep a symmetric tall dataset inside the prior")
    parser.add_argument("--gm-regime", choices=(*gm_datasets.REGIMES, "hierarchical"), default="asymmetric")
    parser.add_argument("--gm-true-g", type=float, default=1.2, help="the shared g of the hierarchical regime")
    parser.add_argument("--methods", nargs="+", default=["clean:1", "clean:0.1"],
                        help="prior:bridge pairs; prior is 'clean' (BayesFlow's (1-t) clean score) or 'diffused' (exact)")
    parser.add_argument("--ik-regime", choices=("selected", "mirror_paired", "hierarchical", "file"), default="selected")
    parser.add_argument("--ik-data", default="", help="(n_max, 2) .npy of observations for --ik-regime file")
    parser.add_argument("--ik-reference", default="", help="precomputed reference .npz (grid, n1..n128) for --ik-data")
    parser.add_argument("--ik-true-angle", type=float, default=None, help="true angle_1 of --ik-data, for the dashed line")
    parser.add_argument("--weights-dir", default="", help="load trained weights from here instead of the output folder")
    parser.add_argument("--tag", default="", help="artifact subfolder for this model setting")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    global IK_DATA_FILE, IK_REFERENCE_FILE
    IK_DATA_FILE = Path(args.ik_data) if args.ik_data else None
    IK_REFERENCE_FILE = Path(args.ik_reference) if args.ik_reference else None
    mixture = None
    if args.benchmark == "gm":
        mixture = gm_model.Mixture(mu_0=args.gm_mu0, sigma_x=args.gm_sigma_x, weights=(0.5, 0.5))
    output = artifact_dir(args)
    if args.stage == "plot":
        replot(args, mixture, output)
        return
    approximator = train_or_load(args, mixture)
    if args.stage == "train":
        return
    curves: dict[int, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
    records: list[RunRecord] = []
    clean_prior = gm_prior_score(mixture) if args.benchmark == "gm" else ik_prior_score
    exact_prior = (diffused_prior_score(approximator, mixture.components, mixture.tau_g ** 2, mixture.weights)
                   if args.benchmark == "gm" else diffused_prior_score(approximator, [0.0], 0.5 ** 2, [1.0]))
    methods = [(prior_kind, float(bridge)) for prior_kind, bridge in (item.split(":") for item in args.methods)]
    for n in args.observations:
        x, grid, exact = gm_tall_problem(n, args.seed, mixture, args.gm_regime, args.gm_true_g) if args.benchmark == "gm" else ik_tall_problem(n, args.seed, args.ik_regime)
        reference_draws = np.random.default_rng(args.seed + 10 * n).choice(grid, size=50_000, p=exact / exact.sum())
        sampled: dict[str, np.ndarray] = {}
        for prior_kind, bridge in methods:
            values = sample_one(approximator, x, exact_prior if prior_kind == "diffused" else clean_prior, args, bridge)
            label = method_label(prior_kind, bridge)
            sampled[label] = values
            suffix = "" if prior_kind == "clean" else f"_{prior_kind}"
            path = output / f"samples_n{n}_bridge{bridge:g}{suffix}.npy"
            np.save(path, values)
            records.append(RunRecord(args.benchmark, n, bridge, float(values.mean()), float(values.std()),
                                     float(wasserstein_distance(values, reference_draws) / reference_draws.std()), str(path),
                                     label))
        curves[n] = (grid, exact, sampled)
    np.savez(output / "reference.npz", grid=next(iter(curves.values()))[0],
             **{f"n{n}": exact for n, (_, exact, _) in curves.items()})
    (output / "results.json").write_text(json.dumps([asdict(item) for item in records], indent=2) + "\n")
    plot_runs(records, curves, args.benchmark, output / "posterior_comparison.png", truth_lines(args, mixture),
              plot_title(args, mixture))
    print(json.dumps([asdict(item) for item in records], indent=2), flush=True)


if __name__ == "__main__":
    main()
