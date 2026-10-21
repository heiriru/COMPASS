#!/usr/bin/env python
"""Experiment 1 -- where the Gaussian-mixture composed field loses the data.

No retraining. This measures, against the benchmark's exact diffused score,
which *part* of the first-order composition the trained network gets wrong.

The object under test is the rule COMPASS actually composes with,

    S_n(g) = (1-n) s_prior,t(g) + sum_i s_{i,g}(g, l_i, x_i, t)
           =      s_prior,t(g) + sum_i [ s_{i,g} - s_prior,t ]
           =      s_prior,t(g) + sum_i Delta s_i ,

verified against ``ComposedField(rule="uncorrected")`` in
``modetree/diffusion.py`` (lines 257-260), which is what the sampler consumes.

Two distinctions drive the whole script.

*Posterior score vs likelihood residual.* The network is trained on
``s_post``; composition only ever uses ``Delta s = s_post - s_prior``. If
``|s_post| >> |Delta s|`` then a small relative error on the trained target is
a large relative error on the used one. Part VIII of the task asks for exactly
those magnitude ratios, and they are reported here pointwise and posterior-
weighted.

*Network error vs rule error.* Both are available in closed form here, so the
composed field is graded twice:

    S_n,phi  vs  S_n^*        (identical rule, oracle rows)   -> network error
    S_n^*    vs  d/dg log p_t (the exact tall diffused score)  -> rule error

Evaluation slice
----------------
Everything is evaluated at ``l_i(g) = a g + b x_i``, the *exact* conditional
mode of ``p(l_i | g, x_i)``. Three reasons: it is closed form, it is shared by
both mixture branches (the locals are conditionally independent of the
component given ``g``), and because ``Var(l_i | g, x_i) = q`` does not depend on
``g``, the Occam factor is constant and the profile log posterior equals the
marginal one up to an additive constant. Integrating the global score along
this slice is therefore exact for the clean target, which makes the oracle arm
a control rather than another approximation. Learned and oracle arms are read
at the *same* states, so no profiling difference can leak into the comparison.

Run:  python ModeTree/experiments/exp1_gm_residual_error.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "Gaussian_mixtures_compositional_score_modelling"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BENCHMARK))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402
import time      # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

import datasets   # noqa: E402
import model as M  # noqa: E402
import oracle     # noqa: E402
import train      # noqa: E402

from compass.priors import GaussianMixturePrior       # noqa: E402
from modetree.compass_adapter import CompassRowSource  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "gm_residual"
CHECKPOINT = (BENCHMARK / "artifacts" / "models"
              / "tokenized_h32d3wide_n200000_six_weighted_log_sigma"
              / "Model_checkpoint.pt")

# lambda, not t: lambda is the quantity every statement below is about, and the
# VE schedule's t is a nearly logarithmic relabelling of it.
SINGLE_LAMBDAS = (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
LADDER_LAMBDAS = (0.5, 0.2, 0.1, 0.05, 0.03167, 0.02, 0.01)
# Trainer.eps = 1e-3 is the smallest diffusion time ever drawn, so the network
# has never seen a noise scale below lambda(1e-3) = 0.03167 for sigma = 15.
# Anything under that is extrapolation and is labelled as such rather than
# quietly averaged in with the rest.
TRAINED_LAMBDA_FLOOR = 0.03167
N_LADDER = (1, 2, 4, 8, 16, 32, 64, 128)
SEED = 2026


def mixture_prior(mixture):
    return GaussianMixturePrior(
        [mixture.weights[0], mixture.weights[1]],
        torch.tensor([[-mixture.mu_0], [mixture.mu_0]], dtype=torch.float64),
        torch.tensor([[[mixture.tau_g ** 2]], [[mixture.tau_g ** 2]]],
                     dtype=torch.float64),
    )


def sources(mixture, device="cpu"):
    if not CHECKPOINT.exists():
        raise SystemExit(f"no trained checkpoint at {CHECKPOINT}")
    learned = train.load_to(CHECKPOINT, device)
    exact = oracle.OracleModel(mixture, sigma=train.SDE_KWARGS["sigma"],
                               device=device)
    return learned, exact


def slice_state(mixture, grid, x):
    """``(len(grid), n, 2)`` states ``(g, a g + b x_i)``."""
    grid = torch.as_tensor(grid, dtype=torch.float64)
    x = torch.as_tensor(np.asarray(x, dtype=np.float64).ravel())
    locals_ = mixture.a * grid.unsqueeze(-1) + mixture.b * x.unsqueeze(0)
    globals_ = grid.unsqueeze(-1).expand_as(locals_)
    return torch.stack([globals_, locals_], dim=-1)


def global_row_scores(row_source, state, lam):
    """The ``g`` component of every row score: ``(len(grid), n)``."""
    return row_source.row_scores(state, lam)[..., 0]


# ---------------------------------------------------------------------------
# exact references
# ---------------------------------------------------------------------------

def exact_tall_global_score(mixture, grid, x, lam):
    """``d/dg log p_t(g, l(g) | x_1..n)`` for the true diffused tall posterior.

    The clean tall posterior is an exact two-component Gaussian mixture in
    ``n + 1`` dimensions (``model.branch_covariance`` / ``branch_mean_vector``);
    the diffusion adds ``lam^2 I`` to it, so this stays closed form. The rule
    error of the first-order composition is the gap between this and ``S_n^*``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    grid = np.asarray(grid, dtype=np.float64)
    covariance = mixture.branch_covariance(n) + (lam ** 2) * np.eye(n + 1)
    factor = np.linalg.cholesky(covariance)
    log_determinant = 2.0 * np.log(np.diag(factor)).sum()
    values = np.concatenate(
        [grid[:, None], mixture.a * grid[:, None] + mixture.b * x[None, :]],
        axis=1,
    )                                                   # (points, n + 1)
    weights = mixture.branch_weights(x)
    terms, directions = [], []
    for index, weight in enumerate(weights):
        delta = values - mixture.branch_mean_vector(x, index)
        solved = np.linalg.solve(covariance, delta.T).T
        quadratic = (delta * solved).sum(axis=1)
        terms.append(np.log(weight) - 0.5 * (log_determinant + quadratic))
        directions.append(-solved[:, 0])
    stacked = np.stack(terms)
    stacked -= stacked.max(axis=0, keepdims=True)
    responsibility = np.exp(stacked)
    responsibility /= responsibility.sum(axis=0, keepdims=True)
    return (responsibility * np.stack(directions)).sum(axis=0)


def diffused_global_posterior(mixture, grid, x, lam):
    """``p_t(g | x_1..n)`` exactly: the clean mixture widened by ``lam^2``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    weights, means, variance = mixture.global_posterior(x)
    grid = np.asarray(grid, dtype=np.float64)
    total = variance + lam ** 2
    density = np.zeros_like(grid)
    for weight, mean in zip(weights, means):
        density += weight * np.exp(-0.5 * (grid - mean) ** 2 / total) / np.sqrt(
            2.0 * np.pi * total)
    return density


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def integrate(grid, score):
    """Normalised density implied by ``d/dg log p = score``."""
    log_p = np.concatenate([[0.0], np.cumsum(
        0.5 * (score[1:] + score[:-1]) * np.diff(grid))])
    log_p -= log_p.max()
    density = np.exp(log_p)
    area = np.trapezoid(density, grid)
    return density / area if area > 0 else density


def w1(grid, density, reference):
    def cdf(values):
        out = np.concatenate([[0.0], np.cumsum(
            0.5 * (values[1:] + values[:-1]) * np.diff(grid))])
        return out / max(out[-1], 1e-300)
    return float(np.trapezoid(np.abs(cdf(density) - cdf(reference)), grid))


def roots_of(grid, score):
    """Maxima of the implied density: downward zero crossings of the score."""
    found = []
    sign = np.sign(score)
    for index in np.nonzero((sign[:-1] > 0) & (sign[1:] <= 0))[0]:
        left, right = grid[index], grid[index + 1]
        a, b = score[index], score[index + 1]
        found.append(float(left + (right - left) * a / (a - b)))
    return found


def summarise(values, weights=None):
    values = np.asarray(values, dtype=np.float64).ravel()
    if weights is None:
        return {"mean": float(values.mean()), "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)),
                "max": float(values.max())}
    weights = np.asarray(weights, dtype=np.float64).ravel()
    weights = weights / weights.sum()
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return {
        "mean": float((weights * values).sum()),
        "median": float(values[order][np.searchsorted(cumulative, 0.5)]),
        "p90": float(values[order][min(np.searchsorted(cumulative, 0.9),
                                       values.size - 1)]),
        "max": float(values.max()),
    }


# ---------------------------------------------------------------------------
# Part I.1, I.2 and VIII -- one observation at a time
# ---------------------------------------------------------------------------

def single_observation(mixture, learned, exact, prior, observations, grid,
                       lambdas):
    rows, fields = [], {}
    for x in observations:
        data = np.array([[x]], dtype=np.float64)
        learned_source = CompassRowSource(
            learned, data, M.HIERARCHY, M.LOCAL_INDICES, M.OBSERVED_INDICES,
            prior)
        exact_source = CompassRowSource(
            exact, data, M.HIERARCHY, M.LOCAL_INDICES, M.OBSERVED_INDICES,
            prior)
        state = slice_state(mixture, grid, [x])
        for lam in lambdas:
            s_learned = global_row_scores(learned_source, state, lam).numpy()[:, 0]
            s_exact = global_row_scores(exact_source, state, lam).numpy()[:, 0]
            s_prior = mixture.diffused_prior_score(grid, lam ** 2)
            delta_exact = s_exact - s_prior
            delta_learned = s_learned - s_prior
            # Delta s is the *only* thing composition uses; the error on it is
            # identical to the error on the posterior score, but the signal it
            # sits on is not.
            error = s_learned - s_exact
            weight = diffused_global_posterior(mixture, grid, [x], lam)
            scale = np.abs(delta_exact)
            floor = 1e-3 * max(scale.max(), 1e-12)
            rows.append({
                "x": float(x), "lambda": float(lam),
                "t": float(learned_source.schedule.time_of_lambda(lam)),
                "rms_posterior_score": float(np.sqrt((error ** 2).mean())),
                "rms_posterior_score_weighted": float(np.sqrt(
                    (weight * error ** 2).sum() / weight.sum())),
                "rms_signal_posterior": float(np.sqrt((s_exact ** 2).mean())),
                "rms_signal_residual": float(np.sqrt((delta_exact ** 2).mean())),
                "rms_signal_prior": float(np.sqrt((s_prior ** 2).mean())),
                "snr_posterior": float(np.sqrt((s_exact ** 2).mean()
                                               / max((error ** 2).mean(), 1e-300))),
                "snr_residual": float(np.sqrt((delta_exact ** 2).mean()
                                              / max((error ** 2).mean(), 1e-300))),
                "snr_residual_weighted": float(np.sqrt(
                    (weight * delta_exact ** 2).sum()
                    / max((weight * error ** 2).sum(), 1e-300))),
                "relative_residual": summarise(np.abs(error) / (scale + floor)),
                "relative_residual_weighted": summarise(
                    np.abs(error) / (scale + floor), weight),
                "cancellation_posterior": summarise(
                    np.abs(s_exact) / (scale + floor), weight),
                "cancellation_prior": summarise(
                    np.abs(s_prior) / (scale + floor), weight),
                "sign_agreement": float(np.mean(
                    np.sign(delta_learned) == np.sign(delta_exact))),
                "sign_agreement_weighted": float(
                    (weight * (np.sign(delta_learned)
                               == np.sign(delta_exact))).sum() / weight.sum()),
                "cosine_residual": float(
                    (delta_learned * delta_exact).sum()
                    / max(np.linalg.norm(delta_learned)
                          * np.linalg.norm(delta_exact), 1e-300)),
            })
            fields[f"{x:+.2f}|{lam:.4g}"] = {
                "s_learned": s_learned.tolist(), "s_exact": s_exact.tolist(),
                "s_prior": s_prior.tolist(), "weight": weight.tolist(),
            }
            print(f"  x={x:+5.2f} lam={lam:6.3g}  |s_post|={rows[-1]['rms_signal_posterior']:8.3f}"
                  f"  |ds|={rows[-1]['rms_signal_residual']:8.3f}"
                  f"  err={rows[-1]['rms_posterior_score']:8.3f}"
                  f"  SNR_post={rows[-1]['snr_posterior']:7.2f}"
                  f"  SNR_res={rows[-1]['snr_residual']:7.2f}"
                  f"  cancel={rows[-1]['cancellation_posterior']['median']:7.2f}",
                  flush=True)
    return rows, fields


# ---------------------------------------------------------------------------
# Part I.3 -- the n ladder
# ---------------------------------------------------------------------------

def ladder(mixture, learned, exact, prior, grid, lambdas, observations):
    rows, fields = [], {}
    for n in observations:
        rng = np.random.default_rng(
            [SEED, datasets.REGIMES.index("asymmetric"), n, 0])
        data = datasets.make_dataset("asymmetric", n, rng, mixture=mixture,
                                     branch_mass=0.75, spread=1.0)
        x = np.asarray(data["x"], dtype=np.float64).ravel()
        column = x.reshape(-1, 1)
        learned_source = CompassRowSource(
            learned, column, M.HIERARCHY, M.LOCAL_INDICES, M.OBSERVED_INDICES,
            prior)
        exact_source = CompassRowSource(
            exact, column, M.HIERARCHY, M.LOCAL_INDICES, M.OBSERVED_INDICES,
            prior)
        state = slice_state(mixture, grid, x)
        clean = mixture.global_density(grid, x)
        reference_modes = np.sort(mixture.branch_means(x))
        posterior_sd = mixture.global_moments(x)[1]
        for lam in lambdas:
            started = time.perf_counter()
            s_learned = global_row_scores(learned_source, state, lam).numpy()
            s_exact = global_row_scores(exact_source, state, lam).numpy()
            s_prior = mixture.diffused_prior_score(grid, lam ** 2)
            composed_learned = (1 - n) * s_prior + s_learned.sum(axis=1)
            composed_exact = (1 - n) * s_prior + s_exact.sum(axis=1)
            truth = exact_tall_global_score(mixture, grid, x, lam)
            diffused = diffused_global_posterior(mixture, grid, x, lam)
            error = composed_learned - composed_exact
            rule_error = composed_exact - truth
            density_learned = integrate(grid, composed_learned)
            density_exact = integrate(grid, composed_exact)
            density_truth = integrate(grid, truth)
            modes_learned = roots_of(grid, composed_learned)
            modes_exact = roots_of(grid, composed_exact)
            near = np.array([np.abs(grid - m).argmin() for m in reference_modes])
            row = {
                "n": int(n), "lambda": float(lam),
                "t": float(learned_source.schedule.time_of_lambda(lam)),
                "rms_network_error": float(np.sqrt((error ** 2).mean())),
                "rms_network_error_weighted": float(np.sqrt(
                    (diffused * error ** 2).sum() / diffused.sum())),
                "rms_rule_error": float(np.sqrt((rule_error ** 2).mean())),
                "rms_composed_signal": float(np.sqrt((composed_exact ** 2).mean())),
                "relative_network_error": float(
                    np.sqrt((error ** 2).mean()
                            / max((composed_exact ** 2).mean(), 1e-300))),
                "cosine": float((composed_learned * composed_exact).sum()
                                / max(np.linalg.norm(composed_learned)
                                      * np.linalg.norm(composed_exact), 1e-300)),
                "sign_agreement": float(np.mean(
                    np.sign(composed_learned) == np.sign(composed_exact))),
                "error_at_modes": np.abs(error[near]).tolist(),
                "modes_learned": modes_learned,
                "modes_exact": modes_exact,
                "modes_reference": reference_modes.tolist(),
                "posterior_sd": float(posterior_sd),
                "w1_learned": w1(grid, density_learned, clean),
                "w1_exact_rule": w1(grid, density_exact, clean),
                "w1_truth": w1(grid, density_truth, clean),
                "w1_learned_vs_diffused": w1(grid, density_learned, diffused),
                "extrapolated": bool(lam < TRAINED_LAMBDA_FLOOR),
                "seconds": time.perf_counter() - started,
            }
            row["w1_learned_over_sd"] = row["w1_learned"] / posterior_sd
            row["w1_exact_rule_over_sd"] = row["w1_exact_rule"] / posterior_sd
            if len(modes_learned) and len(reference_modes):
                row["mode_error"] = float(np.max([
                    np.min(np.abs(np.array(modes_learned) - m))
                    for m in reference_modes]))
                row["mode_recall"] = float(np.mean([
                    np.min(np.abs(np.array(modes_learned) - m)) < 0.5 * posterior_sd
                    + 0.25 for m in reference_modes]))
            else:
                row["mode_error"] = None
                row["mode_recall"] = 0.0
            rows.append(row)
            fields[f"{n}|{lam:.4g}"] = {
                "composed_learned": composed_learned.tolist(),
                "composed_exact": composed_exact.tolist(),
                "truth": truth.tolist(),
                "density_learned": density_learned.tolist(),
                "clean": clean.tolist(),
            }
            print(f"  n={n:4d} lam={lam:6.3g}  |S*|={row['rms_composed_signal']:9.2f}"
                  f"  net_err={row['rms_network_error']:9.2f}"
                  f"  rule_err={row['rms_rule_error']:9.2f}"
                  f"  rel={row['relative_network_error']:6.3f}"
                  f"  W1/sd={row['w1_learned_over_sd']:7.3f}"
                  f"  (rule {row['w1_exact_rule_over_sd']:6.3f})"
                  f"  modes={len(modes_learned)}  {row['seconds']:5.1f}s",
                  flush=True)
    return rows, fields


def fit_exponent(rows, key):
    """``||e_n|| ~ n^alpha`` by least squares in log-log, per lambda."""
    out = {}
    for lam in sorted({row["lambda"] for row in rows}):
        selected = [row for row in rows if row["lambda"] == lam
                    and row[key] > 0]
        if len(selected) < 3:
            continue
        n = np.log([row["n"] for row in selected])
        e = np.log([row[key] for row in selected])
        slope, intercept = np.polyfit(n, e, 1)
        residual = e - (slope * n + intercept)
        out[f"{lam:.4g}"] = {
            "alpha": float(slope), "intercept": float(intercept),
            "r2": float(1.0 - residual.var() / max(e.var(), 1e-300)),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=1201)
    parser.add_argument("--ladder-points", type=int, default=801)
    parser.add_argument("--observations", type=int, nargs="+", default=N_LADDER)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LADDER_LAMBDAS)
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-ladder", action="store_true")
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mixture = M.DEFAULT
    learned, exact = sources(mixture)
    prior = mixture_prior(mixture)
    print(f"checkpoint {CHECKPOINT.parent.name}, "
          f"{sum(p.numel() for p in learned.model.parameters()):,} parameters")

    if not args.skip_single:
        grid = np.linspace(-6.0, 6.0, args.points)
        targets = [-3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.5]
        print("\n== Part I.1/I.2/VIII: single observation ==")
        rows, fields = single_observation(mixture, learned, exact, prior,
                                          targets, grid, SINGLE_LAMBDAS)
        with open(ARTIFACTS / "single_rows.json", "w") as handle:
            json.dump(rows, handle, indent=1)
        with open(ARTIFACTS / "single_fields.json", "w") as handle:
            json.dump({"grid": grid.tolist(), "fields": fields}, handle)

    if not args.skip_ladder:
        grid = np.linspace(-6.0, 6.0, args.ladder_points)
        print("\n== Part I.3: the n ladder ==")
        rows, fields = ladder(mixture, learned, exact, prior, grid,
                              args.lambdas, args.observations)
        scaling = {key: fit_exponent(rows, key) for key in
                   ("rms_network_error", "rms_network_error_weighted",
                    "rms_rule_error", "rms_composed_signal")}
        with open(ARTIFACTS / "ladder_rows.json", "w") as handle:
            json.dump(rows, handle, indent=1)
        with open(ARTIFACTS / "ladder_scaling.json", "w") as handle:
            json.dump(scaling, handle, indent=1)
        with open(ARTIFACTS / "ladder_fields.json", "w") as handle:
            json.dump({"grid": grid.tolist(), "fields": fields}, handle)
        print("\n  ||e_n|| ~ n^alpha")
        for key, value in scaling.items():
            for lam, fit in value.items():
                print(f"    {key:28s} lam={lam:>6s}  alpha={fit['alpha']:+.3f}"
                      f"  r2={fit['r2']:.3f}")

    print(f"\nwrote {ARTIFACTS}")


if __name__ == "__main__":
    main()
