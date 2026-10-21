#!/usr/bin/env python
"""Experiment 1b -- is the GM row-score error a coverage or a capacity failure?

``exp1_gm_residual_error.py`` reads the network along a fixed analytic slice,
``l_i(g) = a g + b x_i``. That is the right slice for grading the composed
field, but it cannot distinguish two very different diagnoses:

* **coverage** -- the network is accurate on the states its training
  distribution actually produced, and the composed field drags it somewhere
  else;
* **capacity/objective** -- the network is inaccurate even on its own training
  distribution, and composition merely amplifies that ``n``-fold.

So this evaluates the same error at states drawn from two distributions, with
no grid and no slice anywhere:

``training``  ``(g, l, x)`` from the prior predictive joint, then ``(g, l)``
              diffused by ``lambda`` -- *exactly* the law of the states the
              denoising objective ever scored;
``posterior`` ``(g, l_1..l_n)`` from the exact diffused tall posterior
              ``p_t(. | x_1..n)`` -- the law of the states a correct sampler
              visits at that noise level.

Both are exact draws from closed forms, so the only approximation in the
comparison is the network itself. The oracle supplies the reference score at
the same states.

Reported per ``lambda``: the RMS error of the row score, of the likelihood
residual ``Delta s = s_post - s_prior``, and the residual's signal-to-error
ratio -- the quantity composition actually consumes.

Run:  python ModeTree/experiments/exp1b_gm_indistribution.py
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

import numpy as np  # noqa: E402
import torch        # noqa: E402

import datasets    # noqa: E402
import model as M  # noqa: E402

from exp1_gm_residual_error import (  # noqa: E402
    ARTIFACTS, SEED, TRAINED_LAMBDA_FLOOR, mixture_prior, sources,
)
from modetree.compass_adapter import CompassRowSource  # noqa: E402

LAMBDAS = (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.03167, 0.02, 0.01)


def training_states(mixture, rng, count, lam):
    """``(g, l)`` diffused by ``lam``, with the ``x`` that generated them."""
    rows = mixture.simulate(count, rng)
    latent = rows[:, M.LATENT_INDICES] + lam * rng.standard_normal((count, 2))
    return latent, rows[:, M.OBSERVED_INDICES]


def posterior_states(mixture, rng, count, x, lam):
    """Exact draws from ``p_t(g, l_1..l_n | x_1..n)``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    covariance = mixture.branch_covariance(n) + (lam ** 2) * np.eye(n + 1)
    factor = np.linalg.cholesky(covariance)
    weights = mixture.branch_weights(x)
    component = rng.random(count) < weights[1]
    means = np.stack([mixture.branch_mean_vector(x, index) for index in (0, 1)])
    draw = means[component.astype(int)] + rng.standard_normal(
        (count, n + 1)) @ factor.T
    return draw


def row_error(learned_source, exact_source, state, lam, mixture):
    """RMS errors of the row score and of the likelihood residual."""
    s_learned = learned_source.row_scores(state, lam).numpy()[..., 0]
    s_exact = exact_source.row_scores(state, lam).numpy()[..., 0]
    s_prior = mixture.diffused_prior_score(state[..., 0].numpy(), lam ** 2)
    delta = s_exact - s_prior
    delta_learned = s_learned - s_prior
    error = s_learned - s_exact
    return {
        "rms_error": float(np.sqrt((error ** 2).mean())),
        "rms_score": float(np.sqrt((s_exact ** 2).mean())),
        "rms_residual": float(np.sqrt((delta ** 2).mean())),
        "snr_score": float(np.sqrt((s_exact ** 2).mean()
                                   / max((error ** 2).mean(), 1e-300))),
        "snr_residual": float(np.sqrt((delta ** 2).mean()
                                      / max((error ** 2).mean(), 1e-300))),
        "cosine_residual": float(
            (delta_learned * delta).sum()
            / max(np.linalg.norm(delta_learned) * np.linalg.norm(delta),
                  1e-300)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mixture = M.DEFAULT
    learned, exact = sources(mixture)
    prior = mixture_prior(mixture)
    rng = np.random.default_rng([SEED, 7])

    dataset_rng = np.random.default_rng(
        [SEED, datasets.REGIMES.index("asymmetric"), args.n, 0])
    data = datasets.make_dataset("asymmetric", args.n, dataset_rng,
                                 mixture=mixture, branch_mass=0.75, spread=1.0)
    x_tall = np.asarray(data["x"], dtype=np.float64).ravel()

    rows = []
    print(f"n={args.n}, {args.draws} draws per cell "
          f"(lambda < {TRAINED_LAMBDA_FLOOR} is outside the trained range)")
    print(f"{'lam':>8} {'regime':>10} | {'|s|':>8} {'|ds|':>8} {'err':>8} "
          f"{'SNR_s':>7} {'SNR_ds':>7} {'cos_ds':>7}")
    for lam in args.lambdas:
        # training regime: one observation per state, its own x
        latent, x_own = training_states(mixture, rng, args.draws, lam)
        state = torch.as_tensor(latent, dtype=torch.float64).unsqueeze(0)
        learned_source = CompassRowSource(
            learned, x_own, M.HIERARCHY, M.LOCAL_INDICES, M.OBSERVED_INDICES,
            prior)
        exact_source = CompassRowSource(
            exact, x_own, M.HIERARCHY, M.LOCAL_INDICES, M.OBSERVED_INDICES,
            prior)
        # One state per observation, not every state against every observation:
        # the training distribution pairs each diffused (g, l) with the single x
        # it generated. Shape (1, draws, 2) makes the row axis do exactly that,
        # in one call of `draws` forwards rather than `draws**2`.
        result = row_error(learned_source, exact_source, state, lam, mixture)
        result.update({"lambda": float(lam), "regime": "training",
                       "extrapolated": bool(lam < TRAINED_LAMBDA_FLOOR)})
        rows.append(result)
        print(f"{lam:8.4g} {'training':>10} | {result['rms_score']:8.3f} "
              f"{result['rms_residual']:8.3f} {result['rms_error']:8.3f} "
              f"{result['snr_score']:7.2f} {result['snr_residual']:7.2f} "
              f"{result['cosine_residual']:7.3f}", flush=True)

        # posterior regime: shared g, the tall dataset's own x
        draw = posterior_states(mixture, rng, args.draws // 8, x_tall, lam)
        state = torch.stack([
            torch.as_tensor(np.stack([np.full(args.n, row[0]), row[1:]], axis=1))
            for row in draw])                            # (S, n, 2)
        learned_source = CompassRowSource(
            learned, x_tall.reshape(-1, 1), M.HIERARCHY, M.LOCAL_INDICES,
            M.OBSERVED_INDICES, prior)
        exact_source = CompassRowSource(
            exact, x_tall.reshape(-1, 1), M.HIERARCHY, M.LOCAL_INDICES,
            M.OBSERVED_INDICES, prior)
        result = row_error(learned_source, exact_source, state, lam, mixture)
        result.update({"lambda": float(lam), "regime": "posterior",
                       "extrapolated": bool(lam < TRAINED_LAMBDA_FLOOR)})
        rows.append(result)
        print(f"{lam:8.4g} {'posterior':>10} | {result['rms_score']:8.3f} "
              f"{result['rms_residual']:8.3f} {result['rms_error']:8.3f} "
              f"{result['snr_score']:7.2f} {result['snr_residual']:7.2f} "
              f"{result['cosine_residual']:7.3f}", flush=True)

    with open(ARTIFACTS / "indistribution_rows.json", "w") as handle:
        json.dump(rows, handle, indent=1)
    print(f"\nwrote {ARTIFACTS / 'indistribution_rows.json'}")


if __name__ == "__main__":
    main()
