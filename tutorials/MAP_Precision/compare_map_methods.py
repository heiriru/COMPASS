"""A/B comparison of `map_estimate` (Tweedie ascent) and `newton_map_estimate`.

Both estimators seek the same fixed point `s(z, t) = 0` and are configured
through the same validated setup block, so any difference here is the ascent
step alone. The score is an *exact* analytic diffused score, so there is no
network error and no composition error either: every number below is optimizer
error measured against a closed-form arrowhead MAP.

Two regimes are swept:

  easy    sigma_global=0.8, sigma_local=0.6, sigma_x=0.25
          the widths baked into tests/test_hierarchical_map.py
  hard    sigma_global=0.3, sigma_local=0.4, sigma_x=0.2
          the widths in tutorials/plot_local_vs_global_joint_map_validation.py,
          whose cached run reaches 43.5 sigma of error at n=200

Candidates start from the **prior mean**, not from the analytic answer. The
existing unit tests initialise at the answer, which is why they pass at n=50
while the tutorial fails on the same n.

    python tutorials/MAP_Precision/compare_map_methods.py --out <dir>
"""

import os

CPU_THREAD_LIMIT = 3
_allowed = sorted(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT]
os.sched_setaffinity(0, set(_allowed))
for _variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = str(len(_allowed))

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from compass.MultiObsSampler import MultiObsSampler
from compass.SDE import VESDE

REGIMES = {
    "easy": dict(sigma_global=0.8, sigma_local=0.6, sigma_x=0.25),
    "hard": dict(sigma_global=0.3, sigma_local=0.4, sigma_x=0.2),
}
PRIOR_MEAN = 0.0


class SharedLocalGaussianScore(torch.nn.Module):
    """Exact diffused score for g,l ~ Normal and x = g + l + noise.

    Same construction as tests/test_hierarchical_map.py, with the widths made
    configurable so both regimes run through identical code.
    """

    sigma_global = 0.8
    sigma_local = 0.6
    sigma_x = 0.25

    def __init__(self, sde):
        super().__init__()
        self.sde = sde
        precision = torch.tensor([
            [1 / self.sigma_global**2 + 1 / self.sigma_x**2, 1 / self.sigma_x**2],
            [1 / self.sigma_x**2, 1 / self.sigma_local**2 + 1 / self.sigma_x**2],
        ])
        self.register_buffer("posterior_covariance", torch.linalg.inv(precision))

    def forward(self, x, t, c, return_attn_weights=False):
        noise_std = self.sde.marginal_prob_std(t).to(x.device)
        alpha = self.sde.alpha_t(t).to(x.device)
        theta = x[:, :2]
        observed = x[:, 2]
        rhs = torch.stack([
            observed / self.sigma_x**2,
            observed / self.sigma_x**2,
        ], dim=1)
        mean = alpha * (rhs @ self.posterior_covariance.T)
        covariance = (
            alpha**2 * self.posterior_covariance
            + noise_std**2 * torch.eye(2, device=x.device)
        )
        score = torch.linalg.solve(covariance, (mean - theta).unsqueeze(-1)).squeeze(-1)
        result = torch.zeros_like(x)
        result[:, :2] = noise_std * score
        return (result, torch.zeros(1)) if return_attn_weights else result


class MockSBIm:
    """Minimal stand-in for ScoreBasedInferenceModel: the contract the sampler needs."""

    def __init__(self, model_cls, nodes_size, sde=None):
        self.sde = VESDE(25.0) if sde is None else sde
        self.model = model_cls(self.sde)
        self.nodes_size = nodes_size

    def output_scale_function(self, t, value):
        return value / self.sde.marginal_prob_std(t)


def make_score_class(regime):
    return type(
        "RegimeScore", (SharedLocalGaussianScore,),
        {k: float(v) for k, v in REGIMES[regime].items()},
    )


def analytic_shared_local_map(observations, score_cls):
    """Closed-form joint mode of [g, l_1..l_n] -- for a Gaussian, mode == mean."""
    n = len(observations)
    sg, sl, sx = score_cls.sigma_global, score_cls.sigma_local, score_cls.sigma_x
    precision = torch.zeros(n + 1, n + 1, dtype=torch.float64)
    precision[0, 0] = 1 / sg**2 + n / sx**2
    precision[1:, 1:] = torch.eye(n, dtype=torch.float64) * (1 / sl**2 + 1 / sx**2)
    precision[0, 1:] = 1 / sx**2
    precision[1:, 0] = 1 / sx**2
    observations = observations.to(torch.float64)
    rhs = torch.cat([
        (observations.sum() / sx**2).reshape(1), observations / sx**2,
    ])
    covariance = torch.linalg.inv(precision)
    return (covariance @ rhs).float(), covariance.diag().sqrt().float()


def make_joint(observations, global_value, local_values):
    result = torch.zeros(len(observations), 3)
    result[:, 0] = global_value
    result[:, 1] = torch.as_tensor(local_values)
    result[:, 2] = observations
    return result


def simulate(n, score_cls, seed):
    """Draw a real problem: true g, true locals, and the resulting observations."""
    generator = torch.Generator().manual_seed(seed)
    true_global = score_cls.sigma_global * torch.randn(1, generator=generator)
    true_local = score_cls.sigma_local * torch.randn(n, generator=generator)
    noise = score_cls.sigma_x * torch.randn(n, generator=generator)
    return true_global, true_local, true_global + true_local + noise


def errors(result, expected, posterior_std):
    """Global error and local RMSE, both in units of the analytic posterior sd."""
    inferred = torch.cat([result[:1, 0].flatten(), result[:, 1]])
    deviation = (inferred - expected) / posterior_std
    return float(deviation[0].abs()), float(deviation[1:].square().mean().sqrt())


def run_case(method, n, regime, seed, timesteps, denoise_clamp, curvature,
             anneal):
    score_cls = make_score_class(regime)
    _, _, observations = simulate(n, score_cls, seed)
    expected, posterior_std = analytic_shared_local_map(observations, score_cls)

    # "long" anneals from the prior width; "short" reproduces the policy in
    # plot_local_vs_global_joint_map_validation.py, which scales sigma_start to
    # the *posterior* width and so shrinks the homotopy as n grows.
    if anneal == "short":
        sigma_start = max(2.0 * float(posterior_std[0]), 1e-3)
    else:
        sigma_start = 2.0 * score_cls.sigma_global

    # Start from the prior mean: the estimator has to *find* the mode.
    initial = make_joint(observations, PRIOR_MEAN, torch.full((n,), PRIOR_MEAN))
    sampler = MultiObsSampler(MockSBIm(score_cls, 3))
    shared = dict(
        data=initial, condition_mask=torch.tensor([0.0, 0.0, 1.0]), init=initial,
        hierarchy=[0], prior=([PRIOR_MEAN], [score_cls.sigma_global]),
        correction="uncorrected", sigma_start=sigma_start,
        timesteps=timesteps, eps=1e-3, device="cpu",
    )

    start = time.perf_counter()
    if method == "tweedie":
        result = sampler.map_estimate(
            denoise_clamp=denoise_clamp, iterations_per_level=3, **shared,
        )[:, 0]
    else:
        result = sampler.newton_map_estimate(
            denoise_clamp=denoise_clamp, iterations_per_level=1,
            curvature=curvature, **shared,
        )[:, 0]
    runtime = time.perf_counter() - start

    global_error, local_rmse = errors(result, expected, posterior_std)
    clamp_edge = PRIOR_MEAN - 5.0 * score_cls.sigma_global
    diagnostics = getattr(sampler, "map_diagnostics", {})
    return {
        "method": method, "curvature": curvature if method == "newton" else "",
        "regime": regime, "anneal": anneal, "sigma_start": sigma_start,
        "n_observations": n, "seed": seed,
        "global_error_analytic_sigma": global_error,
        "local_rmse_analytic_sigma": local_rmse,
        "global_estimate": float(result[0, 0]),
        "analytic_global_map": float(expected[0]),
        "pinned_at_clamp": int(
            denoise_clamp is not None
            and abs(float(result[0, 0]) - clamp_edge) < 1e-4
        ),
        "shared_synchronization_max_abs": float(
            (result[:, 0] - result[0, 0]).abs().max()
        ),
        "score_network_calls": int(sampler.score_network_calls),
        "runtime_seconds": runtime,
        "converged_levels": diagnostics.get("converged_levels", ""),
        "backtracks": diagnostics.get("backtracks", ""),
        "unimproved_steps": diagnostics.get("unimproved_steps", ""),
        "trust_region_hits": diagnostics.get("trust_region_hits", ""),
        "richardson_shift": diagnostics.get("richardson_shift", ""),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).parent / "artifacts"))
    parser.add_argument("--observations", type=int, nargs="+",
                        default=[2, 5, 10, 25, 50, 100, 200])
    parser.add_argument("--regimes", nargs="+", default=["easy", "hard"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--denoise-clamp", type=float, default=5.0,
                        help="Applied to both methods so the A/B is like-for-like. "
                             "Negative disables it.")
    parser.add_argument("--anneal", nargs="+", default=["long", "short"],
                        choices=["long", "short"])
    parser.add_argument(
        "--curvature", nargs="+", default=["jacobian"],
        help="'gaussian' additionally needs a SCHUR_GAUSSIAN correction and a "
             "pilot posterior_covariance -- the dependency 'jacobian' removes.",
    )
    args = parser.parse_args()

    clamp = None if args.denoise_clamp < 0 else args.denoise_clamp
    cases = [("tweedie", "")] + [("newton", c) for c in args.curvature]

    rows = []
    for regime in args.regimes:
        for anneal in args.anneal:
            for n in args.observations:
                for method, curvature in cases:
                    for seed in range(args.repeats):
                        try:
                            row = run_case(method, n, regime, seed, args.timesteps,
                                           clamp, curvature, anneal)
                        except Exception as error:            # noqa: BLE001
                            row = {
                                "method": method, "curvature": curvature,
                                "regime": regime, "anneal": anneal,
                                "n_observations": n, "seed": seed,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        rows.append(row)
                    label = (f"{regime:<5} {anneal:<5} n={n:<4} "
                             f"{method}/{curvature or '-':<9}")
                    done = [r for r in rows[-args.repeats:] if "error" not in r]
                    if done:
                        mean = lambda key: sum(r[key] for r in done) / len(done)
                        print(
                            f"{label} global={mean('global_error_analytic_sigma'):9.4f}"
                            f"  local={mean('local_rmse_analytic_sigma'):9.4f}"
                            f"  calls={mean('score_network_calls'):7.0f}"
                            f"  pinned={sum(r['pinned_at_clamp'] for r in done)}"
                            f"/{len(done)}", flush=True,
                        )
                    else:
                        print(f"{label} FAILED: {rows[-1].get('error')}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    path = out / "map_method_comparison.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
