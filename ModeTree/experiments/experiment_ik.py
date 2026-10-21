#!/usr/bin/env python
"""ModeTree-JAC on the inverse-kinematics benchmark, against a real network.

The fourth rung of the error ladder. ``experiment_gaussian_mixture.py`` put
ModeTree on a learned score for the first time and it failed outright; that
benchmark has a closed-form target but an artificial forward model. This one is
the repository's physical benchmark -- a four-parameter robot arm on a vertical
rail, ``angle_1`` shared across observations and ``(height, angle_2, angle_3)``
private to each -- so the score network, the composition and the reference are
all the ones ``inverse_kinematics/run_inference.py`` already uses.

Two data regimes, both with the **stock priors and the stock checkpoint**:

``stock``    observations drawn from the prior predictive. The arm's reflection
             ``(a1,a2,a3,h) -> (-a1,-a2,-a3,2*x1-h)`` costs about 22 log units
             per row here, so the shared posterior is unimodal. This is the
             regime the published IK tables are in, and it asks whether ModeTree
             is *safe* where there is nothing multimodal to find.

``bimodal``  observations selected to have ``|x1| < eps`` (``bimodality_study``).
             The mirror is then nearly free, the shared posterior is bimodal at
             ``+-angle_1``, and this is the regime ModeTree exists for. The
             per-row penalty still compounds, so how far up the n ladder the
             second mode survives is itself a measurement.

Grading is against ``hierarchical.reference_global``: an ABC posterior computed
by Monte Carlo through the forward model with no network in the loop. It has its
own noise floor, so the shape statistics (mode locations, dip ratio, mode mass
ratio) are reported alongside W1 rather than W1 alone.

Run:  python ModeTree/experiments/experiment_ik.py --regimes bimodal stock
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IK = ROOT / "inverse_kinematics"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(IK))

from modetree.runtime import configure_cpu_limit  # noqa: E402

CPU_INFO = configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402
import time      # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

import churn                # noqa: E402
import hierarchical as H    # noqa: E402
import simulator as sim     # noqa: E402
import train_model as trainer               # noqa: E402
from bimodality_study import bimodality, selected_dataset  # noqa: E402

from compass.priors import GaussianPrior                   # noqa: E402
from modetree.compass_adapter import CompassRowSource      # noqa: E402
from modetree.diffusion import ComposedField               # noqa: E402
from modetree.laplace import assemble                      # noqa: E402
from modetree.tree import ModeTree, ModeTreeConfig, TimeComposer  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
N_LADDER = (8, 16, 32, 64, 128)
# The six-conditional weighted mask menu with log-sigma time sampling: the best
# 200k-simulation checkpoint on this benchmark, and the one the bimodal study
# used. Picking a weaker one would make the network look worse than it is and
# blame ModeTree for it.
CHECKPOINT = (IK / "artifacts" / "models"
              / "legacy_h16d2_n200000_six_weighted_log_sigma"
              / "Model_checkpoint.pt")
T_FINAL = 1e-3
TIMESTEPS = 100

BASELINES = {
    "gauss_jacobian": {
        "label": "DPM2 + gauss_jacobian + churn", "churn": True,
        "kwargs": {"method": "dpm", "order": 2, "correction": "gauss_jacobian",
                   "corrector_steps": 0, "final_corrector_steps": 0,
                   "terminal_corrector_steps": 0},
    },
    "langevin_fnpse": {
        "label": "Langevin + F-NPSE", "churn": False,
        "kwargs": {"method": "langevin", "correction": "fnpe",
                   "corrector_steps": 10, "snr": 0.1},
    },
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def make_dataset(regime, n, seed, true_angle=0.6, epsilon=0.15):
    """``(true angle_1, x)`` for one tall dataset in ``regime``."""
    rng = sim.generator(f"modetree-{regime}", seed)
    if regime == "stock":
        truth, _, x = H.simulate_dataset(n, rng)
        return float(truth), np.asarray(x, dtype=np.float64)
    if regime == "bimodal":
        x = selected_dataset(n, true_angle, epsilon, rng)
        return float(true_angle), np.asarray(x, dtype=np.float64)
    raise ValueError(f"unknown regime {regime!r}")


def reference(x, draws, bandwidth, points, seed):
    span = 4.0 * H.GLOBAL_PRIOR_STD[0]
    grid = np.linspace(-span, span, int(points))
    return H.reference_global(x, grid=grid, draws=draws, bandwidth=bandwidth,
                              seed=seed)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _cdf(grid, density):
    out = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    return out / max(out[-1], 1e-300)


def w1_density(grid, density, reference_density):
    return float(np.trapezoid(
        np.abs(_cdf(grid, density) - _cdf(grid, reference_density)), grid))


def w1_samples(grid, draws, reference_density):
    order = np.sort(np.asarray(draws, dtype=np.float64).reshape(-1))
    empirical = np.searchsorted(order, grid) / order.size
    return float(np.trapezoid(
        np.abs(empirical - _cdf(grid, reference_density)), grid))


def mixture_density(grid, means, sds, weights):
    total = np.zeros_like(grid)
    for mean, sd, weight in zip(means, sds, weights):
        total += weight * np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (
            sd * np.sqrt(2.0 * np.pi))
    return total


def grade(grid, density, reference_density, truth):
    reference_mean, reference_std, _ = H.grid_moments(grid, reference_density)
    mean, std, mode = H.grid_moments(grid, density)
    row = {
        "w1_over_sigma": w1_density(grid, density, reference_density) / reference_std,
        "mean_error_sigma": (mean - reference_mean) / reference_std,
        "width_ratio": std / reference_std,
        "truth_error_sigma": (mean - truth) / reference_std,
        "posterior_mean": mean, "posterior_std": std, "posterior_mode": mode,
    }
    row.update({f"shape_{k}": v for k, v in bimodality(grid, density).items()})
    return row


# ---------------------------------------------------------------------------
# the methods
# ---------------------------------------------------------------------------

def global_prior():
    return GaussianPrior(
        torch.as_tensor(H.GLOBAL_PRIOR_MEAN, dtype=torch.float64),
        torch.as_tensor(H.GLOBAL_PRIOR_STD ** 2, dtype=torch.float64),
    )


def run_modetree(model, x, config=None, curvature_projection=False,
                 lower_floor=0.0, t_final=T_FINAL):
    row_source = CompassRowSource(
        model, x, H.HIERARCHY, H.LOCAL_INDICES, H.OBSERVED_INDICES,
        global_prior(),
    )
    config = config or ModeTreeConfig(
        levels=16, probe_iterations=30, clean_refine_iterations=60,
        sentinel_checkpoints=2, sentinel_sobol=4, jacobian_stride=4,
        switch_to_clean_t=t_final, lambda_max=8.0,
        curvature_projection=curvature_projection,
        curvature_lower_floor=lower_floor,
    )
    field = ComposedField(row_source, "gauss_jacobian",
                          jacobian_refresh=config.jacobian_stride)
    clean = TimeComposer(field, t_final,
                         curvature_projection=curvature_projection,
                         lower_floor=lower_floor)
    started = time.perf_counter()
    tree = ModeTree(field, clean, config,
                    prior_std=torch.as_tensor(H.GLOBAL_PRIOR_STD,
                                              dtype=torch.float64))
    branches = tree.run()
    if not branches:
        return None, time.perf_counter() - started, tree, row_source
    posterior = assemble(
        clean, [b.diagnostics["clean_result"] for b in branches],
        laplace_plus_samples=0,
    )
    return posterior, time.perf_counter() - started, tree, row_source


def run_baseline(model, x, method, draws, seed, device, churn_eta=2.0):
    from compass.MultiObsSampler import MultiObsSampler

    entry = BASELINES[method]
    sampler = MultiObsSampler(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data = torch.as_tensor(x, dtype=torch.float32)
    eta = churn_eta if entry["churn"] else 0.0
    refresh = 5 if method == "gauss_jacobian" else None
    started = time.perf_counter()
    with churn.patched_dpm_sampler(eta), churn.forced_jacobian_refresh(refresh):
        samples = sampler.sample(
            world_size=1, data=data, condition_mask=H.CONDITION_MASK,
            num_samples=draws, timesteps=TIMESTEPS,
            hierarchy=list(H.HIERARCHY),
            prior=(torch.as_tensor(H.GLOBAL_PRIOR_MEAN, dtype=torch.float32),
                   torch.as_tensor(H.GLOBAL_PRIOR_STD, dtype=torch.float32)),
            local_prior=(torch.as_tensor(H.LOCAL_PRIOR_MEAN, dtype=torch.float32),
                         torch.as_tensor(H.LOCAL_PRIOR_STD, dtype=torch.float32)),
            denoise_clamp=5.0, device=device, verbose=False, **entry["kwargs"],
        )
    seconds = time.perf_counter() - started
    samples = samples.detach().cpu().numpy()
    spread = float(np.abs(samples[:, :, H.HIERARCHY]
                          - samples[:1, :, H.HIERARCHY]).max())
    if spread > 1e-4:
        raise AssertionError(f"{method}: shared draws unsynchronized ({spread:.2e})")
    return samples[0, :, H.HIERARCHY[0]], seconds


# ---------------------------------------------------------------------------

def run_cell(model, regime, n, args, device):
    truth, x = make_dataset(regime, n, args.seed)
    started = time.perf_counter()
    grid, exact = reference(x, args.reference_draws, args.reference_bandwidth,
                            args.reference_points, args.seed)
    reference_seconds = time.perf_counter() - started
    reference_shape = bimodality(grid, exact)
    rows, curves = [], {}

    if args.skip_modetree:
        posterior = None
    else:
        posterior, seconds, tree, row_source = run_modetree(
            model, x, curvature_projection=args.curvature_projection,
            lower_floor=args.curvature_lower_floor, t_final=args.t_final)
    if args.skip_modetree:
        pass
    elif posterior is None:
        rows.append({"method": "modetree", "seconds": seconds,
                     "error": "no branches found"})
    else:
        order = np.argsort([float(b.global_state) for b in posterior.branches])
        means = np.array([float(posterior.branches[i].global_state) for i in order])
        sds = np.array([float(torch.as_tensor(
            posterior.branches[i].global_covariance).reshape(-1)[0].sqrt())
            for i in order])
        weights = np.array([float(posterior.weights[i]) for i in order])
        density = mixture_density(grid, means, sds, weights)
        curves["modetree"] = density
        row = {"method": "modetree", "seconds": seconds,
               "branches": len(posterior.branches),
               "branch_means": means.tolist(), "branch_sds": sds.tolist(),
               "branch_weights": weights.tolist(),
               "network_forwards": row_source.network_forwards,
               "score_calls": row_source.score_calls,
               "jacobian_calls": row_source.jacobian_calls}
        row.update(grade(grid, density, exact, truth))
        rows.append(row)

    for method in ([] if args.only_modetree else BASELINES):
        try:
            draws_g, seconds = run_baseline(model, x, method, args.draws,
                                            args.seed, device, args.churn_eta)
        except Exception as error:                        # pragma: no cover
            rows.append({"method": method, "error": repr(error)})
            continue
        counts, edges = np.histogram(draws_g, bins=120,
                                     range=(grid[0], grid[-1]), density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])
        curves[method] = np.interp(grid, centres, counts)
        row = {"method": method, "seconds": seconds, "draws": int(draws_g.size)}
        # W1 straight from the draws (no histogram smoothing); the shape
        # statistics need a density, so they come off the histogram.
        row.update(grade(grid, curves[method], exact, truth))
        row["w1_over_sigma"] = w1_samples(grid, draws_g, exact) / \
            H.grid_moments(grid, exact)[1]
        rows.append(row)

    for row in rows:
        row.update({"regime": regime, "n": n, "truth": truth,
                    "reference_seconds": reference_seconds,
                    "reference_shape": reference_shape})
    return rows, grid, exact, curves, reference_shape


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--observations", type=int, nargs="+", default=N_LADDER)
    result.add_argument("--regimes", nargs="+", default=("bimodal", "stock"))
    result.add_argument("--draws", type=int, default=2000)
    result.add_argument("--seed", type=int, default=3)
    result.add_argument("--churn-eta", type=float, default=2.0)
    result.add_argument("--reference-draws", type=int, default=80_000)
    result.add_argument("--reference-bandwidth", type=float, default=0.05)
    result.add_argument("--reference-points", type=int, default=401)
    result.add_argument("--device", default="cpu")
    result.add_argument("--tag", default="ik")
    result.add_argument("--curvature-projection", action="store_true",
                        help="Build the arrowhead from the Tweedie-admissible "
                             "projection of the row Jacobian.")
    result.add_argument("--curvature-lower-floor", type=float, default=0.0)
    result.add_argument("--t-final", type=float, default=T_FINAL,
                        help="Terminal diffusion time of the clean "
                             "composition. The network's Jacobian is only "
                             "Tweedie-admissible above some lambda; going "
                             "below it is what makes every branch fail the "
                             "curvature test.")
    result.add_argument("--only-modetree", action="store_true",
                        help="ModeTree and the reference only.")
    result.add_argument("--skip-modetree", action="store_true",
                        help="Baselines and the reference only.")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    print(f"CPU affinity: {CPU_INFO['active_cpus']}/{CPU_INFO['host_cpus']} CPUs",
          flush=True)
    if not CHECKPOINT.exists():
        raise SystemExit(f"no checkpoint at {CHECKPOINT}")
    model = trainer.load_to(CHECKPOINT, args.device)
    parameters = sum(p.numel() for p in model.model.parameters())
    print(f"Loaded {CHECKPOINT.parent.name} ({parameters:,} parameters)",
          flush=True)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    everything, panels = [], {}
    for regime in args.regimes:
        for n in args.observations:
            rows, grid, exact, curves, shape = run_cell(
                model, regime, n, args, args.device)
            everything.extend(rows)
            panels[f"{regime}|{n}"] = {
                "grid": grid.tolist(), "exact": exact.tolist(),
                "curves": {k: np.asarray(v).tolist() for k, v in curves.items()},
                "truth": rows[0]["truth"], "reference_shape": shape,
            }
            print(f"  reference {regime} n={n}: modes "
                  f"{[round(m, 3) for m in shape['modes']]} dip "
                  f"{shape['dip_ratio']:.3f}", flush=True)
            for row in rows:
                if "error" in row:
                    print(f"{regime:8s} n={n:4d} {row['method']:16s} FAILED "
                          f"{row['error']}", flush=True)
                    continue
                print(f"{regime:8s} n={n:4d} {row['method']:16s} "
                      f"W1/sd={row['w1_over_sigma']:.4f} "
                      f"width={row['width_ratio']:.3f} "
                      f"dip={row['shape_dip_ratio']:.3f} "
                      f"{row['seconds']:8.2f}s", flush=True)
            with open(ARTIFACTS / f"{args.tag}_modetree.json", "w") as handle:
                json.dump(everything, handle, indent=1)
            with open(ARTIFACTS / f"{args.tag}_panels.json", "w") as handle:
                json.dump(panels, handle)
    print(f"\nwrote {ARTIFACTS / (args.tag + '_modetree.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
