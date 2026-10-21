#!/usr/bin/env python
"""Does the learned first-order score still contain the multimodal topology?

Every failure measured so far -- ``gauss_jacobian``, F-NPSE and ModeTree-JAC --
consumes the network's score *Jacobian* somewhere. The audit showed that object
is not a Hessian (3-31% antisymmetric, 30-35% of the Tweedie spectrum outside
``[0,1]``). That leaves one question unanswered, and it decides what to build
next: is the **first-order** field itself still right?

This diagnostic removes ``J = grad s`` completely. It uses only ordinary forward
evaluations of the score network, through the composition rule the codebase
already implements:

    ComposedField(rule="uncorrected")
        S_g = (1-n) s_prior + sum_i s_{i,g}
            = s_prior + sum_i [s_{i,g} - s_prior]

verified against ``modetree/diffusion.py`` -- that branch calls no Jacobian and
applies no cross-factor. No JVPs, no Hessians, no Tweedie covariance, no Newton,
no ModeTree curvature code anywhere below.

The hierarchical problem
------------------------
The state is ``y_i = (g, l_i)``. Fixing the locals at their prior mean would
test a different function than the one the posterior is about, so for every
candidate ``g`` the locals are **profiled** by first-order ascent on their own
row score,

    l_i*(g) = argmax_{l_i} log p(g, l_i | x_i)     [Adam on s_{i,local}]

and the global score is read at ``(g, l*(g))``.

What integrating that curve gives you -- exactly
------------------------------------------------
At ``l*(g)`` the local score vanishes, so

    d/dg log p(g, l*(g)|X) = partial_g log p + (partial_l log p) . dl*/dg
                           = partial_g log p                    [second term 0]

i.e. the composed global score read at the profiled locals is the exact total
derivative of the **profile** log posterior. Integrating it in 1-D therefore
reconstructs the profile posterior, not the marginal one: the Occam factor from
integrating the locals out is missing. Profile and marginal share their mode
*locations* only when the local curvature is constant in ``g``; their mode
*masses* generally differ. Every reconstructed density below is labelled
``profile`` for that reason, and it is the honest object, not an approximation
that happens to be convenient.

Two problems, because only one of them has an oracle
----------------------------------------------------
``ik``  the arm. Reference is the network-free ABC posterior. No exact score
        exists, so the learned curve cannot be compared to an oracle curve here.
``gm``  the Gaussian-mixture benchmark. Has both an exact diffused score and a
        closed-form tall posterior, so it validates that the profiling and
        integration machinery is correct before any learned-score claim rests
        on it.

Run:
    python ModeTree/experiments/experiment_first_order_topology.py --problem gm
    python ModeTree/experiments/experiment_first_order_topology.py --problem ik
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IK = ROOT / "inverse_kinematics"
GM = ROOT / "Gaussian_mixtures_compositional_score_modelling"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# `bimodality_study.bimodality` is the shape metric every IK table in
# this repository already uses; reuse it for both problems so the dip
# ratios here are comparable with the ones reported elsewhere.
sys.path.insert(0, str(IK))

from modetree.runtime import configure_cpu_limit  # noqa: E402

CPU_INFO = configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402
import time      # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

from modetree.compass_adapter import CompassRowSource  # noqa: E402
from modetree.diffusion import ComposedField           # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "first_order_topology"


# ---------------------------------------------------------------------------
# problem adapters -- each supplies data, a reference, and a row source
# ---------------------------------------------------------------------------

class IKProblem:
    """The robot arm: d_g = 1 (angle_1), d_l = 3, ABC reference."""

    name = "ik"
    has_oracle = False

    def __init__(self, seed=3, regime="bimodal", reference_draws=80_000,
                 checkpoint=None, wrapper="plain"):
        """``checkpoint``/``wrapper`` point the diagnostic at a retrained arm.

        The default is the frozen checkpoint every earlier result used. An arm
        trained by ``exp4_ik_transfer`` saves an ordinary backbone checkpoint
        and has to be re-wrapped on load, which is what ``wrapper`` selects --
        the residual baseline is a closed form, not a stored parameter.
        """
        sys.path.insert(0, str(IK))
        import hierarchical as H
        import train_model as trainer
        from experiment_ik import CHECKPOINT, make_dataset, reference

        if checkpoint is not None:
            CHECKPOINT = Path(checkpoint)

        self.H = H
        self._make_dataset = make_dataset
        self._reference = reference
        self.seed = seed
        self.regime = regime
        self.reference_draws = reference_draws
        self.model = trainer.load_to(CHECKPOINT, "cpu")
        if wrapper != "plain":
            import exp4_ik_transfer as transfer
            transfer.attach(self.model, wrapper,
                            transfer.ARMS.get("Kbpre", ("", 1e-4))[1])
        self.checkpoint = CHECKPOINT.parent.name
        self.prior_std = float(H.GLOBAL_PRIOR_STD[0])
        self.sde_sigma = float(getattr(self.model.sde, "sigma", 25.0))

    def dataset(self, n):
        return self._make_dataset(self.regime, n, self.seed)

    def reference(self, x, n):
        return self._reference(x, self.reference_draws, 0.05, 401, self.seed)

    def source(self, kind, x):
        from compass.priors import GaussianPrior
        assert kind == "learned", "the arm has no analytic score"
        prior = GaussianPrior(
            torch.as_tensor(self.H.GLOBAL_PRIOR_MEAN, dtype=torch.float64),
            torch.as_tensor(self.H.GLOBAL_PRIOR_STD ** 2, dtype=torch.float64),
        )
        return CompassRowSource(self.model, x, self.H.HIERARCHY,
                                self.H.LOCAL_INDICES, self.H.OBSERVED_INDICES,
                                prior)

    def grid(self, points):
        span = 3.0 * self.prior_std
        return np.linspace(-span, span, int(points))


class GMProblem:
    """The Gaussian mixture: exact score *and* closed-form tall posterior."""

    name = "gm"
    has_oracle = True

    def __init__(self, seed=2026):
        sys.path.insert(0, str(GM))
        import datasets
        import model as M
        self.datasets = datasets
        self.M = M
        self.mixture = M.DEFAULT
        self.seed = seed
        self.prior_std = float(self.mixture.global_prior_std)
        import train
        self.sde_sigma = float(train.SDE_KWARGS["sigma"])

    def dataset(self, n):
        rng = np.random.default_rng(
            [self.seed, self.datasets.REGIMES.index("asymmetric"), n, 0])
        data = self.datasets.make_dataset("asymmetric", n, rng,
                                          mixture=self.mixture,
                                          branch_mass=0.75, spread=1.0)
        return 0.0, np.asarray(data["x"], dtype=np.float64)

    def reference(self, x, n):
        grid = np.asarray(self.mixture.default_grid(x, points=2001, pad=3.0))
        return grid, np.asarray(self.mixture.global_density(grid, x))

    def source(self, kind, x):
        from compass.priors import GaussianMixturePrior
        from experiment_gaussian_mixture import load_source, mixture_prior
        model = load_source(kind, self.mixture)
        return CompassRowSource(model, x, self.M.HIERARCHY,
                                self.M.LOCAL_INDICES, self.M.OBSERVED_INDICES,
                                mixture_prior(self.mixture))

    def grid(self, points):
        span = 3.0 * self.prior_std + 3.0
        return np.linspace(-span, span, int(points))


PROBLEMS = {"ik": IKProblem, "gm": GMProblem}


# ---------------------------------------------------------------------------
# first-order local profiling  (no Jacobians anywhere)
# ---------------------------------------------------------------------------

def profile_locals(field, g_values, lam, t, iterations=300, lr=0.05,
                   chunk=256, init=None, verbose=False):
    """``l*(g)`` for a whole grid of ``g`` at once, by Adam on the local score.

    The only network quantity used is ``ComposedField.score``; its local block
    for the ``uncorrected`` rule is the raw per-row local score, which is the
    ascent direction. Adam consumes it as ``-score`` in the ``.grad`` slot --
    the network is never differentiated, only evaluated.

    Grid points are independent, so the whole scan is one batch rather than a
    sequential warm-started sweep; ``init`` allows continuation from a previous
    solution (used for the terminal-noise ladder, where consecutive ``t`` share
    almost the same optimum).
    """
    d_g, d_l, n = field.d_g, field.d_l, field.n
    g_values = torch.as_tensor(g_values, dtype=torch.float64).reshape(-1, 1)
    total = g_values.shape[0]
    out = torch.zeros(total, n * d_l, dtype=torch.float64)
    norms = torch.zeros(total, dtype=torch.float64)
    forwards = 0

    for start in range(0, total, chunk):
        g_block = g_values[start:start + chunk]
        size = g_block.shape[0]
        if init is None:
            locals_ = torch.zeros(size, n * d_l, dtype=torch.float64)
        else:
            locals_ = torch.as_tensor(
                init[start:start + chunk], dtype=torch.float64).clone()
        locals_.requires_grad_(False)
        optimizer = torch.optim.Adam([locals_], lr=lr)
        locals_.grad = torch.zeros_like(locals_)
        for _ in range(iterations):
            state = torch.cat([g_block, locals_.detach()], dim=-1)
            with torch.no_grad():
                composed = field.score(state, t, lam)
            forwards += size
            local_score = composed.score[..., d_g:]
            # ascent on log p  ->  descent on -log p
            locals_.grad = -local_score
            optimizer.step()
        state = torch.cat([g_block, locals_.detach()], dim=-1)
        with torch.no_grad():
            composed = field.score(state, t, lam)
        forwards += size
        out[start:start + size] = locals_.detach()
        norms[start:start + size] = composed.score[..., d_g:].norm(dim=-1)
        if verbose:
            print(f"    profiled {start + size}/{total} "
                  f"max local |s| = {float(norms[start:start + size].max()):.3e}",
                  flush=True)
    return out, norms, forwards


def global_score_curve(field, g_values, locals_, lam, t, chunk=256):
    """``S(g)`` at the profiled locals, via the Jacobian-free composition."""
    g_values = torch.as_tensor(g_values, dtype=torch.float64).reshape(-1, 1)
    out = torch.zeros(g_values.shape[0], dtype=torch.float64)
    forwards = 0
    for start in range(0, g_values.shape[0], chunk):
        state = torch.cat([g_values[start:start + chunk],
                           locals_[start:start + chunk]], dim=-1)
        with torch.no_grad():
            composed = field.score(state, t, lam)
        forwards += state.shape[0]
        out[start:start + state.shape[0]] = composed.score[..., 0]
    return out.numpy(), forwards


# ---------------------------------------------------------------------------
# roots, densities, metrics
# ---------------------------------------------------------------------------

def find_roots(grid, score, refine=None):
    """Sign changes of ``S``, classified as modes (+ -> -) or valleys (- -> +).

    ``refine`` is an optional callable ``g -> S(g)`` used for Brent refinement.
    Newton is deliberately unavailable here: this diagnostic never differentiates
    the score.
    """
    roots = []
    for k in range(len(grid) - 1):
        left, right = score[k], score[k + 1]
        if left == 0.0 or left * right >= 0:
            continue
        a, b = float(grid[k]), float(grid[k + 1])
        if refine is not None:
            try:
                from scipy.optimize import brentq
                root = float(brentq(refine, a, b, xtol=1e-6, rtol=1e-10,
                                    maxiter=60))
            except Exception:
                root = a + (b - a) * abs(left) / (abs(left) + abs(right))
        else:
            root = a + (b - a) * abs(left) / (abs(left) + abs(right))
        roots.append({
            "root": root,
            "type": "mode" if left > 0 > right else "valley",
            "score_left": float(left), "score_right": float(right),
        })
    return roots


def integrate_score(grid, score):
    """Profile log density from ``S``, normalized, in log space."""
    log_p = np.concatenate([[0.0], np.cumsum(
        0.5 * (score[1:] + score[:-1]) * np.diff(grid))])
    log_p -= log_p.max()
    density = np.exp(log_p)
    area = np.trapezoid(density, grid)
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError("score integration did not normalize")
    return density / area


def simpson_check(grid, score):
    """Simpson's rule on the same integrand, as an independent quadrature."""
    if len(grid) % 2 == 0:
        grid, score = grid[:-1], score[:-1]
    log_p = [0.0]
    for k in range(0, len(grid) - 2, 2):
        h = (grid[k + 2] - grid[k]) / 2.0
        panel = h / 3.0 * (score[k] + 4.0 * score[k + 1] + score[k + 2])
        log_p.extend([log_p[-1] + panel / 2.0, log_p[-1] + panel])
    return np.asarray(log_p[:len(grid)])


def describe(grid, density, smooth=9):
    """Modes, valley, dip ratio, masses, mean, sd of a tabulated density."""
    from bimodality_study import bimodality
    shape = bimodality(grid, density)
    mean = float(np.trapezoid(grid * density, grid))
    variance = float(np.trapezoid((grid - mean) ** 2 * density, grid))
    out = {"modes": shape["modes"], "dip_ratio": shape["dip_ratio"],
           "mode_mass_ratio": shape["mode_mass_ratio"],
           "mean": mean, "sd": float(np.sqrt(max(variance, 0.0)))}
    if len(shape["modes"]) == 2:
        left = int(np.argmin(np.abs(grid - shape["modes"][0])))
        right = int(np.argmin(np.abs(grid - shape["modes"][1])))
        valley = left + int(np.argmin(density[left:right + 1]))
        out["valley"] = float(grid[valley])
        below = float(np.trapezoid(density[:valley + 1], grid[:valley + 1]))
        above = float(np.trapezoid(density[valley:], grid[valley:]))
        out["masses"] = [below / (below + above), above / (below + above)]
    return out


def w1(grid, density, reference_density):
    def cdf(values):
        out = np.concatenate([[0.0], np.cumsum(
            0.5 * (values[1:] + values[:-1]) * np.diff(grid))])
        return out / max(out[-1], 1e-300)
    return float(np.trapezoid(np.abs(cdf(density) - cdf(reference_density)), grid))


def reference_score(grid, density, floor=1e-3):
    """``d/dg log p_ref`` from a smoothing spline on the log density.

    The ABC reference is a Monte Carlo estimate, so differentiating it raw
    manufactures structure. A spline on ``log p`` is fitted where the density is
    resolved and differentiated analytically; everywhere else the curve is
    reported as NaN rather than extrapolated.
    """
    from scipy.interpolate import UnivariateSpline
    # Only where the reference is actually resolved. Below ~1e-3 of the peak
    # the ABC estimate is Monte Carlo noise and its log-derivative rings; the
    # task's instruction not to over-interpret tiny fluctuations is enforced
    # here rather than left to the reader of the figure.
    mask = density > floor * density.max()
    if mask.sum() < 20:
        return np.full_like(grid, np.nan)
    log_p = np.log(np.clip(density[mask], 1e-300, None))
    # The smoothing budget is per-point and must be small compared with the
    # Monte Carlo ripple in log p, not compared with log p itself: s scaled by
    # len(log_p) flattened a density whose true d log p/dg reaches O(1e3) near a
    # sharp mode down to O(1). Use a fixed small per-point budget instead.
    spline = UnivariateSpline(grid[mask], log_p, s=len(log_p) * 1e-4, k=4)
    out = np.full_like(grid, np.nan)
    out[mask] = spline.derivative()(grid[mask])
    return out


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

def run_cell(problem, kind, n, t, args, cache=None):
    truth, x = problem.dataset(n)
    grid_ref, reference_density = problem.reference(x, n)
    field_source = problem.source(kind, x)
    field = ComposedField(field_source, "uncorrected")
    lam = field_source.schedule.lam(t)

    grid = problem.grid(args.grid_points)
    started = time.perf_counter()
    locals_, local_norms, forwards_profile = profile_locals(
        field, grid, lam, t, iterations=args.profile_iterations,
        lr=args.profile_lr, chunk=args.chunk,
        init=None if cache is None else cache.get("locals"),
        verbose=args.verbose,
    )
    profile_seconds = time.perf_counter() - started

    started = time.perf_counter()
    score, forwards_score = global_score_curve(field, grid, locals_, lam, t,
                                               chunk=args.chunk)
    score_seconds = time.perf_counter() - started

    def refine(g):
        """``S(g)`` at one point, locals warm-started from the nearest node."""
        index = int(np.argmin(np.abs(grid - g)))
        loc, _, _ = profile_locals(
            field, [g], lam, t, iterations=args.refine_iterations,
            lr=args.profile_lr, chunk=1,
            init=locals_[index: index + 1].numpy(),
        )
        value, _ = global_score_curve(field, [g], loc, lam, t, chunk=1)
        return float(value[0])

    roots = find_roots(grid, score, refine=refine if args.brent else None)
    density = integrate_score(grid, score)

    # the reference lives on its own grid; compare on the score grid
    reference_on_grid = np.interp(grid, grid_ref, reference_density,
                                  left=0.0, right=0.0)
    area = np.trapezoid(reference_on_grid, grid)
    if area > 0:
        reference_on_grid = reference_on_grid / area

    row = {
        "problem": problem.name, "score_source": kind, "n": n, "t": t,
        "lambda": lam, "truth": truth,
        "roots": roots,
        "modes_found": [r["root"] for r in roots if r["type"] == "mode"],
        "valleys_found": [r["root"] for r in roots if r["type"] == "valley"],
        "reconstructed": describe(grid, density),
        "reference": describe(grid, reference_on_grid),
        "w1": w1(grid, density, reference_on_grid),
        "reference_sd": describe(grid, reference_on_grid)["sd"],
        "max_local_score_norm": float(local_norms.max()),
        "median_local_score_norm": float(local_norms.median()),
        "network_forwards": int((forwards_profile + forwards_score) * n),
        "profile_seconds": profile_seconds,
        "score_seconds": score_seconds,
        "grid_points": int(args.grid_points),
    }
    row["w1_over_sigma"] = row["w1"] / max(row["reference_sd"], 1e-12)
    curves = {"grid": grid, "score": score, "density": density,
              "reference": reference_on_grid,
              "reference_score": reference_score(grid, reference_on_grid)}
    return row, curves, locals_.numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="ik", choices=sorted(PROBLEMS))
    parser.add_argument("--observations", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--sources", nargs="+", default=None,
                        help="default: learned (+ oracle where one exists)")
    parser.add_argument("--times", type=float, nargs="+", default=(1e-2,))
    parser.add_argument("--grid-points", type=int, default=1201)
    parser.add_argument("--profile-iterations", type=int, default=300)
    parser.add_argument("--refine-iterations", type=int, default=60)
    parser.add_argument("--profile-lr", type=float, default=0.05)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--brent", action="store_true",
                        help="Refine roots with Brent (extra forward passes).")
    parser.add_argument("--regime", default="bimodal")
    parser.add_argument("--ik-checkpoint", default=None)
    parser.add_argument("--ik-wrapper", default="plain")
    parser.add_argument("--out-suffix", default="",
                        help="Appended to the artifact filenames. Use it for "
                             "side runs (stability checks, resolution sweeps) "
                             "so they cannot overwrite the main sweep.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    kwargs = ({"regime": args.regime, "checkpoint": args.ik_checkpoint,
               "wrapper": args.ik_wrapper} if args.problem == "ik" else {})
    problem = PROBLEMS[args.problem](**kwargs)
    sources = args.sources or (["oracle", "learned"] if problem.has_oracle
                               else ["learned"])
    print(f"problem={problem.name} sources={sources} "
          f"checkpoint={getattr(problem, 'checkpoint', '-')} "
          f"sde_sigma={problem.sde_sigma} prior_sd={problem.prior_std}",
          flush=True)

    rows, store = [], {}
    for kind in sources:
        for n in args.observations:
            cache = None
            for t in sorted(args.times, reverse=True):
                row, curves, locals_ = run_cell(problem, kind, n, t, args, cache)
                cache = {"locals": locals_}
                rows.append(row)
                key = f"{kind}|{n}|{t:g}"
                store[key] = {k: np.asarray(v).tolist()
                              for k, v in curves.items()}
                modes = [round(m, 4) for m in row["modes_found"]]
                ref_modes = [round(m, 4) for m in row["reference"]["modes"]]
                print(f"{kind:8s} n={n:4d} t={t:<8g} lam={row['lambda']:.4f}  "
                      f"modes={modes}  valleys="
                      f"{[round(v, 4) for v in row['valleys_found']]}  "
                      f"ref={ref_modes}  dip={row['reconstructed']['dip_ratio']:.3f}"
                      f" (ref {row['reference']['dip_ratio']:.3f})  "
                      f"W1/sd={row['w1_over_sigma']:.4f}  "
                      f"{row['profile_seconds'] + row['score_seconds']:.0f}s",
                      flush=True)
                with open(OUT / f"{problem.name}{args.out_suffix}_rows.json", "w") as handle:
                    json.dump(rows, handle, indent=1)
                with open(OUT / f"{problem.name}{args.out_suffix}_curves.json", "w") as handle:
                    json.dump(store, handle)
    print(f"\nwrote {OUT / (problem.name + args.out_suffix + '_rows.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
