#!/usr/bin/env python3
"""Which global/local split makes the shared posterior bimodal, and how strongly?

The arm has one exact discrete symmetry. Writing ``S(a) = sum_k l_k sin(cum_k)``
and ``C(a) = sum_k l_k cos(cum_k)``, the forward map is ``x1 = S(a) + h``,
``x2 = C(a)``. Negating **all three** angles flips ``S`` and leaves ``C`` alone,
so::

    (a1, a2, a3, h)  ->  (-a1, -a2, -a3, 2*x1 - h)

lands on the identical end position -- an exact involution of the likelihood,
verified to 4e-16. The angle priors are symmetric and so invariant under it; the
**height prior is the only term that breaks it**, by

    log p(h) - log p(2*x1 - h)  =  2*x1*(h - x1) / sigma_h^2

Three consequences decide the design:

1. **The split must let every row mirror.** The flip is shared across
   observations (all angles negate together) but the compensating height shift is
   per-row. So the shared block must be an *angle* and ``height_arm`` must be
   *local*. With ``height_arm`` global, row ``j`` could only mirror if
   ``h = 2*x1_j - h`` for every ``j`` at once -- impossible for generic data, and
   the bimodality vanishes.
2. **The penalty compounds over rows.** The branch weight ratio is a product of
   ``n`` per-row factors, so with the stock ``sigma_h = 0.25`` one branch wins
   exponentially in ``n``. That is why the eight-observation posterior measured
   earlier is unimodal: the median mirror costs about 22 log units per row.
3. **The penalty vanishes where ``x1`` does.** It is proportional to ``x1_j``,
   so datasets whose end positions sit near the rail height make the reflection
   nearly free *without touching the priors*.

Two routes to a bimodal posterior follow, and only the second works.

**Widening the height prior (``--height-stds``) does not.** It frees the mirror
and destroys the information in ``x1`` at the same time -- at ``sigma_h = 1`` the
posterior has already relaxed back to the prior (sd 0.47 against a prior 0.50).
Worse, it breaks the referee: the ABC kernel's effective sample size falls from
3,396/80,000 at ``sigma_h = 0.25`` to **17/80,000 at ``sigma_h = 16``**, so the
"two clean modes" that appear at the top of that ladder are noise. The tell is a
posterior width that stops contracting with ``n``. The ladder is retained only
because that negative result is worth being able to reproduce.

**Selecting observations with ``|x1| < eps`` (``--select-x1``) does.** Stock
priors throughout -- this is a condition on the data, not a change to the model,
so the estimator stays in its healthy regime and the posterior stays sharp.
Measured at ``eps = 0.15``, true angle 0.6: modes at -0.36/+0.44 with a dip ratio
falling 0.650 -> 0.319 -> 0.129 as ``n`` goes 2 -> 4 -> 8, and near-equal mode
mass at small ``n``.

Both are measured with the SBC-validated ABC reference, so neither answer depends
on a trained network.

Usage:
    python inverse_kinematics/bimodality_study.py --select-x1 0.15
    python inverse_kinematics/bimodality_study.py            # the failed route
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import hierarchical as H  # noqa: E402

LOCAL_SCALE_ANGLE = float(H.LOCAL_PRIOR_STD[1])
import simulator as sim  # noqa: E402

ROOT = Path(__file__).resolve().parent


def bimodality(grid, density, smooth=9):
    """Peak/dip description of a 1-D density.

    Returns the two largest local maxima and the ``dip ratio`` -- the density at
    the lowest point *between* them, divided by the smaller peak. A dip ratio
    near 1 means one broad mode; near 0 means two cleanly separated ones.

    The density is boxcar-smoothed first. Without it the Monte Carlo ripple in
    the reference is read as structure: an early run of this script reported
    "modes" at +0.615 and +0.750 on a density that is plainly unimodal.
    """
    if smooth > 1:
        kernel = np.ones(int(smooth)) / float(smooth)
        density = np.convolve(density, kernel, mode="same")
    interior = np.where(
        (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])
    )[0] + 1
    if len(interior) < 2:
        return {"modes": [float(grid[np.argmax(density)])], "dip_ratio": 1.0,
                "mode_mass_ratio": 0.0}
    order = interior[np.argsort(density[interior])[::-1]][:2]
    left, right = sorted(order)
    valley = left + int(np.argmin(density[left:right + 1]))
    smaller = min(density[left], density[right])
    # Mass on each side of the valley, as a second, shape-free read.
    mass_left = float(np.trapezoid(density[:valley + 1], grid[:valley + 1]))
    mass_right = float(np.trapezoid(density[valley:], grid[valley:]))
    return {
        "modes": [float(grid[left]), float(grid[right])],
        "dip_ratio": float(density[valley] / smaller),
        "mode_mass_ratio": float(min(mass_left, mass_right)
                                 / max(mass_left, mass_right)),
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    # The mirror shifts the height by 2*S, and the arm's reach makes |S| ~ 1-2,
    # so the ladder must reach well past that before the reflection is free. In
    # the limit the height prior is flat, x1 carries no information (any height
    # absorbs it) and only x2 = C(a) constrains the angles -- and C is exactly
    # even under the flip, so the posterior becomes exactly symmetric about
    # zero. The top of this ladder approaches that limit.
    result.add_argument("--height-stds", type=float, nargs="+",
                        default=[0.25, 1.0, 4.0, 16.0])
    result.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8])
    result.add_argument("--true-angle", type=float, default=0.6,
                        help="Well away from zero, so the two branches separate.")
    result.add_argument("--draws", type=int, default=80_000)
    result.add_argument("--bandwidth", type=float, default=0.05)
    result.add_argument("--select-x1", type=float, default=None,
                        help="Keep only observations with |x1| < this. The "
                             "working route to bimodality; leaves priors alone.")
    result.add_argument("--seed", type=int, default=5)
    result.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    return result


def selected_dataset(count, true_angle, epsilon, rng, block=20_000):
    """Prior draws for the locals, kept only where the tip sits near ``x1 = 0``.

    A selection on the *observation*, not a modified prior: the mirror penalty is
    ``2*x1*(h - x1)/sigma_h^2`` and vanishes with ``x1``, so this frees the
    reflection while the informative stock priors keep the modes sharp.
    """
    kept = []
    while len(kept) < count:
        locals_ = rng.normal(loc=H.LOCAL_PRIOR_MEAN, scale=H.LOCAL_PRIOR_STD,
                             size=(block, 3))
        theta = np.concatenate([np.full((block, 1), true_angle), locals_], axis=1)
        x = H.forward(theta)
        kept.extend(list(x[np.abs(x[:, 0]) < epsilon]))
    return np.array(kept[:count])


def main(argv=None):
    arguments = parser().parse_args(argv)
    span = 4.0 * H.GLOBAL_PRIOR_STD[0]
    grid = np.linspace(-span, span, 801)

    print(f"true shared angle_1 = {arguments.true_angle}")
    if arguments.select_x1 is not None:
        print(f"stock priors; selecting observations with |x1| < "
              f"{arguments.select_x1}\n")
    else:
        print("varying the height prior (the route that does not work; see the "
              "module docstring)\n")
    print(f"{'sigma_h':>9}{'n':>4}{'modes':>26}{'dip ratio':>11}"
          f"{'mass ratio':>12}{'sd':>8}")
    rows = []
    if arguments.select_x1 is not None:
        rng = sim.generator("selected", arguments.seed)
        for count in arguments.counts:
            x = selected_dataset(count, arguments.true_angle,
                                 arguments.select_x1, rng)
            _, density = H.reference_global(
                x, grid=grid, draws=arguments.draws,
                bandwidth=arguments.bandwidth, seed=arguments.seed,
            )
            shape = bimodality(grid, density)
            _, std, _ = H.grid_moments(grid, density)
            modes = ", ".join(f"{m:+.3f}" for m in shape["modes"])
            print(f"{H.LOCAL_PRIOR_STD[0]:>9.2f}{count:>4}{modes:>26}"
                  f"{shape['dip_ratio']:>11.3f}{shape['mode_mass_ratio']:>12.3f}"
                  f"{std:>8.3f}", flush=True)
            rows.append({"height_std": float(H.LOCAL_PRIOR_STD[0]),
                         "select_x1": arguments.select_x1,
                         "observations": count, "std": std, **shape})
        output = arguments.artifacts / "inference" / "bimodality_selected.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(
            {"true_angle": arguments.true_angle, "select_x1": arguments.select_x1,
             "draws": arguments.draws, "bandwidth": arguments.bandwidth,
             "rows": rows}, indent=2, sort_keys=True))
        print(f"\nWrote {output}")
        return 0

    for height_std in arguments.height_stds:
        local_std = np.array([height_std, H.LOCAL_PRIOR_STD[1],
                              H.LOCAL_PRIOR_STD[2]])
        # Draw the dataset under the *same* widened prior it will be scored with,
        # so the truth stays a prior draw and the comparison is self-consistent.
        rng = sim.generator(f"bimodal-{height_std}", arguments.seed)
        locals_ = rng.normal(loc=H.LOCAL_PRIOR_MEAN, scale=local_std,
                             size=(max(arguments.counts), 3))
        theta = np.concatenate([
            np.full((len(locals_), 1), arguments.true_angle), locals_,
        ], axis=1)
        x_all = H.forward(theta)

        for count in arguments.counts:
            _, density = H.reference_global(
                x_all[:count], grid=grid, draws=arguments.draws,
                bandwidth=arguments.bandwidth, seed=arguments.seed,
                local_std=local_std,
            )
            shape = bimodality(grid, density)
            _, std, _ = H.grid_moments(grid, density)
            modes = ", ".join(f"{m:+.3f}" for m in shape["modes"])
            print(f"{height_std:>9.2f}{count:>4}{modes:>26}"
                  f"{shape['dip_ratio']:>11.3f}{shape['mode_mass_ratio']:>12.3f}"
                  f"{std:>8.3f}", flush=True)
            rows.append({"height_std": height_std, "observations": count,
                         "std": std, **shape})

    output = arguments.artifacts / "inference" / "bimodality_study.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {"true_angle": arguments.true_angle, "draws": arguments.draws,
         "bandwidth": arguments.bandwidth, "rows": rows},
        indent=2, sort_keys=True,
    ))
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def exact_zero_dataset(count, true_angle, rng, block=20_000):
    """Observations at ``x1 = 0`` with the height at its prior *mode*.

    ``x1 = S(a) + h``, so ``x1 = 0`` can be reached either by solving for ``h``
    or by solving for an angle. Solving for ``h`` is the obvious route and it is
    wrong: it drives the height into the prior tails (measured: -4.9 and -6.2
    prior sd on an eight-observation draw), and the ABC reference then has an
    effective sample size of 4 out of 120,000 for those rows, so it reports
    asymmetry on a posterior that is provably symmetric.

    Here ``h = 0`` instead -- the mode of its own prior -- and ``angle_3`` is
    solved so the angles alone put the tip on the rail height:

        sin(a1 + a2 + a3) = -(l1 sin a1 + l2 sin(a1 + a2)) / l3

    ``angle_2`` is a prior draw and rejected when that has no solution. Every
    coordinate then sits where the prior actually has mass, the reference stays
    supported, and the mirror ``h -> -h`` is exactly free, so
    ``p(a | x) = p(-a | x)`` holds exactly. Any asymmetry a sampler shows here is
    its own.
    """
    kept = []
    while len(kept) < count:
        a2 = rng.normal(0.0, LOCAL_SCALE_ANGLE, size=block)
        target = -(sim.L1 * np.sin(true_angle)
                   + sim.L2 * np.sin(true_angle + a2)) / sim.L3
        ok = np.abs(target) <= 1.0
        a2, target = a2[ok], target[ok]
        a3 = np.arcsin(target) - true_angle - a2
        kept.extend(np.stack([np.zeros(len(a2)), a2, a3], axis=1))
    locals_ = np.array(kept[:count])
    theta = np.concatenate(
        [np.full((count, 1), true_angle), locals_], axis=1
    )
    x = H.forward(theta)
    if np.abs(x[:, 0]).max() > 1e-10:
        raise RuntimeError(f"x1 not zeroed: max |x1| = {np.abs(x[:, 0]).max():.3e}")
    return x, locals_[:, 0]
