"""Exact, noise-free angle_1 posterior (no ABC kernel) for a saved 128-row IK dataset, nested n = 1..128.

Uses inverse_kinematics/recovery/ik_exact.log_marginal_a1: heights and the two joint angles are
integrated out exactly (closed-form elbow solutions + regularised Gauss-Legendre over height), so
the only approximation left is the grid. Also grades the old ABC reference(s) of the same data
against it.

    exact_reference.py DATA.npy|hier OUT.npz [ABC_REFERENCE.npz ...]

`hier` = the built-in hier_a0.6 draw of compare_compass_ik.dataset (true angle_1 0.6, no selection).
With IK_OBSERVE_ABS_X1=1 the data are (|x1|, x2) (the |x1| variant) and the likelihood is
p(x1, x2 | a1) + p(-x1, x2 | a1), which is exactly symmetric in a1.
"""
import os
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:3]))   # 3-thread CPU cap, before numpy
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "inverse_kinematics" / "recovery"), str(ROOT / "inverse_kinematics"),
                str(ROOT / "Arruda")]
import ik_exact as E  # noqa: E402
from natural_bimodal_search import summary  # noqa: E402

NS = (1, 2, 4, 8, 16, 32, 64, 128)
GRID = np.linspace(-2.0, 2.0, 2001)
ABS_X1 = os.environ.get("IK_OBSERVE_ABS_X1") == "1"


def log_lik(grid, x1, x2):
    """Exact log p(x | a1), heights and joint angles integrated out; for |x1| both signs of x1 add."""
    if ABS_X1:
        return np.logaddexp(E.log_marginal_a1(grid, abs(x1), x2), E.log_marginal_a1(grid, -abs(x1), x2))
    return E.log_marginal_a1(grid, x1, x2)


def load(spec):
    if spec == "hier":
        import compare_compass_ik as C
        return C.dataset(C.MAX_N)
    return np.load(spec)


def w1_over_sd(grid_a, p_a, grid_b, p_b):
    """W1 between two gridded densities (CDF difference on a common grid), over the sd of b."""
    common = np.linspace(min(grid_a[0], grid_b[0]), max(grid_a[-1], grid_b[-1]), 8001)
    cdf = []
    for g, p in ((grid_a, p_a), (grid_b, p_b)):
        q = np.interp(common, g, p, left=0, right=0)
        c = np.concatenate([[0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(common))])
        cdf.append(c / c[-1])
    q = np.interp(common, grid_b, p_b, left=0, right=0); q /= np.trapezoid(q, common)
    mean = np.trapezoid(common * q, common)
    sd = np.sqrt(np.trapezoid((common - mean) ** 2 * q, common))
    return float(np.trapezoid(np.abs(cdf[0] - cdf[1]), common) / sd)


def main():
    x = load(sys.argv[1])
    out = Path(sys.argv[2])
    loglik = np.stack([log_lik(GRID, x[j, 0], x[j, 1]) for j in range(len(x))], 1)   # (grid, 128)
    reference = {}
    for n in NS:
        logp = loglik[:, :n].sum(1) + E.log_normal(GRID, E.SD_A1)
        logp -= logp.max()
        p = np.exp(logp); p /= np.trapezoid(p, GRID)
        reference[f"n{n}"] = p
        s = summary(GRID, logp)
        mean = np.trapezoid(GRID * p, GRID)
        print(f"n={n:3d} exact: peaks {np.round(s['peaks'], 3).tolist()} P(a1>0) {np.trapezoid(np.where(GRID > 0, p, 0), GRID):.3f}"
              f" mean {mean:+.3f} sd {np.sqrt(np.trapezoid((GRID - mean) ** 2 * p, GRID)):.3f}", flush=True)
    np.savez(out, grid=GRID, **reference)
    for path in sys.argv[3:]:
        abc = np.load(path)
        print(f"{Path(path).name} vs exact, W1/sd(exact): " + "  ".join(
            f"n={n}:{w1_over_sd(abc['grid'], abc[f'n{n}'], GRID, reference[f'n{n}']):.2f}" for n in NS), flush=True)


if __name__ == "__main__":
    main()
