#!/usr/bin/env python3
"""Is the MCMC reference converged, and is it the right posterior?

Three checks beyond R-hat / ESS (which ``run_reference.py`` already records):

1. **Against an MCMC-free estimate.** Angle layout: the angle_1 marginal on a
   deterministic grid (``ik_exact.grid_posterior_a1``, heights integrated by
   quadrature). Angle+height layout: self-normalised importance sampling from a
   defensive mixture (90% a multivariate t fitted to the draws with 3x the
   scale, 10% the prior) in ``(a1, v = h + L1 sin a1)``; the 2-D grid is *not*
   used, because the likelihood's integrable 1/sqrt edges make a grid
   unreliable once the posterior is narrower than a few cells. The KS distance
   is compared with its 95% noise floor ``1.36 sqrt(1/ESS_mcmc + 1/ESS_check)``.
2. **Against an independent sampler** (angle layout, n <= 2): the full-state
   Metropolis-within-Gibbs sampler ``ik_mcmc.sample_angle`` shares no code path
   with the collapsed one beyond the closed-form likelihood.
3. **Simulation-based calibration** of the reference itself over all datasets
   (the truth came from the simulator): computed in ``plot_recovery.py``.

Usage (from COMPASS/): .COMPASS/bin/python inverse_kinematics/recovery/check_reference.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit()

import json  # noqa: E402

import numpy as np  # noqa: E402

import datasets  # noqa: E402
import ik_exact as E  # noqa: E402
import ik_mcmc as M  # noqa: E402

CHECKED = 20


def ks_to_grid(draws, grid, density):
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    return float(np.abs(np.searchsorted(np.sort(draws), grid, side="right") / len(draws) - cdf).max())


def ks_weighted(draws, values, weights):
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    sorted_draws = np.sort(draws)
    points = np.quantile(sorted_draws, np.linspace(0.0005, 0.9995, 1000))
    empirical = np.searchsorted(sorted_draws, points, side="right") / len(draws)
    return float(np.abs(empirical - np.interp(points, values[order], cdf)).max())


def importance_angle_height(x, a1, h, rng, count=1_000_000, df=3.0):
    """Self-normalised IS for p(a1, h | x) in (a1, v); returns (a1, h, weights, ess)."""
    from scipy.stats import multivariate_normal, multivariate_t
    z = np.stack([a1, h + E.L1 * np.sin(a1)], -1)
    mean, cov = z.mean(0), np.cov(z.T) * 9.0 + 1e-14 * np.eye(2)
    t = multivariate_t(loc=mean, shape=cov, df=df)
    prior_v_sd = np.sqrt(E.SD_H ** 2 + (E.L1 * E.SD_A1) ** 2)          # loose, only for coverage
    wide = multivariate_normal(mean=[0.0, 0.0], cov=np.diag([E.SD_A1 ** 2, prior_v_sd ** 2]))
    from_t = rng.random(count) < 0.9
    y = np.where(from_t[:, None], t.rvs(count, random_state=rng), wide.rvs(count, random_state=rng))
    log_q = np.logaddexp(np.log(0.9) + t.logpdf(y), np.log(0.1) + wide.logpdf(y))
    ya, yh = y[:, 0], y[:, 1] - E.L1 * np.sin(y[:, 0])
    log_p = np.empty(count)
    for s in range(0, count, 100_000):
        sl = slice(s, s + 100_000)
        log_p[sl] = (E.log_lik_a1h(ya[sl, None], yh[sl, None], x[None, :, 0], x[None, :, 1], wraps=(0,)).sum(1)
                     + E.log_normal(ya[sl], E.SD_A1) + E.log_normal(yh[sl], E.SD_H))
    log_w = np.nan_to_num(log_p - log_q, nan=-np.inf)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    return ya, yh, w, float(1.0 / np.sum(w ** 2))


def check(kind, layout, n, rng):
    folder = datasets.folder(kind, layout, n)
    if not (folder / "reference.npz").exists():
        return None
    x = np.load(folder / "data.npz")["x"]
    theta = np.load(folder / "reference.npz")["theta"]
    diag = json.loads((folder / "reference_diagnostics.json").read_text())["per_dataset"]
    rows = []
    for i in range(min(CHECKED, len(x))):
        row = {"dataset": i}
        if layout == "angle_height" and kind == "hier":
            ya, yh, w, ess_is = importance_angle_height(x[i], theta[i, :, 0, 0], theta[i, :, 0, 1], rng)
            for name, column, values in (("angle_1", 0, ya), ("height", 1, yh)):
                ess = diag[i]["ess"][name]
                row[name] = {"ks": ks_weighted(theta[i, :, 0, column], values, w),
                             "ks_floor95": 1.36 * np.sqrt(1.0 / ess + 1.0 / ess_is), "ess": ess,
                             "ess_check": ess_is, "mean_mcmc": float(theta[i, :, 0, column].mean()),
                             "mean_check": float(np.sum(w * values)),
                             "sd_mcmc": float(theta[i, :, 0, column].std())}
        else:
            grid, dens = E.grid_posterior_a1(x[i])
            ess = diag[i]["ess"]["angle_1"]
            row["angle_1"] = {"ks": ks_to_grid(theta[i, :, 0, 0], grid, dens),
                              "ks_floor95": 1.36 / np.sqrt(ess), "ess": ess,
                              "mean_mcmc": float(theta[i, :, 0, 0].mean()),
                              "mean_check": float(np.sum(grid * dens) / dens.sum()),
                              "sd_mcmc": float(theta[i, :, 0, 0].std())}
        rows.append(row)
    independent = None
    if (kind == "single" or layout == "angle") and n <= 2:
        out = M.sample_angle(x[:CHECKED], rng, chains=8, warmup=3000, draws=500, thin=10)
        independent = []
        for i in range(min(CHECKED, len(x))):
            a = out["a1"][i].ravel()
            ref = theta[i, :, 0, 0]
            independent.append({"rhat": M.rhat(out["a1"][i]), "ess": M.ess_bulk(out["a1"][i]),
                                "mean_full_state": float(a.mean()), "mean_collapsed": float(ref.mean()),
                                "sd_full_state": float(a.std()), "sd_collapsed": float(ref.std())})
    within = [r[c]["ks"] <= r[c]["ks_floor95"] for r in rows for c in r if c != "dataset"]
    summary = {"datasets_checked": len(rows), "ks_within_floor95": f"{sum(within)}/{len(within)}",
               "max_ks_over_floor": max(r[c]["ks"] / r[c]["ks_floor95"] for r in rows for c in r if c != "dataset"),
               "rows": rows, "independent_sampler": independent}
    print(f"{folder.name}: KS within its 95% floor for {summary['ks_within_floor95']} "
          f"(max KS/floor {summary['max_ks_over_floor']:.2f})", flush=True)
    if independent:
        dm = [abs(r["mean_full_state"] - r["mean_collapsed"]) / r["sd_collapsed"] for r in independent]
        print(f"   independent full-state sampler: max |mean diff|/sd {max(dm):.3f}, "
              f"max R-hat {max(r['rhat'] for r in independent):.3f}", flush=True)
    return summary


def main():
    rng = np.random.default_rng(99)
    report = {}
    for kind, layout, n in [("single", None, 1)] + [("hier", l, n) for l in ("angle", "angle_height")
                                                      for n in (2, 8, 32)]:
        result = check(kind, layout, n, rng)
        if result is not None:
            report[datasets.folder(kind, layout, n).name] = result
    (datasets.ARTIFACTS / "reference_checks.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
