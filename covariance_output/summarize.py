#!/usr/bin/env python3
"""Summaries for the covariance-output study: GM (exact posterior), PP (recovery), toy.

    python summarize.py            # prints and writes covariance_output/RESULTS_*.md
"""
import os
CPU = sorted(os.sched_getaffinity(0))[:3]
os.sched_setaffinity(0, CPU)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "3"
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
GM = Path("/export/data/rheinric/covout/gm")
PP = Path("/export/home/rheinric/COMPASS/experiments_arruda_schur/artifacts/pp")
NAMES = ["mu_nu", "mu_log_alpha", "mu_log_t0", "log_sigma_nu", "log_sigma_log_alpha",
         "log_sigma_log_t0", "beta_raw"]
GM_METHODS = {"gauss_hierarchical_mixture_prior": "GH (pilot)",
              "gauss_jacobian_mixture_prior_fd": "JAC-FD r9",
              "gauss_jacobian_mixture_prior_netcov": "JAC-netcov"}
PP_METHODS = {"final/gauss_hierarchical-s100-t50": "GH (pilot)",
              "final/gauss_jacobian_FD_r9-s100-t50": "JAC-FD r9",
              "netcov/gauss_jacobian_FD_r1-s100-t50": "JAC-netcov"}


def gm_table():
    rows = ["| net | regime | n | method | W1/sd | branch err | width | local W1 | s |",
            "|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for d in sorted(GM.glob("*_s*")):
        path = d / "scaling" / "eps0.001_cmp" / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        for key, v in sorted(s.items(), key=lambda kv: (kv[0].split("|")[0], int(kv[0].split("|")[1]))):
            regime, n, method = key.split("|")
            if method not in GM_METHODS:
                continue
            m = {k: v[k]["median"] for k in ("w1_over_sigma", "branch_mass_error", "width_ratio",
                                              "local_w1_over_sigma", "seconds")}
            rows.append(f"| {d.name} | {regime} | {n} | {GM_METHODS[method]} | {m['w1_over_sigma']:.3f} | "
                        f"{m['branch_mass_error']:+.3f} | {m['width_ratio']:.3f} | "
                        f"{m['local_w1_over_sigma']:.3f} | {m['seconds']:.0f} |")
            arm = d.name.rsplit("_s", 1)[0]
            agg.setdefault((arm, GM_METHODS[method]), []).append(m)
    summary = ["| net arm | method | configs | mean W1/sd | mean abs branch err | mean width | mean local W1 | mean s |",
               "|---|---|---|---|---|---|---|---|"]
    for (arm, method), ms in sorted(agg.items()):
        summary.append(f"| {arm} | {method} | {len(ms)} | {np.mean([m['w1_over_sigma'] for m in ms]):.3f} | "
                       f"{np.mean([abs(m['branch_mass_error']) for m in ms]):.3f} | "
                       f"{np.mean([m['width_ratio'] for m in ms]):.3f} | "
                       f"{np.mean([m['local_w1_over_sigma'] for m in ms]):.3f} | {np.mean([m['seconds'] for m in ms]):.0f} |")
    return "\n".join(summary), "\n".join(rows)


def pp_load(folder):
    files = sorted(folder.glob("dataset-*.pt"))
    if len(files) < 30:
        return None
    rs = [torch.load(f, map_location="cpu", weights_only=False) for f in files[:30]]
    g = [r["posterior"]["globals"].double().numpy() for r in rs]
    loc = [r["posterior"]["locals"].double().numpy() for r in rs]
    return dict(med=np.array([np.median(x, 0) for x in g]), sd=np.array([x.std(0) for x in g]),
                lo=np.array([np.quantile(x, 0.05, 0) for x in g]),
                hi=np.array([np.quantile(x, 0.95, 0) for x in g]),
                truth=np.array([r["truth_globals"].double().numpy() for r in rs]),
                lmed=[np.median(x, 0) for x in loc],
                ltruth=[r["truth_locals"].double().numpy() for r in rs],
                seconds=float(np.mean([r["runtime_seconds"] for r in rs])))


def pp_table():
    rows = ["| net | method | mean r | global RMSE | cover90 | z_rms | local RMSE | s/dataset |",
            "|---|---|---|---|---|---|---|---|"]
    agg = {}
    for d in sorted(PP.glob("hierarchical_Rg_wide1.5_covout_*")):
        for sub, label in PP_METHODS.items():
            D = pp_load(d / "recovery" / sub)
            if D is None:
                continue
            e = D["med"] - D["truth"]
            r = np.mean([np.corrcoef(D["med"][:, k], D["truth"][:, k])[0, 1] for k in range(7)])
            cover = np.mean((D["truth"] >= D["lo"]) & (D["truth"] <= D["hi"]))
            z = np.mean(np.sqrt(((e / D["sd"]) ** 2).mean(0)))
            lr = np.sqrt(np.mean([((a - b) ** 2).mean() for a, b in zip(D["lmed"], D["ltruth"])]))
            net = d.name.replace("hierarchical_Rg_wide1.5_covout_", "")
            m = dict(r=r, rmse=float(np.sqrt((e ** 2).mean())), cover=cover, z=z, lr=lr, s=D["seconds"])
            rows.append(f"| {net} | {label} | {r:.3f} | {m['rmse']:.3f} | {cover:.2f} | {z:.2f} | {lr:.3f} | {D['seconds']:.0f} |")
            agg.setdefault((net.rsplit("_s", 1)[0], label), []).append(m)
    summary = ["| net arm | method | seeds | mean r | global RMSE | cover90 | z_rms | local RMSE | s/dataset |",
               "|---|---|---|---|---|---|---|---|---|"]
    for (arm, label), ms in sorted(agg.items()):
        f = lambda k: np.mean([m[k] for m in ms])  # noqa: E731
        summary.append(f"| {arm} | {label} | {len(ms)} | {f('r'):.3f} | {f('rmse'):.3f} | {f('cover'):.2f} | "
                       f"{f('z'):.2f} | {f('lr'):.3f} | {f('s'):.0f} |")
    return "\n".join(summary), "\n".join(rows)


if __name__ == "__main__":
    gs, gr = gm_table()
    ps, pr = pp_table()
    text = f"# GM (exact posterior)\n\n{gs}\n\n<details><summary>per config</summary>\n\n{gr}\n\n</details>\n\n# Partial pooling (30 datasets x 100 subjects)\n\n{ps}\n\n{pr}\n"
    (HERE / "RESULTS.md").write_text(text)
    print(text)
