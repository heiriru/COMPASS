"""Roadmap 2.9: why did TSM (tsm2) lower the log_sigma_log_alpha recovery r on partial pooling?
Per-global bias, shrinkage slope, posterior width and 90% coverage, base vs tsm2, final weights,
3 training seeds x 30 datasets (existing runs from experiments_arruda_schur/run_pp_tsm.sh)."""
import os
CPU = sorted(os.sched_getaffinity(0))[:3]; os.sched_setaffinity(0, CPU)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "3"
import json, sys
from pathlib import Path
import numpy as np, torch
A = Path("/export/home/rheinric/COMPASS/experiments_arruda_schur/artifacts/pp")
NAMES = ["mu_nu", "mu_log_alpha", "mu_log_t0", "log_sigma_nu", "log_sigma_log_alpha", "log_sigma_log_t0", "beta_raw"]
arms = sys.argv[1:] or ["base", "tsm2"]
out = {}
for arm in arms:
    for seed in (7, 8, 9):
        d = A / f"hierarchical_Rg_wide1.5_n65536_{arm}_s{seed}/recovery/final/gauss_jacobian_FD-s100-t50"
        files = sorted(d.glob("dataset-*.pt"))
        if len(files) < 30: print(arm, seed, "incomplete", len(files)); continue
        med, sd, lo, hi, truth = [], [], [], [], []
        for f in files:
            r = torch.load(f, map_location="cpu", weights_only=False)
            g = r["posterior"]["globals"].double().numpy()
            med.append(np.median(g, 0)); sd.append(g.std(0))
            lo.append(np.quantile(g, 0.05, 0)); hi.append(np.quantile(g, 0.95, 0))
            truth.append(r["truth_globals"].double().numpy())
        med, sd, lo, hi, truth = map(np.array, (med, sd, lo, hi, truth))
        rows = {}
        for k, n in enumerate(NAMES):
            e = med[:, k] - truth[:, k]
            slope = np.polyfit(truth[:, k], med[:, k], 1)[0]
            rows[n] = dict(r=float(np.corrcoef(med[:, k], truth[:, k])[0, 1]), bias=float(e.mean()),
                           rmse=float(np.sqrt((e ** 2).mean())), slope=float(slope),
                           post_sd=float(sd[:, k].mean()),
                           z_rms=float(np.sqrt(((e / sd[:, k]) ** 2).mean())),
                           cover90=float(((truth[:, k] >= lo[:, k]) & (truth[:, k] <= hi[:, k])).mean()))
        out[f"{arm}_s{seed}"] = rows
for n in NAMES:
    print(f"\n{n}")
    print("  arm/seed      r      bias    rmse   slope  post_sd  z_rms  cover90")
    for key, rows in out.items():
        v = rows[n]
        print(f"  {key:<10} {v['r']:6.3f} {v['bias']:7.3f} {v['rmse']:7.3f} {v['slope']:6.3f} {v['post_sd']:7.3f} {v['z_rms']:6.2f} {v['cover90']:6.2f}")
Path("/export/data/rheinric/roadmap2/pp_tsm_per_global.json").write_text(json.dumps(out, indent=1))

# --- information ceiling and paired comparison for log_sigma_log_alpha (index 4) ---
def load(arm, seed, k=4):
    d = A / f"hierarchical_Rg_wide1.5_n65536_{arm}_s{seed}/recovery/final/gauss_jacobian_FD-s100-t50"
    rs = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(d.glob("dataset-*.pt"))]
    med = np.array([np.median(r["posterior"]["globals"][:, k].double().numpy()) for r in rs])
    var = np.array([r["posterior"]["globals"][:, k].double().var().item() for r in rs])
    tru = np.array([float(r["truth_globals"][k]) for r in rs])
    return med, var, tru
print("\nlog_sigma_log_alpha: information ceiling sqrt(1 - E[post var] / Var(truth)) for a calibrated posterior")
for seed in (7, 8, 9):
    line = []
    for arm in ("base", "tsm2"):
        med, var, tru = load(arm, seed)
        ceiling = np.sqrt(max(0.0, 1 - var.mean() / tru.var()))
        line.append(f"{arm}: r {np.corrcoef(med, tru)[0,1]:.3f} ceiling {ceiling:.3f}")
    mb, _, tru = load("base", seed); mt, _, _ = load("tsm2", seed)
    better = np.mean(np.abs(mt - tru) < np.abs(mb - tru))
    print(f"  s{seed}: " + " | ".join(line) + f" | tsm2 closer on {better:.0%} of datasets; truth sd {tru.std():.2f}")
