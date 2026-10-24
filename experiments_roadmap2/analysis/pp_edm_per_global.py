"""Roadmap 2.4: menu + EDM on partial pooling. Per-global recovery r, RMSE, slope, width, z_rms and
90% coverage for base vs edm (time_loss_weight="edm", six menu, final weights), 3 seeds x 30
datasets, plus the paired per-dataset comparison. Same metrics as pp_tsm_lsla.py."""
import os
CPU = sorted(os.sched_getaffinity(0))[:3]; os.sched_setaffinity(0, CPU)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "3"
import json
from pathlib import Path
import numpy as np, torch
A = Path("/export/home/rheinric/COMPASS/experiments_arruda_schur/artifacts/pp")
NAMES = ["mu_nu", "mu_log_alpha", "mu_log_t0", "log_sigma_nu", "log_sigma_log_alpha", "log_sigma_log_t0", "beta_raw"]
ARMS, SEEDS = ("base", "edm"), (7, 8, 9)


def load(arm, seed):
    d = A / f"hierarchical_Rg_wide1.5_n65536_{arm}_s{seed}/recovery/final/gauss_jacobian_FD-s100-t50"
    files = sorted(d.glob("dataset-*.pt"))
    assert len(files) >= 30, (arm, seed, len(files))
    rs = [torch.load(f, map_location="cpu", weights_only=False) for f in files[:30]]
    g = [r["posterior"]["globals"].double().numpy() for r in rs]
    return dict(med=np.array([np.median(x, 0) for x in g]), sd=np.array([x.std(0) for x in g]),
                lo=np.array([np.quantile(x, 0.05, 0) for x in g]), hi=np.array([np.quantile(x, 0.95, 0) for x in g]),
                truth=np.array([r["truth_globals"].double().numpy() for r in rs]))


data = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
out = {}
for (a, s), D in data.items():
    rows = {}
    for k, n in enumerate(NAMES):
        e = D["med"][:, k] - D["truth"][:, k]
        rows[n] = dict(r=float(np.corrcoef(D["med"][:, k], D["truth"][:, k])[0, 1]),
                       rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
                       slope=float(np.polyfit(D["truth"][:, k], D["med"][:, k], 1)[0]),
                       post_sd=float(D["sd"][:, k].mean()), z_rms=float(np.sqrt(((e / D["sd"][:, k]) ** 2).mean())),
                       cover90=float(((D["truth"][:, k] >= D["lo"][:, k]) & (D["truth"][:, k] <= D["hi"][:, k])).mean()))
    rows["_mean_r"] = float(np.mean([rows[n]["r"] for n in NAMES]))
    e = D["med"] - D["truth"]
    rows["_global_rmse"] = float(np.sqrt((e ** 2).mean()))
    rows["_cover90"] = float(np.mean([rows[n]["cover90"] for n in NAMES]))
    rows["_z_rms"] = float(np.mean([rows[n]["z_rms"] for n in NAMES]))
    out[f"{a}_s{s}"] = rows

print("seed  arm    mean_r  global_rmse  cover90  z_rms")
for s in SEEDS:
    for a in ARMS:
        r = out[f"{a}_s{s}"]
        print(f"s{s}   {a:5s}  {r['_mean_r']:.3f}   {r['_global_rmse']:.3f}       {r['_cover90']:.2f}    {r['_z_rms']:.2f}")
for a in ARMS:
    print(f"mean  {a:5s}  {np.mean([out[f'{a}_s{s}']['_mean_r'] for s in SEEDS]):.3f}   "
          f"{np.mean([out[f'{a}_s{s}']['_global_rmse'] for s in SEEDS]):.3f}       "
          f"{np.mean([out[f'{a}_s{s}']['_cover90'] for s in SEEDS]):.2f}    "
          f"{np.mean([out[f'{a}_s{s}']['_z_rms'] for s in SEEDS]):.2f}")
print("\nper global, mean over seeds: r base -> edm | rmse base -> edm | cover90 base -> edm | edm closer (paired)")
for k, n in enumerate(NAMES):
    m = lambda a, key: np.mean([out[f"{a}_s{s}"][n][key] for s in SEEDS])
    closer = np.mean([np.mean(np.abs(data["edm", s]["med"][:, k] - data["edm", s]["truth"][:, k])
                              < np.abs(data["base", s]["med"][:, k] - data["base", s]["truth"][:, k])) for s in SEEDS])
    print(f"  {n:20s} {m('base','r'):.3f} -> {m('edm','r'):.3f} | {m('base','rmse'):.3f} -> {m('edm','rmse'):.3f} | "
          f"{m('base','cover90'):.2f} -> {m('edm','cover90'):.2f} | {closer:.0%}")
assert all(np.allclose(data["base", s]["truth"], data["edm", s]["truth"]) for s in SEEDS), "unpaired test sets"
Path("/export/data/rheinric/roadmap2/pp_edm_per_global.json").write_text(json.dumps(out, indent=1))
