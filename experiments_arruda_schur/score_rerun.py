"""Score the matched partial-pooling reruns (C2 `none` vs `full`) side by side.

    python score_rerun.py            # every rerun arm, on the datasets all arms have finished

Metrics (all on the original scale, standardised by the joint-theta normaliser):
global / local std-RMSE of the posterior median, and Arruda's mean r (Pearson
correlation of posterior median vs truth across datasets, per global, averaged).
The published C2 files are scored on the same dataset ids for reference.
"""
import glob
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

sys.path[:0] = [str(common.COMPASS)]
import Partial_Pooling.runtime as rt  # noqa: E402

rt.configure_cpu_limit(3)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from Partial_Pooling.config import get_config  # noqa: E402
from Partial_Pooling.model_pipeline import load_normalizers  # noqa: E402
from Partial_Pooling.paths import BenchmarkPaths  # noqa: E402

PUBLISHED = (common.COMPASS / "Partial_Pooling/artifacts/encoder_summary/recovery"
             / "setenc-d16-w64x2s4-h96d6-log_sigma-te64-mask6mix")
METHODS = ("gauss_hierarchical", "gauss_jacobian_FD", "gauss_jacobian_FD_r1",
           "gauss_jacobian_FD_churn4", "langevin_fnpse")


def arms():
    rows = [("published C2", "gauss_hierarchical", PUBLISHED / "dpm2_gauss_hierarchical-s100-t50"),
            ("published C2", "gauss_jacobian (JVP)", PUBLISHED / "dpm2_gauss_jacobian_newton-s100-t50"),
            ("published C2", "langevin_fnpse", PUBLISHED / "langevin_fnpse-s100-t50")]
    # hierarchical_TR: + log-lambda time input + residual output (pp.py --precondition TR)
    for mode in ("none", "hierarchical", "hierarchical_TR", "hierarchical_T", "hierarchical_R",
                 "hierarchical_Rg", "full"):
        for method in METHODS:
            rows.append((f"rerun {mode}", method,
                         HERE / "artifacts/pp" / mode / "recovery/rerun" / f"{method}-s100-t50"))
    return rows


def files(directory):
    out = {}
    for f in glob.glob(str(directory / "**" / "dataset-*.pt"), recursive=True):
        out[int(Path(f).stem.split("-")[1])] = f
    return out


def score(paths, ids, scale):
    G, L, P, T, secs = [], [], [], [], 0.0
    for i in ids:
        o = torch.load(paths[i], map_location="cpu", weights_only=False)
        p = o["posterior"]
        m = p["globals"].median(0).values
        G.append(float((((m - o["truth_globals"]) / scale[:7]) ** 2).mean().sqrt()))
        L.append(float((((p["locals"].median(0).values - o["truth_locals"]) / scale[7:]) ** 2).mean().sqrt()))
        P.append(m.numpy())
        T.append(o["truth_globals"].numpy())
        secs += float(o.get("runtime_seconds") or 0.0)
    P, T = np.array(P), np.array(T)
    r = [np.corrcoef(P[:, k], T[:, k])[0, 1] for k in range(P.shape[1])]
    return np.mean(G), np.mean(L), np.mean(r), r, secs


def main():
    scale = load_normalizers(get_config("small"), BenchmarkPaths.default())["sde_joint_theta"].scale
    found = [(a, b, files(d)) for a, b, d in arms()]
    found = [row for row in found if row[2]]
    common_ids = sorted(set.intersection(*(set(f) for _, _, f in found)))
    print(f"datasets scored (common to every arm with files): {len(common_ids)}\n")
    print(f"{'network':20s} {'rule':26s} {'n':>3s} {'glob RMSE':>9s} {'loc RMSE':>9s} {'mean r':>7s}  per-global r")
    for network, rule, paths in found:
        own = sorted(paths)
        for ids, tag in ((common_ids, ""), (own, " (own)")):
            if tag and own == common_ids:
                continue
            if len(ids) < 3:
                continue
            g, l, r, rs, secs = score(paths, ids, scale)
            print(f"{network:20s} {rule + tag:26s} {len(ids):3d} {g:9.3f} {l:9.3f} {r:7.3f}  "
                  + " ".join(f"{x:.2f}" for x in rs) + f"   {secs / 3600:.1f} h")


if __name__ == "__main__":
    main()
