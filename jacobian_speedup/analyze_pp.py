#!/usr/bin/env python3
"""Partial-pooling speed-up variants against the production FD run (the 0.928 run).

Reference: recovery/rerun (prod schedule, seed 7), reproduced bit-exactly by the
current library. For every recovery/speed_* tag, on the datasets it shares with
the reference:

  dG     mean over datasets and the 7 globals of |median - median_ref| / sd_ref
  dL     the same over the 100 x 3 locals
  sdG    geometric-mean sd ratio of the globals
  r      the paper's mean r (posterior median vs truth, per global, averaged)
  RMSE   global std-RMSE (score_rerun)
  rows   network row-evaluations per dataset (FD copies and warm start included)
  sec    runtime per dataset (shared GPU: compare within one session only)

The prod schedule reseeded (speed_prod_seed8) gives the Monte-Carlo floor of dG/dL.

    python analyze_pp.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch

COMPASS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(COMPASS / "experiments_arruda_schur")]
ROOT = COMPASS / "experiments_arruda_schur/artifacts/pp/hierarchical_Rg_wide1.5/recovery"
METHOD = "gauss_jacobian_FD-s100-t50"


def load_dir(directory):
    out = {}
    for f in glob.glob(str(directory / "dataset-*.pt")):
        o = torch.load(f, map_location="cpu", weights_only=False)
        out[int(o["dataset_id"])] = o
    return out


def main():
    import score_rerun as s
    scale = s.load_normalizers(s.get_config("small"), s.BenchmarkPaths.default())["sde_joint_theta"].scale
    reference = load_dir(ROOT / "rerun" / METHOD)
    tags = sorted(p.name for p in ROOT.iterdir() if p.name.startswith("speed_") and "smoke" not in p.name)
    rows = []
    for tag in ["rerun"] + tags:
        directory = ROOT / tag / METHOD
        if not directory.exists():
            directory = next(iter((ROOT / tag).glob("*-s100-t50")), None)
            if directory is None:
                continue
        runs = load_dir(directory)
        ids = sorted(set(runs) & set(reference))
        if not ids:
            continue
        dg, dl, sdg = [], [], []
        for i in ids:
            mine, ref = runs[i]["posterior"], reference[i]["posterior"]
            ref_sd_g = ref["globals"].std(0)
            dg.append(float(((mine["globals"].median(0).values - ref["globals"].median(0).values).abs()
                             / ref_sd_g).mean()))
            dl.append(float(((mine["locals"].median(0).values - ref["locals"].median(0).values).abs()
                             / ref["locals"].std(0)).mean()))
            sdg.append(float(torch.log(mine["globals"].std(0) / ref_sd_g).mean()))
        paths = {i: f for i, f in ((i, str(directory / f"dataset-{i:04d}.pt")) for i in ids)}
        g, l, r, per, _ = s.score(paths, ids, scale)
        stats = [runs[i].get("solver_stats") or {} for i in ids]
        total = [st.get("evaluated_row_samples", np.nan) + st.get("warm_rows", 0) for st in stats]
        seconds = [runs[i]["runtime_seconds"] for i in ids]
        rows.append((tag.removeprefix("speed_"), len(ids), np.mean(dg), np.mean(dl), float(np.exp(np.mean(sdg))),
                     r, g, np.nanmean(total), np.mean(seconds), [d for d in ids]))
    prod_rows = next((x[7] for x in rows if x[0] == "prod"), np.nan)
    print(f"{'variant':<28}{'n':>3}{'dG':>7}{'dL':>7}{'sdG':>6}{'r':>7}{'RMSE':>7}{'rows/ds':>10}{'x prod':>7}{'sec':>6}")
    for tag, n, dg, dl, sdg, r, g, total, sec, _ in rows:
        print(f"{tag:<28}{n:3d}{dg:7.3f}{dl:7.3f}{sdg:6.2f}{r:7.3f}{g:7.3f}{total / 1e6:9.2f}M"
              f"{total / prod_rows:7.2f}{sec:6.0f}")


if __name__ == "__main__":
    main()
