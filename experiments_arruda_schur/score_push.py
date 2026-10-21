"""Score the attempts to push C2 hierarchical + gauss_jacobian_FD past mean r 0.870, next to
the matched reruns they start from (same 30 datasets, seed, 100 subjects; score_rerun's metric).

    python score_push.py

Adds the log_sigma_nu collapse count (posterior median < -4.5 with truth > -3.5) and, for the
support-truncation arm, the fraction of composed draws it rejected.
"""
import numpy as np
import torch

import score_rerun as sr

GLOBAL = sr.HERE / "artifacts/pp"
ARMS = [
    ("baseline", "100 draws, 50 steps", GLOBAL / "hierarchical/recovery/rerun/gauss_jacobian_FD-s100-t50"),
    ("Rg", "100 draws, 50 steps", GLOBAL / "hierarchical_Rg/recovery/rerun/gauss_jacobian_FD-s100-t50"),
    ("baseline", "400 draws, 100 steps", GLOBAL / "hierarchical/recovery/a400t100/gauss_jacobian_FD-s100-t100"),
    ("Rg", "400 draws, 100 steps", GLOBAL / "hierarchical_Rg/recovery/a400t100/gauss_jacobian_FD-s100-t100"),
    ("Rg", "400 draws, 50 steps (A)", GLOBAL / "hierarchical_Rg/recovery/a400t50/gauss_jacobian_FD-s100-t50"),
    ("Rg", "support 3.5 sd (E)", GLOBAL / "hierarchical_Rg/recovery/support3.5/gauss_jacobian_FD-s100-t50"),
    ("Rg", "wide training prior 1.5x", GLOBAL / "hierarchical_Rg_wide1.5/recovery/rerun/gauss_jacobian_FD-s100-t50"),
    ("baseline", "wide training prior 1.5x", GLOBAL / "hierarchical_wide1.5/recovery/rerun/gauss_jacobian_FD-s100-t50"),
]


def extra(paths, ids):
    collapsed, rejected = 0, []
    for i in ids:
        o = torch.load(paths[i], map_location="cpu", weights_only=False)
        median = float(o["posterior"]["globals"][:, 3].median())
        collapsed += median < -4.5 and float(o["truth_globals"][3]) > -3.5
        if o.get("support"):
            rejected.append(o["support"]["rejected_fraction"])
    return collapsed, (np.mean(rejected) if rejected else None)


def main():
    scale = sr.load_normalizers(sr.get_config("small"), sr.BenchmarkPaths.default())["sde_joint_theta"].scale
    found = [(net, setting, sr.files(d)) for net, setting, d in ARMS]
    found = [row for row in found if row[2]]
    ids = sorted(set.intersection(*(set(f) for _, _, f in found)))
    print(f"datasets scored (common to every arm with files): {len(ids)}\n")
    print(f"{'network':9s} {'setting':26s} {'glob RMSE':>9s} {'loc RMSE':>8s} {'mean r':>7s}  "
          f"{'per-global r':40s} collapses  rejected")
    for net, setting, paths in found:
        g, l, r, per, secs = sr.score(paths, ids, scale)
        collapsed, rejected = extra(paths, ids)
        rej = f"{rejected:.2f}" if rejected is not None else "-"
        print(f"{net:9s} {setting:26s} {g:9.3f} {l:8.3f} {r:7.3f}  {' '.join(f'{v:.2f}' for v in per):40s} "
              f"{collapsed:3d}       {rej}   {secs / 3600:.1f} h")


if __name__ == "__main__":
    main()
