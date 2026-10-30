"""The COMPASS partial-pooling network on upstream BayesFlow's own 100 test datasets.

The fair benchmark (Arruda/partial_pooling_fair) runs both composition rules on BayesFlow's network.
This script puts the final COMPASS method -- its own network and sampler -- on the same test
datasets, so both arms can be scored with BayesFlow's metrics (fair.py score). The model is the
same on both sides: same hierarchical prior, 30 trials, Euler dt 1e-3, 10 s cap, same order of the
7 globals. Only the observation encoding differs: COMPASS uses (choice, log rt, censored).

    python fairtest.py convert            # OUT/fairtest/test.pt from fair's test.npz
    python fairtest.py export <variant> [tag]   # pp.py outputs -> OUT/fairtest/posteriors/compass-<variant>[-<tag>].npz
    python fairtest.py score              # fair.py score over BayesFlow's arms + the exported ones
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jobs as J  # noqa: E402

FAIR_DATA = Path("/export/data/rheinric/compass_pp_fair/data/test.npz")
FT = J.OUT / "fairtest"
TEST = FT / "test.pt"
POST = FT / "posteriors"
GLOBALS = ["mu_nu", "mu_log_alpha", "mu_log_t0", "log_sigma_nu", "log_sigma_log_alpha",
           "log_sigma_log_t0", "beta_raw"]
MAX_DECISION_TIME = 10.0
PP_DIR = J.TREE / "experiments_arruda_schur" / "artifacts" / "pp"


def convert():
    import torch
    sys.path.insert(0, str(J.TREE))
    from Partial_Pooling.schema import transformed_observations
    f = np.load(FAIR_DATA)
    globals_ = np.concatenate([f[k] for k in GLOBALS], axis=1)                       # (100, 7)
    locals_ = np.stack((f["nu"], np.log(f["alpha"]), np.log(f["t0"])), axis=-1)       # (100, 100, 3)
    sim = f["sim_data"]                                                               # (100, 100, 30, 2)
    choice, rt = sim[..., 0], sim[..., 1]
    censored = (rt - f["t0"][..., None] >= MAX_DECISION_TIME - 1e-9).astype(np.float64)
    raw = np.stack((choice, rt, censored), axis=-1)
    physical = np.stack((f["nu"], f["alpha"], f["t0"], np.broadcast_to(f["beta"], f["nu"].shape)), axis=-1)
    payload = {"globals": globals_, "locals": locals_, "physical": physical, "observations_raw": raw,
               "observations": transformed_observations(raw)}
    payload = {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in payload.items()}
    FT.mkdir(parents=True, exist_ok=True)
    torch.save({"payload": {"sde": payload}, "source": str(FAIR_DATA)}, TEST)
    print(f"wrote {TEST}: {int(censored.sum())} censored trials of {censored.size}")


def export(variant, tag="paper_fairtest", method="gauss_jacobian_FD_r9", prefix="hierarchical_Rg_wide1.5"):
    import torch
    folder = PP_DIR / f"{prefix}_{variant}" / "recovery" / tag / f"{method}-s100-t50"
    files = [folder / f"dataset-{i:04d}.pt" for i in range(100)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"{folder}: {len(missing)} datasets missing")
    rs = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    g = np.stack([r["posterior"]["globals"].double().numpy() for r in rs])            # (100, draws, 7)
    loc = np.stack([r["posterior"]["locals"].double().numpy() for r in rs])           # (100, draws, subj, 3)
    out = {f"g_{k}": g[:, :, j] for j, k in enumerate(GLOBALS)}
    out["l_nu"] = loc[..., 0].transpose(0, 2, 1)
    out["l_alpha"] = np.exp(loc[..., 1]).transpose(0, 2, 1)
    out["l_t0"] = np.exp(loc[..., 2]).transpose(0, 2, 1)
    POST.mkdir(parents=True, exist_ok=True)
    suffix = "" if tag == "paper_fairtest" else "-" + tag.replace("paper_fairtest_", "")
    path = POST / f"compass-{variant}{suffix}_n100.npz"
    np.savez(path, seconds=float(sum(r["runtime_seconds"] for r in rs)), **out)
    print(f"wrote {path}")


def score():
    """fair.py score over BayesFlow's own arms and the exported COMPASS ones (symlinked together)."""
    for f in (J.OUT / "fair" / "posteriors").glob("*.npz"):
        if f.name.startswith(("bf_", "paper-final")):
            link = POST / f.name
            if not link.exists():
                link.symlink_to(f)
    env = {**os.environ, "FAIR_POSTERIORS": str(POST), "KERAS_BACKEND": "torch", "CUDA_VISIBLE_DEVICES": "-1",
           "PYTHONPATH": os.pathsep.join([str(J.TREE / "Arruda" / ".vendor"), str(J.TREE)])}
    out = subprocess.run([J.PY, "fair.py", "score"], cwd=J.TREE / "Arruda" / "partial_pooling_fair", env=env,
                         capture_output=True, text=True)
    text = "\n".join(line for line in out.stdout.splitlines() if "arm" in line or line.startswith(("bf_", "compass", "paper", " ")))
    print(text)
    if out.returncode:
        print(out.stderr[-3000:])
    return text


if __name__ == "__main__":
    action, *rest = sys.argv[1:]
    {"convert": convert, "export": export, "score": score}[action](*rest)
