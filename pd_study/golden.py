"""Golden reference: gauss_jacobian_FD draws on a randomized tiny network.

    python golden.py save   # before a library change
    python golden.py check  # after: default settings must reproduce bit for bit
"""
import os
import sys

CPU_THREAD_LIMIT = 3
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = str(CPU_THREAD_LIMIT)
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT])

import torch  # noqa: E402
from compass import ScoreBasedInferenceModel  # noqa: E402

torch.set_num_threads(1)
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.pt")


def model():
    torch.manual_seed(0)
    m = ScoreBasedInferenceModel(nodes_size=6, sigma=5.0, hidden_size=16, depth=2,
                                 num_heads=2, mlp_ratio=2)
    g = torch.Generator().manual_seed(6)
    with torch.no_grad():
        for p in m.model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m


def run(correction, n, **kw):
    m = model()
    torch.manual_seed(1)
    data = torch.randn(n, 3) * 0.5
    torch.manual_seed(2)
    draws = m.sample(x=data, multi_obs_inference=True, hierarchy=[0, 1],
                     correction=correction, timesteps=12, num_samples=64,
                     verbose=False, **kw)
    return draws, dict(m.multi_obs_sampler.covariance_diagnostics)


def all_runs():
    out = {}
    for corr in ("gauss_jacobian_FD", "gauss_jacobian"):
        for n in (1, 4, 16):
            out[(corr, n)] = run(corr, n)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "save":
        torch.save(all_runs(), PATH)
        print("saved", PATH)
    else:
        ref = torch.load(PATH, weights_only=False)
        new = all_runs()
        ok = True
        for k, (d, diag) in ref.items():
            same = torch.equal(d, new[k][0])
            ok &= same
            print(k, "identical" if same else f"DIFF max {float((d - new[k][0]).abs().max()):.3e}",
                  "| neg-info frac", round(diag["negative_information_fraction"], 4),
                  "repairs", diag["repair_count"])
        print("ALL IDENTICAL" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)
