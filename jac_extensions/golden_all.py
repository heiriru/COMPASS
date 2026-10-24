"""Bit-exact regression guard for the jac-extensions branch.

Every existing composition rule, run with default settings on a randomized tiny
network, must reproduce the draws saved on the base commit bit for bit.

    python golden_all.py save   # on the base commit, before any library change
    python golden_all.py check  # after: exit 0 only if every draw is identical
"""
import os
import sys

CPU_THREAD_LIMIT = 3
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = str(CPU_THREAD_LIMIT)
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU_THREAD_LIMIT])

import torch  # noqa: E402
from compass import ScoreBasedInferenceModel  # noqa: E402

torch.set_num_threads(1)
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_all.pt")

# (correction, extra kwargs); method "dpm" unless given.
CASES = [
    ("gauss_jacobian_FD", {}),
    ("gauss_jacobian", {}),
    ("gauss_jacobian_FD", {"jacobian_refresh": 9}),
    ("gauss_hierarchical", {"precision_est_samples": 64}),
    ("Gauss_schur_global", {"precision_est_samples": 64}),
    ("gauss", {"precision_est_samples": 64}),
    ("uncorrected", {}),
    ("fnpe", {"method": "langevin"}),
    ("damping", {"method": "euler"}),
]


def model():
    torch.manual_seed(0)
    m = ScoreBasedInferenceModel(nodes_size=6, sigma=5.0, hidden_size=16, depth=2,
                                 num_heads=2, mlp_ratio=2)
    g = torch.Generator().manual_seed(6)
    with torch.no_grad():
        for p in m.model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m


def run(correction, n, kw):
    m = model()
    torch.manual_seed(1)
    data = torch.randn(n, 3) * 0.5
    torch.manual_seed(2)
    hierarchy = [0, 1] if correction not in ("gauss", "uncorrected", "fnpe", "damping") else [0, 1, 2]
    return m.sample(x=data, multi_obs_inference=True, hierarchy=hierarchy,
                    correction=correction, timesteps=12, num_samples=48,
                    verbose=False, **kw)


def all_runs():
    out = {}
    for correction, kw in CASES:
        for n in (1, 5):
            key = (correction, tuple(sorted(kw.items())), n)
            out[key] = run(correction, n, dict(kw))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "save":
        torch.save(all_runs(), PATH)
        print("saved", PATH)
    else:
        ref = torch.load(PATH, weights_only=False)
        new = all_runs()
        ok = True
        for key, draws in ref.items():
            same = torch.equal(draws, new[key])
            ok &= same
            print(key, "identical" if same else
                  f"DIFF max {float((draws - new[key]).abs().max()):.3e}")
        print("ALL IDENTICAL" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)
