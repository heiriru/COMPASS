#!/usr/bin/env python
"""Oracle-free audit of the IK network's row-score Jacobian.

The Gaussian-mixture rung diagnosed ModeTree's learned-score failure by
comparing the network's Jacobian against an exact one. The arm has no
closed-form score, so that comparison is unavailable here -- but two properties
of the *true* Jacobian hold for any model at all, and can be checked without a
reference:

**Symmetry.** ``J = grad^2 log p_t`` is a Hessian, so ``J = J^T`` exactly. The
network's output is an unconstrained vector field, nothing in the denoising
objective asks it to be a gradient, and every unit of ``||J - J^T||`` is pure
error with no interpretation.

**The Tweedie spectrum.** The second-order identity reads
``Sigma_t = lambda^2 (I + lambda^2 J)`` with ``Sigma_t = Cov(x_0 | x_t)``, a
covariance. So ``I + lambda^2 J`` must be positive semi-definite, and -- since
conditioning cannot increase the variance past the marginal -- its eigenvalues
must lie in ``[0, 1]`` up to the prior's scale. Eigenvalues outside that band
are error, and ``gauss_jacobian``'s ``jacobian_curvature_floor`` exists to clamp
them; ModeTree consumes the curvature unclamped in the Newton step and the
Laplace determinant, so it sees them raw.

Both are measured at the network's own evaluation points: a tall IK dataset
diffused to a ladder of lambda.

Run:  python ModeTree/experiments/experiment_ik_jacobian.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IK = ROOT / "inverse_kinematics"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(IK))

from modetree.runtime import configure_cpu_limit  # noqa: E402

configure_cpu_limit(3)

import argparse  # noqa: E402
import json      # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

import hierarchical as H   # noqa: E402
import train_model as trainer  # noqa: E402

from compass.priors import GaussianPrior               # noqa: E402
from modetree.compass_adapter import CompassRowSource  # noqa: E402
from experiment_ik import CHECKPOINT, make_dataset     # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
LAMBDAS = (2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.003, 0.001)


def audit(model, x, lambdas, states, seed=0):
    prior = GaussianPrior(
        torch.as_tensor(H.GLOBAL_PRIOR_MEAN, dtype=torch.float64),
        torch.as_tensor(H.GLOBAL_PRIOR_STD ** 2, dtype=torch.float64),
    )
    source = CompassRowSource(model, x, H.HIERARCHY, H.LOCAL_INDICES,
                              H.OBSERVED_INDICES, prior)
    n = x.shape[0]
    dim = len(H.HIERARCHY) + len(H.LOCAL_INDICES)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for lam in lambdas:
        # Evaluation points drawn the way the sampler meets them: a prior draw
        # plus lambda * eps, which is exactly the forward diffusion at this
        # noise level.
        scale = torch.as_tensor(
            np.concatenate([H.GLOBAL_PRIOR_STD, H.LOCAL_PRIOR_STD]),
            dtype=torch.float64)
        theta = scale * torch.randn(states, n, dim, generator=generator,
                                    dtype=torch.float64)
        theta = theta + lam * torch.randn(states, n, dim, generator=generator,
                                          dtype=torch.float64)
        jacobian = source.row_jacobians(theta, lam)          # (S, n, D, D)
        symmetric = 0.5 * (jacobian + jacobian.mT)
        antisymmetric = 0.5 * (jacobian - jacobian.mT)
        relative = (antisymmetric.norm(dim=(-2, -1))
                    / jacobian.norm(dim=(-2, -1)).clamp_min(1e-30))
        eye = torch.eye(dim, dtype=torch.float64)
        spectrum = torch.linalg.eigvalsh(eye + (lam ** 2) * symmetric)
        rows.append({
            "lambda": float(lam),
            "antisymmetry_mean": float(relative.mean()),
            "antisymmetry_max": float(relative.max()),
            "tweedie_min_eig": float(spectrum.min()),
            "tweedie_max_eig": float(spectrum.max()),
            "fraction_negative": float((spectrum < 0).double().mean()),
            "fraction_above_one": float((spectrum > 1.0 + 1e-9).double().mean()),
            "jacobian_norm": float(jacobian.norm(dim=(-2, -1)).mean()),
        })
        r = rows[-1]
        print(f"lambda={lam:7.4f}  ||J-J^T||/||J|| mean {r['antisymmetry_mean']:.3f} "
              f"max {r['antisymmetry_max']:.3f}   "
              f"eig(I+l^2 J) in [{r['tweedie_min_eig']:+.3f}, "
              f"{r['tweedie_max_eig']:+.3f}]   "
              f"neg {100 * r['fraction_negative']:5.1f}%  "
              f">1 {100 * r['fraction_above_one']:5.1f}%", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=8)
    parser.add_argument("--states", type=int, default=64)
    parser.add_argument("--regime", default="bimodal")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)

    model = trainer.load_to(CHECKPOINT, "cpu")
    print(f"Loaded {CHECKPOINT.parent.name}", flush=True)
    _, x = make_dataset(args.regime, args.observations, args.seed)
    rows = audit(model, x, LAMBDAS, args.states, args.seed)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "ik_jacobian_audit.json", "w") as handle:
        json.dump({"checkpoint": CHECKPOINT.parent.name,
                   "observations": args.observations,
                   "states": args.states, "rows": rows}, handle, indent=1)
    print(f"\nwrote {ARTIFACTS / 'ik_jacobian_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
