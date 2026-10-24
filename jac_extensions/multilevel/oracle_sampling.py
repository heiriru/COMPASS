"""3-level linear-Gaussian oracle: does the multi-level sampler recover the exact joint posterior?"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
from compass.experimental.multilevel import *
torch.set_num_threads(3)

def run(dims, branching, variants, S=4000, steps=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    tree = NestedTree.balanced(dims, branching)
    N = tree.depth
    A = [0.8 * torch.randn(dims[m], dims[m-1], dtype=torch.float64, generator=g) for m in range(1, N)]
    D = tree.offsets[-1]
    B = torch.randn(2, D, dtype=torch.float64, generator=g)
    model = LinearGaussianTree(tree, A, [0.7] * (N - 1), B, nu=0.5)
    theta, x = model.simulate(generator=g)
    mean, cov = model.posterior(x)
    sd = cov.diagonal().sqrt()
    corr = cov / sd[:, None] / sd[None, :]
    score, jac = model.path_score(x)
    rows = []
    for name, kw in variants.items():
        sampler = MultiLevelSampler(tree, score, model.prior(), jacobian_fn=jac, **kw)
        t0 = time.time()
        out = sampler.sample(S, lam_max=20.0, lam_min=1e-3, steps=steps,
                             generator=torch.Generator().manual_seed(seed + 1))
        flat = tree.flatten(out)
        m_err = ((flat.mean(0) - mean).abs() / sd).max()
        sd_ratio = flat.std(0) / sd
        c = torch.corrcoef(flat.T)
        rows.append((name, float(m_err), float(sd_ratio.min()), float(sd_ratio.max()),
                     float((c - corr).abs().max()), time.time() - t0))
    floor = 1 / S ** 0.5
    print(f"tree dims={dims} branching={branching} latents={tree.total_latents()} S={S} (MC floor/coord ~{floor:.3f})")
    for r in rows:
        print(f"  {r[0]:28s} max|mean err|/sd {r[1]:.3f}  sd ratio [{r[2]:.3f},{r[3]:.3f}]  max|corr err| {r[4]:.3f}  {r[5]:.1f}s")

eye = None
variants = {
    "JAC exact Jacobian": dict(precision="jacobian"),
    "JAC finite differences": dict(precision="jacobian_fd"),
    "JAC FD refresh 9": dict(precision="jacobian_fd", refresh=9),
    "mid-level priors dropped": dict(precision="jacobian", drop_internal_priors=True),
}
run([2, 2, 1], [3, 4], variants)
run([1, 1, 1, 1], [3, 3, 3], variants)
