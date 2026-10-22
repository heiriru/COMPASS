import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
from compass.experimental.multilevel import *
torch.manual_seed(0)

def rand_spd(*shape, d):
    A = torch.randn(*shape, d, d, dtype=torch.float64)
    return A @ A.mT + d * torch.eye(d, dtype=torch.float64)

for dims, branching in [([2, 1], [5]), ([2, 2, 1], [3, 4]), ([1, 2, 0], [3, 2]), ([2, 1, 1, 2], [2, 3, 2])]:
    tree = NestedTree.balanced(dims, branching)
    S = 3
    D = tree.offsets[-1]
    ls = torch.randn(tree.sizes[-1], S, D, dtype=torch.float64)
    lp = rand_spd(tree.sizes[-1], S, d=D)
    ps = [torch.randn(tree.sizes[m], S, tree.offsets[m+1], dtype=torch.float64) for m in range(tree.depth-1)]
    pp = [0.1 * rand_spd(1, 1, d=tree.offsets[m+1]) for m in range(tree.depth-1)]
    u = compose_tree(tree, ls, lp, ps, pp, project=False)
    ref = dense_composition(tree, ls, lp, ps, pp)
    err = max(float((a - b).abs().max()) if a.numel() else 0.0 for a, b in zip(u, ref))
    print("algebra", dims, branching, f"{err:.2e}")

# oracle: exact composed score
for dims, branching in [([2, 1], [6]), ([2, 2, 1], [3, 4]), ([1, 1, 1, 1], [2, 2, 3])]:
    tree = NestedTree.balanced(dims, branching)
    N = tree.depth
    A = [0.8 * torch.randn(dims[m], dims[m-1], dtype=torch.float64) for m in range(1, N)]
    D = tree.offsets[-1]
    B = torch.randn(2, D, dtype=torch.float64)
    model = LinearGaussianTree(tree, A, [0.7] * (N - 1), B, nu=0.5)
    theta, x = model.simulate()
    mean, cov = model.posterior(x)
    score, jac = model.path_score(x)
    prior = model.prior()
    for lam in [3.0, 0.3, 0.01]:
        S = 4
        flat = mean + torch.randn(S, mean.numel(), dtype=torch.float64) * 0.5
        levels = tree.unflatten(flat)
        paths = tree.path(levels, N - 1)
        n = tree.sizes[-1]
        li = torch.arange(n).repeat_interleave(S)
        s = score(paths.reshape(n * S, D), lam, li).reshape(n, S, D)
        P = jacobian_precision(jac(paths.reshape(n*S, D), lam, li), lam, floor=1e-12).reshape(n, S, D, D)
        pss, pps = prior.terms(tree, levels, lam)
        u = compose_tree(tree, s, P, pss, pps, project=True)
        exact = -torch.linalg.solve(cov + lam**2 * torch.eye(cov.shape[0], dtype=torch.float64), (flat - mean).T).T
        got = tree.flatten(u)
        print("oracle", dims, branching, lam, f"{float((got-exact).abs().max() / exact.abs().max()):.2e}")
