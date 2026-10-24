"""Tests for compass.experimental.multilevel (N-level hierarchical GAUSS / JAC).

1. The tree recursion solves the dense nested-arrowhead system exactly (balanced and
   ragged trees, zero-dimensional levels).
2. On a linear-Gaussian tree with exact path scores it returns the exact diffused
   posterior score of every latent, at every noise level.
3. For N = 2 it is the production ``gauss_jacobian`` rule (same composed score through
   ``MultiObsSampler._compositional_score``).
4. Reductions and safeguards: one leaf returns the leaf score; FD Jacobian ~ exact;
   the PSD projection is a no-op on exact factors.
5. The sampler recovers the exact 3-level joint posterior; dropping the mid-level prior
   factors does not (the check has power).
"""
import torch

from compass.experimental.multilevel import (
    LinearGaussianTree, MultiLevelSampler, NestedTree, compose_tree,
    dense_composition, finite_difference_jacobian, jacobian_precision,
)


def rand_spd(*shape, d, generator):
    A = torch.randn(*shape, d, d, dtype=torch.float64, generator=generator)
    return A @ A.mT + d * torch.eye(d, dtype=torch.float64)


def random_factors(tree, S, generator):
    D = tree.offsets[-1]
    leaf_scores = torch.randn(tree.sizes[-1], S, D, dtype=torch.float64, generator=generator)
    leaf_precisions = rand_spd(tree.sizes[-1], S, d=D, generator=generator)
    prior_scores = [torch.randn(tree.sizes[m], S, tree.offsets[m + 1], dtype=torch.float64,
                                generator=generator) for m in range(tree.depth - 1)]
    prior_precisions = [0.1 * rand_spd(1, 1, d=tree.offsets[m + 1], generator=generator)
                        for m in range(tree.depth - 1)]
    return leaf_scores, leaf_precisions, prior_scores, prior_precisions


TREES = [
    NestedTree.balanced([2, 1], [5]),
    NestedTree.balanced([2, 2, 1], [3, 4]),
    NestedTree.balanced([1, 2, 0], [3, 2]),
    NestedTree.balanced([2, 1, 1, 2], [2, 3, 2]),
    NestedTree([1, 2, 1], [None, [0, 0, 0], [0, 0, 1, 2, 2, 2]]),   # ragged
]


def test_tree_recursion_solves_the_dense_system():
    generator = torch.Generator().manual_seed(0)
    for tree in TREES:
        factors = random_factors(tree, 3, generator)
        u = compose_tree(tree, *factors, project=False)
        ref = dense_composition(tree, *factors)
        for a, b in zip(u, ref):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def linear_model(dims, branching, seed=0, X=2):
    generator = torch.Generator().manual_seed(seed)
    tree = NestedTree.balanced(dims, branching)
    N = tree.depth
    A = [0.8 * torch.randn(dims[m], dims[m - 1], dtype=torch.float64, generator=generator)
         for m in range(1, N)]
    B = torch.randn(X, tree.offsets[-1], dtype=torch.float64, generator=generator)
    model = LinearGaussianTree(tree, A, [0.7] * (N - 1), B, nu=0.5)
    _, x = model.simulate(generator=generator)
    return tree, model, x


def exact_terms(tree, model, x, levels, lam):
    score, jac = model.path_score(x)
    n, S = tree.sizes[-1], levels[0].shape[1]
    paths = tree.path(levels, tree.depth - 1)
    D = paths.shape[-1]
    leaf_index = torch.arange(n).repeat_interleave(S)
    flat = paths.reshape(n * S, D)
    s = score(flat, lam, leaf_index).reshape(n, S, D)
    P = jacobian_precision(jac(flat, lam, leaf_index), lam, floor=1e-12).reshape(n, S, D, D)
    prior_scores, prior_precisions = model.prior().terms(tree, levels, lam)
    return s, P, prior_scores, prior_precisions


def test_exact_composed_score_on_linear_gaussian_trees():
    for dims, branching in [([2, 1], [6]), ([2, 2, 1], [3, 4]), ([1, 1, 1, 1], [2, 2, 3])]:
        tree, model, x = linear_model(dims, branching)
        mean, cov = model.posterior(x)
        eye = torch.eye(cov.shape[0], dtype=torch.float64)
        generator = torch.Generator().manual_seed(1)
        for lam in (3.0, 0.3, 0.01):
            flat = mean + 0.5 * torch.randn(4, mean.numel(), dtype=torch.float64, generator=generator)
            levels = tree.unflatten(flat)
            diagnostics = {}
            u = compose_tree(tree, *exact_terms(tree, model, x, levels, lam), project=True,
                             diagnostics=diagnostics)
            exact = -torch.linalg.solve(cov + lam ** 2 * eye, (flat - mean).T).T
            torch.testing.assert_close(tree.flatten(u), exact, rtol=1e-9, atol=1e-9 * float(exact.abs().max()))
            # exact Gaussian information is PSD: the projection never fires
            assert all(c == 0 for c in diagnostics["projected"].values())


def test_two_levels_match_production_gauss_jacobian():
    from types import SimpleNamespace
    from compass.MultiObsSampler import MultiObsSampler
    from compass.SDE import VESDE
    from test_gauss_jacobian_composition import LinearGaussianScore, configure

    sde = VESDE(sigma=5.0)
    t = torch.tensor([[0.4]])
    lam = float(sde.lambda_t(t).reshape(()))
    n, S, nodes = 4, 5, 4                        # nodes: g (2), l (1), x (1)
    generator = torch.Generator().manual_seed(2)
    covariance = rand_spd(n, d=3, generator=generator) / 3.0
    means = 0.3 * torch.randn(n, 3, dtype=torch.float64, generator=generator)
    network = LinearGaussianScore([0, 1, 2], means, covariance, n, sde)
    sampler = MultiObsSampler(SimpleNamespace(sde=sde, model=network,
                                              output_scale_function=lambda t, v: v))
    configure(sampler, "gauss_jacobian", [0, 1], [2], n)
    state = 0.5 * torch.randn(n, S, nodes, generator=generator)
    state[:, :, :2] = state[:1, :, :2]
    mask = torch.zeros(n, S, nodes)
    mask[..., 3] = 1.0
    scores = network(state.reshape(-1, nodes), t, mask.reshape(-1, nodes)).reshape(state.shape)
    production = sampler._compositional_score(scores.clone(), state, t, condition_mask=mask)

    tree = NestedTree.balanced([2, 1], [n])
    eye = torch.eye(3, dtype=torch.float64)
    leaf_precision = torch.linalg.inv(covariance).unsqueeze(1) + eye / lam ** 2
    prior_score = -(state[:1, :, :2].double()) / (1.0 + lam ** 2)
    prior_precision = ((1.0 + 1.0 / lam ** 2) * torch.eye(2, dtype=torch.float64)).reshape(1, 1, 2, 2)
    u = compose_tree(tree, scores[..., :3].double(), leaf_precision, [prior_score], [prior_precision])
    torch.testing.assert_close(u[0][0].float(), production[0, :, :2], rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(u[1][..., 0].float(), production[:, :, 2], rtol=2e-3, atol=2e-4)


def test_single_leaf_returns_the_leaf_score():
    tree = NestedTree.balanced([2, 1, 1], [1, 1])
    generator = torch.Generator().manual_seed(3)
    factors = random_factors(tree, 4, generator)
    u = compose_tree(tree, *factors)
    torch.testing.assert_close(torch.cat([u[0][0], u[1][0], u[2][0]], -1), factors[0][0])


def test_finite_difference_jacobian_matches_exact():
    tree, model, x = linear_model([2, 2, 1], [2, 2])
    score, jac = model.path_score(x)
    paths = torch.randn(8, tree.offsets[-1], dtype=torch.float64)
    leaf_index = torch.arange(4).repeat(2)
    for lam in (2.0, 0.05):
        fd = finite_difference_jacobian(lambda p, l: score(p, l, leaf_index.repeat(6)), paths, lam)
        torch.testing.assert_close(fd, jac(paths, lam, leaf_index), rtol=1e-6, atol=1e-6)


def test_projection_rescues_an_indefinite_merge():
    tree = NestedTree.balanced([1, 1], [6])
    S = 2
    eye = torch.eye(2, dtype=torch.float64)
    leaf_precision = (0.5 * eye).reshape(1, 1, 2, 2).expand(6, S, 2, 2)  # narrower than prior: I < 0
    prior_precision = eye[:1, :1].reshape(1, 1, 1, 1) * 1.0
    leaf_scores = torch.randn(6, S, 2, dtype=torch.float64)
    prior_scores = [torch.zeros(1, S, 1, dtype=torch.float64)]
    try:
        compose_tree(tree, leaf_scores, leaf_precision, prior_scores, [prior_precision], project=False)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    diagnostics = {}
    u = compose_tree(tree, leaf_scores, leaf_precision, prior_scores, [prior_precision],
                     project=True, diagnostics=diagnostics)
    assert all(torch.isfinite(v).all() for v in u)
    assert diagnostics["projected"][1] > 0


def test_sampler_recovers_the_three_level_posterior():
    tree, model, x = linear_model([2, 2, 1], [3, 3], seed=4)
    mean, cov = model.posterior(x)
    sd = cov.diagonal().sqrt()
    score, jac = model.path_score(x)

    def draw(**kw):
        sampler = MultiLevelSampler(tree, score, model.prior(), jacobian_fn=jac, **kw)
        out = sampler.sample(2000, lam_max=20.0, steps=30, generator=torch.Generator().manual_seed(5))
        return tree.flatten(out)

    flat = draw(precision="jacobian_fd", refresh=3)
    mean_error = float(((flat.mean(0) - mean).abs() / sd).max())
    ratio = flat.std(0) / sd
    assert mean_error < 0.12, mean_error            # MC floor/coordinate ~0.022, 15 coordinates
    assert 0.9 < float(ratio.min()) and float(ratio.max()) < 1.1, ratio

    wrong = draw(precision="jacobian", drop_internal_priors=True)
    wrong_ratio = wrong.std(0) / sd
    assert float((wrong_ratio - 1).abs().max()) > 0.2
