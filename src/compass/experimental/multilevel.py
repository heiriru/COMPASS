"""Multi-level (N >= 2) hierarchical GAUSS / JAC composition on nested trees.

Implements the rule derived in ``docs/multilevel_composition/three_level_composition.tex``
(Theorem "N-level composition rule"). The model is a nested tree

    p(theta, X) = p(theta_root) prod_{non-root nodes} p(theta_v | ancestors) prod_leaves p(x_v | path(v)),

with one observation per leaf. The single-observation network is trained on whole
root-to-leaf paths ``(theta_path, x)``. The exact factorisation is

    p(theta | X)  ∝  prod_{internal v} p(theta_path(v))^(1 - #children(v)) prod_leaves p(theta_path(leaf) | x_leaf),

and replacing every backward kernel by a Gaussian (precision from the network's
Jacobian = JAC, or from a pilot covariance = GH) gives a nested arrowhead system
``Lambda u = r`` for the composed score ``u`` of *every* latent at every level. It is
solved by one upward pass (eliminate a level, merge each node's children) and one
downward pass (Gaussian conditional-mean corrections). Cost is linear in the number
of nodes. For N = 2 this is exactly ``gauss_hierarchical`` / ``gauss_jacobian``.

Everything works in the sampler's ``y = x / alpha`` coordinates with noise level
``lam`` (``y = theta_0 + lam * eps``), so VE and VP models are handled alike.

Nothing in ``compass.MultiObsSampler`` is touched; delete this module and its test to
remove the feature.

Layout
------
A :class:`NestedTree` has ``N`` levels with dimensions ``dims[m]`` and ``parents[m]``
(index of each level-m node's parent at level m-1; ``parents[0]`` is ``None`` and level
0 holds the single root). Leaves are the level ``N-1`` nodes, one observation each. A
state is a list ``levels[m]`` of tensors ``(n_m, S, d_m)`` (``S`` = number of samples).
"""
import math

import torch


class NestedTree:
    """Index bookkeeping for a nested hierarchy whose leaves are the observations."""

    def __init__(self, dims, parents):
        if len(dims) != len(parents) or len(dims) < 2:
            raise ValueError("need dims and parents for at least two levels")
        if parents[0] is not None:
            raise ValueError("parents[0] must be None (single root)")
        self.dims = [int(d) for d in dims]
        if any(d < 0 for d in self.dims) or self.dims[0] < 1:
            raise ValueError("dims must be >= 0, root dimension >= 1")
        self.depth = len(dims)
        self.parents = [None] + [torch.as_tensor(p, dtype=torch.long) for p in parents[1:]]
        self.sizes = [1] + [int(p.numel()) for p in self.parents[1:]]
        for m in range(1, self.depth):
            p = self.parents[m]
            if p.numel() and (int(p.min()) < 0 or int(p.max()) >= self.sizes[m - 1]):
                raise ValueError(f"parents[{m}] refers to a missing node")
            counts = torch.bincount(p, minlength=self.sizes[m - 1])
            if bool((counts == 0).any()):
                raise ValueError(f"level {m - 1} has a node without children")
        self.offsets = [0]
        for d in self.dims:
            self.offsets.append(self.offsets[-1] + d)
        # ancestors[m][k]: index of each level-m node's ancestor at level k <= m
        self.ancestors = []
        for m in range(self.depth):
            chain = {m: torch.arange(self.sizes[m])}
            for k in range(m - 1, -1, -1):
                chain[k] = self.parents[k + 1][chain[k + 1]]
            self.ancestors.append(chain)

    @classmethod
    def balanced(cls, dims, branching):
        """Balanced tree: ``branching[m-1]`` children per level-(m-1) node."""
        parents = [None]
        size = 1
        for b in branching:
            parents.append(torch.arange(size).repeat_interleave(int(b)))
            size *= int(b)
        return cls(dims, parents)

    def children_counts(self, m):
        """Number of children of each level-m node (m < depth - 1)."""
        return torch.bincount(self.parents[m + 1], minlength=self.sizes[m])

    def path_dim(self, m):
        return self.offsets[m + 1]

    def path(self, levels, m):
        """Root-to-node path state of every level-m node: ``(n_m, S, D_m)``."""
        parts = [levels[k][self.ancestors[m][k].to(levels[k].device)] for k in range(m + 1)]
        return torch.cat(parts, dim=-1)

    def scatter_path(self, values, m):
        """Inverse bookkeeping of :meth:`path`: split ``(n_m, S, D_m)`` into per-level blocks
        and sum each block onto its node (used for shared coordinates)."""
        out = []
        for k in range(m + 1):
            block = values[..., self.offsets[k]:self.offsets[k + 1]]
            target = torch.zeros((self.sizes[k],) + block.shape[1:], dtype=block.dtype,
                                 device=block.device)
            target.index_add_(0, self.ancestors[m][k].to(block.device), block)
            out.append(target)
        return out

    def total_latents(self):
        return sum(n * d for n, d in zip(self.sizes, self.dims))

    def flatten(self, levels):
        """Concatenate a state into ``(S, total_latents)`` (node-major within a level)."""
        S = levels[0].shape[1]
        return torch.cat([lv.permute(1, 0, 2).reshape(S, -1) for lv in levels], dim=1)

    def unflatten(self, flat):
        S = flat.shape[0]
        out, start = [], 0
        for n, d in zip(self.sizes, self.dims):
            out.append(flat[:, start:start + n * d].reshape(S, n, d).permute(1, 0, 2).contiguous())
            start += n * d
        return out


def _safe_eigh(matrix, chunk=4096):
    """Batched symmetric eigh, chunked (CUDA workspace grows with the batch), with a CPU
    float64 retry plus a 1e-12-relative jitter for chunks the GPU kernel fails to converge."""
    shape = matrix.shape
    flat = matrix.reshape(-1, shape[-2], shape[-1])
    values, vectors = [], []
    for start in range(0, flat.shape[0], chunk):
        block = flat[start:start + chunk]
        try:
            v, w = torch.linalg.eigh(block)
        except RuntimeError:
            cpu = block.detach().to("cpu", torch.float64)
            scale = cpu.diagonal(dim1=-2, dim2=-1).abs().amax(-1).clamp_min(1e-30)
            cpu = cpu + 1e-12 * scale.reshape(-1, 1, 1) * torch.eye(shape[-1], dtype=torch.float64)
            v, w = torch.linalg.eigh(cpu)
            v, w = v.to(block.device, block.dtype), w.to(block.device, block.dtype)
        values.append(v)
        vectors.append(w)
    return torch.cat(values).reshape(shape[:-1]), torch.cat(vectors).reshape(shape)


def _psd_project(matrix):
    """Nearest PSD matrix (eigenvalues clipped at 0) and how many matrices had an
    eigenvalue below rounding level (-1e-10 x their largest magnitude)."""
    sym = 0.5 * (matrix + matrix.mT)
    values, vectors = _safe_eigh(sym)
    scale = values.abs().amax(-1, keepdim=True).clamp_min(1e-300)
    clipped = values < -1e-10 * scale
    if not bool(clipped.any()):
        return sym, 0
    values = values.clamp_min(0.0)
    return (vectors * values.unsqueeze(-2)) @ vectors.mT, int(clipped.any(-1).sum())


def compose_tree(tree, leaf_scores, leaf_precisions, prior_scores, prior_precisions,
                 project=True, diagnostics=None):
    """Composed score of every latent (Theorem "N-level composition rule").

    Args:
        tree: :class:`NestedTree` with ``N`` levels.
        leaf_scores: ``(n_{N-1}, S, D)`` network path scores ``s_leaf`` (y coordinates).
        leaf_precisions: ``(n_{N-1}, S or 1, D, D)`` Gaussian backward precisions of the
            single-observation paths.
        prior_scores: list over internal levels ``m = 0..N-2`` of ``(n_m, S, D_m)`` diffused
            path-prior scores ``grad log p_t(theta_path(v))``.
        prior_precisions: list over ``m = 0..N-2`` of ``(n_m or 1, S or 1, D_m, D_m)``
            Gaussian backward precisions of the path priors.
        project: project every child's information ``S_c - P_0(parent)`` onto the PSD cone
            (the N-level generalisation of the production information projection; with
            exact Gaussian factors it is a no-op). Without it the merged precision can be
            indefinite and the solve raises.
        diagnostics: optional dict; receives ``projected[m]`` counts.

    Returns:
        list ``u[m]`` of ``(n_m, S, d_m)`` composed scores.
    """
    N = tree.depth
    dtype = torch.float64
    S = leaf_scores.shape[1]
    Q = leaf_precisions.to(dtype)
    Q = Q.expand(Q.shape[0], S, Q.shape[-2], Q.shape[-1])
    sigma = leaf_scores.to(dtype)
    q = (Q @ sigma.unsqueeze(-1)).squeeze(-1)
    down = [None] * N
    for m in range(N - 1, 0, -1):
        Da = tree.offsets[m]
        Qaa, Qac = Q[..., :Da, :Da], Q[..., :Da, Da:]
        Qca, Qcc = Q[..., Da:, :Da], Q[..., Da:, Da:]
        qa, qc = q[..., :Da], q[..., Da:]
        if tree.dims[m] > 0:
            factor, info = torch.linalg.cholesky_ex(Qcc)
            if bool((info != 0).any()):
                raise RuntimeError(f"level-{m} pivot is not positive definite")
            gain = torch.cholesky_solve(Qca, factor)                     # Qcc^-1 Qca
            message_precision = Qaa - Qac @ gain
            message_info = qa - (Qac @ torch.cholesky_solve(qc.unsqueeze(-1), factor)).squeeze(-1)
            down[m] = (factor, Qca, qc)
        else:
            message_precision, message_info = Qaa, qa
            down[m] = None
        message_precision = 0.5 * (message_precision + message_precision.mT)
        parent = tree.parents[m].to(Q.device)
        n_parent = tree.sizes[m - 1]
        P0 = prior_precisions[m - 1].to(dtype)
        P0 = P0.expand(n_parent, S, Da, Da)
        s0 = prior_scores[m - 1].to(dtype).expand(n_parent, S, Da)
        P0_child = P0[parent]
        information = message_precision - P0_child
        if project:
            projected, count = _psd_project(information)
            if diagnostics is not None:
                diagnostics.setdefault("projected", {})[m] = count
            # the message keeps its own score sigma_c = S_c^-1 tau_c; the lift moves only
            # its weight, exactly like the two-level projection
            child_score = torch.linalg.solve(message_precision, message_info.unsqueeze(-1)).squeeze(-1)
            message_info = message_info + ((projected - information) @ child_score.unsqueeze(-1)).squeeze(-1)
            information = projected
        merged = torch.zeros(n_parent, S, Da, Da, dtype=dtype, device=Q.device)
        merged.index_add_(0, parent, information)
        Q = P0 + merged
        weighted_prior = (P0 @ s0.unsqueeze(-1)).squeeze(-1)
        q = torch.zeros(n_parent, S, Da, dtype=dtype, device=Q.device)
        q.index_add_(0, parent, message_info - (P0_child @ s0[parent].unsqueeze(-1)).squeeze(-1))
        # q = sum_c tau_c + (1 - n) P0 s0, written as P0 s0 + sum_c (tau_c - P0 s0)
        q = q + weighted_prior
    factor, info = torch.linalg.cholesky_ex(Q)
    if bool((info != 0).any()):
        raise RuntimeError("root precision is not positive definite (try project=True)")
    u = [None] * N
    u[0] = torch.cholesky_solve(q.unsqueeze(-1), factor).squeeze(-1)
    for m in range(1, N):
        if down[m] is None:
            u[m] = torch.zeros(tree.sizes[m], S, 0, dtype=dtype, device=Q.device)
            continue
        factor_m, Qca, qc = down[m]
        ancestors = torch.cat(
            [u[k][tree.ancestors[m][k].to(Q.device)] for k in range(m)], dim=-1)
        rhs = qc - (Qca @ ancestors.unsqueeze(-1)).squeeze(-1)
        u[m] = torch.cholesky_solve(rhs.unsqueeze(-1), factor_m).squeeze(-1)
    return u


def dense_composition(tree, leaf_scores, leaf_precisions, prior_scores, prior_precisions):
    """Reference: assemble ``Lambda`` and ``r`` densely and solve (tests only, O(total^3))."""
    N = tree.depth
    S = leaf_scores.shape[1]
    total = tree.total_latents()
    base = []
    start = 0
    for n, d in zip(tree.sizes, tree.dims):
        base.append(start)
        start += n * d

    def selector(m, node):
        idx = []
        for k in range(m + 1):
            a = int(tree.ancestors[m][k][node])
            idx.extend(range(base[k] + a * tree.dims[k], base[k] + (a + 1) * tree.dims[k]))
        return torch.tensor(idx, dtype=torch.long)

    Lam = torch.zeros(S, total, total, dtype=torch.float64)
    r = torch.zeros(S, total, dtype=torch.float64)
    P = leaf_precisions.to(torch.float64).expand(leaf_scores.shape[0], S, *leaf_precisions.shape[-2:])
    for j in range(tree.sizes[N - 1]):
        idx = selector(N - 1, j)
        Lam[:, idx[:, None], idx[None, :]] += P[j]
        r[:, idx] += (P[j] @ leaf_scores[j].to(torch.float64).unsqueeze(-1)).squeeze(-1)
    for m in range(N - 1):
        counts = tree.children_counts(m)
        P0 = prior_precisions[m].to(torch.float64)
        D = tree.offsets[m + 1]
        P0 = P0.expand(tree.sizes[m], S, D, D)
        s0 = prior_scores[m].to(torch.float64).expand(tree.sizes[m], S, D)
        for v in range(tree.sizes[m]):
            w = 1.0 - float(counts[v])
            idx = selector(m, v)
            Lam[:, idx[:, None], idx[None, :]] += w * P0[v]
            r[:, idx] += w * (P0[v] @ s0[v].unsqueeze(-1)).squeeze(-1)
    flat = torch.linalg.solve(Lam, r.unsqueeze(-1)).squeeze(-1)
    return tree.unflatten(flat)


# --------------------------------------------------------------------------------------
# Path priors
# --------------------------------------------------------------------------------------

class GaussianPathPrior:
    """Gaussian prior on the internal paths: ``theta_path(level m) ~ N(mean[m], cov[m])``.

    ``score`` and ``precision`` are the exact diffused score and Gaussian backward
    precision ``cov^-1 + lam^-2 I`` (exact for a Gaussian prior)."""

    def __init__(self, means, covariances):
        self.means = [torch.as_tensor(m, dtype=torch.float64) for m in means]
        self.covariances = [torch.as_tensor(c, dtype=torch.float64) for c in covariances]
        self.precisions = [torch.linalg.inv(c) for c in self.covariances]

    def terms(self, tree, levels, lam):
        scores, precisions = [], []
        for m in range(tree.depth - 1):
            path = tree.path(levels, m).to(torch.float64)
            D = path.shape[-1]
            eye = torch.eye(D, dtype=torch.float64, device=path.device)
            cov_t = self.covariances[m].to(path.device) + lam ** 2 * eye
            delta = path - self.means[m].to(path.device)
            scores.append(-torch.linalg.solve(cov_t, delta.unsqueeze(-1)).squeeze(-1))
            precisions.append((self.precisions[m].to(path.device) + eye / lam ** 2)
                              .reshape(1, 1, D, D))
        return scores, precisions


# --------------------------------------------------------------------------------------
# Leaf (path) score models
# --------------------------------------------------------------------------------------

def finite_difference_jacobian(score_fn, paths, lam, step=1e-2):
    """Forward-difference Jacobian of ``score_fn`` at ``paths (B, D)``: ``(B, D, D)``,
    one stacked call with all D+1 copies (the library's FD scheme, step ``step * lam``)."""
    B, D = paths.shape
    h = step * lam
    copies = [paths] + [paths + h * torch.nn.functional.one_hot(
        torch.full((B,), k, device=paths.device), D).to(paths.dtype) for k in range(D)]
    out = score_fn(torch.cat(copies, 0), lam).reshape(D + 1, B, D).to(torch.float64)
    columns = (out[1:] - out[:1]) / h                                   # (D_in, B, D_out)
    return columns.permute(1, 2, 0).contiguous()


def jacobian_precision(jacobian, lam, floor=1e-3, lower_clip=True):
    """``Lambda = [lam^2 (I + lam^2 grad s)]^-1`` with the production curvature clamp:
    eigenvalues of ``H = -grad s`` (symmetrised) clipped into ``[0, (1-floor)/lam^2]``."""
    var = lam ** 2
    H = -0.5 * (jacobian + jacobian.mT)
    values, vectors = _safe_eigh(torch.nan_to_num(H.to(torch.float64)))
    values = values.clamp(min=0.0 if lower_clip else None, max=(1.0 - floor) / var)
    precision = 1.0 / (var * (1.0 - var * values))
    return (vectors * precision.unsqueeze(-2)) @ vectors.mT


class SBIMPathModel:
    """Leaf path scores from a trained COMPASS ``ScoreBasedInferenceModel``.

    Args:
        sbim: the model (its ``model`` network, ``sde`` and ``output_scale_function``).
        observations: ``(n_leaves, X)`` observed values, one row per leaf.
        path_nodes: node index of every path coordinate, root first (length D).
        observed_nodes: node indices of the observation (length X).
    """

    def __init__(self, sbim, observations, path_nodes, observed_nodes, device="cpu",
                 max_rows=65536):
        self.sbim = sbim
        self.device = device
        self.observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
        self.path_nodes = list(path_nodes)
        self.observed_nodes = list(observed_nodes)
        self.nodes = len(self.path_nodes) + len(self.observed_nodes)
        mask = torch.zeros(self.nodes, device=device)
        mask[self.observed_nodes] = 1.0
        self.mask = mask
        self.max_rows = int(max_rows)
        self.calls = 0
        self.rows = 0

    def __call__(self, paths, lam, leaf_index=None):
        """Scores ``(B, D)`` for paths ``(B, D)``; row ``b`` belongs to leaf ``leaf_index[b]``
        (default: rows are ``copies x leaves x samples`` in that order, as produced by the
        sampler)."""
        sde = self.sbim.sde
        lam_t = torch.as_tensor(lam, dtype=torch.float32, device=self.device)
        t = sde.time_of_lambda(lam_t).reshape(1, 1)
        alpha = sde.alpha_t(t).to(self.device)
        B = paths.shape[0]
        x = torch.zeros(B, self.nodes, device=self.device)
        x[:, self.path_nodes] = alpha * paths.to(self.device, torch.float32)
        x[:, self.observed_nodes] = self.observations[leaf_index]
        c = self.mask.expand(B, -1)
        out = []
        with torch.no_grad():
            for start in range(0, B, self.max_rows):
                sl = slice(start, start + self.max_rows)
                tt = t.expand(x[sl].shape[0], 1)
                s = self.sbim.model(x=x[sl], t=tt, c=c[sl])
                s = self.sbim.output_scale_function(tt, s)
                out.append(alpha * s[:, self.path_nodes])
                self.calls += 1
        self.rows += B
        return torch.cat(out, 0).to(paths.device, torch.float64)


# --------------------------------------------------------------------------------------
# Exact linear-Gaussian N-level model (oracle + reference posterior)
# --------------------------------------------------------------------------------------

class LinearGaussianTree:
    """``theta_root ~ N(0, s_root^2 I)``, ``theta_v = A_m theta_parent + tau_m eps``,
    ``x_leaf = B path(leaf) + nu eps``. Every quantity the rule needs is closed form."""

    def __init__(self, tree, A, tau, B, nu, s_root=1.0):
        self.tree = tree
        self.A = [None] + [torch.as_tensor(a, dtype=torch.float64) for a in A]
        self.tau = [None] + [float(v) for v in tau]
        self.B = torch.as_tensor(B, dtype=torch.float64)
        self.nu = float(nu)
        self.s_root = float(s_root)
        # path prior covariance per level: path_m = M_m eps_path
        self.path_cov = []
        M = self.s_root * torch.eye(tree.dims[0], dtype=torch.float64)
        self.path_cov.append(M @ M.T)
        for m in range(1, tree.depth):
            d = tree.dims[m]
            prev = M.shape[1]
            last = M[tree.offsets[m - 1]:tree.offsets[m]]                # previous node block
            new = torch.cat([self.A[m] @ last, torch.zeros(d, d, dtype=torch.float64)], 1)
            new[:, prev:] = self.tau[m] * torch.eye(d, dtype=torch.float64)
            M = torch.cat([torch.cat([M, torch.zeros(M.shape[0], d, dtype=torch.float64)], 1), new], 0)
            self.path_cov.append(M @ M.T)
        self.X = self.B.shape[0]
        D = tree.offsets[-1]
        C = self.path_cov[-1]
        # single-observation path posterior p(path | x): Gaussian
        Sxx = self.B @ C @ self.B.T + self.nu ** 2 * torch.eye(self.X, dtype=torch.float64)
        self.gain = C @ self.B.T @ torch.linalg.inv(Sxx)                 # (D, X)
        self.post_cov = C - self.gain @ self.B @ C
        self.post_cov = 0.5 * (self.post_cov + self.post_cov.T)
        self.D = D

    def prior(self):
        D = [self.tree.offsets[m + 1] for m in range(self.tree.depth - 1)]
        return GaussianPathPrior([torch.zeros(d, dtype=torch.float64) for d in D],
                                 self.path_cov[:-1])

    def joint_latent_covariance(self):
        """Prior covariance of all latents in :meth:`NestedTree.flatten` order."""
        tree = self.tree
        total = tree.total_latents()
        base, start = [], 0
        for n, d in zip(tree.sizes, tree.dims):
            base.append(start)
            start += n * d
        # theta = M eps, eps one standard block per node
        M = torch.zeros(total, total, dtype=torch.float64)
        for m in range(tree.depth):
            d = tree.dims[m]
            for v in range(tree.sizes[m]):
                rows = slice(base[m] + v * d, base[m] + (v + 1) * d)
                if m == 0:
                    M[rows, rows] = self.s_root * torch.eye(d, dtype=torch.float64)
                else:
                    p = int(tree.parents[m][v])
                    dp = tree.dims[m - 1]
                    prow = slice(base[m - 1] + p * dp, base[m - 1] + (p + 1) * dp)
                    M[rows] = self.A[m] @ M[prow]
                    M[rows, rows] += self.tau[m] * torch.eye(d, dtype=torch.float64)
        return M @ M.T, base

    def leaf_selectors(self):
        tree = self.tree
        _, base = self.joint_latent_covariance()
        sel = []
        for j in range(tree.sizes[-1]):
            idx = []
            for k in range(tree.depth):
                a = int(tree.ancestors[tree.depth - 1][k][j])
                idx.extend(range(base[k] + a * tree.dims[k], base[k] + (a + 1) * tree.dims[k]))
            sel.append(idx)
        return torch.tensor(sel, dtype=torch.long)

    def simulate(self, generator=None):
        """One dataset: latents (flattened) and observations ``(n_leaves, X)``."""
        cov, _ = self.joint_latent_covariance()
        L = torch.linalg.cholesky(cov + 1e-12 * torch.eye(cov.shape[0], dtype=torch.float64))
        theta = L @ torch.randn(cov.shape[0], generator=generator, dtype=torch.float64)
        sel = self.leaf_selectors()
        paths = theta[sel]                                             # (n_leaves, D)
        x = paths @ self.B.T + self.nu * torch.randn(paths.shape[0], self.X, generator=generator,
                                                     dtype=torch.float64)
        return theta, x

    def simulate_paths(self, count, generator=None):
        """Training units: ``count`` independent (path, x) pairs."""
        L = torch.linalg.cholesky(self.path_cov[-1])
        paths = torch.randn(count, self.D, generator=generator, dtype=torch.float64) @ L.T
        x = paths @ self.B.T + self.nu * torch.randn(count, self.X, generator=generator,
                                                     dtype=torch.float64)
        return paths, x

    def posterior(self, observations):
        """Exact ``p(all latents | X)``: mean ``(total,)`` and covariance."""
        cov, _ = self.joint_latent_covariance()
        sel = self.leaf_selectors()
        obs = torch.as_tensor(observations, dtype=torch.float64)
        n, X = obs.shape
        # x = G theta + noise, G stacks B on each leaf's path coordinates
        G = torch.zeros(n * X, cov.shape[0], dtype=torch.float64)
        for j in range(n):
            G[j * X:(j + 1) * X, sel[j]] = self.B
        precision = torch.linalg.inv(cov) + G.T @ G / self.nu ** 2
        post_cov = torch.linalg.inv(precision)
        post_mean = post_cov @ (G.T @ obs.reshape(-1) / self.nu ** 2)
        return post_mean, 0.5 * (post_cov + post_cov.T)

    def path_score(self, observations):
        """Exact leaf path score/Jacobian functions for the given observations."""
        obs = torch.as_tensor(observations, dtype=torch.float64)
        means = obs @ self.gain.T                                       # (n, D)
        cov = self.post_cov
        eye = torch.eye(self.D, dtype=torch.float64)

        def score(paths, lam, leaf_index):
            delta = paths.to(torch.float64) - means[leaf_index]
            return -torch.linalg.solve(cov + lam ** 2 * eye, delta.T).T

        def jacobian(paths, lam, leaf_index):
            J = -torch.linalg.inv(cov + lam ** 2 * eye)
            return J.expand(paths.shape[0], self.D, self.D)

        return score, jacobian


# --------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------

class MultiLevelSampler:
    """Reverse diffusion on the whole tree, driven by :func:`compose_tree`.

    Args:
        tree: :class:`NestedTree`.
        score_fn: ``score_fn(paths (B, D), lam, leaf_index (B,)) -> (B, D)`` leaf path
            scores in y coordinates (an :class:`SBIMPathModel` or an oracle).
        prior: object with ``terms(tree, levels, lam) -> (scores, precisions)`` for the
            internal paths (e.g. :class:`GaussianPathPrior`).
        precision: ``"jacobian_fd"`` (JAC with finite differences), ``"jacobian"`` (needs
            ``jacobian_fn``), or ``"pilot"`` (constant ``pilot_covariance`` (n_leaves, D, D),
            i.e. the multi-level ``gauss_hierarchical``).
        drop_internal_priors: ablation that drops the mid-level prior factors (exponent
            1 - n_v for every non-root internal node), keeping only the root correction
            with the total leaf count. Wrong by construction; used to show the test has power.
    """

    def __init__(self, tree, score_fn, prior, precision="jacobian_fd", jacobian_fn=None,
                 pilot_covariance=None, refresh=1, fd_step=1e-2, curvature_floor=1e-3,
                 project=True, drop_internal_priors=False, denoise_clamp=None,
                 clamp_center=None, clamp_scale=None):
        if precision not in ("jacobian_fd", "jacobian", "pilot"):
            raise ValueError("precision must be 'jacobian_fd', 'jacobian' or 'pilot'")
        if precision == "jacobian" and jacobian_fn is None:
            raise ValueError("precision='jacobian' needs jacobian_fn")
        if precision == "pilot" and pilot_covariance is None:
            raise ValueError("precision='pilot' needs pilot_covariance")
        self.tree = tree
        self.score_fn = score_fn
        self.prior = prior
        self.precision = precision
        self.jacobian_fn = jacobian_fn
        self.pilot_precision = (None if pilot_covariance is None else
                                torch.linalg.inv(torch.as_tensor(pilot_covariance, dtype=torch.float64)))
        self.refresh = int(refresh)
        self.fd_step = float(fd_step)
        self.curvature_floor = float(curvature_floor)
        self.project = bool(project)
        self.drop_internal_priors = bool(drop_internal_priors)
        self.denoise_clamp = denoise_clamp
        self.clamp_center = clamp_center
        self.clamp_scale = clamp_scale
        self.diagnostics = {}
        self.evaluations = 0
        self._cached_information = None

    def _leaf_precision(self, paths, lam, leaf_index, n, S, D):
        if self.precision == "pilot":
            eye = torch.eye(D, dtype=torch.float64, device=paths.device)
            return (self.pilot_precision.to(paths.device) + eye / lam ** 2).unsqueeze(1)
        if self._cached_information is None or self.evaluations % self.refresh == 0:
            if self.precision == "jacobian":
                J = self.jacobian_fn(paths, lam, leaf_index)
            else:
                J = finite_difference_jacobian(
                    lambda p, l: self.score_fn(p, l, leaf_index.repeat(D + 1)),
                    paths, lam, self.fd_step)
            precision = jacobian_precision(J.to(torch.float64), lam, self.curvature_floor)
            eye = torch.eye(D, dtype=torch.float64, device=paths.device)
            # cache the lambda-free information, as the production refresh does
            self._cached_information = (precision - eye / lam ** 2).reshape(n, S, D, D)
        eye = torch.eye(D, dtype=torch.float64, device=paths.device)
        return self._cached_information + eye / lam ** 2

    def composed_score(self, levels, lam):
        tree = self.tree
        n, S = tree.sizes[-1], levels[0].shape[1]
        paths = tree.path(levels, tree.depth - 1)
        D = paths.shape[-1]
        leaf_index = torch.arange(n, device=paths.device).repeat_interleave(S)
        flat = paths.reshape(n * S, D)
        scores = self.score_fn(flat, lam, leaf_index).reshape(n, S, D).to(torch.float64)
        precision = self._leaf_precision(flat, lam, leaf_index, n, S, D)
        self.evaluations += 1
        if self.denoise_clamp is not None:
            center = self.clamp_center.to(paths.device)
            scale = self.clamp_scale.to(paths.device)
            lo, hi = center - self.denoise_clamp * scale, center + self.denoise_clamp * scale
            denoised = torch.maximum(torch.minimum(paths + lam ** 2 * scores, hi), lo)
            scores = (denoised - paths) / lam ** 2
        prior_scores, prior_precisions = self.prior.terms(tree, levels, lam)
        if self.drop_internal_priors:
            prior_scores, prior_precisions = _drop_internal(tree, prior_scores, prior_precisions)
        diag = {}
        u = compose_tree(tree, scores, precision, prior_scores, prior_precisions,
                         project=self.project, diagnostics=diag)
        for m, c in diag.get("projected", {}).items():
            self.diagnostics[m] = self.diagnostics.get(m, 0) + c
        return u

    def sample(self, num_samples, lam_max, lam_min=1e-3, steps=50, churn_eta=2.0,
               gamma_cap=math.sqrt(2.0) - 1.0, generator=None, device="cpu", init=None):
        """Heun (EDM) reverse sampler on a geometric lambda grid with Karras churn.
        Returns the final state (list of ``(n_m, S, d_m)``), denoised once at ``lam_min``."""
        tree = self.tree
        lams = torch.logspace(math.log10(lam_max), math.log10(lam_min), steps, dtype=torch.float64)
        if init is None:
            levels = [lam_max * torch.randn(n, num_samples, d, dtype=torch.float64,
                                            generator=generator).to(device)
                      for n, d in zip(tree.sizes, tree.dims)]
        else:
            levels = [lv.clone().to(device, torch.float64) for lv in init]
        self.evaluations = 0
        self._cached_information = None
        for i in range(steps - 1):
            lam, lam_next = float(lams[i]), float(lams[i + 1])
            gamma = min(churn_eta * (lam - lam_next) / lam, gamma_cap) if churn_eta > 0 else 0.0
            lam_hat = lam * (1.0 + gamma)
            if gamma > 0:
                spread = math.sqrt(lam_hat ** 2 - lam ** 2)
                levels = [lv + spread * torch.randn(lv.shape, dtype=lv.dtype,
                                                    generator=generator).to(lv.device)
                          for lv in levels]
            u = self.composed_score(levels, lam_hat)
            d1 = [-lam_hat * ui for ui in u]                             # dy/dlam
            trial = [lv + (lam_next - lam_hat) * di for lv, di in zip(levels, d1)]
            u2 = self.composed_score(trial, lam_next)
            d2 = [-lam_next * ui for ui in u2]
            levels = [lv + (lam_next - lam_hat) * 0.5 * (a + b) for lv, a, b in zip(levels, d1, d2)]
        # Tweedie denoise at the last level
        u = self.composed_score(levels, float(lams[-1]))
        return [lv + float(lams[-1]) ** 2 * ui for lv, ui in zip(levels, u)]


def _drop_internal(tree, prior_scores, prior_precisions):
    """Ablation: zero the factor of every non-root internal node, and give the root the
    weight 1 - (total leaves) by rescaling its precision. Implemented by replacing each
    mid-level prior with its root marginal is not expressible here, so we instead set the
    mid-level prior precision to zero (removes those factors) and scale the root factor so
    its total weight is (1 - L) instead of (1 - n_root)."""
    out_s, out_p = list(prior_scores), list(prior_precisions)
    L = tree.sizes[-1]
    n_root = int(tree.children_counts(0)[0])
    for m in range(1, tree.depth - 1):
        out_p[m] = torch.zeros_like(out_p[m])
    if tree.depth > 2 and n_root != 1:
        out_p[0] = out_p[0] * ((1.0 - L) / (1.0 - n_root))
    return out_s, out_p


def single_observation_draws(score_fn, n_leaves, D, num_samples, lam_max, lam_min=1e-3,
                             steps=50, churn_eta=2.0, gamma_cap=math.sqrt(2.0) - 1.0,
                             generator=None, device="cpu"):
    """Independent draws from every leaf's single-observation path posterior (no
    composition), with the same Heun + churn sampler. Returns ``(n_leaves, S, D)``.
    Used for the pilot covariance of the multi-level ``gauss_hierarchical`` arm."""
    lams = torch.logspace(math.log10(lam_max), math.log10(lam_min), steps, dtype=torch.float64)
    leaf_index = torch.arange(n_leaves).repeat_interleave(num_samples)

    def score(y, lam):
        return score_fn(y.reshape(-1, D), lam, leaf_index).reshape(n_leaves, num_samples, D)

    y = lam_max * torch.randn(n_leaves, num_samples, D, dtype=torch.float64,
                              generator=generator).to(device)
    for i in range(steps - 1):
        lam, lam_next = float(lams[i]), float(lams[i + 1])
        gamma = min(churn_eta * (lam - lam_next) / lam, gamma_cap) if churn_eta > 0 else 0.0
        lam_hat = lam * (1.0 + gamma)
        if gamma > 0:
            y = y + math.sqrt(lam_hat ** 2 - lam ** 2) * torch.randn(
                y.shape, dtype=y.dtype, generator=generator).to(device)
        d1 = -lam_hat * score(y, lam_hat)
        trial = y + (lam_next - lam_hat) * d1
        d2 = -lam_next * score(trial, lam_next)
        y = y + (lam_next - lam_hat) * 0.5 * (d1 + d2)
    return y + float(lams[-1]) ** 2 * score(y, float(lams[-1]))
