"""Mode-mixture composition: the problem-independent core.

Replaces the one-Gaussian summary of each ``p(theta | x_i)`` by a K-component
Gaussian mixture whose K is *discovered*, and composes those. See
``docs/mode_mixture.md`` for the derivation, the results and the controls; this
module is only the machinery, so that the arm and the Gaussian-mixture benchmark
run the same code rather than two implementations that could drift.

Everything is batched over a **flat row set**: one tensor of candidate states
plus an ``owner`` vector saying which observation each row belongs to. Rows never
mix inside a score network, which is the property the four-JVP Hessian already
relies on, and it holds just as well across observations as within one. That
matters more than it sounds: a batch of 512 rows costs the same wall-clock as a
batch of 1 (2.83 ms against 2.83 ms on the arm's network), so anything that
processes observations one at a time is paying the full launch cost per
observation for nothing.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Problem:
    """What the core needs to know about a benchmark.

    ``latent`` is the size of the per-observation latent block, ``shared`` the
    index of the shared coordinate inside it, and the prior moments are used
    only to draw Newton starts -- never as the composition prior, which each
    caller supplies itself.
    """

    latent: int
    shared: int
    condition_mask: torch.Tensor
    prior_mean: np.ndarray
    prior_std: np.ndarray
    shared_indices: tuple = ()

    def roles(self):
        """``(shared, local)`` index tensors over the latent block."""
        shared = self.shared_indices or (self.shared,)
        local = [i for i in range(self.latent) if i not in shared]
        return list(shared), local


@dataclass
class Settings:
    """Knobs. Defaults are the ones ``docs/mode_mixture.md`` reports."""

    sigma_top: float = 1.0
    sigma_weights: float = 0.40
    ladder: int = 16
    newton_steps: int = 12
    refine_steps: int = 4
    trust_radius: float = 2.0
    curvature_floor: float = 1e-2
    split_scale: float = 3.0
    split_directions: int = 2
    dedup_tolerance: float = 1.5
    max_modes: int = 8
    top_starts: int = 12
    level_starts: int = 4
    path_nodes: int = 64
    path_bend: float = 1.0
    deconvolve: bool = True
    deconvolve_floor: float = 0.1
    curvature_source: str = "joint"
    min_weight: float = 0.0
    force_k1: bool = False
    keep_saddles: bool = False

    def ladder_sigmas(self):
        return np.geomspace(self.sigma_top, self.sigma_weights, self.ladder)


def forward_ad_attention():
    """Attention backend that implements forward-mode AD.

    Free -- 3.38 ms under ``MATH`` against 3.36 ms fused on the arm's network --
    so it is entered unconditionally rather than only for transformer models.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:                                    # pragma: no cover
        from contextlib import nullcontext
        return nullcontext()
    return sdpa_kernel([SDPBackend.MATH])


def role_masks(problem, device):
    """The two conditional masks that between them carry the whole Hessian.

    ``a``: the shared block latent, every local latent conditioned.
    ``b``: the locals latent, the shared block conditioned.

    Both are entries of ``SIX_CONDITIONALS`` -- ``p(g|l,x)`` and ``p(l|g,x)`` --
    so a network trained on that menu has seen them, and they condition whole
    roles rather than splitting one, which is what the menu can express.
    """
    base = problem.condition_mask.to(device)
    shared, _ = problem.roles()
    index = torch.as_tensor(shared, device=device, dtype=torch.long)
    a = base.clone(); a[:problem.latent] = 1.0; a[index] = 0.0
    b = base.clone(); b[:problem.latent] = 0.0; b[index] = 1.0
    return a, b


def score_function(model, problem, device):
    """``grad_theta log p_sigma(theta | x)`` on the latent block, batched.

    Written in ``sigma`` rather than ``t``: both benchmarks run a VESDE, where
    ``alpha = 1`` and ``marginal_prob_std(time_of_sigma(sigma)) == sigma``
    exactly, so the output rescaling is a division by ``sigma`` and the noise
    scale -- the quantity every statement in the docs is about -- stays in the
    API.
    """
    mask = problem.condition_mask.to(device)
    latent = problem.latent

    def score(theta, x, sigma, condition=None):
        state = torch.cat([theta, x], dim=-1)
        t = model.sde.time_of_sigma(torch.as_tensor(float(sigma))).to(device)
        c = mask.expand_as(state) if condition is None else condition
        raw = model.model(x=state, t=t, c=c)
        return raw[:, :latent] / float(sigma)

    return score


def assembled_score_function(model, problem, device):
    """The joint score assembled from the two role masks, component by component.

    The same identity that gives the Hessian gives the score itself: at
    ``sigma = 0``, ``log p(theta|x) = log p(theta_A | theta_B, x)
    + log p(theta_B | x)``, and the second term has no ``theta_A``, so

        d/d theta_A  log p(theta | x)  =  d/d theta_A  log p(theta_A | theta_B, x)

    Take the shared components from the mask where the shared block is latent
    and the local components from the mask where the locals are, and the result
    is the joint score -- but evaluated through conditionals, where the
    conditioned coordinates are valid attention keys.

    This matters for **mode finding**, not just curvature. Under a decoupled
    score, ``s_g`` depends only on ``g`` and ``s_l`` only on ``l``, so the
    implied density factorises and its critical points are the *Cartesian
    product* of the two coordinates' critical points: two branches in ``g``
    times two in ``l`` gives four modes, three of them spurious pairings that
    the true coupled density does not have. Measured on the tokenized
    checkpoint, mode discovery returned exactly that -- four modes at every
    combination of two ``g`` values and two ``l`` values, identical to the last
    digit. Assembling the score from conditionals removes the artifact at its
    source, because ``s_g`` then genuinely depends on ``l``.

    Costs one extra forward pass: the two masks go through as one batch of twice
    the rows, and the batch dimension is free.
    """
    base = score_function(model, problem, device)
    masks = torch.stack(role_masks(problem, device))
    shared, local = problem.roles()

    def score(theta, x, sigma, condition=None):
        rows = theta.shape[0]
        state = theta.repeat(2, 1)
        observations = x.repeat(2, 1)
        conditions = masks.repeat_interleave(rows, 0).to(theta.dtype)
        out = base(state, observations, sigma, conditions)
        first, second = out[:rows], out[rows:]
        result = torch.empty_like(first)
        result[:, shared] = first[:, shared]
        result[:, local] = second[:, local]
        return result

    return score


def score_and_hessian(score, theta, x, sigma, latent):
    """Score and symmetrised Hessian at every row, in four JVPs total.

    A tangent that perturbs coordinate ``j`` of *every* row returns column ``j``
    of *every* row's Jacobian in one forward-mode pass, because rows do not mix.
    The count is therefore the latent dimension, not the row count.
    """
    from torch.func import jvp

    def evaluate(state):
        return score(state, x, sigma)

    values = None
    columns = []
    with forward_ad_attention():
        for index in range(latent):
            tangent = torch.zeros_like(theta)
            tangent[:, index] = 1.0
            values, column = jvp(evaluate, (theta,), (tangent,))
            columns.append(column)
    jacobian = torch.stack(columns, dim=-1).detach()
    values = values.detach()
    antisymmetry = (jacobian - jacobian.transpose(-1, -2)).abs().max()
    scale = jacobian.abs().max().clamp_min(1e-12)
    hessian = 0.5 * (jacobian + jacobian.transpose(-1, -2))
    return values, hessian, float(antisymmetry / scale)


def floored_precision(hessian, sigma, floor):
    """``-H`` with eigenvalues floored, plus the eigendecomposition.

    At a mode ``H`` is negative definite and ``-H`` is the precision; away from
    one it need not be, and flooring is what turns Newton into a modified-Newton
    that always ascends. The floor is relative to ``1/sigma^2``, the curvature
    the noise kernel alone contributes, so it does not need retuning down the
    ladder.
    """
    eigenvalues, vectors = torch.linalg.eigh(-hessian)
    return torch.clamp(eigenvalues, min=floor / sigma ** 2), vectors, eigenvalues


def _deconvolve_block(C, sigma, floor):
    """From ``C = (Sigma_cond + sigma^2 I)^-1`` recover ``Lambda = Sigma_cond^-1``.

    The masked network returns the score of ``p_sigma(theta_i | theta_-i, x)``
    with the conditioned coordinates *clean*, so only the latent block carries
    the kernel: its covariance is the clean conditional covariance plus
    ``sigma^2 I``, exactly, and subtracting it back is a well-posed inverse
    rather than an approximation. Eigen-floored so a conditional the network
    reports as narrower than the kernel it was measured through cannot produce a
    negative variance.
    """
    values, vectors = torch.linalg.eigh(C)
    values = values.clamp_min(1e-9)
    covariance = (1.0 / values - sigma ** 2).clamp_min(floor / values)
    precision = torch.einsum("rij,rj,rkj->rik", vectors, 1.0 / covariance, vectors)
    inverse = torch.einsum("rij,rj,rkj->rik", vectors, 1.0 / values, vectors)
    return precision, inverse


def conditional_precision(score, theta, x, sigma, problem, settings):
    """The **clean** joint precision, assembled from two conditional masks.

    At ``sigma = 0``, ``log p(theta|x) = log p(theta_A | theta_B, x)
    + log p(theta_B | x)``, and the second term does not contain ``theta_A``,
    so every second derivative with at least one ``theta_A`` index is carried by
    the conditional alone. Two role masks therefore span the whole Hessian: the
    shared-latent mask gives the shared rows, the locals-latent mask gives the
    local rows, and each supplies its own estimate of the off-diagonal block.

    Why bother: under a trunk that normalises per token, a coordinate that is
    *latent* is masked out of attention and cannot influence any other
    coordinate's score, so the all-latent Hessian is exactly diagonal. Conditioned
    it becomes a valid attention key, and the coupling reappears. This recovers
    it without a line integral, so it does not import the non-conservativity that
    a coupling trunk brings with it.

    The two off-diagonal estimates must agree; their discrepancy is returned and
    is a network-error bar that nothing else here provides.
    """
    from torch.func import jvp

    dimension = problem.latent
    rows = theta.shape[0]
    shared, local = problem.roles()
    if not local:
        raise RuntimeError("conditional curvature needs a non-empty local block")
    shared_index = torch.as_tensor(shared, device=theta.device, dtype=torch.long)
    local_index = torch.as_tensor(local, device=theta.device, dtype=torch.long)

    masks = torch.stack(role_masks(problem, theta.device))          # (2, nodes)
    state = theta.repeat(2, 1)
    observations = x.repeat(2, 1)
    condition = masks.repeat_interleave(rows, 0).to(theta.dtype)

    columns = []
    with forward_ad_attention():
        for index in range(dimension):
            tangent = torch.zeros_like(state)
            tangent[:, index] = 1.0
            _, column = jvp(lambda s: score(s, observations, sigma, condition),
                            (state,), (tangent,))
            columns.append(column.detach())
    jacobian = torch.stack(columns, dim=-1).reshape(2, rows, dimension, dimension)

    def block(which, out_index, in_index):
        return jacobian[which][:, out_index][:, :, in_index]

    # Shared rows, from the mask whose latent block is the shared one.
    precision_ss, inverse_ss = _deconvolve_block(
        -block(0, shared_index, shared_index), sigma, settings.deconvolve_floor)
    off_from_shared = -precision_ss @ inverse_ss @ block(0, shared_index,
                                                         local_index)
    # Local rows, from the other mask.
    precision_ll, inverse_ll = _deconvolve_block(
        -block(1, local_index, local_index), sigma, settings.deconvolve_floor)
    off_from_local = -precision_ll @ inverse_ll @ block(1, local_index,
                                                        shared_index)

    scale = max(float(precision_ss.abs().max()),
                float(precision_ll.abs().max()), 1e-30)
    disagreement = float((off_from_shared
                          - off_from_local.transpose(-1, -2)).abs().max() / scale)
    off = 0.5 * (off_from_shared + off_from_local.transpose(-1, -2))

    precision = theta.new_zeros(rows, dimension, dimension)
    grid = torch.meshgrid(shared_index, shared_index, indexing="ij")
    precision[:, grid[0], grid[1]] = precision_ss
    grid = torch.meshgrid(local_index, local_index, indexing="ij")
    precision[:, grid[0], grid[1]] = precision_ll
    grid = torch.meshgrid(shared_index, local_index, indexing="ij")
    precision[:, grid[0], grid[1]] = off
    grid = torch.meshgrid(local_index, shared_index, indexing="ij")
    precision[:, grid[0], grid[1]] = off.transpose(-1, -2)

    # Each block is floored positive on its own, but a matrix assembled from
    # separately-estimated blocks need not be positive definite: a too-large
    # off-diagonal makes the Schur complement negative, the inverse has a
    # negative diagonal entry, and the component width comes out NaN. Floor the
    # eigenvalues of the assembled matrix, which is the only place the
    # constraint can be imposed.
    values, vectors = torch.linalg.eigh(precision)
    floor = settings.curvature_floor * values.abs().amax(dim=-1, keepdim=True)
    values = values.clamp_min(floor.clamp_min(1e-9))
    precision = torch.einsum("rij,rj,rkj->rik", vectors, values, vectors)
    return precision, disagreement


def newton_refine(score, theta, x, sigma, steps, settings, latent):
    """Trust-region modified Newton, row-wise. Returns the last curvature too,
    so callers do not pay for a Hessian the final iteration already computed."""
    values = hessian = None
    for _ in range(steps):
        values, hessian, _ = score_and_hessian(score, theta, x, sigma, latent)
        precision, vectors, _ = floored_precision(hessian, sigma,
                                                  settings.curvature_floor)
        rotated = torch.einsum("rij,ri->rj", vectors, values)
        step = torch.einsum("rij,rj->ri", vectors, rotated / precision)
        norm = step.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # The trust radius is a multiple of the *widest local direction*, not of
        # sigma. Tying it to sigma looks natural on the way down the ladder and
        # is wrong at the bottom: the step cap shrinks with sigma while the
        # distances Newton has to travel do not, so probes cannot walk back into
        # the parent they came from and survive deduplication as modes that are
        # really unconverged iterates. Measured on the exact-score benchmark, a
        # sigma-tied radius reports K = 6 at sigma = 0.05 where the truth is 2.
        # The local width already contains sigma -- the noise kernel contributes
        # 1/sigma^2 to the precision -- so this agrees with the old rule high up
        # the ladder and only differs where the old rule failed.
        width = precision.min(dim=-1).values.clamp_min(1e-12).rsqrt()
        limit = (settings.trust_radius * width).unsqueeze(-1)
        theta = (theta + step * torch.clamp(limit / norm, max=1.0)).detach()
    return theta, values, hessian


def deduplicate(theta, owner, hessian, sigma, settings):
    """Merge modes of the *same* observation closer than the tolerance.

    The metric is the local curvature, so the test is "closer than a fraction of
    the width of the component they would belong to" rather than a distance in
    parameter units, which would need retuning per coordinate and per sigma.
    """
    precision, vectors, _ = floored_precision(hessian, sigma,
                                              settings.curvature_floor)
    keep = []
    owner_list = owner.tolist()
    for index in range(theta.shape[0]):
        duplicate = False
        for other in keep:
            if owner_list[other] != owner_list[index]:
                continue
            delta = theta[index] - theta[other]
            for candidate in (index, other):
                rotated = vectors[candidate].transpose(-1, -2) @ delta
                distance = torch.sqrt((rotated ** 2 * precision[candidate]).sum())
                if float(distance) < settings.dedup_tolerance:
                    duplicate = True
                    break
            if duplicate:
                break
        if not duplicate:
            keep.append(index)
    return torch.as_tensor(keep, device=theta.device, dtype=torch.long)


def _cap_per_owner(theta, owner, rank, cap):
    """Keep the ``cap`` highest-ranked rows of each observation."""
    keep = []
    for value in torch.unique(owner):
        rows = torch.nonzero(owner == value, as_tuple=False).reshape(-1)
        order = rows[torch.argsort(rank[rows], descending=True)]
        keep.append(order[:cap])
    return torch.cat(keep)


def discover_modes(score, data, problem, settings, seed):
    """Mode sets for every observation at once, by continuation down the ladder.

    At the top of the ladder the perturbation kernel dominates and each factor
    is unimodal, so K = 1 is known to be right there for free. Walking sigma
    down, each tracked mode is Newton-refined and then probed for a split along
    its flattest curvature directions; whatever survives deduplication is the
    mode set. K is discovered per observation per level, never supplied.

    Returns ``(theta, owner, trace)`` with ``trace`` one entry per level.
    """
    device = data.device
    count = data.shape[0]
    latent = problem.latent
    generator = np.random.default_rng(seed)

    owner = torch.arange(count, device=device).repeat_interleave(
        settings.top_starts)
    theta = torch.as_tensor(
        generator.normal(loc=problem.prior_mean, scale=problem.prior_std,
                         size=(owner.shape[0], latent)),
        dtype=data.dtype, device=device)

    trace = []
    for level, sigma in enumerate(np.asarray(settings.ladder_sigmas())):
        steps = settings.newton_steps if level == 0 else settings.refine_steps
        theta, _, hessian = newton_refine(score, theta, data[owner], sigma,
                                          steps, settings, latent)

        if level > 0 and not settings.force_k1:
            # Probe every tracked mode for a split along its flattest
            # directions. Always probing, rather than probing when an eigenvalue
            # crosses a threshold, keeps K free of a tuned parameter: a probe
            # that falls back into its parent is removed by the deduplication.
            precision, vectors, _ = floored_precision(
                hessian, sigma, settings.curvature_floor)
            offsets = []
            for direction in range(min(settings.split_directions, latent)):
                # eigh returns ascending eigenvalues, so column 0 is the least
                # curved -- the direction a mode is flattest in.
                width = precision[:, direction].clamp_min(1e-12).rsqrt()
                step = (settings.split_scale * width[:, None]
                        * vectors[:, :, direction])
                offsets.extend([theta + step, theta - step])
            probes = torch.cat(offsets, dim=0)
            probe_owner = owner.repeat(len(offsets))
            if settings.level_starts:
                # Insurance against a branch born between two levels. Whether
                # the continuation needed them is visible in the trace.
                fresh_owner = torch.arange(count, device=device
                                           ).repeat_interleave(
                                               settings.level_starts)
                fresh = torch.as_tensor(
                    generator.normal(
                        loc=problem.prior_mean,
                        scale=np.sqrt(np.asarray(problem.prior_std) ** 2
                                      + sigma ** 2),
                        size=(fresh_owner.shape[0], latent)),
                    dtype=theta.dtype, device=device)
                probes = torch.cat([probes, fresh], dim=0)
                probe_owner = torch.cat([probe_owner, fresh_owner])
            probes, _, _ = newton_refine(score, probes, data[probe_owner],
                                         sigma, settings.refine_steps,
                                         settings, latent)
            theta = torch.cat([theta, probes], dim=0)
            owner = torch.cat([owner, probe_owner])
            _, hessian, _ = score_and_hessian(score, theta, data[owner], sigma,
                                              latent)

        if not settings.keep_saddles:
            # Newton with a floored Hessian ascends, but at small sigma it can
            # come to rest on a saddle rather than a maximum. A saddle has a
            # positive Hessian eigenvalue and is not a mixture component. Only
            # applied where it would not empty an observation.
            top = torch.linalg.eigvalsh(hessian)[:, -1]
            survivors = top < 0.0
            for value in torch.unique(owner):
                rows = owner == value
                if not bool(survivors[rows].any()):
                    survivors = survivors | rows
            theta, owner, hessian = (theta[survivors], owner[survivors],
                                     hessian[survivors])

        keep = deduplicate(theta, owner, hessian, sigma, settings)
        theta, owner, hessian = theta[keep], owner[keep], hessian[keep]

        cap = 1 if settings.force_k1 else settings.max_modes
        counts = torch.bincount(owner, minlength=count)
        if int(counts.max()) > cap:
            precision, _, _ = floored_precision(hessian, sigma,
                                                settings.curvature_floor)
            rank = (relative_log_density(score, theta, owner, data, sigma,
                                         settings)[0]
                    - 0.5 * torch.log(precision).sum(-1))
            keep = _cap_per_owner(theta, owner, rank, cap)
            theta, owner, hessian = theta[keep], owner[keep], hessian[keep]

        eigenvalues = torch.linalg.eigvalsh(hessian)
        trace.append({
            "sigma": float(sigma),
            "modes": torch.bincount(owner, minlength=count).tolist(),
            "top_eigenvalue_scaled": float(
                eigenvalues[:, -1].max() * sigma ** 2),
        })

    order = torch.argsort(owner, stable=True)
    return theta[order], owner[order], trace


def multistart_modes(score, data, problem, settings, seed, starts, sigma):
    """Independent mode discovery at one sigma, as a check on the continuation.

    Many starts from the prior, refined and deduplicated. If this finds a mode
    the ladder did not, the ladder lost a branch on the way down rather than
    never having had it.
    """
    device = data.device
    count = data.shape[0]
    owner = torch.arange(count, device=device).repeat_interleave(starts)
    theta = torch.as_tensor(
        np.random.default_rng(seed).normal(
            loc=problem.prior_mean, scale=problem.prior_std,
            size=(owner.shape[0], problem.latent)),
        dtype=data.dtype, device=device)
    theta, _, hessian = newton_refine(score, theta, data[owner], sigma,
                                      3 * settings.newton_steps, settings,
                                      problem.latent)
    keep = deduplicate(theta, owner, hessian, sigma, settings)
    return torch.bincount(owner[keep], minlength=count).tolist()


def _quadrature(nodes, device, dtype):
    positions, weights = np.polynomial.legendre.leggauss(int(nodes))
    return (torch.as_tensor(0.5 * (positions + 1.0), dtype=dtype, device=device),
            torch.as_tensor(0.5 * weights, dtype=dtype, device=device))


def relative_log_density(score, theta, owner, data, sigma, settings,
                         bend=0.0):
    """``log p(mode) - log p(that observation's first mode)``, by line integral.

    The quantity every score-space rule is missing is the log-partition of each
    factor, and relative log density between two modes is a line integral of the
    score along any path joining them. One batched call covers every path of
    every observation.

    ``bend`` displaces each path transversally by ``bend * sigma`` at its
    midpoint. The integral of a true gradient does not care, so the discrepancy
    between a straight and a bent path measures how far the score is from being
    conservative, in the units the weight is reported in.
    """
    device, dtype, latent = theta.device, theta.dtype, theta.shape[1]
    owner_list = owner.tolist()
    first = {}
    for index, value in enumerate(owner_list):
        first.setdefault(value, index)
    anchor = torch.as_tensor([first[value] for value in owner_list],
                             device=device, dtype=torch.long)
    moving = torch.nonzero(anchor != torch.arange(theta.shape[0], device=device),
                           as_tuple=False).reshape(-1)
    result = torch.zeros(theta.shape[0], dtype=dtype, device=device)
    if moving.numel() == 0:
        return result, 0.0

    start = theta[anchor[moving]]
    delta = theta[moving] - start
    u, quadrature = _quadrature(settings.path_nodes, device, dtype)
    base = start[:, None, :] + u[None, :, None] * delta[:, None, :]
    tangent = delta[:, None, :].expand(-1, u.numel(), -1)

    if bend and latent > 1:
        # Any direction orthogonal to the path will do; take the largest
        # component of delta and rotate it into the next coordinate.
        axis = torch.argmax(delta.abs(), dim=-1)
        transverse = torch.zeros_like(delta)
        transverse[torch.arange(delta.shape[0], device=device),
                   (axis + 1) % latent] = 1.0
        projection = (transverse * delta).sum(-1, keepdim=True)
        transverse = transverse - projection * delta / (
            delta.pow(2).sum(-1, keepdim=True).clamp_min(1e-12))
        transverse = transverse / transverse.norm(
            dim=-1, keepdim=True).clamp_min(1e-12)
        base = base + (bend * sigma * torch.sin(np.pi * u)[None, :, None]
                       * transverse[:, None, :])
        tangent = tangent + (bend * sigma * np.pi
                             * torch.cos(np.pi * u)[None, :, None]
                             * transverse[:, None, :])

    flat = base.reshape(-1, latent)
    rows = data[owner[moving]][:, None, :].expand(-1, u.numel(), -1)
    with torch.no_grad():
        values = score(flat, rows.reshape(flat.shape[0], -1), sigma)
    integrand = (values.reshape(moving.numel(), -1, latent) * tangent).sum(-1)
    result[moving] = (integrand * quadrature[None, :]).sum(-1)
    return result, float("nan")


def laplace_log_mass(score, theta, owner, data, hessian, sigma, settings):
    """Relative log mass of each component, plus the path-integral error bar.

    A Laplace component's mass is its peak density times the Gaussian volume
    factor, so the weight needs the relative peak density -- the path integral --
    and the log determinant of the covariance, which the Jacobian already gives.
    """
    precision, _, _ = floored_precision(hessian, sigma, settings.curvature_floor)
    volume = -0.5 * torch.log(precision).sum(-1)
    straight, _ = relative_log_density(score, theta, owner, data, sigma,
                                       settings)
    bent, _ = relative_log_density(score, theta, owner, data, sigma, settings,
                                   bend=settings.path_bend)
    return straight + volume, (straight - bent).abs()


def build_mixtures(score, data, problem, settings, seed, validate_starts=0,
                   weight_score=None):
    """One 1-D mixture on the shared coordinate per observation.

    Every network call in here is batched across all observations at once.

    ``weight_score`` is the field the branch weights are integrated over, and it
    defaults to ``score``. The two are separated because the conditional
    assembly is exact only at ``sigma = 0``: the masked network returns
    ``p_sigma(theta_A | theta_B clean, x)``, which is the right object for
    locating modes and for curvature, and the wrong one for a path integral at
    ``sigma > 0`` -- there the assembled field is not the gradient of anything,
    and on the exact-score benchmark its path discrepancy rises from 8e-3 to
    0.27 nats. The all-latent mask stays self-consistent for the weights of its
    own density, so it keeps that job.
    """
    sigma = settings.sigma_weights
    latent, shared = problem.latent, problem.shared
    theta, owner, trace = discover_modes(score, data, problem, settings, seed)
    rows = data[owner]
    _, hessian, antisymmetry = score_and_hessian(score, theta, rows, sigma,
                                                 latent)
    log_mass, discrepancy = laplace_log_mass(weight_score or score, theta, owner,
                                             data, hessian, sigma, settings)

    disagreement = float("nan")
    if settings.curvature_source == "conditional":
        # The clean precision straight from the conditional masks: no
        # deconvolution step afterwards, because the sigma correction is already
        # inside it and applied per block where it is exact.
        clean, disagreement = conditional_precision(score, theta, rows, sigma,
                                                    problem, settings)
        variance = torch.linalg.inv(clean)[:, shared, shared]
        covariance = None
    else:
        covariance = torch.linalg.inv(
            _precision_matrix(hessian, sigma, settings))
        variance = covariance[:, shared, shared]
    if settings.deconvolve and covariance is not None:
        # A GMM convolved with the perturbation kernel gains exactly sigma^2 on
        # every component covariance; subtract it back to read the sigma = 0
        # mixture. Floored as a fraction of the measured variance rather than
        # clipped to zero: a component narrower than the kernel it was measured
        # through is a measurement failure, and a spike would dominate the
        # product of n factors.
        variance = torch.maximum(variance - sigma ** 2,
                                 settings.deconvolve_floor * variance)

    validation = None
    if validate_starts:
        validation = multistart_modes(score, data, problem, settings,
                                      seed + 7717, validate_starts, sigma)

    mixtures = []
    for index in range(data.shape[0]):
        rows = torch.nonzero(owner == index, as_tuple=False).reshape(-1)
        weights = torch.softmax(log_mass[rows], dim=0)
        # Optional, and **off by default because it is dangerous**. Dropping
        # components that carry a negligible share of their own factor is
        # tempting on the arm, where better-converged Newton finds shallow
        # maxima of the *learned* field that the exact field does not have
        # (K = 3-8 against a 64-start multi-start's 2). It is wrong in general:
        # composition is a product over n factors, so a factor that says 1e-4
        # at some point is not interchangeable with one that says 0 there. On
        # the exact-score benchmark a single legitimate component of weight
        # ~1e-4 dropped this way takes the composed branch mass from 0.752 to
        # 0.000. Spurious modes need a filter that tests whether a mode is
        # spurious, not how little mass it carries.
        if settings.min_weight > 0 and rows.numel() > 1:
            surviving = weights >= settings.min_weight * float(weights.max())
            if bool(surviving.any()):
                rows = rows[surviving]
                weights = torch.softmax(log_mass[rows], dim=0)
        mixtures.append({
            "mean": theta[rows, shared].detach().cpu().numpy().astype(np.float64),
            "variance": variance[rows].detach().cpu().numpy().astype(np.float64),
            "weight": weights.detach().cpu().numpy().astype(np.float64),
            "modes": theta[rows].detach().cpu().numpy().astype(np.float64),
            "log_mass": log_mass[rows].detach().cpu().numpy().astype(np.float64),
            "path_discrepancy": float(discrepancy[rows].max()),
            "K": int(rows.numel()),
        })
    # How strongly the curvature couples the latent coordinates, as the mean
    # absolute off-diagonal of the Hessian over its mean absolute diagonal.
    # Exactly zero means the score for one latent does not respond to another at
    # all, which is a structural property of some architectures rather than
    # something training can fix, and it caps what any curvature-based rule can
    # represent. Reported because it is the quantity, not a side check.
    used = hessian if settings.curvature_source != "conditional" else clean
    off = used - torch.diag_embed(torch.diagonal(used, dim1=-2, dim2=-1))
    coupling = float(off.abs().mean()
                     / torch.diagonal(used, dim1=-2, dim2=-1).abs().mean()
                     .clamp_min(1e-30))
    return mixtures, {"trace": trace, "antisymmetry": antisymmetry,
                      "coupling": coupling, "validation": validation,
                      "mask_disagreement": disagreement,
                      "continuation": [m["K"] for m in mixtures]}


def _precision_matrix(hessian, sigma, settings):
    precision, vectors, _ = floored_precision(hessian, sigma,
                                              settings.curvature_floor)
    return torch.einsum("rij,rj,rkj->rik", vectors, precision, vectors)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

FLOOR = 1e-300


def mixture_density(mixture, grid, key="weight"):
    mean = mixture["mean"][:, None]
    variance = np.maximum(mixture["variance"][:, None], 1e-12)
    weight = mixture[key][:, None]
    kernel = np.exp(-0.5 * (grid[None, :] - mean) ** 2 / variance)
    kernel /= np.sqrt(2.0 * np.pi * variance)
    return (weight * kernel).sum(axis=0)


def compose_mixture_prior(mixtures, grid, prior_weights, prior_means,
                          prior_variances, key="weight", beam=8192):
    """``prod_i q_i / prior^(n-1)`` for a prior that is itself a mixture.

    A Gaussian prior may be divided out anywhere. A **mixture** prior may not:
    between its modes the density is tiny, so ``prior^(1-n)`` is enormous there,
    and the composition becomes the difference of two large logarithms whose
    residue decides the answer. Measured on the Gaussian-mixture benchmark at
    n = 8, with component widths that are *correct* to 5%::

                            at g = 0      at g = +1.2
        sum log f_i           -152.2          -63.6
        -(n-1) log prior      +136.7          +41.8
        total                  -15.5          -21.7

    so the composed density peaked at ``g = 0``, exactly between the two prior
    modes, where the true posterior has no mass at all.

    The fix is structural, not numerical. Conditioned on a prior component the
    prior *is* a single Gaussian with no interior low-density region, and the
    identity

        p(theta | x_1..n, c)  ~  p(theta | c)^(1-n)  prod_i p(theta | x_i, c)

    is ordinary Gaussian algebra that never evaluates anything between the
    modes. So every factor component is assigned to the prior component it
    belongs to, tuples are enumerated **within** a component, and the branch
    weights come out of the same normalisation that weights the tuples. Tuples
    that mix components across observations are not merely improbable, they are
    not part of the factorisation: the component index is shared by construction,
    exactly like the shared coordinate itself.
    """
    prior_weights = np.asarray(prior_weights, dtype=np.float64)
    prior_means = np.asarray(prior_means, dtype=np.float64)
    prior_variances = np.asarray(prior_variances, dtype=np.float64)

    density = np.zeros_like(grid)
    tables, total_mass = [], []
    for index in range(prior_weights.size):
        # Which prior component does each factor component belong to? The
        # natural answer is the one that best explains its location, allowing
        # for both widths.
        restricted = []
        for mixture in mixtures:
            evidence = (np.log(np.maximum(prior_weights, FLOOR))[None, :]
                        - 0.5 * np.log(prior_variances[None, :]
                                       + mixture["variance"][:, None])
                        - 0.5 * (mixture["mean"][:, None]
                                 - prior_means[None, :]) ** 2
                        / (prior_variances[None, :]
                           + mixture["variance"][:, None]))
            keep = np.argmax(evidence, axis=1) == index
            if not keep.any():
                # This factor has no component of its own on this branch. It
                # still has mass there, carried by whichever component sits
                # closest, so lend that one rather than declaring the branch
                # impossible -- the branch weight is what should decide, and it
                # will come out small. Dropping the branch instead makes the
                # composition undefined whenever one observation disagrees,
                # which is exactly the K = 1 case.
                keep = np.zeros(len(mixture[key]), dtype=bool)
                keep[int(np.argmax(evidence[:, index]))] = True
            restricted.append({k: v[keep] if isinstance(v, np.ndarray) else v
                               for k, v in mixture.items()})
        table = enumerate_components(restricted, prior_variances[index], key,
                                     beam, prior_mean=prior_means[index])
        tables.append(table)
        total_mass.append(table["log_normaliser"])

    total_mass = np.asarray(total_mass)
    if not np.isfinite(total_mass).any():
        raise RuntimeError("no prior component admits a proper composition")
    share = np.exp(total_mass - total_mass[np.isfinite(total_mass)].max())
    share /= share.sum()

    for weight, table in zip(share, tables):
        if table is None or weight <= 0.0:
            continue
        proper = table["proper"]
        sd = table["precision"][proper] ** -0.5
        centre = table["mean"][proper][:, None]
        inner = table["share"][proper][:, None]
        kernel = np.exp(-0.5 * (grid[None, :] - centre) ** 2 / sd[:, None] ** 2)
        kernel /= np.sqrt(2.0 * np.pi) * sd[:, None]
        density += weight * (inner * kernel).sum(axis=0)

    density = density / np.trapezoid(density, grid)
    return density, {"component_weights": share.tolist(),
                     "components": sum(t["components"] for t in tables if t),
                     "improper": sum(t["improper"] for t in tables if t)}


def enumerate_components(mixtures, prior_variance, key="weight", beam=8192,
                         prior_mean=0.0):
    """The composed mixture in closed form, one component per surviving tuple.

    Folded one observation at a time rather than enumerated: ``prod_i K_i``
    reaches a million on the arm as soon as mode discovery is well converged, so
    the full product is not an option. Each component is carried as the
    unnormalised quadratic ``exp(logc - tau g^2 / 2 + s g)``, which makes both
    operations one line -- multiplying in a factor component adds its precision
    and shift, dividing by the prior subtracts the prior's -- and makes the
    component's mass available at every stage for pruning.

    The ``n`` factors are multiplied first and the ``n-1`` prior divisions are
    applied once at the end. That ordering matters: it keeps ``tau`` increasing
    monotonically through the fold, so a partial mass is a sound ranking, which
    it would not be if a division could make an intermediate improper and then a
    later factor make it proper again.

    Pruning to ``beam`` is a beam search and inherits the usual caveat -- a
    tuple that looks weak partway cannot recover later. ``effective_components``
    is reported so a run where the beam actually binds is visible.

    Precisions add, the prior precision is subtracted ``n-1`` times, and the log
    weight picks up the Gaussian overlap between the factors -- the "do these
    observations agree that the answer is on this branch" term the Gaussian
    rules never form, because with one component per factor it is the same for
    every tuple and cancels. A tuple whose final precision is not positive is
    **improper**: dividing by ``prior^(n-1)`` makes its product
    non-normalisable, and it is dropped.
    """
    count = len(mixtures)
    total = float(np.prod([len(m[key]) for m in mixtures]))
    index = np.zeros((1, 0), dtype=np.int64)
    precision = np.zeros(1)
    shift = np.zeros(1)
    logc = np.zeros(1)
    pruned = False

    for position, mixture in enumerate(mixtures):
        weight = np.maximum(mixture[key], FLOOR)
        variance = np.maximum(mixture["variance"], 1e-12)
        mean = mixture["mean"]
        components = len(weight)

        precision = (precision[:, None] + 1.0 / variance[None, :]).reshape(-1)
        shift = (shift[:, None] + (mean / variance)[None, :]).reshape(-1)
        logc = (logc[:, None] + (np.log(weight)
                                 - 0.5 * np.log(2.0 * np.pi * variance)
                                 - 0.5 * mean ** 2 / variance)[None, :]
                ).reshape(-1)
        index = np.concatenate([
            np.repeat(index, components, axis=0),
            np.tile(np.arange(components), index.shape[0])[:, None],
        ], axis=1)

        if position < count - 1 and precision.size > beam:
            mass = logc + 0.5 * shift ** 2 / precision \
                + 0.5 * np.log(2.0 * np.pi / precision)
            keep = np.argpartition(-mass, beam)[:beam]
            precision, shift, logc, index = (precision[keep], shift[keep],
                                             logc[keep], index[keep])
            pruned = True

    prior_precision = (count - 1) / prior_variance
    precision = precision - prior_precision
    shift = shift - prior_precision * prior_mean
    logc = (logc + 0.5 * (count - 1) * np.log(2.0 * np.pi * prior_variance)
            - 0.5 * prior_precision * prior_mean ** 2)
    proper = precision > 0.0
    if not proper.any():
        raise RuntimeError("every composed tuple is improper: no factor "
                           "component is narrower than the prior")

    log_weight = np.full(precision.shape, -np.inf)
    log_weight[proper] = (logc[proper]
                          + 0.5 * shift[proper] ** 2 / precision[proper]
                          + 0.5 * np.log(2.0 * np.pi / precision[proper]))
    finite = log_weight[proper]
    share = np.zeros(precision.shape)
    share[proper] = np.exp(finite - finite.max())
    share /= share.sum()
    return {
        "index": index, "share": share, "proper": proper,
        "mean": np.where(proper, shift / np.where(proper, precision, 1.0), np.nan),
        "precision": precision,
        "log_normaliser": float(finite.max() + np.log(
            np.exp(finite - finite.max()).sum())),
        "components": int(precision.size), "tuples": total,
        "pruned": bool(pruned), "improper": int((~proper).sum()),
        "effective_components": float(1.0 / np.sum(share ** 2)),
    }


def compose_gaussian_prior(mixtures, grid, prior_variance, key="weight"):
    """``prod_i q_i / prior^(n-1)`` for a Gaussian prior, in closed form."""
    table = enumerate_components(mixtures, prior_variance, key)
    proper = table["proper"]
    sd = table["precision"][proper] ** -0.5
    centre = table["mean"][proper][:, None]
    weight = table["share"][proper][:, None]
    kernel = np.exp(-0.5 * (grid[None, :] - centre) ** 2 / sd[:, None] ** 2)
    kernel /= np.sqrt(2.0 * np.pi) * sd[:, None]
    density = (weight * kernel).sum(axis=0)
    return density / np.trapezoid(density, grid), table


def compose_grid_prior(mixtures, grid, prior, key="weight"):
    """``prod_i q_i / prior^(n-1)`` for an arbitrary prior tabulated on ``grid``.

    Needed when the prior is not Gaussian -- the Gaussian-mixture benchmark's
    prior on ``g`` is itself two-component -- so there is no closed form to
    divide by. The tail condition of :func:`enumerate_components` still has to
    hold or the result is a spike at the truncation, so it is checked rather
    than assumed: ``tail_ratio`` is the composed density at the grid edge
    relative to its maximum, and anything not tiny means the readout is invalid.
    """
    densities = np.asarray([mixture_density(m, grid, key) for m in mixtures])
    count = densities.shape[0]
    log_density = (np.log(np.maximum(densities, FLOOR)).sum(axis=0)
                   - (count - 1) * np.log(np.maximum(prior, FLOOR)))
    log_density -= log_density.max()
    density = np.exp(log_density)
    density = density / np.trapezoid(density, grid)
    tail = float(max(density[0], density[-1]) / max(density.max(), 1e-300))
    return density, {"tail_ratio": tail, "components": None}


def refit_components(mixture, draws, iterations=200, floor=1e-4):
    """Refit the K components to draws by EM, K held fixed.

    The attribution control, not the method: mode discovery supplies K and the
    initialisation, EM then gives the best K-component mixture there is for this
    factor. If composing *that* still fails, the failure belongs to the Gaussian
    family and the ``prior^(1-n)`` division rather than to the Laplace curvature
    or to the path integral.
    """
    draws = np.asarray(draws, dtype=np.float64)
    mean = mixture["mean"].copy()
    variance = np.maximum(mixture["variance"].copy(), floor)
    weight = np.full(len(mean), 1.0 / len(mean))
    for _ in range(iterations):
        log_p = (-0.5 * (draws[None, :] - mean[:, None]) ** 2 / variance[:, None]
                 - 0.5 * np.log(2.0 * np.pi * variance[:, None])
                 + np.log(np.maximum(weight, FLOOR))[:, None])
        log_p -= log_p.max(axis=0, keepdims=True)
        responsibility = np.exp(log_p)
        responsibility /= responsibility.sum(axis=0, keepdims=True)
        total = responsibility.sum(axis=1)
        weight = total / total.sum()
        mean = (responsibility * draws[None, :]).sum(1) / np.maximum(total, 1e-12)
        variance = np.maximum(
            (responsibility * (draws[None, :] - mean[:, None]) ** 2).sum(1)
            / np.maximum(total, 1e-12), floor)
    return {**mixture, "mean": mean, "variance": variance, "weight": weight}


def empirical_weights(mixture, draws):
    """Branch weights implied by draws: a referee for the path integral."""
    mean = mixture["mean"][:, None]
    variance = np.maximum(mixture["variance"][:, None], 1e-12)
    logs = (-0.5 * (np.asarray(draws)[None, :] - mean) ** 2 / variance
            - 0.5 * np.log(variance))
    counts = np.bincount(np.argmax(logs, axis=0),
                         minlength=mean.shape[0]).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def positive_mass(mixture, key="weight", threshold=0.0):
    """Mixture mass above ``threshold``: the branch split, free of the widths."""
    from scipy.special import ndtr
    sd = np.sqrt(np.maximum(mixture["variance"], 1e-12))
    return float((mixture[key] * ndtr((mixture["mean"] - threshold) / sd)).sum())
