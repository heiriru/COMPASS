"""Mode-aware Jacobian composition: JAC within each posterior mode, masses from the target.

Every Gaussian composition rule (GAUSS, GH, JAC) carries one Gaussian per row and
cannot represent how posterior mass splits between separated modes (the counting
argument; measured on GM as branch-mass errors of ~0.6 for JAC at n = 16). The *shape*
inside a mode is what the rules do well. This wrapper separates the two jobs:

1. **Discovery.** One short composed run (optionally tempered, ``composition_temperature
   < 1``, so a minor mode is not starved), clustered on the shared coordinates by a
   Gaussian mixture with K chosen by BIC (``K <= max_modes``).
2. **Within-mode sampling.** For each mode, a warm-started run of the same rule
   (``start_lambda`` / ``init_state``) from that cluster's draws re-noised to
   ``start_lambda``, chosen small against the mode separation so the reverse
   diffusion stays in its basin.
3. **Masses.** ``certify_composition`` evaluates the exact tall-data target
   ``p(g)^(1-n) prod_j p(g, l_j | x_j)`` at every within-mode draw with the
   probability-flow ODE. With a block-factorized Gaussian proposal ``q_k`` fitted to
   mode k's draws, ``Z_k ~ mean(pi / q_k)`` (importance sampling), and the modes are
   mixed with weights ``Z_k / sum Z``.

The result is a mixture over modes with JAC shapes and target-derived masses. Costs:
one discovery run, K warm-started (shorter) runs, and ``K x n x M`` PF-ODE density
evaluations. Nothing in ``MultiObsSampler`` is changed. Remove by deleting this module
and its test.
"""
import math

import torch


def fit_gmm(points, max_components=3, iterations=200, seed=0, min_weight=0.02):
    """Full-covariance Gaussian mixture by EM, K in 1..max_components chosen by BIC.

    Returns ``(labels (N,), means (K, d), covariances (K, d, d), weights (K,))``."""
    x = torch.as_tensor(points, dtype=torch.float64)
    N, d = x.shape
    generator = torch.Generator().manual_seed(seed)
    best = None
    for K in range(1, max_components + 1):
        if N < 5 * K * (d + 1):
            break
        # k-means++ style init
        centres = [x[torch.randint(N, (1,), generator=generator)].squeeze(0)]
        for _ in range(1, K):
            dist = torch.stack([((x - c) ** 2).sum(-1) for c in centres]).min(0).values
            centres.append(x[torch.multinomial(dist / dist.sum(), 1, generator=generator)].squeeze(0))
        means = torch.stack(centres)
        cov = torch.cov(x.T).reshape(d, d)
        covs = (cov + 1e-6 * torch.eye(d, dtype=torch.float64)).expand(K, d, d).clone()
        weights = torch.full((K,), 1.0 / K, dtype=torch.float64)
        previous = -math.inf
        for _ in range(iterations):
            dist = torch.distributions.MultivariateNormal(means, covariance_matrix=covs)
            log_r = dist.log_prob(x.unsqueeze(1)) + weights.log()
            log_likelihood = torch.logsumexp(log_r, 1)
            r = torch.exp(log_r - log_likelihood.unsqueeze(1))
            total = float(log_likelihood.sum())
            Nk = r.sum(0).clamp_min(1e-12)
            weights = Nk / N
            means = (r.T @ x) / Nk.unsqueeze(1)
            diff = x.unsqueeze(1) - means
            covs = torch.einsum("nk,nki,nkj->kij", r, diff, diff) / Nk.reshape(-1, 1, 1)
            covs = covs + 1e-6 * torch.eye(d, dtype=torch.float64)
            if abs(total - previous) < 1e-8 * abs(total):
                break
            previous = total
        params = K * (d + d * (d + 1) / 2) + (K - 1)
        bic = -2 * total + params * math.log(N)
        if K > 1 and float(weights.min()) < min_weight:
            continue
        if best is None or bic < best[0]:
            best = (bic, means, covs, weights, r.argmax(1))
    _, means, covs, weights, labels = best
    return labels, means, covs, weights


def merge_close_components(labels, means, covs, weights, merge_sd=3.0):
    """Merge mixture components whose centres are within ``merge_sd`` standard deviations
    (largest of the pair, along the line between them): a spuriously split mode would
    otherwise fix ``start_lambda`` at a fraction of a width and disable the warm start."""
    labels, means, covs, weights = labels.clone(), means.clone(), covs.clone(), weights.clone()
    while means.shape[0] > 1:
        best = None
        for a in range(means.shape[0]):
            for b in range(a + 1, means.shape[0]):
                delta = means[a] - means[b]
                direction = delta / delta.norm().clamp_min(1e-12)
                sd = max(float(direction @ covs[a] @ direction), float(direction @ covs[b] @ direction)) ** 0.5
                ratio = float(delta.norm()) / max(sd, 1e-12)
                if ratio < merge_sd and (best is None or ratio < best[0]):
                    best = (ratio, a, b)
        if best is None:
            break
        _, a, b = best
        wa, wb = float(weights[a]), float(weights[b])
        mean = (wa * means[a] + wb * means[b]) / (wa + wb)
        cov = (wa * (covs[a] + torch.outer(means[a] - mean, means[a] - mean))
               + wb * (covs[b] + torch.outer(means[b] - mean, means[b] - mean))) / (wa + wb)
        keep = [k for k in range(means.shape[0]) if k != b]
        means[a], covs[a], weights[a] = mean, cov, wa + wb
        labels[labels == b] = a
        labels = torch.searchsorted(torch.tensor(keep), labels)
        means, covs, weights = means[keep], covs[keep], weights[keep]
    return labels, means, covs, weights


def mode_aware_sample(sampler, data, condition_mask, hierarchy, num_samples,
                      discovery_samples=1000, max_modes=3, discovery_temperature=1.0,
                      start_fraction=1.0 / 6.0, draws_per_mode=None, prior=None,
                      certify_kwargs=None, log_prior_fn=None, mode_prior_fn=None,
                      discovery_context=None, merge_sd=3.0, generator_seed=0, verbose=False,
                      **sample_kwargs):
    """Mode-aware composition (see module docstring).

    Args:
        sampler: a :class:`compass.MultiObsSampler`.
        data, condition_mask, hierarchy, prior, **sample_kwargs: as for
            ``sampler.sample`` (correction, timesteps, eps, device, ...).
        num_samples: returned draws.
        discovery_samples: draws in the discovery run.
        discovery_temperature: ``composition_temperature`` of the discovery run.
        start_fraction: ``start_lambda = start_fraction x`` (smallest distance between
            mode centres, in prior-standardised shared units x prior std).
        draws_per_mode: draws per within-mode run (default ``num_samples``).
        mode_prior_fn: ``f(centre (d,)) -> prior`` -- compose *within a prior component*:
            the within-mode run of a mode centred at ``centre`` uses this (Gaussian) prior
            instead of ``prior``. For a mixture prior, return the component the mode sits
            in: inside its basin the mixture equals that component up to a constant, and
            the component's Gaussian precision is exact, unlike the moment-matched one.
        merge_sd: clusters closer than this many SDs are one mode (see
            :func:`merge_close_components`).
        discovery_context: zero-argument factory of a context manager wrapped around the
            discovery run only (e.g. an exact mixture prior score patch).
        log_prior_fn: ``f(shared (M, d)) -> (M,)`` log prior density of the shared block,
            replacing ``certify_composition``'s Gaussian prior term (needed when the
            prior is not Gaussian, e.g. a mixture).

    Returns:
        ``(samples (observations, num_samples, nodes), info)``; ``info`` holds the
        modes' centres, discovery fractions, masses, log evidences and ESS.
    """
    device = sample_kwargs.get("device", "cpu")
    discovery_kwargs = dict(sample_kwargs)
    if discovery_temperature != 1.0:
        discovery_kwargs["composition_temperature"] = discovery_temperature
    from contextlib import nullcontext
    with (discovery_context or nullcontext)():
        discovery = sampler.sample(world_size=1, data=data, condition_mask=condition_mask,
                                   hierarchy=hierarchy, prior=prior,
                                   num_samples=discovery_samples, verbose=verbose,
                                   **discovery_kwargs).detach().cpu()
    shared = discovery[0][:, hierarchy].to(torch.float64)
    labels, means, covs, weights = merge_close_components(
        *fit_gmm(shared, max_modes, seed=generator_seed), merge_sd=merge_sd)
    K = means.shape[0]
    info = {"modes": K, "centres": means, "discovery_fraction": weights}
    if K == 1:
        single_prior = prior if mode_prior_fn is None else mode_prior_fn(means[0])
        out = sampler.sample(world_size=1, data=data, condition_mask=condition_mask,
                             hierarchy=hierarchy, prior=single_prior, num_samples=num_samples,
                             verbose=verbose, **sample_kwargs)
        info["masses"] = torch.ones(1, dtype=torch.float64)
        return out, info

    prior_std = torch.as_tensor(sampler.prior_std, dtype=torch.float64).reshape(-1).cpu()
    standardised = means / prior_std
    separation = min(float((standardised[a] - standardised[b]).norm())
                     for a in range(K) for b in range(a + 1, K))
    start_lambda = start_fraction * separation * float(prior_std.mean())
    info["start_lambda"] = start_lambda
    M = int(draws_per_mode or num_samples)
    generator = torch.Generator().manual_seed(generator_seed)
    per_mode, log_evidence, ess = [], [], []
    mask_vector = torch.as_tensor(condition_mask, dtype=torch.float32).reshape(-1)
    latent = (1 - mask_vector).bool()
    for k in range(K):
        members = torch.where(labels == k)[0]
        pick = members[torch.randint(len(members), (M,), generator=generator)]
        init = discovery[:, pick].clone()
        noise = torch.randn(init.shape, generator=generator)
        noise[:, :, hierarchy] = noise[:1, :, hierarchy]          # shared noise is shared
        init[:, :, latent] = init[:, :, latent] + start_lambda * noise[:, :, latent]
        mode_prior = prior if mode_prior_fn is None else mode_prior_fn(means[k])
        draws = sampler.sample(world_size=1, data=data, condition_mask=condition_mask,
                               hierarchy=hierarchy, prior=mode_prior, num_samples=M,
                               start_lambda=start_lambda, init_state=init,
                               verbose=verbose, **sample_kwargs).detach().cpu()
        cert = sampler.certify_composition(draws, data, condition_mask, hierarchy=hierarchy,
                                           prior=prior, proposal="factorized", device=device,
                                           **(certify_kwargs or {}))
        log_weights = cert["log_weights"]
        if log_prior_fn is not None:
            n_obs = cert["num_observations"]
            shared_draws = draws[0][:, hierarchy].to(torch.float64)
            log_prior = (1 - n_obs) * torch.as_tensor(log_prior_fn(shared_draws), dtype=torch.float64)
            log_weights = log_weights - cert["log_prior_term"] + log_prior
            weights = torch.softmax(log_weights, 0)
            cert["ess_fraction"] = float(1.0 / weights.square().sum()) / M
        per_mode.append(draws)
        log_evidence.append(float(torch.logsumexp(log_weights, 0) - math.log(M)))
        ess.append(cert["ess_fraction"])
    log_evidence = torch.tensor(log_evidence, dtype=torch.float64)
    masses = torch.softmax(log_evidence, 0)
    counts = torch.distributions.Multinomial(num_samples, masses).sample()
    chosen = []
    for k in range(K):
        c = int(counts[k])
        if c:
            idx = torch.randint(M, (c,), generator=generator)
            chosen.append(per_mode[k][:, idx])
    out = torch.cat(chosen, 1)
    out = out[:, torch.randperm(out.shape[1], generator=generator)]
    info.update(masses=masses, log_evidence=log_evidence, ess_fraction=ess)
    return out, info
