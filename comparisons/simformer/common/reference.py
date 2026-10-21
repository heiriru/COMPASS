"""Reference samples for arbitrary conditionals, by the paper's own recipes.

Appendix A2.2 specifies, per task:

* Two Moons -- N chains initialised from the joint, 1000 steps of random-direction
  slice sampling, then 3000 Metropolis-Hastings steps at step size 0.01; keep the
  last state of each chain.
* SLCP -- the same with 600 slice steps and 2000 MH steps at step size 0.1.
* Tree, HMM -- 5000 steps of HMC; keep the last state of each chain.

One state per chain is kept, so the reference draws are independent by
construction and ``n_chains`` is the reference sample size. This is expensive
and deliberate: it is what lets them score C2ST against a conditional they have
no closed form for.

Validation before use
---------------------
The paper validates the recipe by checking C2ST ~ 0.5 against the sbibm
reference posterior. Do the same here before any arm is scored against these
samples: ``tests/test_reference.py`` is the place, and an unvalidated reference
set is a silent way to publish a wrong C2ST.

All samplers here target ``task.joint_log_prob`` with the conditioned
coordinates pinned, so they work for *any* condition mask -- posterior,
likelihood, or an arbitrary mixed one.
"""

import torch


def conditional_log_prob(task, mask, conditioned_values):
    """Return ``f(latent) -> log p(z)`` with the masked coordinates pinned.

    Args:
        task: a ``common.tasks.Task``.
        mask: ``(nodes_size,)`` with 1 = conditioned, 0 = latent -- COMPASS's
            convention, identical to the paper's ``M_C``.
        conditioned_values: ``(nodes_size,)``; only the entries where
            ``mask == 1`` are read.
    """
    mask = torch.as_tensor(mask, dtype=torch.float)
    latent_index = (mask == 0).nonzero(as_tuple=True)[0]
    base = torch.as_tensor(conditioned_values, dtype=torch.float).clone()

    def log_prob(latent):
        single = latent.dim() == 1
        latent = latent.unsqueeze(0) if single else latent
        z = base.unsqueeze(0).expand(latent.shape[0], -1).clone()
        z[:, latent_index] = latent
        out = task.joint_log_prob(z)
        return out.squeeze(0) if single else out

    return log_prob, latent_index


def _slice_sample_random_direction(log_prob, state, steps, width=1.0,
                                   max_steps_out=50, generator=None):
    """Random-direction slice sampling (Neal 2003) on a batch of chains."""
    n, dim = state.shape
    current = log_prob(state)
    for _ in range(steps):
        direction = torch.randn(n, dim, generator=generator)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        level = current + torch.log(torch.rand(n, generator=generator).clamp_min(1e-30))

        u = torch.rand(n, generator=generator)
        lower = -width * u
        upper = lower + width
        for _ in range(max_steps_out):
            expand_low = log_prob(state + lower.unsqueeze(-1) * direction) > level
            if not expand_low.any():
                break
            lower = torch.where(expand_low, lower - width, lower)
        for _ in range(max_steps_out):
            expand_high = log_prob(state + upper.unsqueeze(-1) * direction) > level
            if not expand_high.any():
                break
            upper = torch.where(expand_high, upper + width, upper)

        for _ in range(max_steps_out):
            step = lower + (upper - lower) * torch.rand(n, generator=generator)
            candidate = state + step.unsqueeze(-1) * direction
            candidate_lp = log_prob(candidate)
            accepted = candidate_lp > level
            if accepted.all():
                state, current = candidate, candidate_lp
                break
            state = torch.where(accepted.unsqueeze(-1), candidate, state)
            current = torch.where(accepted, candidate_lp, current)
            shrink_low = (step < 0) & ~accepted
            shrink_high = (step >= 0) & ~accepted
            lower = torch.where(shrink_low, step, lower)
            upper = torch.where(shrink_high, step, upper)
    return state


def _metropolis_hastings(log_prob, state, steps, step_size, generator=None):
    """Isotropic random-walk MH on a batch of chains."""
    n, dim = state.shape
    current = log_prob(state)
    accepted_total = 0
    for _ in range(steps):
        proposal = state + step_size * torch.randn(n, dim, generator=generator)
        proposal_lp = log_prob(proposal)
        log_u = torch.log(torch.rand(n, generator=generator).clamp_min(1e-30))
        accept = log_u < (proposal_lp - current)
        state = torch.where(accept.unsqueeze(-1), proposal, state)
        current = torch.where(accept, proposal_lp, current)
        accepted_total += int(accept.sum())
    return state, accepted_total / (steps * n)


def _hmc(log_prob, state, steps, step_size=0.1, leapfrog=20, generator=None):
    """Vanilla HMC with a unit mass matrix, for the differentiable tasks."""
    n, dim = state.shape

    def grad_log_prob(z):
        z = z.detach().requires_grad_(True)
        value = log_prob(z).sum()
        grad, = torch.autograd.grad(value, z)
        return grad

    current_lp = log_prob(state)
    accepted_total = 0
    for _ in range(steps):
        momentum = torch.randn(n, dim, generator=generator)
        new_state = state.clone()
        new_momentum = momentum + 0.5 * step_size * grad_log_prob(new_state)
        for step in range(leapfrog):
            new_state = new_state + step_size * new_momentum
            factor = 1.0 if step < leapfrog - 1 else 0.5
            new_momentum = new_momentum + factor * step_size * grad_log_prob(new_state)
        new_lp = log_prob(new_state)
        delta = (new_lp - 0.5 * (new_momentum ** 2).sum(-1)) - \
                (current_lp - 0.5 * (momentum ** 2).sum(-1))
        log_u = torch.log(torch.rand(n, generator=generator).clamp_min(1e-30))
        accept = log_u < delta
        state = torch.where(accept.unsqueeze(-1), new_state.detach(), state)
        current_lp = torch.where(accept, new_lp.detach(), current_lp)
        accepted_total += int(accept.sum())
    return state, accepted_total / (steps * n)


def reference_conditional_samples(task, mask, conditioned_values, n_samples,
                                  recipe=None, seed=0):
    """Reference draws from ``p(latent | conditioned)`` by the paper's recipe.

    Args:
        recipe: overrides ``common.paper.REFERENCE_MCMC[task.name]``. Keys are
            ``slice_steps``/``mh_steps``/``mh_step_size`` or ``hmc_steps``
            (plus optional ``hmc_step_size``/``hmc_leapfrog``).

    Returns ``(samples, diagnostics)``; ``samples`` is ``(n_samples, n_latent)``
    in the order the latent coordinates appear in the flat node vector.
    """
    from . import paper

    if recipe is None:
        recipe = paper.REFERENCE_MCMC.get(task.name)
    if recipe is None:
        raise KeyError(
            f"No reference recipe for task {task.name!r}. The paper specifies "
            f"one for {sorted(paper.REFERENCE_MCMC)}; add it explicitly."
        )

    generator = torch.Generator().manual_seed(seed)
    log_prob, latent_index = conditional_log_prob(task, mask, conditioned_values)

    # "We initialized N Markov chains with samples from the joint distribution."
    theta, x = task.sample_joint(n_samples, generator=generator)
    state = torch.cat([theta, x], dim=-1)[:, latent_index].contiguous()

    diagnostics = {"recipe": dict(recipe), "n_chains": int(n_samples)}
    if "hmc_steps" in recipe:
        state, rate = _hmc(
            log_prob, state, recipe["hmc_steps"],
            step_size=recipe.get("hmc_step_size", 0.1),
            leapfrog=recipe.get("hmc_leapfrog", 20), generator=generator,
        )
        diagnostics["hmc_accept_rate"] = rate
    else:
        state = _slice_sample_random_direction(
            log_prob, state, recipe["slice_steps"], generator=generator)
        state, rate = _metropolis_hastings(
            log_prob, state, recipe["mh_steps"], recipe["mh_step_size"],
            generator=generator)
        diagnostics["mh_accept_rate"] = rate
    return state, diagnostics
