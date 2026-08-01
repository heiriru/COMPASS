"""Euler--Maruyama DDM adapted from diffusion-experiments case_study4.

The scientific equations, parameter transformations, and defaults follow upstream;
randomness, censoring, decision-time accounting, and batching are benchmark additions.
"""

import numpy as np

from ..schema import physical_from_joint


def simulate_ddm_trial(nu, alpha, t0, beta, rng, dt=0.001, max_decision_time=10.0):
    """Direct single-trial reference returning (choice, RT, censored)."""
    position = float(beta) * float(alpha)
    decision_time = 0.0
    while 0.0 < position < alpha and decision_time < max_decision_time:
        position += nu * dt + np.sqrt(dt) * rng.normal()
        decision_time += dt
    censored = decision_time >= max_decision_time and 0.0 < position < alpha
    choice = float(position >= alpha) if not censored else float(position >= alpha / 2.0)
    return choice, float(t0 + min(decision_time, max_decision_time)), float(censored)


def simulate_ddm(parameters, rng, trials=30, dt=0.001, max_decision_time=10.0):
    """Vectorized subjects/trials simulation with output (..., trials, 3)."""
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape[-1] != 4:
        raise ValueError("DDM parameters must end in (nu, alpha, t0, beta).")
    leading = parameters.shape[:-1]
    count = int(np.prod(leading))
    flat = parameters.reshape(count, 4)
    nu, alpha, t0, beta = (flat[:, index, None] for index in range(4))
    positions = np.broadcast_to(beta * alpha, (count, trials)).copy()
    decision_times = np.zeros((count, trials), dtype=np.float64)
    active = (positions > 0.0) & (positions < alpha)
    max_steps = int(np.ceil(max_decision_time / dt))
    for _ in range(max_steps):
        if not active.any():
            break
        active_indices = np.flatnonzero(active)
        positions.flat[active_indices] += (
            np.broadcast_to(nu, positions.shape).flat[active_indices] * dt
            + np.sqrt(dt) * rng.normal(size=active_indices.size)
        )
        decision_times[active] += dt
        active = (positions > 0.0) & (positions < np.broadcast_to(alpha, positions.shape)) & (decision_times < max_decision_time)
    censored = (
        (decision_times >= max_decision_time)
        & (positions > 0.0)
        & (positions < np.broadcast_to(alpha, positions.shape))
    )
    choices = np.where(censored, positions >= alpha / 2.0, positions >= alpha).astype(float)
    reaction_times = decision_times + np.broadcast_to(t0, positions.shape)
    result = np.stack((choices, reaction_times, censored.astype(float)), axis=-1)
    return result.reshape(*leading, trials, 3)


def simulate_hierarchical(globals_, locals_, rng, trials=30, **kwargs):
    return simulate_ddm(physical_from_joint(globals_[..., None, :], locals_), rng, trials, **kwargs)
