"""A2 critical-thinking check: multi-start on a likelihood with two unequal maxima.

Likelihood (analytic, differentiable; stands in for the PF-ODE log_prob):
    x1 = theta^2 + nu1 e1,   x2 = a theta + nu2 e2
For x1 > 0 the likelihood has maxima near theta = +-sqrt(x1); x2 tilts them, so the
global maximum is on the side sign(x2) favours. The exact global maximum is
found on a dense grid.

Arms (R rows each):
    single start, wrong basin      the start sits at -sign(theta_global) sqrt(x1)
    single start, at the MAP       a prior N(0, 1) MAP-like point near 0
    4 starts from posterior draws  drawn from the exact (grid) posterior
Prediction: a single start in the wrong basin stays there (local optimizer);
posterior-draw multi-start recovers the global maximum unless one branch
has posterior mass so small that no draw lands in it; screening 32 draws and
optimizing the best 3 (compare()'s default) closes most of that gap.

    .COMPASS/bin/python -m validation.a2_maximum_likelihood.nonconcave
"""

import validation.common.runtime  # noqa: F401

import json
import math
import os

import torch

from compass.likelihood_maximization import maximize_likelihood

OUT = os.path.join(os.path.dirname(__file__), "results")
NU1, NU2, A = 0.3, 1.0, 0.3
R = 400


class BimodalLikelihood:
    """``log p(x | theta)`` over joint vectors ``[theta, x1, x2]``."""

    def log_prob(self, data, condition_mask, **kwargs):
        theta, x1, x2 = data[:, 0], data[:, 1], data[:, 2]
        return (-0.5 * ((x1 - theta ** 2) / NU1) ** 2 - 0.5 * ((x2 - A * theta) / NU2) ** 2
                - math.log(2 * math.pi * NU1 * NU2))


def main():
    os.makedirs(OUT, exist_ok=True)
    g = torch.Generator().manual_seed(20261013)
    theta_true = 1.0 + 0.5 * torch.rand(R, generator=g)          # well-separated branches
    theta_true = theta_true * torch.where(torch.rand(R, generator=g) < 0.5, -1.0, 1.0)
    x1 = theta_true ** 2 + NU1 * torch.randn(R, generator=g)
    x2 = A * theta_true + NU2 * torch.randn(R, generator=g)
    model = BimodalLikelihood()
    mask = torch.tensor([1.0, 0.0, 0.0])

    grid = torch.linspace(-4, 4, 80001)
    ll_grid = model.log_prob(torch.stack([grid.expand(R, -1).reshape(-1),
                                          x1.repeat_interleave(len(grid)),
                                          x2.repeat_interleave(len(grid))], 1), mask
                             ).reshape(R, -1)
    best = ll_grid.argmax(1)
    theta_global, ll_global = grid[best], ll_grid.max(1).values
    # Exact posterior under a N(0, 1) prior, for posterior-draw starts.
    log_post = ll_grid - 0.5 * grid ** 2
    post = torch.softmax(log_post, 1)
    mass_global_side = (post * (torch.sign(grid) == torch.sign(theta_global).unsqueeze(1))).sum(1)

    def arm(starts_theta, keep=None):
        data = torch.stack([starts_theta[0], x1, x2], 1)
        starts = None
        if len(starts_theta) > 1:
            starts = torch.stack([torch.stack([s, x1, x2], 1) for s in starts_theta[1:]])
        res = maximize_likelihood(model, data, mask, starts=starts, max_iterations=100,
                                  keep_best_starts=keep)
        gap = (ll_global - res.log_likelihood.double()).clamp_min(0)
        return {"found_global": float((gap < 1e-3).float().mean()),
                "mean_gap_nats": float(gap.mean()), "max_gap_nats": float(gap.max())}

    wrong = -torch.sign(theta_global) * x1.clamp_min(0).sqrt()
    near_zero = 0.05 * torch.sign(x2)        # a MAP-like start, tilted the way x2 points
    draws = torch.stack([grid[torch.multinomial(post, 1, generator=g).squeeze(1)]
                         for _ in range(4)])
    draws32 = torch.stack([grid[torch.multinomial(post, 1, generator=g).squeeze(1)]
                           for _ in range(32)])
    results = {
        "single_start_wrong_basin": arm([wrong]),
        "single_start_near_map": arm([near_zero]),
        "posterior_draws_x4": arm(list(draws)),
        "posterior_draws_x4_plus_wrong": arm([wrong] + list(draws)),
        # compare()'s default: MAP-like start + best 3 of 32 screened draws
        "wrong_plus_screen32_keep3": arm([wrong] + list(draws32), keep=3),
        "posterior_mass_on_global_side": {
            "median": float(mass_global_side.median()),
            "fraction_rows_below_0.05": float((mass_global_side < 0.05).float().mean())},
    }
    with open(os.path.join(OUT, "nonconcave.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
