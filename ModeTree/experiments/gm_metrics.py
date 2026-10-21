"""Posterior metrics for the oracle bridge experiment.

The target is bimodal, so no summary built from a mean and a covariance can
grade it.  Everything below is computed on the shared coordinate ``g`` of the
returned samples against the *exact* two-component posterior, and the whole set
is reported for every arm -- the decision rule in the report is not allowed to
rest on ``W1`` alone.

Branch assignment uses the midpoint between the two exact branch means, which
for this model is the exact posterior's own density minimum whenever the
branches are symmetric and is within a small fraction of a within-branch sd of
it otherwise.
"""

from __future__ import annotations

import math

import torch

from gm_oracle import interp


def _empirical_quantiles(samples, probabilities):
    ordered, _ = torch.sort(samples)
    count = ordered.numel()
    levels = (torch.arange(count, dtype=torch.float64) + 0.5) / count
    return interp(probabilities, levels, ordered)


def wasserstein(samples, tall, order=1, points=4001):
    """``W_order`` between the samples and the exact posterior, in units of sd."""
    probabilities = (torch.arange(points, dtype=torch.float64) + 0.5) / points
    empirical = _empirical_quantiles(samples, probabilities)
    exact = tall.global_quantiles(probabilities)
    _, sd = tall.global_moments()
    gap = (empirical - exact).abs()
    if order == 1:
        return float(gap.mean()) / sd
    return float((gap ** order).mean() ** (1.0 / order)) / sd


def kde(samples, points, bandwidth):
    z = (points.unsqueeze(-1) - samples) / bandwidth
    return (torch.exp(-0.5 * z ** 2).mean(-1)
            / (bandwidth * math.sqrt(2.0 * math.pi)))


def summarise(samples, tall):
    """Every metric the comparison table needs, for one arm at one ``n``."""
    samples = torch.as_tensor(samples, dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(samples)
    escaped = float((~finite).double().mean())
    samples = samples[finite]
    mean_exact, sd_exact = tall.global_moments()
    means = tall.global_means
    order = torch.argsort(means)
    means = means[order]
    weights = tall.weights[order]
    within = math.sqrt(tall.global_variance)

    # Exact modality, from the closed-form density rather than from a kernel
    # estimate.  The branch *means* are not the density's modes once the two
    # branches come within a few within-branch sds of each other, and at
    # n = 128 in this benchmark they are 2.9 sds apart -- so the split point,
    # the dip and the mode locations are all read off the exact density.
    wide = torch.linspace(float(means[0]) - 6.0 * within,
                          float(means[1]) + 6.0 * within, 200_001,
                          dtype=torch.float64)
    exact_density = torch.exp(tall.global_log_prob(wide))
    interior = exact_density[1:-1]
    peaks = torch.nonzero((interior > exact_density[:-2])
                          & (interior > exact_density[2:])).reshape(-1) + 1
    exact_modes = int(peaks.numel())
    if exact_modes >= 2:
        first, last = int(peaks[0]), int(peaks[-1])
        valley_index = int(exact_density[first:last].argmin()) + first
        midpoint = float(wide[valley_index])
        exact_dip = float(exact_density[valley_index]
                          / exact_density[peaks].min())
        mode_locations = wide[peaks[[0, -1]]]
    else:
        midpoint = float(means.mean())
        exact_dip = 1.0
        mode_locations = means.clone()

    if samples.numel() == 0:
        # A diverged arm still has to fill every column of the table; returning
        # a short dict instead made the driver raise KeyError and lose the
        # whole stage's results.
        nan = float("nan")
        return {
            "escaped_fraction": escaped, "w1_over_sd": nan, "w2_over_sd": nan,
            "mode_recall": 0.0, "mode_location_error_sd": [nan, nan],
            "weight_tv": nan, "positive_branch_mass": nan,
            "exact_positive_branch_mass": nan, "branch_weights": weights.tolist(),
            "valley_mass": nan, "exact_valley_mass": nan, "dip_ratio": nan,
            "exact_dip_ratio": exact_dip, "exact_num_modes": exact_modes,
            "exact_mode_locations": mode_locations.tolist(),
            "split_point": midpoint, "mean_error_sd": nan, "width_ratio": nan,
            "diverged": True,
        }

    # Branch mass is compared at the *density valley*, not at the midpoint of
    # the branch means: those coincide only while the branches are far apart,
    # and the exact mass on each side of the valley is what a sample can
    # actually be graded against.
    assignment = (samples > midpoint).long()
    empirical_weights = torch.stack([
        (assignment == index).double().mean() for index in range(2)
    ])
    split_cdf = float(tall.global_cdf(
        torch.tensor([midpoint], dtype=torch.float64))[0])
    exact_side = torch.tensor([split_cdf, 1.0 - split_cdf], dtype=torch.float64)
    location_error = []
    for index in range(2):
        member = samples[assignment == index]
        location_error.append(
            float("nan") if member.numel() == 0
            else abs(float(member.mean()) - float(means[index])) / within
        )

    # Valley mass on a window a quarter of the mode separation wide, and the
    # sample dip estimated with a kernel a sixth of that separation -- narrow
    # enough to resolve the structure the exact density actually has.
    separation = float(mode_locations[1] - mode_locations[0])
    half = 0.25 * separation
    low, high = midpoint - half, midpoint + half
    valley_mass = float(((samples > low) & (samples < high)).double().mean())
    cdf = tall.global_cdf(torch.tensor([low, high], dtype=torch.float64))
    exact_valley = float(cdf[1] - cdf[0])

    bandwidth = separation / 6.0
    probes = torch.cat([mode_locations,
                        torch.tensor([midpoint], dtype=torch.float64)])
    density = kde(samples, probes, bandwidth)
    dip = (float(density[2] / density[:2].min())
           if float(density[:2].min()) > 0 else float("inf"))

    recalled = int(((empirical_weights > 1e-3) & (exact_side > 1e-3)).sum())
    expected = int((exact_side > 1e-3).sum())

    w1 = wasserstein(samples, tall, 1)
    return {
        "escaped_fraction": escaped,
        "diverged": bool(escaped > 0.0 or not math.isfinite(w1) or w1 > 1e3),
        "w1_over_sd": w1,
        "w2_over_sd": wasserstein(samples, tall, 2),
        "mode_recall": recalled / expected,
        "mode_location_error_sd": location_error,
        "weight_tv": float(0.5 * (empirical_weights - exact_side).abs().sum()),
        "positive_branch_mass": float(empirical_weights[1]),
        "exact_positive_branch_mass": float(exact_side[1]),
        "branch_weights": weights.tolist(),
        "valley_mass": valley_mass,
        "exact_valley_mass": exact_valley,
        "dip_ratio": dip,
        "exact_dip_ratio": exact_dip,
        "exact_num_modes": exact_modes,
        "exact_mode_locations": mode_locations.tolist(),
        "split_point": midpoint,
        "mode_separation_in_within_sd": separation / within,
        "mean_error_sd": abs(float(samples.mean()) - mean_exact) / sd_exact,
        "width_ratio": float(samples.std(unbiased=True)) / sd_exact,
    }


def monte_carlo_floor(tall, count, repeats=8, seed=0):
    """What ``count`` *exact* draws score -- the numerical floor for W1/sd."""
    generator = torch.Generator().manual_seed(seed)
    values = [wasserstein(tall.sample_global(count, generator), tall, 1)
              for _ in range(repeats)]
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean()), float(tensor.std(unbiased=True))
