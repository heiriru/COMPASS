import torch


TIME_GRID_TYPES = ("linear_t", "log_snr", "log_sigma")


def _invert_monotonic(fn, targets, eps, increasing):
    """Invert a scalar monotonic noise schedule by vectorized bisection."""
    low = torch.full_like(targets, eps)
    high = torch.ones_like(targets)
    for _ in range(64):
        mid = (low + high) / 2
        values = fn(mid)
        if increasing:
            low = torch.where(values < targets, mid, low)
            high = torch.where(values < targets, high, mid)
        else:
            low = torch.where(values > targets, mid, low)
            high = torch.where(values > targets, high, mid)
    return (low + high) / 2


def make_time_grid(sde, grid_type, timesteps, eps, device):
    """Return descending diffusion times for the requested schedule coordinate."""
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2.")
    dtype = torch.get_default_dtype()
    if grid_type == "linear_t":
        return torch.linspace(1.0, eps, timesteps, device=device, dtype=dtype)

    endpoint_t = torch.tensor([1.0, eps], device=device, dtype=dtype)
    tiny = torch.finfo(dtype).tiny
    if grid_type == "log_sigma":
        if not hasattr(sde, "sigma_t"):
            raise ValueError("time_grid_type='log_sigma' requires an SDE with sigma_t(t).")
        fn = lambda t: torch.log(sde.sigma_t(t).clamp_min(tiny))
        endpoints = fn(endpoint_t)
        targets = torch.linspace(endpoints[0], endpoints[1], timesteps, device=device)
        return _invert_monotonic(fn, targets, eps, increasing=True)

    if grid_type == "log_snr":
        mean_fn = getattr(sde, "marginal_prob_mean_coeff", None)
        if mean_fn is None:
            mean_fn = getattr(sde, "marginal_prob_mean", None)
        if mean_fn is None or not hasattr(sde, "marginal_prob_std"):
            raise ValueError(
                "time_grid_type='log_snr' requires a VP-style SDE exposing "
                "marginal_prob_mean_coeff(t) (or marginal_prob_mean(t)) and "
                "marginal_prob_std(t)."
            )

        def fn(t):
            return 2 * (
                torch.log(mean_fn(t).abs().clamp_min(tiny))
                - torch.log(sde.marginal_prob_std(t).clamp_min(tiny))
            )

        endpoints = fn(endpoint_t)
        targets = torch.linspace(endpoints[0], endpoints[1], timesteps, device=device)
        return _invert_monotonic(fn, targets, eps, increasing=False)

    raise ValueError(f"time_grid_type {grid_type!r} not recognized; choose one of {TIME_GRID_TYPES}.")
