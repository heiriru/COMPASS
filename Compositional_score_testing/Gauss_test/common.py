"""Shared COMPASS training, sampling, reference-MCMC, and metric utilities."""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from compass import ScoreBasedInferenceModel


Tensor = torch.Tensor
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_rows(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Refusing to write an empty result table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def standardize_x(x: Tensor, mean: Tensor | None = None,
                  std: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    mean = x.mean(0) if mean is None else mean
    std = x.std(0).clamp_min(1e-6) if std is None else std
    return torch.nan_to_num((x - mean) / std), mean, std


def new_compass_model(theta_dim: int, x_dim: int, device: torch.device) -> ScoreBasedInferenceModel:
    """Paper-scale score model expressed with COMPASS's transformer backbone."""
    return ScoreBasedInferenceModel(
        nodes_size=theta_dim + x_dim,
        sde_type="vpsde",
        beta_min=0.1,
        beta_max=40.0,
        hidden_size=256,
        depth=3,
        num_heads=8,
        mlp_ratio=4,
        device=str(device),
    )


def train_compass_model(
    z: Tensor,
    x: Tensor,
    output_dir: Path,
    device: torch.device,
    *,
    epochs: int = 5000,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    seed: int = 42,
) -> Path:
    """Train through :class:`ScoreBasedInferenceModel` and save normalization."""
    seed_everything(seed)
    permutation = torch.randperm(z.shape[0])
    split = max(1, int(0.8 * z.shape[0]))
    train_indices, validation_indices = permutation[:split], permutation[split:]
    x_norm, x_mean, x_std = standardize_x(x)
    model = new_compass_model(z.shape[1], x.shape[1], device)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train(
        z[train_indices], x_norm[train_indices],
        z[validation_indices], x_norm[validation_indices],
        batch_size=batch_size,
        max_epochs=epochs,
        lr=learning_rate,
        device=str(device),
        verbose=True,
        path=str(output_dir),
        name="model",
        early_stopping_patience=20,
        time_sampling="uniform",  # paper: t ~ U(0, 1)
    )
    checkpoint = output_dir / "model_checkpoint.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"COMPASS training did not create {checkpoint}")
    atomic_torch_save({"x_mean": x_mean, "x_std": x_std,
                       "theta_dim": z.shape[1], "x_dim": x.shape[1]},
                      output_dir / "normalization.pt")
    return checkpoint


def load_compass_model(model_dir: Path, device: torch.device) -> tuple[ScoreBasedInferenceModel, dict]:
    checkpoint = model_dir / "model_checkpoint.pt"
    normalization = torch.load(model_dir / "normalization.pt", map_location="cpu", weights_only=True)
    model = ScoreBasedInferenceModel.load(str(checkpoint), device=str(device))
    model.model.eval().to(device)
    return model, normalization


@torch.no_grad()
def compass_gauss(
    model,
    context: Tensor,
    theta_dim: int,
    device: torch.device,
    *,
    num_samples: int,
    timesteps: int = 1000,
    correction: str = "gauss",
    posterior_precision: Tensor | None = None,
    posterior_covariance: Tensor | None = None,
    precision_est_samples: int = 1000,
    precision_est_timesteps: int = 100,
    denoise_clamp: float | None = 5.0,
) -> Tensor:
    """Sample with COMPASS DPM and a diagonal or full Gaussian correction."""
    if correction not in ("gauss", "full_gaussian"):
        raise ValueError(f"Unsupported Gaussian correction: {correction}")
    context = context.to(device)
    samples = model.sample(
        x=context,
        multi_obs_inference=True,
        hierarchy=list(range(theta_dim)),
        prior=(torch.zeros(theta_dim), torch.ones(theta_dim)),
        correction=correction,
        posterior_precision=posterior_precision,
        posterior_covariance=posterior_covariance,
        precision_est_samples=precision_est_samples,
        precision_est_timesteps=precision_est_timesteps,
        precision_est_batch_size=128,
        denoise_clamp=denoise_clamp,
        timesteps=timesteps,
        num_samples=num_samples,
        method="dpm",
        order=2,
        corrector_steps=0,
        final_corrector_steps=0,
        device=str(device),
        verbose=True,
    )
    return samples[0].detach().cpu()


@torch.no_grad()
def paper_langevin(
    model,
    context: Tensor,
    theta_dim: int,
    device: torch.device,
    *,
    num_samples: int,
    steps: int = 400,
    langevin_steps: int = 5,
    tau: float = 0.5,
    clip: float | None = None,
    prior_mean: Tensor | None = None,
    prior_std: Tensor | None = None,
) -> Tensor:
    """F-NPSE ULA exactly as configured in Appendix E of the paper.

    This intentionally does not use COMPASS's generic Langevin corrector: the
    baseline uses its published uniform time grid and
    ``delta=tau*(1-gamma)/sqrt(gamma)`` step size.  Scores still come from the
    same COMPASS-trained network as the GAUSS comparison.
    """
    context = torch.as_tensor(context, dtype=torch.float32, device=device)
    n_observations = context.shape[0]
    samples = torch.randn(num_samples, theta_dim, device=device)
    mask = torch.cat((torch.zeros(theta_dim, device=device),
                      torch.ones(context.shape[1], device=device)))
    prior_mean = torch.zeros(theta_dim, device=device) if prior_mean is None else prior_mean.to(device)
    prior_std = torch.ones(theta_dim, device=device) if prior_std is None else prior_std.to(device)
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    model.model.eval()
    for index, time in enumerate(times[:-1]):
        next_time = times[index + 1]
        alpha = model.sde.alpha_t(time)
        gamma = alpha / model.sde.alpha_t(next_time) if index < steps - 1 else alpha
        delta = tau * (1.0 - gamma) / torch.sqrt(gamma)
        t_batch = time.reshape(1, 1)
        for _ in range(langevin_steps):
            theta = samples[:, None, :].expand(-1, n_observations, -1)
            x = context[None, :, :].expand(num_samples, -1, -1)
            joint = torch.cat((theta, x), -1).reshape(-1, theta_dim + context.shape[1])
            masks = mask.expand(joint.shape[0], -1)
            scaled_score = model.model(x=joint, t=t_batch, c=masks)
            score = model.output_scale_function(t_batch, scaled_score)[:, :theta_dim]
            score = score.reshape(num_samples, n_observations, theta_dim)
            sigma = model.sde.sigma_t(time)
            prior_variance = alpha.square() * prior_std.square() + sigma.square()
            prior_score = -(samples - alpha * prior_mean) / prior_variance
            composed_score = (1 - n_observations) * prior_score + score.sum(1)
            samples = samples + delta * composed_score + torch.sqrt(2.0 * delta) * torch.randn_like(samples)
        if clip is not None:
            samples.clamp_(-clip, clip)
    return samples.cpu()


def _value_and_grad(log_density: Callable[[Tensor], Tensor], value: Tensor) -> tuple[Tensor, Tensor]:
    point = value.detach().requires_grad_(True)
    log_prob = log_density(point)
    gradient, = torch.autograd.grad(log_prob.sum(), point)
    return log_prob.detach(), gradient.detach()


def mala_reference(
    log_density: Callable[[Tensor], Tensor],
    dimension: int,
    *,
    num_samples: int = 1000,
    chains: int = 64,
    burnin: int = 1000,
    thinning: int = 5,
    step_size: float = 0.08,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Vectorized Metropolis-adjusted Langevin reference sampler."""
    seed_everything(seed)
    current = torch.randn(chains, dimension, device=device)
    current_lp, current_grad = _value_and_grad(log_density, current)
    draws: list[Tensor] = []
    iterations = burnin + thinning * math.ceil(num_samples / chains)
    scale = float(step_size)
    accepted_window = 0.0
    for iteration in range(iterations):
        mean_forward = current + 0.5 * scale**2 * current_grad
        proposal = mean_forward + scale * torch.randn_like(current)
        proposal_lp, proposal_grad = _value_and_grad(log_density, proposal)
        mean_reverse = proposal + 0.5 * scale**2 * proposal_grad
        forward_q = -0.5 * ((proposal - mean_forward) / scale).square().sum(-1)
        reverse_q = -0.5 * ((current - mean_reverse) / scale).square().sum(-1)
        log_accept = proposal_lp + reverse_q - current_lp - forward_q
        accept = torch.log(torch.rand(chains, device=device)) < log_accept
        accepted_window += float(accept.float().mean())
        current = torch.where(accept[:, None], proposal, current)
        current_lp = torch.where(accept, proposal_lp, current_lp)
        current_grad = torch.where(accept[:, None], proposal_grad, current_grad)
        if iteration < burnin and (iteration + 1) % 50 == 0:
            rate = accepted_window / 50.0
            scale *= math.exp(max(-0.2, min(0.2, rate - 0.574)))
            accepted_window = 0.0
        if iteration >= burnin and (iteration - burnin) % thinning == 0:
            draws.append(current.detach().cpu().clone())
    return torch.cat(draws, 0)[:num_samples]


def sliced_wasserstein(first: Tensor, second: Tensor, *, projections: int = 1000,
                       seed: int = 0) -> float:
    first, second = first.float().cpu(), second.float().cpu()
    count = min(first.shape[0], second.shape[0])
    generator = torch.Generator().manual_seed(seed)
    first = first[torch.randperm(first.shape[0], generator=generator)[:count]]
    second = second[torch.randperm(second.shape[0], generator=generator)[:count]]
    directions = torch.randn(projections, first.shape[1], generator=generator)
    directions = directions / directions.norm(dim=1, keepdim=True)
    projected_first = torch.sort(first @ directions.T, dim=0).values
    projected_second = torch.sort(second @ directions.T, dim=0).values
    return float(torch.sqrt((projected_first - projected_second).square().mean()))


def scaled_mmd_to_dirac(samples: Tensor, truth: Tensor) -> Tensor:
    """Paper's marginal MMD-to-Dirac proxy from its public plotting code."""
    variance = samples.var(0)
    return (variance + (samples.mean(0) - truth).square()) / variance.sqrt().clamp_min(1e-12)
