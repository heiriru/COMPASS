"""Simulators and analytic toy scores used by the Figure 1--4 experiments.

The benchmark definitions follow the paper's Appendix D/J and its public
reference implementation.  Parameters are represented internally in a latent
``z`` coordinate with a standard-normal prior.  This makes the Gaussian prior
assumption of COMPASS's ``correction='full_gaussian'`` exact: uniform priors use
a probit transform and log-normal priors use their underlying normal variable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch


Tensor = torch.Tensor
SQRT_TWO = math.sqrt(2.0)


def normal_cdf(z: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(z / SQRT_TWO))


def uniform_from_normal(z: Tensor, low: Tensor, high: Tensor) -> Tensor:
    # Avoid exact bounds, where likelihood implementations can become singular.
    u = normal_cdf(z).clamp(1e-7, 1.0 - 1e-7)
    return low.to(z) + (high.to(z) - low.to(z)) * u


class BenchmarkTask(Protocol):
    name: str
    theta_dim: int
    x_dim: int
    labels: tuple[str, ...]

    def theta_from_z(self, z: Tensor) -> Tensor: ...
    def simulate(self, z: Tensor, generator: torch.Generator | None = None) -> Tensor: ...
    def log_likelihood(self, z: Tensor, observations: Tensor) -> Tensor: ...


@dataclass
class SLCPTask:
    """The 5-parameter, 8-observable SLCP benchmark."""

    name: str = "slcp"
    theta_dim: int = 5
    x_dim: int = 8
    labels: tuple[str, ...] = (r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$", r"$\theta_5$")

    def __post_init__(self) -> None:
        self.low = torch.full((5,), -3.0)
        self.high = torch.full((5,), 3.0)

    def theta_from_z(self, z: Tensor) -> Tensor:
        return uniform_from_normal(z, self.low, self.high)

    @staticmethod
    def _mean_cholesky(theta: Tensor) -> tuple[Tensor, Tensor]:
        mean = theta[..., :2]
        s1 = theta[..., 2].square().clamp_min(1e-6)
        s2 = theta[..., 3].square().clamp_min(1e-6)
        rho = torch.tanh(theta[..., 4]).clamp(-0.999, 0.999)
        chol = theta.new_zeros(*theta.shape[:-1], 2, 2)
        chol[..., 0, 0] = s1
        chol[..., 1, 0] = rho * s2
        chol[..., 1, 1] = s2 * torch.sqrt(1.0 - rho.square())
        return mean, chol

    def simulate(self, z: Tensor, generator: torch.Generator | None = None) -> Tensor:
        theta = self.theta_from_z(z)
        mean, chol = self._mean_cholesky(theta)
        eps = torch.randn(*theta.shape[:-1], 4, 2, device=theta.device,
                          dtype=theta.dtype, generator=generator)
        values = mean[..., None, :] + torch.einsum("...ij,...kj->...ki", chol, eps)
        return values.reshape(*theta.shape[:-1], 8)

    def log_likelihood(self, z: Tensor, observations: Tensor) -> Tensor:
        theta = self.theta_from_z(z)
        mean, chol = self._mean_cholesky(theta)
        obs = observations.reshape(-1, 4, 2)
        delta = obs[None, :, :, :] - mean[:, None, None, :]
        solved = torch.linalg.solve_triangular(
            chol[:, None, None, :, :], delta.unsqueeze(-1), upper=False
        ).squeeze(-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        return (-0.5 * solved.square().sum(-1) - 0.5 * logdet[:, None, None]).sum((1, 2))


def _rk4_step(state: Tensor, dt: float, derivative, theta: Tensor) -> Tensor:
    k1 = derivative(state, theta)
    k2 = derivative(state + 0.5 * dt * k1, theta)
    k3 = derivative(state + 0.5 * dt * k2, theta)
    k4 = derivative(state + dt * k3, theta)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclass
class SIRTask:
    """SIR benchmark with ten binomial observations."""

    name: str = "sir"
    theta_dim: int = 2
    x_dim: int = 10
    labels: tuple[str, ...] = (r"$\beta$", r"$\gamma$")

    def __post_init__(self) -> None:
        self.log_loc = torch.log(torch.tensor([0.4, 0.125]))
        self.log_scale = torch.tensor([0.50, 0.20])

    def theta_from_z(self, z: Tensor) -> Tensor:
        return torch.exp(self.log_loc.to(z) + self.log_scale.to(z) * z)

    @staticmethod
    def _derivative(state: Tensor, theta: Tensor) -> Tensor:
        population = 1e6
        susceptible, infected, recovered = state.unbind(-1)
        beta, gamma = theta.unbind(-1)
        infections = beta * susceptible * infected / population
        return torch.stack((-infections, infections - gamma * infected, gamma * infected), -1)

    def _probabilities(self, z: Tensor) -> Tensor:
        theta = self.theta_from_z(z)
        state = theta.new_tensor([1e6 - 1.0, 1.0, 0.0]).expand(theta.shape[0], -1).clone()
        recorded = []
        for day in range(161):
            if day % 17 == 0:
                recorded.append((state[:, 1] / 1e6).clamp(1e-8, 1.0 - 1e-8))
            if day < 160:
                state = _rk4_step(state, 0.5, self._derivative, theta)
                state = _rk4_step(state, 0.5, self._derivative, theta)
                state = state.clamp_min(0.0)
        return torch.stack(recorded, -1)

    def simulate(self, z: Tensor, generator: torch.Generator | None = None) -> Tensor:
        probs = self._probabilities(z)
        # torch.distributions does not accept a Generator; inverse-CDF is not
        # available for Binomial, so reproducibility is controlled by torch's seed.
        return torch.distributions.Binomial(total_count=1000, probs=probs).sample()

    def log_likelihood(self, z: Tensor, observations: Tensor) -> Tensor:
        probs = self._probabilities(z)[:, None, :]
        obs = observations.to(probs)[None, :, :]
        return (obs * torch.log(probs) + (1000.0 - obs) * torch.log1p(-probs)).sum((1, 2))


@dataclass
class LotkaVolterraTask:
    """Lotka--Volterra benchmark with the paper's 20-dimensional summary."""

    name: str = "lotka_volterra"
    theta_dim: int = 4
    x_dim: int = 20
    labels: tuple[str, ...] = (r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$")

    def __post_init__(self) -> None:
        self.log_loc = torch.tensor([-0.125, -3.0, -0.125, -3.0])
        self.log_scale = torch.full((4,), 0.50)

    def theta_from_z(self, z: Tensor) -> Tensor:
        return torch.exp(self.log_loc.to(z) + self.log_scale.to(z) * z)

    @staticmethod
    def _derivative(state: Tensor, theta: Tensor) -> Tensor:
        prey, predator = state.unbind(-1)
        alpha, beta, gamma, delta = theta.unbind(-1)
        return torch.stack(((alpha - beta * predator) * prey,
                            (-gamma + delta * prey) * predator), -1)

    def _mean_trajectory(self, z: Tensor) -> Tensor:
        theta = self.theta_from_z(z)
        state = theta.new_tensor([30.0, 1.0]).expand(theta.shape[0], -1).clone()
        recorded = []
        # Original grid: 0..20 by .1, with every 21st point retained.
        for index in range(201):
            if index % 21 == 0:
                recorded.append(state)
            if index < 200:
                state = _rk4_step(state, 0.1, self._derivative, theta)
                state = state.clamp(1e-8, 1e4)
        return torch.stack(recorded, 1).transpose(1, 2).reshape(theta.shape[0], -1)

    def simulate(self, z: Tensor, generator: torch.Generator | None = None) -> Tensor:
        mean = self._mean_trajectory(z)
        return torch.exp(torch.log(mean.clamp_min(1e-10)) + 0.1 * torch.randn_like(mean))

    def log_likelihood(self, z: Tensor, observations: Tensor) -> Tensor:
        log_mean = torch.log(self._mean_trajectory(z).clamp_min(1e-10))[:, None, :]
        log_obs = torch.log(observations.clamp_min(1e-10)).to(log_mean)[None, :, :]
        return (-0.5 * ((log_obs - log_mean) / 0.1).square() - log_obs).sum((1, 2))


@dataclass
class JRNMMTask:
    """Three-parameter stochastic Jansen--Rit neural mass model (gain fixed at 0)."""

    name: str = "jrnnm_3d"
    theta_dim: int = 3
    x_dim: int = 33
    labels: tuple[str, ...] = (r"$C$", r"$\mu$", r"$\sigma$")
    integration_batch_size: int = 256

    def __post_init__(self) -> None:
        self.low = torch.tensor([10.0, 50.0, 100.0])
        self.high = torch.tensor([250.0, 500.0, 5000.0])

    def theta_from_z(self, z: Tensor) -> Tensor:
        return uniform_from_normal(z, self.low, self.high)

    @staticmethod
    def _sigmoid(v: Tensor) -> Tensor:
        return 5.0 / (1.0 + torch.exp((0.56 * (6.0 - v)).clamp(-60.0, 60.0)))

    def _simulate_batch(self, z: Tensor) -> Tensor:
        theta = self.theta_from_z(z)
        c, mu, sigma = theta.unbind(-1)
        state = torch.randn(theta.shape[0], 6, device=theta.device, dtype=theta.dtype)
        dt = 1.0 / 1024.0
        sqrt_dt = math.sqrt(dt)
        outputs = []
        total_steps = 10 * 1024
        burnin_steps = 2 * 1024
        for step in range(total_steps):
            x0, x1, x2, x3, x4, x5 = state.unbind(-1)
            drift = torch.stack((
                x3,
                x4,
                x5,
                3.25 * 100.0 * self._sigmoid(x1 - x2) - 200.0 * x3 - 10000.0 * x0,
                3.25 * 100.0 * (mu + 0.8 * c * self._sigmoid(c * x0)) - 200.0 * x4 - 10000.0 * x1,
                22.0 * 50.0 * (0.25 * c * self._sigmoid(0.25 * c * x0)) - 100.0 * x5 - 2500.0 * x2,
            ), -1)
            noise = torch.zeros_like(state)
            noise[:, 3] = 0.01 * torch.randn_like(c)
            noise[:, 4] = sigma * torch.randn_like(c)
            noise[:, 5] = torch.randn_like(c)
            state = state + dt * drift + sqrt_dt * noise
            if step >= burnin_steps and (step - burnin_steps) % 8 == 0:
                outputs.append(state[:, 1] - state[:, 2])
        signal = torch.stack(outputs, -1)
        return signal - signal.mean(-1, keepdim=True)

    @staticmethod
    def _log_psd(signal: Tensor) -> Tensor:
        # Matches the reference summary: 65-point, 50%-overlapping windows,
        # yielding 33 real FFT bins between 0 and 64 Hz.
        windows = signal.unfold(-1, 65, 32)
        power = torch.fft.rfft(windows, dim=-1).abs().square().mean(-2)
        return torch.log10(power.clamp_min(1e-12))

    def simulate(self, z: Tensor, generator: torch.Generator | None = None) -> Tensor:
        chunks = []
        for start in range(0, z.shape[0], self.integration_batch_size):
            chunks.append(self._log_psd(self._simulate_batch(z[start:start + self.integration_batch_size])))
        return torch.cat(chunks, 0)

    def log_likelihood(self, z: Tensor, observations: Tensor) -> Tensor:
        raise NotImplementedError("Figure 4 does not require an analytic JRNMM reference posterior.")


TASKS: dict[str, type] = {
    "slcp": SLCPTask,
    "sir": SIRTask,
    "lotka_volterra": LotkaVolterraTask,
    "jrnnm_3d": JRNMMTask,
}


def get_task(name: str) -> BenchmarkTask:
    try:
        return TASKS[name]()
    except KeyError as error:
        raise ValueError(f"Unknown task {name!r}; choose from {sorted(TASKS)}") from error
