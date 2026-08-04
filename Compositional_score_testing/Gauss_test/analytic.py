"""Analytic Gaussian/GMM score models for Figures 1 and 2."""
from __future__ import annotations

import math
import torch
from torch import nn

from compass.SDE import VPSDE
from compass.Sampler import Sampler

Tensor = torch.Tensor


def gaussian_log_prob(value: Tensor, mean: Tensor, covariance: Tensor) -> Tensor:
    delta = value - mean
    solved = torch.linalg.solve(covariance, delta.unsqueeze(-1)).squeeze(-1)
    _, logdet = torch.linalg.slogdet(covariance)
    return -.5 * (delta * solved).sum(-1) - .5 * logdet


class FrozenErrorNet(nn.Module):
    """Fixed random bounded perturbation used for the epsilon experiment."""
    def __init__(self, input_dim: int, output_dim: int, seed: int = 1234):
        super().__init__()
        state = torch.random.get_rng_state(); torch.manual_seed(seed)
        self.network = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh(),
                                     nn.Linear(64, output_dim), nn.Tanh())
        torch.random.set_rng_state(state)
        for parameter in self.parameters(): parameter.requires_grad_(False)

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class AnalyticPosteriorScore(nn.Module):
    def __init__(self, toy, sde: VPSDE, epsilon: float):
        super().__init__(); self.toy, self.sde, self.epsilon = toy, sde, float(epsilon)
        self.error = FrozenErrorNet(2 * toy.dimension + 1, toy.dimension)

    def forward(self, x: Tensor, t: Tensor, c: Tensor, return_attn_weights: bool = False):
        del c
        theta, observation = x[:, :self.toy.dimension], x[:, self.toy.dimension:]
        time = t.reshape(-1, 1).expand(theta.shape[0], 1)
        std = self.sde.sigma_t(time).to(theta)
        scaled_score = std * self.toy.diffused_single_score(theta, observation, time, self.sde)
        if self.epsilon:
            alpha = self.sde.alpha_t(time).to(theta)
            scaled_score = scaled_score + self.epsilon * self.error(torch.cat((theta, observation, alpha), -1))
        output = torch.zeros_like(x); output[:, :self.toy.dimension] = scaled_score
        return (output, output.new_zeros(1)) if return_attn_weights else output


class AnalyticSBIm:
    """Minimal COMPASS wrapper around an exact single-observation score."""
    def __init__(self, toy, epsilon: float, device: torch.device):
        self.sde = VPSDE(beta_min=.1, beta_max=40.)
        self.model = AnalyticPosteriorScore(toy, self.sde, epsilon).to(device)
        self.nodes_size = 2 * toy.dimension
        self.sampler = Sampler(self)

    def output_scale_function(self, t: Tensor, value: Tensor) -> Tensor:
        return value / self.sde.sigma_t(t).to(value.device)


class GaussianToy:
    """Correlated Gaussian simulator with the paper's N(0,I) prior."""
    def __init__(self, dimension: int, prior_mean: Tensor | None = None,
                 prior_std: Tensor | None = None, rho: float = .8):
        # The final paper explicitly fixes N(0,I). The optional arguments remain
        # accepted so older calls based on a pre-publication script stay valid.
        del prior_mean, prior_std
        self.dimension = dimension
        self.prior_mean, self.prior_std = torch.zeros(dimension), torch.ones(dimension)
        self.prior_covariance = torch.eye(dimension)
        self.likelihood_covariance = torch.eye(dimension) * (1 - rho) + rho
        self.likelihood_precision = torch.linalg.inv(self.likelihood_covariance)
        self.prior_precision = torch.eye(dimension)
        self.single_covariance = torch.linalg.inv(self.prior_precision + self.likelihood_precision)

    def sample_problem(self, n: int, seed: int) -> tuple[Tensor, Tensor]:
        generator = torch.Generator().manual_seed(seed)
        theta = torch.randn(self.dimension, generator=generator)
        chol = torch.linalg.cholesky(self.likelihood_covariance)
        return theta, theta + torch.randn(n, self.dimension, generator=generator) @ chol.T

    def posterior(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        precision = self.prior_precision + len(observations) * self.likelihood_precision
        covariance = torch.linalg.inv(precision)
        mean = covariance @ (self.likelihood_precision @ observations.sum(0))
        return mean, covariance

    def sample_reference(self, observations: Tensor, count: int, seed: int) -> Tensor:
        mean, covariance = self.posterior(observations)
        generator = torch.Generator().manual_seed(seed)
        return mean + torch.randn(count, self.dimension, generator=generator) @ torch.linalg.cholesky(covariance).T

    def single_covariance_estimate(self, observations: Tensor) -> Tensor:
        del observations
        return self.single_covariance

    def single_mean_estimate(self, observations: Tensor) -> Tensor:
        information = torch.einsum(
            "ij,nj->ni", self.likelihood_precision, observations,
        )
        return torch.einsum("ij,nj->ni", self.single_covariance, information)

    def single_precision_diagonal(self, observations: Tensor) -> Tensor:
        return 1.0 / torch.diagonal(
            self.single_covariance_estimate(observations)
        )[None]

    def diffused_single_score(self, theta: Tensor, observation: Tensor,
                              time: Tensor, sde: VPSDE) -> Tensor:
        covariance = self.single_covariance.to(theta)
        mean = (covariance @ torch.einsum("ij,bj->bi", self.likelihood_precision.to(theta), observation).T).T
        alpha, sigma = sde.alpha_t(time).to(theta), sde.sigma_t(time).to(theta)
        diffused_covariance = (alpha[..., None].square() * covariance
                              + sigma[..., None].square() * torch.eye(self.dimension, device=theta.device))
        return -torch.linalg.solve(diffused_covariance, (theta - alpha * mean).unsqueeze(-1)).squeeze(-1)


class GaussianMixtureToy:
    def __init__(self, dimension: int):
        self.dimension = dimension
        diagonal = torch.linspace(.6, 1.4, dimension)
        self.component_covariances = torch.stack((torch.diag(2.25 * diagonal),
                                                   torch.diag(diagonal / 9.)))
        self.component_precisions = torch.linalg.inv(self.component_covariances)
        self.posterior_covariances = torch.linalg.inv(self.component_precisions + torch.eye(dimension))

    @property
    def prior_mean(self): return torch.zeros(self.dimension)

    @property
    def prior_std(self): return torch.ones(self.dimension)

    def sample_problem(self, n: int, seed: int) -> tuple[Tensor, Tensor]:
        generator = torch.Generator().manual_seed(seed)
        theta = torch.randn(self.dimension, generator=generator)
        components = torch.randint(0, 2, (n,), generator=generator)
        observations = [theta + torch.randn(self.dimension, generator=generator)
                        @ torch.linalg.cholesky(self.component_covariances[int(k)]).T
                        for k in components]
        return theta, torch.stack(observations)

    def _components(self, observation: Tensor):
        information = torch.einsum("kij,...j->...ki", self.component_precisions.to(observation), observation)
        means = torch.einsum("kij,...kj->...ki", self.posterior_covariances.to(observation), information)
        marginal = self.component_covariances.to(observation) + torch.eye(self.dimension, device=observation.device)
        expanded = observation[..., None, :].expand(*observation.shape[:-1], 2, self.dimension)
        weights = torch.softmax(gaussian_log_prob(expanded, torch.zeros_like(expanded), marginal), -1)
        return means, self.posterior_covariances.to(observation), weights

    def diffused_single_score(self, theta: Tensor, observation: Tensor,
                              time: Tensor, sde: VPSDE) -> Tensor:
        means, covariances, weights = self._components(observation)
        alpha, sigma = sde.alpha_t(time).to(theta), sde.sigma_t(time).to(theta)
        diffused_means = alpha[:, None, :] * means
        diffused_covariances = (alpha[:, None, None].square() * covariances[None]
                                + sigma[:, None, None].square()
                                * torch.eye(self.dimension, device=theta.device))
        delta = theta[:, None, :] - diffused_means
        scores = -torch.linalg.solve(diffused_covariances, delta.unsqueeze(-1)).squeeze(-1)
        log_density = gaussian_log_prob(theta[:, None, :], diffused_means, diffused_covariances)
        responsibilities = torch.softmax(torch.log(weights.clamp_min(1e-30)) + log_density, -1)
        return (responsibilities[..., None] * scores).sum(1)

    def single_covariance_estimate(self, observations: Tensor) -> Tensor:
        means, covariances, weights = self._components(observations)
        second_moment = (
            covariances[None]
            + means.unsqueeze(-1) * means.unsqueeze(-2)
        )
        mean = (weights[..., None] * means).sum(1)
        return (
            weights[..., None, None] * second_moment
        ).sum(1) - mean.unsqueeze(-1) * mean.unsqueeze(-2)

    def single_precision_diagonal(self, observations: Tensor) -> Tensor:
        covariance = self.single_covariance_estimate(observations)
        return 1.0 / torch.diagonal(
            covariance, dim1=-2, dim2=-1
        ).clamp_min(1e-8)

    def log_posterior(self, theta: Tensor, observations: Tensor) -> Tensor:
        delta = observations[None, :, None, :] - theta[:, None, None, :]
        component = gaussian_log_prob(delta, torch.zeros_like(delta), self.component_covariances.to(theta)) - math.log(2)
        return -.5 * theta.square().sum(-1) + torch.logsumexp(component, -1).sum(-1)
