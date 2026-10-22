"""Covariance head (compass.experimental.covariance_head) in place of the score Jacobian.

1. A head returning the exact information C^-1 reproduces the Jacobian rule's draws on a
   Gaussian oracle (the hook is wired exactly where the Jacobian precision was).
2. Trained by second-order DSM against the oracle denoiser, the head recovers C^-1.
3. K is symmetric PSD by construction.
"""
import torch

from compass.MultiObsSampler import MultiObsSampler
from compass.experimental.covariance_head import (
    CovarianceHead, gaussian_nll, train_covariance_head,
)
from compass.experimental.oracles import LinearGaussianHierarchy, oracle_model


def hierarchy():
    generator = torch.Generator().manual_seed(0)
    A = torch.randn(1, 2, generator=generator, dtype=torch.float64) * 0.7
    B = torch.randn(2, 1, generator=generator, dtype=torch.float64)
    C = torch.randn(2, 2, generator=generator, dtype=torch.float64) * 0.5
    return LinearGaussianHierarchy(A, B, C, tau=0.6, nu=0.5)


MASK = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32)


class ExactHead(CovarianceHead):
    def __init__(self, information):
        super().__init__([0, 1, 2], [3, 4])
        self.register_buffer("information", information.float())

    def forward(self, y, x, lam):
        return self.information.expand(y.shape[0], 3, 3)


def draw(model, head=None, correction="gauss_jacobian_FD"):
    torch.manual_seed(3)
    data = torch.randn(5, 2) * 0.8
    sampler = MultiObsSampler(model)
    if head is not None:
        sampler.set_covariance_head(head)
    torch.manual_seed(4)
    return sampler.sample(world_size=1, data=data, condition_mask=MASK, timesteps=20,
                          num_samples=32, hierarchy=[0, 1], correction=correction,
                          prior=(torch.zeros(2), torch.ones(2)), verbose=False)


def exact_information(h):
    _, _, cov = h.conditional(MASK)
    return torch.linalg.inv(cov)


def test_exact_head_reproduces_the_jacobian_rule():
    h = hierarchy()
    model = oracle_model(h, sigma=10.0)
    reference = draw(model)
    with_head = draw(model, ExactHead(exact_information(h)))
    torch.testing.assert_close(with_head, reference, rtol=2e-3, atol=2e-3)


def test_head_output_is_symmetric_psd():
    head = CovarianceHead([0, 1, 2], [3, 4])
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0, 0.5)
    K = head(torch.randn(64, 3), torch.randn(64, 2), torch.rand(64) + 0.01)
    torch.testing.assert_close(K, K.mT)
    values = torch.linalg.eigvalsh(K.detach().double())
    assert bool((values >= -1e-6 * values.abs().amax(-1, keepdim=True)).all())


def test_trained_head_recovers_the_gaussian_information():
    h = hierarchy()
    model = oracle_model(h, sigma=10.0)
    U, gain, cov = h.conditional(MASK)

    def simulate(count, generator):
        z = h.simulate(count, generator)
        return z[:, :3], z[:, 3:]

    def denoiser(y, x, lam):        # exact Tweedie mean of the oracle posterior
        mean = x.double() @ gain.T
        eye = torch.eye(3, dtype=torch.float64)
        lam2 = lam.double().reshape(-1, 1, 1) ** 2
        g = cov @ torch.linalg.inv(cov + lam2 * eye)
        return (mean.unsqueeze(-1) + g @ (y.double() - mean).unsqueeze(-1)).squeeze(-1)

    head = train_covariance_head(model, simulate, [0, 1, 2], [3, 4], lam_min=0.05,
                                 lam_max=5.0, steps=1500, batch=1024, lr=3e-3,
                                 hidden=64, depth=2, denoiser=denoiser, verbose=False)
    target = torch.linalg.inv(cov)
    with torch.no_grad():
        K = head(0.3 * torch.randn(256, 3), torch.randn(256, 2), torch.full((256,), 0.5)).double()
    error = (K.mean(0) - target).norm() / target.norm()
    assert float(error) < 0.15, float(error)
    # and the NLL at the truth is no larger than at K = 0
    theta = torch.randn(10, 3, dtype=torch.float64)
    lam = torch.full((10,), 0.5, dtype=torch.float64)
    assert float(gaussian_nll(theta, theta, target.expand(10, 3, 3), lam).mean()) < \
        float(gaussian_nll(theta, theta, torch.zeros(10, 3, 3, dtype=torch.float64), lam).mean()) + 10
