"""Covariance output on the score network (compass.covariance_output)."""
import pytest
import torch

from compass import ScoreBasedInferenceModel
from compass.MultiObsSampler import MultiObsSampler
from compass.covariance_output import (
    CovarianceObjective, NetworkCovarianceHead, enable_covariance_output,
    load_with_covariance_output,
)
from compass.experimental.oracles import LinearGaussianHierarchy


def legacy():
    torch.manual_seed(0)
    return ScoreBasedInferenceModel(nodes_size=5, sigma=10.0, hidden_size=16, depth=2,
                                    num_heads=2, mlp_ratio=2)


def tokenized():
    torch.manual_seed(0)
    return ScoreBasedInferenceModel(nodes_size=5, sigma=10.0, hidden_size=16, depth=2,
                                    num_heads=2, mlp_ratio=2, arch="tokenized")


def hierarchy():
    g = torch.Generator().manual_seed(0)
    return LinearGaussianHierarchy(torch.randn(1, 2, generator=g, dtype=torch.float64) * 0.7,
                                   torch.randn(2, 1, generator=g, dtype=torch.float64),
                                   torch.randn(2, 2, generator=g, dtype=torch.float64) * 0.5,
                                   tau=0.6, nu=0.5)


MASK = torch.tensor([0.0, 0, 0, 1, 1])


@pytest.mark.parametrize("build", [legacy, tokenized])
def test_output_is_spd_and_leaves_the_score_untouched(build):
    m = build()
    x, t, c = torch.randn(7, 5), torch.full((7, 1), 0.4), MASK.expand(7, -1)
    with torch.no_grad():
        before = m.model(x=x, t=t, c=c)
    head = enable_covariance_output(m, rank=3)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0, 0.3)
        after = m.model(x=x, t=t, c=c)
        K = head.information([0, 1, 2]).double()
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(K, K.mT)
    assert float(torch.linalg.eigvalsh(K).min()) > 0


def test_objective_is_zero_at_zero_information_and_trains_the_readout_only_when_detached():
    m = tokenized()
    head = enable_covariance_output(m, rank=3)
    data = torch.randn(16, 5)
    t = torch.full((16, 1), 0.3)
    x_t = data + 0.5 * torch.randn(16, 5) * (1 - MASK)
    out = m.model(x=x_t, t=t, c=MASK.expand(16, -1))
    trainer = m.trainer
    trainer.sde = m.sde
    loss = CovarianceObjective(head, detach_trunk=True)(trainer, data, x_t, t, out, MASK.expand(16, -1))
    loss.backward()
    trunk = [p for n, p in m.model.named_parameters() if "covariance_output" not in n]
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in trunk)
    assert float(head.weight.grad.abs().sum()) > 0


def test_joint_training_is_calibrated_to_the_networks_own_residual(tmp_path):
    """The NLL target is Cov(theta_0 - D_net)^-1 - lam^-2 I: the information of the network's
    *own* denoising residual (wider than the true posterior's when the score is imperfect),
    which is what a composition should weight a row by."""
    from compass import ConditionalMenu
    h = hierarchy()
    z = h.simulate(8000, torch.Generator().manual_seed(1)).float()
    theta, x = z[:, :3], z[:, 3:]
    m = tokenized()
    head = enable_covariance_output(m, rank=3, node_scale=z.std(0))
    menu = ConditionalMenu(m.schema, {"theta": range(3), "x": range(3, 5)},
                           [("posterior", ["x"])], bernoulli_fraction=0.1)
    m.train(theta=theta, x=x, theta_val=theta[:500], x_val=x[:500], batch_size=256,
            max_epochs=25, lr=2e-3, verbose=False, path=str(tmp_path), device="cpu",
            covariance_objective=CovarianceObjective(head), mask_sampler=menu)
    net = NetworkCovarianceHead(m, [0, 1, 2], [3, 4])
    lam, N = 1.0, 4000
    y = theta[:N] + lam * torch.randn(N, 3)
    t = m.sde.time_of_lambda(torch.full((N, 1), lam))
    state = torch.cat([y, x[:N]], 1)
    with torch.no_grad():
        s = m.output_scale_function(t, m.model(x=state, t=t, c=MASK.expand(N, -1)))[:, :3]
    residual = (theta[:N] - (y + lam ** 2 * s)).double()
    empirical = torch.linalg.inv(torch.cov(residual.T)) - torch.eye(3, dtype=torch.float64) / lam ** 2
    K = net.information(y, x[:N], torch.full((N,), lam)).mean(0)
    error = float((K - empirical).norm() / empirical.norm())
    assert error < 0.3, error
    # round trip
    m.save(path=str(tmp_path), name="cov")
    fresh = load_with_covariance_output(tokenized(), tmp_path / "cov.pt", rank=3)
    K2 = NetworkCovarianceHead(fresh, [0, 1, 2], [3, 4]).information(y, x[:N], torch.full((N,), lam)).mean(0)
    torch.testing.assert_close(K2, K, rtol=1e-5, atol=1e-5)


def test_sampler_uses_one_forward_per_refresh_instead_of_the_jacobian():
    m = tokenized()
    head = enable_covariance_output(m, rank=3)
    net = NetworkCovarianceHead(m, [0, 1, 2], [3, 4])
    sampler = MultiObsSampler(m)
    sampler.set_covariance_head(net)
    out = sampler.sample(world_size=1, data=torch.randn(4, 2), condition_mask=MASK, timesteps=10,
                         num_samples=16, hierarchy=[0, 1], correction="gauss_jacobian_FD",
                         prior=(torch.zeros(2), torch.ones(2)), verbose=False, jacobian_refresh=1)
    assert torch.isfinite(out).all()
    assert net.calls > 0
    assert sampler.solver_stats["jacobian_row_samples"] == 0      # no FD copies were evaluated


def test_sbim_load_reattaches_the_output(tmp_path):
    m = tokenized()
    head = enable_covariance_output(m, rank=3)
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0, 0.3)
    m.save(path=str(tmp_path), name="cov")
    loaded = ScoreBasedInferenceModel.load(str(tmp_path / "cov.pt"), device="cpu")
    x, t = torch.randn(6, 5), torch.full((6, 1), 0.4)
    with torch.no_grad():
        m.model(x=x, t=t, c=MASK.expand(6, -1))
        a = m.covariance_output.information([0, 1, 2])
        loaded.model(x=x, t=t, c=MASK.expand(6, -1))
        b = loaded.covariance_output.information([0, 1, 2])
    torch.testing.assert_close(a, b)


def test_output_lives_on_the_backbone_behind_a_forwarding_wrapper(tmp_path):
    """A wrapper that forwards __getattr__ and state_dict to its backbone (as the
    preconditioning wrappers do) must not receive the output: it would not be saved."""
    class Wrapped(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x, t, c, **kw):
            return self.backbone(x=x, t=t, c=c)

        def state_dict(self, *a, **k):
            return self.backbone.state_dict(*a, **k)

        def load_state_dict(self, *a, **k):
            return self.backbone.load_state_dict(*a, **k)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__("backbone"), name)

    m = tokenized()
    m.model = Wrapped(m.model)
    enable_covariance_output(m, rank=3)
    assert any(k.startswith("covariance_output.") for k in m.model.state_dict())
