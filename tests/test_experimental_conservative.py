"""Conservative score wrapper (compass.experimental.conservative)."""
import torch

from compass import ScoreBasedInferenceModel
from compass.experimental.conservative import ConservativeScore, enable_conservative, load_conservative


def model():
    torch.manual_seed(0)
    m = ScoreBasedInferenceModel(nodes_size=4, sigma=5.0, hidden_size=16, depth=2,
                                 num_heads=2, mlp_ratio=2)
    with torch.no_grad():
        for p in m.model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return m


def score(m, x, t, c):
    return m.output_scale_function(t, m.model(x=x, t=t, c=c))


def test_jacobian_is_symmetric():
    m = enable_conservative(model())
    m.model.eval()
    c = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    t = torch.tensor([[0.4]])
    x0 = torch.randn(1, 4)

    def f(latent):
        x = torch.cat([latent, x0[:, 3:]], -1)
        return score(m, x, t, c)[:, :3]
    J = torch.autograd.functional.jacobian(f, x0[:, :3]).reshape(3, 3)
    asym = (J - J.T).norm() / J.norm()
    assert float(asym) < 1e-5, float(asym)
    plain = model()
    Jp = torch.autograd.functional.jacobian(
        lambda l: score(plain, torch.cat([l, x0[:, 3:]], -1), t, c)[:, :3], x0[:, :3]).reshape(3, 3)
    assert float((Jp - Jp.T).norm() / Jp.norm()) > 1e-3       # the plain net is not


class LinearDenoiser(torch.nn.Module):
    """raw r = -(x - d0) / sigma with constant d0: J_d = 0, so conservative == plain."""

    def __init__(self, sde, d0):
        super().__init__()
        self.sde, self.d0 = sde, d0

    def forward(self, x, t, c, **kw):
        sigma = self.sde.marginal_prob_std(t).reshape(-1, 1)
        return -(x - self.d0) / sigma * (1 - c)


def test_flat_denoiser_gives_the_plain_score():
    m = model()
    base = LinearDenoiser(m.sde, torch.tensor([0.3, -0.2, 0.1, 0.0]))
    wrapped = ConservativeScore(base, m.sde).eval()
    x = torch.randn(5, 4)
    t = torch.full((5, 1), 0.3)
    c = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(5, -1)
    torch.testing.assert_close(wrapped(x=x, t=t, c=c), base(x=x, t=t, c=c), rtol=1e-5, atol=1e-5)


def test_training_step_and_round_trip(tmp_path):
    m = enable_conservative(model())
    theta, x = torch.randn(256, 3), torch.randn(256, 1)
    m.train(theta=theta, x=x, theta_val=theta[:64], x_val=x[:64], batch_size=64,
            max_epochs=1, verbose=False, path=str(tmp_path))
    m.save(path=str(tmp_path), name="cons")
    fresh = load_conservative(model(), tmp_path / "cons.pt")
    fresh.model.eval()
    m.model.eval()
    probe = torch.randn(4, 4)
    t = torch.full((4, 1), 0.5)
    c = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(4, -1)
    torch.testing.assert_close(fresh.model(x=probe, t=t, c=c), m.model(x=probe, t=t, c=c))
