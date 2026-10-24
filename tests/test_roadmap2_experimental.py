"""Experimental roadmap items: 2.10 EDM output parameterisation + training noise
floor, 2.8 Fisher-identity consistency penalty. CPU, contracts and identities only."""

import pytest
import torch

from compass import Atomic, ScoreBasedInferenceModel
from compass.SDE import VESDE
from compass.consistency import FisherConsistency, gaussian_prior_score


def tokenized(parameterization="eps", seed=0, sigma_data=1.0):
    torch.manual_seed(seed)
    model = ScoreBasedInferenceModel(theta_size=2, x_spec=[Atomic("x", 2)], arch="tokenized",
                                     sigma=15.0, hidden_size=16, depth=2, num_heads=2,
                                     mlp_ratio=2, parameterization=parameterization,
                                     parameterization_sigma_data=sigma_data)
    for module in model.model.modules():
        if hasattr(module, "adaLN_modulation"):
            torch.nn.init.normal_(module.adaLN_modulation[-1].weight, std=0.3)
            torch.nn.init.normal_(module.adaLN_modulation[-1].bias, std=0.3)
    for parameter in model.model.final_layer.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    return model


def batch(seed=1, rows=6):
    torch.manual_seed(seed)
    x = torch.randn(rows, 4) * 3
    t = torch.rand(rows, 1) * 0.9 + 0.05
    c = torch.zeros(rows, 4)
    c[:, 2:] = 1
    c[0] = 0
    return x, t, c


# ----- 2.10 -------------------------------------------------------------------------

@pytest.mark.parametrize("sigma_data", [1.0, 2.5])
def test_edm_output_is_the_documented_transform(sigma_data):
    edm, eps = tokenized("edm", sigma_data=sigma_data), tokenized("eps")
    x, t, c = batch()
    lam = edm.sde.lambda_t(t)
    latent = 1 - c
    x_in = latent * x / torch.sqrt(lam ** 2 + sigma_data ** 2) + c * x
    F = eps.model(x=x_in, t=t, c=c)
    expected = latent * (-lam * x / (lam ** 2 + sigma_data ** 2)
                         + sigma_data / torch.sqrt(lam ** 2 + sigma_data ** 2) * F) + c * F
    assert torch.allclose(edm.model(x=x, t=t, c=c), expected, atol=1e-5)
    # positional calls and attention-weight calls see the same transform
    assert torch.allclose(edm.model(x, t, c), expected, atol=1e-5)
    out, _ = edm.model(x, t, c, return_attn_weights=True)
    assert torch.allclose(out, expected, atol=1e-5)


def test_edm_with_zero_network_is_the_gaussian_baseline_score():
    model = tokenized("edm")
    for parameter in model.model.final_layer.parameters():
        torch.nn.init.zeros_(parameter)
    x, t, c = batch()
    score = model.output_scale_function(t, model.model(x=x, t=t, c=c))
    lam = model.sde.lambda_t(t)
    assert torch.allclose((score * (1 - c))[:, :2], (-x / (1 + lam ** 2))[:, :2], atol=1e-5)


def test_edm_round_trips_and_eps_checkpoints_are_unchanged(tmp_path):
    model = tokenized("edm", sigma_data=2.0)
    model.save(str(tmp_path))
    payload = torch.load(tmp_path / "Model.pt")
    assert payload["parameterization"] == "edm" and payload["parameterization_sigma_data"] == 2.0
    loaded = ScoreBasedInferenceModel.load(str(tmp_path / "Model.pt"), device="cpu")
    x, t, c = batch()
    assert torch.allclose(loaded.model(x=x, t=t, c=c), model.model(x=x, t=t, c=c), atol=1e-6)
    tokenized("eps").save(str(tmp_path / ".."), name="plain")
    assert "parameterization" not in torch.load(tmp_path / ".." / "plain.pt")


def test_edm_composes_with_the_log_lambda_time_map():
    torch.manual_seed(0)
    model = ScoreBasedInferenceModel(theta_size=2, x_spec=[Atomic("x", 2)], arch="tokenized",
                                     hidden_size=16, depth=1, num_heads=2,
                                     parameterization="edm", time_embedding="log_lambda")
    x, t, c = batch()
    out = model.model(x=x, t=t, c=c)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_time_eps_sets_the_training_noise_floor(tmp_path):
    model = tokenized("edm")
    theta, x = torch.randn(32, 2), torch.randn(32, 2)
    model.train(theta, x, batch_size=16, max_epochs=1, device="cpu", verbose=False,
                path=str(tmp_path), time_eps=1e-5, time_sampling="log_sigma")
    assert model.trainer.eps == 1e-5
    torch.manual_seed(0)
    t = model.trainer._sample_timesteps(20000)
    assert float(t.min()) < 1e-4 and float(t.min()) >= 1e-5 * 0.999


# ----- 2.8 --------------------------------------------------------------------------
# g ~ N(0, 1), l | g ~ N(g, a^2), x | l ~ N(l, b^2): x | g ~ N(g, v), v = a^2 + b^2, so
# g | x ~ N(x / (1 + v), v / (1 + v)) and its diffused score is closed form.

A2, B2 = 0.25, 0.25
V = A2 + B2


class FakeTrainer:
    def __init__(self, bias=0.0):
        self.sde = VESDE(sigma=15.0)
        self.device = "cpu"
        self.bias = bias

    def _sample_timesteps(self, n):
        lam = torch.exp(torch.empty(n, 1).uniform_(torch.log(torch.tensor(0.05)).item(),
                                                   torch.log(torch.tensor(3.0)).item()))
        return self.sde.time_of_lambda(lam)

    def _get_score(self, x, t, c):
        lam = self.sde.lambda_t(t).reshape(-1, 1)
        mean = x[:, 2:3] / (1 + V)
        var = V / (1 + V)
        score = torch.zeros_like(x)
        score[:, :1] = -(x[:, :1] - mean) / (var + lam ** 2) + self.bias
        return score


def pool(rows=40000):
    torch.manual_seed(3)
    g = torch.randn(rows, 1)
    l = g + A2 ** 0.5 * torch.randn(rows, 1)
    x = l + B2 ** 0.5 * torch.randn(rows, 1)
    return torch.cat([g, l, x], 1)


def mean_penalty(bias, calls=40):
    penalty = FisherConsistency(gaussian_prior_score([0.0], [1.0]), indices=[0], observed=[2],
                                anchors=16, neighbours=256, min_ess=8)
    penalty.bind(None, pool())
    trainer = FakeTrainer(bias)
    torch.manual_seed(5)
    return sum(float(penalty(trainer, "cpu")) for _ in range(calls)) / calls, penalty


def test_exact_score_satisfies_the_fisher_identity():
    exact, penalty = mean_penalty(0.0)
    biased, _ = mean_penalty(0.1)
    # a 0.1 bias costs lambda^2 * 0.01 per anchor (measured 0.012 against 0.0005 for
    # the exact score, whose penalty is resampling noise only).
    assert biased > 10 * exact
    assert all(kept > 0 for _, kept, _ in penalty.history)


def test_penalty_has_a_gradient_and_respects_every():
    torch.manual_seed(0)
    network = torch.nn.Linear(3, 3)
    trainer = FakeTrainer()
    trainer._get_score = lambda x, t, c: network(x)
    penalty = FisherConsistency(gaussian_prior_score([0.0], [1.0]), indices=[0], observed=[2],
                                anchors=4, neighbours=64, min_ess=1.0, every=2)
    penalty.bind(None, pool(2000))
    assert float(penalty(trainer, "cpu")) == 0.0          # step 1 skipped
    value = penalty(trainer, "cpu")
    value.backward()
    assert network.weight.grad is not None and network.weight.grad.abs().sum() > 0


def test_penalty_trains_end_to_end(tmp_path):
    torch.manual_seed(0)
    model = ScoreBasedInferenceModel(nodes_size=3, sigma=15.0, hidden_size=8, depth=1,
                                     num_heads=2, mlp_ratio=2)
    data = pool(256)
    penalty = FisherConsistency(gaussian_prior_score([0.0], [1.0]), indices=[0], observed=[2],
                                anchors=4, neighbours=32, min_ess=1.0)
    model.train(data[:, :2], data[:, 2:], batch_size=64, max_epochs=1, device="cpu",
                verbose=False, path=str(tmp_path), fisher_consistency=penalty)
    assert len(penalty.history) == 4 and torch.isfinite(torch.tensor(model.trainer.train_loss)).all()
