"""Focused tests for the optional instantaneous divergence head."""

import os


CPU_USAGE_LIMIT_FRACTION = 0.06
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_cpu_usage_limit(fraction=CPU_USAGE_LIMIT_FRACTION):
    """Hard-limit this process and its children to host CPU capacity."""
    logical_cpus = os.cpu_count()
    if logical_cpus is None:
        raise RuntimeError("Cannot enforce the CPU cap: os.cpu_count() is unavailable.")

    cpu_limit = int(logical_cpus * fraction)
    if cpu_limit < 1:
        raise RuntimeError(
            f"Cannot enforce a {fraction:.1%} CPU limit on a "
            f"{logical_cpus}-CPU host: one CPU would exceed the limit.")
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError(
            "Cannot enforce the CPU cap: OS affinity controls are unavailable.")

    allowed_cpus = sorted(os.sched_getaffinity(0))
    selected_cpus = tuple(allowed_cpus[:cpu_limit])
    if not selected_cpus:
        raise RuntimeError("The process has no CPUs available in its affinity mask.")

    os.sched_setaffinity(0, selected_cpus)
    active_cpus = tuple(sorted(os.sched_getaffinity(0)))
    if active_cpus != selected_cpus:
        raise RuntimeError(
            "Failed to enforce the CPU affinity cap: requested "
            f"{selected_cpus}, active {active_cpus}.")

    thread_limit = len(active_cpus)
    for variable in CPU_THREAD_ENV_VARS:
        os.environ[variable] = str(thread_limit)
    return logical_cpus, active_cpus


# Apply the hard cap before importing autocvd, PyTorch, or another native
# thread-pool library. Pytest imports are capped as well as direct execution.
CPU_LIMIT_INFO = configure_cpu_usage_limit()
_logical_cpus, _active_cpus = CPU_LIMIT_INFO
print(
    "[setup] CPU limited to {} of {} logical CPUs ({:.2%} capacity)".format(
        len(_active_cpus), _logical_cpus, len(_active_cpus) / _logical_cpus),
    flush=True,
)

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from autocvd import autocvd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compass.ConditionTransformer import ConditionTransformer
from compass.PFODE import PFODE
from compass.SDE import VESDE, VPSDE
from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel


def test_cpu_usage_cap_is_active():
    logical_cpus, selected_cpus = CPU_LIMIT_INFO
    active_cpus = tuple(sorted(os.sched_getaffinity(0)))
    assert active_cpus == selected_cpus
    assert len(active_cpus) <= int(logical_cpus * CPU_USAGE_LIMIT_FRACTION)
    assert len(active_cpus) / logical_cpus <= CPU_USAGE_LIMIT_FRACTION
    for variable in CPU_THREAD_ENV_VARS:
        assert os.environ[variable] == str(len(active_cpus))


def _small_model(**kwargs):
    return ScoreBasedInferenceModel(
        nodes_size=4, hidden_size=8, depth=1, num_heads=1, mlp_ratio=1,
        **kwargs)


def test_transformer_legacy_and_optional_return_contracts():
    torch.manual_seed(0)
    model = ConditionTransformer(
        nodes_size=4, hidden_size=8, depth=1, num_heads=1, mlp_ratio=1)
    x = torch.randn(3, 4)
    t = torch.rand(3, 1)
    c = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ])

    legacy_score = model(x=x, t=t, c=c)
    score, drift = model(x=x, t=t, c=c, return_divergence=True)
    score_attn, attn = model(x=x, t=t, c=c, return_attn_weights=True)
    score_both, drift_both, attn_both = model(
        x=x, t=t, c=c, return_attn_weights=True,
        return_divergence=True)

    assert legacy_score.shape == (3, 4)
    assert drift.shape == (3,)
    assert torch.equal(legacy_score, score)
    assert torch.equal(legacy_score, score_attn)
    assert torch.equal(legacy_score, score_both)
    assert torch.equal(drift, drift_both)
    assert torch.count_nonzero(drift) == 0
    assert torch.equal(attn, attn_both)

    # Once the zero-initialized scalar projection receives a non-zero weight,
    # divergence gradients reach both the head and shared transformer backbone.
    with torch.no_grad():
        model.divergence_head.mlp[-1].weight.fill_(0.1)
    model.zero_grad(set_to_none=True)
    _, drift = model(x=x, t=t, c=c, return_divergence=True)
    drift.sum().backward()
    assert model.divergence_head.mlp[-1].weight.grad is not None
    assert model.x_embedder.embedding_params.grad is not None


def test_instantaneous_target_exact_hutchinson_and_masks():
    model = _small_model(sde_type="vpsde")
    trainer = model.trainer
    x = torch.randn(3, 4, requires_grad=True)
    t = torch.tensor([[0.2], [0.5], [0.8]])
    mask = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    diagonal = torch.tensor([1.5, -0.5, 2.0, 0.25])
    raw_score = x * diagonal
    alpha = model.sde.alpha_t(t).reshape(-1)
    expected = alpha * torch.tensor([1.0, -0.25, 0.0])

    exact = trainer.instantaneous_divergence_target(
        raw_score, x, t, mask, estimator="exact")
    hutchinson = trainer.instantaneous_divergence_target(
        raw_score, x, t, mask, estimator="hutchinson",
        hutchinson_samples=1)

    assert torch.allclose(exact, expected)
    # A diagonal Jacobian is estimated exactly by every Rademacher probe.
    assert torch.allclose(hutchinson, expected)
    prediction = torch.tensor([2.0, 1.0, 10.0], requires_grad=True)
    loss = trainer.divergence_loss_fn(prediction, expected, mask)
    manual = torch.mean(torch.stack([
        ((prediction[0] - expected[0]) / 2) ** 2,
        ((prediction[1] - expected[1]) / 2) ** 2,
    ]))
    assert torch.allclose(loss, manual)


class GaussianScoreWithDrift(torch.nn.Module):
    """Independent Gaussian score with an analytic learned drift head."""

    def __init__(self, sde, mean, variance):
        super().__init__()
        self.sde = sde
        self.mean = torch.as_tensor(mean)
        self.variance = torch.as_tensor(variance)
        self.forward_calls = 0

    def forward(self, x, t, c, return_attn_weights=False,
                return_divergence=False):
        self.forward_calls += 1
        sigma = self.sde.sigma_t(t).to(x.device)
        alpha = self.sde.alpha_t(t).to(x.device)
        variance_t = alpha**2 * self.variance.to(x.device) + sigma**2
        raw_score = -sigma * (
            x - alpha * self.mean.to(x.device)) / variance_t

        if return_divergence:
            assert not torch.is_grad_enabled()
            latent = 1 - c
            raw_diagonal = -sigma / variance_t
            drift = alpha.reshape(-1) * (
                raw_diagonal * latent).sum(dim=-1)
            return raw_score, drift
        if return_attn_weights:
            return raw_score, torch.zeros(1)
        return raw_score


class MockSBIm:
    def __init__(self, sde):
        self.sde = sde
        self.model = GaussianScoreWithDrift(
            sde, mean=torch.tensor([-1.0, 0.5, 2.0, -0.25]),
            variance=torch.tensor(0.4))
        self.divergence_head_trained = True


@torch.no_grad()
def _evaluation_data():
    points = torch.tensor([
        [-1.0, 0.5, 2.0, -0.25],
        [-0.5, 0.1, 2.0, -0.25],
        [-1.8, 1.0, 2.0, -0.25],
    ])
    return points, torch.tensor([0.0, 0.0, 1.0, 1.0])


def test_learned_pfode_matches_exact_for_ve_and_vp():
    data, mask = _evaluation_data()
    for sde in (VESDE(sigma=25.0), VPSDE()):
        sbim = MockSBIm(sde)
        pfode = PFODE(sbim)
        sbim.model.forward_calls = 0
        exact = pfode.log_prob(
            data, mask, timesteps=40, divergence="exact")
        exact_calls = sbim.model.forward_calls
        sbim.model.forward_calls = 0
        learned = pfode.log_prob(
            data, mask, timesteps=40, divergence="learned")
        learned_calls = sbim.model.forward_calls

        assert torch.allclose(learned, exact, atol=2e-5, rtol=2e-5)
        assert learned_calls == exact_calls


def test_learned_mode_requires_trained_head():
    model = _small_model(sde_type="vesde")
    data, mask = _evaluation_data()
    try:
        model.log_prob(data[:1], mask, timesteps=2, divergence="learned")
    except RuntimeError as exc:
        assert "has not been trained" in str(exc)
    else:
        raise AssertionError("An untrained divergence head must not be used silently.")


def test_legacy_and_new_checkpoint_loading(tmp_path):
    torch.manual_seed(4)
    model = _small_model(sde_type="vesde")
    x = torch.randn(2, 4)
    t = torch.rand(2, 1)
    c = torch.tensor([[0.0, 0.0, 1.0, 1.0]]).repeat(2, 1)
    expected_score = model.model(x=x, t=t, c=c).detach()

    legacy_state = {
        key: value for key, value in model.model.state_dict().items()
        if not key.startswith("divergence_head.")
    }
    legacy_checkpoint = {
        "model_state_dict": legacy_state,
        "nodes_size": model.nodes_size,
        "sde_type": model.sde_type,
        "sigma": model.sigma,
        "beta_min": model.beta_min,
        "beta_max": model.beta_max,
        "hidden_size": model.hidden_size,
        "depth": model.depth,
        "num_heads": model.num_heads,
        "mlp_ratio": model.mlp_ratio,
    }
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy_checkpoint, legacy_path)
    loaded_legacy = ScoreBasedInferenceModel.load(legacy_path, device="cpu")
    assert not loaded_legacy.divergence_head_trained
    assert torch.equal(
        loaded_legacy.model(x=x, t=t, c=c).detach(), expected_score)

    with torch.no_grad():
        model.model.divergence_head.mlp[-1].bias.fill_(1.25)
    model.divergence_head_trained = True
    model.save(tmp_path, name="new")
    loaded_new = ScoreBasedInferenceModel.load(
        tmp_path / "new.pt", device="cpu")
    _, expected_drift = model.model(
        x=x, t=t, c=c, return_divergence=True)
    _, loaded_drift = loaded_new.model(
        x=x, t=t, c=c, return_divergence=True)
    assert loaded_new.divergence_head_trained
    assert torch.equal(loaded_drift, expected_drift)


def test_legacy_trainer_batch_return_and_opt_in_head_update():
    torch.manual_seed(7)
    model = _small_model(sde_type="vesde")
    trainer = model.trainer
    trainer.device = "cpu"
    trainer.eps = 1e-3
    trainer.time_sampling = "uniform"
    trainer.model = model.model
    data = torch.randn(16, 4)
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]]).repeat(16, 1)
    batch = (data, mask, torch.arange(16))

    trainer.train_divergence = False
    trainer.divergence_loss_weight = 1.0
    trainer.divergence_warmup_epochs = 0
    legacy_loss = trainer._run_batch(batch)
    assert legacy_loss.ndim == 0

    trainer.train_divergence = True
    trainer.divergence_target = "exact"
    trainer.hutchinson_samples = 1
    # Mimic the intended warm-start workflow with a non-trivial trained score
    # projection; a freshly constructed COMPASS score head is exactly zero.
    with torch.no_grad():
        model.model.final_layer.embedding_params.normal_(std=0.1)
    before = model.model.divergence_head.mlp[-1].weight.detach().clone()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)
    torch.manual_seed(8)
    loss, _, divergence_loss = trainer._run_batch(
        batch, return_components=True)
    assert torch.isfinite(divergence_loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = model.model.divergence_head.mlp[-1].weight.detach()
    assert not torch.equal(after, before)


def test_public_training_api_is_opt_in(tmp_path):
    torch.manual_seed(9)
    theta = torch.randn(32, 2)
    observations = theta + 0.2 * torch.randn(32, 2)

    legacy = _small_model(sde_type="vesde")
    legacy.train(
        theta, observations, batch_size=16, max_epochs=1,
        early_stopping_patience=1, device="cpu", verbose=False,
        path=str(tmp_path), name="legacy_default")
    assert not legacy.divergence_head_trained
    assert legacy.trainer.train_divergence_loss == [0.0]

    joint = _small_model(sde_type="vesde")
    joint.train(
        theta, observations, batch_size=16, max_epochs=1,
        early_stopping_patience=1, device="cpu", verbose=False,
        path=str(tmp_path), name="joint", train_divergence=True,
        divergence_warmup_epochs=0)
    assert joint.divergence_head_trained
    assert len(joint.trainer.train_score_loss) == 1
    assert len(joint.trainer.train_divergence_loss) == 1
    restored = ScoreBasedInferenceModel.load(
        tmp_path / "joint_checkpoint.pt", device="cpu")
    assert restored.divergence_head_trained


if __name__ == "__main__":
    autocvd(num_gpus=1, interval=1)
    test_cpu_usage_cap_is_active()
    test_transformer_legacy_and_optional_return_contracts()
    test_instantaneous_target_exact_hutchinson_and_masks()
    test_learned_pfode_matches_exact_for_ve_and_vp()
    test_learned_mode_requires_trained_head()
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_legacy_and_new_checkpoint_loading(tmp_path)
        test_public_training_api_is_opt_in(tmp_path)
    test_legacy_trainer_batch_return_and_opt_in_head_update()
    print("All instantaneous-divergence tests passed.")
