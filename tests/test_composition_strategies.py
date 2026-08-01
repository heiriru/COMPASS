from types import SimpleNamespace
import pytest
import torch

from compass.MultiObsSampler import MultiObsSampler
from compass.SDE import VESDE, VPSDE


def sampler(correction, sde=None, n=4, batch=None):
    instance = MultiObsSampler(SimpleNamespace(sde=sde or VESDE(), model=None))
    instance.hierarchy = [0]
    instance.correction = correction
    instance.prior_mean = torch.tensor([0.25])
    instance.prior_std = torch.tensor([1.5])
    instance.denoise_clamp = None
    instance.num_observations = n
    instance.composition_batch_size = n if batch is None else batch
    instance.damping_at_data = 1.0
    instance.damping_at_noise = n ** -0.5
    instance.world_size = 1
    instance.score_network_calls = 0
    instance.evaluated_subject_rows = 0
    return instance


@pytest.mark.parametrize("name", ["legacy_mean", "prior_corrected_sum", "damped_sum", "minibatch_damped"])
def test_r1_required_names_reduce_to_single_score(name):
    instance = sampler(name, n=1)
    scores = torch.tensor([[[2.0]]])
    x = torch.tensor([[[0.5]]])
    result = instance._compositional_score(scores.clone(), x, torch.tensor([[0.2]]))
    torch.testing.assert_close(result, scores)


def test_exact_required_formulas():
    scores = torch.tensor([[[1.]], [[2.]], [[3.]], [[4.]]])
    x = torch.zeros(4, 1, 1)
    t = torch.tensor([[0.3]])
    base = sampler("prior_corrected_sum")
    prior = base._diffused_gaussian_prior_score(x[:1], t)
    expected = (1 - 4) * prior + scores.sum(0, keepdim=True)
    torch.testing.assert_close(base._compositional_score(scores.clone(), x, t)[:1], expected)
    legacy = sampler("legacy_mean")
    torch.testing.assert_close(legacy._compositional_score(scores.clone(), x, t)[:1], prior + scores.mean(0, keepdim=True))
    damped = sampler("damped_sum")
    torch.testing.assert_close(damped._compositional_score(scores.clone(), x, t)[:1], expected / 4)


def test_minibatch_damped_has_full_score_expectation():
    full_scores = torch.tensor([1.0, 2.0, 5.0, 8.0]).reshape(4, 1, 1)
    x = torch.zeros(4, 1, 1)
    t = torch.tensor([[0.3]])
    full = sampler("damped_sum")
    target = full._compositional_score(full_scores.clone(), x, t)[0, 0, 0]
    estimates = []
    for left in range(4):
        for right in range(left + 1, 4):
            mini = sampler("minibatch_damped", n=4, batch=2)
            selected = full_scores[[left, right]]
            estimates.append(mini._compositional_score(
                selected.clone(), x, t, num_observations=4,
                minibatch_selected=True,
            )[0, 0, 0])
    torch.testing.assert_close(torch.stack(estimates).mean(), target)


def test_composition_changes_only_declared_global_coordinates():
    instance = sampler("prior_corrected_sum", n=3)
    scores = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    x = torch.zeros_like(scores)
    result = instance._compositional_score(scores.clone(), x, torch.tensor([[0.4]]))
    torch.testing.assert_close(result[:, :, 1:], scores[:, :, 1:])
    torch.testing.assert_close(result[:, :, :1], result[:1, :, :1].expand(3, -1, -1))


@pytest.mark.parametrize("sde", [VESDE(), VPSDE()])
@pytest.mark.parametrize("name", ["legacy_mean", "prior_corrected_sum", "damped_sum", "minibatch_damped"])
def test_required_methods_are_finite_across_diffusion_time(name, sde):
    instance = sampler(name, sde=sde, n=4)
    scores = torch.randn(4, 3, 1)
    x = torch.randn(4, 3, 1)
    for time in (1e-3, 0.1, 0.5, 1.0):
        result = instance._compositional_score(
            scores.clone(), x, torch.tensor([[time]]), num_observations=4,
            minibatch_selected=name == "minibatch_damped",
        )
        assert torch.isfinite(result).all()


@pytest.mark.parametrize("sde", [VESDE(), VPSDE()])
def test_diffused_prior_score_matches_analytic_ve_vp(sde):
    instance = sampler("prior_corrected_sum", sde=sde)
    y = torch.tensor([[[0.7]]])
    t = torch.tensor([[0.4]])
    alpha = sde.alpha_t(t)
    sigma = sde.sigma_t(t)
    x_t = alpha * y
    score_x = -(x_t - alpha * instance.prior_mean) / (alpha**2 * instance.prior_std**2 + sigma**2)
    torch.testing.assert_close(instance._diffused_gaussian_prior_score(y, t), alpha * score_x)


def test_minibatch_is_selected_before_network_forward_and_counted():
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.rows = []
        def forward(self, x, t, c): self.rows.append(x.shape[0]); return torch.ones_like(x)
    model = Model()
    sbim = SimpleNamespace(sde=VESDE(), model=model, output_scale_function=lambda t, x: x)
    instance = MultiObsSampler(sbim)
    instance.hierarchy = [0, 1]
    instance.correction = "minibatch_damped"
    instance.prior_mean = torch.zeros(2)
    instance.prior_std = torch.ones(2)
    instance.denoise_clamp = None
    instance.composition_batch_size = 3
    instance.world_size = 1
    instance.score_network_calls = 0
    instance.evaluated_subject_rows = 0
    result = instance._get_score(torch.zeros(8, 5, 2), torch.tensor([[0.4]]), torch.zeros(8, 5, 2), torch.arange(8))
    assert model.rows == [15]
    assert instance.score_network_calls == 1
    assert instance.evaluated_subject_rows == 3
    assert result.shape == (8, 5, 2)


def test_true_minibatch_rejects_row_specific_latents_before_forward():
    instance = sampler("minibatch_damped", n=4, batch=2)
    with pytest.raises(ValueError, match="every latent coordinate is shared"):
        instance.sample(
            world_size=1, data=torch.zeros(4, 1),
            condition_mask=torch.tensor([0., 0., 1.]), hierarchy=[0],
            correction="minibatch_damped", composition_batch_size=2,
            timesteps=2, num_samples=1,
        )
