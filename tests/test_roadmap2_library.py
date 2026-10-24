"""Library changes of roadmap items 2.1-2.6: eps forwarding, the seeded validation
pass, final/best weight restore, the prior-score callable, and latent attention as a
constructor option. All CPU, tiny untrained networks: these check contracts, not
statistical accuracy."""

import pytest
import torch

from compass import (
    Atomic, GaussianPrior, ScoreBasedInferenceModel, ScorePrior, hierarchical_roles,
)

NODES, OBSERVED, HIERARCHY = 4, 2, [0]


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return ScoreBasedInferenceModel(nodes_size=NODES, sigma=5.0, hidden_size=8, depth=1,
                                    num_heads=2, mlp_ratio=2)


def observations(count=3):
    torch.manual_seed(1)
    return torch.randn(count, OBSERVED) * 0.5


# ----- 2.1: eps reaches the samplers -------------------------------------------------

@pytest.mark.parametrize("multi", [False, True])
def test_sample_forwards_eps(multi, monkeypatch):
    model = tiny_model()
    target = model.multi_obs_sampler if multi else model.sampler
    seen = {}
    original = target.sample

    def spy(*args, **kwargs):
        seen["eps"] = kwargs.get("eps")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, "sample", spy)
    kwargs = dict(multi_obs_inference=True, hierarchy=HIERARCHY, prior=([0.0], [1.0]),
                  correction="gauss_jacobian_FD") if multi else {}
    model.sample(x=observations(), timesteps=3, num_samples=2, eps=2e-4, verbose=False, **kwargs)
    assert seen["eps"] == 2e-4


def test_eps_changes_the_single_observation_samples():
    model = tiny_model()
    x = observations(1)
    out = []
    for eps in (1e-3, 1e-5):
        torch.manual_seed(3)
        out.append(model.sample(x=x, timesteps=4, num_samples=3, eps=eps, verbose=False))
    assert not torch.allclose(out[0], out[1])


# ----- 2.2 / 2.3: seeded validation pass, restore ------------------------------------

def _train(seed, tmp_path, with_validation=True, **kwargs):
    model = tiny_model()
    torch.manual_seed(5)
    theta, x = torch.randn(64, 2), torch.randn(64, 2)
    val = (torch.randn(32, 2), torch.randn(32, 2)) if with_validation else (None, None)
    torch.manual_seed(seed)
    model.train(theta, x, *val, batch_size=16, max_epochs=3, lr=kwargs.pop("lr", 1e-3),
                device="cpu", verbose=False, path=str(tmp_path), early_stopping_patience=100,
                **kwargs)
    return model


def test_seeded_validation_pass_is_identical_every_epoch(tmp_path):
    # lr=0: the weights never move, so any epoch-to-epoch change is the validation draw.
    seeded = _train(0, tmp_path, lr=0.0)
    assert len(set(seeded.trainer.val_loss)) == 1
    legacy = _train(0, tmp_path, lr=0.0, validation_noise_seed=None)
    assert len(set(legacy.trainer.val_loss)) == 3


def test_seeded_validation_does_not_consume_training_draws(tmp_path):
    with_val = _train(0, tmp_path / "a")
    without_val = _train(0, tmp_path / "b", with_validation=False)
    assert with_val.trainer.train_loss == without_val.trainer.train_loss


@pytest.mark.parametrize("restore", ["final", "best"])
def test_restore_selects_the_weights_kept(tmp_path, restore):
    model = _train(0, tmp_path, restore=restore)
    final = torch.load(tmp_path / "Model.pt")["model_state_dict"]
    best = torch.load(tmp_path / "Model_checkpoint.pt")["model_state_dict"]
    live = model.model.state_dict()
    expected = final if restore == "final" else best
    assert all(torch.equal(live[k].cpu(), expected[k].cpu()) for k in live)


def test_new_training_defaults():
    import inspect
    signature = inspect.signature(ScoreBasedInferenceModel.train).parameters
    assert signature["early_stopping_patience"].default == 100
    assert signature["restore"].default == "final"
    with pytest.raises(ValueError):
        _train(0, "/tmp/never", restore="latest")


# ----- 2.4: menu + EDM as the default pair -------------------------------------------

@pytest.mark.parametrize("kwargs, menu, weight", [
    ({}, False, None),                                              # no roles: unchanged
    ({"mask_roles": 2}, True, "edm"),                               # inference menu
    ({"mask_roles": (1, 1)}, True, "edm"),                          # six menu
    ({"mask_roles": 2, "time_loss_weight": None}, True, None),      # explicit opt-out
    ({"mask_menu": "bernoulli", "mask_roles": 2}, False, None),     # explicit Bernoulli
])
def test_auto_menu_and_edm_defaults(kwargs, menu, weight, monkeypatch):
    model, seen = tiny_model(), {}
    monkeypatch.setattr(model.trainer, "train", lambda **kw: seen.update(kw))
    model.train(torch.randn(8, 2), torch.randn(8, 2), device="cpu", verbose=False, **kwargs)
    assert (seen["mask_sampler"] is not None) == menu
    assert seen["time_loss_weight"] == weight


def test_explicit_sampler_gets_edm(monkeypatch):
    model, seen = tiny_model(), {}
    sampler = model._resolve_mask_sampler(None, "inference", 2, None, 0.0, 0.33)
    monkeypatch.setattr(model.trainer, "train", lambda **kw: seen.update(kw))
    model.train(torch.randn(8, 2), torch.randn(8, 2), device="cpu", verbose=False, mask_sampler=sampler)
    assert seen["mask_sampler"] is sampler and seen["time_loss_weight"] == "edm"


# ----- 2.5: prior-score callable -----------------------------------------------------

def _compose(prior, correction="gauss_jacobian_FD"):
    model = tiny_model()
    torch.manual_seed(11)
    return model.sample(x=observations(), multi_obs_inference=True, hierarchy=HIERARCHY,
                        prior=prior, correction=correction, timesteps=4, num_samples=5,
                        verbose=False)


@pytest.mark.parametrize("correction", ["gauss_jacobian_FD", "gauss_hierarchical"])
def test_score_callable_reproduces_the_tuple_prior(correction):
    mean, std = torch.tensor([0.3]), torch.tensor([1.7])
    gaussian = GaussianPrior(mean, std.square())
    two_arg = ScorePrior(lambda theta_y, lam: gaussian.score(theta_y, None, lam),
                         mean=mean, covariance=std.square())
    three_arg = ScorePrior(gaussian.score, mean=mean, covariance=std.square())
    reference = _compose((mean.tolist(), std.tolist()), correction)
    for prior in (two_arg, three_arg):
        assert torch.allclose(_compose(prior, correction), reference, atol=1e-5)


def test_score_callable_is_what_the_composition_reads():
    calls = []

    def score(theta_y, lam):
        calls.append(float(lam))
        return -theta_y / (1.0 + lam ** 2)

    _compose(ScorePrior(score, mean=[0.0], covariance=[1.0]))
    assert calls and len(set(calls)) > 1


def test_bare_callable_and_bad_shapes_are_refused():
    with pytest.raises(ValueError, match="ScorePrior"):
        _compose(lambda theta_y, lam: -theta_y)
    with pytest.raises(ValueError, match="shape"):
        _compose(ScorePrior(lambda theta_y, lam: theta_y.sum(-1), mean=[0.0], covariance=[1.0]))


def test_score_prior_from_samples_moments():
    torch.manual_seed(0)
    draws = torch.randn(20000, 2) * torch.tensor([1.0, 2.0]) + torch.tensor([3.0, -1.0])
    prior = ScorePrior.from_samples(lambda y, lam: -y, draws)
    assert torch.allclose(prior.mean, torch.tensor([3.0, -1.0], dtype=torch.float64), atol=0.05)
    assert torch.allclose(torch.diagonal(prior.covariance),
                          torch.tensor([1.0, 4.0], dtype=torch.float64), rtol=0.05)


# ----- 2.6: latent attention as a library option ------------------------------------

G, L = 2, 2


def tokenized(mode="hierarchical", seed=0):
    torch.manual_seed(seed)
    model = ScoreBasedInferenceModel(theta_size=G + L, x_spec=[Atomic("x", 3)], arch="tokenized",
                                     hidden_size=16, depth=2, num_heads=2, mlp_ratio=2,
                                     latent_attention=mode, latent_roles=(G, L))
    for module in model.model.modules():                 # wake the adaLN-Zero trunk
        if hasattr(module, "adaLN_modulation"):
            torch.nn.init.normal_(module.adaLN_modulation[-1].weight, std=0.3)
    return model


def _outputs(model):
    torch.manual_seed(4)
    x, t = torch.randn(6, model.nodes_size), torch.rand(6, 1)
    c = torch.zeros(6, model.nodes_size)
    c[:, G + L:] = 1
    return model.model(x=x, t=t, c=c)


def test_constructor_matches_the_experimental_switch():
    from compass.experimental.latent_attention import enable_latent_attention
    built = tokenized("hierarchical")
    switched = tokenized("none")
    enable_latent_attention(switched, "hierarchical", hierarchical_roles(switched.schema, G, L))
    assert torch.allclose(_outputs(built), _outputs(switched))
    assert not torch.allclose(_outputs(built), _outputs(tokenized("none")))


def test_latent_attention_round_trips_through_the_checkpoint(tmp_path):
    model = tokenized("hierarchical")
    model.save(str(tmp_path))
    payload = torch.load(tmp_path / "Model.pt")
    assert payload["arch_version"] == 2 and payload["latent_attention"] == "hierarchical"
    loaded = ScoreBasedInferenceModel.load(str(tmp_path / "Model.pt"), device="cpu")
    assert loaded.model.latent_attention == "hierarchical"
    assert torch.allclose(_outputs(loaded), _outputs(model))


def test_plain_checkpoints_keep_arch_version_1(tmp_path):
    tokenized("none").save(str(tmp_path))
    payload = torch.load(tmp_path / "Model.pt")
    assert payload["arch_version"] == 1 and "latent_attention" not in payload


def test_latent_attention_argument_errors():
    with pytest.raises(ValueError, match="latent_roles"):
        ScoreBasedInferenceModel(theta_size=4, x_spec=[Atomic("x", 3)], arch="tokenized",
                                 hidden_size=8, depth=1, num_heads=2, latent_attention="global")
    with pytest.raises(ValueError, match="trunk"):
        ScoreBasedInferenceModel(nodes_size=7, hidden_size=8, depth=1, num_heads=2,
                                 latent_attention="global", latent_roles=(2, 2))
