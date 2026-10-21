"""``SBIm.train(mask_menu=...)``: choosing the training condition-mask law.

The six conditionals and ``ConditionalMenu`` already existed; what is new is that
picking between them and the Bernoulli draw is one keyword on ``train`` rather
than a hand-built sampler per benchmark.
"""
import pytest
import torch

import compass
from compass import ScoreBasedInferenceModel as SBIm


# [g(1), l(3), x(2)] -- the inverse-kinematics layout.
GLOBAL_SIZE, LOCAL_SIZE, OBSERVED = 1, 3, 2
NODES = GLOBAL_SIZE + LOCAL_SIZE + OBSERVED


def _model():
    torch.manual_seed(0)
    return SBIm(nodes_size=NODES, hidden_size=16, depth=2, num_heads=2, sigma=5.0)


def _resolve(model, **kwargs):
    defaults = dict(mask_sampler=None, mask_menu="bernoulli", mask_roles=None,
                    mask_weights=None, bernoulli_fraction=0.0,
                    bernoulli_probability=0.33)
    defaults.update(kwargs)
    return model._resolve_mask_sampler(
        defaults["mask_sampler"], defaults["mask_menu"], defaults["mask_roles"],
        defaults["mask_weights"], defaults["bernoulli_fraction"],
        defaults["bernoulli_probability"],
    )


def test_bernoulli_is_the_default_and_keeps_the_trainer_default():
    assert _resolve(_model()) is None


def test_six_builds_a_uniform_conditional_menu():
    menu = _resolve(_model(), mask_menu="six", mask_roles=(GLOBAL_SIZE, LOCAL_SIZE))
    assert isinstance(menu, compass.ConditionalMenu)
    assert menu.names == [name for name, _ in compass.SIX_CONDITIONALS]
    # Uniform: each of the six is drawn 1/6 of the time.
    assert torch.allclose(menu.weights, torch.full((6,), 1.0 / 6.0))
    # Menu-only by default: no Bernoulli fallback unless asked for.
    assert menu.bernoulli_fraction == 0.0


def test_six_draws_each_conditional_about_a_sixth_of_the_time():
    menu = _resolve(_model(), mask_menu="six", mask_roles=(GLOBAL_SIZE, LOCAL_SIZE))
    torch.manual_seed(0)
    drawn = menu.sample(torch.Size([60_000, NODES]))

    # The deployed posterior p(g,l|x): parameters latent, observations conditioned.
    posterior = torch.tensor([0.0] * (GLOBAL_SIZE + LOCAL_SIZE) + [1.0] * OBSERVED)
    share = (drawn == posterior).all(dim=-1).float().mean()
    assert share == pytest.approx(1.0 / 6.0, abs=0.01)


def test_six_lifts_the_posterior_mask_far_above_its_bernoulli_probability():
    """The point of the menu, stated as a test.

    Under Bernoulli(p) the deployed posterior mask has probability
    ``(1-p)^params * p^observations``, which falls off exponentially in the node
    count -- 2.2% at this 6-node layout and 6.5e-17 at partial pooling's 100.
    """
    p = 0.33
    bernoulli = (1 - p) ** (GLOBAL_SIZE + LOCAL_SIZE) * p ** OBSERVED
    assert bernoulli == pytest.approx(0.0219, abs=1e-4)
    assert (1.0 / 6.0) / bernoulli > 7.0


def test_weights_and_bernoulli_fraction_are_honoured():
    menu = _resolve(
        _model(), mask_menu="six", mask_roles=(GLOBAL_SIZE, LOCAL_SIZE),
        mask_weights=[6, 1, 1, 1, 1, 1], bernoulli_fraction=0.25,
    )
    assert menu.weights[0] == pytest.approx(6.0 / 11.0)
    assert menu.bernoulli_fraction == 0.25


def test_explicit_roles_dict_is_accepted():
    roles = {"g": [0], "l": [1, 2, 3], "x": [4, 5]}
    menu = _resolve(_model(), mask_menu="six", mask_roles=roles)
    assert menu.names == [name for name, _ in compass.SIX_CONDITIONALS]


def test_six_without_roles_is_refused():
    with pytest.raises(ValueError, match="needs mask_roles"):
        _resolve(_model(), mask_menu="six")


def test_unknown_menu_is_refused():
    with pytest.raises(ValueError, match="Unknown mask_menu"):
        _resolve(_model(), mask_menu="seven", mask_roles=(1, 3))


def test_mask_sampler_and_mask_menu_together_is_refused():
    sampler = _resolve(_model(), mask_menu="six", mask_roles=(GLOBAL_SIZE, LOCAL_SIZE))
    with pytest.raises(ValueError, match="not both"):
        _resolve(_model(), mask_sampler=sampler, mask_menu="six",
                 mask_roles=(GLOBAL_SIZE, LOCAL_SIZE))


def test_explicit_mask_sampler_still_works():
    sampler = _resolve(_model(), mask_menu="six", mask_roles=(GLOBAL_SIZE, LOCAL_SIZE))
    assert _resolve(_model(), mask_sampler=sampler) is sampler
