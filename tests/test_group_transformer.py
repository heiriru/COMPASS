"""The two-level trunk: the symmetry it claims, the one it refuses, and drop-in.

Two things are under test, and they are different in kind.

**Symmetry.** ``Group`` claims exactly one invariance -- trials permuted within
one repeat -- and explicitly declines to promise anything about permuting the
repeats. Both halves are asserted: the first must hold, and under the default
``slot="learned"`` the second must *not*, because a slot embedding that changed
nothing would be a silent promise of a symmetry the schema does not make. The
``slot="none"`` ablation is equivariant and is asserted to be so, so that
whichever mode is configured, the test says which.

**Drop-in.** The diffusion state stays a flat ``(batch, nodes_size)`` vector, so
``Sampler``, ``PFODE`` and the ~3.9k-line ``MultiObsSampler`` should need no
changes at all. This file is the verification of that claim rather than the
assumption of it, mirroring ``tests/test_tokenized_stack_integration.py`` one
level of nesting further down -- including the two paths the benchmark actually
runs, ``gauss_jacobian`` composition and ``newton_map_estimate``, both of which
push forward-mode AD through the new trunk.
"""

import pytest
import torch

torch.set_num_threads(1)

from compass import Atomic, Group, IID, NodeSchema, ScoreBasedInferenceModel
from compass.GroupedTransformer import grouped_parameter_breakdown

GLOBALS, LOCALS, TRIALS, ITEM = 7, 3, 4, 3
SUBJECTS = 3
INNER_WIDTH = LOCALS + TRIALS * ITEM
INNER_TOKENS = LOCALS + TRIALS
NODES = GLOBALS + SUBJECTS * INNER_WIDTH
TOKENS = GLOBALS + SUBJECTS * INNER_TOKENS


def wake(model, seed=11):
    """Give the adaLN modulation the non-zero weights a trained model has.

    Every block is adaLN-**Zero**: ``adaLN_modulation[-1]`` is zero-initialized,
    so ``gate_msa`` and ``gate_mlp`` are zero and each block is exactly the
    identity at construction. A freshly built trunk therefore ignores attention
    completely, and every "this input must change the output" assertion below
    would pass vacuously -- while every "this must not change" assertion would
    pass for the wrong reason. The properties under test belong to the
    architecture, not to the initialization, so the modulation is woken first.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for block in model.model.blocks:
            weight = block.adaLN_modulation[-1].weight
            weight.copy_(torch.empty_like(weight).normal_(0, 0.05, generator=generator))
    return model


def build(slot="learned", pool="mean", seed=0, woken=True, **kwargs):
    torch.manual_seed(seed)
    spec = [Group("subjects", SUBJECTS, [
        Atomic("local", LOCALS), IID("trials", ITEM, TRIALS),
    ], slot=slot, pool=pool)]
    model = ScoreBasedInferenceModel(
        theta_size=GLOBALS, x_spec=spec, hidden_size=8, num_heads=2,
        mlp_ratio=2, depth=2, device="cpu", **kwargs,
    )
    model.model.eval()
    return wake(model) if woken else model


@pytest.fixture(scope="module")
def model():
    return build()


@pytest.fixture(scope="module")
def observations():
    generator = torch.Generator().manual_seed(1)
    return torch.randn(4, SUBJECTS * TRIALS * ITEM, generator=generator)


def inputs(model, batch=4, seed=2):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, NODES, generator=generator)
    t = torch.rand(batch, 1, generator=generator) * 0.9 + 0.05
    c = torch.zeros(batch, NODES)
    for repeat in range(SUBJECTS):                    # deployed posterior mask
        base = GLOBALS + repeat * INNER_WIDTH
        c[:, base + LOCALS:base + INNER_WIDTH] = 1.0
    return x, t, c


def posterior_mask():
    """Theta latent, trials conditioned -- built explicitly.

    There is no contiguous ``[theta, x]`` split to infer this from: each
    repeat's latent locals sit *inside* its observed trials. ``sample()``
    refuses to guess for a grouped schema, which is what
    ``test_sample_refuses_to_guess_a_nested_mask`` pins.
    """
    mask = torch.zeros(NODES)
    for repeat in range(SUBJECTS):
        mask[trial_slice(repeat)] = 1.0
    return mask


def joint(observations):
    """Full-width node vectors: zeros on the latent theta, data on the trials.

    ``log_prob`` and ``newton_map_estimate`` take the whole node vector.
    ``sample`` does **not** -- it fills ``joint_data[condition_mask == 1]`` from
    a flat ``data``, so there it is the observed values alone that get passed,
    with the full-width mask alongside.
    """
    batch = observations.shape[0]
    out = torch.zeros(batch, NODES)
    for repeat in range(SUBJECTS):
        source = observations[:, repeat * TRIALS * ITEM:(repeat + 1) * TRIALS * ITEM]
        out[:, trial_slice(repeat)] = source
    return out


def repeat_slice(repeat):
    base = GLOBALS + repeat * INNER_WIDTH
    return slice(base, base + INNER_WIDTH)


def trial_slice(repeat):
    base = GLOBALS + repeat * INNER_WIDTH + LOCALS
    return slice(base, base + TRIALS * ITEM)


#############################################
# ----- Shapes, budget, dispatch -----
#############################################

def test_shapes_and_dispatch(model):
    assert model.arch == "grouped"
    assert model.nodes_size == NODES
    assert model.token_count == TOKENS
    x, t, c = inputs(model)
    with torch.no_grad():
        out = model.model(x=x, t=t, c=c)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_all_latent_input_stays_finite(model):
    """No attention key at all; survives only via add_bias_kv."""
    x, t, _ = inputs(model)
    with torch.no_grad():
        out = model.model(x=x, t=t, c=torch.zeros_like(x))
    assert torch.isfinite(out).all()


def test_the_parameter_count_is_nearly_independent_of_k():
    """So a K sweep measures the representation, not the capacity.

    Only the ``K x hidden_size`` slot table varies with K; every trunk weight is
    shared across repeats.
    """
    counts = {}
    for subjects in (2, 8, 32):
        torch.manual_seed(0)
        model = ScoreBasedInferenceModel(
            theta_size=GLOBALS, hidden_size=8, num_heads=2, mlp_ratio=2, depth=2,
            x_spec=[Group("subjects", subjects, [
                Atomic("local", LOCALS), IID("trials", ITEM, TRIALS)])],
            device="cpu",
        )
        counts[subjects] = sum(p.numel() for p in model.model.parameters())
    assert counts[32] - counts[2] == (32 - 2) * 8      # exactly the slot table
    assert (counts[32] - counts[2]) / counts[2] < 0.05


def test_breakdown_accounts_for_every_parameter(model):
    breakdown = grouped_parameter_breakdown(model.model)
    assert breakdown["unaccounted"] == 0
    assert breakdown["total"] == sum(p.numel() for p in model.model.parameters())


#############################################
# ----- The symmetry that IS claimed -----
#############################################

def test_permuting_trials_within_a_repeat_permutes_only_its_trial_scores(model):
    x, t, c = inputs(model)
    order = torch.tensor([2, 0, 3, 1])
    permuted = x.clone()
    view = x[:, trial_slice(1)].reshape(-1, TRIALS, ITEM)
    permuted[:, trial_slice(1)] = view[:, order].reshape(-1, TRIALS * ITEM)

    with torch.no_grad():
        base = model.model(x=x, t=t, c=c)
        after = model.model(x=permuted, t=t, c=c)

    # Repeat 1's trial scores permute with their trials...
    expected = base[:, trial_slice(1)].reshape(-1, TRIALS, ITEM)[:, order]
    assert torch.allclose(
        after[:, trial_slice(1)].reshape(-1, TRIALS, ITEM), expected, atol=1e-6)
    # ...and nothing else moves at all.
    assert torch.allclose(after[:, :GLOBALS], base[:, :GLOBALS], atol=1e-6)
    for repeat in (0, 2):
        assert torch.allclose(
            after[:, repeat_slice(repeat)], base[:, repeat_slice(repeat)], atol=1e-6)


def test_the_nesting_is_load_bearing(model):
    """Move one repeat's trials under another's locals: the scores must change.

    A trunk that ignored the nesting -- that treated the whole thing as one flat
    bag of trials -- would pass every other test in this file.
    """
    x, t, c = inputs(model)
    swapped = x.clone()
    swapped[:, trial_slice(0)] = x[:, trial_slice(1)]
    swapped[:, trial_slice(1)] = x[:, trial_slice(0)]
    with torch.no_grad():
        base = model.model(x=x, t=t, c=c)
        after = model.model(x=swapped, t=t, c=c)
    assert not torch.allclose(after[:, :GLOBALS], base[:, :GLOBALS], atol=1e-6)


#############################################
# ----- The symmetry that is NOT claimed -----
#############################################

def permute_subjects(x, order):
    out = x.clone()
    for target, source in enumerate(order):
        out[:, repeat_slice(target)] = x[:, repeat_slice(source)]
    return out


def test_learned_slots_break_subject_permutation(model):
    """The default promises no subject symmetry, and must actually not have it."""
    x, t, c = inputs(model)
    order = [1, 2, 0]
    with torch.no_grad():
        base = model.model(x=x, t=t, c=c)
        after = model.model(x=permute_subjects(x, order), t=t, c=c)
    expected = permute_subjects(base, order)
    assert not torch.allclose(after[:, :GLOBALS], base[:, :GLOBALS], atol=1e-6)
    assert not torch.allclose(after, expected, atol=1e-6)


def test_slot_embedding_is_not_zero_at_initialization():
    """A zero table would leave the trunk equivariant at step 0, i.e. silently
    asserting the symmetry the schema declines to promise."""
    fresh = build(woken=False)
    assert fresh.model.slot_embedding.abs().max() > 0


def test_slot_none_is_equivariant_and_that_is_the_ablation():
    ablation = build(slot="none")
    x, t, c = inputs(ablation)
    order = [1, 2, 0]
    with torch.no_grad():
        base = ablation.model(x=x, t=t, c=c)
        after = ablation.model(x=permute_subjects(x, order), t=t, c=c)
    assert torch.allclose(after, permute_subjects(base, order), atol=1e-6)
    assert torch.allclose(after[:, :GLOBALS], base[:, :GLOBALS], atol=1e-6)


def test_k1_is_not_asserted_equal_to_the_flat_trunk():
    """Deliberately weaker than an equality check -- see the plan's tier 3.2.

    The topologies differ: the flat trunk is one transformer over
    ``[globals, locals, trials]``, while K=1 is ``within -> pool -> across ->
    broadcast``. The pooling bottleneck alone makes bit-identity unavailable, so
    what is asserted is that K=1 is *well formed*: right shapes, right masking,
    and the inner invariance intact.
    """
    torch.manual_seed(0)
    grouped = ScoreBasedInferenceModel(
        theta_size=GLOBALS, hidden_size=8, num_heads=2, mlp_ratio=2, depth=2,
        x_spec=[Group("subjects", 1, [
            Atomic("local", LOCALS), IID("trials", ITEM, TRIALS)])],
        device="cpu",
    )
    grouped.model.eval()
    assert grouped.nodes_size == GLOBALS + INNER_WIDTH
    assert grouped.token_count == GLOBALS + INNER_TOKENS
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(2, grouped.nodes_size, generator=generator)
    t = torch.rand(2, 1, generator=generator) * 0.9 + 0.05
    c = torch.cat([torch.zeros(2, GLOBALS + LOCALS), torch.ones(2, TRIALS * ITEM)], 1)
    with torch.no_grad():
        out = grouped.model(x=x, t=t, c=c)
    assert out.shape == x.shape and torch.isfinite(out).all()


#############################################
# ----- Masking -----
#############################################

def test_a_subject_is_a_key_if_any_of_its_tokens_is():
    """``min`` would make every subject a non-key under the deployed posterior
    mask (locals latent), so the globals would attend to nothing."""
    model = build()
    x, t, c = inputs(model)
    inner_mask = model.schema.token_mask(c)[:, GLOBALS:].reshape(-1, INNER_TOKENS)
    subject_mask = model.model._subject_key_mask(inner_mask)
    assert inner_mask.min(dim=-1).values.sum() == 0     # every subject has latent locals
    assert subject_mask.min() == 1.0                    # yet every subject is a key


def test_a_partially_observed_trial_is_latent_as_a_whole(model):
    x, t, c = inputs(model)
    partial = c.clone()
    partial[:, GLOBALS + LOCALS] = 0.0                  # one scalar of trial 0
    with torch.no_grad():
        assert not torch.allclose(
            model.model(x=x, t=t, c=partial), model.model(x=x, t=t, c=c), atol=1e-6)
    token = model.schema.token_mask(partial)
    assert token[0, GLOBALS + LOCALS] == 0.0


#############################################
# ----- Drop-in for every consumer of the score -----
#############################################

def test_sample_refuses_to_guess_a_nested_mask(model, observations):
    """The [theta, x] convenience split does not exist for a nested schema."""
    with pytest.raises(ValueError, match="no contiguous"):
        model.sample(x=observations, timesteps=4, num_samples=2, device="cpu",
                     verbose=False)


def test_single_observation_sampler(model, observations):
    torch.manual_seed(4)
    draws = model.sample(x=observations, condition_mask=posterior_mask(),
                         timesteps=8, num_samples=4, method="dpm",
                         device="cpu", verbose=False, capture_attention=False)
    # Every latent node comes back: the globals and every repeat's locals.
    assert draws.shape[-1] == GLOBALS + SUBJECTS * LOCALS
    assert torch.isfinite(draws).all()


def test_attention_capture(model, observations):
    torch.manual_seed(4)
    model.sample(
        x=observations, condition_mask=posterior_mask(), timesteps=8,
        num_samples=4, method="dpm", device="cpu",
        verbose=False, capture_attention=True,
    )
    weights = model.sampler.all_attn_weights
    assert len(weights) > 0, "no attention weights captured"
    # The across-subject level is the one worth capturing: this is where the
    # group-level parameters attend to several subjects at once, which is the
    # whole hypothesis. One extra key column comes from add_bias_kv.
    assert weights[0].shape[-2:] == (GLOBALS + SUBJECTS, GLOBALS + SUBJECTS + 1)


def test_pfode_log_prob(model, observations):
    """The likelihood layout: every parameter conditioned, trials latent."""
    condition_mask = torch.ones(NODES)
    for repeat in range(SUBJECTS):
        condition_mask[trial_slice(repeat)] = 0.0
    data = joint(observations) + 0.1
    values = model.log_prob(data=data, condition_mask=condition_mask, timesteps=8,
                            divergence="exact", device="cpu", verbose=False)
    assert values.shape == (4,)
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("correction, method", [
    ("gauss_jacobian", "dpm"),
    ("gauss_hierarchical", "dpm"),
    ("fnpe", "langevin"),
])
def test_multi_observation_composition(model, observations, correction, method):
    """Composition is over *groups* now; the identity is unchanged."""
    torch.manual_seed(5)
    draws = model.sample(
        x=observations, condition_mask=posterior_mask(),
        timesteps=8, num_samples=4, method=method, order=2,
        multi_obs_inference=True, hierarchy=list(range(GLOBALS)),
        correction=correction, jacobian_refresh=20,
        precision_est_samples=32, precision_est_timesteps=8,
        corrector_steps=0, final_corrector_steps=0, device="cpu", verbose=False,
    )
    assert draws.shape[-1] == GLOBALS + SUBJECTS * LOCALS
    assert torch.isfinite(draws).all()


def test_arrow_newton_map_with_jacobian_curvature(model, observations):
    """Forward-mode AD through the pool, the two trunks and the broadcast."""
    locals_count = SUBJECTS * LOCALS
    condition_mask = torch.zeros(NODES)
    for repeat in range(SUBJECTS):
        condition_mask[trial_slice(repeat)] = 1.0
    start = joint(observations)
    refined = model.multi_obs_sampler.newton_map_estimate(
        data=start, condition_mask=condition_mask, init=start,
        hierarchy=list(range(GLOBALS)),
        prior=([0.0] * GLOBALS, [1.0] * GLOBALS),
        local_prior=([0.0] * locals_count, [1.0] * locals_count),
        correction="gauss_jacobian", curvature="jacobian",
        timesteps=8, device="cpu",
    )
    refined = refined.reshape(start.shape)
    assert torch.isfinite(refined).all()
    # Conditioned trial features must come back untouched.
    for repeat in range(SUBJECTS):
        assert torch.allclose(
            refined[:, trial_slice(repeat)], start[:, trial_slice(repeat)], atol=1e-6)


def test_training_runs_and_round_trips(tmp_path):
    """One short training run: nested masks, checkpointing, reload.

    ``train`` concatenates ``[theta, x]``, which is still exactly right here:
    ``theta`` is the group-level block and ``x`` is everything the Group covers,
    repeats' own latent locals included. The training condition mask comes from
    the mask sampler, not from this split.
    """
    torch.manual_seed(6)
    model = build(seed=6, woken=False)
    generator = torch.Generator().manual_seed(7)
    theta = torch.randn(32, GLOBALS, generator=generator)
    x = torch.randn(32, SUBJECTS * INNER_WIDTH, generator=generator)
    model.train(
        theta=theta, x=x, theta_val=theta[:8], x_val=x[:8], max_epochs=1,
        batch_size=8, path=str(tmp_path), name="grouped", device="cpu",
        verbose=False,
    )
    saved = tmp_path / "grouped_checkpoint.pt"
    assert saved.exists()
    reloaded = ScoreBasedInferenceModel.load(str(saved), device="cpu")
    assert reloaded.arch == "grouped"
    assert reloaded.schema == model.schema
    assert reloaded.depth_within == model.depth_within
    assert reloaded.depth_across == model.depth_across
