"""Tests for set_pd_strategy: how the composed precision Lambda_S is kept PD.

Every reference here is computed independently of the sampler, from the exact
Gaussian row posteriors N(mu_j, Sigma_j) of ``LinearGaussianScore``:

- "row_projection" is the default and must not change (golden draws in
  pd_study/golden.py; here: row_projection_post reproduces it).
- "composed_check" counts samples whose raw Lambda_S is not PD.
- "gloeckler" equals a line-by-line port of markovsbi's CorrectedScoreFn
  (Gloeckler et al. 2025) with the exact row precisions.
- "row_projection_consistent" equals GAUSS composed from the *modified*
  Gaussians N(mu_j, (K_j + Delta_j)^-1): the lifted precision together with
  the score of that same lifted Gaussian.
"""
import numpy as np
import pytest
import torch

from compass.MultiObsSampler import MultiObsSampler
from test_gauss_jacobian_composition import (
    SDE, TIME, VARIANCE, LinearGaussianScore, build_state, configure, make_sbim,
)

HIERARCHY = [0, 1]
NODES = 3


def wide_setup(observations=6, locals_=(), seed=5, widen=10.0, narrow=1.0):
    """Row posteriors over the globals, some wider than the N(0, I) prior."""
    features = HIERARCHY + list(locals_)
    d = len(features)
    generator = torch.Generator().manual_seed(seed)
    covariance = []
    for j in range(observations):
        root = torch.randn(d, d, generator=generator, dtype=torch.float64)
        c = root @ root.mT / d + 0.2 * torch.eye(d, dtype=torch.float64)
        covariance.append(c * (widen if j % 2 == 0 else narrow))
    covariance = torch.stack(covariance)
    means = torch.randn(observations, d, generator=generator, dtype=torch.float64) * 0.5
    model = LinearGaussianScore(features, means, covariance, observations, SDE)
    return {"features": features, "covariance": covariance, "means": means, "model": model,
            "nodes": NODES + len(locals_), "observations": observations,
            "hierarchy": HIERARCHY, "locals": list(locals_)}


def sampler_for(setup, strategy, fallback="repair", nugget=None):
    sampler = MultiObsSampler(make_sbim(setup["model"]))
    configure(sampler, "gauss_jacobian", setup["hierarchy"], setup["locals"],
              setup["observations"])
    sampler.set_pd_strategy(strategy, fallback, nugget)
    sampler._reset_covariance_diagnostics()
    return sampler


def compose(setup, strategy, fallback="repair", nugget=None, num_samples=6):
    state, mask = build_state(setup, num_samples=num_samples)
    sampler = sampler_for(setup, strategy, fallback, nugget)
    scores = setup["model"](state.reshape(-1, setup["nodes"]), TIME,
                            mask.reshape(-1, setup["nodes"])).reshape(state.shape)
    out = sampler._compositional_score(scores.clone(), state, TIME, condition_mask=mask)
    return out[0, :, HIERARCHY].to(torch.float64), state, sampler


def gaussian_terms(setup, state):
    """Exact per-row K_j, mu_j, Lambda_j(t), s_j(y) on the globals (no locals)."""
    y = state[0, :, HIERARCHY].to(torch.float64)                      # (S, H)
    sigma = setup["covariance"]                                       # (n, H, H)
    identity = torch.eye(len(HIERARCHY), dtype=torch.float64)
    k = torch.linalg.inv(sigma)
    lam = k + identity / VARIANCE
    s = -torch.einsum("nij,nsj->nsi", torch.linalg.inv(sigma + VARIANCE * identity),
                      y[None] - setup["means"][:, None])
    prior_lam = identity + identity / VARIANCE
    prior_s = -y / (1.0 + VARIANCE)
    return y, k, lam, s, prior_lam, prior_s


def gauss(lams, scores, prior_lam, prior_s):
    n = lams.shape[0]
    precision = lams.sum(0) + (1 - n) * prior_lam
    numerator = torch.einsum("nij,nsj->si", lams, scores) + (1 - n) * prior_s @ prior_lam.T
    return torch.linalg.solve(precision, numerator.T).T


def test_setup_has_indefinite_information_and_nonpd_lambda_s():
    setup = wide_setup()
    _, k, lam, _, prior_lam, _ = gaussian_terms(setup, build_state(setup)[0])
    information = k - torch.eye(2, dtype=torch.float64)
    assert (torch.linalg.eigvalsh(information)[:, 0] < 0).any()
    n = setup["observations"]
    lambda_s = lam.sum(0) + (1 - n) * prior_lam
    assert torch.linalg.eigvalsh(lambda_s)[0] < 0


def test_default_strategy_is_the_projected_rule():
    """row_projection == GAUSS with I_j -> I_j^+ and the unmodified row scores."""
    setup = wide_setup()
    out, state, _ = compose(setup, "row_projection")
    y, k, lam, s, prior_lam, prior_s = gaussian_terms(setup, state)
    identity = torch.eye(2, dtype=torch.float64)
    values, vectors = torch.linalg.eigh(k - identity)
    delta = (vectors * (-values).clamp_min(0).unsqueeze(-2)) @ vectors.mT
    expected = gauss(lam + delta, s, prior_lam, prior_s)
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)


def test_row_projection_post_reproduces_the_default():
    setup = wide_setup(locals_=(2,))
    a, _, _ = compose(setup, "row_projection")
    b, _, _ = compose(setup, "row_projection_post")
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_composed_check_counts_nonpd_lambda_s():
    setup = wide_setup()
    _, state, sampler = compose(setup, "composed_check")
    y, k, lam, s, prior_lam, prior_s = gaussian_terms(setup, state)
    n = setup["observations"]
    bad = torch.linalg.eigvalsh(lam.sum(0) + (1 - n) * prior_lam)[0] <= 0
    diag = sampler.covariance_diagnostics
    assert diag["lambda_s_checks"] == 1
    assert diag["lambda_s_nonpd_samples"] == (state.shape[1] if bad else 0)
    assert diag["lambda_s_trace"][0][1] == float(bad)


def test_composed_check_is_raw_gauss_when_lambda_s_is_pd():
    """Narrow rows: Lambda_S PD although some I_j may be indefinite -> raw GAUSS, untouched."""
    setup = wide_setup(widen=3.0, narrow=0.8)
    out, state, sampler = compose(setup, "composed_check")
    y, k, lam, s, prior_lam, prior_s = gaussian_terms(setup, state)
    n = setup["observations"]
    assert torch.linalg.eigvalsh(lam.sum(0) + (1 - n) * prior_lam)[0] > 0
    assert sampler.covariance_diagnostics["lambda_s_nonpd_samples"] == 0
    torch.testing.assert_close(out, gauss(lam, s, prior_lam, prior_s), rtol=2e-3, atol=2e-3)


def markovsbi_corrected_score(lams, scores, prior_lam, prior_s, nugget):
    """Port of markovsbi.sampling.score_fn.CorrectedScoreFn (ensure_valid, d > 1)."""
    lams, scores = lams.numpy(), scores.numpy()
    prior_lam, prior_s = prior_lam.numpy(), prior_s.numpy()
    n, d = lams.shape[0], lams.shape[-1]
    lam_total = (1 - n) * prior_lam + lams.sum(0)
    w, v = np.linalg.eigh(lam_total)
    w = np.where(w <= 0, -w, 0.0) / (n - 1)
    corr = v @ np.diag(w) @ v.T + nugget * np.eye(d)
    lams = lams + corr[None]
    lam_total = (1 - n) * prior_lam + lams.sum(0)
    out = []
    for i in range(scores.shape[1]):
        num = (1 - n) * prior_lam @ prior_s[i] + sum(lams[j] @ scores[j, i] for j in range(n))
        out.append(np.linalg.solve(lam_total, num))
    return torch.from_numpy(np.stack(out))


@pytest.mark.parametrize("nugget", [0.1, 0.0])
def test_gloeckler_matches_markovsbi(nugget):
    setup = wide_setup()
    out, state, _ = compose(setup, "gloeckler", nugget=nugget)
    y, k, lam, s, prior_lam, prior_s = gaussian_terms(setup, state)
    expected = markovsbi_corrected_score(lam, s, prior_lam, prior_s, nugget)
    torch.testing.assert_close(out, expected, rtol=3e-3, atol=3e-3)


def test_gloeckler_lift_makes_lambda_s_pd():
    sampler = MultiObsSampler.__new__(MultiObsSampler)
    sampler.pd_nugget = 0.0
    lam_s = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)  # eigenvalues -1, 3
    values, vectors = torch.linalg.eigh(lam_s)
    n = 5
    lifted = lam_s + n * sampler._gloeckler_lift(values, vectors, n)
    torch.testing.assert_close(torch.linalg.eigvalsh(lifted),
                               torch.tensor([1.0 / (n - 1), 3.0], dtype=torch.float64))


def test_consistent_projection_composes_the_modified_gaussians():
    """(K_j + Delta_j) with the score of N(mu_j, (K_j + Delta_j)^-1) -- not of N(mu_j, K_j^-1)."""
    setup = wide_setup()
    out, state, _ = compose(setup, "row_projection_consistent")
    y, k, lam, s, prior_lam, prior_s = gaussian_terms(setup, state)
    identity = torch.eye(2, dtype=torch.float64)
    values, vectors = torch.linalg.eigh(k - identity)
    k_mod = identity + (vectors * values.clamp_min(0).unsqueeze(-2)) @ vectors.mT
    sigma_mod = torch.linalg.inv(k_mod)
    s_mod = -torch.einsum("nij,nsj->nsi", torch.linalg.inv(sigma_mod + VARIANCE * identity),
                          y[None] - setup["means"][:, None])
    expected = gauss(k_mod + identity / VARIANCE, s_mod, prior_lam, prior_s)
    torch.testing.assert_close(out, expected, rtol=3e-3, atol=3e-3)
    # and it is a different rule from the default one here
    default, _, _ = compose(setup, "row_projection")
    assert float((default - out).abs().max()) > 1e-2


def test_strategies_agree_when_every_row_is_informative():
    """Rows narrower than the prior: I_j PSD, Lambda_S PD -> all strategies are plain GAUSS."""
    setup = wide_setup(widen=0.3, narrow=0.3)
    ref, state, _ = compose(setup, "row_projection")
    for strategy in ("composed_check", "row_projection_post", "row_projection_consistent"):
        torch.testing.assert_close(compose(setup, strategy)[0], ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(compose(setup, "gloeckler", nugget=0.0)[0], ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("fallback", ["repair", "row_projection", "gloeckler"])
def test_composed_check_fallbacks_act_only_on_nonpd_samples(fallback):
    """Lambda_S non-PD in every sample -> the fallback rule; with locals, still finite."""
    setup = wide_setup()
    out, state, sampler = compose(setup, "composed_check", fallback)
    assert sampler.covariance_diagnostics["lambda_s_nonpd_fraction"] == 1.0
    if fallback == "row_projection":
        torch.testing.assert_close(out, compose(setup, "row_projection")[0], rtol=1e-5, atol=1e-5)
    elif fallback == "gloeckler":
        torch.testing.assert_close(out, compose(setup, "gloeckler")[0], rtol=1e-5, atol=1e-5)
    else:
        assert sampler.covariance_diagnostics["repair_count"] == 1
    with_locals, _, _ = compose(wide_setup(locals_=(2,)), "composed_check", fallback)
    assert torch.all(torch.isfinite(with_locals))


def test_unknown_strategy_is_rejected():
    setup = wide_setup()
    sampler = MultiObsSampler(make_sbim(setup["model"]))
    with pytest.raises(ValueError):
        sampler.set_pd_strategy("nope")
    with pytest.raises(ValueError):
        sampler.set_pd_strategy("composed_check", "nope")


# --- truncation, blend, gloeckler_consistent, and the probe --------------------------------

def row_information(k):
    identity = torch.eye(2, dtype=torch.float64)
    values, vectors = torch.linalg.eigh(k - identity)
    delta = (vectors * (-values).clamp_min(0).unsqueeze(-2)) @ vectors.mT
    keep = (vectors * (values >= 0).to(values.dtype).unsqueeze(-2)) @ vectors.mT   # P_j^+
    return delta, keep


def test_truncation_drops_the_negative_directions():
    """Each row's message is (I_j^+, P_j^+ u_j), u_j = Lambda_j s_j - Lambda_0 s_0: no lift term."""
    setup = wide_setup()
    out, state, _ = compose(setup, "row_truncation")
    y, k, lam, s, prior_lam, prior_s = gaussian_terms(setup, state)
    delta, keep = row_information(k)
    prior_weighted = prior_s @ prior_lam.T                                       # (S, H)
    messages = torch.einsum("nij,nsj->nsi", lam, s) - prior_weighted[None]
    precision = prior_lam + (lam + delta - prior_lam).sum(0)
    numerator = prior_weighted + torch.einsum("nij,nsj->si", keep, messages)
    expected = torch.linalg.solve(precision, numerator.T).T
    torch.testing.assert_close(out, expected, rtol=3e-3, atol=3e-3)
    # a Gaussian row's kept message is -lambda^-2 I_j^+ (y - m_j) for its likelihood mean m_j:
    # it never involves K_j^-1, unlike the consistent rule
    for rule in ("row_projection", "row_projection_consistent"):
        assert float((compose(setup, rule)[0] - out).abs().max()) > 1e-2


def test_blend_formula_and_limits():
    setup = wide_setup()
    out, state, _ = compose(setup, "row_projection_blend")
    y, k, lam, s, prior_lam, prior_s = gaussian_terms(setup, state)
    delta, _ = row_information(k)
    lifted = lam + delta
    means = torch.linalg.solve((k + delta).unsqueeze(1), torch.einsum("nij,nsj->nsi", lifted, s).unsqueeze(-1))
    extra = (delta.unsqueeze(1) @ means).squeeze(-1).sum(0)
    n = setup["observations"]
    precision = lifted.sum(0) + (1 - n) * prior_lam
    numerator = torch.einsum("nij,nsj->si", lam, s) + extra + (1 - n) * prior_s @ prior_lam.T
    torch.testing.assert_close(out, torch.linalg.solve(precision, numerator.T).T, rtol=3e-3, atol=3e-3)


def test_blend_stays_bounded_where_the_row_mean_is_undetermined():
    """A row with K_j ~ 0 along v: the consistent rule divides by the floor, the blend by P_0."""
    sampler = MultiObsSampler.__new__(MultiObsSampler)
    inverse_variance = torch.tensor(4.0, dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    clean = torch.diag(torch.tensor([2.0, 1e-9], dtype=torch.float64))         # K_j
    effective = (clean + inverse_variance * identity).unsqueeze(0)              # Lambda_j
    delta = torch.diag(torch.tensor([0.0, 1.0 - 1e-9], dtype=torch.float64)).unsqueeze(0)
    scores = torch.tensor([[[0.3, 0.5]]], dtype=torch.float64)
    consistent = sampler._consistent_extra(delta, effective, scores, inverse_variance)
    assert float(consistent.abs().max()) > 1e3
    lifted = effective + delta
    blend = delta[0] @ torch.linalg.solve(clean + delta[0], lifted[0] @ scores[0, 0])
    assert float(blend.abs().max()) < 10


def test_gloeckler_consistent_weights_the_row_means():
    setup = wide_setup()
    out, state, _ = compose(setup, "gloeckler_consistent", nugget=0.1)
    y, k, lam, s, prior_lam, prior_s = gaussian_terms(setup, state)
    n = setup["observations"]
    lam_s = lam.sum(0) + (1 - n) * prior_lam
    w, v = torch.linalg.eigh(lam_s)
    lift = v @ torch.diag((-w).clamp_min(0) / (n - 1)) @ v.T + 0.1 * torch.eye(2, dtype=torch.float64)
    weighted_means = torch.linalg.solve(k.unsqueeze(1), torch.einsum("nij,nsj->nsi", lam, s).unsqueeze(-1))
    numerator = (torch.einsum("nij,nsj->si", lam, s) + (lift @ weighted_means).squeeze(-1).sum(0)
                 + (1 - n) * prior_s @ prior_lam.T)
    expected = torch.linalg.solve(lam_s + n * lift, numerator.T).T
    torch.testing.assert_close(out, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("strategy", ["row_truncation", "row_projection_blend",
                                      "gloeckler_consistent"])
def test_new_strategies_run_with_locals(strategy):
    setup = wide_setup(locals_=(2,))
    out, _, _ = compose(setup, strategy)
    assert torch.isfinite(out).all()


def test_new_strategies_agree_when_every_row_is_informative():
    setup = wide_setup(widen=0.3, narrow=0.3)
    ref, _, _ = compose(setup, "row_projection")
    for strategy in ("row_truncation", "row_projection_blend"):
        torch.testing.assert_close(compose(setup, strategy)[0], ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(compose(setup, "gloeckler_consistent", nugget=0.0)[0], ref,
                               rtol=1e-5, atol=1e-5)


def test_probe_measures_the_consistent_minus_default_score(tmp_path, monkeypatch):
    """The probe leaves the draws alone and its dscore is |s_cons - s_def| / |s_def|."""
    import json
    setup = wide_setup()
    default, state, _ = compose(setup, "row_projection")
    consistent, _, _ = compose(setup, "row_projection_consistent")
    path = tmp_path / "probe.jsonl"
    monkeypatch.setenv("COMPASS_PD_PROBE", str(path))
    probed, _, _ = compose(setup, "row_projection")
    assert torch.equal(probed, default)
    record = json.loads(path.read_text().splitlines()[0])
    ratio = (consistent - default).norm(dim=-1) / default.norm(dim=-1)
    expected = torch.quantile(ratio, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    torch.testing.assert_close(torch.tensor(record["dscore"], dtype=torch.float64), expected, rtol=5e-3, atol=5e-3)
    assert 0 < record["active_fraction"] <= 1 and len(record["k_rel"]) == 3


def test_consistent_gate_switches_to_the_default_above_the_threshold():
    """COMPASS_PD_CONSISTENT_BELOW = lambda*: consistent for lambda < lambda*, default above."""
    setup = wide_setup(locals_=(2,))
    lam = VARIANCE ** 0.5
    default, _, _ = compose(setup, "row_projection")
    consistent, _, _ = compose(setup, "row_projection_consistent")
    assert float((default - consistent).abs().max()) > 1e-2

    def gated(threshold):
        state, mask = build_state(setup, num_samples=6)
        sampler = sampler_for(setup, "row_projection_consistent")
        assert sampler.pd_consistent_below == float("inf")          # off unless set
        sampler.pd_consistent_below = threshold
        scores = setup["model"](state.reshape(-1, setup["nodes"]), TIME,
                                mask.reshape(-1, setup["nodes"])).reshape(state.shape)
        out = sampler._compositional_score(scores.clone(), state, TIME, condition_mask=mask)
        return out[0, :, HIERARCHY].to(torch.float64)

    torch.testing.assert_close(gated(0.5 * lam), default, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gated(2.0 * lam), consistent, rtol=1e-6, atol=1e-6)


def guarded(setup, guard, box, strategy="row_projection_consistent"):
    state, mask = build_state(setup, num_samples=6)
    sampler = sampler_for(setup, strategy)
    sampler.pd_consistent_guard = guard
    sampler.denoise_clamp = box
    scores = setup["model"](state.reshape(-1, setup["nodes"]), TIME,
                            mask.reshape(-1, setup["nodes"])).reshape(state.shape)
    out = sampler._compositional_score(scores.clone(), state, TIME, condition_mask=mask)
    return out[0, :, HIERARCHY].to(torch.float64), sampler


@pytest.mark.parametrize("guard", ["fallback", "clamp"])
def test_consistent_guard_limits(guard):
    """A box that holds every row mean leaves the consistent rule unchanged; with fallback, a
    box that holds none gives the default rule."""
    setup = wide_setup(locals_=(2,))
    consistent, _, _ = compose(setup, "row_projection_consistent")
    wide, sampler = guarded(setup, guard, 1e6)
    torch.testing.assert_close(wide, consistent, rtol=1e-6, atol=1e-6)
    assert sampler.covariance_diagnostics["pd_guard_outside_fraction"] == 0.0
    if guard == "fallback":
        # the same box also clamps the row predictions: compare with the default rule under it
        default, _ = guarded(setup, "none", 1e-9, strategy="row_projection")
        empty, sampler = guarded(setup, guard, 1e-9)
        torch.testing.assert_close(empty, default, rtol=1e-5, atol=1e-5)
        assert sampler.covariance_diagnostics["pd_guard_outside_fraction"] == 1.0


def test_consistent_guard_clamp_is_the_metric_clamp_of_the_row_means():
    """extra = lambda^-2 sum_j Delta_j (clamp(mu_j) - y), clamp in the metric K_j + Delta_j."""
    sampler = MultiObsSampler.__new__(MultiObsSampler)
    sampler.pd_consistent_guard = "clamp"
    sampler.denoise_clamp = 0.5
    sampler.training_prior = None
    sampler.prior_mean = torch.zeros(2, dtype=torch.float64)
    sampler.prior_std = torch.ones(2, dtype=torch.float64)
    generator = torch.Generator().manual_seed(7)
    rows, samples, inverse_variance = 3, 5, 4.0
    rotation = torch.tensor([[0.8, -0.6], [0.6, 0.8]], dtype=torch.float64)
    k = torch.stack([rotation @ torch.diag(torch.tensor(v, dtype=torch.float64)) @ rotation.T
                     for v in ([0.3, 2.0], [1.5, 0.05], [0.8, 0.8])])         # clean precisions K_j
    delta = torch.stack([row_information(kj)[0] for kj in k])
    effective = k + inverse_variance * torch.eye(2, dtype=torch.float64)
    scores = torch.randn(rows, samples, 2, generator=generator, dtype=torch.float64)
    state = 2 * torch.randn(1, samples, 2, generator=generator, dtype=torch.float64)
    got = sampler._guarded_consistent_extra(delta, effective, scores, inverse_variance, state)[0]
    y = state[0]
    mean = y + torch.einsum("nij,nsj->nsi", torch.linalg.inv(k) @ effective, scores) / inverse_variance
    lower, upper = -0.5 * torch.ones(2, dtype=torch.float64), 0.5 * torch.ones(2, dtype=torch.float64)
    clamped = MultiObsSampler._box_qp(mean, (k + delta).unsqueeze(1), lower, upper)
    assert ((clamped >= lower - 1e-12) & (clamped <= upper + 1e-12)).all()
    expected = inverse_variance * torch.einsum("nij,nsj->si", delta, clamped - y)
    torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-9)
