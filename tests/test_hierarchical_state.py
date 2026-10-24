"""Tests for the compact sampler state and the faster composition numerics (2026-10-23).

1. HierarchicalState arithmetic and the layout's assemble/split are exact inverses.
2. The compact layout reproduces the synchronised (dense) sampler draw for draw when the
   noise comes from the legacy stream, for every sampler method.
3. The compact noise stream has the hierarchical covariance: one global vector per draw.
4. "elimination" Schur factors equal the "cholesky" route (with the information
   projection active), across the noise range.
5. FD base reuse and the network row budget leave the Jacobian and the draws unchanged.
"""
import pytest
import torch

from compass import ScoreBasedInferenceModel
from compass.MultiObsSampler import MultiObsSampler
from compass.hierarchical_state import HierarchicalLayout, HierarchicalState

NODES, OBSERVED = 4, 2
MASK = torch.tensor([0., 0., 1., 1.])
LOCAL_PRIOR = (torch.zeros(1), torch.ones(1) * 1.5)


def tiny_model():
    torch.manual_seed(0)
    return ScoreBasedInferenceModel(nodes_size=NODES, sigma=5.0, hidden_size=8, depth=1,
                                    num_heads=2, mlp_ratio=2)


def observations(count=4):
    torch.manual_seed(1)
    return torch.randn(count, OBSERVED) * 0.5


def draw(layout="compact", noise="legacy", setup=None, **kwargs):
    sampler = MultiObsSampler(tiny_model())
    sampler.set_state_layout(layout)
    sampler.set_noise_stream(noise)
    if setup is not None:
        setup(sampler)
    arguments = dict(world_size=1, data=observations(), condition_mask=MASK, hierarchy=[0],
                     timesteps=8, num_samples=6, verbose=False, device="cpu",
                     prior=(torch.zeros(1), torch.ones(1)), correction="gauss_jacobian_FD",
                     local_prior=LOCAL_PRIOR)
    arguments.update(kwargs)
    torch.manual_seed(7)
    return sampler.sample(**arguments)


# ---------------------------------------------------------------------------
# 1. state arithmetic and layout
# ---------------------------------------------------------------------------

def layout_fixture(rows=3, samples=5):
    torch.manual_seed(4)
    mask = MASK.expand(rows, samples, NODES)
    frame = torch.randn(rows, samples, NODES) * mask
    return HierarchicalLayout(frame, mask, [0], [1])


def test_assemble_split_roundtrip_and_synchronisation():
    layout = layout_fixture()
    state = HierarchicalState(g=torch.randn(5, 1), l=torch.randn(3, 5, 1))
    rows = layout.assemble(state)
    assert rows.shape == (3, 5, NODES)
    assert torch.equal(rows[:, :, 0], state.g[:, 0].expand(3, 5))     # globals in every row
    assert torch.equal(rows[:, :, 2:], layout.frame[:, :, 2:])          # observed values in place
    back = layout.split(rows)
    assert torch.equal(back.g, state.g) and torch.equal(back.l, state.l)


def test_state_arithmetic_matches_the_synchronised_array():
    layout = layout_fixture()
    x = HierarchicalState(g=torch.randn(5, 1), l=torch.randn(3, 5, 1))
    u = HierarchicalState(g=torch.randn(5, 1), l=torch.randn(3, 5, 1))
    a, b = torch.tensor([[0.7]]), torch.tensor([[-1.3]])
    rows = layout.assemble(a * x + b * u)                                # G' = aG + bu^g, L' = aL + bu^l
    dense = a * layout.assemble(x) + b * layout.assemble_latent(u)
    assert torch.equal(rows[:, :, :2], dense[:, :, :2])
    assert torch.equal(rows[:, :, 2:], layout.frame[:, :, 2:])
    assert torch.equal(torch.sqrt(x.abs()).g, torch.sqrt(x.g.abs()))
    assert float((x - x).max()) == 0.0


# ---------------------------------------------------------------------------
# 2. compact == dense with the legacy noise stream
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(),                                                     # DPM-2 + churn, gauss_jacobian_FD
    dict(order=1), dict(order=3),
    dict(churn_eta=0.0, corrector_steps=2),                     # predictor-corrector
    dict(method="euler"),
    dict(method="adaptive", correction="uncorrected", adaptive_rel_tol=0.3),
    dict(method="langevin", correction="fnpe", corrector_steps=2),
    dict(correction="gauss_hierarchical", precision_est_samples=40),
    dict(correction="gauss", method="euler", precision_est_samples=40),   # terminal denoise
    dict(churn_eta=0.0, terminal_corrector_counts=(1, 3)),
], ids=lambda kwargs: ",".join(f"{k}={v}" for k, v in kwargs.items()) or "default")
def test_compact_layout_reproduces_the_dense_sampler(kwargs):
    dense = draw("dense", **kwargs)
    compact = draw("compact", **kwargs)
    if isinstance(dense, dict):
        for count in dense:
            assert torch.equal(dense[count], compact[count])
    else:
        assert torch.equal(dense, compact)


def test_compact_layout_reproduces_dense_trajectory_and_warm_start():
    torch.manual_seed(5)
    init = torch.randn(4, 6, NODES) * 0.5
    warm = dict(start_lambda=0.5, init_state=init)
    assert torch.equal(draw("dense", **warm), draw("compact", **warm))
    sampler_d, sampler_c = MultiObsSampler(tiny_model()), MultiObsSampler(tiny_model())
    for sampler, layout in ((sampler_d, "dense"), (sampler_c, "compact")):
        sampler.set_state_layout(layout); sampler.set_noise_stream("legacy")
        torch.manual_seed(7)
        sampler.sample(world_size=1, data=observations(), condition_mask=MASK, hierarchy=[0],
                       timesteps=6, num_samples=5, verbose=False, device="cpu",
                       prior=(torch.zeros(1), torch.ones(1)), correction="gauss_jacobian_FD",
                       save_trajectory=True)
    assert torch.equal(sampler_d.data_t, sampler_c.data_t)


# ---------------------------------------------------------------------------
# 3. compact noise stream
# ---------------------------------------------------------------------------

def test_compact_noise_has_one_global_vector_per_draw():
    layout = layout_fixture(rows=3, samples=4000)
    torch.manual_seed(0)
    noise = layout.noise("compact")
    assert noise.g.shape == (4000, 1) and noise.l.shape == (3, 4000, 1)
    rows = layout.assemble_latent(noise)
    assert torch.equal(rows[0, :, 0], rows[2, :, 0])                     # shared across rows
    assert abs(float(noise.l.std()) - 1.0) < 0.05 and abs(float(noise.g.std()) - 1.0) < 0.05
    corr = torch.corrcoef(torch.stack([noise.l[0, :, 0], noise.l[1, :, 0]]))[0, 1]
    assert abs(float(corr)) < 0.06                                       # independent locals


def test_compact_noise_stream_gives_statistically_equal_draws():
    a = draw("compact", noise="compact", num_samples=400)
    b = draw("dense", noise="legacy", num_samples=400)
    # The untrained net's draws are far from Gaussian, so compare against the Monte-Carlo
    # error of the difference of the two means (draws are independent across samples).
    for feature in (0, 1):
        x, y = a[0, :, feature], b[0, :, feature]
        error = float((x.var() / len(x) + y.var() / len(y)).sqrt())
        assert abs(float(x.mean() - y.mean())) < 4 * error
        assert 0.8 < float(x.std() / y.std()) < 1.25


# ---------------------------------------------------------------------------
# 4. elimination Schur factors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [1e-3, 3e-2, 1.0])
def test_elimination_schur_equals_cholesky_route(lam):
    sampler = MultiObsSampler(tiny_model())
    sampler.correction = "gauss_jacobian"
    h, dim = 3, 5
    sampler.prior_precision_matrix = torch.eye(h, dtype=torch.float64) * 0.5
    generator = torch.Generator().manual_seed(2)
    root = torch.randn(6, 4, dim, dim, generator=generator, dtype=torch.float64)
    information = root @ root.mT * 0.3
    information[..., :h, :h] -= 0.4 * torch.eye(h, dtype=torch.float64)   # some I_j indefinite
    information = information + 0.5 * torch.eye(dim, dtype=torch.float64)
    joint = information + torch.eye(dim, dtype=torch.float64) / lam**2
    results = {}
    for mode in ("cholesky", "elimination"):
        sampler.set_schur_mode(mode)
        results[mode] = sampler._lifted_schur_factors(joint, 1.0 / lam**2, 6, h)
        assert sampler._last_row_delta is not None                       # the projection fired
    (k_c, r_c), (k_e, r_e) = results["cholesky"], results["elimination"]
    scale = 1.0 / lam**2
    assert torch.allclose(k_e, k_c, rtol=0, atol=1e-9 * scale)
    assert torch.allclose(r_e, r_c, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# 5. FD base reuse and the network row budget
# ---------------------------------------------------------------------------

def test_fd_base_reuse_and_row_budget_leave_draws_unchanged():
    reference = draw("compact", setup=lambda s: setattr(s, "fd_reuse_base", False))
    assert torch.equal(draw("compact"), reference)
    assert torch.equal(draw("compact", setup=lambda s: s.set_network_row_budget(7)), reference)


def test_prepare_data_does_not_copy_per_draw():
    sampler = MultiObsSampler(tiny_model())
    data, mask = observations(), MASK.repeat(4, 1)
    joint, expanded, _ = sampler._prepare_data((data, mask, torch.arange(4)), 1000, "cpu")
    assert joint.shape == (4, 1000, NODES) and joint.stride(1) == 0 and expanded.stride(1) == 0
    assert torch.equal(joint[:, 0, 2:], data)
