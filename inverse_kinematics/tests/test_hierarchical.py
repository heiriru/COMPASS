"""Contract tests for the tall-data layout and its reference posterior.

Run from the repository root:
    python -m pytest inverse_kinematics/tests -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hierarchical as H  # noqa: E402
import simulator as sim  # noqa: E402


def test_node_order_round_trips():
    """A silent transposition here would train on a different forward model."""
    rng = sim.generator("test-order")
    theta = rng.normal(size=(64, 4))
    assert np.array_equal(
        H.from_simulator_order(H.to_simulator_order(theta)), theta
    )


def test_forward_agrees_with_the_simulator_under_conversion():
    rng = sim.generator("test-forward-nodes")
    theta = rng.normal(size=(64, 4))
    assert np.allclose(H.forward(theta), sim.forward(H.to_simulator_order(theta)))


def test_prior_scales_follow_the_simulator():
    """Node-order scales are derived, not retyped, so they cannot drift."""
    assert H.PRIOR_STD[0] == sim.PRIOR_SCALES[1]      # angle_1
    assert H.PRIOR_STD[1] == sim.PRIOR_SCALES[0]      # height_arm
    assert np.array_equal(H.PRIOR_STD[2:], sim.PRIOR_SCALES[2:])


def test_layout_constants_are_consistent():
    assert H.NODES == 6
    assert H.LATENT_INDICES == [0, 1, 2, 3]
    assert H.CONDITION_MASK.tolist() == [0, 0, 0, 0, 1, 1]
    assert len(H.NODE_NAMES) == H.NODES


def test_dataset_shares_one_angle_and_varies_the_locals():
    rng = sim.generator("test-dataset")
    truth, locals_, x = H.simulate_dataset(12, rng)
    assert locals_.shape == (12, 3)
    assert x.shape == (12, 2)
    # Every row reproduces its own endpoint from the *shared* angle.
    theta = np.concatenate([np.full((12, 1), truth), locals_], axis=1)
    assert np.allclose(H.forward(theta), x)
    # ...and the locals really do differ between observations.
    assert locals_.std(axis=0).min() > 0


def test_dataset_honours_a_supplied_angle():
    rng = sim.generator("test-fixed-angle")
    truth, _, _ = H.simulate_dataset(4, rng, angle_1=0.31)
    assert truth == 0.31


def test_training_rows_draw_a_fresh_angle_each():
    """Training is single-observation; sharing is a property of the tall data."""
    rng = sim.generator("test-rows")
    theta, x = H.simulate(512, rng)
    assert theta.shape == (512, 4) and x.shape == (512, 2)
    assert float(theta[:, 0].std()) > 0.3          # angle_1 varies row to row
    assert np.allclose(H.forward(theta.numpy()), x.numpy(), atol=1e-6)


def test_reference_recovers_a_known_angle_and_contracts_with_n():
    """The reference must concentrate on the truth, and do so faster with more data.

    Both halves matter: a reference that merely peaked near the truth could be
    the prior in disguise, and one that contracted without being centred would
    be confidently wrong.
    """
    rng = sim.generator("test-reference")
    truth = 0.4
    _, _, x_small = H.simulate_dataset(2, rng, angle_1=truth)
    _, _, x_large = H.simulate_dataset(16, rng, angle_1=truth)

    _, small_std, _ = H.grid_moments(*H.reference_global(x_small, draws=20_000))
    grid, density = H.reference_global(x_large, draws=20_000)
    large_mean, large_std, _ = H.grid_moments(grid, density)

    assert large_std < small_std < H.GLOBAL_PRIOR_STD[0]
    assert abs(large_mean - truth) < 4 * large_std


def test_reference_integrates_to_one():
    rng = sim.generator("test-normalization")
    _, _, x = H.simulate_dataset(6, rng)
    grid, density = H.reference_global(x, draws=20_000)
    assert abs(float(np.trapezoid(density, grid)) - 1.0) < 1e-6
    assert (density >= 0).all()
