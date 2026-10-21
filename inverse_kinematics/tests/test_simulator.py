"""Contract tests for the inverse-kinematics simulator.

Run from the repository root:
    python -m pytest inverse_kinematics/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import simulator as sim  # noqa: E402


def test_forward_matches_the_notebook_scalar_form():
    """The vectorized forward model against a literal transcription of the source.

    Grades the vectorization, which is the only thing this package changed about
    the equations; the transcription below is the notebook's body verbatim.
    """
    rng = sim.generator("test-forward")
    parameters = sim.sample_prior(64, rng)
    for row in parameters:
        height_arm, angle_1, angle_2, angle_3 = row
        l1, l2, l3 = 0.5, 0.5, 1.0
        x1 = l1 * np.sin(angle_1)
        x1 += l2 * np.sin(angle_1 + angle_2)
        x1 += l3 * np.sin(angle_1 + angle_2 + angle_3) + height_arm
        x2 = l1 * np.cos(angle_1)
        x2 += l2 * np.cos(angle_1 + angle_2)
        x2 += l3 * np.cos(angle_1 + angle_2 + angle_3)
        # Not bit-for-bit: the vectorized form sums the three segments in one
        # expression where the notebook accumulates with `+=`, so the rounding
        # order differs. A few ulp is the correct expectation here.
        assert np.allclose(sim.forward(row), np.array([x1, x2]), atol=1e-15, rtol=1e-13)


def test_joint_chain_terminates_at_the_forward_endpoint():
    """The drawn arm and the closed form must agree, in reversed plot order.

    This is the identity the plotting code silently relies on: if it broke, the
    figures would show arms whose tips miss the observation they encode, and
    nothing else would fail.
    """
    rng = sim.generator("test-joints")
    parameters = sim.sample_prior(512, rng)
    endpoints = sim.joint_positions(parameters)[:, -1, :]
    assert np.abs(endpoints - sim.forward(parameters)[:, ::-1]).max() < 1e-12


def test_joint_chain_has_the_declared_segment_lengths():
    rng = sim.generator("test-lengths")
    joints = sim.joint_positions(sim.sample_prior(256, rng))
    lengths = np.linalg.norm(np.diff(joints, axis=1), axis=2)
    assert np.allclose(lengths, [sim.L1, sim.L2, sim.L3])


def test_shoulder_sits_on_the_rail_at_the_height_parameter():
    rng = sim.generator("test-shoulder")
    parameters = sim.sample_prior(128, rng)
    shoulders = sim.joint_positions(parameters)[:, 0, :]
    assert np.allclose(shoulders[:, 0], 0.0)
    assert np.allclose(shoulders[:, 1], parameters[:, 0])


def test_reach_is_bounded_by_the_summed_segment_lengths():
    """No configuration can put the tip further from its shoulder than l1+l2+l3."""
    rng = sim.generator("test-reach")
    parameters = sim.sample_prior(4096, rng)
    joints = sim.joint_positions(parameters)
    reach = np.linalg.norm(joints[:, -1, :] - joints[:, 0, :], axis=1)
    assert reach.max() <= sim.L1 + sim.L2 + sim.L3 + 1e-12


def test_prior_scales_are_recovered_from_a_large_draw():
    rng = sim.generator("test-prior")
    parameters = sim.sample_prior(200_000, rng)
    assert np.allclose(parameters.mean(axis=0), 0.0, atol=0.01)
    assert np.allclose(parameters.std(axis=0), sim.PRIOR_SCALES, rtol=0.02)


def test_named_generators_are_reproducible_and_independent():
    first = sim.sample_prior(32, sim.generator("alpha"))
    again = sim.sample_prior(32, sim.generator("alpha"))
    other = sim.sample_prior(32, sim.generator("beta"))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_rejection_sampling_lands_inside_the_tolerance():
    rng = sim.generator("test-rejection")
    target = np.array([0.0, 1.7])
    accepted, drawn = sim.sample_consistent_with(target, 24, rng, tolerance=0.02)
    assert len(accepted) == 24
    assert drawn > 0
    distance = np.linalg.norm(sim.forward(accepted) - target, axis=1)
    assert distance.max() < 0.02


def test_rejection_sampling_reports_an_unreachable_target():
    """A target beyond the arm's reach must fail loudly, not spin forever."""
    rng = sim.generator("test-unreachable")
    with pytest.raises(RuntimeError, match="landed within"):
        sim.sample_consistent_with(
            np.array([50.0, 50.0]), 1, rng, tolerance=0.01, max_draws=400_000,
        )
