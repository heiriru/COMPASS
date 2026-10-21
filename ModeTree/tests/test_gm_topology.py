"""Tests for the bridge-topology machinery of ``exp7_gm_topology``.

The experiment's conclusions rest on three claims that are cheap to check
exactly and expensive to get wrong silently:

* the topology reader finds the modes, the valley, the barrier and the branch
  mass of a curve whose answers are known in closed form;
* the ``alpha`` field really does interpolate between ``naive`` and ``jac``,
  with both endpoints reproduced to machine precision -- otherwise "alpha = 1
  is JAC" is an assumption rather than a fact;
* the level-dependent Langevin step degenerates to exactly the two schedules
  it is supposed to interpolate, so a schedule's effect is never confused with
  a change of sampler.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path("/export/home/rheinric/COMPASS")
sys.path.insert(0, str(ROOT / "ModeTree"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Gaussian_mixtures_compositional_score_modelling"))
sys.path.insert(0, str(ROOT / "ModeTree" / "experiments"))

import datasets                      # noqa: E402
import model as M                    # noqa: E402

import gm_topology as TP             # noqa: E402
from gm_bridges import (ComposedField, annealed_langevin,  # noqa: E402
                        power_schedule, sigmoid_schedule)
from gm_oracle import GMOracle       # noqa: E402

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def oracle():
    rng = np.random.default_rng(0)
    dataset = datasets.make_dataset("asymmetric", 8, rng, M.DEFAULT,
                                    branch_mass=0.75)
    return GMOracle(M.DEFAULT, dataset["x"], 15.0)


@pytest.fixture(scope="module")
def bounds(oracle):
    return float(oracle.schedule.lam(1.0)), float(oracle.schedule.lam(1e-3))


# ---------------------------------------------------------------------------
# The topology reader
# ---------------------------------------------------------------------------

def test_describe_recovers_a_known_two_component_mixture():
    """A mixture with known modes, valley and weights is read back correctly."""
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    centres = torch.tensor([-2.0, 1.5], dtype=torch.float64)
    sd = 0.3
    grid = torch.linspace(-8.0, 8.0, 40001, dtype=torch.float64)

    def log_density(points):
        z = (points.unsqueeze(-1) - centres) / sd
        return torch.logsumexp(torch.log(weights) - 0.5 * z ** 2, dim=-1)

    # The slope this module works with is d/dg log q.
    z = (grid.unsqueeze(-1) - centres) / sd
    responsibility = torch.softmax(torch.log(weights) - 0.5 * z ** 2, dim=-1)
    slope = (responsibility * (-(grid.unsqueeze(-1) - centres) / sd ** 2)).sum(-1)

    result = TP.describe(grid, slope)
    assert result["num_modes"] == 2
    assert result["mode_minus"] == pytest.approx(-2.0, abs=2e-3)
    assert result["mode_plus"] == pytest.approx(1.5, abs=2e-3)
    # Well separated components, so the mass either side of the valley is the
    # mixture weight to a very good approximation.
    assert result["w_plus"] == pytest.approx(0.75, abs=1e-6)
    # The barrier is read against the *shallower* mode; here that is the
    # smaller component, and the closed-form depth is available directly.
    valley = result["split_point"]
    expected = float(log_density(torch.tensor([-2.0]))[0]
                     - log_density(torch.tensor([valley]))[0])
    assert result["barrier"] == pytest.approx(expected, rel=2e-3)


def test_unimodal_curve_reports_no_barrier():
    grid = torch.linspace(-6.0, 6.0, 20001, dtype=torch.float64)
    step = float(grid[1] - grid[0])
    slope = -grid
    result = TP.describe(grid, slope)
    assert result["num_modes"] == 1
    assert result["barrier"] == 0.0
    # Without the continuous slope the root is only bracketed, so it is
    # accurate to half a grid spacing; with it, Brent refines to machine
    # precision.  ``profile`` always passes the continuous slope.
    assert result["mode_plus"] == pytest.approx(0.0, abs=step)
    refined = TP.describe(grid, slope, function=lambda value: -float(value))
    assert refined["mode_plus"] == pytest.approx(0.0, abs=1e-10)


def test_mass_floor_rejects_a_negligible_shoulder():
    """A bump carrying far less than the floor is not counted as a branch."""
    grid = torch.linspace(-8.0, 8.0, 40001, dtype=torch.float64)
    weights = torch.tensor([1.0 - 1e-8, 1e-8], dtype=torch.float64)
    centres = torch.tensor([0.0, 5.0], dtype=torch.float64)
    z = (grid.unsqueeze(-1) - centres) / 0.3
    responsibility = torch.softmax(torch.log(weights) - 0.5 * z ** 2, dim=-1)
    slope = (responsibility
             * (-(grid.unsqueeze(-1) - centres) / 0.09)).sum(-1)
    assert TP.describe(grid, slope, mass_floor=1e-3)["num_modes"] == 1
    assert TP.describe(grid, slope, mass_floor=1e-12)["num_modes"] == 2


def test_tau_and_lambda_round_trip():
    hi, lo = 6.431, 0.03167
    for tau in (0.0, 0.1, 0.5, 0.9, 1.0):
        lam = TP.lam_of_tau(tau, hi, lo)
        assert TP.tau_of_lam(lam, hi, lo) == pytest.approx(tau, abs=1e-12)


def test_topology_loss_vanishes_on_a_perfect_match():
    truth = {"num_modes": 2, "mode_minus": -1.0, "mode_plus": 1.0,
             "barrier": 3.0, "w_plus": 0.75, "normalisable": True}
    assert TP.topology_loss(dict(truth), truth, within=0.25) == 0.0
    wrong_count = dict(truth, num_modes=3)
    assert TP.topology_loss(wrong_count, truth, within=0.25) >= 10.0


# ---------------------------------------------------------------------------
# The alpha field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.05, 0.5, 3.0])
def test_alpha_field_reproduces_both_endpoints(oracle, bounds, lam):
    hi, lo = bounds
    t = float(oracle.schedule.time_of_lam(lam))
    state = torch.randn(11, oracle.dim, dtype=torch.float64)
    naive = ComposedField(oracle, "naive", lam_max=hi, lam_min=lo)(state, t, lam)
    jac = ComposedField(oracle, "jac", lam_max=hi, lam_min=lo)(state, t, lam)
    at_zero = ComposedField(oracle, "alpha", alpha=0.0,
                            lam_max=hi, lam_min=lo)(state, t, lam)
    at_one = ComposedField(oracle, "alpha", alpha=1.0,
                           lam_max=hi, lam_min=lo)(state, t, lam)
    assert torch.allclose(at_zero, naive, atol=1e-12)
    assert torch.allclose(at_one, jac, atol=1e-12)


@pytest.mark.parametrize("weight", [0.25, 0.5, 0.75])
def test_alpha_field_is_affine_in_alpha(oracle, bounds, weight):
    hi, lo = bounds
    lam, t = 0.4, float(oracle.schedule.time_of_lam(0.4))
    state = torch.randn(9, oracle.dim, dtype=torch.float64)
    naive = ComposedField(oracle, "naive", lam_max=hi, lam_min=lo)(state, t, lam)
    jac = ComposedField(oracle, "jac", lam_max=hi, lam_min=lo)(state, t, lam)
    blended = ComposedField(oracle, "alpha", alpha=weight,
                            lam_max=hi, lam_min=lo)(state, t, lam)
    assert torch.allclose(blended, naive + weight * (jac - naive), atol=1e-12)


def test_alpha_schedule_reads_the_log_lambda_coordinate(oracle, bounds):
    """``alpha(tau)`` is evaluated at the log-lambda coordinate, not at ``t``."""
    hi, lo = bounds
    seen = []
    field = ComposedField(oracle, "alpha", alpha=lambda tau: seen.append(tau) or 1.0,
                          lam_max=hi, lam_min=lo)
    for lam in (lo, math.sqrt(hi * lo), hi):
        field(torch.zeros(1, oracle.dim, dtype=torch.float64),
              float(oracle.schedule.time_of_lam(lam)), lam)
    assert seen[0] == pytest.approx(0.0, abs=1e-12)
    assert seen[1] == pytest.approx(0.5, abs=1e-12)
    assert seen[2] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# The level-dependent Langevin step
# ---------------------------------------------------------------------------

def _langevin(oracle, bounds, **kwargs):
    hi, lo = bounds
    field = ComposedField(oracle, "fnpe", fnpe_prior="diffused",
                          lam_max=hi, lam_min=lo)
    generator = torch.Generator().manual_seed(7)
    samples, _ = annealed_langevin(field, 16, levels=5, steps_per_level=3,
                                   snr=0.1, lam_max=hi, lam_min=lo,
                                   generator=generator, **kwargs)
    return samples


def test_schedule_at_w_zero_is_the_fixed_prescription(oracle, bounds):
    reference = _langevin(oracle, bounds, hierarchy_scale=None)
    scheduled = _langevin(oracle, bounds, shared_schedule=lambda tau: 0.0)
    assert torch.allclose(reference, scheduled, atol=1e-12)


def test_schedule_at_w_one_is_the_unscaled_step(oracle, bounds):
    reference = _langevin(oracle, bounds, hierarchy_scale=1.0)
    scheduled = _langevin(oracle, bounds, shared_schedule=lambda tau: 1.0)
    assert torch.allclose(reference, scheduled, atol=1e-12)


def test_power_and_sigmoid_schedules_have_the_right_orientation():
    """``w`` must be ~0 at the noise end and ~1 at the data end."""
    for p in (0.5, 1, 2, 4):
        schedule = power_schedule(p)
        assert schedule(1.0) == pytest.approx(0.0, abs=1e-12)
        assert schedule(0.0) == pytest.approx(1.0, abs=1e-12)
    for k, tau0 in ((5, 0.25), (10, 0.5), (20, 0.75)):
        schedule = sigmoid_schedule(k, tau0)
        assert schedule(1.0) < 0.3
        assert schedule(0.0) > 0.7
        assert schedule(tau0) == pytest.approx(0.5, abs=1e-12)


def test_variance_step_stays_inside_its_clip(oracle, bounds):
    """The geometry-aware step never leaves ``[1/n, 1]`` times the base step."""
    hi, lo = bounds
    field = ComposedField(oracle, "fnpe", fnpe_prior="diffused",
                          lam_max=hi, lam_min=lo)
    generator = torch.Generator().manual_seed(3)
    _, stats = annealed_langevin(field, 32, levels=8, steps_per_level=4,
                                 snr=0.1, lam_max=hi, lam_min=lo,
                                 generator=generator, variance_step=True,
                                 variance_snr=1e6, trace=True)
    shared = [row["shared_scale"] for row in stats["level_trace"]]
    assert min(shared) >= 1.0 / oracle.n - 1e-12
    assert max(shared) <= 1.0 + 1e-12
