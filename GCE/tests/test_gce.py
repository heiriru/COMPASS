"""Fast checks for the GCE application (CPU, ~10 s):

    CUDA_VISIBLE_DEVICES=-1 ../.COMPASS/bin/python -m pytest tests -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gce.paths import configure  # noqa: E402

configure(cpus=3)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import norm  # noqa: E402

from gce import compass_setup, models  # noqa: E402
from gce.observation import PROFILES, make_observations, observed_vector  # noqa: E402
from gce.paths import EXTERNAL  # noqa: E402
from gce.simulator import TRACED, StarParameters, simulate_star, simulate_star_stock  # noqa: E402

ROWS = [[-2.3, -2.89, -0.3, 0.55, 0.5, 8.0], [-2.0, -3.1, -0.1, 0.6, 0.4, 3.0]]


@pytest.mark.skipif(not EXTERNAL["chempy_tng_val"].exists(), reason="old-numpy reference not readable")
def test_port_reproduces_old_numpy_outputs():
    stored = np.load(EXTERNAL["chempy_tng_val"])
    for i in (0, 1):
        new = simulate_star_stock(stored["params"][i])
        assert np.abs(new - stored["abundances"][i]).max() < 1e-12


@pytest.mark.parametrize("row", ROWS)
def test_constant_schedule_is_the_stock_run(row):
    star = StarParameters(*row)
    stock = simulate_star_stock(row)
    ours = simulate_star(star)
    hooked = simulate_star(star, slope_schedule=lambda log_z, t: row[0])
    assert np.array_equal(stock, ours.abundances)
    assert np.array_equal(ours.abundances, hooked.abundances)


def test_metallicity_schedule_changes_the_abundances():
    star = StarParameters(*ROWS[0])
    base = simulate_star(star).abundances
    varied = simulate_star(star, slope_schedule=lambda log_z, t: -2.3 - 0.5 * log_z, record_slopes=True)
    assert np.abs(varied.abundances - base).max() > 0.01
    assert -3.5 <= min(varied.slopes) and max(varied.slopes) <= -1.5


def test_birth_time_coordinate_round_trip():
    t = np.linspace(1.01, 13.69, 50)
    assert np.allclose(models.birth_time_from_u(models.u_from_birth_time(t)), t)


@pytest.mark.parametrize("name", sorted(models.MODELS))
def test_model_layout_and_draws(name):
    model = models.get(name)
    rng = np.random.default_rng(0)
    g = model.sample_globals(rng, 5, 1.5)
    l = model.sample_locals(rng, g)
    assert g.shape == (5, len(model.global_names)) and l.shape == (5, len(model.local_names))
    for key in model.nested_null:
        assert key in model.global_names
    star, schedule = model.star(g[0], l[0])
    assert 1.0 <= star.birth_time <= 13.7
    assert (schedule is None) == (model.schedule == "constant")


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_observation_layouts_agree(profile):
    p = PROFILES[profile]
    rng = np.random.default_rng(0)
    x = make_observations(np.zeros((4, len(TRACED))), np.full(4, 8.0), p, rng)
    assert x.shape == (4, p.x_size)
    errors = np.full((4, len(p.elements)), 0.03)
    x_real, _ = observed_vector(np.zeros((4, len(p.elements))), errors, p,
                                age=np.full(4, 5.0) if p.time_node else None,
                                age_error=np.full(4, 1.0) if p.time_node else None)
    assert x_real.shape == x.shape


def test_savage_dickey_matches_the_conjugate_answer():
    rng = np.random.default_rng(1)
    m, s = 0.4, 0.2                        # posterior N(m, s^2); prior N(0, 0.3^2)
    out = compass_setup.savage_dickey(rng.normal(m, s, 200_000), 0.0, 0.3, 0.0)
    exact = norm.logpdf(0.0, m, s) - norm.logpdf(0.0, 0.0, 0.3)
    assert out["reliable"] and abs(out["log_bf_null"] - exact) < 0.05


def test_interval_null_bayes_factor():
    rng = np.random.default_rng(2)
    prior_mean, prior_sd = np.log(0.1), 1.0
    draws = rng.normal(np.log(0.01), 0.5, 100_000)
    out = compass_setup.interval_null_bayes_factor(draws, prior_mean, prior_sd, (-np.inf, np.log(0.02)))
    exact = np.log(norm.cdf(np.log(0.02), np.log(0.01), 0.5)) - np.log(norm.cdf(np.log(0.02), prior_mean, prior_sd))
    assert abs(out["log_bf_null"] - exact) < 0.02
