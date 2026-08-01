import numpy as np

from Partial_Pooling.rng import generator
from Partial_Pooling.schema import transformed_observations
from Partial_Pooling.simulators.ddm_sde import simulate_ddm, simulate_ddm_trial


def test_vectorized_single_trial_matches_reference():
    parameters = np.array([[0.4, 1.3, 0.15, 0.48]])
    expected = simulate_ddm_trial(*parameters[0], np.random.default_rng(41))
    actual = simulate_ddm(parameters, np.random.default_rng(41), trials=1)[0, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_shapes_validity_reproducibility_and_censoring():
    parameters = np.array([[0.0, 100.0, 0.3, 0.5], [1.0, 1.0, 0.1, 0.5]])
    first = simulate_ddm(parameters, np.random.default_rng(9), trials=4, max_decision_time=0.002)
    second = simulate_ddm(parameters, np.random.default_rng(9), trials=4, max_decision_time=0.002)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 4, 3)
    assert set(np.unique(first[..., 0])).issubset({0.0, 1.0})
    assert set(np.unique(first[..., 2])).issubset({0.0, 1.0})
    assert np.all(first[..., 1] >= parameters[:, 2, None])
    assert np.all(first[..., 1] <= parameters[:, 2, None] + 0.002 + 1e-12)
    assert transformed_observations(first).shape == (2, 12)


def test_named_splits_are_independent_and_reproducible():
    train = generator(7, "data", "train").normal(size=8)
    same = generator(7, "data", "train").normal(size=8)
    validation = generator(7, "data", "validation").normal(size=8)
    np.testing.assert_array_equal(train, same)
    assert not np.array_equal(train, validation)
