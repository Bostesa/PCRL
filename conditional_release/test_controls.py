"""Data-free tests for actual joint, independent, and frozen-stage laws."""
import numpy as np
from scipy import sparse


def test_product_map_is_independent_coin_law():
    from conditional_release.controls import product_map
    fixed = np.array([[.25, .75], [.4, .6]])
    groups = np.array([0, 0])
    matrix = product_map(groups, fixed, 1, 2)
    q = (matrix @ np.array([.3, .7])).reshape(2, 4)
    expected = (np.array([.3, .7])[None, :, None] * fixed[:, None, :]).reshape(2, 4)
    np.testing.assert_allclose(q, expected)
    pair = q.reshape(2, 2, 2)
    np.testing.assert_allclose(pair[:, 0, 0] * pair[:, 1, 1], pair[:, 0, 1] * pair[:, 1, 0])


def test_product_second_map_keeps_first_frozen():
    from conditional_release.controls import product_map
    fixed = np.array([[.25, .75], [.4, .6]])
    matrix = product_map(np.array([0, 1]), fixed, 2, 2)
    q = (matrix @ np.array([.3, .7, .2, .8])).reshape(2, 2, 2)
    np.testing.assert_allclose(q.sum(2), fixed)
    np.testing.assert_allclose(q.sum(1), [[.3, .7], [.2, .8]])


def test_sequential_map_conditions_on_emitted_first_not_action_position():
    from conditional_release.controls import sequential_map
    first = np.array([[.3, .7], [.4, .6]])
    # Distinct first slots which emit the same wire must share second weights.
    second_group = np.array([[0, 0], [1, 0]])
    matrix = sequential_map(first, second_group, 1, 2)
    q = (matrix @ np.array([.2, .8, .9, .1])).reshape(2, 2, 2)
    np.testing.assert_allclose(q.sum(2), first)
    np.testing.assert_allclose(q[0, 0] / first[0, 0], q[0, 1] / first[0, 1])
    assert abs(q[1, 0, 0] / first[1, 0] - .9) < 1e-12


def test_categorical_map_trains_true_one_hot_control():
    from conditional_release.controls import categorical_map, simplex_rows
    groups = np.array([0, 1, 0])
    matrix = categorical_map(groups, 4)
    theta = np.array([0, 1, 0, 0, 0, 0, 1, 0.])
    q = (matrix @ theta).reshape(3, 4)
    assert np.array_equal(q.argmax(1), [1, 2, 1])
    inequalities, rhs = simplex_rows(2, 4)
    assert np.max(inequalities @ theta - rhs) == 0


def test_invalid_template_is_not_allowed_for_shared_group():
    from conditional_release.controls import template_bounds
    valid = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], bool)
    bounds = template_bounds(np.array([0, 0]), valid)
    assert np.array_equal(bounds[:, 1], [1, 1, 0, 0])


def test_rejects_nonfinite_factor_and_negative_group():
    from conditional_release.controls import product_map
    import pytest
    with pytest.raises(ValueError):
        product_map(np.array([-1]), np.array([[1., 0]]), 1, 2)
    with pytest.raises(ValueError):
        product_map(np.array([0]), np.array([[1., np.nan]]), 1, 2)
