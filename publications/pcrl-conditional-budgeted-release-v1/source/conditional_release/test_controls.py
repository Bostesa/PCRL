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


def test_registered_second_groups_alias_codes_and_keep_parent_fallback():
    from conditional_release.controls import register_second_groups, apply_second_groups
    contexts = np.array([0, 0, 1], np.int64)
    emitted = np.array([[10, 11], [10, 11], [12, 12]], np.int64)
    valid = np.ones(emitted.shape, bool)
    groups, keys, counts = register_second_groups(contexts, emitted, valid, floor=1., ceiling=4)
    assert keys.shape == (4, 2)
    assert np.array_equal(groups[0], groups[1])
    assert groups[2, 0] == groups[2, 1]
    unseen = apply_second_groups(np.array([0, 1]), np.array([[900, 10], [901, 12]]), keys)
    assert np.array_equal(keys[unseen[:, 0], 1], [-1, -1])


def test_product_start_is_exact_keep_or_uniform_valid():
    from conditional_release.controls import initial_factor
    bounds = np.array([[0, 1], [0, 0], [0, 1], [0, 0]], np.float64)
    assert np.array_equal(initial_factor(bounds, "keep"), [[1, 0, 0, 0]])
    assert np.array_equal(initial_factor(bounds, "uniform"), [[.5, 0, .5, 0]])


def _space():
    from incumbent_view.r2_space import Space
    p1 = np.array([[.6,.4], [.65,.35], [.7,.3], [.75,.25]], np.float64)
    p2 = np.array([[.5,.1,.1,.1,.1,.1], [.6,.08,.08,.08,.08,.08],
                   [.7,.06,.06,.06,.06,.06], [.8,.04,.04,.04,.04,.04]], np.float64)
    actions = np.array([[i]+[j for j in range(4) if j != i] for i in range(4)], np.int64)
    return Space(p1,p2,actions,actions,np.ones((4,4),bool),np.ones((4,4),bool))


def test_local_deployment_marginal_ignores_other_input():
    from conditional_release.controls import ControlPolicy
    space = _space()
    one = np.tile([.3,.7,0,0], (4,1))
    two = np.tile([.2,0,.8,0], (4,1))
    policy = ControlPolicy(space, "LOCAL", None, one, two)
    p = space.prepare(np.array([1,1],np.int64), np.array([0,3],np.int64))
    q = policy.law_prepared(p).reshape(2,4,4)
    np.testing.assert_allclose(q.sum(2), one[[1,1]])
    assert np.array_equal(q.sum(2)[0], q.sum(2)[1])


def test_deterministic_deployment_rejects_fractional_policy():
    from conditional_release.controls import ControlPolicy
    import pytest
    with pytest.raises(ValueError):
        ControlPolicy(_space(), "DETERMINISTIC", None, np.tile([.5,.5]+[0]*14, (1,1)))
