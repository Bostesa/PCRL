"""Small independent checks of the paired interval replay arithmetic."""

import hashlib

import numpy as np

from results.pcrl_utility_reference_repair_v1.r3.utility_interval_replay_worker import (
    _oriented_costs, group_draw_means, one_sided_bounds,
)


def test_group_draw_means_match_expanded_people_and_draw_digest():
    group = np.array([9, 3, 9, 5, 7], dtype=np.int64)
    costs = np.array([[2.0, 1.0], [3.0, 5.0], [4.0, 1.0],
                      [6.0, 3.0], [8.0, 2.0]])
    actual, digest, count = group_draw_means(costs, group, draws=7,
                                             seed=41, batch_size=3)
    ordered = np.array([3, 5, 7, 9])
    rng = np.random.default_rng(41)
    expected = []
    raw = hashlib.sha256()
    for take in (3, 3, 1):
        index = rng.integers(0, 4, size=(take, 4), dtype=np.int32)
        raw.update(index.tobytes())
        for row in index:
            people = np.concatenate([np.flatnonzero(group == ordered[i]) for i in row])
            expected.append(costs[people].mean(axis=0))
    assert count == 4 and digest == raw.hexdigest()
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)


def test_one_sided_bounds_contain_points_and_apply_inclusive_caps():
    point = np.array([0.0, 0.02, 0.009])
    boot = np.array([[0.0, 0.02, 0.004], [0.0, 0.02, 0.014],
                     [0.0, 0.02, 0.005], [0.0, 0.02, 0.013]])
    out = one_sided_bounds(point, boot, np.array([0.01, 0.01, 0.01]))
    assert [row['status'] for row in out['rows']] == ['PASS', 'VIOLATION', 'UNRESOLVED']
    assert all(row['lower'] <= row['point'] <= row['upper'] for row in out['rows'])
    shifted = one_sided_bounds(np.array([0.0]),
                               np.array([[1.0], [2.0], [3.0], [4.0]]),
                               np.array([0.01]))
    assert shifted['upper_critical'] == 0.0
    assert shifted['rows'][0]['upper'] == 0.0


def test_oriented_costs_use_disclosed_probabilities_and_summed_brier():
    candidate = np.array([[0.8, 0.2], [0.4, 0.6]])
    reference = np.array([[0.7, 0.3], [0.9, 0.1]])
    label = np.array([0, 1], dtype=np.int64)
    accuracy, nll, brier = _oriented_costs(candidate, reference, label)
    np.testing.assert_array_equal(accuracy, [0, -1])
    np.testing.assert_allclose(nll, [np.log(.7 / .8), np.log(.1 / .6)])
    np.testing.assert_allclose(brier, [.08 - .18, .32 - 1.62])
