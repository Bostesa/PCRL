"""Synthetic, data-free checks of paired group resampling and boundaries."""

import hashlib
import importlib

import numpy as np


def test_group_bootstrap_carries_both_people_and_pairs_costs():
    interval = importlib.import_module("utility_intervals")
    group = np.array([10, 10, 20, 30, 40], dtype=np.int64)
    cost = np.array([[1., 0.], [3., 0.], [0., 1.],
                     [2., 2.], [4., 3.]])
    actual, digest, groups = interval.bootstrap_group_means(
        cost, group, draws=7, seed=7, batch_size=3)
    rng = np.random.default_rng(7)
    expected = []
    stream = hashlib.sha256()
    for block_size in (3, 3, 1):
        sampled = rng.integers(0, 4, size=(block_size, 4), dtype=np.int32)
        stream.update(np.ascontiguousarray(sampled).tobytes())
        for draw in sampled:
            person_indices = np.concatenate([np.flatnonzero(group == (10, 20, 30, 40)[i])
                                             for i in draw])
            expected.append(cost[person_indices].mean(axis=0))
    assert groups == 4
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=0, atol=1e-15)
    assert digest == stream.hexdigest()


def test_simultaneous_one_sided_boundaries_keep_unresolved_cost():
    interval = importlib.import_module("utility_intervals")
    point = np.array([0.0, 0.02, 0.009])
    bootstrap = np.array([[0.0, 0.02, 0.005], [0.0, 0.02, 0.013],
                          [0.0, 0.02, 0.006], [0.0, 0.02, 0.012],
                          [0.0, 0.02, 0.004], [0.0, 0.02, 0.014]])
    result = interval.simultaneous_cost_bounds(
        point, bootstrap, np.array([0.01, 0.01, 0.01]))
    assert [row["status"] for row in result["rows"]] == [
        "PASS", "VIOLATION", "UNRESOLVED"]
    assert result["rows"][0]["lower"] == result["rows"][0]["upper"] == 0.0
    assert result["rows"][1]["lower"] == result["rows"][1]["upper"] == 0.02
    assert result["rows"][2]["lower"] < 0.01 < result["rows"][2]["upper"]
