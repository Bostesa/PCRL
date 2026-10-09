"""Data-free checks of R3 privacy endpoint arithmetic."""

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r3.privacy_math import (
    alias_auc, attack_tune_indices, rank_auc, independent_group_shuffle,
)


def test_rank_auc_handles_perfect_reverse_constant_and_ties():
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    assert rank_auc(y, [0, .1, .8, 1]) == 1
    assert rank_auc(y, [1, .8, .1, 0]) == 0
    assert rank_auc(y, [.5] * 4) == .5
    assert rank_auc(np.array([0, 1, 0, 1]), [0, 0, 1, 1]) == .5
    assert alias_auc(y, [1, .8, .1, 0])["fixed_alias_max_auc"] == 1
    with pytest.raises(ValueError):
        rank_auc(np.zeros(4, dtype=np.int64), [.1, .2, .3, .4])


def test_fixed_att_group_split_is_disjoint_and_deterministic():
    group = np.arange(4061, dtype=np.int64)
    a_fit, a_tune = attack_tune_indices(group)
    b_fit, b_tune = attack_tune_indices(group)
    assert len(a_fit) == 3046 and len(a_tune) == 1015
    np.testing.assert_array_equal(a_fit, b_fit)
    np.testing.assert_array_equal(a_tune, b_tune)
    assert not np.intersect1d(a_fit, a_tune).size
    assert len(np.union1d(a_fit, a_tune)) == 4061


def test_group_shuffle_preserves_whole_equal_size_vectors():
    groups = np.array([10, 10, 20, 20, 30, 30, 40], dtype=np.int64)
    labels = np.array([0, 1, 1, 1, 0, 0, 1], dtype=np.int64)
    first = independent_group_shuffle(labels, groups)
    second = independent_group_shuffle(labels, groups)
    np.testing.assert_array_equal(first, second)
    assert first[6] == 1  # the size-one bucket can only map to itself
    assert sorted(tuple(first[i:i + 2]) for i in (0, 2, 4)) == [
        (0, 0), (0, 1), (1, 1)]
