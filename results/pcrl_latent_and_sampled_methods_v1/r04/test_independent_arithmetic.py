"""Finite fixtures fixed before real candidate outcomes."""

import math

import numpy as np
import pytest

from independent_arithmetic import (
    aggregate_public_actions,
    biased_mmd2,
    binary_auc,
    expected_task_cost,
    gaussian_binary_kernel,
    pack_ten_bit_pair,
    product_pair_mmd2,
)


def test_xor_single_views_match_but_pair_is_revealing():
    # SEX 0: 00/11; SEX 1: 01/10. Each recipient bit is marginally fair.
    group0 = [([0], [0]), ([1], [1])]
    group1 = [([0], [1]), ([1], [0])]
    sigma = 1.0
    r = gaussian_binary_kernel([0], [1], sigma)
    assert biased_mmd2([[0], [1]], [[0], [1]], [sigma]) == pytest.approx(0)
    assert product_pair_mmd2(group0, group1, [sigma]) == pytest.approx((1 - r) ** 2)


def test_proposed_four_scale_xor_kernel():
    gamma = (0.25, 0.5, 1.0, 2.0)
    bandwidths = [1 / math.sqrt(2 * g) for g in gamma]
    group0 = [([0], [0]), ([1], [1])]
    group1 = [([0], [1]), ([1], [0])]
    mixed_off_diagonal = sum(math.exp(-g) for g in gamma) / 4
    assert biased_mmd2([[0], [1]], [[0], [1]], bandwidths) == pytest.approx(0)
    assert product_pair_mmd2(group0, group1, bandwidths) == pytest.approx(
        (1 - mixed_off_diagonal) ** 2
    )


def test_biased_mmd_includes_diagonals_and_is_nonnegative():
    r = math.exp(-0.5)
    assert biased_mmd2([[0]], [[1]], [1.0]) == pytest.approx(2 * (1 - r))
    assert biased_mmd2([[0], [1]], [[0], [1]], [1.0]) == pytest.approx(0)


def test_channel_single_draw_expected_loss_differs_from_mean_vector_loss():
    q = [0.5, 0.5]
    bank = [[0.1, 0.9], [0.9, 0.1]]
    nll, brier = expected_task_cost(q, bank, 1)
    assert nll == pytest.approx(-0.5 * (math.log(0.9) + math.log(0.1)))
    assert brier == pytest.approx(0.82)
    averaged = np.average(np.asarray(bank), axis=0, weights=q)
    assert -math.log(averaged[1]) < nll
    assert sum((averaged - [0, 1]) ** 2) < brier


def test_duplicate_actions_use_actual_global_public_identity():
    assert aggregate_public_actions([0.2, 0.3, 0.5], [7, 7, 9]) == {7: 0.5, 9: 0.5}
    assert aggregate_public_actions([0.2, 0.3, 0.5], [(7, 8), (7, 8), (9, 8)]) == {
        (7, 8): 0.5, (9, 8): 0.5
    }


def test_auc_ties_and_reversal():
    assert binary_auc([0.2, 0.2, 0.9, 0.1], [0, 1, 1, 0]) == pytest.approx(0.875)
    assert binary_auc([0.8, 0.8, 0.1, 0.9], [0, 1, 1, 0]) == pytest.approx(0.125)


def test_pair_code_packing_promotes_before_multiplication():
    # A uint16 multiplication would alias these two different public pairs.
    packed = pack_ten_bit_pair(np.array([0, 64, 1023], dtype=np.uint16),
                               np.array([1, 1, 1023], dtype=np.uint16))
    assert packed.tolist() == [1, 65537, 1048575]
    assert len(np.unique(packed)) == 3
