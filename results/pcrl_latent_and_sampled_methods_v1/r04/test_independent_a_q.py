"""Actual public code lookup keeps zero-prototype-support people."""

from __future__ import annotations

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_q import actual_q_features


def test_zero_support_attaches_all_zero_row_without_dropping_code():
    local = np.zeros((2, 19), np.float64)
    local[0, 2] = .5
    pair = np.zeros((2, 2, 19), np.float64)
    pair[0, 0, 2] = .5
    table = {"r1": local, "r1_mass": np.array([.5, 0.]),
             "r2": local, "r2_mass": np.array([.5, 0.]),
             "pair": pair, "pair_mass": np.array([[.5, 0.], [0., 0.]])}
    codes = np.array([0, 1, 0], np.int64)
    result, unsupported = actual_q_features(table, view="r1", code1=codes)
    assert result.shape == (3, 19) and unsupported == 1
    assert result[1].sum() == 0 and result[0, 2] == .5
    both, unsupported_pair = actual_q_features(
        table, view="pair", code1=codes, code2=np.array([0, 0, 1], np.int64))
    assert both.shape == (3, 19) and unsupported_pair == 2
