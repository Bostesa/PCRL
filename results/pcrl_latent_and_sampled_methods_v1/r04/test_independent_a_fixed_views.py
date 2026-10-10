"""Data-free fixtures for independently encoding fixed A reference wires."""

import numpy as np

from .independent_a_fixed_views import build_fixed_views


def _release(code1, code2, *, continuous=False):
    n = len(code1)
    return {
        "row_id": np.arange(10, 10 + n, dtype=np.int64),
        "unit": np.arange(20, 20 + n, dtype=np.int64),
        "prob_r1": np.tile(np.array([[0.8, 0.2]], np.float64), (n, 1)),
        "prob_r2": np.tile(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]], np.float64), (n, 1)),
        "decision_r1": np.zeros(n, np.int64),
        "decision_r2": np.full(n, 5, np.int64),
        "code_r1": np.full(n, -1, np.int64) if continuous else np.array(code1, np.int64),
        "code_r2": np.full(n, -1, np.int64) if continuous else np.array(code2, np.int64),
    }


def _receipt(continuous=False):
    return {
        "interface_kind": "continuous" if continuous else "finite",
        "release_id": "tiny-public-law",
        "source_binding_sha256": "a" * 64,
        "bank_sha256_by_recipient": {"r1": None if continuous else "b" * 64,
                                     "r2": None if continuous else "c" * 64},
    }


def test_finite_unknown_inner_code_keeps_person_and_lsb_bits():
    att = _release([0, 1], [1, 0])
    inner = _release([2, 1], [1, 0])
    q = {"r1": np.zeros((3, 19)), "r2": np.zeros((2, 19)),
         "pair": np.zeros((3, 2, 19)),
         "r1_mass": np.ones(3), "r2_mass": np.ones(2),
         "pair_mass": np.ones((3, 2))}
    views = build_fixed_views(att, inner, _receipt(), bank_sizes=(3, 2), q_table=q)
    assert views["r1"]["inner"]["x"].shape[0] == 2
    assert views["r1"]["inner"]["cell_key"].tolist() == ['["2"]', '["1"]']
    assert views["r1"]["inner"]["x_code_bits"].tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert views["pair"]["inner"]["cell_key"].tolist() == ['["2","1"]', '["1","0"]']
    assert views["r1"]["inner"]["x_defense_aware"].shape[1] == views["r1"]["inner"]["x"].shape[1] + 19


def test_continuous_defense_view_uses_only_public_log_probabilities():
    att = _release([0, 0], [0, 0], continuous=True)
    inner = _release([0, 0], [0, 0], continuous=True)
    views = build_fixed_views(att, inner, _receipt(True), bank_sizes=None, q_table=None)
    assert views["r1"]["inner"]["cell_key"] is None
    assert views["r1"]["inner"]["x_code_bits"] is None
    np.testing.assert_array_equal(views["r1"]["inner"]["x_defense_aware"][:, -2:],
                                  np.log(inner["prob_r1"]))
    np.testing.assert_array_equal(views["pair"]["inner"]["x_defense_aware"][:, -8:],
                                  np.column_stack((np.log(inner["prob_r1"]),
                                                   np.log(inner["prob_r2"]))))


def test_zero_support_q_keeps_the_emitted_pair_and_attaches_zero_features():
    att = _release([0, 1], [1, 0])
    inner = _release([2, 1], [1, 0])
    q = {"r1": np.zeros((3, 19)), "r2": np.zeros((2, 19)),
         "pair": np.zeros((3, 2, 19)),
         "r1_mass": np.ones(3), "r2_mass": np.ones(2),
         "pair_mass": np.ones((3, 2))}
    q["pair_mass"][2, 1] = 0
    views = build_fixed_views(att, inner, _receipt(), bank_sizes=(3, 2), q_table=q)
    assert views["pair"]["zero_q_support_count"]["inner"] == 1
    assert views["pair"]["inner"]["x"].shape[0] == 2
    np.testing.assert_array_equal(views["pair"]["inner"]["x_defense_aware"][0, -19:],
                                  np.zeros(19))
