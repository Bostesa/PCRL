"""Data-free contract tests for the bank-anchored public wire."""

import numpy as np
import pytest

from bank_anchored.r1_release import (emit_public_pair, release_pair_from_scores,
                                      verify_identity_pair)


def test_emitted_pair_contains_only_code_derived_outputs():
    tables = (
        np.array([[.8, .2], [.2, .8]], dtype=np.float64),
        np.array([[.7, .1, .1, .05, .03, .02], [.1, .7, .1, .05, .03, .02],
                  [.1, .1, .7, .05, .03, .02]], dtype=np.float64),
    )
    left, right = emit_public_pair(
        np.array([1, 0], dtype=np.int64),
        np.array([2, 1], dtype=np.int64), tables,
    )
    for view, table, ids in ((left, tables[0], [1, 0]), (right, tables[1], [2, 1])):
        assert set(view) == {"code", "prob", "decision"}
        assert view["code"].shape == (2, 10)
        assert view["code"].dtype == np.uint8
        assert np.array_equal(view["prob"], table[ids])
        assert np.array_equal(view["decision"], table[ids].argmax(axis=1))


def test_identity_checkpoint_checks_actual_both_recipient_wires():
    tables = (
        np.array([[.8, .2], [.2, .8]], dtype=np.float64),
        np.array([[.7, .1, .1, .05, .03, .02], [.1, .7, .1, .05, .03, .02],
                  [.1, .1, .7, .05, .03, .02]], dtype=np.float64),
    )
    ids = (np.array([1, 0], dtype=np.int64), np.array([2, 1], dtype=np.int64))
    left, right = emit_public_pair(*ids, tables)
    assert verify_identity_pair((left, right), ids, tables)
    changed = dict(right)
    changed["decision"] = changed["decision"].copy()
    changed["decision"][0] = 0
    with pytest.raises(ValueError, match="identity"):
        verify_identity_pair((left, changed), ids, tables)


def test_public_wire_rejects_invalid_or_noncanonical_states():
    tables = (
        np.array([[.8, .2], [.2, .8]], dtype=np.float64),
        np.array([[.7, .1, .1, .05, .03, .02], [.1, .7, .1, .05, .03, .02],
                  [.1, .1, .7, .05, .03, .02]], dtype=np.float64),
    )
    with pytest.raises(ValueError):
        emit_public_pair(np.array([2], dtype=np.int64), np.array([0], dtype=np.int64), tables)
    with pytest.raises(ValueError):
        emit_public_pair(np.array([0], dtype=np.int64), np.array([0], dtype=np.int64),
                         (np.vstack([tables[0][0], tables[0][0]]), tables[1]))
    with pytest.raises(ValueError):
        emit_public_pair(np.array([0.0]), np.array([0], dtype=np.int64), tables)


def test_private_policy_context_never_enters_emitted_wire():
    class _Policy:
        def sample_public_pair(self, score0, score1, cell0, cell1, uniform):
            assert uniform == (.25, .75)
            assert (cell0, cell1) == (4, 9)
            return 1, 2

    tables = (
        np.array([[.8, .2], [.2, .8]], dtype=np.float64),
        np.array([[.7, .1, .1, .05, .03, .02], [.1, .7, .1, .05, .03, .02],
                  [.1, .1, .7, .05, .03, .02]], dtype=np.float64),
    )
    wires = release_pair_from_scores(
        _Policy(), np.array([[.2, .8]], dtype=np.float64),
        np.array([[.1, .2, .7, 0., 0., 0.]], dtype=np.float64),
        np.array([4], dtype=np.int64), np.array([9], dtype=np.int64),
        tables, np.array([[.25, .75]], dtype=np.float64))
    assert all(set(wire) == {"code", "prob", "decision"} for wire in wires)
    assert np.array_equal(wires[0]["decision"], [1])
    assert np.array_equal(wires[1]["decision"], [2])


def test_private_policy_cannot_change_source_decision():
    class _Bad:
        def sample_public_pair(self, *args):
            return 0, 0

    tables = (
        np.array([[.8, .2], [.2, .8]], dtype=np.float64),
        np.array([[.7, .1, .1, .05, .03, .02], [.1, .7, .1, .05, .03, .02],
                  [.1, .1, .7, .05, .03, .02]], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="class"):
        release_pair_from_scores(
            _Bad(), np.array([[.2, .8]], dtype=np.float64),
            np.array([[.1, .2, .7, 0., 0., 0.]], dtype=np.float64),
            np.array([4], dtype=np.int64), np.array([9], dtype=np.int64),
            tables, np.array([.25], dtype=np.float64))
