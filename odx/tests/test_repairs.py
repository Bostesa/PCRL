"""Regression tests for the reporting defects repaired on this branch (2026-10-03)."""
import numpy as np
import pytest


def test_role_labels_not_truncated():
    """Bug: roles stored as <U16 truncated 'excluded_exposure' (17 chars) -> exclusion counts reported as 0."""
    arr = np.array(["defense_fit", "assessment"]).astype("<U32")
    arr[0] = "excluded_exposure"
    assert arr[0] == "excluded_exposure"
    bad = np.array(["defense_fit"]).astype("<U16")
    bad[0] = "excluded_exposure"
    assert bad[0] != "excluded_exposure"          # documents the original failure mode


def test_load_world_counts_exclusions_from_identities():
    import oar.study as S
    if not (S.BENCH / "inputs" / "adult_labels.npz").exists():
        pytest.skip("private inputs not present")
    for ds, n in (("adult", 17), ("hmda", 42)):
        W = S.load_world(ds)
        assert int((W["role"] == "excluded_exposure").sum()) == n
        assert int(W["exposed"][np.isin(W["role_bench"], ("attacker_fit", "attacker_val", "assessment"))].sum()) == n


def test_feature_equality_counts_rows_and_distinct_vectors():
    from odx.equality import feature_equality_counts
    ref = np.array([[0.0, 1.0], [2.0, 3.0]])
    q = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [5.0, 5.0], [2.0, 3.0]])
    c = feature_equality_counts(q, ref)
    assert c == {"query_rows": 5, "affected_rows": 4, "distinct_query_vectors": 3, "distinct_vectors_shared_with_reference": 2}
