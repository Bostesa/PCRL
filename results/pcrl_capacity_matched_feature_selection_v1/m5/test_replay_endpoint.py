"""Small, independent endpoint checks using synthetic paired people."""

import numpy as np
import pytest

from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_endpoint import (
    replay_pair_endpoint,
)


def _case():
    row = np.arange(6, dtype=np.int64)
    unit = np.arange(10, 16, dtype=np.int64)
    truth = {"row_id": row, "unit": unit,
             "income": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
             "occupation": np.arange(6, dtype=np.int64),
             "sex": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)}
    fit = {"income": np.array([0, 0, 1], dtype=np.int64),
           "occupation": np.array([2, 2, 3], dtype=np.int64)}
    candidate = {}
    references = {}
    for purpose, label, width in (("income", "income", 2),
                                  ("occupation_group", "occupation", 6)):
        y = truth[label]
        good = np.full((6, width), 0.1 / (width - 1), dtype=np.float64)
        good[np.arange(6), y] = 0.9
        record = {"row_id": row.copy(), "unit": unit.copy(), "prob": good}
        candidate[purpose] = record
        uniform = np.full((6, width), 1 / width, dtype=np.float64)
        references[purpose] = {name: {"row_id": row.copy(), "unit": unit.copy(),
                                      "prob": uniform.copy()}
                               for name in ("U0", "Ucal", "F")}
    readers = {view: {"R_A": np.array([0, 1, 0, 1, 0, 1], dtype=np.float64),
                      "R_B": np.full(6, 0.5, dtype=np.float64)}
               for view in ("income", "occupation_group", "coalition")}
    return truth, fit, candidate, references, readers


def test_independent_endpoint_replays_three_active_references_and_reader_envelope():
    truth, fit, candidate, references, readers = _case()
    result = replay_pair_endpoint(
        seed=0, pair_id="pair-0", truth=truth, fit_labels=fit,
        candidate=candidate, references=references, reader_scores=readers,
        v2_status_by_view={view: "PASS" for view in readers},
    )
    assert result["task"]["income"]["point_eligible_all_references"]
    assert set(result["task"]["income"]["references"]) == {"U0", "Ucal", "F"}
    assert result["task"]["income"]["fit_majority_constant_baseline"]["predicted_class"] == 0
    assert result["task"]["occupation_group"]["fit_majority_constant_baseline"]["predicted_class"] == 2
    assert all(item["recall_status"] == "SUPPRESSED_SUPPORT_LT5"
               for item in result["task"]["occupation_group"]["per_class"])
    assert result["recovery"]["coalition"]["primary_envelope_auc"] == 1.0
    assert result["recovery"]["coalition"]["readers"]["R_B"]["fixed_alias_max_auc"] == 0.5


def test_float_identity_and_failed_v2_are_refused_before_scoring():
    truth, fit, candidate, references, readers = _case()
    statuses = {view: "PASS" for view in readers}
    truth["row_id"] = truth["row_id"].astype(np.float64)
    with pytest.raises(ValueError, match="int64"):
        replay_pair_endpoint(seed=0, pair_id="pair-0", truth=truth,
                             fit_labels=fit, candidate=candidate,
                             references=references, reader_scores=readers,
                             v2_status_by_view=statuses)
    truth["row_id"] = np.arange(6, dtype=np.int64)
    statuses["coalition"] = "PENDING"
    with pytest.raises(ValueError, match="V2"):
        replay_pair_endpoint(seed=0, pair_id="pair-0", truth=truth,
                             fit_labels=fit, candidate=candidate,
                             references=references, reader_scores=readers,
                             v2_status_by_view=statuses)


def test_reference_row_order_and_float32_scores_are_refused():
    truth, fit, candidate, references, readers = _case()
    statuses = {view: "PASS" for view in readers}
    references["income"]["F"]["row_id"] = references["income"]["F"]["row_id"][::-1]
    with pytest.raises(ValueError, match="row|identity"):
        replay_pair_endpoint(seed=0, pair_id="pair-0", truth=truth,
                             fit_labels=fit, candidate=candidate,
                             references=references, reader_scores=readers,
                             v2_status_by_view=statuses)
    references["income"]["F"]["row_id"] = truth["row_id"].copy()
    readers["coalition"]["R_A"] = readers["coalition"]["R_A"].astype(np.float32)
    with pytest.raises(ValueError, match="float64"):
        replay_pair_endpoint(seed=0, pair_id="pair-0", truth=truth,
                             fit_labels=fit, candidate=candidate,
                             references=references, reader_scores=readers,
                             v2_status_by_view=statuses)
