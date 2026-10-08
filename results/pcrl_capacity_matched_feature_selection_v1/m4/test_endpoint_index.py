"""Data-free checks for final point endpoint shape and frozen readers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "results/pcrl_method_portfolio_v1/shared"))

from endpoint_index import score_selected_rule


def _role(n=12):
    return {"row_id": np.arange(n, dtype=np.int64) + 1,
            "unit": np.arange(n, dtype=np.int64) + 101}


def _fixture():
    truth = {**_role(), "income": np.asarray([0, 1] * 6, dtype=np.int64),
             "occupation": np.asarray(list(range(6)) * 2, dtype=np.int64),
             "sex": np.asarray([0, 1] * 6, dtype=np.int64)}
    fit = {**_role(), "income": truth["income"].copy(),
           "occupation": truth["occupation"].copy()}
    income = {**_role(), "prob": np.eye(2, dtype=np.float64)[truth["income"]] * .8 + .1}
    occupation = {**_role(), "prob": np.eye(6, dtype=np.float64)[truth["occupation"]] * .7 + .05}
    candidate = {"income": income, "occupation_group": occupation}
    refs = {name: {ref: record for ref in ("U0", "Ucal", "F")}
            for name, record in candidate.items()}
    reader = {view: {"LR_C1": np.linspace(.1, .9, 12, dtype=np.float64)}
              for view in ("income", "occupation_group", "coalition")}
    choice = {"status": "SELECTED", "pair_id": "M_DROP_A__M_DROP_B",
              "income_mask_id": "M_DROP_A", "occupation_mask_id": "M_DROP_B",
              "union_sha256": "a" * 64,
              "recipient_capacity_signatures": {"income": {}, "occupation_group": {}},
              "union_capacity_signature": {}}
    return truth, fit, candidate, refs, reader, choice


def test_selected_rule_maps_occupation_key_and_suppresses_small_classes():
    truth, fit, candidate, refs, reader, choice = _fixture()
    result = score_selected_rule(
        seed=0, rule_id="J", selection=choice, truth=truth, fit_labels=fit,
        candidate=candidate, references=refs, reader_predictions=reader,
        v2_status_by_view={view: "PASS" for view in reader},
    )
    assert set(result["task"]) == {"income", "occupation_group"}
    assert result["task"]["occupation_group"]["scorer_task_literal"] == "occupation"
    assert all(row["recall"] is None for row in result["task"]["occupation_group"]["per_class"])
    assert all(result["task"][task]["references"][ref]["point_eligible"]
               for task in result["task"] for ref in ("U0", "Ucal", "F"))
    assert result["recovery"]["coalition"]["reader_ids"] == ["LR_C1"]


def test_selected_rule_refuses_incomplete_v2_or_reader_view():
    truth, fit, candidate, refs, reader, choice = _fixture()
    with pytest.raises(ValueError, match="V2 controls"):
        score_selected_rule(
            seed=0, rule_id="J", selection=choice, truth=truth, fit_labels=fit,
            candidate=candidate, references=refs, reader_predictions=reader,
            v2_status_by_view={view: ("FAIL" if view == "coalition" else "PASS")
                               for view in reader},
        )
    reader.pop("coalition")
    with pytest.raises(ValueError, match="coverage"):
        score_selected_rule(
            seed=0, rule_id="J", selection=choice, truth=truth, fit_labels=fit,
            candidate=candidate, references=refs, reader_predictions=reader,
            v2_status_by_view={view: "PASS" for view in
                               ("income", "occupation_group", "coalition")},
        )
