"""Pure final development endpoint and inference-array construction.

Inputs are the actual frozen releases, reference probabilities and saved
reader scores, all aligned to INNER_SELECTION. This module opens no role
files, chooses no mask or reader, and performs no fitting.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from attack_eval import oriented_auc
from common_eval import fit_constant_probability
from endpoints import public_per_class, score_task_bundle


RULES = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")
VIEWS = ("income", "occupation_group", "coalition")
REFERENCES = ("U0", "Ucal", "F")


def _int64_record(record: Mapping[str, object], fields: tuple[str, ...]) -> int:
    if not {"row_id", "unit", *fields} <= set(record):
        raise ValueError("endpoint role record lacks required fields")
    row = np.asarray(record["row_id"])
    unit = np.asarray(record["unit"])
    if (row.dtype != np.dtype("int64") or unit.dtype != np.dtype("int64")
            or row.ndim != 1 or unit.shape != row.shape
            or len(np.unique(row)) != len(row)):
        raise ValueError("endpoint role identity is not exact ordered int64")
    for field in fields:
        value = np.asarray(record[field])
        if value.dtype != np.dtype("int64") or value.shape != row.shape:
            raise ValueError("endpoint labels need exact aligned int64 arrays")
    return len(row)


def _same_people(first: Mapping[str, object], second: Mapping[str, object]) -> None:
    for key in ("row_id", "unit"):
        if not np.array_equal(first[key], second[key]):
            raise ValueError("endpoint rows or groups differ from admitted order")


def _prob(record: Mapping[str, object], n: int, width: int) -> np.ndarray:
    value = np.asarray(record["prob"])
    if (value.dtype != np.dtype("float64") or value.shape != (n, width)
            or not np.isfinite(value).all() or np.any((value < 0) | (value > 1))
            or not np.allclose(value.sum(axis=1), 1, rtol=0, atol=1e-8)):
        raise ValueError("endpoint probability vector differs from actual release")
    return value


def score_selected_rule(
    *, seed: int, rule_id: str, selection: Mapping[str, object],
    truth: Mapping[str, object], fit_labels: Mapping[str, object],
    candidate: Mapping[str, Mapping[str, object]],
    references: Mapping[str, Mapping[str, Mapping[str, object]]],
    reader_predictions: Mapping[str, Mapping[str, np.ndarray]],
    v2_status_by_view: Mapping[str, str],
) -> dict:
    """Score a prefit-selected pair on INNER without any new selection."""
    if rule_id not in RULES or selection.get("status") != "SELECTED":
        raise ValueError("only one frozen selected rule can be scored")
    n = _int64_record(truth, ("income", "occupation", "sex"))
    _int64_record(fit_labels, ("income", "occupation"))
    if (set(candidate) != {"income", "occupation_group"}
            or set(references) != {"income", "occupation_group"}
            or set(reader_predictions) != set(VIEWS)
            or set(v2_status_by_view) != set(VIEWS)):
        raise ValueError("candidate/reference/reader/control view coverage differs")
    if any(v2_status_by_view[view] != "PASS" for view in VIEWS):
        raise ValueError("real V2 controls failed or are incomplete")
    task = {}
    for purpose, label, width in (("income", "income", 2),
                                  ("occupation_group", "occupation", 6)):
        local = candidate[purpose]
        _same_people(truth, local)
        _prob(local, n, width)
        if set(references[purpose]) != set(REFERENCES):
            raise ValueError("all U0/Ucal/F references are required")
        for ref in REFERENCES:
            _same_people(truth, references[purpose][ref])
            _prob(references[purpose][ref], n, width)
        constant = fit_constant_probability(fit_labels[label], task=label)
        scored = score_task_bundle(
            local, truth, references[purpose], task=label,
            fit_constant_prob=constant,
        )
        candidate_metrics = dict(scored["candidate"])
        per_class = public_per_class(candidate_metrics)
        candidate_metrics.pop("per_class_recall")
        task[purpose] = {
            "candidate": candidate_metrics,
            "references": scored["references"],
            "point_eligible_all_references":
                scored["point_eligible_all_references"],
            "per_class": per_class,
            "scorer_task_literal": label,
        }
    sex = np.asarray(truth["sex"])
    if not np.array_equal(np.unique(sex), [0, 1]):
        raise ValueError("INNER SEX labels need both classes")
    recovery = {}
    for view in VIEWS:
        bank = reader_predictions[view]
        if not isinstance(bank, Mapping) or not bank:
            raise ValueError("complete-view reader bank missing")
        readers = {}
        for reader_id, raw_score in bank.items():
            scores = np.asarray(raw_score)
            if (not isinstance(reader_id, str) or not reader_id
                    or scores.dtype != np.dtype("float64")
                    or scores.shape != (n,) or not np.isfinite(scores).all()
                    or np.any((scores < 0) | (scores > 1))):
                raise ValueError("saved reader score is not a complete aligned vector")
            readers[reader_id] = oriented_auc(sex, scores)
        recovery[view] = {
            "primary_envelope_auc": max(
                row["fixed_alias_max_auc"] for row in readers.values()),
            "readers": readers, "reader_ids": list(bank),
            "v2_status": v2_status_by_view[view],
        }
    keys = ("pair_id", "income_mask_id", "occupation_mask_id",
            "union_sha256", "recipient_capacity_signatures",
            "union_capacity_signature")
    if any(key not in selection for key in keys):
        raise ValueError("FIT selection lacks capacity or mask identity")
    return {
        "schema": "pcrl-focused-rule-endpoint-v1",
        "status": "DEVELOPMENT_POINT_ONLY", "seed": seed,
        "rule_id": rule_id, **{key: selection[key] for key in keys},
        "task": task, "recovery": recovery,
    }
