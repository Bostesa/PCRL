"""Pure focused-study task endpoints against U0, Ucal and full-input F.

This module reads no role files. Callers supply aligned admitted records and
the fitting-role constant probability; all three utility references are active.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from common_eval import assert_same_people, task_metrics


REFERENCES = ("U0", "Ucal", "F")
CAPS = {"accuracy_difference": -0.010, "nll_excess": 0.010,
        "brier_excess": 0.005}


def score_task_bundle(
    candidate: Mapping[str, object], truth: Mapping[str, object],
    references: Mapping[str, Mapping[str, object]], *, task: str,
    fit_constant_prob: np.ndarray,
) -> dict:
    """Score one actual calibrated task release and its three fixed references."""
    if task not in ("income", "occupation") or set(references) != set(REFERENCES):
        raise ValueError("task and exactly U0/Ucal/F references required")
    for record in (candidate, truth, *[references[name] for name in REFERENCES]):
        for identity in ("row_id", "unit"):
            values = np.asarray(record[identity])
            if values.dtype != np.dtype("int64") or values.ndim != 1:
                raise ValueError(f"{identity} needs exact int64 identity")
    assert_same_people(candidate, truth, *[references[name] for name in REFERENCES])
    y = np.asarray(truth[task])
    cand = task_metrics(candidate["prob"], y, task=task,
                        constant_prob=fit_constant_prob)
    comparisons = {}
    for name in REFERENCES:
        baseline = task_metrics(references[name]["prob"], y, task=task)
        diff = {"accuracy_difference": cand["accuracy"] - baseline["accuracy"],
                "nll_excess": cand["nll"] - baseline["nll"],
                "brier_excess": cand["brier"] - baseline["brier"]}
        comparisons[name] = {
            "accuracy": baseline["accuracy"], "nll": baseline["nll"],
            "brier": baseline["brier"], **diff,
            "point_eligible": (diff["accuracy_difference"] >= CAPS["accuracy_difference"]
                               and diff["nll_excess"] <= CAPS["nll_excess"]
                               and diff["brier_excess"] <= CAPS["brier_excess"]),
        }
    return {"task": task, "candidate": cand, "references": comparisons,
            "point_eligible_all_references": all(
                comparisons[name]["point_eligible"] for name in REFERENCES)}


def public_per_class(task_metrics_record: Mapping[str, object]) -> list[dict]:
    """Suppress published recall for any class with fewer than five people."""
    counts = list(task_metrics_record["class_counts"])
    recalls = list(task_metrics_record["per_class_recall"])
    if not counts or len(counts) != len(recalls):
        raise ValueError("per-class metric schema differs")
    result = []
    for class_id, (support, recall) in enumerate(zip(counts, recalls)):
        if type(support) is not int or support < 0:
            raise ValueError("invalid class support")
        if support == 0:
            if recall is not None:
                raise ValueError("zero-support class has a recall")
            public_recall, status = None, "UNDEFINED_NO_SUPPORT"
        elif support < 5:
            public_recall, status = None, "SUPPRESSED_SUPPORT_LT5"
        else:
            if type(recall) not in (int, float) or not math.isfinite(recall) or \
                    not 0 <= recall <= 1:
                raise ValueError("supported class lacks a valid recall")
            public_recall, status = float(recall), "REPORTED"
        result.append({"class_id": class_id, "support": support,
                       "recall": public_recall, "recall_status": status})
    return result
