"""Independent numerical replay of one frozen pair's INNER endpoints.

Inputs are admitted arrays supplied by the guarded M5 worker. This module
imports no production evaluator, attacker scorer, or selector.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .independent_metrics import binary_auc, task_metrics


PURPOSES = (("income", "income", 2), ("occupation_group", "occupation", 6))
REFERENCES = ("U0", "Ucal", "F")
VIEWS = ("income", "occupation_group", "coalition")
CAPS = {"accuracy_difference": -0.010, "nll_excess": 0.010,
        "brier_excess": 0.005}


def _identity(record: Mapping, *, labels: tuple[str, ...] = ()) -> int:
    if not isinstance(record, Mapping):
        raise ValueError("role identity needs a record")
    row = np.asarray(record["row_id"])
    unit = np.asarray(record["unit"])
    if (row.dtype != np.dtype("int64") or unit.dtype != np.dtype("int64")
            or row.ndim != 1 or unit.shape != row.shape or not len(row)
            or len(np.unique(row)) != len(row)):
        raise ValueError("ordered row/group identity needs exact int64 arrays")
    for name in labels:
        value = np.asarray(record[name])
        if value.dtype != np.dtype("int64") or value.shape != row.shape:
            raise ValueError(f"{name} needs exact aligned int64 labels")
    return len(row)


def _same_people(first: Mapping, second: Mapping) -> None:
    _identity(second)
    if (not np.array_equal(first["row_id"], second["row_id"])
            or not np.array_equal(first["unit"], second["unit"])):
        raise ValueError("role row/group identity differs from INNER")


def _probability(record: Mapping, n: int, width: int) -> np.ndarray:
    probability = np.asarray(record["prob"])
    if (probability.dtype != np.dtype("float64")
            or probability.shape != (n, width)
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
            or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-8)):
        raise ValueError("release probability needs exact float64 simplex rows")
    return probability


def _public_classes(metrics: Mapping) -> list[dict]:
    out = []
    for cls, (support, recall) in enumerate(zip(
            metrics["class_counts"], metrics["per_class_recall"], strict=True)):
        if support == 0:
            status, public = "UNDEFINED_NO_SUPPORT", None
        elif support < 5:
            status, public = "SUPPRESSED_SUPPORT_LT5", None
        else:
            status, public = "REPORTED", float(recall)
        out.append({"class_id": cls, "support": support,
                    "recall": public, "recall_status": status})
    return out


def replay_pair_endpoint(*, seed: int, pair_id: str, truth: Mapping,
                         fit_labels: Mapping, candidate: Mapping,
                         references: Mapping, reader_scores: Mapping,
                         v2_status_by_view: Mapping) -> dict:
    """Recompute tasks, FIT-majority constants and fixed-reader AUC envelopes."""
    if (type(seed) is not int or seed not in (0, 1, 2)
            or not isinstance(pair_id, str) or not pair_id):
        raise ValueError("frozen seed/pair identity invalid")
    n = _identity(truth, labels=("income", "occupation", "sex"))
    if (set(candidate) != {item[0] for item in PURPOSES}
            or set(references) != set(candidate)
            or set(reader_scores) != set(VIEWS)
            or set(v2_status_by_view) != set(VIEWS)):
        raise ValueError("task/reference/reader view coverage incomplete")
    if any(v2_status_by_view[view] != "PASS" for view in VIEWS):
        raise ValueError("V2 control failed or is incomplete")
    tasks = {}
    for purpose, label, width in PURPOSES:
        fit = np.asarray(fit_labels[label])
        if (fit.dtype != np.dtype("int64") or fit.ndim != 1 or not len(fit)
                or np.any((fit < 0) | (fit >= width))):
            raise ValueError("FIT task labels need exact int64 class values")
        _same_people(truth, candidate[purpose])
        cand_prob = _probability(candidate[purpose], n, width)
        y = np.asarray(truth[label])
        candidate_metrics = task_metrics(y, cand_prob, width=width)
        classes = _public_classes(candidate_metrics)
        candidate_metrics.pop("per_class_recall")
        majority = int(np.argmax(np.bincount(fit, minlength=width)))
        constant_accuracy = float(np.mean(y == majority))
        candidate_metrics["constant_accuracy"] = constant_accuracy
        candidate_metrics["accuracy_gain_over_constant"] = (
            candidate_metrics["accuracy"] - constant_accuracy)
        if set(references[purpose]) != set(REFERENCES):
            raise ValueError("U0/Ucal/F references incomplete")
        comparisons = {}
        for name in REFERENCES:
            record = references[purpose][name]
            _same_people(truth, record)
            metric = task_metrics(y, _probability(record, n, width), width=width)
            difference = {
                "accuracy_difference": candidate_metrics["accuracy"] - metric["accuracy"],
                "nll_excess": candidate_metrics["nll"] - metric["nll"],
                "brier_excess": candidate_metrics["brier"] - metric["brier"],
            }
            comparisons[name] = {
                "accuracy": metric["accuracy"], "nll": metric["nll"],
                "brier": metric["brier"], **difference,
                "point_eligible": (
                    difference["accuracy_difference"] >= CAPS["accuracy_difference"]
                    and difference["nll_excess"] <= CAPS["nll_excess"]
                    and difference["brier_excess"] <= CAPS["brier_excess"]),
            }
        tasks[purpose] = {
            "candidate": candidate_metrics, "references": comparisons,
            "point_eligible_all_references": all(
                item["point_eligible"] for item in comparisons.values()),
            "per_class": classes, "scorer_task_literal": label,
            "fit_majority_constant_baseline": {
                "status": "FIT_MAJORITY_REPLAYED",
                "predicted_class": majority,
                "accuracy": constant_accuracy,
                "accuracy_gain": candidate_metrics["accuracy_gain_over_constant"],
            },
        }
    sex = np.asarray(truth["sex"])
    if not np.array_equal(np.unique(sex), [0, 1]):
        raise ValueError("INNER SEX needs both classes")
    recovery = {}
    for view in VIEWS:
        bank = reader_scores[view]
        if not isinstance(bank, Mapping) or not bank:
            raise ValueError("complete reader bank missing")
        readers = {}
        for reader_id, raw in bank.items():
            score = np.asarray(raw)
            if (not isinstance(reader_id, str) or not reader_id
                    or score.dtype != np.dtype("float64") or score.shape != (n,)
                    or not np.isfinite(score).all()
                    or np.any((score < 0) | (score > 1))):
                raise ValueError("reader score needs exact aligned float64 probabilities")
            positive = binary_auc(sex, score)
            readers[reader_id] = {
                "positive_alias_auc": positive,
                "reverse_alias_auc": 1.0 - positive,
                "fixed_alias_max_auc": max(positive, 1.0 - positive),
            }
        recovery[view] = {
            "primary_envelope_auc": max(
                item["fixed_alias_max_auc"] for item in readers.values()),
            "readers": readers, "reader_ids": list(bank),
            "v2_status": v2_status_by_view[view],
        }
    return {"schema": "pcrl-focused-m5-pair-endpoint-v1", "status": "REPLAYED",
            "seed": seed, "pair_id": pair_id, "task": tasks,
            "recovery": recovery}
