"""Independent point endpoint replay for C's saved INNER rows.

This module consumes verified release wires and frozen reader predictions. It
imports no Team C or shared scorer. Adult execution requires an S3 lease.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from replay_c_readers_v1 import READERS, oriented_auc


VIEWS = ("r1", "r2", "pair")
TASKS = (("income", 2), ("occupation", 6))


def task_metrics(probability: np.ndarray, label: np.ndarray, width: int) -> dict:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(label, dtype=np.int64)
    if (p.ndim != 2 or p.shape[1] != width or y.shape != (len(p),) or
            len(p) == 0 or not np.isfinite(p).all() or
            np.any((p < 0) | (p > 1)) or
            not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0) or
            np.any((y < 0) | (y >= width))):
        raise ValueError("C task probability or label geometry differs")
    decision = p.argmax(axis=1)
    observed = p[np.arange(len(y)), y]
    onehot = np.eye(width, dtype=np.float64)[y]
    return {
        "rows": len(y),
        "accuracy": float(np.mean(decision == y)),
        "nll": float(np.mean(-np.log(np.maximum(observed, 1e-12)))),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "per_class_recall": [
            float(np.mean(decision[y == label_id] == label_id))
            if np.any(y == label_id) else None
            for label_id in range(width)
        ],
        "class_counts": [int(np.count_nonzero(y == label_id))
                         for label_id in range(width)],
    }


def pair_aliases(local_one: Mapping[str, np.ndarray],
                 local_two: Mapping[str, np.ndarray],
                 coalition: Mapping[str, np.ndarray]) -> tuple[dict, list[dict]]:
    """Attach the twelve zero-fit local predictions to the coalition slate."""
    if any(tuple(bank) != READERS for bank in (local_one, local_two, coalition)):
        raise ValueError("C six-reader order differs")
    out = dict(coalition)
    aliases = []
    for recipient, source in ((1, local_one), (2, local_two)):
        for reader in READERS:
            alias = f"IGNORE_R{2 if recipient == 1 else 1}__{reader}"
            if alias in out:
                raise ValueError("C ignore alias collision")
            out[alias] = source[reader]
            aliases.append({"alias": alias, "source_view": f"r{recipient}",
                            "source_reader": reader, "new_fit": False})
    return out, aliases


def score_view(predictions: Mapping[str, np.ndarray],
               sex: np.ndarray, winner: str, orientation: str) -> dict:
    if (not predictions or winner not in predictions or
            orientation not in ("positive", "reverse")):
        raise ValueError("C frozen validation winner or orientation differs")
    scores = {reader: oriented_auc(sex, value) for reader, value in
              predictions.items()}
    selected = scores[winner]["positive_alias_auc"]
    return {"schema": "pcrl-portfolio-attack-slate-v1",
            "primary_envelope_auc": max(item["fixed_alias_max_auc"]
                                        for item in scores.values()),
            "readers": scores,
            "validation_winner": winner,
            "validation_winner_orientation": orientation,
            "validation_winner_inner_auc": selected if orientation == "positive"
            else 1.0 - selected}


def _winner(validation: Mapping[str, Mapping[str, object]]) -> tuple[str, str]:
    if not validation:
        raise ValueError("C frozen validation bank is empty")
    winner, best = next(iter(validation)), -1.0
    for name, item in validation.items():
        value = float(item["fixed_alias_max_auc"])
        if value > best + 1e-12:
            winner, best = name, value
    return winner, str(validation[winner]["selected_orientation"])


def _same(actual: object, expected: object, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"C saved {path} keys differ")
        for key, value in expected.items():
            _same(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"C saved {path} list differs")
        for index, value in enumerate(expected):
            _same(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        if (not isinstance(actual, (float, int)) or
                not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"C saved {path} numerical endpoint differs")
    elif actual != expected:
        raise ValueError(f"C saved {path} differs")


def replay_score_receipt(
    receipt: Mapping[str, object],
    probabilities: Mapping[str, np.ndarray],
    task_labels: Mapping[str, np.ndarray],
    sex: np.ndarray,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    frozen_validation: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict:
    """Replay all score-stage values from independently verified inputs."""
    if (set(probabilities) != {task for task, _ in TASKS} or
            set(task_labels) != {task for task, _ in TASKS} or
            set(predictions) != set(VIEWS) or
            set(frozen_validation) != set(VIEWS)):
        raise ValueError("C endpoint input bank differs")
    task = {name: task_metrics(probabilities[name], task_labels[name], width)
            for name, width in TASKS}
    pair, aliases = pair_aliases(predictions["r1"], predictions["r2"],
                                 predictions["pair"])
    banks = dict(predictions)
    banks["pair"] = pair
    checks = {view: dict(frozen_validation[view]) for view in VIEWS}
    if any(tuple(checks[view]) != READERS for view in VIEWS):
        raise ValueError("C frozen six-reader validation order differs")
    for alias in aliases:
        checks["pair"][alias["alias"]] = checks[alias["source_view"]][
            alias["source_reader"]]
    scores = {}
    for view in VIEWS:
        winner, orientation = _winner(checks[view])
        scores[view] = score_view(banks[view], sex, winner, orientation)
    if receipt.get("schema") != "team-c-arm-score-stage-v1" or receipt.get("status") != "SCORED":
        raise ValueError("C saved score-stage schema/status differs")
    _same(receipt.get("task_metrics"), task, "task_metrics")
    _same(receipt.get("view_scores"), scores, "view_scores")
    _same(receipt.get("ignore_aliases"), aliases, "ignore_aliases")
    return {"task_metrics": task, "view_scores": scores,
            "ignore_aliases": aliases}
