"""Pure shared calibration and task utility functions for admitted in-memory rows.

This module opens no data. A scientific runner must read roles through
SafeRoleStore, acquire the shared queue for every fit, and verify the effective
S1 seal before calling fit_global_temperature on Adult arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from safe_data_guard import sha_arr, sha_file


CAL_SCHEMA = "pcrl-portfolio-global-temp-v1"
CLIP = 1e-12
TEMP_BOUNDS = (0.25, 4.0)
TASK_WIDTH = {"income": 2, "occupation": 6}


def assert_same_people(*records: dict) -> None:
    """Refuse row permutations, group swaps, or mismatched recipient views."""
    if not records:
        raise ValueError("at least one record required")
    rid = np.asarray(records[0]["row_id"], dtype=np.int64)
    grp = np.asarray(records[0]["unit"], dtype=np.int64)
    if rid.ndim != 1 or grp.shape != rid.shape or np.unique(rid).size != rid.size:
        raise ValueError("invalid row/group identities")
    for record in records[1:]:
        if not np.array_equal(np.asarray(record["row_id"], dtype=np.int64), rid) or \
                not np.array_equal(np.asarray(record["unit"], dtype=np.int64), grp):
            raise ValueError("ordered row/group identity mismatch")


def _probabilities(prob, width: int) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != width or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("invalid probability schema")
    if not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("probability rows do not sum to one")
    return p


def _labels(labels, n: int, width: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (n,) or np.any((y < 0) | (y >= width)):
        raise ValueError("invalid task label schema")
    return y


def _log_input(p: np.ndarray) -> np.ndarray:
    clipped = np.maximum(p, CLIP)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return np.log(clipped)


def _scaled_from_logs(logp: np.ndarray, alpha: float) -> np.ndarray:
    scaled = alpha * logp
    scaled -= scaled.max(axis=1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(axis=1, keepdims=True)


def fit_global_temperature(prob, labels, *, task: str, arm_id: str, model_seed: int,
                           row_id, unit, draw_id: str = "deterministic") -> dict:
    """Fit one held-out NLL-optimal inverse temperature, no grid or SEX input."""
    if task not in TASK_WIDTH or not arm_id or not draw_id:
        raise ValueError("task/arm/draw identity required")
    p = _probabilities(prob, TASK_WIDTH[task])
    y = _labels(labels, len(p), TASK_WIDTH[task])
    rid = np.asarray(row_id, dtype=np.int64)
    grp = np.asarray(unit, dtype=np.int64)
    if rid.shape != y.shape or grp.shape != y.shape or np.unique(rid).size != rid.size:
        raise ValueError("calibration role identities invalid")
    logp = _log_input(p)

    def derivative(alpha: float) -> float:
        q = _scaled_from_logs(logp, alpha)
        return float(np.mean(np.sum(q * logp, axis=1) - logp[np.arange(len(y)), y]))

    lo, hi = TEMP_BOUNDS
    dlo, dhi = derivative(lo), derivative(hi)
    if dlo >= 0:
        alpha, boundary = lo, "LOWER"
    elif dhi <= 0:
        alpha, boundary = hi, "UPPER"
    else:
        for _ in range(80):
            mid = (lo + hi) / 2
            if derivative(mid) > 0:
                hi = mid
            else:
                lo = mid
        alpha, boundary = (lo + hi) / 2, "INTERIOR"
    q = p.copy() if alpha == 1.0 else _scaled_from_logs(logp, alpha)
    if not np.array_equal(np.argmax(q, axis=1), np.argmax(p, axis=1)):
        raise ValueError("positive temperature changed decisions")
    return {"schema": CAL_SCHEMA, "task": task, "arm_id": arm_id, "model_seed": int(model_seed),
            "fit_role": "CALIBRATION_HELDOUT", "draw_id": draw_id,
            "alpha": float(alpha), "bounds": list(TEMP_BOUNDS), "boundary": boundary,
            "fit_rows": int(len(y)), "fit_row_unit_pair_sha256": sha_arr(np.column_stack((rid, grp))),
            "input_probability_sha256": sha_arr(p), "task_label_sha256": sha_arr(y),
            "fitted_nll": float(np.mean(-np.log(np.maximum(q[np.arange(len(y)), y], CLIP)))),
            "derivative_at_alpha": derivative(alpha), "argmax_preserved": True}


def apply_global_temperature(prob, record: dict) -> np.ndarray:
    if record.get("schema") != CAL_SCHEMA or record.get("task") not in TASK_WIDTH:
        raise ValueError("calibrator identity mismatch")
    p = _probabilities(prob, TASK_WIDTH[record["task"]])
    alpha = float(record["alpha"])
    if not math.isfinite(alpha) or not TEMP_BOUNDS[0] <= alpha <= TEMP_BOUNDS[1]:
        raise ValueError("calibrator temperature outside frozen bounds")
    q = p.copy() if alpha == 1.0 else _scaled_from_logs(_log_input(p), alpha)
    if not np.array_equal(np.argmax(q, axis=1), np.argmax(p, axis=1)):
        raise ValueError("calibrated decisions differ")
    return q


def save_calibrator(record: dict, path: str | Path) -> str:
    """Write once to a private path; return SHA256 for release-package binding."""
    if record.get("schema") != CAL_SCHEMA:
        raise ValueError("invalid calibrator record")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
    except BaseException:
        p.unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def load_calibrator(path: str | Path, expected_sha256: str) -> dict:
    p = Path(path)
    if sha_file(p) != expected_sha256:
        raise ValueError("calibrator file hash mismatch")
    record = json.loads(p.read_text())
    if record.get("schema") != CAL_SCHEMA:
        raise ValueError("calibrator schema mismatch")
    return record


def task_metrics(prob, labels, *, task: str, constant_prob=None) -> dict:
    if task not in TASK_WIDTH:
        raise ValueError("unknown task")
    p = _probabilities(prob, TASK_WIDTH[task])
    y = _labels(labels, len(p), TASK_WIDTH[task])
    pred = np.argmax(p, axis=1)
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], CLIP))
    onehot = np.eye(TASK_WIDTH[task], dtype=np.float64)[y]
    brier = np.sum((p - onehot) ** 2, axis=1)
    recall = [float(np.mean(pred[y == c] == c)) if np.any(y == c) else None for c in range(TASK_WIDTH[task])]
    out = {"rows": int(len(y)), "accuracy": float(np.mean(pred == y)), "nll": float(np.mean(nll)),
           "brier": float(np.mean(brier)), "per_class_recall": recall,
           "class_counts": [int(np.sum(y == c)) for c in range(TASK_WIDTH[task])]}
    if constant_prob is not None:
        cp = _probabilities(np.asarray(constant_prob, dtype=np.float64).reshape(1, -1), TASK_WIDTH[task])[0]
        constant_acc = float(np.mean(y == np.argmax(cp)))
        out["constant_accuracy"] = constant_acc
        out["accuracy_gain_over_constant"] = out["accuracy"] - constant_acc
    return out


def fit_constant_probability(labels, *, task: str) -> np.ndarray:
    if task not in TASK_WIDTH:
        raise ValueError("unknown task")
    y = np.asarray(labels, dtype=np.int64)
    _labels(y, len(y), TASK_WIDTH[task])
    counts = np.bincount(y, minlength=TASK_WIDTH[task]).astype(np.float64)
    return counts / counts.sum()


def utility_against_references(candidate: dict, u0: dict, ucal: dict) -> dict:
    """Point differences and inclusive gates; uncertainty is a separate step."""
    out = {}
    for name, ref in (("U0", u0), ("Ucal", ucal)):
        delta = {"accuracy": candidate["accuracy"] - ref["accuracy"],
                 "nll": candidate["nll"] - ref["nll"],
                 "brier": candidate["brier"] - ref["brier"]}
        out[name] = {"delta": delta,
                     "point_eligible": delta["accuracy"] >= -0.010 and delta["nll"] <= 0.010
                     and delta["brier"] <= 0.005}
    out["point_eligible_both"] = out["U0"]["point_eligible"] and out["Ucal"]["point_eligible"]
    return out
