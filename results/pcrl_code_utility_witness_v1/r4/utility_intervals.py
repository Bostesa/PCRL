"""Fixed-family utility intervals on paired INNER duplicate groups.

Pure arrays only. The queued caller must verify role, source, release and
decoder identity before supplying probabilities and labels. Missing arm or
reference cells retain null slots in the original 2,376-clause family.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

from .finite_family import ordered_clause_ids
from .inference_rule import BOOTSTRAP_DRAWS, K_FULL, interval_bounds, utility_status


BOOTSTRAP_SEED = 24681357
TASK_CLASSES = {"income": 2, "occupation": 6}
FLOOR = 1e-12


def cost_vectors(candidate: np.ndarray, reference: np.ndarray,
                 labels: np.ndarray, task: str) -> dict[str, np.ndarray]:
    """Accuracy uses candidate minus reference; losses use candidate excess."""
    if task not in TASK_CLASSES:
        raise ValueError("unknown task")
    width = TASK_CLASSES[task]
    cand, ref, y = np.asarray(candidate), np.asarray(reference), np.asarray(labels)
    if (cand.dtype != np.float64 or ref.dtype != np.float64
            or cand.ndim != 2 or cand.shape != ref.shape
            or cand.shape[1] != width or y.dtype != np.int64
            or y.shape != (len(cand),) or not len(y)
            or np.any((y < 0) | (y >= width))
            or not np.isfinite(cand).all() or not np.isfinite(ref).all()
            or np.any(cand < 0) or np.any(ref < 0)
            or not np.allclose(cand.sum(axis=1), 1, rtol=0, atol=1e-8)
            or not np.allclose(ref.sum(axis=1), 1, rtol=0, atol=1e-8)):
        raise ValueError("candidate/reference/label schema differs")
    rows = np.arange(len(y))
    onehot = np.eye(width, dtype=np.float64)[y]
    return {
        "accuracy": (cand.argmax(axis=1) == y).astype(np.float64)
                    - (ref.argmax(axis=1) == y).astype(np.float64),
        "nll": -np.log(np.maximum(cand[rows, y], FLOOR))
               + np.log(np.maximum(ref[rows, y], FLOOR)),
        "brier": np.sum((cand - onehot) ** 2, axis=1)
                 - np.sum((ref - onehot) ** 2, axis=1),
    }


def _weights(units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(units)
    if (ids.dtype != np.int64 or ids.ndim != 1 or not len(ids)):
        raise ValueError("duplicate-group IDs must be nonempty int64")
    groups, inverse = np.unique(ids, return_inverse=True)
    if len(groups) < 2:
        raise ValueError("paired-group inference needs at least two groups")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    sampled = rng.integers(0, len(groups),
                           size=(BOOTSTRAP_DRAWS, len(groups)), dtype=np.int64)
    draw = np.stack([np.bincount(row, minlength=len(groups))
                     for row in sampled])
    return draw, inverse


def _replicates(values: np.ndarray, draw: np.ndarray,
                inverse: np.ndarray) -> np.ndarray:
    group_count = np.bincount(inverse).astype(np.float64)
    group_sum = np.bincount(inverse, weights=values, minlength=len(group_count))
    return (draw @ group_sum) / (draw @ group_count)


def evaluate_utility_family(
    candidates: Mapping[tuple[int, str, str], np.ndarray],
    references: Mapping[tuple[int, str, str], np.ndarray],
    labels: Mapping[str, np.ndarray], units: np.ndarray,
) -> dict:
    """Score admitted candidate/reference arrays, preserving all missing slots.

    A single 2,048-draw group count matrix is reused across every available
    arm, reference, task and seed. Seed means are equal-weight means of
    per-seed points and of same-draw replicates, not stacked populations.
    """
    names = ordered_clause_ids()
    draw, inverse = _weights(units)
    n = len(inverse)
    for task, y in labels.items():
        if task not in TASK_CLASSES or np.asarray(y).shape != (n,):
            raise ValueError("task label/people inventory differs")
    null = {"point": None, "se": None, "lower": None, "upper": None,
            "status": "NOT_ESTIMABLE_DEPENDENCY"}
    clauses = {name: null.copy() for name in names}
    rep_by_base: dict[str, dict[int, np.ndarray]] = {}
    point_by_base: dict[str, dict[int, float]] = {}
    registered = set(names)
    for (seed, arm, task), cand in sorted(candidates.items()):
        if seed not in (0, 1) or task not in TASK_CLASSES:
            raise ValueError("candidate seed or task differs from finite family")
        for ref_name in ("F", "U0", "Ucal"):
            key = (seed, task, ref_name)
            if key not in references:
                continue
            vectors = cost_vectors(cand, references[key], labels[task], task)
            for metric, values in vectors.items():
                base = f"utility:{arm}:{task}:{ref_name}:{metric}"
                name = f"s{seed}:{base}"
                if name not in registered:
                    raise ValueError("candidate/reference outside sealed finite family")
                point = float(np.mean(values))
                replicate = _replicates(values, draw, inverse)
                se = float(np.std(replicate, ddof=1))
                lower, upper = interval_bounds(point, se)
                clauses[name] = {"point": point, "se": se, "lower": lower,
                                 "upper": upper,
                                 "status": utility_status(metric, point, lower, upper)}
                rep_by_base.setdefault(base, {})[seed] = replicate
                point_by_base.setdefault(base, {})[seed] = point
    for base, per_seed in rep_by_base.items():
        if set(per_seed) != {0, 1}:
            continue
        metric = base.rsplit(":", 1)[-1]
        point = (point_by_base[base][0] + point_by_base[base][1]) / 2
        replicate = (per_seed[0] + per_seed[1]) / 2
        se = float(np.std(replicate, ddof=1))
        lower, upper = interval_bounds(point, se)
        clauses[f"mean:{base}"] = {"point": point, "se": se,
                                   "lower": lower, "upper": upper,
                                   "status": utility_status(metric, point, lower, upper)}
    order_hash = hashlib.sha256(json.dumps(names, ensure_ascii=False,
                                           separators=(",", ":")).encode()).hexdigest()
    return {"schema": "pcrl-code-witness-utility-family-v1",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "groups": int(len(np.unique(units))), "rows": n,
            "family_k": K_FULL, "ordered_ids_sha256": order_hash,
            "ordered_ids": list(names), "clauses": clauses,
            "estimable_clauses": sum(v["point"] is not None for v in clauses.values()),
            "scope": "approximate conditional paired-group development intervals; full fixed-K critical"}
