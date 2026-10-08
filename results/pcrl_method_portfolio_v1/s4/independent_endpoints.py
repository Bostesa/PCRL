"""Independent, in-memory replay of the portfolio's registered endpoints.

This file deliberately imports no candidate, release-builder, attack, common
evaluator, or selection implementation. A separate S4 custody adapter must
read immutable private packages and admitted INNER labels, verify their hashes
and ordered row/group identities, then construct the objects below. Numerical
execution on Adult arrays requires an S3 queue lease.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BOOTSTRAP_REPLICATES = 2048
STUDENTIZATION_FLOOR = 1e-12
TASKS = ("income", "occupation")
REFERENCES = ("U0", "Ucal")
VIEWS = ("r1", "r2", "pair")


def _array(value, *, ndim: int, kind: str) -> np.ndarray:
    out = np.asarray(value)
    if out.ndim != ndim or (kind == "integer" and not np.issubdtype(out.dtype, np.integer)):
        raise ValueError(f"expected {ndim}-D {kind} array")
    if kind == "finite" and not np.isfinite(out).all():
        raise ValueError("nonfinite array")
    return out


def _probability_rows(probability, width: int, n: int) -> np.ndarray:
    p = _array(probability, ndim=2, kind="finite").astype(np.float64, copy=False)
    if p.shape != (n, width) or np.any(p < 0) or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-8):
        raise ValueError("invalid released probability rows")
    return p


@dataclass(frozen=True)
class TaskRows:
    """Actual released task probabilities and public decisions, in role order."""

    probability: np.ndarray
    decision: np.ndarray

    def checked(self, n: int, width: int) -> "TaskRows":
        p = _probability_rows(self.probability, width, n)
        d = _array(self.decision, ndim=1, kind="integer")
        if d.shape != (n,) or np.any((d < 0) | (d >= width)) or not np.array_equal(d, p.argmax(1)):
            raise ValueError("released decision/probability mismatch")
        return TaskRows(p, d.astype(np.int64, copy=False))


@dataclass(frozen=True)
class ArmRows:
    """Complete-view reader predictions are frozen before INNER scoring."""

    tasks: dict[str, TaskRows]
    readers: dict[str, dict[str, np.ndarray]]

    def checked(self, n: int) -> "ArmRows":
        if set(self.tasks) != set(TASKS) or set(self.readers) != set(VIEWS):
            raise ValueError("arm task/view declaration is incomplete")
        tasks = {task: self.tasks[task].checked(n, 2 if task == "income" else 6)
                 for task in TASKS}
        readers = {}
        for view in VIEWS:
            predictions = self.readers[view]
            if not predictions or len(set(predictions)) != len(predictions):
                raise ValueError("reader slate missing")
            readers[view] = {}
            for name, value in predictions.items():
                p = _array(value, ndim=1, kind="finite").astype(np.float64, copy=False)
                if p.shape != (n,) or not name or np.any((p < 0) | (p > 1)):
                    raise ValueError("frozen reader prediction invalid")
                readers[view][name] = p
        return ArmRows(tasks, readers)


@dataclass(frozen=True)
class SeedRows:
    """One paired model seed; all arrays refer to the same INNER people."""

    row_id: np.ndarray
    unit: np.ndarray
    labels: dict[str, np.ndarray]
    sex: np.ndarray
    references: dict[str, dict[str, TaskRows]]
    candidate: ArmRows
    controls: dict[str, ArmRows]

    def checked(self, control_ids: tuple[str, ...]) -> "SeedRows":
        rid = _array(self.row_id, ndim=1, kind="integer").astype(np.int64, copy=False)
        unit = _array(self.unit, ndim=1, kind="integer").astype(np.int64, copy=False)
        n = len(rid)
        if n == 0 or unit.shape != (n,) or np.unique(rid).size != n or np.unique(unit).size != 2234:
            raise ValueError("INNER identity/group schema mismatch")
        if set(self.labels) != set(TASKS) or set(self.references) != set(REFERENCES):
            raise ValueError("task labels or frozen references missing")
        labels = {}
        for task, width in (("income", 2), ("occupation", 6)):
            y = _array(self.labels[task], ndim=1, kind="integer").astype(np.int64, copy=False)
            if y.shape != (n,) or np.any((y < 0) | (y >= width)):
                raise ValueError("task label vocabulary mismatch")
            labels[task] = y
        sex = _array(self.sex, ndim=1, kind="integer").astype(np.int64, copy=False)
        if sex.shape != (n,) or not np.isin(sex, (0, 1)).all() or np.unique(sex).size != 2:
            raise ValueError("SEX label schema mismatch")
        refs = {}
        for ref in REFERENCES:
            if set(self.references[ref]) != set(TASKS):
                raise ValueError("reference task missing")
            refs[ref] = {task: self.references[ref][task].checked(n, 2 if task == "income" else 6)
                         for task in TASKS}
        if tuple(self.controls) != control_ids:
            raise ValueError("control IDs or frozen order mismatch")
        return SeedRows(rid, unit, labels, sex, refs, self.candidate.checked(n),
                        {key: self.controls[key].checked(n) for key in control_ids})


def _task_metric(rows: TaskRows, label: np.ndarray, weight: np.ndarray) -> tuple[float, float, float]:
    total = float(weight.sum())
    if total <= 0:
        raise ValueError("empty bootstrap")
    n = len(label)
    observed = rows.probability[np.arange(n), label]
    accuracy = float(np.dot(weight, rows.decision == label) / total)
    nll = float(np.dot(weight, -np.log(np.maximum(observed, 1e-12))) / total)
    brier_row = np.sum(rows.probability * rows.probability, axis=1) + 1 - 2 * observed
    brier = float(np.dot(weight, brier_row) / total)
    return accuracy, nll, brier


class _RankedReader:
    """Weighted tie-correct Mann–Whitney AUC for repeated bootstrap groups."""

    def __init__(self, prediction: np.ndarray, sex: np.ndarray):
        order = np.argsort(prediction, kind="stable")
        sorted_score = prediction[order]
        self.order = order
        self.sex = sex[order]
        self.starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]

    def auc(self, weight: np.ndarray) -> float:
        w = weight[self.order]
        pos = np.add.reduceat(w * self.sex, self.starts)
        neg = np.add.reduceat(w * (1 - self.sex), self.starts)
        n_pos, n_neg = float(pos.sum()), float(neg.sum())
        if n_pos <= 0 or n_neg <= 0:
            raise ValueError("bootstrap sample lacks a SEX class")
        numerator = np.dot(pos, np.cumsum(neg) - 0.5 * neg)
        return float(numerator / (n_pos * n_neg))


class _PreparedSeed:
    def __init__(self, seed: SeedRows, control_ids: tuple[str, ...]):
        self.seed = seed
        self.ranked = {}
        for arm_id, arm in (("candidate", seed.candidate), *seed.controls.items()):
            self.ranked[arm_id] = {view: tuple(_RankedReader(p, seed.sex)
                                                for p in arm.readers[view].values())
                                   for view in VIEWS}
        self.control_ids = control_ids

    def envelope(self, arm_id: str, view: str, weight: np.ndarray) -> float:
        return max(max(a, 1.0 - a) for a in
                   (reader.auc(weight) for reader in self.ranked[arm_id][view]))

    def margins(self, weight: np.ndarray) -> np.ndarray:
        s = self.seed
        values = []
        for task in TASKS:
            y = s.labels[task]
            ca, cn, cb = _task_metric(s.candidate.tasks[task], y, weight)
            for ref in REFERENCES:
                ra, rn, rb = _task_metric(s.references[ref][task], y, weight)
                values.extend((ca - ra + 0.010, 0.010 - (cn - rn), 0.005 - (cb - rb)))
        candidate_auc = {view: self.envelope("candidate", view, weight) for view in VIEWS}
        for control_id in self.control_ids:
            control_auc = {view: self.envelope(control_id, view, weight) for view in VIEWS}
            values.extend((0.010 - (candidate_auc["r1"] - control_auc["r1"]),
                           0.010 - (candidate_auc["r2"] - control_auc["r2"]),
                           control_auc["pair"] - candidate_auc["pair"] - 0.020))
        return np.asarray(values, dtype=np.float64)


def margin_names(control_ids: tuple[str, ...]) -> tuple[str, ...]:
    names = []
    for task in TASKS:
        for ref in REFERENCES:
            names.extend(f"{task}/{ref}/{metric}" for metric in ("ACC", "NLL", "BRIER"))
    for control in control_ids:
        names.extend(f"{control}/{metric}" for metric in
                     ("R1_LOCAL", "R2_LOCAL", "PAIR_BENEFIT"))
    return tuple(names)


def replay_family(seeds: dict[int, SeedRows], control_ids: tuple[str, ...],
                  private_seed: bytes, *, replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    """Replay one predeclared candidate family without selecting a control."""
    if not seeds or not control_ids or len(set(control_ids)) != len(control_ids):
        raise ValueError("seed/control family missing or duplicated")
    if len(private_seed) != 32 or replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("frozen bootstrap seed/replicate contract mismatch")
    checked = {sid: item.checked(control_ids) for sid, item in sorted(seeds.items())}
    first = next(iter(checked.values()))
    for item in checked.values():
        if not np.array_equal(item.row_id, first.row_id) or not np.array_equal(item.unit, first.unit):
            raise ValueError("model seeds do not contain the same ordered people/groups")
    prepared = tuple(_PreparedSeed(item, control_ids) for item in checked.values())
    unit = first.unit
    groups, group_code = np.unique(unit, return_inverse=True)
    if len(groups) != 2234:
        raise ValueError("INNER group count changed")
    names = margin_names(control_ids)

    def evaluate(weight: np.ndarray) -> np.ndarray:
        result = np.mean([item.margins(weight) for item in prepared], axis=0)
        if result.shape != (len(names),) or not np.isfinite(result).all():
            raise ValueError("nonfinite or incomplete margin vector")
        return result

    point = evaluate(np.ones(len(unit), dtype=np.float64))
    boot = np.empty((replicates, len(names)), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(private_seed[:16], "big")))
    for b in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sampled_code = np.searchsorted(groups, sampled)
        group_weight = np.bincount(sampled_code, minlength=len(groups)).astype(np.float64)
        boot[b] = evaluate(group_weight[group_code])
    se = boot.std(axis=0, ddof=1)
    # This is a fixed floating-point tolerance, not a structural identity.
    zero = np.max(np.abs(boot - point), axis=0) <= STUDENTIZATION_FLOOR
    se[zero] = 0.0
    scale = np.maximum(se, STUDENTIZATION_FLOOR)
    deviation = np.zeros_like(boot)
    deviation[:, ~zero] = np.abs((boot[:, ~zero] - point[~zero]) / scale[~zero])
    critical = float(np.quantile(deviation.max(axis=1), 0.95, method="higher"))
    width = critical * np.where(zero, 0.0, scale)
    lower, upper = point - width, point + width
    clauses = []
    for j, name in enumerate(names):
        status = "PASS" if lower[j] >= 0 else "VIOLATION" if upper[j] < 0 else "UNRESOLVED"
        clauses.append({"name": name, "point_margin": float(point[j]), "se": float(se[j]),
                        "lower": float(lower[j]), "upper": float(upper[j]), "status": status,
                        "numerical_se_status": "NUMERICALLY_CONSTANT" if zero[j] else "VARIABLE"})
    return {"schema": "pcrl-s4-independent-endpoint-replay-v1",
            "seed_ids": list(checked), "control_ids": list(control_ids),
            "groups": len(groups), "replicates": replicates, "critical_95": critical,
            "clauses": clauses}
