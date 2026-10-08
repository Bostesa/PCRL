"""Independent paired INNER inference over S4-verified C R1 release arrays.

This module consumes only S4's replayed per-arm artifacts and the admitted
reference arrays. It imports no Team C or shared scoring implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from replay_c_readers_v1 import READERS, oriented_auc
from replay_c_values_v1 import sha_array


TASKS = (("income", "prob_1", "U0_1", "Ucal_1", 2),
         ("occupation", "prob_2", "U0_2", "Ucal_2", 6))
VIEWS = ("r1", "r2", "pair")
REPLICATES = 2048
FLOOR = 1e-12


@dataclass(frozen=True)
class Prepared:
    arms: tuple[str, ...]
    names: tuple[str, ...]
    by_seed: dict[int, dict]
    unit: np.ndarray
    rows: int


def _intern(bank: dict[str, np.ndarray], value: np.ndarray,
            shape: tuple[int, ...]) -> str:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("C inference endpoint array shape or finite values differ")
    digest = sha_array(array)
    if digest in bank and not np.array_equal(bank[digest], array):
        raise ValueError("C inference array content hash collision")
    bank.setdefault(digest, array)
    return digest


def margin_names(arms: tuple[str, ...]) -> tuple[str, ...]:
    if len(arms) != 8 or len(set(arms)) != 8:
        raise ValueError("C inference needs eight distinct frozen arms")
    names = []
    for arm in arms:
        for task, *_ in TASKS:
            for reference in ("U0", "Ucal"):
                for metric in ("ACC", "NLL", "BRIER"):
                    names.append(f"{arm}|{task}:{reference}:{metric}")
        for other in arms:
            if other != arm:
                for view, metric in (("r1", "LOCAL"), ("r2", "LOCAL"),
                                     ("pair", "BENEFIT")):
                    names.append(f"{arm}|{other}:{view}:{metric}")
    return tuple(names)


def prepare(records: dict[int, dict], arms: tuple[str, ...]) -> Prepared:
    """Validate all paired rows and intern duplicate predictions by content."""
    if set(records) != {0, 1}:
        raise ValueError("C inference paired seeds differ")
    names = margin_names(arms)
    first_row, first_unit, first_labels = None, None, None
    prepared = {}
    for seed in (0, 1):
        record = records[seed]
        if set(record["arms"]) != set(arms):
            raise ValueError("C inference arm coverage differs")
        truth = record["truth"]
        row = np.asarray(truth["row_id"], dtype=np.int64)
        unit = np.asarray(truth["unit"], dtype=np.int64)
        labels = {task: np.asarray(truth[task], dtype=np.int64)
                  for task in ("income", "occupation", "sex")}
        if (row.shape != (2235,) or unit.shape != row.shape or
                len(np.unique(row)) != 2235 or len(np.unique(unit)) != 2234 or
                any(value.shape != row.shape for value in labels.values()) or
                not np.isin(labels["income"], (0, 1)).all() or
                not np.isin(labels["occupation"], tuple(range(6))).all() or
                not np.isin(labels["sex"], (0, 1)).all()):
            raise ValueError("C inference INNER row, group or label schema differs")
        if first_row is None:
            first_row, first_unit, first_labels = row, unit, labels
        elif (not np.array_equal(row, first_row) or
              not np.array_equal(unit, first_unit) or
              any(not np.array_equal(labels[key], first_labels[key])
                  for key in labels)):
            raise ValueError("C inference seeds do not share ordered people")
        bank: dict[str, np.ndarray] = {}
        references = {}
        for task, _, u0, ucal, width in TASKS:
            references[task] = {
                "U0": _intern(bank, record["references"][u0], (2235, width)),
                "Ucal": _intern(bank, record["references"][ucal], (2235, width)),
            }
        release = {}
        for arm in arms:
            item = record["arms"][arm]
            if not np.array_equal(item["row_id"], row) or not np.array_equal(item["unit"], unit):
                raise ValueError("C inference arm person order differs")
            if any(not np.array_equal(item[key], labels[key]) for key in labels):
                raise ValueError("C inference arm truth differs")
            probs = {task: _intern(bank, item[prob], (2235, width))
                     for task, prob, _, _, width in TASKS}
            readers = {}
            for view in VIEWS:
                readers[view] = tuple(_intern(
                    bank, item[f"reader_{view}__{name}"], (2235,))
                    for name in READERS)
            readers["pair"] += readers["r1"] + readers["r2"]
            release[arm] = {"prob": probs, "readers": readers}
        prepared[seed] = {"labels": labels, "reference": references,
                          "release": release, "bank": bank}
    return Prepared(arms, names, prepared, first_unit, 2235)


def _task(probability: np.ndarray, labels: np.ndarray, width: int) -> tuple[float, float, float]:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (p.shape != (len(y), width) or not np.isfinite(p).all() or
            np.any((p < 0) | (p > 1)) or
            not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("C inference task simplex differs")
    observed = p[np.arange(len(y)), y]
    onehot = np.eye(width, dtype=np.float64)[y]
    return (float(np.mean(p.argmax(axis=1) == y)),
            float(np.mean(-np.log(np.maximum(observed, 1e-12)))),
            float(np.mean(np.sum((p - onehot) ** 2, axis=1))))


def evaluate(prepared: Prepared, positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    if pos.ndim != 1 or not len(pos) or np.any((pos < 0) | (pos >= prepared.rows)):
        raise ValueError("C inference resample positions differ")
    across = []
    for seed in (0, 1):
        data = prepared.by_seed[seed]
        labels, bank = data["labels"], data["bank"]
        sex = labels["sex"][pos]
        task_cache, auc_cache = {}, {}
        def metric(digest: str, task: str, width: int):
            key = (digest, task)
            if key not in task_cache:
                task_cache[key] = _task(bank[digest][pos], labels[task][pos], width)
            return task_cache[key]
        def auc(digest: str):
            if digest not in auc_cache:
                auc_cache[digest] = oriented_auc(sex, bank[digest][pos])["fixed_alias_max_auc"]
            return auc_cache[digest]
        reference = {task: {name: metric(digest, task, width)
                            for name, digest in data["reference"][task].items()}
                     for task, _, _, _, width in TASKS}
        scored = {}
        for arm in prepared.arms:
            item = data["release"][arm]
            scored[arm] = {
                "task": {task: metric(item["prob"][task], task, width)
                         for task, _, _, _, width in TASKS},
                "auc": {view: max(auc(digest) for digest in item["readers"][view])
                        for view in VIEWS},
            }
        margins = []
        for arm in prepared.arms:
            candidate = scored[arm]
            for task, *_ in TASKS:
                for ref in ("U0", "Ucal"):
                    c, u = candidate["task"][task], reference[task][ref]
                    margins.extend((c[0] - u[0] + .010,
                                    .010 - (c[1] - u[1]),
                                    .005 - (c[2] - u[2])))
            for other in prepared.arms:
                if other != arm:
                    control = scored[other]
                    margins.extend((.010 - (candidate["auc"]["r1"] - control["auc"]["r1"]),
                                    .010 - (candidate["auc"]["r2"] - control["auc"]["r2"]),
                                    control["auc"]["pair"] - candidate["auc"]["pair"] - .020))
        across.append(np.asarray(margins, dtype=np.float64))
    vector = np.mean(np.stack(across), axis=0)
    if vector.shape != (len(prepared.names),) or not np.isfinite(vector).all():
        raise ValueError("C inference margin vector differs")
    return vector


def simultaneous(prepared: Prepared, private_seed: bytes) -> dict:
    """Independently recompute 2,048 paired-group max-studentized bounds."""
    if len(private_seed) != 32:
        raise ValueError("C inference seed is not 32 bytes")
    groups = np.unique(prepared.unit)
    if len(groups) != 2234:
        raise ValueError("C inference paired group count differs")
    members = {int(group): np.flatnonzero(prepared.unit == group) for group in groups}
    rng = np.random.Generator(np.random.PCG64DXSM(
        int.from_bytes(private_seed[:16], "big")))
    point = evaluate(prepared, np.arange(prepared.rows, dtype=np.int64))
    boot = np.empty((REPLICATES, len(prepared.names)), dtype=np.float64)
    for index in range(REPLICATES):
        selected = rng.choice(groups, size=len(groups), replace=True)
        positions = np.concatenate([members[int(group)] for group in selected])
        boot[index] = evaluate(prepared, positions)
    se = boot.std(axis=0, ddof=1)
    constant = np.max(np.abs(boot - point), axis=0) <= FLOOR
    se[constant] = 0.0
    denominator = np.maximum(se, FLOOR)
    scaled = np.zeros_like(boot)
    scaled[:, ~constant] = np.abs((boot[:, ~constant] - point[~constant]) /
                                  denominator[~constant])
    critical = float(np.quantile(scaled.max(axis=1), .95, method="higher"))
    width = critical * np.where(constant, 0.0, denominator)
    lower, upper = point - width, point + width
    clauses = []
    for index, name in enumerate(prepared.names):
        status = ("PASS" if lower[index] >= 0 else
                  "VIOLATION" if upper[index] < 0 else "UNRESOLVED")
        clauses.append({"name": name, "point_margin": float(point[index]),
                        "se": float(se[index]), "lower": float(lower[index]),
                        "upper": float(upper[index]), "status": status,
                        "numerical_se_status": "NUMERICALLY_CONSTANT"
                        if constant[index] else "VARIABLE"})
    return {"schema": "pcrl-portfolio-paired-inference-v1",
            "bootstrap_replicates": REPLICATES, "inner_groups": len(groups),
            "mechanism_draws_in_primary_interval": 1,
            "critical_family": "two-sided-maximum-absolute-studentized",
            "critical_95": critical, "studentization_floor": FLOOR,
            "numpy_version": np.__version__,
            "private_seed_sha256": hashlib.sha256(private_seed).hexdigest(),
            "draw0_is_point": True, "clauses": clauses}
