"""Independent paired-group simultaneous inference from M5 replayed vectors.

This module does not import M3 inference, M4 endpoint code, or fitting selection.
It recomputes every possible point-selected contrast in a fixed 207-clause family.
"""

from __future__ import annotations

import hashlib

import numpy as np


RULES = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")
TASKS = ("income", "occupation_group")
REFS = ("U0", "Ucal", "F")
VIEWS = ("income", "occupation_group", "coalition")
METRICS = ("accuracy", "nll", "brier")
LOCAL_IDS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
             "RAW_LR_C1", "RAW_HGB_15x200")
READER_IDS = {
    "income": frozenset(LOCAL_IDS),
    "occupation_group": frozenset(LOCAL_IDS),
    "coalition": frozenset(LOCAL_IDS) | frozenset(
        alias + reader for alias in ("IGNORE_R1__", "IGNORE_R2__")
        for reader in LOCAL_IDS),
}
CLAUSE_NAMES = tuple(
    f"{rule}|{task}|{ref}|{metric}"
    for rule in RULES for task in TASKS for ref in REFS for metric in METRICS
) + tuple(
    f"J|vs|{rule}|{kind}"
    for rule in RULES[1:]
    for kind in ("coalition_benefit", "income_local_cap", "occupation_local_cap")
)
FLOOR = 1e-12


def _wire(raw: object, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    observed = np.asarray(raw)
    if observed.dtype != np.dtype(dtype) or observed.shape != shape:
        raise ValueError("M5 inference saved wire dtype or shape differs")
    value = np.array(observed, order="C", copy=True)
    if dtype == "float64" and not np.all(np.isfinite(value)):
        raise ValueError("M5 inference saved wire is nonfinite")
    value.setflags(write=False)
    return value


def _fingerprint(array: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes(order="C"))
    return h.hexdigest()


def _prob(raw: object, rows: int, classes: int) -> np.ndarray:
    p = _wire(raw, "float64", (rows, classes))
    if np.any((p < 0) | (p > 1)) or not np.allclose(
            p.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("M5 inference probability simplex differs")
    return p


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Tie-aware Mann–Whitney AUC with vectorized tied-block counts."""
    positives = int(np.count_nonzero(y))
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("M5 inference resample lacks both SEX classes")
    order = np.argsort(score, kind="mergesort")
    sorted_s = score[order]
    sorted_y = y[order]
    starts = np.r_[0, np.flatnonzero(sorted_s[1:] != sorted_s[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(y)])
    pos = np.add.reduceat(sorted_y, starts)
    neg = sizes - pos
    before = np.cumsum(neg) - neg
    auc = float(np.sum(pos * (before + .5 * neg)) / (positives * negatives))
    return max(auc, 1.0 - auc)


def _task(y: np.ndarray, p: np.ndarray, positions: np.ndarray) -> tuple[float, float, float]:
    chosen_y = y[positions]
    chosen_p = p[positions]
    rows = np.arange(len(positions))
    correct = chosen_p[rows, chosen_y]
    squared = np.square(chosen_p).sum(axis=1) - 2 * correct + 1
    return (float(np.mean(np.argmax(chosen_p, axis=1) == chosen_y)),
            float(np.mean(-np.log(np.maximum(correct, 1e-12)))),
            float(np.mean(squared)))


def _prepare(saved_by_seed: dict, plans: dict, expected_seeds: tuple[int, ...],
             expected_rows: int, expected_groups: int) -> tuple[dict, np.ndarray, np.ndarray]:
    if (set(saved_by_seed) != set(expected_seeds)
            or set(plans) != set(expected_seeds) or len(CLAUSE_NAMES) != 207):
        raise ValueError("M5 paired seed or clause family differs")
    frozen = {}
    reference_truth = None
    active = np.ones(207, dtype=bool)
    for seed in expected_seeds:
        plan = plans[seed]
        if (plan.get("schema") != "pcrl-focused-m3-control-plan-v1"
                or set(plan.get("rules", {})) != set(RULES)
                or plan["rules"]["J"].get("status") != "SELECTED"):
            raise ValueError("M5 FIT rule slate is incomplete")
        selected_pairs = {row["pair_id"] for row in plan["rules"].values()
                          if row.get("status") == "SELECTED"}
        if set(saved_by_seed[seed]) != selected_pairs:
            raise ValueError("M5 saved selected pair set differs")
        first = saved_by_seed[seed][plan["rules"]["J"]["pair_id"]]
        truth = {field: _wire(first[f"truth__{field}"], "int64", (expected_rows,))
                 for field in ("row_id", "unit", "income", "occupation", "sex")}
        if (len(np.unique(truth["row_id"])) != expected_rows
                or len(np.unique(truth["unit"])) != expected_groups
                or np.any((truth["income"] < 0) | (truth["income"] > 1))
                or np.any((truth["occupation"] < 0) | (truth["occupation"] > 5))
                or np.any((truth["sex"] < 0) | (truth["sex"] > 1))):
            raise ValueError("M5 inference identity or labels differ")
        if reference_truth is None:
            reference_truth = truth
        elif any(not np.array_equal(reference_truth[key], truth[key]) for key in truth):
            raise ValueError("M5 paired seed rows, groups or labels differ")
        cache: dict[str, np.ndarray] = {}

        def intern(array: np.ndarray) -> str:
            digest = _fingerprint(array)
            if digest in cache and not np.array_equal(cache[digest], array):
                raise ValueError("M5 inference array hash collision")
            cache[digest] = array
            return digest

        refs = {}
        for task, width in (("income", 2), ("occupation_group", 6)):
            refs[task] = {ref: intern(_prob(first[f"reference__{task}__{ref}"],
                                            expected_rows, width)) for ref in REFS}
        rules = {}
        for index, rule in enumerate(RULES):
            choice = plan["rules"][rule]
            if choice.get("status") != "SELECTED":
                active[index * 18:(index + 1) * 18] = False
                if index:
                    active[180 + (index - 1) * 3:183 + (index - 1) * 3] = False
                continue
            pair = choice["pair_id"]
            values = saved_by_seed[seed][pair]
            if any(not np.array_equal(values[f"truth__{field}"], truth[field])
                   for field in truth):
                raise ValueError("M5 same-seed pair truth changed")
            for task, width in (("income", 2), ("occupation_group", 6)):
                for ref in REFS:
                    if not np.array_equal(values[f"reference__{task}__{ref}"],
                                          cache[refs[task][ref]]):
                        raise ValueError("M5 same-seed reference vector changed")
            probs = {task: intern(_prob(values[f"candidate__{task}"], expected_rows,
                                        width)) for task, width in
                     (("income", 2), ("occupation_group", 6))}
            readers = {}
            for view in VIEWS:
                prefix = f"reader__{view}__"
                ids = {key[len(prefix):] for key in values if key.startswith(prefix)}
                if ids != READER_IDS[view]:
                    raise ValueError("M5 fixed reader IDs differ")
                readers[view] = tuple(intern(_wire(values[prefix + reader],
                                                  "float64", (expected_rows,)))
                                      for reader in sorted(ids))
                if any(np.any((cache[digest] < 0) | (cache[digest] > 1))
                       for digest in readers[view]):
                    raise ValueError("M5 saved reader score outside [0,1]")
            rules[rule] = (probs, readers)
        frozen[seed] = {"truth": truth, "refs": refs, "rules": rules, "cache": cache}
    return frozen, reference_truth["unit"], active


def _margins(frozen: dict, active: np.ndarray, positions: np.ndarray) -> np.ndarray:
    per_seed = []
    for seed in sorted(frozen):
        item = frozen[seed]
        truth, arrays = item["truth"], item["cache"]
        task_cache: dict[tuple[str, str], tuple[float, float, float]] = {}
        auc_cache: dict[str, float] = {}

        def task(digest: str, purpose: str) -> tuple[float, float, float]:
            key = (digest, purpose)
            if key not in task_cache:
                labels = truth["income" if purpose == "income" else "occupation"]
                task_cache[key] = _task(labels, arrays[digest], positions)
            return task_cache[key]

        def auc(digest: str) -> float:
            if digest not in auc_cache:
                auc_cache[digest] = _auc(truth["sex"][positions],
                                         arrays[digest][positions])
            return auc_cache[digest]

        stats = {}
        for rule, (prob, readers) in item["rules"].items():
            stats[rule] = ({purpose: task(digest, purpose) for purpose, digest in prob.items()},
                           {view: max(auc(digest) for digest in ids)
                            for view, ids in readers.items()})
        values = np.full(207, np.nan, dtype=np.float64)
        k = 0
        for rule in RULES:
            if rule not in stats:
                k += 18
                continue
            for purpose in TASKS:
                candidate = stats[rule][0][purpose]
                for ref in REFS:
                    reference = task(item["refs"][purpose][ref], purpose)
                    values[k:k + 3] = (candidate[0] - reference[0] + .010,
                                       reference[1] + .010 - candidate[1],
                                       reference[2] + .005 - candidate[2])
                    k += 3
        joint = stats["J"][1]
        for control in RULES[1:]:
            if control not in stats:
                k += 3
                continue
            other = stats[control][1]
            values[k:k + 3] = (other["coalition"] - joint["coalition"] - .020,
                               other["income"] + .010 - joint["income"],
                               other["occupation_group"] + .010 - joint["occupation_group"])
            k += 3
        per_seed.append(values)
    result = np.full(207, np.nan, dtype=np.float64)
    result[active] = np.mean(np.stack(per_seed)[:, active], axis=0)
    if not np.all(np.isfinite(result[active])):
        raise ValueError("M5 paired numerical margins are not finite")
    return result


def infer_from_saved_values(saved_by_seed: dict, plans: dict, private_seed: bytes, *,
                            expected_seeds: tuple[int, ...] = (0, 1),
                            expected_rows: int = 2235, expected_groups: int = 2234,
                            replicates: int = 2048) -> dict:
    if (not isinstance(private_seed, bytes) or len(private_seed) != 32
            or type(replicates) is not int or replicates < 2):
        raise ValueError("M5 inference requires all 32 seed bytes and replicates")
    frozen, groups, active = _prepare(saved_by_seed, plans, expected_seeds,
                                      expected_rows, expected_groups)
    observed = _margins(frozen, active, np.arange(expected_rows, dtype=np.int64))
    active_index = np.flatnonzero(active)
    original = observed[active_index]
    unique = np.unique(groups)
    group_positions = {int(group): np.flatnonzero(groups == group) for group in unique}
    generator = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(private_seed, "big")))
    samples = np.empty((replicates, len(active_index)), dtype=np.float64)
    for draw_index in range(replicates):
        drawn = generator.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([group_positions[int(group)] for group in drawn])
        try:
            samples[draw_index] = _margins(frozen, active, positions)[active_index]
        except ValueError as error:
            if "both SEX classes" not in str(error):
                raise
            return {"schema": "pcrl-focused-simultaneous-inference-v1",
                    "status": "INCOMPLETE_UNDEFINED_REPLICATE",
                    "bootstrap_replicates": replicates,
                    "undefined_replicate": draw_index,
                    "clauses": [{"name": name, "status": (
                        "UNRESOLVED_INVALID_FIT_RULE" if not valid else
                        "UNRESOLVED_UNDEFINED_REPLICATE")}
                        for name, valid in zip(CLAUSE_NAMES, active)]}
    standard = samples.std(axis=0, ddof=1)
    zero = np.max(np.abs(samples - original), axis=0) <= FLOOR
    standard[zero] = 0
    scale = np.maximum(standard, FLOOR)
    student = np.zeros_like(samples)
    student[:, ~zero] = np.abs((samples[:, ~zero] - original[~zero]) / scale[~zero])
    critical = float(np.quantile(student.max(axis=1), .95, method="higher"))
    width = critical * np.where(zero, 0, scale)
    lookup = {int(value): j for j, value in enumerate(active_index)}
    clauses = []
    for i, name in enumerate(CLAUSE_NAMES):
        if i not in lookup:
            clauses.append({"name": name, "status": "UNRESOLVED_INVALID_FIT_RULE"})
            continue
        j = lookup[i]
        lower, upper = original[j] - width[j], original[j] + width[j]
        clauses.append({"name": name, "point_margin": float(original[j]),
                        "se": float(standard[j]), "lower": float(lower),
                        "upper": float(upper),
                        "status": "PASS" if lower >= 0 else "VIOLATION" if upper < 0
                        else "UNRESOLVED",
                        "numerical_se_status": "NUMERICALLY_CONSTANT" if zero[j]
                        else "VARIABLE"})
    return {"schema": "pcrl-focused-simultaneous-inference-v1",
            "status": "DEVELOPMENT_SIMULTANEOUS_INTERVALS",
            "bootstrap_replicates": replicates, "inner_groups": len(unique),
            "critical_family": "two-sided-maximum-absolute-studentized",
            "critical_95": critical, "studentization_floor": FLOOR,
            "private_seed_sha256": hashlib.sha256(private_seed).hexdigest(),
            "numpy_version": np.__version__,
            "registered_clause_count": len(CLAUSE_NAMES),
            "active_clause_count": int(active.sum()),
            "invalid_fit_rule_ids": [rule for rule in RULES
                                     if any(plans[seed]["rules"][rule]["status"] != "SELECTED"
                                            for seed in expected_seeds)],
            "clauses": clauses,
            "interpretation": "development inference conditional on fitted models and exposed rows"}
