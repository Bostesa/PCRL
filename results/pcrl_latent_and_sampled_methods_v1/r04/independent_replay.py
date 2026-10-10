"""Independent task, recovery, selection, and interval arithmetic.

This file imports no candidate, runner, shared scoring, or selection module.
The caller must hash and align admitted private arrays before using it, and
must run real-data numerical replay only through R03 after R00's effective seal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np


TASK_CAPS = {"accuracy_drop": 0.010, "nll_excess": 0.010, "brier_excess": 0.005}
LOCAL_AUC_CAP = 0.010
COALITION_BENEFIT_TARGET = 0.020
NLL_FLOOR = 1e-12


def d_finite_family_ids() -> tuple[str, ...]:
    """Independently enumerate every prospective seven-arm D interval slot."""
    arms = ("D_T", "D_K", "D_L", "D_S12", "D_S21", "D_C", "D_F")
    refs = ("U0", "Ucal")
    views = ("r1", "r2", "pair")
    base = []
    for arm in arms:
        for task in ("income", "occupation"):
            for ref in refs:
                for metric in ("accuracy", "nll", "brier"):
                    base.append(f"utility:{arm}:{task}:{ref}:{metric}")
    for first, second in combinations(arms, 2):
        for view in views:
            base.append(f"arm_pair:{first}:{second}:{view}:auc")
    for arm in arms:
        for ref in refs:
            for view in views:
                base.append(f"u_anchor:{arm}:{ref}:{view}:auc")
    names = tuple(f"{summary}:{name}" for summary in ("s0", "s1", "mean")
                  for name in base)
    if len(names) != 567 or len(set(names)) != len(names):
        raise AssertionError("independent D family count or identities differ")
    return names


def _probabilities(values: np.ndarray, n: int, classes: int) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if (p.shape != (n, classes) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or np.max(np.abs(p.sum(axis=1) - 1)) > 1e-8):
        raise ValueError("invalid probability matrix")
    return p


def _labels(values: np.ndarray, n: int, classes: int) -> np.ndarray:
    y = np.asarray(values)
    if (y.shape != (n,) or not np.issubdtype(y.dtype, np.integer)
            or np.any((y < 0) | (y >= classes))):
        raise ValueError("invalid ordered labels")
    return y.astype(np.int64, copy=False)


def _weights(values: np.ndarray | None, n: int) -> np.ndarray:
    w = np.ones(n, dtype=np.float64) if values is None else np.asarray(values, dtype=np.float64)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid nonnegative row multiplicities")
    return w


def task_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    decision: np.ndarray | None = None,
    row_weight: np.ndarray | None = None,
    majority_class: int | None = None,
) -> dict:
    """Accuracy, natural-log NLL, summed Brier, class recall, task usefulness."""
    y0 = np.asarray(labels)
    if y0.ndim != 1 or not len(y0):
        raise ValueError("nonempty ordered labels required")
    n = len(y0)
    raw = np.asarray(probabilities)
    if raw.ndim != 2 or raw.shape[0] != n or raw.shape[1] < 2:
        raise ValueError("probability dimensions invalid")
    classes = raw.shape[1]
    y = _labels(y0, n, classes)
    p = _probabilities(raw, n, classes)
    d = np.argmax(p, axis=1)
    if decision is not None:
        declared = _labels(decision, n, classes)
        if not np.array_equal(declared, d):
            raise ValueError("wire decision differs from public probability argmax")
    w = _weights(row_weight, n)
    denom = float(w.sum())
    chosen = p[np.arange(n), y]
    acc = float(np.dot(w, d == y) / denom)
    nll = float(np.dot(w, -np.log(np.maximum(chosen, NLL_FLOOR))) / denom)
    # Equivalent to sum_j (p_j - 1{j=y})^2, without constructing one-hot rows.
    brier = float(np.dot(w, np.sum(p * p, axis=1) + 1 - 2 * chosen) / denom)
    recalls: list[float | None] = []
    counts: list[int] = []
    for cls in range(classes):
        in_class = y == cls
        support = float(np.dot(w, in_class))
        counts.append(int(np.count_nonzero(in_class)))
        recalls.append(float(np.dot(w, in_class & (d == cls)) / support) if support else None)
    result = {"accuracy": acc, "nll": nll, "brier": brier,
              "class_recall": recalls, "class_counts": counts}
    if majority_class is not None:
        if not 0 <= majority_class < classes:
            raise ValueError("majority class outside task vocabulary")
        base_acc = float(np.dot(w, y == majority_class) / denom)
        result["majority_accuracy"] = base_acc
        result["accuracy_gain_over_majority"] = acc - base_acc
    return result


def task_costs(candidate: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    return {
        "accuracy_drop": float(reference["accuracy"] - candidate["accuracy"]),
        "nll_excess": float(candidate["nll"] - reference["nll"]),
        "brier_excess": float(candidate["brier"] - reference["brier"]),
    }


def task_point_gate(
    candidate_by_task: Mapping[str, Mapping[str, float]],
    references_by_task: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict:
    """Inclusive point gate against both U0 and Ucal for both fixed tasks."""
    # The saved-array key is "occupation"; its source label is occupation_group.
    if set(candidate_by_task) != {"income", "occupation"}:
        raise ValueError("both tasks required")
    rows = {}
    for task in ("income", "occupation"):
        if set(references_by_task[task]) != {"U0", "Ucal"}:
            raise ValueError("both original frozen references required")
        rows[task] = {}
        for ref in ("U0", "Ucal"):
            cost = task_costs(candidate_by_task[task], references_by_task[task][ref])
            rows[task][ref] = {
                "cost": cost,
                "margin": {name: TASK_CAPS[name] - value for name, value in cost.items()},
                "point_pass": all(value <= TASK_CAPS[name] for name, value in cost.items()),
            }
    return {"rows": rows, "point_pass": all(row["point_pass"]
            for task_rows in rows.values() for row in task_rows.values())}


def weighted_binary_auc(scores: np.ndarray, sex: np.ndarray, row_weight: np.ndarray | None = None) -> float:
    """Tie-correct weighted Mann-Whitney AUC, used for duplicate-group resampling."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(sex)
    if (s.ndim != 1 or y.shape != s.shape or not len(s) or not np.isfinite(s).all()
            or not np.issubdtype(y.dtype, np.integer) or not np.isin(y, (0, 1)).all()):
        raise ValueError("invalid aligned SEX scores")
    w = _weights(row_weight, len(y))
    order = np.argsort(s, kind="stable")
    sorted_score, sorted_label, sorted_weight = s[order], y[order], w[order]
    block_starts = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0) + 1]
    pos = np.add.reduceat(sorted_weight * sorted_label, block_starts)
    neg = np.add.reduceat(sorted_weight * (1 - sorted_label), block_starts)
    if pos.sum() <= 0 or neg.sum() <= 0:
        raise ValueError("positive and negative SEX weight required")
    lower_negative = np.cumsum(neg) - neg
    wins = float(np.dot(pos, lower_negative + 0.5 * neg))
    return wins / float(pos.sum() * neg.sum())


def reader_recovery(
    reader_predictions: Mapping[str, np.ndarray],
    sex: np.ndarray,
    *,
    row_weight: np.ndarray | None = None,
    reverse_allowed: bool = True,
) -> dict:
    """Score every predeclared fixed reader; report the finite-reader envelope."""
    if not reader_predictions:
        raise ValueError("reader bank empty")
    values = {}
    for name, scores in reader_predictions.items():
        if not name or name in values:
            raise ValueError("reader ID missing or repeated")
        auc = weighted_binary_auc(scores, sex, row_weight)
        values[name] = max(auc, 1 - auc) if reverse_allowed else auc
    winner = max(values, key=lambda name: (values[name], -list(values).index(name)))
    return {"reader_auc": values, "winner": winner, "auc": values[winner]}


def comparison_margins(
    candidate_auc: Mapping[str, float], control_auc: Mapping[str, float]
) -> dict[str, float]:
    """Nonnegative margins meet the fixed local and coalition allowances."""
    if set(candidate_auc) != {"r1", "r2", "pair"} or set(control_auc) != set(candidate_auc):
        raise ValueError("all three aligned views required")
    return {
        "r1_local": LOCAL_AUC_CAP - (candidate_auc["r1"] - control_auc["r1"]),
        "r2_local": LOCAL_AUC_CAP - (candidate_auc["r2"] - control_auc["r2"]),
        "pair_benefit": control_auc["pair"] - candidate_auc["pair"] - COALITION_BENEFIT_TARGET,
    }


def diagnostic_fallback(
    costs_by_arm: Mapping[str, Sequence[Mapping[str, float]]],
    tie_order: Sequence[str],
) -> dict:
    """Task-only fallback; all supplied seed/task/reference costs enter the max."""
    if not costs_by_arm or set(costs_by_arm) != set(tie_order):
        raise ValueError("fallback arm registry or deterministic tie order differs")
    violations = {}
    for arm, rows in costs_by_arm.items():
        if not rows:
            raise ValueError("fallback lacks fixed task/reference costs")
        normalized = [max(0.0, (float(row[name]) - cap) / cap)
                      for row in rows for name, cap in TASK_CAPS.items()]
        if not np.isfinite(normalized).all():
            raise ValueError("nonfinite fallback cost")
        violations[arm] = max(normalized)
    selected = min(tie_order, key=lambda arm: (violations[arm], tie_order.index(arm)))
    return {"arm": selected, "worst_normalized_violation": violations[selected],
            "all_violations": violations, "status": "INELIGIBLE_DESCRIPTIVE"}


def strongest_matched_control(
    pair_auc_by_control: Mapping[str, float],
    eligible_ids: Sequence[str],
    tie_order: Sequence[str],
) -> dict:
    """The strongest privacy comparator has the *lowest* coalition AUC."""
    eligible = tuple(eligible_ids)
    if (not eligible or len(set(eligible)) != len(eligible)
            or not set(eligible).issubset(pair_auc_by_control)
            or set(tie_order) != set(pair_auc_by_control)):
        raise ValueError("matched control membership incomplete")
    for value in pair_auc_by_control.values():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("control coalition AUC invalid")
    chosen = min(eligible, key=lambda arm: (pair_auc_by_control[arm], tie_order.index(arm)))
    return {"control": chosen, "pair_auc": float(pair_auc_by_control[chosen]),
            "eligible_ids": list(eligible)}


def nominate_within_family(
    candidates: Mapping[str, Mapping],
    *,
    required_control_ids: Sequence[str],
    tie_order: Sequence[str],
) -> dict:
    """Replay fixed validity, point, local, worst-benefit and simplicity order.

    Each candidate record contains `admission_valid`, `completed_controls`,
    `point_feasible_all_seeds`, `local_margins_by_control`,
    `coalition_benefit_by_control`, and a predeclared integer
    `simplicity_rank` (smaller is simpler). Missing coverage cannot win.
    """
    required = tuple(required_control_ids)
    if (not candidates or not required or len(set(required)) != len(required)
            or set(candidates) != set(tie_order)):
        raise ValueError("candidate or comparator registry differs")
    ranking = []
    excluded = {}
    for arm, row in candidates.items():
        if not row["admission_valid"]:
            excluded[arm] = "INVALID_ADMISSION"
            continue
        if not set(required).issubset(row["completed_controls"]):
            excluded[arm] = "INCOMPLETE_MANDATORY_CONTROLS"
            continue
        if not row["point_feasible_all_seeds"]:
            excluded[arm] = "POINT_UTILITY_INFEASIBLE"
            continue
        local = row["local_margins_by_control"]
        benefit = row["coalition_benefit_by_control"]
        if not set(required).issubset(local) or not set(required).issubset(benefit):
            excluded[arm] = "INCOMPLETE_MATCHED_COMPARISON"
            continue
        if any(set(local[control]) != {"r1", "r2"}
               or any(not np.isfinite(local[control][view]) for view in ("r1", "r2"))
               for control in required):
            raise ValueError("local comparison invalid")
        if any(local[control][view] < 0 for control in required for view in ("r1", "r2")):
            excluded[arm] = "POINT_LOCAL_GUARD_FAILED"
            continue
        worst_benefit = min(float(benefit[control]) for control in required)
        if not np.isfinite(worst_benefit):
            raise ValueError("nonfinite coalition benefit")
        simplicity = row["simplicity_rank"]
        if not isinstance(simplicity, int) or simplicity < 0:
            raise ValueError("sealed simplicity rank invalid")
        ranking.append((arm, worst_benefit, simplicity))
    if not ranking:
        return {"nominee": None, "status": "NO_ELIGIBLE_NOMINEE", "excluded": excluded}
    winner = min(ranking, key=lambda item: (-item[1], item[2], tie_order.index(item[0])))
    return {"nominee": winner[0], "worst_control_benefit": winner[1],
            "status": "POINT_LEAD" if winner[1] >= COALITION_BENEFIT_TARGET else "BELOW_BENEFIT_TARGET",
            "eligible": [arm for arm, _, _ in ranking], "excluded": excluded}


@dataclass(frozen=True)
class SimultaneousBounds:
    names: tuple[str, ...]
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    critical: float
    numerical_constant: np.ndarray

    def statuses(self) -> dict[str, str]:
        """All names are favorable margins, so PASS needs lower >= 0."""
        return {name: ("PASS" if self.lower[i] >= 0 else
                       "RULED_OUT_OR_VIOLATION" if self.upper[i] < 0 else "UNRESOLVED")
                for i, name in enumerate(self.names)}


def simultaneous_group_bounds(
    group_id: np.ndarray,
    names: Sequence[str],
    evaluate: Callable[[np.ndarray], Sequence[float]],
    *,
    seed: int,
    replicates: int,
    confidence: float = 0.95,
    floor: float = 1e-12,
) -> SimultaneousBounds:
    """Paired group bootstrap over the entire caller-supplied finite family.

    Every replicate applies the same person weights to every arm and model
    seed. The caller must seal `names`, seed, replicate count, and all arm
    membership before seeing outcomes. A missing arm cannot shrink this family.
    """
    g = np.asarray(group_id)
    labels = tuple(names)
    if (g.ndim != 1 or not len(g) or not labels or len(set(labels)) != len(labels)
            or replicates < 2 or not (0 < confidence < 1) or floor <= 0):
        raise ValueError("invalid simultaneous family")
    groups, inverse = np.unique(g, return_inverse=True)
    point = np.asarray(evaluate(np.ones(len(g), dtype=np.float64)), dtype=np.float64)
    if point.shape != (len(labels),) or not np.isfinite(point).all():
        raise ValueError("invalid point margin vector")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    sample = np.empty((replicates, len(labels)), dtype=np.float64)
    for b in range(replicates):
        indices = rng.integers(0, len(groups), size=len(groups))
        multiplicity = np.bincount(indices, minlength=len(groups))[inverse]
        sample[b] = evaluate(multiplicity.astype(np.float64))
    if not np.isfinite(sample).all():
        raise ValueError("nonfinite bootstrap margins")
    nearly_fixed = np.max(np.abs(sample - point), axis=0) <= floor
    standard_error = np.std(sample, axis=0, ddof=1)
    scales = np.maximum(standard_error, floor)
    standardized = np.abs((sample - point) / scales)
    standardized[:, nearly_fixed] = 0.0
    maxima = np.max(standardized, axis=1)
    critical = float(np.quantile(maxima, confidence, method="higher"))
    halfwidth = critical * scales
    halfwidth[nearly_fixed] = 0.0
    return SimultaneousBounds(labels, point, point - halfwidth, point + halfwidth,
                              critical, nearly_fixed)
