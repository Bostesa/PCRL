"""Prospective A endpoint family and paired people × realized-release inference.

Ten same-bank arms, four fixed references, two source seeds, and three actual
sampled releases give 4,455 ordered clauses. No candidate model is imported.
The complete saved INNER release and fixed reader predictions are the inputs.
"""

from __future__ import annotations

import hashlib
import itertools
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.stats import t as student_t

_SHARED = Path(__file__).resolve().parents[2] / "pcrl_method_portfolio_v1" / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
from attack_eval import oriented_auc
from common_eval import assert_same_people, task_metrics

A_ARMS = ("A_JOINT_STOCH", "A_TASK_STOCH", "A_TASK_DET", "A_JOINT_DET",
          "A_CENTRAL_PRODUCT", "A_LOCAL", "A_SEQ12", "A_SEQ21",
          "A_IDENTITY", "A_CLASS_ONLY")
STOCHASTIC_ARMS = frozenset(("A_JOINT_STOCH", "A_TASK_STOCH", "A_CENTRAL_PRODUCT",
                             "A_LOCAL", "A_SEQ12", "A_SEQ21"))
A_REFS = ("T_STAR", "P_STAR", "U0", "Ucal")
VIEWS = ("r1", "r2", "pair")
TASKS = (("income", "prob_1"), ("occupation", "prob_2"))
METRICS = ("accuracy", "nll", "brier")
SUMMARIES = ("s0_d0", "s0_d1", "s0_d2", "s1_d0", "s1_d1", "s1_d2",
             "s0_mean", "s1_mean", "mean")
BASE_CLAUSES = 495
A_CLAUSES = 4455
REPLICATES = 2048
INNER_GROUPS = 2234
STUDENTIZATION_FLOOR = 1e-12
ALPHA = 0.05
GROUP_BOOTSTRAP_SEED = 20261010
RELEASE_BOOTSTRAP_SEED = 20261011
GROUP_BOOTSTRAP_SEED_BYTES = GROUP_BOOTSTRAP_SEED.to_bytes(16, "big") + hashlib.sha256(
    b"pcrl-latent-sampled-inference-v1").digest()[:16]
GROUP_BOOTSTRAP_SEED_SHA256 = hashlib.sha256(GROUP_BOOTSTRAP_SEED_BYTES).hexdigest()
RELEASE_BOOTSTRAP_SEED_BYTES = RELEASE_BOOTSTRAP_SEED.to_bytes(16, "big") + hashlib.sha256(
    b"pcrl-a-release-inference-v1").digest()[:16]
RELEASE_BOOTSTRAP_SEED_SHA256 = hashlib.sha256(RELEASE_BOOTSTRAP_SEED_BYTES).hexdigest()


def a_margin_names() -> tuple[str, ...]:
    """Ordered K=4,455 clauses. This order is sealed before A outcome access."""
    base = []
    for arm in A_ARMS:
        for task, _ in TASKS:
            for ref in A_REFS:
                for metric in METRICS:
                    base.append(f"utility:{arm}:{task}:{ref}:{metric}")
    for first, second in itertools.combinations(A_ARMS, 2):
        for view in VIEWS:
            base.append(f"arm_pair:{first}:{second}:{view}:auc")
    for arm in A_ARMS:
        for ref in A_REFS:
            for view in VIEWS:
                base.append(f"ref_pair:{arm}:{ref}:{view}:auc")
    names = tuple(f"{summary}:{name}" for summary in SUMMARIES for name in base)
    if len(base) != BASE_CLAUSES or len(names) != A_CLAUSES or len(set(names)) != A_CLAUSES:
        raise AssertionError("A finite family dimension or uniqueness changed")
    return names


def _release_scores(release: Mapping, truth: Mapping, positions: np.ndarray) -> dict:
    tasks = {task: task_metrics(np.asarray(release[field])[positions],
                                np.asarray(truth[task])[positions], task=task)
             for task, field in TASKS}
    sex = np.asarray(truth["sex"], dtype=np.int64)[positions]
    privacy = {}
    for view in VIEWS:
        bank = release["reader_predictions"][view]
        if not bank:
            raise ValueError("fixed reader bank missing")
        privacy[view] = max(oriented_auc(sex, np.asarray(pred)[positions])["fixed_alias_max_auc"]
                            for pred in bank.values())
    return {"task": tasks, "auc": privacy}


def _validate_records(seed_records: Mapping[int, Mapping]) -> None:
    if set(seed_records) != {0, 1}:
        raise ValueError("A family requires paired seeds zero and one")
    try:
        assert_same_people(seed_records[0]["truth"], seed_records[1]["truth"])
    except (KeyError, ValueError) as exc:
        raise ValueError("ordered A seed people/groups differ") from exc
    allowed = set(A_ARMS).union(A_REFS)
    for seed in (0, 1):
        item = seed_records[seed]
        releases = item["releases"]
        if not set(releases).issubset(allowed):
            raise ValueError("unregistered A release entered family")
        for arm, realized in releases.items():
            expected = {0, 1, 2} if arm in STOCHASTIC_ARMS else {0}
            if set(realized) != expected:
                raise ValueError(f"{arm} incomplete or extra realized release set")
            try:
                assert_same_people(item["truth"], *realized.values())
            except (KeyError, ValueError) as exc:
                raise ValueError("ordered A release people/groups differ") from exc


def _base_vector(scores: Mapping[str, dict | None]) -> np.ndarray:
    vals = []

    def difference(first: str, second: str, scope: str, target: str) -> float:
        a, b = scores.get(first), scores.get(second)
        return float("nan") if a is None or b is None else a[scope][target] - b[scope][target]

    for arm in A_ARMS:
        for task, _ in TASKS:
            for ref in A_REFS:
                for metric in METRICS:
                    a, b = scores.get(arm), scores.get(ref)
                    vals.append(float("nan") if a is None or b is None else
                                a["task"][task][metric] - b["task"][task][metric])
    for first, second in itertools.combinations(A_ARMS, 2):
        for view in VIEWS:
            vals.append(difference(first, second, "auc", view))
    for arm in A_ARMS:
        for ref in A_REFS:
            for view in VIEWS:
                vals.append(difference(arm, ref, "auc", view))
    out = np.asarray(vals, dtype=np.float64)
    if out.shape != (BASE_CLAUSES,) or np.any(np.isinf(out)):
        raise ValueError("A base margin shape or value invalid")
    return out


def _margin_at(seed_records: Mapping[int, Mapping], positions: np.ndarray,
               draw_indices: Mapping[int, tuple[int, int, int]] | None) -> np.ndarray:
    selected = {0: (0, 1, 2), 1: (0, 1, 2)} if draw_indices is None else draw_indices
    if set(selected) != {0, 1} or any(len(selected[s]) != 3 or
                                       any(d not in (0, 1, 2) for d in selected[s])
                                       for s in (0, 1)):
        raise ValueError("A draw-index resample must have three indices per seed")
    scored = {}
    for seed in (0, 1):
        item = seed_records[seed]
        for arm in (*A_ARMS, *A_REFS):
            realized = item["releases"].get(arm)
            if realized is None:
                continue
            for draw, release in realized.items():
                scored[seed, arm, draw] = _release_scores(release, item["truth"], positions)
    base = {}
    for seed in (0, 1):
        for draw in range(3):
            view = {arm: scored.get((seed, arm, draw if arm in STOCHASTIC_ARMS else 0))
                    for arm in (*A_ARMS, *A_REFS)}
            base[seed, draw] = _base_vector(view)
    mean0 = np.mean([base[0, d] for d in selected[0]], axis=0)
    mean1 = np.mean([base[1, d] for d in selected[1]], axis=0)
    result = np.concatenate([base[seed, draw] for seed in (0, 1) for draw in range(3)] +
                            [mean0, mean1, (mean0 + mean1) / 2])
    if result.shape != (A_CLAUSES,):
        raise ValueError("A family margin vector width changed")
    return result


def a_margin_at(seed_records: Mapping[int, Mapping], positions,
                draw_indices: Mapping[int, tuple[int, int, int]] | None = None) -> tuple[tuple[str, ...], np.ndarray]:
    """Evaluate fixed readers on paired INNER positions; absent arms yield NaNs."""
    _validate_records(seed_records)
    pos = np.asarray(positions, dtype=np.int64)
    n = len(seed_records[0]["truth"]["row_id"])
    if pos.ndim != 1 or not len(pos) or np.any((pos < 0) | (pos >= n)):
        raise ValueError("paired A resample positions invalid")
    return a_margin_names(), _margin_at(seed_records, pos, draw_indices)


def crossed_resample(unit: np.ndarray, group_rng: np.random.Generator,
                     release_rng: np.random.Generator) -> tuple[np.ndarray, dict[int, tuple[int, int, int]]]:
    """One group sample shared by every arm/seed; coupled draw IDs per seed."""
    groups = np.asarray(unit, dtype=np.int64)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("A group vector invalid")
    ordered = np.unique(groups)
    locations = {int(g): np.flatnonzero(groups == g) for g in ordered}
    sampled = group_rng.choice(ordered, size=len(ordered), replace=True)
    positions = np.concatenate([locations[int(g)] for g in sampled])
    draws = {seed: tuple(int(v) for v in release_rng.integers(0, 3, size=3))
             for seed in (0, 1)}
    return positions, draws


def summarize_a_bootstrap(names: tuple[str, ...], point: np.ndarray,
                          boot: np.ndarray) -> tuple[list[dict], str, float]:
    """Keep all K slots even when an entire source arm is unavailable."""
    if names != a_margin_names():
        raise ValueError("A finite family order changed")
    point = np.asarray(point, dtype=np.float64)
    boot = np.asarray(boot, dtype=np.float64)
    if point.shape != (A_CLAUSES,) or np.any(np.isinf(point)):
        raise ValueError("A point vector invalid")
    available = np.isfinite(point)
    if not np.any(available) or boot.shape != (REPLICATES, int(available.sum())) or not np.isfinite(boot).all():
        raise ValueError("A bootstrap vector invalid")
    estimate = point[available]
    se = boot.std(axis=0, ddof=1)
    zero = np.max(np.abs(boot - estimate), axis=0) <= STUDENTIZATION_FLOOR
    se[zero] = 0.0
    scale = np.maximum(se, STUDENTIZATION_FLOOR)
    if available.all():
        scaled = np.zeros_like(boot)
        scaled[:, ~zero] = np.abs((boot[:, ~zero] - estimate[~zero]) / scale[~zero])
        critical = float(np.quantile(scaled.max(axis=1), 1 - ALPHA, method="higher"))
        clause_critical = np.full(len(estimate), critical)
        critical_family = "two-sided-maximum-absolute-studentized-full-K"
    else:
        # The denominator remains the predeclared K, including every missing
        # arm clause. A max over the available subset would silently shrink it.
        # Three-release means have only two release degrees of freedom, even
        # though each fixed release scores 2,234 independent INNER groups.
        level = 1 - ALPHA / (2 * A_CLAUSES)
        group_critical = float(student_t.ppf(level, INNER_GROUPS - 1))
        release_critical = float(student_t.ppf(level, 2))
        clause_critical = np.asarray([
            group_critical if name.startswith(("s0_d", "s1_d")) else release_critical
            for name, included in zip(names, available) if included], dtype=np.float64)
        critical = float(clause_critical.max())
        critical_family = "two-sided-fixed-K-Bonferroni-t-approximation"
    width = clause_critical * np.where(zero, 0.0, scale)
    present = iter(zip(estimate, se, estimate - width, estimate + width, zero))
    clauses = []
    for name, is_available in zip(names, available):
        if not is_available:
            clauses.append({"name": name, "point_margin": None, "se": None,
                            "lower": None, "upper": None,
                            "status": "INCOMPLETE_MISSING_ARM", "numerical_se_status": None})
            continue
        p, s, lo, hi, z = next(present)
        clauses.append({"name": name, "point_margin": float(p), "se": float(s),
                        "lower": float(lo), "upper": float(hi), "status": "BOUND_ONLY",
                        "numerical_se_status": "NUMERICALLY_CONSTANT" if z else "VARIABLE"})
    return clauses, critical_family, critical


def run_a_inference(seed_records: Mapping[int, Mapping]) -> dict:
    """2048 crossed resamples; max-|t| when complete, fixed-K Bonferroni if incomplete."""
    _validate_records(seed_records)
    unit = np.asarray(seed_records[0]["truth"]["unit"], dtype=np.int64)
    if unit.ndim != 1 or np.unique(unit).size != INNER_GROUPS:
        raise ValueError("A INNER_SELECTION requires exactly 2234 groups")
    names = a_margin_names()
    point = _margin_at(seed_records, np.arange(len(unit), dtype=np.int64), None)
    available = np.isfinite(point)
    if np.any(np.isinf(point)) or not np.any(available):
        raise ValueError("no finite A clauses")
    group_rng = np.random.Generator(np.random.PCG64DXSM(GROUP_BOOTSTRAP_SEED))
    release_rng = np.random.Generator(np.random.PCG64DXSM(RELEASE_BOOTSTRAP_SEED))
    boot = np.empty((REPLICATES, int(available.sum())), dtype=np.float64)
    for b in range(REPLICATES):
        idx, draw_indices = crossed_resample(unit, group_rng, release_rng)
        sampled = _margin_at(seed_records, idx, draw_indices)
        if not np.array_equal(np.isfinite(sampled), available):
            raise ValueError("A finite-family availability changed during bootstrap")
        boot[b] = sampled[available]
    clauses, critical_family, critical = summarize_a_bootstrap(names, point, boot)
    missing_arms = {str(seed): [arm for arm in (*A_ARMS, *A_REFS)
                                if arm not in seed_records[seed]["releases"]] for seed in (0, 1)}
    return {"schema": "pcrl-a-crossed-group-release-inference-v1", "finite_family": "A_10_ARMS_4_REFS_9_SUMMARIES_V1",
            "clause_count": A_CLAUSES, "available_clause_count": int(available.sum()),
            "missing_clause_count": int((~available).sum()), "missing_arms_by_seed": missing_arms,
            "family_complete": bool(available.all()), "bootstrap_replicates": REPLICATES,
            "inner_groups": INNER_GROUPS, "source_seeds": [0, 1], "realizations_per_stochastic_arm": 3,
            "seed_summaries": list(SUMMARIES), "mechanism_draws_in_primary_interval": 3,
            "release_resample": "three coupled complete draw IDs with replacement per seed for mean summaries; draw-specific clauses keep original release ID",
            "critical_family": critical_family, "critical_95": critical,
            "bonferroni_degrees_of_freedom_if_incomplete": {
                "fixed_draw_group_only": INNER_GROUPS - 1,
                "three_release_seed_or_grand_mean": 2},
            "fixed_family_denominator": A_CLAUSES, "studentization_floor": STUDENTIZATION_FLOOR,
            "numpy_version": np.__version__, "group_bootstrap_seed_rule": "PCG64DXSM(20261010)",
            "group_bootstrap_seed_bytes_sha256": GROUP_BOOTSTRAP_SEED_SHA256,
            "release_bootstrap_seed_rule": "PCG64DXSM(20261011)",
            "release_bootstrap_seed_bytes_sha256": RELEASE_BOOTSTRAP_SEED_SHA256,
            "contrast_convention": "raw first-minus-second; apply sealed allowances/directions to bounds",
            "missing_rule": "keep all K names and K denominator; mark unavailable clauses incomplete; available clauses use fixed-K Bonferroni t approximation",
            "clauses": clauses}
