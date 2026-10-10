"""Exactly 567 prospective D margins for one paired INNER max-t family.

Nine fixed releases (seven D arms and U0/Ucal) are scored once per seed and
bootstrap draw, then all finite contrasts reuse those scores. The same group
resample positions are passed to both seeds and every arm. No fit or data load.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

_SHARED = Path(__file__).resolve().parents[2] / "pcrl_method_portfolio_v1" / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
from attack_eval import oriented_auc
from common_eval import assert_same_people, task_metrics
from paired_inference import paired_group_inference

D_ARMS = ("D_T", "D_K", "D_L", "D_S12", "D_S21", "D_C", "D_F")
U_ANCHORS = ("U0", "Ucal")
VIEWS = ("r1", "r2", "pair")
TASKS = (("income", "prob_1"), ("occupation", "prob_2"))
METRICS = ("accuracy", "nll", "brier")
SUMMARIES = ("s0", "s1", "mean")


def d_margin_names() -> tuple[str, ...]:
    """252 task + 189 all-arm privacy + 126 U-anchor privacy clauses."""
    base = []
    for arm in D_ARMS:
        for task, _ in TASKS:
            for ref in U_ANCHORS:
                for metric in METRICS:
                    base.append(f"utility:{arm}:{task}:{ref}:{metric}")
    for first, second in itertools.combinations(D_ARMS, 2):
        for view in VIEWS:
            base.append(f"arm_pair:{first}:{second}:{view}:auc")
    for arm in D_ARMS:
        for ref in U_ANCHORS:
            for view in VIEWS:
                base.append(f"u_anchor:{arm}:{ref}:{view}:auc")
    names = tuple(f"{summary}:{name}" for summary in SUMMARIES for name in base)
    if len(names) != 567 or len(set(names)) != len(names):
        raise AssertionError("D finite family size differs")
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


def _one_seed(item: Mapping, positions: np.ndarray) -> np.ndarray:
    truth = item["truth"]
    releases = item["releases"]
    if set(releases) != set(D_ARMS).union(U_ANCHORS):
        raise ValueError("fixed D release registry coverage missing")
    try:
        assert_same_people(truth, *releases.values())
    except (KeyError, ValueError) as exc:
        raise ValueError("ordered D release people/groups differ") from exc
    n = len(truth["row_id"])
    if positions.ndim != 1 or not len(positions) or np.any((positions < 0) | (positions >= n)):
        raise ValueError("paired resample positions invalid")
    scored = {name: _release_scores(releases[name], truth, positions)
              for name in (*D_ARMS, *U_ANCHORS)}
    values = []
    for arm in D_ARMS:
        for task, _ in TASKS:
            for ref in U_ANCHORS:
                for metric in METRICS:
                    values.append(scored[arm]["task"][task][metric] -
                                  scored[ref]["task"][task][metric])
    for first, second in itertools.combinations(D_ARMS, 2):
        for view in VIEWS:
            values.append(scored[first]["auc"][view] - scored[second]["auc"][view])
    for arm in D_ARMS:
        for ref in U_ANCHORS:
            for view in VIEWS:
                values.append(scored[arm]["auc"][view] - scored[ref]["auc"][view])
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (189,) or not np.isfinite(result).all():
        raise ValueError("D seed margin vector differs")
    return result


def d_margin_at(seed_records: Mapping[int, Mapping], positions) -> tuple[tuple[str, ...], np.ndarray]:
    """Raw signed contrasts for s0, s1 and their equally weighted mean."""
    if set(seed_records) != {0, 1}:
        raise ValueError("fixed D screen requires paired seeds zero and one")
    try:
        assert_same_people(seed_records[0]["truth"], seed_records[1]["truth"])
    except (KeyError, ValueError) as exc:
        raise ValueError("ordered D seed people/groups differ") from exc
    pos = np.asarray(positions, dtype=np.int64)
    s0, s1 = (_one_seed(seed_records[seed], pos) for seed in (0, 1))
    return d_margin_names(), np.concatenate((s0, s1, (s0 + s1) / 2))


def run_d_inference(seed_records: Mapping[int, Mapping], private_seed: bytes) -> dict:
    """Inherited 2048 paired-group simultaneous max-|t| over all 567 margins."""
    if set(seed_records) != {0, 1}:
        raise ValueError("fixed D screen requires paired seeds zero and one")
    unit = np.asarray(seed_records[0]["truth"]["unit"], dtype=np.int64)
    names = d_margin_names()

    def at(positions: np.ndarray, draw: int) -> np.ndarray:
        if draw != 0:
            raise ValueError("D has no stochastic release draw")
        got, values = d_margin_at(seed_records, positions)
        if got != names:
            raise ValueError("D finite margin schema differs")
        return values

    report = paired_group_inference(unit, names, at, private_seed)
    # These are raw signed differences. The inherited helper's zero-threshold
    # PASS/VIOLATION labels do not encode this campaign's nonzero utility and
    # privacy allowances, nor the chosen direction of each arm pair.
    for clause in report["clauses"]:
        clause["status"] = "BOUND_ONLY"
    report["finite_family"] = "D_7_ARMS_2_U_ANCHORS_2_SEEDS_AND_MEAN_V1"
    report["clause_count"] = len(names)
    report["seed_summaries"] = SUMMARIES
    report["contrast_convention"] = "raw first-minus-second; apply sealed allowances and desired direction to simultaneous lower/upper bounds"
    return report
