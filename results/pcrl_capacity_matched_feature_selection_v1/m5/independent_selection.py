"""Independent fitting-stage selection replay from saved aggregate score inputs.

No M2/M3 selector, table, or scorer is imported. This verifies the finite
bank's decisions; final INNER endpoints and strongest controls are separate.
"""

from __future__ import annotations

from itertools import product
from math import isclose, isfinite
from typing import Mapping


PURPOSES = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal", "F")
UTILITY_CAPS = {"accuracy": 0.010, "nll": 0.010, "brier": 0.005}
LOCAL_CAP = 0.010


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _task_masks(table: Mapping, references: Mapping) -> dict[str, dict[str, dict]]:
    if set(references) != set(PURPOSES):
        raise ValueError("task references incomplete")
    result = {}
    for purpose, key in (("income", "income_masks"),
                         ("occupation_group", "occupation_masks")):
        if set(references[purpose]) != set(REFERENCES):
            raise ValueError("U0/Ucal/F references incomplete")
        rows = table[key]
        if not isinstance(rows, list) or not rows:
            raise ValueError("empty mask bank")
        by_id = {}
        for row in rows:
            mid = row["mask_id"]
            if mid in by_id or not mid or not row["mask_sha256"]:
                raise ValueError("duplicate or unidentified mask")
            accuracy = _finite(row["task_validation_accuracy"], "accuracy")
            nll = _finite(row["task_validation_nll"], "nll")
            brier = _finite(row["task_validation_brier"], "brier")
            auc = _finite(row["local_search_auc"], "local AUC")
            if not (0 <= accuracy <= 1 and nll >= 0 and 0 <= brier <= 2
                    and 0.5 <= auc <= 1):
                raise ValueError("mask endpoint out of range")
            margins = row["utility_margins_by_ref_metric"]
            if set(margins) != set(REFERENCES):
                raise ValueError("utility reference margins incomplete")
            feasible = True
            for ref_id in REFERENCES:
                ref = references[purpose][ref_id]
                if set(ref) != {"accuracy", "nll", "brier"} or set(margins[ref_id]) != set(ref):
                    raise ValueError("utility metric schema differs")
                expected = {
                    "accuracy": accuracy - (_finite(ref["accuracy"], "ref accuracy")
                                            - UTILITY_CAPS["accuracy"]),
                    "nll": _finite(ref["nll"], "ref NLL") + UTILITY_CAPS["nll"] - nll,
                    "brier": _finite(ref["brier"], "ref Brier")
                             + UTILITY_CAPS["brier"] - brier,
                }
                for metric, value in expected.items():
                    reported = _finite(margins[ref_id][metric], "utility margin")
                    if not isclose(reported, value, rel_tol=0, abs_tol=1e-12):
                        raise ValueError("utility margin differs from raw metric/reference")
                    feasible &= value >= 0
            if type(row["utility_feasible"]) is not bool or row["utility_feasible"] != feasible:
                raise ValueError("utility feasibility differs from independent margins")
            by_id[mid] = row
        result[purpose] = by_id
    return result


def _pair_rows(table: Mapping, masks: dict) -> dict[tuple[str, str], dict]:
    pairs = {}
    union_score = {}
    expected = set(product(masks["income"], masks["occupation_group"]))
    for row in table["pairs"]:
        a, b = row["income_mask_id"], row["occupation_mask_id"]
        key = (a, b)
        if key not in expected or key in pairs or row["pair_id"] != a + "__" + b:
            raise ValueError("pair bank is duplicate, incomplete or foreign")
        first, second = masks["income"][a], masks["occupation_group"][b]
        if (row["income_mask_sha256"] != first["mask_sha256"] or
                row["occupation_mask_sha256"] != second["mask_sha256"]):
            raise ValueError("pair mask identity differs")
        expected_loss = (first["task_validation_nll"] +
                         second["task_validation_nll"])
        if not isclose(_finite(row["task_loss"], "task loss"), expected_loss,
                       rel_tol=0, abs_tol=1e-12):
            raise ValueError("pair task loss differs")
        for field, value in (("income_local_search_auc", first["local_search_auc"]),
                             ("occupation_local_search_auc", second["local_search_auc"])):
            if not isclose(_finite(row[field], field), value, rel_tol=0, abs_tol=1e-12):
                raise ValueError("pair local AUC differs")
        feasible = first["utility_feasible"] and second["utility_feasible"]
        if type(row["pair_feasible"]) is not bool or row["pair_feasible"] != feasible:
            raise ValueError("pair utility feasibility differs")
        auc = _finite(row["coalition_search_auc"], "coalition AUC")
        if not 0.5 <= auc <= 1:
            raise ValueError("coalition search AUC is not oriented")
        union = row["union_sha256"]
        groups = row["union_group_count"]
        columns = row["union_encoded_columns"]
        if type(groups) is not int or type(columns) is not int or groups < 1 or columns < groups:
            raise ValueError("union capacity invalid")
        if "union_columns" in row and len(row["union_columns"]) != columns:
            raise ValueError("union column count differs")
        if union in union_score and union_score[union] != (auc, groups, columns):
            raise ValueError("raw union critic cache or geometry forks by allocation")
        union_score[union] = (auc, groups, columns)
        pairs[key] = row
    if set(pairs) != expected:
        raise ValueError("pair bank does not cover full Cartesian product")
    return pairs


def replay_fit_selection(table: Mapping, references: Mapping) -> dict:
    """Return all ten rule IDs from independently validated saved table inputs."""
    if table.get("critic_input_kind") != "RAW_UNION_ONLY":
        raise ValueError("union-cached coalition critic must be RAW_UNION_ONLY")
    masks = _task_masks(table, references)
    pairs = _pair_rows(table, masks)
    income, occupation = masks["income"], masks["occupation_group"]
    first_t = min((m for m in income.values() if m["utility_feasible"]),
                  key=lambda m: (m["task_validation_nll"], m["mask_sha256"]))
    second_t = min((m for m in occupation.values() if m["utility_feasible"]),
                   key=lambda m: (m["task_validation_nll"], m["mask_sha256"]))
    t = pairs[first_t["mask_id"], second_t["mask_id"]]

    def legal(quota=None):
        return [row for row in pairs.values() if (
            row["pair_feasible"] and
            row["income_local_search_auc"] <= t["income_local_search_auc"] + LOCAL_CAP and
            row["occupation_local_search_auc"] <= t["occupation_local_search_auc"] + LOCAL_CAP and
            (quota is None or (row["union_group_count"], row["union_encoded_columns"]) == quota)
        )]

    def tie(row):
        return (row["task_loss"], row["income_mask_sha256"],
                row["occupation_mask_sha256"], row["pair_id"])

    def seq(rows, purpose):
        if not rows:
            return None
        field = "income_mask_id" if purpose == "income" else "occupation_mask_id"
        scores = masks[purpose]
        candidate_ids = {row[field] for row in rows}
        first = min(candidate_ids, key=lambda mid: (
            scores[mid]["local_search_auc"], scores[mid]["task_validation_nll"],
            scores[mid]["mask_sha256"], mid))
        return min((row for row in rows if row[field] == first),
                   key=lambda row: (row["coalition_search_auc"], *tie(row)))

    valid = legal()
    if not valid or t not in valid:
        raise ValueError("task-only reference or legal joint bank invalid")
    joint = min(valid, key=lambda row: (row["coalition_search_auc"], *tie(row)))
    legal_income = {row["income_mask_id"] for row in valid}
    legal_occupation = {row["occupation_mask_id"] for row in valid}
    i_first = min(legal_income, key=lambda mid: (
        income[mid]["local_search_auc"], income[mid]["task_validation_nll"],
        income[mid]["mask_sha256"], mid))
    i_second = min(legal_occupation, key=lambda mid: (
        occupation[mid]["local_search_auc"], occupation[mid]["task_validation_nll"],
        occupation[mid]["mask_sha256"], mid))
    independent = pairs[i_first, i_second]
    if independent not in valid:
        raise ValueError("local-independent choice lacks a legal pair")
    quota = joint["union_group_count"], joint["union_encoded_columns"]
    matched = legal(quota)
    selected = {
        "J": joint, "T": t, "I": independent,
        "S12": seq(valid, "income"), "S21": seq(valid, "occupation_group"),
        "SHARED": min((row for row in valid if row["income_mask_id"] == row["occupation_mask_id"]),
                      key=lambda row: (row["coalition_search_auc"], *tie(row)), default=None),
        "T_U": min(matched, key=tie, default=None),
        "I_U": min(matched, key=lambda row: (
            row["income_local_search_auc"] + row["occupation_local_search_auc"],
            *tie(row)), default=None),
        "S12_U": seq(matched, "income"),
        "S21_U": seq(matched, "occupation_group"),
    }
    selected_ids = {rule: None if row is None else row["pair_id"]
                    for rule, row in selected.items()}
    return {
        "selected_pair_ids": selected_ids,
        "union_quota": quota,
        "legal_pair_count": len(valid),
        "union_matched_legal_pair_count": len(matched),
        "same_pair_as_joint": [rule for rule, pair_id in selected_ids.items()
                               if rule != "J" and pair_id == joint["pair_id"]],
    }
