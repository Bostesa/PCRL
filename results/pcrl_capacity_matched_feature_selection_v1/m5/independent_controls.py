"""Independent strongest-control replay from saved aggregate endpoints.

This code imports no M3 selection rule or M4 scorer. It verifies identities and
recomputes every point utility flag from reported numerical differences.
"""

from __future__ import annotations

from math import isfinite
from typing import Mapping, Sequence


RULES = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")
MAIN_ORDER = RULES[1:]
ATTRIBUTION_ORDER = ("T_U", "I_U", "S12_U", "S21_U", "T", "I", "S12", "S21", "SHARED")
PURPOSES = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal", "F")
PROFILE_FIELDS = ("numeric_groups", "categorical_groups", "encoded_columns_max")


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _utility(endpoint: Mapping) -> bool:
    tasks = endpoint.get("task")
    if not isinstance(tasks, Mapping) or set(tasks) != set(PURPOSES):
        raise ValueError("endpoint task set incomplete")
    decisions = []
    for purpose in PURPOSES:
        refs = tasks[purpose].get("references")
        if not isinstance(refs, Mapping) or set(refs) != set(REFERENCES):
            raise ValueError("endpoint U0/Ucal/F reference set incomplete")
        for ref in REFERENCES:
            row = refs[ref]
            accuracy = _number(row["accuracy_difference"], "accuracy difference")
            nll = _number(row["nll_excess"], "NLL excess")
            brier = _number(row["brier_excess"], "Brier excess")
            eligible = accuracy >= -0.010 and nll <= 0.010 and brier <= 0.005
            if type(row.get("point_eligible")) is not bool or row["point_eligible"] != eligible:
                raise ValueError("reported point utility flag differs from numeric clause")
            decisions.append(eligible)
    return all(decisions)


def _profile(selection: Mapping) -> tuple[tuple[int, int, int], ...]:
    profiles = selection.get("recipient_capacity_signatures")
    if not isinstance(profiles, Mapping) or set(profiles) != set(PURPOSES):
        raise ValueError("recipient capacity profiles incomplete")
    result = []
    for purpose in PURPOSES:
        item = profiles[purpose]
        if not all(type(item.get(field)) is int for field in PROFILE_FIELDS):
            raise ValueError("recipient profile fields invalid")
        result.append(tuple(item[field] for field in PROFILE_FIELDS))
    return tuple(result)


def replay_strongest_controls(
    plans_by_seed: Mapping[int, Mapping],
    endpoints_by_seed: Mapping[int, Mapping[str, Mapping]], *,
    expected_seeds: Sequence[int],
) -> dict:
    """Independently rank eligible main and equal-union controls by mean AUC."""
    seeds = tuple(expected_seeds)
    if (not seeds or len(set(seeds)) != len(seeds) or
            set(plans_by_seed) != set(seeds) or set(endpoints_by_seed) != set(seeds)):
        raise ValueError("paired seed set differs from frozen study")
    utility = {}
    for seed in seeds:
        plan = plans_by_seed[seed]
        if (plan.get("schema") != "pcrl-focused-m3-control-plan-v1" or
                set(plan.get("rules", {})) != set(RULES)):
            raise ValueError("per-seed control plan differs")
        rules = plan["rules"]
        if rules["J"].get("status") != "SELECTED":
            raise ValueError("joint candidate was not selected")
        reported = endpoints_by_seed[seed]
        selected = {rule for rule in RULES if rules[rule].get("status") == "SELECTED"}
        if set(reported) != selected:
            raise ValueError("final endpoint set differs from selected rules")
        utility[seed] = {}
        pair_auc = {}
        for rule in selected:
            choice, endpoint = rules[rule], reported[rule]
            fields = ("pair_id", "income_mask_id", "occupation_mask_id",
                      "union_sha256", "recipient_capacity_signatures",
                      "union_capacity_signature")
            if (endpoint.get("seed") != seed or endpoint.get("rule_id") != rule or
                    any(endpoint.get(field) != choice.get(field) for field in fields)):
                raise ValueError("final endpoint identity differs from fitting selection")
            _profile(choice)
            utility[seed][rule] = _utility(endpoint)
            auc = _number(endpoint.get("recovery", {}).get("coalition", {}).get(
                "primary_envelope_auc"), "coalition AUC")
            if not 0.5 <= auc <= 1:
                raise ValueError("final coalition AUC is not oriented")
            pair = choice["pair_id"]
            if pair in pair_auc and abs(pair_auc[pair] - auc) > 1e-12:
                raise ValueError("same fitted mask pair has different final coalition AUC")
            pair_auc[pair] = auc
    pools = {"main": MAIN_ORDER, "attribution": ATTRIBUTION_ORDER}
    result = {"candidate_point_eligible_all_seeds": all(
        utility[seed]["J"] for seed in seeds)}
    for pool, order in pools.items():
        eligible = []
        excluded = {}
        for rule in order:
            reasons = []
            for seed in seeds:
                choice = plans_by_seed[seed]["rules"][rule]
                joint = plans_by_seed[seed]["rules"]["J"]
                if choice.get("status") != "SELECTED":
                    reasons.append(f"seed {seed}: {choice.get('status')}")
                    continue
                if _profile(choice) != _profile(joint):
                    reasons.append(f"seed {seed}: recipient capacity mismatch")
                if (pool == "attribution" and choice["union_capacity_signature"] !=
                        joint["union_capacity_signature"]):
                    reasons.append(f"seed {seed}: union capacity mismatch")
                if not utility[seed][rule]:
                    reasons.append(f"seed {seed}: point utility shortfall")
            if reasons:
                excluded[rule] = reasons
            else:
                eligible.append(rule)
        if not eligible:
            result[pool] = {"status": "INCOMPLETE_NO_ELIGIBLE_CONTROL",
                            "rule_id": None, "eligible_rule_ids": [],
                            "excluded_rule_ids": excluded}
            continue
        def mean_auc(rule: str) -> float:
            return sum(endpoints_by_seed[seed][rule]["recovery"]["coalition"][
                "primary_envelope_auc"] for seed in seeds) / len(seeds)
        winner = min(eligible, key=lambda rule: (mean_auc(rule), order.index(rule)))
        result[pool] = {
            "status": "SELECTED", "rule_id": winner,
            "paired_seed_mean_coalition_auc": mean_auc(winner),
            "eligible_rule_ids": eligible, "excluded_rule_ids": excluded,
            "ties_joint_mask_pair": all(
                plans_by_seed[seed]["rules"][winner]["pair_id"] ==
                plans_by_seed[seed]["rules"]["J"]["pair_id"] for seed in seeds),
        }
    return result
