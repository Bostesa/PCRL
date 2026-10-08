"""Pure matched-control selection from M2's immutable fitting-role pair table.

This module performs no role access, model fitting, attack fitting, or INNER
scoring. All selectors receive the same cached M2 table.
"""

from __future__ import annotations

import hashlib
import json
from math import isfinite


LOCAL_AUC_ALLOWANCE = 0.010


def _legal_rows(table, t_pair, union_quota=None):
    """Apply the frozen utility, local-reader and optional union-size clauses."""
    income = {mask.mask_id: mask for mask in table.income_masks}
    occupation = {mask.mask_id: mask for mask in table.occupation_masks}
    return tuple(row for row in table.pairs if (
        row.pair_feasible
        and income[row.income_mask_id].utility_feasible
        and occupation[row.occupation_mask_id].utility_feasible
        and row.income_local_search_auc <= t_pair.income_local_search_auc + LOCAL_AUC_ALLOWANCE
        and row.occupation_local_search_auc <= t_pair.occupation_local_search_auc + LOCAL_AUC_ALLOWANCE
        and (union_quota is None or row.union_size_signature == union_quota)
    ))


def _tie(row):
    return row.task_loss, row.income_mask_sha256, row.occupation_mask_sha256


def _local_key(mask):
    return mask.local_search_auc, mask.task_validation_nll, mask.mask_sha256


def select_task_only(table):
    """Select T independently by each task's validation NLL and mask SHA.

    Search-reader AUC fields are deliberately absent from the key. The
    returned pair is an existing row in the shared bank.
    """
    income = tuple(mask for mask in table.income_masks if mask.utility_feasible)
    occupation = tuple(mask for mask in table.occupation_masks if mask.utility_feasible)
    if not income or not occupation:
        raise ValueError("no utility-feasible mask for task-only T")
    first = min(income, key=lambda mask: (
        mask.task_validation_nll, mask.mask_sha256,
    ))
    second = min(occupation, key=lambda mask: (
        mask.task_validation_nll, mask.mask_sha256,
    ))
    pair = table.pair(table.pair_id(first.mask_id, second.mask_id))
    if not pair.pair_feasible:
        raise ValueError("task-only pair is not utility-feasible")
    return pair


def select_independent(table, t_pair):
    """Choose each local-privacy mask independently, then combine the pair."""
    rows = _legal_rows(table, t_pair)
    if not rows:
        return None
    legal_income = {row.income_mask_id for row in rows}
    legal_occupation = {row.occupation_mask_id for row in rows}
    first = min((mask for mask in table.income_masks if mask.mask_id in legal_income),
                key=_local_key)
    second = min((mask for mask in table.occupation_masks if mask.mask_id in legal_occupation),
                 key=_local_key)
    pair = table.pair(table.pair_id(first.mask_id, second.mask_id))
    if pair not in rows:
        raise ValueError("shared table is not a complete legal Cartesian bank")
    return pair


def select_sequential(table, t_pair, *, first, union_quota=None):
    """Freeze a local first step, then choose a coalition-aware completion."""
    if first not in ("income", "occupation"):
        raise ValueError("first must be income or occupation")
    rows = _legal_rows(table, t_pair, union_quota)
    if not rows:
        return None
    if first == "income":
        legal = {row.income_mask_id for row in rows}
        chosen = min((mask for mask in table.income_masks if mask.mask_id in legal),
                     key=_local_key)
        completions = (row for row in rows if row.income_mask_id == chosen.mask_id)
    else:
        legal = {row.occupation_mask_id for row in rows}
        chosen = min((mask for mask in table.occupation_masks if mask.mask_id in legal),
                     key=_local_key)
        completions = (row for row in rows if row.occupation_mask_id == chosen.mask_id)
    return min(completions, key=lambda row: (row.coalition_search_auc, *_tie(row)))


def select_shared(table, t_pair, *, union_quota=None):
    rows = (row for row in _legal_rows(table, t_pair, union_quota)
            if row.income_mask_id == row.occupation_mask_id)
    return min(rows, key=lambda row: (row.coalition_search_auc, *_tie(row)), default=None)


def select_union_task_only(table, t_pair, union_quota):
    """Minimize task loss under J's union size and the frozen local guard."""
    return min(_legal_rows(table, t_pair, union_quota), key=_tie, default=None)


def select_union_independent(table, t_pair, union_quota):
    """Minimize summed local recovery under a pairwise union-size constraint."""
    return min(_legal_rows(table, t_pair, union_quota),
               key=lambda row: (row.income_local_search_auc + row.occupation_local_search_auc,
                                *_tie(row)), default=None)


RULE_ORDER = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")
MAIN_CONTROL_ORDER = ("T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")
ATTRIBUTION_CONTROL_ORDER = ("T_U", "I_U", "S12_U", "S21_U", "T", "I", "S12", "S21", "SHARED")


def _canonical_sha(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def build_control_plan(table, bank, j_pair_id):
    """Register J and every matched rule from one immutable fitting-role table.

    Same mask pair and same raw union are only equivalence candidates here.
    Complete release aliases require M4's head/calibration/wire identity check.
    """
    if table.bank_sha256 != bank.bank_sha256:
        raise ValueError("pair table and mask bank identities differ")
    if table.critic_input_kind != "RAW_UNION_ONLY":
        raise ValueError("pair critic input kind must be RAW_UNION_ONLY for union cache")
    t_pair = select_task_only(table)
    j_pair = table.pair(j_pair_id)
    if j_pair not in _legal_rows(table, t_pair):
        raise ValueError("J pair fails task eligibility or the T local guard")
    quota = j_pair.union_size_signature
    picked = {
        "J": j_pair,
        "T": t_pair,
        "I": select_independent(table, t_pair),
        "S12": select_sequential(table, t_pair, first="income"),
        "S21": select_sequential(table, t_pair, first="occupation"),
        "SHARED": select_shared(table, t_pair),
        "T_U": select_union_task_only(table, t_pair, quota),
        "I_U": select_union_independent(table, t_pair, quota),
        "S12_U": select_sequential(table, t_pair, first="income", union_quota=quota),
        "S21_U": select_sequential(table, t_pair, first="occupation", union_quota=quota),
    }
    income = {mask.mask_id: mask for mask in table.income_masks}
    occupation = {mask.mask_id: mask for mask in table.occupation_masks}
    rules = {}
    for rule_id in RULE_ORDER:
        row = picked[rule_id]
        if row is None:
            rules[rule_id] = {
                "status": "INVALID_NO_ELIGIBLE_PAIR",
                "reason": ("no diagonal pair meets utility and local guards" if rule_id == "SHARED"
                           else "no pair meets utility, local guards and union quota"),
                "pair_id": None,
            }
            continue
        first = bank.mask(row.income_mask_id)
        second = bank.mask(row.occupation_mask_id)
        score_first = income[row.income_mask_id]
        score_second = occupation[row.occupation_mask_id]
        if (getattr(first, "mask_sha256", row.income_mask_sha256) != row.income_mask_sha256
                or getattr(second, "mask_sha256", row.occupation_mask_sha256) !=
                row.occupation_mask_sha256):
            raise ValueError("pair table mask identity differs from bank")
        rules[rule_id] = {
            "status": "SELECTED", "pair_id": row.pair_id,
            "income_mask_id": row.income_mask_id,
            "occupation_mask_id": row.occupation_mask_id,
            "income_mask_sha256": row.income_mask_sha256,
            "occupation_mask_sha256": row.occupation_mask_sha256,
            "head_ids": {"income": score_first.head_id,
                         "occupation_group": score_second.head_id},
            "recipient_capacity_signatures": {
                "income": {"numeric_groups": first.capacity_signature[0],
                           "categorical_groups": first.capacity_signature[1],
                           "encoded_columns_max": first.capacity_signature[2],
                           "actual_source_groups": first.source_group_count,
                           "actual_encoded_columns": first.encoded_columns},
                "occupation_group": {"numeric_groups": second.capacity_signature[0],
                                     "categorical_groups": second.capacity_signature[1],
                                     "encoded_columns_max": second.capacity_signature[2],
                                     "actual_source_groups": second.source_group_count,
                                     "actual_encoded_columns": second.encoded_columns},
            },
            "recipient_encoded_columns": {"income": first.encoded_columns,
                                           "occupation_group": second.encoded_columns},
            "union_capacity_signature": {"source_groups": row.union_group_count,
                                         "encoded_columns": row.union_encoded_columns},
            "union_sha256": row.union_sha256,
            "matches_joint_mask_pair": row.pair_id == j_pair_id,
        }
    for rule_id, item in rules.items():
        if item["status"] != "SELECTED":
            continue
        item["same_mask_pair_rule_ids"] = [other for other in RULE_ORDER
                                           if other != rule_id and
                                           rules[other]["pair_id"] == item["pair_id"]]
        item["same_raw_union_rule_ids"] = [other for other in RULE_ORDER
                                           if other != rule_id and
                                           rules[other].get("union_sha256") == item["union_sha256"]]
    plan = {
        "schema": "pcrl-focused-m3-control-plan-v1",
        "bank_sha256": table.bank_sha256,
        "table_sha256": table.table_sha256,
        "joint_pair_id": j_pair_id,
        "task_only_pair_id": t_pair.pair_id,
        "joint_union_quota": {"source_groups": quota[0], "encoded_columns": quota[1]},
        "main_control_order": list(MAIN_CONTROL_ORDER),
        "attribution_control_order": list(ATTRIBUTION_CONTROL_ORDER),
        "rules": rules,
        "dropped_invalid_rules": [rule_id for rule_id in RULE_ORDER
                                  if rules[rule_id]["status"] != "SELECTED"],
        "release_alias_status": "PENDING_M4_COMPLETE_WIRE_AND_MODEL_IDENTITY",
    }
    plan["selection_receipt_sha256"] = _canonical_sha(plan)
    return plan


def _endpoint_utility_eligible(endpoint):
    task = endpoint.get("task")
    if not isinstance(task, dict) or set(task) != {"income", "occupation_group"}:
        raise ValueError("endpoint task references are incomplete")
    eligible = []
    for purpose in ("income", "occupation_group"):
        refs = task[purpose].get("references")
        if not isinstance(refs, dict) or set(refs) != {"U0", "Ucal", "F"}:
            raise ValueError("endpoint U0/Ucal/F references are incomplete")
        for reference in ("U0", "Ucal", "F"):
            value = refs[reference].get("point_eligible")
            if type(value) is not bool:
                raise ValueError("endpoint point utility eligibility must be boolean")
            eligible.append(value)
    return all(eligible)


def _recipient_profile(selection):
    return tuple(tuple(selection["recipient_capacity_signatures"][purpose][name]
                       for name in ("numeric_groups", "categorical_groups", "encoded_columns_max"))
                 for purpose in ("income", "occupation_group"))


def strongest_controls(plans_by_seed, endpoints_by_seed):
    """Select strongest final controls with frozen pools; never select masks.

    M4 supplies sealed per-rule aggregate final endpoints. Complete release
    aliases stay in the pool; M4 separately proves them from restored wire IDs.
    """
    seeds = tuple(sorted(plans_by_seed))
    if not seeds or set(endpoints_by_seed) != set(seeds):
        raise ValueError("paired seed plans and endpoints differ")
    for seed in seeds:
        plan = plans_by_seed[seed]
        if plan.get("schema") != "pcrl-focused-m3-control-plan-v1":
            raise ValueError("control plan schema differs from frozen contract")
        for rule_id, selection in plan["rules"].items():
            if selection["status"] != "SELECTED":
                continue
            if rule_id not in endpoints_by_seed[seed]:
                raise ValueError(f"missing endpoint for seed {seed} rule {rule_id}")
            endpoint = endpoints_by_seed[seed][rule_id]
            if endpoint.get("seed") != seed or endpoint.get("rule_id") != rule_id:
                raise ValueError("endpoint seed/rule identity differs from registry")
            if endpoint.get("pair_id") != selection["pair_id"]:
                raise ValueError("endpoint pair differs from fitting selection")
            if (endpoint.get("income_mask_id") != selection["income_mask_id"] or
                    endpoint.get("occupation_mask_id") != selection["occupation_mask_id"]):
                raise ValueError("endpoint mask identities differ from fitting selection")
            if endpoint.get("union_sha256") != selection["union_sha256"]:
                raise ValueError("endpoint raw union identity differs from fitting selection")
            if (endpoint.get("recipient_capacity_signatures") !=
                    selection["recipient_capacity_signatures"] or
                    endpoint.get("union_capacity_signature") !=
                    selection["union_capacity_signature"]):
                raise ValueError("endpoint capacity signatures differ from fitting selection")
            _endpoint_utility_eligible(endpoint)
            auc = endpoint.get("recovery", {}).get("coalition", {}).get("primary_envelope_auc")
            if type(auc) not in (float, int) or not isfinite(auc) or not 0.5 <= auc <= 1:
                raise ValueError("final coalition AUC missing, unoriented or invalid")
    pools = {
        "main": MAIN_CONTROL_ORDER,
        "attribution": ATTRIBUTION_CONTROL_ORDER,
    }
    result = {}
    for pool_name, order in pools.items():
        eligible = []
        excluded = {}
        for rule_id in order:
            reasons = []
            for seed in seeds:
                plan = plans_by_seed[seed]
                choice = plan["rules"][rule_id]
                joint = plan["rules"]["J"]
                if choice["status"] != "SELECTED":
                    reasons.append(f"seed {seed}: {choice['status']}")
                    continue
                if _recipient_profile(choice) != _recipient_profile(joint):
                    reasons.append(f"seed {seed}: recipient profile mismatch")
                if (pool_name == "attribution" and choice["union_capacity_signature"] !=
                        joint["union_capacity_signature"]):
                    reasons.append(f"seed {seed}: union size mismatch")
                if not _endpoint_utility_eligible(endpoints_by_seed[seed][rule_id]):
                    reasons.append(f"seed {seed}: point utility shortfall")
            if reasons:
                excluded[rule_id] = reasons
            else:
                eligible.append(rule_id)
        if not eligible:
            result[pool_name] = {
                "status": "INCOMPLETE_NO_ELIGIBLE_CONTROL",
                "rule_id": None, "eligible_rule_ids": [],
                "excluded_rule_reasons": excluded,
            }
            continue
        chosen = min(eligible, key=lambda rule_id: (
            sum(endpoints_by_seed[seed][rule_id]["recovery"]["coalition"]["primary_envelope_auc"]
                for seed in seeds) / len(seeds),
            order.index(rule_id),
        ))
        pair_ids = [plans_by_seed[seed]["rules"][chosen]["pair_id"] for seed in seeds]
        result[pool_name] = {
            "status": "SELECTED", "rule_id": chosen,
            "paired_seed_mean_coalition_auc": sum(
                endpoints_by_seed[seed][chosen]["recovery"]["coalition"]["primary_envelope_auc"]
                for seed in seeds) / len(seeds),
            "pair_ids": pair_ids,
            "ties_joint_mask_pair": all(
                pair_id == plans_by_seed[seed]["joint_pair_id"]
                for seed, pair_id in zip(seeds, pair_ids)),
            "eligible_rule_ids": eligible,
            "excluded_rule_reasons": excluded,
            "release_alias_status": "PENDING_M4_COMPLETE_WIRE_AND_MODEL_IDENTITY",
        }
    result["schema"] = "pcrl-focused-m3-strongest-controls-v1"
    result["paired_seeds"] = list(seeds)
    result["registry_sha256"] = _canonical_sha(result)
    return result
