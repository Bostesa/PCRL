"""Independent, frozen-order nomination and comparison logic for R5 replay."""
from __future__ import annotations

import math

from .r5_attack_math import ARMS, VIEWS, oriented_bounds


V_ARMS = ("V_J", "V_P", "V_D", "V_L", "V_S12", "V_S21")
NON_JOINT = ("V_P", "V_D", "V_L", "V_S12", "V_S21")


def fit_nominee(gates: dict[str, dict | None],
                fit_records: dict[tuple[str, int], dict]) -> dict:
    """Use only FIT reader losses and original INNER point gates."""
    slate = []
    for arm in V_ARMS:
        gate = gates.get(arm)
        if gate is None or gate.get("all_two_seed_task_reference_point_caps_pass") is not True:
            continue
        gains = []
        alias = []
        for seed in (0, 1):
            record = fit_records.get((arm, seed))
            if record is None:
                break
            prior = record["constant_prior_ce"]
            losses = record["pair_reader_losses"]
            if not isinstance(losses, dict) or not losses or not math.isfinite(prior):
                raise ValueError("FIT retained pair-reader slate incomplete")
            if any(not math.isfinite(x) for x in losses.values()):
                raise ValueError("FIT reader CE is nonfinite")
            best = min([prior, *losses.values()])
            if (abs(best - record["pair_min_ce"]) > 1e-12 or
                    abs((prior - best) - record["reader_recovery_gap"]) > 1e-12 or
                    len(record["crossfit_manifest_sha256"]) != 64 or
                    type(record["exact_p0_alias"]) is not bool):
                raise ValueError("FIT retained-reader inventory/summary differs")
            gains.append(prior - best)
            alias.append(record["exact_p0_alias"])
        if len(gains) == 2:
            slate.append({"arm_id": arm, "seed_gain": gains,
                          "equal_seed_mean_gap": sum(gains) / 2,
                          "both_seed_exact_p0_alias": all(alias)})
    nonidentity = [record for record in slate if not record["both_seed_exact_p0_alias"]]
    winner = min(nonidentity, key=lambda x: (x["equal_seed_mean_gap"],
                                             V_ARMS.index(x["arm_id"]))) if nonidentity else None
    return {"nominee": None if winner is None else winner["arm_id"],
            "fallback": "P0" if winner is None else None,
            "slate": slate, "rule": "minimum equal-seed FIT CE-prior gap in fixed V-arm order"}


def strongest_control(candidate: str, gates: dict[str, dict | None],
                      attacked_arms: set[str], mean_pair_auc: dict[str, float]) -> dict:
    if candidate not in ARMS:
        raise ValueError("candidate differs from fixed family")
    eligible = [a for a in ARMS if a != candidate and
                gates.get(a) is not None and
                gates[a].get("all_two_seed_task_reference_point_caps_pass") is True]
    missing = [a for a in eligible if a not in attacked_arms or a not in mean_pair_auc]
    if missing:
        return {"status": "INCOMPLETE", "selected": None,
                "missing_eligible_controls": missing}
    if not eligible:
        return {"status": "UNTESTED", "selected": None, "eligible": []}
    if any(not math.isfinite(mean_pair_auc[a]) for a in eligible):
        raise ValueError("nonfinite final pair AUC")
    best = min(eligible, key=lambda a: (mean_pair_auc[a], ARMS.index(a)))
    return {"status": "SELECTED", "selected": best, "eligible": eligible,
            "point_auc": mean_pair_auc[best]}


def minimum_benefit(slots: dict, candidate: str, control: str, required: float,
                    *, view: str = "pair", strict: bool = False) -> dict:
    point, lower, upper = oriented_bounds(slots, candidate, control, view)
    if strict:
        verdict = "PASS" if lower > required else ("VIOLATION" if upper <= required else "UNRESOLVED")
    else:
        verdict = "PASS" if lower >= required else ("VIOLATION" if upper < required else "UNRESOLVED")
    return {"candidate": candidate, "control": control, "view": view,
            "benefit_control_minus_candidate": point, "lower": lower,
            "upper": upper, "required": required, "status": verdict}


def maximum_local_worsening(slots: dict, candidate: str, control: str,
                            *, view: str) -> dict:
    if view not in VIEWS[:2]:
        raise ValueError("local reader view must be one recipient")
    point, lower, upper = oriented_bounds(slots, candidate, control, view)
    # Oriented value is control AUC minus candidate AUC, so the upper bound
    # on candidate worsening is the negative of its lower bound.
    worsen_lower, worsen_upper = -upper, -lower
    verdict = "PASS" if worsen_upper <= 0.010 else (
        "VIOLATION" if worsen_lower > 0.010 else "UNRESOLVED")
    return {"candidate": candidate, "control": control, "view": view,
            "worsening_candidate_minus_control": -point,
            "lower": worsen_lower, "upper": worsen_upper,
            "cap": 0.010, "status": verdict}


def all_required_status(records: list[dict], missing: list[str] | None = None) -> str:
    if missing:
        return "INCOMPLETE_DEPENDENCY"
    states = {r["status"] for r in records}
    if "VIOLATION" in states:
        return "VIOLATION"
    if "UNRESOLVED" in states:
        return "UNRESOLVED"
    if states == {"PASS"}:
        return "PASS"
    return "INCOMPLETE_DEPENDENCY"


def claim_recovery_clauses(nominee: str | None, gates: dict[str, dict | None],
                           attacked_arms: set[str], mean_pair_auc: dict[str, float],
                           slots: dict) -> dict:
    """Recovery clauses only; utility conjunction belongs to utility replay."""
    def comparisons(candidate, controls, threshold, *, strict=False):
        entries, missing = [], []
        for control in controls:
            if control not in attacked_arms:
                missing.append(control)
                continue
            entries.append(minimum_benefit(slots, candidate, control, threshold, strict=strict))
            entries.extend(maximum_local_worsening(slots, candidate, control, view=view)
                           for view in VIEWS[:2])
        return {"status": all_required_status(entries, missing),
                "candidate": candidate,
                "required_controls": list(controls), "missing": missing,
                "clauses": entries}

    if nominee is None or nominee not in attacked_arms:
        strongest = {"status": "INCOMPLETE", "selected": None}
        claim_a = {"status": "INCOMPLETE_DEPENDENCY", "nominee": nominee}
    else:
        strongest = strongest_control(nominee, gates, attacked_arms, mean_pair_auc)
        a_controls = ["P0"]
        if strongest["status"] == "SELECTED" and strongest["selected"] not in a_controls:
            a_controls.append(strongest["selected"])
        claim_a = comparisons(nominee, a_controls, 0.020)
        if strongest["status"] != "SELECTED":
            claim_a["status"] = "INCOMPLETE_DEPENDENCY"
    # B applies specifically to the new joint objective, independent of FIT
    # nominee.  It is an explicit objective-package diagnostic.
    claim_b = comparisons("V_J", ["O_J"], 0.0, strict=True) if "V_J" in attacked_arms else {
        "status": "INCOMPLETE_DEPENDENCY", "required_controls": ["O_J"]}
    required_c = [a for a in NON_JOINT if gates.get(a) is not None and
                  gates[a].get("all_two_seed_task_reference_point_caps_pass") is True]
    claim_c = comparisons("V_J", required_c, 0.020) if "V_J" in attacked_arms else {
        "status": "INCOMPLETE_DEPENDENCY", "required_controls": required_c}
    if any(a not in attacked_arms for a in ("V_S12", "V_S21")):
        claim_c["status"] = "INCOMPLETE_DEPENDENCY"
        claim_c["missing_required_sequential"] = [a for a in ("V_S12", "V_S21")
                                                  if a not in attacked_arms]
    return {"A": claim_a, "B": claim_b, "C": claim_c,
            "strongest_control": strongest,
            "utility_conjunction_pending": True}
