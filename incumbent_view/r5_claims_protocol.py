"""Independent A/B/C claim accounting against the fixed K948 family.

This follows the three comparisons in the user protocol. It imports no R3
claim, scoring, attack, or selection implementation.
"""
from __future__ import annotations

from .r5_attack_math import ARMS, oriented_bounds

TASKS = ("income", "occupation")
REFERENCES = ("U0", "Ucal")
METRICS = ("accuracy", "nll", "brier")
NONJOINT = ("V_P", "V_D", "V_L", "V_S12", "V_S21")


def _costs(arms: tuple[str, ...], utility_slots: dict) -> list[dict]:
    out = []
    for arm in dict.fromkeys(arms):
        for seed in (0, 1):
            for task in TASKS:
                for ref in REFERENCES:
                    for metric in METRICS:
                        key = f"s{seed}:utility:{arm}:{task}:{ref}:{metric}"
                        record = utility_slots.get(key)
                        out.append({"slot": key, "status": "INCOMPLETE" if record is None or
                                    record.get("point") is None else
                                    "PASS" if record.get("status") == "PASS" else "NOT_ESTABLISHED"})
    return out


def _comparison(candidate: str, control: str, recovery_slots: dict,
                threshold: float, strict: bool) -> list[dict]:
    if candidate == control:
        raise ValueError("candidate cannot compare with itself")
    result = []
    try:
        point, lower, upper = oriented_bounds(recovery_slots, candidate, control, "pair")
        passed = lower > threshold if strict else lower >= threshold
        result.append({"kind": "coalition_benefit", "candidate": candidate,
                       "control": control, "point": point, "lower": lower,
                       "upper": upper, "required": threshold, "strict": strict,
                       "status": "PASS" if passed else "NOT_ESTABLISHED"})
    except (KeyError, ValueError):
        result.append({"kind": "coalition_benefit", "candidate": candidate,
                       "control": control, "status": "INCOMPLETE"})
    for reference in dict.fromkeys(("P0", control)):
        if candidate == reference:
            continue
        for view in ("r1", "r2"):
            try:
                point, lower, upper = oriented_bounds(recovery_slots, candidate, reference, view)
                # oriented benefit = reference minus candidate; candidate local
                # worsening therefore has upper bound equal to -lower.
                result.append({"kind": "local_worsening", "candidate": candidate,
                               "control": reference, "view": view,
                               "point": -point, "lower": -upper, "upper": -lower,
                               "cap": 0.010,
                               "status": "PASS" if -lower <= 0.010 else "NOT_ESTABLISHED"})
            except (KeyError, ValueError):
                result.append({"kind": "local_worsening", "candidate": candidate,
                               "control": reference, "view": view, "status": "INCOMPLETE"})
    return result


def _claim(candidate: str | None, controls: tuple[str, ...], gates: dict,
           utility_slots: dict, recovery_slots: dict, threshold: float,
           *, strict: bool = False) -> dict:
    if candidate is None or candidate not in ARMS:
        return {"status": "UNTESTED", "reason": "no eligible nominated candidate", "components": []}
    required = tuple(dict.fromkeys((candidate, *controls)))
    missing = [arm for arm in required if gates.get(arm, {}).get(
        "all_two_seed_task_reference_point_caps_pass") is not True]
    if missing:
        return {"status": "UNTESTED", "reason": "required original INNER point gate absent",
                "arms": missing, "components": []}
    components = _costs(required, utility_slots)
    for control in controls:
        components.extend(_comparison(candidate, control, recovery_slots, threshold, strict))
    status = ("INCOMPLETE" if any(x["status"] == "INCOMPLETE" for x in components)
              else "PASSED" if all(x["status"] == "PASS" for x in components)
              else "NOT_ESTABLISHED")
    return {"status": status, "candidate": candidate, "controls": list(controls),
            "components": components}


def all_claims(nominee: str | None, gates: dict, utility_slots: dict,
               recovery_slots: dict, *, nominee_exact_p0_alias: bool = False) -> dict:
    """A: P0 margin; B: objective package; C: all eligible nonjoint arms."""
    a = ({"status": "UNTESTED", "reason": "nominee is exact P0 alias", "components": []}
         if nominee_exact_p0_alias else
         _claim(nominee, ("P0",), gates, utility_slots, recovery_slots, 0.020))
    b = _claim("V_J", ("O_J",), gates, utility_slots, recovery_slots, 0.0, strict=True)
    missing_sequential = [arm for arm in ("V_S12", "V_S21") if
                          gates.get(arm, {}).get("all_two_seed_task_reference_point_caps_pass") is not True]
    controls = tuple(arm for arm in NONJOINT if
                     gates.get(arm, {}).get("all_two_seed_task_reference_point_caps_pass") is True)
    c = ({"status": "UNTESTED", "reason": "required sequential control absent or point-ineligible",
          "arms": missing_sequential, "components": []}
         if missing_sequential else
         _claim("V_J", controls, gates, utility_slots, recovery_slots, 0.020))
    return {"schema": "IVR1-r5-protocol-claims-v1", "A": a, "B": b, "C": c,
            "scope": "selected on reused development data; conditional intervals are not confirmation"}
