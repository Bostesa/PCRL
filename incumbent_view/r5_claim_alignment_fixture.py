"""Data-free A/B/C protocol alignment check; run only in the global queue."""
import json

from .r5_attack_math import ARMS
from .r5_claims_protocol import all_claims


def _set(slots, candidate, control, view, benefit):
    first, second = sorted((candidate, control), key=ARMS.index)
    signed = benefit if control == first else -benefit
    slots[f"mean:recovery:{first}:{second}:{view}:auc"] = {
        "point": signed, "lower": signed, "upper": signed, "status": "BOUND_ONLY"}


def run():
    gates = {arm: {"all_two_seed_task_reference_point_caps_pass": True} for arm in ARMS}
    utility = {f"s{seed}:utility:{arm}:{task}:{ref}:{metric}":
               {"point": 0.0, "status": "PASS"}
               for seed in (0, 1) for arm in ARMS
               for task in ("income", "occupation") for ref in ("U0", "Ucal")
               for metric in ("accuracy", "nll", "brier")}
    slots = {}
    for candidate, control, benefit in [
        ("V_L", "P0", .03), ("V_J", "O_J", .001),
        *(("V_J", other, .03) for other in ("V_P", "V_D", "V_L", "V_S12", "V_S21"))]:
        _set(slots, candidate, control, "pair", benefit)
        for reference in dict.fromkeys(("P0", control)):
            for view in ("r1", "r2"):
                _set(slots, candidate, reference, view, 0.0)
    result = all_claims("V_L", gates, utility, slots)
    assert [result[k]["status"] for k in ("A", "B", "C")] == ["PASSED"] * 3
    # No V_L-versus-V_P pair slot exists. A still follows the user's specific
    # incumbent comparison with P0; strongest-control reporting is separate.
    assert "mean:recovery:V_P:V_L:pair:auc" not in slots
    reduced = {k: v for k, v in slots.items() if not (
        k.startswith("mean:recovery:P0:V_J:") and (":r1:" in k or ":r2:" in k))}
    missing = all_claims("V_L", gates, utility, reduced)
    assert missing["A"]["status"] == "PASSED"
    assert missing["B"]["status"] == "INCOMPLETE"
    assert missing["C"]["status"] == "INCOMPLETE"
    return {"schema": "IVR1-r5-claim-alignment-fixture-v1", "status": "PASS",
            "checks": ["A requires P0 only", "B and C retain P0 local guards"],
            "real_people_read": 0, "labels_read": 0}


def main():
    from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
    assert_active_from_env(stage="verification")
    print(json.dumps(run(), sort_keys=True))


if __name__ == "__main__":
    main()
