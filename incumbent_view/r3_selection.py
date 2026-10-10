"""Point admission and FIT-reader nomination; no final attacker outcomes."""
import math
from .r3_family import ARMS, V_FAMILIES, TASKS, REFERENCES, METRICS

GATE_SCHEMA = "pcrl-incumbent-r3-original-inner-point-gate-v1"


def original_inner_gate(arm, clauses, source_receipts, fit_proxy=None):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    required = {(s, t, r, m) for s in (0, 1) for t in TASKS for r in REFERENCES for m in METRICS}
    observed = {}
    for row in clauses:
        key = (row["seed"], row["task"], row["reference"], row["metric"])
        if key not in required or key in observed or row["status"] not in ("PASS", "FAIL"):
            raise ValueError("original INNER clause inventory differs")
        observed[key] = row["status"]
    if set(observed) != required:
        raise ValueError("24 original clauses required")
    passed = sum(v == "PASS" for v in observed.values())
    return {"schema": GATE_SCHEMA, "arm_id": arm, "seeds": [0, 1],
            "all_two_seed_task_reference_point_caps_pass": passed == 24,
            "original_inner": {"required": 24, "passed": passed, "status": "PASS" if passed == 24 else "FAIL"},
            "fit_proxy": fit_proxy,
            "source_receipts": source_receipts,
            "gate_semantics": "only original INNER 24 clauses; FIT proxy remains separate",
            "scope": "development point attack eligibility, not simultaneous utility pass"}


def nominate_family(gates, reader_records):
    """Minimize equal-seed strongest retained pair-reader gain over prior.

    reader_records[(arm,seed)] supplies fitted training-prior CE evaluated on
    the three fixed group-disjoint folds aggregated over all FIT rows, all retained reader CE values, and an exact P0
    release-alias flag. The prior is included in the minimum even if omitted
    from the retained list; no final attack result enters this function.
    """
    slate = []
    for arm in V_FAMILIES:
        gate = gates.get(arm)
        if gate is None or gate.get("schema") != GATE_SCHEMA or gate.get("arm_id") != arm or gate.get("all_two_seed_task_reference_point_caps_pass") is not True:
            continue
        by_seed = []
        aliases = []
        for seed in (0, 1):
            record = reader_records.get((arm, seed))
            if record is None:
                break
            fold_hash = record.get("crossfit_manifest_sha256")
            if not isinstance(fold_hash, str) or len(fold_hash) != 64 or any(c not in "0123456789abcdef" for c in fold_hash):
                raise ValueError("fixed FIT crossfit manifest hash required")
            prior = record.get("constant_prior_ce")
            readers = record.get("pair_reader_losses")
            if not isinstance(readers, dict) or not readers or not isinstance(prior, (float, int)) or not math.isfinite(prior):
                raise ValueError("internal-reader record incomplete")
            if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in readers.values()):
                raise ValueError("reader CE must be finite")
            best = min(prior, *readers.values())
            if (not isinstance(record.get("pair_min_ce"), (float, int))
                    or not isinstance(record.get("reader_recovery_gap"), (float, int))
                    or abs(float(record["pair_min_ce"]) - best) > 1e-12
                    or abs(float(record["reader_recovery_gap"]) - (prior - best)) > 1e-12):
                raise ValueError("R2 internal reader summary differs from full retained CE slate")
            by_seed.append({"seed": seed, "prior_ce": prior, "strongest_reader_ce": best, "gain": prior - best, "crossfit_manifest_sha256": fold_hash})
            alias = record.get("exact_p0_alias")
            if type(alias) is not bool:
                raise ValueError("exact alias declaration required")
            aliases.append(alias)
        if len(by_seed) == 2:
            slate.append({"arm_id": arm, "seed_reader_gains": by_seed,
                          "equal_seed_mean_reader_gain": sum(v["gain"] for v in by_seed) / 2,
                          "both_seed_exact_p0_alias": all(aliases)})
    competitive = [v for v in slate if not v["both_seed_exact_p0_alias"]]
    selected = (min(competitive, key=lambda v: (v["equal_seed_mean_reader_gain"], V_FAMILIES.index(v["arm_id"])))
                if competitive else None)
    return {"schema": "pcrl-incumbent-r3-internal-nomination-v1",
            "nominee": None if selected is None else selected["arm_id"],
            "fallback": "P0" if selected is None else None,
            "status": "NOMINATED" if selected else "NO_NONIDENTITY_ELIGIBLE_NOMINEE",
            "rule": "minimum equal-seed CE_prior-minus-min-retained-pair-CE; fixed family tie order",
            "slate": slate, "final_attack_results_used": False}


def strongest_control(candidate, gates, pair_auc, pool=None):
    """Frozen final-AUC control rule, never candidate nomination."""
    if candidate not in ARMS:
        raise ValueError("unknown candidate")
    pool = tuple(a for a in ARMS if a != candidate) if pool is None else tuple(pool)
    if any(a not in ARMS or a == candidate for a in pool):
        raise ValueError("control pool differs")
    eligible = [a for a in pool if gates.get(a, {}).get("all_two_seed_task_reference_point_caps_pass") is True]
    missing = [a for a in eligible if a not in pair_auc]
    if missing:
        return {"status": "INCOMPLETE", "selected": None, "missing_eligible_controls": missing}
    if any(not math.isfinite(pair_auc[a]) for a in eligible):
        raise ValueError("control AUC invalid")
    selected = min(eligible, key=lambda a: (pair_auc[a], ARMS.index(a))) if eligible else None
    return {"status": "SELECTED" if selected else "UNTESTED", "selected": selected,
            "pool": list(pool), "eligible_pool": eligible}
