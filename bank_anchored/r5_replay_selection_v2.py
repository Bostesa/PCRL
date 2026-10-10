"""Versioned public selection replay with explicit ineligible S21 null arm.

This consumes only completed public R4 and R5 receipts. All private attack
prediction and label arithmetic is checked by the separate R5 replay units.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "results/pcrl_bank_anchored_release_v1"
ARM_ORDER = ("B0", "T", "D", "P", "L", "J", "S12", "S21", "W_Q", "P_STAR")
ESTIMABLE_ARMS = tuple(arm for arm in ARM_ORDER if arm != "S21")
VIEWS = ("r1", "r2", "pair")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def equal_seed_mean(values: list[dict]) -> dict[str, dict]:
    by_arm = defaultdict(lambda: defaultdict(list))
    for row in values:
        if row["arm_id"] not in ARM_ORDER or row["source_seed"] not in (0, 1):
            raise ValueError("unregistered arm/seed in public attack family")
        by_arm[row["arm_id"]][row["source_seed"]].append(row)
    if set(by_arm) != set(ESTIMABLE_ARMS):
        raise ValueError("public attack family has missing arm")
    result = {}
    for arm in ESTIMABLE_ARMS:
        if set(by_arm[arm]) != {0, 1}:
            raise ValueError("paired source seeds are incomplete")
        seed = {}
        for s in (0, 1):
            expected_draws = {0} if arm in ("B0", "D", "W_Q", "P_STAR") else {0, 1, 2}
            if {item["realization"] for item in by_arm[arm][s]} != expected_draws:
                raise ValueError("arm has missing or selected cached draws")
            seed[s] = {view: sum(float(item["auc_by_view"][view])
                                 for item in by_arm[arm][s]) / len(by_arm[arm][s])
                       for view in VIEWS}
        result[arm] = {"s0": seed[0], "s1": seed[1],
                       "mean": {view: (seed[0][view] + seed[1][view]) / 2
                                for view in VIEWS}}
    result["S21"] = {"s0": None, "s1": None, "mean": None,
                     "status": "CONDITIONAL_NULL",
                     "reason": "S21_INNER_POINT_GATE_FAILED_NO_ATTACK_FIT"}
    return result


def replay(plan_path: Path, aggregate_path: Path, csv_path: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    aggregate = json.loads(aggregate_path.read_text())
    if (plan.get("schema") != "BAR1-r5-effective-attack-replay-plan-v1" or
            aggregate.get("schema") !=
            "pcrl-bank-r4-mixed-h1-h2-public-attack-aggregate-v1" or
            aggregate.get("unit_count") != 38 or aggregate.get("view_rows") != 114 or
            aggregate.get("attack_auc_csv_sha256") != sha(csv_path) or
            aggregate.get("all_control_gates_pass") is not True):
        raise PermissionError("effective public attack source family differs")
    units = plan["source_attack_unit_ids"]
    public = aggregate["units"]
    if (len(units) != 38 or len(set(units)) != 38 or
            {row["unit_id"] for row in public} != set(units) or
            len(public) != 38):
        raise PermissionError("38 effective attack units differ")
    by_uid = {row["unit_id"]: row for row in public}
    source_rows = {}
    for uid in units:
        row = by_uid[uid]
        receipt_path = CAMP / "r4" / f"{uid}_RECEIPT.json"
        r5_path = CAMP / "r5" / f"R5_REPLAY_{uid}_RECEIPT.json"
        if sha(receipt_path) != row["receipt_sha256"]:
            raise PermissionError("R4 source receipt changed after aggregate")
        r4 = json.loads(receipt_path.read_text())
        r5 = json.loads(r5_path.read_text())
        if (r5.get("status") != "PASS" or r5.get("source_unit_id") != uid or
                r5.get("r4_receipt_sha256") != row["receipt_sha256"] or
                r4.get("unit_id") != uid or r4.get("status") !=
                "COMPLETE_PRIVATE_DEVELOPMENT_ONLY" or
                (r4["arm_id"], r4["source_seed"], r4["realization"]) !=
                (row["arm_id"], row["source_seed"], row["realization"]) or
                r4["attack"]["freeze_receipt_sha256"] !=
                row["freeze_receipt_sha256"]):
            raise PermissionError("R4/R5 frozen attack receipt identity differs")
        for view in VIEWS:
            score = r4["attack"]["scores"][view]
            replayed = r5["checked_views"][view]
            value = float(score["primary_envelope_auc"])
            if (abs(value - float(row["auc_by_view"][view])) > 1e-12 or
                    abs(value - float(replayed["envelope_auc"])) > 1e-12 or
                    score["validation_winner"] != replayed["validation_winner"] or
                    len(score["readers"]) != replayed["reader_count"] or
                    row["control_gates_by_view"][view] !=
                    r4["attack"]["controls"][view]["scores"]["gates"] or
                    not all(row["control_gates_by_view"][view].values())):
                raise ValueError("independent AUC, reader or control gate differs")
        source_rows[uid] = r4
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 114 or len({(r["unit_id"], r["view"]) for r in rows}) != 114:
        raise ValueError("public attack CSV lacks complete 38-by-three views")
    for record in rows:
        uid, view = record["unit_id"], record["view"]
        if uid not in source_rows or view not in VIEWS:
            raise ValueError("public attack CSV contains unknown unit/view")
        source = source_rows[uid]
        score = source["attack"]["scores"][view]
        if (record["arm_id"] != source["arm_id"] or
                int(record["source_seed"]) != source["source_seed"] or
                int(record["realization"]) != source["realization"] or
                abs(float(record["primary_envelope_auc"]) -
                    float(score["primary_envelope_auc"])) > 1e-12 or
                record["validation_winner"] != score["validation_winner"] or
                int(record["reader_count"]) != len(score["readers"])):
            raise ValueError("public attack CSV differs from frozen source receipt")
    eligible = {}
    for arm in ARM_ORDER:
        gate_path = (CAMP / "r3" /
                     (f"{arm}_HISTORICAL_ORIGINAL_INNER_GATE.json" if
                      arm in ("W_Q", "P_STAR") else
                      f"{arm}_TWO_SEED_ATTACK_GATE_V2.json"))
        gate = json.loads(gate_path.read_text())
        if gate.get("arm_id") != arm:
            raise PermissionError("attack admission gate arm differs")
        eligible[arm] = gate.get("all_two_seed_task_reference_point_caps_pass") is True
    if eligible["S21"]:
        raise PermissionError("S21 conditional null contradicts frozen utility gate")
    means = equal_seed_mean(public)
    nominee = "J" if eligible["J"] else None
    controls = [arm for arm in ARM_ORDER if arm != "J" and eligible[arm]]
    comparator = min(controls, key=lambda arm: (means[arm]["mean"]["pair"],
                                                 ARM_ORDER.index(arm))) if controls else None
    benefit = (means[comparator]["mean"]["pair"] -
               means["J"]["mean"]["pair"] if nominee and comparator else None)
    local = ({view: means["J"]["mean"][view] -
             means[comparator]["mean"][view] for view in ("r1", "r2")}
             if nominee and comparator else None)
    return {"schema": "BAR1-r5-independent-public-selection-replay-v2",
            "status": "PASS", "source_release_units": len(units),
            "checked_view_rows": len(rows), "all_control_gates_pass": True,
            "conditional_null_arm": "S21",
            "eligible_by_arm": eligible, "equal_seed_means": means,
            "nominee": nominee, "selected_strongest_control": comparator,
            "selected_control_minus_j_pair_auc": benefit,
            "j_minus_control_local_auc": local,
            "scope": "exploratory development point recovery, selected from reused people"}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for arg in ("plan", "plan-sha256", "aggregate", "aggregate-sha256",
                "csv", "csv-sha256", "output"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
    active = assert_active_from_env(stage="verification")
    for path, digest in ((Path(args.plan), args.plan_sha256),
                         (Path(args.aggregate), args.aggregate_sha256),
                         (Path(args.csv), args.csv_sha256)):
        if sha(path) != digest:
            raise PermissionError("R5 public selection source hash changed")
    result = replay(Path(args.plan), Path(args.aggregate), Path(args.csv))
    result["unit_id"] = active["unit_id"]
    output = Path(args.output).resolve(strict=False)
    if (output.parent != CAMP / "r5" or output.exists()):
        raise PermissionError("R5 public selection receipt output differs")
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
