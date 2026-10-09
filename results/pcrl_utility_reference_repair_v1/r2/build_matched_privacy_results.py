"""Build the compact rule-by-seed matched privacy table from replayed aggregates."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
from math import isfinite
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
PRIVATE = STUDY / ".private"
REPO = STUDY.parents[1]
RESULT = STUDY / "MATCHED_PRIVACY_RESULTS.csv"
RECEIPT = STUDY / "MATCHED_PRIVACY_RESULTS_RECEIPT.json"
VIEWS = ("income", "occupation_group", "coalition")
TASKS = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal", "F")
CAPS = {"accuracy_loss": .010, "nll_excess": .010,
        "brier_excess": .005}
MASK_UTILITY_COMMIT = "d4deaa86e0332f8278e777a94e524ada2c83b0f6"
FIELDS = (
    "seed", "rule_id", "rule_status", "pair_id", "income_mask_id",
    "occupation_mask_id", "same_pair_as_J", "same_pair_as_T",
    "union_sha256", "income_point_U0", "income_point_Ucal",
    "income_point_F", "occupation_point_U0", "occupation_point_Ucal",
    "occupation_point_F", "point_utility_all_refs", "view",
    "primary_envelope_auc", "validation_winner", "validation_winner_auc",
    "main_control_rule_id", "main_control_view_auc",
    "main_control_minus_rule_auc", "joint_view_auc",
    "joint_minus_rule_auc", "reader_count", "controls_status",
    "development_status",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    if path.suffix != ".json":
        raise PermissionError("matched table reads JSON only")
    return json.loads(path.read_text())


def _write_new(path: Path, payload: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _utility_map(path: Path) -> tuple[dict, dict]:
    bank_path = REPO / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    gate_path = STUDY / "FULL_INPUT_GATE.json"
    lead_path = STUDY / "r3/LEAD_MASK_TASK_NUMERICAL_VERIFICATION.json"
    bank_ver_path = STUDY / "r3/BANK_MASK_TASK_NUMERICAL_VERIFICATION.json"
    bank = read_json(bank_path)
    mask_ids = {item["mask_id"] for item in bank["masks"]}
    if len(mask_ids) != 5:
        raise PermissionError("fixed mask bank identity differs")
    expected_gate = sha(gate_path)
    verified = {}
    for name, source, status, schema in (
        ("LEAD", lead_path, "PASS_POINT_GATE_REPLAY",
         "pcrl-utility-r3-lead-mask-task-numerical-verification-v1"),
        ("BANK", bank_ver_path, "PASS_NUMERICAL_REPLAY",
         "pcrl-utility-r3-bank-mask-task-numerical-verification-v1"),
    ):
        record = read_json(source)
        if (record.get("schema") != schema or
                record.get("status") != status or
                record.get("full_input_gate_sha256") != expected_gate):
            raise PermissionError(f"{name} mask replay identity differs")
        verified[name] = record["mask_gate_receipt_sha256_by_seed_mask"]
    expected_receipts = {}
    for seed in (0, 1):
        lead = verified["LEAD"].get(str(seed), {})
        rest = verified["BANK"].get(str(seed), {})
        if set(lead) & set(rest) or set(lead) | set(rest) != mask_ids:
            raise PermissionError("independent five-mask coverage differs")
        expected_receipts.update({(seed, key): value for key, value in
                                  (lead | rest).items()})
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 60:
        raise PermissionError("fixed five-mask utility inventory differs")
    result = {}
    for row in rows:
        key = (int(row["seed"]), row["mask_id"], row["task"],
               row["reference"])
        if (key in result or row["point_eligible"] not in ("True", "False")
                or row["gate_receipt_sha256"] != expected_receipts.get(key[:2])
                or row["full_input_gate_sha256"] != expected_gate):
            raise PermissionError("mask utility identity duplicates or differs")
        deltas = {name: float(row[name]) for name in CAPS}
        if (not all(isfinite(value) for value in deltas.values())
                or (row["point_eligible"] == "True") !=
                all(deltas[name] <= cap + 1e-12
                    for name, cap in CAPS.items())):
            raise PermissionError("mask utility point arithmetic differs")
        result[key] = row["point_eligible"] == "True"
    expected_keys = {(seed, mask_id, task, ref)
                     for seed in (0, 1) for mask_id in mask_ids
                     for task in TASKS for ref in REFERENCES}
    if set(result) != expected_keys:
        raise PermissionError("exact five-mask utility key coverage differs")
    sources = {"mask_bank": sha(bank_path), "full_input_gate": expected_gate,
               "r3_lead_mask_verification": sha(lead_path),
               "r3_bank_mask_verification": sha(bank_ver_path)}
    return result, sources


def _reader_map(path: Path) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 120:
        raise PermissionError("complete frozen reader inventory differs")
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["controls_status"] != "PASS":
            raise PermissionError("a reader control failed")
        key = (int(row["seed"]), row["pair_id"], row["view"])
        grouped.setdefault(key, []).append(row)
    if len(grouped) != 12:
        raise PermissionError("selected pair/view inventory differs")
    for (_, _, view), block in grouped.items():
        if (len(block) != (18 if view == "coalition" else 6)
                or len({row["reader_id"] for row in block}) != len(block)
                or len({row["primary_envelope_auc"] for row in block}) != 1
                or len({row["validation_winner"] for row in block}) != 1
                or len({row["validation_winner_inner_auc"] for row in block}) != 1
                or abs(max(float(row["fixed_alias_max_auc"]) for row in block) -
                       float(block[0]["primary_envelope_auc"])) > 1e-12):
            raise PermissionError("reader envelope or view custody differs")
    return grouped


def build() -> tuple[list[dict], dict]:
    selection_path = STUDY / "MATCHED_PRIVACY_CONTROL_SELECTION.json"
    reader_path = STUDY / "PRIVACY_READERS.csv"
    utility_path = STUDY / "MASK_UTILITY.csv"
    independent_path = STUDY / "r3/MATCHED_CONTROL_NUMERICAL_VERIFICATION.json"
    source_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--",
         str(utility_path.relative_to(REPO))],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if source_commit != MASK_UTILITY_COMMIT:
        raise PermissionError("published mask utility source commit differs")
    selection = read_json(selection_path)
    independent = read_json(independent_path)
    if (selection.get("schema") !=
            "pcrl-utility-matched-privacy-control-selection-v1"
            or selection.get("reader_csv_sha256") != sha(reader_path)
            or independent.get("schema") !=
            "pcrl-utility-r3-matched-control-numerical-verification-v1"
            or independent.get("status") !=
            "PASS_INDEPENDENT_AGGREGATE_CONTROL_REPLAY"
            or independent["source_sha256"].get(
                "MATCHED_PRIVACY_CONTROL_SELECTION.json") !=
            sha(selection_path)
            or independent["source_sha256"].get("PRIVACY_READERS.csv") !=
            sha(reader_path)):
        raise PermissionError("independent matched-control replay differs")
    main = selection["selection"]["main"]
    if main["status"] != "SELECTED":
        raise PermissionError("no strongest eligible main control")
    readers = _reader_map(reader_path)
    utility, utility_sources = _utility_map(utility_path)
    plans = {}
    source_hashes = {
        "matched_control_selection": sha(selection_path),
        "independent_control_verification": sha(independent_path),
        "privacy_reader_table": sha(reader_path),
        "mask_utility_table": sha(utility_path),
        "mask_utility_builder_source": sha(STUDY / "r2/build_mask_utility.py"),
        "mask_utility_publisher_commit": source_commit,
    }
    source_hashes.update(utility_sources)
    for seed in (0, 1):
        plan_path = PRIVATE / "pair_select" / f"seed{seed}" / "CONTROL_PLAN.json"
        plans[seed] = read_json(plan_path)
        source_hashes[f"seed{seed}_plan"] = sha(plan_path)
        if (plans[seed].get("schema") != "pcrl-focused-m3-control-plan-v1"
                or independent["source_sha256"].get(
                    f"CONTROL_PLAN.seed{seed}") != sha(plan_path)):
            raise PermissionError("pair selection plan differs")
    output = []
    for seed in (0, 1):
        plan = plans[seed]
        control = plan["rules"][main["rule_id"]]
        if control["status"] != "SELECTED":
            raise PermissionError("frozen main control absent in seed")
        for rule_id, rule in sorted(plan["rules"].items()):
            selected = rule["status"] == "SELECTED"
            eligible = {}
            if selected:
                for purpose in TASKS:
                    mask = rule[("income_mask_id" if purpose == "income"
                                 else "occupation_mask_id")]
                    for ref in REFERENCES:
                        eligible[(purpose, ref)] = utility[
                            (seed, mask, purpose, ref)]
            for view in VIEWS:
                block = readers[(seed, rule["pair_id"], view)] if selected else []
                control_block = readers[(seed, control["pair_id"], view)]
                joint_block = readers[(seed, plan["joint_pair_id"], view)]
                auc = float(block[0]["primary_envelope_auc"]) if block else None
                control_auc = float(control_block[0]["primary_envelope_auc"])
                joint_auc = float(joint_block[0]["primary_envelope_auc"])
                output.append({
                    "seed": seed, "rule_id": rule_id,
                    "rule_status": rule["status"],
                    "pair_id": rule.get("pair_id") or "",
                    "income_mask_id": rule.get("income_mask_id", ""),
                    "occupation_mask_id": rule.get("occupation_mask_id", ""),
                    "same_pair_as_J": (rule.get("pair_id") ==
                                       plan["joint_pair_id"]) if selected else "",
                    "same_pair_as_T": (rule.get("pair_id") ==
                                       plan["task_only_pair_id"]) if selected else "",
                    "union_sha256": rule.get("union_sha256", ""),
                    "income_point_U0": eligible.get(("income", "U0"), ""),
                    "income_point_Ucal": eligible.get(("income", "Ucal"), ""),
                    "income_point_F": eligible.get(("income", "F"), ""),
                    "occupation_point_U0": eligible.get(
                        ("occupation_group", "U0"), ""),
                    "occupation_point_Ucal": eligible.get(
                        ("occupation_group", "Ucal"), ""),
                    "occupation_point_F": eligible.get(
                        ("occupation_group", "F"), ""),
                    "point_utility_all_refs": all(eligible.values())
                    if selected else "", "view": view,
                    "primary_envelope_auc": auc if selected else "",
                    "validation_winner": block[0]["validation_winner"]
                    if selected else "",
                    "validation_winner_auc": block[0][
                        "validation_winner_inner_auc"] if selected else "",
                    "main_control_rule_id": main["rule_id"],
                    "main_control_view_auc": control_auc,
                    "main_control_minus_rule_auc": control_auc - auc
                    if selected else "",
                    "joint_view_auc": joint_auc,
                    "joint_minus_rule_auc": joint_auc - auc
                    if selected else "",
                    "reader_count": len(block) if selected else 0,
                    "controls_status": block[0]["controls_status"]
                    if selected else "NOT_RUN_INVALID_RULE",
                    "development_status": "exploratory_reused_Adult_people",
                })
    if len(output) != 60:
        raise PermissionError("ten rules x two seeds x three views expected")
    receipt = {
        "schema": "pcrl-utility-matched-privacy-results-receipt-v1",
        "status": "DERIVED_FROM_INDEPENDENT_REPLAY",
        "rows": len(output),
        "main_strongest_control_rule_id": main["rule_id"],
        "joint_paired_seed_mean_coalition_auc": independent[
            "joint_paired_seed_mean_coalition_auc"],
        "main_control_paired_seed_mean_coalition_auc": independent[
            "strongest_main_control_paired_seed_mean_coalition_auc"],
        "control_minus_joint_coalition_auc": independent[
            "coalition_auc_benefit_control_minus_joint"],
        "input_sha256": source_hashes,
        "development_only": True,
    }
    return output, receipt


def main() -> None:
    rows, receipt = build()
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode()
    receipt["csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    receipt_bytes = (json.dumps(receipt, sort_keys=True, indent=2,
                                allow_nan=False) + "\n").encode()
    if RESULT.exists() or RECEIPT.exists():
        raise FileExistsError("matched result output already exists")
    _write_new(RESULT, csv_bytes)
    _write_new(RECEIPT, receipt_bytes)
    print(json.dumps({"rows": len(rows), "csv_sha256": sha(RESULT),
                      "receipt_sha256": sha(RECEIPT)}, sort_keys=True))


if __name__ == "__main__":
    main()
