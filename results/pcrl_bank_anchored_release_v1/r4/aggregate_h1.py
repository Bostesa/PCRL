"""Aggregate all H1 public attack receipts without opening person-level data.

This records each frozen reader envelope and control gate. It performs no
arm selection, uncertainty calculation, or numerical replay of private fits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parent
VIEWS = ("r1", "r2", "pair")
SCHEMAS = {
    "B0": "pcrl-bank-r4-witness-fresh-attack-v1",
    "W_Q": "pcrl-bank-r4-witness-fresh-attack-v1",
    "P_STAR": "pcrl-bank-r4-pstar-fresh-attack-v1",
}
DEFAULT_SCHEMA = "pcrl-bank-r4-policy-fresh-attack-v1"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_pinned(path: Path, expected: str) -> dict:
    if path.is_symlink() or sha(path) != expected:
        raise PermissionError(f"public aggregate input differs: {path.name}")
    return json.loads(path.read_text())


def rows_for_receipt(receipt: Mapping, entry: Mapping,
                     stage_lock_sha: str, receipt_sha: str) -> tuple[list[dict], dict]:
    arm = entry["arm_id"]
    seed = entry["source_seed"]
    draw = entry["realization"]
    uid = entry["unit_id"]
    if (receipt.get("schema") != SCHEMAS.get(arm, DEFAULT_SCHEMA) or
            receipt.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_ONLY" or
            receipt.get("unit_id") != uid or
            receipt.get("arm_id") != arm or
            receipt.get("source_seed") != seed or
            receipt.get("realization") != draw or
            receipt.get("attacker_seed") != 100000 + 1000 * seed + draw or
            receipt.get("unit_spec_sha256") != entry["spec_sha256"] or
            receipt.get("stage_lock_sha256") != stage_lock_sha):
        raise PermissionError(f"H1 attack receipt identity differs: {uid}")
    attack = receipt.get("attack", {})
    if (attack.get("schema") != "pcrl-next-frozen-attack-unit-v1" or
            set(attack.get("scores", {})) != set(VIEWS) or
            set(attack.get("controls", {})) != set(VIEWS) or
            len(attack.get("freeze_receipt_sha256", "")) != 64):
        raise PermissionError(f"H1 attack view or freeze receipt missing: {uid}")
    rows = []
    for view in VIEWS:
        score = attack["scores"][view]
        gates = attack["controls"][view]["scores"]["gates"]
        auc = score.get("primary_envelope_auc")
        if (not isinstance(auc, (int, float)) or not math.isfinite(auc) or
                not 0 <= auc <= 1 or not isinstance(score.get("readers"), dict) or
                not score["readers"] or not isinstance(gates, dict) or
                not gates or any(type(x) is not bool for x in gates.values())):
            raise PermissionError(f"H1 AUC/readers/control gates differ: {uid}/{view}")
        rows.append({
            "unit_id": uid, "arm_id": arm, "source_seed": seed,
            "realization": draw, "view": view,
            "primary_envelope_auc": auc,
            "validation_winner": score.get("validation_winner"),
            "validation_winner_inner_auc": score.get("validation_winner_inner_auc"),
            "reader_count": len(score["readers"]),
            "all_control_gates_pass": all(gates.values()),
            "control_gates_json": json.dumps(gates, sort_keys=True),
            "receipt_sha256": receipt_sha,
        })
    unit = {"unit_id": uid, "arm_id": arm, "source_seed": seed,
            "realization": draw, "attacker_seed": receipt["attacker_seed"],
            "receipt_sha256": receipt_sha,
            "freeze_receipt_sha256": attack["freeze_receipt_sha256"],
            "auc_by_view": {row["view"]: row["primary_envelope_auc"] for row in rows},
            "control_gates_by_view": {
                view: attack["controls"][view]["scores"]["gates"] for view in VIEWS},
            "reader_names_by_view": {
                view: sorted(attack["scores"][view]["readers"]) for view in VIEWS}}
    return rows, unit


def _write_once(path: Path, blob: bytes) -> str:
    if path.exists():
        if path.read_bytes() != blob:
            raise FileExistsError("immutable H1 aggregate output differs")
        return sha(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(blob).hexdigest()


def aggregate(index_sha: str, queue_index_sha: str, lock_sha: str) -> dict:
    index_path = ROOT / "ATTACK_UNIT_INDEX_H1.json"
    queue_path = ROOT / "ATTACK_QUEUE_INDEX_H1.json"
    lock_path = ROOT.parent / "r0" / "ATTACK_STAGE_LOCK_H1.json"
    index = read_pinned(index_path, index_sha)
    queue = read_pinned(queue_path, queue_index_sha)
    lock = read_pinned(lock_path, lock_sha)
    if (index.get("schema") != "pcrl-bank-r4-attack-unit-index-v3-h1" or
            queue.get("schema") != "pcrl-bank-r4-queue-unit-index-v3-h1" or
            lock.get("schema") != "pcrl-bank-attack-stage-lock-v1" or
            queue.get("attack_unit_index_sha256") != index_sha or
            queue.get("stage_lock_sha256") != lock_sha or
            queue.get("amendment_sha256") != lock_sha or
            lock.get("units") != index.get("stage_lock_units") or
            len(index.get("units", [])) != 38 or
            len(queue.get("units", [])) != 38):
        raise PermissionError("H1 aggregate index/lock binding differs")
    queue_units = {row["unit_id"]: row for row in queue["units"]}
    if len(queue_units) != 38:
        raise PermissionError("H1 queue unit identities repeat")
    rows, units = [], []
    for entry in index["units"]:
        uid = entry["unit_id"]
        q = queue_units.get(uid)
        if q is None or q["attack_spec_sha256"] != entry["spec_sha256"]:
            raise PermissionError("H1 attack/queue spec linkage differs")
        receipt_path = ROOT / f"{uid}_RECEIPT.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"H1 attack unit still lacks public receipt: {uid}")
        receipt_sha = sha(receipt_path)
        unit_rows, unit = rows_for_receipt(
            read_pinned(receipt_path, receipt_sha), entry, lock_sha, receipt_sha)
        rows.extend(unit_rows)
        units.append(unit)
    if len(rows) != 114 or len({unit["unit_id"] for unit in units}) != 38:
        raise AssertionError("H1 aggregate does not contain 38 three-view releases")
    fieldnames = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_sha = _write_once(ROOT / "ATTACK_AUC_H1.csv", stream.getvalue().encode())
    summary = {
        "schema": "pcrl-bank-r4-h1-public-attack-aggregate-v1",
        "status": "COMPLETE_DEVELOPMENT_RECEIPTS_ONLY",
        "attack_unit_index_sha256": index_sha,
        "attack_queue_index_sha256": queue_index_sha,
        "stage_lock_sha256": lock_sha,
        "attack_auc_csv_sha256": csv_sha,
        "unit_count": 38, "view_rows": 114,
        "all_control_gates_pass": all(row["all_control_gates_pass"] for row in rows),
        "units": units,
        "scope": "Frozen public R4 receipt aggregates on reused INNER_SELECTION; no selection or independent confirmation.",
    }
    summary_blob = (json.dumps(summary, indent=2, sort_keys=True,
                               allow_nan=False) + "\n").encode()
    summary_sha = _write_once(ROOT / "ATTACK_AGGREGATE_H1.json", summary_blob)
    return {"status": summary["status"], "unit_count": 38,
            "view_rows": 114, "csv_sha256": csv_sha,
            "aggregate_sha256": summary_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--attack-unit-index-sha256", required=True)
    parser.add_argument("--attack-queue-index-sha256", required=True)
    parser.add_argument("--stage-lock-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.attack_unit_index_sha256,
                               args.attack_queue_index_sha256,
                               args.stage_lock_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
