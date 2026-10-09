"""Publish every fixed-bank task-only gate aggregate; opens JSON only."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
BANK = REPO / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
PRIVATE = STUDY / ".private"
OUTPUT = STUDY / "MASK_UTILITY.csv"
CAPS = {"accuracy_loss": .010, "nll_excess": .010, "brier_excess": .005}
FIELDS = (
    "seed", "mask_id", "encoded_columns", "task", "reference", "rows",
    "mask_accuracy", "reference_accuracy", "accuracy_loss",
    "mask_nll", "reference_nll", "nll_excess",
    "mask_brier", "reference_brier", "brier_excess", "point_eligible",
    "class_counts_json", "mask_per_class_recall_json",
    "reference_per_class_recall_json", "gate_receipt_sha256",
    "full_input_gate_sha256", "development_status",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows() -> list[dict]:
    bank = json.loads(BANK.read_text())
    masks = {item["mask_id"]: item for item in bank["masks"]}
    if (sha(BANK) !=
            "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c"
            or len(masks) != 5):
        raise PermissionError("fixed five-mask bank identity differs")
    gate_sha = sha(STUDY / "FULL_INPUT_GATE.json")
    verification = {}
    expected_status = {"LEAD": "PASS_POINT_GATE_REPLAY",
                       "BANK": "PASS_NUMERICAL_REPLAY"}
    for scope in ("LEAD", "BANK"):
        path = STUDY / "r3" / f"{scope}_MASK_TASK_NUMERICAL_VERIFICATION.json"
        record = json.loads(path.read_text())
        if (record.get("status") != expected_status[scope]
                or record.get("full_input_gate_sha256") != gate_sha):
            raise PermissionError(f"{scope} independent task replay differs")
        verification[scope] = record["mask_gate_receipt_sha256_by_seed_mask"]
    expected_leads = {"M_DROP_NATIVE_COUNTRY", "M_DROP_MARITAL_STATUS"}
    for seed in ("0", "1"):
        if (set(verification["LEAD"].get(seed, {})) != expected_leads
                or set(verification["BANK"].get(seed, {})) !=
                set(masks) - expected_leads):
            raise PermissionError("independent five-mask replay coverage differs")
    rows = []
    for seed in (0, 1):
        for mask_id in sorted(masks):
            receipt_path = (PRIVATE / "mask_gate" / mask_id / f"seed{seed}" /
                            "MASK_GATE_RECEIPT.json")
            receipt_sha = sha(receipt_path)
            source = "LEAD" if mask_id in verification["LEAD"][str(seed)] else "BANK"
            if verification[source][str(seed)].get(mask_id) != receipt_sha:
                raise PermissionError("mask gate differs from independent replay")
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("schema") != "pcrl-utility-masked-common-task-gate-v1"
                    or receipt.get("mask_id") != mask_id
                    or receipt.get("seed") != seed
                    or receipt.get("full_input_gate_sha256") != gate_sha
                    or receipt.get("selected_recipe") != "A"
                    or receipt.get("common_role") != "INNER_SELECTION"):
                raise PermissionError("mask gate aggregate identity differs")
            for task in ("income", "occupation_group"):
                candidate = receipt["metrics"]["mask"][task]
                for reference in ("U0", "Ucal", "F"):
                    ref = receipt["metrics"][reference][task]
                    point = receipt["point_eligibility"][task]["references"][reference]
                    delta = {
                        "accuracy_loss": ref["accuracy"] - candidate["accuracy"],
                        "nll_excess": candidate["nll"] - ref["nll"],
                        "brier_excess": candidate["brier"] - ref["brier"],
                    }
                    eligible = all(delta[key] <= cap + 1e-12
                                   for key, cap in CAPS.items())
                    if (candidate["rows"] != 2235 or ref["rows"] != 2235
                            or candidate["class_counts"] != ref["class_counts"]
                            or point["point_eligible"] is not eligible
                            or any(abs(delta[key] - point[key]) > 1e-12
                                   for key in CAPS)):
                        raise PermissionError("task gate point arithmetic differs")
                    rows.append({
                        "seed": seed, "mask_id": mask_id,
                        "encoded_columns": masks[mask_id]["encoded_columns"],
                        "task": task, "reference": reference,
                        "rows": 2235,
                        "mask_accuracy": candidate["accuracy"],
                        "reference_accuracy": ref["accuracy"],
                        "accuracy_loss": delta["accuracy_loss"],
                        "mask_nll": candidate["nll"],
                        "reference_nll": ref["nll"],
                        "nll_excess": delta["nll_excess"],
                        "mask_brier": candidate["brier"],
                        "reference_brier": ref["brier"],
                        "brier_excess": delta["brier_excess"],
                        "point_eligible": eligible,
                        "class_counts_json": json.dumps(candidate["class_counts"],
                                                       separators=(",", ":")),
                        "mask_per_class_recall_json": json.dumps(
                            candidate["per_class_recall"], separators=(",", ":")),
                        "reference_per_class_recall_json": json.dumps(
                            ref["per_class_recall"], separators=(",", ":")),
                        "gate_receipt_sha256": receipt_sha,
                        "full_input_gate_sha256": gate_sha,
                        "development_status": "exploratory_reused_Adult_people_point_only",
                    })
    if len(rows) != 60:
        raise PermissionError("task utility table is incomplete")
    return rows


def main() -> None:
    rows = build_rows()
    fd, temporary = tempfile.mkstemp(prefix=".MASK_UTILITY.", dir=STUDY)
    try:
        with os.fdopen(fd, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, OUTPUT)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps({"rows": len(rows), "sha256": sha(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
