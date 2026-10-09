"""Independent aggregate-only replay of five-mask task eligibility and T.

No role file, person-level vector, or runner selection/scoring module is read.
Run after the sealed q12 queue finishes and before the first FIT SEX read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


TASKS = (("income", "income_masks"), ("occupation_group", "occupation_masks"))
REFERENCES = ("U0", "Ucal", "F")
CAPS = {"accuracy": .010, "nll": .010, "brier": .005}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def select_task_mask(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["utility_feasible"]]
    if not eligible:
        raise ValueError("no point utility-feasible mask")
    return min(eligible, key=lambda row: (row["metrics"]["nll"],
                                           row["mask_sha256"]))


def _private_record(private: Path, record: dict) -> Path:
    relative = Path(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json":
        raise PermissionError("task package is not a bounded JSON receipt")
    path = (private / relative).resolve(strict=True)
    if not path.is_relative_to(private) or sha_file(path) != record["sha256"]:
        raise PermissionError("task package receipt bytes differ")
    return path


def audit(study: Path) -> dict:
    study = study.resolve(strict=True)
    private = study / ".private"
    lock_path = study / "CAMPAIGN_LOCK.json"
    lock = json.loads(lock_path.read_text())
    if (lock.get("stage") != "pre_sensitive_five_mask_task_table"
            or lock.get("allowed_roles") != []):
        raise PermissionError("task-table source lock is not derived-only")
    queue_path = private / "queue_g12/QUEUE.json"
    queue = json.loads(queue_path.read_text())
    if (set(queue["units"]) != {"R2_TASK_TABLE_S0", "R2_TASK_TABLE_S1"}
            or any(unit["status"] != "DONE" or unit["exit_code"] != 0
                   for unit in queue["units"].values())):
        raise PermissionError("both sealed task-only table units must finish")
    bank_path = study.parents[1] / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    if sha_file(bank_path) != lock["mask_bank_sha256"]:
        raise PermissionError("frozen bank SHA differs")
    bank = json.loads(bank_path.read_text())
    masks = {item["mask_id"]: item for item in bank["masks"]}
    if set(masks) != set(lock["admitted_mask_ids"]) or len(masks) != 5:
        raise PermissionError("five-mask task-only bank differs")
    by_seed = {}
    for seed in (0, 1):
        directory = private / "task_table" / f"seed{seed}"
        unit = queue["units"][f"R2_TASK_TABLE_S{seed}"]
        paths = {name: directory / name for name in
                 ("TASK_TABLE_RECEIPT.json", "TASK_ONLY_TABLE.json", "T_SELECTION.json")}
        for path in paths.values():
            if unit["output_sha256"].get(str(path.resolve())) != sha_file(path):
                raise PermissionError("task-only output differs from sealed queue")
        receipt = json.loads(paths["TASK_TABLE_RECEIPT.json"].read_text())
        table = json.loads(paths["TASK_ONLY_TABLE.json"].read_text())
        choice = json.loads(paths["T_SELECTION.json"].read_text())
        if (receipt.get("schema") != "pcrl-utility-pre-sensitive-task-table-receipt-v1"
                or receipt.get("status") != "TASK_ONLY_T_FROZEN"
                or receipt.get("seed") != seed
                or receipt.get("sex_read") is not False
                or receipt.get("conditional_lock_sha256") != sha_file(lock_path)
                or receipt.get("bank_sha256") != table.get("bank_sha256")
                or not isinstance(receipt.get("bank_sha256"), str)
                or len(receipt["bank_sha256"]) != 64
                or receipt.get("task_table_file_sha256") !=
                sha_file(paths["TASK_ONLY_TABLE.json"])
                or receipt.get("T_selection_sha256") !=
                sha_file(paths["T_SELECTION.json"])):
            raise PermissionError("pre-sensitive table receipt identity differs")
        bare = {key: value for key, value in table.items()
                if key != "task_table_sha256"}
        if table.get("task_table_sha256") != content_sha(bare):
            raise PermissionError("task table content SHA differs")
        replayed_rows = {}
        selected = {}
        for task, row_key in TASKS:
            observed = table[row_key]
            if len(observed) != 5 or {row["mask_id"] for row in observed} != set(masks):
                raise PermissionError("task table mask coverage differs")
            replayed_rows[task] = []
            for row in observed:
                mask_id = row["mask_id"]
                gate_path = private / "mask_gate" / mask_id / f"seed{seed}" / "MASK_GATE_RECEIPT.json"
                fit_path = private / "masks" / mask_id / f"seed{seed}" / "MASK_FIT_RECEIPT.json"
                if (sha_file(gate_path) != receipt["mask_gate_receipt_sha256_by_id"][mask_id]
                        or sha_file(gate_path) != lock["derived_input_sha256_by_path"].get(str(gate_path))
                        or sha_file(fit_path) != lock["derived_input_sha256_by_path"].get(str(fit_path))):
                    raise PermissionError("mask task aggregate differs from sealed table input")
                gate = json.loads(gate_path.read_text())
                fit = json.loads(fit_path.read_text())
                package_path = _private_record(private, fit["task_packages"][task])
                package = json.loads(package_path.read_text())
                head_id = f"{package['model_id']}:SHA256:{package['model']['sha256']}"
                metrics = gate["metrics"]["mask"][task]
                if (row["mask_sha256"] != masks[mask_id]["mask_sha256"]
                        or row["head_id"] != head_id
                        or any(abs(row["metrics"][key] - metrics[key]) > 1e-12
                               for key in CAPS)):
                    raise PermissionError("task table head or metric differs from gate")
                feasible = True
                for reference in REFERENCES:
                    baseline = gate["metrics"][reference][task]
                    margins = {
                        "accuracy": CAPS["accuracy"] - (baseline["accuracy"] - metrics["accuracy"]),
                        "nll": CAPS["nll"] - (metrics["nll"] - baseline["nll"]),
                        "brier": CAPS["brier"] - (metrics["brier"] - baseline["brier"]),
                    }
                    if any(abs(row["utility_margins_by_ref_metric"][reference][key] - value)
                           > 1e-12 for key, value in margins.items()):
                        raise PermissionError("task table margin differs from gate")
                    expected_flag = all(value >= 0 for value in margins.values())
                    if gate["point_eligibility"][task]["references"][reference]["point_eligible"] is not expected_flag:
                        raise PermissionError("task gate point flag differs")
                    feasible &= expected_flag
                if type(row["utility_feasible"]) is not bool or row["utility_feasible"] is not feasible:
                    raise PermissionError("task table utility feasibility differs")
                replayed_rows[task].append({"mask_id": mask_id,
                                            "utility_feasible": feasible,
                                            "nll": metrics["nll"]})
            selected[task] = select_task_mask(observed)
        first, second = selected["income"], selected["occupation_group"]
        expected_pair = f"{first['mask_id']}__{second['mask_id']}"
        if (choice.get("schema") != "pcrl-focused-pre-sensitive-T-v1"
                or choice.get("task_table_sha256") != table["task_table_sha256"]
                or choice.get("bank_sha256") != table["bank_sha256"]
                or choice.get("pair_id") != expected_pair
                or choice.get("income_mask_id") != first["mask_id"]
                or choice.get("occupation_mask_id") != second["mask_id"]
                or choice.get("head_ids") != {
                    "income": first["head_id"],
                    "occupation_group": second["head_id"]}
                or receipt.get("T_pair_id") != expected_pair):
            raise PermissionError("pre-sensitive T differs from independent ranking")
        bare_choice = {key: value for key, value in choice.items()
                       if key != "selection_receipt_sha256"}
        if choice.get("selection_receipt_sha256") != content_sha(bare_choice):
            raise PermissionError("pre-sensitive T content SHA differs")
        by_seed[str(seed)] = {"T_pair_id": expected_pair,
                             "task_masks": replayed_rows,
                             "receipt_sha256": sha_file(paths["TASK_TABLE_RECEIPT.json"]),
                             "table_sha256": sha_file(paths["TASK_ONLY_TABLE.json"]),
                             "T_selection_sha256": sha_file(paths["T_SELECTION.json"])}
    return {"schema": "pcrl-utility-r3-task-table-selection-audit-v1",
            "status": "PASS_AGGREGATE_SELECTION_REPLAY",
            "scope": "frozen task-only JSON aggregates; no role arrays or SEX labels",
            "source_lock_sha256": sha_file(lock_path),
            "source_queue_state_sha256": sha_file(queue_path),
            "full_input_gate_sha256": lock["full_input_gate_sha256"],
            "mask_bank_sha256": sha_file(bank_path), "by_seed": by_seed,
            "development_only": True}


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    result = audit(study)
    output = study / "r3/TASK_TABLE_SELECTION_AUDIT.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(result["status"], {seed: value["T_pair_id"]
                              for seed, value in result["by_seed"].items()})


if __name__ == "__main__":
    main()
