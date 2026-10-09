"""One sealed FIT SEX search after five task gates and pre-sensitive T custody.

This unit fits exactly the source-matched 22 raw proxy readers per seed. It
selects J and all nine controls from one immutable 25-pair table. No ATT or
INNER SEX array is opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r1.fit_splits import outer_split
from results.pcrl_utility_reference_repair_v1.r1.role_store import (
    assert_same_people, sha_file,
)
from results.pcrl_utility_reference_repair_v1.r2.admission import admit_conditional_gate
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.pair_selection import (
    build_matched_selection,
)
from results.pcrl_utility_reference_repair_v1.r2.privacy_store import PrivacyRoleStore
from results.pcrl_utility_reference_repair_v1.r2.raw_search import fit_fixed_raw_search


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
MASK_BANK = REPO / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
TASKS = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal", "F")


def _write_once_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _private_bound_json(private_root: Path, relative: str,
                        expected_sha256: str, label: str) -> dict:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise PermissionError(f"{label} path escapes private root")
    path = (private_root / value).resolve(strict=True)
    if (not path.is_relative_to(private_root)
            or sha_file(path) != expected_sha256):
        raise PermissionError(f"{label} bytes differ")
    return json.loads(path.read_text())


def _task_input(private_root: Path, seed: int, expected_receipt_sha: str,
                expected_source_lock_sha: str, bank_sha256: str,
                gate_sha256: str, common_sha256: str) -> tuple[dict, dict, dict]:
    folder = private_root / "task_table" / f"seed{seed}"
    receipt_path = folder / "TASK_TABLE_RECEIPT.json"
    if sha_file(receipt_path) != expected_receipt_sha:
        raise PermissionError("pre-sensitive task-table receipt bytes differ")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "pcrl-utility-pre-sensitive-task-table-receipt-v1"
            or receipt.get("status") != "TASK_ONLY_T_FROZEN"
            or receipt.get("seed") != seed
            or receipt.get("bank_sha256") != bank_sha256
            or receipt.get("full_input_gate_sha256") != gate_sha256
            or receipt.get("conditional_lock_sha256") != expected_source_lock_sha
            or receipt.get("common_row_unit_pair_sha256") != common_sha256
            or receipt.get("sex_read") is not False):
        raise PermissionError("pre-sensitive task table/T was not frozen")
    table = _private_bound_json(
        private_root, receipt["task_table_relative_path"],
        receipt["task_table_file_sha256"], "task-only table")
    choice = _private_bound_json(
        private_root, receipt["T_selection_relative_path"],
        receipt["T_selection_sha256"], "pre-sensitive T")
    if (table.get("task_table_sha256") != receipt["task_table_content_sha256"]
            or choice.get("pair_id") != receipt["T_pair_id"]
            or choice.get("task_table_sha256") != table["task_table_sha256"]):
        raise PermissionError("pre-sensitive T/table content identity differs")
    return receipt, table, choice


def select_pair_unit(store: PrivacyRoleStore, bank, seed: int,
                     private_root: Path, output: Path, *,
                     task_receipt: dict, task_table: dict, t_choice: dict,
                     task_receipt_sha256: str, gate_sha256: str,
                     lock_sha256: str) -> dict:
    # The task-only table and T are already published and verified. FIT SEX
    # remains unopened until the grouped split and person alignment are fixed.
    feature = store.read("OSF_DEFENSE_FIT", "features")
    income = store.read("OSF_DEFENSE_FIT", "income")
    occupation = store.read("OSF_DEFENSE_FIT", "occupation")
    assert_same_people(feature, income, occupation)
    yi, yo = income["income"], occupation["occupation"]
    fit_rows, val_rows, split_receipt = outer_split(
        feature["unit"], yi, yo, seed)
    first_mask = bank.masks[0]
    gate_path = private_root / "mask_gate" / first_mask.mask_id / f"seed{seed}" / "MASK_GATE_RECEIPT.json"
    expected_first_sha = task_receipt["mask_gate_receipt_sha256_by_id"][first_mask.mask_id]
    if sha_file(gate_path) != expected_first_sha:
        raise PermissionError("task-only reference gate aggregate differs")
    first_gate = json.loads(gate_path.read_text())
    if first_gate.get("full_input_gate_sha256") != gate_sha256:
        raise PermissionError("task-only F gate reference differs")
    references = {
        task: {reference: first_gate["metrics"][reference][task]
               for reference in REFERENCES}
        for task in TASKS
    }
    sex_record = store.read("OSF_DEFENSE_FIT", "sex")
    assert_same_people(feature, sex_record)
    local, unions, search = fit_fixed_raw_search(
        bank, feature["X"], sex_record["sex"], fit_rows, val_rows, seed)
    table, joint, plan = build_matched_selection(
        bank, task_table, t_choice, references, local, unions)
    folder = output.parent
    files = {}
    for filename, content in (
        ("PAIR_TABLE.json", table.to_payload()),
        ("JOINT_SELECTION.json", joint.to_payload()),
        ("CONTROL_PLAN.json", plan),
        ("RAW_SEARCH_RECEIPT.json", {
            **search, "task_table_receipt_sha256": task_receipt_sha256,
            "FIT_outer_split": split_receipt,
            "fit_role": "OSF_DEFENSE_FIT:outer_group_fit",
            "validation_role": "OSF_DEFENSE_FIT:outer_group_validation",
            "privacy_label": "SEX", "development_only": True,
        }),
    ):
        path = folder / filename
        files[filename] = _write_once_json(path, content)
    receipt = {
        "schema": "pcrl-utility-pair-selection-receipt-v1",
        "seed": seed, "bank_sha256": bank.bank_sha256,
        "full_input_gate_sha256": gate_sha256,
        "conditional_lock_sha256": lock_sha256,
        "task_table_receipt_sha256": task_receipt_sha256,
        "task_only_T_pair_id": t_choice["pair_id"],
        "joint_pair_id": joint.pair_id,
        "selected_rule_count": sum(
            item["status"] == "SELECTED" for item in plan["rules"].values()),
        "fitted_raw_search_readers": search["fitted_reader_count"],
        "FIT_outer_split": split_receipt,
        "files_sha256": files,
        "sex_roles_read": ["OSF_DEFENSE_FIT"],
        "no_ATT_or_INNER_SEX_read": True,
        "development_only": True,
    }
    _write_once_json(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = assert_running_unit(args.unit_id, args.queue_root, args.lock,
                               "r2_pair_select", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("R2 pair search requires one numerical thread")
    root = args.private_root.resolve(strict=True)
    if root != STUDY / ".private":
        raise PermissionError("R2 pair search custody differs")
    lock = json.loads(args.lock.read_text())
    lock_sha = sha_file(args.lock)
    if (lock.get("mask_bank_sha256") != sha_file(MASK_BANK)
            or lock.get("source_map_sha256") != sha_file(args.manifest)
            or lock.get("selected_recipe") != "A"):
        raise PermissionError("R2 pair search lock or source identity differs")
    admitted = admit_conditional_gate(
        args.gate, expected_gate_sha256=lock["full_input_gate_sha256"],
        expected_source_lock_sha256=lock["source_science_lock_sha256"],
        expected_common_row_unit_pair_sha256=lock["common_row_unit_pair_sha256"],
        private_root=root)
    if admitted.selected_recipe != "A":
        raise PermissionError("R2 pair search uses the selected A recipe only")
    bank = load_frozen_bank(MASK_BANK, expected_sha256=lock["mask_bank_sha256"])
    if set(lock.get("admitted_mask_ids", [])) != {mask.mask_id for mask in bank.masks}:
        raise PermissionError("five-mask bank not admitted for pair selection")
    expected = root / "pair_select" / f"seed{args.seed}" / "PAIR_SELECT_RECEIPT.json"
    files = {str(expected)} | {str(expected.parent / name) for name in (
        "PAIR_TABLE.json", "JOINT_SELECTION.json", "CONTROL_PLAN.json",
        "RAW_SEARCH_RECEIPT.json")}
    if args.output.resolve() != expected or set(spec["outputs"]) != files:
        raise PermissionError("R2 pair selection output inventory differs")
    task_path = root / "task_table" / f"seed{args.seed}" / "TASK_TABLE_RECEIPT.json"
    task_sha = spec.get("inputs", {}).get(str(task_path))
    if task_sha is None or sha_file(task_path) != task_sha:
        raise PermissionError("task table was not a sealed pair-search input")
    source_sha = (lock_sha if lock.get("task_table_source_lock_sha256") ==
                  "CURRENT_LOCK" else lock.get("task_table_source_lock_sha256"))
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise PermissionError("task-table source lock is incomplete")
    task_receipt, task_table, t_choice = _task_input(
        root, args.seed, task_sha, source_sha, bank.bank_sha256,
        admitted.gate_sha256, admitted.common_row_unit_pair_sha256)
    store = PrivacyRoleStore(args.export_root, args.manifest,
                             stage="r2_pair_select")
    select_pair_unit(
        store, bank, args.seed, root, expected,
        task_receipt=task_receipt, task_table=task_table, t_choice=t_choice,
        task_receipt_sha256=task_sha, gate_sha256=admitted.gate_sha256,
        lock_sha256=lock_sha)


if __name__ == "__main__":
    main()
