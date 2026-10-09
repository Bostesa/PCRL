"""Publish five-mask task utility and T before any FIT SEX access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r1.role_store import sha_file
from results.pcrl_utility_reference_repair_v1.r2.admission import admit_conditional_gate
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.task_table import (
    build_pre_sensitive_task_table,
)
from results.pcrl_utility_reference_repair_v1.r2.lead_branch import LEADS


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
MASK_BANK = REPO / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
TASKS = ("income", "occupation_group")


def _write_once_json(path: Path, record: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"),
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


def _input_receipt(path: Path, spec: dict) -> dict:
    expected = spec.get("inputs", {}).get(str(path))
    if expected is None or sha_file(path) != expected:
        raise PermissionError("task-only predecessor receipt differs from sealed unit")
    return json.loads(path.read_text())


def _private_record_file(root: Path, record: dict, label: str) -> Path:
    if (not isinstance(record, dict)
            or not isinstance(record.get("relative_path"), str)
            or not isinstance(record.get("sha256"), str)):
        raise PermissionError(f"{label} identity is incomplete")
    relative = Path(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError(f"{label} path escapes private root")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or sha_file(path) != record["sha256"]:
        raise PermissionError(f"{label} path or bytes differ")
    return path


def resolve_mask_lock_lineage(lock: dict, bank, current_sha256: str) -> dict[str, str]:
    mapping = lock.get("mask_lock_sha256_by_id")
    mask_ids = {mask.mask_id for mask in bank.masks}
    def full_sha(value: object) -> bool:
        return (isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value))
    if (not full_sha(current_sha256)
            or not isinstance(mapping, dict) or set(mapping) != mask_ids
            or any(not (full_sha(value) or
                        (value == "CURRENT_LOCK" and mask_id not in LEADS))
                   for mask_id, value in mapping.items())):
        raise PermissionError("bank mask-to-lock lineage map differs")
    return {mask_id: (current_sha256 if value == "CURRENT_LOCK" else value)
            for mask_id, value in mapping.items()}


def publish_task_table(bank, seed: int, private_root: Path, output: Path,
                       spec: dict, common_row_unit_sha256: str,
                       *, gate_sha256: str, lock_sha256: str,
                       source_locks: dict[str, str],
                       r1_gate_receipt_sha256: str,
                       mask_lock_source_spec: dict[str, str]) -> dict:
    gates, heads, shas = {}, {}, {}
    for mask in bank.masks:
        mask_id = mask.mask_id
        gate_path = private_root / "mask_gate" / mask_id / f"seed{seed}" / "MASK_GATE_RECEIPT.json"
        fit_path = private_root / "masks" / mask_id / f"seed{seed}" / "MASK_FIT_RECEIPT.json"
        gate = _input_receipt(gate_path, spec)
        fit = _input_receipt(fit_path, spec)
        if (gate.get("full_input_gate_sha256") != gate_sha256
                or gate.get("conditional_lock_sha256") != source_locks[mask_id]
                or gate.get("mask_fit_source_lock_sha256") !=
                fit.get("conditional_lock_sha256")
                or gate.get("mask_fit_receipt_sha256") != sha_file(fit_path)
                or fit.get("mask_id") != mask_id
                or fit.get("seed") != seed
                or set(fit.get("task_packages", {})) != set(TASKS)):
            raise PermissionError("mask gate and fit package identities differ")
        gates[mask_id] = gate
        shas[mask_id] = sha_file(gate_path)
        heads[mask_id] = {}
        for task in TASKS:
            entry = fit["task_packages"][task]
            package_path = _private_record_file(
                private_root, entry, "task-only head package")
            package = json.loads(package_path.read_text())
            if (package.get("schema") != "pcrl-utility-masked-task-package-v1"
                    or package.get("task") != task
                    or package.get("mask_id") != mask_id
                    or package.get("seed") != seed
                    or package.get("conditional_lock_sha256") !=
                    gate["mask_fit_source_lock_sha256"]
                    or package.get("selected_recipe") != "A"):
                raise PermissionError("task-only head package identity differs")
            model = package["model"]
            _private_record_file(private_root, model,
                                 "task-only head checkpoint")
            heads[mask_id][task] = f"{package['model_id']}:SHA256:{model['sha256']}"
    table, t_choice = build_pre_sensitive_task_table(
        bank, seed, common_row_unit_sha256, gates, heads, shas,
        full_input_gate_sha256=gate_sha256, selected_recipe="A",
        r1_gate_receipt_sha256=r1_gate_receipt_sha256,
        mask_lock_sha256_by_id=source_locks)
    table_path = output.parent / "TASK_ONLY_TABLE.json"
    table_sha = _write_once_json(table_path, table)
    selection_path = output.parent / "T_SELECTION.json"
    selection = (t_choice if t_choice is not None else {
        "schema": "pcrl-utility-pre-sensitive-T-no-go-v1",
        "status": "NO_UTILITY_FEASIBLE_TASK_ONLY_PAIR",
        "seed": seed, "task_table_sha256": table["task_table_sha256"],
        "sex_read": False,
    })
    selection_sha = _write_once_json(selection_path, selection)
    receipt = {
        "schema": "pcrl-utility-pre-sensitive-task-table-receipt-v1",
        "seed": seed,
        "status": "TASK_ONLY_T_FROZEN" if t_choice is not None else
                  "NO_UTILITY_FEASIBLE_TASK_ONLY_PAIR",
        "bank_sha256": bank.bank_sha256,
        "full_input_gate_sha256": gate_sha256,
        "conditional_lock_sha256": lock_sha256,
        "common_row_unit_pair_sha256": common_row_unit_sha256,
        "task_table_relative_path": str(table_path.relative_to(private_root)),
        "task_table_file_sha256": table_sha,
        "task_table_content_sha256": table["task_table_sha256"],
        "T_selection_relative_path": str(selection_path.relative_to(private_root)),
        "T_selection_sha256": selection_sha,
        "T_pair_id": None if t_choice is None else t_choice["pair_id"],
        "mask_gate_receipt_sha256_by_id": shas,
        "mask_lock_sha256_by_id": source_locks,
        "mask_lock_source_spec_by_id": mask_lock_source_spec,
        "sex_read": False,
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
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = assert_running_unit(args.unit_id, args.queue_root, args.lock,
                               "r2_task_table", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("R2 task-only table requires one numerical thread")
    private_root = args.private_root.resolve(strict=True)
    if private_root != STUDY / ".private":
        raise PermissionError("R2 task-only table custody differs")
    lock = json.loads(args.lock.read_text())
    lock_sha = sha_file(args.lock)
    if lock.get("mask_bank_sha256") != sha_file(MASK_BANK):
        raise PermissionError("task-only table mask bank differs")
    admitted = admit_conditional_gate(
        args.gate, expected_gate_sha256=lock["full_input_gate_sha256"],
        expected_source_lock_sha256=lock["source_science_lock_sha256"],
        expected_common_row_unit_pair_sha256=lock["common_row_unit_pair_sha256"],
        private_root=private_root)
    if (admitted.selected_recipe != "A"
            or lock.get("selected_recipe") != "A"):
        raise PermissionError("task-only table requires the selected A recipe")
    bank = load_frozen_bank(MASK_BANK, expected_sha256=lock["mask_bank_sha256"])
    if set(lock.get("admitted_mask_ids", [])) != {mask.mask_id for mask in bank.masks}:
        raise PermissionError("five-mask bank is not sealed and admitted")
    source_locks = resolve_mask_lock_lineage(lock, bank, lock_sha)
    lead_relative = Path(lock.get("lead_branch_receipt_relative_path", ""))
    if (lead_relative.is_absolute() or ".." in lead_relative.parts
            or not lead_relative.parts):
        raise PermissionError("lead qualification receipt path is invalid")
    lead_path = (STUDY / lead_relative).resolve(strict=True)
    if (not lead_path.is_relative_to(STUDY)
            or sha_file(lead_path) != lock.get("lead_branch_receipt_sha256")):
        raise PermissionError("lead qualification receipt bytes differ")
    lead = json.loads(lead_path.read_text())
    if (lead.get("status") != "LEAD_QUALIFIED"
            or not set(lead.get("qualified_leads", [])).intersection(LEADS)):
        raise PermissionError("two-seed lead qualification is not sealed")
    expected = private_root / "task_table" / f"seed{args.seed}" / "TASK_TABLE_RECEIPT.json"
    outputs = {str(expected), str(expected.parent / "TASK_ONLY_TABLE.json"),
               str(expected.parent / "T_SELECTION.json")}
    if args.output.resolve() != expected or set(spec["outputs"]) != outputs:
        raise PermissionError("task-only table output inventory differs")
    publish_task_table(
        bank, args.seed, private_root, expected, spec,
        admitted.common_row_unit_pair_sha256,
        gate_sha256=admitted.gate_sha256, lock_sha256=lock_sha,
        source_locks=source_locks,
        r1_gate_receipt_sha256={key: sha for key, _, sha in
                                admitted.r1_gate_receipt_paths}[str(args.seed)],
        mask_lock_source_spec=lock["mask_lock_sha256_by_id"])


if __name__ == "__main__":
    main()
