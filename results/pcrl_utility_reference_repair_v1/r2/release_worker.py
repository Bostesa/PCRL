"""Sealed, SEX-free complete release for one selected pair and paired seed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r1.role_store import sha_file
from results.pcrl_utility_reference_repair_v1.r2.admission import admit_conditional_gate
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.privacy_store import PrivacyRoleStore
from results.pcrl_utility_reference_repair_v1.r2.release_custody import (
    write_complete_release,
)
from results.pcrl_utility_reference_repair_v1.r2.release_loader import (
    restore_mask_task_release,
)


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
MASK_BANK = REPO / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
ROLE_INPUTS = {"ATTACK_FIT_NEW:features", "INNER_SELECTION:features"}


def _private_json(root: Path, path: Path, expected_sha: str, label: str) -> dict:
    real = path.resolve(strict=True)
    if (not real.is_relative_to(root) or not isinstance(expected_sha, str)
            or len(expected_sha) != 64 or sha_file(real) != expected_sha):
        raise PermissionError(f"{label} private bytes differ")
    return json.loads(real.read_text())


def _selected_pair_and_packages(
    spec: dict, lock: dict, lock_sha: str, root: Path, bank: object,
    gate_sha: str, seed: int, pair_id: str,
) -> tuple[object, dict]:
    select_path = root / "pair_select" / f"seed{seed}" / "PAIR_SELECT_RECEIPT.json"
    select_sha = spec.get("inputs", {}).get(str(select_path))
    if select_sha is None:
        raise PermissionError("release is missing sealed pair-selection input")
    selected = _private_json(root, select_path, select_sha, "pair selection")
    source_lock = lock.get("pair_select_source_lock_sha256")
    if source_lock == "CURRENT_LOCK":
        source_lock = lock_sha
    if (selected.get("schema") != "pcrl-utility-pair-selection-receipt-v1"
            or selected.get("seed") != seed
            or selected.get("bank_sha256") != bank.bank_sha256
            or selected.get("full_input_gate_sha256") != gate_sha
            or selected.get("conditional_lock_sha256") != source_lock
            or selected.get("fitted_raw_search_readers") != 22
            or selected.get("no_ATT_or_INNER_SEX_read") is not True):
        raise PermissionError("pair selection is not frozen for this release")
    folder = select_path.parent
    files = selected.get("files_sha256", {})
    plan = _private_json(root, folder / "CONTROL_PLAN.json",
                         files.get("CONTROL_PLAN.json"), "control plan")
    if (plan.get("schema") != "pcrl-focused-m3-control-plan-v1"
            or plan.get("bank_sha256") != bank.bank_sha256
            or plan.get("joint_pair_id") != selected.get("joint_pair_id")):
        raise PermissionError("selected control plan differs from source pair search")
    selected_rules = [item for item in plan["rules"].values()
                      if item["status"] == "SELECTED"]
    if pair_id not in {item["pair_id"] for item in selected_rules}:
        raise PermissionError("release pair is not one of the frozen selected controls")
    pair = next((item for item in bank.pairs if item.pair_id == pair_id), None)
    if pair is None:
        raise PermissionError("selected pair is absent from frozen bank")
    table_path = root / "task_table" / f"seed{seed}" / "TASK_TABLE_RECEIPT.json"
    table_receipt = _private_json(
        root, table_path, selected["task_table_receipt_sha256"],
        "pre-sensitive task table")
    if (table_receipt.get("schema") !=
            "pcrl-utility-pre-sensitive-task-table-receipt-v1"
            or table_receipt.get("status") != "TASK_ONLY_T_FROZEN"
            or table_receipt.get("seed") != seed
            or table_receipt.get("bank_sha256") != bank.bank_sha256
            or table_receipt.get("full_input_gate_sha256") != gate_sha
            or table_receipt.get("sex_read") is not False
            or set(table_receipt.get("mask_gate_receipt_sha256_by_id", {})) !=
            {mask.mask_id for mask in bank.masks}):
        raise PermissionError("release task-only table or mask gate lineage differs")
    packages = {}
    for task, mask_id in (("income", pair.income_mask_id),
                          ("occupation_group", pair.occupation_mask_id)):
        gate_path = (root / "mask_gate" / mask_id / f"seed{seed}" /
                     "MASK_GATE_RECEIPT.json")
        mask_gate = _private_json(
            root, gate_path,
            table_receipt["mask_gate_receipt_sha256_by_id"][mask_id],
            "selected mask utility gate")
        mask = bank.mask(mask_id)
        if (mask_gate.get("schema") != "pcrl-utility-masked-common-task-gate-v1"
                or mask_gate.get("mask_id") != mask_id
                or mask_gate.get("mask_sha256") != mask.mask_sha256
                or mask_gate.get("seed") != seed
                or mask_gate.get("full_input_gate_sha256") != gate_sha
                or mask_gate.get("conditional_lock_sha256") !=
                table_receipt["mask_lock_sha256_by_id"][mask_id]
                or mask_gate.get("selected_recipe") != "A"):
            raise PermissionError("selected mask gate identity differs")
        fit_path = root / "masks" / mask_id / f"seed{seed}" / "MASK_FIT_RECEIPT.json"
        fit_sha = mask_gate["mask_fit_receipt_sha256"]
        fit = _private_json(root, fit_path, fit_sha, "selected mask fit")
        if fit.get("conditional_lock_sha256") != mask_gate.get(
                "mask_fit_source_lock_sha256"):
            raise PermissionError("selected mask fit/gate source lock differs")
        packages[task] = restore_mask_task_release(
            root, {"relative_path": str(fit_path.relative_to(root)),
                   "sha256": fit_sha}, mask, task, seed, "A",
            gate_sha256=gate_sha,
            conditional_lock_sha256=fit["conditional_lock_sha256"])
    head_ids = {name: value.head_id for name, value in packages.items()}
    for item in selected_rules:
        if item["pair_id"] == pair_id and item["head_ids"] != head_ids:
            raise PermissionError("release model heads differ from frozen rule")
    return pair, packages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--pair-id", required=True)
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
                               "r2_release", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("R2 complete release requires one numerical thread")
    root = args.private_root.resolve(strict=True)
    if root != STUDY / ".private":
        raise PermissionError("R2 release private root differs")
    output = root / "releases" / f"seed{args.seed}" / args.pair_id / "RELEASE_RECEIPT.json"
    outputs = {str(output), str(output.parent / "ATTACK_FIT_NEW.npz"),
               str(output.parent / "INNER_SELECTION.npz")}
    if (args.output.resolve() != output or set(spec["outputs"]) != outputs
            or set(spec.get("role_inputs", {})) != ROLE_INPUTS):
        raise PermissionError("R2 selected release output or role inventory differs")
    lock = json.loads(args.lock.read_text())
    lock_sha = sha_file(args.lock)
    manifest_sha = sha_file(args.manifest)
    if (lock.get("source_map_sha256") != manifest_sha
            or lock.get("mask_bank_sha256") != sha_file(MASK_BANK)
            or lock.get("selected_recipe") != "A"):
        raise PermissionError("R2 release source or recipe lock differs")
    admitted = admit_conditional_gate(
        args.gate, expected_gate_sha256=lock["full_input_gate_sha256"],
        expected_source_lock_sha256=lock["source_science_lock_sha256"],
        expected_common_row_unit_pair_sha256=lock[
            "common_row_unit_pair_sha256"], private_root=root)
    if admitted.selected_recipe != "A":
        raise PermissionError("selected F recipe differs from release recipe")
    bank = load_frozen_bank(MASK_BANK, expected_sha256=lock["mask_bank_sha256"])
    if set(lock.get("admitted_mask_ids", [])) != {mask.mask_id for mask in bank.masks}:
        raise PermissionError("selected release requires the full admitted bank")
    pair, packages = _selected_pair_and_packages(
        spec, lock, lock_sha, root, bank, admitted.gate_sha256,
        args.seed, args.pair_id)
    store = PrivacyRoleStore(args.export_root, args.manifest, stage="r2_release")
    sources = {role: store.read(role, "features")
               for role in ("ATTACK_FIT_NEW", "INNER_SELECTION")}
    write_complete_release(
        sources, packages, pair, bank, output.parent, private_root=root,
        seed=args.seed, lock_sha256=lock_sha,
        gate_sha256=admitted.gate_sha256, manifest_sha256=manifest_sha,
        role_feature_sha256={role: store.manifest["role_file_sha256"][
            f"{role}:features"] for role in sources})


if __name__ == "__main__":
    main()
