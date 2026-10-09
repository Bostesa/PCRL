"""Sealed fresh RAW_ONLY readers for one selected canonical union and seed."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r1.role_store import sha_file
from results.pcrl_utility_reference_repair_v1.r2.admission import admit_conditional_gate
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.privacy_store import PrivacyRoleStore


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
MASK_BANK = REPO / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
M4 = REPO / "results/pcrl_capacity_matched_feature_selection_v1/m4"
SHARED = REPO / "results/pcrl_method_portfolio_v1/shared"
for source in (M4, SHARED):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
from raw_union import canonical_raw_union, fit_raw_union_readers  # noqa: E402


THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
ROLE_INPUTS = {"ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
               "INNER_SELECTION:features"}
RAW_IDS = ("RAW_LR_C1", "RAW_HGB_15x200")


def expected_outputs(output: Path) -> set[str]:
    parent = output.parent
    names = {"RAW_UNION_INDEX.json"}
    for reader in RAW_IDS:
        names.add(f"model__{reader}.joblib")
        for role in ("att", "inner"):
            names.add(f"prediction__{reader}__{role}.npy")
            names.add(f"transport__{reader}__{role}.json")
    return {str(parent / name) for name in names}


def required_union_shas(bank: object, plan: dict) -> set[str]:
    """All canonical local and coalition unions disclosed by selected pairs."""
    pairs = {pair.pair_id: pair for pair in bank.pairs}
    diagonal = {(pair.income_mask_id, pair.occupation_mask_id): pair
                for pair in bank.pairs
                if pair.income_mask_id == pair.occupation_mask_id}
    selected = {row["pair_id"] for row in plan["rules"].values()
                if row.get("status") == "SELECTED"}
    if not selected or not selected <= set(pairs):
        raise PermissionError("selected canonical union pair set differs")
    required = set()
    for pair_id in selected:
        pair = pairs[pair_id]
        required.add(pair.union_sha256)
        for mask_id in (pair.income_mask_id, pair.occupation_mask_id):
            required.add(diagonal[(mask_id, mask_id)].union_sha256)
    return required


def _selected_union(spec: dict, lock: dict, lock_sha: str, root: Path,
                    bank: object, seed: int, gate_sha: str,
                    union_sha: str):
    path = root / "pair_select" / f"seed{seed}" / "PAIR_SELECT_RECEIPT.json"
    expected = spec.get("inputs", {}).get(str(path))
    if expected is None or sha_file(path) != expected:
        raise PermissionError("RAW_ONLY source pair-selection receipt differs")
    receipt = json.loads(path.read_text())
    source_lock = lock.get("pair_select_source_lock_sha256")
    if source_lock == "CURRENT_LOCK":
        source_lock = lock_sha
    if (receipt.get("schema") != "pcrl-utility-pair-selection-receipt-v1"
            or receipt.get("seed") != seed
            or receipt.get("bank_sha256") != bank.bank_sha256
            or receipt.get("conditional_lock_sha256") != source_lock
            or receipt.get("full_input_gate_sha256") != gate_sha
            or receipt.get("fitted_raw_search_readers") != 22):
        raise PermissionError("RAW_ONLY selected pair source differs")
    plan_path = path.parent / "CONTROL_PLAN.json"
    if sha_file(plan_path) != receipt.get("files_sha256", {}).get(
            "CONTROL_PLAN.json"):
        raise PermissionError("RAW_ONLY selected control plan bytes differ")
    plan = json.loads(plan_path.read_text())
    if (plan.get("schema") != "pcrl-focused-m3-control-plan-v1"
            or plan.get("bank_sha256") != bank.bank_sha256
            or union_sha not in required_union_shas(bank, plan)):
        raise PermissionError("RAW_ONLY union is not in a selected complete view")
    matches = [pair for pair in bank.pairs if pair.union_sha256 == union_sha]
    if (not matches or len({(pair.union_atoms, pair.union_columns)
                            for pair in matches}) != 1):
        raise PermissionError("RAW_ONLY union geometry is not unique")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--union-sha256", required=True)
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
                               "r2_raw_union", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("R2 RAW_ONLY fit requires one numerical thread")
    root = args.private_root.resolve(strict=True)
    output = root / "raw_union" / f"seed{args.seed}" / args.union_sha256 / "RAW_UNION_INDEX.json"
    if (root != STUDY / ".private" or args.output.resolve() != output
            or set(spec["outputs"]) != expected_outputs(output)
            or set(spec.get("role_inputs", {})) != ROLE_INPUTS):
        raise PermissionError("RAW_ONLY output or role inventory differs")
    lock = json.loads(args.lock.read_text())
    lock_sha, manifest_sha = sha_file(args.lock), sha_file(args.manifest)
    if (lock.get("source_map_sha256") != manifest_sha
            or lock.get("mask_bank_sha256") != sha_file(MASK_BANK)
            or lock.get("selected_recipe") != "A"):
        raise PermissionError("RAW_ONLY source lock or recipe differs")
    admitted = admit_conditional_gate(
        args.gate, expected_gate_sha256=lock["full_input_gate_sha256"],
        expected_source_lock_sha256=lock["source_science_lock_sha256"],
        expected_common_row_unit_pair_sha256=lock[
            "common_row_unit_pair_sha256"], private_root=root)
    bank = load_frozen_bank(MASK_BANK, expected_sha256=lock["mask_bank_sha256"])
    if set(lock.get("admitted_mask_ids", [])) != {mask.mask_id for mask in bank.masks}:
        raise PermissionError("RAW_ONLY requires completed five-mask bank")
    pair = _selected_union(spec, lock, lock_sha, root, bank, args.seed,
                           admitted.gate_sha256, args.union_sha256)
    store = PrivacyRoleStore(args.export_root, args.manifest, stage="r2_raw_union")
    att = store.read("ATTACK_FIT_NEW", "features")
    sex = store.read("ATTACK_FIT_NEW", "sex")
    inner = store.read("INNER_SELECTION", "features")
    names = tuple(str(value) for value in att["feature_names"])
    if tuple(str(value) for value in inner["feature_names"]) != names:
        raise PermissionError("RAW_ONLY role feature schema differs")
    columns = pair.union_columns
    feature_names = tuple(names[j] for j in columns)
    att_union = canonical_raw_union(att, source_columns=columns,
                                    feature_names=feature_names)
    inner_union = canonical_raw_union(inner, source_columns=columns,
                                      feature_names=feature_names)
    fit_raw_union_readers(
        att_union, sex, inner_union, seed=args.seed,
        union_sha256=pair.union_sha256, union_atoms=pair.union_atoms,
        source_schema_sha256=bank.source_schema_sha256,
        lock_sha256=lock_sha, manifest_sha256=manifest_sha,
        private_output=output.parent)


if __name__ == "__main__":
    main()
