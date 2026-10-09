"""Sealed fresh complete-view readers using ATT SEX and no INNER SEX."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r1.role_store import sha_file
from results.pcrl_utility_reference_repair_v1.r2.admission import admit_conditional_gate
from results.pcrl_utility_reference_repair_v1.r2.attack_fit import (
    expected_outputs, fit_complete_attack_view,
)
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.privacy_store import PrivacyRoleStore
from results.pcrl_utility_reference_repair_v1.r2.release_custody import (
    restore_complete_release,
)
from results.pcrl_utility_reference_repair_v1.r2.release_worker import (
    _selected_pair_and_packages,
)


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
MASK_BANK = REPO / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
M4 = REPO / "results/pcrl_capacity_matched_feature_selection_v1/m4"
SHARED = REPO / "results/pcrl_method_portfolio_v1/shared"
for source in (M4, SHARED):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
from raw_union import canonical_raw_union  # noqa: E402


THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
ROLE_INPUTS = {"ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
               "INNER_SELECTION:features"}
VIEWS = ("income", "occupation_group", "coalition")


def _source_lock(lock: dict, name: str, current: str) -> str:
    value = lock.get(name)
    source = current if value == "CURRENT_LOCK" else value
    if (not isinstance(source, str) or len(source) != 64
            or any(char not in "0123456789abcdef" for char in source)):
        raise PermissionError(f"{name} is incomplete")
    return source


def _spec_input(spec: dict, path: Path, label: str) -> str:
    expected = spec.get("inputs", {}).get(str(path))
    if expected is None or sha_file(path) != expected:
        raise PermissionError(f"{label} sealed bytes differ")
    return expected


def _union_pair(bank: object, pair: object, view: str):
    if view == "coalition":
        return pair
    mask_id = pair.income_mask_id if view == "income" else pair.occupation_mask_id
    matches = [candidate for candidate in bank.pairs
               if candidate.income_mask_id == mask_id
               and candidate.occupation_mask_id == mask_id]
    if len(matches) != 1:
        raise PermissionError("local canonical RAW_ONLY union missing")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--view", choices=VIEWS, required=True)
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
                               "r2_attack_fit", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("R2 fresh ATT reader fit requires one thread")
    root = args.private_root.resolve(strict=True)
    output = (root / "attacks" / f"seed{args.seed}" / args.pair_id /
              args.view / "ATTACK_FIT_INDEX.json")
    if (root != STUDY / ".private" or args.output.resolve() != output
            or set(spec.get("role_inputs", {})) != ROLE_INPUTS
            or set(spec["outputs"]) != expected_outputs(output)):
        raise PermissionError("R2 ATT reader output or role inventory differs")
    lock = json.loads(args.lock.read_text())
    lock_sha, manifest_sha = sha_file(args.lock), sha_file(args.manifest)
    if (lock.get("source_map_sha256") != manifest_sha
            or lock.get("mask_bank_sha256") != sha_file(MASK_BANK)
            or lock.get("selected_recipe") != "A"):
        raise PermissionError("R2 ATT reader source lock differs")
    admitted = admit_conditional_gate(
        args.gate, expected_gate_sha256=lock["full_input_gate_sha256"],
        expected_source_lock_sha256=lock["source_science_lock_sha256"],
        expected_common_row_unit_pair_sha256=lock[
            "common_row_unit_pair_sha256"], private_root=root)
    bank = load_frozen_bank(MASK_BANK, expected_sha256=lock["mask_bank_sha256"])
    if set(lock.get("admitted_mask_ids", [])) != {mask.mask_id for mask in bank.masks}:
        raise PermissionError("R2 ATT reader needs completed five-mask bank")
    pair, packages = _selected_pair_and_packages(
        spec, lock, lock_sha, root, bank, admitted.gate_sha256,
        args.seed, args.pair_id)
    release_path = (root / "releases" / f"seed{args.seed}" / args.pair_id /
                    "RELEASE_RECEIPT.json")
    release_sha = _spec_input(spec, release_path, "complete release")
    raw_pair = _union_pair(bank, pair, args.view)
    raw_path = (root / "raw_union" / f"seed{args.seed}" /
                raw_pair.union_sha256 / "RAW_UNION_INDEX.json")
    raw_sha = _spec_input(spec, raw_path, "RAW_ONLY reader")
    raw_index = json.loads(raw_path.read_text())
    raw_source_lock = _source_lock(lock, "raw_union_source_lock_sha256",
                                   lock_sha)
    if (raw_index.get("schema") != "pcrl-focused-raw-union-reader-index-v1"
            or raw_index.get("seed") != args.seed
            or raw_index.get("union_sha256") != raw_pair.union_sha256
            or raw_index.get("lock_sha256") != raw_source_lock
            or raw_index.get("manifest_sha256") != manifest_sha):
        raise PermissionError("RAW_ONLY reader source lock or union differs")
    release_source_lock = _source_lock(lock, "release_source_lock_sha256",
                                       lock_sha)
    local_paths = {}
    if args.view == "coalition":
        for local in VIEWS[:2]:
            path = (root / "attacks" / f"seed{args.seed}" / args.pair_id /
                    local / "ATTACK_FIT_INDEX.json")
            local_paths[local] = (path, _spec_input(
                spec, path, f"{local} ignore-recipient source"))
    store = PrivacyRoleStore(args.export_root, args.manifest,
                             stage="r2_attack_fit")
    sources = {role: store.read(role, "features")
               for role in ("ATTACK_FIT_NEW", "INNER_SELECTION")}
    restored = restore_complete_release(
        release_path, expected_index_sha256=release_sha,
        source_roles=sources, public_releases=packages, pair=pair, bank=bank,
        private_root=root, seed=args.seed,
        lock_sha256=release_source_lock,
        gate_sha256=admitted.gate_sha256, manifest_sha256=manifest_sha,
        role_feature_sha256={role: store.manifest["role_file_sha256"][
            f"{role}:features"] for role in sources})
    sex_att = store.read("ATTACK_FIT_NEW", "sex")
    view_att = restored["ATTACK_FIT_NEW"]["views"]
    view_inner = restored["INNER_SELECTION"]["views"]
    columns = raw_pair.union_columns
    feature_names = tuple(str(value) for value in
                          sources["ATTACK_FIT_NEW"]["feature_names"])
    names = tuple(feature_names[j] for j in columns)
    att_union = canonical_raw_union(sources["ATTACK_FIT_NEW"],
                                    source_columns=columns, feature_names=names)
    inner_union = canonical_raw_union(sources["INNER_SELECTION"],
                                      source_columns=columns, feature_names=names)
    dependencies = None
    local_source_lock = None
    if args.view == "coalition":
        local_source_lock = _source_lock(
            lock, "local_attack_source_lock_sha256", lock_sha)
        dependencies = {
            local: (path, sha, view_att[local], view_inner[local])
            for local, (path, sha) in local_paths.items()}
    fit_complete_attack_view(
        att_view=view_att[args.view], sex_att=sex_att,
        inner_view=view_inner[args.view], att_union=att_union,
        inner_union=inner_union, raw_index_path=raw_path,
        raw_index_sha256=raw_sha, release_index_sha256=release_sha,
        seed=args.seed, pair_id=args.pair_id, view_name=args.view,
        lock_sha256=lock_sha, manifest_sha256=manifest_sha,
        private_output=output.parent, local_dependencies=dependencies,
        local_attack_source_lock_sha256=local_source_lock)


if __name__ == "__main__":
    main()
