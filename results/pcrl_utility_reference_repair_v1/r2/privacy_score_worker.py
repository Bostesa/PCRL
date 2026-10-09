"""One sealed seed-level INNER SEX score after all fresh ATT fits are frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r1.role_store import sha_file
from results.pcrl_utility_reference_repair_v1.r2.admission import admit_conditional_gate
from results.pcrl_utility_reference_repair_v1.r2.attack_fit import (
    expected_outputs as attack_output_paths,
)
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.privacy_score import (
    score_saved_attack_view,
)
from results.pcrl_utility_reference_repair_v1.r2.privacy_store import PrivacyRoleStore


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
MASK_BANK = REPO / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
VIEWS = ("income", "occupation_group", "coalition")


def _source_lock(lock: dict, name: str, current: str) -> str:
    value = lock.get(name)
    source = current if value == "CURRENT_LOCK" else value
    if (not isinstance(source, str) or len(source) != 64
            or any(char not in "0123456789abcdef" for char in source)):
        raise PermissionError(f"{name} is incomplete")
    return source


def _input(spec: dict, path: Path, label: str) -> str:
    expected = spec.get("inputs", {}).get(str(path))
    if expected is None or sha_file(path) != expected:
        raise PermissionError(f"{label} sealed input bytes differ")
    return expected


def _private_component(root: Path, relative_value: object, label: str) -> Path:
    if not isinstance(relative_value, str):
        raise PermissionError(f"{label} path is incomplete")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise PermissionError(f"{label} path escapes private custody")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise PermissionError(f"{label} path escapes private custody")
    return path


def sealed_common_input_paths(
    spec: dict, gate_path: Path, manifest_path: Path, bank_path: Path,
    export_root: Path, private_root: Path,
) -> set[str]:
    """Bind every common file opened by deep F admission and INNER scoring."""
    root = private_root.resolve(strict=True)
    gate_path = gate_path.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    bank_path = bank_path.resolve(strict=True)
    sex_path = (export_root.resolve(strict=True) /
                "INNER_SELECTION" / "sex.npz").resolve(strict=True)
    common = {str(gate_path), str(manifest_path), str(bank_path), str(sex_path)}
    for path, label in ((gate_path, "full input gate"),
                        (manifest_path, "role manifest"),
                        (bank_path, "frozen mask bank")):
        _input(spec, path, label)
    manifest = json.loads(manifest_path.read_text())
    sex_sha = manifest.get("role_file_sha256", {}).get("INNER_SELECTION:sex")
    if (not isinstance(sex_sha, str) or len(sex_sha) != 64
            or spec.get("inputs", {}).get(str(sex_path)) != sex_sha):
        raise PermissionError("INNER SEX sealed role identity differs")
    gate = json.loads(gate_path.read_text())
    if gate.get("selected_recipe") != "A":
        raise PermissionError("privacy score full-input recipe differs")
    for seed in ("0", "1"):
        r1_dir = root / "r1" / "gate" / f"seed_{seed}"
        for name, key in (("GATE_RECEIPT.json",
                           "r1_gate_receipt_sha256_by_seed"),
                          ("GATE_PREDICTIONS.npz",
                           "r1_gate_prediction_sha256_by_seed")):
            path = (r1_dir / name).resolve(strict=True)
            expected = gate.get(key, {}).get(seed)
            if _input(spec, path, "R1 gate " + name) != expected:
                raise PermissionError("R1 gate source bytes differ")
            common.add(str(path))
        f_receipt = gate.get("f_receipts", {}).get(seed, {})
        receipt_path = _private_component(
            root, f_receipt.get("relative_path"), "F package receipt")
        if _input(spec, receipt_path, "F package receipt") != f_receipt.get("sha256"):
            raise PermissionError("F package receipt source bytes differ")
        common.add(str(receipt_path))
        for task in ("income", "occupation_group"):
            package_record = gate.get("f_packages", {}).get(seed, {}).get(task, {})
            package_path = _private_component(
                root, package_record.get("relative_path"), "F task package")
            if _input(spec, package_path, "F task package") != package_record.get("sha256"):
                raise PermissionError("F task package source bytes differ")
            common.add(str(package_path))
            package = json.loads(package_path.read_text())
            for key in ("model", "calibrator"):
                component = package.get(key, {})
                path = _private_component(root, component.get("relative_path"),
                                          "F " + key)
                if _input(spec, path, "F " + key) != component.get("sha256"):
                    raise PermissionError("F " + key + " source bytes differ")
                common.add(str(path))
    return common


def _write_once_json(path: Path, content: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(content, sort_keys=True, separators=(",", ":"),
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
                               "r2_privacy_score", args.output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("R2 privacy score needs one numerical thread")
    root = args.private_root.resolve(strict=True)
    output = root / "privacy_score" / f"seed{args.seed}" / "PRIVACY_SCORE_RECEIPT.json"
    scores_path = output.parent / "PRIVACY_SCORES.json"
    if (root != STUDY / ".private" or args.output.resolve() != output
            or set(spec["outputs"]) != {str(output), str(scores_path)}
            or set(spec.get("role_inputs", {})) != {"INNER_SELECTION:sex"}):
        raise PermissionError("R2 privacy score output or role inventory differs")
    lock = json.loads(args.lock.read_text())
    lock_sha, manifest_sha = sha_file(args.lock), sha_file(args.manifest)
    if (lock.get("source_map_sha256") != manifest_sha
            or lock.get("mask_bank_sha256") != sha_file(MASK_BANK)
            or lock.get("selected_recipe") != "A"):
        raise PermissionError("R2 privacy score source lock differs")
    common_inputs = sealed_common_input_paths(
        spec, args.gate, args.manifest, MASK_BANK, args.export_root, root)
    admitted = admit_conditional_gate(
        args.gate, expected_gate_sha256=lock["full_input_gate_sha256"],
        expected_source_lock_sha256=lock["source_science_lock_sha256"],
        expected_common_row_unit_pair_sha256=lock[
            "common_row_unit_pair_sha256"], private_root=root)
    bank = load_frozen_bank(MASK_BANK, expected_sha256=lock["mask_bank_sha256"])
    if set(lock.get("admitted_mask_ids", [])) != {mask.mask_id for mask in bank.masks}:
        raise PermissionError("privacy score requires completed mask bank")
    selection_path = root / "pair_select" / f"seed{args.seed}" / "PAIR_SELECT_RECEIPT.json"
    selection_sha = _input(spec, selection_path, "pair selection")
    selection = json.loads(selection_path.read_text())
    select_lock = _source_lock(lock, "pair_select_source_lock_sha256", lock_sha)
    if (selection.get("schema") != "pcrl-utility-pair-selection-receipt-v1"
            or selection.get("seed") != args.seed
            or selection.get("bank_sha256") != bank.bank_sha256
            or selection.get("full_input_gate_sha256") != admitted.gate_sha256
            or selection.get("conditional_lock_sha256") != select_lock):
        raise PermissionError("privacy score pair selection differs")
    plan_path = selection_path.parent / "CONTROL_PLAN.json"
    if _input(spec, plan_path, "selected control plan") != selection[
            "files_sha256"]["CONTROL_PLAN.json"]:
        raise PermissionError("privacy score selected control plan bytes differ")
    plan = json.loads(plan_path.read_text())
    if (plan.get("schema") != "pcrl-focused-m3-control-plan-v1"
            or plan.get("bank_sha256") != bank.bank_sha256):
        raise PermissionError("privacy score selected control plan differs")
    pairs = sorted({row["pair_id"] for row in plan["rules"].values()
                    if row["status"] == "SELECTED"})
    if not pairs or not set(pairs) <= {pair.pair_id for pair in bank.pairs}:
        raise PermissionError("privacy score pair inventory is incomplete")
    attack_lock = _source_lock(lock, "attack_fit_source_lock_sha256", lock_sha)
    release_lock = _source_lock(lock, "release_source_lock_sha256", lock_sha)
    required_inputs = common_inputs | {str(selection_path), str(plan_path)}
    indices = {}
    for pair_id in pairs:
        release = root / "releases" / f"seed{args.seed}" / pair_id / "RELEASE_RECEIPT.json"
        required_inputs.update({str(release),
                                str(release.parent / "ATTACK_FIT_NEW.npz"),
                                str(release.parent / "INNER_SELECTION.npz")})
        release_sha = _input(spec, release, "complete release")
        receipt = json.loads(release.read_text())
        if (receipt.get("schema") != "pcrl-utility-conditional-complete-release-v1"
                or receipt.get("pair_id") != pair_id
                or receipt.get("seed") != args.seed
                or receipt.get("lock_sha256") != release_lock
                or receipt.get("gate_sha256") != admitted.gate_sha256):
            raise PermissionError("privacy score release source differs")
        for view in VIEWS:
            path = root / "attacks" / f"seed{args.seed}" / pair_id / view / "ATTACK_FIT_INDEX.json"
            required_inputs.update(attack_output_paths(path))
            attack_sha = _input(spec, path, "ATT reader")
            indices[(pair_id, view)] = (path, attack_sha, release_sha)
    if set(spec.get("inputs", {})) != required_inputs:
        raise PermissionError("privacy score sealed input inventory differs")
    for name in sorted(required_inputs):
        _input(spec, Path(name), "privacy score source")
    # Every selected reader and release has been sealed before INNER SEX opens.
    store = PrivacyRoleStore(args.export_root, args.manifest,
                             stage="r2_privacy_score")
    sex_inner = store.read("INNER_SELECTION", "sex")
    scores = {}
    for pair_id in pairs:
        scores[pair_id] = {}
        for view in VIEWS:
            path, attack_sha, release_sha = indices[(pair_id, view)]
            scores[pair_id][view] = score_saved_attack_view(
                path, attack_sha, sex_inner, private_root=root,
                seed=args.seed, pair_id=pair_id, view_name=view,
                attack_fit_lock_sha256=attack_lock,
                manifest_sha256=manifest_sha,
                release_index_sha256=release_sha)
    score_payload = {
        "schema": "pcrl-utility-seed-privacy-scores-v1",
        "seed": args.seed, "development_only": True,
        "bank_sha256": bank.bank_sha256,
        "full_input_gate_sha256": admitted.gate_sha256,
        "pair_selection_receipt_sha256": selection_sha,
        "pairs": scores,
        "selected_rule_pair_ids": {
            rule: row["pair_id"] if row["status"] == "SELECTED" else None
            for rule, row in plan["rules"].items()},
    }
    score_sha = _write_once_json(scores_path, score_payload)
    receipt = {
        "schema": "pcrl-utility-seed-privacy-score-receipt-v1",
        "status": "SCORED" if all(
            scores[pair][view]["controls"]["status"] == "PASS"
            for pair in pairs for view in VIEWS) else "CONTROL_FAILED",
        "seed": args.seed, "selected_pair_count": len(pairs),
        "complete_view_count": 3 * len(pairs),
        "score_file_sha256": score_sha,
        "pair_selection_receipt_sha256": selection_sha,
        "full_input_gate_sha256": admitted.gate_sha256,
        "conditional_lock_sha256": lock_sha,
        "attack_fit_source_lock_sha256": attack_lock,
        "release_source_lock_sha256": release_lock,
        "role_inputs": ["INNER_SELECTION:sex"],
        "development_only": True,
    }
    _write_once_json(output, receipt)


if __name__ == "__main__":
    main()
