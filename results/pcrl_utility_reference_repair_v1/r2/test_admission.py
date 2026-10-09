"""No-role-read admission checks for R2's conditional branch."""

from __future__ import annotations

import hashlib
import json

import pytest

from results.pcrl_utility_reference_repair_v1.r2.admission import (
    admit_conditional_gate,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path, recipe="A"):
    private = tmp_path / ".private"
    private.mkdir()
    packages = {}
    receipts = {}
    r1_gate_shas = {}
    for seed in (0, 1):
        packages[str(seed)] = {}
        components = {}
        for task in ("income", "occupation_group"):
            relative = f"r1/full/seed_{seed}/F_{task}_PACKAGE.json"
            path = private / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            model_path = private / f"r1/model_{seed}_{task}.npz"
            model_data = f"{seed}:{task}:weights".encode()
            if recipe == "A":
                model_path.write_bytes(model_data)
                model_record = {
                    "relative_path": str(model_path.relative_to(private)),
                    "sha256": _sha(model_data),
                }
            else:
                bundle = private / f"r1/bundle_{seed}"
                bundle.mkdir(exist_ok=True)
                hashes = {}
                for filename in ("encoder.pt", "head_0.joblib", "head_1.joblib"):
                    payload = f"{seed}:joint:weights".encode() + filename.encode()
                    (bundle / filename).write_bytes(payload)
                    hashes[filename] = _sha(payload)
                model_record = {
                    "relative_directory": str(bundle.relative_to(private)),
                    "sha256": hashes,
                }
            cal_path = private / f"r1/cal_{seed}_{task}.json"
            cal_data = f"{seed}:{task}:calibration".encode()
            cal_path.write_bytes(cal_data)
            manifest = {
                "schema": "pcrl-utility-full-F-task-package-v1",
                "seed": seed, "task": task, "selected_recipe": recipe,
                "selection_lock_sha256": "f" * 64,
                "model": model_record,
                "calibrator": {"relative_path": str(cal_path.relative_to(private)),
                               "sha256": _sha(cal_data)},
                "feature_schema_sha256": "c" * 64,
                "fit_row_unit_sha256": "d" * 64,
                "calibration_row_unit_sha256": "e" * 64,
            }
            content = (json.dumps(manifest, sort_keys=True) + "\n").encode()
            path.write_bytes(content)
            packages[str(seed)][task] = {
                "relative_path": relative, "sha256": _sha(content),
            }
            components[task if recipe == "A" else "B_source_bundle"] = model_record
            components[f"calibrator_{task}"] = manifest["calibrator"]
        receipt_path = private / f"r1/full/seed_{seed}/F_PACKAGE_RECEIPT.json"
        receipt = {
            "schema": "pcrl-utility-full-F-package-v1",
            "seed": seed, "selected_recipe": recipe,
            "selection_lock_sha256": "f" * 64,
            "fit_row_unit_sha256": "d" * 64,
            "calibration_row_unit_sha256": "e" * 64,
            "packages": components, "task_packages": packages[str(seed)],
        }
        receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode()
        receipt_path.write_bytes(receipt_bytes)
        receipts[str(seed)] = {
            "relative_path": str(receipt_path.relative_to(private)),
            "sha256": _sha(receipt_bytes),
        }
        gate_folder = private / f"r1/gate/seed_{seed}"
        gate_folder.mkdir(parents=True)
        prediction_path = gate_folder / "GATE_PREDICTIONS.npz"
        prediction_bytes = f"synthetic-gate-predictions-{seed}".encode()
        prediction_path.write_bytes(prediction_bytes)
        r1_gate_path = gate_folder / "GATE_RECEIPT.json"
        r1_gate = {
            "schema": "pcrl-utility-full-input-gate-result-v1",
            "seed": seed, "selected_recipe": recipe,
            "inner_row_unit_sha256": "b" * 64,
            "predictions_relative_path": str(prediction_path.relative_to(private)),
            "predictions_sha256": _sha(prediction_bytes),
        }
        r1_gate_bytes = (json.dumps(r1_gate, sort_keys=True) + "\n").encode()
        r1_gate_path.write_bytes(r1_gate_bytes)
        r1_gate_shas[str(seed)] = _sha(r1_gate_bytes)
    gate = {
        "schema": "pcrl-utility-full-input-gate-v1",
        "status": "PASS_FULL_INPUT_POINT_GATE",
        "selected_recipe": recipe,
        "common_role": "INNER_SELECTION",
        "common_row_unit_pair_sha256": "b" * 64,
        "source_science_lock_sha256": "a" * 64,
        "per_seed_task": {
            str(seed): {
                task: {"U0": True, "Ucal": True}
                for task in ("income", "occupation_group")
            } for seed in (0, 1)
        },
        "f_packages": packages,
        "f_receipts": receipts,
        "r1_gate_receipt_sha256_by_seed": r1_gate_shas,
    }
    path = tmp_path / "FULL_INPUT_GATE.json"
    raw = (json.dumps(gate, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return path, private, gate, _sha(raw)


def test_conditional_gate_admits_only_a_complete_hashed_f_receipt(tmp_path):
    gate_path, private, _, gate_sha = _fixture(tmp_path)
    admitted = admit_conditional_gate(
        gate_path, expected_gate_sha256=gate_sha,
        expected_source_lock_sha256="a" * 64,
        expected_common_row_unit_pair_sha256="b" * 64,
        private_root=private,
    )
    assert admitted.selected_recipe == "A"
    assert len(admitted.f_package_paths) == 4
    assert len(admitted.r1_gate_receipt_paths) == 2
    assert admitted.gate_sha256 == gate_sha


def test_conditional_gate_rejects_failed_task_and_changed_package(tmp_path):
    gate_path, private, gate, _ = _fixture(tmp_path)
    gate["per_seed_task"]["1"]["occupation_group"]["Ucal"] = False
    raw = (json.dumps(gate, sort_keys=True) + "\n").encode()
    gate_path.write_bytes(raw)
    with pytest.raises(PermissionError, match="point eligibility"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256=_sha(raw),
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )
    gate["per_seed_task"]["1"]["occupation_group"]["Ucal"] = True
    raw = (json.dumps(gate, sort_keys=True) + "\n").encode()
    gate_path.write_bytes(raw)
    (private / "r1/full/seed_0/F_income_PACKAGE.json").write_text("changed")
    with pytest.raises(PermissionError, match="F package"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256=_sha(raw),
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )


def test_conditional_gate_rejects_escape_and_wrong_receipt_sha(tmp_path):
    gate_path, private, gate, gate_sha = _fixture(tmp_path)
    with pytest.raises(PermissionError, match="gate SHA"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256="c" * 64,
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )
    gate["f_packages"]["0"]["income"]["relative_path"] = "../outside"
    raw = (json.dumps(gate, sort_keys=True) + "\n").encode()
    gate_path.write_bytes(raw)
    with pytest.raises(PermissionError, match="receipt/task identity|private root"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256=_sha(raw),
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )


def test_conditional_gate_rehashes_model_component_inside_f_manifest(tmp_path):
    gate_path, private, _, gate_sha = _fixture(tmp_path)
    (private / "r1/model_0_income.npz").write_text("mutated weights")
    with pytest.raises(PermissionError, match="F model component"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256=gate_sha,
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )


def test_conditional_gate_rehashes_all_B_bundle_files(tmp_path):
    gate_path, private, _, gate_sha = _fixture(tmp_path, recipe="B")
    assert admit_conditional_gate(
        gate_path, expected_gate_sha256=gate_sha,
        expected_source_lock_sha256="a" * 64,
        expected_common_row_unit_pair_sha256="b" * 64,
        private_root=private,
    ).selected_recipe == "B"
    (private / "r1/bundle_0/head_1.joblib").write_text("mutated")
    with pytest.raises(PermissionError, match="F model component"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256=gate_sha,
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )


def test_conditional_gate_rehashes_f_receipt_before_task_arrays(tmp_path):
    gate_path, private, _, gate_sha = _fixture(tmp_path)
    (private / "r1/full/seed_1/F_PACKAGE_RECEIPT.json").write_text("changed")
    with pytest.raises(PermissionError, match="F package receipt"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256=gate_sha,
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )


def test_conditional_gate_rehashes_r1_common_prediction_bytes(tmp_path):
    gate_path, private, _, gate_sha = _fixture(tmp_path)
    (private / "r1/gate/seed_1/GATE_PREDICTIONS.npz").write_text("changed")
    with pytest.raises(PermissionError, match="R1 common-role gate prediction"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256=gate_sha,
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )
