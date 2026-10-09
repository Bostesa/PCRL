"""Data-free proof that the sealed score unit inventories all opened bytes."""

import hashlib
import json
from pathlib import Path

import pytest

from results.pcrl_utility_reference_repair_v1.r2 import privacy_score_worker


def _put(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _fixture(tmp_path):
    root = tmp_path / ".private"
    export = tmp_path / "roles"
    gate_path = tmp_path / "FULL_INPUT_GATE.json"
    manifest_path = tmp_path / "DATA_ACCESS_MANIFEST.json"
    bank_path = tmp_path / "MASK_BANK.json"
    sex_path = export / "INNER_SELECTION" / "sex.npz"
    inputs = {}
    for path in (manifest_path, bank_path, sex_path):
        inputs[str(path)] = _put(path, path.name.encode())
    gate = {"selected_recipe": "A", "f_receipts": {}, "f_packages": {},
            "r1_gate_receipt_sha256_by_seed": {},
            "r1_gate_prediction_sha256_by_seed": {}}
    for seed in ("0", "1"):
        receipt_path = root / "r1" / "gate" / f"seed_{seed}" / "GATE_RECEIPT.json"
        prediction_path = receipt_path.with_name("GATE_PREDICTIONS.npz")
        inputs[str(receipt_path)] = _put(receipt_path, b"{}")
        inputs[str(prediction_path)] = _put(prediction_path, b"predictions")
        gate["r1_gate_receipt_sha256_by_seed"][seed] = inputs[str(receipt_path)]
        gate["r1_gate_prediction_sha256_by_seed"][seed] = inputs[str(prediction_path)]
        f_receipt = root / "r1" / "full" / f"seed_{seed}" / "F_PACKAGE_RECEIPT.json"
        inputs[str(f_receipt)] = _put(f_receipt, b"{}")
        gate["f_receipts"][seed] = {
            "relative_path": str(f_receipt.relative_to(root)),
            "sha256": inputs[str(f_receipt)]}
        gate["f_packages"][seed] = {}
        for task in ("income", "occupation_group"):
            directory = f_receipt.parent
            model = directory / f"A_{task}_head.npz"
            cal = directory / f"{task}_calibrator.json"
            package = directory / f"F_{task}_PACKAGE.json"
            inputs[str(model)] = _put(model, b"model")
            inputs[str(cal)] = _put(cal, b"{}")
            content = {"model": {"relative_path": str(model.relative_to(root)),
                                 "sha256": inputs[str(model)]},
                       "calibrator": {"relative_path": str(cal.relative_to(root)),
                                      "sha256": inputs[str(cal)]}}
            inputs[str(package)] = _put(package, json.dumps(content).encode())
            gate["f_packages"][seed][task] = {
                "relative_path": str(package.relative_to(root)),
                "sha256": inputs[str(package)]}
    inputs[str(gate_path)] = _put(gate_path, json.dumps(gate).encode())
    return ({"inputs": inputs}, gate_path, manifest_path, bank_path,
            export, root, sex_path)


def test_common_inputs_include_inner_sex_and_every_deep_f_component(tmp_path):
    assert hasattr(privacy_score_worker, "sealed_common_input_paths")
    spec, gate, manifest, bank, export, root, sex = _fixture(tmp_path)
    paths = privacy_score_worker.sealed_common_input_paths(
        spec, gate, manifest, bank, export, root)
    assert paths == set(spec["inputs"])
    assert len(paths) == 22
    assert str(sex) in paths
    assert len([name for name in paths if name.endswith("_head.npz")]) == 4


def test_common_inputs_fail_when_declared_inner_sex_is_missing(tmp_path):
    spec, gate, manifest, bank, export, root, sex = _fixture(tmp_path)
    del spec["inputs"][str(sex)]
    with pytest.raises(PermissionError, match="INNER SEX"):
        privacy_score_worker.sealed_common_input_paths(
            spec, gate, manifest, bank, export, root)


def test_common_inputs_fail_on_mutated_f_model_bytes(tmp_path):
    spec, gate, manifest, bank, export, root, _ = _fixture(tmp_path)
    model = next(Path(name) for name in spec["inputs"] if name.endswith("_head.npz"))
    model.write_bytes(b"altered")
    with pytest.raises(PermissionError, match="F model"):
        privacy_score_worker.sealed_common_input_paths(
            spec, gate, manifest, bank, export, root)
