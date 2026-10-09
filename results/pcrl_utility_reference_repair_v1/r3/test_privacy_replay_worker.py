"""Data-free checks for independent frozen-model privacy replay custody."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r3.privacy_replay_worker import (
    CORE, _expected_output, _reforward_core_models, _semantic_bank_sha, _sex,
)
from results.pcrl_utility_reference_repair_v1.r3.task_replay import array_sha


class ToyProbabilityModel:
    def predict_proba(self, x):
        p = 1 / (1 + np.exp(-np.asarray(x, dtype=np.float64).sum(axis=1)))
        return np.column_stack((1 - p, p))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_core_models_are_reforwarded_on_frozen_tune_and_inner_views(tmp_path):
    folder = tmp_path / "core"
    folder.mkdir()
    att = {"x": np.array([[0.0], [1.0], [-1.0]]),
           "x_defense_aware": np.array([[0.1], [1.1], [-0.9]])}
    inner = {"x": np.array([[0.2], [-0.2]]),
             "x_defense_aware": np.array([[0.3], [-0.1]])}
    tune_idx = np.array([0, 2])
    spec = {"inputs": {}}
    tune, predictions = {}, {}
    for name in CORE:
        model = ToyProbabilityModel()
        path = folder / f"model__{name}.joblib"
        joblib.dump(model, path)
        spec["inputs"][str(path)] = _sha(path)
        key = "x_defense_aware" if name == "DA_MLP_64x32" else "x"
        tune[name] = model.predict_proba(att[key][tune_idx])[:, 1]
        predictions[name] = model.predict_proba(inner[key])[:, 1]
    assert _reforward_core_models(spec, tmp_path, att, inner, tune_idx,
                                  tune, predictions) == 0
    predictions["HGB_15x200"][0] += 1e-5
    with pytest.raises(PermissionError, match="does not reproduce"):
        _reforward_core_models(spec, tmp_path, att, inner, tune_idx,
                               tune, predictions)


def test_sex_role_requires_manifest_person_and_group_identity(tmp_path):
    export = tmp_path / "export"
    path = export / "INNER_SELECTION" / "sex.npz"
    path.parent.mkdir(parents=True)
    row = np.array([1, 2, 3, 4], dtype=np.int64)
    unit = np.array([10, 11, 12, 12], dtype=np.int64)
    sex = np.array([0, 1, 1, 0], dtype=np.int64)
    np.savez_compressed(path, row_id=row, unit=unit, sex=sex)
    manifest = {"role_file_sha256": {"INNER_SELECTION:sex": _sha(path)},
                "roles": {"INNER_SELECTION": {
                    "rows": 4, "groups": 3,
                    "row_unit_pair_sha256": array_sha(np.column_stack((row, unit)))}}}
    spec = {"inputs": {str(path): _sha(path)}}
    assert np.array_equal(_sex(spec, export, manifest, "INNER_SELECTION")["sex"],
                          sex)
    manifest["roles"]["INNER_SELECTION"]["groups"] = 4
    with pytest.raises(PermissionError, match="person/label geometry"):
        _sex(spec, export, manifest, "INNER_SELECTION")


def test_semantic_bank_sha_is_not_public_bank_file_sha(tmp_path):
    public = tmp_path / "MASK_BANK.json"
    public.write_text('{"public geometry": true}\n')
    semantic_sha = "7" * 64
    assert semantic_sha != _sha(public)
    receipt = tmp_path / "pair_select/seed0/PAIR_SELECT_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({
        "schema": "pcrl-utility-pair-selection-receipt-v1",
        "seed": 0, "full_input_gate_sha256": "a" * 64,
        "bank_sha256": semantic_sha,
    }) + "\n")
    spec = {"inputs": {str(receipt): _sha(receipt)}}
    audit = {"by_seed": {"0": {"pair_receipt_sha256": _sha(receipt)}}}
    assert _semantic_bank_sha(spec, tmp_path, 0, audit, "a" * 64) == semantic_sha
    audit["by_seed"]["0"]["pair_receipt_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="independent audit"):
        _semantic_bank_sha(spec, tmp_path, 0, audit, "a" * 64)


def test_retry_has_distinct_write_once_output(tmp_path):
    assert _expected_output(tmp_path, "R3_PRIVACY_REPLAY") != _expected_output(
        tmp_path, "R3_PRIVACY_REPLAY_RETRY")
    assert _expected_output(tmp_path, "R3_PRIVACY_REPLAY_RETRY").name == (
        "PRIVACY_REPLAY_RETRY.json")
    with pytest.raises(PermissionError, match="unit identity"):
        _expected_output(tmp_path, "R3_PRIVACY_REPLAY_UNSEALED")
