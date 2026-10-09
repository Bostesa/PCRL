"""Data-free checks for independent frozen-model privacy replay custody."""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r3.privacy_replay_worker import (
    CORE, _reforward_core_models, _sex,
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
