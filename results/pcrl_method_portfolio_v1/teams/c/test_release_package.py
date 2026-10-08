"""Synthetic deployment checks for the private head/calibrator binding."""

import hashlib
import json

import numpy as np
import pytest

from prototype import AdultGroupedSchema
from release_package import restore_release_package, save_release_package
from training import FittedTaskHead, HeadSpec, _network, save_head


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path, *, alter_scored_head_after_checkpoint=False):
    root = tmp_path / "private"
    root.mkdir()
    schema = AdultGroupedSchema()
    schema_sha = _sha("\n".join(schema.feature_names).encode())
    model = _network(1, 2, 64)
    head = FittedTaskHead(
        "income", "team-c-s19-income-test", ("age",), 1, HeadSpec(),
        {"feature_schema_sha256": schema_sha, "seed": 19,
         "selected_atoms": ("age",)}, model,
    )
    head_path = root / "head.npz"
    head_sha = save_head(head, head_path, private_root=root)
    if alter_scored_head_after_checkpoint:
        head.model[0].weight.data.add_(1.0)
    record = {
        "schema": "pcrl-portfolio-global-temp-v1", "task": "income",
        "arm_id": head.model_id, "model_seed": 19,
        "fit_role": "CALIBRATION_HELDOUT", "draw_id": "deterministic",
        "alpha": 1.0, "fit_row_unit_pair_sha256": "c" * 64,
    }
    cal_path = root / "calibrator.json"
    cal_path.write_text(json.dumps(record, sort_keys=True))
    cal_sha = _sha(cal_path.read_bytes())
    sources = {
        "fit_features": "1" * 64, "fit_label": "2" * 64,
        "calibration_features": "3" * 64, "calibration_label": "4" * 64,
    }
    package_path = root / "release-package.json"
    package_sha = save_release_package(
        head, head_path, head_sha, record, cal_path, cal_sha,
        package_path=package_path, private_root=root,
        campaign_lock_sha256="a" * 64, data_manifest_sha256="b" * 64,
        source_role_hashes=sources,
    )
    return root, package_path, package_sha, cal_path, sources


def _load_calibrator(path, expected_sha256):
    if _sha(path.read_bytes()) != expected_sha256:
        raise ValueError("calibrator file hash mismatch")
    return json.loads(path.read_text())


def _restore(root, path, package_sha, sources):
    return restore_release_package(
        path, expected_sha256=package_sha, private_root=root,
        expected_campaign_lock_sha256="a" * 64,
        expected_data_manifest_sha256="b" * 64,
        expected_source_role_hashes=sources,
        load_calibrator=_load_calibrator,
        apply_global_temperature=lambda probabilities, record: probabilities.copy(),
    )


def test_restored_release_uses_selected_head_and_bound_calibrator(tmp_path):
    root, path, package_sha, _, sources = _fixture(tmp_path)
    release = _restore(root, path, package_sha, sources)
    X = np.zeros((3, 83), dtype=np.float32)
    X[:, 0] = [20, 30, 40]
    view = release.view(X, AdultGroupedSchema.feature_names)
    assert view["feature_names"] == ("age",)
    np.testing.assert_array_equal(view["selected_values"][:, 0], X[:, 0])
    assert view["probabilities"].shape == (3, 2)
    assert view["head_id"].startswith("team-c-s19-income-test:")
    assert view["calibrator_id"].startswith("H-GLOBAL-TEMP:")


def test_restore_refuses_changed_or_missing_calibrator_and_wrong_role_binding(tmp_path):
    root, path, package_sha, cal_path, sources = _fixture(tmp_path)
    with pytest.raises(ValueError, match="source role hashes"):
        _restore(root, path, package_sha, {**sources, "fit_label": "f" * 64})
    cal_path.write_text(cal_path.read_text() + "\n")
    with pytest.raises(ValueError, match="calibrator.*hash"):
        _restore(root, path, package_sha, sources)
    cal_path.unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        _restore(root, path, package_sha, sources)


def test_package_refuses_scored_head_with_different_weights_than_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="saved head weights differ"):
        _fixture(tmp_path, alter_scored_head_after_checkpoint=True)
