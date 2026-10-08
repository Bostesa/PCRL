"""Data-free deployment checks for the locked Team B screen worker."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bottleneck import schema_fingerprint
from control_artifacts import file_sha256
from screen_driver import (_save_attempt_index, _verify_sequential_deployment,
                           _write_json)
from test_attack_bridge import _model
from train_bottleneck import FrozenPurposeView


@pytest.mark.parametrize("arm,purpose", [("B-SEQ12", "income"),
                                        ("B-SEQ21", "occupation")])
def test_final_sequential_draw_equals_frozen_first_and_rejects_change(tmp_path, arm, purpose):
    model = _model()
    schema_sha = schema_fingerprint(model.feature_names)
    data = SimpleNamespace(
        feature_names=model.feature_names,
        feature_schema_sha256=schema_sha,
        x=torch.zeros((3, 83), dtype=torch.float32),
        row_ids=("1", "2", "3"),
        row_order_sha256="a" * 64,
        group_order_sha256="b" * 64,
    )
    secret = b"s" * 32
    frozen = FrozenPurposeView.from_model(
        model, data, purpose=purpose, source_id="synthetic-source",
        first_checkpoint_sha256="c" * 64,
        sampler_plan_sha256="d" * 64, sampler_secret=secret)
    attempt = tmp_path / ".private" / "attempt"
    (attempt / "fit").mkdir(parents=True, mode=0o700)
    frozen_path = attempt / "fit" / "first_fixed_draw.pt"
    frozen.save(frozen_path)
    fitted = SimpleNamespace(
        arm=arm, model=model,
        stage_receipts=({}, {"frozen_view_sha256": frozen.sha256}),
        private_artifact_sha256={"first_fixed_draw_file": file_sha256(frozen_path)},
    )
    receipt = _verify_sequential_deployment(
        fitted, data, secret=secret, source_id="synthetic-source", attempt=attempt)
    assert receipt["checked"] and receipt["first_purpose_forward_passes"] == 1
    changed = copy.deepcopy(model)
    with torch.no_grad():
        next(changed.encoders[purpose].parameters()).add_(0.1)
    fitted.model = changed
    with pytest.raises(ValueError, match="differs from fixed"):
        _verify_sequential_deployment(
            fitted, data, secret=secret, source_id="synthetic-source", attempt=attempt)


def test_private_attempt_index_binds_releases_readers_and_endpoint_arrays(tmp_path):
    attempt = tmp_path / ".private" / "attempt"
    attempt.mkdir(parents=True, mode=0o700)
    model = _model()
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        (attempt / f"release__{role}.json").write_text(role)
    cal_hashes = {}
    for purpose in ("income", "occupation"):
        folder = attempt / "calibration"
        folder.mkdir(exist_ok=True)
        path = folder / f"{purpose}.json"
        path.write_text(purpose)
        cal_hashes[purpose] = file_sha256(path)
    reader_receipts = {}
    for view in ("income", "occupation", "pair"):
        folder = attempt / f"readers__{view}"
        folder.mkdir()
        path = folder / "READERS_RECEIPT.json"
        path.write_text(view)
        reader_receipts[view] = {"private_receipt_sha256": file_sha256(path)}
    endpoint_npz = attempt / "ENDPOINT_INPUTS.npz"
    endpoint_npz.write_bytes(b"synthetic-private-arrays")
    _write_json(attempt / "ENDPOINT_INPUTS.json",
                {"npz_sha256": file_sha256(endpoint_npz)})
    _write_json(attempt / "AGGREGATE.json", {"status": "EXPLORATORY_DEVELOPMENT"})
    releases = {role: file_sha256(attempt / f"release__{role}.json")
                for role in ("ATTACK_FIT_NEW", "INNER_SELECTION")}
    fitted = SimpleNamespace(model=model, private_artifact_sha256={"stage": "a" * 64})
    kwargs = dict(
        attempt=attempt, arm="B-JVIB", seed=0,
        screen_unit_id="B-SCREEN-B-JVIB-S0", source_id="b" * 64,
        manifest_sha256="c" * 64, campaign_lock_sha256="d" * 64,
        screen_config_sha256="e" * 64, fitted=fitted,
        calibration=SimpleNamespace(record_file_sha256=cal_hashes),
        release_sha256=releases, attack_receipts=reader_receipts,
        endpoint_receipt_sha256=file_sha256(attempt / "ENDPOINT_INPUTS.json"),
    )
    digest = _save_attempt_index(**kwargs)
    assert digest == file_sha256(attempt / "ATTEMPT_INDEX.json")
    (attempt / "release__INNER_SELECTION.json").write_text("tampered")
    with pytest.raises(ValueError, match="Saved release changed"):
        _save_attempt_index(**kwargs)
