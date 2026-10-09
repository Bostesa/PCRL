"""Data-free masked checkpoint and exact point-boundary replay checks."""

import json

import numpy as np
import torch

from results.pcrl_utility_reference_repair_v1.r3.mask_task_replay_worker import (
    _a_forward, _expected_output, _gate_source_lock, _network, _point,
)
import pytest
from results.pcrl_utility_reference_repair_v1.r3.source_reference import file_sha


def test_selected_input_a_head_restore_matches_synthetic_forward(tmp_path):
    root = tmp_path / ".private"
    root.mkdir()
    model = _network(3, 2)
    model.eval()
    x = np.arange(15, dtype=np.float32).reshape(5, 3) / 10
    with torch.no_grad():
        expected = torch.softmax(model(torch.from_numpy(x)).double(), dim=1).numpy()
    receipt = {"selected_columns": 3}
    model_id = "synthetic-mask-head"
    meta = {"schema": "team-c-head-checkpoint-v1", "purpose": "income",
            "model_id": model_id, "selected_atoms": ["age"],
            "selected_columns": 3, "spec": {"hidden_width": 64},
            "receipt": receipt, "state_keys": list(model.state_dict())}
    path = root / "head.npz"
    np.savez_compressed(path, meta=np.array(json.dumps(meta)),
                        **{f"state_{i}": value.detach().numpy()
                           for i, value in enumerate(model.state_dict().values())})
    entry = {"relative_path": "head.npz", "sha256": file_sha(path),
             "training_receipt": receipt}
    spec = {"inputs": {str(path): file_sha(path)}}
    actual = _a_forward(spec, root, entry, {"model_id": model_id},
                        {"atoms": ["age"], "column_indices": [0, 1, 2]},
                        "income", 2, x)
    assert np.array_equal(actual, expected)


def test_point_caps_are_inclusive():
    baseline = {"accuracy": 0.8, "nll": 0.5, "brier": 0.4}
    candidate = {"accuracy": 0.79, "nll": 0.51, "brier": 0.405}
    assert _point(candidate, baseline)["point_eligible"] is True
    candidate["brier"] += 1e-6
    assert _point(candidate, baseline)["point_eligible"] is False


def test_lead_and_remaining_bank_outputs_are_distinct_and_fixed(tmp_path):
    root = tmp_path / ".private"
    assert _expected_output(root, ["M_DROP_NATIVE_COUNTRY", "M_DROP_MARITAL_STATUS"]) == \
        root / "r3/LEAD_MASK_TASK_REPLAY.json"
    assert _expected_output(root, ["M_DROP_WORKCLASS", "M_DROP_RELATIONSHIP",
                                   "M_DROP_EDUCATION_FACT"]) == \
        root / "r3/BANK_MASK_TASK_REPLAY.json"
    with pytest.raises(PermissionError):
        _expected_output(root, ["M_DROP_WORKCLASS", "M_DROP_WORKCLASS"])
    with pytest.raises(PermissionError):
        _expected_output(root, ["M_DROP_WORKCLASS"])


def test_replay_gate_lock_lineage_is_cohort_specific():
    leads = {"M_DROP_NATIVE_COUNTRY", "M_DROP_MARITAL_STATUS"}
    bank = {"M_DROP_EDUCATION_FACT", "M_DROP_RELATIONSHIP",
            "M_DROP_WORKCLASS"}
    old = "a" * 64
    new = "b" * 64
    assert _gate_source_lock({"prior_lead_gate_lock_sha256": old}, leads) == old
    assert _gate_source_lock({"prior_mask_gate_lock_sha256": new}, bank) == new
    with pytest.raises(PermissionError):
        _gate_source_lock({"prior_lead_gate_lock_sha256": old}, bank)
    with pytest.raises(PermissionError):
        _gate_source_lock({"prior_lead_gate_lock_sha256": old,
                           "prior_mask_gate_lock_sha256": new}, leads)
