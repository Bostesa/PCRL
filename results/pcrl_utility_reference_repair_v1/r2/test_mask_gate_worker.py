"""Small sealed-gate arithmetic fixture with no Adult archive or SEX label."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r2 import mask_gate_worker as gate
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_method_portfolio_v1.teams.c.prototype import AdultGroupedSchema


ROOT = Path(__file__).resolve().parents[3]
MASK_BANK = ROOT / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
MASK_SHA = "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c"


class _RoleStore:
    def __init__(self, arrays):
        self.arrays = arrays
        self.calls = []

    def read(self, role, field):
        self.calls.append((role, field))
        assert role == "INNER_SELECTION"
        if field == "features":
            return {k: self.arrays[k] for k in ("row_id", "unit", "X", "feature_names")}
        if field == "income":
            return {k: self.arrays[k] for k in ("row_id", "unit", "income")}
        if field == "occupation":
            return {k: self.arrays[k] for k in ("row_id", "unit", "occupation")}
        if field == "references_s0":
            return {k: self.arrays[k] for k in (
                "row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2",
                "d_1", "d_2")}
        raise AssertionError("forbidden role field: " + field)


class _Release:
    def __init__(self, task, mask):
        self.task = task
        self.mask = mask

    def view(self, x83, names):
        width = 2 if self.task == "income" else 6
        return {"selected_values": x83[:, self.mask.column_indices].copy(),
                "probabilities": np.full((len(x83), width), 1 / width),
                "decisions": np.zeros(len(x83), dtype=np.int64),
                "mask_id": self.mask.mask_id}


def _fixture(tmp_path):
    private = tmp_path / ".private"
    private.mkdir()
    n = 24
    row = np.arange(100, 100 + n, dtype=np.int64)
    income = np.arange(n, dtype=np.int64) % 2
    occupation = np.arange(n, dtype=np.int64) % 6
    p1 = np.full((n, 2), 0.5, dtype=np.float64)
    p2 = np.full((n, 6), 1 / 6, dtype=np.float64)
    arrays = {
        "row_id": row, "unit": row.copy(),
        "income": income, "occupation": occupation,
        "X": np.zeros((n, 83), dtype=np.float32),
        "feature_names": np.asarray(AdultGroupedSchema().feature_names),
        "F_income": p1, "F_occupation_group": p2,
        "U0_1": p1, "U0_2": p2,
        "Ucal_1": p1, "Ucal_2": p2,
        "d_1": np.zeros(n, dtype=np.int64),
        "d_2": np.zeros(n, dtype=np.int64),
    }
    folder = private / "r1/gate/seed_0"
    folder.mkdir(parents=True)
    f_path = folder / "GATE_PREDICTIONS.npz"
    np.savez_compressed(f_path, **{key: value for key, value in arrays.items()
                                   if key not in ("X", "feature_names", "d_1", "d_2")})
    common_sha = gate.array_sha(np.column_stack((row, row)))
    f_receipt = {
        "schema": "pcrl-utility-full-input-gate-result-v1", "seed": 0,
        "selected_recipe": "A", "inner_row_unit_sha256": common_sha,
        "predictions_relative_path": str(f_path.relative_to(private)),
        "predictions_sha256": gate.sha_file(f_path),
    }
    f_receipt_path = folder / "GATE_RECEIPT.json"
    f_receipt_path.write_text(json.dumps(f_receipt))
    mask = load_frozen_bank(MASK_BANK, expected_sha256=MASK_SHA).mask(
        "M_DROP_NATIVE_COUNTRY")
    output = private / "mask_gate" / mask.mask_id / "seed0" / "MASK_GATE_RECEIPT.json"
    return private, arrays, mask, output, f_receipt_path, common_sha


def test_mask_gate_matches_all_three_references_without_sex(tmp_path, monkeypatch):
    private, arrays, mask, output, f_receipt, common_sha = _fixture(tmp_path)
    monkeypatch.setattr(gate, "restore_mask_task_release",
                        lambda root, record, actual_mask, task, seed, recipe,
                        **kwargs: _Release(task, actual_mask))
    store = _RoleStore(arrays)
    receipt = gate.gate_mask_unit(
        store, mask, 0, "A", private, output,
        fit_receipt_sha256="a" * 64,
        f_gate_receipt_sha256=gate.sha_file(f_receipt),
        gate_sha256="b" * 64, lock_sha256="c" * 64,
        common_row_unit_sha256=common_sha)
    assert receipt["all_task_reference_points_eligible"] is True
    assert set(receipt["point_eligibility"]["income"]["references"]) == {
        "U0", "Ucal", "F"}
    assert store.calls == [("INNER_SELECTION", field) for field in (
        "features", "income", "occupation", "references_s0")]
    assert (output.parent / "MASK_GATE_PREDICTIONS.npz").exists()


def test_gate_prediction_reader_rejects_extra_member_before_load(tmp_path):
    private, arrays, mask, output, f_receipt, common_sha = _fixture(tmp_path)
    path = private / "r1/gate/seed_0/GATE_PREDICTIONS.npz"
    with np.load(path, allow_pickle=False) as z:
        copies = {key: np.array(z[key], copy=True) for key in z.files}
    copies["SEX"] = np.zeros(len(arrays["row_id"]), dtype=np.int64)
    np.savez_compressed(path, **copies)
    with pytest.raises(PermissionError, match="members differ"):
        gate._f_predictions(path, gate.sha_file(path), 0, common_sha)
