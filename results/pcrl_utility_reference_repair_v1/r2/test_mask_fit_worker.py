"""Synthetic custody and access-order checks; no Adult role archive is opened."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r2 import mask_fit_worker as worker
from results.pcrl_utility_reference_repair_v1.r2 import mask_train
from results.pcrl_utility_reference_repair_v1.r2 import release_loader
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank


ROOT = Path(__file__).resolve().parents[3]
MASK_BANK = ROOT / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
MASK_SHA = "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c"


class _Store:
    def __init__(self, done):
        self.done = done
        self.manifest = {"source_schema_sha256": hashlib.sha256(
            "\n".join(mask_train.AdultGroupedSchema().feature_names).encode()
        ).hexdigest()}
        self.calls = []

    def read(self, role, field):
        self.calls.append((role, field))
        if role == "CALIBRATION_HELDOUT" and not self.done[0]:
            raise AssertionError("CAL was read before final head custody")
        n = 120 if role == "OSF_DEFENSE_FIT" else 24
        row = np.arange(n, dtype=np.int64) + (0 if n == 120 else 1000)
        common = {"row_id": row, "unit": row.copy()}
        if field == "features":
            x = np.zeros((n, 83), dtype=np.float32)
            x[:, 82] = np.arange(n, dtype=np.float32)
            return {**common, "X": x, "feature_names": np.array(
                mask_train.AdultGroupedSchema().feature_names)}
        if field == "income":
            return {**common, "income": np.arange(n, dtype=np.int64) % 2}
        if field == "occupation":
            return {**common, "occupation": np.arange(n, dtype=np.int64) % 6}
        raise AssertionError(field)


class _Head:
    def __init__(self, task, stage):
        self.purpose = task
        self.model_id = f"synthetic:{task}:{stage}"
        self.receipt = {"best_epoch": 2, "model_id": self.model_id}

    def predict_proba(self, selected):
        width = 2 if self.purpose == "income" else 6
        return np.full((len(selected), width), 1 / width, dtype=np.float64)


def test_A_mask_fit_custody_and_calibration_open_order(tmp_path, monkeypatch):
    mask = load_frozen_bank(MASK_BANK, expected_sha256=MASK_SHA).mask(
        "M_DROP_NATIVE_COUNTRY")
    private = tmp_path / ".private"
    private.mkdir()
    output = private / "masks" / mask.mask_id / "seed0" / "MASK_FIT_RECEIPT.json"
    done = [False]
    store = _Store(done)

    def fake_fit(recipe, plan, yi, yo, group, *, seed, feature_names):
        assert recipe == "A" and seed == 0
        assert np.all(plan.full_matrix("outer", "validation")[:, 82] ==
                      plan.fill_constants["outer"][82])
        return mask_train.MaskFitCoreResult(
            final_kind="A", mask_id=mask.mask_id, seed=seed,
            outer_models=(_Head("income", "outer"),
                          _Head("occupation_group", "outer")),
            final_models=(_Head("income", "final"),
                          _Head("occupation_group", "final")),
            outer_raw_metrics={"income": {"nll": 0.7},
                               "occupation_group": {"nll": 1.8}},
            selected_C=None, C_search=None, inner_encoder=None,
            outer_split_receipt=plan.outer_split_receipt,
            inner_split_receipt=plan.inner_split_receipt,
        )

    def fake_save(head, path, *, private_root):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(head.model_id.encode())
        if path.name == "final_occupation_group_head.npz":
            done[0] = True
        return worker.sha_file(path)

    monkeypatch.setattr(worker, "fit_mask_task_only", fake_fit)
    monkeypatch.setattr(worker, "save_head", fake_save)
    receipt = worker.fit_mask_unit(
        store, mask, 0, "A", private, output,
        gate_sha256="a" * 64, lock_sha256="b" * 64,
        source_lock_sha256="c" * 64)
    expected = set(worker.expected_output_paths("A", private, mask.mask_id, 0))
    actual = {path for path in output.parent.rglob("*") if path.is_file()}
    assert actual == expected
    assert done[0]
    assert receipt["task_packages"]["income"]["sha256"] == worker.sha_file(
        output.parent / "FINAL_income_PACKAGE.json")
    assert store.calls[:3] == [
        ("OSF_DEFENSE_FIT", "features"),
        ("OSF_DEFENSE_FIT", "income"),
        ("OSF_DEFENSE_FIT", "occupation"),
    ]
    assert store.calls[3:] == [
        ("CALIBRATION_HELDOUT", "features"),
        ("CALIBRATION_HELDOUT", "income"),
        ("CALIBRATION_HELDOUT", "occupation"),
    ]
    package = json.loads((output.parent / "FINAL_income_PACKAGE.json").read_text())
    assert package["source_indices"] == list(mask.column_indices)
    assert package["model"]["sha256"] == worker.sha_file(
        output.parent / "final_income_head.npz")
    class RestoredHead(_Head):
        def __init__(self):
            super().__init__("income", "final")
            self.model_id = package["model_id"]
            self.selected_atoms = mask.atoms
            self.selected_columns = len(mask.column_indices)

    monkeypatch.setattr(release_loader, "restore_head",
                        lambda path, *, expected_sha256, private_root: RestoredHead())
    fit_record = {"relative_path": str(output.relative_to(private)),
                  "sha256": worker.sha_file(output)}
    restored = release_loader.restore_mask_task_release(
        private, fit_record, mask, "income", 0, "A",
        gate_sha256="a" * 64, conditional_lock_sha256="b" * 64)
    view = restored.view(np.zeros((3, 83), dtype=np.float32),
                         tuple(mask_train.AdultGroupedSchema().feature_names))
    assert view["selected_values"].shape == (3, 42)
    assert view["mask_id"] == mask.mask_id
    assert view["task_package_sha256"] == receipt["task_packages"]["income"]["sha256"]
    (output.parent / "final_income_head.npz").write_text("changed")
    with pytest.raises(PermissionError, match="A masked checkpoint"):
        release_loader.restore_mask_task_release(
            private, fit_record, mask, "income", 0, "A",
            gate_sha256="a" * 64, conditional_lock_sha256="b" * 64)


def test_recipe_specific_output_inventories_are_closed(tmp_path):
    root = tmp_path / ".private"
    a = worker.expected_output_paths("A", root, "M_DROP_NATIVE_COUNTRY", 0)
    b = worker.expected_output_paths("B", root, "M_DROP_NATIVE_COUNTRY", 0)
    assert len(a) == 13
    assert len(b) == 16
    assert set(a) != set(b)
    assert b[-1].name == "MASK_FIT_RECEIPT.json"
