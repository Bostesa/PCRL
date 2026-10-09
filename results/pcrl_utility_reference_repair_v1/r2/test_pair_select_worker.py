"""Synthetic proof that FIT SEX is the last role opened after frozen T."""

from __future__ import annotations

import json

import numpy as np

from results.pcrl_utility_reference_repair_v1.r2 import pair_select_worker as worker
from results.pcrl_utility_reference_repair_v1.r2.test_pair_selection import _inputs


class _Store:
    def __init__(self):
        self.calls = []
        n = 120
        row = np.arange(n, dtype=np.int64)
        self.common = {"row_id": row, "unit": row.copy()}
        self.x = np.zeros((n, 83), dtype=np.float32)

    def read(self, role, field):
        assert role == "OSF_DEFENSE_FIT"
        self.calls.append(field)
        if field == "features":
            return {**self.common, "X": self.x,
                    "feature_names": np.array([f"x{i}" for i in range(83)])}
        if field == "income":
            return {**self.common,
                    "income": np.arange(120, dtype=np.int64) % 2}
        if field == "occupation":
            return {**self.common,
                    "occupation": np.arange(120, dtype=np.int64) % 6}
        if field == "sex":
            assert self.calls == ["features", "income", "occupation", "sex"]
            return {**self.common,
                    "sex": np.arange(120, dtype=np.int64) % 2}
        raise AssertionError(field)


def test_pair_worker_opens_sex_after_task_table_and_binds_all_controls(tmp_path,
                                                                        monkeypatch):
    bank, task_table, t_choice, references, local, unions = _inputs()
    private = tmp_path / ".private"
    private.mkdir()
    first = bank.masks[0]
    gate_path = private / "mask_gate" / first.mask_id / "seed0" / "MASK_GATE_RECEIPT.json"
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text(json.dumps({
        "full_input_gate_sha256": "a" * 64,
        "metrics": {reference: {task: references[task][reference]
                                for task in worker.TASKS}
                    for reference in worker.REFERENCES},
    }))
    task_receipt = {"mask_gate_receipt_sha256_by_id": {
        first.mask_id: worker.sha_file(gate_path)}}

    def fake_search(bank_arg, x, sex, fit, val, seed):
        assert len(fit) + len(val) == len(x) == len(sex)
        return local, unions, {"fitted_reader_count": 22}

    monkeypatch.setattr(worker, "fit_fixed_raw_search", fake_search)
    output = private / "pair_select/seed0/PAIR_SELECT_RECEIPT.json"
    store = _Store()
    receipt = worker.select_pair_unit(
        store, bank, 0, private, output, task_receipt=task_receipt,
        task_table=task_table, t_choice=t_choice,
        task_receipt_sha256="b" * 64,
        gate_sha256="a" * 64, lock_sha256="c" * 64)
    assert store.calls == ["features", "income", "occupation", "sex"]
    assert receipt["fitted_raw_search_readers"] == 22
    assert receipt["selected_rule_count"] == sum(
        item["status"] == "SELECTED" for item in
        json.loads((output.parent / "CONTROL_PLAN.json").read_text())["rules"].values())
    assert set(receipt["files_sha256"]) == {
        "PAIR_TABLE.json", "JOINT_SELECTION.json", "CONTROL_PLAN.json",
        "RAW_SEARCH_RECEIPT.json"}
