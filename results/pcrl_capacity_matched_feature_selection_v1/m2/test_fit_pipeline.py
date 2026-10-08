"""Data-free checks for the FIT-stage task/privacy access boundary."""

from pathlib import Path

import numpy as np
import pytest

from bank import build_shared_bank
from test_bank_joint import _records
from fit_pipeline import run_fit_pipeline


class _Release:
    def __init__(self, width):
        self.width = width

    def view(self, x, feature_names):
        return {"probabilities": np.full((len(x), self.width), 1 / self.width)}


class _Package:
    def __init__(self, purpose, mask_id, stage):
        self.model_id = f"{purpose}:{mask_id}:{stage}"
        self.restored_release = _Release(2 if purpose == "income" else 6)
        self.cache_key = "a" * 64
        self.head_sha256 = "b" * 64
        self.calibrator_sha256 = "c" * 64
        self.package_sha256 = "d" * 64
        self.receipt_sha256 = "e" * 64
        self.relative_package_path = f"heads/{purpose}/{mask_id}/{stage}.json"


def _data():
    x = np.zeros((60, 83), dtype=np.float32)
    rid = np.arange(60, dtype=np.int64)
    unit = rid.copy()
    income = rid % 2
    occupation = rid % 6
    refs = {name: np.full((60, width), 1 / width)
            for name, width in (("U0_1", 2), ("Ucal_1", 2),
                                ("U0_2", 6), ("Ucal_2", 6))}
    return x, rid, unit, income, occupation, refs


def test_task_only_receipt_is_published_before_sensitive_label_is_requested():
    bank = build_shared_bank(_records())
    x, rid, unit, income, occupation, refs = _data()
    events = []

    def fit(purpose, mask, stage, rows):
        events.append(("fit", stage, purpose, mask["mask_id"]))
        return _Package(purpose, mask["mask_id"], stage)

    def on_task(table, choice):
        assert not any(key == "local_search_auc" for row in
                       table["income_masks"] for key in row)
        assert choice["pair_id"] in {pair.pair_id for pair in bank.pairs}
        events.append(("task_receipt", table["task_table_sha256"]))

    def read_sex():
        assert events[-1][0] == "task_receipt"
        events.append(("read_sex",))
        return rid % 2

    def critic(columns, fit_rows, validation_rows, sex):
        assert events.count(("read_sex",)) == 1
        return 0.70

    result = run_fit_pipeline(
        bank=bank, seed=0, fit_x=x, fit_row_id=rid, fit_unit=unit,
        fit_income=income, fit_occupation=occupation, fit_references=refs,
        feature_names=tuple(f"c{i}" for i in range(83)),
        fit_package=fit, read_sex=read_sex, fit_raw_search_auc=critic,
        select_task_only=lambda table, bank: {
            "pair_id": bank.pairs[0].pair_id,
            "income_mask_id": bank.pairs[0].income_mask_id,
            "occupation_mask_id": bank.pairs[0].occupation_mask_id,
        },
        on_task_complete=on_task,
    )
    assert result.pair_table.critic_input_kind == "RAW_UNION_ONLY"
    assert len(result.pair_table.pairs) == 25
    assert len(result.final_packages["income"]) == 5
    assert len(result.final_packages["occupation_group"]) == 5
    assert set(result.full_f_packages) == {"income", "occupation_group"}
    assert events.index(("read_sex",)) > 0
    assert events[-1][0] == "fit" and events[-1][1] == "final"


def test_failed_task_receipt_prevents_sensitive_read():
    bank = build_shared_bank(_records())
    x, rid, unit, income, occupation, refs = _data()
    requested = []
    with pytest.raises(RuntimeError, match="write failure"):
        run_fit_pipeline(
            bank=bank, seed=0, fit_x=x, fit_row_id=rid, fit_unit=unit,
            fit_income=income, fit_occupation=occupation, fit_references=refs,
            feature_names=tuple(f"c{i}" for i in range(83)),
            fit_package=lambda p, m, s, r: _Package(p, m["mask_id"], s),
            read_sex=lambda: requested.append("sex"),
            fit_raw_search_auc=lambda *args: 0.7,
            select_task_only=lambda table, bank: {"pair_id": bank.pairs[0].pair_id},
            on_task_complete=lambda *args: (_ for _ in ()).throw(
                RuntimeError("write failure")),
        )
    assert requested == []


def test_m3_task_only_selector_consumes_m2_table_before_sex():
    from results.pcrl_capacity_matched_feature_selection_v1.m3.task_only import (
        select_task_only_pre_sensitive,
    )

    bank = build_shared_bank(_records())
    x, rid, unit, income, occupation, refs = _data()
    state = {"sealed": False, "selected": None}

    def task_sink(table, choice):
        assert choice == select_task_only_pre_sensitive(table)
        state["sealed"] = True
        state["selected"] = choice["pair_id"]

    def read_sex():
        assert state["sealed"] is True
        return rid % 2

    result = run_fit_pipeline(
        bank=bank, seed=0, fit_x=x, fit_row_id=rid, fit_unit=unit,
        fit_income=income, fit_occupation=occupation, fit_references=refs,
        feature_names=tuple(f"c{i}" for i in range(83)),
        fit_package=lambda p, m, s, r: _Package(p, m["mask_id"], s),
        read_sex=read_sex, fit_raw_search_auc=lambda *args: 0.7,
        select_task_only=lambda table, bank: select_task_only_pre_sensitive(table),
        on_task_complete=task_sink,
    )
    assert result.task_only_choice["pair_id"] == state["selected"]
    assert result.pair_table.pair(state["selected"]).pair_feasible


def test_no_utility_feasible_t_is_a_reportable_prefit_no_go():
    from results.pcrl_capacity_matched_feature_selection_v1.m3.task_only import (
        select_task_only_pre_sensitive,
    )

    bank = build_shared_bank(_records())
    x, rid, unit, income, occupation, refs = _data()
    for name, labels in (("U0_1", income), ("Ucal_1", income),
                         ("U0_2", occupation), ("Ucal_2", occupation)):
        width = 2 if name.endswith("1") else 6
        refs[name] = np.eye(width, dtype=np.float64)[labels]
    seen = []

    def read_sex():
        seen.append("forbidden_sex_read")
        raise AssertionError("NO_FEASIBLE_J must stop before SEX")

    result = run_fit_pipeline(
        bank=bank, seed=0, fit_x=x, fit_row_id=rid, fit_unit=unit,
        fit_income=income, fit_occupation=occupation, fit_references=refs,
        feature_names=tuple(f"c{i}" for i in range(83)),
        fit_package=lambda p, m, s, r: _Package(p, m["mask_id"], s),
        read_sex=read_sex, fit_raw_search_auc=lambda *args: 0.7,
        select_task_only=lambda table, bank: select_task_only_pre_sensitive(table),
        on_task_complete=lambda table, choice: seen.append(
            "task_no_go_published" if choice["status"] == "NO_UTILITY_FEASIBLE_T"
            else "unexpected"),
    )
    assert result.task_only_choice["pair_id"] is None
    assert result.pair_table is None
    assert result.final_packages == {}
    assert seen == ["task_no_go_published"]
