"""Data-free orchestration checks before any Adult mask fit is admitted."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.mask_inputs import build_mask_input_plan
from results.pcrl_utility_reference_repair_v1.r2 import mask_train


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
SOURCE_SHA = "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c"


def _fixture():
    n = 120
    x = np.zeros((n, 83), dtype=np.float32)
    x[:, 0] = np.arange(n, dtype=np.float32)
    x[:, 82] = np.arange(n, dtype=np.float32) + 10
    yi = np.arange(n, dtype=np.int64) % 2
    yo = np.arange(n, dtype=np.int64) % 6
    unit = np.arange(n, dtype=np.int64)
    mask = load_frozen_bank(SOURCE, expected_sha256=SOURCE_SHA).mask(
        "M_DROP_NATIVE_COUNTRY")
    return x, yi, yo, unit, mask


class _UniformAHead:
    def __init__(self, purpose):
        self.purpose = purpose

    def predict_proba(self, selected):
        width = 2 if self.purpose == "income" else 6
        return np.full((len(selected), width), 1 / width)


def test_A_has_two_scoped_head_fits_per_task_and_raw_outer_score(monkeypatch):
    x, yi, yo, unit, mask = _fixture()
    plan = build_mask_input_plan(x, yi, yo, unit, mask, seed=0)
    calls = []

    def fake_fit(X, y, groups, names, selected_atoms, *, purpose, model_id, seed, spec):
        calls.append((purpose, len(X), X[:, 82].copy(), tuple(selected_atoms)))
        assert len(names) == 83 and len(y) == len(X) == len(groups)
        return _UniformAHead(purpose)

    monkeypatch.setattr(mask_train, "fit_task_head", fake_fit)
    result = mask_train.fit_mask_task_only(
        "A", plan, yi, yo, unit, seed=0, feature_names=tuple(
            mask_train.AdultGroupedSchema().feature_names))
    assert len(calls) == 4
    assert [item[0] for item in calls] == [
        "income", "occupation_group", "income", "occupation_group"]
    assert [item[1] for item in calls] == [len(plan.outer_fit)] * 2 + [len(x)] * 2
    assert all(np.all(item[2] == item[2][0]) for item in calls)
    assert result.outer_raw_metrics["income"]["nll"] > 0
    assert result.final_kind == "A"


def test_B_fits_inner_outer_final_and_reuses_frozen_C(monkeypatch):
    x, yi, yo, unit, mask = _fixture()
    plan = build_mask_input_plan(x, yi, yo, unit, mask, seed=0)
    encoder_rows = []
    fitted_C = []

    def fake_encoder(xx, income, occupation, seed):
        encoder_rows.append((len(xx), xx[:, 82].copy()))
        return {"rows": len(xx)}

    def fake_select(model, xfit, yifit, yofit, xval, yival, yoval):
        assert model["rows"] == len(xfit)
        assert len(xval) == len(plan.inner_val)
        return {"income": {"selected_C": 0.1, "validation_nll": 1.0,
                            "grid": []},
                "occupation_group": {"selected_C": 1.0,
                                     "validation_nll": 2.0, "grid": []}}

    def fake_heads(model, xx, income, occupation, c_by_task):
        fitted_C.append(dict(c_by_task))
        assert model["rows"] == len(xx)
        return ("income_head", "occupation_head")

    def fake_predict(model, heads, xx):
        return (np.full((len(xx), 2), 0.5),
                np.full((len(xx), 6), 1 / 6))

    monkeypatch.setattr(mask_train, "train_source_encoder", fake_encoder)
    monkeypatch.setattr(mask_train, "select_source_head_C", fake_select)
    monkeypatch.setattr(mask_train, "fit_source_heads", fake_heads)
    monkeypatch.setattr(mask_train, "predict_source", fake_predict)
    result = mask_train.fit_mask_task_only(
        "B", plan, yi, yo, unit, seed=0, feature_names=tuple(
            mask_train.AdultGroupedSchema().feature_names))
    assert [n for n, _ in encoder_rows] == [len(plan.inner_fit),
                                           len(plan.outer_fit), len(x)]
    assert all(np.all(values == values[0]) for _, values in encoder_rows)
    assert fitted_C == [{"income": 0.1, "occupation_group": 1.0}] * 2
    assert result.selected_C == {"income": 0.1, "occupation_group": 1.0}
    assert result.final_kind == "B"

