"""Data-free masking across B's inner, outer and final FIT stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank
from results.pcrl_utility_reference_repair_v1.r2.mask_inputs import (
    build_mask_input_plan,
    mask_calibration_matrix,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
SOURCE_SHA = "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c"


def _fixture():
    n = 120
    x = np.zeros((n, 83), dtype=np.float32)
    x[:, 0] = np.arange(n, dtype=np.float32)
    x[:, 82] = np.arange(n, dtype=np.float32) + 10
    cal = np.zeros((24, 83), dtype=np.float32)
    cal[:, 82] = 9999.0
    income = np.arange(n, dtype=np.int64) % 2
    occupation = np.arange(n, dtype=np.int64) % 6
    groups = np.arange(n, dtype=np.int64)
    bank = load_frozen_bank(SOURCE, expected_sha256=SOURCE_SHA)
    return x, cal, income, occupation, groups, bank.mask("M_DROP_NATIVE_COUNTRY")


def test_each_stage_uses_its_own_fit_only_constant_and_no_person_omission():
    x, cal, income, occupation, groups, mask = _fixture()
    plan = build_mask_input_plan(x, income, occupation, groups, mask, seed=0)
    assert plan.selected_fit.shape == (120, 42)
    assert not set(plan.outer_fit).intersection(plan.outer_val)
    assert set(plan.inner_fit).issubset(set(plan.outer_fit))
    assert set(plan.inner_val).issubset(set(plan.outer_fit))
    for stage, subset in (("inner", "train"), ("inner", "validation"),
                          ("outer", "fit"), ("outer", "validation"),
                          ("final", "fit")):
        reconstructed = plan.full_matrix(stage, subset)
        assert reconstructed.shape[1] == 83
        assert np.all(reconstructed[:, 82] == plan.fill_constants[stage][82])
    assert plan.fill_constants["inner"][82] == np.mean(x[plan.inner_fit, 82])
    assert plan.fill_constants["outer"][82] == np.mean(x[plan.outer_fit, 82])
    assert plan.fill_constants["final"][82] == np.mean(x[:, 82])
    final_cal = mask_calibration_matrix(cal, plan)
    assert final_cal.shape == cal.shape
    assert np.all(final_cal[:, 82] != 9999.0)


def test_mask_plan_rejects_label_or_extra_predictor_columns():
    x, cal, income, occupation, groups, mask = _fixture()
    with pytest.raises(ValueError, match="83-column"):
        build_mask_input_plan(np.column_stack((x, income)).astype(np.float32),
                              income, occupation, groups, mask, seed=0)
    plan = build_mask_input_plan(x, income, occupation, groups, mask, seed=0)
    with pytest.raises(ValueError, match="calibration"):
        mask_calibration_matrix(cal[:, :-1], plan)
