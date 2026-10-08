"""Synthetic full-wire deployment checks; no Adult role values or labels."""

from __future__ import annotations

import numpy as np
import pytest

from results.pcrl_method_portfolio_v1.teams.c.prototype import (
    AdultGroupedSchema, IdentityCalibrator, PurposeRelease,
)
from results.pcrl_capacity_matched_feature_selection_v1.m1.capacity import (
    capacity_geometry,
)

from bank import build_shared_bank
import release as release_module
from release import release_selected_pair


class SyntheticHead:
    def __init__(self, classes: int):
        self.classes = classes

    def predict_proba(self, selected: np.ndarray) -> np.ndarray:
        base = np.abs(selected[:, 0].astype(np.float64)) + 1.0
        logits = np.repeat(base[:, None], self.classes, axis=1)
        logits[:, 0] += 0.2
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)


def _fixture():
    schema = AdultGroupedSchema()
    bank = build_shared_bank(capacity_geometry(schema.feature_names)["masks"])
    first = next(mask for mask in bank.masks if "relationship" not in mask.atoms)
    second = first
    pair = next(pair for pair in bank.pairs
                if pair.income_mask_id == first.mask_id
                and pair.occupation_mask_id == second.mask_id)
    income = PurposeRelease(
        schema, first.atoms, SyntheticHead(2), "income", "head-income",
        "cal-income", IdentityCalibrator(),
    )
    occupation = PurposeRelease(
        schema, second.atoms, SyntheticHead(6), "occupation_group",
        "head-occupation", "cal-occupation", IdentityCalibrator(),
    )
    X = np.arange(4 * 83, dtype=np.float32).reshape(4, 83) / 100.0
    row_id = np.array((10, 11, 12, 13), dtype=np.int64)
    unit = np.array((20, 21, 22, 23), dtype=np.int64)
    return schema, bank, pair, income, occupation, X, row_id, unit


def test_restored_wire_exposes_exact_raw_union_and_no_ids():
    schema, bank, pair, income, occupation, X, row_id, unit = _fixture()
    result = release_selected_pair(
        X, schema.feature_names, row_id, unit, "INNER_SELECTION", pair,
        bank, income, occupation,
    )
    wire = result["wire"]
    assert np.array_equal(wire["income"]["selected_values"],
                          X[:, bank.mask(pair.income_mask_id).column_indices])
    assert np.array_equal(wire["coalition"]["selected_values"],
                          X[:, pair.union_columns])
    assert set(result["private_alignment"]) == {
        "income", "occupation_group", "coalition",
    }
    assert all("row_id" not in local and "unit" not in local
               for local in wire.values())
    assert result["receipt"]["union_sha256"] == pair.union_sha256
    assert result["receipt"]["role"] == "INNER_SELECTION"


def test_withheld_column_perturbation_cannot_change_wire():
    schema, bank, pair, income, occupation, X, row_id, unit = _fixture()
    original = release_selected_pair(X, schema.feature_names, row_id, unit,
                                     "INNER_SELECTION", pair, bank,
                                     income, occupation)["wire"]
    modified = X.copy()
    withheld = sorted(set(range(83)) - set(pair.union_columns))
    assert withheld
    modified[:, withheld] += 999.0
    later = release_selected_pair(modified, schema.feature_names, row_id,
                                  unit, "INNER_SELECTION", pair, bank,
                                  income, occupation)["wire"]
    for name in ("income", "occupation_group", "coalition"):
        for field in ("selected_values", "probabilities", "decisions") if name != "coalition" else ("selected_values", "income_probabilities", "occupation_group_probabilities", "income_decisions", "occupation_group_decisions"):
            assert np.array_equal(original[name][field], later[name][field])


def test_mask_and_source_schema_mismatch_refuse_before_release():
    schema, bank, pair, income, occupation, X, row_id, unit = _fixture()
    wrong = next(mask for mask in bank.masks if mask.mask_id != pair.income_mask_id)
    wrong_income = PurposeRelease(
        schema, wrong.atoms, SyntheticHead(2), "income", "head-income-wrong",
        "cal-income", IdentityCalibrator(),
    )
    with pytest.raises(ValueError, match="selected mask"):
        release_selected_pair(X, schema.feature_names, row_id, unit,
                              "INNER_SELECTION", pair, bank,
                              wrong_income, occupation)
    with pytest.raises(ValueError, match="83-column schema"):
        release_selected_pair(X, schema.feature_names + ("sex",), row_id,
                              unit, "INNER_SELECTION", pair, bank,
                              income, occupation)


def test_signed_zero_wire_marker_is_rejected(monkeypatch):
    schema, bank, pair, income, occupation, X, row_id, unit = _fixture()
    original = release_module.release_pair

    def marked(*args):
        wire = original(*args)
        assert wire["income"]["selected_values"][0, 0] == 0.0
        wire["income"]["selected_values"][0, 0] = -0.0
        return wire

    monkeypatch.setattr(release_module, "release_pair", marked)
    with pytest.raises(ValueError, match="raw source values"):
        release_selected_pair(X, schema.feature_names, row_id, unit,
                              "INNER_SELECTION", pair, bank,
                              income, occupation)
