"""Data-free checks for the conditional selected-input deployment boundary."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from results.pcrl_method_portfolio_v1.teams.c.prototype import AdultGroupedSchema
from results.pcrl_utility_reference_repair_v1.r2.mask_adapter import (
    FixedFillSelectedHead,
    fit_public_fill_constants,
)


class _UsesLastColumn:
    def predict_proba(self, full_input: np.ndarray) -> np.ndarray:
        assert full_input.shape[1] == 83
        positive = 1.0 / (1.0 + np.exp(-full_input[:, 82]))
        return np.column_stack((1.0 - positive, positive))


def _mask():
    schema = AdultGroupedSchema()
    atoms = tuple(atom for atom in schema.atoms if atom != "native-country")
    return schema, atoms, schema.column_indices(atoms)


def test_public_fill_uses_only_supplied_fit_features():
    fit = np.zeros((3, 83), dtype=np.float32)
    fit[:, 82] = np.array([1.0, 2.0, 6.0], dtype=np.float32)
    constants = fit_public_fill_constants(fit)
    assert constants.shape == (83,)
    assert constants.dtype == np.float32
    assert constants[82] == np.float32(3.0)
    fit[0, 82] = 90.0
    assert constants[82] == np.float32(3.0)


def test_fixed_fill_head_is_invariant_to_omitted_person_values():
    schema, atoms, columns = _mask()
    fit = np.zeros((3, 83), dtype=np.float32)
    fit[:, 82] = np.array([1.0, 2.0, 6.0], dtype=np.float32)
    constants = fit_public_fill_constants(fit)
    head = FixedFillSelectedHead(
        purpose="income", selected_atoms=atoms, source_indices=columns,
        full_input_head=_UsesLastColumn(), base_model_id="B-TEST",
        base_model_sha256="a" * 64,
        fill_constants=constants,
        source_schema_sha256=hashlib.sha256(
            "\n".join(schema.feature_names).encode()
        ).hexdigest(),
    )
    people = np.zeros((2, 83), dtype=np.float32)
    people[0, 82] = -500.0
    people[1, 82] = 500.0
    selected = people[:, columns]
    probability = head.predict_proba(selected)
    assert np.array_equal(probability[0], probability[1])
    assert np.isclose(probability[0, 1], 1 / (1 + np.exp(-3.0)))
    assert head.selected_columns == len(columns)
    assert head.model_id.startswith("B-TEST:FIXED-FILL:")


def test_adapter_refuses_wrong_mask_or_nonfinite_selected_values():
    schema, atoms, columns = _mask()
    constants = np.zeros(83, dtype=np.float32)
    source_sha = hashlib.sha256("\n".join(schema.feature_names).encode()).hexdigest()
    with pytest.raises(ValueError, match="mask columns"):
        FixedFillSelectedHead(
            purpose="income", selected_atoms=atoms, source_indices=columns[:-1],
            full_input_head=_UsesLastColumn(), base_model_id="B-TEST",
            base_model_sha256="a" * 64,
            fill_constants=constants, source_schema_sha256=source_sha,
        )
    head = FixedFillSelectedHead(
        purpose="income", selected_atoms=atoms, source_indices=columns,
        full_input_head=_UsesLastColumn(), base_model_id="B-TEST",
        base_model_sha256="a" * 64,
        fill_constants=constants, source_schema_sha256=source_sha,
    )
    selected = np.zeros((1, len(columns)), dtype=np.float32)
    selected[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite selected"):
        head.predict_proba(selected)


def test_adapter_identity_binds_the_underlying_weight_sha():
    schema, atoms, columns = _mask()
    source_sha = hashlib.sha256("\n".join(schema.feature_names).encode()).hexdigest()
    kwargs = dict(
        purpose="income", selected_atoms=atoms, source_indices=columns,
        full_input_head=_UsesLastColumn(), base_model_id="B-TEST",
        fill_constants=np.zeros(83, dtype=np.float32),
        source_schema_sha256=source_sha,
    )
    first = FixedFillSelectedHead(**kwargs, base_model_sha256="a" * 64)
    second = FixedFillSelectedHead(**kwargs, base_model_sha256="b" * 64)
    assert first.model_id != second.model_id
    assert first.base_model_sha256 == "a" * 64
