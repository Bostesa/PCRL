"""Data-free checks for the conditional release's complete attacker view."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "results/pcrl_method_portfolio_v1/teams/c"))
from prototype import AdultGroupedSchema, IdentityCalibrator, PurposeRelease  # noqa: E402

from results.pcrl_utility_reference_repair_v1.r2.complete_views import (
    build_conditional_attack_views, release_conditional_pair,
)
from results.pcrl_utility_reference_repair_v1.r2.release_loader import BoundPurposeRelease


class BinaryHead:
    def predict_proba(self, selected):
        p = np.clip(0.2 + 0.1 * np.asarray(selected)[:, 0], .05, .95)
        return np.column_stack((1 - p, p))


class SixHead:
    def predict_proba(self, selected):
        return np.tile(np.array([.5, .1, .1, .1, .1, .1]), (len(selected), 1))


def fixture():
    schema = AdultGroupedSchema()
    x = np.zeros((4, 83), dtype=np.float32)
    x[:, 0] = [0, 1, 2, 3]
    x[:, schema.atoms["relationship"][0]] = [1, 0, 1, 0]
    source = {"X": x, "feature_names": schema.feature_names,
              "row_id": np.array([101, 102, 103, 104], dtype=np.int64),
              "unit": np.array([21, 22, 23, 24], dtype=np.int64)}
    first = BoundPurposeRelease(PurposeRelease(
        schema, ("age", "relationship"), BinaryHead(), "income", "head-i",
        "cal-i", IdentityCalibrator()), "MASK-I", "a" * 64, "b" * 64)
    second = BoundPurposeRelease(PurposeRelease(
        schema, ("capital-gain",), SixHead(), "occupation_group", "head-o",
        "cal-o", IdentityCalibrator()), "MASK-O", "c" * 64, "d" * 64)
    public = {"income": first, "occupation_group": second}
    wire = release_conditional_pair(source, public)
    return source, public, wire


def test_complete_view_covers_all_transmitted_fields_and_raw_values():
    source, public, wire = fixture()
    views, union = build_conditional_attack_views(source, wire, public)
    assert set(views) == {"income", "occupation_group", "coalition"}
    assert set(views["income"].wire_components) == set(wire["income"])
    assert set(views["coalition"].wire_components) == set(wire["coalition"])
    assert np.array_equal(views["coalition"].x_raw_only, union.x)
    width = wire["income"]["selected_values"].shape[1]
    assert np.array_equal(views["income"].x[:, :width],
                          wire["income"]["selected_values"])
    assert ("task_package_sha256", '"' + "b" * 64 + '"') in views["income"].wire_constants
    assert views["income"].x_defense_aware.shape[1] == views["income"].x.shape[1] + 2


def test_complete_view_rejects_clean_output_bypass():
    source, public, wire = fixture()
    wire["income"]["probabilities"] = np.tile([.9, .1], (4, 1))
    wire["income"]["decisions"] = np.zeros(4, dtype=np.int64)
    wire["coalition"]["income_probabilities"] = wire["income"]["probabilities"].copy()
    wire["coalition"]["income_decisions"] = wire["income"]["decisions"].copy()
    with pytest.raises(ValueError, match="replay"):
        build_conditional_attack_views(source, wire, public)


def test_complete_view_rejects_missing_package_identity_and_changed_raw_value():
    source, public, wire = fixture()
    del wire["income"]["task_package_sha256"]
    with pytest.raises(ValueError, match="wire fields"):
        build_conditional_attack_views(source, wire, public)
    source, public, wire = fixture()
    wire["coalition"]["selected_values"][0, 0] = 5
    with pytest.raises(ValueError, match="coalition.*source"):
        build_conditional_attack_views(source, wire, public)


def test_complete_view_rejects_package_or_head_swap():
    source, public, wire = fixture()
    wire["occupation_group"]["task_package_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="package"):
        build_conditional_attack_views(source, wire, public)
    source, public, wire = fixture()
    wire["income"]["head_id"] = "other"
    with pytest.raises(ValueError, match="head"):
        build_conditional_attack_views(source, wire, public)
