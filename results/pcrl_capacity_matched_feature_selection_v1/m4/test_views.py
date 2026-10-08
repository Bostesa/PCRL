"""Synthetic contracts for canonical raw unions and safe reader transport."""

from __future__ import annotations

import json
import sys
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "results/pcrl_method_portfolio_v1/teams/c"))
sys.path.insert(0, str(ROOT / "results/pcrl_method_portfolio_v1/shared"))

from prototype import AdultGroupedSchema, IdentityCalibrator, PurposeRelease, release_pair
from s2_bridge import to_s2_attack_views
from attack_eval import AttackView
from safe_data_guard import sha_arr

from views import (build_complete_attack_views, canonical_union_from_source,
                   transport_raw_reader)


class BinaryHead:
    def predict_proba(self, selected):
        p = np.clip(0.2 + 0.1 * np.asarray(selected)[:, 0], 0.05, 0.95)
        return np.column_stack((1.0 - p, p))


class SixHead:
    def predict_proba(self, selected):
        return np.tile(np.asarray([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]), (len(selected), 1))


class LinearRawReader:
    def predict_proba(self, x):
        p = np.clip(0.2 + 0.05 * np.asarray(x)[:, 0], 0.05, 0.95)
        return np.column_stack((1.0 - p, p))


def fixture(first=("age", "relationship"), second=("capital-gain",)):
    schema = AdultGroupedSchema()
    x = np.zeros((4, 83), dtype=np.float64)
    x[:, 0] = [0, 1, 2, 3]
    x[:, 2] = [2, 3, 4, 5]
    x[:, schema.atoms["relationship"][0]] = [1, 0, 1, 0]
    row_id = np.asarray([101, 102, 103, 104], dtype=np.int64)
    unit = np.asarray([21, 22, 23, 24], dtype=np.int64)
    a = PurposeRelease(schema, first, BinaryHead(), "income", "head-i", "cal-i",
                       IdentityCalibrator())
    b = PurposeRelease(schema, second, SixHead(), "occupation_group", "head-o", "cal-o",
                       IdentityCalibrator())
    release = release_pair(x, schema.feature_names, a, b)
    source = {"X": x, "feature_names": schema.feature_names,
              "row_id": row_id, "unit": unit}
    return schema, source, release


def public_releases(schema, release):
    return {
        "income": PurposeRelease(
            schema, tuple(release["income"]["selected_groups"]), BinaryHead(),
            "income", "head-i", "cal-i", IdentityCalibrator(),
        ),
        "occupation_group": PurposeRelease(
            schema, tuple(release["occupation_group"]["selected_groups"]), SixHead(),
            "occupation_group", "head-o", "cal-o", IdentityCalibrator(),
        ),
    }


def custody(tmp_path, union, model, prediction, *, input_kind="RAW_ONLY"):
    private = tmp_path / ".private"
    private.mkdir()
    path = private / "raw_reader.joblib"
    joblib.dump(model, path)
    record = {
        "schema": "pcrl-focused-raw-reader-transport-v1",
        "reader_id": "RAW_LR_C1", "input_kind": input_kind,
        "source_union_sha256": union.content_sha256,
        "source_columns": list(union.source_columns),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_artifact_name": path.name,
        "source_prediction_sha256": sha_arr(np.asarray(prediction, dtype=np.float64)),
        "attack_split_sha256": "a" * 64,
    }
    receipt = private / "raw_reader_receipt.json"
    receipt.write_text(json.dumps(record, sort_keys=True) + "\n")
    return path, receipt, hashlib.sha256(receipt.read_bytes()).hexdigest()


def test_canonical_union_is_original_column_order_and_row_exact():
    schema, source, release = fixture()
    union = canonical_union_from_source(source, release, schema)
    expected = schema.column_indices(("age", "capital-gain", "relationship"))
    assert union.source_columns == expected
    assert union.feature_names == tuple(schema.feature_names[j] for j in expected)
    assert np.array_equal(union.x, source["X"][:, expected])
    assert union.x.flags.c_contiguous
    assert np.array_equal(union.row_id, source["row_id"])


def test_canonical_union_rejects_coalition_or_local_value_not_in_source():
    schema, source, release = fixture()
    release["coalition"]["selected_values"][0, 0] = 9
    with pytest.raises(ValueError, match="coalition.*source"):
        canonical_union_from_source(source, release, schema)
    schema, source, release = fixture()
    release["income"]["selected_values"][0, 0] = 9
    with pytest.raises(ValueError, match="income.*source"):
        canonical_union_from_source(source, release, schema)


@pytest.mark.parametrize("wire_name", ["income", "coalition"])
def test_canonical_union_rejects_signed_zero_wire_marker(wire_name):
    schema, source, release = fixture()
    assert source["X"][0, 0] == 0 and not np.signbit(source["X"][0, 0])
    release[wire_name]["selected_values"][0, 0] = -0.0
    with pytest.raises(ValueError, match=f"{wire_name}.*source"):
        canonical_union_from_source(source, release, schema)


def test_transport_uses_identical_source_columns_not_equal_width(tmp_path):
    schema, source, release = fixture(first=("age",), second=("capital-gain",))
    first = canonical_union_from_source(source, release, schema)
    model = LinearRawReader()
    prediction = model.predict_proba(first.x)[:, 1]
    other_schema, other_source, other_release = fixture(
        first=("capital-loss",), second=("hours-per-week",))
    other = canonical_union_from_source(other_source, other_release, other_schema)
    assert first.x.shape[1] == other.x.shape[1]
    path, receipt, receipt_sha = custody(tmp_path, first, model, prediction)
    with pytest.raises(ValueError, match="source columns"):
        transport_raw_reader(first, other, prediction, receipt, receipt_sha, path,
                             expected_attack_split_sha256="a" * 64)


def test_transport_same_union_with_different_allocation_keeps_predictions(tmp_path):
    schema, source, release = fixture(first=("age",), second=("capital-gain",))
    first = canonical_union_from_source(source, release, schema)
    schema2, source2, release2 = fixture(first=("capital-gain",), second=("age",))
    second = canonical_union_from_source(source2, release2, schema2)
    model = LinearRawReader()
    prediction = model.predict_proba(first.x)[:, 1]
    path, receipt, receipt_sha = custody(tmp_path, first, model, prediction)
    carried = transport_raw_reader(first, second, prediction, receipt, receipt_sha, path,
                                   expected_attack_split_sha256="a" * 64)
    assert np.array_equal(carried.prediction, prediction)
    assert carried.max_abs_replay_delta == 0
    assert first.source_columns == second.source_columns


def test_transport_rejects_changed_people_or_unreplayable_predictions(tmp_path):
    schema, source, release = fixture(first=("age",), second=("capital-gain",))
    first = canonical_union_from_source(source, release, schema)
    model = LinearRawReader()
    prediction = model.predict_proba(first.x)[:, 1]
    path, receipt, receipt_sha = custody(tmp_path, first, model, prediction)
    source["row_id"] = source["row_id"][::-1]
    other = canonical_union_from_source(source, release, schema)
    with pytest.raises(ValueError, match="ordered people"):
        transport_raw_reader(first, other, prediction, receipt, receipt_sha, path,
                             expected_attack_split_sha256="a" * 64)
    with pytest.raises(ValueError, match="prediction replay"):
        changed_path = tmp_path / ".private" / "changed_receipt.json"
        changed_record = json.loads(receipt.read_text())
        changed_record["source_prediction_sha256"] = sha_arr(prediction + 1e-6)
        changed_path.write_text(json.dumps(changed_record, sort_keys=True) + "\n")
        changed_sha = hashlib.sha256(changed_path.read_bytes()).hexdigest()
        transport_raw_reader(first, first, prediction + 1e-6, changed_path,
                             changed_sha, path,
                             expected_attack_split_sha256="a" * 64)


def test_transport_refuses_complete_view_model_even_when_width_matches(tmp_path):
    schema, source, release = fixture(first=("age",), second=("capital-gain",))
    union = canonical_union_from_source(source, release, schema)
    model = LinearRawReader()
    prediction = model.predict_proba(union.x)[:, 1]
    path, receipt, receipt_sha = custody(tmp_path, union, model, prediction,
                                         input_kind="COMPLETE_VIEW")
    with pytest.raises(ValueError, match="RAW_ONLY"):
        transport_raw_reader(union, union, prediction, receipt, receipt_sha, path,
                             expected_attack_split_sha256="a" * 64)


def test_complete_s2_views_cover_wire_and_use_canonical_contiguous_raw_columns():
    schema, source, release = fixture()
    views, union = build_complete_attack_views(
        source, release, public_releases(schema, release), schema,
        bridge=to_s2_attack_views, attack_view_type=AttackView,
    )
    assert set(views) == {"income", "occupation_group", "coalition"}
    assert np.array_equal(views["coalition"].x_raw_only, union.x)
    assert views["coalition"].x_raw_only.flags.c_contiguous
    assert views["coalition"].x.flags.c_contiguous
    assert views["coalition"].x_defense_aware.flags.c_contiguous
    assert len(views["coalition"].wire_components) == len(release["coalition"])


def test_complete_view_rejects_clean_output_bypass():
    schema, source, release = fixture()
    release["income"]["probabilities"] = np.tile([0.9, 0.1], (4, 1))
    release["income"]["decisions"] = np.zeros(4, dtype=np.int64)
    release["coalition"]["income_probabilities"] = release["income"]["probabilities"].copy()
    release["coalition"]["income_decisions"] = release["income"]["decisions"].copy()
    with pytest.raises(ValueError, match="recomputation"):
        build_complete_attack_views(
            source, release, public_releases(schema, release), schema,
            bridge=to_s2_attack_views, attack_view_type=AttackView,
        )
