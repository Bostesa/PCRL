"""Small source-identity checks; no reader fit or Adult role access."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "results/pcrl_method_portfolio_v1/teams/c"))

from prototype import AdultGroupedSchema, IdentityCalibrator, PurposeRelease, release_pair
from raw_union import canonical_raw_union
from views import canonical_union_from_source


class Head:
    def __init__(self, classes):
        self.classes = classes

    def predict_proba(self, x):
        return np.tile(np.full(self.classes, 1 / self.classes), (len(x), 1))


def test_raw_union_matches_deployed_canonical_source_hash():
    schema = AdultGroupedSchema()
    x = np.zeros((3, 83), dtype=np.float32)
    x[:, 0] = [0.0, -0.0, 1.0]
    first = PurposeRelease(schema, ("age", "relationship"), Head(2), "income",
                           "head-i", "cal-i", IdentityCalibrator())
    second = PurposeRelease(schema, ("capital-gain",), Head(6),
                            "occupation_group", "head-o", "cal-o", IdentityCalibrator())
    wire = release_pair(x, schema.feature_names, first, second)
    source = {"X": x, "feature_names": schema.feature_names,
              "row_id": np.array([1, 2, 3], dtype=np.int64),
              "unit": np.array([11, 12, 13], dtype=np.int64)}
    deployed = canonical_union_from_source(source, wire, schema)
    raw = canonical_raw_union(source, source_columns=deployed.source_columns,
                              feature_names=deployed.feature_names)
    assert raw.content_sha256 == deployed.content_sha256
    assert np.array_equal(raw.x, deployed.x)


def test_raw_union_rejects_float_identity_and_wrong_source_columns():
    schema = AdultGroupedSchema()
    source = {"X": np.zeros((2, 83), dtype=np.float32),
              "feature_names": schema.feature_names,
              "row_id": np.array([1, 2], dtype=np.int64),
              "unit": np.array([11, 12], dtype=np.int64)}
    with pytest.raises(ValueError, match="feature names"):
        canonical_raw_union(source, source_columns=(0,),
                            feature_names=(schema.feature_names[1],))
    source["row_id"] = np.array([1.2, 2.2])
    with pytest.raises(ValueError, match="int64"):
        canonical_raw_union(source, source_columns=(0,),
                            feature_names=(schema.feature_names[0],))
