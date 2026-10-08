"""Synthetic complete-wire custody; no Adult role reads."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "results/pcrl_method_portfolio_v1/teams/c"))
sys.path.insert(0, str(ROOT / "results/pcrl_method_portfolio_v1/shared"))
sys.path.insert(0, str(ROOT / "results/pcrl_capacity_matched_feature_selection_v1/m2"))

from attack_eval import AttackView
from bank import build_shared_bank
from prototype import AdultGroupedSchema, IdentityCalibrator, PurposeRelease
from release import release_selected_pair
from results.pcrl_capacity_matched_feature_selection_v1.m1.capacity import capacity_geometry
from s2_bridge import to_s2_attack_views
from release_bundle import restore_release_role, save_release_bundle


class Head:
    def __init__(self, classes):
        self.classes = classes

    def predict_proba(self, x):
        scores = np.full((len(x), self.classes), 1 / self.classes)
        return scores


def test_selected_release_custody_roundtrip_binds_source_and_wire(tmp_path):
    schema = AdultGroupedSchema()
    bank = build_shared_bank(capacity_geometry(schema.feature_names)["masks"])
    pair = bank.pairs[0]
    first = bank.mask(pair.income_mask_id)
    second = bank.mask(pair.occupation_mask_id)
    releases = {
        "income": PurposeRelease(schema, first.atoms, Head(2), "income",
                                 "head-i", "cal-i", IdentityCalibrator()),
        "occupation_group": PurposeRelease(
            schema, second.atoms, Head(6), "occupation_group",
            "head-o", "cal-o", IdentityCalibrator()),
    }
    sources = {}
    results = {}
    for role, first_id in (("ATTACK_FIT_NEW", 1), ("INNER_SELECTION", 101)):
        x = np.zeros((3, 83), dtype=np.float32)
        x[:, 0] = [0, -0.0, 1]
        source = {"X": x, "feature_names": schema.feature_names,
                  "row_id": np.arange(first_id, first_id + 3, dtype=np.int64),
                  "unit": np.arange(first_id + 1000, first_id + 1003,
                                    dtype=np.int64)}
        sources[role] = source
        results[role] = release_selected_pair(
            x, schema.feature_names, source["row_id"], source["unit"],
            role, pair, bank, releases["income"], releases["occupation_group"],
        )
    private = tmp_path / ".private"
    private.mkdir()
    output = private / "release"
    package_shas = {"income": "a" * 64, "occupation_group": "b" * 64}
    feature_shas = {"ATTACK_FIT_NEW": "c" * 64,
                    "INNER_SELECTION": "d" * 64}
    save_release_bundle(
        role_sources=sources, release_results=results,
        public_releases=releases, pair=pair, bank=bank, schema=schema,
        bridge=to_s2_attack_views, attack_view_type=AttackView,
        private_root=private, output_dir=output, seed=0,
        lock_sha256="e" * 64, manifest_sha256="f" * 64,
        role_feature_sha256=feature_shas,
        package_sha256_by_purpose=package_shas,
    )
    index_path = output / "RELEASE_RECEIPT.json"
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    for role in sources:
        views, union = restore_release_role(
            index_path=index_path, expected_index_sha256=index_sha,
            role=role, source_record=sources[role],
            public_releases=releases, pair=pair, bank=bank, schema=schema,
            bridge=to_s2_attack_views, attack_view_type=AttackView,
            private_root=private, lock_sha256="e" * 64,
            manifest_sha256="f" * 64,
            role_feature_sha256=feature_shas[role],
        )
        assert set(views) == {"income", "occupation_group", "coalition"}
        assert np.array_equal(views["coalition"].x_raw_only, union.x)
