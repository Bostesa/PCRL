"""Synthetic exact-wire custody replay through the actual S2 AttackView."""

import numpy as np
import pytest

from test_s2_bridge import S2AttackView, alignment, public_releases, sample_views
from s2_bridge import to_s2_attack_views
from view_custody import restore_view_bundle, save_view_bundle


def test_private_view_bundle_replays_all_wire_arrays_and_reader_matrices(tmp_path):
    release = sample_views()
    packages = public_releases()
    original = to_s2_attack_views(release, alignment(), packages, S2AttackView)
    private = tmp_path / ".private"
    package_hashes = {"income": "a" * 64, "occupation_group": "b" * 64}
    receipt_path, checksum = save_view_bundle(
        release, original, npz_path=private / "view.npz",
        private_root=private, role="ATTACK_FIT_NEW", arm_id="C-SIMPLE-REL0",
        seed=2, campaign_lock_sha256="c" * 64,
        data_manifest_sha256="d" * 64,
        role_feature_sha256="e" * 64,
        package_sha256_by_purpose=package_hashes,
    )
    restored, restored_wire = restore_view_bundle(
        receipt_path, expected_receipt_sha256=checksum,
        private_root=private, expected_campaign_lock_sha256="c" * 64,
        expected_data_manifest_sha256="d" * 64,
        expected_role="ATTACK_FIT_NEW", expected_arm_id="C-SIMPLE-REL0",
        expected_seed=2,
        expected_role_feature_sha256="e" * 64,
        expected_package_sha256_by_purpose=package_hashes,
        public_releases=packages, attack_view_type=S2AttackView,
        return_release=True,
    )
    assert set(restored) == set(original)
    for name in original:
        assert np.array_equal(restored[name].x, original[name].x)
        assert np.array_equal(restored[name].x_raw_only, original[name].x_raw_only)
        assert np.array_equal(restored[name].x_defense_aware,
                              original[name].x_defense_aware)
        assert restored[name].wire_components == original[name].wire_components
        assert set(restored_wire[name]) == set(release[name])
        for field, value in release[name].items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(restored_wire[name][field], value)
            else:
                assert restored_wire[name][field] == value

    with pytest.raises(PermissionError, match="source or package identity"):
        restore_view_bundle(
            receipt_path, expected_receipt_sha256=checksum,
            private_root=private, expected_campaign_lock_sha256="c" * 64,
            expected_data_manifest_sha256="d" * 64,
            expected_role="INNER_SELECTION", expected_arm_id="C-SIMPLE-REL0",
            expected_seed=2,
            expected_role_feature_sha256="e" * 64,
            expected_package_sha256_by_purpose=package_hashes,
            public_releases=packages, attack_view_type=S2AttackView,
        )

    receipt_path.write_text(receipt_path.read_text() + " ")
    with pytest.raises(PermissionError, match="receipt SHA"):
        restore_view_bundle(
            receipt_path, expected_receipt_sha256=checksum,
            private_root=private, expected_campaign_lock_sha256="c" * 64,
            expected_data_manifest_sha256="d" * 64,
            expected_role="ATTACK_FIT_NEW", expected_arm_id="C-SIMPLE-REL0",
            expected_seed=2,
            expected_role_feature_sha256="e" * 64,
            expected_package_sha256_by_purpose=package_hashes,
            public_releases=packages, attack_view_type=S2AttackView,
        )
