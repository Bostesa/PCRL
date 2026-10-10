"""Data-free A release admission and private stream custody checks."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import run_release


class ReleaseWorkerTest(unittest.TestCase):
    def test_queue_denial_precedes_policy_root_and_role_access(self):
        with patch.object(run_release, "assert_active_from_env",
                          side_effect=PermissionError("R03 slot")), \
             patch.object(run_release, "TStarSourceStore",
                          side_effect=AssertionError("source opened")), \
             patch.object(run_release, "RoleOnlyStore",
                          side_effect=AssertionError("role opened")):
            with self.assertRaisesRegex(PermissionError, "R03 slot"):
                run_release.admit_release_unit(
                    unit_id="A-RELEASE-JOINT-S0-ATT-R0",
                    lock_path="missing", amendment_path="missing",
                    amendment_sha256="a" * 64, output_dir="missing")

    def test_root_manifest_requires_distinct_hashes_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "roots.json"
            public = root / "hashes.json"
            values = {str(i): bytes([i + 1]) * 32 for i in range(3)}
            seq = b"\x04" * 32
            private.write_text(json.dumps({
                "schema": "scv1-private-stream-roots-v1",
                "release_realization_roots_hex": {
                    key: value.hex() for key, value in values.items()},
                "seq_fit_root_hex": seq.hex()}))
            private.chmod(0o600)
            public.write_text(json.dumps({
                "schema": "scv1-public-stream-root-hashes-v1",
                "private_file_sha256": run_release._sha(private),
                "private_file_mode": "0600",
                "release_realization_root_sha256": {
                    key: hashlib.sha256(value).hexdigest()
                    for key, value in values.items()},
                "seq_fit_root_sha256": hashlib.sha256(seq).hexdigest(),
                "seq_fit_hmac_domain": "SCV1-seq-fit-first-v1",
                "release_hmac_message": "compact ASCII JSON [SCV1-uniform-v1,source_seed,role,realization,unit]"}))
            self.assertEqual(run_release.load_selected_stream_root(
                private, public, realization=1,
                expected_public_sha256=run_release._sha(public)), values["1"])
            linked = root / "linked_roots.json"
            linked.symlink_to(private)
            with self.assertRaises(ValueError):
                run_release.load_selected_stream_root(
                    linked, public, realization=1,
                    expected_public_sha256=run_release._sha(public))
            private.write_text(private.read_text().replace("02" * 32, "01" * 32))
            with self.assertRaises(ValueError):
                run_release.load_selected_stream_root(
                    private, public, realization=1,
                    expected_public_sha256=run_release._sha(public))

    def test_selected_policy_must_match_fit_source_action_and_checkpoint(self):
        fit = {
            "schema": "SCV1-private-stochastic-fit-v1",
            "status": "SELECTED_FIT_POLICY", "unit_id": "fit-0",
            "arm_id": "A_JOINT_STOCH", "source_seed": 0,
            "a_amendment_sha256": "a" * 64,
            "source_binding_sha256": "b" * 64,
            "row_unit_pair_sha256": "c" * 64,
            "bank_sha256_by_recipient": {"r1": "d" * 64, "r2": "e" * 64},
            "action_sha256_by_recipient": {"r1": "f" * 64, "r2": "0" * 64},
            "selected": {"start": 0, "update": 25},
        }
        policy = {key: fit[key] for key in (
            "status", "arm_id", "source_seed", "source_binding_sha256",
            "row_unit_pair_sha256", "bank_sha256_by_recipient",
            "action_sha256_by_recipient")}
        policy.update(fit_unit_id="fit-0", selected_start="identity_biased",
                      selected_update=25)
        expected = dict(unit_id="fit-0", arm_id="A_JOINT_STOCH", seed=0,
                        amendment_sha256="a" * 64)
        run_release._selected_parent_identity(fit, policy, **expected)
        for key, replacement in (("source_binding_sha256", "1" * 64),
                                 ("row_unit_pair_sha256", "2" * 64),
                                 ("action_sha256_by_recipient",
                                  {"r1": "3" * 64, "r2": "0" * 64}),
                                 ("selected_update", 50),
                                 ("fit_unit_id", "fit-1")):
            swapped = dict(policy)
            swapped[key] = replacement
            with self.subTest(key=key), self.assertRaises(PermissionError):
                run_release._selected_parent_identity(fit, swapped, **expected)


if __name__ == "__main__":
    unittest.main()
