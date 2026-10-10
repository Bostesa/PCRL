"""Data-free A release admission and private stream custody checks."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import run_release
from .admitted_fit import REQUIRED as A_REQUIRED


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

    def test_queue_output_is_receipt_file_not_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "CAMPAIGN_LOCK.json"
            queue = root / "queue"
            queue.mkdir()
            output = root / ".private" / "release_s0"
            output.parent.mkdir(mode=0o700)
            public = root / "sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json"
            public.parent.mkdir(parents=True)
            roots = {str(i): str(i + 1) * 64 for i in range(3)}
            public.write_text(json.dumps({
                "schema": "scv1-public-stream-root-hashes-v1",
                "release_realization_root_sha256": roots,
                "seq_fit_root_sha256": "4" * 64,
                "seq_fit_hmac_domain": "SCV1-seq-fit-first-v1"}))
            lock.write_text(json.dumps({
                "schema": "pcrl-latent-and-sampled-campaign-lock-v1",
                "status": "LOCALLY_SEALED_D_STAGE",
                "source_files_sha256": {"source_role_manifest": "a" * 64,
                                        "r01/SOURCE_AUDIT.json": "b" * 64,
                                        "r03/compute_queue.py": "c" * 64}}))
            amendment = root / "A_STAGE.json"
            plan = {"kind": "release", "arm_id": "A_JOINT_STOCH", "seed": 0,
                    "role": "ATTACK_FIT_NEW", "realization": 0,
                    "output_id": "joint_s0_att_0", "cpu_seconds": 5.,
                    "wall_seconds": 5.}
            payload = {key: {} for key in A_REQUIRED}
            payload.update(schema="pcrl-latent-sampled-A-stage-amendment-v1",
                           status="LOCALLY_SEALED_A_STAGE",
                           dated_utc="2026-10-10T02:31:25Z",
                           frozen_candidate_sha256="e" * 64,
                           independent_prefit_audit_sha256="f" * 64,
                           input_scope="ADMITTED_DEVELOPMENT_ROLES_ONLY",
                           owner="R00", previous_a_amendment_sha256="1" * 64,
                           scope_note="synthetic release admission", stage_scope=["a"],
                           original_lock_sha256=run_release._sha(lock),
                           role_bindings={"repair_manifest_sha256": "a" * 64,
                                          "source_audit_sha256": "b" * 64},
                           implementation_source_sha256={"r03/compute_queue.py": "c" * 64},
                           unit_plan={"A-RELEASE-0": plan},
                           realizations={"ids": [0, 1, 2],
                                         "stream_root_hash_manifest_sha256": run_release._sha(public),
                                         "release_root_sha256_by_realization": roots,
                                         "seq_fit_root_sha256": "4" * 64,
                                         "seq_fit_hmac_domain": "SCV1-seq-fit-first-v1"})
            amendment.write_text(json.dumps(payload))
            spec = {"stage": "a", "mode": "real_data", "arm_id": "A_JOINT_STOCH",
                    "seed": 0, "role": "ATTACK_FIT_NEW", "realization": 0,
                    "output_id": "joint_s0_att_0", "cpu_reserve_s": 5.,
                    "max_wall_s": 5., "amendment_sha256": run_release._sha(amendment),
                    "outputs": [str(output.resolve() / "RELEASE_RECEIPT.json")],
                    "sources": [{"path": str(path.resolve()),
                                 "sha256": run_release._sha(path)}
                                for path in (lock, amendment, public)]}
            state = {"amendments": [{"path": str(amendment.resolve()),
                                      "sha256": run_release._sha(amendment),
                                      "stages": ["a"]}],
                     "units": {"A-RELEASE-0": {"spec": spec}}}
            (queue / "state.json").write_text(json.dumps(state))
            with patch.dict("os.environ", {"PCRL_QUEUE_ROOT": str(queue)}), \
                 patch.object(run_release, "assert_active_from_env",
                              return_value={"seal_sha256": run_release._sha(lock)}), \
                 patch.object(run_release, "RELEASE_IMPLEMENTATION", frozenset()):
                run_release.admit_release_unit(
                    unit_id="A-RELEASE-0", lock_path=lock, amendment_path=amendment,
                    amendment_sha256=run_release._sha(amendment), output_dir=output)
                spec["outputs"] = [str(output.resolve())]
                (queue / "state.json").write_text(json.dumps(state))
                with self.assertRaisesRegex(PermissionError, "unit/role/draw"):
                    run_release.admit_release_unit(
                        unit_id="A-RELEASE-0", lock_path=lock, amendment_path=amendment,
                        amendment_sha256=run_release._sha(amendment), output_dir=output)

    def test_parent_fit_receipt_matches_completed_queue_output(self):
        path = Path("/private/fit/FIT_RECEIPT.json")
        fit_sha = "a" * 64
        spec = {"deps": ["A-FIT-0"]}
        parent = {"status": "done", "spec": {
            "arm_id": "A_JOINT_STOCH", "seed": 0, "outputs": [str(path)]},
            "output_sha256": {str(path): fit_sha}}
        expected = dict(dependency="A-FIT-0", arm_id="A_JOINT_STOCH",
                        seed=0, fit_receipt_path=path, fit_sha256=fit_sha)
        run_release._completed_fit_dependency(spec, parent, **expected)
        for key, changed in (("outputs", ["/private/other/FIT_RECEIPT.json"]),
                             ("seed", 1)):
            bad = {**parent, "spec": {**parent["spec"], key: changed}}
            with self.subTest(key=key), self.assertRaises(PermissionError):
                run_release._completed_fit_dependency(spec, bad, **expected)
        bad = {**parent, "output_sha256": {str(path): "b" * 64}}
        with self.assertRaises(PermissionError):
            run_release._completed_fit_dependency(spec, bad, **expected)


if __name__ == "__main__":
    unittest.main()
