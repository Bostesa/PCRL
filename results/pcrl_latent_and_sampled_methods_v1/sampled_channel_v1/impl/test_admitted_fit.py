"""Synthetic access-order and task capability checks; no Adult archive read."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

if __package__:
    from .admitted_fit import admit_fit_unit, load_fit_capability, MANDATORY_IMPLEMENTATION
    MODULE = f"{__package__}.admitted_fit"
else:
    from admitted_fit import admit_fit_unit, load_fit_capability, MANDATORY_IMPLEMENTATION
    MODULE = "admitted_fit"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AdmittedFitTest(unittest.TestCase):
    def test_queue_denial_precedes_amendment_and_role_read(self):
        with patch(f"{MODULE}.assert_active_from_env", side_effect=PermissionError("queue")), \
             patch(f"{MODULE}.RoleOnlyStore", side_effect=AssertionError("role opened")):
            with self.assertRaisesRegex(PermissionError, "queue"):
                admit_fit_unit(unit_id="A-JOINT-STOCH-S0", lock_path="missing",
                               amendment_path="missing", amendment_sha256="a" * 64,
                               output_dir="missing")

    def test_task_loader_never_reads_sex_and_protected_reads_last(self):
        class Source:
            source_binding_sha256 = "a" * 64

            def read_bank(self, recipient):
                return {"bank": recipient}

            def read_role_cells(self, role):
                assert role == "OSF_DEFENSE_FIT"
                return {"row_id": np.array([1], np.int64), "unit": np.array([1], np.int64)}

        class Role:
            def __init__(self):
                self.calls = []

            def read(self, role, field, procedure=None):
                self.calls.append((role, field, procedure))
                if field == "sex":
                    return {"row_id": np.array([1], np.int64),
                            "unit": np.array([1], np.int64),
                            "sex": np.array([0], np.int64)}
                return {"field": field}

        source, role = Source(), Role()
        expected = SimpleNamespace(row_unit_pair_sha256="b" * 64)
        with patch(f"{MODULE}.bind_task_inputs", return_value=("task", "policy", {}, {})):
            task = load_fit_capability(source, role, seed=0, objective_kind="task_rate",
                                       expected=expected)
            self.assertEqual(task[0], "task")
            self.assertEqual([x[1] for x in role.calls],
                             ["references_s0", "income", "occupation"])
            with patch(f"{MODULE}.attach_protected", return_value="protected"):
                got = load_fit_capability(source, role, seed=0,
                                          objective_kind="protected_bayes",
                                          expected=expected)
                self.assertEqual(got[0], "protected")
                self.assertEqual(role.calls[-1], ("OSF_DEFENSE_FIT", "sex", "defense_fit"))

    def test_effective_amendment_latest_and_source_hash_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            lock = root / "CAMPAIGN_LOCK.json"
            lock.write_text(json.dumps({
                "schema": "pcrl-latent-and-sampled-campaign-lock-v1",
                "status": "LOCALLY_SEALED_D_STAGE",
                "source_files_sha256": {
                    "source_role_manifest": "a" * 64,
                    "r01/SOURCE_AUDIT.json": "b" * 64,
                    "r03/compute_queue.py": hashlib.sha256(b"source\n").hexdigest()}}))
            amendment = root / "A_STAGE.json"
            output = root / ".private" / "private-output"
            output.parent.mkdir(mode=0o700)
            sources = {}
            for relative in MANDATORY_IMPLEMENTATION:
                source = root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("source\n")
                sources[relative] = sha(source)
            stream = root / "sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json"
            stream.parent.mkdir(parents=True, exist_ok=True)
            root_hashes = {"0": "1" * 64, "1": "2" * 64, "2": "3" * 64}
            stream.write_text(json.dumps({
                "schema": "scv1-public-stream-root-hashes-v1",
                "release_realization_root_sha256": root_hashes,
                "seq_fit_root_sha256": "4" * 64,
                "seq_fit_hmac_domain": "SCV1-seq-fit-first-v1"}))
            payload = {"schema": "pcrl-latent-sampled-A-stage-amendment-v1",
                       "status": "LOCALLY_SEALED_A_STAGE",
                       "dated_utc": "2026-10-10T02:31:25Z",
                       "frozen_candidate_sha256": "e" * 64,
                       "independent_prefit_audit_sha256": "f" * 64,
                       "input_scope": "ADMITTED_DEVELOPMENT_ROLES_ONLY",
                       "owner": "R00", "previous_a_amendment_sha256": "1" * 64,
                       "scope_note": "synthetic admission test", "stage_scope": ["a"],
                       "original_lock_sha256": sha(lock),
                       "source_sha256": {}, "role_bindings": {
                           "repair_manifest_sha256": "a" * 64,
                           "source_audit_sha256": "b" * 64},
                       "bank_action_sha256_by_seed": {}, "optimizer_recipe": {},
                       "control_recipe": {}, "utility_caps": {}, "attack_plan": {},
                       "realizations": {
                           "ids": [0, 1, 2],
                           "stream_root_hash_manifest_sha256": sha(stream),
                           "release_root_sha256_by_realization": root_hashes,
                           "seq_fit_root_sha256": "4" * 64,
                           "seq_fit_hmac_domain": "SCV1-seq-fit-first-v1"},
                       "interval_family": {}, "cpu_budget": {},
                       "implementation_source_sha256": sources,
                       "unit_plan": {"A-JOINT-STOCH-S0": {
                           "arm_id": "A_JOINT_STOCH", "seed": 0,
                           "objective_kind": "protected_bayes",
                           "output_id": "joint_s0"}}}
            amendment.write_text(json.dumps(payload))
            queue = root / "queue"
            queue.mkdir()
            spec = {"stage": "a", "seed": 0, "arm_id": "A_JOINT_STOCH",
                    "output_id": "joint_s0",
                    "amendment_sha256": sha(amendment),
                    "outputs": [str(output.resolve() / "FIT_RECEIPT.json")],
                    "sources": [{"path": str(p.resolve()), "sha256": sha(p)}
                                for p in (lock, amendment,
                                          *(root / name for name in MANDATORY_IMPLEMENTATION),
                                          stream)]}
            state = {"amendments": [{"path": str(amendment.resolve()),
                                     "sha256": sha(amendment), "stages": ["a"]}],
                     "units": {"A-JOINT-STOCH-S0": {"spec": spec}}}
            (queue / "state.json").write_text(json.dumps(state))
            with patch.dict("os.environ", {"PCRL_QUEUE_ROOT": str(queue)}), \
                 patch(f"{MODULE}.assert_active_from_env",
                       return_value={"seal_sha256": sha(lock)}):
                got, unit = admit_fit_unit(unit_id="A-JOINT-STOCH-S0",
                                           lock_path=lock, amendment_path=amendment,
                                           amendment_sha256=sha(amendment),
                                           output_dir=output)
                self.assertEqual(got["unit_plan"]["A-JOINT-STOCH-S0"]["seed"], 0)
                self.assertEqual(unit["stage"], "a")
                public = root / "public-fit-output"
                spec["outputs"] = [str(public / "FIT_RECEIPT.json")]
                (queue / "state.json").write_text(json.dumps(state))
                with self.assertRaisesRegex(PermissionError, "owner-only private"):
                    admit_fit_unit(unit_id="A-JOINT-STOCH-S0", lock_path=lock,
                                   amendment_path=amendment,
                                   amendment_sha256=sha(amendment), output_dir=public)
                spec["outputs"] = [str(output / "FIT_RECEIPT.json")]
                (queue / "state.json").write_text(json.dumps(state))
                spec["outputs"] = [str(output.resolve())]
                (queue / "state.json").write_text(json.dumps(state))
                with self.assertRaisesRegex(PermissionError, "unit/amendment/output"):
                    admit_fit_unit(unit_id="A-JOINT-STOCH-S0", lock_path=lock,
                                   amendment_path=amendment,
                                   amendment_sha256=sha(amendment), output_dir=output)
                spec["outputs"] = [str(output.resolve() / "FIT_RECEIPT.json")]
                (queue / "state.json").write_text(json.dumps(state))
                saved_stream = stream.read_bytes()
                stream.write_text('{"tampered":true}')
                with self.assertRaisesRegex(PermissionError, "stream"):
                    admit_fit_unit(unit_id="A-JOINT-STOCH-S0", lock_path=lock,
                                   amendment_path=amendment,
                                   amendment_sha256=sha(amendment), output_dir=output)
                stream.write_bytes(saved_stream)
                source = root / "sampled_channel_v1/impl/fit.py"
                source.write_text("changed\n")
                with self.assertRaisesRegex(PermissionError, "source"):
                    admit_fit_unit(unit_id="A-JOINT-STOCH-S0", lock_path=lock,
                                   amendment_path=amendment,
                                   amendment_sha256=sha(amendment), output_dir=output)


if __name__ == "__main__":
    unittest.main()
