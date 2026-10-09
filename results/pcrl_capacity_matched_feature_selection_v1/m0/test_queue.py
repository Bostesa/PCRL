"""Data-free admission and shared-slot tests for the focused study queue."""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import psutil

from results.pcrl_capacity_matched_feature_selection_v1.m0.queue import FocusedQueue


class FocusedQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve(strict=True)
        self.slot = self.base / "shared-slot0.lock"
        self.slot.touch()
        self.root = self.base / "study-queue"
        self.start = dt.datetime.now(dt.timezone.utc).isoformat()
        self.q = FocusedQueue(self.root, self.slot, epoch_start_utc=self.start)
        self.script = self.base / "worker.py"
        self.out = self.base / "done.json"
        self.script.write_text(
            "import json,os,pathlib\n"
            f"pathlib.Path({str(self.out)!r}).write_text(json.dumps({{'threads':os.environ.get('OMP_NUM_THREADS')}}))\n"
        )

    def spec(self, mode: str = "synthetic") -> dict:
        return {
            "unit_id": "TEST-ONE", "mode": mode,
            "data_access": "NONE" if mode == "synthetic" else "OSF_DEFENSE_FIT",
            "argv": [sys.executable, str(self.script)],
            "cwd": str(self.base), "script": str(self.script),
            "script_sha256": hashlib.sha256(self.script.read_bytes()).hexdigest(),
            "inputs": {}, "outputs": [str(self.out)], "deps": [],
            "cpu_reserve_s": 10, "wall_limit_s": 10, "max_rss_gib": 1,
        }

    def test_archived_queue_carries_failed_unit_and_full_cpu_charge(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test fixture")
        self.script.write_text("raise SystemExit(1)\n")
        bad = self.spec()
        self.q.enqueue(bad)
        self.assertEqual(self.q.run_once()["status"], "FAILED")
        old = self.q.status()
        archived = self.base / "study-queue-first-attempt"
        self.root.rename(archived)
        next_q = FocusedQueue(self.root, self.slot, epoch_start_utc=self.start)
        carry = next_q.carry_forward_archived_budget(archived)
        self.assertEqual(carry["charged_cpu_s"], old["used_cpu_s"])
        self.assertEqual(next_q.status()["used_cpu_s"], old["used_cpu_s"])
        self.assertEqual(next_q.status()["units"], {})
        with self.assertRaisesRegex(ValueError, "fresh queue"):
            next_q.carry_forward_archived_budget(archived)
        second_archive = self.base / "study-queue-second-attempt"
        self.root.rename(second_archive)
        third_q = FocusedQueue(self.root, self.slot, epoch_start_utc=self.start)
        nested = third_q.carry_forward_archived_budget(second_archive)
        self.assertEqual(nested["charged_cpu_s"], old["used_cpu_s"])

    def test_real_unit_cannot_run_without_science_seal(self) -> None:
        with self.assertRaisesRegex(ValueError, "science gate"):
            self.q.enqueue(self.spec("real"))
        self.assertFalse(self.out.exists())

    def test_presealed_placeholder_binds_exact_local_seal_for_real_unit(self) -> None:
        study = self.base / "repo" / "results" / "study"
        private = study / ".private"
        private.mkdir(parents=True)
        queue = FocusedQueue(private / "queue", self.slot, epoch_start_utc=self.start)
        seal = study / "LOCAL_SEAL_RECEIPT.json"
        seal.write_text('{"status":"PASS"}')
        source = self.base / "repo" / "prior" / "role.npz"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"synthetic permitted fixture")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest_dir = study / "m1"
        manifest_dir.mkdir()
        manifest = {"role_file_relative_path": {"OSF_DEFENSE_FIT:features": "role.npz"},
                    "role_file_sha256": {"OSF_DEFENSE_FIT:features": digest}}
        (manifest_dir / "DATA_ACCESS_MANIFEST.json").write_text(json.dumps(manifest))
        fixture_script = self.base / "repo" / "worker.py"
        real_output = private / "real_done.json"
        fixture_script.write_text(f"import pathlib\npathlib.Path({str(real_output)!r}).write_text('done')\n")
        spec = self.spec("real")
        canonical_source = str(source.resolve())
        spec.update(script=str(fixture_script), argv=[sys.executable, str(fixture_script)],
                    cwd=str(self.base / "repo"),
                    script_sha256=hashlib.sha256(fixture_script.read_bytes()).hexdigest(),
                    inputs={canonical_source: digest}, outputs=[str(real_output)],
                    role_inputs={"OSF_DEFENSE_FIT:features":
                        {"path": canonical_source, "sha256": digest}},
                    role_access=["OSF_DEFENSE_FIT"], stage="FIT",
                    allow_child_processes=False,
                    source_map_sha256="manifest-hash", seal_sha256="effective-seal")
        (private / "UNIT_SPECS.json").write_text(json.dumps({"unit_specs": {
            "TEST-ONE": {**spec, "seal_sha256": "$LOCAL_SEAL_SHA256"}}}))
        (study / "CAMPAIGN_LOCK.json").write_text(json.dumps({
            "prior_package_relative_to_worktree": "prior", "source_map_sha256": "manifest-hash",
            "allowed_roles": ["OSF_DEFENSE_FIT"],
            "governed_source_sha256_by_path": {"worker.py": spec["script_sha256"]}}))
        with patch.object(FocusedQueue, "_seal_ok", return_value=True):
            queue.admit_science(seal_path=seal, seal_sha256="effective-seal",
                                external_heavy=1, evidence="fixture")
            queue.enqueue(spec)
            self.assertEqual(queue.run_once()["status"], "DONE")
        self.assertEqual(real_output.read_text(), "done")

    def test_full_local_git_seal_admits_exact_real_fixture(self) -> None:
        repo = self.base / "repo"
        study = repo / "results" / "study"
        private = study / ".private"
        (study / "m1").mkdir(parents=True)
        (study / "m5").mkdir()
        private.mkdir()
        source = repo / "prior" / "role.npz"
        source.parent.mkdir()
        source.write_bytes(b"allowed test-role bytes")
        role_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        output = private / "completed.json"
        script = repo / "worker.py"
        real_source_root = Path(__file__).resolve().parents[3]
        script.write_text(
            "import pathlib,sys\n"
            f"sys.path.insert(0,{str(real_source_root)!r})\n"
            "from results.pcrl_capacity_matched_feature_selection_v1.m0.admission "
            "import assert_running_unit\n"
            f"assert_running_unit('TEST-ONE',{str(private / 'queue')!r},"
            f"{str(study / 'CAMPAIGN_LOCK.json')!r},'FIT',{str(output)!r})\n"
            f"pathlib.Path({str(output)!r}).write_text('ok')\n"
        )
        manifest = {"role_file_relative_path": {"OSF_DEFENSE_FIT:features": "role.npz"},
                    "role_file_sha256": {"OSF_DEFENSE_FIT:features": role_sha}}
        manifest_path = study / "m1" / "DATA_ACCESS_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest))
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        protocol_path = study / "COMMON_PROTOCOL.md"
        protocol_path.write_text("fixed synthetic protocol\n")
        protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        spec = self.spec("real")
        spec.update(script=str(script), script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                    argv=[sys.executable, str(script)], cwd=str(repo), outputs=[str(output)],
                    inputs={str(source.resolve()): role_sha},
                    role_inputs={"OSF_DEFENSE_FIT:features":
                                 {"path": str(source.resolve()), "sha256": role_sha}},
                    role_access=["OSF_DEFENSE_FIT"], stage="FIT",
                    source_map_sha256=manifest_sha, allow_child_processes=False,
                    seal_sha256="$LOCAL_SEAL_SHA256")
        registry = private / "UNIT_SPECS.json"
        registry.write_text(json.dumps({"unit_specs": {"TEST-ONE": spec}}))
        registry_sha = hashlib.sha256(registry.read_bytes()).hexdigest()
        lock = study / "CAMPAIGN_LOCK.json"
        lock.write_text(json.dumps({
            "status": "LOCAL_GIT_SEALED_EFFECTIVE", "adult_fit_permission": True,
            "private_unit_registry_sha256": registry_sha,
            "source_map_sha256": manifest_sha,
            "governed_source_sha256_by_path": {"worker.py": spec["script_sha256"]},
            "governed_protocol_sha256_by_relative_path": {
                "COMMON_PROTOCOL.md": protocol_sha},
            "prior_package_relative_to_worktree": "prior",
            "allowed_roles": ["OSF_DEFENSE_FIT"]}))
        lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Queue Test"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "worker.py", "results/study/CAMPAIGN_LOCK.json",
                        "results/study/m1/DATA_ACCESS_MANIFEST.json"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "synthetic seal"], check=True)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        review = study / "m5" / "PREFIT_REVIEW.json"
        review.write_text(json.dumps({"status": "PASS", "lock_sha256": lock_sha,
                                      "lock_commit_sha": commit}))
        seal = study / "LOCAL_SEAL_RECEIPT.json"
        seal.write_text(json.dumps({"status": "PASS", "lock_sha256": lock_sha,
                                    "lock_commit_sha": commit,
                                    "m5_review_sha256": hashlib.sha256(review.read_bytes()).hexdigest()}))
        seal_sha = hashlib.sha256(seal.read_bytes()).hexdigest()
        self.assertTrue(FocusedQueue._seal_ok(seal, seal_sha))
        with patch("results.pcrl_capacity_matched_feature_selection_v1.m0.queue.resource.getrlimit",
                   return_value=(0, 0)), patch(
                   "results.pcrl_capacity_matched_feature_selection_v1.m0.queue.subprocess.run",
                   side_effect=AssertionError("contained worker must not spawn Git")):
            self.assertTrue(FocusedQueue._seal_ok(
                seal, seal_sha, contained_worker=True))
        queue = FocusedQueue(private / "queue", self.slot, epoch_start_utc=self.start)
        queue.admit_science(seal_path=seal, seal_sha256=seal_sha,
                            external_heavy=1, evidence="synthetic local git fixture")
        spec["seal_sha256"] = seal_sha
        queue.enqueue(spec)
        # The contained worker revalidates its own RUNNING spec; its output
        # reservation must not be mistaken for a different unit's collision.
        queue._validate_spec(spec, queue.status())
        self.assertEqual(queue.run_once()["status"], "DONE")
        self.assertEqual(output.read_text(), "ok")
        output.unlink()
        verification_lock = json.loads(lock.read_text())
        verification_lock.update(adult_fit_permission=False,
                                 verification_only=True)
        lock.write_text(json.dumps(verification_lock))
        with self.assertRaisesRegex(ValueError, "verification-only lock denies"):
            queue._validate_spec(spec, queue.status())
        replay_spec = {**spec, "stage": "independent_task_no_go"}
        registry.write_text(json.dumps({"unit_specs": {
            "TEST-ONE": {**replay_spec, "seal_sha256": "$LOCAL_SEAL_SHA256"}}}))
        verification_lock["private_unit_registry_sha256"] = hashlib.sha256(
            registry.read_bytes()).hexdigest()
        lock.write_text(json.dumps(verification_lock))
        queue._validate_spec(replay_spec, queue.status())
        protocol_path.write_text("mutated after real unit\n")
        self.assertFalse(FocusedQueue._seal_ok(seal, seal_sha))
        with patch("results.pcrl_capacity_matched_feature_selection_v1.m0.queue.resource.getrlimit",
                   return_value=(0, 0)):
            self.assertFalse(FocusedQueue._seal_ok(
                seal, seal_sha, contained_worker=True))

    def test_derived_only_real_stage_needs_no_role_file(self) -> None:
        study = self.base / "repo" / "results" / "study"
        private = study / ".private"
        (study / "m1").mkdir(parents=True)
        private.mkdir()
        manifest = study / "m1/DATA_ACCESS_MANIFEST.json"
        manifest.write_text(json.dumps({"role_file_relative_path": {},
                                        "role_file_sha256": {}}))
        script = self.base / "repo" / "inference.py"
        output = private / "derived.json"
        script.write_text(f"import pathlib\npathlib.Path({str(output)!r}).write_text('derived')\n")
        spec = self.spec("real")
        spec.update(unit_id="DERIVED-ONE", stage="simultaneous_inference",
                    data_access="NONE_PRIVATE_DERIVED_ONLY", role_inputs={},
                    role_access=[], inputs={}, outputs=[str(output)],
                    script=str(script), argv=[sys.executable, str(script)],
                    cwd=str(self.base / "repo"),
                    script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                    source_map_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    allow_child_processes=False,
                    seal_sha256="$LOCAL_SEAL_SHA256")
        (private / "UNIT_SPECS.json").write_text(json.dumps({"unit_specs": {
            "DERIVED-ONE": spec}}))
        (study / "CAMPAIGN_LOCK.json").write_text(json.dumps({
            "prior_package_relative_to_worktree": "prior",
            "source_map_sha256": spec["source_map_sha256"],
            "allowed_roles": [],
            "governed_source_sha256_by_path": {
                "inference.py": spec["script_sha256"]}}))
        (self.base / "repo" / "prior").mkdir()
        seal = study / "LOCAL_SEAL_RECEIPT.json"
        seal.write_text('{"status":"PASS"}')
        queue = FocusedQueue(private / "queue", self.slot, epoch_start_utc=self.start)
        with patch.object(FocusedQueue, "_seal_ok", return_value=True):
            queue.admit_science(seal_path=seal, seal_sha256="derived-seal",
                                external_heavy=1, evidence="derived-only fixture")
            queue.enqueue({**spec, "seal_sha256": "derived-seal"})
            self.assertEqual(queue.run_once()["status"], "DONE")
        self.assertEqual(output.read_text(), "derived")

    def test_synthetic_worker_uses_shared_slot_and_single_thread(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test-only observation")
        self.q.enqueue(self.spec())
        with self.slot.open("r+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.q.run_once()["status"], "SLOT_BUSY")
            self.assertFalse(self.out.exists())
        result = self.q.run_once()
        self.assertEqual(result["status"], "DONE")
        self.assertEqual(json.loads(self.out.read_text()), {"threads": "1"})
        self.assertGreaterEqual(self.q.status()["used_cpu_s"], 0)

    def test_requested_unit_cannot_launch_an_earlier_queued_unit(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="data-free fixture")
        foreign = self.spec()
        foreign["unit_id"] = "FOREIGN-ONE"
        self.q.enqueue(foreign)
        second_output = self.base / "second.json"
        second_script = self.base / "second.py"
        second_script.write_text(
            f"import pathlib\npathlib.Path({str(second_output)!r}).write_text('second')\n")
        target = self.spec()
        target.update(unit_id="TARGET-ONE", script=str(second_script),
                      script_sha256=hashlib.sha256(second_script.read_bytes()).hexdigest(),
                      argv=[sys.executable, str(second_script)],
                      outputs=[str(second_output)])
        self.q.enqueue(target)
        denied = self.q.run_once(expected_unit_id="TARGET-ONE")
        self.assertEqual(denied["status"], "UNEXPECTED_PENDING")
        self.assertEqual(denied["unit_id"], "FOREIGN-ONE")
        self.assertFalse(self.out.exists())
        self.assertFalse(second_output.exists())
        self.assertEqual(self.q.status()["units"]["TARGET-ONE"]["status"], "QUEUED")

    def test_changed_script_is_rejected_before_launch(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test-only observation")
        self.q.enqueue(self.spec())
        self.script.write_text("raise RuntimeError('changed')\n")
        self.assertEqual(self.q.run_once()["status"], "INPUT_IDENTITY_CHANGED")
        self.assertFalse(self.out.exists())

    def test_existing_output_is_rejected_before_enqueue(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test-only observation")
        self.out.write_text("historical output")
        with self.assertRaisesRegex(ValueError, "output already exists"):
            self.q.enqueue(self.spec())
        self.assertEqual(self.out.read_text(), "historical output")

    def test_synthetic_unit_cannot_declare_role_input(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test-only observation")
        fake_role = self.base / "OSF_DEFENSE_FIT.npz"
        fake_role.write_text("synthetic fixture")
        unit = self.spec()
        unit["inputs"] = {str(fake_role): hashlib.sha256(fake_role.read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError, "synthetic inputs"):
            self.q.enqueue(unit)

    def test_unit_cannot_start_if_wall_cap_exceeds_remaining_epoch(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test-only observation")
        self.q.enqueue(self.spec())
        self.q.start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=35995)
        self.assertEqual(self.q.run_once()["status"], "BUDGET_CEILING")
        self.assertFalse(self.out.exists())

    def test_output_appearing_after_enqueue_blocks_launch(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test-only observation")
        self.q.enqueue(self.spec())
        self.out.write_text("do not overwrite")
        self.assertEqual(self.q.run_once()["status"], "INPUT_IDENTITY_CHANGED")
        self.assertEqual(self.out.read_text(), "do not overwrite")

    def test_failed_executable_gets_terminal_receipt(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test-only observation")
        unit = self.spec()
        unit["argv"][0] = str(self.base / "no-python")
        self.q.enqueue(unit)
        result = self.q.run_once()
        self.assertEqual(result["status"], "FAILED_START")
        self.assertEqual(self.q.status()["units"]["TEST-ONE"]["status"], "FAILED_START")

    def test_memory_cap_terminates_own_worker(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test-only observation")
        self.script.write_text(
            "import time\nblob=bytearray(128*1024*1024)\ntime.sleep(5)\n"
        )
        unit = self.spec()
        unit["max_rss_gib"] = 0.04
        unit["wall_limit_s"] = 4
        self.q.enqueue(unit)
        result = self.q.run_once()
        self.assertEqual(result["status"], "FAILED_MEMORY_LIMIT")
        self.assertTrue(self.q.status()["units"]["TEST-ONE"]["memory_exceeded"])
        self.assertFalse(self.out.exists())

    def test_probe_failure_is_terminal_and_kills_own_group(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test-only observation")
        self.script.write_text("import time\ntime.sleep(3)\n")
        self.q.enqueue(self.spec())
        with patch.object(psutil, "Process", side_effect=PermissionError("probe denied")):
            result = self.q.run_once()
        self.assertEqual(result["status"], "FAILED_PROBE")
        self.assertEqual(self.q.status()["units"]["TEST-ONE"]["status"], "FAILED_PROBE")
        self.assertFalse(self.out.exists())

    def test_live_descendant_is_not_reported_done(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test-only observation")
        self.script.write_text(
            "import os,pathlib,subprocess,sys,time\n"
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'])\n"
            f"pathlib.Path({str(self.out)!r}).write_text(str(p.pid))\n"
            "time.sleep(0.2)\n"
        )
        unit = self.spec()
        try:
            self.q.enqueue(unit)
            result = self.q.run_once()
            self.assertEqual(result["status"], "FAILED_ORPHAN_CHILD")
            self.assertNotEqual(self.q.status()["units"]["TEST-ONE"]["status"], "DONE")
        finally:
            if self.out.exists():
                try:
                    os.kill(int(self.out.read_text()), 9)
                except (ProcessLookupError, ValueError):
                    pass

    def test_nochild_worker_cannot_spawn_and_wait(self) -> None:
        self.q.admit_synthetic(external_heavy=1, evidence="test-only observation")
        self.script.write_text(
            "import pathlib,subprocess,sys\n"
            "subprocess.run([sys.executable,'-c','print(1)'],check=True)\n"
            f"pathlib.Path({str(self.out)!r}).write_text('unexpected')\n"
        )
        unit = self.spec()
        unit["allow_child_processes"] = False
        self.q.enqueue(unit)
        self.assertEqual(self.q.run_once()["status"], "FAILED")
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
