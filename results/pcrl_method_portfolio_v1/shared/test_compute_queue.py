"""Lightweight behavior checks for the portfolio's local compute queue."""

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compute_queue import Queue


class ComputeQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.queue = Queue(self.root)
        self.marker = self.root / "ran.txt"
        self.synthetic_test = self.root / "test_fixture.py"
        self.synthetic_test.write_text("# fixed queue fixture\n")
        law = self.root / "law.json"
        law.write_text("{}")
        self.phase1_lock = self.root / "PHASE1_INDEX.json"
        self.phase1_lock.write_text(json.dumps({
            "shared_laws": {"path": law.name, "sha256": hashlib.sha256(law.read_bytes()).hexdigest()},
            "pre_algorithm_tests": {"S3_FIXTURE": [{"path": self.synthetic_test.name,
                                                  "sha256": hashlib.sha256(self.synthetic_test.read_bytes()).hexdigest()}]},
        }))
        self.spec = {
            "unit_id": "synthetic_smoke_s0",
            "argv": [sys.executable, "-c", f"from pathlib import Path; Path({str(self.marker)!r}).write_text('ok')"],
            "cwd": str(self.root),
            "deps": [],
            "max_wall_s": 4,
            "cpu_reserve_s": 4,
            "max_rss_gib": 0.25,
            "stage": "setup",
            "mode": "prelock_synthetic", "team": "S3_FIXTURE", "data_access": "NONE",
            "test_file": str(self.synthetic_test),
            "test_sha256": hashlib.sha256(self.synthetic_test.read_bytes()).hexdigest(),
        }
        self.queue.enqueue(self.spec)
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=False,
                               external_heavy=2, phase1_lock_path=str(self.phase1_lock),
                               phase1_lock_sha256=hashlib.sha256(self.phase1_lock.read_bytes()).hexdigest())

    def _suppress_default_fixture(self):
        state = self.queue.status()
        state["units"]["synthetic_smoke_s0"]["status"] = "skipped"
        self.queue._save(state)

    def test_gate_is_closed_until_all_admissions_are_recorded(self):
        self.assertEqual(self.queue.run_once()["status"], "gate_closed")
        self.assertFalse(self.marker.exists())
        self.queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True, external_heavy=2)
        self.assertEqual(self.queue.run_once()["status"], "external_capacity_full")
        self.assertFalse(self.marker.exists())

    def test_unit_runs_once_with_receipt_and_complete_marker(self):
        self.queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True, external_heavy=0)
        result = self.queue.run_once()
        self.assertEqual(result["status"], "done")
        self.assertEqual(self.marker.read_text(), "ok")
        self.assertEqual(self.queue.run_once()["status"], "empty")
        state = self.queue.status()
        self.assertEqual(state["units"]["synthetic_smoke_s0"]["status"], "done")
        self.assertGreater(state["used_cpu_s"], 0)
        receipts = [json.loads(line) for line in (self.root / "RECEIPTS.jsonl").read_text().splitlines()]
        self.assertEqual([r["event"] for r in receipts if r.get("unit_id") == "synthetic_smoke_s0"],
                         ["enqueue", "start", "child_started", "finish"])

    def test_both_flock_slots_prevent_an_unregistered_third_job(self):
        self.queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True, external_heavy=0)
        handles = [(self.root / f"slot{i}.lock").open("a+") for i in (0, 1)]
        self.addCleanup(lambda: [h.close() for h in handles])
        for handle in handles:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertEqual(self.queue.run_once()["status"], "portfolio_capacity_full")
        self.assertFalse(self.marker.exists())

    def test_prelock_synthetic_unit_requires_registered_law_and_test_hash(self):
        law = self.root / "law.json"
        law.write_text('{"fixed": true}')
        test_file = self.root / "test_tiny.py"
        test_file.write_text("# frozen synthetic-only test\n")
        law_hash = hashlib.sha256(law.read_bytes()).hexdigest()
        test_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()
        lock = self.root / "PHASE1_LOCK.json"
        lock.write_text(json.dumps({
            "shared_laws": {"path": "law.json", "sha256": law_hash},
            "pre_algorithm_tests": {"team_c": {"path": "test_tiny.py", "sha256": test_hash}},
        }))
        spec = dict(self.spec, unit_id="synthetic_c", mode="prelock_synthetic",
                    team="team_c", data_access="NONE", test_file=str(test_file),
                    test_sha256=test_hash)
        self.queue.enqueue(spec)
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, phase1_lock_path=str(lock),
                               phase1_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
        result = self.queue.run_once()
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["unit_id"], "synthetic_c")
        self.assertEqual(self.queue.status()["units"]["synthetic_smoke_s0"]["status"], "planned")

    def test_prelock_synthetic_unit_rejects_unregistered_test(self):
        law = self.root / "law.json"
        law.write_text("{}")
        lock = self.root / "PHASE1_LOCK.json"
        lock.write_text(json.dumps({"shared_laws": {"path": "law.json", "sha256": hashlib.sha256(law.read_bytes()).hexdigest()},
                                    "pre_algorithm_tests": {}}))
        test_file = self.root / "unregistered.py"
        test_file.write_text("# not locked\n")
        self.queue.enqueue(dict(self.spec, unit_id="bad_synthetic", mode="prelock_synthetic",
                                team="team_c", data_access="NONE", test_file=str(test_file),
                                test_sha256=hashlib.sha256(test_file.read_bytes()).hexdigest()))
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, phase1_lock_path=str(lock),
                               phase1_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
        self.assertEqual(self.queue.run_once()["status"], "phase1_test_unregistered")
        self.assertFalse(self.marker.exists())

    def test_legacy_unqueued_checks_are_charged_once(self):
        self.queue.record_deviation("prequeue-light-tests", 180, "synthetic engineering checks")
        self.queue.record_deviation("prequeue-light-tests", 180, "synthetic engineering checks")
        state = self.queue.status()
        self.assertEqual(state["used_cpu_s"], 180)
        with self.assertRaises(ValueError):
            self.queue.record_deviation("prequeue-light-tests", 181, "changed charge")

    def test_prelock_accepts_one_of_multiple_registered_team_tests(self):
        law = self.root / "law.json"
        law.write_text("{}")
        first = self.root / "test_first.py"
        first.write_text("# first\n")
        second = self.root / "other_worktree" / "results" / "pcrl_method_portfolio_v1" / "teams" / "b" / "test_second.py"
        second.parent.mkdir(parents=True)
        second.write_text("# second\n")
        lock = self.root / "PHASE1_LOCK_AMENDMENT.json"
        lock.write_text(json.dumps({
            "shared_laws": {"path": "law.json", "sha256": hashlib.sha256(law.read_bytes()).hexdigest()},
            "pre_algorithm_tests": {"team_b": [
                {"path": "test_first.py", "sha256": hashlib.sha256(first.read_bytes()).hexdigest()},
                {"path": "results/pcrl_method_portfolio_v1/teams/b/test_second.py", "sha256": hashlib.sha256(second.read_bytes()).hexdigest()},
            ]},
        }))
        self.queue.enqueue(dict(self.spec, unit_id="b_second", mode="prelock_synthetic",
                                team="team_b", data_access="NONE", test_file=str(second),
                                test_sha256=hashlib.sha256(second.read_bytes()).hexdigest()))
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, phase1_lock_path=str(lock),
                               phase1_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
        self.assertEqual(self.queue.run_once()["status"], "done")

    def test_orphaned_running_claim_blocks_new_admission(self):
        self.queue.enqueue(dict(self.spec, unit_id="second"))
        self.queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True, external_heavy=0)
        state = self.queue.status()
        state["units"][self.spec["unit_id"]].update(status="running", slot=0, worker_pid=999999)
        self.queue._save(state)
        self.assertEqual(self.queue.run_once()["status"], "orphan_unresolved")
        self.assertFalse(self.marker.exists())

    def test_stale_lock_file_without_flock_does_not_block(self):
        (self.root / "slot0.lock").write_text("stale text is not a lock")
        self.queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True, external_heavy=0)
        self.assertEqual(self.queue.run_once()["status"], "done")

    def test_concurrent_workers_start_one_unit_once(self):
        self.queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True, external_heavy=0)
        script = str(Path(__file__).parent / "compute_queue.py")
        argv = [sys.executable, script, "--root", str(self.root), "run-once"]
        workers = [subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
        for worker in workers:
            _, err = worker.communicate(timeout=10)
            self.assertEqual(worker.returncode, 0, err.decode())
        receipts = [json.loads(line) for line in (self.root / "RECEIPTS.jsonl").read_text().splitlines()]
        self.assertEqual(len([r for r in receipts if r["event"] == "start"]), 1)
        self.assertEqual(self.queue.status()["units"][self.spec["unit_id"]]["status"], "done")

    def test_killed_worker_keeps_child_slot_and_then_fails_closed(self):
        self.spec["argv"] = [sys.executable, "-c", "import time; time.sleep(1.2)"]
        # Recreate the queue because unit specs are immutable.
        root = self.root / "crash"
        queue = Queue(root)
        spec = dict(self.spec, unit_id="crash_job", cwd=str(root), max_wall_s=3, cpu_reserve_s=3)
        queue.enqueue(spec)
        queue.enqueue(dict(self.spec, unit_id="later_job", cwd=str(root), max_wall_s=3, cpu_reserve_s=3))
        queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True, external_heavy=1,
                          phase1_lock_path=str(self.phase1_lock),
                          phase1_lock_sha256=hashlib.sha256(self.phase1_lock.read_bytes()).hexdigest())
        script = str(Path(__file__).parent / "compute_queue.py")
        worker = subprocess.Popen([sys.executable, script, "--root", str(root), "run-once"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: worker.poll() is None and worker.kill())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if any(r.get("event") == "child_started" for r in
                   (json.loads(line) for line in (root / "RECEIPTS.jsonl").read_text().splitlines())):
                break
            time.sleep(0.01)
        else:
            self.fail("child did not start")
        os.kill(worker.pid, signal.SIGKILL)
        worker.communicate(timeout=3)
        self.assertEqual(queue.run_once()["status"], "portfolio_capacity_full")
        time.sleep(1.3)
        self.assertEqual(queue.run_once()["status"], "orphan_unresolved")
        self.assertEqual(queue.status()["units"]["later_job"]["status"], "planned")

    def test_continuation_budget_preserves_prior_charges(self):
        self.queue.record_deviation("prior", 180, "prior engineering")
        self.queue.begin_continuation("2026-10-08T15:10:00+00:00")
        state = self.queue.status()
        self.assertEqual(state["used_cpu_s"], 180)
        self.assertEqual(state["budget_epoch"]["start_utc"], "2026-10-08T15:10:00+00:00")
        self.assertEqual(state["budget_epoch"]["used_cpu_s"], 0)
        self.queue.record_deviation("continuation", 2, "queue bootstrap")
        state = self.queue.status()
        self.assertEqual(state["used_cpu_s"], 182)
        self.assertEqual(state["budget_epoch"]["used_cpu_s"], 2)

    def test_reviewed_single_role_export_has_separate_gate(self):
        self._suppress_default_fixture()
        script = self.root / "export_roles.py"
        script.write_text(f"from pathlib import Path\nPath({str(self.marker)!r}).write_text('exported')\n")
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        export = dict(self.spec, unit_id="role_export", mode="role_export",
                      argv=[sys.executable, str(script)],
                      data_access="S4_REVIEWED_SINGLE_PASS_SOURCE_ZIP",
                      reviewed_script=str(script), reviewed_sha256=digest)
        self.queue.enqueue(export)
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0)
        self.assertEqual(self.queue.run_once()["status"], "export_review_missing")
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, export_authorized=True,
                               export_reviewed_sha256=digest)
        self.assertEqual(self.queue.run_once()["status"], "done")
        self.assertEqual(self.marker.read_text(), "exported")

    def test_prelock_blocks_changed_registered_source(self):
        law = self.root / "law.json"
        law.write_text("{}")
        test_file = self.root / "test_source.py"
        test_file.write_text("# unchanged test\n")
        source = self.root / "results/pcrl_method_portfolio_v1/teams/a/algorithm.py"
        source.parent.mkdir(parents=True)
        source.write_text("# original source\n")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        lock = self.root / "PHASE1_QUEUE_INDEX.json"
        lock.write_text(json.dumps({
            "shared_laws": {"path": "law.json", "sha256": hashlib.sha256(law.read_bytes()).hexdigest()},
            "pre_algorithm_tests": {"A": [{"path": "test_source.py", "sha256": hashlib.sha256(test_file.read_bytes()).hexdigest()}]},
            "source_code_hashes": {"A": [{"path": "teams/a/algorithm.py", "sha256": source_hash}]},
        }))
        self.queue.enqueue(dict(self.spec, unit_id="changed_source", mode="prelock_synthetic",
                                team="A", data_access="NONE", test_file=str(test_file),
                                test_sha256=hashlib.sha256(test_file.read_bytes()).hexdigest()))
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, phase1_lock_path=str(lock),
                               phase1_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
        source.write_text("# changed source\n")
        self.assertEqual(self.queue.run_once()["status"], "phase1_source_changed")
        self.assertFalse(self.marker.exists())

    def test_independent_role_audit_requires_export_and_exact_reviewed_script(self):
        self._suppress_default_fixture()
        script = self.root / "audit.py"
        script.write_text(f"from pathlib import Path\nPath({str(self.marker)!r}).write_text('audited')\n")
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        audit = dict(self.spec, unit_id="role_audit", argv=[sys.executable, str(script)],
                     mode="role_audit", data_access="IDENTITY_ONLY_SOURCE_AND_ALLOWED_ROLE_EXPORT",
                     reviewed_script=str(script), reviewed_sha256=digest)
        self.queue.enqueue(audit)
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, audit_authorized=True, audit_reviewed_sha256=digest)
        self.assertEqual(self.queue.run_once()["status"], "export_not_complete")
        state = self.queue.status()
        state["units"]["S2-SAFE-ROLE-EXPORT-V1"] = {"status": "done", "spec": {"mode": "role_export"}}
        self.queue._save(state)
        self.assertEqual(self.queue.run_once()["status"], "done")
        self.assertEqual(self.marker.read_text(), "audited")

    def test_hash_registered_preseal_smoke_can_reserve_longer_synthetic_run(self):
        law = self.root / "law.json"
        law.write_text("{}")
        script = self.root / "smoke.py"
        script.write_text("# synthetic timing only\n")
        lock = self.root / "PHASE1_QUEUE_INDEX.json"
        lock.write_text(json.dumps({
            "shared_laws": {"path": "law.json", "sha256": hashlib.sha256(law.read_bytes()).hexdigest()},
            "pre_algorithm_tests": {"C": [{"path": "smoke.py", "sha256": hashlib.sha256(script.read_bytes()).hexdigest()}]},
        }))
        self.queue.enqueue(dict(self.spec, unit_id="long_smoke", mode="preseal_smoke",
                                data_access="NONE", team="C", test_file=str(script),
                                test_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                                max_wall_s=900, cpu_reserve_s=900, max_rss_gib=2.0))
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, phase1_lock_path=str(lock),
                               phase1_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
        self.assertEqual(self.queue.run_once()["status"], "done")

    def test_role_only_byte_replay_requires_independent_audit(self):
        self._suppress_default_fixture()
        script = self.root / "replay.py"
        script.write_text(f"from pathlib import Path\nPath({str(self.marker)!r}).write_text('replayed')\n")
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        replay = dict(self.spec, unit_id="role_replay", argv=[sys.executable, str(script)],
                      mode="role_replay", data_access="ADMITTED_ROLE_EXPORT_ONLY",
                      reviewed_script=str(script), reviewed_sha256=digest)
        self.queue.enqueue(replay)
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, replay_authorized=True, replay_reviewed_sha256=digest)
        self.assertEqual(self.queue.run_once()["status"], "role_audit_not_complete")
        state = self.queue.status()
        state["units"]["S4-INDEPENDENT-ROLE-AUDIT-V1"] = {"status": "done", "spec": {"mode": "role_audit"}}
        self.queue._save(state)
        self.assertEqual(self.queue.run_once()["status"], "done")

    def test_second_export_requires_preserved_first_and_exact_original_script(self):
        self._suppress_default_fixture()
        script = self.root / "export_once.py"
        script.write_text(f"from pathlib import Path\nPath({str(self.marker)!r}).write_text('second')\n")
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        first = self.root / "first"
        second = self.root / "second"
        replay = dict(self.spec, unit_id="export_replay", argv=[sys.executable, str(script)],
                      mode="role_export_replay", data_access="S4_REVIEWED_REPEAT_FOR_BYTE_DETERMINISM",
                      reviewed_script=str(script), reviewed_sha256=digest,
                      preserved_first_path=str(first), replay_output_path=str(second))
        self.queue.enqueue(replay)
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, export_replay_authorized=True,
                               export_replay_reviewed_sha256=digest)
        state = self.queue.status()
        state["units"]["S2-SAFE-ROLE-EXPORT-V1"] = {
            "status": "done", "spec": {"mode": "role_export", "reviewed_sha256": digest}}
        self.queue._save(state)
        self.assertEqual(self.queue.run_once()["status"], "first_export_not_preserved")
        first.mkdir()
        self.assertEqual(self.queue.run_once()["status"], "done")

    def test_independent_replay_compare_requires_second_export(self):
        self._suppress_default_fixture()
        script = self.root / "compare.py"
        script.write_text(f"from pathlib import Path\nPath({str(self.marker)!r}).write_text('compared')\n")
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        compare = dict(self.spec, unit_id="compare", argv=[sys.executable, str(script)],
                       mode="role_compare", data_access="TWO_ALLOWED_EXPORT_TREES_ONLY",
                       reviewed_script=str(script), reviewed_sha256=digest,
                       max_wall_s=60, cpu_reserve_s=60, max_rss_gib=0.5)
        self.queue.enqueue(compare)
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, compare_authorized=True,
                               compare_reviewed_sha256=digest)
        self.assertEqual(self.queue.run_once()["status"], "second_export_not_complete")
        state = self.queue.status()
        state["units"]["S2-SAFE-ROLE-EXPORT-REPLAY-V1"] = {
            "status": "done", "spec": {"mode": "role_export_replay"}}
        self.queue._save(state)
        self.assertEqual(self.queue.run_once()["status"], "done")

    def test_orphan_recovery_charges_ceiling_and_never_restarts_same_unit(self):
        self.queue.enqueue(dict(self.spec, unit_id="later"))
        self.queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True,
                               external_heavy=0)
        state = self.queue.status()
        state["units"][self.spec["unit_id"]].update(status="running", slot=0,
                                                     worker_pid=999999, child_pgid=999998)
        self.queue._save(state)
        self.assertEqual(self.queue.run_once()["status"], "orphan_unresolved")
        result = self.queue.recover_orphan(self.spec["unit_id"], "owned synthetic worker crashed")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(self.queue.status()["used_cpu_s"], self.spec["cpu_reserve_s"])
        self.assertEqual(self.queue.run_once()["status"], "done")
        self.assertEqual(self.queue.status()["units"][self.spec["unit_id"]]["status"], "incomplete")

    def test_local_git_seal_needs_committed_lock_and_s4_review(self):
        repo = self.root / "repo"
        campaign = repo / "results/pcrl_method_portfolio_v1"
        campaign.mkdir(parents=True)
        marker = self.root / "sealed_ran.txt"
        script = campaign / "teams/b/screen_driver.py"
        script.parent.mkdir(parents=True)
        script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ok')\n")
        script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
        manifest = campaign / "DATA_ACCESS_MANIFEST.json"
        manifest.write_text('{"status":"ROLE_FILES_ADMITTED_S4_VERIFIED"}')
        audit = campaign / "S4_ROLE_EXPORT_VERIFICATION.json"
        audit.write_text('{"status":"PASS"}')
        lock = campaign / "CAMPAIGN_LOCK.json"
        lock.write_text(json.dumps({
            "status": "LOCAL_GIT_SEALED_EFFECTIVE", "adult_fit_permission": True,
            "fit_admission": "B_ONLY", "admitted_arms": {"B": ["B-JVIB"]},
            "families": {k: ({"fit_admission": "ALLOWED",
                              "screen_unit_ids": {"B-JVIB:s0": "real_fit_fixture",
                                                  "B-JVIB:s1": "later_real_fit_fixture"},
                              "real_control_unit_ids": {},
                              "screen_driver_sha256": script_sha}
                             if k == "B" else {"fit_admission": "DENIED"})
                         for k in "ABCD"},
            "data_admission": {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "s4_audit_receipt_sha256": hashlib.sha256(audit.read_bytes()).hexdigest()},
        }))
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "results"], check=True)
        env = dict(os.environ, GIT_AUTHOR_NAME="Queue Test", GIT_AUTHOR_EMAIL="queue@test.invalid",
                   GIT_COMMITTER_NAME="Queue Test", GIT_COMMITTER_EMAIL="queue@test.invalid")
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seal fixture"], check=True, env=env)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
        review = campaign / "S4_PREFIT_SEAL_REVIEW.json"
        review.write_text(json.dumps({"status": "PASS", "fit_admission": "B_ONLY", "lock_commit_sha": commit,
                                      "lock_sha256": lock_hash}))
        receipt = campaign / "LOCAL_SEAL_RECEIPT.json"
        receipt.write_text(json.dumps({"status": "PASS", "fit_admission": "B_ONLY", "commit_sha": commit,
                                       "lock_commit_sha": commit,
                                       "campaign_lock_sha256": lock_hash,
                                       "lock_sha256": lock_hash,
                                       "review_path": "S4_PREFIT_SEAL_REVIEW.json",
                                       "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest()}))
        queue = Queue(self.root / "sealed_queue")
        unit = dict(self.spec, unit_id="real_fit_fixture", mode="real_data", family="B", arm_id="B-JVIB",
                    team="B", arm="B-JVIB", seed=0, stage="screen", cwd=str(repo),
                    argv=[sys.executable, str(script), "--arm", "B-JVIB", "--seed", "0",
                          "--unit-id", "real_fit_fixture"])
        queue.update_gate(safe_inputs=True, lock_pushed=False, occupancy_confirmed=True,
                          external_heavy=0, local_seal_path=str(receipt),
                          local_seal_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
        queue.enqueue(unit)
        self.assertEqual(queue.run_once()["status"], "done")
        self.assertEqual(marker.read_text(), "ok")
        queue.enqueue(dict(unit, unit_id="later_real_fit_fixture", seed=1,
                           argv=[sys.executable, str(script), "--arm", "B-JVIB", "--seed", "1",
                                 "--unit-id", "later_real_fit_fixture"]))
        complete_receipt = json.loads(receipt.read_text())
        for missing_key in ("lock_commit_sha", "lock_sha256"):
            incomplete = dict(complete_receipt)
            del incomplete[missing_key]
            receipt.write_text(json.dumps(incomplete))
            queue.update_gate(safe_inputs=True, lock_pushed=False, occupancy_confirmed=True,
                              external_heavy=0, local_seal_path=str(receipt),
                              local_seal_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
            self.assertEqual(queue.run_once()["status"], "local_seal_invalid")
        receipt.write_text(json.dumps(complete_receipt))
        queue.update_gate(safe_inputs=True, lock_pushed=False, occupancy_confirmed=True,
                          external_heavy=0, local_seal_path=str(receipt),
                          local_seal_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
        lock.write_text(lock.read_text() + "\n")
        self.assertEqual(queue.run_once()["status"], "local_seal_invalid")

    def test_b_only_seal_rejects_non_b_and_unlisted_arms_at_enqueue(self):
        repo = self.root / "b_only_repo"
        campaign = repo / "results/pcrl_method_portfolio_v1"
        campaign.mkdir(parents=True)
        marker = self.root / "b_only_ran.txt"
        script = campaign / "teams/b/screen_driver.py"
        script.parent.mkdir(parents=True)
        script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
        script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
        manifest = campaign / "DATA_ACCESS_MANIFEST.json"
        manifest.write_text('{"status":"ROLE_FILES_ADMITTED_S4_VERIFIED"}')
        audit = campaign / "S4_ROLE_EXPORT_VERIFICATION.json"
        audit.write_text('{"status":"PASS"}')
        lock = campaign / "CAMPAIGN_LOCK.json"
        lock.write_text(json.dumps({
            "status": "LOCAL_GIT_SEALED_EFFECTIVE", "adult_fit_permission": True,
            "fit_admission": "B_ONLY",
            "admitted_arms": {"B": ["B-JVIB", "B-TASK"]},
            "families": {k: ({"fit_admission": "ALLOWED",
                              "screen_unit_ids": {"B-JVIB:s0": "b_admitted"},
                              "real_control_unit_ids": {},
                              "screen_driver_sha256": script_sha,
                              "real_control_probe_sha256": "1" * 64}
                             if k == "B" else {"fit_admission": "DENIED"})
                         for k in "ABCD"},
            "data_admission": {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "s4_audit_receipt_sha256": hashlib.sha256(audit.read_bytes()).hexdigest()},
        }))
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "results"], check=True)
        env = dict(os.environ, GIT_AUTHOR_NAME="Queue Test", GIT_AUTHOR_EMAIL="queue@test.invalid",
                   GIT_COMMITTER_NAME="Queue Test", GIT_COMMITTER_EMAIL="queue@test.invalid")
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "b-only fixture"], check=True, env=env)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
        review = campaign / "S4_PREFIT_SEAL_REVIEW.json"
        review.write_text(json.dumps({"status": "PASS", "fit_admission": "B_ONLY",
                                      "lock_commit_sha": commit, "lock_sha256": lock_sha}))
        receipt = campaign / "LOCAL_SEAL_RECEIPT.json"
        receipt.write_text(json.dumps({"status": "PASS", "fit_admission": "B_ONLY",
                                       "commit_sha": commit, "lock_commit_sha": commit,
                                       "campaign_lock_sha256": lock_sha, "lock_sha256": lock_sha,
                                       "review_path": review.name,
                                       "review_sha256": hashlib.sha256(review.read_bytes()).hexdigest()}))
        queue = Queue(self.root / "b_only_queue")
        queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True,
                          external_heavy=0, local_seal_path=str(receipt),
                          local_seal_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
        base = dict(self.spec, mode="real_data", cwd=str(repo),
                    team="B", arm="B-JVIB", seed=0, stage="screen",
                    argv=[sys.executable, str(script), "--arm", "B-JVIB", "--seed", "0",
                          "--unit-id", "b_admitted"])
        with self.assertRaises(ValueError):
            queue.enqueue(dict(base, unit_id="a_denied", family="A", arm_id="A-JOINT-STOCH"))
        with self.assertRaises(ValueError):
            queue.enqueue(dict(base, unit_id="b_unlisted", family="B", arm_id="B-RAWJ"))
        with self.assertRaises(ValueError):
            queue.enqueue(dict(base, unit_id="missing_scope"))
        with self.assertRaises(ValueError):
            queue.enqueue(dict(base, unit_id="extra_b_fit", family="B", arm_id="B-JVIB",
                               team="B", arm="B-JVIB", seed=0, stage="screen"))
        with self.assertRaises(ValueError):
            queue.enqueue(dict(base, unit_id="wrong_stage", family="B", arm_id="B-JVIB",
                               team="B", arm="B-JVIB", seed=0, stage="postrelease_control"))
        with self.assertRaises(ValueError):
            queue.enqueue(dict(base, unit_id="wrong_seed", family="B", arm_id="B-JVIB",
                               team="B", arm="B-JVIB", seed=1, stage="screen"))
        with self.assertRaises(ValueError):
            queue.enqueue(dict(base, unit_id="mismatched_team", family="B", arm_id="B-JVIB",
                               team="A", arm="B-JVIB"))
        with self.assertRaises(ValueError):
            queue.enqueue(dict(base, unit_id="mismatched_arm", family="B", arm_id="B-JVIB",
                               team="B", arm="B-TASK"))
        with self.assertRaises(ValueError):
            queue.enqueue(dict(base, unit_id="script_only_decoy", family="B", arm_id="B-JVIB",
                               argv=[sys.executable, "-c", "pass", str(script)]))
        with self.assertRaises(ValueError):
            queue.enqueue(dict(base, unit_id="b_admitted", family="B", arm_id="B-JVIB",
                               argv=[sys.executable, str(script), "--arm", "B-JVIB", "--seed", "1",
                                     "--unit-id", "b_admitted"]))
        queue.enqueue(dict(base, unit_id="b_admitted", family="B", arm_id="B-JVIB"))
        self.assertEqual(queue.run_once()["status"], "done")
        self.assertEqual(marker.read_text(), "ran")
        # A previously queued non-B unit must also be refused at launch.
        state = queue.status()
        state["units"]["a_legacy"] = {"spec": dict(base, unit_id="a_legacy", family="A",
                                                     team="A", arm="A-JOINT-STOCH",
                                                     arm_id="A-JOINT-STOCH"), "status": "planned"}
        queue._save(state)
        self.assertEqual(queue.run_once()["status"], "family_not_admitted")

    def test_role_probe_requires_admitted_role_manifest_and_audit(self):
        self._suppress_default_fixture()
        script = self.root / "results/pcrl_method_portfolio_v1/shared/probe_att_group_split.py"
        script.parent.mkdir(parents=True)
        script.write_text(f"from pathlib import Path\nPath({str(self.marker)!r}).write_text('probed')\n")
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        sex_file = self.root / "shared/.private/role_export_v1/ATTACK_FIT_NEW/sex.npz"
        sex_file.parent.mkdir(parents=True)
        sex_file.write_bytes(b"fixed admitted role fixture")
        manifest = self.root / "DATA_ACCESS_MANIFEST.json"
        manifest.write_text(json.dumps({"status":"ROLE_FILES_ADMITTED_S4_VERIFIED",
                                        "roles":{"ATTACK_FIT_NEW":{"rows":4}},
                                        "real_file_specs":{"ATTACK_FIT_NEW:sex":{
                                            "role":"ATTACK_FIT_NEW", "field":"sex",
                                            "members":["row_id","unit","sex"],
                                            "path":str(sex_file.relative_to(self.root)),
                                            "sha256":hashlib.sha256(sex_file.read_bytes()).hexdigest()}}}))
        audit = self.root / "S4_ROLE_EXPORT_VERIFICATION.json"
        audit.write_text('{"status":"PASS","roles":{"ATTACK_FIT_NEW":{"rows":4}}}')
        index = json.loads(self.phase1_lock.read_text())
        index["source_code_hashes"] = {"ATT_PROBE": [{"path":"shared/probe_att_group_split.py",
                                                       "sha256":digest}]}
        self.phase1_lock.write_text(json.dumps(index))
        probe = dict(self.spec, unit_id="att_role_probe", mode="role_probe",
                     team="ATT_PROBE", cwd=str(self.root),
                     argv=[sys.executable, str(script)],
                     data_access="ADMITTED_ATTACK_FIT_SEX_ONLY",
                     reviewed_script=str(script), reviewed_sha256=digest,
                     admission_manifest_path=str(manifest),
                     admission_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                     s4_audit_path=str(audit),
                     s4_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
        self.queue.enqueue(probe)
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, phase1_lock_path=str(self.phase1_lock),
                               phase1_lock_sha256=hashlib.sha256(self.phase1_lock.read_bytes()).hexdigest())
        self.assertEqual(self.queue.run_once()["status"], "role_audit_not_complete")
        state = self.queue.status()
        state["units"]["S4-INDEPENDENT-ROLE-AUDIT-V1"] = {"spec": {}, "status": "done"}
        self.queue._save(state)
        self.assertEqual(self.queue.run_once()["status"], "done")
        self.assertEqual(self.marker.read_text(), "probed")

    def test_shared_anchor_binds_seed_id_script_and_argv(self):
        repo = self.root / "anchor_repo"
        script = repo / "results/pcrl_method_portfolio_v1/shared/run_feature_ref_raw83_u.py"
        script.parent.mkdir(parents=True)
        script.write_text("# synthetic anchor script\n")
        lock = script.parent.parent / "CAMPAIGN_LOCK.json"
        lock.write_text(json.dumps({
            "fit_admission": "B_ONLY",
            "families": {k: {"fit_admission": "ALLOWED" if k == "B" else "DENIED"}
                         for k in "ABCD"},
            "admitted_shared_unit_ids": ["S2-FEATURE-REF-RAW83-U-S0", "S2-FEATURE-REF-RAW83-U-S1"],
            "shared_anchor_runner_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        }))
        gate = {"safe_inputs": True, "local_seal_path": str(lock.parent / "LOCAL_SEAL_RECEIPT.json")}
        spec = dict(self.spec, mode="real_data", cwd=str(repo), family="SHARED", team="SHARED",
                    arm_id="FEATURE_REF_RAW83_U", arm="FEATURE_REF_RAW83_U", seed=0,
                    stage="feature_anchor", unit_id="S2-FEATURE-REF-RAW83-U-S0",
                    argv=[sys.executable, str(script), "--seed", "0"])
        with mock.patch.object(Queue, "_local_seal_ok", return_value=True):
            self.assertIsNone(Queue._real_admission_reason(gate, spec))
            self.assertEqual(Queue._real_admission_reason(gate, dict(spec, argv=[sys.executable, str(script), "--seed", "1"])),
                             "shared_seed_argv_mismatch")
            self.assertEqual(Queue._real_admission_reason(gate, dict(spec, unit_id="unlisted")),
                             "shared_unit_not_admitted")
            self.assertEqual(Queue._real_admission_reason(gate, dict(spec, argv=[sys.executable, "-c", "pass", str(script), "--seed", "0"])),
                             "shared_script_not_admitted")


if __name__ == "__main__":
    unittest.main()
