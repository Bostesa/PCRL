"""Data-free R2 replay queue proof with committed synthetic ancestry."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest

from compute_queue import (Queue, S4_C_ARMS, S4_C_PARENT_LABELS,
                           S4_C_R2_SOURCE_NAMES, canonical, sha256_file)
from test_compute_queue_s4_c_adult_green import (S4CReplayQueueTests,
                                                 THREAD_NAMES, digest, write_json)


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
R2_DRAFT = CAMPAIGN / "S4_C_R2_REPLAY_ADMISSION_DRAFT.json"
R2_SCHEMA = CAMPAIGN / "s4/S4_C_R2_REPLAY_UNIT_SCHEMA.json"
R2_BUILDER = CAMPAIGN / "s4/build_c_arm_result_registry_v2.py"


class R2ReplayRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        # Reuse the reviewed R1 synthetic Git/parent fixture without inheriting
        # its tests. All files are disposable; no role archive is opened.
        old = S4CReplayQueueTests("test_exact_arm_enqueues_and_mutations_are_denied")
        old.setUp()
        self.addCleanup(old.doCleanups)
        self.old = old
        self.repo, self.campaign, self.s4 = old.repo, old.campaign, old.s4
        self.queue, self.venv = old.queue, old.venv
        self._historical_failure_and_diagnostic()

        self.schema = self.s4 / R2_SCHEMA.name
        self.builder = self.s4 / R2_BUILDER.name
        shutil.copy2(R2_SCHEMA, self.schema)
        shutil.copy2(R2_BUILDER, self.builder)
        self.sources = {}
        for name in S4_C_R2_SOURCE_NAMES:
            path = self.s4 / f"{name}.py"
            if not path.exists():
                path.write_text(f"# synthetic R2 {name} identity\n")
            self.sources[f"results/pcrl_method_portfolio_v1/s4/{name}.py"] = sha256_file(path)
        copied_queue = self.campaign / "shared/compute_queue.py"
        copied_queue.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / "compute_queue.py", copied_queue)
        old._commit("synthetic R2 queue and source identity")
        self.queue_commit = old._git_output("rev-parse", "HEAD")

        draft = json.loads(R2_DRAFT.read_text())
        self.units = deepcopy(draft["unit_specs"])
        # Synthetic parent bytes replace the real metadata hashes for FULL:S0.
        old_uid = "S4-C-R1-C-SIMPLE-FULL-S0"
        self.arm_id = "S4-C-R2-C-SIMPLE-FULL-S0"
        self.units[self.arm_id]["parents"] = deepcopy(old.units[old_uid]["parents"])
        self.units["S4-C-R2-AGGREGATE"]["parents"]["c_finalizer"] = deepcopy(
            old.units["S4-C-R1-AGGREGATE"]["parents"]["c_finalizer"])
        self.admission = self.campaign / "S4_C_R2_REPLAY_ADMISSION.json"
        self.admission_data = {
            **{k: draft[k] for k in (
                "schema", "scope", "lock_sha256", "lock_commit_sha",
                "fit_seal_relative_path", "fit_review_relative_path",
                "data_manifest_sha256", "parent_c_unit_count", "unit_count",
                "cpu_reserve_s")},
            "status": "LOCAL_GIT_SEALED_EFFECTIVE", "adult_role_access": True,
            "fit_seal_sha256": sha256_file(old.fit_seal),
            "fit_review_sha256": sha256_file(old.fit_review),
            "lock_sha256": old.fit_sha, "lock_commit_sha": old.fit_commit,
            "data_manifest_sha256": sha256_file(old.manifest),
            "unit_schema_sha256": sha256_file(self.schema),
            "source_sha256_by_path": self.sources,
            "queue_source_sha256": sha256_file(HERE / "compute_queue.py"),
            "queue_source_commit_sha": self.queue_commit,
            "post_arm_registry_builder_sha256": sha256_file(self.builder),
            "unit_specs": self.units,
            "prior_r1_admission_sha256": sha256_file(old.admission),
            "prior_r1_review_sha256": sha256_file(old.review),
            "prior_r1_seal_sha256": sha256_file(old.seal),
            "prior_r1_failed_queue_receipt_sha256": sha256_file(self.failed_receipt),
            "diagnostic_v2_admission_sha256": sha256_file(self.diag_admission),
            "diagnostic_v2_review_sha256": sha256_file(self.diag_review),
            "diagnostic_v2_seal_sha256": sha256_file(self.diag_seal),
            "diagnostic_v2_queue_receipt_sha256": sha256_file(self.diag_receipt),
            "diagnostic_v2_report_sha256": sha256_file(self.diag_report),
        }
        write_json(self.admission, self.admission_data)
        old._commit("synthetic R2 admission")
        admission_commit = old._git_output("rev-parse", "HEAD")
        self.review = self.campaign / "S4_C_R2_REPLAY_PREFIT_REVIEW.json"
        write_json(self.review, {
            "schema": "pcrl-s4-c-r2-replay-prefit-review-v2", "status": "PASS",
            "scope": "S4_C_R2_ADULT_REPLAY", "admission_sha256": sha256_file(self.admission),
            "admission_commit_sha": admission_commit,
        })
        self.seal = self.campaign / "S4_C_R2_REPLAY_LOCAL_SEAL_RECEIPT.json"
        write_json(self.seal, {
            "schema": "pcrl-s4-c-r2-replay-local-seal-v2", "status": "PASS",
            "scope": "S4_C_R2_ADULT_REPLAY", "admission_sha256": sha256_file(self.admission),
            "review_sha256": sha256_file(self.review),
            "admission_commit_sha": admission_commit,
        })
        with self.queue._mutex():
            state = self.queue._read()
            state["gate"].update(
                s4_c_r2_local_seal_path=str(self.seal),
                s4_c_r2_local_seal_sha256=sha256_file(self.seal))
            self.queue._save(state)

    def _historical_failure_and_diagnostic(self) -> None:
        old = self.old
        self.failed_receipt = self.queue.root / "units/S4-C-R1-C-SIMPLE-FULL-S0/RECEIPT.json"
        write_json(self.failed_receipt, {
            "unit_id": "S4-C-R1-C-SIMPLE-FULL-S0", "status": "failed",
            "returncode": 1, "output_sha256": {},
        })
        diag_id = "S4-C-R1-READER-BANK-DIAG-FULL-S0-R1-R2"
        self.diag_report = (self.s4 / ".private/c_r1_reader_bank_diag_v2" /
                            f"{diag_id}.json")
        write_json(self.diag_report, {
            "status": "DIAGNOSTIC_RECORDED_NO_TOLERANCE_DECISION",
            "summary": {"first_mismatch_reader": "RAW_LR_C1",
                        "exact_mismatch_readers": ["RAW_LR_C1"]},
        })
        self.diag_receipt = self.queue.root / "units" / diag_id / "RECEIPT.json"
        write_json(self.diag_receipt, {
            "unit_id": diag_id, "status": "done", "returncode": 0,
            "output_sha256": {str(self.diag_report): sha256_file(self.diag_report)},
        })
        self.diag_admission = self.campaign / "S4_C_READER_BANK_DIAGNOSTIC_V2_ADMISSION.json"
        source_map = {}
        for name in ("run_c_reader_bank_diagnostic_v2", "run_c_adult_replay_v1",
                     "replay_c_readers_v1", "replay_c_release_v1", "replay_c_values_v1"):
            path = self.s4 / f"{name}.py"
            if not path.exists():
                path.write_text(f"# synthetic {name} identity\n")
            source_map[f"results/pcrl_method_portfolio_v1/s4/{name}.py"] = sha256_file(path)
        write_json(self.diag_admission, {
            "schema": "pcrl-s4-c-reader-bank-diagnostic-admission-v2",
            "status": "LOCAL_GIT_SEALED_EFFECTIVE",
            "data_manifest_sha256": sha256_file(old.manifest),
            "source_sha256_by_path": source_map,
        })
        old._commit("synthetic diagnostic ancestry")
        diag_commit = old._git_output("rev-parse", "HEAD")
        self.diag_review = self.campaign / "S4_C_READER_BANK_DIAGNOSTIC_V2_PREFIT_REVIEW.json"
        write_json(self.diag_review, {
            "status": "PASS", "admission_sha256": sha256_file(self.diag_admission),
        })
        self.diag_seal = self.campaign / "S4_C_READER_BANK_DIAGNOSTIC_V2_LOCAL_SEAL_RECEIPT.json"
        write_json(self.diag_seal, {
            "status": "PASS", "admission_sha256": sha256_file(self.diag_admission),
            "review_sha256": sha256_file(self.diag_review),
            "admission_commit_sha": diag_commit,
        })

    def _spec(self, uid: str, registry_sha: str | None = None) -> dict:
        unit = self.units[uid]
        aggregate = uid == "S4-C-R2-AGGREGATE"
        runner = self.s4 / ("run_c_aggregate_replay_v2.py" if aggregate else
                            "run_c_adult_replay_v2.py")
        report = self.s4 / ".private/c_r2_replay_v2" / f"{uid}.json"
        deps = ([unit["parents"]["c_finalizer"]["unit_id"],
                 *unit["parents"]["arm_reports"]] if aggregate else
                [unit["parents"][label]["unit_id"] for label in S4_C_PARENT_LABELS])
        argv = [str(self.venv), str(runner), "--unit-id", uid,
                "--admission", str(self.admission),
                "--admission-sha256", sha256_file(self.admission),
                "--local-seal", str(self.seal),
                "--local-seal-sha256", sha256_file(self.seal),
                "--queue-root", str(self.queue.root),
                "--private-report", str(report)]
        if aggregate:
            argv += ["--arm-result-index",
                     str(self.campaign / "S4_C_R2_ARM_RESULT_HASH_REGISTRY.json"),
                     "--arm-result-index-sha256", registry_sha or ""]
        spec = {
            "unit_id": uid, "mode": "s4_c_r2_adult_replay", "team": "S4", "family": "S4",
            "stage": unit["stage"], "arm": unit["arm"], "arm_id": unit["arm"],
            "seed": unit["seed"], "data_access": unit["data_access"],
            "admission_unit_sha256": digest(canonical(unit).encode()),
            "deps": deps, "cwd": str(self.repo), "outputs": [str(report)],
            "max_wall_s": unit["max_wall_s"], "cpu_reserve_s": unit["cpu_reserve_s"],
            "max_rss_gib": unit["max_rss_gib"], "runner_sha256": sha256_file(runner),
            "env": {name: "1" for name in THREAD_NAMES} |
                   {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(self.s4)},
            "argv": argv,
        }
        if aggregate:
            spec["arm_result_index_sha256"] = registry_sha
        return spec

    def _completed_arms(self) -> str:
        units = {}
        for uid in self.units["S4-C-R2-AGGREGATE"]["parents"]["arm_reports"]:
            unit = self.units[uid]
            report = self.s4 / ".private/c_r2_replay_v2" / f"{uid}.json"
            artifact = report.with_suffix(".npz")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"synthetic artifact {uid}".encode())
            write_json(report, {
                "status": "REPLAYED_C_R2_PASS", "unit_id": uid,
                "arm": unit["arm"], "seed": unit["seed"],
                "verified_inner_artifacts": {
                    "filename": artifact.name, "sha256": sha256_file(artifact)},
            })
            entry = self.old._completed(uid, report, self._spec(uid))
            units[uid] = {"unit_id": uid,
                          "queue_receipt_sha256": entry["queue_receipt_sha256"],
                          "report_sha256": entry["output_sha256"],
                          "artifact_npz_sha256": sha256_file(artifact),
                          "status": "REPLAYED_C_R2_PASS"}
        registry = self.campaign / "S4_C_R2_ARM_RESULT_HASH_REGISTRY.json"
        write_json(registry, {
            "schema": "pcrl-s4-c-r2-arm-result-hash-registry-v2",
            "status": "S4_HASH_CHECKED_16_OF_16",
            "admission_sha256": sha256_file(self.admission),
            "fit_lock_sha256": self.old.fit_sha,
            "registry_builder_sha256": sha256_file(self.builder),
            "units": units,
        })
        self.old._commit("synthetic R2 arm result registry")
        with self.queue._mutex():
            state = self.queue._read()
            state["units"].update(self.old.parent_states)
            self.queue._save(state)
        return sha256_file(registry)

    def test_arm_scope_parents_and_enqueue(self) -> None:
        spec = self._spec(self.arm_id)
        state = self.queue.status()
        self.assertTrue(Queue._s4_c_r2_local_seal_ok(state["gate"]))
        self.assertIsNone(self.queue._s4_c_r2_admission_reason(state["gate"], spec))
        self.assertIsNone(self.queue._s4_c_r2_parent_reason(state, spec))
        self.assertIsNone(self.queue._gate_status(state, spec))
        self.assertEqual(self.queue.enqueue(spec)["status"], "planned")
        for key, value, reason in (
            ("data_access", "WRONG_ROLE", "s4_c_r2_unit_scope_mismatch"),
            ("argv", [*spec["argv"], "--extra"], "s4_c_r2_argv_mismatch"),
            ("outputs", [str(self.repo / "wrong.json")], "s4_c_r2_output_changed"),
        ):
            changed = deepcopy(spec)
            changed[key] = value
            self.assertEqual(self.queue._s4_c_r2_admission_reason(state["gate"], changed),
                             reason)
        parent_id = self.units[self.arm_id]["parents"]["reader_r1"]["unit_id"]
        receipt = self.queue.root / "units" / parent_id / "RECEIPT.json"
        receipt.write_text(receipt.read_text() + " ")
        self.assertEqual(self.queue._s4_c_r2_parent_reason(self.queue.status(), spec),
                         "s4_c_parent_not_complete")
        self.assertEqual(self.queue._gate_status(self.queue.status(), spec),
                         "s4_c_parent_not_complete")

    def test_aggregate_registry_required_at_enqueue_and_launch(self) -> None:
        uid = "S4-C-R2-AGGREGATE"
        missing = self._spec(uid)
        self.assertEqual(self.queue._s4_c_r2_admission_reason(self.queue.status()["gate"], missing),
                         "s4_c_r2_arm_result_registry_invalid")
        registry_sha = self._completed_arms()
        spec = self._spec(uid, registry_sha)
        state = self.queue.status()
        self.assertIsNone(self.queue._s4_c_r2_admission_reason(state["gate"], spec))
        self.assertIsNone(self.queue._s4_c_r2_parent_reason(state, spec))
        self.assertIsNone(self.queue._gate_status(state, spec))
        self.assertEqual(self.queue.enqueue(spec)["status"], "planned")
        registry = self.campaign / "S4_C_R2_ARM_RESULT_HASH_REGISTRY.json"
        changed = json.loads(registry.read_text())
        changed["status"] = "INCOMPLETE"
        write_json(registry, changed)
        self.assertEqual(self.queue._s4_c_r2_admission_reason(self.queue.status()["gate"], spec),
                         "s4_c_r2_arm_result_registry_invalid")
        self.assertEqual(self.queue._gate_status(self.queue.status(), spec),
                         "s4_c_r2_arm_result_registry_invalid")


if __name__ == "__main__":
    unittest.main()
