"""Data-free Git/seal and parent-custody proof for the S4 C replay queue lane."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compute_queue import Queue, S4_C_ARMS, S4_C_PARENT_LABELS, S4_C_SOURCE_NAMES, canonical, sha256_file


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
CANDIDATE = CAMPAIGN / "S4_C_REPLAY_ADMISSION_CANDIDATE_R2.json"
SCHEMA_SHA = "7f1bdacaaa920a252b0867c857f7c06aeb6625e9689d24f9962ce1830b0a04d7"
THREAD_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


class S4CReplayQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = (Path(self.temp.name) / "PCRL/.worktrees/coordinator").resolve()
        self.campaign = self.repo / "results/pcrl_method_portfolio_v1"
        self.s4 = self.campaign / "s4"
        self.team_c = self.repo.parent / "team-c/results/pcrl_method_portfolio_v1/teams/c"
        self.private_c = self.team_c / ".private/team_c/synthetic-lock"
        self.s4.mkdir(parents=True)
        self.private_c.mkdir(parents=True)
        self.queue = Queue(self.campaign / "shared/.private/compute")
        self.venv = self.repo.parent.parent / ".venv/bin/python"
        self.venv.parent.mkdir(parents=True, exist_ok=True)
        self.venv.symlink_to(sys.executable)
        self.c_worker = self.team_c / "run_c_worker.py"
        self.c_screen = self.team_c / "screen_orchestrator.py"
        self.c_worker.write_text("# synthetic C worker identity\n")
        self.c_screen.write_text("# synthetic C screen identity\n")
        self.sources = {}
        for name in S4_C_SOURCE_NAMES:
            path = self.s4 / f"{name}.py"
            path.write_text(f"# synthetic S4 {name} identity\n")
            self.sources[f"results/pcrl_method_portfolio_v1/s4/{name}.py"] = sha256_file(path)
        self.builder = self.s4 / "build_c_arm_result_registry_v1.py"
        self.builder.write_text("# synthetic post-arm registry builder\n")
        self.manifest = self.campaign / "DATA_ACCESS_MANIFEST.json"
        write_json(self.manifest, {"schema": "synthetic-role-manifest", "values": "none"})
        self.fit_path = self.campaign / "CAMPAIGN_LOCK.json"
        write_json(self.fit_path, {
            "fit_admission": "ALL_FAMILIES",
            "families": {
                "A": {"fit_admission": "DENIED"},
                "C": {"fit_admission": "ALLOWED",
                      "worker_runner_sha256": sha256_file(self.c_worker),
                      "screen_orchestration": {"source_sha256": sha256_file(self.c_screen)}},
                "D": {"fit_admission": "DENIED"},
            },
            "admitted_arms": {"C": list(S4_C_ARMS)},
            "data_access_manifest_sha256": sha256_file(self.manifest),
        })
        self._git("init", "-q")
        self._commit("synthetic historical C fit")
        self.fit_commit = self._git_output("rev-parse", "HEAD")
        self.fit_sha = sha256_file(self.fit_path)
        self.fit_review = self.campaign / "S4_PREFIT_SEAL_REVIEW.json"
        write_json(self.fit_review, {
            "schema": "pcrl-s4-prefit-seal-review-v1", "status": "PASS",
            "lock_commit_sha": self.fit_commit, "lock_sha256": self.fit_sha,
        })
        self.fit_seal = self.campaign / "LOCAL_SEAL_RECEIPT.json"
        write_json(self.fit_seal, {
            "status": "PASS", "commit_sha": self.fit_commit,
            "lock_sha256": self.fit_sha,
            "review_sha256": sha256_file(self.fit_review),
        })
        self.admission = self.campaign / "S4_C_REPLAY_ADMISSION.json"
        candidate = json.loads(CANDIDATE.read_text())
        self.units = deepcopy(candidate["unit_specs"])
        self.arm_id = "S4-C-R1-C-SIMPLE-FULL-S0"
        self._make_c_parents(self.arm_id)
        self._make_c_finalizer()
        self.admission_data = {
            "schema": "pcrl-s4-c-replay-admission-v1",
            "status": "LOCAL_GIT_SEALED_EFFECTIVE",
            "scope": "S4_C_R1_INDEPENDENT_ARM_AND_AGGREGATE_REPLAY",
            "adult_role_access": True,
            "lock_commit_sha": self.fit_commit, "lock_sha256": self.fit_sha,
            "fit_review_relative_path": self.fit_review.name,
            "fit_review_sha256": sha256_file(self.fit_review),
            "fit_seal_relative_path": self.fit_seal.name,
            "fit_seal_sha256": sha256_file(self.fit_seal),
            "data_manifest_sha256": sha256_file(self.manifest),
            "source_sha256_by_path": self.sources,
            "queue_source_sha256": sha256_file(HERE / "compute_queue.py"),
            "unit_schema_sha256": SCHEMA_SHA,
            "unit_count": 17, "parent_c_unit_count": 135,
            "cpu_reserve_s": 3900,
            "post_arm_registry_builder_sha256": sha256_file(self.builder),
            "unit_specs": self.units,
        }
        write_json(self.admission, self.admission_data)
        self._commit("synthetic S4 C admission")
        self.admission_commit = self._git_output("rev-parse", "HEAD")
        self.review = self.campaign / "S4_C_REPLAY_PREFIT_REVIEW.json"
        write_json(self.review, {
            "schema": "pcrl-s4-c-replay-prefit-review-v1", "status": "PASS",
            "scope": "S4_C_R1_ADULT_REPLAY",
            "admission_sha256": sha256_file(self.admission),
            "admission_commit_sha": self.admission_commit,
        })
        self.seal = self.campaign / "S4_C_REPLAY_LOCAL_SEAL_RECEIPT.json"
        write_json(self.seal, {
            "schema": "pcrl-s4-c-replay-local-seal-v1", "status": "PASS",
            "admission_sha256": sha256_file(self.admission),
            "review_sha256": sha256_file(self.review),
            "admission_commit_sha": self.admission_commit,
        })
        with self.queue._mutex():
            state = self.queue._read()
            state["gate"].update(safe_inputs=True, occupancy_confirmed=True,
                                 occupancy_checked_utc=datetime.now(timezone.utc).isoformat(),
                                 external_heavy=1,
                                 s4_c_local_seal_path=str(self.seal),
                                 s4_c_local_seal_sha256=sha256_file(self.seal))
            state["disk_floor_gib"] = 0.0
            state["units"].update(self.parent_states)
            self.queue._save(state)

    def _git(self, *args: str) -> None:
        env = dict(os.environ, GIT_AUTHOR_NAME="S3 Synthetic",
                   GIT_AUTHOR_EMAIL="s3@test.invalid",
                   GIT_COMMITTER_NAME="S3 Synthetic",
                   GIT_COMMITTER_EMAIL="s3@test.invalid")
        subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                       env=env, stdout=subprocess.DEVNULL)

    def _git_output(self, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(self.repo), *args],
                                       text=True).strip()

    def _commit(self, message: str) -> None:
        self._git("add", "results")
        self._git("commit", "-qm", message)

    def _completed(self, uid: str, output: Path, spec: dict) -> dict:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            write_json(output, {"schema": "synthetic-parent", "unit_id": uid})
        output_sha = sha256_file(output)
        receipt = {"unit_id": uid, "status": "done", "returncode": 0,
                   "output_sha256": {str(output): output_sha}}
        receipt_path = self.queue.root / "units" / uid / "RECEIPT.json"
        write_json(receipt_path, receipt)
        self.parent_states[uid] = {
            "status": "done", "returncode": 0,
            "spec": spec | {"unit_id": uid, "outputs": [str(output)]},
            "output_sha256": receipt["output_sha256"],
        }
        return {"unit_id": uid, "queue_receipt_sha256": sha256_file(receipt_path),
                "output_sha256": output_sha}

    def _make_c_parents(self, arm_id: str) -> None:
        self.parent_states = {}
        unit = self.units[arm_id]
        fit_id = unit["parents"]["fit"]["unit_id"]
        arm, seed = unit["arm"], unit["seed"]
        for label in S4_C_PARENT_LABELS:
            uid = unit["parents"][label]["unit_id"]
            stage = ("privacy_fit_cal_batch" if label == "fit" else
                     "forward_from_handoff" if label == "forward" else
                     "screen_reader" if label.startswith("reader_") else
                     "postrelease_control_v2" if label.startswith("v2_") else
                     "screen_score")
            view = label.split("_", 1)[1] if label.startswith(("reader_", "v2_")) else None
            script = self.c_worker if label in ("fit", "forward") else self.c_screen
            output = self.private_c / ("worker_receipts" if script == self.c_worker
                                       else "screen_outputs") / f"{uid}.json"
            spec = {"mode": "real_data", "family": "C", "team": "C",
                    "seed": seed, "arm": "C-SIMPLE-PRIVACY-BATCH" if label == "fit" else arm,
                    "stage": stage, "view": view,
                    "argv": [str(self.venv), str(script)],
                    "runner_sha256": sha256_file(script),
                    "deps": [fit_id] if label == "forward" else []}
            unit["parents"][label] = self._completed(uid, output, spec)

    def _make_c_finalizer(self) -> None:
        unit = self.units["S4-C-R1-AGGREGATE"]
        uid = "C-R1-SCREEN-FINALIZE"
        output = self.private_c / "screen_outputs" / f"{uid}.json"
        spec = {"mode": "real_data", "family": "C", "team": "C",
                "stage": "screen_finalize", "seed": None,
                "argv": [str(self.venv), str(self.c_screen)],
                "runner_sha256": sha256_file(self.c_screen), "deps": []}
        unit["parents"]["c_finalizer"] = self._completed(uid, output, spec)

    def _spec(self, uid: str, registry_sha: str | None = None) -> dict:
        unit = self.units[uid]
        aggregate = uid == "S4-C-R1-AGGREGATE"
        runner = self.s4 / ("run_c_aggregate_replay_v1.py" if aggregate else
                            "run_c_adult_replay_v1.py")
        report = self.s4 / ".private/c_r1_replay_v1" / f"{uid}.json"
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
                     str(self.campaign / "S4_C_R1_ARM_RESULT_HASH_REGISTRY.json"),
                     "--arm-result-index-sha256", registry_sha or ""]
        return {
            "unit_id": uid, "mode": "s4_c_adult_replay", "team": "S4",
            "family": "S4", "stage": unit["stage"], "arm": unit["arm"],
            "arm_id": unit["arm"], "seed": unit["seed"],
            "data_access": unit["data_access"],
            "admission_unit_sha256": digest(canonical(unit).encode()),
            "deps": deps, "cwd": str(self.repo), "outputs": [str(report)],
            "max_wall_s": unit["max_wall_s"],
            "cpu_reserve_s": unit["cpu_reserve_s"],
            "max_rss_gib": unit["max_rss_gib"],
            "runner_sha256": sha256_file(runner),
            "env": {name: "1" for name in THREAD_NAMES}
                   | {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(self.s4)},
            "argv": argv,
        } | ({"arm_result_index_sha256": registry_sha} if aggregate else {})

    def _make_arm_registry(self) -> str:
        records = {}
        runner = self.s4 / "run_c_adult_replay_v1.py"
        for uid in self.units["S4-C-R1-AGGREGATE"]["parents"]["arm_reports"]:
            unit = self.units[uid]
            report = self.s4 / ".private/c_r1_replay_v1" / f"{uid}.json"
            artifact = report.with_suffix(".npz")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"synthetic NPZ identity {uid}".encode())
            write_json(report, {"status": "REPLAYED_C_R1_PASS", "unit_id": uid,
                                "arm": unit["arm"], "seed": unit["seed"],
                                "verified_inner_artifacts": {"sha256": sha256_file(artifact)}})
            spec = {"mode": "s4_c_adult_replay", "team": "S4", "family": "S4",
                    "stage": "independent_c_r1_arm_replay", "arm": unit["arm"],
                    "seed": unit["seed"], "data_access": unit["data_access"],
                    "runner_sha256": sha256_file(runner)}
            entry = self._completed(uid, report, spec)
            records[uid] = {"queue_receipt_sha256": entry["queue_receipt_sha256"],
                            "report_sha256": entry["output_sha256"],
                            "artifact_npz_sha256": sha256_file(artifact)}
        registry = self.campaign / "S4_C_R1_ARM_RESULT_HASH_REGISTRY.json"
        write_json(registry, {"schema": "pcrl-s4-c-r1-arm-result-hash-registry-v1",
                              "status": "S4_HASH_CHECKED_16_OF_16",
                              "admission_sha256": sha256_file(self.admission),
                              "fit_lock_sha256": self.fit_sha,
                              "registry_builder_sha256": sha256_file(self.builder),
                              "units": records})
        self._commit("synthetic completed S4 C arm registry")
        with self.queue._mutex():
            state = self.queue._read()
            state["units"].update(self.parent_states)
            self.queue._save(state)
        return sha256_file(registry)

    def test_exact_arm_enqueues_and_mutations_are_denied(self) -> None:
        gate = self.queue.status()["gate"]
        self.assertTrue(Queue._s4_c_local_seal_ok(gate))
        spec = self._spec(self.arm_id)
        self.assertIsNone(self.queue._s4_c_admission_reason(gate, spec))
        self.assertIsNone(self.queue._s4_c_parent_reason(self.queue.status(), spec))
        self.assertEqual(self.queue.enqueue(spec)["status"], "planned")
        for key, value, reason in (
            ("data_access", "WRONG_ROLE", "s4_c_unit_scope_mismatch"),
            ("argv", [*spec["argv"], "--extra"], "s4_c_argv_mismatch"),
            ("outputs", [str(self.repo / "wrong.json")], "s4_c_output_changed"),
        ):
            changed = deepcopy(spec)
            changed[key] = value
            self.assertEqual(self.queue._s4_c_admission_reason(gate, changed), reason)
        source = self.s4 / "run_c_adult_replay_v1.py"
        source.write_text("# changed synthetic runner\n")
        self.assertFalse(Queue._s4_c_local_seal_ok(gate))

    def test_parent_receipt_or_output_mutation_denies_launch(self) -> None:
        spec = self._spec(self.arm_id)
        self.assertIsNone(self.queue._gate_status(self.queue.status(), spec))
        uid = self.units[self.arm_id]["parents"]["reader_r1"]["unit_id"]
        receipt = self.queue.root / "units" / uid / "RECEIPT.json"
        receipt.write_text(receipt.read_text() + " ")
        self.assertEqual(self.queue._s4_c_parent_reason(self.queue.status(), spec),
                         "s4_c_parent_not_complete")
        write_json(receipt, json.loads(receipt.read_text()))
        # A changed output is caught independently of the S3 receipt bytes.
        output = Path(self.parent_states[uid]["spec"]["outputs"][0])
        output.write_text(output.read_text() + " ")
        self.assertEqual(self.queue._s4_c_parent_reason(self.queue.status(), spec),
                         "s4_c_parent_output_changed")

    def test_aggregate_requires_committed_exact_arm_registry(self) -> None:
        uid = "S4-C-R1-AGGREGATE"
        missing = self._spec(uid)
        self.assertEqual(self.queue._s4_c_admission_reason(self.queue.status()["gate"], missing),
                         "s4_c_arm_result_registry_invalid")
        registry_sha = self._make_arm_registry()
        spec = self._spec(uid, registry_sha)
        self.assertIsNone(self.queue._s4_c_admission_reason(self.queue.status()["gate"], spec))
        self.assertIsNone(self.queue._s4_c_parent_reason(self.queue.status(), spec))
        self.assertEqual(self.queue.enqueue(spec)["status"], "planned")
        registry = self.campaign / "S4_C_R1_ARM_RESULT_HASH_REGISTRY.json"
        changed = json.loads(registry.read_text())
        changed["status"] = "INCOMPLETE"
        write_json(registry, changed)
        self.assertEqual(self.queue._s4_c_admission_reason(self.queue.status()["gate"], spec),
                         "s4_c_arm_result_registry_invalid")


if __name__ == "__main__":
    unittest.main()
