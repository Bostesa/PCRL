"""Data-free admission proof for the sealed six-reader C diagnostic."""

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

from compute_queue import Queue, canonical, sha256_file


PRIOR_UID = "S4-C-R1-LR-INNER-DIAG-FULL-S0-R1"
UID = "S4-C-R1-READER-BANK-DIAG-FULL-S0-R1-R2"
FAILED = "S4-C-R1-C-SIMPLE-FULL-S0"
SOURCE_NAMES = (
    "run_c_reader_bank_diagnostic_v2", "run_c_reader_diagnostic_v1",
    "run_c_adult_replay_v1",
    "replay_c_readers_v1", "replay_c_release_v1", "replay_c_values_v1",
)
READERS = ["LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
           "RAW_LR_C1", "RAW_HGB_15x200"]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


class ReaderBankDiagnosticQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = (Path(self.tmp.name) / "PCRL/.worktrees/coordinator").resolve()
        self.campaign = self.repo / "results/pcrl_method_portfolio_v1"
        self.s4 = self.campaign / "s4"
        self.s4.mkdir(parents=True)
        self.queue = Queue(self.campaign / "shared/.private/compute")
        self.venv = self.repo.parent.parent / ".venv/bin/python"
        self.venv.parent.mkdir(parents=True, exist_ok=True)
        self.venv.symlink_to(sys.executable)
        self.team_c = self.repo.parent / "team-c/results/pcrl_method_portfolio_v1/teams/c"
        self.team_c.mkdir(parents=True)
        self.worker = self.team_c / "run_c_worker.py"
        self.screen = self.team_c / "screen_orchestrator.py"
        self.worker.write_text("# synthetic worker\n")
        self.screen.write_text("# synthetic screen\n")
        self.sources = {}
        for name in SOURCE_NAMES:
            path = self.s4 / f"{name}.py"
            path.write_text(f"# synthetic source {name}\n")
            relative = f"results/pcrl_method_portfolio_v1/s4/{name}.py"
            self.sources[relative] = sha256_file(path)
        manifest = self.campaign / "DATA_ACCESS_MANIFEST.json"
        write_json(manifest, {"schema": "synthetic-metadata-only"})
        self.fit_lock = self.campaign / "CAMPAIGN_LOCK.json"
        write_json(self.fit_lock, {
            "families": {"C": {"fit_admission": "ALLOWED",
                               "worker_runner_sha256": sha256_file(self.worker),
                               "screen_orchestration": {"source_sha256": sha256_file(self.screen)}}},
        })
        self._git("init", "-q")
        self._commit("synthetic historical fit")
        self.fit_commit = self._git_out("rev-parse", "HEAD")
        self.fit_sha = sha256_file(self.fit_lock)
        self.parents = {}
        parent_ids = {
            "fit": "C-R1-FIT-PRIVACY-S0",
            "forward": "C-R1-FORWARD-C-SIMPLE-FULL-S0",
            "reader_r1": "C-R1-SCREEN-READ-C-SIMPLE-FULL-S0-r1",
        }
        parent_states = {}
        for label, pid in parent_ids.items():
            script = self.screen if label == "reader_r1" else self.worker
            output = (self.team_c / ".private/team_c/synthetic" /
                      f"{pid}.json")
            write_json(output, {"schema": "synthetic-parent", "unit_id": pid})
            output_sha = sha256_file(output)
            receipt_path = self.queue.root / "units" / pid / "RECEIPT.json"
            receipt = {"unit_id": pid, "status": "done", "returncode": 0,
                       "output_sha256": {str(output): output_sha}}
            write_json(receipt_path, receipt)
            stage = {"fit": "privacy_fit_cal_batch", "forward": "forward_from_handoff",
                     "reader_r1": "screen_reader"}[label]
            pspec = {
                "unit_id": pid, "mode": "real_data", "team": "C", "family": "C",
                "stage": stage, "seed": 0,
                "arm": "C-SIMPLE-PRIVACY-BATCH" if label == "fit" else "C-SIMPLE-FULL",
                "view": "r1" if label == "reader_r1" else None,
                "argv": [str(self.venv), str(script)],
                "runner_sha256": sha256_file(script), "outputs": [str(output)],
                "deps": [parent_ids["fit"]] if label == "forward" else [],
            }
            parent_states[pid] = {"status": "done", "returncode": 0,
                                  "spec": pspec, "output_sha256": receipt["output_sha256"]}
            self.parents[label] = {"unit_id": pid,
                                   "queue_receipt_sha256": sha256_file(receipt_path),
                                   "output_sha256": output_sha}
        failed_path = self.queue.root / "units" / FAILED / "RECEIPT.json"
        write_json(failed_path, {"unit_id": FAILED, "status": "failed",
                                 "returncode": 1, "output_sha256": {}})
        parent_states[FAILED] = {"status": "failed", "returncode": 1,
                                 "spec": {"unit_id": FAILED, "mode": "s4_c_adult_replay"}}
        self.failed_sha = sha256_file(failed_path)
        self.historical = self.campaign / "S4_C_REPLAY_ADMISSION.json"
        write_json(self.historical, {
            "schema": "pcrl-s4-c-replay-admission-v1",
            "status": "LOCAL_GIT_SEALED_EFFECTIVE",
            "lock_commit_sha": self.fit_commit, "lock_sha256": self.fit_sha,
            "data_manifest_sha256": sha256_file(manifest),
            "source_sha256_by_path": {k: v for k, v in self.sources.items()
                                     if not k.endswith(("run_c_reader_diagnostic_v1.py",
                                                        "run_c_reader_bank_diagnostic_v2.py"))},
            "unit_specs": {FAILED: {"parents": deepcopy(self.parents)}},
        })
        self._commit("synthetic historical S4 admission")
        historical_commit = self._git_out("rev-parse", "HEAD")
        historical_review = self.campaign / "S4_C_REPLAY_PREFIT_REVIEW.json"
        write_json(historical_review, {"status": "PASS"})
        historical_seal = self.campaign / "S4_C_REPLAY_LOCAL_SEAL_RECEIPT.json"
        write_json(historical_seal, {"status": "PASS",
                                     "admission_sha256": sha256_file(self.historical),
                                     "admission_commit_sha": historical_commit})
        self.unit = {
            "stage": "independent_c_r1_reader_diagnostic", "arm": "C-SIMPLE-FULL",
            "seed": 0, "view": "r1", "reader": "LR_C1",
            "data_access": "S4_C_R1_INNER_FEATURES_SAVED_READER_DIAGNOSTIC",
            "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 1.0,
            "parents": deepcopy(self.parents),
        }
        self.admission = self.campaign / "S4_C_READER_DIAGNOSTIC_ADMISSION.json"
        self.admission_data = {
            "schema": "pcrl-s4-c-reader-diagnostic-admission-v1",
            "status": "LOCAL_GIT_SEALED_EFFECTIVE",
            "scope": "S4_C_R1_LR_C1_R1_NUMERICAL_DIAGNOSTIC",
            "adult_role_access": True,
            "historical_s4_admission_sha256": sha256_file(self.historical),
            "historical_s4_review_sha256": sha256_file(historical_review),
            "historical_s4_seal_sha256": sha256_file(historical_seal),
            "historical_fit_lock_sha256": self.fit_sha,
            "historical_fit_commit_sha": self.fit_commit,
            "data_manifest_sha256": sha256_file(manifest),
            "failed_s4_queue_receipt_sha256": self.failed_sha,
            "queue_source_sha256": sha256_file(Path(__file__).with_name("compute_queue.py")),
            "source_sha256_by_path": {k: v for k, v in self.sources.items()
                                      if not k.endswith("run_c_reader_bank_diagnostic_v2.py")},
            "unit_specs": {PRIOR_UID: self.unit},
        }
        write_json(self.admission, self.admission_data)
        self._commit("synthetic diagnostic admission")
        admission_commit = self._git_out("rev-parse", "HEAD")
        self.review = self.campaign / "S4_C_READER_DIAGNOSTIC_PREFIT_REVIEW.json"
        write_json(self.review, {
            "schema": "pcrl-s4-c-reader-diagnostic-prefit-review-v1",
            "status": "PASS", "scope": self.admission_data["scope"],
            "admission_sha256": sha256_file(self.admission),
            "admission_commit_sha": admission_commit,
        })
        self.seal = self.campaign / "S4_C_READER_DIAGNOSTIC_LOCAL_SEAL_RECEIPT.json"
        write_json(self.seal, {
            "schema": "pcrl-s4-c-reader-diagnostic-local-seal-v1",
            "status": "PASS", "scope": self.admission_data["scope"],
            "admission_sha256": sha256_file(self.admission),
            "review_sha256": sha256_file(self.review),
            "admission_commit_sha": admission_commit,
        })
        prior_report = (self.s4 / ".private/c_r1_reader_diag_v1" /
                        f"{PRIOR_UID}.json")
        write_json(prior_report, {
            "unit_id": PRIOR_UID,
            "status": "DIAGNOSTIC_RECORDED_NO_TOLERANCE_DECISION",
            "saved_vs_rebuilt_model_predictions": {"exact_equal": True},
        })
        prior_report_sha = sha256_file(prior_report)
        prior_receipt_path = (self.queue.root / "units" / PRIOR_UID /
                              "RECEIPT.json")
        prior_output_sha = {str(prior_report): prior_report_sha}
        write_json(prior_receipt_path, {
            "unit_id": PRIOR_UID, "status": "done", "returncode": 0,
            "output_sha256": prior_output_sha,
        })
        parent_states[PRIOR_UID] = {
            "status": "done", "returncode": 0,
            "spec": {"unit_id": PRIOR_UID, "mode": "s4_c_reader_diagnostic"},
            "output_sha256": prior_output_sha,
        }
        self.bank_unit = {
            "stage": "independent_c_r1_reader_bank_diagnostic",
            "arm": "C-SIMPLE-FULL", "seed": 0, "view": "r1",
            "readers": READERS,
            "data_access": "S4_C_R1_INNER_FEATURES_SAVED_READER_BANK_DIAGNOSTIC",
            "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 1.5,
            "parents": deepcopy(self.parents),
        }
        self.bank_admission = (self.campaign /
                               "S4_C_READER_BANK_DIAGNOSTIC_V2_ADMISSION.json")
        self.bank_admission_data = {
            "schema": "pcrl-s4-c-reader-bank-diagnostic-admission-v2",
            "status": "LOCAL_GIT_SEALED_EFFECTIVE",
            "scope": "S4_C_R1_SIX_READER_INNER_DIAGNOSTIC_V2",
            "adult_role_access": True,
            "historical_s4_admission_sha256": sha256_file(self.historical),
            "historical_s4_review_sha256": sha256_file(historical_review),
            "historical_s4_seal_sha256": sha256_file(historical_seal),
            "historical_fit_lock_sha256": self.fit_sha,
            "historical_fit_commit_sha": self.fit_commit,
            "failed_s4_queue_receipt_sha256": self.failed_sha,
            "prior_single_diagnostic_admission_sha256": sha256_file(self.admission),
            "prior_single_diagnostic_review_sha256": sha256_file(self.review),
            "prior_single_diagnostic_seal_sha256": sha256_file(self.seal),
            "prior_single_diagnostic_queue_receipt_sha256": sha256_file(prior_receipt_path),
            "prior_single_diagnostic_report_sha256": prior_report_sha,
            "data_manifest_sha256": sha256_file(manifest),
            "queue_source_sha256": sha256_file(Path(__file__).with_name("compute_queue.py")),
            "source_sha256_by_path": {k: v for k, v in self.sources.items()
                                      if not k.endswith("run_c_reader_diagnostic_v1.py")},
            "unit_specs": {UID: self.bank_unit},
        }
        write_json(self.bank_admission, self.bank_admission_data)
        self._commit("synthetic six-reader diagnostic admission")
        bank_commit = self._git_out("rev-parse", "HEAD")
        self.bank_review = (self.campaign /
                            "S4_C_READER_BANK_DIAGNOSTIC_V2_PREFIT_REVIEW.json")
        write_json(self.bank_review, {
            "schema": "pcrl-s4-c-reader-bank-diagnostic-prefit-review-v2",
            "status": "PASS", "scope": self.bank_admission_data["scope"],
            "admission_sha256": sha256_file(self.bank_admission),
            "admission_commit_sha": bank_commit,
        })
        self.bank_seal = (self.campaign /
                          "S4_C_READER_BANK_DIAGNOSTIC_V2_LOCAL_SEAL_RECEIPT.json")
        write_json(self.bank_seal, {
            "schema": "pcrl-s4-c-reader-bank-diagnostic-local-seal-v2",
            "status": "PASS", "scope": self.bank_admission_data["scope"],
            "admission_sha256": sha256_file(self.bank_admission),
            "review_sha256": sha256_file(self.bank_review),
            "admission_commit_sha": bank_commit,
        })
        with self.queue._mutex():
            state = self.queue._read()
            state["gate"].update(safe_inputs=True, manually_closed=False,
                                 occupancy_confirmed=True, external_heavy=1,
                                 occupancy_checked_utc=datetime.now(timezone.utc).isoformat(),
                                 s4_c_diag_local_seal_path=str(self.seal),
                                 s4_c_diag_local_seal_sha256=sha256_file(self.seal),
                                 s4_c_bank_diag_local_seal_path=str(self.bank_seal),
                                 s4_c_bank_diag_local_seal_sha256=sha256_file(self.bank_seal))
            state["units"].update(parent_states)
            state["disk_floor_gib"] = 0.0
            self.queue._save(state)

    def _git(self, *args: str) -> None:
        env = dict(os.environ, GIT_AUTHOR_NAME="S3 Synthetic",
                   GIT_AUTHOR_EMAIL="s3@test.invalid",
                   GIT_COMMITTER_NAME="S3 Synthetic",
                   GIT_COMMITTER_EMAIL="s3@test.invalid")
        subprocess.run(["git", "-C", str(self.repo), *args], env=env,
                       check=True, stdout=subprocess.DEVNULL)

    def _git_out(self, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(self.repo), *args],
                                       text=True).strip()

    def _commit(self, message: str) -> None:
        self._git("add", "results")
        self._git("commit", "-qm", message)

    def spec(self) -> dict:
        runner = self.s4 / "run_c_reader_bank_diagnostic_v2.py"
        output = self.s4 / ".private/c_r1_reader_bank_diag_v2" / f"{UID}.json"
        env = {name: "1" for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS")}
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(self.s4))
        return {
            "unit_id": UID, "mode": "s4_c_reader_bank_diagnostic",
            "team": "S4", "family": "S4", "stage": self.bank_unit["stage"],
            "arm": self.bank_unit["arm"], "arm_id": self.bank_unit["arm"],
            "seed": 0, "view": "r1", "readers": READERS,
            "data_access": self.bank_unit["data_access"],
            "deps": [self.parents["reader_r1"]["unit_id"]],
            "admission_unit_sha256": hashlib.sha256(
                canonical(self.bank_unit).encode()).hexdigest(),
            "runner_sha256": sha256_file(runner),
            "cwd": str(self.repo), "env": env, "outputs": [str(output)],
            "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 1.5,
            "argv": [str(self.venv), str(runner), "--unit-id", UID,
                     "--admission", str(self.bank_admission),
                     "--admission-sha256", sha256_file(self.bank_admission),
                     "--local-seal", str(self.bank_seal),
                     "--local-seal-sha256", sha256_file(self.bank_seal),
                     "--queue-root", str(self.queue.root),
                     "--private-report", str(output)],
        }

    def test_exact_spec_admits_at_enqueue_and_launch(self) -> None:
        spec = self.spec()
        self.assertEqual(self.queue.enqueue(spec)["status"], "planned")
        self.assertIsNone(self.queue._gate_status(self.queue.status(), spec))

    def test_wrong_role_source_argv_and_output_are_denied(self) -> None:
        cases = (("data_access", "OTHER", "s4_c_bank_diag_scope_invalid"),
                 ("runner_sha256", "0" * 64, "s4_c_bank_diag_runner_changed"),
                 ("argv", [*self.spec()["argv"][:-1], "elsewhere.json"],
                  "s4_c_bank_diag_argv_mismatch"),
                 ("outputs", [str(self.s4 / "other.json")],
                  "s4_c_bank_diag_output_changed"))
        for key, value, reason in cases:
            with self.subTest(key=key):
                changed = self.spec()
                changed[key] = value
                with self.assertRaisesRegex(ValueError, reason):
                    self.queue.enqueue(changed)

    def test_changed_parent_output_and_source_set_are_denied(self) -> None:
        spec = self.spec()
        state = self.queue.status()
        parent = self.parents["reader_r1"]["unit_id"]
        altered = deepcopy(state)
        altered["units"][parent]["output_sha256"] = {"changed": "0" * 64}
        self.assertEqual(self.queue._s4_c_bank_diag_parent_reason(altered, spec),
                         "s4_c_diag_parent_changed")
        self.assertEqual(self.queue._gate_status(altered, spec),
                         "s4_c_diag_parent_changed")
        with self.queue._mutex():
            stored = self.queue._read()
            original = deepcopy(stored["units"][parent]["output_sha256"])
            stored["units"][parent]["output_sha256"] = {"changed": "0" * 64}
            self.queue._save(stored)
        with self.assertRaisesRegex(ValueError, "s4_c_diag_parent_changed"):
            self.queue.enqueue(spec)
        with self.queue._mutex():
            stored = self.queue._read()
            stored["units"][parent]["output_sha256"] = original
            self.queue._save(stored)
        self.bank_admission_data["source_sha256_by_path"]["results/pcrl_method_portfolio_v1/s4/extra.py"] = "0" * 64
        write_json(self.bank_admission, self.bank_admission_data)
        self._commit("synthetic changed diagnostic source set")
        review = json.loads(self.bank_review.read_text())
        review["admission_sha256"] = sha256_file(self.bank_admission)
        review["admission_commit_sha"] = self._git_out("rev-parse", "HEAD")
        write_json(self.bank_review, review)
        seal = json.loads(self.bank_seal.read_text())
        seal["admission_sha256"] = review["admission_sha256"]
        seal["admission_commit_sha"] = review["admission_commit_sha"]
        seal["review_sha256"] = sha256_file(self.bank_review)
        write_json(self.bank_seal, seal)
        state["gate"]["s4_c_bank_diag_local_seal_sha256"] = sha256_file(self.bank_seal)
        self.assertFalse(self.queue._s4_c_bank_diag_local_seal_ok(state["gate"]))


if __name__ == "__main__":
    unittest.main()
