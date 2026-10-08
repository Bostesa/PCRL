"""Synthetic test-first boundary for independent S4 Adult replay admission.

The fixture contains only committed metadata and a dummy Python script. It does
not open any Adult role file or run an S4 replay worker.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from compute_queue import Queue


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
MAP_SHA = "f6100b8fb3ca9f7b2b72d89f486375ed7f4c0b79218db87ec05694018149bce4"
SCHEMA_SHA = "6aed1ed024153211a15f6cd5be000169a77db393314fe71fa18781f744d896fc"
SOURCE_NAMES = (
    "run_adult_replay_v3", "replay_b_values_v3", "replay_anchor_values_v3",
    "replay_b_attempt", "replay_anchor_attempt", "replay_b_readers",
    "replay_b_release", "replay_feature_anchor", "replay_v2_controls",
    "verifier_identity_v3", "independent_endpoints",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class S4AdultSealTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = (Path(self.temp.name) / "PCRL/.worktrees/coordinator").resolve()
        self.campaign = self.repo / "results/pcrl_method_portfolio_v1"
        self.s4 = self.campaign / "s4"
        self.s4.mkdir(parents=True)
        (self.repo / "../../../PCRL_eval_cache_private").resolve().mkdir(parents=True)
        source_map = CAMPAIGN / "s4/S4_ADULT_REPLAY_PARENT_MAP_V2.json"
        source_schema = CAMPAIGN / "s4/S4_ADULT_REPLAY_UNIT_SCHEMA_V2.json"
        self.assertEqual(sha(source_map), MAP_SHA)
        self.assertEqual(sha(source_schema), SCHEMA_SHA)
        shutil.copyfile(source_map, self.s4 / source_map.name)
        shutil.copyfile(source_schema, self.s4 / source_schema.name)
        self.runner = self.s4 / "run_adult_replay_v3.py"
        for name in SOURCE_NAMES:
            (self.s4 / f"{name}.py").write_text(f"# synthetic {name} identity\n")
        self.lock = self.campaign / "S4_VERIFIER_ADMISSION.json"
        parent_map = json.loads(source_map.read_text())
        historical = {
            lock_sha: {"kind": kind, "lock_sha256": lock_sha}
            for kind, values in (
                ("FIT", parent_map["accepted_fit_lock_sha256"]),
                ("CONTROL_V2", parent_map["accepted_control_v2_lock_sha256"]),
            )
            for lock_sha in values
        }
        write_json(self.lock, {
            "schema": "pcrl-s4-verifier-admission-v3",
            "status": "LOCAL_GIT_SEALED_EFFECTIVE",
            "scope": "S4_ADULT_REPLAY",
            "parent_map_sha256": MAP_SHA,
            "unit_schema_sha256": SCHEMA_SHA,
            "unit_specs": parent_map["unit_specs"],
            "runner_sha256": sha(self.runner),
            "scheduler_source_sha256": sha(HERE / "compute_queue.py"),
            "source_sha256_by_file": {
                f"results/pcrl_method_portfolio_v1/s4/{name}.py":
                    sha(self.s4 / f"{name}.py") for name in SOURCE_NAMES},
            "accepted_fit_lock_sha256": parent_map["accepted_fit_lock_sha256"],
            "accepted_control_v2_lock_sha256": parent_map["accepted_control_v2_lock_sha256"],
            "historical_seal_by_lock_sha256": historical,
            "queue_root_relative_to_coordinator":
                "results/pcrl_method_portfolio_v1/shared/.private/compute",
            "private_output_root_relative_to_coordinator":
                "results/pcrl_method_portfolio_v1/s4/.private/adult_replay",
            "hcal_cache_root_relative_to_coordinator":
                "../../../PCRL_eval_cache_private",
        })
        subprocess.run(["git", "-C", str(self.repo), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "results"], check=True)
        git_env = dict(os.environ, GIT_AUTHOR_NAME="S3 Test",
                       GIT_AUTHOR_EMAIL="s3@test.invalid",
                       GIT_COMMITTER_NAME="S3 Test",
                       GIT_COMMITTER_EMAIL="s3@test.invalid")
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "synthetic S4 admission"],
                       check=True, env=git_env)
        self.commit = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        self.review = self.campaign / "S4_VERIFIER_PREFIT_REVIEW.json"
        write_json(self.review, {
            "schema": "pcrl-s4-verifier-prefit-review-v3", "status": "PASS",
            "scope": "S4_ADULT_REPLAY", "admission_sha256": sha(self.lock),
            "commit_sha": self.commit,
        })
        self.seal = self.campaign / "S4_VERIFIER_LOCAL_SEAL_RECEIPT.json"
        write_json(self.seal, {
            "schema": "pcrl-s4-verifier-local-seal-v3", "status": "PASS",
            "scope": "S4_ADULT_REPLAY", "admission_sha256": sha(self.lock),
            "commit_sha": self.commit, "review_path": self.review.name,
            "review_sha256": sha(self.review),
        })
        self.gate = {"safe_inputs": True, "s4_local_seal_path": str(self.seal),
                     "s4_local_seal_sha256": sha(self.seal)}

    def exact_spec(self) -> dict:
        admission = json.loads(self.lock.read_text())
        unit = admission["unit_specs"]["S4-B-TASK-S0"]
        uid = "S4-B-TASK-S0"
        venv = self.repo.parent.parent / ".venv/bin/python"
        venv.parent.mkdir(parents=True, exist_ok=True)
        if not venv.exists():
            venv.symlink_to(sys.executable)
        queue_root = self.repo / admission["queue_root_relative_to_coordinator"]
        report = (self.repo / admission["private_output_root_relative_to_coordinator"] /
                  uid / "REPORT.json")
        cache = (self.repo / admission["hcal_cache_root_relative_to_coordinator"]).resolve()
        return {
            "unit_id": uid, "mode": "s4_adult_replay", "family": "S4", "team": "S4",
            "kind": "B_SCREEN", "arm": "B-TASK", "arm_id": "B-TASK", "seed": 0,
            "stage": "independent_b_v2_replay", "data_access": unit["data_access"],
            "parent_map_unit_sha256": hashlib.sha256(json.dumps(
                unit, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
            "parent_unit_id": unit["parent_unit_id"],
            "control_v2_unit_id": unit["control_v2_unit_id"],
            "parent_queue_receipt_sha256": unit["parent_queue_receipt_sha256"],
            "control_v2_queue_receipt_sha256": unit["control_v2_queue_receipt_sha256"],
            "fit_lock_sha256": unit["fit_lock_sha256"],
            "control_v2_lock_sha256": unit["control_v2_lock_sha256"],
            "runner_sha256": sha(self.runner),
            "deps": [unit["parent_unit_id"], unit["control_v2_unit_id"]],
            "cwd": str(self.repo), "outputs": [str(report)],
            "max_wall_s": 600, "cpu_reserve_s": 600, "max_rss_gib": 2.0,
            "env": {name: "1" for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS",
                "PYTHONDONTWRITEBYTECODE")}
                   | {"PYTHONPATH": str(self.s4)},
            "argv": [str(venv), str(self.runner),
                     "--unit-id", uid, "--admission", str(self.lock),
                     "--admission-sha256", sha(self.lock),
                     "--local-seal", str(self.seal),
                     "--local-seal-sha256", sha(self.seal),
                     "--queue-root", str(queue_root),
                     "--private-report", str(report),
                     "--cache-root", str(cache)],
        }

    def test_separate_committed_s4_seal_is_required(self) -> None:
        self.assertTrue(Queue._s4_local_seal_ok(self.gate))
        changed = dict(self.gate, s4_local_seal_sha256="0" * 64)
        self.assertFalse(Queue._s4_local_seal_ok(changed))
        review = json.loads(self.review.read_text())
        review["scope"] = "B_ONLY"
        write_json(self.review, review)
        changed = dict(self.gate, s4_local_seal_sha256=sha(self.seal))
        self.assertFalse(Queue._s4_local_seal_ok(changed))

    def test_alternate_committed_cache_root_is_rejected(self) -> None:
        admission = json.loads(self.lock.read_text())
        admission["hcal_cache_root_relative_to_coordinator"] = "../../../other_cache"
        (self.repo / "../../../other_cache").resolve().mkdir(parents=True)
        write_json(self.lock, admission)
        subprocess.run(["git", "-C", str(self.repo), "add", "results"], check=True)
        git_env = dict(os.environ, GIT_AUTHOR_NAME="S3 Test",
                       GIT_AUTHOR_EMAIL="s3@test.invalid",
                       GIT_COMMITTER_NAME="S3 Test",
                       GIT_COMMITTER_EMAIL="s3@test.invalid")
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "alternate cache"],
                       check=True, env=git_env)
        commit = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        review = json.loads(self.review.read_text())
        review.update(admission_sha256=sha(self.lock), commit_sha=commit)
        write_json(self.review, review)
        seal = json.loads(self.seal.read_text())
        seal.update(admission_sha256=sha(self.lock), commit_sha=commit,
                    review_sha256=sha(self.review))
        write_json(self.seal, seal)
        gate = dict(self.gate, s4_local_seal_sha256=sha(self.seal))
        self.assertFalse(Queue._s4_local_seal_ok(gate))

    def test_old_b_seal_cannot_enqueue_s4_adult_mode(self) -> None:
        queue = Queue(Path(self.temp.name) / "queue")
        spec = {
            "unit_id": "S4-B-TASK-S0", "mode": "s4_adult_replay",
            "family": "S4", "team": "S4", "arm": "B-TASK", "arm_id": "B-TASK",
            "seed": 0, "stage": "independent_b_v2_replay",
            "data_access": "S4_B_FIT_TASK_ATT_INNER_ROLE_ONLY_REPLAY",
            "cwd": str(self.repo), "argv": [sys.executable, str(self.runner)],
            "deps": [], "outputs": [], "max_wall_s": 600,
            "cpu_reserve_s": 600, "max_rss_gib": 2.0,
        }
        with self.assertRaises(ValueError):
            queue.enqueue(spec)

    def test_exact_s4_unit_and_argv_are_bound_before_parent_access(self) -> None:
        spec = self.exact_spec()
        self.assertIsNone(Queue._s4_admission_reason(self.gate, spec))
        for change in (
            {"unit_id": "S4-B-TASK-S1"},
            {"data_access": "S4_B_ATT_INNER_ROLE_ONLY_REPLAY"},
            {"deps": []},
            {"max_wall_s": 601, "cpu_reserve_s": 601},
            {"outputs": [str(self.campaign / "s4/.private/other/REPORT.json")]},
            {"argv": [spec["argv"][0], "-c", "pass", *spec["argv"][1:]]},
        ):
            changed = dict(spec, **change)
            self.assertIsNotNone(Queue._s4_admission_reason(self.gate, changed), change)


if __name__ == "__main__":
    unittest.main()
