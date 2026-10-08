"""Synthetic admission checks for A score-bank preprocessing under B_ONLY."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
from compute_queue import Queue


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AScoreBankAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.campaign = self.repo / "results/pcrl_method_portfolio_v1"
        self.shared = self.campaign / "shared"
        self.shared.mkdir(parents=True)
        self.script = self.shared / "admit_a_score_bank.py"
        self.script.write_text("# synthetic reviewed runner\n")
        self.manifest = self.campaign / "DATA_ACCESS_MANIFEST.json"
        self.manifest.write_text("{}\n")
        self.cache = Path(self.temp.name) / "source_cache"
        self.cache.mkdir()
        self.queue_root = self.shared / ".private" / "compute"
        self.queue_root.mkdir(parents=True)
        self.unit_id = "S2-A-SCORE-BANK-TSTAR-S0"
        self.lock = self.campaign / "CAMPAIGN_LOCK.json"
        self.lock.write_text(json.dumps({
            "status": "LOCAL_GIT_SEALED_EFFECTIVE",
            "adult_fit_permission": True,
            "fit_admission": "B_ONLY",
            "families": {"A": {"preprocessing_admission": "ALLOWED",
                               "fit_admission": "DENIED"}},
            "admitted_a_score_bank_unit_ids": [
                f"S2-A-SCORE-BANK-{key.upper().replace('_', '-')}-S{seed}"
                for key in ("tstar", "pstar", "lra_seq21", "hcal_local_d0", "hcal_seq12_d0")
                for seed in (0, 1)
            ],
            "a_score_bank_runner_sha256": sha(self.script),
            "a_score_bank_cache_root": "../source_cache",
            "data_admission": {"manifest_sha256": sha(self.manifest)},
        }))
        self.seal = self.campaign / "LOCAL_SEAL_RECEIPT.json"
        self.seal.write_text("{}\n")
        self.gate = {"safe_inputs": True, "local_seal_path": str(self.seal)}
        self.output = (self.shared / ".private" / "tstar_score_bank_s0_v1" /
                       "BANK_ADMISSION_RECEIPT.json")
        self.spec = {
            "unit_id": self.unit_id, "mode": "real_data", "family": "SHARED",
            "team": "SHARED", "arm_id": "A_SCORE_BANK_PREPROCESS",
            "arm": "A_SCORE_BANK_PREPROCESS", "stage": "a_score_bank_preprocess",
            "data_access": "A_ROLE_ONLY_REFERENCES_PREPROCESS", "seed": 0,
            "release_key": "tstar", "cwd": str(self.repo),
            "runner_sha256": sha(self.script), "cache_root": str(self.cache),
            "deps": [], "outputs": [str(self.output)],
            "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 1.5,
            "argv": [sys.executable, str(self.script),
                     "--cache-root", str(self.cache),
                     "--seed", "0", "--release", "tstar",
                     "--campaign-lock-sha256", sha(self.lock),
                     "--queue-root", str(self.queue_root)],
        }

    def reason(self, spec=None):
        with mock.patch.object(Queue, "_local_seal_ok", return_value=True):
            return Queue._real_admission_reason(self.gate, self.spec if spec is None else spec)

    def test_exact_preprocessing_unit_is_admitted_while_a_fit_is_denied(self):
        self.assertIsNone(self.reason())
        self.assertIsNotNone(self.reason(dict(self.spec, stage="a_task_fit")))
        self.assertIsNotNone(self.reason(dict(self.spec, arm_id="A-TASK-RD")))

    def test_exact_preprocessing_unit_passes_actual_enqueue(self):
        queue = Queue(Path(self.temp.name) / "queue")
        with queue._mutex():
            state = queue._read()
            state["gate"] = self.gate
            queue._save(state)
        with mock.patch.object(Queue, "_local_seal_ok", return_value=True):
            self.assertEqual(queue.enqueue(self.spec)["status"], "planned")

    def test_wrong_seed_release_unit_or_source_is_denied(self):
        for change in (
            {"seed": 2},
            {"release_key": "unknown"},
            {"unit_id": "S2-A-SCORE-BANK-TSTAR-S0-R1"},
            {"runner_sha256": "0" * 64},
            {"arm": "A-TASK-RD"},
            {"arm_id": "A-TASK-RD"},
        ):
            with self.subTest(change=change):
                self.assertIsNotNone(self.reason(dict(self.spec, **change)))

    def test_script_argument_and_output_bindings_are_exact(self):
        decoy = dict(self.spec, argv=[sys.executable, "other.py", *self.spec["argv"][2:],
                                       str(self.script)])
        self.assertIsNotNone(self.reason(decoy))
        old_output = dict(self.spec, outputs=[str(self.output).replace("_v1", "_old")])
        self.assertIsNotNone(self.reason(old_output))
        wrong_cache = dict(self.spec, argv=[x if x != str(self.cache) else str(self.repo / "wrong")
                                                  for x in self.spec["argv"]])
        self.assertIsNotNone(self.reason(wrong_cache))


if __name__ == "__main__":
    unittest.main()
