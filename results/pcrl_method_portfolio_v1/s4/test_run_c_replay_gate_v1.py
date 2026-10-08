"""Data-free S4 C preaccess and private custody refusal fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_c_adult_replay_v1 as gate
from run_c_aggregate_replay_v1 import _load_arm, _same
from replay_c_values_v1 import sha_array


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IndependentCReplayGateSynthetic(unittest.TestCase):
    def test_parent_stage_and_source_before_role(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            private = root / "teams/c/.private"
            output = private / "source.json"
            output.parent.mkdir(parents=True)
            output.write_text('{"schema":"synthetic"}\n')
            script = root / "screen.py"
            script.write_text("# synthetic\n")
            unit_id = "C-R1-SCREEN-SCORE-C-SIMPLE-FULL-S0"
            expected = {str(output): sha(output)}
            receipt = {"unit_id": unit_id, "status": "done",
                       "returncode": 0, "output_sha256": expected}
            receipt_path = root / "queue/units" / unit_id / "RECEIPT.json"
            receipt_path.parent.mkdir(parents=True)
            receipt_path.write_text(json.dumps(receipt))
            spec = {"unit_id": unit_id, "mode": "real_data", "team": "C",
                    "family": "C", "seed": 0, "stage": "screen_score",
                    "arm": "C-SIMPLE-FULL", "view": None,
                    "outputs": [str(output)], "argv": ["python", str(script)],
                    "runner_sha256": sha(script)}
            state = {"units": {unit_id: {"status": "done", "returncode": 0,
                                          "output_sha256": expected, "spec": spec}}}
            entry = {"unit_id": unit_id, "queue_receipt_sha256": sha(receipt_path),
                     "output_sha256": sha(output)}
            lock = {"families": {"C": {"screen_orchestration": {
                "source_sha256": sha(script)}}}}
            with patch.object(gate, "REPO", root):
                path, private_root = gate._parent_output(root / "queue", state, entry,
                    lock, 0, "C-SIMPLE-FULL", "score")
                self.assertEqual(path, output)
                self.assertEqual(private_root, private)
                spec["stage"] = "screen_reader"
                with self.assertRaisesRegex(PermissionError, "stage"):
                    gate._parent_output(root / "queue", state, entry,
                        lock, 0, "C-SIMPLE-FULL", "score")

    def test_s4_arm_artifact_hash_and_member_refusal(self):
        with tempfile.TemporaryDirectory() as scratch:
            report_path = Path(scratch) / "S4-C-R1-C-SIMPLE-FULL-S0.json"
            artifact_path = report_path.with_suffix(".npz")
            arrays = {"row_id": np.arange(2, dtype=np.int64)}
            np.savez(artifact_path, **arrays)
            report = {"status": "REPLAYED_C_R1_PASS", "arm": "C-SIMPLE-FULL",
                      "seed": 0, "unit_id": report_path.stem,
                      "verified_inner_artifacts": {
                          "filename": artifact_path.name,
                          "sha256": sha(artifact_path),
                          "member_sha256": {"row_id": sha_array(arrays["row_id"])}}}
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(PermissionError, "members"):
                _load_arm(report_path, "C-SIMPLE-FULL", 0)

    def test_aggregate_comparison_rejects_changed_interval(self):
        _same({"lower": .1}, {"lower": .1}, "interval")
        with self.assertRaisesRegex(ValueError, "numerical"):
            _same({"lower": -.1}, {"lower": .1}, "interval")


if __name__ == "__main__":
    unittest.main()
