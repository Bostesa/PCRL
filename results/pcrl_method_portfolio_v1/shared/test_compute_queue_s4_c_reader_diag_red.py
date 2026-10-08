"""Data-free RED gate for one separately sealed S4 C reader diagnostic.

This fixture has no Adult roles, model, prediction or SEX file. It never
starts a child; a valid diagnostic requires its own local Git seal.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

from compute_queue import Queue


class S4CReaderDiagnosticAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve()
        self.script = self.work / "dummy_reader_diag.py"
        self.script.write_text("raise AssertionError('diagnostic child must not launch')\n")
        self.queue = Queue(self.work / "queue")

    def spec(self) -> dict:
        uid = "S4-C-R1-LR-INNER-DIAG-FULL-S0-R1"
        return {
            "unit_id": uid, "mode": "s4_c_reader_diagnostic",
            "family": "S4", "team": "S4",
            "stage": "independent_c_r1_reader_diagnostic",
            "arm": "C-SIMPLE-FULL", "arm_id": "C-SIMPLE-FULL",
            "seed": 0, "view": "r1", "reader": "LR_C1",
            "data_access": "S4_C_R1_INNER_FEATURES_SAVED_READER_DIAGNOSTIC",
            "deps": ["C-R1-SCREEN-READ-C-SIMPLE-FULL-S0-r1"],
            "argv": [sys.executable, str(self.script)],
            "cwd": str(self.work), "env": {},
            "outputs": [str(self.work / "private/DIAGNOSTIC.json")],
            "max_wall_s": 120, "cpu_reserve_s": 120,
            "max_rss_gib": 1.0,
        }

    def test_unsealed_diagnostic_cannot_enqueue(self) -> None:
        with self.assertRaisesRegex(ValueError, "s4_c_diag_local_seal_invalid"):
            self.queue.enqueue(self.spec())

    def test_unsealed_diagnostic_is_denied_at_launch(self) -> None:
        state = self.queue.status()
        state["gate"].update(
            manually_closed=False, occupancy_confirmed=True,
            occupancy_checked_utc=datetime.now(timezone.utc).isoformat(),
            external_heavy=1, safe_inputs=True,
        )
        state["disk_floor_gib"] = 0.0
        self.assertEqual(
            self.queue._gate_status(state, self.spec()),
            "s4_c_diag_local_seal_invalid",
        )


if __name__ == "__main__":
    unittest.main()
