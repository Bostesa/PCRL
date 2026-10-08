"""Synthetic test-first boundary for the separate S4 C replay lane.

The fixture has a dummy script and no Adult role files. It never launches a
worker; both enqueue and launch admission must refuse an unsealed unit.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

from compute_queue import Queue


class S4CAdultSealTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.script = self.work / "dummy_replay.py"
        self.script.write_text("raise AssertionError('must not launch')\n")
        self.queue = Queue(self.work / "queue")

    def spec(self, *, aggregate: bool = False) -> dict:
        uid = ("S4-C-R1-AGGREGATE" if aggregate else
               "S4-C-R1-C-SIMPLE-FULL-S0")
        return {
            "unit_id": uid, "mode": "s4_c_adult_replay",
            "family": "S4", "team": "S4",
            "arm": "C-AGGREGATE" if aggregate else "C-SIMPLE-FULL",
            "arm_id": "C-AGGREGATE" if aggregate else "C-SIMPLE-FULL",
            "seed": None if aggregate else 0,
            "stage": ("independent_c_r1_aggregate" if aggregate else
                      "independent_c_r1_arm_replay"),
            "data_access": ("S4_C_R1_VERIFIED_PRIVATE_AGGREGATE" if aggregate else
                            "S4_C_R1_ATT_INNER_FEATURES_TASK_SEX_REPLAY"),
            "argv": [sys.executable, str(self.script)],
            "cwd": str(self.work), "deps": [],
            "env": {}, "outputs": [str(self.work / uid / "REPORT.json")],
            "max_wall_s": 600 if aggregate else 180,
            "cpu_reserve_s": 600 if aggregate else 180,
            "max_rss_gib": 2.0,
        }

    def test_unsealed_arm_and_aggregate_cannot_enqueue(self) -> None:
        for aggregate in (False, True):
            with self.subTest(aggregate=aggregate):
                with self.assertRaisesRegex(ValueError, "s4_c_local_seal_invalid"):
                    self.queue.enqueue(self.spec(aggregate=aggregate))

    def test_unsealed_unit_is_denied_again_at_launch(self) -> None:
        state = self.queue.status()
        state["gate"].update(
            manually_closed=False, occupancy_confirmed=True,
            occupancy_checked_utc=datetime.now(timezone.utc).isoformat(),
            external_heavy=1, safe_inputs=True,
        )
        self.assertEqual(
            self.queue._gate_status(state, self.spec()),
            "s4_c_local_seal_invalid",
        )


if __name__ == "__main__":
    unittest.main()
