"""Data-free RED for a distinct, separately sealed C R3 replay lane."""

from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

from compute_queue import Queue


class S4CR3AdultSealTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name).resolve()
        self.script = self.work / "dummy_r3_replay.py"
        self.script.write_text("raise AssertionError('child must not launch')\n")
        self.queue = Queue(self.work / "queue")

    def spec(self, *, aggregate: bool = False) -> dict:
        uid = ("S4-C-R3-AGGREGATE" if aggregate else
               "S4-C-R3-C-SIMPLE-FULL-S0")
        arm = "C-AGGREGATE" if aggregate else "C-SIMPLE-FULL"
        return {
            "unit_id": uid, "mode": "s4_c_r3_adult_replay",
            "family": "S4", "team": "S4", "arm": arm, "arm_id": arm,
            "seed": None if aggregate else 0,
            "stage": ("independent_c_r3_aggregate" if aggregate else
                      "independent_c_r3_arm_replay"),
            "data_access": ("S4_C_R3_VERIFIED_PRIVATE_AGGREGATE" if aggregate else
                            "S4_C_R3_ATT_INNER_FEATURES_TASK_SEX_REPLAY"),
            "argv": [sys.executable, str(self.script)],
            "cwd": str(self.work), "deps": [], "env": {},
            "outputs": [str(self.work / uid / "REPORT.json")],
            "max_wall_s": 600 if aggregate else 180,
            "cpu_reserve_s": 600 if aggregate else 180,
            "max_rss_gib": 2.0,
        }

    def test_unsealed_r3_arm_and_aggregate_cannot_enqueue(self) -> None:
        for aggregate in (False, True):
            with self.subTest(aggregate=aggregate):
                with self.assertRaisesRegex(ValueError,
                                            "s4_c_r3_local_seal_invalid"):
                    self.queue.enqueue(self.spec(aggregate=aggregate))

    def test_unsealed_r3_unit_is_denied_again_at_launch(self) -> None:
        state = self.queue.status()
        state["gate"].update(
            manually_closed=False, occupancy_confirmed=True,
            occupancy_checked_utc=datetime.now(timezone.utc).isoformat(),
            external_heavy=1, safe_inputs=True,
        )
        state["disk_floor_gib"] = 0.0
        for aggregate in (False, True):
            with self.subTest(aggregate=aggregate):
                self.assertEqual(
                    self.queue._gate_status(state, self.spec(aggregate=aggregate)),
                    "s4_c_r3_local_seal_invalid",
                )


if __name__ == "__main__":
    unittest.main()
