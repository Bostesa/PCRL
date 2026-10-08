"""Data-free admission boundary for the corrected S4 C replay preaccess test."""

from pathlib import Path
import tempfile
import unittest

from compute_queue import Queue
from test_compute_queue_verifier_synthetic import S4, registered_spec


V13 = "S4_VERIFIER_QUEUE_INDEX_V13.json"
V12 = "S4_VERIFIER_QUEUE_INDEX_V12.json"
UNIT = "S4-C-R1-PREACCESS-SYNTHETIC-GREEN-R1-013"


class S4CVerifierV13AdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_exact_corrected_preaccess_unit_enqueues(self) -> None:
        from json import loads

        index = loads((S4 / V13).read_text())
        self.assertEqual(set(index["units"]), {UNIT})
        registered = index["units"][UNIT]
        spec = registered_spec(V13)
        spec.update(
            unit_id=UNIT,
            stage=registered["stage"],
            argv=[*spec["argv"][:4], registered["test_path"]],
            test_file=str(S4.parent.parent.parent / registered["test_path"]),
            test_sha256=registered["test_sha256"],
            source_sha256_by_file=registered["source_sha256_by_file"],
        )
        self.assertEqual(Queue(Path(self.temp.name) / "queue").enqueue(spec)["status"], "planned")

    def test_old_preaccess_test_bytes_and_v13_tamper_are_denied(self) -> None:
        from json import loads

        old = registered_spec(V12)
        prior = loads((S4 / V12).read_text())["units"]["S4-C-R1-PREACCESS-SYNTHETIC-GREEN-012"]
        old.update(
            unit_id="S4-C-R1-PREACCESS-SYNTHETIC-GREEN-012",
            argv=[*old["argv"][:4], prior["test_path"]],
            test_file=str(S4.parent.parent.parent / prior["test_path"]),
            test_sha256=prior["test_sha256"],
        )
        self.assertEqual(Queue._verifier_synthetic_reason(old), "verifier_test_changed")
        changed = registered_spec(V13)
        changed["test_sha256"] = "0" * 64
        self.assertEqual(Queue._verifier_synthetic_reason(changed), "verifier_test_changed")


if __name__ == "__main__":
    unittest.main()
