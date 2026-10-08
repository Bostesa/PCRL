"""Bind the corrected S4 C canonical queue-root test to immutable V14."""

from pathlib import Path
import tempfile
import unittest

from compute_queue import Queue
from test_compute_queue_verifier_synthetic import registered_spec


V14 = "S4_VERIFIER_QUEUE_INDEX_V14.json"
V13 = "S4_VERIFIER_QUEUE_INDEX_V13.json"


class S4CVerifierV14AdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_exact_v14_unit_enqueues_without_adult_access(self) -> None:
        spec = registered_spec(V14)
        self.assertEqual(spec["unit_id"], "S4-C-R1-PREACCESS-QUEUE-ROOT-GREEN-014")
        self.assertEqual(Queue(Path(self.temp.name) / "queue").enqueue(spec)["status"], "planned")

    def test_changed_v14_and_historical_v13_are_denied(self) -> None:
        old = registered_spec(V13)
        self.assertEqual(Queue._verifier_synthetic_reason(old), "verifier_test_changed")
        tampered = registered_spec(V14)
        tampered["test_sha256"] = "0" * 64
        self.assertEqual(Queue._verifier_synthetic_reason(tampered), "verifier_test_changed")


if __name__ == "__main__":
    unittest.main()
