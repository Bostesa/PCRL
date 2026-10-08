"""Data-free admission boundary for the committed S4 C replay V12 suite."""

from pathlib import Path
import tempfile
import unittest

from compute_queue import Queue
from test_compute_queue_verifier_synthetic import registered_spec


V12 = "S4_VERIFIER_QUEUE_INDEX_V12.json"
V11 = "S4_VERIFIER_QUEUE_INDEX_V11.json"
V12_UNITS = (
    "S4-C-R1-VALUES-SYNTHETIC-GREEN-012",
    "S4-C-R1-RELEASE-SYNTHETIC-GREEN-012",
    "S4-C-R1-READERS-SYNTHETIC-GREEN-012",
    "S4-C-R1-ENDPOINTS-SYNTHETIC-GREEN-012",
    "S4-C-R1-INFERENCE-SYNTHETIC-GREEN-012",
    "S4-C-R1-PREACCESS-SYNTHETIC-GREEN-012",
)


class S4CVerifierV12AdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_all_six_exact_v12_units_enqueue_without_adult_access(self) -> None:
        from json import loads
        from test_compute_queue_verifier_synthetic import S4

        index = loads((S4 / V12).read_text())
        self.assertEqual(set(index["units"]), set(V12_UNITS))
        queue = Queue(Path(self.temp.name) / "queue")
        for unit_id in V12_UNITS:
            with self.subTest(unit_id=unit_id):
                spec = registered_spec(V12)
                registered = index["units"][unit_id]
                spec.update(
                    unit_id=unit_id,
                    stage=registered["stage"],
                    argv=[*spec["argv"][:4], registered["test_path"]],
                    test_file=str(S4.parent.parent.parent / registered["test_path"]),
                    test_sha256=registered["test_sha256"],
                    source_sha256_by_file=registered["source_sha256_by_file"],
                )
                self.assertEqual(queue.enqueue(spec)["status"], "planned")

    def test_v11_and_changed_v12_identity_are_denied(self) -> None:
        old = registered_spec(V11)
        self.assertEqual(Queue._verifier_synthetic_reason(old),
                         "verifier_index_unregistered")
        spec = registered_spec(V12)
        spec["test_sha256"] = "0" * 64
        self.assertEqual(Queue._verifier_synthetic_reason(spec),
                         "verifier_test_changed")


if __name__ == "__main__":
    unittest.main()
