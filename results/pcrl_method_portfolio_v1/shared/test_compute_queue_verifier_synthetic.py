"""Synthetic checks for the independent S4 verifier queue lane."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from compute_queue import Queue


REPO = Path(__file__).resolve().parents[3]
S4 = REPO / "results/pcrl_method_portfolio_v1/s4"
VENV = REPO.parent.parent / ".venv/bin/python"
INDEX_NAMES = ("S4_VERIFIER_QUEUE_INDEX_V3.json", "S4_VERIFIER_QUEUE_INDEX_V5.json",
               "S4_VERIFIER_QUEUE_INDEX_V6.json", "S4_VERIFIER_QUEUE_INDEX_V7.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registered_spec(index_name: str) -> dict:
    index_path = S4 / index_name
    index = json.loads(index_path.read_text())
    unit_id, registered = next(iter(index["units"].items()))
    test_path = REPO / registered["test_path"]
    return {
        "unit_id": unit_id,
        "mode": "verifier_synthetic",
        "stage": registered["stage"],
        "family": "S4", "team": "S4",
        "data_access": "NONE", "cwd": str(REPO),
        "argv": [str(VENV), "-m", "pytest", "-q", registered["test_path"]],
        "env": {**{k: v for k, v in index["env"].items() if k != "PYTHONPATH"},
                "PYTHONPATH": str(S4), "PYTHONDONTWRITEBYTECODE": "1"},
        "deps": [], "outputs": [],
        "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 1.0,
        "test_file": str(test_path), "test_sha256": registered["test_sha256"],
        "source_sha256_by_file": registered["source_sha256_by_file"],
        "verifier_index_path": str(index_path), "verifier_index_sha256": sha(index_path),
    }


class VerifierSyntheticAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.queue = Queue(Path(self.temp.name) / "queue")

    def test_registered_red_and_green_units_enqueue_without_adult_seal(self):
        for name in INDEX_NAMES:
            with self.subTest(index=name):
                spec = registered_spec(name)
                self.assertEqual(self.queue.enqueue(spec)["status"], "planned")

    def test_wrong_test_argv_and_superseded_index_are_denied(self):
        spec = registered_spec(INDEX_NAMES[0])
        for change in (
            {"test_sha256": "0" * 64},
            {"argv": [str(VENV), "-m", "pytest", "-q", "other.py"]},
            {"verifier_index_path": str(S4 / "S4_VERIFIER_QUEUE_INDEX_V2.json")},
            {"data_access": "ADMITTED_INNER_LABELS"},
            {"cpu_reserve_s": 121},
        ):
            with self.subTest(change=change):
                altered = dict(spec, **change)
                with self.assertRaises((ValueError, KeyError)):
                    Queue(Path(self.temp.name) / f"negative-{len(str(change))}").enqueue(altered)

    def test_green_source_map_is_exact(self):
        spec = registered_spec(INDEX_NAMES[2])
        wrong = dict(spec, source_sha256_by_file={
            "results/pcrl_method_portfolio_v1/s4/verifier_identity_v3.py": "0" * 64})
        with self.assertRaises(ValueError):
            self.queue.enqueue(wrong)


if __name__ == "__main__":
    unittest.main()
