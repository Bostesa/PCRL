"""Data-free input and custody tests for queued R3 interval reporting."""

import unittest

from bank_anchored.r3_interval_worker import validate_config, validate_entry_receipt


def config():
    return {
        "schema": "pcrl-bank-r3-interval-job-v1",
        "unit_id": "R3_INTERVALS_V1",
        "stage": "verification",
        "seal": {"path": "/tmp/seal", "sha256": "a" * 64},
        "entries": [
            {"arm": "B0", "seed": seed,
             "vector": {"path": f"/tmp/v{seed}", "sha256": "b" * 64},
             "receipt": {"path": f"/tmp/r{seed}", "sha256": "c" * 64}}
            for seed in (0, 1)
        ],
        "outputs": {"results": "/tmp/results", "receipt": "/tmp/receipt"},
    }


class IntervalWorkerTests(unittest.TestCase):
    def test_config_requires_paired_b0_and_exact_entry_order(self):
        good = config()
        validate_config(good, {"unit_id": "R3_INTERVALS_V1", "stage": "verification"})
        duplicate = config()
        duplicate["entries"].append(duplicate["entries"][1])
        with self.assertRaises(PermissionError):
            validate_config(duplicate, {"unit_id": "R3_INTERVALS_V1", "stage": "verification"})
        missing = config()
        missing["entries"].pop()
        with self.assertRaises(PermissionError):
            validate_config(missing, {"unit_id": "R3_INTERVALS_V1", "stage": "verification"})

    def test_receipt_binds_inner_vector_and_same_seal(self):
        entry = config()["entries"][0]
        receipt = {"schema": "pcrl-bank-r3-utility-receipt-v1",
                   "role": "INNER_SELECTION", "stage": "witness",
                   "arm": "B0", "seed": 0, "primary_clause_count": 12,
                   "source_seal_sha256": "a" * 64,
                   "private_vector_sha256": "b" * 64}
        validate_entry_receipt(entry, receipt, "a" * 64)
        for key, value in (("role", "OSF_DEFENSE_FIT"),
                           ("private_vector_sha256", "d" * 64),
                           ("source_seal_sha256", "e" * 64)):
            bad = dict(receipt, **{key: value})
            with self.assertRaises(PermissionError):
                validate_entry_receipt(entry, bad, "a" * 64)

    def test_queue_guard_precedes_vector_read(self):
        import inspect
        from bank_anchored import r3_interval_worker
        source = inspect.getsource(r3_interval_worker.run)
        self.assertLess(source.index("assert_active_from_env"), source.index("np.load"))


if __name__ == "__main__":
    unittest.main()
