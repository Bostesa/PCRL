"""Data-free, test-first checks for the six-reader C INNER diagnostic."""

import unittest

import numpy as np

from run_c_reader_bank_diagnostic_v2 import summarize_reader_bank


READERS = (
    "LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
    "RAW_LR_C1", "RAW_HGB_15x200",
)


class ReaderBankDiagnosticTests(unittest.TestCase):
    def test_identifies_later_reader_without_exposing_vectors(self) -> None:
        frozen = {name: np.array([0.1, 0.2, 0.2, 0.8], dtype=np.float64)
                  for name in READERS}
        replayed = {name: value.copy() for name, value in frozen.items()}
        replayed["MLP_64x32"][2] = np.nextafter(0.2, 1.0)
        report = summarize_reader_bank(frozen, replayed)
        self.assertEqual(report["reader_order"], list(READERS))
        self.assertEqual(report["exact_mismatch_readers"], ["MLP_64x32"])
        self.assertEqual(report["first_mismatch_reader"], "MLP_64x32")
        self.assertEqual(report["by_reader"]["LR_C1"]["mismatch_count"], 0)
        self.assertEqual(report["by_reader"]["MLP_64x32"]["rank_changed_count"], 2)
        self.assertNotIn("predictions", str(report))

    def test_missing_reader_fails_closed(self) -> None:
        values = {name: np.array([0.1, 0.9]) for name in READERS}
        del values["RAW_HGB_15x200"]
        with self.assertRaises(ValueError):
            summarize_reader_bank(values, values)


if __name__ == "__main__":
    unittest.main()
