"""Data-free receipt formatting check after the recorded DET smoke failure."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deterministic_trainer import FitResult
from smoke_controls import _result_status


class SmokeStatusTest(unittest.TestCase):
    def test_dataclass_and_dictionary_fit_results(self):
        self.assertEqual(_result_status(FitResult(True, False, None, None, (), 1.0)),
                         (True, False))
        self.assertEqual(_result_status({"complete": True, "success": True}),
                         (True, True))
        with self.assertRaises(TypeError):
            _result_status({"complete": True})


if __name__ == "__main__":
    unittest.main()
