"""The queue must launch the virtual-environment entry, not its base target."""

from pathlib import Path
import sys
import tempfile
import unittest

from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_build_protection_specs import (
    INDEX, UNITS, python_entry, python_import_smoke,
)


class InterpreterEntryTests(unittest.TestCase):
    def test_g1_keeps_completed_b0_seed_zero_out_of_new_queue(self):
        self.assertEqual(len(UNITS), 15)
        self.assertNotIn(("B0", 0), UNITS)
        self.assertIn(("B0", 1), UNITS)
        self.assertEqual(INDEX.name, "R2_PROTECTION_SOURCE_INDEX_G1.json")

    def test_symlink_entry_is_preserved_and_can_import_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "python_alias"
            alias.symlink_to(Path(sys.executable))
            self.assertEqual(python_entry(alias), str(alias))
            self.assertNotEqual(python_entry(alias), str(alias.resolve()))
        entry = python_entry(sys.executable)
        python_import_smoke(entry)


if __name__ == "__main__":
    unittest.main()
