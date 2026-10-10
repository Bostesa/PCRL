"""Data-free identity check for P* H2 successor construction."""

import json
from pathlib import Path
import unittest

from . import pstar_h2_builder, reference_worker_h2


class PStarH2BuilderTests(unittest.TestCase):
    def test_only_view_source_and_immutable_destinations_change(self):
        root = Path(__file__).resolve().parent
        old_path = root / ".private" / "units" / "R4_ATTACK_P_STAR_S0_R0_H1.json"
        if not old_path.exists():
            self.skipTest("campaign H1 predecessor receipt not available")
        old = json.loads(old_path.read_text())
        row = {"predecessor_unit_id": old["unit_id"],
               "successor_unit_id": "R4_ATTACK_P_STAR_S0_R0_H2",
               "source_seed": 0, "attacker_seed": old["attacker_seed"]}
        new = pstar_h2_builder._new_spec(old, row)
        self.assertEqual(new["schema"], reference_worker_h2.SPEC_SCHEMA)
        self.assertEqual(new["unit_id"], row["successor_unit_id"])
        self.assertEqual(new["fixed_map_receipt_sha256"],
                         old["fixed_map_receipt_sha256"])
        self.assertEqual(new["att_receipt_sha256"], old["att_receipt_sha256"])
        self.assertEqual(new["inner_receipt_sha256"], old["inner_receipt_sha256"])
        self.assertEqual({Path(pin["path"]).name for pin in new["source_pins"]},
                         reference_worker_h2.REQUIRED_SOURCES)
        with self.assertRaises(PermissionError):
            pstar_h2_builder._new_spec(old, row | {"attacker_seed": 0})


if __name__ == "__main__":
    unittest.main()
