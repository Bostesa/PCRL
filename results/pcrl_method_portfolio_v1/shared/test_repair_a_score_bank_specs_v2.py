"""Metadata-only checks for the one-field A score-bank spec repair."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import repair_a_score_bank_specs_v2 as repair
from safe_data_guard import sha_file


class AScoreBankSpecRepairTests(unittest.TestCase):
    def test_only_cache_root_is_added(self) -> None:
        old = {"unit_id": "fixed", "argv": ["python", "runner"], "seed": 0}
        updated = repair._repair_one(old, Path("/private/cache"))
        self.assertEqual(updated, {**old, "cache_root": "/private/cache"})
        self.assertEqual(old, {"unit_id": "fixed", "argv": ["python", "runner"], "seed": 0})
        with self.assertRaises(ValueError):
            repair._repair_one(updated, Path("/private/cache"))

    def test_preserved_ten_specs_match_sealed_plan(self) -> None:
        index_path = repair.V1_ROOT / "SPEC_INDEX.json"
        self.assertEqual(sha_file(index_path), repair.V1_INDEX_SHA256)
        index = json.loads(index_path.read_text())
        specs, preserved = repair.build_repaired_specs(
            index["campaign_lock_sha256"], index["local_seal_receipt_sha256"])
        self.assertEqual(preserved, index)
        self.assertEqual(len(specs), 10)
        self.assertEqual({s["unit_id"] for s in specs},
                         set(json.loads((repair.CAMPAIGN / "CAMPAIGN_LOCK.json").read_text())
                             ["admitted_a_score_bank_unit_ids"]))
        cache = str((repair.ROOT / "../../../PCRL_eval_cache_private").resolve())
        for spec in specs:
            original_path = repair.V1_ROOT / f"{spec['unit_id']}.json"
            original = json.loads(original_path.read_text())
            self.assertEqual({k: v for k, v in spec.items() if k != "cache_root"}, original)
            self.assertEqual(spec["cache_root"], cache)
            self.assertEqual(spec["argv"][3], cache)
        self.assertFalse(repair.V2_ROOT.exists())


if __name__ == "__main__":
    unittest.main()
