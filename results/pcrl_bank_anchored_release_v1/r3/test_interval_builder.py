"""Data-free fixed-family interval queue spec assembly."""

import unittest

from bank_anchored.r3_make_interval_config import derive_documents


class IntervalBuilderTests(unittest.TestCase):
    def test_paired_b0_required_and_missing_arms_left_absent(self):
        entries = [{"arm": "B0", "seed": seed,
                    "vector": {"path": f"/x/v{seed}", "sha256": "b" * 64},
                    "receipt": {"path": f"/x/r{seed}", "sha256": "c" * 64}}
                   for seed in (0, 1)]
        args = {"unit_id": "R3_INTERVALS", "stage": "verification",
                "seal_record": {"path": "/x/seal", "sha256": "a" * 64},
                "entries": entries, "config_path": "/x/.private/config",
                "outputs": {"results": "/x/out", "receipt": "/x/receipt"},
                "sources": [{"path": "/x/worker", "sha256": "d" * 64}],
                "amendment_record": {"path": "/x/amend", "sha256": "e" * 64},
                "deps": ["R3_B0_S0_INNER", "R3_B0_S1_INNER"]}
        cfg, spec = derive_documents(**args)
        self.assertEqual(len(cfg["entries"]), 2)
        self.assertEqual(spec["amendment_sha256"], "e" * 64)
        with self.assertRaises(PermissionError):
            derive_documents(**(args | {"entries": entries[:1]}))


if __name__ == "__main__":
    unittest.main()
