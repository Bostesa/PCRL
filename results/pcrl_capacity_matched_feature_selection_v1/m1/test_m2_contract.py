"""Data-free exact handoff check against M2's shared finite bank parser."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

from capacity import capacity_geometry


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from m2.bank import build_shared_bank  # noqa: E402


def names_from_prior_public_source() -> tuple[str, ...]:
    path = ROOT / "results/pcrl_method_portfolio_v1/teams/c/prototype.py"
    spec = importlib.util.spec_from_file_location("focused_m1_prior_prototype", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FEATURE_NAMES


class M2ContractTests(unittest.TestCase):
    def test_every_mask_and_pair_identity(self) -> None:
        m1 = capacity_geometry(names_from_prior_public_source())
        m2 = build_shared_bank(m1["masks"])
        self.assertEqual(len(m2.masks), 5)
        self.assertEqual(len(m2.pairs), 25)
        by_id = {mask["mask_id"]: mask for mask in m1["masks"]}
        for mask in m2.masks:
            prior = by_id[mask.mask_id]
            self.assertEqual(mask.mask_sha256, prior["mask_sha256"])
            self.assertEqual(mask.column_indices, tuple(prior["column_indices"]))
            self.assertEqual(mask.atoms, tuple(prior["atoms"]))
            self.assertEqual(mask.encoded_columns, prior["encoded_columns"])
        pairs = {pair["pair_id"]: pair for pair in m1["pairs"]}
        for pair in m2.pairs:
            prior = pairs[pair.pair_id]
            self.assertEqual(pair.union_sha256, prior["union_sha256"])
            self.assertEqual(pair.union_columns, tuple(prior["union_column_indices"]))
            self.assertEqual(pair.union_group_count, prior["union_group_count"])
            self.assertEqual(pair.union_encoded_columns, prior["union_encoded_columns"])


if __name__ == "__main__":
    unittest.main()
