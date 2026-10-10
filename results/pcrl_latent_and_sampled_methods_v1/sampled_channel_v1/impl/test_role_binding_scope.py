"""Data-free check that historical source pins do not widen FIT role access."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .run_admitted import _binding


FIT_KEYS = {"role_store_id", "repair_manifest_sha256", "source_audit_sha256",
            "fit_row_unit_pair_sha256", "source_binding_sha256_by_seed",
            "tstar_bank_receipt_sha256_by_seed"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RoleBindingScopeTest(unittest.TestCase):
    def test_exact_fit_binding_and_separate_historical_source_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            audit = root / "r01/SOURCE_AUDIT.json"
            audit.parent.mkdir()
            audit.write_text("{}\n")
            repair = root / "repair.json"
            repair.write_text("{}\n")
            portfolio = root / "portfolio"
            receipt = (portfolio /
                       "shared/.private/tstar_score_bank_s0_v1/BANK_ADMISSION_RECEIPT.json")
            receipt.parent.mkdir(parents=True)
            receipt.write_text("{}\n")
            historical = root / "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"
            historical.write_text("{}\n")
            binding = {"role_store_id": "UTILITY_REPAIR_ROLE_EXPORT_V1",
                       "repair_manifest_sha256": _sha(repair),
                       "source_audit_sha256": _sha(audit),
                       "fit_row_unit_pair_sha256": "a" * 64,
                       "source_binding_sha256_by_seed": {"s0": "b" * 64,
                                                        "s1": "c" * 64},
                       "tstar_bank_receipt_sha256_by_seed": {"s0": _sha(receipt),
                                                             "s1": "d" * 64}}
            self.assertEqual(set(binding), FIT_KEYS)
            amendment = {"role_bindings": binding,
                         "source_sha256": {
                             "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json":
                                 _sha(historical)}}
            args = Namespace(repair_manifest=str(repair), source_audit=str(audit),
                             portfolio_package=str(portfolio), campaign_root=str(root))
            spec = {"sources": [{"path": str(path), "sha256": _sha(path)}
                                for path in (repair, audit, receipt)]}
            got = _binding(amendment, spec, args, 0)
            self.assertEqual(got[3], binding)
            self.assertEqual(amendment["source_sha256"][
                "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"], _sha(historical))
            widened = {**binding, "historical_audit_sha256": _sha(historical)}
            with self.assertRaisesRegex(PermissionError, "incomplete"):
                _binding({"role_bindings": widened,
                          "source_sha256": amendment["source_sha256"]}, spec, args, 0)


if __name__ == "__main__":
    unittest.main()
