"""Data-free replay of the exact A seal metadata that blocked the first FIT."""

import hashlib
import json
from pathlib import Path
import unittest

from .admitted_fit import validate_a_amendment_envelope


CORRECTED_SEAL_SHA256 = "a780c8b328c8649a064534e1bec91a07a0b4d01eb4ef3a74dd7d5d5a548c976a"
CORRECTED_SEAL = (Path(__file__).resolve().parents[2] /
                  "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_PATH_CORRECTION.json")


class AmendmentEnvelopeTest(unittest.TestCase):
    def test_actual_corrected_seal_and_unknown_key_refusal(self):
        payload = CORRECTED_SEAL.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), CORRECTED_SEAL_SHA256)
        amendment = json.loads(payload)
        validate_a_amendment_envelope(amendment)
        self.assertEqual(amendment["status"], "LOCALLY_SEALED_A_STAGE")
        self.assertEqual(amendment["stage_scope"], ["a"])
        with self.assertRaisesRegex(PermissionError, "top-level keys"):
            validate_a_amendment_envelope({**amendment, "unexpected_science_change": 1})
        without_correction_pair = {k: v for k, v in amendment.items()
                                   if k != "engineering_correction"}
        with self.assertRaisesRegex(PermissionError, "top-level keys"):
            validate_a_amendment_envelope(without_correction_pair)


if __name__ == "__main__":
    unittest.main()
