"""Synthetic public-table replay for inherited score-routed bank attacks."""

import unittest

import numpy as np

from .attack_worker import validate_archive_arrays
from .bank_service import make_bank_service
from .test_attack_worker import fixture


def table_fixture():
    t0 = np.array([[.9, .1], [.5, .5]], dtype=np.float64)
    t1 = np.array([[.9, .02, .02, .02, .02, .02],
                   [1 / 6] * 6,
                   [.1, .9, 0, 0, 0, 0]], dtype=np.float64)
    item = {"policy_sha256": "p" * 64, "policy_fingerprint": "f" * 64,
            "decoder_sha256": None,
            "table0": t0.tolist(), "table1": t1.tolist(),
            "token_class0": [0, 0], "token_class1": [0, 0, 1]}
    return {"schema": "code-utility-public-bank-tables-v1",
            "seed": 0, "bit_count": 10,
            "source_model_sha256": "s" * 64,
            "feature_schema_sha256": "x" * 64,
            "families": {"W_Q": item}}


class BankServiceTest(unittest.TestCase):
    def test_exact_public_table_reconstructs_code_only_wire(self):
        doc = table_fixture()
        service = make_bank_service(doc, arm_id="W_Q", seed=0,
                                    source_model_sha256="s" * 64,
                                    feature_schema_sha256="x" * 64,
                                    public_table_sha256="t" * 64,
                                    policy_sha256="p" * 64,
                                    decoder_sha256=None)
        arrays, meta, _ = fixture(bits=10)
        arrays["prob0"] = service.probabilities_from_code(0, arrays["code0"])
        arrays["prob1"] = service.probabilities_from_code(1, arrays["code1"])
        arrays["decision0"] = arrays["prob0"].argmax(axis=1).astype(np.int64)
        arrays["decision1"] = arrays["prob1"].argmax(axis=1).astype(np.int64)
        wire = validate_archive_arrays(arrays, meta, service, bits=10)
        self.assertEqual(wire["first"]["metadata"]["public_table_sha256"], "t" * 64)

    def test_rejects_invalid_table_binding_and_unknown_code(self):
        doc = table_fixture()
        with self.assertRaisesRegex(ValueError, "policy"):
            make_bank_service(doc, arm_id="W_Q", seed=0,
                              source_model_sha256="s" * 64,
                              feature_schema_sha256="x" * 64,
                              public_table_sha256="t" * 64,
                              policy_sha256="z" * 64,
                              decoder_sha256=None)
        service = make_bank_service(doc, arm_id="W_Q", seed=0,
                                    source_model_sha256="s" * 64,
                                    feature_schema_sha256="x" * 64,
                                    public_table_sha256="t" * 64,
                                    policy_sha256="p" * 64,
                                    decoder_sha256=None)
        code = np.zeros((1, 10), dtype=np.uint8)
        code[0, 2] = 1
        with self.assertRaisesRegex(ValueError, "table"):
            service.probabilities_from_code(0, code)


if __name__ == "__main__":
    unittest.main()
