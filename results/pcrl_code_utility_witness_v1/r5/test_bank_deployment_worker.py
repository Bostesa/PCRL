import unittest

import numpy as np

from results.pcrl_code_utility_witness_v1.r5.bank_deployment_worker import decode_public


class BankDeploymentDecoderTest(unittest.TestCase):
    def test_public_code_and_table_independently_determine_wire(self):
        code = np.zeros((3, 10), dtype=np.uint8)
        code[1, 0] = 1
        code[2, 1] = 1
        table = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        probability, decision = decode_public(code, table)
        np.testing.assert_array_equal(probability, table)
        np.testing.assert_array_equal(decision, [0, 1, 0])

    def test_rejects_fractional_and_out_of_bank_codes(self):
        table = np.array([[0.9, 0.1]])
        with self.assertRaisesRegex(ValueError, "hard bits"):
            decode_public(np.full((1, 10), 0.5), table)
        code = np.zeros((1, 10), dtype=np.uint8)
        code[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "outside public table"):
            decode_public(code, table)


if __name__ == "__main__":
    unittest.main()
