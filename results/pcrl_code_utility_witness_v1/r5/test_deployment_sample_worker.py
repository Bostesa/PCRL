"""Data-free guards for the measured T0 deployment batch comparator."""

import unittest

import numpy as np

from results.pcrl_code_utility_witness_v1.r5.deployment_sample_worker import check_recipient_wire


class DeploymentWireComparatorTest(unittest.TestCase):
    def setUp(self):
        self.saved = {"code": np.array([[0, 1], [1, 0]], np.uint8),
                      "probabilities": np.array([[.7, .3], [.2, .8]], np.float64),
                      "decision": np.array([0, 1], np.int64)}
        self.got = {key: value.copy() for key, value in self.saved.items()}

    def test_tiny_batch_arithmetic_gap_passes_with_exact_code_and_decision(self):
        self.got["probabilities"][0, 0] += 9e-8
        self.got["probabilities"][0, 1] -= 9e-8
        value = check_recipient_wire(self.got, self.saved,
                                     self.got["probabilities"].copy(), 2e-7)
        self.assertAlmostEqual(value, 9e-8)

    def test_code_decision_decoder_or_large_probability_gap_refused(self):
        with self.assertRaisesRegex(ValueError, "code"):
            bad = dict(self.got, code=np.array([[1, 1], [1, 0]], np.uint8))
            check_recipient_wire(bad, self.saved, bad["probabilities"], 2e-7)
        with self.assertRaisesRegex(ValueError, "decision"):
            bad = dict(self.got, decision=np.array([1, 1], np.int64))
            check_recipient_wire(bad, self.saved, bad["probabilities"], 2e-7)
        with self.assertRaisesRegex(ValueError, "decoder"):
            check_recipient_wire(self.got, self.saved,
                                 np.array([[.8, .2], [.2, .8]]), 2e-7)
        with self.assertRaisesRegex(ValueError, "probability"):
            bad = dict(self.got, probabilities=np.array([[.7000003, .2999997], [.2, .8]]))
            check_recipient_wire(bad, self.saved, bad["probabilities"], 2e-7)


if __name__ == "__main__":
    unittest.main()
