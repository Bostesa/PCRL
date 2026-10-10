import unittest

from bank_anchored.r5_replay_utility import _boundary_diagnostic


class IndependentBoundaryDiagnosticTests(unittest.TestCase):
    def test_inclusive_loss_boundary_disagreement_is_retained(self):
        own, prior, disputed = _boundary_diagnostic(
            0.001 - 2e-16, 0.001 + 2e-16, 0.001, "nll")
        self.assertEqual((own, prior, disputed), ("PASS", "FAIL", True))

    def test_no_cap_relaxation(self):
        own, prior, disputed = _boundary_diagnostic(
            0.001 + 2e-16, 0.001 + 4e-16, 0.001, "nll")
        self.assertEqual((own, prior, disputed), ("FAIL", "FAIL", False))

    def test_accuracy_inclusive_reverse_boundary(self):
        own, prior, disputed = _boundary_diagnostic(
            -0.002, -0.002 - 2e-16, -0.002, "accuracy")
        self.assertEqual((own, prior, disputed), ("PASS", "FAIL", True))


if __name__ == "__main__":
    unittest.main()
