import unittest

import numpy as np

from bank_anchored.r5_refit_reader import fit_fixed_hgb


class IndependentRefitDiagnostic(unittest.TestCase):
    def test_fixed_hgb_recovers_planted_signal(self):
        rng = np.random.default_rng(20261010)
        x = rng.normal(size=(160, 3)).astype(np.float64)
        y = (x[:, 0] > 0).astype(np.int64)
        result = fit_fixed_hgb(x[:120], y[:120], x[120:], y[120:],
                               max_leaf_nodes=15, max_iter=200, seed=7)
        self.assertEqual(result["prediction"].shape, (40,))
        self.assertGreater(result["alias_max_auc"], .85)


if __name__ == "__main__":
    unittest.main()
