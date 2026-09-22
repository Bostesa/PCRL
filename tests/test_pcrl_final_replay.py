"""A verifier must pair all three anchors on the union of final households."""
import unittest

import numpy as np

from experiments.pcrl_final_prospective_v1.independent_replay import bootstrap_se


class ReplayBootstrapTest(unittest.TestCase):
    def test_household_union_covers_each_anchor(self):
        role = 'attack:A/SEX'
        cache = {}
        for anchor in range(3):
            households = np.array([f'h{i}' for i in range(10) if i != anchor])
            loss = np.arange(len(households), dtype=float)/10
            cache[('Q', anchor, role)] = (loss, np.ones(len(loss)), households)
            cache[('J', anchor, role)] = (loss[::-1], np.ones(len(loss)), households)
        endpoint = {'id': 'paired', 'plus': 'Q', 'minus': 'J',
                    'role': role, 'weighting': 'unweighted'}
        result = bootstrap_se([endpoint], cache, n_boot=100, seed=9)
        self.assertTrue(np.isfinite(result['paired']))


if __name__ == '__main__':
    unittest.main()
