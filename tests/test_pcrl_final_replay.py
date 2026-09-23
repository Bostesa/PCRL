"""A verifier must pair all three anchors on the union of final households."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.pcrl_final_prospective_v1.independent_replay import bootstrap_se, point_estimates
from experiments.pcrl_final_prospective_v1.audit_panel import unit_dir


class ReplayBootstrapTest(unittest.TestCase):
    def test_rejects_mismatched_person_order(self):
        role = 'attack:A/SEX'
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for anchor in range(3):
                for release, ids in (('Q', ['a', 'b']), ('J', ['b', 'a'])):
                    d = unit_dir(root, release, anchor, role)
                    d.mkdir(parents=True)
                    np.savez_compressed(d/'score_final.npz', ids=np.array(ids),
                                        loss=np.array([.2, .3]), weights=np.ones(2),
                                        households=np.array(['h1', 'h2']))
            endpoint = {'id': 'paired', 'plus': 'Q', 'minus': 'J',
                        'role': role, 'weighting': 'unweighted'}
            with self.assertRaisesRegex(ValueError, 'person order'):
                point_estimates(root, [endpoint])

    def test_household_union_covers_each_anchor(self):
        role = 'attack:A/SEX'
        cache = {}
        for anchor in range(3):
            households = np.array([f'h{i}' for i in range(10) if i != anchor])
            loss = np.arange(len(households), dtype=float)/10
            ids = np.array([f'{anchor}-{i}' for i in range(len(households))])
            cache[('Q', anchor, role)] = (loss, np.ones(len(loss)), households, ids)
            cache[('J', anchor, role)] = (loss[::-1], np.ones(len(loss)), households, ids)
        endpoint = {'id': 'paired', 'plus': 'Q', 'minus': 'J',
                    'role': role, 'weighting': 'unweighted'}
        result = bootstrap_se([endpoint], cache, n_boot=100, seed=9)
        self.assertTrue(np.isfinite(result['paired']))


if __name__ == '__main__':
    unittest.main()
