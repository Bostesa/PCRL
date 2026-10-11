"""Additional fixed data-free laws, written before engineering lock."""
import unittest
import numpy as np
from conditional_release_independent.contexts import hierarchy_from_counts, independent_mixture
from conditional_release_independent.inference import prepare_group_score_mass, weighted_rank_replicates, slate_seed_summary
from conditional_release_independent.arithmetic import exhaustive_auc


class ContextInferenceTests(unittest.TestCase):
    def test_context_unsupported_and_parent(self):
        counts=[0]*96
        counts[:8]=[10,70,80,80,80,80,80,80]
        counts[8:16]=[0,0,0,0,0,0,0,30]
        report=hierarchy_from_counts(counts)
        self.assertEqual(report['raw_to_context'][:2],[0,0])
        self.assertEqual(report['merged_counts'][0],80)
        self.assertEqual(report['raw_to_context'][8:16],[-1]*8)

    def test_mixture_fallback_and_alias(self):
        rows=independent_mixture([(0,0),(0,0)], [{(0,0):.2,(1,1):.8}]*2, [-1,0],[.5])
        self.assertEqual(rows[0][(0,0)],1)
        self.assertAlmostEqual(rows[1][(0,0)],.6)
        self.assertAlmostEqual(rows[1][(1,1)],.4)

    def test_sparse_group_weighted_rank(self):
        laws=[{0:.25,1:.75},{0:.25,1:.75},{0:.75,1:.25},{.5:1}]
        labels=[1,1,0,0]; groups=['a','a','b','c']
        prepared=prepare_group_score_mass(laws,labels,groups)
        values=weighted_rank_replicates(prepared,[[1,1,1],[2,1,3]])
        self.assertAlmostEqual(values[0],exhaustive_auc(laws,labels))
        self.assertAlmostEqual(values[1],exhaustive_auc(laws,labels,[2,2,1,3]))

    def test_seed_max_before_mean(self):
        value=slate_seed_summary([[np.array([.8,.6]),np.array([.7,.7])],
                                 [np.array([.6,.8]),np.array([.7,.7])]])
        np.testing.assert_array_equal(value,np.array([.75,.75]))


if __name__=='__main__': unittest.main()
