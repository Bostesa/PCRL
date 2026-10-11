"""Fixed synthetic action-cost parity; no optimizer or Adult input."""
import unittest
from types import SimpleNamespace
import numpy as np
from conditional_release.secondary import bank_action_costs
from conditional_release.preparation import task_costs
from conditional_release.reader_extension import losses_for_actions

class SecondaryCostTests(unittest.TestCase):
    def test_full_bank_lookup_matches_exact_action_loss(self):
        space=SimpleNamespace(probability1=np.array([[.9,.1],[.6,.4],[.2,.8]],np.float64),
                              probability2=np.array([[.7,.1,.05,.05,.05,.05],[.1,.6,.1,.1,.05,.05]],np.float64))
        p={'code_r1':np.array([[0,1,2],[2,0,1]],np.int64),'code_r2':np.array([[0,1,0],[1,0,1]],np.int64)}
        y=(np.array([0,1],np.int64),np.array([0,4],np.int64))
        original=task_costs(space,p,y);lookup=bank_action_costs(space,p,y)
        for key in original:np.testing.assert_array_equal(original[key],lookup[key])
    def test_lazy_guard_lookup_preserves_fold_sensitive_class(self):
        table=np.arange(3*2*4,dtype=np.float64).reshape(3,2,4)
        tables={'guard':{'view':'r1','table':table}}
        ids=np.array([[0,1],[2,3]],np.int64);fold=np.array([0,2],np.int64);sex=np.array([1,0],np.int64)
        losses=losses_for_actions(tables,ids,ids,fold,sex)
        np.testing.assert_array_equal(losses['guard'],np.array([[4,5],[18,19]],np.float64))
        self.assertEqual(len(losses),1)

if __name__=='__main__':unittest.main()
