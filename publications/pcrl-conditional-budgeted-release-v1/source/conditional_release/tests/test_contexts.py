"""Locked label-free context and common cost correctness fixtures."""
import unittest
import numpy as np
from conditional_release.contexts import ContextMap,fit_context_map,raw_context,RAW_COUNT
from conditional_release.preparation import probability_costs,constraint_rows
from conditional_release.lp import build_mixture_problem


def inputs(classpair=0, margin=.375, highentropy=False, count=64):
    ic,oc=divmod(classpair,6)
    p1=np.array([(1+margin)/2,(1-margin)/2],np.float64)
    if ic:p1=p1[::-1]
    p2=np.full(6,.08 if highentropy else .005,np.float64);p2[oc]=1-p2.sum()+p2[oc]
    return np.tile(p1,(count,1)),np.tile(p2,(count,1))


class ContextTests(unittest.TestCase):
    def test_grid_merges_and_supported_unseen(self):
        xs=[];ys=[]
        for m in (.125,.375,.625,.875):
            for entropy in (False,True):
                x,y=inputs(margin=m,highentropy=entropy);xs.append(x);ys.append(y)
        x,y=np.vstack(xs),np.vstack(ys);model=fit_context_map(x,y)
        self.assertEqual(len(model.labels),8)
        self.assertEqual(model.raw_fit_counts.sum(),512)
        self.assertTrue(np.all(model.apply(x,y)>=0))
        xx,yy=inputs(margin=.376,count=1)
        self.assertGreaterEqual(model.apply(xx,yy)[0],0)
        xx,yy=inputs(classpair=1,count=1)
        self.assertEqual(model.apply(xx,yy)[0],-1)

    def test_hierarchical_sparse_parent(self):
        x,y=inputs(count=64);model=fit_context_map(x,y)
        self.assertEqual(model.labels,("classpair/0",))
        self.assertTrue(np.all(model.mapping[:8]==0))
        x,y=inputs(count=63);model=fit_context_map(x,y)
        self.assertEqual(len(model.labels),0)
        self.assertTrue(np.all(model.mapping==-1))

    def test_bin_ties_go_right(self):
        x,y=inputs(margin=.25,count=1)
        self.assertEqual(raw_context(x,y)[0]//2%4,1)

    def test_expected_loss_is_not_loss_of_mean(self):
        p=np.array([[[.9,.1],[.6,.4]]],np.float64);y=np.array([0],np.int64)
        actual=probability_costs(p,y)["nll"].mean()
        mean=probability_costs(p.mean(1),y)["nll"][0]
        self.assertGreater(actual,mean)
        self.assertAlmostEqual(probability_costs(p,y)["brier"][0,0],.02)

    def test_negative_reference_remaining_is_preserved(self):
        n=6;e1=np.tile(np.array([0,1],np.int64),(n,1));e2=e1.copy();i=np.zeros(n,np.int64)
        law=build_mixture_problem(e1,e2,i,i,np.tile([.5,.5],(n,1)),np.zeros(n,np.int64),"scalar")
        loss=np.tile([.1,.2],(n,1));cost={"r1_nll":loss}
        refs={"U0":{"r1_nll":np.zeros(n)},"Ucal":{"r1_nll":np.zeros(n)}}
        constraints,translations=constraint_rows(law,cost,np.arange(n,dtype=np.int64)%3,refs,{},"ORIGINAL")
        self.assertTrue(np.all(constraints.rhs<0))
        self.assertEqual(len(translations),8)
        self.assertTrue(all(x["negative_rhs"] for x in translations))

    def test_actual_alias_change_and_guard_sign(self):
        e1=np.array([[0,0,1]],np.int64);e2=np.zeros_like(e1);i=np.array([0],np.int64)
        law=build_mixture_problem(e1,e2,i,i,np.array([[.2,.3,.5]]),i,"scalar")
        intercept,coef=law.change_coefficients();self.assertEqual(intercept,0.)
        self.assertEqual(coef[0],.5)


if __name__=="__main__":unittest.main()
