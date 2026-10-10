"""Frozen data-free correctness fixtures; queue-only numerical test entry.

No role data read. Tests assess implementation, not synthetic method success.
"""
from __future__ import annotations
import unittest
import numpy as np
from .r2_space import Space,group_folds,costs,identity_q,utility_receipt,decimal_cost_check,accurate_margin
from .r2_policy import Policy
from .r2_train import initial_weights,backoff,_torch_law,_bayes_torch,_constraints,_objective
from .r2_space import bayes_objective
from .r2_torch_cache import FrozenRound
from .r2_readers import weighted_counts,categorical_probability,wire_features,fit_round,reader_guard_receipt,Reader
from .r2_projection import law_coefficients,constraint_coefficients


def tiny_space():
    p1=np.asarray([[.45,.55],[.15,.85],[.4,.6],[.2,.8]],np.float64)
    p2=np.asarray([[.7,.3],[.6,.4],[.8,.2],[.9,.1]],np.float64)
    a=np.asarray([[i]+[j for j in range(4) if j!=i] for i in range(4)],np.int64)
    return Space(p1,p2,a,a,np.ones((4,4),bool),np.ones((4,4),bool))


class Tests(unittest.TestCase):
    def test_identity_and_float_decimal(self):
        space=tiny_space();code=np.arange(4,dtype=np.int64);p=space.prepare(code,code)
        q=Policy(space,"P0",()).law_prepared(p)
        self.assertTrue(np.array_equal(q,identity_q(p)))
        y1=np.ones(4,np.int64);y2=np.zeros(4,np.int64)
        cost=costs(space,p,y1,y2)
        self.assertTrue(all(accurate_margin(q,c,k)[0]==0 for k,c in cost.items()))
        self.assertTrue(all(v==0 for v in decimal_cost_check(space,p,q,y1,y2).values()))

    def test_product_joint_deterministic_boundaries(self):
        s=tiny_space();c=np.arange(4,dtype=np.int64);p=s.prepare(c,c)
        # Nonfactorable correlated slots, same shared 480-param gate.
        w=list(initial_weights(8,16,16,1,1));w[3][0]=3;w[3][5]=3
        j=Policy(s,"V_J",(tuple(w),),1.,1.).law_prepared(p).reshape(-1,4,4)
        product=Policy(s,"V_P",(tuple(w),),.4,.8).law_prepared(p).reshape(-1,4,4)
        self.assertGreater(abs(j[0,0,0]-j[0].sum(0)[0]*j[0].sum(1)[0]),.01)
        self.assertTrue(np.allclose(product,product.sum(2)[:,:,None]*product.sum(1)[:,None,:],rtol=0,atol=1e-15))
        d=Policy(s,"V_D",(tuple(w),),1.,1.).law_prepared(p)
        self.assertTrue(np.isin(d,(0,1)).all())
        for arm in ("V_J","V_P"):
            self.assertTrue(np.array_equal(Policy(s,arm,(tuple(w),),0.,0.).law_prepared(p),identity_q(p)))

    def test_sequential_freeze_and_support(self):
        s=tiny_space();c=np.arange(4,dtype=np.int64);p=s.prepare(c,c)
        ww=(initial_weights(8,12,4,2,1),initial_weights(8,12,4,3,1))
        for arm in ("V_S12","V_S21"):
            a=Policy(s,arm,ww,.3,0.).law_prepared(p).reshape(-1,4,4)
            b=Policy(s,arm,ww,.3,.7).law_prepared(p).reshape(-1,4,4)
            axis=2 if arm=="V_S12" else 1
            self.assertTrue(np.allclose(a.sum(axis),b.sum(axis),rtol=0,atol=1e-15))
            self.assertTrue(np.array_equal(Policy(s,arm,ww,0.,0.).law_prepared(p),identity_q(p)))

    def test_family_coin_api(self):
        s=tiny_space();w=(initial_weights(8,16,16,5,1),);c=np.arange(4,dtype=np.int64)
        p=Policy(s,"V_P",w,.5,.5)
        with self.assertRaises(ValueError):p.sample_codes(c,c,np.full(4,.5))
        codes=p.sample_codes(c,c,np.full((4,2),.5))
        self.assertEqual(codes[0].shape,(4,))
        x=p.sample_public_pair(s.probability1[0],s.probability2[0],0,0,(.1,.9))
        self.assertTrue(all(isinstance(z,int) for z in x))

    def test_groups_and_all_local_guards(self):
        units=np.repeat(np.arange(30,dtype=np.int64),2);fold=group_folds(units)
        self.assertTrue(np.array_equal(fold[::2],fold[1::2]))
        self.assertEqual(set(fold),{0,1,2})
        q=np.zeros((60,16),np.float64);q[:,1]=1
        better=np.ones_like(q);better[:,1]=2
        worse=np.ones_like(q);worse[:,1]=0
        readers=[Reader("good","r1",better,1.,{},0,0),Reader("bad","r1",worse,1.,{},0,0)]
        self.assertFalse(reader_guard_receipt(q,readers,fold)["feasible"])

    def test_mean_trap_weighted_actual_categorical(self):
        s=tiny_space();code=np.asarray([0,1,2,3],np.int64);p=s.prepare(code,np.zeros(4,np.int64))
        q=identity_q(p);sex=np.asarray([0,0,1,1],np.int64)
        self.assertEqual(np.mean(s.probability1[:2,1]),np.mean(s.probability1[2:,1]))
        count=weighted_counts(p["code_r1"],4,q,sex,np.ones(4,bool))
        prediction=categorical_probability(count,.5)
        self.assertTrue(np.all(prediction[:2]<.5) and np.all(prediction[2:]>.5))
        self.assertGreater(np.linalg.norm(wire_features(s,"r1",np.arange(4))[0]-wire_features(s,"r1",np.arange(4))[2]),0)

    def test_expected_actual_loss_not_mean_probability(self):
        s=tiny_space();p=s.prepare(np.asarray([0],np.int64),np.asarray([0],np.int64))
        q=np.zeros((1,16),np.float64);q[0,0]=.5;q[0,4]=.5
        actual=.5*(-np.log(.55)-np.log(.85));mean_loss=-np.log(.7)
        c=costs(s,p,np.asarray([1],np.int64),np.asarray([0],np.int64))
        self.assertAlmostEqual(float(np.sum(q*c["r1_nll"])),actual,places=15)
        self.assertGreater(actual-mean_loss,.01)

    def test_prefit_decimal_safety_envelope(self):
        s=tiny_space();code=np.tile(np.arange(4,dtype=np.int64),5);p=s.prepare(code,code)
        rng=np.random.default_rng(3819);q=rng.dirichlet(np.ones(16),len(code)).astype(np.float64)
        y1=np.ones(len(code),np.int64);y2=np.zeros(len(code),np.int64)
        high=decimal_cost_check(s,p,q,y1,y2);cc=costs(s,p,y1,y2)
        error=max(abs(accurate_margin(q,c,k)[0]-high[k]) for k,c in cc.items())
        self.assertLess(error,1e-12)
        print("Decimal50 numerical engineering max error",format(error,".17g"),"safety margin",1e-10)

    def test_weighted_readers_constant_null(self):
        s=tiny_space();c=np.tile(np.arange(4,dtype=np.int64),9);p=s.prepare(c,c)
        # Every duplicated input has both SEX labels, folds contain group pairs.
        unit=np.repeat(np.arange(18,dtype=np.int64),2)
        sex=np.tile(np.asarray([0,1],np.int64),18)
        fold=group_folds(unit);q=np.full((36,16),1/16,np.float64)
        readers=fit_round(s,p,q,sex,fold,round_index=0,start_index=0,seed=7,views=("r1",))
        self.assertEqual(len(readers),3)
        for reader in readers:
            self.assertTrue(np.isfinite(reader.loss).all())
        prior=next(r for r in readers if r.name.endswith("/prior"))
        self.assertTrue(np.allclose(prior.loss,np.log(2),atol=1e-15,rtol=0))
        self.assertTrue(reader_guard_receipt(identity_q(p),readers,fold)["feasible"])

    def test_polynomial_vs_direct_all_constraints(self):
        s=tiny_space();code=np.tile(np.arange(4,dtype=np.int64),9);p=s.prepare(code,code)
        fold=group_folds(np.repeat(np.arange(18,dtype=np.int64),2))
        cc=costs(s,p,np.ones(36,np.int64),np.zeros(36,np.int64))
        readers=[Reader("r1/test","r1",np.arange(16,dtype=np.float64)[None,:].repeat(36,0),0.,{},0,0)]
        weights=(initial_weights(8,16,16,8,0),)
        for arm in ("V_J","V_P"):
            q=[Policy(s,arm,weights,e,e).law_prepared(p) for e in (0.,.5,1.)]
            coeff=law_coefficients(*q)
            constraints=constraint_coefficients(coeff,cc,fold,readers)
            for eta in (0.,.03125,.05,.5,1.):
                direct=Policy(s,arm,weights,eta,eta).law_prepared(p)
                poly=coeff[0]+eta*coeff[1]+eta*eta*coeff[2]
                self.assertTrue(np.allclose(poly,direct,atol=5e-16,rtol=0))
                direct_constraints=constraint_coefficients((direct,np.zeros_like(direct),np.zeros_like(direct)),cc,fold,readers)
                for (v,_,_,_), (d,_,_,_) in zip(constraints,direct_constraints):
                    self.assertAlmostEqual(v[0]+eta*v[1]+eta*eta*v[2],d[0],places=14)

    def test_old_objective_torch_matches_numpy_on_2d_ids(self):
        import torch
        s=tiny_space();code=np.tile(np.arange(4,dtype=np.int64),9);p=s.prepare(code,code)
        sex=np.tile(np.asarray([0,1],np.int64),18)
        rng=np.random.default_rng(12);q=rng.dirichlet(np.ones(16),len(code))
        actual=float(_bayes_torch(p,torch.as_tensor(q),sex,torch))
        expected=bayes_objective(p,q,sex)[0]
        self.assertAlmostEqual(actual,expected,places=13)

    def test_s21_uses_recipient_domain_coins(self):
        s=tiny_space();code=np.asarray([0],np.int64)
        ww=(initial_weights(8,12,4,2,1),initial_weights(8,12,4,3,1))
        policy=Policy(s,"V_S21",ww,1.,1.);law=policy.law_prepared(s.prepare(code,code)).reshape(4,4)
        for coin1,coin2 in ((.05,.12),(.9,.2),(.1,.95)):
            # Recipient 2 draws first from coin2; recipient 1 uses coin1
            # conditional on its ACTUAL emitted recipient-2 output.
            b=int((np.cumsum(law.sum(0))<=coin2).sum().clip(0,3))
            a=int((np.cumsum(law[:,b]/law[:,b].sum())<=coin1).sum().clip(0,3))
            expected=(int(s.actions1[0,a]),int(s.actions2[0,b]))
            actual=tuple(int(x[0]) for x in policy.sample_codes(code,code,np.asarray([[coin1,coin2]])))
            self.assertEqual(actual,expected)

    def test_cached_round_matches_direct_loss_guards(self):
        import torch
        s=tiny_space();code=np.tile(np.arange(4,dtype=np.int64),9);p=s.prepare(code,code)
        fold=group_folds(np.repeat(np.arange(18,dtype=np.int64),2));rng=np.random.default_rng(414)
        q=torch.as_tensor(rng.dirichlet(np.ones(16),36))
        c=costs(s,p,np.ones(36,np.int64),np.zeros(36,np.int64))
        readers=[Reader(f"{v}/fake{i}",v,rng.random((36,16)),0.,{},0,0)
                 for v in ("r1","r2","pair") for i in range(3)]
        cache=FrozenRound(c,fold,readers,torch=torch)
        expected=_constraints(q,c,fold,readers,torch)
        actual=cache.constraints(q)
        self.assertEqual(len(actual),len(expected))
        self.assertLess(max(abs(float(a-b)) for a,b in zip(actual,expected)),1e-13)
        self.assertAlmostEqual(float(cache.objective(q,"V_J")),float(_objective(q,p,None,readers,"V_J",torch)),places=13)


def main():
    # Even these reader fixture fits use the new shared queue.
    import argparse,hashlib,io,json
    from pathlib import Path
    from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
    parser=argparse.ArgumentParser();parser.add_argument("--unit-id",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();context=assert_active_from_env(args.unit_id,"admission")
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt={"schema":"IVR1-r2-correctness-fixtures-v1","status":"PASS" if result.wasSuccessful() else "FAIL",
             "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
             "queue":context,"log":stream.getvalue(),"real_data_reads":False,"numeric_safety":1e-10}
    path=Path(args.output)
    with path.open("x") as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps({"status":receipt["status"],"tests_run":result.testsRun,"receipt_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=="__main__":main()
