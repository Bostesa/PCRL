import unittest
import numpy as np
from task_conditioned_representation import attacks, inference
from task_conditioned_representation import utility_worker

class ContractTests(unittest.TestCase):
    def test_scale_retains_small_finite_variance(self):
        x=np.array([[0.,0.],[1e-10,2.],[2e-10,4.]])
        z=attacks.FiniteScaler().fit(x).transform(x)
        self.assertGreater(np.std(z[:,0]),.9)
        self.assertEqual(z.shape,x.shape)
    def test_planted_signal_at_1e_minus_25_keeps_unit_variance(self):
        y=np.tile(np.array([0,1]),64)
        x=np.column_stack((1e-25*y,np.ones(len(y))))
        scaler=attacks.FiniteScaler().fit(x)
        z=scaler.transform(x)
        self.assertGreater(scaler.raw_scale_[0],0.)
        self.assertAlmostEqual(float(np.std(z[:,0])),1.)
        self.assertEqual(float(scaler.scale_[1]),1.)
        self.assertTrue(np.array_equal(z[:,1],np.zeros(len(y))))
    def test_fresh_reader_recovers_planted_1e_minus_25_bit(self):
        train_y=np.tile(np.array([0,1]),64)
        train_x=np.column_stack((1e-25*train_y,np.ones(len(train_y))))
        record=attacks.fit_reader(train_x,train_y,'LR_C1',2026101205)
        validation_y=np.tile(np.array([0,1]),16)
        validation_x=np.column_stack((1e-25*validation_y,np.ones(len(validation_y))))
        scores=attacks.predict_reader(record,validation_x)
        self.assertEqual(inference.auc(validation_y,scores),1.)
    def test_nomination_membership_requires_exact_arm_seed_checkpoint(self):
        checkpoint='a'*64
        lock={'instances':[{'arm_id':'JOINT_CONDITIONAL','seed':0,'checkpoint_sha256':checkpoint}],
              'unrelated_notes':{'checkpoint_sha256':'b'*64}}
        self.assertTrue(utility_worker.admitted_checkpoint(lock,'JOINT_CONDITIONAL',0,checkpoint))
        self.assertFalse(utility_worker.admitted_checkpoint(lock,'LOCAL_TWIN',0,checkpoint))
        self.assertFalse(utility_worker.admitted_checkpoint(lock,'JOINT_CONDITIONAL',1,checkpoint))
        self.assertFalse(utility_worker.admitted_checkpoint(lock,'JOINT_CONDITIONAL',0,'b'*64))
        lock['instances'].append(dict(lock['instances'][0]))
        self.assertFalse(utility_worker.admitted_checkpoint(lock,'JOINT_CONDITIONAL',0,checkpoint))
    def test_utility_point_cap_has_no_positive_tolerance(self):
        cap=.01
        self.assertTrue(utility_worker.utility_clause_pass(cap,cap))
        self.assertFalse(utility_worker.utility_clause_pass(np.nextafter(cap,np.inf),cap))
    def test_weighted_1e_minus_25_scaling_retains_variance(self):
        x=np.array([[0.,1.],[1e-25,1.]])
        weight=np.array([3.,1.])
        scaler=attacks.FiniteScaler().fit(x,sample_weight=weight)
        z=scaler.transform(x)
        self.assertAlmostEqual(float(np.average(z[:,0],weights=weight)),0.)
        self.assertAlmostEqual(float(np.average(z[:,0]**2,weights=weight)),1.)
        self.assertEqual(float(scaler.scale_[1]),1.)
    def test_legacy_adapter_uses_positive_empirical_scaler(self):
        from task_conditioned_representation.legacy_attack_worker import legacy_adapter
        self.assertIs(legacy_adapter().StandardScaler,attacks.FiniteScaler)
    def test_primary_complete_wire(self):
        a=np.arange(4*21,dtype=float).reshape(4,21);b=np.arange(4*29,dtype=float).reshape(4,29)
        t=attacks.feature_tables(a,b)
        self.assertEqual(t['pair']['full'].shape,(4,50))
        self.assertTrue(np.array_equal(t['r1']['full'],a))
        self.assertTrue(np.array_equal(t['r2']['full'],b))
        self.assertEqual(t['pair']['representation'].shape,(4,32))
        self.assertEqual(t['pair']['output'].shape,(4,10))
    def test_ties_auc_and_duplicate_weights(self):
        self.assertEqual(inference.auc(np.array([0,1]),np.array([.5,.5])),.5)
        self.assertEqual(inference.auc(np.array([0,1]),np.array([.1,.9])),1.)
        self.assertEqual(inference.auc(np.array([0,1]),np.array([.9,.1])),0.)
    def test_metric_summed_brier(self):
        m=inference.task_metrics(np.array([0,1]),np.array([[.8,.2],[.2,.8]]))
        self.assertAlmostEqual(m['brier'],.08)
        self.assertEqual(m['accuracy'],1.)
    def test_split_group_disjoint(self):
        unit=np.array([1,1,2,3,4,5,6,7])
        train,val=attacks.attack_split(unit,False)
        self.assertFalse(set(unit[train])&set(unit[val]))
    def test_group_bootstrap_shared_duplicates(self):
        unit=np.array([10,10,20,30])
        w=inference.group_bootstrap(unit,8,42)
        self.assertTrue(np.array_equal(w[:,0],w[:,1]))
        self.assertTrue(np.array_equal(w[:,[0,2,3]].sum(1),np.full(8,3)))
    def test_missing_class_rejected(self):
        with self.assertRaises(ValueError):inference.auc(np.zeros(3,dtype=int),np.ones(3))
    def test_family_status_never_promotes_fallback(self):
        self.assertEqual(inference.family_status(False,True),'diagnostic_ineligible')
        self.assertEqual(inference.family_status(True,False),'incomplete')
    def test_output_bypass_and_nonfinite_refused(self):
        with self.assertRaises(ValueError):attacks.feature_tables(np.zeros((3,22)),np.zeros((3,29)))
        a=np.zeros((3,21));b=np.zeros((3,29));a[0,0]=np.inf
        with self.assertRaises(ValueError):attacks.feature_tables(a,b)
    def test_validation_orientation_then_frozen_application(self):
        sign=attacks.orientation(np.array([0,1]),np.array([.9,.1]))
        self.assertEqual(sign,-1)
        self.assertTrue(np.allclose(attacks.orient(np.array([.2,.8]),sign),[.8,.2]))
    def test_shared_rank_auc_bootstrap_matches_explicit_rows(self):
        y=np.array([0,0,1,1]);s=np.array([.1,.5,.5,.9]);w=np.array([[2.,0.,1.,1.],[1.,2.,1.,0.]])
        got=inference.bootstrap_auc(y,s,w)
        for i in range(2):
            index=np.repeat(np.arange(4),w[i].astype(int))
            self.assertAlmostEqual(got[i],inference.auc(y[index],s[index]))
    def test_pair_ignore_other_envelope_monotone(self):
        scores={'r1':{'LR':np.array([.1,.9])},'r2':{'LR':np.array([.9,.1])},'pair':{'LR':np.array([.5,.5])}}
        out=attacks.add_ignore_other(scores)
        self.assertEqual(max(inference.auc(np.array([0,1]),v) for v in out['pair'].values()),1.)
