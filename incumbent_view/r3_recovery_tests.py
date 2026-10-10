"""Data-free frozen reader and crossed inference fixtures for queue admission."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .r3_recovery_intervals import weighted_auc_batch,bootstrap_plan,frozen_envelope,recovery_intervals
from .r3_recovery_worker import inference_lock_bridge,sha


class RecoveryFixtures(unittest.TestCase):
    def test_weighted_ties_and_reverse_orientation(self):
        y=np.array([0,1,0,1],np.int64);g=np.arange(4,dtype=np.int64)
        weights=np.ones((2,4),np.int64)
        a=weighted_auc_batch(y,np.array([0.,1.,1.,2.]),g,weights,groups=4)
        self.assertTrue(np.allclose(a,.875))
        reverse=frozen_envelope(y,{'direct':np.array([0.,1.,1.,2.]),'reverse':np.array([2.,1.,1.,0.])},
                                ('direct','reverse'),g,weights,groups=4)
        self.assertTrue(np.allclose(reverse,.875))

    def test_group_plan_couples_people_and_draws(self):
        unit=np.array([10,10,20,30],np.int64)
        inverse,weights,draws=bootstrap_plan(unit,bootstrap_draws=8,groups=3)
        np.testing.assert_array_equal(inverse,[0,0,1,2])
        np.testing.assert_array_equal(weights[0],[1,1,1])
        self.assertEqual(weights.shape,(9,3));self.assertEqual(draws.shape,(8,2,3))
        self.assertTrue(np.all(weights[1:].sum(1)==3))
        self.assertTrue(np.array_equal(draws,bootstrap_plan(unit,bootstrap_draws=8,groups=3)[2]))

    def test_deterministic_broadcast_and_null_family(self):
        b=4;draws=np.zeros((b,2,3),np.int64);values={}
        for arm,level in (('P0',.71),('V_J',.67)):
            for seed in (0,1):
                for d in (range(3) if arm=='V_J' else (0,)):
                    for view in ('r1','r2','pair'):
                        values[(arm,seed,d,view)]=np.full(b+1,level,np.float64)
        slots,_=recovery_intervals(values,{'P0','V_J'},draws,bootstrap_draws=b,groups=3)
        self.assertEqual(len(slots),756)
        self.assertAlmostEqual(slots['mean:recovery:P0:V_J:pair:auc']['point'],.04)
        self.assertEqual(slots['s0_d1:recovery:P0:V_J:pair:auc']['status'],'BOUND_ONLY')
        self.assertEqual(slots['mean:recovery:P0:V_P:pair:auc']['status'],'NOT_ESTIMABLE_DEPENDENCY')

    def test_inference_science_is_separate_from_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            admission=root/'admission.json';admission.write_text('{}')
            science=root/'science.json';science.write_text('{"science":true}')
            amend=root/'inference.json';amend.write_text(json.dumps({'campaign_lock_sha256':sha(science)}))
            pins={'admission_lock':{'path':str(admission),'sha256':sha(admission)},
                  'campaign_seal':{'path':str(science),'sha256':sha(science)}}
            state={'seal':{'sha256':sha(admission)},'amendments':[
                {'path':str(science),'sha256':sha(science),'stages':['protection']},
                {'path':str(amend),'sha256':sha(amend),'stages':['verification']}],
                'units':{'unit':{'spec':{'amendment_sha256':sha(amend)}}}}
            registered={p.resolve():sha(p) for p in (admission,science,amend)}
            active={'unit_id':'unit','seal_sha256':sha(admission)}
            self.assertEqual(inference_lock_bridge(pins,active,state,registered),(sha(admission),sha(science)))
            bad={'unit_id':'unit','seal_sha256':sha(science)}
            with self.assertRaises(PermissionError):inference_lock_bridge(pins,bad,state,registered)
            amend.write_text(json.dumps({'campaign_lock_sha256':'0'*64}))
            state['amendments'][-1]['sha256']=sha(amend)
            state['units']['unit']['spec']['amendment_sha256']=sha(amend)
            registered[amend.resolve()]=sha(amend)
            with self.assertRaises(PermissionError):inference_lock_bridge(pins,active,state,registered)


if __name__=='__main__':unittest.main()
