"""Queue-only data-free receipts for R2 activity accounting and lock bridge."""
from __future__ import annotations
import argparse,hashlib,io,json,unittest
from pathlib import Path
import numpy as np
from .r2_tests import tiny_space
from .r2_space import (group_folds,costs,identity_q,reset_preprocess_counts,
                       preprocess_counts)
from .r2_policy import Policy,reset_forward_counts,forward_counts
from .r2_train import initial_weights,_torch_law,reset_training_counts,training_counts
from .r2_worker import reader_fit_counts,_science_lock_bridge
from .r2_readers import Reader


class Tests(unittest.TestCase):
    def test_counters_leave_exact_identity_and_torch_law_unchanged(self):
        import torch
        s=tiny_space();reset_preprocess_counts();reset_forward_counts();reset_training_counts()
        c=np.arange(4,dtype=np.int64)
        p=s.prepare(c,c);q=Policy(s,"P0",()).law_prepared(p)
        group_folds(c);costs(s,p,np.ones(4,np.int64),np.zeros(4,np.int64))
        w=tuple(torch.as_tensor(v,dtype=torch.float64) for v in initial_weights(8,16,16,1,0))
        tq=_torch_law(s,p,(w,),"O_J",0.,0.,torch=torch)
        self.assertTrue(np.array_equal(q,identity_q(p)))
        self.assertTrue(np.array_equal(tq.detach().numpy(),q))
        self.assertEqual(preprocess_counts(),{"group_fold_calls":1,"group_fold_rows":4,
            "prepared_calls":1,"prepared_rows":4,"prepared_action_positions":64,
            "cost_matrix_calls":1,"cost_matrix_rows":4,"cost_matrix_action_positions":64})
        self.assertEqual(forward_counts(),{"numpy_policy_forward_calls":1,
            "numpy_policy_forward_rows":4,"numpy_policy_forward_action_positions":64})
        self.assertEqual(training_counts(),{"torch_policy_forward_calls":1,
            "torch_policy_forward_rows":4,"torch_policy_forward_action_positions":64,
            "optimizer_steps":0,"factor_optimizer_steps":0})

    def test_exact_family_fold_reader_counts_ignore_stage_marker(self):
        loss=np.zeros((4,16),np.float64)
        readers=[Reader(f"stage2/s0r0/pair/{family}","pair",loss,0.,
                        {"f0":np.zeros(1),"f1":np.zeros(1),"f2":np.zeros(1),
                         "stage1_frozen_marker":np.ones(1)},0,0)
                 for family in ("prior","categorical","nonlinear")]
        self.assertEqual(reader_fit_counts(readers),{
            "reader_family_fold_fits":{"prior":3,"categorical":3,"nonlinear":3},
            "trained_reader_fits":6,"constant_prior_fits":3})

    def test_science_and_admission_hashes_are_separate(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            base=Path(temp);admission=base/"ADMISSION_LOCK.json";science=base/"CAMPAIGN_LOCK.json"
            admission.write_text("admission");science.write_text("science")
            ash=hashlib.sha256(admission.read_bytes()).hexdigest();ssh=hashlib.sha256(science.read_bytes()).hexdigest()
            ah=hashlib.sha256(b"engineering-amendment").hexdigest()
            cfg={"pins":{"admission_lock":{"path":str(admission),"sha256":ash},
                         "science_lock":{"path":str(science),"sha256":ssh}},
                 "admission_lock_sha256":ash,"science_lock_sha256":ssh,
                 "source_lock_sha256":ssh,"protection_amendment_sha256":ah}
            active={"seal_sha256":ash,"unit_id":"TEST"}
            registered={admission.resolve():ash,science.resolve():ssh}
            queue={"amendments":[{"sha256":ssh,"stages":["protection"]},
                                 {"sha256":ah,"stages":["protection"]}],
                   "units":{"TEST":{"spec":{"amendment_sha256":ah}}}}
            self.assertEqual(_science_lock_bridge(cfg,active,queue,registered),(ssh,ash))
            cfg["science_lock_sha256"]=ash
            with self.assertRaises(PermissionError):_science_lock_bridge(cfg,active,queue,registered)


def main():
    from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
    ap=argparse.ArgumentParser();ap.add_argument("--unit-id",required=True);ap.add_argument("--output",required=True)
    a=ap.parse_args();context=assert_active_from_env(a.unit_id,"admission")
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt={"schema":"IVR1-r2-accounting-fixtures-v1","status":"PASS" if result.wasSuccessful() else "FAIL",
             "tests_run":result.testsRun,"errors":len(result.errors),"failures":len(result.failures),
             "real_data_reads":False,"queue":context,"log":stream.getvalue()}
    path=Path(a.output)
    with path.open("x") as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps({"status":receipt["status"],"tests_run":result.testsRun,
                      "receipt_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__=="__main__":main()
