"""Data-free admission negative tests; execute only in the shared queue."""
import unittest
import numpy as np
from incumbent_view.r4_worker import validate_gate, validate_seal_bridge, load_sex, check_law


class AdmissionTests(unittest.TestCase):
    def gate(self,passed=24):
        return {"schema":"pcrl-incumbent-r3-original-inner-point-gate-v1","arm_id":"V_J",
                "seeds":[0,1],"all_two_seed_task_reference_point_caps_pass":passed==24,
                "original_inner":{"required":24,"passed":passed,"status":"PASS" if passed==24 else "FAIL"},
                "fit_proxy":{"status":"FAIL"},
                "source_receipts":{k:{} for k in ("s0_fit","s0_inner","s1_fit","s1_inner")}}

    def test_original_inner_pass_does_not_conjoin_fit(self):
        validate_gate(self.gate(),{"arm_id":"V_J"})

    def test_inner_failure_blocks_fresh_fit(self):
        with self.assertRaises(PermissionError):validate_gate(self.gate(22),{"arm_id":"V_J"})

    def test_wrong_arm_gate_rejected(self):
        with self.assertRaises(PermissionError):validate_gate(self.gate(),{"arm_id":"V_S21"})

    def test_admission_queue_seal_is_distinct_from_campaign_science_lock(self):
        spec={"queue_seal":{"sha256":"a"*64},"campaign_seal":{"sha256":"b"*64}}
        lock={"queue_seal_sha256":"a"*64,"campaign_seal_sha256":"b"*64}
        validate_seal_bridge(spec,"a"*64,lock)
        with self.assertRaises(PermissionError):validate_seal_bridge(spec,"b"*64,lock)
        with self.assertRaises(PermissionError):validate_seal_bridge(spec,"a"*64,{**lock,"campaign_seal_sha256":"c"*64})
        with self.assertRaises(PermissionError):validate_seal_bridge({**spec,"campaign_seal":spec["queue_seal"]},"a"*64,
                                                                        {**lock,"campaign_seal_sha256":"a"*64})

    def test_unadmitted_sex_role_refused_before_file_access(self):
        calls=[]
        def pinned(pin):calls.append(pin);raise AssertionError("file should remain unopened")
        with self.assertRaises(PermissionError):load_sex({"sha256":"a"*64},"HEAD_VALIDATION",{},pinned)
        self.assertEqual(calls,[])

    def test_wrong_sex_hash_refused_before_file_access(self):
        calls=[]
        def pinned(pin):calls.append(pin);raise AssertionError("file should remain unopened")
        manifest={"roles":{"ATTACK_FIT_NEW":{"source_files":{"sex":{"sha256":"b"*64}}}}}
        with self.assertRaises(PermissionError):load_sex({"sha256":"a"*64},"ATTACK_FIT_NEW",manifest,pinned)
        self.assertEqual(calls,[])

    def test_sample_outside_positive_support_rejected(self):
        bank={"bank_r1":np.array([[.8,.2],[.7,.3]],np.float64),
              "bank_r2":np.array([[.5,.1,.1,.1,.1,.1]],np.float64)}
        law={"row_id":np.array([1],np.int64),"unit":np.array([1],np.int64),
             "code_r1":np.zeros((1,16),np.int64),"code_r2":np.zeros((1,16),np.int64),
             "mass":np.zeros((1,16),np.float64),"valid":np.ones((1,16),bool),
             "b0_code_r1":np.zeros(1,np.int64),"b0_code_r2":np.zeros(1,np.int64)}
        law["mass"][0,0]=1
        arrays={"row_id":law["row_id"],"unit":law["unit"],
                "code_r1":np.ones(1,np.int64),"code_r2":np.zeros(1,np.int64)}
        with self.assertRaises(PermissionError):check_law(law,arrays,bank)
        arrays["code_r1"][:]=0;check_law(law,arrays,bank)


if __name__=="__main__":unittest.main()
