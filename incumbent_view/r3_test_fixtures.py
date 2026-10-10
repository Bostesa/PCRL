"""Data-free frozen fixture algorithms, executed only inside an R0 queue unit."""
import unittest
import numpy as np
from incumbent_view.r3_utility import evaluate_law, evaluate_realization, decimal_fit_check, stable_mean, point_clause
from incumbent_view.r3_family import ordered_clause_ids, oriented_recovery_slot, K, ARMS
from incumbent_view.r3_selection import original_inner_gate, nominate_family, strongest_control
from incumbent_view.r3_intervals import expected_utility_intervals, FIELDS


def fixture():
    tables = (np.array([[.55,.45],[.85,.15],[.60,.40],[.80,.20]],np.float64),
              np.array([[.5,.1,.1,.1,.1,.1],[.4,.12,.12,.12,.12,.12]],np.float64))
    law={"code_r1":np.array([[0,1],[2,3]],np.int64),"code_r2":np.zeros((2,2),np.int64),
         "mass":np.full((2,2),.5,np.float64),"valid":np.ones((2,2),bool),
         "b0_code_r1":np.array([0,2],np.int64),"b0_code_r2":np.zeros(2,np.int64)}
    labels=(np.zeros(2,np.int64),np.zeros(2,np.int64))
    return law,tables,labels


def gate(arm,fail=False,fit_fail=False):
    clauses=[{"seed":s,"task":t,"reference":r,"metric":m,"status":"PASS"}
             for s in (0,1) for t in ("income","occupation") for r in ("U0","Ucal") for m in ("accuracy","nll","brier")]
    if fail: clauses[-1]["status"]="FAIL"
    return original_inner_gate(arm,clauses,{}, {"status":"FAIL" if fit_fail else "PASS"})


class UtilityFixtures(unittest.TestCase):
    def test_expected_loss_is_not_loss_of_mean(self):
        law,tables,labels=fixture(); out=evaluate_law(law,tables,labels)
        self.assertAlmostEqual(out["income"]["expected"]["nll"][0],(-np.log(.55)-np.log(.85))/2)
        self.assertGreater(out["income"]["expected"]["nll"][0],-np.log(.70))
    def test_expected_equals_enumerated_draw_costs(self):
        law,tables,labels=fixture(); out=evaluate_law(law,tables,labels)
        draws=[evaluate_realization((law["code_r1"][:,j],law["code_r2"][:,j]),tables,labels) for j in (0,1)]
        for task in ("income","occupation"):
            for metric in ("accuracy","nll","brier"):
                np.testing.assert_array_equal(out[task]["expected"][metric],(draws[0][task][metric]+draws[1][task][metric])/2)
    def test_identity_exact(self):
        law,tables,labels=fixture(); law["mass"][:]=[1,0]
        out=evaluate_law(law,tables,labels)
        for task in out:
            for m in out[task]["expected"]: np.testing.assert_array_equal(out[task]["expected"][m],out[task]["p0"][m])
        for rec in decimal_fit_check(law,tables,labels,out).values(): self.assertEqual(rec["margin"],0)
    def test_decimal_error_validation(self):
        law,tables,labels=fixture();out=evaluate_law(law,tables,labels)
        for rec in decimal_fit_check(law,tables,labels,out).values(): self.assertEqual(rec["numerical_admission"],"PASS")
    def test_invalid_mass_rejected(self):
        law,tables,labels=fixture();law["valid"][0,1]=False
        with self.assertRaises(ValueError):evaluate_law(law,tables,labels)
    def test_invalid_public_id_rejected(self):
        law,tables,labels=fixture();law["code_r1"][0,1]=4
        with self.assertRaises(ValueError):evaluate_law(law,tables,labels)
    def test_accuracy_and_brier_convention(self):
        law,tables,labels=fixture();out=evaluate_law(law,tables,labels)
        self.assertEqual(out["income"]["expected"]["accuracy"][0],1.)
        self.assertAlmostEqual(out["income"]["expected"]["brier"][0],(.45**2*2+.15**2*2)/2)
    def test_stable_paired_accumulation(self): self.assertEqual(stable_mean(np.array([1e16,1,-1e16],np.float64)),1/3)
    def test_inclusive_boundaries(self):
        for metric,cap in (("accuracy",-.01),("nll",.01),("brier",.005)):
            self.assertEqual(point_clause(metric,cap)["status"],"PASS")
    def test_mean_output_trap(self):
        _,tables,_=fixture();p=tables[0][:,0]
        self.assertAlmostEqual(p[:2].mean(),p[2:].mean());self.assertFalse(set(p[:2])&set(p[2:]))


class SelectionFixtures(unittest.TestCase):
    def test_fit_failure_does_not_block_inner_gate(self): self.assertTrue(gate("V_J",fit_fail=True)["all_two_seed_task_reference_point_caps_pass"])
    def test_23_of_24_negative_gate_is_valid(self): self.assertFalse(gate("V_S21",fail=True)["all_two_seed_task_reference_point_caps_pass"])
    def test_nomination_minimizes_reader_gain(self):
        gates={a:gate(a) for a in ("V_J","V_P")};records={}
        for a,ce in (("V_J",.60),("V_P",.65)):
            for s in (0,1): records[(a,s)]={"constant_prior_ce":.7,"crossfit_manifest_sha256":"a"*64,"pair_reader_losses":{"reader":ce},"pair_min_ce":ce,"reader_recovery_gap":.7-ce,"exact_p0_alias":False}
        self.assertEqual(nominate_family(gates,records)["nominee"],"V_P")
    def test_prior_included_and_alias_not_gain(self):
        records={("V_J",s):{"constant_prior_ce":.7,"crossfit_manifest_sha256":"a"*64,"pair_reader_losses":{"bad":.9},"pair_min_ce":.7,"reader_recovery_gap":0.,"exact_p0_alias":True} for s in (0,1)}
        result=nominate_family({"V_J":gate("V_J")},records)
        self.assertIsNone(result["nominee"]);self.assertEqual(result["slate"][0]["equal_seed_mean_reader_gain"],0)
    def test_strongest_control_missing_is_incomplete(self):
        self.assertEqual(strongest_control("V_J",{"P0":gate("P0")},{})["status"],"INCOMPLETE")
    def test_complete_fixed_family(self): self.assertEqual(len(ordered_clause_ids()),948)
    def test_contrast_orientation(self):
        name,sign=oriented_recovery_slot("V_J","P0","pair");self.assertEqual(sign,1)
        self.assertEqual(oriented_recovery_slot("P0","V_J","pair"),(name,-1))
    def test_empty_family_retains_all_utility_nulls(self): self.assertEqual(len(expected_utility_intervals({})),192)
    def test_duplicate_group_paired_interval_fixture(self):
        row=np.arange(4,dtype=np.int64);unit=np.array([0,0,1,2],np.int64)
        record={f:np.zeros(4,np.float64) for f in FIELDS-{ "row_id","unit"}};record.update(row_id=row,unit=unit)
        out=expected_utility_intervals({("P0",0):record},expected_groups=3,bootstrap_draws=16)
        self.assertEqual(out["s0:utility:P0:income:U0:nll"]["se"],0)
        self.assertEqual(out["s1:utility:P0:income:U0:nll"]["status"],"NOT_ESTIMABLE_DEPENDENCY")


if __name__=="__main__": unittest.main()
