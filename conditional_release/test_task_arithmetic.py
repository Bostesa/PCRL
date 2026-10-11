import json
from pathlib import Path
import unittest

from .task_aggregate_worker import absolute_cost,point_margins


class TestTaskArithmetic(unittest.TestCase):
    def test_frozen_dual_reference_arithmetic(self):
        p=Path(__file__).resolve().parents[1]/"results/pcrl_conditional_budgeted_release_v1/r5/TASK_ARITHMETIC_FIXTURES.json"
        f=json.loads(p.read_text())
        def flatten(value):return {(ref,key):margin for ref,values in value.items() for key,margin in values.items()}
        p0,arm=flatten(f["p0_margin"]),flatten(f["arm_margin"])
        for key,expected in f["expected_cost"].items():
            with self.subTest(metric=key):
                actual,diff=absolute_cost(f["p0_cost"],p0,arm,key)
                self.assertAlmostEqual(actual,expected,places=14)
                self.assertLessEqual(diff,f["agreement_limit"])
        arm[("Ucal","r1_nll")]+=f["rejected_calibration_disagreement"]
        with self.assertRaises(PermissionError):absolute_cost(f["p0_cost"],p0,arm,"r1_nll")
        with self.assertRaises(PermissionError):point_margins({"inner_point_utility":[]})


if __name__=="__main__":unittest.main()
