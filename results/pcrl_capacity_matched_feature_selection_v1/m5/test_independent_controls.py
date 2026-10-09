"""Synthetic independent strongest-control arithmetic and identity checks."""

from __future__ import annotations

import copy
import unittest

from independent_controls import replay_strongest_controls


RULES = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")


def fixture():
    profiles = {"income": {"numeric_groups": 4, "categorical_groups": 4,
                            "encoded_columns_max": 77},
                "occupation_group": {"numeric_groups": 4, "categorical_groups": 4,
                                     "encoded_columns_max": 77}}
    pair = {rule: ("B__B" if rule == "I" else "A__A" if rule in ("T", "SHARED")
                   else "A__B") for rule in RULES}
    union = {rule: ({"source_groups": 8, "encoded_columns": 66}
                    if rule == "I" else {"source_groups": 8, "encoded_columns": 42}
                    if rule in ("T", "SHARED") else
                    {"source_groups": 9, "encoded_columns": 83}) for rule in RULES}
    auc = {rule: (0.60 if rule == "I" else 0.66 if rule in ("T", "SHARED")
                  else 0.70) for rule in RULES}
    plans, endpoints = {}, {}
    for seed in (0, 1):
        choices = {}
        reports = {}
        for rule in RULES:
            a, b = pair[rule].split("__")
            choices[rule] = {
                "status": "SELECTED", "pair_id": pair[rule],
                "income_mask_id": a, "occupation_mask_id": b,
                "recipient_capacity_signatures": copy.deepcopy(profiles),
                "union_capacity_signature": copy.deepcopy(union[rule]),
                "union_sha256": ("f" if union[rule]["source_groups"] == 9 else a.lower()) * 64,
            }
            task = {}
            for purpose in ("income", "occupation_group"):
                refs = {}
                for ref in ("U0", "Ucal", "F"):
                    ineligible = rule == "I_U" and seed == 1 and ref == "F"
                    refs[ref] = {
                        "accuracy_difference": -0.02 if ineligible else 0.0,
                        "nll_excess": 0.0, "brier_excess": 0.0,
                        "point_eligible": not ineligible,
                    }
                task[purpose] = {"references": refs}
            reports[rule] = {**choices[rule], "seed": seed, "rule_id": rule,
                             "task": task,
                             "recovery": {"coalition": {"primary_envelope_auc": auc[rule]}}}
        plans[seed] = {"schema": "pcrl-focused-m3-control-plan-v1", "rules": choices}
        endpoints[seed] = reports
    return plans, endpoints


class IndependentControlTests(unittest.TestCase):
    def test_distinct_main_and_union_pools_keep_aliases(self):
        plans, endpoints = fixture()
        out = replay_strongest_controls(plans, endpoints, expected_seeds=(0, 1))
        self.assertEqual(out["main"]["rule_id"], "I")
        self.assertEqual(out["attribution"]["rule_id"], "T_U")
        self.assertTrue(out["attribution"]["ties_joint_mask_pair"])
        self.assertIn("I_U", out["attribution"]["excluded_rule_ids"])
        self.assertIn("S21", out["attribution"]["eligible_rule_ids"])

    def test_numeric_utility_and_endpoint_identity_are_recomputed(self):
        plans, endpoints = fixture()
        endpoints[1]["I_U"]["task"]["income"]["references"]["F"]["point_eligible"] = True
        with self.assertRaisesRegex(ValueError, "utility flag"):
            replay_strongest_controls(plans, endpoints, expected_seeds=(0, 1))
        plans, endpoints = fixture()
        endpoints[0]["T"]["union_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            replay_strongest_controls(plans, endpoints, expected_seeds=(0, 1))


if __name__ == "__main__":
    unittest.main()
