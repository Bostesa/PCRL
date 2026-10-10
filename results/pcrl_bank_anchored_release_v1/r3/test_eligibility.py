"""Data-free per-arm point gate tests."""

import unittest

from bank_anchored.r3_eligibility import summarize_arm, point_nomination_rows


def unit(arm, seed, role, *, pass_all=True):
    count = 6 if role == "OSF_DEFENSE_FIT" else 12
    receipt = {"schema": "pcrl-bank-r3-utility-receipt-v1",
               "arm": arm, "seed": seed, "role": role,
               "stage": "protection", "source_seal_sha256": "a" * 64,
               "primary_clause_count": count,
               "primary_clause_pass_count": count if pass_all else count-1,
               "primary_point_pass": pass_all,
               "output_csv_sha256": "b" * 64}
    rows = []
    refs = ("B0",) if role == "OSF_DEFENSE_FIT" else ("U0", "Ucal")
    for task in ("income", "occupation"):
        for ref in refs:
            for metric in ("accuracy", "nll", "brier"):
                rows.append({"role": role, "arm": arm, "seed": str(seed),
                             "task": task, "reference": ref, "metric": metric,
                             "primary_gate": "True", "point_status": "PASS"})
    if not pass_all:
        rows[-1]["point_status"] = "VIOLATION"
    return receipt, rows


class EligibilityTests(unittest.TestCase):
    def test_all_four_units_required_and_all_point_clauses_counted(self):
        docs = {(seed, role): unit("J", seed, role)
                for seed in (0, 1)
                for role in ("OSF_DEFENSE_FIT", "INNER_SELECTION")}
        result = summarize_arm("J", docs)
        self.assertTrue(result["all_two_seed_task_reference_point_caps_pass"])
        self.assertEqual(result["fit_clauses"], {"passed": 12, "total": 12})
        self.assertEqual(result["inner_clauses"], {"passed": 24, "total": 24})
        docs.pop((1, "INNER_SELECTION"))
        with self.assertRaises(PermissionError):
            summarize_arm("J", docs)

    def test_failed_fit_or_inner_blocks_attack_eligibility(self):
        docs = {(seed, role): unit("J", seed, role)
                for seed in (0, 1)
                for role in ("OSF_DEFENSE_FIT", "INNER_SELECTION")}
        docs[(0, "OSF_DEFENSE_FIT")] = unit("J", 0, "OSF_DEFENSE_FIT", pass_all=False)
        self.assertFalse(summarize_arm("J", docs)["all_two_seed_task_reference_point_caps_pass"])
        docs[(0, "OSF_DEFENSE_FIT")] = unit("J", 0, "OSF_DEFENSE_FIT")
        docs[(1, "INNER_SELECTION")] = unit("J", 1, "INNER_SELECTION", pass_all=False)
        self.assertFalse(summarize_arm("J", docs)["all_two_seed_task_reference_point_caps_pass"])

    def test_nomination_table_retains_missing_arms_and_only_j_as_nominee(self):
        gate = {"schema": "pcrl-bank-r3-arm-point-eligibility-v1", "arm_id": "J",
                "all_two_seed_task_reference_point_caps_pass": True,
                "fit_bank_proxy_pass": True, "inner_original_caps_pass": True}
        rows = point_nomination_rows({"J": gate})
        self.assertEqual(len(rows), 10)
        self.assertEqual(next(r for r in rows if r["arm_id"] == "J")["nomination"], "J_NOMINEE")
        self.assertEqual(next(r for r in rows if r["arm_id"] == "T")["utility_gate"], "NOT_RUN")
        self.assertEqual(next(r for r in rows if r["arm_id"] == "W_Q")["utility_gate"],
                         "HISTORICAL_SEPARATE")
        incomplete = point_nomination_rows({}, {"J"})
        self.assertEqual(next(r for r in incomplete if r["arm_id"] == "J")["nomination"],
                         "INCOMPLETE_SEARCH")


if __name__ == "__main__":
    unittest.main()
