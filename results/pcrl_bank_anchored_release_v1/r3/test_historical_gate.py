"""Data-free separate historical original-INNER admission checks."""

import unittest

from bank_anchored.r3_historical_gate import original_inner_pass


class HistoricalGateTests(unittest.TestCase):
    def test_wq_requires_all_eight_two_seed_cells_and_24_clauses(self):
        rows = [{"arm": "W_Q", "seed": seed, "task": task,
                 "reference": ref,
                 "point_clauses": {"accuracy": True, "nll": True, "brier": True}}
                for seed in (0, 1) for task in ("income", "occupation")
                for ref in ("U0", "Ucal")]
        audit = {"schema": "pcrl-r5-bank-old-point-audit-v1", "status": "PASS",
                 "summary": {"W_Q": {"clauses_total": 24, "clauses_pass": 24,
                                     "cells_total": 8}}, "rows": rows}
        self.assertTrue(original_inner_pass("W_Q", audit))
        with self.assertRaises(PermissionError):
            original_inner_pass("W_Q", audit | {"rows": rows[:-1]})
        bad = dict(audit, rows=[*rows[:-1], rows[-1] | {
            "point_clauses": {"accuracy": True, "nll": False, "brier": True}}])
        with self.assertRaises(PermissionError):
            original_inner_pass("W_Q", bad)

    def test_pstar_has_separate_original_inner_gate(self):
        cells = [{"seed": seed, "task": task, "reference": ref,
                  "point_pass": True}
                 for seed in (0, 1) for task in ("income", "occupation")
                 for ref in ("U0", "Ucal")]
        audit = {"schema": "pcrl-bank-r5-pstar-common-inner-replay-v1",
                 "role": "INNER_SELECTION", "status": "PASS_UTILITY_POINT_ONLY",
                 "all_24_point_clauses_pass": True, "cells": cells}
        self.assertTrue(original_inner_pass("P_STAR", audit))
        self.assertFalse(original_inner_pass("P_STAR", audit | {
            "cells": [*cells[:-1], cells[-1] | {"point_pass": False}]}))


if __name__ == "__main__":
    unittest.main()
