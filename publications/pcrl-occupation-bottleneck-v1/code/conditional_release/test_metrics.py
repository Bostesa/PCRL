"""Read only locked data-free fixtures. Run solely under the shared queue."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import t

from .metrics import integrated_auc, score_group_matrices, matrix_auc
from .inference import paired_group_counts, envelope_distribution, mean_distribution, simultaneous_interval

FIXTURES = Path(__file__).resolve().parents[1] / "results/pcrl_conditional_budgeted_release_v1/r5/EXACT_LAW_FIXTURES.json"


class ExactLawTests(unittest.TestCase):
    def setUp(self):
        self.laws = json.loads(FIXTURES.read_text())

    def test_exact_weighted_ranks_and_exhaustive_outcomes(self):
        for case in self.laws["cases"]:
            scores = np.asarray(case["scores"])
            mass = np.asarray(case["mass"])
            y = np.asarray(case["sex"], dtype=np.int64)
            groups = np.asarray(case["groups"], dtype=np.int64)
            actual = integrated_auc(scores, mass, y, groups=groups)
            total = 0.0
            for a in np.flatnonzero(y == 1):
                for b in np.flatnonzero(y == 0):
                    for i in range(mass.shape[1]):
                        for j in range(mass.shape[1]):
                            total += mass[a, i] * mass[b, j] * (int(scores[a, i] > scores[b, j]) + 0.5 * int(scores[a, i] == scores[b, j]))
            expected = total / (sum(y == 1) * sum(y == 0))
            self.assertAlmostEqual(actual, expected, places=13, msg=case["name"])
            self.assertAlmostEqual(actual, case["auc"], places=13, msg=case["name"])
            matrix = score_group_matrices(scores, mass, y, groups)
            self.assertAlmostEqual(float(matrix_auc(matrix, np.ones(len(matrix["groups"])))), actual, places=13)

    def test_alias_mass_and_reversal(self):
        case = self.laws["cases"][2]
        scores, mass, y = np.array(case["scores"]), np.array(case["mass"]), np.array(case["sex"])
        self.assertAlmostEqual(integrated_auc(1 - scores, mass, y), 1 - case["auc"])

    def test_monte_carlo_matches_integrated_law(self):
        plan = self.laws["monte_carlo"]
        case = next(c for c in self.laws["cases"] if c["name"] == plan["case"])
        rng = np.random.default_rng(plan["seed"])
        neg = rng.choice(2, plan["draws"], p=case["mass"][0])
        pos = rng.choice(2, plan["draws"], p=case["mass"][1])
        auc = np.mean((pos > neg) + 0.5 * (pos == neg))
        self.assertLess(abs(auc - case["auc"]), plan["absolute_tolerance"])

    def test_invalid_law_and_duplicate_cache(self):
        case = self.laws["cases"][-1]
        scores, mass, y, g = map(np.array, (case["scores"], case["mass"], case["sex"], case["groups"]))
        bad = mass.copy(); bad[0, 0] = -0.1
        with self.assertRaises(ValueError): integrated_auc(scores, bad, y)
        with self.assertRaises(ValueError): integrated_auc(scores, mass * 0.9, y)
        bad_y = y.copy(); bad_y[1] = 1
        with self.assertRaises(ValueError): integrated_auc(scores, mass, bad_y, groups=g)
        bad_scores = scores.copy(); bad_scores[1, 0] = 0.4
        with self.assertRaises(ValueError): integrated_auc(bad_scores, mass, y, groups=g)

    def test_group_bootstrap_pairing_and_counts(self):
        g = np.arange(40, dtype=np.int64)
        y = np.repeat(np.arange(2), 20)
        score = (0.1 + 0.8 * y)[:, None]
        mass = np.ones_like(score)
        values, counts = paired_group_counts(g)
        self.assertEqual(counts.shape, (2048, 40))
        np.testing.assert_array_equal(counts.sum(1), 40)
        dist = envelope_distribution({"one": score, "constant": np.full_like(score, 0.5)}, mass, y, g, counts=counts, group_values=values)
        np.testing.assert_allclose(dist["bootstrap"], 1)
        one = mean_distribution(np.arange(40, dtype=float), g, counts=counts, group_values=values)
        two = mean_distribution(np.arange(40, dtype=float) + 2, g, counts=counts, group_values=values)
        np.testing.assert_allclose(two["bootstrap"] - one["bootstrap"], 2)
        with self.assertRaises(ValueError): envelope_distribution({"one": score}, mass, y, g + 1, counts=counts, group_values=values)

    def test_interval_df_and_boundary(self):
        vals = np.linspace(0, 1, 2048)
        out = simultaneous_interval(0.5, vals, K=495, reporting_groups=2234)
        self.assertEqual(out["df"], 2233)
        self.assertAlmostEqual(out["critical_value"], t.ppf(1 - 0.05 / 990, 2233))
        self.assertAlmostEqual(out["upper"] - out["point"], vals.std(ddof=1) * out["critical_value"])


if __name__ == "__main__":
    unittest.main()
