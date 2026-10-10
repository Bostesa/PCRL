"""Tiny data-free fit API and exact public-law replay checks."""

import unittest

import numpy as np

from bank import FixedBank
from channel import GateWeights, JointPolicy, expected_cost
from fit import (TaskOnlyData, ProtectedData, FitRecipe, prepare_law,
                 TASK_CAP_ORDER, PROTECTED_CAP_ORDER,
                 exact_metrics, fit_gate, _probabilities_numpy, _torch_metrics)


def fixture():
    p1 = np.array([[.9, .1], [.6, .4]], dtype=np.float64)
    p2 = np.array([[.8, .2], [.7, .3]], dtype=np.float64)
    b1 = FixedBank.from_source(p1, np.arange(2), np.zeros(2, np.int64))
    b2 = FixedBank.from_source(p2, np.arange(2), np.zeros(2, np.int64))
    policy = JointPolicy(b1, b2, (tuple(b1.old_to_global),),
                         (tuple(b2.old_to_global),),
                         GateWeights.zeros(input_width=8))
    task = TaskOnlyData(
        score1=np.array([[.9, .1], [.8, .2], [.7, .3]], dtype=np.float64),
        score2=np.array([[.8, .2], [.7, .3], [.8, .2]], dtype=np.float64),
        cell1=np.zeros(3, dtype=np.int64), cell2=np.zeros(3, dtype=np.int64),
        y1=np.zeros(3, dtype=np.int64), y2=np.zeros(3, dtype=np.int64))
    return task, policy


class FitAPITest(unittest.TestCase):
    def test_task_only_object_has_no_sensitive_attribute(self):
        task, policy = fixture()
        self.assertFalse(hasattr(task, "sex"))
        with self.assertRaises(TypeError):
            fit_gate(task, policy, objective_kind="protected_bayes", caps={},
                     recipe=FitRecipe(updates=1, checkpoint_every=1,
                                      cpu_seconds=1, seed=4))

    def test_prepared_law_uses_fixed_padded_slots_and_full_rows(self):
        task, policy = fixture()
        prepared = prepare_law(task, policy)
        self.assertEqual(prepared.features.shape, (3, 8))
        self.assertEqual(prepared.valid.shape, (3, 16))
        self.assertEqual(prepared.valid.sum(axis=1).tolist(), [4, 4, 4])
        self.assertTrue(np.all(prepared.valid[:, [0, 1, 4, 5]]))
        self.assertEqual(prepared.public_pair.shape, (3, 16))

    def test_exact_metrics_are_expectations_of_draws(self):
        task, policy = fixture()
        metrics = exact_metrics(task, policy)
        one = expected_cost(policy.bank1, policy.bank2,
                            policy.sparse_law(task.score1[0], task.score2[0], 0, 0),
                            y1=0, y2=0)
        self.assertGreater(metrics["r1_nll"], 0)
        self.assertAlmostEqual(one["r1"]["nll"],
                               -0.5 * np.log(.9) - 0.5 * np.log(.6))
        self.assertAlmostEqual(metrics["r1_nll"], sum(
            expected_cost(policy.bank1, policy.bank2,
                          policy.sparse_law(task.score1[i], task.score2[i], 0, 0),
                          y1=0, y2=0)["r1"]["nll"] for i in range(3)) / 3)

    def test_stochastic_task_fit_requires_explicit_recipe_and_caps(self):
        task, policy = fixture()
        caps = {"r1_nll": 1.0, "r1_brier": 1.0,
                "r2_nll": 1.0, "r2_brier": 1.0}
        result = fit_gate(task, policy, objective_kind="task_rate", caps=caps,
                          recipe=FitRecipe(updates=1, checkpoint_every=1,
                                           cpu_seconds=5, seed=4))
        self.assertEqual(len(result["starts"]), 2)
        self.assertTrue(all("checkpoints" in s for s in result["starts"]))
        self.assertEqual(tuple(result["starts"][0]["checkpoints"][0]["residual"]),
                         TASK_CAP_ORDER)
        self.assertAlmostEqual(result["starts"][1]["checkpoints"][0]["metrics"]["task_rate"],
                               0.0, places=12)
        for start in result["starts"]:
            for checkpoint in start["checkpoints"]:
                self.assertIsInstance(checkpoint["weights"], GateWeights)
                replayed = exact_metrics(task, JointPolicy(
                    policy.bank1, policy.bank2, policy.actions1, policy.actions2,
                    checkpoint["weights"]))
                self.assertAlmostEqual(replayed["r1_nll"],
                                       checkpoint["metrics"]["r1_nll"])
        if result["success"]:
            self.assertIsInstance(result["policy"], JointPolicy)
            self.assertTrue(all(result["metrics"][key] <= cap for key, cap in caps.items()))

    def test_protected_fit_requires_true_sensitive_capability(self):
        task, policy = fixture()
        protected = ProtectedData(task=task, sex=np.array([0, 1, 0], np.int64))
        caps = {"r1_nll": 1.0, "r1_brier": 1.0,
                "r2_nll": 1.0, "r2_brier": 1.0}
        result = fit_gate(protected, policy, objective_kind="protected_bayes",
                          caps=caps, recipe=FitRecipe(updates=1,
                          checkpoint_every=1, cpu_seconds=5, seed=4))
        self.assertEqual(len(result["starts"]), 2)
        self.assertIn("pair_bayes", result["starts"][0]["checkpoints"][0]["metrics"])
        self.assertEqual(tuple(result["starts"][0]["checkpoints"][0]["residual"]),
                         PROTECTED_CAP_ORDER)

    def test_torch_and_numpy_use_identical_unpacked_two_by_three_slots(self):
        import torch
        p1 = np.array([[.9, .1], [.6, .4]], np.float64)
        p2 = np.array([[.8, .2], [.7, .3], [.65, .35]], np.float64)
        b1 = FixedBank.from_source(p1, np.arange(2), np.zeros(2, np.int64))
        b2 = FixedBank.from_source(p2, np.arange(3), np.zeros(3, np.int64))
        gate = GateWeights(np.zeros((8, 16)), np.zeros(16), np.zeros((16, 16)),
                           np.arange(16, dtype=np.float64) / 4)
        policy = JointPolicy(b1, b2, (tuple(b1.old_to_global),),
                             (tuple(b2.old_to_global),), gate)
        task = TaskOnlyData(np.array([[.9, .1]], np.float64),
                            np.array([[.8, .2]], np.float64),
                            np.array([0], np.int64), np.array([0], np.int64),
                            np.array([0], np.int64), np.array([0], np.int64))
        prepared = prepare_law(task, policy)
        numpy_q = _probabilities_numpy(prepared, gate)
        logits = torch.as_tensor(gate.b2.copy()).reshape(1, 16)
        torch_q = torch.softmax(logits.masked_fill(
            ~torch.as_tensor(prepared.valid.copy()), -torch.inf), dim=1)
        self.assertTrue(np.allclose(numpy_q, torch_q.numpy(), atol=1e-15, rtol=0))
        self.assertAlmostEqual(float(_torch_metrics(prepared, torch_q,
            sex=None, torch=torch)["r1_nll"]),
            exact_metrics(task, policy)["r1_nll"])
        self.assertEqual(np.flatnonzero(prepared.valid[0]).tolist(),
                         [0, 1, 2, 4, 5, 6])

    def test_cpu_cutoff_suppresses_a_partial_two_start_selection(self):
        task, policy = fixture()
        caps = {"r1_nll": 1.0, "r1_brier": 1.0,
                "r2_nll": 1.0, "r2_brier": 1.0}
        result = fit_gate(task, policy, objective_kind="task_rate", caps=caps,
                          recipe=FitRecipe(updates=1, checkpoint_every=1,
                                           cpu_seconds=1e-9, seed=4))
        self.assertFalse(result["complete"])
        self.assertFalse(result["success"])
        self.assertIsNone(result["policy"])

    def test_checkpoint_sink_preserves_law_and_order_and_fails_closed(self):
        task, policy = fixture()
        caps = {"r1_nll": 1.0, "r1_brier": 1.0,
                "r2_nll": 1.0, "r2_brier": 1.0}
        recipe = FitRecipe(updates=1, checkpoint_every=1,
                           cpu_seconds=5, seed=4)
        control = fit_gate(task, policy, objective_kind="task_rate",
                           caps=caps, recipe=recipe)
        observed = []

        def sink(start, checkpoint):
            observed.append((start, checkpoint["update"],
                             checkpoint["weights"].b2.copy()))

        saved = fit_gate(task, policy, objective_kind="task_rate", caps=caps,
                         recipe=recipe, checkpoint_sink=sink)
        self.assertEqual([(s, u) for s, u, _ in observed],
                         [("identity_biased", 0), ("identity_biased", 1),
                          ("uniform_logit", 0), ("uniform_logit", 1)])
        self.assertTrue(np.array_equal(observed[0][2][1:], np.zeros(15)))
        self.assertEqual(observed[0][2][0], 4.0)
        self.assertTrue(np.array_equal(observed[2][2], np.zeros(16)))
        self.assertEqual(control["diagnostic_best"], saved["diagnostic_best"])
        self.assertTrue(np.array_equal(control["policy"].gate.b2,
                                       saved["policy"].gate.b2))

        def fail(*_):
            raise OSError("checkpoint custody failed")

        with self.assertRaisesRegex(OSError, "custody"):
            fit_gate(task, policy, objective_kind="task_rate", caps=caps,
                     recipe=recipe, checkpoint_sink=fail)

    def test_exact_metrics_scores_actual_hard_policy_law(self):
        class HardFirstPolicy(JointPolicy):
            law_kind = "deterministic_argmax_v1"

            def sparse_law(self, score1, score2, cell1, cell2):
                pairs, _ = self._pairs_and_probabilities(score1, score2, cell1, cell2)
                return ((pairs[0][0], pairs[0][1], 1.0),)

        task, soft = fixture()
        hard = HardFirstPolicy(soft.bank1, soft.bank2, soft.actions1,
                               soft.actions2, soft.gate)
        soft_metric = exact_metrics(task, soft)
        hard_metric = exact_metrics(task, hard)
        first1 = hard.bank1.probability[hard.actions1[0][0], 0]
        self.assertAlmostEqual(hard_metric["r1_nll"], -np.log(first1))
        self.assertNotAlmostEqual(soft_metric["r1_nll"], hard_metric["r1_nll"])


if __name__ == "__main__":
    unittest.main()
