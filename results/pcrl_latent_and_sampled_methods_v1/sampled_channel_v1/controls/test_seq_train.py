"""Data-free checks for freeze-first sequential fitting and FIT draw custody."""

from pathlib import Path
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "impl"))
from bank import FixedBank
from channel import GateWeights
from fit import FitRecipe, ProtectedData, TaskOnlyData, exact_metrics

from factor_controls import FactorGate, SequentialPolicy, public_probability_embedding
from seq_train import _draw_uniform, fit_sequential


def _fixture(order="12"):
    b1 = FixedBank.from_source(np.array([[.95, .05], [.85, .15], [.75, .25],
                                         [.65, .35]]), np.arange(4),
                               np.zeros(4, dtype=np.int64))
    b2 = FixedBank.from_source(np.array([[.90, .10], [.70, .30]]), np.arange(2),
                               np.zeros(2, dtype=np.int64))
    # Two of the four first-bank IDs are never offered by this source mask.
    a1 = (tuple(map(int, b1.old_to_global[:2])),)
    a2 = (tuple(map(int, b2.old_to_global)),)
    features = b1.probability[a1[0][0]]
    other = b2.probability[a2[0][0]]
    task = TaskOnlyData(np.tile(features, (4, 1)), np.tile(other, (4, 1)),
                        np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64),
                        np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64))
    protected = ProtectedData(task, np.array([0, 0, 1, 1], dtype=np.int64))
    width = 8
    first = FactorGate(np.zeros((width, 16)), np.zeros(16),
                       np.zeros((16, 4)), np.zeros(4))
    first_bank = b1 if order == "12" else b2
    second = FactorGate(np.zeros((width, 16)), np.zeros(16),
                        np.zeros((16, 4)), np.zeros(4),
                        public_probability_embedding(first_bank.probability))
    template = SequentialPolicy(b1, b2, a1, a2, GateWeights.zeros(input_width=8),
                                first, second, order=order,
                                first_checkpoint_sha256="0" * 64)
    return protected, template


def _run(data, template, path=None):
    recipe1 = FitRecipe(updates=2, checkpoint_every=1, cpu_seconds=30,
                        seed=17, dual_every=1)
    recipe2 = FitRecipe(updates=2, checkpoint_every=1, cpu_seconds=30,
                        seed=23, dual_every=1)
    return fit_sequential(
        data, template, caps={"r1_nll": 2.0, "r1_brier": 2.0,
                              "r2_nll": 2.0, "r2_brier": 2.0},
        recipe_stage1=recipe1, recipe_stage2=recipe2, total_cpu_seconds=60,
        unit_ids=("person-A", "person-A", "person-B", "person-C"),
        private_root=b"K" * 32, source_seed=0,
        source_binding_sha256="a" * 64,
        private_draw_cache_path=path,
    )


class SequentialFitTest(unittest.TestCase):
    def test_frozen_compact_json_hmac_vector(self):
        self.assertEqual(
            _draw_uniform(b"K" * 32, 1, "21", "person-A"),
            0.9364098057452417,
        )

    def test_two_stage_fit_uses_one_frozen_first_draw_per_unit(self):
        data, template = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "first-draw.json"
            result = _run(data, template, path)
            self.assertTrue(result["complete"])
            self.assertTrue(result["success"])
            self.assertEqual(len(result["stage1_starts"]), 2)
            self.assertEqual(len(result["stage2_starts"]), 2)
            self.assertEqual(result["private_first_codes"][0],
                             result["private_first_codes"][1])
            self.assertTrue(set(result["private_first_codes"]).issubset(
                set(template.actions1[0])))
            self.assertEqual(result["policy"].first_checkpoint_sha256,
                             result["first_checkpoint_sha256"])
            self.assertEqual(result["metrics"], exact_metrics(data, result["policy"]))
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertGreater(result["coverage"]["unseen_first_code_count"], 0)
            initial = public_probability_embedding(template.bank1.probability)
            for code in result["coverage"]["unseen_first_codes"]:
                np.testing.assert_array_equal(result["policy"].second.embedding[code],
                                              initial[code])

    def test_same_frozen_identity_reuses_cache_and_tampering_refuses(self):
        data, template = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "first-draw.json"
            first = _run(data, template, path)
            second = _run(data, template, path)
            self.assertEqual(first["first_checkpoint_sha256"],
                             second["first_checkpoint_sha256"])
            np.testing.assert_array_equal(first["private_first_codes"],
                                          second["private_first_codes"])
            self.assertEqual(first["draw_cache_sha256"], second["draw_cache_sha256"])
            path.write_text("{}")
            with self.assertRaises(ValueError):
                _run(data, template, path)

    def test_duplicate_unit_with_different_source_scores_refuses(self):
        data, template = _fixture()
        task = data.task
        changed = task.score1.copy()
        changed[1] = [0.9, 0.1]
        different = ProtectedData(TaskOnlyData(changed, task.score2, task.cell1,
                                               task.cell2, task.y1, task.y2), data.sex)
        with self.assertRaises(ValueError):
            _run(different, template)

    def test_duplicate_unit_with_different_label_or_sex_refuses(self):
        data, template = _fixture()
        task = data.task
        changed_y2 = task.y2.copy()
        changed_y2[1] = 1
        different_label = ProtectedData(TaskOnlyData(task.score1, task.score2,
            task.cell1, task.cell2, task.y1, changed_y2), data.sex)
        with self.assertRaises(ValueError):
            _run(different_label, template)
        changed_sex = data.sex.copy()
        changed_sex[1] = 1
        different_sex = ProtectedData(task, changed_sex)
        with self.assertRaises(ValueError):
            _run(different_sex, template)

    def test_symlinked_private_cache_parent_refuses(self):
        data, template = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "owner"
            parent.mkdir(mode=0o700)
            link = Path(directory) / "alias"
            link.symlink_to(parent, target_is_directory=True)
            with self.assertRaises((PermissionError, ValueError)):
                _run(data, template, link / "first-draw.json")

    def test_reverse_order_uses_second_recipient_first_public_code(self):
        data, template = _fixture(order="21")
        result = _run(data, template)
        self.assertTrue(result["success"])
        self.assertEqual(result["policy"].order, "21")
        self.assertTrue(set(result["private_first_codes"]).issubset(
            set(template.actions2[0])))
        self.assertEqual(result["metrics"], exact_metrics(data, result["policy"]))


if __name__ == "__main__":
    unittest.main()
