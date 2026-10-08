"""Synthetic specialist cell-reader checks; run only through S3's queue."""

import unittest
import tempfile
import hashlib
import json
from pathlib import Path

import numpy as np

from attack_views import AlignedDViews
from d_cell_slate import fit_d_cell_readers, load_private_cell_reader
from s2_attack_adapter import as_s2_attack_view
from test_s2_attack_adapter import StubAttackView, code_release


class TinyS2Split:
    @staticmethod
    def attack_fit_tune_split(unit):
        assert len(unit) == 12
        return {"fit": np.arange(8), "tune": np.arange(8, 12)}


def fixture():
    row = np.arange(12, dtype=np.int64)
    r0 = code_release(0, np.tile([0, 0, 1, 1], 3))
    r1 = code_release(1, np.tile([0, 1, 0, 1], 3))
    aligned = AlignedDViews(row, row, row, row, r0, r1, "a" * 64)
    view = as_s2_attack_view(aligned, "pair", StubAttackView)
    sex = {"row_id": row, "unit": row, "sex": np.tile([0, 1, 1, 0], 3)}
    return aligned, view, sex


class DCellSlateTests(unittest.TestCase):
    def test_three_locked_reader_fits_use_attack_train_only(self):
        att, att_view, sex = fixture()
        inner, inner_view, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".private"
            root.mkdir(mode=0o700)
            models = root / "models"
            readers = fit_d_cell_readers(att, inner, "pair", att_view, inner_view,
                                         sex, TinyS2Split, model_dir=models, private_root=root)
            self._check_readers(readers, att, models, root)
            changed_tune = {**sex, "sex": sex["sex"].copy()}
            changed_tune["sex"][8:] = 1 - changed_tune["sex"][8:]
            again = fit_d_cell_readers(att, inner, "pair", att_view, inner_view,
                                       changed_tune, TinyS2Split,
                                       model_dir=root / "models2", private_root=root)
            self.assertEqual([r["model_sha256"] for r in readers],
                             [r["model_sha256"] for r in again])
            for a, b in zip(readers, again):
                np.testing.assert_array_equal(a["prediction_inner"], b["prediction_inner"])

    def _check_readers(self, readers, att, models, root):
        self.assertEqual([r["name"] for r in readers], ["D_CELL_A0.5", "D_CELL_A1", "D_CELL_A5"])
        for reader in readers:
            self.assertEqual(reader["fit_role"], "ATTACK_FIT_NEW/ATTACK_TRAIN")
            self.assertEqual(len(reader["model_sha256"]), 64)
            self.assertEqual(Path(reader["model_artifact_path"]).parent.resolve(),
                             models.resolve())
            self.assertEqual(hashlib.sha256(Path(reader["model_artifact_path"]).read_bytes()).hexdigest(),
                             reader["model_sha256"])
            self.assertEqual(hashlib.sha256(Path(reader["prediction_artifact_path"]).read_bytes()).hexdigest(),
                             reader["prediction_sha256"])
            complete = json.loads(Path(reader["completion_receipt_path"]).read_text())
            self.assertEqual(complete["status"], "COMPLETE")
            self.assertEqual(complete["fit_rows"], 8)
            self.assertEqual(hashlib.sha256(Path(reader["completion_receipt_path"]).read_bytes()).hexdigest(),
                             reader["completion_receipt_sha256"])
            restored = load_private_cell_reader(reader["model_artifact_path"],
                                                 reader["model_sha256"], private_root=root)
            np.testing.assert_array_equal(restored.predict_proba(att.codes("pair"))[:, 1],
                                          reader["prediction_inner"])
            self.assertEqual(reader["prediction_tune"].shape, (4,))
            self.assertEqual(reader["prediction_inner"].shape, (12,))
            self.assertGreater(reader["prediction_tune"][1], reader["prediction_tune"][0])

    def test_misaligned_attack_sex_is_refused(self):
        att, att_view, sex = fixture()
        bad = {**sex, "row_id": sex["row_id"][::-1]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".private"
            root.mkdir(mode=0o700)
            models = root / "models"
            with self.assertRaises(ValueError):
                fit_d_cell_readers(att, att, "pair", att_view, att_view,
                                   bad, TinyS2Split, model_dir=models, private_root=root)
            att_view.cell_key = att_view.cell_key[::-1]
            with self.assertRaises(ValueError):
                fit_d_cell_readers(att, att, "pair", att_view, att_view,
                                   sex, TinyS2Split, model_dir=models, private_root=root)


if __name__ == "__main__":
    unittest.main()
