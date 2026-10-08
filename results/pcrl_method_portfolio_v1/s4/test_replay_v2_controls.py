"""Synthetic oracle for independent V2 control replay; no Adult role access.

This test intentionally precedes replay_v2_controls.py. It creates a tiny
feature law at the real ATT/INNER row geometry, then writes a private probe
package with the same member names as the queued V2 control protocol.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from replay_v2_controls import (ATT_SALT, INNER_SALT, group_shuffle,
                                replay_probe)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(value) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256((str(a.dtype) + str(a.shape)).encode() + a.tobytes()).hexdigest()


def reference_shuffle(y, groups, salt):
    seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    output = np.empty_like(y)
    by_size = {}
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        by_size.setdefault(len(idx), []).append(idx)
    for size in sorted(by_size):
        bucket = by_size[size]
        for dest, source in zip(bucket, rng.permutation(len(bucket))):
            output[dest] = y[bucket[int(source)]]
    return output


def oriented(y, p):
    raw = roc_auc_score(y, p)
    return max(raw, 1.0 - raw)


class FixedPublicReader:
    """Synthetic score reader with the same predict_proba custody interface."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float64)

    def predict_proba(self, x):
        score = np.asarray(x, dtype=np.float64) @ self.weights
        positive = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack((1.0 - positive, positive))


class ReplayV2Test(unittest.TestCase):
    def fixture(self, root: Path):
        att_unit = np.r_[np.arange(4061, dtype=np.int64), [0, 1, 2]]
        inner_unit = np.r_[np.arange(2234, dtype=np.int64) + 10000, [10000]]
        att_id = np.arange(4064, dtype=np.int64) + 1
        inner_id = np.arange(2235, dtype=np.int64) + 100000
        y_att = (att_unit % 2).astype(np.int64)
        y_inner = (inner_unit % 2).astype(np.int64)
        x_att = y_att.astype(np.float64)[:, None]
        x_inner = y_inner.astype(np.float64)[:, None]

        # Recompute the frozen ATT group split without the V2 implementation.
        salt = "pcrl-method-portfolio-v1|attack-tune|20261008|"
        ranked = sorted((hashlib.sha256((salt + str(int(u))).encode()).hexdigest(),
                         int(u)) for u in np.unique(att_unit))
        tune_groups = np.asarray([u for _, u in ranked[:1015]], dtype=np.int64)
        fit = ~np.isin(att_unit, tune_groups)
        shuffled_att = reference_shuffle(y_att[fit], att_unit[fit], ATT_SALT)
        shuffled_inner = reference_shuffle(y_inner, inner_unit, INNER_SALT)

        # The real probe fits LR. This replay test checks saved-model custody
        # and AUC logic with deterministic models, without a solver dependency.
        shuffled_model = FixedPublicReader([1.0])
        planted_model = FixedPublicReader([0.0, 1.0])
        shuffled_pred = shuffled_model.predict_proba(x_inner)[:, 1]
        planted_pred = planted_model.predict_proba(np.column_stack((x_inner, y_inner)))[:, 1]
        expected = {
            "null_auc": 0.5,
            "shuffled_att_true_inner_auc": oriented(y_inner, shuffled_pred),
            "shuffled_att_independent_inner_auc": oriented(shuffled_inner, shuffled_pred),
            "planted_auc": oriented(y_inner, planted_pred),
        }
        gates = {"null_exact": True,
                 "shuffled_independent_inner_at_most_0.60":
                     expected["shuffled_att_independent_inner_auc"] <= 0.60,
                 "planted_at_least_0.99": expected["planted_auc"] >= 0.99}
        expected["gates"] = gates
        expected["status"] = "PASS" if all(gates.values()) else "FAIL"

        custody = root / "custody"
        custody.mkdir()
        files = {}
        for name, model in (("shuffled_model", shuffled_model),
                            ("planted_model", planted_model)):
            path = custody / f"{name}.joblib"
            joblib.dump(model, path, compress=3)
            files[path.name] = digest(path)
        arrays = {
            "att_fit_row_id": att_id[fit], "att_fit_unit": att_unit[fit],
            "att_fit_x": x_att[fit], "att_fit_true_sex": y_att[fit],
            "att_fit_shuffled_sex": shuffled_att,
            "inner_row_id": inner_id, "inner_unit": inner_unit,
            "inner_x": x_inner, "inner_true_sex": y_inner,
            "inner_shuffled_sex": shuffled_inner,
            "shuffled_prediction_inner": shuffled_pred,
            "planted_prediction_inner": planted_pred,
        }
        package = custody / "probe_replay.npz"
        np.savez_compressed(package, **arrays)
        files[package.name] = digest(package)
        receipt = {
            "schema": "pcrl-portfolio-private-control-v2-custody-v1",
            "status": "FROZEN", "files": files,
            "arrays_sha256": {key: array_digest(value) for key, value in arrays.items()},
            "att_view_content_sha256": {"x": array_digest(x_att)},
            "inner_view_content_sha256": {"x": array_digest(x_inner)},
            "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
            "score_role": "INNER_SELECTION",
            "att_shuffle_salt": ATT_SALT,
            "inner_shuffle_salt": INNER_SALT,
        }
        receipt_path = custody / "CUSTODY_RECEIPT.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        report = {"schema": "pcrl-team-b-real-attack-controls-v2",
                  "status": expected["status"], "view": "income",
                  "custody_receipt_sha256": digest(receipt_path),
                  "probe": {"schema": "pcrl-portfolio-attack-control-probes-v2",
                            "att_shuffle_salt": ATT_SALT,
                            "inner_shuffle_salt": INNER_SALT,
                            "att_shuffle_count": 1, "inner_shuffle_count": 1,
                            **expected}}
        att_view = {"row_id": att_id, "unit": att_unit, "x": x_att}
        inner_view = {"row_id": inner_id, "unit": inner_unit, "x": x_inner}
        return custody, report, att_view, inner_view, y_att, y_inner

    def test_counterexample_and_exact_custody_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = self.fixture(Path(temporary))
            self.assertEqual(case[1]["probe"]["shuffled_att_true_inner_auc"], 1.0)
            self.assertLessEqual(case[1]["probe"]["shuffled_att_independent_inner_auc"], .60)
            result = replay_probe(*case)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["model_prediction_parity"], "PASS")

    def test_distinct_salts_and_tampered_prediction_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            custody, report, att, inner, y_att, y_inner = self.fixture(Path(temporary))
            self.assertFalse(np.array_equal(group_shuffle(y_inner, inner["unit"],
                                                          INNER_SALT),
                                            group_shuffle(y_inner, inner["unit"],
                                                          ATT_SALT)))
            report["probe"]["shuffled_att_independent_inner_auc"] += .01
            with self.assertRaises(ValueError):
                replay_probe(custody, report, att, inner, y_att, y_inner)


if __name__ == "__main__":
    unittest.main()
