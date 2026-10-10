"""Data-free worker boundary tests: admitted law and immutable B0 wire."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bank_anchored.r3_worker import (identity_law_from_wire, load_law_npz,
                                     reference_probabilities, run, score_rows,
                                     validate_job_config,
                                     validate_source_replay, verify_bank_wire)


class UtilityWorkerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.row = np.array([10, 11], dtype=np.int64)
        self.unit = np.array([10, 11], dtype=np.int64)
        self.law = {
            "row_id": self.row, "unit": self.unit,
            "code_r1": np.array([[2, 2], [1, 1]], dtype=np.int64),
            "code_r2": np.array([[4, 4], [7, 7]], dtype=np.int64),
            "mass": np.array([[1., 0.], [1., 0.]], dtype=np.float64),
            "valid": np.array([[True, False], [True, False]]),
            "b0_code_r1": np.array([2, 1], dtype=np.int64),
            "b0_code_r2": np.array([4, 7], dtype=np.int64),
        }
        def bits(values):
            return ((values[:, None] >> np.arange(10)) & 1).astype(np.uint8)
        self.wire = {"row_id": self.row, "unit": self.unit,
                     "code0": bits(self.law["b0_code_r1"]),
                     "code1": bits(self.law["b0_code_r2"]),
                     "prob0": np.array([[.8, .2], [.1, .9]]),
                     "prob1": np.tile(np.array([[.8, .04, .04, .04, .04, .04]]), (2, 1)),
                     "decision0": np.array([0, 1], dtype=np.int64),
                     "decision1": np.array([0, 0], dtype=np.int64)}
        self.tables = (np.array([[.5, .5], [.1, .9], [.8, .2]]),
                       np.array([[.8, .04, .04, .04, .04, .04]] * 8))

    def test_role_law_rejects_extra_sensitive_member_before_scoring(self):
        # Break caught: loading a mixed role archive that includes SEX.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "law.npz"
            np.savez(path, **self.law, sex=np.array([0, 1]))
            with self.assertRaises(PermissionError):
                load_law_npz(path, self.row, self.unit)

    def test_identity_ids_must_match_saved_bank_wire(self):
        # Break caught: treating a new route as the frozen B0 reference.
        verify_bank_wire(self.wire, self.law, self.tables, self.row, self.unit)
        switched = {**self.law, "b0_code_r1": np.array([1, 1], dtype=np.int64)}
        with self.assertRaises(PermissionError):
            verify_bank_wire(self.wire, switched, self.tables, self.row, self.unit)

    def test_wire_probabilities_must_be_fixed_table_decodes(self):
        # Break caught: a clean or recalibrated probability bypasses the bank.
        wire = {**self.wire, "prob0": np.array([[.7, .3], [.1, .9]])}
        with self.assertRaises(PermissionError):
            verify_bank_wire(wire, self.law, self.tables, self.row, self.unit)

    def test_fit_and_inner_use_distinct_primary_references(self):
        # Break caught: old FIT U0 constraint accidentally blocks exact B0.
        labels = (np.array([0, 1], dtype=np.int64),
                  np.array([0, 0], dtype=np.int64))
        refs = {"U0_1": np.array([[.95, .05], [.05, .95]]),
                "Ucal_1": np.array([[.95, .05], [.05, .95]]),
                "U0_2": self.wire["prob1"],
                "Ucal_2": self.wire["prob1"]}
        fit, _ = score_rows(self.law, self.tables, labels, refs,
                            role="OSF_DEFENSE_FIT", arm="B0", seed=0)
        inner, _ = score_rows(self.law, self.tables, labels, refs,
                              role="INNER_SELECTION", arm="B0", seed=0)
        self.assertEqual(len(fit), 18)
        self.assertEqual(len(inner), 18)
        b0_fit = [r for r in fit if r["reference"] == "B0"]
        self.assertEqual(len(b0_fit), 6)
        self.assertTrue(all(r["primary_gate"] and r["point_status"] == "PASS"
                            and r["margin"] == 0 for r in b0_fit))
        old_fit = [r for r in fit if r["reference"] == "U0"
                   and r["task"] == "income" and r["metric"] == "nll"]
        self.assertEqual(len(old_fit), 1)
        self.assertFalse(old_fit[0]["primary_gate"])
        self.assertGreater(old_fit[0]["margin"], 0)
        self.assertTrue(all(r["primary_gate"] for r in inner
                            if r["reference"] in ("U0", "Ucal")))

    def test_source_replay_must_show_all_bank_point_clauses(self):
        # Break caught: admitting an incomplete inherited bank as B0.
        good = {"schema": "pcrl-r5-bank-old-point-audit-v1", "status": "PASS",
                "summary": {"W_TSTAR": {"clauses_pass": 24, "clauses_total": 24}}}
        validate_source_replay(good)
        bad = {**good, "summary": {"W_TSTAR": {"clauses_pass": 23,
                                                 "clauses_total": 24}}}
        with self.assertRaises(PermissionError):
            validate_source_replay(bad)

    def test_run_requires_current_campaign_queue_guard_first(self):
        # Break caught: importing the predecessor campaign's queue guard.
        from results.pcrl_bank_anchored_release_v1.r0 import compute_queue

        class CurrentQueueReached(Exception):
            pass

        with patch.object(compute_queue, "assert_active_from_env",
                          side_effect=CurrentQueueReached):
            with self.assertRaises(CurrentQueueReached):
                run(Path("/nonexistent/r3_job.json"))

    def test_witness_mode_builds_exact_identity_without_r2_law(self):
        # Break caught: making the B0 prefit gate depend on a protected fit.
        identity = identity_law_from_wire(self.wire)
        self.assertEqual(identity["mass"].shape, (2, 16))
        np.testing.assert_array_equal(identity["mass"][:, 0], [1., 1.])
        self.assertEqual(np.count_nonzero(identity["mass"][:, 1:]), 0)
        self.assertTrue(identity["valid"][:, 0].all())
        self.assertFalse(identity["valid"][:, 1:].any())
        verify_bank_wire(self.wire, identity, self.tables, self.row, self.unit)

    def test_witness_config_only_allows_b0_and_no_fitted_law(self):
        # Break caught: protected policy scoring before bank identity gate.
        record = {"path": "/tmp/test", "sha256": "a" * 64}
        config = {"schema": "pcrl-bank-r3-utility-job-v1", "unit_id": "R3_B0_FIT_S0",
                  "stage": "witness", "role": "OSF_DEFENSE_FIT", "arm": "B0",
                  "seed": 0, "seal": record, "role_manifest": record,
                  "public_table": record, "bank_receipt": record,
                  "bank_wire": record, "r5_source_replay": record,
                  "law": None, "outputs": {"csv": "/tmp/x", "private_vectors": "/tmp/y",
                                          "receipt": "/tmp/z"}}
        active = {"unit_id": "R3_B0_FIT_S0", "stage": "witness"}
        validate_job_config(config, active)
        with self.assertRaises(PermissionError):
            validate_job_config({**config, "arm": "J"}, active)
        with self.assertRaises(PermissionError):
            validate_job_config({**config, "law": record}, active)

    def test_role_reference_archive_strips_only_nonprobability_members(self):
        # Break caught: passing the role archive's row/decision keys into scorer.
        archive = {"row_id": self.row, "unit": self.unit,
                   "U0_1": self.wire["prob0"], "Ucal_1": self.wire["prob0"],
                   "U0_2": self.wire["prob1"], "Ucal_2": self.wire["prob1"],
                   "d_1": self.wire["decision0"], "d_2": self.wire["decision1"]}
        selected = reference_probabilities(archive)
        self.assertEqual(set(selected), {"U0_1", "Ucal_1", "U0_2", "Ucal_2"})
        with self.assertRaises(PermissionError):
            reference_probabilities({**archive, "sex": np.array([0, 1])})


if __name__ == "__main__":
    unittest.main()
