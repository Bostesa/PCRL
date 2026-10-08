"""Synthetic tests for the independent Team A score-bank parity oracle.

These tests are numerical work. Run them only through S3's admitted queue.
No test opens an Adult artifact.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_bank_parity_oracle_b4 as oracle


def fine():
    return {
        "kind": "qpc.FinePartition", "K": 2, "eps": 1e-12,
        "cell_class": [0, 0, 1],
        "centroid": [[.9, .1], [.7, .3], [.2, .8]],
        "n": [2, 1, 0], "S": [[1.8, .2], [.7, .3], [0., 0.]],
        "A": [0., 0., 0.], "fallback": [False, False, True],
    }


def policy():
    f = fine()
    proto = oracle.token_prototypes(f, [0, 0, 1])[1]
    return {
        "kind": "qpc.Policy", "recipient": 1, "fine": f,
        "cell_token": [0, 0, 1], "token_class": [0, 1],
        "token_proto": proto.tolist(), "token_n": [3, 0],
        "meta": {},
    }


class ScoreBankParityOracleTests(unittest.TestCase):
    def test_fine_assignment_ties_choose_lower_cell_and_zero_components(self):
        f = fine()
        f["centroid"] = [[.8, .2], [.8, .2], [.2, .8]]
        p = np.array([[1., 0.], [.75, .25], [.2, .8]], dtype=np.float64)
        got = oracle.assign_fine_cells(p, np.array([0, 0, 1]), f)
        np.testing.assert_array_equal(got, [0, 0, 2])

    def test_reserved_token_and_canonical_ids_are_preserved(self):
        pol = policy()
        bank, classes, ids = oracle.rebuild_policy_bank(pol)
        np.testing.assert_array_equal(classes, [0, 1])
        np.testing.assert_array_equal(ids, [0, 1])
        self.assertEqual(bank.shape, (2, 2))
        self.assertEqual(int(np.argmax(bank[1])), 1)
        bad = dict(pol, cell_token=[1, 1, 0])
        with self.assertRaises(oracle.ParityError):
            oracle.rebuild_policy_bank(bad)

    def test_temperature_one_is_bitwise_identity_and_other_alpha_is_independent(self):
        q0 = np.array([[.8, .2], [.2, .8]], dtype=np.float64)
        self.assertEqual(oracle.temperature_bank(q0, [0, 1], 1.0).tobytes(), q0.tobytes())
        got = oracle.temperature_bank(q0, [0, 1], [2.0, 1.0])
        self.assertGreater(got[0, 0], q0[0, 0])
        self.assertEqual(got[1].tobytes(), q0[1].tobytes())

    def test_duplicate_vectors_do_not_merge_distinct_token_ids(self):
        q = np.array([[.8, .2], [.8, .2], [.2, .8]], dtype=np.float64)
        token = np.array([0, 1], dtype=np.int64)
        self.assertEqual(oracle.compare_arrays(token, np.array([0, 1]))["mismatch_count"], 0)
        self.assertEqual(oracle.compare_arrays(token, np.array([0, 0]))["mismatch_count"], 1)
        self.assertEqual(q[0].tobytes(), q[1].tobytes())

    def test_cell_and_token_mismatch_cannot_be_hidden_by_equal_probabilities(self):
        independent = {"cell": np.array([0, 1]), "token": np.array([0, 1]),
                       "bank": np.array([[.8, .2], [.8, .2]]), "decision": np.array([0, 0])}
        produced = {"cell": np.array([0, 1]), "token": np.array([0, 0]),
                    "bank": independent["bank"], "decision": np.array([0, 0])}
        report = oracle.compare_release(independent, produced)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["token"]["mismatch_count"], 1)
        self.assertEqual(report["probability"]["mismatch_count"], 0)

    def test_nonbitwise_float_is_unresolved_not_pass(self):
        x = np.array([[.8, .2]], dtype=np.float64)
        y = x.copy()
        y[0, 0] = np.nextafter(y[0, 0], np.inf)
        report = oracle.compare_arrays(x, y)
        self.assertEqual(report["status"], "UNRESOLVED_NUMERICAL_PARITY")
        self.assertEqual(report["mismatch_count"], 1)
        self.assertGreaterEqual(report["max_ulp_difference"], 1)

    def test_rebuilt_pinned_table_float_difference_is_unresolved(self):
        pol = policy()
        pol["token_proto"][0][0] = float(np.nextafter(pol["token_proto"][0][0], np.inf))
        with self.assertRaises(oracle.NumericalParityError) as caught:
            oracle.rebuild_policy_bank(pol)
        self.assertEqual(caught.exception.stage, "policy token prototype")
        self.assertGreaterEqual(caught.exception.max_ulp_difference, 1)

    def test_digest_declares_dtype_shape_and_bytes(self):
        a = np.array([1, 2], dtype=np.int64)
        b = a.astype(np.int32)
        self.assertNotEqual(oracle.array_digest(a), oracle.array_digest(b))
        self.assertEqual(oracle.array_digest(a), oracle.array_digest(np.ascontiguousarray(a)))

    def test_source_binding_refuses_wrong_output_hash_or_role_members(self):
        with self.assertRaises(oracle.ParityError):
            oracle.require_members({"row_id", "unit", "sex"}, {"row_id", "unit"})
        with self.assertRaises(oracle.ParityError):
            oracle.require_sha("0" * 64, "1" * 64, "fixture")

    def test_partial_or_selected_output_set_never_passes(self):
        report = oracle.assemble_public_report([], "a" * 64, "b" * 64, "c" * 64)
        self.assertEqual(report["overall_status"], "INCOMPLETE")
        self.assertEqual(len(report["missing_units"]), 10)
        with self.assertRaises(oracle.ParityError):
            oracle.assemble_public_report([
                {"seed": 0, "release_key": "tstar", "status": "PASS"},
                {"seed": 0, "release_key": "tstar", "status": "PASS"},
            ], "a" * 64, "b" * 64, "c" * 64)

    def test_standard_paths_construct_only_ten_registered_units(self):
        specs = {f"{role}:references_s{seed}":
                 {"path": f"shared/.private/{role}/references_s{seed}.npz"}
                 for role in oracle.ROLES for seed in (0, 1)}
        public, output = oracle.standard_paths(Path("/tmp/results/pcrl_method_portfolio_v1"),
                                               Path("/tmp/cache"), Path("/tmp/queue"),
                                               {"real_file_specs": specs})
        self.assertEqual(set(output), set(oracle.OUTPUT_PINS))
        self.assertEqual(len(output), 10)
        self.assertEqual(public["ordered_schema"],
                         Path("/tmp/cache/hcal_v1/admitted/inputs/schema.json"))
        self.assertIsNone(output[(0, "lra_seq21")]["decoder_json"])


if __name__ == "__main__":
    unittest.main()
