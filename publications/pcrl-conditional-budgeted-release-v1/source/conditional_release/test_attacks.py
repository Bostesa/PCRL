"""Data-free attack codec/actual-law checks; no reader fitting in this module."""
import json
from pathlib import Path
import unittest

import numpy as np

from .attacks import (wire_feature_tables, law_arrays, aggregated_training, sample_training_actions,
                      group_shuffle, SHUFFLE_ATT_SALT)

FIXTURE = Path(__file__).resolve().parents[1] / "results/pcrl_conditional_budgeted_release_v1/r5/WIRE_READER_FIXTURES.json"


class WireReaderTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text())
        f = self.fixture
        self.bank = {"release_id":"fixture-public-bank", "source_binding_sha256":"0" * 64,
            "recipients":{str(r):{"probability":f[f"bank_r{r}"], "bank_sha256":str(r) * 64} for r in (1, 2)}}
        self.law = {key:np.asarray(f[key], dtype=np.float64 if key == "mass" else np.int64)
                    for key in ("row_id", "unit", "code_r1", "code_r2", "mass")}
        self.y = np.asarray(f["sex"])
        self.uniform = np.full((4, 6), 1 / 24)

    def test_lossless_code_probability_decision_metadata(self):
        tables, schema = wire_feature_tables(self.bank, self.uniform)
        for r in (1, 2):
            b = np.asarray(self.bank["recipients"][str(r)]["probability"])
            tab = tables[f"r{r}"]
            np.testing.assert_array_equal(tab["x"][:, :len(b)], np.eye(len(b)))
            np.testing.assert_array_equal(tab["probability"], b)
            bits = tab["bits"]
            np.testing.assert_array_equal(bits @ (1 << np.arange(bits.shape[1])), np.arange(len(b)))
            self.assertEqual(set(schema[f"r{r}"]["public_metadata_fields"]), {"model_id", "source_binding_sha256", "bank_sha256", "code_length_bits"})
        self.assertEqual(len(tables["pair"]["x"]), 24)
        self.assertGreater(tables["pair"]["defense"].shape[1], tables["pair"]["x"].shape[1])

    def test_pair_aliases_and_exact_training_mass(self):
        _, _, indexes, mass = law_arrays(self.law, self.bank)
        np.testing.assert_array_equal(indexes["pair"] // 6, indexes["r1"])
        np.testing.assert_array_equal(indexes["pair"] % 6, indexes["r2"])
        for view, size in (("r1", 4), ("r2", 6), ("pair", 24)):
            _, _, weight, counts = aggregated_training(indexes[view], mass, self.y, np.arange(6), size)
            self.assertAlmostEqual(weight.sum(), 6)
            np.testing.assert_allclose(counts.sum(0), np.bincount(self.y))

    def test_duplicate_cache_sampling(self):
        slots = sample_training_actions(self.law, seed=0)
        self.assertEqual(slots.shape, (6, 4))
        np.testing.assert_array_equal(slots[0], slots[1])
        np.testing.assert_array_equal(slots, sample_training_actions(self.law, seed=0))
        changed = {k:v.copy() for k, v in self.law.items()}
        changed["code_r1"][1] = changed["code_r1"][1, ::-1]
        with self.assertRaises(ValueError): sample_training_actions(changed, seed=0)

    def test_public_law_and_malformed_ids_reject(self):
        with self.assertRaises(ValueError): wire_feature_tables(self.bank, self.uniform * 0.9)
        changed = {k:v.copy() for k, v in self.law.items()}
        changed["code_r1"][0, 1] = 4
        with self.assertRaises(ValueError): law_arrays(changed, self.bank)

    def test_group_shuffle_and_mean_output_trap(self):
        out = group_shuffle(self.y, self.law["unit"], SHUFFLE_ATT_SALT)
        np.testing.assert_array_equal(np.sort(out), np.sort(self.y))
        self.assertEqual(out[0], out[1])
        table = np.asarray(self.bank["recipients"]["1"]["probability"])
        exact = -np.sum(self.law["mass"][0] * np.log(table[self.law["code_r1"][0], 0]))
        wrong = -np.log(np.sum(self.law["mass"][0] * table[self.law["code_r1"][0], 0]))
        self.assertGreater(exact, wrong)


if __name__ == "__main__":
    unittest.main()
