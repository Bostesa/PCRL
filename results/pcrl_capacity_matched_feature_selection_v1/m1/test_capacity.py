"""Synthetic and public-schema checks; no Adult person values."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from capacity import (assert_role_alignment, capacity_geometry, digest, reconstruct_union, source_groups,
                      validate_release_input)


def portfolio_feature_names() -> tuple[str, ...]:
    root = Path(__file__).resolve().parents[3]
    source = root / "results/pcrl_method_portfolio_v1/teams/c/prototype.py"
    spec = importlib.util.spec_from_file_location("focused_prior_c_prototype", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FEATURE_NAMES


class CapacityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.names = portfolio_feature_names()
        self.geometry = capacity_geometry(self.names)

    def test_schema_and_exact_feasibility(self) -> None:
        self.assertEqual(len(self.names), 83)
        groups = source_groups(self.names)
        self.assertEqual({name: group["encoded_columns"] for name, group in groups.items()}, {
            "age": 1, "education_fact": 17, "capital-gain": 1,
            "capital-loss": 1, "hours-per-week": 1, "workclass": 8,
            "marital-status": 7, "relationship": 6, "native-country": 41,
        })
        self.assertEqual(self.geometry["mask_count"], 5)
        self.assertEqual(self.geometry["pair_count"], 25)
        self.assertEqual(sorted(m["encoded_columns"] for m in self.geometry["masks"]),
                         [42, 66, 75, 76, 77])
        self.assertEqual(set(self.geometry["exact_width_multiplicity"].values()), {1})
        self.assertEqual(sum(p["union_encoded_columns"] == 83 for p in self.geometry["pairs"]), 20)
        self.assertEqual(sum(p["union_encoded_columns"] < 83 for p in self.geometry["pairs"]), 5)

    def test_partial_group_and_reordered_schema_refused(self) -> None:
        with self.assertRaises(ValueError):
            source_groups(self.names[:-1])
        changed = list(self.names)
        changed[7], changed[8] = changed[8], changed[7]
        with self.assertRaises(ValueError):
            source_groups(changed)

    def test_release_and_canonical_union(self) -> None:
        x = np.arange(2 * 83, dtype=np.float32).reshape(2, 83)
        a, b = self.geometry["masks"][:2]
        xa, xb = x[:, a["column_indices"]], x[:, b["column_indices"]]
        validate_release_input(x, self.names, a, xa)
        columns, union = reconstruct_union(a, xa, b, xb, x)
        self.assertEqual(columns, tuple(range(83)))
        np.testing.assert_array_equal(union, x)
        bad = xa.copy()
        bad[0, 0] += 1
        with self.assertRaises(ValueError):
            validate_release_input(x, self.names, a, bad)
        with self.assertRaises(ValueError):
            reconstruct_union(a, bad, b, xb, x)
        signed_zero = xa.copy()
        self.assertEqual(float(signed_zero[0, 0]), 0.0)
        signed_zero[0, 0] = np.float32(-0.0)
        with self.assertRaises(ValueError):
            validate_release_input(x, self.names, a, signed_zero)
        other_signed_zero = xb.copy()
        other_signed_zero[0, 0] = np.float32(-0.0)
        with self.assertRaises(ValueError):
            reconstruct_union(a, xa, b, other_signed_zero, x)
        incomplete = dict(a)
        incomplete["column_indices"] = a["column_indices"][1:]
        incomplete["mask_sha256"] = digest({
            "source_schema_sha256": incomplete["source_schema_sha256"],
            "atoms": incomplete["atoms"],
            "column_indices": incomplete["column_indices"],
        })
        with self.assertRaises(ValueError):
            validate_release_input(x, self.names, incomplete, x[:, incomplete["column_indices"]])

    def test_ordered_role_identity(self) -> None:
        rows = np.array([1, 2], dtype=np.int64)
        groups = np.array([9, 9], dtype=np.int64)
        assert_role_alignment(rows, groups, rows.copy(), groups.copy())
        with self.assertRaises(ValueError):
            assert_role_alignment(rows, groups, rows[::-1], groups)
        with self.assertRaises(ValueError):
            assert_role_alignment(rows, groups, rows, np.array([9, 10], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
