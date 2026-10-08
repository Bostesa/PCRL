"""Data-free checks for M5's independent grouped-capacity arithmetic."""

import unittest

from capacity_geometry import enumerate_masks, pair_union, validate_partition


ATOMS = {
    "age": (0,),
    "education_fact": (1, *range(13, 29)),
    "capital-gain": (2,),
    "capital-loss": (3,),
    "hours-per-week": (4,),
    "workclass": tuple(range(5, 13)),
    "marital-status": tuple(range(29, 36)),
    "relationship": tuple(range(36, 42)),
    "native-country": tuple(range(42, 83)),
}
NUMERIC = ("age", "capital-gain", "capital-loss", "hours-per-week")
CATEGORICAL = ("education_fact", "workclass", "marital-status",
               "relationship", "native-country")


class CapacityGeometryTests(unittest.TestCase):
    def test_primary_cap_yields_five_masks_and_distinct_pair_union(self):
        validate_partition(ATOMS, 83)
        masks = enumerate_masks(ATOMS, NUMERIC, CATEGORICAL, 4, 4, 77)
        self.assertEqual(len(masks), 5)
        self.assertEqual(sorted(m.encoded_columns for m in masks), [42, 66, 75, 76, 77])
        self.assertEqual(len({m.columns for m in masks}), 5)
        self.assertEqual(len([pair_union(a, b) for a in masks for b in masks]), 25)
        for a in masks:
            for b in masks:
                union = pair_union(a, b)
                self.assertEqual(len(union), a.encoded_columns if a == b else 83)

    def test_exact_width_is_singleton_and_invalid_group_partition_refused(self):
        masks = enumerate_masks(ATOMS, NUMERIC, CATEGORICAL, 4, 4, 77)
        self.assertEqual(sum(m.encoded_columns == 77 for m in masks), 1)
        broken = dict(ATOMS, relationship=(35, *range(37, 42)))
        with self.assertRaises(ValueError):
            validate_partition(broken, 83)


if __name__ == "__main__":
    unittest.main()
