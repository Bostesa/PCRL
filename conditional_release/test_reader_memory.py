import json
from pathlib import Path
import unittest

from .attacks import hgb_dense_projection


class TestDenseMemoryBoundary(unittest.TestCase):
    def test_exact_fixed_boundary_without_allocation(self):
        p=Path(__file__).resolve().parents[1]/"results/pcrl_conditional_budgeted_release_v1/r5/HGB_MEMORY_BOUNDARY_FIXTURES.json"
        fixture=json.loads(p.read_text())
        for item in fixture["cases"]:
            with self.subTest(item=item):
                value=hgb_dense_projection(item["rows"],item["columns"])
                self.assertEqual(value["dense_float64_bytes"],item["expected_bytes"])
                self.assertEqual(value["within_limit"],item["within_limit"])
                self.assertEqual(value["limit_bytes"],fixture["limit_bytes"])
        for rows,columns in fixture["invalid_dimensions"]:
            with self.subTest(rows=rows,columns=columns):
                with self.assertRaises(ValueError):hgb_dense_projection(rows,columns)


if __name__=="__main__":unittest.main()
