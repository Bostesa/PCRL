"""Tiny Team D view integration against S2's actual AttackView class.

The S3 queued command must include both this team directory and the admitted
shared evaluator directory on PYTHONPATH; no role export or Adult array is read.
"""

import unittest

import numpy as np

from attack_eval import AttackView
from attack_views import AlignedDViews
from s2_attack_adapter import as_s2_attack_view
from test_s2_attack_adapter import code_release, continuous_release


class ActualS2AttackViewTests(unittest.TestCase):
    def test_pair_code_and_continuous_wire_bindings_validate(self):
        row = np.arange(4, dtype=np.int64)
        for releases in (
            (code_release(0, [1, 12, 0, 1023]), code_release(1, [11, 2, 1023, 0])),
            (continuous_release(0, 4), continuous_release(1, 4)),
        ):
            with self.subTest(code=releases[0]["code_ids"].shape[1] == 1):
                aligned = AlignedDViews(row, row, row, row, *releases, "a" * 64)
                for name in ("recipient0", "recipient1", "pair"):
                    view = as_s2_attack_view(aligned, name, AttackView)
                    self.assertIs(view.validate(), view)
                    self.assertEqual(len(view.components), len(view.component_columns))
                    self.assertEqual(set(view.wire_components),
                                     set(view.components).union(name for name, _ in view.wire_constants))
                    self.assertTrue(np.isfinite(view.x).all())
                    self.assertGreater(view.x_defense_aware.shape[1], view.x.shape[1])
                    self.assertEqual(view.x_metadata.shape[0], len(row))
                    if releases[0]["code_ids"].shape[1] == 1:
                        self.assertIsNotNone(view.cell_key)
                        self.assertIsNotNone(view.x_code_bits)
                    else:
                        self.assertIsNone(view.cell_key)
                        self.assertIsNone(view.x_code_bits)


if __name__ == "__main__":
    unittest.main()
