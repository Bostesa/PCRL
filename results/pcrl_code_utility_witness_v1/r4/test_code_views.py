"""Data-free complete-view invariants for the code-utility follow-up."""

import unittest

import numpy as np

from .code_views import prepare_complete_views


def role(start=0, bits=16, *, as_id=False, extra=None):
    row_id = np.arange(start, start + 4, dtype=np.int64)
    unit = np.array([start, start, start + 2, start + 3], dtype=np.int64)
    code1 = np.zeros((4, bits), dtype=np.uint8)
    code2 = np.zeros((4, bits), dtype=np.uint8)
    code1[:, 0] = [0, 1, 0, 1]
    code2[:, 1] = [0, 0, 1, 1]

    def wire(code, classes, purpose):
        ids = np.sum(code.astype(np.uint64) << np.arange(bits, dtype=np.uint64), axis=1)
        q = np.full((4, classes), 0.1 / (classes - 1), dtype=np.float64)
        q[:, 0] = 0.9
        q[ids != 0] = 1.0 / classes
        release = {
            "code_id" if as_id else "code": ids.astype(np.uint16) if as_id else code,
            "probabilities": q,
            "decision": np.zeros(4, dtype=np.int64),
            "metadata": {"purpose": purpose, "bits": bits, "fallback": np.array([0, 0, 1, 0], dtype=np.uint8)},
        }
        if extra:
            release.update(extra)
        return release

    return {"row_id": row_id, "unit": unit, "first": wire(code1, 2, 0),
            "second": wire(code2, 6, 1)}


class CodeViewsTest(unittest.TestCase):
    def test_10_bit_and_variable_metadata_are_visible(self):
        att, inner = role(bits=10), role(10, bits=10)
        views = prepare_complete_views(att, inner, bits=10)
        self.assertEqual(views["pair"].att.x_code_bits.shape, (4, 20))
        self.assertIn("r1.metadata.fallback", views["pair"].att.wire_components)
        self.assertGreater(np.unique(views["r1"].att.x_metadata, axis=0).shape[0], 1)

    def test_16_bit_complete_pair_and_public_id_equivalence(self):
        att, inner = role(0, as_id=False), role(10, as_id=True)
        views = prepare_complete_views(att, inner, bits=16)
        self.assertEqual(set(views), {"r1", "r2", "pair"})
        self.assertEqual(views["r1"].att.x_code_bits.shape, (4, 16))
        self.assertEqual(views["pair"].att.x_code_bits.shape, (4, 32))
        self.assertIn("pair_bit_products", views["pair"].schema["public_derived"])
        self.assertIn("r1.metadata.fallback", [f["name"] for f in views["pair"].schema["fields"]])
        self.assertEqual(views["pair"].att.cell_key.tolist(), views["pair"].inner.cell_key.tolist())

    def test_rejects_clean_output_or_private_extra(self):
        att, inner = role(), role(10)
        att["first"]["teacher_score"] = np.ones((4, 2))
        with self.assertRaisesRegex(ValueError, "exact public wire"):
            prepare_complete_views(att, inner, bits=16)

    def test_rejects_soft_and_wrong_width(self):
        att, inner = role(), role(10)
        att["first"]["code"] = att["first"]["code"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "hard bits"):
            prepare_complete_views(att, inner, bits=16)
        att, inner = role(bits=10), role(10, bits=10)
        with self.assertRaisesRegex(ValueError, "exact-width"):
            prepare_complete_views(att, inner, bits=16)

    def test_rejects_hidden_head_bypass_and_misalignment(self):
        att, inner = role(), role(10)
        inner["first"]["probabilities"][0] = [0.2, 0.8]
        inner["first"]["decision"][0] = 1
        with self.assertRaisesRegex(ValueError, "code-only"):
            prepare_complete_views(att, inner, bits=16)
        att, inner = role(), role(10)
        inner["second"]["decision"] = np.array([1, 0, 0, 0], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "argmax"):
            prepare_complete_views(att, inner, bits=16)
        att, inner = role(), role(10)
        att["second"]["decision"] = np.zeros(3, dtype=np.int64)
        with self.assertRaises(ValueError):
            prepare_complete_views(att, inner, bits=16)

    def test_rejects_noncanonical_id_and_metadata_gap(self):
        att, inner = role(as_id=True), role(10, as_id=True)
        att["first"]["code_id"] = np.array([0, 1, 2, 65536], dtype=np.uint32)
        with self.assertRaisesRegex(ValueError, "public code ID"):
            prepare_complete_views(att, inner, bits=16)
        att, inner = role(), role(10)
        inner["first"]["metadata"].pop("fallback")
        with self.assertRaisesRegex(ValueError, "metadata"):
            prepare_complete_views(att, inner, bits=16)

    def test_rejects_vector_capacity_metadata(self):
        att, inner = role(), role(10)
        att["first"]["metadata"]["bits"] = np.full(4, 16, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "capacity"):
            prepare_complete_views(att, inner, bits=16)


if __name__ == "__main__":
    unittest.main()
