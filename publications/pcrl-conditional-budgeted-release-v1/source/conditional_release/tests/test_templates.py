"""Tiny fixed template law tests; coordinator seals before running."""
import unittest
import numpy as np
from incumbent_view.r2_space import Space
from conditional_release.templates import build_template_law, template_probabilities


def tiny_space():
    p = np.array([[.8, .2], [.6, .4], [.1, .9]], np.float64)
    action = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [2, 2, 2, 2]], np.int64)
    valid = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]], bool)
    return Space(p, p, action, action, valid, valid)


class TemplateFixtures(unittest.TestCase):
    def test_narrow_and_wide_identity_class_and_alias(self):
        space = tiny_space()
        codes = np.array([0, 1], np.int64)
        contexts = np.zeros(2, np.int64)
        for support, parameters in (("NARROW", 16), ("WIDE", 9)):
            law, metadata = build_template_law(space, codes, codes, contexts, support, 1_000_000)
            self.assertEqual(law.parameters, parameters)
            theta = np.zeros(parameters); theta[0] = 1.
            q = law.probabilities(theta)
            a, b, deployed = template_probabilities(metadata, theta, space, codes, codes, contexts)
            np.testing.assert_array_equal(q, deployed)
            np.testing.assert_array_equal(a, law.emitted1)
            np.testing.assert_array_equal(b, law.emitted2)
            self.assertTrue(np.all(space.probability1[a].argmax(2) == 0))
        theta = np.zeros(9); theta[0] = .5; theta[4] = .5
        a, b, q = template_probabilities(metadata, theta, space, codes, codes, contexts)
        # Explicit global0 and KEEP alias for input0, but not input1.
        self.assertEqual(q[0, (a[0] == 0) & (b[0] == 0)].sum(), 1.)
        self.assertEqual(q[1, (a[1] == 0) & (b[1] == 0)].sum(), .5)

    def test_memory_preflight(self):
        space = tiny_space(); codes = np.array([0, 1], np.int64)
        with self.assertRaises(MemoryError):
            build_template_law(space, codes, codes, np.zeros(2, np.int64), "WIDE", 1)

    def test_unsupported_exact_identity(self):
        space = tiny_space(); codes = np.array([0, 1], np.int64)
        law, metadata = build_template_law(space, codes, codes, np.array([-1, 0], np.int64), "WIDE", 1_000_000)
        theta = np.zeros(9); theta[4] = 1.
        q = law.probabilities(theta)
        a, b, deployed = template_probabilities(metadata, theta, space, codes, codes, np.array([-1, 0], np.int64))
        np.testing.assert_array_equal(q, deployed)
        self.assertEqual(q[0, 0], 1.)
        self.assertEqual(a[0, 0], codes[0]); self.assertEqual(b[0, 0], codes[0])

    def test_shared_class_mismatch_rejected(self):
        space = tiny_space(); codes = np.array([0, 2], np.int64)
        with self.assertRaises(ValueError):
            build_template_law(space, codes, codes, np.zeros(2, np.int64), "WIDE", 1_000_000)


if __name__ == "__main__":
    unittest.main()
