"""New data-free runtime fixtures; run only through the campaign queue."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from incumbent_view.r2_space import Space
from conditional_release.templates import build_template_law
from conditional_release.control_wide_policy import (
    ControlWidePolicy, load_wide_control_policy, save_wide_control_policy,
)


class FrozenContext:
    labels = ("classpair/0",)
    mapping = np.zeros(96, np.int64)
    raw_fit_counts = np.zeros(96, np.int64)
    merged_fit_counts = np.array([64], np.int64)

    def apply(self, p1, p2):
        return np.where((p1.argmax(1) == 0) & (p2.argmax(1) == 0), 0, -1).astype(np.int64)


def fixture():
    p = np.array([[.8, .2], [.6, .4], [.1, .9]], np.float64)
    actions = np.repeat(np.arange(3, dtype=np.int64)[:, None], 4, axis=1)
    valid = np.zeros((3, 4), bool); valid[:, 0] = True
    space = Space(p, p, actions, actions, valid, valid)
    _, metadata = build_template_law(space, np.array([0, 1], np.int64),
                                    np.array([0, 1], np.int64), np.zeros(2, np.int64),
                                    "WIDE", 1_000_000)
    return space, FrozenContext(), metadata


def emitted_dict(a, b, q):
    result = {}
    for x, y, mass in zip(a, b, q):
        result[(int(x), int(y))] = result.get((int(x), int(y)), 0.) + mass
    return {key: value for key, value in result.items() if value}


class WideRuntimeFixtures(unittest.TestCase):
    def test_product_emits_named_global_ids_and_full_axis(self):
        space, ctx, meta = fixture()
        policy = ControlWidePolicy(space, "PRODUCT", ctx, np.array([[.2, .3, .5]]),
                                   np.array([[.4, .1, .5]]), metadata=meta)
        a, b, q = policy.law_codes(np.array([1], np.int64), np.array([0], np.int64))
        self.assertEqual(a.shape, (1, 9))
        expected = {(0, 0): .3*.5, (0, 1): .3*.5, (1, 0): .7*.5, (1, 1): .7*.5}
        self.assertEqual(set(emitted_dict(a[0], b[0], q[0])), set(expected))
        for key, value in expected.items():
            self.assertAlmostEqual(emitted_dict(a[0], b[0], q[0])[key], value)

    def test_local_never_reads_central_context(self):
        space, _, _ = fixture()
        class Forbidden:
            def apply(self, *args): raise AssertionError("local consulted other inputs")
        left = np.array([[0., 0., 1.], [1., 0., 0.], [1., 0., 0.]])
        right = np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
        policy = ControlWidePolicy(space, "LOCAL", Forbidden(), left, right)
        a, b, q = policy.law_codes(np.array([0, 0], np.int64), np.array([0, 2], np.int64))
        self.assertEqual([emitted_dict(a[i], b[i], q[i]) for i in range(2)], [{(1, 0): 1.}, {(1, 2): 1.}])

    def test_unseen_local_code_has_supplied_keep_table(self):
        space, _, _ = fixture()
        table = np.zeros((3, 3)); table[:, 0] = 1.
        policy = ControlWidePolicy(space, "LOCAL", None, table, table)
        a, b, q = policy.law_codes(np.array([2], np.int64), np.array([1], np.int64))
        self.assertEqual(emitted_dict(a[0], b[0], q[0]), {(2, 1): 1.})

    def test_unsupported_central_context_is_exact_keep(self):
        space, ctx, meta = fixture()
        policy = ControlWidePolicy(space, "PRODUCT", ctx, np.array([[0., 1., 0.]]),
                                   np.array([[0., 1., 0.]]), metadata=meta)
        a, b, q = policy.law_codes(np.array([2], np.int64), np.array([0], np.int64))
        self.assertEqual(emitted_dict(a[0], b[0], q[0]), {(2, 0): 1.})

    def test_deterministic_binary_only_and_global_aliases(self):
        space, ctx, meta = fixture()
        theta = np.zeros(9); theta[8] = 1.
        policy = ControlWidePolicy(space, "DETERMINISTIC", ctx, theta, metadata=meta)
        a, b, q = policy.law_codes(np.array([0], np.int64), np.array([0], np.int64))
        self.assertEqual(emitted_dict(a[0], b[0], q[0]), {(1, 1): 1.})
        with self.assertRaises(ValueError):
            ControlWidePolicy(space, "DETERMINISTIC", ctx, np.full(9, 1/9), metadata=meta)

    def test_sequential_conditions_on_actual_code_not_action_position(self):
        space, ctx, meta = fixture()
        policy = ControlWidePolicy(space, "SEQUENTIAL", ctx, np.array([[.4, .6, 0.]]),
            np.array([[1., 0., 0.], [0., 0., 1.]]), np.array([[0, -1], [0, 0]], np.int64),
            1, metadata=meta)
        a, b, q = policy.law_codes(np.array([0], np.int64), np.array([0], np.int64))
        self.assertEqual(emitted_dict(a[0], b[0], q[0]), {(0, 1): 1.})

    def test_sequential_missing_child_uses_context_parent(self):
        space, ctx, meta = fixture()
        policy = ControlWidePolicy(space, "SEQUENTIAL", ctx, np.array([[1., 0., 0.]]),
            np.array([[0., 1., 0.]]), np.array([[0, -1]], np.int64), 1, metadata=meta)
        a, b, q = policy.law_codes(np.array([1], np.int64), np.array([1], np.int64))
        self.assertEqual(emitted_dict(a[0], b[0], q[0]), {(1, 0): 1.})

    def test_sequential_recipient_two_order_is_correct(self):
        space, ctx, meta = fixture()
        policy = ControlWidePolicy(space, "SEQUENTIAL", ctx, np.array([[0., 0., 1.]]),
            np.array([[0., 1., 0.]]), np.array([[0, -1]], np.int64), 2, metadata=meta)
        a, b, q = policy.law_codes(np.array([1], np.int64), np.array([0], np.int64))
        self.assertEqual(emitted_dict(a[0], b[0], q[0]), {(0, 1): 1.})

    def test_padding_and_changed_bank_axis_are_rejected(self):
        space, ctx, meta = fixture()
        table = np.zeros((3, 3)); table[:, 0] = 1.; table[2] = [.5, 0., .5]
        with self.assertRaises(ValueError): ControlWidePolicy(space, "LOCAL", None, table, table)
        changed = dict(meta); changed["contexts"] = [dict(meta["contexts"][0])]
        changed["contexts"][0]["recipient1_templates"] = [-1, 1, 0]
        with self.assertRaises(ValueError):
            ControlWidePolicy(space, "PRODUCT", ctx, np.array([[1., 0., 0.]]),
                              np.array([[1., 0., 0.]]), metadata=changed)

    def test_private_serialization_binds_bank_and_exact_fields(self):
        space, ctx, meta = fixture()
        policy = ControlWidePolicy(space, "PRODUCT", ctx, np.array([[.5, .25, .25]]),
                                   np.array([[.5, .25, .25]]), metadata=meta)
        manifest = {"seeds": {"0": {"public_bank": {"sha256": "fixture-bank"}}}}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/".private"/"policy"
            record, receipt = save_wide_control_policy(path, policy, seed=0, arm_id="toy", budget="fixture", source_manifest=manifest)
            self.assertEqual(record["schema"], "CBR1-wide-control-policy-v1")
            with patch("conditional_release.contexts.ContextMap", return_value=ctx), patch(
                    "conditional_release.source.load_historical_policy", return_value=SimpleNamespace(space=space)):
                restored = load_wide_control_policy(path, manifest, expected_receipt_sha256=receipt)
                for actual, expected in zip(restored.law_codes(np.array([0], np.int64), np.array([1], np.int64)),
                                            policy.law_codes(np.array([0], np.int64), np.array([1], np.int64))):
                    np.testing.assert_array_equal(actual, expected)
                bad = {"seeds": {"0": {"public_bank": {"sha256": "other-bank"}}}}
                with self.assertRaises(PermissionError): load_wide_control_policy(path, bad)


if __name__ == "__main__": unittest.main()
