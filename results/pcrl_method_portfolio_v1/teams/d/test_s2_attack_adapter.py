"""Synthetic D-to-S2 view binding checks; numerical run requires S3 admission."""

import unittest

import numpy as np

from attack_views import AlignedDViews, view_matrix
from s2_attack_adapter import DA_TRANSFORM_SHA256, as_s2_attack_view


class StubAttackView:
    """Check S2's component contract without importing its moving draft."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def validate(self):
        assert len(set(self.components)) == len(self.components)
        constants = [name for name, _ in self.wire_constants]
        assert not set(self.components).intersection(constants)
        assert set(self.wire_components) == set(self.components).union(constants)
        covered = np.zeros(self.x.shape[1], dtype=np.int8)
        for name, start, end in self.component_columns:
            assert name in self.components and 0 <= start < end <= self.x.shape[1]
            covered[start:end] += 1
        assert np.all(covered == 1)
        assert self.x.shape[0] == len(self.row_id) == len(self.unit)
        return self


def code_release(purpose, ids):
    ids = np.asarray(ids, dtype=np.int64).reshape(-1, 1)
    p = np.column_stack((np.where(ids[:, 0] % 2 == 0, .8, .2),
                         np.where(ids[:, 0] % 2 == 0, .2, .8)))
    return {
        "code_ids": ids, "decoded": ids.astype(np.float32),
        "probabilities": p, "decision": p.argmax(1),
        "metadata": {"purpose": purpose, "code_length_bits": 10,
                     "block_count": 1, "fallback": False, "stochastic": False,
                     "kind": "LOCAL", "calibration": "H-GLOBAL-TEMP",
                     "calibrator_sha256": str(purpose) * 64},
    }


def continuous_release(purpose, n):
    p = np.tile([.7, .3], (n, 1)) if purpose == 0 else np.tile([.4, .6], (n, 1))
    return {
        "code_ids": np.empty((n, 0), dtype=np.int64),
        "decoded": np.arange(n, dtype=np.float32)[:, None],
        "probabilities": p, "decision": p.argmax(1),
        "metadata": {"purpose": purpose, "code_length_bits": 0,
                     "block_count": 0, "fallback": False, "stochastic": False,
                     "kind": "LEACE", "calibration": "H-GLOBAL-TEMP",
                     "calibrator_sha256": str(purpose) * 64},
    }


class S2AttackAdapterTests(unittest.TestCase):
    def test_pair_has_exact_wire_bindings_bits_metadata_and_pair_cells(self):
        row = np.arange(4, dtype=np.int64)
        releases = (code_release(0, [1, 12, 0, 1023]),
                    code_release(1, [11, 2, 1023, 0]))
        views = AlignedDViews(row, row, row, row, *releases, "a" * 64)
        attack = as_s2_attack_view(views, "pair", StubAttackView)
        self.assertEqual(len(attack.components), 8)
        self.assertEqual(attack.components[0], "recipient0.code_ids")
        self.assertEqual(attack.components[-1], "recipient1.decision")
        self.assertIn(("recipient0.metadata.calibrator_sha256", '"' + "0" * 64 + '"'),
                      attack.wire_constants)
        parity = []
        for name, start, end in attack.component_columns:
            block = attack.x[:, start:end]
            parity.append(block[:, :1024] if name.endswith(".code_ids") else block)
            if name.endswith(".decision"):
                purpose = 0 if name.startswith("recipient0") else 1
                parity.append(attack.x_metadata[:, purpose * 101:(purpose + 1) * 101])
        np.testing.assert_array_equal(np.concatenate(parity, axis=1), view_matrix(views, "pair"))
        self.assertEqual(attack.x.shape[1], view_matrix(views, "pair").shape[1] + 20 - 202)
        self.assertEqual(attack.x_metadata.shape, (4, 2 * 101))
        self.assertEqual(attack.x_code_bits.shape, (4, 20))
        self.assertEqual(attack.cell_key.tolist(), ["1|11", "12|2", "0|1023", "1023|0"])
        self.assertEqual(len(DA_TRANSFORM_SHA256), 64)
        self.assertEqual(attack.x_defense_aware.shape[1], attack.x.shape[1] + 106)
        extra = attack.x_defense_aware[:, attack.x.shape[1]:]
        np.testing.assert_allclose(extra[0, :2], np.log(releases[0]["probabilities"][0]))
        self.assertEqual(extra[0, 2], releases[0]["decoded"][0, 0] ** 2)
        self.assertEqual(extra[0, -100:].reshape(10, 10)[0, 0], 1.0)
        other_row = row + 100
        other = AlignedDViews(other_row, other_row, other_row, other_row,
                              *releases, "b" * 64)
        np.testing.assert_array_equal(
            attack.x_defense_aware,
            as_s2_attack_view(other, "pair", StubAttackView).x_defense_aware)

    def test_continuous_empty_code_field_remains_declared(self):
        row = np.arange(3, dtype=np.int64)
        releases = (continuous_release(0, 3), continuous_release(1, 3))
        views = AlignedDViews(row, row, row, row, *releases, "b" * 64)
        attack = as_s2_attack_view(views, "recipient0", StubAttackView)
        self.assertEqual(attack.components[0], "recipient0.decoded")
        self.assertIn(("recipient0.code_ids", "[]"), attack.wire_constants)
        self.assertEqual(attack.component_columns[0], ("recipient0.decoded", 0, 1))
        self.assertIsNone(attack.cell_key)
        self.assertIsNone(attack.x_code_bits)
        np.testing.assert_array_equal(attack.x[:, 0], releases[0]["decoded"][:, 0])
        self.assertEqual(attack.x_metadata.shape, (3, 101))
        self.assertEqual(attack.x_defense_aware.shape[1], attack.x.shape[1] + 3)

    def test_float64_calibrated_confidence_distinction_is_not_rounded(self):
        row = np.arange(2, dtype=np.int64)
        r0 = code_release(0, [0, 0])
        r1 = code_release(1, [1, 1])
        r0["probabilities"] = np.asarray([[0.5000000001, 0.4999999999],
                                          [0.5000000002, 0.4999999998]], dtype=np.float64)
        r0["decision"] = np.zeros(2, dtype=np.int64)
        views = AlignedDViews(row, row, row, row, r0, r1, "a" * 64)
        attack = as_s2_attack_view(views, "recipient0", StubAttackView)
        name, start, end = next(part for part in attack.component_columns
                                if part[0] == "recipient0.probabilities")
        self.assertEqual(name, "recipient0.probabilities")
        np.testing.assert_array_equal(attack.x[:, start:end], r0["probabilities"])
        self.assertNotEqual(float(attack.x[0, start]), float(attack.x[1, start]))


if __name__ == "__main__":
    unittest.main()
