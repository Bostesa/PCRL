"""Data-free regression for the mandatory P* defense-aware reader view."""

import unittest

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.reference_wire import (
    ReferenceRoleRelease, reference_recipient_batch,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.reference_materialize import (
    reconstruct_reference_wires,
)
from .pstar_views_h2 import _local, _pair, prepare_pstar_three_views_h2


def _package(role):
    bank1 = np.array([[.9, .1], [.2, .8]], np.float64)
    bank2 = np.array([[.9, .02, .02, .02, .02, .02],
                      [.02, .9, .02, .02, .02, .02]], np.float64)
    code = np.array([0, 1], np.int64)
    receipt = {
        "role": role, "arm_id": "P_STAR", "source_seed": 0,
        "release_id": "P_STAR-s0", "source_binding_sha256": "a" * 64,
        "interface_kind": "finite",
        "bank_sha256_by_recipient": {"r1": sha_array(bank1),
                                     "r2": sha_array(bank2)}}
    wires = []
    for recipient, bank in ((1, bank1), (2, bank2)):
        wires.append(reconstruct_reference_wires(
            bank[code], code, code, recipient=recipient,
            interface_kind="finite", release_id="P_STAR-s0",
            source_binding_sha256="a" * 64,
            bank_sha256=sha_array(bank), bank_size=len(bank)))
    return ReferenceRoleRelease(receipt, "b" * 64,
                                np.array([11, 12], np.int64),
                                np.array([21, 22], np.int64), *wires)


class PStarH2ViewsTests(unittest.TestCase):
    def test_complete_wire_and_public_derived_defense_view(self):
        att = _package("ATTACK_FIT_NEW")
        inner = _package("INNER_SELECTION")
        left = _local(reference_recipient_batch(att, 1))
        right = _local(reference_recipient_batch(att, 2))
        pair = _pair(left, right)
        self.assertEqual(tuple(pair.public_derived),
                         ("r1.public_log_probability",
                          "r2.public_log_probability", "pair_code_bit_products"))
        np.testing.assert_array_equal(
            left.public_derived["public_log_probability"],
            np.log(np.array([[.9, .1], [.2, .8]], np.float64)))
        np.testing.assert_array_equal(
            pair.public_derived["pair_code_bit_products"],
            np.array([[0.], [1.]]))
        prepared = prepare_pstar_three_views_h2(att, inner)
        for view in ("r1", "r2", "pair"):
            p = prepared[view]
            self.assertIsNotNone(p.att.x_defense_aware)
            self.assertIsNotNone(p.inner.x_defense_aware)
            self.assertEqual(p.att.x_defense_aware.shape[0], 2)
            self.assertEqual(p.inner.x_defense_aware.shape[0], 2)
            self.assertIsNotNone(p.att.cell_key)
            self.assertEqual(p.att.x_defense_aware.shape[1] - p.att.x.shape[1],
                             2 if view == "r1" else 6 if view == "r2" else 9)

    def test_rejects_non_pstar_and_nonfinite_public_probability(self):
        att = _package("ATTACK_FIT_NEW")
        inner = _package("INNER_SELECTION")
        bad = ReferenceRoleRelease(att.receipt | {"arm_id": "W_Q"},
                                   att.receipt_sha256, att.row_id, att.unit,
                                   att.wires1, att.wires2)
        with self.assertRaises(ValueError):
            prepare_pstar_three_views_h2(bad, inner)
        left = reference_recipient_batch(att, 1)
        left.fields["probability"].values[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            _local(left)


if __name__ == "__main__":
    unittest.main()
