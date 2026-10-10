"""Data-free regression for the mandatory continuous defense-aware view."""

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.reference_materialize import (
    reconstruct_reference_wires,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import (
    pair_batch, prepare_views,
)
from .reference_wire import ReferenceRoleRelease, reference_recipient_batch
from .retry_reference_view import corrected_continuous_recipient_batch


def _package(role: str) -> ReferenceRoleRelease:
    row = np.array([11, 12], dtype=np.int64)
    unit = np.array([21, 22], dtype=np.int64)
    p1 = np.array([[1.0 - 1e-14, 1e-14], [0.2, 0.8]], dtype=np.float64)
    p2 = np.array([
        [1.0 - 5e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14],
        [0.02, 0.9, 0.02, 0.02, 0.02, 0.02],
    ], dtype=np.float64)
    receipt = {
        "role": role,
        "arm_id": "U0",
        "source_seed": 0,
        "release_id": "U0-s0-v1",
        "source_binding_sha256": "a" * 64,
        "interface_kind": "continuous",
        "bank_sha256_by_recipient": {"r1": None, "r2": None},
    }
    codes = np.full(2, -1, dtype=np.int64)
    wires1 = reconstruct_reference_wires(
        p1, np.argmax(p1, axis=1).astype(np.int64), codes,
        recipient=1, interface_kind="continuous", release_id="U0-s0-v1",
        source_binding_sha256="a" * 64, bank_sha256=None, bank_size=None,
    )
    wires2 = reconstruct_reference_wires(
        p2, np.argmax(p2, axis=1).astype(np.int64), codes,
        recipient=2, interface_kind="continuous", release_id="U0-s0-v1",
        source_binding_sha256="a" * 64, bank_sha256=None, bank_size=None,
    )
    return ReferenceRoleRelease(receipt, "b" * 64, row, unit, wires1, wires2)


def test_continuous_reference_has_public_defense_aware_columns_in_all_views():
    att, inner = _package("ATTACK_FIT_NEW"), _package("INNER_SELECTION")
    a1, a2 = (corrected_continuous_recipient_batch(att, recipient)
              for recipient in (1, 2))
    i1, i2 = (corrected_continuous_recipient_batch(inner, recipient)
              for recipient in (1, 2))
    for recipient, corrected in ((1, a1), (2, a2)):
        original = reference_recipient_batch(att, recipient)
        assert tuple(corrected.fields) == tuple(original.fields)
        assert corrected.code_fields == original.code_fields == ()
        expected = np.log(np.maximum(
            np.asarray(original.fields["probability"].values), 1e-12))
        np.testing.assert_array_equal(
            corrected.public_derived["log_probabilities"], expected)
    prepared = {
        "r1": prepare_views(a1, i1),
        "r2": prepare_views(a2, i2),
        "pair": prepare_views(pair_batch(a1, a2), pair_batch(i1, i2)),
    }
    for name, width in (("r1", 2), ("r2", 6), ("pair", 8)):
        view = prepared[name]
        assert view.att.x_defense_aware is not None
        assert view.inner.x_defense_aware is not None
        assert view.att.x_defense_aware.shape[1] == view.att.x.shape[1] + width
        assert view.inner.x_defense_aware.shape[1] == view.inner.x.shape[1] + width
        expected = np.log(np.maximum(view.probability_att, 1e-12))
        np.testing.assert_array_equal(view.att.x_defense_aware[:, -width:], expected)
        np.testing.assert_array_equal(view.inner.x_defense_aware[:, -width:], expected)
        assert view.att.cell_key is None
        assert view.inner.cell_key is None


def test_corrected_continuous_view_refuses_finite_interface():
    package = _package("ATTACK_FIT_NEW")
    package.receipt["interface_kind"] = "finite"
    import pytest
    with pytest.raises(ValueError, match="continuous"):
        corrected_continuous_recipient_batch(package, 1)
