"""Continuous U0/Ucal score wire must support the mandatory DA reader.

This is a data-free regression against the original saved-wire adapter. The
log-probability transformation matches the established D U-anchor adapter.
"""

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.reference_anchor import (
    wire_batch_anchor,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.reference_wire import (
    ReferenceRoleRelease, WIRE_SCHEMA, prepare_reference_three_views,
    reference_recipient_batch,
)


def _role(anchor: str, role: str) -> ReferenceRoleRelease:
    row = np.array([11, 12], np.int64)
    unit = np.array([21, 22], np.int64)
    q1 = np.array([[1., 0.], [.2, .8]], np.float64)
    q2 = np.array([[1., 0., 0., 0., 0., 0.],
                   [.05, .8, .05, .05, .025, .025]], np.float64)
    release_id = f"{anchor}-s0"
    source = "e" * 64
    receipt = {"role": role, "arm_id": anchor, "source_seed": 0,
               "release_id": release_id, "source_binding_sha256": source,
               "interface_kind": "continuous",
               "bank_sha256_by_recipient": {"r1": None, "r2": None}}

    def wires(recipient, values):
        metadata = {"model_id": release_id, "source_binding_sha256": source,
                    "bank_sha256": None, "code_length_bits": None}
        return tuple({"schema": WIRE_SCHEMA, "recipient": recipient,
                      "probability": q.tolist(), "decision": int(np.argmax(q)),
                      "release_id": release_id, "public_metadata": metadata}
                     for q in values)

    return ReferenceRoleRelease(receipt, "a" * 64, row, unit,
                                wires(1, q1), wires(2, q2))


@pytest.mark.parametrize("anchor", ["U0", "Ucal"])
def test_continuous_reference_da_matches_public_log_probability_anchor(anchor):
    att = _role(anchor, "ATTACK_FIT_NEW")
    inner = _role(anchor, "INNER_SELECTION")
    q1 = np.asarray([wire["probability"] for wire in att.wires1], np.float64)
    q2 = np.asarray([wire["probability"] for wire in att.wires2], np.float64)
    record = {"row_id": att.row_id, "unit": att.unit,
              "U0_1": q1, "U0_2": q2, "Ucal_1": q1, "Ucal_2": q2,
              "d_1": np.argmax(q1, axis=1).astype(np.int64),
              "d_2": np.argmax(q2, axis=1).astype(np.int64)}
    for recipient, q in ((1, q1), (2, q2)):
        batch = reference_recipient_batch(att, recipient)
        anchor_batch = wire_batch_anchor(record, anchor, recipient)
        assert batch.code_fields == ()
        assert "code_id" not in batch.fields
        assert set(batch.public_derived or ()) == {"log_probabilities"}
        np.testing.assert_array_equal(batch.public_derived["log_probabilities"],
                                      anchor_batch.public_derived["log_probabilities"])
        np.testing.assert_array_equal(batch.public_derived["log_probabilities"],
                                      np.log(np.maximum(q, 1e-12)))
    views = prepare_reference_three_views(att, inner)
    assert set(views) == {"r1", "r2", "pair"}
    for name, extra in (("r1", 2), ("r2", 6), ("pair", 8)):
        for role in ("att", "inner"):
            view = getattr(views[name], role)
            assert view.x_defense_aware is not None
            assert view.x_defense_aware.shape == (2, view.x.shape[1] + extra)
            np.testing.assert_array_equal(view.x_defense_aware[:, :view.x.shape[1]],
                                          view.x)
