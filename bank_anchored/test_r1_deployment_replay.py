"""Data-free checks for queue-only replay of the actual selected-policy CLI."""

import numpy as np
import pytest

from bank_anchored.r1_deployment_replay import audit_wire, derive_uniforms, run


def _wire(path, **overrides):
    code = np.zeros((2, 10), dtype=np.uint8)
    code[1, 0] = 1
    fields = {
        "code": code,
        "prob": np.array([[.8, .2], [.7, .3]], dtype=np.float64),
        "decision": np.array([0, 0], dtype=np.int64),
    }
    fields.update(overrides)
    np.savez_compressed(path, **fields)


def test_replay_rejects_wire_not_exactly_decoded_from_public_ids(tmp_path):
    table = np.array([[.8, .2], [.7, .3]], dtype=np.float64)
    original = tmp_path / "original.npz"
    changed = tmp_path / "changed.npz"
    _wire(original)
    assert audit_wire(original, table, original, identity=True)["rows"] == 2
    _wire(changed, prob=np.array([[.8, .2], [.6, .4]], dtype=np.float64))
    with pytest.raises(ValueError, match="fixed bank table"):
        audit_wire(changed, table, original, identity=False)


def test_replay_rejects_hidden_field_and_direct_nonqueue_call(tmp_path):
    table = np.array([[.8, .2], [.7, .3]], dtype=np.float64)
    original = tmp_path / "original.npz"
    hidden = tmp_path / "hidden.npz"
    _wire(original)
    _wire(hidden, fine_cell=np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="hidden field"):
        audit_wire(hidden, table, original, identity=False)
    with pytest.raises(RuntimeError, match="active queue"):
        run(tmp_path / "not_read.json")


def test_replay_draws_match_sealed_unit_cached_hmac_domains():
    root = bytes(range(32))
    units = np.array([7, 7, *range(8, 38)], dtype=np.int64)
    joint = derive_uniforms(root, seed=0, arm_id="J", units=units)
    assert joint.dtype == np.float64 and joint.shape == (32,)
    assert joint[0] == joint[1] == 0.8224904208972221
    factor = derive_uniforms(root, seed=0, arm_id="P", units=units)
    assert factor.dtype == np.float64 and factor.shape == (32, 2)
    assert tuple(factor[0]) == (0.10723990052591165, 0.6892582348827188)
    assert np.array_equal(factor[0], factor[1])
    assert derive_uniforms(root, seed=0, arm_id="B0", units=units) is None
    assert derive_uniforms(root, seed=0, arm_id="D", units=units) is None
