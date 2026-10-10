import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import (
    Field, WireBatch, prepare_views, pair_batch,
)


def batch(ids=(1, 2, 3), code=(0, 1, 0)):
    return WireBatch(
        row_id=np.asarray(ids, dtype=np.int64),
        unit=np.asarray(ids, dtype=np.int64),
        fields={
            "code": Field("category", np.asarray(code)),
            "bits": Field("bits", np.asarray([[0, 0], [0, 1], [0, 0]])),
            "probabilities": Field("probability", np.asarray([[.8, .2], [.4, .6], [.7, .3]])),
            "decision": Field("decision", np.asarray([0, 1, 0])),
            "fallback": Field("metadata", np.asarray([0, 1, 0])),
        },
        code_fields=("code",),
    )


def test_exact_wire_fields_are_bound_and_unseen_codes_have_fallback():
    att = batch()
    inner = batch(ids=(4, 5, 6), code=(0, 9, 1))
    result = prepare_views(att, inner)
    result.att.validate()
    result.inner.validate()
    assert result.att.wire_components == tuple(att.fields)
    assert result.att.cell_key.tolist() == ['["0"]', '["1"]', '["0"]']
    assert result.inner.cell_key.tolist() == ['["0"]', '["9"]', '["1"]']
    assert result.inner.x.shape == result.att.x.shape
    assert result.inner.x[1, 2] == 1  # code unknown indicator, not an ordinal code
    assert result.probability_att.shape == (3, 2)
    assert result.att.x_code_bits.shape == (3, 2)
    assert result.att.x_metadata.shape == (3, 1)


def test_schema_changes_or_misaligned_pair_fail():
    att = batch()
    altered = batch(ids=(4, 5, 6))
    altered.fields.pop("fallback")
    with pytest.raises(ValueError, match="schema"):
        prepare_views(att, altered)
    with pytest.raises(ValueError, match="ordered"):
        pair_batch(att, batch(ids=(3, 2, 1)))


def test_pair_has_both_wire_names_and_exact_joint_code_key():
    left = batch()
    right = batch(code=(1, 0, 1))
    prepared = prepare_views(pair_batch(left, right), pair_batch(left, right))
    assert len(prepared.att.wire_components) == 10
    assert prepared.att.cell_key[0] == '["0","1"]'
    assert prepared.att.x.shape[1] == prepared.inner.x.shape[1]


def test_probability_and_bit_contract_reject_invalid_values():
    broken = batch()
    broken.fields["probabilities"] = Field("probability", [[1.1, -.1]] * 3)
    with pytest.raises(ValueError, match="probability"):
        prepare_views(broken, broken)
    broken = batch()
    broken.fields["bits"] = Field("bits", [[0, 2]] * 3)
    with pytest.raises(ValueError, match="bit"):
        prepare_views(broken, broken)


def test_actual_float32_softmax_rounding_is_accepted_without_modifying_wire():
    rounded = batch()
    p = np.asarray([[.7, .3], [.1, .9], [.2, .8]], dtype=np.float32)
    rounded.fields["probabilities"] = Field("probability", p)
    result = prepare_views(rounded, rounded)
    np.testing.assert_array_equal(result.probability_att, p.astype(np.float64))
