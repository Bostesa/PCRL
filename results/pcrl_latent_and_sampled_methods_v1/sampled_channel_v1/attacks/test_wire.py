"""Synthetic complete-view and public-channel attack contract checks."""

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import (
    pair_batch, prepare_views,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.wire import (
    CacheIdentity, PublicBank, canonical_bank_sha256,
    materialize_recipient_batch, materialize_pair_batch,
    pair_posterior_lookup,
)


def bank(recipient: int) -> PublicBank:
    if recipient == 1:
        probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    else:
        probabilities = np.asarray([
            [0.7, 0.1, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.7, 0.1, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.7, 0.1, 0.05, 0.05],
        ])
    return PublicBank(recipient=recipient, probability=probabilities,
                      release_id="sampled-s0",
                      public_metadata={"model_id": "sampled-s0",
                                       "source_binding_sha256": "a" * 64,
                                       "canonical_bank_sha256": canonical_bank_sha256(probabilities),
                                       "mechanism_sha256": "c" * 64,
                                       "code_length_bits": 2,
                                       "fallback": False,
                                       "sampling_contract": "cached_one_joint_draw_v1"})


def release(public_bank: PublicBank, code_id: int) -> dict:
    probability = public_bank.probability[code_id].tolist()
    return {"schema": "SCV1-score-release-v1", "recipient": public_bank.recipient,
            "code_id": code_id, "probability": probability,
            "decision": int(np.argmax(probability)), "release_id": public_bank.release_id,
            "public_metadata": dict(public_bank.public_metadata)}


def test_actual_draw_code_probability_decision_and_metadata_enter_view():
    b = bank(1)
    att = materialize_recipient_batch(
        [release(b, 0), release(b, 1)],
        row_id=np.asarray([11, 12], dtype=np.int64),
        unit=np.asarray([21, 22], dtype=np.int64), bank=b)
    inner = materialize_recipient_batch(
        [release(b, 2)],
        row_id=np.asarray([13], dtype=np.int64),
        unit=np.asarray([23], dtype=np.int64), bank=b)
    prepared = prepare_views(att, inner)
    assert "code_id" in att.fields
    assert "code_bits" in att.fields
    assert "probability" in att.fields
    assert "decision" in att.fields
    assert "public_metadata.model_id" in att.fields
    assert "public_metadata.fallback" in att.fields
    assert "public_metadata.canonical_json" in att.fields
    assert prepared.probability_att.tolist() == [b.probability[0].tolist(),
                                                  b.probability[1].tolist()]
    assert prepared.att.cell_key[0] != prepared.att.cell_key[1]
    assert prepared.inner.cell_key[0] != prepared.att.cell_key[0]
    assert prepared.att.x_code_bits.shape == (2, 2)


def test_bank_rejects_duplicate_vector_even_when_ids_differ():
    duplicate = np.asarray([[0.9, 0.1], [0.9, 0.1]])
    with pytest.raises(ValueError, match="duplicate"):
        PublicBank(recipient=1, probability=duplicate, release_id="sampled-s0",
                   public_metadata=dict(bank(1).public_metadata))


def test_person_varying_metadata_outside_public_code_is_refused():
    b = bank(1)
    first, second = release(b, 0), release(b, 1)
    second["public_metadata"]["mechanism_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="registered public metadata"):
        materialize_recipient_batch([first, second],
                                    np.asarray([11, 12], dtype=np.int64),
                                    np.asarray([21, 22], dtype=np.int64), b)


def test_extra_unregistered_metadata_key_is_refused():
    b = bank(1)
    observed = release(b, 0)
    observed["public_metadata"]["route"] = "person-specific-route"
    with pytest.raises(ValueError, match="registered public metadata"):
        materialize_recipient_batch([observed], np.asarray([11], dtype=np.int64),
                                    np.asarray([21], dtype=np.int64), b)


def test_registered_metadata_has_exact_seven_key_schema():
    b = bank(1)
    missing = dict(b.public_metadata)
    missing.pop("sampling_contract")
    with pytest.raises(ValueError, match="metadata key set"):
        PublicBank(recipient=1, probability=b.probability, release_id=b.release_id,
                   public_metadata=missing)
    wrong_model = dict(b.public_metadata, model_id="other-release")
    with pytest.raises(ValueError, match="model_id"):
        PublicBank(recipient=1, probability=b.probability, release_id=b.release_id,
                   public_metadata=wrong_model)
    wrong_hash = dict(b.public_metadata, canonical_bank_sha256="b" * 64)
    with pytest.raises(ValueError, match="canonical bank SHA"):
        PublicBank(recipient=1, probability=b.probability, release_id=b.release_id,
                   public_metadata=wrong_hash)


def test_wire_rejects_averaged_probability_for_one_sampled_code():
    b = bank(1)
    observed = release(b, 0)
    observed["probability"] = ((b.probability[0] + b.probability[1]) / 2).tolist()
    with pytest.raises(ValueError, match="bank row"):
        materialize_recipient_batch([observed], np.asarray([11], dtype=np.int64),
                                    np.asarray([21], dtype=np.int64), b)


@pytest.mark.parametrize("replacement", ["0.9", True, 0])
def test_wire_refuses_non_float_probability_element(replacement):
    b = bank(1)
    observed = release(b, 0)
    observed["probability"][0] = replacement
    with pytest.raises(ValueError, match="float64 JSON numbers"):
        materialize_recipient_batch([observed], np.asarray([11], dtype=np.int64),
                                    np.asarray([21], dtype=np.int64), b)


def test_wire_refuses_hidden_input_or_action_field():
    b = bank(1)
    observed = release(b, 0)
    observed["input_cell"] = 7
    with pytest.raises(ValueError, match="wire fields"):
        materialize_recipient_batch([observed], np.asarray([11], dtype=np.int64),
                                    np.asarray([21], dtype=np.int64), b)


def test_duplicate_group_has_one_cached_release():
    b = bank(1)
    ids = np.asarray([11, 12], dtype=np.int64)
    groups = np.asarray([21, 21], dtype=np.int64)
    materialize_recipient_batch([release(b, 0), release(b, 0)], ids, groups, b)
    with pytest.raises(ValueError, match="cached"):
        materialize_recipient_batch([release(b, 0), release(b, 1)], ids, groups, b)


def test_pair_preserves_both_actual_recipient_views_and_alignment():
    ids = np.asarray([11, 12], dtype=np.int64)
    groups = np.asarray([21, 22], dtype=np.int64)
    r1 = materialize_recipient_batch([release(bank(1), 0), release(bank(1), 1)],
                                     ids, groups, bank(1))
    r2 = materialize_recipient_batch([release(bank(2), 0), release(bank(2), 1)],
                                     ids, groups, bank(2))
    pair = pair_batch(r1, r2)
    assert "r1.code_id" in pair.fields and "r2.code_id" in pair.fields
    assert pair.code_fields == ("r1.code_id", "r2.code_id")
    with pytest.raises(ValueError, match="ordered recipient rows/groups"):
        pair_batch(r1, materialize_recipient_batch(
            [release(bank(2), 0), release(bank(2), 1)],
            ids[::-1].copy(), groups, bank(2)))


def test_paired_materializer_refuses_cross_arm_bank():
    b1, b2 = bank(1), bank(2)
    changed = dict(b2.public_metadata, model_id="different-arm")
    other_arm = PublicBank(2, b2.probability, "different-arm", changed)
    ids = np.asarray([11], dtype=np.int64)
    groups = np.asarray([21], dtype=np.int64)
    first = CacheIdentity(0, "ATTACK_FIT_NEW", 0, b1.release_id,
                          b1.public_metadata["source_binding_sha256"], "e" * 64)
    second = CacheIdentity(0, "ATTACK_FIT_NEW", 0, other_arm.release_id,
                           other_arm.public_metadata["source_binding_sha256"], "e" * 64)
    with pytest.raises(ValueError, match="same paired arm"):
        materialize_pair_batch([release(b1, 0)], [release(other_arm, 0)],
                               ids, groups, b1, other_arm,
                               left_cache=first, right_cache=second)


def test_paired_materializer_requires_same_verified_joint_cache_receipt():
    b1, b2 = bank(1), bank(2)
    ids = np.asarray([11], dtype=np.int64)
    groups = np.asarray([21], dtype=np.int64)
    first = CacheIdentity(0, "ATTACK_FIT_NEW", 0, b1.release_id,
                          b1.public_metadata["source_binding_sha256"], "e" * 64)
    second = CacheIdentity(0, "ATTACK_FIT_NEW", 0, b2.release_id,
                           b2.public_metadata["source_binding_sha256"], "e" * 64)
    materialize_pair_batch([release(b1, 0)], [release(b2, 0)], ids, groups,
                           b1, b2, left_cache=first, right_cache=second)
    changed = CacheIdentity(0, "ATTACK_FIT_NEW", 1, b2.release_id,
                            b2.public_metadata["source_binding_sha256"], "f" * 64)
    with pytest.raises(ValueError, match="same verified joint cache"):
        materialize_pair_batch([release(b1, 0)], [release(b2, 0)], ids, groups,
                               b1, b2, left_cache=first, right_cache=changed)


def test_paired_materializer_refuses_mismatched_public_mechanism():
    b1, b2 = bank(1), bank(2)
    altered = dict(b2.public_metadata, mechanism_sha256="d" * 64)
    b2_other = PublicBank(2, b2.probability, b2.release_id, altered)
    ids = np.asarray([11], dtype=np.int64)
    groups = np.asarray([21], dtype=np.int64)
    cache = CacheIdentity(0, "ATTACK_FIT_NEW", 0, b1.release_id,
                          b1.public_metadata["source_binding_sha256"], "e" * 64)
    with pytest.raises(ValueError, match="same paired arm/source/mechanism"):
        materialize_pair_batch([release(b1, 0)], [release(b2_other, 0)],
                               ids, groups, b1, b2_other,
                               left_cache=cache, right_cache=cache)


def test_channel_aware_posterior_uses_only_public_output_pair_and_law():
    # Two private source states; rows sum to one over four public output pairs.
    prior = np.asarray([0.75, 0.25])
    law = np.asarray([[0.5, 0.2, 0.2, 0.1],
                      [0.1, 0.2, 0.2, 0.5]])
    posterior = pair_posterior_lookup(prior, law, (2, 2))
    assert posterior.shape == (2, 2, 2)
    assert np.allclose(posterior.sum(axis=2), 1.0)
    assert np.allclose(posterior[0, 0], [0.9375, 0.0625])
