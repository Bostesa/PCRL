"""Synthetic public-prototype Q reader checks; no Adult role values."""

from dataclasses import replace

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.q_aware import (
    compile_uniform_prototype_features, fit_q_posterior_reader, with_q_features,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.test_wire import (
    bank, release,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.wire import (
    CacheIdentity, materialize_pair_batch, materialize_recipient_batch,
)


def public_prototypes():
    c1 = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    c2 = np.asarray([
        [0.7, 0.1, 0.05, 0.05, 0.05, 0.05],
        [0.05, 0.7, 0.1, 0.05, 0.05, 0.05],
    ], dtype=np.float64)
    return c1, c2


def public_law(score1, score2, cell1, cell2):
    assert np.array_equal(score1, public_prototypes()[0][cell1])
    assert np.array_equal(score2, public_prototypes()[1][cell2])
    if cell1 == 0:
        return ((0, cell2, 0.75), (1, cell2, 0.25))
    return ((2, cell2, 1.0),)


def table():
    c1, c2 = public_prototypes()
    return compile_uniform_prototype_features(
        bank(1), bank(2), c1, c2, public_law,
        public_centroids_sha256="d" * 64)


def test_uniform_public_prototype_lookup_uses_emitted_codes_only():
    b1, b2 = bank(1), bank(2)
    features = table()
    assert features.prototype_count == 4
    assert features.pair.shape[:2] == (3, 3)
    assert features.r1.shape[0] == 3 and features.r2.shape[0] == 3
    assert np.isclose(features.pair_mass.sum(), 1.0)
    assert np.isclose(features.pair_mass[0, 0], 0.1875)
    ids = np.asarray([11, 12], dtype=np.int64)
    groups = np.asarray([21, 22], dtype=np.int64)
    cache = CacheIdentity(0, "ATTACK_FIT_NEW", 0, b1.release_id,
                          b1.public_metadata["source_binding_sha256"], "e" * 64)
    pair = materialize_pair_batch([release(b1, 0), release(b1, 1)],
                                  [release(b2, 0), release(b2, 1)],
                                  ids, groups, b1, b2,
                                  left_cache=cache, right_cache=cache)
    augmented = with_q_features(pair, features, view="pair")
    assert np.array_equal(augmented.public_derived["q_uniform_proto_v1"],
                          features.pair[[0, 1], [0, 1]])
    assert "r1.code_id" in augmented.fields
    assert "r1.public_metadata.canonical_json" in augmented.fields
    assert with_q_features(augmented, features, view="pair") is augmented
    planted = dict(augmented.public_derived)
    planted["q_uniform_proto_v1"] = planted["q_uniform_proto_v1"].copy()
    planted["q_uniform_proto_v1"][0, 0] += 0.01
    from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import WireBatch
    altered = WireBatch(augmented.row_id, augmented.unit, augmented.fields,
                        augmented.code_fields, planted)
    with pytest.raises(ValueError, match="public Q feature parity"):
        with_q_features(altered, features, view="pair")
    local = materialize_recipient_batch([release(b1, 0), release(b1, 1)],
                                        ids, groups, b1)
    assert np.array_equal(with_q_features(local, features, view="r1")
                          .public_derived["q_uniform_proto_v1"],
                          features.r1[[0, 1]])


def test_uniform_lookup_refuses_private_or_unreachable_code_state():
    b1, b2 = bank(1), bank(2)
    features = table()
    ids = np.asarray([11], dtype=np.int64)
    groups = np.asarray([21], dtype=np.int64)
    # Occupation code 2 exists in the bank but the prototype law never emits it.
    local = materialize_recipient_batch([release(b2, 2)], ids, groups, b2)
    with pytest.raises(ValueError, match="zero public prototype support"):
        with_q_features(local, features, view="r2")
    wrong = dict(features.identity, public_centroids_sha256="f" * 64)
    with pytest.raises(ValueError, match="source identity"):
        compile_uniform_prototype_features(b1, b2, *public_prototypes(),
                                           public_law, public_centroids_sha256="d" * 64,
                                           expected_identity=wrong)


def test_public_q_table_bytes_are_rehashed_before_lookup():
    features = table()
    changed = features.r1.copy()
    changed[0, 0] += 0.01
    tampered = replace(features, r1=changed)
    b1 = bank(1)
    local = materialize_recipient_batch([release(b1, 0)],
                                        np.asarray([11], dtype=np.int64),
                                        np.asarray([21], dtype=np.int64), b1)
    with pytest.raises(ValueError, match="public Q table content SHA"):
        with_q_features(local, tampered, view="r1")


def test_q_reader_refuses_fit_without_active_a_queue_before_sex_access(tmp_path):
    features = table()
    b1 = bank(1)
    local = materialize_recipient_batch([release(b1, 0)],
                                        np.asarray([11], dtype=np.int64),
                                        np.asarray([21], dtype=np.int64), b1)
    lock = tmp_path / "A_LOCK.json"
    lock.write_text("{}\n")
    import hashlib
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    class NoSex(dict):
        def __getitem__(self, key):
            raise AssertionError("ATT SEX accessed before A queue gate")
    with pytest.raises(RuntimeError, match="active queue slot environment missing"):
        fit_q_posterior_reader(local, local, NoSex(), features,
                               view="r1", attacker_seed=0,
                               private_dir=tmp_path / ".private" / "reader",
                               lock_path=lock, lock_sha256=lock_sha,
                               expected_table_sha256=features.table_sha256,
                               queue_unit_id="A-SYNTHETIC-NOT-RUNNING",
                               queue_guard=lambda: None)
