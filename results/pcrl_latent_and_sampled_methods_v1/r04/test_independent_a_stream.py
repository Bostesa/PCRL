"""Data-free HMAC/cache/wire checks for the independent A verifier."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_source import sha_array, sha_file
from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_law import (
    IndependentAPolicy, _bank_hash)
from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_stream import (
    _cache_key, _input_digest, _selected_root, _uniform, _wire_hash,
    replay_sampled_release)


def test_replay_one_sticky_pair_and_complete_wire(tmp_path):
    p1 = np.array([[.8, .2]], dtype=np.float64)
    p2 = np.array([[.7, .3, 0, 0, 0, 0]], dtype=np.float64)
    d = np.array([0], dtype=np.int64)
    binding = "b" * 64
    mechanism = "c" * 64
    bank_hashes = {"r1": _bank_hash(p1, d), "r2": _bank_hash(p2, d)}
    policy = IndependentAPolicy(
        {"law_kind": "stochastic_joint_v1", "source_seed": 0,
         "release_id": "toy", "source_binding_sha256": binding,
         "mechanism_sha256": mechanism, "bank_sha256_by_recipient": bank_hashes},
        {"bank1_probability": p1, "bank2_probability": p2,
         "bank1_decision": d, "bank2_decision": d,
         "w1": np.zeros((12, 16)), "b1": np.zeros(16),
         "w2": np.zeros((16, 16)), "b2": np.zeros(16)},
        ((0,),), ((0,),))
    root = b"x" * 32
    root_path = tmp_path / "root.bin"
    root_path.write_bytes(root)
    ids = np.array([3, 4], dtype=np.int64)
    groups = np.array([10, 10], dtype=np.int64)
    score1 = np.repeat(np.array([[.9, .1]]), 2, axis=0)
    score2 = np.repeat(np.array([[.7, .3, 0, 0, 0, 0]]), 2, axis=0)
    cells = np.zeros(2, dtype=np.int64)
    source_identity = {"tstar_bank_admission_receipt_sha256": "1" * 64,
                       "tstar_role_tokens_file_sha256": "2" * 64,
                       "reference_file_sha256": "3" * 64}
    receipt = {"schema": "SCV1-private-realization-v1", "status": "FROZEN_ONE_DRAW",
               "release_id": "toy", "source_seed": 0, "role": "INNER_SELECTION",
               "realization": 1, "rows": 2, "groups": 1,
               "row_unit_pair_sha256": sha_array(np.column_stack((ids, groups))),
               "source_binding_sha256": binding,
               "private_stream_root_sha256": hashlib.sha256(root).hexdigest(),
               "source_identity": source_identity,
               "canonical_bank_sha256_by_recipient": bank_hashes,
               "mechanism_sha256": mechanism}
    cache = tmp_path / "cache.sqlite3"
    connection = sqlite3.connect(cache)
    config = {"schema": "SCV1-joint-cache-v1", "release_id": "toy",
              "source_seed": 0, "role": "INNER_SELECTION", "realization": 1,
              "source_binding_sha256": binding,
              "bank_sha256": [bank_hashes["r1"], bank_hashes["r2"]],
              "mechanism_sha256": mechanism,
              "private_root_sha256": hashlib.sha256(root).hexdigest(),
              "offline": True}
    connection.execute("CREATE TABLE config (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE releases (key TEXT PRIMARY KEY, input_digest TEXT NOT NULL, z1 INTEGER NOT NULL, z2 INTEGER NOT NULL)")
    connection.execute("INSERT INTO config VALUES (1,?)", (json.dumps(config),))
    key = _cache_key(root, receipt, 10)
    digest = _input_digest(score1[0], score2[0], 0, 0, binding)
    connection.execute("INSERT INTO releases VALUES (?,?,0,0)", (key, digest))
    connection.commit()
    connection.close()
    assert 0 <= _uniform(root, 0, "INNER_SELECTION", 1, 10) < 1
    codes = tmp_path / "release_codes.npz"
    np.savez_compressed(codes, row_id=ids, unit=groups,
                        code_r1=np.zeros(2, np.int64), code_r2=np.zeros(2, np.int64))
    receipt["cache_file_sha256"] = sha_file(cache)
    receipt["codes_file_sha256"] = sha_file(codes)
    receipt["wire_sha256_by_recipient"] = {
        f"r{r}": _wire_hash(policy, receipt, r, np.zeros(2, np.int64)) for r in (1, 2)}
    receipt_path = tmp_path / "RELEASE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    result = replay_sampled_release(
        policy, directory=tmp_path, expected_receipt_sha256=sha_file(receipt_path),
        root_path=root_path, row_id=ids, unit=groups,
        score1=score1, score2=score2, cell1=cells, cell2=cells,
        expected_source_identity=source_identity)
    assert result["hmac_cache_exact"] and result["groups"] == 1


def test_public_stream_precommitment_selects_one_draw(tmp_path):
    roots = {str(i): bytes([i + 1]) * 32 for i in range(3)}
    seq = b"s" * 32
    private = tmp_path / "roots.json"
    private.write_text(json.dumps({"schema": "scv1-private-stream-roots-v1",
                                   "release_realization_roots_hex": {
                                       key: value.hex() for key, value in roots.items()},
                                   "seq_fit_root_hex": seq.hex()}))
    private.chmod(0o600)
    public = tmp_path / "public.json"
    public.write_text(json.dumps({
        "schema": "scv1-public-stream-root-hashes-v1", "private_file_mode": "0600",
        "private_file_sha256": sha_file(private),
        "release_hmac_message": "compact ASCII JSON [SCV1-uniform-v1,source_seed,role,realization,unit]",
        "seq_fit_hmac_domain": "SCV1-seq-fit-first-v1",
        "release_realization_root_sha256": {
            key: hashlib.sha256(value).hexdigest() for key, value in roots.items()},
        "seq_fit_root_sha256": hashlib.sha256(seq).hexdigest()}))
    assert _selected_root(private, 2, public, sha_file(public)) == roots["2"]
