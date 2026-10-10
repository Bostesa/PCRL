"""Data-free deterministic full-wire restoration and attack-view checks."""

import hashlib
import json

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.reference_materialize import (
    materialize_reference_role,
)
from .reference_wire import (
    load_public_fixed_map, load_reference_role_package, prepare_reference_three_views,
    reference_recipient_batch, with_reference_q_features,
)
from .q_aware import compile_uniform_prototype_features
from .wire import PublicBank, canonical_bank_sha256
from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import pair_batch


def _source():
    return {"role_source_file_sha256": "a" * 64,
            "reference_file_sha256": "b" * 64,
            "bank_receipt_sha256": "c" * 64,
            "source_audit_sha256": "d" * 64}


def _banks():
    return (np.array([[.9, .1], [.2, .8]], np.float64),
            np.array([[.9, .02, .02, .02, .02, .02],
                      [.02, .9, .02, .02, .02, .02]], np.float64))


def _materialize(tmp_path, role):
    b1, b2 = _banks()
    row = np.array([11, 12], np.int64)
    unit = np.array([21, 22], np.int64)
    code = np.array([0, 1], np.int64)
    output = tmp_path / ".private" / role
    output.parent.mkdir(exist_ok=True, mode=0o700)
    materialize_reference_role(
        row_id=row, unit=unit, prob1=b1[code], prob2=b2[code],
        decision1=code, decision2=code, code1=code, code2=code,
        arm_id="T_STAR", seed=0, role=role, release_id="T_STAR-s0",
        source_binding_sha256="e" * 64, source_identity=_source(),
        interface_kind="finite",
        bank_sha256_by_recipient={"r1": sha_array(b1), "r2": sha_array(b2)},
        expected_row_unit_sha256=sha_array(np.column_stack((row, unit))),
        output_dir=output, bank1=b1, bank2=b2)
    sha = hashlib.sha256((output / "REFERENCE_RELEASE_RECEIPT.json").read_bytes()).hexdigest()
    return output, sha, b1, b2


def test_finite_reference_restores_actual_code_probability_metadata_and_pair(tmp_path):
    att_dir, att_sha, b1, b2 = _materialize(tmp_path, "ATTACK_FIT_NEW")
    inner_dir, inner_sha, _, _ = _materialize(tmp_path, "INNER_SELECTION")
    att = load_reference_role_package(att_dir, receipt_sha256=att_sha,
                                      full_bank1=b1, full_bank2=b2)
    inner = load_reference_role_package(inner_dir, receipt_sha256=inner_sha,
                                        full_bank1=b1, full_bank2=b2)
    assert set(att.wires1[0]) == {"schema", "recipient", "code_id", "probability",
                                  "decision", "release_id", "public_metadata"}
    assert att.wires1[0]["code_id"] == 0
    assert att.wires2[1]["probability"] == b2[1].tolist()
    assert att.wires1[0]["public_metadata"]["code_length_bits"] == 1
    batch = reference_recipient_batch(att, 1)
    assert batch.fields["code_bits"].values.tolist() == [[0.0], [1.0]]
    prepared = prepare_reference_three_views(att, inner)
    assert set(prepared) == {"r1", "r2", "pair"}
    assert prepared["pair"].att.cell_key.tolist() == ['["0","0"]', '["1","1"]']
    assert prepared["pair"].probability_att.shape == (2, 8)


def test_reference_loader_refuses_hidden_npz_field_and_forged_wire_sha(tmp_path):
    root, receipt_sha, b1, b2 = _materialize(tmp_path, "ATTACK_FIT_NEW")
    path = root / "reference_release.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays, hidden_cell=np.array([0, 1], np.int64))
    record_path = root / "REFERENCE_RELEASE_RECEIPT.json"
    record = json.loads(record_path.read_text())
    record["codes_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="member set"):
        load_reference_role_package(
            root, receipt_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
            full_bank1=b1, full_bank2=b2)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    record["codes_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["wire_sha256_by_recipient"]["r1"] = "0" * 64
    record_path.write_text(json.dumps(record, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="wire SHA"):
        load_reference_role_package(
            root, receipt_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
            full_bank1=b1, full_bank2=b2)


def test_fixed_map_q_features_use_observed_code_without_extra_wire_fields(tmp_path):
    root, sha, b1, b2 = _materialize(tmp_path, "ATTACK_FIT_NEW")
    package = load_reference_role_package(root, receipt_sha256=sha,
                                          full_bank1=b1, full_bank2=b2)
    metadata = lambda bank: {
        "model_id": "T_STAR-s0", "source_binding_sha256": "e" * 64,
        "canonical_bank_sha256": canonical_bank_sha256(bank),
        "mechanism_sha256": "f" * 64,
        "code_length_bits": 1, "fallback": False,
        "sampling_contract": "cached_one_joint_draw_v1"}
    public1 = PublicBank(1, b1, "T_STAR-s0", metadata(b1))
    public2 = PublicBank(2, b2, "T_STAR-s0", metadata(b2))
    table = compile_uniform_prototype_features(
        public1, public2, b1, b2,
        lambda _s1, _s2, i, j: ((i, j, 1.0),),
        public_centroids_sha256="a" * 64)
    pair = pair_batch(reference_recipient_batch(package, 1),
                      reference_recipient_batch(package, 2))
    attached = with_reference_q_features(
        pair, table, view="pair", full_bank1=b1, full_bank2=b2,
        role_release=package, expected_mechanism_sha256="f" * 64)
    assert tuple(attached.fields) == tuple(pair.fields)
    assert "q_uniform_proto_v1" in attached.public_derived
    assert attached.public_derived["q_uniform_proto_v1"].shape[0] == 2
    with pytest.raises(ValueError, match="mechanism"):
        with_reference_q_features(pair, table, view="pair", full_bank1=b1,
                                  full_bank2=b2, role_release=package,
                                  expected_mechanism_sha256="0" * 64)


def test_continuous_reference_has_no_fabricated_code_and_pair_is_aligned(tmp_path):
    b1, b2 = _banks()
    row = np.array([11, 12], np.int64)
    unit = np.array([21, 22], np.int64)
    packages = []
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        output = tmp_path / ".private" / role
        output.parent.mkdir(exist_ok=True, mode=0o700)
        source = _source()
        source["bank_receipt_sha256"] = None
        materialize_reference_role(
            row_id=row, unit=unit, prob1=b1, prob2=b2,
            decision1=np.array([0, 1], np.int64),
            decision2=np.array([0, 1], np.int64),
            code1=np.full(2, -1, np.int64), code2=np.full(2, -1, np.int64),
            arm_id="U0", seed=0, role=role, release_id="U0-s0",
            source_binding_sha256="e" * 64, source_identity=source,
            interface_kind="continuous",
            bank_sha256_by_recipient={"r1": None, "r2": None},
            expected_row_unit_sha256=sha_array(np.column_stack((row, unit))),
            output_dir=output)
        digest = hashlib.sha256(
            (output / "REFERENCE_RELEASE_RECEIPT.json").read_bytes()).hexdigest()
        packages.append(load_reference_role_package(output, receipt_sha256=digest))
    att, inner = packages
    assert set(att.wires1[0]) == {"schema", "recipient", "probability",
                                  "decision", "release_id", "public_metadata"}
    assert att.wires1[0]["public_metadata"]["code_length_bits"] is None
    assert att.wires1[0]["public_metadata"]["bank_sha256"] is None
    batch = reference_recipient_batch(att, 1)
    assert "code_id" not in batch.fields and "code_bits" not in batch.fields
    assert batch.code_fields == ()
    prepared = prepare_reference_three_views(att, inner)
    assert prepared["pair"].att.cell_key is None
    assert prepared["pair"].probability_att.shape == (2, 8)
    assert prepared["pair"].att.row_id.tolist() == row.tolist()


def test_public_fixed_map_checks_exact_label_free_banks_and_member_set(tmp_path):
    b1, b2 = _banks()
    values = {"bank_r1": b1, "bank_r2": b2,
              "fine_to_code_r1": np.arange(64, dtype=np.int64) % 2,
              "fine_to_code_r2": np.arange(641, dtype=np.int64) % 2}
    archive = tmp_path / "A_FIXED_MAP_T_STAR_S0.npz"
    np.savez_compressed(archive, **values)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {"schema": "pcrl-r01-public-fixed-fine-map-v1",
               "status": "LABEL_FREE_FIXED_PUBLIC_SOURCE",
               "arm_id": "T_STAR", "seed": 0, "release_id": "T_STAR-s0",
               "source_kind": "historical_finite_reference",
               "source_binding_sha256": "a" * 64,
               "source_audit_sha256": "b" * 64,
               "original_bank_receipt_sha256": "c" * 64,
               "original_bank_file_sha256_by_recipient": {
                   "r1": "d" * 64, "r2": "e" * 64},
               "array_sha256": {name: sha_array(value)
                                for name, value in values.items()},
               "array_shape": {name: list(value.shape)
                               for name, value in values.items()},
               "npz_sha256": sha(archive)}
    receipt_path = tmp_path / "A_FIXED_MAP_T_STAR_S0.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    restored, got = load_public_fixed_map(
        receipt_path, archive, expected_receipt_sha256=sha(receipt_path),
        expected_archive_sha256=sha(archive), arm_id="T_STAR", seed=0)
    assert got["source_binding_sha256"] == "a" * 64
    assert np.array_equal(restored["fine_to_code_r2"], values["fine_to_code_r2"])
    with archive.open("wb") as stream:
        np.savez_compressed(stream, **values, hidden_role=np.array([1], np.int64))
    receipt["npz_sha256"] = sha(archive)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="member set"):
        load_public_fixed_map(
            receipt_path, archive, expected_receipt_sha256=sha(receipt_path),
            expected_archive_sha256=sha(archive), arm_id="T_STAR", seed=0)
