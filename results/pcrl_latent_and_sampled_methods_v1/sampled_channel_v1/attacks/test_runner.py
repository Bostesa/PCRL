"""Synthetic A attack assembly; no Adult role values or reader fits."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.runner import (
    SampledRoleRelease, assert_a_attack_admission,
    load_fixed_saved_release_pair, load_role_release_package,
    prepare_sampled_three_views, run_sampled_attack,
    validate_attack_identity,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.test_wire import (
    bank, release,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.wire import (
    CacheIdentity, PublicBank,
)
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array


def role(role_name: str, codes1=(0, 1), codes2=(0, 1), realization=0):
    b1, b2 = bank(1), bank(2)
    ids = np.asarray([11, 12], dtype=np.int64)
    units = np.asarray([21, 22], dtype=np.int64)
    cache = CacheIdentity(0, role_name, realization, b1.release_id,
                          b1.public_metadata["source_binding_sha256"],
                          ("e" if role_name == "ATTACK_FIT_NEW" else "f") * 64)
    return SampledRoleRelease(role_name, tuple(release(b1, i) for i in codes1),
                              tuple(release(b2, j) for j in codes2),
                              ids, units, cache, cache)


def synthetic_release_package(tmp_path, role_name="ATTACK_FIT_NEW"):
    root = tmp_path / ".private" / role_name
    root.mkdir(parents=True, mode=0o700)
    sample = role(role_name)
    cache_path = root / "cache.sqlite3"
    cache_path.write_bytes(b"synthetic cache custody bytes")
    codes_path = root / "release_codes.npz"
    np.savez_compressed(codes_path, row_id=sample.row_id, unit=sample.unit,
                        code_r1=np.asarray([0, 1], np.int64),
                        code_r2=np.asarray([0, 1], np.int64))
    cache_path.chmod(0o600)
    codes_path.chmod(0o600)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def wire_digest(wires):
        h = hashlib.sha256()
        for wire in wires:
            h.update(json.dumps(wire, sort_keys=True, separators=(",", ":"),
                                allow_nan=False).encode() + b"\n")
        return h.hexdigest()
    receipt = {
        "schema": "SCV1-private-realization-v1", "status": "FROZEN_ONE_DRAW",
        "release_id": sample.left_cache.release_id, "source_seed": 0,
        "role": role_name, "realization": 0, "rows": 2, "groups": 2,
        "row_unit_pair_sha256": sha_array(np.column_stack((sample.row_id, sample.unit))),
        "source_binding_sha256": sample.left_cache.source_binding_sha256,
        "source_identity": {name: "1" * 64 for name in (
            "tstar_bank_admission_receipt_sha256", "tstar_role_tokens_file_sha256",
            "reference_file_sha256")},
        "canonical_bank_sha256_by_recipient": {
            "r1": bank(1).public_metadata["canonical_bank_sha256"],
            "r2": bank(2).public_metadata["canonical_bank_sha256"]},
        "mechanism_sha256": bank(1).public_metadata["mechanism_sha256"],
        "private_stream_root_sha256": "2" * 64,
        "cache_file_sha256": digest(cache_path),
        "codes_file_sha256": digest(codes_path),
        "wire_sha256_by_recipient": {
            "r1": wire_digest(sample.wires1), "r2": wire_digest(sample.wires2)},
    }
    receipt_path = root / "RELEASE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    receipt_path.chmod(0o600)
    return root, receipt


def test_fixed_saved_release_loader_reconstructs_exact_wire_and_refuses_tamper(tmp_path):
    root, receipt = synthetic_release_package(tmp_path)
    receipt_path = root / "RELEASE_RECEIPT.json"
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    loaded = load_role_release_package(root, bank(1), bank(2),
                                       expected_receipt_sha256=receipt_sha)
    assert loaded.wires1 == role("ATTACK_FIT_NEW").wires1
    assert loaded.wires2 == role("ATTACK_FIT_NEW").wires2
    assert loaded.left_cache.cache_sha256 == receipt["cache_file_sha256"]
    receipt["wire_sha256_by_recipient"]["r2"] = "0" * 64
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="wire SHA"):
        load_role_release_package(root, bank(1), bank(2),
                                  expected_receipt_sha256=hashlib.sha256(
                                      receipt_path.read_bytes()).hexdigest())


def test_saved_release_cache_bytes_and_exact_member_schema_are_checked(tmp_path):
    root, _ = synthetic_release_package(tmp_path)
    receipt = root / "RELEASE_RECEIPT.json"
    sha = lambda: hashlib.sha256(receipt.read_bytes()).hexdigest()
    (root / "cache.sqlite3").write_bytes(b"changed")
    with pytest.raises(ValueError, match="cache or code artifact SHA"):
        load_role_release_package(root, bank(1), bank(2),
                                  expected_receipt_sha256=sha())
    root2, _ = synthetic_release_package(tmp_path / "second")
    code_path = root2 / "release_codes.npz"
    with code_path.open("wb") as output:
        np.savez_compressed(output, row_id=np.array([11, 12], np.int64),
                            unit=np.array([21, 22], np.int64),
                            code_r1=np.array([0, 1], np.int64),
                            code_r2=np.array([0, 1], np.int64),
                            hidden_token=np.array([3, 4], np.int64))
    record = json.loads((root2 / "RELEASE_RECEIPT.json").read_text())
    record["codes_file_sha256"] = hashlib.sha256(code_path.read_bytes()).hexdigest()
    (root2 / "RELEASE_RECEIPT.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="member set"):
        load_role_release_package(
            root2, bank(1), bank(2), expected_receipt_sha256=hashlib.sha256(
                (root2 / "RELEASE_RECEIPT.json").read_bytes()).hexdigest())


def test_attack_loader_matches_r12_actual_materializer(tmp_path, monkeypatch):
    from pathlib import Path
    impl = Path(__file__).resolve().parents[1] / "impl"
    monkeypatch.syspath_prepend(str(impl))
    from bank import FixedBank
    from channel import GateWeights, JointPolicy
    from release import CachedJointRelease
    from materialize import materialize_role, reconstruct_wire_from_codes

    source1 = np.array([[.9, .1], [.7, .3]], np.float64)
    source2 = np.array([[.8, .04, .04, .04, .04, .04],
                        [.7, .06, .06, .06, .06, .06]], np.float64)
    b1 = FixedBank.from_source(source1, np.arange(2, dtype=np.int64),
                               np.zeros(2, np.int64))
    b2 = FixedBank.from_source(source2, np.arange(2, dtype=np.int64),
                               np.zeros(2, np.int64))
    policy = JointPolicy(b1, b2, (tuple(b1.old_to_global),),
                         (tuple(b2.old_to_global),),
                         GateWeights.zeros(input_width=12))
    root = tmp_path / ".private" / "r12"
    root.mkdir(parents=True, mode=0o700)
    service = CachedJointRelease(
        policy, cache_path=root / "cache.sqlite3", release_id="arm-s0",
        source_seed=0, role="ATTACK_FIT_NEW", realization=1,
        source_binding_sha256="a" * 64, private_root=b"x" * 32, offline=True)
    row_id = np.array([11, 12], np.int64)
    unit = np.array([21, 22], np.int64)
    materialize_role(
        service, row_id=row_id, unit=unit, score1=source1, score2=source2,
        cell1=np.zeros(2, np.int64), cell2=np.zeros(2, np.int64),
        output_dir=root,
        expected_row_unit_sha256=sha_array(np.column_stack((row_id, unit))),
        source_identity={name: "b" * 64 for name in (
            "tstar_bank_admission_receipt_sha256", "tstar_role_tokens_file_sha256",
            "reference_file_sha256")})

    def public_bank(recipient, fixed):
        return PublicBank(recipient, fixed.probability, service.release_id, {
            "model_id": service.release_id,
            "source_binding_sha256": service.source_binding_sha256,
            "canonical_bank_sha256": service.bank_sha256[recipient - 1],
            "mechanism_sha256": service.mechanism_sha256,
            "code_length_bits": (len(fixed.probability) - 1).bit_length(),
            "fallback": False,
            "sampling_contract": "cached_one_joint_draw_v1",
        })
    public1, public2 = public_bank(1, b1), public_bank(2, b2)
    receipt_path = root / "RELEASE_RECEIPT.json"
    package = load_role_release_package(
        root, public1, public2,
        expected_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest())
    with np.load(root / "release_codes.npz", allow_pickle=False) as stored:
        expected1 = reconstruct_wire_from_codes(service, stored["code_r1"], recipient=1)
        expected2 = reconstruct_wire_from_codes(service, stored["code_r2"], recipient=2)
    assert package.wires1 == expected1
    assert package.wires2 == expected2


def test_attack_identity_binds_both_receipts_streams_policy_and_draw(tmp_path):
    att_dir, _ = synthetic_release_package(tmp_path, "ATTACK_FIT_NEW")
    inner_dir, _ = synthetic_release_package(tmp_path, "INNER_SELECTION")
    def load(path):
        return load_role_release_package(
            path, bank(1), bank(2),
            expected_receipt_sha256=hashlib.sha256(
                (path / "RELEASE_RECEIPT.json").read_bytes()).hexdigest())
    att, inner = load(att_dir), load(inner_dir)
    identity = {
        "arm_id": "A_JOINT_STOCH", "source_seed": 0, "realization": 0,
        "release_id": bank(1).release_id,
        "source_binding_sha256": bank(1).public_metadata["source_binding_sha256"],
        "mechanism_sha256": bank(1).public_metadata["mechanism_sha256"],
        "canonical_bank_sha256_by_recipient": {
            "r1": bank(1).public_metadata["canonical_bank_sha256"],
            "r2": bank(2).public_metadata["canonical_bank_sha256"]},
        "release_receipt_sha256_by_role": {
            "ATTACK_FIT_NEW": att.release_receipt_sha256,
            "INNER_SELECTION": inner.release_receipt_sha256},
        "private_stream_root_sha256_by_role": {
            "ATTACK_FIT_NEW": att.private_stream_root_sha256,
            "INNER_SELECTION": inner.private_stream_root_sha256},
        "policy_receipt_sha256": "a" * 64,
    }
    assert validate_attack_identity(
        {"attack_identity": identity}, att, inner, bank(1), bank(2),
        policy_receipt_sha256="a" * 64,
        policy_arm_id="A_JOINT_STOCH") == identity
    for mutation in (
            {"realization": 1},
            {"arm_id": "A_TASK_STOCH"},
            {"release_receipt_sha256_by_role": {
                **identity["release_receipt_sha256_by_role"],
                "INNER_SELECTION": "f" * 64}},
            {"private_stream_root_sha256_by_role": {
                **identity["private_stream_root_sha256_by_role"],
                "ATTACK_FIT_NEW": "f" * 64}}):
        with pytest.raises(ValueError, match="immutable attack identity"):
            validate_attack_identity(
                {"attack_identity": {**identity, **mutation}},
                att, inner, bank(1), bank(2),
                policy_receipt_sha256="a" * 64,
                policy_arm_id="A_JOINT_STOCH")


def test_fixed_outer_loader_uses_r12_selected_policy_and_two_receipts(tmp_path, monkeypatch):
    from pathlib import Path
    impl = Path(__file__).resolve().parents[1] / "impl"
    monkeypatch.syspath_prepend(str(impl))
    from bank import FixedBank
    from channel import GateWeights, JointPolicy
    from release import CachedJointRelease
    from materialize import materialize_role
    from policy_package import save_policy_package
    from source_bridge import action_map_sha256

    x1 = np.array([[.9, .1], [.7, .3]], np.float64)
    x2 = np.array([[.8, .04, .04, .04, .04, .04],
                   [.7, .06, .06, .06, .06, .06]], np.float64)
    b1 = FixedBank.from_source(x1, np.arange(2, dtype=np.int64),
                               np.zeros(2, np.int64))
    b2 = FixedBank.from_source(x2, np.arange(2, dtype=np.int64),
                               np.zeros(2, np.int64))
    policy = JointPolicy(b1, b2, (tuple(b1.old_to_global),),
                         (tuple(b2.old_to_global),),
                         GateWeights.zeros(input_width=12))
    policy_dir = tmp_path / ".private" / "policy"
    policy_dir.mkdir(parents=True, mode=0o700)
    row = np.array([11, 12], np.int64)
    unit = np.array([21, 22], np.int64)
    row_sha = sha_array(np.column_stack((row, unit)))
    source_sha = "a" * 64
    services = {}
    receipt_shas, stream_shas = {}, {}
    for role_name in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        directory = tmp_path / ".private" / role_name
        directory.mkdir(mode=0o700)
        service = CachedJointRelease(
            policy, cache_path=directory / "cache.sqlite3",
            release_id="A_JOINT_STOCH-s0", source_seed=0,
            role=role_name, realization=0,
            source_binding_sha256=source_sha,
            private_root=b"x" * 32, offline=True)
        receipt = materialize_role(
            service, row_id=row, unit=unit, score1=x1, score2=x2,
            cell1=np.zeros(2, np.int64), cell2=np.zeros(2, np.int64),
            output_dir=directory, expected_row_unit_sha256=row_sha,
            source_identity={name: "b" * 64 for name in (
                "tstar_bank_admission_receipt_sha256", "tstar_role_tokens_file_sha256",
                "reference_file_sha256")})
        receipt_shas[role_name] = hashlib.sha256(
            (directory / "RELEASE_RECEIPT.json").read_bytes()).hexdigest()
        stream_shas[role_name] = receipt["private_stream_root_sha256"]
        services[role_name] = service
    from release import _bank_hash
    policy_record = save_policy_package(
        policy_dir, policy, arm_id="A_JOINT_STOCH",
        release_id="A_JOINT_STOCH-s0", source_seed=0,
        source_binding_sha256=source_sha, row_unit_pair_sha256=row_sha,
        bank_sha256=services["ATTACK_FIT_NEW"].bank_sha256,
        action_sha256=(action_map_sha256(policy.actions1),
                       action_map_sha256(policy.actions2)),
        fit_unit_id="A-FIT-S0", start="identity_biased", update=0,
        fit_receipt_sha256="c" * 64)
    assert policy_record["bank_sha256_by_recipient"] == {
        "r1": _bank_hash(b1), "r2": _bank_hash(b2)}
    identity = {
        "arm_id": "A_JOINT_STOCH", "source_seed": 0, "realization": 0,
        "release_id": "A_JOINT_STOCH-s0",
        "source_binding_sha256": source_sha,
        "mechanism_sha256": policy_record["mechanism_sha256"],
        "canonical_bank_sha256_by_recipient":
            policy_record["bank_sha256_by_recipient"],
        "release_receipt_sha256_by_role": receipt_shas,
        "private_stream_root_sha256_by_role": stream_shas,
        "policy_receipt_sha256": policy_record["receipt_sha256"],
    }
    att, inner, public1, public2, restored = load_fixed_saved_release_pair(
        policy_dir, tmp_path / ".private" / "ATTACK_FIT_NEW",
        tmp_path / ".private" / "INNER_SELECTION", attack_identity=identity)
    assert restored["arm_id"] == "A_JOINT_STOCH"
    assert att.wires1[0]["public_metadata"]["mechanism_sha256"] == \
        inner.wires2[0]["public_metadata"]["mechanism_sha256"]
    assert prepare_sampled_three_views(att, inner, public1, public2).realization == 0


def test_actual_sampled_local_and_pair_views_are_prepared_together():
    views = prepare_sampled_three_views(role("ATTACK_FIT_NEW"),
                                        role("INNER_SELECTION"), bank(1), bank(2))
    assert set(views.prepared) == {"r1", "r2", "pair"}
    assert views.prepared["pair"].att.cell_key.tolist() == ['["0","0"]', '["1","1"]']
    assert views.prepared["r1"].probability_att.tolist() == [
        bank(1).probability[0].tolist(), bank(1).probability[1].tolist()]
    assert views.prepared["pair"].att.x_code_bits.shape == (2, 4)
    assert all(views.prepared[name].att.row_id.tolist() == [11, 12]
               for name in ("r1", "r2", "pair"))


def test_mixed_realizations_or_roles_refused_before_attack_fit():
    att = role("ATTACK_FIT_NEW")
    inner = role("INNER_SELECTION", realization=1)
    with pytest.raises(ValueError, match="same source seed and realization"):
        prepare_sampled_three_views(att, inner, bank(1), bank(2))
    with pytest.raises(ValueError, match="ATTACK_FIT_NEW and INNER_SELECTION"):
        prepare_sampled_three_views(replace(att, role="INNER_SELECTION"),
                                    role("INNER_SELECTION"), bank(1), bank(2))


def test_outer_worker_blocks_before_saved_role_loader_without_a_queue(
        tmp_path, monkeypatch):
    # The test itself may run inside R03's registered synthetic queue unit.
    # Remove only the inherited slot variables to exercise direct-launch refusal.
    for name in ("PCRL_QUEUE_ROOT", "PCRL_QUEUE_UNIT_ID", "PCRL_QUEUE_STAGE",
                 "PCRL_QUEUE_SLOT", "PCRL_QUEUE_SLOT_FD", "PCRL_QUEUE_SEAL_SHA256",
                 "PCRL_QUEUE_SPEC_SHA256"):
        monkeypatch.delenv(name, raising=False)
    lock = tmp_path / "A_LOCK.json"
    lock.write_text("{}\n")
    import hashlib
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="active queue slot environment missing"):
        run_sampled_attack(
            policy_package_dir=tmp_path / "absent_policy",
            att_release_dir=tmp_path / "absent_att",
            inner_release_dir=tmp_path / "absent_inner",
            q_table_dir=None,
            role_manifest_path=tmp_path / "absent_manifest.json",
            role_manifest_sha256="a" * 64,
            original_lock_path=lock, original_lock_sha256=lock_sha,
            a_amendment_path=tmp_path / "absent_amendment.json",
            a_amendment_sha256="b" * 64,
            queue_unit_id="A-NOT-RUNNING",
            private_root=tmp_path / ".private" / "attempt")


def dummy_saved_dirs(tmp_path):
    audit = tmp_path / "r01" / "SOURCE_AUDIT.json"
    audit.parent.mkdir()
    audit.write_text('{"synthetic_audit":true}\n')
    policy = tmp_path / ".private" / "policy"
    att = tmp_path / ".private" / "att"
    inner = tmp_path / ".private" / "inner"
    policy.mkdir(parents=True, mode=0o700)
    att.mkdir(mode=0o700)
    inner.mkdir(mode=0o700)
    for directory, names in (
            (policy, ("POLICY_RECEIPT.json", "policy.npz")),
            (att, ("RELEASE_RECEIPT.json", "release_codes.npz", "cache.sqlite3")),
            (inner, ("RELEASE_RECEIPT.json", "release_codes.npz", "cache.sqlite3"))):
        for name in names:
            (directory / name).write_bytes(b"synthetic placeholder")
    return policy, att, inner


def test_outer_worker_refuses_unregistered_output_before_role_access(tmp_path, monkeypatch):
    from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks import runner

    policy, att_dir, inner_dir = dummy_saved_dirs(tmp_path)
    manifest = tmp_path / "role_manifest.json"
    manifest.write_text("{}\n")
    lock = tmp_path / "lock.json"
    lock.write_text("{}\n")
    stream = tmp_path / "sampled_channel_v1" / "design" / "STREAM_ROOT_HASH_MANIFEST.json"
    stream.parent.mkdir(parents=True)
    stream.write_text("{}\n")
    amendment = tmp_path / "amendment.json"
    amendment.write_text("{}\n")
    monkeypatch.setattr(runner, "assert_active_from_env", lambda **kwargs: {})
    spec = {"outputs": ["/other/ATTACK_ATTEMPT_INDEX.json"]}
    monkeypatch.setattr(runner, "assert_a_attack_admission",
                        lambda **kwargs: ({}, spec))
    with pytest.raises(ValueError, match="output differs"):
        run_sampled_attack(
            policy_package_dir=policy, att_release_dir=att_dir,
            inner_release_dir=inner_dir, q_table_dir=None,
            role_manifest_path=manifest,
            role_manifest_sha256="a" * 64,
            original_lock_path=lock, original_lock_sha256="b" * 64,
            a_amendment_path=amendment, a_amendment_sha256="c" * 64,
            queue_unit_id="A-SYNTHETIC",
            private_root=tmp_path / ".private" / "attempt")
    parent = tmp_path / ".private"
    spec["outputs"] = [str(parent / "attempt" / "ATTACK_ATTEMPT_INDEX.json")]
    parent.chmod(0o755)
    with pytest.raises(ValueError, match="private queue custody"):
        run_sampled_attack(
            policy_package_dir=policy, att_release_dir=att_dir,
            inner_release_dir=inner_dir, q_table_dir=None,
            role_manifest_path=manifest, role_manifest_sha256="a" * 64,
            original_lock_path=lock, original_lock_sha256="b" * 64,
            a_amendment_path=amendment, a_amendment_sha256="c" * 64,
            queue_unit_id="A-SYNTHETIC", private_root=parent / "attempt")
    parent.chmod(0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    with pytest.raises(PermissionError, match="ancestry"):
        run_sampled_attack(
            policy_package_dir=policy, att_release_dir=att_dir,
            inner_release_dir=inner_dir, q_table_dir=None,
            role_manifest_path=manifest, role_manifest_sha256="a" * 64,
            original_lock_path=lock, original_lock_sha256="b" * 64,
            a_amendment_path=amendment, a_amendment_sha256="c" * 64,
            queue_unit_id="A-SYNTHETIC", private_root=alias / "attempt")


def test_outer_worker_uses_actual_wires_and_delays_inner_sex(tmp_path, monkeypatch):
    from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks import runner

    policy, att_dir, inner_dir = dummy_saved_dirs(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    lock = tmp_path / "lock.json"
    lock.write_text("{}\n")
    stream = tmp_path / "sampled_channel_v1" / "design" / "STREAM_ROOT_HASH_MANIFEST.json"
    stream.parent.mkdir(parents=True)
    stream.write_text("{}\n")
    amendment = tmp_path / "amendment.json"
    amendment.write_text("{}\n")
    root = tmp_path / ".private" / "attempt"
    root.parent.mkdir(mode=0o700)
    att = role("ATTACK_FIT_NEW")
    inner = role("INNER_SELECTION")
    pair_sha = sha_array(np.column_stack((att.row_id, att.unit)))
    label_reads = []

    class SyntheticStore:
        def __init__(self, path):
            assert path == manifest.resolve()
            self.manifest = {"roles": {name: {"row_unit_pair_sha256": pair_sha}
                                       for name in ("ATTACK_FIT_NEW", "INNER_SELECTION")}}

        def read(self, role_name, field, procedure):
            assert field == "sex"
            label_reads.append((role_name, procedure))
            if role_name == "INNER_SELECTION":
                assert (root / "common" / "FROZEN_COMPLETE.json").is_file()
            return {"row_id": att.row_id, "unit": att.unit,
                    "sex": np.asarray([0, 1], dtype=np.int64)}

    def fake_common(prepared, sex_att, *, load_inner_sex, private_root, **kwargs):
        assert set(prepared) == {"r1", "r2", "pair"}
        assert label_reads == [("ATTACK_FIT_NEW", "attack")]
        common = root / "common"
        assert private_root == common
        common.mkdir()
        (common / "FROZEN_COMPLETE.json").write_text('{"status":"FROZEN_COMPLETE"}\n')
        load_inner_sex()
        return {"public": {"controls": {
                    name: {"scores": {"status": "PASS"}}
                    for name in ("r1", "r2", "pair")},
                "freeze_receipt_sha256": "f" * 64,
                "scores": {"r1": {"auc": 0.5}, "r2": {"auc": 0.5},
                           "pair": {"auc": 0.5}}}}

    monkeypatch.setattr(runner, "assert_active_from_env", lambda **kwargs: {})
    monkeypatch.setattr(runner, "assert_a_attack_admission",
                        lambda **kwargs: ({"attack_plan": {
                            "reader_ids": ["LR_C1", "MLP_64x32", "HGB_15x200",
                                           "DA_MLP_64x32", "CELL_A1", "CODE_BIT_LR_C1",
                                           "PROB_LR_C1", "PROB_HGB_15x200"],
                            "q_inapplicable_reason": "synthetic no-Q fixture"}}, {
                            "outputs": [str(root / "ATTACK_ATTEMPT_INDEX.json")],
                            "attack_identity": {"arm_id": "A_JOINT_STOCH",
                                                "policy_receipt_sha256": "a" * 64},
                            "q_table_identity": None,
                            "deps": ["A-FIT-S0"]}))
    monkeypatch.setattr(runner, "RoleOnlyStore", SyntheticStore)
    monkeypatch.setattr(runner, "run_three_view_attack", fake_common)
    monkeypatch.setattr(runner, "load_fixed_saved_release_pair",
                        lambda *args, **kwargs: (
                            att, inner, bank(1), bank(2), {"fit_unit_id": "A-FIT-S0"}))
    result = run_sampled_attack(
        policy_package_dir=policy, att_release_dir=att_dir,
        inner_release_dir=inner_dir, q_table_dir=None,
        role_manifest_path=manifest,
        role_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        original_lock_path=lock, original_lock_sha256="a" * 64,
        a_amendment_path=amendment, a_amendment_sha256="b" * 64,
        queue_unit_id="A-SYNTHETIC",
        private_root=root)
    assert label_reads == [("ATTACK_FIT_NEW", "attack"),
                           ("INNER_SELECTION", "screen")]
    assert result["status"] == "COMPLETE"
    index = json.loads((root / "ATTACK_ATTEMPT_INDEX.json").read_text())
    assert index["wire_content_sha256"]["att"]
    assert index["cache_identity"]["att"]["cache_sha256"] == "e" * 64


def test_a_admission_binds_latest_amendment_and_every_saved_source(tmp_path, monkeypatch):
    from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks import runner

    lock = tmp_path / "CAMPAIGN_LOCK.json"
    audit = tmp_path / "r01" / "SOURCE_AUDIT.json"
    audit.parent.mkdir()
    audit.write_text('{"synthetic_audit":true}\n')
    audit_sha = hashlib.sha256(audit.read_bytes()).hexdigest()
    stream = tmp_path / "sampled_channel_v1" / "design" / "STREAM_ROOT_HASH_MANIFEST.json"
    stream.parent.mkdir(parents=True)
    roots = {"0": "1" * 64, "1": "2" * 64, "2": "3" * 64}
    stream.write_text(json.dumps({
        "schema": "scv1-public-stream-root-hashes-v1",
        "release_realization_root_sha256": roots}) + "\n")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    role_manifest_sha = "e" * 64
    lock.write_text(json.dumps({
        "schema": "pcrl-latent-and-sampled-campaign-lock-v1",
        "status": "LOCALLY_SEALED_D_STAGE",
        "source_files_sha256": {
            "source_role_manifest": role_manifest_sha,
            "r01/SOURCE_AUDIT.json": audit_sha}}) + "\n")
    lock_sha = sha(lock)
    implementation = {}
    for relative in (
            "sampled_channel_v1/attacks/runner.py",
            "sampled_channel_v1/attacks/wire.py",
            "sampled_channel_v1/attacks/q_aware.py",
            "sampled_channel_v1/attacks/q_artifact.py",
            "sampled_channel_v1/controls/control_package.py",
            "r02/complete_view.py", "r02/protocol.py", "r03/compute_queue.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic {relative}\n")
        implementation[relative] = sha(path)
    amendment = tmp_path / "A_STAGE_AMENDMENT.json"
    deps = ["A-FIT-S0", "A-RELEASE-S0-ATT-R1", "A-RELEASE-S0-INNER-R1", "A-Q-S0"]
    amendment.write_text(json.dumps({
        "schema": "pcrl-latent-sampled-A-stage-amendment-v1",
        "status": "LOCALLY_SEALED_A_STAGE",
        "original_lock_sha256": lock_sha,
        "implementation_source_sha256": implementation,
        "unit_plan": {
            "A-SYNTHETIC": {"kind": "attack", "output_id": "A-SYNTHETIC",
                            "arm_id": "A_JOINT_STOCH", "seed": 0, "realization": 1,
                            "fit_unit_id": deps[0],
                            "release_unit_ids": {"ATT": deps[1], "INNER": deps[2]},
                            "q_unit_id": deps[3], "deps": deps,
                            "trained_reader_and_probe_fits": 33},
            "A-Q-S0": {"kind": "q_preprocess", "output_id": "A-Q-S0",
                         "arm_id": "A_JOINT_STOCH", "seed": 0,
                         "fit_unit_id": deps[0]}},
        "realizations": {
            "ids": [0, 1, 2],
            "stream_root_hash_manifest_sha256": sha(stream),
            "release_root_sha256_by_realization": roots},
    }))
    amendment_sha = sha(amendment)
    saved = tmp_path / "saved-package.json"
    saved.write_text('{"synthetic":true}\n')
    spec = {"amendment_sha256": amendment_sha,
            "deps": deps,
            "attack_identity": {"arm_id": "A_JOINT_STOCH", "source_seed": 0,
                                "realization": 1,
                                "private_stream_root_sha256_by_role": {
                                    "ATTACK_FIT_NEW": roots["1"],
                                    "INNER_SELECTION": roots["1"]}},
            "sources": [{"path": str(path.resolve()), "sha256": sha(path)}
                        for path in (lock, amendment, stream, saved,
                                     *(tmp_path / name for name in implementation))]}
    queue = tmp_path / "queue"
    queue.mkdir()
    state = {"units": {"A-SYNTHETIC": {"spec": spec}},
             "amendments": [{"path": str(amendment.resolve()),
                             "sha256": amendment_sha, "stages": ["a"]}]}
    (queue / "state.json").write_text(json.dumps(state))
    monkeypatch.setenv("PCRL_QUEUE_ROOT", str(queue))
    monkeypatch.setattr(runner, "assert_active_from_env",
                        lambda **kwargs: {"seal_sha256": lock_sha})
    got, got_spec = assert_a_attack_admission(
        queue_unit_id="A-SYNTHETIC", original_lock_path=lock,
        original_lock_sha256=lock_sha, a_amendment_path=amendment,
        a_amendment_sha256=amendment_sha,
        role_manifest_sha256=role_manifest_sha,
        source_paths=(saved,))
    assert got["status"] == "LOCALLY_SEALED_A_STAGE"
    assert got_spec == spec
    old_amendment = amendment.read_text()
    altered = json.loads(old_amendment)
    altered["implementation_source_sha256"]["sampled_channel_v1/attacks/runner.py"] = "f" * 64
    amendment.write_text(json.dumps(altered))
    altered_sha = sha(amendment)
    spec["amendment_sha256"] = altered_sha
    next(item for item in spec["sources"] if item["path"] == str(amendment.resolve()))[
        "sha256"] = altered_sha
    state["amendments"][-1]["sha256"] = altered_sha
    (queue / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="implementation source differs"):
        assert_a_attack_admission(
            queue_unit_id="A-SYNTHETIC", original_lock_path=lock,
            original_lock_sha256=lock_sha, a_amendment_path=amendment,
            a_amendment_sha256=altered_sha,
            role_manifest_sha256=role_manifest_sha, source_paths=(saved,))
    amendment.write_text(old_amendment)
    spec["amendment_sha256"] = amendment_sha
    next(item for item in spec["sources"] if item["path"] == str(amendment.resolve()))[
        "sha256"] = amendment_sha
    state["amendments"][-1]["sha256"] = amendment_sha
    (queue / "state.json").write_text(json.dumps(state))
    spec["deps"] = deps[:-1]
    (queue / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="sealed arm/seed/draw/Q unit plan"):
        assert_a_attack_admission(
            queue_unit_id="A-SYNTHETIC", original_lock_path=lock,
            original_lock_sha256=lock_sha, a_amendment_path=amendment,
            a_amendment_sha256=amendment_sha,
            role_manifest_sha256=role_manifest_sha,
            source_paths=(saved,))
    spec["deps"] = deps
    (queue / "state.json").write_text(json.dumps(state))
    spec["attack_identity"]["private_stream_root_sha256_by_role"]["INNER_SELECTION"] = roots["0"]
    (queue / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="precommitted realization"):
        assert_a_attack_admission(
            queue_unit_id="A-SYNTHETIC", original_lock_path=lock,
            original_lock_sha256=lock_sha, a_amendment_path=amendment,
            a_amendment_sha256=amendment_sha,
            role_manifest_sha256=role_manifest_sha,
            source_paths=(saved,))
    spec["attack_identity"]["private_stream_root_sha256_by_role"]["INNER_SELECTION"] = roots["1"]
    (queue / "state.json").write_text(json.dumps(state))
    saved.write_text('{"synthetic":false}\n')
    with pytest.raises(ValueError, match="source path/SHA"):
        assert_a_attack_admission(
            queue_unit_id="A-SYNTHETIC", original_lock_path=lock,
            original_lock_sha256=lock_sha, a_amendment_path=amendment,
            a_amendment_sha256=amendment_sha,
            role_manifest_sha256=role_manifest_sha,
            source_paths=(saved,))
    saved.write_text('{"synthetic":true}\n')
    state["amendments"].append({"path": str(tmp_path / "later.json"),
                                "sha256": "c" * 64, "stages": ["a"]})
    (queue / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="latest registered"):
        assert_a_attack_admission(
            queue_unit_id="A-SYNTHETIC", original_lock_path=lock,
            original_lock_sha256=lock_sha, a_amendment_path=amendment,
            a_amendment_sha256=amendment_sha,
            role_manifest_sha256=role_manifest_sha,
            source_paths=(saved,))
    with pytest.raises(ValueError, match="role manifest/source audit"):
        assert_a_attack_admission(
            queue_unit_id="A-SYNTHETIC", original_lock_path=lock,
            original_lock_sha256=lock_sha, a_amendment_path=amendment,
            a_amendment_sha256=amendment_sha,
            role_manifest_sha256="f" * 64,
            source_paths=(saved,))


def test_q_preprocess_dependency_binds_completed_queue_outputs(tmp_path, monkeypatch):
    from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks import runner

    q_dir = tmp_path / ".private" / "q"
    q_dir.mkdir(parents=True)
    receipt = q_dir / "Q_TABLE_RECEIPT.json"
    table = q_dir / "q_table.npz"
    receipt.write_text("synthetic public Q receipt\n")
    table.write_bytes(b"synthetic public Q table")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    q_id = "A-Q-S0"
    queue = tmp_path / "queue"
    (queue / "units" / q_id).mkdir(parents=True)
    outputs = [str(receipt), str(table)]
    state = {"units": {q_id: {"status": "done", "spec_sha256": "a" * 64,
                              "spec": {"outputs": outputs}}}}
    (queue / "state.json").write_text(json.dumps(state))
    queue_receipt = {"status": "done", "returncode": 0,
                     "spec_sha256": "a" * 64,
                     "output_sha256": {str(path): sha(path) for path in (receipt, table)}}
    receipt_path = queue / "units" / q_id / "RECEIPT.json"
    receipt_path.write_text(json.dumps(queue_receipt))
    monkeypatch.setenv("PCRL_QUEUE_ROOT", str(queue))
    spec = {"deps": [q_id]}
    amendment = {"unit_plan": {"A-ATTACK": {"q_unit_id": q_id}}}
    identity = {"receipt_sha256": sha(receipt), "content_sha256": "b" * 64}
    runner._assert_q_unit_custody(q_dir, spec=spec, amendment=amendment,
                                  queue_unit_id="A-ATTACK", q_identity=identity)
    queue_receipt["output_sha256"][str(table)] = "c" * 64
    receipt_path.write_text(json.dumps(queue_receipt))
    with pytest.raises(ValueError, match="dependency receipt/content"):
        runner._assert_q_unit_custody(q_dir, spec=spec, amendment=amendment,
                                      queue_unit_id="A-ATTACK", q_identity=identity)
