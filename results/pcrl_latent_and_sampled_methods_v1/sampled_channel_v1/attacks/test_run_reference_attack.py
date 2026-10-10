"""Data-free gates for deterministic historical ATT/INNER reference attacks."""

import hashlib
import json

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks import (
    run_reference_attack as worker,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.wire import (
    canonical_bank_sha256,
)


def test_reference_output_requires_private_owner_only_ancestry(tmp_path):
    private = tmp_path / ".private"
    private.mkdir(mode=0o700)
    parent = private / "attempts"
    parent.mkdir(mode=0o755)
    output = parent / "reference__T_STAR__s0"
    spec = {"outputs": [str(output / "ATTACK_ATTEMPT_INDEX.json")]}
    with pytest.raises(PermissionError, match="custody"):
        worker._checked_output(output, spec)
    parent.chmod(0o700)
    assert worker._checked_output(output, spec) == output
    wrong = {"outputs": [str(output / "OTHER.json")]}
    with pytest.raises(PermissionError, match="custody"):
        worker._checked_output(output, wrong)


def test_reference_worker_refuses_shadowed_critical_module_before_role_access(
        tmp_path, monkeypatch):
    from results.pcrl_latent_and_sampled_methods_v1.r01 import role_store
    shadow = tmp_path / "shadow.py"
    shadow.write_text("# synthetic shadow\n")
    monkeypatch.setattr(role_store, "__file__", str(shadow))
    campaign = worker.Path(worker.__file__).resolve().parents[2]
    with pytest.raises(PermissionError, match="loaded module"):
        worker._checked_module_origins(campaign, {})


def test_fixed_public_q_mechanism_matches_separate_q_compiler_descriptor():
    # The maps contain no people, frequencies or protected labels.
    bank1 = np.asarray([[.9, .1], [.1, .9]], np.float64)
    bank2 = np.asarray([[.9, .02, .02, .02, .02, .02],
                        [.02, .9, .02, .02, .02, .02]], np.float64)
    fixed = {"bank_r1": bank1, "bank_r2": bank2,
             "fine_to_code_r1": np.arange(64, dtype=np.int64) % 2,
             "fine_to_code_r2": np.arange(641, dtype=np.int64) % 2}
    receipt = {"arm_id": "T_STAR", "seed": 0, "release_id": "historical-tstar",
               "source_binding_sha256": "a" * 64,
               "original_bank_receipt_sha256": "b" * 64}
    mechanism, descriptor_sha = worker._fixed_q_mechanism(fixed, receipt)
    base = {
        "source_kind": "historical_finite_reference", "arm_id": "T_STAR",
        "seed": 0, "release_id": "historical-tstar",
        "source_binding_sha256": "a" * 64,
        "source_receipt_sha256": "b" * 64,
        "canonical_bank_sha256_by_recipient": {
            "r1": canonical_bank_sha256(bank1),
            "r2": canonical_bank_sha256(bank2)},
        "full_bank_array_sha256_by_recipient": {
            "r1": sha_array(bank1), "r2": sha_array(bank2)},
        "fine_to_code_sha256_by_recipient": {
            "r1": sha_array(fixed["fine_to_code_r1"]),
            "r2": sha_array(fixed["fine_to_code_r2"])},
    }
    canonical = lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    expected = hashlib.sha256(b"SCV1-fixed-public-Q-law-v1\n" + canonical(base)).hexdigest()
    assert mechanism == expected
    assert descriptor_sha == hashlib.sha256(
        canonical({**base, "mechanism_sha256": mechanism})).hexdigest()


def test_reference_worker_requires_active_queue_before_source_or_label_read(monkeypatch):
    def no_slot(**kwargs):
        raise PermissionError("synthetic active slot absent")
    monkeypatch.setattr(worker, "assert_active_from_env", no_slot)
    with pytest.raises(PermissionError, match="active slot absent"):
        worker.admit_reference_attack(
            unit_id="A-REFERENCE-ATTACK-T_STAR-S0",
            original_lock_path="unread", original_lock_sha256="a" * 64,
            a_amendment_path="unread", a_amendment_sha256="b" * 64,
            role_manifest_path="unread", role_manifest_sha256="c" * 64,
            att_release_dir="unread", inner_release_dir="unread",
            fixed_map_receipt="unread", fixed_map_npz="unread",
            q_table_dir="unread", private_root="unread")


def test_reference_gate_refuses_npz_not_pinned_by_effective_amendment(
        tmp_path, monkeypatch):
    """Queue and live NPZ can agree while the effective science lock disagrees."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    manifest = tmp_path / "role_manifest.json"
    manifest.write_text("{}\n")
    manifest_sha = worker._sha(manifest)
    original = campaign / "CAMPAIGN_LOCK.json"
    original.write_text(json.dumps({
        "schema": "pcrl-latent-and-sampled-campaign-lock-v1",
        "status": "LOCALLY_SEALED_D_STAGE",
        "source_files_sha256": {
            "source_role_manifest": manifest_sha,
            "r01/SOURCE_AUDIT.json": "a" * 64}}))
    original_sha = worker._sha(original)
    unit_id = "A-REFERENCE-ATTACK-T_STAR-S0"
    deps = ["A-REFERENCE-RELEASE-T_STAR-S0-ATT",
            "A-REFERENCE-RELEASE-T_STAR-S0-INNER",
            "A-Q-PREPROCESS-T_STAR-S0"]
    implementation = {
        name: "e" * 64 for name in (
            "sampled_channel_v1/attacks/run_reference_attack.py",
            "sampled_channel_v1/attacks/reference_wire.py",
            "sampled_channel_v1/attacks/q_aware.py", "r02/protocol.py",
            "r03/compute_queue.py")}
    for name in implementation:
        path = campaign / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source\n")
    fixed_receipt = campaign / "r01/A_FIXED_MAP_T_STAR_S0.json"
    fixed_archive = campaign / "r01/A_FIXED_MAP_T_STAR_S0.npz"
    fixed_receipt.write_text("{}\n")
    fixed_archive.write_bytes(b"synthetic public map")
    amendment = campaign / "A_STAGE_AMENDMENT.json"
    amendment.write_text(json.dumps({
        "schema": "pcrl-latent-sampled-A-stage-amendment-v1",
        "status": "LOCALLY_SEALED_A_STAGE",
        "original_lock_sha256": original_sha,
        "implementation_source_sha256": implementation,
        "source_sha256": {
            "r01/SOURCE_AUDIT.json": "a" * 64,
            "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json": "b" * 64,
            "r01/A_FIXED_MAP_T_STAR_S0.json": "1" * 64,
            "r01/A_FIXED_MAP_T_STAR_S0.npz": "3" * 64},
        "attack_plan": {
            "reader_ids": list(worker.READER_IDS),
            "reference_ids": ["T_STAR", "P_STAR", "U0", "Ucal",
                              "D0_LOCAL", "D0_SEQ12", "D0_SEQ21"]},
        "unit_plan": {unit_id: {
            "kind": "reference_attack", "output_id": unit_id,
            "reference_id": "T_STAR", "seed": 0, "realization": 0,
            "trained_reader_and_probe_fits": 33,
            "release_unit_ids": {"ATT": deps[0], "INNER": deps[1]},
            "q_unit_id": deps[2], "deps": deps}}}))
    amendment_sha = worker._sha(amendment)
    output_base = tmp_path / ".private"
    output_base.mkdir(mode=0o700)
    output = output_base / "reference_attempt"
    spec = {"amendment_sha256": amendment_sha, "stage": "a",
            "mode": "real_data", "deps": deps,
            "outputs": [str(output / "ATTACK_ATTEMPT_INDEX.json")],
            "sources": [],
            "reference_identity": {
                "reference_id": "T_STAR", "source_seed": 0,
                "release_id": "historical-tstar", "interface_kind": "finite",
                "source_binding_sha256": "f" * 64,
                "release_receipt_sha256_by_role": {
                    "ATTACK_FIT_NEW": "b" * 64,
                    "INNER_SELECTION": "c" * 64},
                "fixed_map_identity": {"receipt_sha256": "1" * 64,
                                       "npz_sha256": "2" * 64},
                "q_table_identity": {"receipt_sha256": "4" * 64,
                                     "content_sha256": "5" * 64}}}
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "state.json").write_text(json.dumps({
        "amendments": [{"stages": ["a"], "sha256": amendment_sha,
                        "path": str(amendment)}],
        "units": {unit_id: {"spec": spec}}}))
    monkeypatch.setenv("PCRL_QUEUE_ROOT", str(queue))
    monkeypatch.setattr(worker, "assert_active_from_env",
                        lambda **kwargs: {"seal_sha256": original_sha})
    # Unrelated source/dependency gates are covered elsewhere; this fixture
    # isolates the immutable amendment's NPZ pin before any SEX label access.
    monkeypatch.setattr(worker, "_checked_source", lambda *args: None)
    monkeypatch.setattr(worker, "_checked_module_origins", lambda *args: None)
    monkeypatch.setattr(worker, "_completed_dependency", lambda *args: None)
    role_dirs = []
    for role in ("ATT", "INNER"):
        directory = tmp_path / role
        directory.mkdir()
        (directory / "REFERENCE_RELEASE_RECEIPT.json").write_text("{}\n")
        (directory / "reference_release.npz").write_bytes(b"synthetic")
        role_dirs.append(directory)
    with pytest.raises(PermissionError, match="fixed-map source pin"):
        worker.admit_reference_attack(
            unit_id=unit_id, original_lock_path=original,
            original_lock_sha256=original_sha,
            a_amendment_path=amendment, a_amendment_sha256=amendment_sha,
            role_manifest_path=manifest, role_manifest_sha256=manifest_sha,
            att_release_dir=role_dirs[0], inner_release_dir=role_dirs[1],
            fixed_map_receipt=fixed_receipt, fixed_map_npz=fixed_archive,
            q_table_dir=tmp_path / "never_opened_q",
            private_root=output)
