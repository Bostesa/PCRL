"""Data-free custody contract for a restored D deployment package."""

from __future__ import annotations

import hashlib
from pathlib import Path
import stat

import pytest

from results.pcrl_latent_and_sampled_methods_v1.r18.run_restored_d_replay import (
    materialize_verified_restore,
)

try:
    from results.pcrl_latent_and_sampled_methods_v1.r18.run_restored_d_replay import (
        validate_completed_parents,
    )
except ImportError:
    validate_completed_parents = None


ARCHIVE_NAMES = (
    "d_first_public_receipt.json",
    "d_one_row_result.json",
    "d_service_manifest.json",
    "d_service_weights.npz",
    "role_admission_manifest.json",
    "u0_source_model.pt",
)


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    restored = tmp_path / "restore"
    restored.mkdir(mode=0o700)
    expected = {}
    for name in ARCHIVE_NAMES:
        payload = ("invented:" + name).encode()
        target = restored / name
        target.write_bytes(payload)
        target.chmod(0o600)
        expected[name] = hashlib.sha256(payload).hexdigest()
    return restored, expected


def test_restored_package_uses_verified_bytes_and_canonical_names(tmp_path: Path) -> None:
    restored, expected = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    actual = materialize_verified_restore(restored, scratch, expected)
    assert actual == {
        "d_first_public_receipt.json": scratch / "d_first_public_receipt.json",
        "d_one_row_result.json": scratch / "d_one_row_result.json",
        "d_service_manifest.json": scratch / "service" / "manifest.json",
        "d_service_weights.npz": scratch / "service" / "weights.npz",
        "role_admission_manifest.json": scratch / "role_admission_manifest.json",
        "u0_source_model.pt": scratch / "rel__s0__U" / "model.pt",
    }
    assert stat.S_IMODE(scratch.stat().st_mode) == 0o700
    assert stat.S_IMODE((scratch / "service").stat().st_mode) == 0o700
    assert stat.S_IMODE((scratch / "rel__s0__U").stat().st_mode) == 0o700
    for archive_name, copied in actual.items():
        assert copied.read_bytes() == (restored / archive_name).read_bytes()
        assert stat.S_IMODE(copied.stat().st_mode) == 0o600


def test_changed_snapshot_member_refuses_without_scratch(tmp_path: Path) -> None:
    restored, expected = _fixture(tmp_path)
    (restored / "d_service_weights.npz").write_bytes(b"changed")
    scratch = tmp_path / "scratch"
    with pytest.raises((PermissionError, ValueError), match="snapshot|hash|SHA"):
        materialize_verified_restore(restored, scratch, expected)
    assert not scratch.exists()


def test_existing_scratch_refuses_without_overwrite(tmp_path: Path) -> None:
    restored, expected = _fixture(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    sentinel = scratch / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises((FileExistsError, PermissionError)):
        materialize_verified_restore(restored, scratch, expected)
    assert sentinel.read_text() == "keep"


def test_symlinked_snapshot_ancestor_refuses(tmp_path: Path) -> None:
    (tmp_path / "actual").mkdir(mode=0o700)
    restored, expected = _fixture(tmp_path / "actual")
    (tmp_path / "alias").symlink_to(tmp_path / "actual", target_is_directory=True)
    scratch = tmp_path / "scratch"
    with pytest.raises(PermissionError, match="symlink"):
        materialize_verified_restore(tmp_path / "alias" / "restore", scratch, expected)
    assert not scratch.exists()


def _parent_fixture(tmp_path: Path) -> tuple[dict, dict, dict]:
    first_path = str(tmp_path / "D_FIRST_S0_R1_RECEIPT.json")
    result_path = str(tmp_path / "R18-D-DEPLOY-REAL-S0.json")
    hashes = {"d_first_public_receipt.json": "a" * 64,
              "d_one_row_result.json": "b" * 64}
    state = {"units": {
        "D_FIRST_T_DK_S0_R1": {"status": "done", "returncode": 0,
                               "spec": {"outputs": [first_path]},
                               "output_sha256": {first_path: hashes["d_first_public_receipt.json"]}},
        "R18-D-DEPLOY-REAL-S0": {"status": "done", "returncode": 0,
                                  "spec": {"outputs": [result_path]},
                                  "output_sha256": {result_path: hashes["d_one_row_result.json"]}},
    }}
    spec = {"deps": ["D_FIRST_T_DK_S0_R1", "R18-D-DEPLOY-REAL-S0"]}
    return state, spec, hashes


def test_completed_parent_outputs_need_exact_ids_paths_and_rc0(tmp_path: Path) -> None:
    state, spec, hashes = _parent_fixture(tmp_path)
    validate_completed_parents(state, spec, hashes,
                               first_receipt_path=tmp_path / "D_FIRST_S0_R1_RECEIPT.json",
                               one_row_path=tmp_path / "R18-D-DEPLOY-REAL-S0.json")
    state["units"]["R18-D-DEPLOY-REAL-S0"]["returncode"] = 1
    with pytest.raises(PermissionError, match="parent"):
        validate_completed_parents(state, spec, hashes,
                                   first_receipt_path=tmp_path / "D_FIRST_S0_R1_RECEIPT.json",
                                   one_row_path=tmp_path / "R18-D-DEPLOY-REAL-S0.json")
    state["units"]["R18-D-DEPLOY-REAL-S0"]["returncode"] = 0
    spec["deps"].append("unrelated")
    with pytest.raises(PermissionError, match="parent"):
        validate_completed_parents(state, spec, hashes,
                                   first_receipt_path=tmp_path / "D_FIRST_S0_R1_RECEIPT.json",
                                   one_row_path=tmp_path / "R18-D-DEPLOY-REAL-S0.json")
    spec["deps"].pop()
    state["units"]["R18-D-DEPLOY-REAL-S0"]["spec"]["outputs"] = [str(tmp_path / "wrong.json")]
    with pytest.raises(PermissionError, match="parent"):
        validate_completed_parents(state, spec, hashes,
                                   first_receipt_path=tmp_path / "D_FIRST_S0_R1_RECEIPT.json",
                                   one_row_path=tmp_path / "R18-D-DEPLOY-REAL-S0.json")
