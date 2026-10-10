"""Data-free custody checks for versioned R18 byte snapshots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from results.pcrl_latent_and_sampled_methods_v1.r18.local_snapshot import copy_and_restore


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    private = tmp_path / ".private"
    private.mkdir(mode=0o700)
    source = private / "source.bin"
    source.write_bytes(b"invented fixed bank bytes\x00\x01")
    source.chmod(0o600)
    return {"bank.bin": {"source_path": str(source), "sha256": _sha(source)}}, private


def test_snapshot_and_fresh_restore_are_exact_and_owner_only(tmp_path: Path) -> None:
    entries, private = _fixture(tmp_path)
    result = copy_and_restore(entries, private / "v1", required_names={"bank.bin"})
    assert result["status"] == "LOCAL_UNCACHED_SNAPSHOT_AND_RESTORE_PASS"
    assert result["uncached_hash_read"] is True
    assert result["members"]["bank.bin"]["sha256"] == entries["bank.bin"]["sha256"]
    for part in ("snapshot", "restore"):
        file = private / "v1" / part / "bank.bin"
        assert _sha(file) == entries["bank.bin"]["sha256"]
        assert file.stat().st_mode & 0o777 == 0o600
    assert (private / "v1" / "snapshot" / "MANIFEST.json").exists()
    assert (private / "v1" / "RESTORE_PROOF.json").exists()


def test_wrong_source_hash_refuses_before_version_created(tmp_path: Path) -> None:
    entries, private = _fixture(tmp_path)
    entries["bank.bin"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        copy_and_restore(entries, private / "v1", required_names={"bank.bin"})
    assert not (private / "v1").exists()


def test_symlink_source_and_existing_destination_refuse(tmp_path: Path) -> None:
    entries, private = _fixture(tmp_path)
    link = private / "link.bin"
    link.symlink_to(entries["bank.bin"]["source_path"])
    entries["bank.bin"]["source_path"] = str(link)
    with pytest.raises(PermissionError, match="symlink"):
        copy_and_restore(entries, private / "v1", required_names={"bank.bin"})
    assert not (private / "v1").exists()
    entries["bank.bin"]["source_path"] = str(private / "source.bin")
    (private / "v1").mkdir(mode=0o700)
    with pytest.raises(FileExistsError):
        copy_and_restore(entries, private / "v1", required_names={"bank.bin"})


def test_inventory_names_and_private_modes_fail_closed(tmp_path: Path) -> None:
    entries, private = _fixture(tmp_path)
    with pytest.raises(ValueError, match="inventory names"):
        copy_and_restore(entries, private / "v1", required_names={"another.bin"})
    source = Path(entries["bank.bin"]["source_path"])
    source.chmod(0o644)
    with pytest.raises(PermissionError, match="owner-only"):
        copy_and_restore(entries, private / "v1", required_names={"bank.bin"})
    assert not (private / "v1").exists()
