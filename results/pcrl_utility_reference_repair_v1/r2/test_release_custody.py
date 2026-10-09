"""Data-free, byte-level checks for conditional complete release custody."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r2.release_custody import (
    restore_complete_release, write_complete_release,
)
from results.pcrl_utility_reference_repair_v1.r2.test_complete_views import fixture


@dataclass(frozen=True)
class Mask:
    mask_id: str
    mask_sha256: str
    atoms: tuple[str, ...]


class Bank:
    bank_sha256 = "f" * 64

    def mask(self, mask_id):
        return {"MASK-I": Mask("MASK-I", "a" * 64,
                               ("age", "relationship")),
                "MASK-O": Mask("MASK-O", "c" * 64,
                               ("capital-gain",))}[mask_id]


@dataclass(frozen=True)
class Pair:
    pair_id: str = "MASK-I__MASK-O"
    income_mask_id: str = "MASK-I"
    occupation_mask_id: str = "MASK-O"


def inputs():
    source, public, _ = fixture()
    inner = {key: value.copy() if isinstance(value, np.ndarray) else value
             for key, value in source.items()}
    inner["row_id"] += 1000
    inner["unit"] += 1000
    return {"ATTACK_FIT_NEW": source, "INNER_SELECTION": inner}, public


def write(tmp_path):
    roles, public = inputs()
    root = tmp_path / ".private"
    root.mkdir()
    output = root / "releases" / "seed0" / Pair.pair_id
    hashes = {role: "1" * 64 for role in roles}
    receipt = write_complete_release(
        roles, public, Pair(), Bank(), output, private_root=root,
        seed=0, lock_sha256="2" * 64, gate_sha256="3" * 64,
        manifest_sha256="4" * 64, role_feature_sha256=hashes)
    return root, output, receipt, roles, public, hashes


def restore(root, output, receipt, roles, public, hashes):
    return restore_complete_release(
        output / "RELEASE_RECEIPT.json", expected_index_sha256=receipt,
        source_roles=roles, public_releases=public, pair=Pair(), bank=Bank(),
        private_root=root, seed=0, lock_sha256="2" * 64,
        gate_sha256="3" * 64, manifest_sha256="4" * 64,
        role_feature_sha256=hashes)


def test_write_restore_exact_all_three_views_both_roles(tmp_path):
    args = write(tmp_path)
    rebuilt = restore(*args)
    assert set(rebuilt) == {"ATTACK_FIT_NEW", "INNER_SELECTION"}
    assert set(rebuilt["INNER_SELECTION"]["views"]) == {
        "income", "occupation_group", "coalition"}
    schema = args[4]["income"].release.schema
    assert rebuilt["ATTACK_FIT_NEW"]["union"].source_columns == \
        schema.column_indices(("age", "capital-gain", "relationship"))
    assert (args[1] / "ATTACK_FIT_NEW.npz").exists()
    assert (args[1] / "INNER_SELECTION.npz").exists()


def test_restore_rejects_changed_file_and_public_identity(tmp_path):
    args = write(tmp_path)
    root, output, receipt, roles, public, hashes = args
    public["income"] = public["income"].__class__(
        public["income"].release, "OTHER", "a" * 64, "b" * 64)
    with pytest.raises(PermissionError, match="mask"):
        restore(*args)
    public["income"] = inputs()[1]["income"]
    path = output / "ATTACK_FIT_NEW.npz"
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(PermissionError, match="bytes"):
        restore(*args)


def test_restore_rejects_changed_source_values(tmp_path):
    args = write(tmp_path)
    root, output, receipt, roles, public, hashes = args
    changed = dict(roles)
    att = dict(roles["ATTACK_FIT_NEW"])
    att["X"] = att["X"].copy()
    att["X"][0, 0] = 77
    changed["ATTACK_FIT_NEW"] = att
    with pytest.raises(PermissionError, match="source|view"):
        restore_complete_release(
            output / "RELEASE_RECEIPT.json", expected_index_sha256=receipt,
            source_roles=changed, public_releases=public, pair=Pair(), bank=Bank(),
            private_root=root, seed=0, lock_sha256="2" * 64,
            gate_sha256="3" * 64, manifest_sha256="4" * 64,
            role_feature_sha256=hashes)


def test_restore_rejects_builder_source_identity_change(tmp_path):
    args = write(tmp_path)
    root, output, _, roles, public, hashes = args
    path = output / "RELEASE_RECEIPT.json"
    receipt = json.loads(path.read_text())
    receipt["builder_source_sha256"]["release_custody.py"] = "0" * 64
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    altered_index_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(PermissionError, match="builder"):
        restore(root, output, altered_index_sha, roles, public, hashes)
