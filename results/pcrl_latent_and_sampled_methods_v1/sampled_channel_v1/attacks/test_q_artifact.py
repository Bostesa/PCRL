"""Data-free write-once custody for the public-prototype Q table."""

import hashlib

import pytest

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.q_artifact import (
    load_q_table_artifact, save_q_table_artifact,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.test_q_aware import table


def test_q_table_roundtrip_and_changed_bytes_refused(tmp_path):
    root = tmp_path / ".private" / "q"
    root.mkdir(parents=True, mode=0o700)
    original = table()
    receipt = save_q_table_artifact(original, root, policy_receipt_sha256="a" * 64)
    restored = load_q_table_artifact(
        root, expected_receipt_sha256=receipt["receipt_sha256"],
        expected_content_sha256=original.table_sha256,
        expected_policy_receipt_sha256="a" * 64)
    assert restored.table_sha256 == original.table_sha256
    assert restored.identity == original.identity
    with pytest.raises(FileExistsError, match="write-once"):
        save_q_table_artifact(original, root, policy_receipt_sha256="a" * 64)
    with pytest.raises(ValueError, match="source policy"):
        load_q_table_artifact(
            root, expected_receipt_sha256=receipt["receipt_sha256"],
            expected_content_sha256=original.table_sha256,
            expected_policy_receipt_sha256="b" * 64)
    data = root / "q_table.npz"
    with data.open("ab") as output:
        output.write(b"planted extra bytes")
    assert hashlib.sha256(data.read_bytes()).hexdigest() != receipt["q_table_file_sha256"]
    with pytest.raises(ValueError, match="file SHA"):
        load_q_table_artifact(
            root, expected_receipt_sha256=receipt["receipt_sha256"],
            expected_content_sha256=original.table_sha256,
            expected_policy_receipt_sha256="a" * 64)
