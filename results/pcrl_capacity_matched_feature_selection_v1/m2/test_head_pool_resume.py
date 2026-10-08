"""Data-free refusal checks for an interrupted or altered head cache."""

import json

import pytest

from head_pool import _check_existing_cache


def test_checkpoint_without_immutable_receipt_is_rejected(tmp_path):
    (tmp_path / "head.npz").write_bytes(b"old checkpoint")
    with pytest.raises(PermissionError, match="without immutable receipt"):
        _check_existing_cache(tmp_path, cache_key="a" * 64)


def test_existing_receipt_must_bind_checkpoint_and_cache_key(tmp_path):
    head = tmp_path / "head.npz"
    head.write_bytes(b"checkpoint")
    receipt = {
        "schema": "pcrl-focused-head-pool-receipt-v1",
        "cache_identity": {"cache_key": "a" * 64},
        "head_sha256": "0" * 64,
        "calibrator_sha256": "1" * 64,
        "package_sha256": "2" * 64,
    }
    (tmp_path / "HEAD_POOL_RECEIPT.json").write_text(json.dumps(receipt))
    with pytest.raises(PermissionError, match="checkpoint hash"):
        _check_existing_cache(tmp_path, cache_key="a" * 64)
