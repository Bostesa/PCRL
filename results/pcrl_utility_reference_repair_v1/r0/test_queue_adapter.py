"""The repair queue must reject a missing or malformed study seal."""

import json

from results.pcrl_utility_reference_repair_v1.r0.queue_adapter import RepairQueue


def test_missing_seal_fails_closed(tmp_path):
    assert not RepairQueue._seal_ok(tmp_path / "missing.json", "0" * 64)


def test_wrong_hash_fails_before_lock_access(tmp_path):
    seal = tmp_path / "LOCAL_SEAL_RECEIPT.json"
    seal.write_text(json.dumps({"schema": "pcrl-utility-local-seal-v1", "status": "PASS"}))
    assert not RepairQueue._seal_ok(seal, "0" * 64)
