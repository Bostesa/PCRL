"""The repair queue must reject a missing or malformed study seal."""

import json
import hashlib
import subprocess
from pathlib import Path

import resource

from results.pcrl_utility_reference_repair_v1.r0.queue_adapter import RepairQueue


def test_missing_seal_fails_closed(tmp_path):
    assert not RepairQueue._seal_ok(tmp_path / "missing.json", "0" * 64)


def test_wrong_hash_fails_before_lock_access(tmp_path):
    seal = tmp_path / "LOCAL_SEAL_RECEIPT.json"
    seal.write_text(json.dumps({"schema": "pcrl-utility-local-seal-v1", "status": "PASS"}))
    assert not RepairQueue._seal_ok(seal, "0" * 64)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_positive_seal_and_contained_worker(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    study = repo / "results" / "pcrl_utility_reference_repair_v1"
    (study / "m1").mkdir(parents=True)
    (study / "r3").mkdir()
    (study / "r1").mkdir()
    (study / ".private").mkdir()
    manifest = study / "m1" / "DATA_ACCESS_MANIFEST.json"
    registry = study / ".private" / "UNIT_SPECS.json"
    source = study / "r1" / "worker.py"
    protocol = study / "EXECUTION_PLAN.md"
    manifest.write_text('{"schema":"test"}\n')
    registry.write_text('{"unit_specs":{}}\n')
    source.write_text("pass\n")
    protocol.write_text("# Test protocol\n")
    lock = study / "CAMPAIGN_LOCK.json"
    lock.write_text(json.dumps({
        "schema": "pcrl-utility-campaign-lock-v1",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True,
        "verification_only": False,
        "private_unit_registry_sha256": _sha(registry),
        "source_map_sha256": _sha(manifest),
        "governed_source_sha256_by_path": {
            "results/pcrl_utility_reference_repair_v1/r1/worker.py": _sha(source)},
        "governed_protocol_sha256_by_relative_path": {
            "EXECUTION_PLAN.md": _sha(protocol)},
    }, sort_keys=True) + "\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "results"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seal"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    review = study / "r3" / "PREFIT_REVIEW.json"
    review.write_text(json.dumps({
        "status": "PASS", "lock_sha256": _sha(lock), "lock_commit_sha": commit,
    }) + "\n")
    seal = study / "LOCAL_SEAL_RECEIPT.json"
    seal.write_text(json.dumps({
        "schema": "pcrl-utility-local-seal-v1", "status": "PASS",
        "lock_sha256": _sha(lock), "lock_commit_sha": commit,
        "r3_review_sha256": _sha(review),
    }) + "\n")
    assert RepairQueue._seal_ok(seal, _sha(seal))
    monkeypatch.setattr(resource, "getrlimit", lambda kind: (0, 0))
    assert RepairQueue._seal_ok(seal, _sha(seal), contained_worker=True)
    source.write_text("changed\n")
    assert not RepairQueue._seal_ok(seal, _sha(seal))
