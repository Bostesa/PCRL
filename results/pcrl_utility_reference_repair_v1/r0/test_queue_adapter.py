"""The repair queue must reject a missing or malformed study seal."""

import json
import hashlib
import subprocess
import sys
import datetime as dt
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


def test_real_stage_admission_enqueues_and_runs(tmp_path):
    """The actual queue contract admits a role-bound, sealed worker end to end."""
    repo = tmp_path / "repo"
    study = repo / "results" / "pcrl_utility_reference_repair_v1"
    for part in ("m1", "r1", "r3", ".private"):
        (study / part).mkdir(parents=True, exist_ok=True)
    role = study / ".private" / "role.bin"
    role.write_bytes(b"synthetic permitted role\n")
    manifest = study / "m1" / "DATA_ACCESS_MANIFEST.json"
    role_key = "OSF_DEFENSE_FIT:features"
    manifest.write_text(json.dumps({
        "role_file_relative_path": {role_key: ".private/role.bin"},
        "role_file_sha256": {role_key: _sha(role)},
    }, sort_keys=True) + "\n")
    worker = study / "r1" / "worker.py"
    actual_repo = Path(__file__).resolve().parents[3]
    worker.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(actual_repo)!r})\n"
        "from results.pcrl_utility_reference_repair_v1.r0.admission "
        "import assert_running_unit\n"
        "u,q,l,o,r = sys.argv[1:]\n"
        "spec = assert_running_unit(u,q,l,'reference_parity',o)\n"
        "assert r in spec['inputs']\n"
        "Path(o).write_bytes(Path(r).read_bytes())\n")
    registry = study / ".private" / "UNIT_SPECS.json"
    protocol = study / "EXECUTION_PLAN.md"
    protocol.write_text("# Synthetic prefit fixture\n")
    slot = tmp_path / "shared-slot.lock"
    slot.touch()
    queue_root = study / ".private" / "queue"
    output = study / ".private" / "out.bin"
    lock_path = study / "CAMPAIGN_LOCK.json"
    source_map_sha = _sha(manifest)
    spec = {
        "unit_id": "REFERENCE_PARITY_FIXTURE", "mode": "real",
        "stage": "reference_parity", "script": str(worker),
        "script_sha256": _sha(worker), "cwd": str(repo),
        "argv": [sys.executable, str(worker), "REFERENCE_PARITY_FIXTURE",
                 str(queue_root), str(lock_path), str(output), str(role)],
        "cpu_reserve_s": 60, "wall_limit_s": 30, "max_rss_gib": 1,
        "inputs": {str(role): _sha(role)}, "outputs": [str(output)],
        "deps": [], "source_map_sha256": source_map_sha,
        "role_access": ["OSF_DEFENSE_FIT"],
        "role_inputs": {role_key: {"path": str(role), "sha256": _sha(role)}},
        "allow_child_processes": False, "data_access": "ROLE_ONLY",
        "seal_sha256": "$LOCAL_SEAL_SHA256",
    }
    registry.write_text(json.dumps({"unit_specs": {spec["unit_id"]: spec}},
                                   sort_keys=True) + "\n")
    lock_path.write_text(json.dumps({
        "schema": "pcrl-utility-campaign-lock-v1",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "adult_fit_permission": True, "verification_only": False,
        "private_unit_registry_sha256": _sha(registry),
        "source_map_sha256": source_map_sha,
        "prior_package_relative_to_worktree":
            "results/pcrl_utility_reference_repair_v1",
        "allowed_roles": ["OSF_DEFENSE_FIT"],
        "governed_source_sha256_by_path": {
            "results/pcrl_utility_reference_repair_v1/r1/worker.py": _sha(worker)},
        "governed_protocol_sha256_by_relative_path": {
            "EXECUTION_PLAN.md": _sha(protocol)},
    }, sort_keys=True) + "\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "results"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seal"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    review = study / "r3" / "PREFIT_REVIEW.json"
    review.write_text(json.dumps({
        "status": "PASS", "lock_sha256": _sha(lock_path),
        "lock_commit_sha": commit,
    }) + "\n")
    seal = study / "LOCAL_SEAL_RECEIPT.json"
    seal.write_text(json.dumps({
        "schema": "pcrl-utility-local-seal-v1", "status": "PASS",
        "lock_sha256": _sha(lock_path), "lock_commit_sha": commit,
        "r3_review_sha256": _sha(review),
    }) + "\n")
    queue = RepairQueue(queue_root, slot,
        epoch_start_utc=(dt.datetime.now(dt.timezone.utc) -
                         dt.timedelta(seconds=1)).isoformat())
    queue.admit_science(seal_path=seal, seal_sha256=_sha(seal),
                        external_heavy=1, evidence="synthetic fixture slot reservation")
    spec["seal_sha256"] = _sha(seal)
    queue.enqueue(spec)
    outcome = queue.run_once(expected_unit_id="REFERENCE_PARITY_FIXTURE")
    assert outcome["status"] == "DONE", (outcome, (queue_root / "REFERENCE_PARITY_FIXTURE.stderr").read_text())
    assert output.read_bytes() == role.read_bytes()
