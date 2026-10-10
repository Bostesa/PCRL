"""Data-free predecessor FIT and downstream sidecar checks."""

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from .downstream_sources import (_unit_plan_sha, verify_downstream_sidecar,
                                 verify_downstream_registration,
                                 verify_predecessor_fit, SHARED, STAGE_SCOPE)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _fixture(tmp_path):
    campaign = tmp_path / "results/pcrl_latent_and_sampled_methods_v1"
    shared = tmp_path / "results/pcrl_method_portfolio_v1/shared"
    shared.mkdir(parents=True)
    lock = _json(campaign / "CAMPAIGN_LOCK.json", {"status": "sealed"})
    old = campaign / "old_worker.py"
    old.write_text("# unchanged producer source\n")
    new_names = frozenset({
        "sampled_channel_v1/impl/run_release_guarded.py",
        "sampled_channel_v1/impl/release_specs.py",
        "sampled_channel_v1/impl/downstream_sources.py",
    })
    new = {}
    for relative in new_names:
        path = campaign / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic {relative}\n")
        new[relative] = _sha(path)
    inherited = {}
    for name in SHARED:
        path = shared / name
        path.write_text(f"# frozen {name}\n")
        inherited[name] = _sha(path)
    plan = {"A-FIT-0": {"kind": "fit", "arm_id": "A_JOINT_STOCH", "seed": 0}}
    producer = _json(campaign / "PRODUCER.json", {
        "unit_plan": plan,
        "implementation_source_sha256": {"old_worker.py": _sha(old)},
    })
    sidecar = _json(campaign / "A_DOWNSTREAM_SOURCE_MANIFEST_20261010_001.json", {
        "schema": "pcrl-a-downstream-source-manifest-v1",
        "status": "LOCALLY_SEALED_DOWNSTREAM",
        "dated_utc": "2026-10-10T12:00:00Z",
        "producer_amendment_sha256": _sha(producer),
        "original_lock_sha256": _sha(lock),
        "unit_plan_sha256": _unit_plan_sha(plan),
        "new_source_sha256": new,
        "attack_plan": {"inherited_shared_source_sha256": inherited},
        "stage_scope": STAGE_SCOPE,
        "scientific_scope_unchanged": True,
    })
    paths = [lock, producer, sidecar, old]
    paths += [campaign / name for name in new]
    paths += [shared / name for name in inherited]
    spec = {
        "amendment_sha256": _sha(producer),
        "downstream_manifest_path": str(sidecar),
        "downstream_manifest_sha256": _sha(sidecar),
        "sources": [{"path": str(path), "sha256": _sha(path)} for path in paths],
    }
    kwargs = dict(sidecar_path=sidecar, sidecar_sha256=_sha(sidecar),
                  producer_path=producer, producer_sha256=_sha(producer),
                  original_lock_path=lock, original_lock_sha256=_sha(lock),
                  campaign_root=campaign, spec=spec,
                  required_new_paths=new_names)
    return kwargs, shared


def test_downstream_sidecar_keeps_original_source_and_four_reader_hashes(tmp_path):
    kwargs, shared = _fixture(tmp_path)
    value = verify_downstream_sidecar(**kwargs)
    assert value["producer_amendment_sha256"] == kwargs["producer_sha256"]
    (shared / "common_eval.py").write_text("tampered reader\n")
    with pytest.raises(PermissionError, match="inherited reader"):
        verify_downstream_sidecar(**kwargs)


def test_wrong_downstream_manifest_sha_refuses_before_source_access(tmp_path):
    kwargs, _ = _fixture(tmp_path)
    kwargs["sidecar_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="sidecar"):
        verify_downstream_sidecar(**kwargs)


def test_selected_predecessor_fit_keeps_producer_sha(tmp_path):
    fit_dir = tmp_path / ".private/A-FIT-0"
    fit_dir.mkdir(parents=True)
    receipt = _json(fit_dir / "FIT_RECEIPT.json", {
        "status": "SELECTED_FIT_POLICY", "unit_id": "A-FIT-0",
        "arm_id": "A_JOINT_STOCH", "source_seed": 0,
        "a_amendment_sha256": "a" * 64,
    })
    parent = {"status": "done", "spec": {
        "unit_id": "A-FIT-0", "stage": "a", "mode": "real_data",
        "amendment_sha256": "a" * 64,
        "arm_id": "A_JOINT_STOCH", "seed": 0,
        "outputs": [str(receipt)],
    }, "output_sha256": {str(receipt): _sha(receipt)}}
    kwargs = dict(parent=parent, fit_receipt_path=receipt,
                  producer_sha256="a" * 64, fit_unit_id="A-FIT-0",
                  arm_id="A_JOINT_STOCH", seed=0)
    assert verify_predecessor_fit(**kwargs)["status"] == "SELECTED_FIT_POLICY"
    parent["spec"]["amendment_sha256"] = "b" * 64
    with pytest.raises(PermissionError, match="producer FIT"):
        verify_predecessor_fit(**kwargs)


def _registration_fixture(tmp_path):
    kwargs, _ = _fixture(tmp_path)
    sidecar = kwargs["sidecar_path"]
    repo = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    relative = sidecar.relative_to(repo).as_posix()
    subprocess.run(["git", "add", relative], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm",
                    "sealed sidecar"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=repo, text=True).strip()
    sidecar_value = json.loads(sidecar.read_text())
    source_map = {
        "new_source_sha256": sidecar_value["new_source_sha256"],
        "inherited_shared_source_sha256":
            sidecar_value["attack_plan"]["inherited_shared_source_sha256"],
    }
    source_map_sha = hashlib.sha256(json.dumps(
        source_map, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode()).hexdigest()
    registration = _json(kwargs["campaign_root"] /
                         "A_DOWNSTREAM_SOURCE_REGISTRATION_20261010_001.json", {
        "schema": "pcrl-a-downstream-source-registration-v1",
        "status": "LOCALLY_REGISTERED_BY_R00",
        "registered_utc": "2026-10-10T13:00:00Z",
        "producer_amendment_sha256": kwargs["producer_sha256"],
        "manifest_relative_path": relative,
        "manifest_sha256": kwargs["sidecar_sha256"],
        "manifest_commit_full_sha": commit,
        "stage_scope": STAGE_SCOPE,
        "independent_review_sha256": "a" * 64,
        "source_map_sha256": source_map_sha,
    })
    kwargs["spec"]["sources"].append({"path": str(registration),
                                      "sha256": _sha(registration)})
    kwargs["spec"].update({
        "downstream_registration_path": str(registration),
        "downstream_registration_sha256": _sha(registration),
    })
    subprocess.run(["git", "add", registration.relative_to(repo).as_posix()],
                   cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm",
                    "registered sidecar"], cwd=repo, check=True)
    return kwargs, registration, repo


def test_r00_registration_binds_manifest_commit_and_source_map(tmp_path):
    kwargs, registration, repo = _registration_fixture(tmp_path)
    value = verify_downstream_registration(
        registration_path=registration, registration_sha256=_sha(registration),
        sidecar_path=kwargs["sidecar_path"],
        sidecar_sha256=kwargs["sidecar_sha256"],
        producer_sha256=kwargs["producer_sha256"], repo_root=repo,
        spec=kwargs["spec"])
    assert subprocess.run(["git", "merge-base", "--is-ancestor",
                           value["manifest_commit_full_sha"], "HEAD"],
                          cwd=repo, check=False).returncode == 0


def test_r00_registration_refuses_wrong_sha_and_uncommitted_manifest(tmp_path):
    kwargs, registration, repo = _registration_fixture(tmp_path)
    with pytest.raises(PermissionError, match="registration SHA"):
        verify_downstream_registration(
            registration_path=registration, registration_sha256="0" * 64,
            sidecar_path=kwargs["sidecar_path"],
            sidecar_sha256=kwargs["sidecar_sha256"],
            producer_sha256=kwargs["producer_sha256"], repo_root=repo,
            spec=kwargs["spec"])
    value = json.loads(registration.read_text())
    value["manifest_commit_full_sha"] = "0" * 40
    _json(registration, value)
    kwargs["spec"]["downstream_registration_sha256"] = _sha(registration)
    next(row for row in kwargs["spec"]["sources"]
         if row["path"] == str(registration))["sha256"] = _sha(registration)
    subprocess.run(["git", "add", registration.relative_to(repo).as_posix()],
                   cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm",
                    "different manifest reference"], cwd=repo, check=True)
    with pytest.raises(PermissionError, match="manifest commit"):
        verify_downstream_registration(
            registration_path=registration, registration_sha256=_sha(registration),
            sidecar_path=kwargs["sidecar_path"],
            sidecar_sha256=kwargs["sidecar_sha256"],
            producer_sha256=kwargs["producer_sha256"], repo_root=repo,
            spec=kwargs["spec"])


def test_r00_registration_refuses_uncommitted_local_bytes(tmp_path):
    kwargs, registration, repo = _registration_fixture(tmp_path)
    registration.write_text(registration.read_text() + " ")
    kwargs["spec"]["downstream_registration_sha256"] = _sha(registration)
    next(row for row in kwargs["spec"]["sources"]
         if row["path"] == str(registration))["sha256"] = _sha(registration)
    with pytest.raises(PermissionError, match="registration.*Git custody"):
        verify_downstream_registration(
            registration_path=registration, registration_sha256=_sha(registration),
            sidecar_path=kwargs["sidecar_path"],
            sidecar_sha256=kwargs["sidecar_sha256"],
            producer_sha256=kwargs["producer_sha256"], repo_root=repo,
            spec=kwargs["spec"])


def test_r00_registration_refuses_manifest_commit_outside_head_history(tmp_path):
    kwargs, registration, repo = _registration_fixture(tmp_path)
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"],
                                   cwd=repo, text=True).strip()
    orphan = subprocess.check_output([
        "git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
        "commit-tree", tree, "-m", "orphan manifest"], cwd=repo,
        text=True).strip()
    value = json.loads(registration.read_text())
    value["manifest_commit_full_sha"] = orphan
    _json(registration, value)
    kwargs["spec"]["downstream_registration_sha256"] = _sha(registration)
    next(row for row in kwargs["spec"]["sources"]
         if row["path"] == str(registration))["sha256"] = _sha(registration)
    subprocess.run(["git", "add", registration.relative_to(repo).as_posix()],
                   cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm",
                    "register orphan manifest"], cwd=repo, check=True)
    with pytest.raises(PermissionError, match="outside HEAD ancestry"):
        verify_downstream_registration(
            registration_path=registration, registration_sha256=_sha(registration),
            sidecar_path=kwargs["sidecar_path"],
            sidecar_sha256=kwargs["sidecar_sha256"],
            producer_sha256=kwargs["producer_sha256"], repo_root=repo,
            spec=kwargs["spec"])
