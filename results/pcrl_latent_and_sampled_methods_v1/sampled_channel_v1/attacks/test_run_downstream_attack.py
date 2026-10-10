"""Data-free downstream sidecar and imported-worker custody checks."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks import (
    run_downstream_attack as downstream,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return _sha(path)


def _fixture(tmp_path, monkeypatch, *, stub_registration=True):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    shared = tmp_path / "pcrl_method_portfolio_v1" / "shared"
    lock = campaign / "CAMPAIGN_LOCK.json"
    lock_sha = _write(lock, '{}\n')
    producer_path = campaign / "A_STAGE_AMENDMENT.json"
    unit_id = "A-ATTACK-A_JOINT_DET-S0-R0"
    plan = {"kind": "attack", "arm_id": "A_JOINT_DET", "seed": 0,
            "realization": 0, "output_id": unit_id,
            "deps": ["FIT", "ATT", "INNER", "Q"]}
    original_worker = campaign / downstream.SELECTED_MODULE
    worker_sha = _write(original_worker, "frozen selected worker\n")
    producer = {"status": "LOCALLY_SEALED_A_STAGE",
                "original_lock_sha256": lock_sha,
                "unit_plan": {unit_id: plan},
                "implementation_source_sha256": {
                    downstream.SELECTED_MODULE: worker_sha}}
    producer_sha = _write(producer_path, json.dumps(producer) + "\n")
    new_sources = {}
    for relative in (downstream.THIS_MODULE, downstream.RENDERER_MODULE):
        new_sources[relative] = _write(campaign / relative,
                                       f"frozen {relative}\n")
    inherited = {}
    for name in downstream.SHARED:
        inherited[name] = _write(shared / name, f"frozen {name}\n")
    plan_sha = hashlib.sha256(json.dumps(
        producer["unit_plan"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    manifest = {"schema": downstream.SCHEMA,
                "status": downstream.STATUS, "dated_utc": "2026-10-10T00:00:00Z",
                "producer_amendment_sha256": producer_sha,
                "original_lock_sha256": lock_sha,
                "unit_plan_sha256": plan_sha,
                "new_source_sha256": new_sources,
                "attack_plan": {
                    "inherited_shared_source_sha256": inherited},
                "stage_scope": ["attack", "diagnostic", "q", "reference",
                                "release"],
                "scientific_scope_unchanged": True}
    manifest_path = campaign / "A_DOWNSTREAM_SOURCE_MANIFEST_20261010_001.json"
    manifest_sha = _write(manifest_path, json.dumps(manifest) + "\n")
    registration_path = campaign / "A_DOWNSTREAM_SOURCE_REGISTRATION_20261010_001.json"
    registration_sha = _write(registration_path, '{"synthetic":"registered"}\n')
    output = tmp_path / ".private" / unit_id
    raw = ["--unit-id", unit_id, "--kind", "attack"]
    source_files = [lock, producer_path, manifest_path, registration_path,
                    original_worker,
                    *(campaign / name for name in new_sources),
                    *(shared / name for name in inherited)]
    spec = {"unit_id": unit_id, "stage": "a", "mode": "real_data",
            "data_access": "ATTACK_FIT_NEW_SEX_THEN_INNER_SELECTION_SEX",
            "amendment_sha256": producer_sha,
            "downstream_manifest_sha256": manifest_sha,
            "downstream_manifest_path": str(manifest_path),
            "downstream_registration_sha256": registration_sha,
            "downstream_registration_path": str(registration_path),
            "deps": plan["deps"], "arm_id": "A_JOINT_DET", "seed": 0,
            "realization": 0, "output_id": unit_id,
            "outputs": [str(output / "ATTACK_ATTEMPT_INDEX.json")],
            "argv": ["python", "-m", downstream.MODULE_ID, *raw],
            "sources": [{"path": str(path.resolve()), "sha256": _sha(path)}
                        for path in source_files]}
    from results.pcrl_latent_and_sampled_methods_v1.r03 import a_graph, compute_queue
    monkeypatch.setattr(a_graph, "compile_graph", lambda *args, **kwargs: {})
    monkeypatch.setattr(compute_queue, "assert_active_from_env",
                        lambda **kwargs: {"status": "active"})
    if stub_registration:
        monkeypatch.setattr(downstream, "validate_registration",
                            lambda **kwargs: {"status": "LOCALLY_REGISTERED_BY_R00"})
    monkeypatch.setattr(downstream, "__file__",
                        str(campaign / downstream.THIS_MODULE))
    kwargs = {"unit_id": unit_id,
              "producer_amendment_path": producer_path,
              "producer_amendment_sha256": producer_sha,
              "downstream_manifest_path": manifest_path,
              "downstream_manifest_sha256": manifest_sha,
              "downstream_registration_path": registration_path,
              "downstream_registration_sha256": registration_sha,
              "original_lock_path": lock,
              "original_lock_sha256": lock_sha,
              "private_root": output, "argv": raw, "spec": spec}
    return kwargs, manifest, shared, campaign


def test_sidecar_accepts_exact_sources_and_rejects_postseal_reader_edit(
        tmp_path, monkeypatch):
    kwargs, _, shared, _ = _fixture(tmp_path, monkeypatch)
    producer, manifest, spec = downstream.validate_downstream(**kwargs)
    assert producer["unit_plan"][kwargs["unit_id"]]["kind"] == "attack"
    assert manifest["producer_amendment_sha256"] == kwargs[
        "producer_amendment_sha256"]
    assert spec is kwargs["spec"]
    original_outputs = spec["outputs"]
    spec["outputs"] = [str(kwargs["private_root"] / "unregistered.json")]
    with pytest.raises(PermissionError, match="differs from original graph"):
        downstream.validate_downstream(**kwargs)
    spec["outputs"] = original_outputs
    (shared / "attack_eval.py").write_text("unsealed reader edit\n")
    with pytest.raises(PermissionError, match="differs from sealed queue"):
        downstream.validate_downstream(**kwargs)


def test_sidecar_rejects_unpinned_manifest_and_wrong_imported_worker(
        tmp_path, monkeypatch):
    kwargs, _, _, campaign = _fixture(tmp_path, monkeypatch)
    altered = dict(kwargs)
    altered["downstream_manifest_sha256"] = "f" * 64
    with pytest.raises(PermissionError, match="protocol bytes differ"):
        downstream.validate_downstream(**altered)
    producer, _, spec = downstream.validate_downstream(**kwargs)
    sources = downstream._sources(spec)
    with pytest.raises(PermissionError, match="imported A attack worker origin"):
        downstream._worker_module("attack", campaign, producer, sources)


def test_registration_refuses_wrong_source_map_before_git_or_labels(
        tmp_path, monkeypatch):
    kwargs, manifest, _, campaign = _fixture(
        tmp_path, monkeypatch, stub_registration=False)
    path = kwargs["downstream_registration_path"]
    record = {
        "schema": downstream.REGISTRATION_SCHEMA,
        "status": downstream.REGISTRATION_STATUS,
        "registered_utc": "2026-10-10T00:00:00Z",
        "producer_amendment_sha256": kwargs["producer_amendment_sha256"],
        "manifest_relative_path": kwargs["downstream_manifest_path"].relative_to(
            campaign.parents[1]).as_posix(),
        "manifest_sha256": kwargs["downstream_manifest_sha256"],
        "manifest_commit_full_sha": "a" * 40,
        "stage_scope": downstream.STAGE_SCOPE,
        "independent_review_sha256": "b" * 64,
        "source_map_sha256": "f" * 64,
    }
    source_map = {
        "new_source_sha256": manifest["new_source_sha256"],
        "inherited_shared_source_sha256": manifest["attack_plan"][
            "inherited_shared_source_sha256"],
    }
    correct_map_sha = hashlib.sha256(json.dumps(
        source_map, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    path.write_text(json.dumps(record))
    with pytest.raises(PermissionError, match="does not bind manifest"):
        downstream.validate_registration(
            campaign=campaign,
            producer_sha256=kwargs["producer_amendment_sha256"],
            manifest_path=kwargs["downstream_manifest_path"],
            manifest_sha256=kwargs["downstream_manifest_sha256"],
            manifest=manifest,
            registration_path=path, registration_sha256=_sha(path))
    record["source_map_sha256"] = correct_map_sha
    path.write_text(json.dumps(record))
    monkeypatch.setattr(downstream, "_committed_blob",
                        lambda repo, revision, target: target.read_bytes())
    monkeypatch.setattr(downstream.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=0))
    got = downstream.validate_registration(
        campaign=campaign,
        producer_sha256=kwargs["producer_amendment_sha256"],
        manifest_path=kwargs["downstream_manifest_path"],
        manifest_sha256=kwargs["downstream_manifest_sha256"], manifest=manifest,
        registration_path=path, registration_sha256=_sha(path))
    assert got["source_map_sha256"] == correct_map_sha
