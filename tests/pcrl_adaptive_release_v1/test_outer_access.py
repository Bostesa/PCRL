import hashlib
import json

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import inference, outer_access, roles


def _synthetic_outer():
    household = next(f"synthetic-outer-{i}" for i in range(10000)
                     if roles.role_of(f"synthetic-outer-{i}") == "outer_assessment")
    pool = {"x": np.zeros((1, 32)), "ha": np.zeros((1, 4)),
            "hb": np.zeros((1, 2)), "weights": np.ones(1),
            "ids": np.array(["synthetic-person"]),
            "households": np.array([household]),
            "labels": {"same_residence": np.array([1]),
                       "SEX": np.array([0]), "RAC1P": np.array([2])}}
    return {"ctx": {"pools": {"downstream_fit": pool,
                                "attacker_validation": {}}},
            "encoded": {"downstream_fit": {
                "codes": {"T0": np.array([0])},
                "p": np.array([.5]), "r": np.array([0.]),
                "risk": np.zeros((1, 11))}}}


def _lock_files(tmp_path, monkeypatch):
    directory = tmp_path / "pcrl_adaptive_release_v1"
    directory.mkdir()
    lock_path = directory / "SELECTION_LOCK.json"
    unlock_path = directory / "private" / "OUTER_UNLOCK.json"
    unlock_path.parent.mkdir()
    index_path = tmp_path / "2018-index.json"
    index_path.write_text("{}")
    census_path = tmp_path / "DATA_ROLE_COUNTS.json"
    census_path.write_text(json.dumps({"schema": "pcrl-adaptive-role-summary-v1",
        "outer_labels_accessed": False, "global_role_overlap": 0,
        "anchors": {str(a): {"outer_assessment": {
            "people": 1, "households": 1, "weight_sum": 1.}}
            for a in (0, 1, 2)}}))
    monkeypatch.setattr(outer_access, "CENSUS_PATH", census_path)
    slots = [{"id": "candidate", "arm": "U", "source_name": "candidate",
              "comparators": ["D17", "simple", "task_only", "gradient", "deterministic"]}]
    lock = {"schema": "pcrl-adaptive-selection-lock-v1", "status": "LOCKED",
            "study": "pcrl_adaptive_release_v1", "assessment_year": 2018,
            "outer_assessment_authorized": True,
            "protocol_sha256": "a"*64,
            "input_index_sha256": hashlib.sha256(b"{}").hexdigest(),
            "scoring_code_sha256": outer_access.scoring_code_hashes(),
            "outer_population": {
                "census_sha256": hashlib.sha256(census_path.read_bytes()).hexdigest(),
                "pools": list(outer_access.data.POOLS), "role": "outer_assessment"},
            "slots": slots, "alias_of": {},
            "family_representatives": {name: {"U": name} for name in
                                       ("simple", "task_only", "gradient", "deterministic")},
            "family_manifest": inference.family_manifest(slots),
            "capability_manifest": {
                "schema": 1,
                "endpoints": inference.capability_endpoints(["candidate"]),
                "n_endpoints": 2,
                "multiplicity": "separate two-sided Bonferroni H-capability family",
                "scope": "2018 development; not a primary-clause rescue"},
            "anchors": {str(anchor): {
                "inner_panel_complete_sha256": "b"*64,
                "inner_audit_sha256": "c"*64,
                "logical_to_canonical": {name: name for name in
                    ("candidate", "D17", "simple", "task_only", "gradient", "deterministic")},
                "releases": {"candidate": {"channel_array_sha256": "d"*64},
                             **{name: {"channel_array_sha256": "e"*64} for name in
                                ("D17", "simple", "task_only", "gradient", "deterministic")}}
            } for anchor in (0, 1, 2)}}
    lock_path.write_text(json.dumps(lock))
    digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    unlock = {"schema": "pcrl-adaptive-outer-unlock-v1",
              "lock_sha256": digest, "remote_verified": True,
              "branch": outer_access.BRANCH, "remote_commit_sha": "f"*40}
    unlock_path.write_text(json.dumps(unlock))
    monkeypatch.setattr(outer_access, "LOCK_PATH", lock_path)
    monkeypatch.setattr(outer_access, "UNLOCK_PATH", unlock_path)
    return lock_path, unlock_path, index_path, digest


def test_outer_loader_refuses_unverified_unlock_before_deserializing(tmp_path, monkeypatch):
    lock, receipt, index, digest = _lock_files(tmp_path, monkeypatch)
    value = json.loads(receipt.read_text())
    value["remote_verified"] = False
    receipt.write_text(json.dumps(value))
    monkeypatch.setattr(outer_access.data, "index", lambda *_: pytest.fail("index read before lock"))
    with pytest.raises(PermissionError, match="remotely verified"):
        outer_access.load_locked_outer_role(index, 0, lock, digest)
    with pytest.raises(PermissionError, match="SHA-256"):
        outer_access.load_locked_outer_role(index, 0, lock, "0"*64)


def test_verified_outer_gate_opens_only_globally_assigned_2018_rows(tmp_path, monkeypatch):
    lock, _, index, digest = _lock_files(tmp_path, monkeypatch)
    monkeypatch.setattr(outer_access.data, "index", lambda *_: {})
    monkeypatch.setattr(outer_access.data, "_sanitized_record", lambda *_: {"verified": True})
    monkeypatch.setattr(outer_access.outer_pool, "load_verified_outer",
                        lambda *args: roles._pooled_role(_synthetic_outer(),
                                             "outer_assessment", allow_outer=True))
    with pytest.raises(PermissionError):
        roles.pooled_role(_synthetic_outer(), "outer_assessment")
    rows = outer_access.load_locked_outer_role(index, 0, lock, digest)
    assert rows["ids"].tolist() == ["synthetic-person"]
    assert rows["labels"]["RAC1P"].tolist() == [2]
    assert rows["token_codes"].tolist() == [0]


def test_outer_gate_refuses_unmapped_or_unscored_endpoint_before_data_read(tmp_path, monkeypatch):
    lock, receipt, index, _ = _lock_files(tmp_path, monkeypatch)
    monkeypatch.setattr(outer_access.data, "index", lambda *_: pytest.fail("index read before mapping check"))
    for mutation in ("unmapped", "unscored", "missing_family", "changed_scorer"):
        value = json.loads(lock.read_text())
        if mutation == "unmapped":
            del value["anchors"]["1"]["logical_to_canonical"]["candidate"]
        elif mutation == "unscored":
            value["anchors"]["2"]["logical_to_canonical"]["gradient"] = "absent"
        else:
            if mutation == "missing_family":
                del value["family_representatives"]["gradient"]
            else:
                value["scoring_code_sha256"]["outer_pool.py"] = "0"*64
        lock.write_text(json.dumps(value))
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        unlock = json.loads(receipt.read_text())
        unlock["lock_sha256"] = digest
        receipt.write_text(json.dumps(unlock))
        with pytest.raises(PermissionError):
            outer_access.load_locked_outer_role(index, 0, lock, digest)


def test_postlock_scorer_repair_requires_remote_pinned_exact_hash_change(tmp_path, monkeypatch):
    lock_path, unlock_path, _, _ = _lock_files(tmp_path, monkeypatch)
    current = outer_access.scoring_code_hashes()
    frozen = dict(current)
    frozen["outer_pool.py"] = "1" * 64
    frozen["outer_access.py"] = "2" * 64
    lock = json.loads(lock_path.read_text())
    lock["scoring_code_sha256"] = frozen
    lock_path.write_text(json.dumps(lock))
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    unlock = json.loads(unlock_path.read_text())
    unlock["lock_sha256"] = lock_sha
    unlock_path.write_text(json.dumps(unlock))

    public = lock_path.with_name("SCORER_CORRECTION.json")
    private = unlock_path.with_name("SCORER_CORRECTION_UNLOCK.json")
    monkeypatch.setattr(outer_access, "CORRECTION_PATH", public, raising=False)
    monkeypatch.setattr(outer_access, "CORRECTION_UNLOCK_PATH", private, raising=False)
    with pytest.raises(PermissionError):
        outer_access.verify_outer_unlock(lock_path, lock_sha)

    correction = {
        "schema": "pcrl-adaptive-postlock-scorer-correction-v1",
        "status": "TECHNICAL_SCORER_REPAIR",
        "lock_sha256": lock_sha,
        "original_scoring_code_sha256": frozen,
        "corrected_scoring_code_sha256": current,
        "changed_modules": ["outer_access.py", "outer_pool.py"],
        "score_outputs_written_before_repair": False,
        "candidate_panel_unchanged": True,
    }
    public.write_text(json.dumps(correction))
    private.write_text(json.dumps({
        "schema": "pcrl-adaptive-scorer-correction-unlock-v1",
        "lock_sha256": lock_sha,
        "correction_sha256": hashlib.sha256(public.read_bytes()).hexdigest(),
        "remote_verified": True,
        "branch": outer_access.BRANCH,
        "remote_commit_sha": "e" * 40,
    }))
    assert outer_access.verify_outer_unlock(lock_path, lock_sha) == lock
    bad = json.loads(private.read_text())
    bad["correction_sha256"] = "0" * 64
    private.write_text(json.dumps(bad))
    with pytest.raises(PermissionError):
        outer_access.verify_outer_unlock(lock_path, lock_sha)
