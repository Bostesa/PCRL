"""Three-anchor locked outer inference from synthetic completed score archives."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import audit, evaluate, inference
from experiments.pcrl_adaptive_release_v1 import inference_run


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_outer(tmp_path):
    slot = {"id": "P_candidate", "arm": "P", "source_name": "B_selected",
            "comparators": ["D17"]}
    lock = {"schema": "pcrl-adaptive-selection-lock-v1", "status": "LOCKED",
            "assessment_year": 2018, "slots": [slot], "alias_of": {},
            "family_manifest": inference.family_manifest([slot]), "anchors": {}}
    descriptors = {"A_selected": {"toy": "A"},
                   "B_selected": {"toy": "B"}, "D17": {"toy": "D17"}}
    for anchor in range(3):
        candidate = "A_selected" if anchor == 0 else "B_selected"
        lock["anchors"][str(anchor)] = {
            "inner_panel_complete_sha256": "a"*64,
            "inner_audit_sha256": "b"*64,
            "logical_to_canonical": {"P_candidate": candidate,
                                     "B_selected": candidate, "D17": "D17"},
            "releases": {candidate: descriptors[candidate], "D17": descriptors["D17"]}}
    lock_path = tmp_path / "SELECTION_LOCK.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True))
    lock_sha = _sha(lock_path)
    outer_dirs = {}
    ids = np.array(["person-1", "person-2", "person-3", "person-4"])
    households = np.array(["house-1", "house-1", "house-2", "house-2"])
    weights = np.array([1., 2., 1., 2.])
    for anchor in range(3):
        root = tmp_path / "private" / f"outer_{anchor}"
        root.mkdir(parents=True, exist_ok=True)
        candidate = "A_selected" if anchor == 0 else "B_selected"
        records = {}; contributions = {}
        for name in (candidate, "D17"):
            prefix = hashlib.sha256(name.encode()).hexdigest()[:12]
            role_records = {}
            for role in audit.ROLES:
                loss = np.full(4, .45 if anchor == 0 and name == candidate else
                               .4 if name == candidate else .5)
                h_loss = np.full(4, .6)
                score = {"ids": ids, "households": households,
                         "weights": weights, "loss": loss,
                         "scores": audit.score_weightings(loss, weights)}
                h = {"ids": ids, "households": households,
                     "weights": weights, "loss": h_loss,
                     "scores": audit.score_weightings(h_loss, weights)}
                contributions.update(evaluate._private_contributions(prefix, role, score, h))
                role_records[role] = {"candidate": score["scores"], "H": h["scores"]}
            records[name] = {"source": descriptors[name], "roles": role_records,
                             "private_contribution_prefix": prefix}
        archive = root / "OUTER_CONTRIBUTIONS.npz"
        np.savez_compressed(archive, **contributions)
        report = {"schema": "pcrl-adaptive-outer-audit-v1", "anchor": anchor,
                  "selection_lock_sha256": lock_sha, "releases": records,
                  "assessment_year": 2018, "development_only": True,
                  "no_outer_fit_or_selection": True,
                  "inner_panel_complete_sha256": "a"*64,
                  "inner_audit_sha256": "b"*64,
                  "contributions_relative_path": archive.name,
                  "contributions_sha256": _sha(archive)}
        (root / "OUTER_AUDIT.json").write_text(json.dumps(report, sort_keys=True))
        receipt = {"schema": "pcrl-adaptive-outer-audit-complete-v1",
                   "anchor": anchor, "release_ids": sorted(records),
                   "selection_lock_sha256": lock_sha,
                   "artifacts": evaluate._inventory(root)}
        (root / "COMPLETE.json").write_text(json.dumps(receipt, sort_keys=True))
        outer_dirs[anchor] = root
    return lock_path, lock_sha, outer_dirs


def test_replay_resolves_anchor_specific_alias_and_common_household_family(tmp_path, monkeypatch):
    lock_path, lock_sha, outer_dirs = _synthetic_outer(tmp_path)
    monkeypatch.setattr(inference_run, "_verify_remote_lock", lambda *args: None)
    output = tmp_path / "INFERENCE.json"
    private = tmp_path / "private" / "INFERENCE_REPLAY_INDEX.json"
    report = inference_run.run_inference(lock_path, lock_sha, outer_dirs,
                                         output, private)
    assert report["family_size"] == 10
    assert report["bootstrap"]["requested"] == 10000
    assert report["decisions"]["P_candidate"]["all_primary_clauses_passed"] is False
    row = next(item for item in report["rows"]
               if item["role"] == "utility:A/same_residence" and
               item["weighting"] == "U")
    assert row["estimate"] == pytest.approx((-.05-.1-.1)/3)
    assert report["anchor_resolution"]["0"]["P_candidate"] == "A_selected"
    assert report["anchor_resolution"]["1"]["P_candidate"] == "B_selected"
    assert "person-1" not in output.read_text() and "house-1" not in output.read_text()
    assert json.loads(private.read_text())["outer_artifacts"]["0"]["contributions_sha256"]
    with pytest.raises(FileExistsError):
        inference_run.run_inference(lock_path, lock_sha, outer_dirs, output, private)


def test_unmapped_endpoint_or_tampered_contributions_fails_before_output(tmp_path, monkeypatch):
    lock_path, lock_sha, outer_dirs = _synthetic_outer(tmp_path)
    monkeypatch.setattr(inference_run, "_verify_remote_lock", lambda *args: None)
    lock = json.loads(lock_path.read_text())
    del lock["anchors"]["1"]["logical_to_canonical"]["P_candidate"]
    with pytest.raises(ValueError, match="unmapped|resolve"):
        inference_run.validate_locked_resolution(lock)
    lock["anchors"]["1"]["logical_to_canonical"]["P_candidate"] = "B_selected"
    lock["anchors"]["1"]["logical_to_canonical"]["unused_full_spec_name"] = "unscored"
    assert inference_run.validate_locked_resolution(lock)[1]["P_candidate"] == "B_selected"
    lock_path, lock_sha, outer_dirs = _synthetic_outer(tmp_path)
    path = outer_dirs[2] / "OUTER_CONTRIBUTIONS.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="artifact|inventory|SHA"):
        inference_run.run_inference(
            lock_path, lock_sha, outer_dirs, tmp_path / "INFERENCE.json",
            tmp_path / "private" / "INFERENCE_REPLAY_INDEX.json")
    assert not (tmp_path / "INFERENCE.json").exists()
