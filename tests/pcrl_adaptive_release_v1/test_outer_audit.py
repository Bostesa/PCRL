"""Synthetic outer scorer gates; no ACS outer rows are opened in tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import outer_audit
from experiments.pcrl_adaptive_release_v1 import audit, evaluate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(role: str) -> str:
    return role.replace(":", "_").replace("/", "_")


def _slate(root: Path, role: str, release_id: str, *, slate="catchup") -> dict:
    directory = root / release_id / _slug(role)
    model = directory / "models" / "m"
    model.mkdir(parents=True)
    (model / "synthetic.bin").write_bytes(b"frozen-test-model")
    from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited
    target = role.split("/")[1]
    view = role.split(":")[1].split("/")[0]
    registry = {
        "role": role, "release_id": release_id, "slate": slate,
        "models": {"m": {
            "kind": "model", "target": target, "source_view": view,
            "wire": "H" if release_id == "H" or view == "B" else "release",
            "source_release_id": release_id,
            "model_directory": "/stale/restored/path",
            "model_relative_directory": "models/m",
            "model_sha256": inherited.model_directory_hash(model),
        }},
    }
    registry["artifacts_sha256"] = inherited._artifact_inventory(directory)
    (directory / "own_registry.json").write_text(json.dumps(registry, sort_keys=True))
    return registry


def _panel(tmp_path):
    index = tmp_path / "index.json"
    index.write_text("{}")
    root = tmp_path / "private" / "inner"
    h_roles = (*audit.ROLES, "attack:B/SEX", "attack:B/RAC1P")
    h_reg = {role: _slate(root, role, "H") for role in h_roles}
    own_reg = {role: _slate(root, role, "candidate") for role in audit.ROLES}
    roles = {}
    for role in audit.ROLES:
        own_routes = audit.candidate_route_bank(
            role, "candidate", own_reg[role], h_reg[role],
            **({"a_same": own_reg[f"attack:A/{role.split('/')[1]}"],
                "b_h_only": h_reg[f"attack:B/{role.split('/')[1]}"]}
               if role.startswith("attack:AB/") else {}))
        h_routes = audit.candidate_route_bank(
            role, "H", h_reg[role], h_reg[role],
            **({"a_same": h_reg[f"attack:A/{role.split('/')[1]}"],
                "b_h_only": h_reg[f"attack:B/{role.split('/')[1]}"]}
               if role.startswith("attack:AB/") else {}))
        roles[role] = {
            "selected_candidate": "own/m", "candidate_count": len(own_routes),
            "selected_route": {key: own_routes["own/m"].get(key)
                               for key in ("source_view", "wire", "model_sha256")},
            "H_selected_candidate": "own/m",
            "H_selected_route": {key: h_routes["own/m"].get(key)
                                 for key in ("source_view", "wire", "model_sha256")},
        }
    source = {"router_kind": "T0", "channel_array_sha256": "synthetic"}
    report = {"schema": "pcrl-adaptive-inner-audit-v1", "anchor": 0,
              "outer_pool_opened": False, "slate": "catchup",
              "index_sha256": _sha(index),
              "source_code_sha256": outer_audit.current_inner_code_hashes(),
              "releases": {"candidate": {"source": source, "roles": roles,
                                         "private_contribution_prefix": "123456789abc"}}}
    (root / "INNER_AUDIT.json").write_text(json.dumps(report, sort_keys=True))
    receipt = {"schema": "pcrl-adaptive-inner-audit-complete-v1", "anchor": 0,
               "release_ids": ["candidate"], "artifacts": evaluate._inventory(root)}
    (root / "COMPLETE.json").write_text(json.dumps(receipt, sort_keys=True))
    lock = {"schema": "pcrl-adaptive-selection-lock-v1", "status": "LOCKED",
            "assessment_year": 2018,
            "anchors": {"0": {"inner_panel_complete_sha256": _sha(root / "COMPLETE.json"),
                               "inner_audit_sha256": _sha(root / "INNER_AUDIT.json"),
                               "releases": {"candidate": source}}}}
    lock_path = tmp_path / "SELECTION_LOCK.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True))
    return root, lock_path, report


def _rows():
    return {"x": np.zeros((4, 32)), "ha": np.zeros((4, 4)),
            "hb": np.zeros((4, 2)), "weights": np.array([1., 2., 1., 2.]),
            "ids": np.array(["a", "b", "c", "d"]),
            "households": np.array(["h1", "h1", "h2", "h2"]),
            "token_codes": np.array([0, 1, 0, 1]),
            "teacher_p": np.full(4, .5), "residual": np.zeros(4),
            "risk": np.zeros((4, 11)),
            "labels": {"same_residence": np.array([0, 1, 0, 1]),
                       "SEX": np.array([0, 1, 0, 1]),
                       "RAC1P": np.array([0, 1, 8, 0])}}


def test_inner_lock_and_slate_replay_before_unlocked_outer_load(tmp_path, monkeypatch):
    inner, lock_path, _ = _panel(tmp_path)
    q = np.zeros((32, 17)); q[:, 0] = 1.
    channel = tmp_path / "private" / "Q.npz"
    np.savez_compressed(channel, Q=q)
    spec = {"Q": q, "channel_artifact_path": str(channel),
            "channel_artifact_sha256": _sha(channel), "router": "T0"}
    # The synthetic inner descriptor is replaced with the actual pinned spec.
    descriptor = outer_audit.release_descriptor(spec, tmp_path / "index.json")
    report = json.loads((inner / "INNER_AUDIT.json").read_text())
    report["releases"]["candidate"]["source"] = descriptor
    (inner / "INNER_AUDIT.json").write_text(json.dumps(report, sort_keys=True))
    receipt = json.loads((inner / "COMPLETE.json").read_text())
    receipt["artifacts"] = evaluate._inventory(inner)
    (inner / "COMPLETE.json").write_text(json.dumps(receipt, sort_keys=True))
    lock = json.loads(lock_path.read_text())
    lock["anchors"]["0"] = {
        "inner_panel_complete_sha256": _sha(inner / "COMPLETE.json"),
        "inner_audit_sha256": _sha(inner / "INNER_AUDIT.json"),
        "releases": {"candidate": descriptor}}
    lock_path.write_text(json.dumps(lock, sort_keys=True))
    index = tmp_path / "index.json"
    calls = []
    def unlocked(*args):
        calls.append("outer")
        return _rows()
    monkeypatch.setattr(outer_audit, "_load_unlocked_rows", unlocked)
    def score(rows, law, route_lock):
        assert route_lock["selected"] == "own/m"
        assert Path(route_lock["route"]["model_directory"]).is_file() is False
        assert Path(route_lock["route"]["model_directory"]).exists()
        value = .6 if route_lock["release_id"] == "H" else .5
        loss = np.full(4, value)
        return {"ids": rows["ids"], "households": rows["households"],
                "weights": rows["weights"], "loss": loss,
                "scores": audit.score_weightings(loss, rows["weights"])}
    monkeypatch.setattr(outer_audit.audit, "score_frozen_route", score)
    out = tmp_path / "private" / "outer"
    result = outer_audit.score_locked_outer(
        0, {"candidate": spec}, index, inner, lock_path, _sha(lock_path), out)
    assert calls == ["outer"]
    assert result["releases"]["candidate"]["roles"]["attack:AB/SEX"]["candidate"]["U"] == pytest.approx(.5)
    assert (out / "COMPLETE.json").exists()
    with pytest.raises(FileExistsError):
        outer_audit.score_locked_outer(
            0, {"candidate": spec}, index, inner, lock_path, _sha(lock_path), out)


def test_changed_inner_report_or_lock_stops_before_outer_gate(tmp_path, monkeypatch):
    inner, lock_path, _ = _panel(tmp_path)
    calls = []
    monkeypatch.setattr(outer_audit, "_load_unlocked_rows", lambda *args: calls.append("outer"))
    with pytest.raises(ValueError, match="lock|descriptor|release"):
        outer_audit.score_locked_outer(
            0, {}, tmp_path / "index.json", inner, lock_path, _sha(lock_path),
            tmp_path / "private" / "outer")
    assert calls == []


def test_index_or_scoring_source_drift_stops_before_outer_gate(tmp_path, monkeypatch):
    inner, lock_path, report = _panel(tmp_path)
    index = tmp_path / "index.json"
    calls = []
    monkeypatch.setattr(outer_audit, "_load_unlocked_rows", lambda *args: calls.append("outer"))
    with pytest.raises(ValueError, match="index"):
        outer_audit.verify_scoring_provenance(report, index.with_name("other.json"))
    report["source_code_sha256"]["audit.py"] = "0"*64
    with pytest.raises(ValueError, match="source"):
        outer_audit.verify_scoring_provenance(report, index)
    assert calls == []
    (inner / "INNER_AUDIT.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA|hash|report"):
        outer_audit.verify_locked_inner(
            inner, lock_path, _sha(lock_path), anchor=0,
            release_ids={"candidate"})
    assert calls == []


def test_cli_builds_only_locked_release_ids_and_delegates_once(tmp_path, monkeypatch):
    _, lock_path, _ = _panel(tmp_path)
    from experiments.pcrl_adaptive_release_v1 import release_specs
    calls = []
    monkeypatch.setattr(release_specs, "build_release_specs", lambda *args, **kwargs: {
        "releases": {"candidate": {"synthetic": True}, "other": {"synthetic": True}}})
    def score(anchor, releases, *args):
        calls.append((anchor, set(releases)))
        output = Path(args[-1]); output.mkdir(parents=True)
        (output / "COMPLETE.json").write_text("{}")
        return {"schema": "synthetic"}
    monkeypatch.setattr(outer_audit, "score_locked_outer", score)
    result = outer_audit.main([
        "--anchor", "0", "--delta", "0.001",
        "--index", str(tmp_path / "index.json"),
        "--inner-panel-dir", str(tmp_path / "private" / "inner"),
        "--selection-lock", str(lock_path), "--lock-sha256", _sha(lock_path),
        "--a-center-dir", str(tmp_path / "private" / "a"),
        "--output-dir", str(tmp_path / "private" / "outer")])
    assert result["schema"] == "synthetic"
    assert calls == [(0, {"candidate"})]
