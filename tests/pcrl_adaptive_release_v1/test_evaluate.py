"""Synthetic executable panel and immutable receipt checks; no ACS rows."""
from __future__ import annotations

import hashlib
import importlib
import json
import stat

import numpy as np
import pytest


def evaluate():
    try:
        return importlib.import_module("experiments.pcrl_adaptive_release_v1.evaluate")
    except ModuleNotFoundError as error:
        pytest.fail(f"inner panel implementation missing: {error}")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(role_name):
    position = {"audit_fit": 0, "inner_selection": 1, "inner_check": 2}[role_name]
    return {"x": np.zeros((4, 32)), "ha": np.zeros((4, 4)), "hb": np.zeros((4, 2)),
            "token_codes": np.array([0, 1, 0, 1]),
            "teacher_p": np.full(4, .5), "residual": np.zeros(4),
            "labels": {"same_residence": np.array([0, 1, 0, 1]),
                       "SEX": np.array([0, 1, 0, 1]),
                       "RAC1P": np.array([0, 1, 8, 0])},
            "weights": np.array([1., 2., 1., 2.]),
            "ids": np.array([f"{position}-{i}" for i in range(4)]),
            "households": np.array([f"{position}-h{i}" for i in range(4)])}


def _fake_registry(role, release_id, slate):
    _, view, target = role.replace("attack:B/", "attack:B/").split(":")[0], role.split(":")[1].split("/")[0], role.split("/")[1]
    return {"role": role, "release_id": release_id, "slate": slate,
            "fit_missing_classes": [], "validation_missing_classes": [],
            "models": {"m": {"kind": "model", "target": target,
                              "source_view": view,
                              "wire": "H" if release_id == "H" or view == "B" else "release",
                              "source_release_id": release_id,
                              "model_sha256": "a" * 64}}}


def test_inner_panel_uses_only_three_roles_and_writes_immutable_private_receipt(tmp_path, monkeypatch):
    e = evaluate()
    index_path = tmp_path / "index.json"
    index_path.write_text('{"synthetic":true}')
    channel_path = tmp_path / "channel.npz"
    q = np.zeros((32, 17)); q[:, 0] = 1.
    np.savez_compressed(channel_path, Q=q)
    input_calls, fit_calls, route_keys = [], [], []
    monkeypatch.setattr(e.data, "index", lambda path: {"synthetic": True})
    monkeypatch.setattr(e.data, "_sanitized_record", lambda index, anchor: (index_path, {}))
    monkeypatch.setattr(e.data, "load_prepared", lambda index, anchor: {"anchor": anchor})
    def pooled(prepared, role):
        input_calls.append(role)
        return _rows(role)
    monkeypatch.setattr(e.roles, "pooled_role", pooled)
    def fit(fit_rows, fit_p, selection_rows, selection_p, role,
            output_dir, seed, *, release_id, slate):
        fit_calls.append((role, release_id, fit_p.shape[1], selection_p.shape[1]))
        return _fake_registry(role, release_id, slate)
    monkeypatch.setattr(e.audit, "fit_role_slate", fit)
    def select(rows, law, role, routes, *, release_id):
        route_keys.append((role, release_id, set(routes)))
        return {"role": role, "release_id": release_id,
                "route": next(iter(routes.values())), "selected": "m",
                "scores": {"m": {"balanced": .5}},
                "candidate_count": len(routes), "rule": "synthetic"}
    monkeypatch.setattr(e.audit, "select_frozen_routes", select)
    def score(rows, law, lock):
        target = lock["role"].split("/")[1]
        mask = rows["labels"][target] >= 0
        value = .6 if lock["release_id"] == "H" else .5
        loss = np.full(mask.sum(), value)
        weights = rows["weights"][mask]
        return {"ids": rows["ids"][mask], "households": rows["households"][mask],
                "weights": weights, "loss": loss,
                "scores": e.audit.score_weightings(loss, weights)}
    monkeypatch.setattr(e.audit, "score_frozen_route", score)
    release = {"candidate": {"Q": q, "channel_artifact_path": str(channel_path),
                             "channel_artifact_sha256": _sha(channel_path),
                             "router": "T0"}}
    output = tmp_path / "private" / "panel"
    report = e.audit_panel(0, release, index_path, output)
    assert input_calls == ["audit_fit", "inner_selection", "inner_check"]
    assert len(fit_calls) == 12  # seven shared H ancestors, five candidate roles
    assert len(route_keys) == 10  # candidate and H selections for five roles
    assert any(keys == {"own/m", "H/m", "A/m", "B/m"}
               for role, release_id, keys in route_keys
               if role == "attack:AB/SEX" and release_id == "candidate")
    assert report["outer_pool_opened"] is False
    assert "inherited_audit.py" in report["source_code_sha256"]
    assert "data.py" in report["source_code_sha256"]
    assert report["releases"]["candidate"]["roles"]["attack:AB/SEX"]["candidate"]["U"] == pytest.approx(.5)
    receipt = json.loads((output / "COMPLETE.json").read_text())
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "PANEL_CONTRIBUTIONS.npz").stat().st_mode) == 0o600
    assert receipt["artifacts"]["INNER_AUDIT.json"] == _sha(output / "INNER_AUDIT.json")
    assert receipt["artifacts"]["PANEL_CONTRIBUTIONS.npz"] == _sha(output / "PANEL_CONTRIBUTIONS.npz")
    with np.load(output / "PANEL_CONTRIBUTIONS.npz", allow_pickle=False) as private:
        assert any(key.endswith("_household") for key in private.files)
        prefix = report["releases"]["candidate"]["private_contribution_prefix"]
        role_key = f"{prefix}__attack_AB_SEX"
        assert private[f"{role_key}_household_candidate_U_num"].sum() == pytest.approx(2.)
        assert private[f"{role_key}_household_candidate_W_num"].sum() == pytest.approx(3.)
    with pytest.raises(FileExistsError):
        e.audit_panel(0, release, index_path, output)


def test_inner_panel_refuses_raw_prepared_fallback_before_deserialization(tmp_path, monkeypatch):
    e = evaluate()
    index_path = tmp_path / "index.json"
    index_path.write_text('{"synthetic":true}')
    q = np.zeros((32, 17)); q[:, 0] = 1.
    channel_path = tmp_path / "q.npz"; np.savez_compressed(channel_path, Q=q)
    monkeypatch.setattr(e.data, "index", lambda path: {"synthetic": True})
    monkeypatch.setattr(e.data, "_sanitized_record", lambda index, anchor: None)
    def forbidden_load(index, anchor):
        raise AssertionError("raw prepared object was deserialized before sanitized gate")
    monkeypatch.setattr(e.data, "load_prepared", forbidden_load)
    release = {"candidate": {"Q": q, "channel_artifact_path": str(channel_path),
                             "channel_artifact_sha256": _sha(channel_path),
                             "router": "T0"}}
    with pytest.raises(FileNotFoundError, match="sanitized"):
        e.audit_panel(0, release, index_path, tmp_path / "private" / "panel")


def test_callable_router_requires_verified_artifact_and_private_leaf_range(tmp_path, monkeypatch):
    e = evaluate()
    q = np.ones((33, 17))/17
    channel_path = tmp_path / "q.npz"; np.savez_compressed(channel_path, Q=q)
    partition_path = tmp_path / "partition.json"; partition_path.write_text('{"synthetic":true}')
    spec = {"Q": q, "channel_artifact_path": str(channel_path),
            "channel_artifact_sha256": _sha(channel_path),
            "router": lambda rows: np.full(len(rows["token_codes"]), 33, dtype=int),
            "router_artifact_path": str(partition_path),
            "router_sha256": _sha(partition_path),
            "router_entrypoint": "synthetic:router"}
    with pytest.raises(ValueError):
        e.token_law_for_release(spec, _rows("audit_fit"))


def test_callable_router_cannot_read_labels_hb_or_household(tmp_path):
    e = evaluate()
    q = np.ones((32, 17))/17
    channel_path = tmp_path / "q.npz"; np.savez_compressed(channel_path, Q=q)
    partition_path = tmp_path / "partition.json"; partition_path.write_text('{"synthetic":true}')
    seen = []
    def legal_router(rows):
        seen.append(set(rows))
        return rows["token_codes"]
    spec = {"Q": q, "channel_artifact_path": str(channel_path),
            "channel_artifact_sha256": _sha(channel_path),
            "router": legal_router,
            "router_artifact_path": str(partition_path),
            "router_sha256": _sha(partition_path),
            "router_entrypoint": "synthetic:legal_router"}
    law = e.token_law_for_release(spec, _rows("audit_fit"))
    assert law.shape == (4, 17)
    assert seen == [{"x", "ha", "token_codes", "teacher_p", "residual"}]
    spec["router_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        e.token_law_for_release(spec, _rows("audit_fit"))


def test_channel_bytes_must_match_pinned_artifact(tmp_path):
    e = evaluate()
    frozen = np.zeros((32, 17)); frozen[:, 0] = 1.
    changed = np.zeros((32, 17)); changed[:, 1] = 1.
    channel_path = tmp_path / "q.npz"; np.savez_compressed(channel_path, Q=frozen)
    spec = {"Q": changed, "channel_artifact_path": str(channel_path),
            "channel_artifact_sha256": _sha(channel_path), "router": "T0"}
    with pytest.raises(ValueError):
        e.token_law_for_release(spec, _rows("audit_fit"))


def _saved_task_only(tmp_path):
    from experiments.pcrl_adaptive_release_v1 import roles, task_baselines

    households = []
    index = 0
    while len(households) < 40:
        name = f"synthetic-task-fit-{index}"
        if roles.role_of(name) == "nuisance_train":
            households.append(name)
        index += 1
    x = np.zeros((40, 32)); ha = np.zeros((40, 4))
    x[:, 0] = np.linspace(-2, 2, 40)
    ha[:, 0] = np.linspace(-2, 2, 40)
    rows = {"x": x, "ha": ha, "weights": np.ones(40),
            "households": np.array(households),
            "labels": {"same_residence": (ha[:, 0] > 0).astype(int)}}
    model, receipt = task_baselines.fit_task_only(rows, seed=41)
    directory = tmp_path / "private" / "task_only"
    task_baselines.save_task_only(directory, model, receipt)
    return model, directory


def test_task_only_person_law_uses_pinned_model_and_legal_local_inputs_only(tmp_path):
    from experiments.pcrl_adaptive_release_v1 import task_baselines
    from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs

    e = evaluate()
    model, directory = _saved_task_only(tmp_path)
    rows = _rows("audit_fit")
    rows["token_codes"][:] = 0  # same historical code, different legal X_A/H_A
    rows["x"][:, 0] = [-1.5, -.5, .5, 1.5]
    rows["ha"][:, 0] = [-1.5, -.5, .5, 1.5]
    spec = {"task_only_model_dir": str(directory),
            "task_only_receipt_sha256": _sha(directory / "TASK_ONLY.json"),
            "mode": "constant_replacement", "publish": .5}
    actual = e.token_law_for_release(spec, rows)
    expected = task_baselines.task_only_control_law(
        model, RuntimeInputs(rows["x"], rows["ha"]),
        mode="constant_replacement", publish=.5, constant_token=0)
    assert actual.shape == (4, 17)
    assert np.array_equal(actual, expected)
    assert len(np.unique(actual, axis=0)) > 1
    changed_hidden = {**rows, "hb": np.full((4, 2), 99.),
                      "households": np.array(["changed"]*4),
                      "labels": {"same_residence": np.ones(4),
                                 "SEX": np.ones(4), "RAC1P": np.ones(4)}}
    assert np.array_equal(actual, e.token_law_for_release(spec, changed_hidden))
    rr = {**spec, "mode": "randomized_response", "publish": .5}
    assert np.allclose(e.token_law_for_release(rr, rows),
                       .5*np.eye(17)[model.token_codes(RuntimeInputs(rows["x"], rows["ha"]))]+.5/17)
    tampered = {**spec, "task_only_receipt_sha256": "0"*64}
    with pytest.raises(ValueError, match="pin|artifact|hash"):
        e.token_law_for_release(tampered, rows)


def test_inner_panel_audits_task_only_person_law_without_publishing_it(tmp_path, monkeypatch):
    e = evaluate()
    _, directory = _saved_task_only(tmp_path)
    index_path = tmp_path / "index.json"
    index_path.write_text('{"synthetic":true}')
    monkeypatch.setattr(e.data, "index", lambda path: {"synthetic": True})
    monkeypatch.setattr(e.data, "_sanitized_record", lambda index, anchor: (index_path, {}))
    monkeypatch.setattr(e.data, "load_prepared", lambda index, anchor: {"anchor": anchor})
    pools = {name: _rows(name) for name in ("audit_fit", "inner_selection", "inner_check")}
    for rows in pools.values():
        rows["x"][:, 0] = [-1.5, -.5, .5, 1.5]
        rows["ha"][:, 0] = [-1.5, -.5, .5, 1.5]
        rows["token_codes"][:] = 0
    monkeypatch.setattr(e.roles, "pooled_role", lambda prepared, role: pools[role])
    seen_laws = []
    def fit(fit_rows, fit_p, selection_rows, selection_p, role,
            output_dir, seed, *, release_id, slate):
        if release_id == "task_rr":
            seen_laws.append(fit_p.copy())
            assert fit_p.shape == (4, 17)
        return _fake_registry(role, release_id, slate)
    monkeypatch.setattr(e.audit, "fit_role_slate", fit)
    monkeypatch.setattr(e.audit, "select_frozen_routes", lambda rows, law, role,
            routes, *, release_id: {"role": role, "release_id": release_id,
            "route": next(iter(routes.values())), "selected": "m",
            "scores": {"m": {"balanced": .5}}, "candidate_count": len(routes),
            "rule": "synthetic"})
    def score(rows, law, lock):
        n = len(rows["ids"])
        loss = np.full(n, .6 if lock["release_id"] == "H" else .5)
        return {"ids": rows["ids"], "households": rows["households"],
                "weights": rows["weights"], "loss": loss,
                "scores": e.audit.score_weightings(loss, rows["weights"])}
    monkeypatch.setattr(e.audit, "score_frozen_route", score)
    spec = {"task_only_model_dir": str(directory),
            "task_only_receipt_sha256": _sha(directory / "TASK_ONLY.json"),
            "mode": "randomized_response", "publish": .5}
    report = e.audit_panel(0, {"task_rr": spec}, index_path,
                           tmp_path / "private" / "task_panel")
    assert len(seen_laws) == 5
    assert len(np.unique(seen_laws[0], axis=0)) > 1
    assert report["releases"]["task_rr"]["source"]["law_kind"] == "task_only_original_person_17_token"
    assert "task_baselines.py" in report["source_code_sha256"]
    public_text = (tmp_path / "private" / "task_panel" / "INNER_AUDIT.json").read_text()
    assert "token_law" not in public_text
    assert "person_probability_rows" not in public_text
