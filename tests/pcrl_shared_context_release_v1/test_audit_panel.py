"""Per-person audit panel on a tiny synthetic anchor; no ACS rows, no outer rows."""
from __future__ import annotations

import hashlib
import json
import sys
import types

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import roles
from experiments.pcrl_shared_context_release_v1 import audit_panel, laws

ROLE_NAMES = audit_panel.POOL_ROLES


def _households(role, count, start=0):
    found, i = [], start
    while len(found) < count:
        name = f"hh{i}"
        if roles.role_of(name) == role:
            found.append(name)
        i += 1
    return found


def synthetic_pools(n=48, seed=0):
    rng = np.random.default_rng(seed)
    pools = {}
    for position, role in enumerate(ROLE_NAMES):
        houses = np.repeat(_households(role, n // 2), 2)
        ha = rng.normal(size=(n, 4))
        sex = (ha[:, 0] + rng.normal(scale=.5, size=n) > 0).astype(int)
        pools[role] = {
            "x": rng.normal(size=(n, 32)), "ha": ha, "hb": rng.normal(size=(n, 2)),
            "token_codes": rng.integers(0, 32, n), "teacher_p": rng.uniform(.1, .9, n),
            "residual": rng.normal(size=n), "risk": rng.uniform(size=(n, 11)),
            "labels": {"same_residence": (ha[:, 1] > 0).astype(int), "SEX": sex,
                       "RAC1P": rng.integers(0, 9, n)},
            "weights": rng.uniform(.5, 2., n),
            "ids": np.array([f"{position}-{i}" for i in range(n)]),
            "households": houses}
    return pools


def d17_file(tmp_path):
    q = np.zeros((32, 17))
    q[np.arange(32), np.arange(32) % 17] = 1.
    path = tmp_path / "D17.npz"
    np.savez(path, Q=q)
    return laws.historical_spec("D17", 0, map_path=path, map_sha256=laws.sha256_file(path)), q


def nested_unit(tmp_path, monkeypatch, name, law_fn, module="release", kind="nested"):
    fake = sys.modules.get(f"{laws.PACKAGE}.{module}")
    if not isinstance(fake, types.ModuleType) or not hasattr(fake, "_registry"):
        fake = types.ModuleType(f"{laws.PACKAGE}.{module}")
        fake._registry = {}
        fake.load_law = lambda unit_dir: fake._registry[str(unit_dir).rsplit("/", 1)[-1]]
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
    fake._registry[name] = law_fn
    root = tmp_path / name
    root.mkdir()
    (root / "B.npz").write_bytes(name.encode())
    laws.write_descriptor(root, kind, release_id=name, anchor=0)
    return root


def mock_slate(monkeypatch):
    fits = []

    def fit(fit_rows, fit_p, sel_rows, sel_p, role, output_dir, seed, *, release_id, slate):
        fits.append((role, release_id, fit_p.shape[1], seed))
        _, view, target = audit_panel.inherited_audit.parse_role(role)
        return {"role": role, "release_id": release_id, "slate": slate,
                "fit_missing_classes": [], "validation_missing_classes": [],
                "models": {"m": {"kind": "model", "target": target, "source_view": view,
                                 "wire": "H" if release_id == "H" or view == "B" else "release",
                                 "source_release_id": release_id, "model_sha256": "a" * 64}}}

    def select(rows, law, role, routes, *, release_id):
        return {"role": role, "release_id": release_id, "route": next(iter(routes.values())),
                "selected": next(iter(routes)), "scores": {k: {"balanced": .5} for k in routes},
                "candidate_count": len(routes), "rule": "synthetic"}

    def score(rows, law, lock):
        target = lock["role"].split("/")[1]
        mask = rows["labels"][target] >= 0
        loss = np.full(mask.sum(), .6 if lock["release_id"] == "H" else .5)
        return {"ids": rows["ids"][mask], "households": rows["households"][mask],
                "weights": rows["weights"][mask], "loss": loss,
                "scores": audit_panel.ar_audit.score_weightings(loss, rows["weights"][mask])}
    monkeypatch.setattr(audit_panel.ar_audit, "fit_role_slate", fit)
    monkeypatch.setattr(audit_panel.ar_audit, "select_frozen_routes", select)
    monkeypatch.setattr(audit_panel.ar_audit, "score_frozen_route", score)
    return fits


def test_alias_ledger_hashes_per_person_laws_on_role_rows(tmp_path, monkeypatch):
    pools = synthetic_pools()
    d17_spec, q = d17_file(tmp_path)
    collapsed = nested_unit(tmp_path, monkeypatch, "NM1_U",
                            lambda legal: q[legal["token_codes"]].copy())
    mixed = nested_unit(tmp_path, monkeypatch, "NM4_U",
                        lambda legal: .9 * q[legal["token_codes"]] + .1 / 17)
    rd_same = nested_unit(tmp_path, monkeypatch, "RD_TASK",
                          lambda legal: q[legal["token_codes"]].copy(), module="rd",
                          kind="deterministic_policy")
    canonical, ledger, _ = audit_panel.canonicalize(
        {"D17": d17_spec, "NM1_U": collapsed, "NM4_U": mixed, "RD_TASK": rd_same}, pools)
    assert list(canonical) == ["D17", "NM4_U"]
    declared = ledger["declared"]
    assert declared["NM1_U"]["canonical"] == "D17" and declared["RD_TASK"]["canonical"] == "D17"
    assert declared["NM1_U"]["max_abs_difference_to_canonical"] == 0.
    assert declared["NM4_U"]["nearest_distinct_canonical"]["release_id"] == "D17"
    assert set(declared["D17"]["law_sha256_by_role"]) == set(ROLE_NAMES)
    # Same law on different people is a different identity.
    other = synthetic_pools(seed=1)
    _, ledger2, _ = audit_panel.canonicalize({"D17": d17_spec}, other)
    assert ledger2["declared"]["D17"]["law_identity"] != declared["D17"]["law_identity"]


def test_mocked_panel_audits_each_distinct_law_once_with_common_seeds(tmp_path, monkeypatch):
    fits = mock_slate(monkeypatch)
    pools = synthetic_pools()
    d17_spec, q = d17_file(tmp_path)
    sources = {"D17": d17_spec,
               "NM1_U": nested_unit(tmp_path, monkeypatch, "NM1_U",
                                    lambda legal: q[legal["token_codes"]].copy()),
               "NM4_U": nested_unit(tmp_path, monkeypatch, "NM4_U",
                                    lambda legal: .9 * q[legal["token_codes"]] + .1 / 17)}
    out = tmp_path / "private" / "panel"
    report = audit_panel.run_panel(1, pools, sources, out, index_sha256="0" * 64)
    assert len(fits) == 7 + 5 * 2  # shared H ancestors + two canonical releases
    assert sorted(report["releases"]) == ["D17", "NM4_U"]
    assert report["logical_to_canonical"]["NM1_U"] == "D17"
    assert report["releases"]["D17"]["aliases"] == ["D17", "NM1_U"]
    seeds = {(role, rid): seed for role, rid, _, seed in fits}
    assert seeds[("attack:AB/SEX", "D17")] == seeds[("attack:AB/SEX", "NM4_U")] == \
        audit_panel.SEED_BASE + 1000 + 3
    leaf = report["hist_gb_min_leaf"]
    assert leaf["D17"]["attack:A/SEX"]["effective_min_samples_leaf"] == {"hist_gb_20": 20, "hist_gb_5": 5}
    assert leaf["NM4_U"]["attack:A/SEX"]["max_support"] == 17
    assert leaf["NM4_U"]["attack:A/SEX"]["effective_min_samples_leaf"]["hist_gb_20"] == 340
    assert leaf["NM4_U"]["attack:A/SEX"]["coarsened"] is True
    assert report["outer_pool_opened"] is False
    receipt = json.loads((out / "COMPLETE.json").read_text())
    assert receipt["declared_ids"] == ["D17", "NM1_U", "NM4_U"]
    assert receipt["artifacts"]["ALIAS_LEDGER.json"] == laws.sha256_file(out / "ALIAS_LEDGER.json")
    audit_panel.verify_complete(out)
    with pytest.raises(FileExistsError):
        audit_panel.run_panel(1, pools, sources, out, index_sha256="0" * 64)


def test_panel_refuses_outer_or_misassigned_rows(tmp_path, monkeypatch):
    mock_slate(monkeypatch)
    pools = synthetic_pools()
    d17_spec, _ = d17_file(tmp_path)
    with pytest.raises(PermissionError):
        audit_panel.run_panel(0, {**pools, "outer_assessment": pools["inner_check"]},
                              {"D17": d17_spec}, tmp_path / "private" / "a", index_sha256="0" * 64)
    swapped = {**pools, "inner_check": pools["audit_fit"]}
    with pytest.raises(PermissionError):
        audit_panel.run_panel(0, swapped, {"D17": d17_spec}, tmp_path / "private" / "b",
                              index_sha256="0" * 64)
    with pytest.raises(ValueError, match="H-only"):
        audit_panel.canonicalize({"H": d17_spec}, pools)


def test_outer_gate_requires_coordinator_lock_and_remote_unlock(tmp_path, monkeypatch):
    lock = tmp_path / "SELECTION_LOCK.json"
    unlock = tmp_path / "OUTER_UNLOCK.json"
    kwargs = {"lock_path": lock, "unlock_path": unlock}
    with pytest.raises(PermissionError):
        audit_panel.verify_outer_gate(lock, "0" * 64, **kwargs)
    lock.write_text(json.dumps({"schema": audit_panel.LOCK_SCHEMA, "status": "LOCKED",
                                "study": audit_panel.STUDY, "outer_assessment_authorized": True,
                                "anchors": {}}))
    sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    with pytest.raises(PermissionError, match="unlock receipt"):
        audit_panel.verify_outer_gate(lock, sha, **kwargs)
    receipt = {"schema": audit_panel.UNLOCK_SCHEMA, "lock_sha256": sha, "remote_verified": False,
               "branch": audit_panel.BRANCH, "remote_commit_sha": "a" * 40}
    unlock.write_text(json.dumps(receipt))
    with pytest.raises(PermissionError, match="remotely verified"):
        audit_panel.verify_outer_gate(lock, sha, **kwargs)
    unlock.write_text(json.dumps({**receipt, "remote_verified": True}))
    checked = []
    monkeypatch.setattr(audit_panel, "remote_lock_check",
                        lambda commit, digest: checked.append((commit, digest)))
    assert audit_panel.verify_outer_gate(lock, sha, **kwargs)["status"] == "LOCKED"
    assert checked == [("a" * 40, sha)]  # the receipt's flag alone is never trusted
    with pytest.raises(PermissionError):
        audit_panel.verify_outer_gate(lock, "f" * 64, **kwargs)
    with pytest.raises(PermissionError):  # an unregistered lock path is refused
        audit_panel.verify_outer_gate(lock, sha)


def test_real_standard_slate_on_tiny_anchor_records_min_leaf(tmp_path, monkeypatch):
    """Unmocked AR/TAC/TDR slate end to end on 48 synthetic people per role."""
    pools = synthetic_pools()
    d17_spec, q = d17_file(tmp_path)
    sources = {"D17": d17_spec,
               "NM4_U": nested_unit(tmp_path, monkeypatch, "NM4_U",
                                    lambda legal: .5 * q[legal["token_codes"]]
                                    + .5 * np.eye(17)[legal["token_codes"] % 3])}
    out = tmp_path / "private" / "panel"
    report = audit_panel.run_panel(0, pools, sources, out, index_sha256="0" * 64)
    record = report["hist_gb_min_leaf"]["NM4_U"]["attack:A/SEX"]
    assert record["observed_min_samples_leaf"] == record["effective_min_samples_leaf"]
    assert report["hist_gb_min_leaf"]["D17"]["attack:A/SEX"]["observed_min_samples_leaf"] == \
        {"hist_gb_20": 20, "hist_gb_5": 5}
    for release in report["releases"].values():
        for role in release["roles"].values():
            assert np.isfinite(role["candidate"]["U"]) and np.isfinite(role["H"]["U"])
    audit_panel.verify_complete(out)


def test_outer_j_runs_only_behind_this_study_gate(tmp_path, monkeypatch):
    """No lock/unlock -> refused before any index, prepared or route is opened."""
    opened = []
    monkeypatch.setattr(audit_panel.data, "index", lambda *a, **k: opened.append("index"))
    monkeypatch.setattr(audit_panel.data, "load_prepared", lambda *a, **k: opened.append("prepared"))
    with pytest.raises(PermissionError):
        audit_panel.score_outer_j(0, tmp_path / "index.json", tmp_path / "private" / "j",
                                  audit_panel.LOCK_PATH, "0" * 64, tmp_path / "private" / "out")
    assert opened == [] and not (tmp_path / "private" / "out").exists()


def _git(repo, *args):
    import subprocess
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


def test_remote_lock_check_against_a_real_origin(tmp_path):
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(tmp_path, "init", "-q", "-b", "main", str(work))
    _git(work, "remote", "add", "origin", str(origin))
    lock_file = work / audit_panel.LOCK_RELATIVE
    lock_file.parent.mkdir(parents=True)
    lock_file.write_text('{"status": "LOCKED"}\n')
    _git(work, "add", ".")
    _git(work, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "lock")
    commit = _git(work, "rev-parse", "HEAD")
    sha = hashlib.sha256(lock_file.read_bytes()).hexdigest()
    with pytest.raises(Exception):  # not pushed yet
        audit_panel.remote_lock_check(commit, sha, repo=work)
    _git(work, "push", "-q", "origin", f"HEAD:refs/heads/{audit_panel.BRANCH}")
    evidence = audit_panel.remote_lock_check(commit, sha, repo=work)
    assert evidence["remote_tip"] == commit
    with pytest.raises(PermissionError, match="differs"):
        audit_panel.remote_lock_check(commit, "0" * 64, repo=work)


def test_score_outer_alias_guard(tmp_path, monkeypatch):
    """A law aliased on inner rows but different on outer rows is scored separately."""
    from experiments.pcrl_adaptive_release_v1 import outer_audit

    mock_slate(monkeypatch)
    pools = synthetic_pools()
    d17_spec, q = d17_file(tmp_path)
    shifted = lambda legal: np.where((legal["x"][:, 0] > 5)[:, None],
                                     np.full((len(legal["x"]), 17), 1 / 17), q[legal["token_codes"]])
    sources = {"D17": d17_spec,
               "NM1_U": nested_unit(tmp_path, monkeypatch, "NM1_U", shifted),
               "RD_TASK": nested_unit(tmp_path, monkeypatch, "RD_TASK",
                                      lambda legal: q[legal["token_codes"]].copy())}
    inner_dir = tmp_path / "private" / "panel"
    inner = audit_panel.run_panel(0, pools, sources, inner_dir, index_sha256="0" * 64)
    assert inner["logical_to_canonical"] == {"D17": "D17", "NM1_U": "D17", "RD_TASK": "D17"}
    locked = {"anchors": {"0": {"inner_panel_complete_sha256": audit_panel._sha(inner_dir / "COMPLETE.json"),
                                "inner_audit_sha256": audit_panel._sha(inner_dir / "INNER_AUDIT.json")}}}
    monkeypatch.setattr(outer_audit, "reconstruct_selected_routes", lambda root, report, rid: (
        {r: {"role": r, "release_id": rid, "selected": "m"} for r in audit_panel.ar_audit.ROLES},
        {r: {"role": r, "release_id": "H", "selected": "m"} for r in audit_panel.ar_audit.ROLES}))
    index = tmp_path / "index.json"
    index.write_text("{}")
    monkeypatch.setattr(audit_panel, "_sha", lambda p: "0" * 64 if str(p) == str(index.resolve())
                        else laws.sha256_file(p))
    locked["anchors"]["0"] = {"inner_panel_complete_sha256": laws.sha256_file(inner_dir / "COMPLETE.json"),
                              "inner_audit_sha256": laws.sha256_file(inner_dir / "INNER_AUDIT.json")}
    outer_rows = synthetic_pools(seed=5)["inner_check"]
    outer_rows["x"][:3, 0] = 10.
    report = audit_panel.score_outer(0, sources, index, inner_dir, "lock", "f" * 64,
                                     tmp_path / "private" / "outer", _gate=lambda p, s: locked,
                                     _load_outer_rows=lambda: outer_rows)
    assert report["outer_alias_confirmed"] == {"RD_TASK": "D17"}
    assert report["outer_alias_breaks"]["NM1_U"]["outer_people_differing"] == 3
    assert report["releases"]["NM1_U"]["scored_separately_from"] == "D17"
    assert set(report["releases"]) == {"D17", "NM1_U"}
    with pytest.raises(ValueError, match="exactly the inner panel"):
        audit_panel.score_outer(0, {"D17": d17_spec}, index, inner_dir, "lock", "f" * 64,
                                tmp_path / "private" / "outer2", _gate=lambda p, s: locked,
                                _load_outer_rows=lambda: outer_rows)
