"""Tests for dpc.closeout (data/custody owner): drive detection by content, versioned copies, same-device labelling,
dry runs that write nothing, policy re-encoding from a copy, and redirection of the osf custody routines away from
the closed osf results directory. All fixtures live in pytest's tmp_path; no private store is written.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q dpc/tests/test_closeout.py
"""
from __future__ import annotations

import hashlib
import json
import time
import types
from pathlib import Path

import numpy as np
import pytest

from dpc import closeout as CO

SUMS = b"deadbeef  smf_v1/run/units/x/COMPLETE.json\n"
WANT = hashlib.sha256(SUMS).hexdigest()


def _volumes(tmp: Path, match=True):
    root = tmp / "Volumes"
    for n in ("Macintosh HD", "BackgroundSyncService Setup", "Unrelated Stick", "Backup Disk"):
        (root / n).mkdir(parents=True)
    # a matching file on a SKIPPED volume must never be selected (skipped by name before any filesystem call)
    (root / "Macintosh HD" / CO.SMF_DRIVE_NAME).mkdir()
    (root / "Macintosh HD" / CO.SMF_DRIVE_NAME / "SHA256SUMS").write_bytes(SUMS)
    (root / "Unrelated Stick" / CO.SMF_DRIVE_NAME).mkdir()
    (root / "Unrelated Stick" / CO.SMF_DRIVE_NAME / "SHA256SUMS").write_bytes(b"other")
    if match:
        (root / "Backup Disk" / CO.SMF_DRIVE_NAME).mkdir()
        (root / "Backup Disk" / CO.SMF_DRIVE_NAME / "SHA256SUMS").write_bytes(SUMS)
    return root


def test_locate_drive_by_content_and_names_withheld(tmp_path):
    root = _volumes(tmp_path)
    vol, ev = CO.locate_drive(root, WANT)
    assert vol == root / "Backup Disk"
    assert ev["skipped_by_name"] == 2 and ev["candidate_volumes"] == 2 and ev["matching_volumes"] == 1
    txt = json.dumps(ev)
    assert "Backup Disk" not in txt and "Unrelated" not in txt and "Macintosh" not in txt


def test_locate_drive_absent(tmp_path):
    vol, ev = CO.locate_drive(_volumes(tmp_path, match=False), WANT)
    assert vol is None and ev["mounted"] is False


def test_versioned_never_reuses(tmp_path):
    a = CO.versioned(tmp_path, "x")
    a.mkdir()
    b = CO.versioned(tmp_path, "x")
    assert b.name == "x_v2"
    with pytest.raises(SystemExit):
        CO.copy_tree(tmp_path, [], a)


def _store(tmp: Path):
    src = tmp / "PCRL_eval_cache_private" / "dpc_v1"
    (src / "run" / "units" / "u").mkdir(parents=True)
    (src / "run" / "units" / "u" / "f.bin").write_bytes(b"x" * 100)
    (src / "run" / "units" / "u" / "f.bin.tmp").write_bytes(b"partial")          # in-progress write: excluded
    (src / "START.txt").write_text("t0\n")
    return src


def _stub_restore(monkeypatch):
    monkeypatch.setattr(CO, "restore_all", lambda copy, live, targets, seed: (
        {"teacher U (seed 1)": {"status": "PASS", "copy_seen": (copy / "START.txt").exists()},
         "final attacker": {"status": "PENDING"}}, {"wall_s": 0.0, "cpu_s": 0.0}))


def test_same_device_copy_is_labelled_and_never_a_drive_restore(tmp_path, monkeypatch):
    src = _store(tmp_path)
    out = tmp_path / "pkg"
    _stub_restore(monkeypatch)
    bv = CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path, match=False), out_pkg=out)
    assert bv["status"] == "LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING"
    assert "not a drive restore" in bv["restore_kind"] and bv["off_device_backup"].startswith("PENDING")
    assert bv["files"] == 2 and bv["uncached_readback_match"] == 2
    date = time.strftime("%Y%m%d", time.gmtime())
    root = src.parent / f"dpc_v1_local_copy_{date}"
    assert (root / "dpc_v1" / "START.txt").exists() and not (root / "dpc_v1" / "run" / "units" / "u" / "f.bin.tmp").exists()
    lines = (root / "SHA256SUMS").read_text().splitlines()
    assert all(ln.split("  ")[1].startswith("dpc_v1/") for ln in lines) and len(lines) == 2
    pub = (out / "BACKUP_VERIFICATION.json").read_text() + (out / "RESTORE_INDEX.json").read_text()
    assert "/Users/" not in pub and "<PRIVATE_CACHE>" in pub and "DRIVE_COPY_VERIFIED" not in pub
    assert bv["restore_checks"]["teacher U (seed 1)"]["copy_seen"]
    bv2 = CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path / "b", match=False), out_pkg=out)
    assert bv2["destination"].endswith(f"dpc_v1_local_copy_{date}_v2/dpc_v1")


def test_drive_copy_goes_to_the_content_matched_volume(tmp_path, monkeypatch):
    src = _store(tmp_path)
    vols = _volumes(tmp_path)
    monkeypatch.setattr(CO, "smf_recorded_sums_sha", lambda: WANT)
    _stub_restore(monkeypatch)
    bv = CO.backup(src=src, cache=src.parent, volumes_root=vols, out_pkg=tmp_path / "pkg")
    date = time.strftime("%Y%m%d", time.gmtime())
    assert bv["status"] == "DRIVE_COPY_VERIFIED" and bv["destination"] == f"<DRIVE_ROOT>/private_dpc_v1_{date}/dpc_v1"
    assert (vols / "Backup Disk" / f"private_dpc_v1_{date}" / "dpc_v1" / "START.txt").exists()
    assert not (vols / "Unrelated Stick" / f"private_dpc_v1_{date}").exists()
    assert (vols / "Backup Disk" / f"private_dpc_v1_{date}" / "BACKUP_RECORD.json").exists()


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    src = _store(tmp_path)
    out = tmp_path / "pkg"
    monkeypatch.setattr(CO, "rehearse", lambda s, t, k: {"statuses": {}})
    before = sorted(p.name for p in src.parent.iterdir())
    plan = CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path, match=False), out_pkg=out,
                     dry_run=True)
    assert plan["same_device"] and sorted(p.name for p in src.parent.iterdir()) == before and not out.exists()


def test_targets_must_be_unit_names(tmp_path):
    t = tmp_path / "t.json"
    t.write_text(json.dumps({"policies": {"x": "../../etc/passwd"}}))
    with pytest.raises(SystemExit):
        CO.load_targets(t)


# ---------------------------------------------------------------- policy re-encoding from a copy (toy policy)
def toy_loader(path):
    z = json.loads(Path(path).read_text())
    return z["p1"], z["p2"]


def toy_encode(pol, P, d):
    """class-preserving toy code: token = 2 * decision + (max prob above the policy threshold)."""
    hi = (P.max(1) > pol["thr"]).astype(np.int64)
    tok = 2 * d + hi
    q = np.full(P.shape, 0.0)
    q[np.arange(len(d)), d] = 1.0
    return tok, q, d.copy()


def _policy_unit(root: Path, name, T, thr=(0.7, 0.4), tamper=False):
    d = root / "run" / "units" / name
    d.mkdir(parents=True)
    pol = {"p1": {"thr": thr[0]}, "p2": {"thr": thr[1]}}
    (d / "policy.json").write_text(json.dumps(pol))
    out = {}
    for i, p in ((1, pol["p1"]), (2, pol["p2"])):
        tok, q, h = toy_encode(p, T[f"p{i}"], T[f"d{i}"])
        out.update({f"tok{i}": tok, f"q{i}": q, f"hard{i}": h})
    if tamper:
        out["tok2"] = out["tok2"].copy()
        out["tok2"][0] ^= 1
    np.savez(d / "release.npz", **out)
    (d / "record.json").write_text(json.dumps({"seed": 1, "cfg": {"teacher": "U"}, "config": "toy"}))
    files = {f: CO.sha(d / f) for f in ("policy.json", "release.npz", "record.json")}
    (d / "COMPLETE.json").write_text(json.dumps({"files": files}))
    return d


def _teacher(n=50, seed=0):
    rng = np.random.default_rng(seed)
    P1 = rng.dirichlet(np.ones(2), n)
    P2 = rng.dirichlet(np.ones(6), n)
    return {"p1": P1, "p2": P2, "d1": P1.argmax(1), "d2": P2.argmax(1)}


def test_restore_policy_reencodes_and_detects_differences(tmp_path):
    T = _teacher()
    copy, live = tmp_path / "copy", tmp_path / "live"
    for root in (copy, live):
        _policy_unit(root, "pol__s1__toy", T)
    loader = "dpc.tests.test_closeout:toy_loader"
    r = CO.restore_policy(copy, live, "pol__s1__toy", {("U", 1): T}, loader, toy_encode)
    assert r["status"] == "PASS" and all(r["class_preserved_vs_restored_teacher"].values())
    # a saved release that the backed-up policy does not reproduce (hash-consistent) must FAIL
    _policy_unit(copy, "pol__s1__bad", T, tamper=True)
    _policy_unit(live, "pol__s1__bad", T, tamper=True)
    r = CO.restore_policy(copy, live, "pol__s1__bad", {("U", 1): T}, loader, toy_encode)
    assert r["status"] == "FAIL" and r["vs_copy_release"]["tok2"]["mismatches"] == 1
    # a corrupted copied file fails before anything is unpickled
    (copy / "run" / "units" / "pol__s1__toy" / "release.npz").write_bytes(b"corrupt")
    r = CO.restore_policy(copy, live, "pol__s1__toy", {("U", 1): T}, loader, toy_encode)
    assert r["status"] == "FAIL" and r["copy_files_hash_ok_uncached"] is False
    # the teacher must have been restored from the copy
    r = CO.restore_policy(live, live, "pol__s1__toy", {}, loader, toy_encode)
    assert r["status"] == "FAIL" and "not restored" in r["reason"]


def test_attacker_restore_pending_without_entry_point(tmp_path):
    assert CO.restore_attacker(tmp_path, tmp_path, None, {})["status"] == "PENDING"


def toy_attacker(units_root, D, scale=1.0):
    return np.load(Path(units_root) / "att" / "preds.npz")["P"] * scale


def test_attacker_restore_compares_with_saved(tmp_path):
    for root in (tmp_path / "copy", tmp_path / "live"):
        (root / "run" / "units" / "att").mkdir(parents=True)
        np.savez(root / "run" / "units" / "att" / "preds.npz", P=np.linspace(0, 1, 11))
    spec = {"fn": "dpc.tests.test_closeout:toy_attacker", "saved": {"unit": "att", "key": "P"}}
    assert CO.restore_attacker(tmp_path / "copy", tmp_path / "live", spec, {})["status"] == "PASS"
    spec["kwargs"] = {"scale": 0.5}
    r = CO.restore_attacker(tmp_path / "copy", tmp_path / "live", spec, {})
    assert r["status"] == "FAIL" and r["max_abs_diff_vs_saved_copy"] == pytest.approx(0.5)


# ---------------------------------------------------------------- predecessor custody redirection
def _fake_osf(tmp: Path):
    pkg = tmp / "results" / "pcrl_online_strength_frontier_v1"
    pkg.mkdir(parents=True)

    def write_public(path, obj):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj))

    OC = types.SimpleNamespace(PKG=pkg, OUT_REPAIR=pkg / "PREDECESSOR_CUSTODY_REPAIR.json",
                               REPLAY_OUT=pkg / "predecessor_custody" / "R.json", write_public=write_public)

    def backup(dest=None, seed=1, **kw):
        OC.write_public(OC.PKG / "BACKUP_VERIFICATION.json", {"status": "DRIVE_COPY_VERIFIED", "files": 3,
                                                               "uncached_readback_match": 3, "restore_checks": {}})
        OC.write_public(OC.PKG / "RESTORE_INDEX.json", {"x": 1})
        return {"status": "DRIVE_COPY_VERIFIED", "files": 3, "uncached_readback_match": 3, "restore_checks": {}}

    def predecessor(restore=False, dry_run=False):
        assert OC.REPLAY_OUT.parent.name == "pred"
        OC.write_public(OC.OUT_REPAIR, {"status": "RESTORED"})
        return {"status": "RESTORED", "unresolved_gap": []}

    OC.backup, OC.predecessor = backup, predecessor
    return OC


def test_redirected_writes_never_touch_the_closed_osf_directory(tmp_path):
    OC = _fake_osf(tmp_path)
    out = tmp_path / "pred"
    saved, writes = CO.redirected(OC, out)
    OC.write_public(OC.PKG / "BACKUP_VERIFICATION.json", {"a": 1})
    OC.write_public(OC.OUT_REPAIR, {"b": 2})
    assert (out / "osf_BACKUP_VERIFICATION.json").exists() and (out / "PREDECESSOR_CUSTODY_REPAIR.json").exists()
    assert not any(OC.PKG.iterdir())
    with pytest.raises(SystemExit):
        OC.write_public(OC.PKG / "RESEARCH_DECISION.json", {"c": 3})
    CO.restore_bindings(OC, saved)
    assert OC.OUT_REPAIR == OC.PKG / "PREDECESSOR_CUSTODY_REPAIR.json"


def test_predecessor_runs_both_routines_redirected_with_fake_drive(tmp_path, monkeypatch):
    OC = _fake_osf(tmp_path)
    monkeypatch.setattr(CO, "smf_recorded_sums_sha", lambda: WANT)
    monkeypatch.setattr(CO, "lock_pushed", lambda: (False, "test: no dpc EVALUATION_LOCK"))
    out = tmp_path / "pred"
    rec = CO.predecessor(restore=True, volumes_root=_volumes(tmp_path), out_dir=out, OC=OC)
    assert rec["status"] == "RUN" and rec["closed_results_unchanged"]
    assert rec["runs"]["osf_backup"]["status"] == "DRIVE_COPY_VERIFIED"
    assert rec["runs"]["osf_backup"]["attacker_restore_unsealed"] is False
    assert rec["runs"]["osf_predecessor_restore"]["status"] == "RESTORED"
    assert not any(OC.PKG.iterdir()) and (out / "STATUS.json").exists()
    assert OC.OUT_REPAIR == OC.PKG / "PREDECESSOR_CUSTODY_REPAIR.json"           # bindings restored


def test_predecessor_pending_when_drive_absent(tmp_path):
    out = tmp_path / "pred"
    rec = CO.predecessor(volumes_root=_volumes(tmp_path, match=False), out_dir=out)
    assert rec["status"] == "PENDING" and "osf_backup_and_predecessor_restore" in rec["pending_commands"]
    assert json.loads((out / "STATUS.json").read_text())["status"] == "PENDING"


def test_status_withholds_volume_names():
    txt = json.dumps(CO.status())
    assert "BackgroundSyncService" not in txt and "Macintosh" not in txt
