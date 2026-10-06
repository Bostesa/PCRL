"""Tests for qpc.closeout (data/custody owner E). Everything runs on synthetic fixtures in pytest's tmp_path: fake
volumes (drive detection by content, skipped names), a fake private store, stubbed restore checks, toy policies, a
synthetic teacher, and fake dpc/osf closeout modules. No private store, drive or closed results tree is written.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q qpc/tests/test_closeout.py
"""
from __future__ import annotations

import hashlib
import json
import time
import types
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dpc import admit as DA
from qpc import closeout as CO

SUMS = b"deadbeef  smf_v1/run/units/x/COMPLETE.json\n"
WANT = hashlib.sha256(SUMS).hexdigest()
DATE = time.strftime("%Y%m%d", time.gmtime())
REAL_MARKER = CO.smf_marker_sha                     # captured before the autouse fixture patches it


def _volumes(tmp: Path, match=True):
    root = tmp / "Volumes"
    for n in ("Macintosh HD", "BackgroundSyncService Setup", "Unrelated Stick", "Owner Named Disk"):
        (root / n).mkdir(parents=True)
    for skipped in ("Macintosh HD", "BackgroundSyncService Setup"):      # a match on a SKIPPED volume never counts
        (root / skipped / CO.SMF_DRIVE_NAME).mkdir()
        (root / skipped / CO.SMF_DRIVE_NAME / "SHA256SUMS").write_bytes(SUMS)
    (root / "Unrelated Stick" / CO.SMF_DRIVE_NAME).mkdir()
    (root / "Unrelated Stick" / CO.SMF_DRIVE_NAME / "SHA256SUMS").write_bytes(b"other")
    if match:
        d = root / "Owner Named Disk" / CO.SMF_DRIVE_NAME
        d.mkdir()
        (d / "SHA256SUMS").write_bytes(SUMS)
        (root / "Owner Named Disk" / "Owner Personal Folder").mkdir()     # an identifying name: never published
    return root


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Marker, closed trees and receipt folders all point into tmp_path."""
    monkeypatch.setattr(CO, "smf_marker_sha", lambda: WANT)
    closed = {k: tmp_path / "closed" / k for k in ("dpc_results", "osf_results", "smf_results")}
    for d in closed.values():
        d.mkdir(parents=True)
        (d / "R.json").write_text("{}")
    monkeypatch.setattr(CO, "CLOSED_RESULTS", closed)
    monkeypatch.setattr(CO, "PRED_DIR", tmp_path / "pkg" / "provenance" / "predecessor_custody")
    monkeypatch.setattr(CO, "DPC_DIR", tmp_path / "pkg" / "provenance" / "dpc_custody")
    monkeypatch.setattr(CO, "PKG", tmp_path / "pkg")
    monkeypatch.setattr(CO, "diskutil_probe", lambda: {"ran": False})
    monkeypatch.setattr(CO.QD, "evaluation_lock_pushed", lambda *a, **k: (False, "not pushed (test)"))
    return closed


def _store(tmp: Path):
    src = tmp / "PCRL_eval_cache_private" / "qpc_v1"
    (src / "run" / "units" / "u").mkdir(parents=True)
    (src / "run" / "units" / "u" / "f.bin").write_bytes(b"x" * 100)
    (src / "run" / "units" / "u" / "f.bin.tmp").write_bytes(b"partial")          # in-progress write: excluded
    (src / "START.txt").write_text("t0\n")
    return src


def _stub_restore(monkeypatch, status="PASS"):
    monkeypatch.setattr(CO, "restore_all", lambda copy, live, targets, seed: (
        {"teacher U (seed 1)": {"status": status, "copy_seen": (copy / "START.txt").exists(),
                                "copy_is_not_live": copy != live},
         "Q*": {"status": "PASS"}, "privacy/control code": {"status": "PASS"}, "attacker": {"status": "PASS"}},
        {"wall_s": 0.0, "cpu_s": 0.0}))


# ------------------------------------------------------------------ drive detection
def test_locate_drive_by_content_skips_names_and_withholds_them(tmp_path):
    root = _volumes(tmp_path)
    vol, ev = CO.locate_drive(root)
    assert vol == root / "Owner Named Disk"
    assert ev["skipped_by_name"] == 2 and ev["candidate_volumes"] == 2 and ev["matching_volumes"] == 1
    txt = json.dumps(ev)
    assert "Owner" not in txt and "Unrelated" not in txt and "Background" not in txt and "Macintosh" not in txt


def test_locate_drive_absent_and_skipped_image_never_read(tmp_path, monkeypatch):
    root = _volumes(tmp_path, match=False)
    seen = []
    orig = CO.DC.sha
    monkeypatch.setattr(CO.DC, "sha", lambda p, nocache=False: (seen.append(str(p)), orig(p, nocache))[1])
    vol, ev = CO.locate_drive(root)
    assert vol is None and ev["mounted"] is False
    assert not [s for s in seen if "BackgroundSyncService" in s or "Macintosh" in s]


def test_marker_is_read_at_the_pin():
    assert REAL_MARKER() == "28705e242f77b6333561f66117934b3ebc427072e31dc5a38b3a9d9e8f02cd05"


# ------------------------------------------------------------------ same-device and drive copies
def test_same_device_copy_is_labelled_pending_and_restored_from_the_copy(tmp_path, monkeypatch):
    src = _store(tmp_path)
    out = tmp_path / "pkg"
    _stub_restore(monkeypatch)
    bv = CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path, match=False), out_pkg=out)
    assert bv["status"] == CO.STATUS_LOCAL == "LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING"
    assert "not a drive restore" in bv["restore_kind"] and bv["off_device_backup"].startswith("PENDING")
    assert "not off-device custody" in bv["custody_gap"]
    assert bv["pending"] == CO.PENDING and all("<DRIVE_ROOT>" in v for v in CO.PENDING.values())
    assert bv["files"] == 2 and bv["uncached_readback_match"] == 2
    root = src.parent / f"qpc_v1_local_copy_{DATE}"
    assert (root / "qpc_v1" / "START.txt").exists() and not (root / "qpc_v1/run/units/u/f.bin.tmp").exists()
    lines = (root / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == 2 and all(ln.split("  ")[1].startswith("qpc_v1/") for ln in lines)
    assert CO.verify_sums(root)["pass"]
    pub = (out / "BACKUP_VERIFICATION.json").read_text() + (out / "RESTORE_INDEX.json").read_text()
    assert "/Users/" not in pub and str(tmp_path) not in pub and "<PRIVATE_CACHE>" in pub
    assert CO.STATUS_DRIVE not in pub and json.loads((out / "RESTORE_INDEX.json").read_text())["not_off_device"]
    chk = bv["restore_checks"]["teacher U (seed 1)"]
    assert chk["copy_seen"] and chk["copy_is_not_live"]
    assert bv["required_restores"] == {"U teacher": "PASS", "Q*": "PASS", "privacy/control code": "PASS",
                                       "attacker": "PASS"}
    assert (root / "BACKUP_RECORD.json").exists()
    bv2 = CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path / "b", match=False), out_pkg=out)
    assert bv2["destination"].endswith(f"qpc_v1_local_copy_{DATE}_v2/qpc_v1")
    assert (root / "qpc_v1" / "START.txt").exists()                                # the first copy is kept


def test_drive_copy_goes_to_the_content_matched_volume_only(tmp_path, monkeypatch):
    src = _store(tmp_path)
    vols = _volumes(tmp_path)
    _stub_restore(monkeypatch)
    bv = CO.backup(src=src, cache=src.parent, volumes_root=vols, out_pkg=tmp_path / "pkg")
    assert bv["status"] == CO.STATUS_DRIVE and bv["destination"] == f"<DRIVE_ROOT>/private_qpc_v1_{DATE}/qpc_v1"
    assert "custody_gap" not in bv and "Owner" not in json.dumps(bv)
    assert (vols / "Owner Named Disk" / f"private_qpc_v1_{DATE}" / "qpc_v1" / "START.txt").exists()
    assert not (vols / "Unrelated Stick" / f"private_qpc_v1_{DATE}").exists()


def test_backup_refuses_when_free_space_would_drop_below_5_gib(tmp_path, monkeypatch):
    src = _store(tmp_path)
    monkeypatch.setattr(CO, "free_gib", lambda p: 5.0)
    _stub_restore(monkeypatch)
    with pytest.raises(SystemExit, match="5 GiB"):
        CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path, match=False), out_pkg=tmp_path / "pkg")
    assert not list(src.parent.glob("qpc_v1_local_copy_*"))


def test_failed_restore_is_not_reported_as_passing(tmp_path, monkeypatch):
    src = _store(tmp_path)
    _stub_restore(monkeypatch, status="FAIL")
    bv = CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path, match=False), out_pkg=tmp_path / "pkg")
    assert bv["restore_all_pass"] is False and bv["required_restores"]["U teacher"] == "FAIL"


def test_dry_runs_write_nothing(tmp_path, monkeypatch):
    src = _store(tmp_path)
    out = tmp_path / "pkg"
    monkeypatch.setattr(CO, "rehearse", lambda s, t, k: {"statuses": {}})
    before = sorted(p.name for p in src.parent.iterdir())
    for vols in (_volumes(tmp_path / "a", match=False), _volumes(tmp_path / "b", match=True)):
        drive_before = sorted(p.name for p in (vols / "Owner Named Disk").iterdir())
        CO.backup(src=src, cache=src.parent, volumes_root=vols, out_pkg=out, dry_run=True)
        CO.dpc_backup(dry_run=True, volumes_root=vols)
        CO.predecessor(restore=True, dry_run=True, volumes_root=vols)
        CO.run_all(dry_run=True, volumes_root=vols, src=src, cache=src.parent, out_pkg=out)
        assert sorted(p.name for p in (vols / "Owner Named Disk").iterdir()) == drive_before
    assert sorted(p.name for p in src.parent.iterdir()) == before and not out.exists()


def test_targets_must_be_qpc_unit_names_and_entry_points(tmp_path):
    t = tmp_path / "t.json"
    for bad in ({"policies": {"x": "../../etc/passwd"}}, {"policies": {"x": "ref__s1__F"}},
                {"policies": {}, "encode": "os:system"}, {"policies": {}, "attacker": {"fn": "subprocess:run"}}):
        t.write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            CO.load_targets(t)
    t.write_text(json.dumps({"seed": 1, "policies": {"Q*": "pol__s1__U_DIRECT-TASK_i8o16"},
                             "attacker": {"fn": "qpc.audit:refit_selected_attacker"}}))
    assert CO.load_targets(t)["policies"]["Q*"] == "pol__s1__U_DIRECT-TASK_i8o16"


def test_verify_sums_detects_corruption(tmp_path):
    d = tmp_path / "f"
    (d / "a").mkdir(parents=True)
    (d / "a" / "x").write_bytes(b"hello")
    (d / "SHA256SUMS").write_text(f"{hashlib.sha256(b'hello').hexdigest()}  a/x\n")
    assert CO.verify_sums(d)["pass"]
    (d / "a" / "x").write_bytes(b"hellO")
    r = CO.verify_sums(d)
    assert not r["pass"] and r["entries"] == 1 and r["match"] == 0
    assert CO.verify_sums(tmp_path / "missing")["pass"] is False


# ------------------------------------------------------------------ restores (synthetic teacher, toy policy)
def _state(seed=0):
    g = torch.Generator().manual_seed(seed)
    st = {}
    for i in (0, 1):
        for j, (o, n) in zip((0, 2, 4), ((64, 83), (64, 64), (16, 64))):
            st[f"enc.{i}.{j}.weight"] = torch.randn(o, n, generator=g) * 0.2
            st[f"enc.{i}.{j}.bias"] = torch.randn(o, generator=g) * 0.1
    for i, K in enumerate((2, 6)):
        st[f"head.{i}.weight"] = torch.randn(K, 16, generator=g)
        st[f"head.{i}.bias"] = torch.randn(K, generator=g)
    return st


def _complete(d: Path):
    files = {str(p.relative_to(d)): DA.sha(p) for p in sorted(d.rglob("*")) if p.is_file() and p.name != "COMPLETE.json"}
    (d / "COMPLETE.json").write_text(json.dumps({"id": d.name, "files": files}))


def _synthetic_store(root: Path, k=1, n=200):
    """admitted rel__s{k}__U + run/units tea__s{k}__U, both hash-complete; returns (D, teacher outputs)."""
    rng = np.random.default_rng(k)
    D = {"X": rng.normal(size=(n, 83)), "row_id": np.arange(n, dtype=np.int64),
         "idx": {"OSF_DEFENSE_FIT": np.arange(120)}}
    st = _state(k)
    d = root / "admitted" / f"rel__s{k}__U"
    d.mkdir(parents=True)
    torch.save(st, d / "model.pt")
    H = DA.forward(st, D["X"])
    rel, T = {"row_id": D["row_id"]}, {"row_id": D["row_id"]}
    for i, K in enumerate((2, 6)):
        y = rng.integers(0, K, 120)
        y[:K] = np.arange(K)
        head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(H[i][:120], y)
        joblib.dump(head, d / f"head_{i}.joblib")
        c, P, hard = DA.head_outputs(head, H[i])
        rel.update({f"r{i + 1}": H[i], f"c{i + 1}": c, f"p{i + 1}": P, f"hard{i + 1}": hard})
        T.update({f"r{i + 1}": H[i], f"c{i + 1}": c, f"p{i + 1}": P, f"d{i + 1}": hard})
    np.savez(d / "release.npz", **rel)
    (d / "record.json").write_text(json.dumps({"seed": k}))
    _complete(d)
    u = root / "run" / "units" / f"tea__s{k}__U"
    u.mkdir(parents=True)
    np.savez(u / "teacher.npz", **T)
    (u / "record.json").write_text("{}")
    _complete(u)
    return D, T


def test_restore_teacher_from_copy_bitwise_and_vs_pinned_dpc_unit(tmp_path, monkeypatch):
    import shutil
    live = tmp_path / "live"
    D, T = _synthetic_store(live)
    copy = tmp_path / "copy"
    shutil.copytree(live, copy)
    monkeypatch.setattr(CO, "DPC_UNITS", live / "run" / "units")       # stands in for the pinned dpc teacher unit
    info, out = CO.restore_teacher(copy, live, "U", 1, D)
    assert info["status"] == "PASS" and all(c["bitwise"] for c in info["vs_pinned_dpc_teacher_unit"].values())
    assert np.array_equal(out["p2"], T["p2"])
    z = dict(np.load(copy / "run" / "units" / "tea__s1__U" / "teacher.npz"))
    z["p1"] = z["p1"].copy()
    z["p1"][3, 0] = np.nextafter(z["p1"][3, 0], 2.0)
    np.savez(copy / "run" / "units" / "tea__s1__U" / "teacher.npz", **z)
    info2, _ = CO.restore_teacher(copy, live, "U", 1, D)
    assert info2["status"] == "FAIL"


def test_restore_teacher_refuses_a_copy_that_does_not_verify(tmp_path):
    import shutil
    live = tmp_path / "live"
    D, _ = _synthetic_store(live)
    copy = tmp_path / "copy"
    shutil.copytree(live, copy)
    (copy / "admitted" / "rel__s1__U" / "head_0.joblib").write_bytes(b"tampered")
    info, out = CO.restore_teacher(copy, live, "U", 1, D)
    assert info["status"] == "FAIL" and out is None and "nothing unpickled" in info["reason"]


def toy_loader(path):
    z = json.loads(Path(path).read_text())
    return z["p1"], z["p2"]


def toy_encode(pol, P, d):
    hi = (P.max(1) > pol["thr"]).astype(np.int64)
    q = np.zeros(P.shape)
    q[np.arange(len(d)), d] = 1.0
    return 2 * d + hi, q, d.copy()


def _policy_unit(root: Path, name, T, thr=(0.7, 0.4)):
    d = root / "run" / "units" / name
    d.mkdir(parents=True)
    pol = {"p1": {"thr": thr[0]}, "p2": {"thr": thr[1]}}
    (d / "policy.json").write_text(json.dumps(pol))
    out = {"row_id": T["row_id"]}
    for i, p in ((1, pol["p1"]), (2, pol["p2"])):
        tok, q, h = toy_encode(p, T[f"p{i}"], T[f"d{i}"])
        out.update({f"tok{i}": tok, f"q{i}": q, f"hard{i}": h, f"alpha{i}": np.int64(4)})
    np.savez(d / "release.npz", **out)
    (d / "record.json").write_text(json.dumps({"seed": 1, "teacher": "U", "config": "U|DIRECT-TASK|i8o16"}))
    _complete(d)


def test_restore_policy_re_encodes_from_the_restored_teacher(tmp_path):
    import shutil
    live = tmp_path / "live"
    D, T = _synthetic_store(live)
    _policy_unit(live, "pol__s1__U_DIRECT-TASK_i8o16", T)
    copy = tmp_path / "copy"
    shutil.copytree(live, copy)
    info = CO.restore_policy(copy, live, "pol__s1__U_DIRECT-TASK_i8o16", {("U", 1): T}, D, encode=toy_encode,
                             loader=toy_loader, deploy=False)
    assert info["status"] == "PASS" and all(info["decision_preserved_vs_restored_teacher"].values())
    assert info["teacher"] == "U" and info["seed"] == 1
    # a restored teacher that differs by one decision is caught
    T2 = {**T, "d2": T["d2"].copy()}
    T2["d2"][0] = (T2["d2"][0] + 1) % 6
    bad = CO.restore_policy(copy, live, "pol__s1__U_DIRECT-TASK_i8o16", {("U", 1): T2}, D, encode=toy_encode,
                            loader=toy_loader, deploy=False)
    assert bad["status"] == "FAIL"
    # missing teacher -> FAIL with a reason, never a silent pass
    miss = CO.restore_policy(copy, live, "pol__s1__U_DIRECT-TASK_i8o16", {}, D, encode=toy_encode, loader=toy_loader)
    assert miss["status"] == "FAIL" and "not restored" in miss["reason"]


def test_unit_teacher_seed_parses_raw_j_names(tmp_path):
    (tmp_path / "pol__s2__RAW-J_b0.3_DIRECT-TASK_i4o8").mkdir()
    t, k, _ = CO._unit_teacher_seed(tmp_path, "pol__s2__RAW-J_b0.3_DIRECT-TASK_i4o8")
    assert (t, k) == ("RAW-J_b0.3", 2)


def test_required_restore_classes():
    st = {"teacher U (seed 1)": "PASS", "Q*": "PASS", "privacy/control code (U|LOCAL)": "FAIL", "attacker": "PENDING"}
    assert CO.required_targets_status(st) == {"U teacher": "PASS", "Q*": "PASS", "privacy/control code": "FAIL",
                                              "attacker": "PENDING"}
    assert CO.required_targets_status({})["Q*"].startswith("PENDING")


# ------------------------------------------------------------------ dpc backup and predecessor repair (redirected)
def test_dpc_backup_pending_without_drive_writes_only_the_qpc_receipt(tmp_path, _isolate):
    before = {k: CO.tree_state(v) for k, v in _isolate.items()}
    rec = CO.dpc_backup(volumes_root=_volumes(tmp_path, match=False))
    assert rec["status"] == "PENDING" and "dpc.closeout backup --dest <DRIVE_ROOT>" in rec["pending_command"]
    assert (CO.DPC_DIR / "STATUS.json").exists()
    assert {k: CO.tree_state(v) for k, v in _isolate.items()} == before


def test_dpc_backup_with_drive_redirects_receipts(tmp_path, _isolate):
    vols = _volumes(tmp_path)
    calls = {}

    def fake_backup(dest, targets_file, seed, out_pkg):
        calls.update(dest=dest, out_pkg=Path(out_pkg), seed=seed)
        (Path(dest) / f"private_dpc_v1_{DATE}").mkdir()
        return {"status": "DRIVE_COPY_VERIFIED", "files": 3, "uncached_readback_match": 3,
                "restore_statuses": {"teacher U (seed 1)": "PASS"}, "restore_all_pass": True}
    rec = CO.dpc_backup(volumes_root=vols, DCmod=types.SimpleNamespace(backup=fake_backup))
    assert rec["status"] == "RUN" and rec["closed_results_unchanged"]
    assert calls["out_pkg"] == CO.DPC_DIR and calls["dest"] == str(vols / "Owner Named Disk")
    assert "Owner" not in (CO.DPC_DIR / "STATUS.json").read_text()

    def bad_backup(dest, targets_file, seed, out_pkg):
        (_isolate["dpc_results"] / "BACKUP_VERIFICATION.json").write_text("{}")    # writes into a closed tree
        return {"status": "DRIVE_COPY_VERIFIED"}
    rec2 = CO.dpc_backup(volumes_root=vols, DCmod=types.SimpleNamespace(backup=bad_backup))
    assert rec2["status"] == "FAILED_CLOSED_RESULTS_CHANGED"


def _fake_osf(tmp_path, closed_pkg: Path, write_into_closed=False):
    OC = types.SimpleNamespace(PKG=closed_pkg, OUT_REPAIR=closed_pkg / "PREDECESSOR_CUSTODY_REPAIR.json",
                               REPLAY_OUT=closed_pkg / "predecessor_custody" / "X.json")
    log = []

    def write_public(path, obj):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj))
        log.append(str(path))
    OC.write_public = write_public

    def backup(dest, seed):
        OC.write_public(OC.PKG / "BACKUP_VERIFICATION.json", {"status": "DRIVE_COPY_VERIFIED"})
        if write_into_closed:
            OC.write_public(OC.PKG / "SOMETHING_ELSE.json", {})
        return {"status": "DRIVE_COPY_VERIFIED", "files": 1, "restore_checks": {"U": {"status": "PASS"}}}

    def predecessor(restore, dry_run):
        OC.write_public(OC.OUT_REPAIR, {"status": "RESTORED"})
        return {"status": "RESTORED", "unresolved_gap": None}
    OC.backup, OC.predecessor = backup, predecessor
    return OC, log


def test_predecessor_redirects_osf_receipts_and_keeps_closed_trees(tmp_path, monkeypatch, _isolate):
    monkeypatch.setattr(CO, "osf_assessment_sealed_until_qpc_lock", lambda: __import__("contextlib").nullcontext(False))
    OC, log = _fake_osf(tmp_path, _isolate["osf_results"])
    orig = OC.write_public
    rec = CO.predecessor(restore=True, volumes_root=_volumes(tmp_path), OC=OC)
    assert rec["status"] == "RUN" and rec["closed_results_unchanged"]
    assert rec["runs"]["osf_backup"]["status"] == "DRIVE_COPY_VERIFIED"
    assert rec["runs"]["osf_predecessor_restore"]["status"] == "RESTORED"
    assert (CO.PRED_DIR / "osf_BACKUP_VERIFICATION.json").exists()
    assert (CO.PRED_DIR / "PREDECESSOR_CUSTODY_REPAIR.json").exists()
    assert not (_isolate["osf_results"] / "BACKUP_VERIFICATION.json").exists()
    assert OC.write_public is orig and OC.OUT_REPAIR == _isolate["osf_results"] / "PREDECESSOR_CUSTODY_REPAIR.json"
    assert "Owner" not in (CO.PRED_DIR / "STATUS.json").read_text()


def test_predecessor_refuses_unredirected_write_into_a_closed_tree(tmp_path, monkeypatch, _isolate):
    monkeypatch.setattr(CO, "osf_assessment_sealed_until_qpc_lock", lambda: __import__("contextlib").nullcontext(False))
    OC, _ = _fake_osf(tmp_path, _isolate["osf_results"], write_into_closed=True)
    rec = CO.predecessor(restore=False, volumes_root=_volumes(tmp_path), OC=OC)
    assert rec["runs"]["osf_backup"]["status"] == "FAILED" and "unredirected" in rec["runs"]["osf_backup"]["reason"]
    assert not (_isolate["osf_results"] / "SOMETHING_ELSE.json").exists() and rec["closed_results_unchanged"]


def test_predecessor_pending_without_drive(tmp_path):
    rec = CO.predecessor(restore=True, volumes_root=_volumes(tmp_path, match=False))
    assert rec["status"] == "PENDING" and len(rec["gaps"]) == 3
    assert json.loads((CO.PRED_DIR / "STATUS.json").read_text())["pending_command"].startswith(CO.CMD)


def test_osf_assessment_stays_sealed_until_the_qpc_lock_is_pushed(monkeypatch):
    from osf import assess as AS
    orig = AS.open_assessment
    with CO.osf_assessment_sealed_until_qpc_lock() as ok:
        assert ok is False
        with pytest.raises(SystemExit, match="sealed by qpc custody"):
            AS.open_assessment()
    assert AS.open_assessment is orig
    monkeypatch.setattr(CO.QD, "evaluation_lock_pushed", lambda *a, **k: (True, "pushed"))
    with CO.osf_assessment_sealed_until_qpc_lock() as ok:
        assert ok is True and AS.open_assessment is orig


# ------------------------------------------------------------------ the full sequence
def test_run_all_with_drive_runs_in_order_and_rereads_new_folders(tmp_path, monkeypatch, _isolate):
    src = _store(tmp_path)
    vols = _volumes(tmp_path)
    disk = vols / "Owner Named Disk"
    order = []

    def fake_dpc(dry_run, volumes_root):
        order.append("dpc")
        f = disk / f"private_dpc_v1_{DATE}"
        (f / "dpc_v1").mkdir(parents=True)
        (f / "dpc_v1" / "a").write_bytes(b"a")
        (f / "SHA256SUMS").write_text(f"{hashlib.sha256(b'a').hexdigest()}  dpc_v1/a\n")
        return {"status": "RUN"}

    def fake_pred(restore, dry_run, volumes_root):
        order.append("predecessor")
        (disk / "Owner Extra Folder").mkdir()                     # an unexpected new folder: name never published
        return {"status": "RUN"}
    monkeypatch.setattr(CO, "dpc_backup", fake_dpc)
    monkeypatch.setattr(CO, "predecessor", fake_pred)
    _stub_restore(monkeypatch)
    out = tmp_path / "pkg"
    bv = CO.run_all(volumes_root=vols, src=src, cache=src.parent, out_pkg=out)
    assert order == ["dpc", "predecessor"] and bv["status"] == CO.STATUS_DRIVE
    rr = bv["uncached_reread_of_drive_folders"]
    assert rr["created_by_this_closeout"][f"private_dpc_v1_{DATE}"]["pass"]
    assert rr["created_by_this_closeout"][f"private_qpc_v1_{DATE}"]["pass"]
    assert "<other new folder>" in rr["created_by_this_closeout"]
    assert rr["all_pass"] is False                                 # the unknown folder has no SHA256SUMS
    assert bv["closed_results_unchanged"] and bv["components"]["dpc_off_device_backup"]["status"] == "RUN"
    pub = (out / "BACKUP_VERIFICATION.json").read_text()
    assert "Owner" not in pub and str(tmp_path) not in pub


def test_run_all_without_drive_records_pending_and_makes_a_local_copy(tmp_path, monkeypatch, _isolate):
    src = _store(tmp_path)
    _stub_restore(monkeypatch)
    out = tmp_path / "pkg"
    bv = CO.run_all(volumes_root=_volumes(tmp_path, match=False), src=src, cache=src.parent, out_pkg=out)
    assert bv["status"] == CO.STATUS_LOCAL and bv["components"]["drive_mounted"] is False
    assert bv["components"]["dpc_off_device_backup"]["status"] == "PENDING"
    assert bv["components"]["predecessor_repair"]["status"] == "PENDING"
    assert (CO.DPC_DIR / "STATUS.json").exists() and (CO.PRED_DIR / "STATUS.json").exists()
    assert (src.parent / f"qpc_v1_local_copy_{DATE}" / "SHA256SUMS").exists()


def test_plan_only_dry_run_loads_no_data(tmp_path, monkeypatch):
    src = _store(tmp_path)
    monkeypatch.setattr(CO, "rehearse", lambda *a: (_ for _ in ()).throw(AssertionError("rehearsal ran")))
    plan = CO.run_all(dry_run=True, plan_only=True, volumes_root=_volumes(tmp_path, match=False), src=src,
                      cache=src.parent, out_pkg=tmp_path / "pkg")
    assert plan["restore_rehearsal_read_only_on_live_store"]["status"].startswith("SKIPPED")
    with pytest.raises(SystemExit, match="requires --dry-run"):
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        CO.main(["backup", "--plan-only"])


def test_copy_records_a_live_file_that_keeps_changing(tmp_path, monkeypatch):
    src = _store(tmp_path)
    log = src / "run" / "SEMA_LOG.jsonl"
    log.write_text("a\n")
    import shutil
    real = shutil.copy2

    def appending_copy(s, d, **kw):                         # another process appends during every copy of the log
        out = real(s, d, **kw)
        if Path(s).name == "SEMA_LOG.jsonl":
            with open(s, "a") as fh:
                fh.write("x\n")
        return out
    monkeypatch.setattr(shutil, "copy2", appending_copy)
    copy, cp = CO.copy_study(src, tmp_path / "out" / "qpc_v1_local_copy_x")
    assert cp["live_files_changed_during_copy"] == ["run/SEMA_LOG.jsonl"]
    assert cp["source_stable_during_copy"] == cp["files"] - 1 and cp["uncached_readback_match"] == cp["files"]
    assert CO.verify_sums(tmp_path / "out" / "qpc_v1_local_copy_x")["pass"]       # the snapshot is self-consistent
    with pytest.raises(SystemExit, match="never reused"):
        CO.copy_study(src, tmp_path / "out" / "qpc_v1_local_copy_x")


def test_restore_and_deploy_a_real_qpc_policy_from_the_copy(tmp_path):
    """End to end on synthetic data: a bound qpc.PolicyPair (direct KL k-means codebooks) is re-encoded from the teacher
    restored from the copy, deployed through qpc.deploy.release from the copy's teacher unit and compared bitwise."""
    import shutil
    from dpc import deploy as DDP
    from qpc import kmeans as KM
    from qpc import release as RL
    live = tmp_path / "live"
    D, T = _synthetic_store(live)
    D["feature_names"] = np.asarray([f"f{j}" for j in range(83)])
    msha = DA.sha(live / "admitted" / "rel__s1__U" / "model.pt")
    fsha = DDP.schema_sha256([str(x) for x in D["feature_names"]])
    fit = D["idx"]["OSF_DEFENSE_FIT"]
    pols = []
    for r, K, m in ((1, 2, 2), (2, 6, 2)):
        f = KM.fit_recipient(T[f"p{r}"][fit], T[f"d{r}"][fit], K, m)
        pols.append(RL.make_policy(r, f.partition if hasattr(f, "partition") else f[0]))
    pair = RL.make_pair(pols[0], pols[1], "DIRECT-TASK", 2, 2,
                        meta={"teacher_model_sha256": msha, "feature_names_sha256": fsha,
                              "config": "U|DIRECT-TASK|i2o2"})
    name = "pol__s1__U_DIRECT-TASK_i2o2"
    u = live / "run" / "units" / name
    u.mkdir(parents=True)
    RL.save_policy(pair, u / "policy.json")
    np.savez(u / "release.npz", **RL.release_arrays(pair, D["row_id"], T["p1"], T["d1"], T["p2"], T["d2"]))
    (u / "record.json").write_text(json.dumps({"seed": 1, "teacher": "U", "config": "U|DIRECT-TASK|i2o2"}))
    _complete(u)
    copy = tmp_path / "copy"
    shutil.copytree(live, copy)
    info = CO.restore_policy(copy, live, name, {("U", 1): T}, D)
    assert info["status"] == "PASS", info
    assert info["deployment_from_copy"]["binding"] == "BOUND"
    assert all(info["deployment_from_copy"]["bitwise_equal_to_re_encoded_release"].values())
    assert info["deployment_from_copy"]["outputs"] == sorted(DDP.ALLOWED_OUTPUT)
    # a policy bound to a different (valid) teacher model is refused at deployment; a corrupt one fails, not crashes
    torch.save(_state(99), copy / "admitted" / "rel__s1__U" / "model.pt")
    _complete(copy / "admitted" / "rel__s1__U")
    bad = CO.deploy_from_copy(copy, pair, "U", 1, D, {})
    assert bad["status"] == "FAIL" and "different teacher" in bad["reason"]
    (copy / "admitted" / "rel__s1__U" / "model.pt").write_bytes(b"corrupt")
    _complete(copy / "admitted" / "rel__s1__U")
    assert CO.deploy_from_copy(copy, pair, "U", 1, D, {})["status"] == "FAIL"
