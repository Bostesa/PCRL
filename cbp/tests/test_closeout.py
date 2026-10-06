"""Tests for cbp.closeout (role F, custody). Everything runs on synthetic fixtures in pytest's tmp_path: fake volumes
(drive detection by content, skipped names), a fake private store and pinned input, stubbed restores, a fake qpc
closeout module for the redirected qpc custody run, and a synthetic teacher + qpc-format policy for an end-to-end restore
from a copy. No private store, drive, closed results tree or git ref is written. Adapted from qpc/tests/test_closeout.py
(the synthetic teacher-store helpers are imported from it unchanged).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m cbp.sema \\
        --label F:closeout-tests -- env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q \\
        cbp/tests/test_closeout.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import types
from pathlib import Path

import numpy as np
import pytest

from cbp import closeout as CO
from qpc import closeout as QC
from qpc.tests.test_closeout import _complete, _synthetic_store

SUMS = b"deadbeef  smf_v1/run/units/x/COMPLETE.json\n"
WANT = hashlib.sha256(SUMS).hexdigest()
DATE = time.strftime("%Y%m%d", time.gmtime())
SMF = "private_smf_v1_20261005"
REAL_MARKER = CO.smf_marker_sha
INPUT_BYTES = b"pinned-input-standin"


def _volumes(tmp: Path, match=True):
    root = tmp / "Volumes"
    for n in ("Macintosh HD", "BackgroundSyncService Setup", "Unrelated Stick", "Owner Named Disk"):
        (root / n).mkdir(parents=True)
    for skipped in ("Macintosh HD", "BackgroundSyncService Setup"):      # a match on a SKIPPED volume never counts
        (root / skipped / SMF).mkdir()
        (root / skipped / SMF / "SHA256SUMS").write_bytes(SUMS)
    (root / "Unrelated Stick" / SMF).mkdir()
    (root / "Unrelated Stick" / SMF / "SHA256SUMS").write_bytes(b"other")
    if match:
        d = root / "Owner Named Disk" / SMF
        d.mkdir()
        (d / "SHA256SUMS").write_bytes(SUMS)
        (root / "Owner Named Disk" / "Owner Personal Folder").mkdir()     # an identifying name: never published
    return root


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Marker, lock state, closed trees, receipt folders and the pinned input all point into tmp_path."""
    monkeypatch.setattr(CO, "smf_marker_sha", lambda: WANT)
    monkeypatch.setattr(CO, "cbp_lock_pushed", lambda: (False, "not pushed (test)"))
    closed = {k: tmp_path / "closed" / k for k in ("qpc_results", "dpc_results", "osf_results", "smf_results")}
    for d in closed.values():
        d.mkdir(parents=True)
        (d / "R.json").write_text("{}")
    monkeypatch.setattr(CO, "CLOSED_RESULTS", closed)
    monkeypatch.setattr(CO, "PKG", tmp_path / "pkg")
    monkeypatch.setattr(CO, "QPC_CUSTODY_DIR", tmp_path / "pkg" / "provenance" / "qpc_custody")
    monkeypatch.setattr(CO, "diskutil_probe", lambda: {"ran": False})
    inp = tmp_path / "jcv" / "adult_jcv.npz"
    inp.parent.mkdir(parents=True)
    inp.write_bytes(INPUT_BYTES)
    monkeypatch.setattr(CO, "INPUT", inp)
    monkeypatch.setattr(CO, "INPUT_SHA", hashlib.sha256(INPUT_BYTES).hexdigest())
    monkeypatch.setattr(CO, "CBP_TARGETS", tmp_path / "no_targets.json")
    monkeypatch.setattr(CO, "QPC_TARGETS", tmp_path / "no_qpc_targets.json")
    return closed


def _store(tmp: Path):
    src = tmp / "PCRL_eval_cache_private" / "cbp_v1"
    (src / "run" / "units" / "u").mkdir(parents=True)
    (src / "run" / "units" / "u" / "f.bin").write_bytes(b"x" * 100)
    (src / "run" / "units" / "u" / "f.bin.tmp").write_bytes(b"partial")          # in-progress write: excluded
    (src / "START.txt").write_text("t0\n")
    return src


def _stub_restore(monkeypatch, status="PASS"):
    def fake(copy, live, targets, seed, input_path):
        return ({"teacher U (seed 1)": {"status": status, "copy_seen": (Path(copy) / "START.txt").exists(),
                                        "copy_is_not_live": Path(copy) != Path(live),
                                        "input_in_copy": Path(input_path).exists() and Path(copy).parent in
                                        Path(input_path).parents},
                 "Q (DIRECT-TASK i8o64)": {"status": "PASS"}, "P* (U|SEQ-21|i8o64|l0.04)": {"status": "PASS"},
                 "attacker": {"status": "PASS"}}, {"wall_s": 0.0, "cpu_s": 0.0})
    monkeypatch.setattr(CO, "restore_all", fake)


# ------------------------------------------------------------------ drive detection, pins, code identity
def test_locate_drive_by_content_skips_names_and_withholds_them(tmp_path):
    root = _volumes(tmp_path)
    vol, ev = CO.locate_drive(root)
    assert vol == root / "Owner Named Disk"
    assert ev["skipped_by_name"] == 2 and ev["candidate_volumes"] == 2 and ev["matching_volumes"] == 1
    txt = json.dumps(ev)
    assert "Owner" not in txt and "Unrelated" not in txt and "Background" not in txt and "Macintosh" not in txt


def test_skipped_installer_image_is_never_read(tmp_path, monkeypatch):
    root = _volumes(tmp_path, match=False)
    seen = []
    orig = CO.DC.sha
    monkeypatch.setattr(CO.DC, "sha", lambda p, nocache=False: (seen.append(str(p)), orig(p, nocache))[1])
    vol, ev = CO.locate_drive(root)
    assert vol is None and ev["mounted"] is False
    assert not [s for s in seen if "BackgroundSyncService" in s or "Macintosh" in s]


def test_marker_is_read_at_the_pin():
    assert REAL_MARKER() == "28705e242f77b6333561f66117934b3ebc427072e31dc5a38b3a9d9e8f02cd05"


def test_pinned_packages_are_byte_identical_to_the_source_tip():
    ident = CO.source_code_identity()
    assert ident["identical"] and ident["files"] > 20 and not ident["differing_from_source_tip"]


# ------------------------------------------------------------------ same-device and drive copies
def test_same_device_copy_bundles_the_input_is_labelled_pending_and_restored_from_the_copy(tmp_path, monkeypatch):
    src = _store(tmp_path)
    out = tmp_path / "pkg"
    _stub_restore(monkeypatch)
    bv = CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path, match=False), out_pkg=out)
    assert bv["status"] == CO.STATUS_LOCAL == "LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING"
    assert "not a drive restore" in bv["restore_kind"] and bv["off_device_backup"].startswith("PENDING")
    assert "not off-device custody" in bv["custody_gap"]
    assert bv["pending"] == CO.PENDING and "<DRIVE_ROOT>" in CO.PENDING["cbp_off_device_backup_and_restore"]
    assert all(v.startswith(CO.SEMA) for k, v in CO.PENDING.items() if not k.endswith("NOT_recommended"))
    assert "qpc.closeout all --targets <PRIVATE_CACHE>/qpc_v1/run/closeout_targets.json" in \
        CO.PENDING["qpc_dpc_osf_smf_pending_custody"]
    assert bv["files"] == 3 and bv["store_files"] == 2 and bv["dependency_files"] == 1 and bv["uncached_readback_match"] == 3
    root = src.parent / f"cbp_v1_local_copy_{DATE}"
    assert (root / "cbp_v1" / "START.txt").exists() and not (root / "cbp_v1/run/units/u/f.bin.tmp").exists()
    assert (root / CO.DEP_REL).read_bytes() == INPUT_BYTES
    lines = (root / "SHA256SUMS").read_text().splitlines()
    assert len(lines) == 3 and {ln.split("  ")[1].split("/")[0] for ln in lines} == {"cbp_v1", "dependencies"}
    assert CO.verify_sums(root)["pass"]
    pub = (out / "BACKUP_VERIFICATION.json").read_text() + (out / "RESTORE_INDEX.json").read_text()
    assert "/Users/" not in pub and str(tmp_path) not in pub and "<PRIVATE_CACHE>" in pub
    assert CO.STATUS_DRIVE not in pub and json.loads((out / "RESTORE_INDEX.json").read_text())["not_off_device"]
    chk = bv["restore_checks"]["teacher U (seed 1)"]
    assert chk["copy_seen"] and chk["copy_is_not_live"] and chk["input_in_copy"]
    assert bv["required_restores"] == {"U teacher": "PASS", "Q": "PASS", "P* or fallback": "PASS", "attacker": "PASS"}
    assert bv["read_capability"]["F_NOCACHE_uncached_read"] == "available"
    assert "not performed" in bv["read_capability"]["physical_cold_read"]
    assert (root / "BACKUP_RECORD.json").exists()
    bv2 = CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path / "b", match=False), out_pkg=out)
    assert bv2["destination"].endswith(f"cbp_v1_local_copy_{DATE}_v2/cbp_v1")
    assert (root / "cbp_v1" / "START.txt").exists()                                # the first copy is kept


def test_drive_copy_goes_to_the_content_matched_volume_only(tmp_path, monkeypatch):
    src = _store(tmp_path)
    vols = _volumes(tmp_path)
    _stub_restore(monkeypatch)
    bv = CO.backup(src=src, cache=src.parent, volumes_root=vols, out_pkg=tmp_path / "pkg")
    assert bv["status"] == CO.STATUS_DRIVE and bv["destination"] == f"<DRIVE_ROOT>/private_cbp_v1_{DATE}/cbp_v1"
    assert "custody_gap" not in bv and "Owner" not in json.dumps(bv)
    assert (vols / "Owner Named Disk" / f"private_cbp_v1_{DATE}" / "cbp_v1" / "START.txt").exists()
    assert (vols / "Owner Named Disk" / f"private_cbp_v1_{DATE}" / CO.DEP_REL).exists()
    assert not (vols / "Unrelated Stick" / f"private_cbp_v1_{DATE}").exists()
    assert not (vols / "BackgroundSyncService Setup" / f"private_cbp_v1_{DATE}").exists()


def test_backup_refuses_when_free_space_would_drop_below_5_gib(tmp_path, monkeypatch):
    src = _store(tmp_path)
    monkeypatch.setattr(CO, "free_gib", lambda p: 5.0)
    _stub_restore(monkeypatch)
    with pytest.raises(SystemExit, match="5 GiB"):
        CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path, match=False), out_pkg=tmp_path / "pkg")
    assert not list(src.parent.glob("cbp_v1_local_copy_*"))


def test_backup_refuses_a_dependency_that_fails_its_pin(tmp_path, monkeypatch):
    src = _store(tmp_path)
    _stub_restore(monkeypatch)
    CO.INPUT.write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="pinned hash"):
        CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path, match=False), out_pkg=tmp_path / "pkg")
    assert not list(src.parent.glob("cbp_v1_local_copy_*"))


def test_failed_restore_is_not_reported_as_passing(tmp_path, monkeypatch):
    src = _store(tmp_path)
    _stub_restore(monkeypatch, status="FAIL")
    bv = CO.backup(src=src, cache=src.parent, volumes_root=_volumes(tmp_path, match=False), out_pkg=tmp_path / "pkg")
    assert bv["restore_all_pass"] is False and bv["required_restores"]["U teacher"] == "FAIL"


def test_required_restore_classes():
    st = {"teacher U (seed 1)": "PASS", "Q (DIRECT-TASK i8o64)": "PASS", "fallback (INELIGIBLE) U|JOINT": "FAIL",
          "attacker": "PENDING"}
    assert CO.required_targets_status(st) == {"U teacher": "PASS", "Q": "PASS", "P* or fallback": "FAIL",
                                              "attacker": "PENDING"}
    assert CO.required_targets_status({})["Q"].startswith("PENDING")
    assert CO.required_targets_status({})["P* or fallback"].startswith("PENDING")


# ------------------------------------------------------------------ dry runs, targets, sums, live files
def test_dry_runs_write_nothing(tmp_path, monkeypatch, _isolate):
    src = _store(tmp_path)
    out = tmp_path / "pkg"
    monkeypatch.setattr(CO, "rehearse", lambda s, t, k: {"statuses": {}})
    monkeypatch.setattr(CO, "source_code_identity", lambda: {"identical": True, "files": 1, "differing_from_source_tip": []})
    before = sorted(p.name for p in src.parent.iterdir())
    trees = {k: CO.tree_state(v) for k, v in _isolate.items()}
    for vols in (_volumes(tmp_path / "a", match=False), _volumes(tmp_path / "b", match=True)):
        drive_before = sorted(p.name for p in (vols / "Owner Named Disk").iterdir())
        CO.backup(src=src, cache=src.parent, volumes_root=vols, out_pkg=out, dry_run=True)
        CO.qpc_custody(dry_run=True, volumes_root=vols)
        CO.run_all(dry_run=True, volumes_root=vols, src=src, cache=src.parent, out_pkg=out)
        assert sorted(p.name for p in (vols / "Owner Named Disk").iterdir()) == drive_before
    assert sorted(p.name for p in src.parent.iterdir()) == before and not out.exists()
    assert {k: CO.tree_state(v) for k, v in _isolate.items()} == trees


def test_plan_only_dry_run_loads_no_data(tmp_path, monkeypatch):
    src = _store(tmp_path)
    monkeypatch.setattr(CO, "rehearse", lambda *a: (_ for _ in ()).throw(AssertionError("rehearsal ran")))
    plan = CO.run_all(dry_run=True, plan_only=True, volumes_root=_volumes(tmp_path, match=False), src=src,
                      cache=src.parent, out_pkg=tmp_path / "pkg")
    assert plan["restore_rehearsal_read_only_on_live_store"]["status"].startswith("SKIPPED")
    assert plan["self_contained"]["required"] == 20
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with pytest.raises(SystemExit, match="requires --dry-run"):
        CO.main(["backup", "--plan-only"])


def test_targets_must_be_cbp_unit_names_and_entry_points(tmp_path):
    t = tmp_path / "t.json"
    for bad in ({"policies": {"x": "../../etc/passwd"}}, {"policies": {"x": "ref__s1__F"}},
                {"policies": {}, "encode": "os:system"}, {"policies": {}, "attacker": {"fn": "subprocess:run"}},
                {"policies": {}, "attacker": {"fn": "dpc.audit:x"}}):
        t.write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            CO.load_targets(t)
    t.write_text(json.dumps({"seed": 1, "policies": {"Q (DIRECT-TASK i8o64)": "pol__s1__U_DIRECT-TASK_i8o64",
                                                     "P* (SEQ-21 0.04)": "pol__s1__U_SEQ-21_i8o64_l0.04"},
                             "attacker": {"fn": "cbp.assess:refit_selected_attacker"}}))
    assert CO.load_targets(t)["policies"]["P* (SEQ-21 0.04)"] == "pol__s1__U_SEQ-21_i8o64_l0.04"
    assert CO.load_targets(tmp_path / "absent.json")["policies"] == {}


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


def test_copy_records_a_live_file_that_keeps_changing(tmp_path, monkeypatch):
    src = _store(tmp_path)
    log = src / "run" / "SEMA_LOG.jsonl"
    log.write_text("a\n")
    real = shutil.copy2

    def appending_copy(s, d, **kw):                         # another process appends during every copy of the log
        out = real(s, d, **kw)
        if Path(s).name == "SEMA_LOG.jsonl":
            with open(s, "a") as fh:
                fh.write("x\n")
        return out
    monkeypatch.setattr(shutil, "copy2", appending_copy)
    copy, cp = CO.copy_study(src, tmp_path / "out" / "cbp_v1_local_copy_x", {CO.DEP_REL: CO.INPUT})
    assert cp["live_files_changed_during_copy"] == ["cbp_v1/run/SEMA_LOG.jsonl"]
    assert cp["source_stable_during_copy"] == cp["files"] - 1 and cp["uncached_readback_match"] == cp["files"]
    assert CO.verify_sums(tmp_path / "out" / "cbp_v1_local_copy_x")["pass"]
    with pytest.raises(SystemExit, match="never reused"):
        CO.copy_study(src, tmp_path / "out" / "cbp_v1_local_copy_x")


# ------------------------------------------------------------------ the pending qpc / dpc / osf / smf custody
def test_qpc_custody_pending_without_drive_never_calls_the_source_closeout(tmp_path, _isolate):
    called = []
    fake = types.SimpleNamespace(run_all=lambda **k: called.append(k))
    rec = CO.qpc_custody(volumes_root=_volumes(tmp_path, match=False), QCmod=fake)
    assert rec["status"] == "PENDING" and not called
    assert "qpc.closeout all --targets <PRIVATE_CACHE>/qpc_v1/run/closeout_targets.json" in rec["pending_command"]
    assert "no redundant qpc same-device copy" in rec["note"] and len(rec["gaps"]) == 3
    txt = (CO.QPC_CUSTODY_DIR / "STATUS.json").read_text()
    assert "Owner" not in txt and str(tmp_path) not in txt


def _fake_qpc(tmp_path, write_closed=None):
    """A stand-in for qpc.closeout.run_all that writes through qpc.closeout's own (rebound) receipt locations."""
    seen = {}

    def run_all(targets_file=None, volumes_root=None, out_pkg=None, **kw):
        seen.update(out_pkg=Path(out_pkg), dpc_dir=QC.DPC_DIR, pred_dir=QC.PRED_DIR, pkg=QC.PKG,
                    seal=QC.osf_assessment_sealed_until_qpc_lock, closed=dict(QC.CLOSED_RESULTS))
        QC.write_public(QC.DPC_DIR / "STATUS.json", {"status": "RUN"})
        QC.write_public(QC.PRED_DIR / "STATUS.json", {"status": "RUN"})
        QC.write_public(Path(out_pkg) / "BACKUP_VERIFICATION.json", {"status": "DRIVE_COPY_VERIFIED"})
        CO.DC.write_public(QC.DPC_DIR / "BACKUP_VERIFICATION.json", {"status": "DRIVE_COPY_VERIFIED"})
        if write_closed is not None:
            QC.write_public(write_closed / "BACKUP_VERIFICATION.json", {"status": "x"})
        return {"status": "DRIVE_COPY_VERIFIED", "files": 3, "uncached_readback_match": 3, "restore_all_pass": True,
                "required_restores": {"U teacher": "PASS"}, "components": {"dpc_off_device_backup": {"status": "RUN"}},
                "uncached_reread_of_drive_folders": {"all_pass": True}}
    return types.SimpleNamespace(run_all=run_all), seen


def test_qpc_custody_with_drive_redirects_every_receipt_and_restores_bindings(tmp_path, monkeypatch, _isolate):
    monkeypatch.setattr(CO, "source_code_identity", lambda: {"identical": True, "files": 1, "differing_from_source_tip": []})
    orig = {k: getattr(QC, k) for k in ("DPC_DIR", "PRED_DIR", "PKG", "CLOSED_RESULTS", "write_public",
                                        "osf_assessment_sealed_until_qpc_lock")}
    orig_dc = CO.DC.write_public
    fake, seen = _fake_qpc(tmp_path)
    rec = CO.qpc_custody(volumes_root=_volumes(tmp_path), QCmod=fake)
    out = CO.QPC_CUSTODY_DIR
    assert rec["status"] == "RUN" and rec["closed_results_unchanged"]
    assert seen["out_pkg"] == out and seen["dpc_dir"] == out / "dpc_custody" and seen["pred_dir"] == out / \
        "predecessor_custody" and seen["pkg"] == out
    assert seen["seal"] is CO.osf_assessment_sealed_until_cbp_lock and "qpc_results" in seen["closed"]
    for f in ("dpc_custody/STATUS.json", "predecessor_custody/STATUS.json", "BACKUP_VERIFICATION.json",
              "dpc_custody/BACKUP_VERIFICATION.json", "STATUS.json"):
        assert (out / f).exists(), f
    assert {k: getattr(QC, k) for k in orig} == orig and CO.DC.write_public is orig_dc
    assert "Owner" not in (out / "STATUS.json").read_text()


def test_qpc_custody_refuses_a_write_into_a_closed_tree(tmp_path, monkeypatch, _isolate):
    monkeypatch.setattr(CO, "source_code_identity", lambda: {"identical": True, "files": 1, "differing_from_source_tip": []})
    before = {k: CO.tree_state(v) for k, v in _isolate.items()}
    fake, _ = _fake_qpc(tmp_path, write_closed=_isolate["qpc_results"])
    rec = CO.qpc_custody(volumes_root=_volumes(tmp_path), QCmod=fake)
    assert rec["runs"]["qpc_run_all"]["status"] == "FAILED" and "closed results" in rec["runs"]["qpc_run_all"]["reason"]
    assert rec["closed_results_unchanged"] and {k: CO.tree_state(v) for k, v in _isolate.items()} == before
    assert not (_isolate["qpc_results"] / "BACKUP_VERIFICATION.json").exists()
    assert QC.write_public is CO.DC.write_public                     # bindings restored after the failure


def test_qpc_custody_refuses_when_a_pinned_package_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(CO, "source_code_identity", lambda: {"identical": False, "files": 9,
                                                             "differing_from_source_tip": ["qpc/run.py"]})
    called = []
    rec = CO.qpc_custody(volumes_root=_volumes(tmp_path), QCmod=types.SimpleNamespace(run_all=lambda **k: called.append(1)))
    assert rec["status"] == "REFUSED" and not called


def test_osf_assessment_stays_sealed_until_the_cbp_lock_is_pushed(monkeypatch):
    from osf import assess as AS
    orig = AS.open_assessment
    with CO.osf_assessment_sealed_until_cbp_lock() as ok:
        assert ok is False
        with pytest.raises(SystemExit, match="sealed by cbp custody"):
            AS.open_assessment()
    assert AS.open_assessment is orig
    monkeypatch.setattr(CO, "cbp_lock_pushed", lambda: (True, "pushed"))
    with CO.osf_assessment_sealed_until_cbp_lock() as ok:
        assert ok is True and AS.open_assessment is orig


# ------------------------------------------------------------------ the full sequence
def test_run_all_with_drive_runs_in_order_and_rereads_new_folders(tmp_path, monkeypatch, _isolate):
    src = _store(tmp_path)
    vols = _volumes(tmp_path)
    disk = vols / "Owner Named Disk"
    order = []

    def fake_qpc_custody(dry_run, volumes_root, qpc_targets=None):
        order.append("qpc")
        for name in (f"private_qpc_v1_{DATE}", f"private_dpc_v1_{DATE}"):
            f = disk / name
            (f / "x").mkdir(parents=True)
            (f / "x" / "a").write_bytes(b"a")
            (f / "SHA256SUMS").write_text(f"{hashlib.sha256(b'a').hexdigest()}  x/a\n")
        (disk / "Owner Extra Folder").mkdir()                     # an unexpected new folder: name never published
        return {"status": "RUN"}
    monkeypatch.setattr(CO, "qpc_custody", fake_qpc_custody)
    _stub_restore(monkeypatch)
    out = tmp_path / "pkg"
    bv = CO.run_all(volumes_root=vols, src=src, cache=src.parent, out_pkg=out)
    assert order == ["qpc"] and bv["status"] == CO.STATUS_DRIVE
    rr = bv["uncached_reread_of_drive_folders"]
    for n in (f"private_qpc_v1_{DATE}", f"private_dpc_v1_{DATE}", f"private_cbp_v1_{DATE}"):
        assert rr["created_by_this_closeout"][n]["pass"], n
    assert "<other new folder>" in rr["created_by_this_closeout"] and rr["all_pass"] is False
    assert rr["created_before_the_cbp_copy"] == 3
    assert bv["closed_results_unchanged"] and bv["components"]["qpc_custody"]["status"] == "RUN"
    pub = (out / "BACKUP_VERIFICATION.json").read_text()
    assert "Owner" not in pub and str(tmp_path) not in pub


def test_run_all_without_drive_records_pending_and_makes_a_local_copy(tmp_path, monkeypatch, _isolate):
    src = _store(tmp_path)
    _stub_restore(monkeypatch)
    out = tmp_path / "pkg"
    bv = CO.run_all(volumes_root=_volumes(tmp_path, match=False), src=src, cache=src.parent, out_pkg=out)
    assert bv["status"] == CO.STATUS_LOCAL and bv["components"]["drive_mounted"] is False
    assert bv["components"]["qpc_custody"]["status"] == "PENDING"
    assert "qpc.closeout all" in bv["components"]["qpc_custody"]["pending_command"]
    assert (CO.QPC_CUSTODY_DIR / "STATUS.json").exists()
    assert (src.parent / f"cbp_v1_local_copy_{DATE}" / "SHA256SUMS").exists()


# ------------------------------------------------------------------ an end-to-end restore from a synthetic copy
def test_restore_from_a_synthetic_copy_alone(tmp_path, monkeypatch):
    """A bound qpc-format policy pair (KL k-means codebooks, the format of every cbp code) is re-encoded from the teacher
    restored from the COPY (own forward pass), deployed from the copy through qpc.deploy and compared bitwise with the
    copied and live releases; tampering with the copy is caught."""
    from dpc import admit as DA
    from dpc import deploy as DDP
    from qpc import kmeans as KM
    from qpc import release as RL
    live = tmp_path / "PCRL_eval_cache_private" / "cbp_v1"
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
                        meta={"teacher_model_sha256": msha, "feature_names_sha256": fsha, "config": "U|DIRECT-TASK|i2o2"})
    name = "pol__s1__U_DIRECT-TASK_i2o2"
    u = live / "run" / "units" / name
    u.mkdir(parents=True)
    RL.save_policy(pair, u / "policy.json")
    np.savez(u / "release.npz", **RL.release_arrays(pair, D["row_id"], T["p1"], T["d1"], T["p2"], T["d2"]))
    (u / "record.json").write_text(json.dumps({"seed": 1, "teacher": "U", "config": "U|DIRECT-TASK|i2o2"}))
    _complete(u)
    monkeypatch.setattr(QC, "DPC_UNITS", live / "run" / "units")   # stands in for the pinned dpc teacher unit
    monkeypatch.setattr(CO, "load_D_from_input", lambda p: (D if Path(p).read_bytes() == INPUT_BYTES else
                                                            (_ for _ in ()).throw(SystemExit("wrong input"))))
    copy, cp = CO.copy_study(live, tmp_path / "copies" / "cbp_v1_local_copy_t", {CO.DEP_REL: CO.INPUT})
    targets = {"seed": 1, "policies": {"Q (synthetic DIRECT-TASK)": name}}
    checks, _ = CO.restore_all(copy, live, targets, 1, copy.parent / CO.DEP_REL)
    st = {k: v.get("status") for k, v in checks.items()}
    assert st["teacher U (seed 1)"] == "PASS" and st["Q (synthetic DIRECT-TASK)"] == "PASS", checks
    assert st["attacker"] == "PENDING"
    dep = checks["Q (synthetic DIRECT-TASK)"]["deployment_from_copy"]
    assert dep["binding"] == "BOUND" and all(dep["bitwise_equal_to_re_encoded_release"].values())
    assert dep["outputs"] == sorted(DDP.ALLOWED_OUTPUT)
    # tampering with the copied release is caught (the copy no longer verifies against live)
    z = dict(np.load(copy / "run" / "units" / name / "release.npz"))
    z["tok2"] = z["tok2"].copy()
    z["tok2"][0] = (z["tok2"][0] + 1) % 2
    np.savez(copy / "run" / "units" / name / "release.npz", **z)
    bad, _ = CO.restore_all(copy, live, targets, 1, copy.parent / CO.DEP_REL)
    assert bad["Q (synthetic DIRECT-TASK)"]["status"] == "FAIL"


def test_the_copy_input_loader_refuses_a_missing_or_unpinned_file(tmp_path):
    with pytest.raises(SystemExit, match="no bundled input"):
        CO.load_D_from_input(tmp_path / "missing.npz")
    other = tmp_path / "other.npz"
    other.write_bytes(b"not the pinned input")
    with pytest.raises(SystemExit, match="pinned hash"):
        CO.load_D_from_input(other)
