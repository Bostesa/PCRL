"""Custody-owner checks for osf.closeout that run without the external drive: the independent forward pass and restore
reproduce an admitted release bitwise from a copy alone, tampering is detected, versioned folders are never reused,
skipped volumes are never opened, and both dry-runs write nothing."""
import shutil

import numpy as np
import pytest

from osf import admit as AD
from osf import closeout as CO
from osf import data as OD


@pytest.fixture(scope="module")
def D():
    return OD.load()


def _fake_store(tmp_path, name):
    """A 'drive' copy and a 'local' store holding the same admitted smf unit (local layout run/units/<name>)."""
    src = AD.admitted_path(name)
    drive = tmp_path / "drive" / "run" / "units"
    local = tmp_path / "local"
    shutil.copytree(src, drive / name)
    shutil.copytree(src, local / "run" / "units" / name)
    return drive, local


@pytest.mark.parametrize("name", ["raw__s1__RAW-J__b0.3__e40", "lc__s1__E"])
def test_restore_from_copy_alone_is_bitwise(tmp_path, monkeypatch, D, name):
    drive, local = _fake_store(tmp_path, name)
    monkeypatch.setattr(CO, "SRC", local)
    r = CO.restore_unit(drive, name, D)
    assert r["status"] == "PASS", r
    assert all(v["features_max_abs_diff"] == 0 and v["hard_mismatches"] == 0
               for v in r["rebuild_from_drive_vs_local_release"].values())


def test_restore_detects_tampered_drive_file(tmp_path, monkeypatch, D):
    name = "tl__s1__e40"
    drive, local = _fake_store(tmp_path, name)
    monkeypatch.setattr(CO, "SRC", local)
    p = drive / name / "head_1.joblib"
    b = bytearray(p.read_bytes())
    b[-1] ^= 1
    p.write_bytes(bytes(b))
    assert CO.restore_unit(drive, name, D)["drive_files_hash_ok_uncached"] is False


def test_versioned_folders_never_reused(tmp_path):
    (tmp_path / "x").mkdir()
    assert CO.versioned(tmp_path, "x").name == "x_v2"
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a").write_text("a")
    sums, reread = CO.copy_files(tmp_path / "src", ["a"], tmp_path / "dst")
    assert all(reread.values())
    with pytest.raises(SystemExit):
        CO.copy_files(tmp_path / "src", ["a"], tmp_path / "dst")


def test_skipped_volumes_are_not_candidates(monkeypatch):
    monkeypatch.setattr(CO.os, "listdir", lambda p: ["Macintosh HD", "BackgroundSyncService Setup", "EXT"])
    assert [v.name for v in CO.volumes()] == ["EXT"]


def test_dry_runs_write_nothing():
    before = {p: p.stat().st_mtime_ns for p in CO.PKG.glob("*.json")}
    plan = CO.backup(dry_run=True)
    assert plan["files"] > 0 and "fallback_if_drive_absent" in plan
    rec = CO.predecessor(dry_run=True)
    assert rec["drive_copy"]["root"].startswith("<DRIVE_ROOT>/")
    assert {p: p.stat().st_mtime_ns for p in CO.PKG.glob("*.json")} == before


def test_public_scrub_refuses_paths():
    with pytest.raises(SystemExit):
        CO.scrub("x " + str(CO.HOME) + "/y")
    with pytest.raises(SystemExit):
        CO.scrub("/Volumes/anything")
    assert CO.scrub("<DRIVE_ROOT>/private_osf_v1") == "<DRIVE_ROOT>/private_osf_v1"


def test_forward_equals_released_features(D):
    import torch
    d = AD.admitted_path("raw__s0__RAW-L__b0.1__e20")
    z = np.load(d / "release.npz")
    pos = {int(r): j for j, r in enumerate(D["row_id"])}
    ix = np.array([pos[int(r)] for r in z["row_id"]])
    H = CO.forward(torch.load(d / "model.pt", weights_only=True), D["X"][ix])
    assert np.array_equal(H[0], z["r1"]) and np.array_equal(H[1], z["r2"])
