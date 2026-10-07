"""Tests for hcal.closeout (role F: custody, resources, reporting). SYNTHETIC ONLY: everything runs in pytest's
tmp_path with fake volumes (drive detection by content, skipped names), a fake hcal_v1 store, a stand-in pinned input,
a synthetic teacher (jcv.train.Model + sklearn heads, restored through the real dpc.deploy path), synthetic frozen-bank
tables, units, ledgers and ps output. No private store, drive, git ref or tracked result is written; no Adult data or
label is loaded; nothing is killed.

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m hcal.sema --label F:closeout-tests -- env OMP_NUM_THREADS=1 \\
        PYTHONPATH=. <python> -m pytest -q tests/pcrl_heldout_calibration_v1/test_closeout.py
"""
from __future__ import annotations

import calendar
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from hcal import closeout as CO
from hcal import ids as I

DATE = time.strftime("%Y%m%d", time.gmtime())
SMF = "private_smf_v1_20261005"
INPUT_BYTES = b"pinned-input-standin"
INPUT_SHA = hashlib.sha256(INPUT_BYTES).hexdigest()
USER = "synthuser"
WT = Path(__file__).resolve().parents[2]


def _h(b):
    return hashlib.sha256(b).hexdigest()


def _complete(d: Path):
    files = {str(p.relative_to(d)): _h(p.read_bytes()) for p in sorted(d.rglob("*"))
             if p.is_file() and p.name != "COMPLETE.json"}
    (d / "COMPLETE.json").write_text(json.dumps({"id": d.name, "files": files}))


def _smf_folder(vol: Path):
    d = vol / SMF
    (d / "smf_v1").mkdir(parents=True)
    (d / "smf_v1" / "a.txt").write_bytes(b"x")
    sums = f"{_h(b'x')}  smf_v1/a.txt\n".encode()
    (d / "SHA256SUMS").write_bytes(sums)
    return _h(sums)


def _volumes(tmp: Path, match=True):
    """Volumes: the two skipped names (each holding a matching marker that must never count), an unrelated stick with a
    wrong marker, and (match=True) an owner-named disk with the verified smf folder and a personal folder."""
    root = tmp / "Volumes"
    for n in ("Macintosh HD", "BackgroundSyncService Setup", "Unrelated Stick", "Owner Named Disk"):
        (root / n).mkdir(parents=True)
    want = _smf_folder(root / "Macintosh HD")
    _smf_folder(root / "BackgroundSyncService Setup")
    (root / "Unrelated Stick" / SMF).mkdir()
    (root / "Unrelated Stick" / SMF / "SHA256SUMS").write_bytes(b"other")
    if match:
        _smf_folder(root / "Owner Named Disk")
        (root / "Owner Named Disk" / "Owner Personal Folder").mkdir()
    return root, want


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(CO, "HOME_NAME", USER)
    monkeypatch.setattr(CO, "diskutil_probe", lambda: {"ran": False})
    monkeypatch.setattr(CO, "smf_marker_sha", lambda: "0" * 64)
    monkeypatch.setattr(CO, "PKG", tmp_path / "pkg")
    monkeypatch.setattr(CO, "pinned_input", lambda: (tmp_path / "jcv" / "adult_jcv.npz", INPUT_SHA))


def _input(tmp: Path):
    p = tmp / "jcv" / "adult_jcv.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(INPUT_BYTES)
    return {CO.DEP_REL: p}


def _teacher(store: Path, k=0, t="U", n=48):
    """A synthetic admitted teacher: jcv.train.Model + sklearn heads, its teacher.npz from the real dpc.deploy path."""
    import joblib
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from dpc import deploy as DD
    from jcv.train import Model
    rng = np.random.default_rng(k)
    X = rng.normal(size=(n, 83))
    names = [f"c{j:02d}" for j in range(83)]
    inp = store / "admitted" / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    np.savez(inp / "deploy_input.npz", X=X, feature_names=np.asarray(names))
    (inp / "schema.json").write_text(json.dumps(names))
    d = store / "admitted" / f"rel__s{k}__{t}"
    d.mkdir(parents=True)
    m = Model(83, [2, 6], k)
    torch.save(m.state_dict(), d / "model.pt")
    with torch.no_grad():
        R = [m.encode(i, torch.from_numpy(X.astype(np.float32))).double().numpy() for i in (0, 1)]
    for i, K in enumerate((2, 6)):
        head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)).fit(R[i], np.arange(n) % K)
        joblib.dump(head, d / f"head_{i}.joblib")
    (d / "record.json").write_text(json.dumps({"seed": k}))
    _complete(d)
    model, heads, msha = DD.load_teacher(d, seed=k)
    P = DD.teacher_probs(model, heads, X)
    u = store / "admitted" / "units" / f"tea__s{k}__{t}"
    u.mkdir(parents=True)
    np.savez(u / "teacher.npz", row_id=np.arange(n), p1=P[0], p2=P[1], d1=P[0].argmax(1), d2=P[1].argmax(1))
    (u / "record.json").write_text("{}")
    _complete(u)
    return msha


def _store(tmp: Path, teacher=False):
    """A fake hcal_v1 store: admitted bank tables + receipt, run units, ledgers (and optionally a synthetic teacher)."""
    s = tmp / "cache" / "hcal_v1"
    (s / "run" / "units").mkdir(parents=True)
    (s / "START.txt").write_text("2026-10-07T00:00:00Z\n")
    (s / "run" / "SEMA_LOG.jsonl").write_text("")
    bank = s / "admitted" / "bank"
    bank.mkdir(parents=True)
    rec = {"verdict": "ADMITTED", "bank": {}, "teachers": {}, "inputs": {}}
    for key in ("0|U|DIRECT-TASK|i8o64", "1|U|JOINT|i8o64|l0.1"):
        k, p = key.split("|", 1)
        path = bank / f"bank__s{k}__{I.safe(p)}.npz"
        np.savez_compressed(path, tok1=np.arange(5), q01=np.full((3, 2), 0.5))
        rec["bank"][key] = {"bank_sha256": _h(path.read_bytes()), "ok": True}
    if teacher:
        msha = _teacher(s, 0, "U")
        rec["teachers"]["U|0"] = {"model_sha256": msha}
        rec["inputs"] = {f: _h((s / "admitted" / "inputs" / f).read_bytes()) for f in ("deploy_input.npz", "schema.json")}
    (s / "admitted" / "ADMISSION_RECEIPT.json").write_text(json.dumps(rec))
    for u, arrays in (("cal__s0__H-GLOBAL-TEMP", {"alpha": np.array([1.25, 0.8])}),
                      ("cal__s0__H-CLASS-TEMP", {"alpha": np.array([1.0, 1.5, 0.9])}),
                      ("cal__s0__H-TOKEN32", {"q1": np.full((4, 2), 0.25)})):
        d = s / "run" / "units" / u
        d.mkdir()
        np.savez(d / "table.npz", **arrays)
        (d / "record.json").write_text("{}")
        _complete(d)
    d = s / "run" / "units" / "rel__s0__P"
    d.mkdir()
    np.savez(d / "release.npz", tok1=np.array([0, 1, 1]), hard1=np.array([0, 1, 1]), hard2=np.array([2, 0, 5]))
    np.savez(d / "preds.npz", p=np.array([0.1, 0.7, 0.4]))
    (d / "record.json").write_text("{}")
    _complete(d)
    return s, rec


def _copy(tmp: Path, teacher=False):
    s, rec = _store(tmp, teacher)
    out = CO.same_device_copy(src=s, deps=_input(tmp), dep_sha=INPUT_SHA)
    return s, rec, out


# ------------------------------------------------------------------ drive detection
def test_locate_drive_by_content_skips_names_and_withholds_them(tmp_path):
    root, want = _volumes(tmp_path)
    vol, ev = CO.locate_drive(root, want)
    assert vol is not None and vol.name == "Owner Named Disk"
    assert ev["skipped_by_name"] == 2 and ev["candidate_volumes"] == 2 and ev["matching_volumes"] == 1
    assert "Owner" not in json.dumps(ev) and "Stick" not in json.dumps(ev)
    vol, ev = CO.locate_drive(_volumes(tmp_path / "b", match=False)[0], want)
    assert vol is None and ev["mounted"] is False and ev["matching_volumes"] == 0


def test_skipped_installer_image_is_never_read(tmp_path, monkeypatch):
    from dpc import closeout as DC
    root, want = _volumes(tmp_path)
    seen = []
    real = DC.sha

    def spy(p, nocache=False):
        seen.append(str(p))
        return real(p, nocache)
    monkeypatch.setattr(DC, "sha", spy)
    CO.locate_drive(root, want)
    assert seen and not any("BackgroundSyncService" in x or "Macintosh HD" in x for x in seen)


def test_off_device_copy_is_pending_without_the_drive_and_writes_nothing(tmp_path):
    root, want = _volumes(tmp_path, match=False)
    s, _ = _store(tmp_path)
    before = sorted(str(p) for p in root.rglob("*"))
    r = CO.off_device_copy(root, src=s, deps=_input(tmp_path), dep_sha=INPUT_SHA, want=want)
    assert r["status"] == "PENDING" and "hcal.closeout offdevice" in r["pending_command"]
    assert sorted(str(p) for p in root.rglob("*")) == before
    assert not list((tmp_path / "cache").glob("hcal_v1_local_copy_*"))


def test_off_device_copy_goes_to_the_content_matched_volume_and_rereads_prior_folders(tmp_path):
    root, want = _volumes(tmp_path)
    s, _ = _store(tmp_path)
    r = CO.off_device_copy(root, src=s, deps=_input(tmp_path), dep_sha=INPUT_SHA, want=want)
    assert r["status"] == "OFF_DEVICE_COPY_VERIFIED" and r["label"] == "OFF_DEVICE_COPY"
    dest = root / "Owner Named Disk" / f"private_hcal_v1_{DATE}"
    assert (dest / "hcal_v1" / "START.txt").is_file() and (dest / CO.DEP_REL).read_bytes() == INPUT_BYTES
    assert r["prior_known_folders_uncached_reread"][SMF]["pass"] and r["other_folders_on_drive"] == 1
    assert not list((root / "Unrelated Stick").glob("private_hcal*"))
    txt = json.dumps(CO.public(r))
    assert "Owner" not in txt and "<DRIVE_ROOT>" in txt


# ------------------------------------------------------------------ same-device copy
def test_same_device_copy_is_versioned_verified_and_labelled_same_device(tmp_path):
    s, _, r = _copy(tmp_path)
    root = tmp_path / "cache" / f"hcal_v1_local_copy_{DATE}"
    assert r["label"] == "SAME_DEVICE_COPY" and r["verified"] and r["_root"] == root
    assert r["off_device_copy"].startswith("PENDING") and r["destination"] == f"<PRIVATE_CACHE>/{root.name}"
    sums = (root / "SHA256SUMS").read_text().splitlines()
    n_store = sum(1 for p in s.rglob("*") if p.is_file())
    assert len(sums) == n_store + 1 and any(ln.endswith(CO.DEP_REL) for ln in sums)
    assert r["uncached_readback_match"] == r["files"] == n_store + 1 and r["independent_uncached_reread"]["pass"]
    rec = json.loads((root / "BACKUP_RECORD.json").read_text())
    assert rec["label"] == "SAME_DEVICE_COPY" and "_root" not in rec
    txt = json.dumps(CO.public(r))
    assert "OFF_DEVICE_COPY_VERIFIED" not in txt and str(tmp_path) not in txt
    r2 = CO.same_device_copy(src=s, deps=_input(tmp_path), dep_sha=INPUT_SHA)
    assert r2["_root"].name == f"hcal_v1_local_copy_{DATE}_v2" and (root / "SHA256SUMS").read_text().splitlines() == sums


def test_same_device_copy_refusals_write_nothing(tmp_path, monkeypatch):
    s, _ = _store(tmp_path)
    deps = _input(tmp_path)
    with pytest.raises(SystemExit):
        CO.same_device_copy("../elsewhere", src=s, deps=deps, dep_sha=INPUT_SHA)
    with pytest.raises(SystemExit):
        CO.same_device_copy(src=s, deps=deps, dep_sha="f" * 64)                 # dependency fails its pin
    monkeypatch.setattr(CO, "free_gib", lambda p: 1.0)
    with pytest.raises(SystemExit):
        CO.same_device_copy(src=s, deps=deps, dep_sha=INPUT_SHA)                # would leave < 5 GiB free
    plan = CO.same_device_copy(src=s, deps=deps, dep_sha=INPUT_SHA, dry_run=True)
    assert plan["dry_run"] and plan["written"] == "nothing"
    assert not list((tmp_path / "cache").glob("hcal_v1_local_copy_*"))


def test_verify_sums_detects_corruption_of_the_copy(tmp_path):
    _, _, r = _copy(tmp_path)
    root = r["_root"]
    assert CO.verify_sums(root)["pass"]
    (root / "hcal_v1" / "START.txt").write_text("tampered\n")
    v = CO.verify_sums(root)
    assert not v["pass"] and v["match"] == v["entries"] - 1


# ------------------------------------------------------------------ restores from the copy alone
def test_teacher_restore_from_the_copy_alone_bitwise(tmp_path):
    s, rec, r = _copy(tmp_path, teacher=True)
    shutil.move(str(s), str(tmp_path / "live_moved_away"))                     # the restore may not need the live store
    pub = {"teacher_parity": {"U|0": {"model_sha256": rec["teachers"]["U|0"]["model_sha256"]}}}
    res = CO.restore_teacher(r["_root"], 0, "U", public_admission=pub)
    assert res["status"] == "PASS", res
    assert res["p1_bitwise"] and res["p2_bitwise"] and res["d1_bitwise"] and res["d2_bitwise"]
    assert res["model_sha256_equals_copy_receipt"] and res["input_sha256_equals_copy_receipt"]
    assert res["model_sha256_equals_tracked_admission"]
    bad = CO.restore_teacher(r["_root"], 0, "U", public_admission={"teacher_parity": {"U|0": {"model_sha256": "x"}}})
    assert bad["status"] == "FAIL"
    assert CO.restore_teacher(r["_root"], 1, "U", public_admission=False)["status"] == "FAIL"   # absent seed


def test_teacher_restore_catches_a_tampered_teacher_table(tmp_path):
    _, _, r = _copy(tmp_path, teacher=True)
    u = r["_root"] / "hcal_v1" / "admitted" / "units" / "tea__s0__U"
    z = dict(np.load(u / "teacher.npz"))
    z["p1"] = z["p1"] + 1e-15
    np.savez(u / "teacher.npz", **z)
    res = CO.restore_teacher(r["_root"], 0, "U", public_admission=False)
    assert res["status"] == "FAIL" and not res["teacher_unit"]["ok"]
    _complete(u)                                                                 # even with a re-forged COMPLETE.json
    res = CO.restore_teacher(r["_root"], 0, "U", public_admission=False)
    assert res["status"] == "FAIL" and not res["p1_bitwise"]


def test_bank_tables_match_the_admission_receipts(tmp_path):
    _, rec, r = _copy(tmp_path)
    keys = list(rec["bank"])
    pub = {"frozen_bank": rec["bank"]}
    res = CO.restore_bank_tables(r["_root"], keys, public_admission=pub)
    assert res["status"] == "PASS" and res["checked"] == 2 and res["tracked_admission_compared"]
    other = {"frozen_bank": {keys[0]: {"bank_sha256": "0" * 64}, keys[1]: rec["bank"][keys[1]]}}
    res = CO.restore_bank_tables(r["_root"], keys, public_admission=other)
    assert res["status"] == "FAIL" and res["tables"][keys[0]]["status"] == "FAIL"
    assert CO.restore_bank_tables(r["_root"], ["2|U|CLASS|i1o1"], public_admission=pub)["status"] == "FAIL"
    p = r["_root"] / "hcal_v1" / "admitted" / "bank" / "bank__s0__U_DIRECT-TASK_i8o64.npz"
    p.write_bytes(p.read_bytes() + b"\0")
    assert CO.restore_bank_tables(r["_root"], keys, public_admission=pub)["status"] == "FAIL"


def _rebuild_from(src_store):
    """A stand-in rebuild hook that recomputes nothing: it returns the ORIGINAL arrays captured before the copy."""
    cache = {}
    for d in (src_store / "run" / "units").iterdir():
        for f in ("table.npz", "release.npz"):
            if (d / f).is_file():
                cache[(d.name, f)] = dict(np.load(d / f))

    def table(store, unit, **kw):
        return cache[(unit, "table.npz")]

    def release(store, unit, **kw):
        return cache[(unit, "release.npz")]
    return table, release


def test_family_tables_hook_passes_fails_and_reports_missing_families(tmp_path):
    s, _, r = _copy(tmp_path)
    table, _ = _rebuild_from(s)
    fams = {"H-GLOBAL-TEMP": ["cal__s0__H-GLOBAL-TEMP"], "H-CLASS-TEMP": ["cal__s0__H-CLASS-TEMP"],
            "H-TOKEN32": ["cal__s0__H-TOKEN32"]}
    res = CO.restore_family_tables(r["_root"], fams, table, live_root=s)
    assert res["status"] == "PASS" and set(res["families"]) == set(I.NEW_DECODERS)
    res = CO.restore_family_tables(r["_root"], {k: v for k, v in fams.items() if k != "H-TOKEN32"}, table)
    assert res["status"] == "FAIL" and res["families"]["H-TOKEN32"]["reason"].startswith("MISSING")
    u = r["_root"] / "hcal_v1" / "run" / "units" / "cal__s0__H-CLASS-TEMP"
    np.savez(u / "table.npz", alpha=np.array([1.0, 1.5, 0.9000001]))
    _complete(u)
    res = CO.restore_family_tables(r["_root"], fams, table)
    assert res["families"]["H-CLASS-TEMP"]["status"] == "FAIL" and res["status"] == "FAIL"
    assert CO.restore_family_tables(r["_root"], fams, lambda store, unit: {})["status"] == "FAIL"   # key missing


def test_release_hook_checks_bitwise_and_decision_preservation(tmp_path):
    s, _, r = _copy(tmp_path, teacher=True)
    _, release = _rebuild_from(s)
    res = CO.restore_release(r["_root"], "rel__s0__P", release, role="control")
    assert res["status"] == "PASS" and res["role"] == "control"
    res = CO.restore_release(r["_root"], "rel__s0__P", release, decision_teacher=(0, "U"))
    assert res["status"] == "FAIL" and "decisions_preserved" in res    # synthetic hard arrays != teacher decisions
    res = CO.restore_release(r["_root"], "rel__s0__P", lambda store, unit: {**release(store, unit), "tok1": np.zeros(3)})
    assert res["status"] == "FAIL"
    assert CO.restore_release(r["_root"], "rel__s0__absent", release)["status"] == "FAIL"


def test_reader_refit_hook_compares_with_saved_predictions(tmp_path):
    s, _, r = _copy(tmp_path)
    saved = np.array([0.1, 0.7, 0.4])
    seen = {}

    def refit(store, unit, D=None, **kw):
        seen["store"] = store
        return saved + kw.get("eps", 0.0)
    res = CO.restore_reader_refit(r["_root"], "rel__s0__P", refit, "p", live_root=s)
    assert res["status"] == "PASS" and res["max_abs_diff"] == {"copy": 0.0, "live": 0.0}
    assert seen["store"] == r["_root"] / "hcal_v1"
    assert CO.restore_reader_refit(r["_root"], "rel__s0__P", refit, "p", kwargs={"eps": 1e-3})["status"] == "FAIL"
    assert CO.restore_reader_refit(r["_root"], "rel__s0__P", refit, "p", tolerance=1e-2,
                                   kwargs={"eps": 1e-3})["status"] == "PASS"
    assert CO.restore_reader_refit(r["_root"], "rel__s0__P", refit, "missing_key")["status"] == "FAIL"


def test_hooks_accept_only_callables_or_hcal_entry_points(tmp_path):
    _, _, r = _copy(tmp_path)
    for spec in ("os:system", "lra.closeout:backup", "hcal.closeout", "hcal.closeout:public; rm"):
        with pytest.raises(SystemExit):
            CO.restore_unit_tables(r["_root"], ["cal__s0__H-TOKEN32"], spec, "table.npz")
    assert CO._resolve("hcal.closeout:public") is CO.public


def test_restore_summary_is_truthful():
    s = CO.restore_summary({"U teacher": {"t": {"status": "PASS"}}, "frozen bank": {"status": "PASS"}})
    assert s["required_restores"]["calibrator families"] == "PENDING" and not s["restore_all_pass"]
    assert not s["complete"]
    na = {"selected release": "NO_ELIGIBLE_COMPETITIVE_NOMINEE", "selected reader refit": "no nominee"}
    s = CO.restore_summary({"U teacher": {"status": "PASS"}, "frozen bank": {"status": "PASS"},
                            "calibrator families": {"status": "PASS"}, "control release": {"status": "PASS"}}, na)
    assert s["restore_all_pass"] and s["complete"]
    assert s["required_restores"]["selected release"].startswith("NOT_APPLICABLE (NO_ELIGIBLE")
    s = CO.restore_summary({"U teacher": {"a": {"status": "PASS"}, "b": {"status": "FAIL"}}})
    assert s["required_restores"]["U teacher"] == "FAIL"
    for bad in ({"U teacher": "x"}, {"frozen bank": "x"}, {"control release": " "}):
        with pytest.raises(SystemExit):
            CO.restore_summary({}, bad)
    with pytest.raises(SystemExit):
        CO.restore_summary({"control release": {"status": "PASS"}}, {"control release": "why"})


def test_backup_verification_never_upgrades_a_same_device_copy(tmp_path):
    _, _, r = _copy(tmp_path)
    bv = CO.backup_verification(r, checks={"U teacher": {"status": "PASS"}})
    assert bv["custody_status"] == "SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_PENDING"
    assert bv["off_device"]["status"] == "PENDING" and bv["restore_kind"].startswith("restore test from a SAME-DEVICE")
    assert not bv["restore_all_pass"] and "_root" not in bv["copy"]
    ri = CO.restore_index(r, CO.restore_summary({}))
    assert ri["copy"].endswith("(same device; off-device copy PENDING)")
    CO.write_public(tmp_path / "pkg" / "BACKUP_VERIFICATION.json", bv)
    assert json.loads((tmp_path / "pkg" / "BACKUP_VERIFICATION.json").read_text())["copy_label"] == "SAME_DEVICE_COPY"


def test_load_D_from_copy_refuses_a_missing_or_unpinned_input_without_loading(tmp_path):
    _, _, r = _copy(tmp_path)
    with pytest.raises(SystemExit, match="fails its pinned hash"):
        CO.load_D_from_copy(r["_root"], want_sha="e" * 64)
    with pytest.raises(SystemExit, match="no bundled input"):
        CO.load_D_from_copy(tmp_path / "nowhere")


def test_write_public_refuses_identifying_text_and_nulls_nonfinite(tmp_path):
    p = tmp_path / "pkg" / "R.json"
    CO.write_public(p, {"a": float("nan"), "b": np.float64(np.inf), "_private": "/" + "Users" + "/x", "c": Path("/x")})
    assert json.loads(p.read_text()) == {"a": None, "b": None, "c": "<PATH>"}
    for bad in ({"x": "/" + "Users" + "/someone/f"}, {"x": "/" + "Volumes" + "/Disk"}, {"x": f"made by {USER}"}):
        with pytest.raises(SystemExit):
            CO.write_public(p, bad)


# ------------------------------------------------------------------ accounting, scan, processes, status
def _t(s):
    return f"2026-10-07T{s}Z"


def test_accounting_sums_cpu_by_role_and_finds_the_peak_concurrency(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    ev = [("acquire", "A:admit", 1, 0, "00:00:00", {}), ("acquire", "F:tests", 2, 1, "00:05:00", {}),
          ("release", "F:tests", 2, 1, "00:06:00", {"cpu_s": 50.0, "wall_s": 60.0}),
          ("acquire", "E:verify", 3, 1, "00:06:00", {}),
          ("release", "A:admit", 1, 0, "00:10:00", {"cpu_s": 590.0, "wall_s": 600.0, "child_maxrss_bytes": 7}),
          ("release", "E:verify", 3, 1, "00:20:00", {"cpu_s": 800.0, "wall_s": 840.0}),
          ("acquire", "A:audit", 4, 0, "00:30:00", {})]
    with open(run / "SEMA_LOG.jsonl", "w") as f:
        for e, lab, pid, slot, at, extra in ev:
            f.write(json.dumps({"at": _t(at), "event": e, "label": lab, "slot": slot, "wrapper_pid": pid, **extra}) + "\n")
        f.write("not json\n")
    (run / "COMPUTE_LEDGER.jsonl").write_text(json.dumps({"at": _t("00:10:00"), "stage": "admit", "shard": None,
                                                          "wall_s": 590.0, "cpu_s": 580.0, "maxrss_bytes": 5}) + "\n")
    start = tmp_path / "START.txt"
    start.write_text(_t("00:00:00") + "\n")
    now_ts = calendar.timegm(time.strptime(_t("01:00:00"), "%Y-%m-%dT%H:%M:%SZ"))
    a = CO.accounting(run, start, now_ts)
    assert a["cpu_s_by_role"]["A"]["cpu_s"] == 590.0 and a["cpu_s_by_role"]["E"]["cpu_s"] == 800.0
    assert a["cpu_s_by_role"]["F"]["holds"] == 1 and a["released_holds"] == 3
    assert a["max_concurrent_holds"] == 2 and a["slot_limit_respected"]                 # F ends as E starts
    assert a["elapsed_h"] == 1.0 and a["semaphore_child_cpu_h"] == round(1440 / 3600, 3)
    assert len(a["active_holds"]) == 1 and a["active_holds"][0]["label"] == "A:audit"
    assert a["stage_ledger_cpu_s_by_stage"]["admit"]["cpu_s"] == 580.0 and a["unparsed_lines"] == 1
    assert a["budget"]["cpu_h_remaining"] == round(20 - 1440 / 3600, 3)


def test_accounting_flags_more_than_two_concurrent_holds(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "SEMA_LOG.jsonl", "w") as f:
        for pid in (1, 2, 3):
            f.write(json.dumps({"at": _t("00:00:00"), "event": "acquire", "label": "X:y", "slot": pid,
                                "wrapper_pid": pid}) + "\n")
    a = CO.accounting(run, tmp_path / "no_start", calendar.timegm(time.strptime(_t("00:01:00"), "%Y-%m-%dT%H:%M:%SZ")))
    assert a["max_concurrent_holds"] == 3 and not a["slot_limit_respected"] and a["elapsed_h"] is None


def test_identity_scan_finds_and_masks_identifying_content(tmp_path):
    secret = "AKIA" + "Q" * 16
    files = {"paths.md": "see /" + "Users" + "/someone/notes and /" + "Volumes" + "/Disk/x",
             "user.md": f"written by {USER} today",
             "mail.md": "contact a.person" + "@" + "example.org",
             "cred.txt": f"key {secret}\n" + "-----BEGIN " + "RSA PRIVATE KEY-----",
             "rec.csv": "39, State-gov, Bachelors, Male, " + "<" + "=50K",
             "rows.json": '{"row_id": [' + ", ".join(str(i) for i in range(60)) + "]}",
             "home.md": "run " + "~" + "/PCRL/.venv/bin/python",
             "clean.md": "store <PRIVATE_CACHE>/hcal_v1; remote git" + "@" + "github.com:org/repo; Male share 0.67"}
    for n, txt in files.items():
        (tmp_path / n).write_text(txt)
    (tmp_path / "blob.npz").write_bytes(b"PK\0\0binary")
    out = CO.identity_scan([tmp_path / n for n in list(files) + ["blob.npz"]], user_name=USER)
    kinds = {}
    for f in out["findings"]:
        kinds.setdefault(f["file"].split("/")[-1], set()).add(f["kind"])
    assert {"home_path", "volume_or_private_path"} <= kinds["paths.md"]
    assert kinds["user.md"] == {"user_name"} and kinds["mail.md"] == {"email"}
    assert {"credential:aws_access_key_id", "credential:private_key_block"} <= kinds["cred.txt"]
    assert kinds["rec.csv"] == {"per_person_record:adult_record_line"}
    assert kinds["rows.json"] == {"per_person_record:per_row_array"}
    assert kinds["home.md"] == {"home_relative_path"} and "clean.md" not in kinds
    assert out["skipped_binary"][0]["reason"] == "binary"
    txt = json.dumps(out)
    for leak in (secret, USER, "someone", "a.person", "State-gov"):
        assert leak not in txt
    assert all(f["severity"] == "review" for f in out["findings"] if f["kind"] == "home_relative_path")


def test_identity_scan_of_the_owned_files_finds_nothing_identifying():
    own = [WT / "hcal" / "closeout.py", Path(__file__)] + [WT / I.REL / f for f in
                                                          ("PRIOR_ART_AND_CLAIM_SCOPE.md", "TEAM_PLAN.md")]
    out = CO.identity_scan([p for p in own if p.is_file()], user_name=Path.home().name)
    assert out["identifying"] == 0, [f for f in out["findings"] if f["severity"] == "identifying"]


def test_owned_processes_filters_scrubs_and_never_kills(monkeypatch):
    def no_kill(*a, **k):
        raise AssertionError("closeout must never kill a process")
    monkeypatch.setattr(os, "kill", no_kill)
    home = "/" + "Users" + "/" + USER
    ps = "\n".join([
        f"  101     1   01:02:03   00:10.50  20480  12.5 python -m hcal.sema --label A:audit -- env x",
        f"  102   101   01:00:00 1-02:03:04.00  40960  99.0 python -m hcal.run --lock L --stage audit",
        f"  103     1      00:05   00:00.01   1024   0.0 /bin/zsh -c something else",
        f"  104     1      00:05   00:01.00   2048   1.0 python -P {home}/w/hcal/sema.py --label E:verify -- y",
        f"  105     1      00:05   00:01.00   2048   1.0 python -m lra.run --stage audit"])
    out = CO.owned_processes(ps_output=ps)
    assert [p["pid"] for p in out["processes"]] == [101, 102, 104] and out["count"] == 3
    assert out["processes"][1]["cpu_s"] == 93784.0 and out["processes"][0]["rss_mib"] == 20.0
    assert USER not in json.dumps(out) and out["killed"].startswith("nothing")


def test_status_report_composes_without_writing(tmp_path, monkeypatch):
    s, _ = _store(tmp_path)
    (s / "run" / "ACTIVITY_LOG.jsonl").write_text(
        json.dumps({"at": _t("00:00:00"), "event": "start calibrate", "shard": "0/2"}) + "\n" +
        json.dumps({"at": _t("00:01:00"), "event": "unit complete", "unit": "x"}) + "\n")
    monkeypatch.setattr(CO, "owned_processes", lambda: {"processes": [], "count": 0, "rss_mib_total": 0.0})
    monkeypatch.setattr(CO, "drive_status", lambda: {"mounted": False, "off_device_copy": "PENDING"})
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    r = CO.status_report(expected={"cal": 5, "rel": 1}, findings=["B-3 open"], priv=s)
    assert r["units_completed"] == {"cal": 3, "rel": 1} and r["units_remaining"] == {"cal": 2, "rel": 0}
    assert r["stage"]["stage"] == "calibrate" and r["stage"]["event"] == "start"
    assert r["unresolved_findings"] == ["B-3 open"] and r["free_disk_gib"] is not None
    assert r["accounting"]["start"] == "2026-10-07T00:00:00Z"
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
