"""Custody-owner negative controls for osf.admit: each admission check must FAIL on a planted defect (real roles,
admitted smf units; light: one loader call, a few units). Assessment labels stay sealed."""
import json
import shutil

import joblib
import numpy as np
import pytest

from osf import admit as AD


@pytest.fixture(scope="module")
def ctx():
    c = AD.Ctx()
    return c


def test_head_scaler_check_fails_on_wrong_fit_rows(ctx):
    d = AD.SMF_UNITS / "tl__s0__e40"
    z = np.load(d / "release.npz")
    ix = ctx.rows_of(z["row_id"])
    rec = json.loads((d / "record.json").read_text())
    h = joblib.load(d / "head_0.joblib")
    good = AD.head_checks(ctx, h, z["r1"], ix, ctx.D["y"]["income"], 2, rec["heads"]["0"])
    assert good["scaler_mean_var_bitwise_OSF_DEFENSE_FIT"] and good["HEAD_VALIDATION_log_loss_equals_recorded"]
    real = ctx.fit
    try:                                  # one OSF_DEFENSE_FIT row swapped for an AUDIT_FIT row
        ctx.fit = np.sort(np.concatenate([real[1:], ctx.D["idx"]["AUDIT_FIT"][:1]]))
        bad = AD.head_checks(ctx, h, z["r1"], ix, ctx.D["y"]["income"], 2, rec["heads"]["0"])
    finally:
        ctx.fit = real
    assert not bad["scaler_mean_var_bitwise_OSF_DEFENSE_FIT"]


def test_head_validation_check_fails_on_wrong_recorded_table(ctx):
    d = AD.SMF_UNITS / "raw__s1__RAW-J__b0.3__e40"
    z = np.load(d / "release.npz")
    ix = ctx.rows_of(z["row_id"])
    rec = json.loads((d / "record.json").read_text())["heads"]["1"]
    rec = {"selected_C": rec["selected_C"], "table": [dict(t) for t in rec["table"]]}
    for t in rec["table"]:
        t["defense_val_log_loss"] += 1e-9
    r = AD.head_checks(ctx, joblib.load(d / "head_1.joblib"), z["r2"], ix, ctx.D["y"]["occupation_group"], 6, rec)
    assert not r["HEAD_VALIDATION_log_loss_equals_recorded"]


def test_fare_tree_check_fails_when_one_fit_row_differs(ctx):
    uid = "smf__s0__p0__c1"
    assert AD.check_tree(ctx, uid)["pass"]
    X0 = ctx.D["X"].copy()
    try:
        ctx.D["X"][ctx.fit[0], 0] += np.float32(1e-3)
        t = AD.check_tree(ctx, uid)
    finally:
        ctx.D["X"] = X0
    assert not t["fit_rows_sha256_equals_OSF_DEFENSE_FIT"] and not t["pass"]


def test_copy_refuses_a_differing_admitted_file(tmp_path):
    src = AD.SMF_UNITS / "warm__s0"
    files = json.loads((src / "COMPLETE.json").read_text())["files"]
    dst = tmp_path / "warm__s0"
    rec = AD.copy_unit(src, dst, files)
    assert rec["uncached_reread_matches_source"] and rec["copied_now"] == len(files) + 1
    assert AD.copy_unit(src, dst, files)["already_identical"] == len(files) + 1
    (dst / "record.json").write_text("{}")
    with pytest.raises(SystemExit):
        AD.copy_unit(src, dst, files)
    shutil.rmtree(dst)


def test_api_refuses_unadmitted_and_tampered(tmp_path, monkeypatch):
    with pytest.raises(KeyError):
        AD.admitted_path("B__s0__J-F__r0.25__e20")
    with pytest.raises(KeyError):
        AD.admitted_path("tl__s0__e20")
    rec = AD.admission_record()
    src = AD.ADMITTED / "warm__s0"
    fake = tmp_path / "admitted"
    shutil.copytree(src, fake / "warm__s0")
    (fake / "warm__s0" / "warm.pt").write_bytes(b"tampered")
    monkeypatch.setattr(AD, "ADMITTED", fake)
    with pytest.raises(RuntimeError):
        AD.admitted_path("warm__s0")
    assert rec["verdict"] == "ADMITTED"
