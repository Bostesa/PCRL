"""Synthetic end-to-end tests of hcal.infer (role A): 23-slot family preserved, nominee/comparator status cases,
denominator and seed alignment, diagnostic slots scored without a nominee. Temporary unit stores only."""
from __future__ import annotations

import json

import numpy as np
import pytest

from hcal import family as FAM
from hcal import ids as I
from hcal import infer as INF

N, G = 600, 400
DP = FAM.DIAG_PARTITION


def _write(root, name, arrays):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "preds.npz", **arrays)


def _store(tmp_path, rids, seed_shift=None, conf=None):
    rng = np.random.default_rng(0)
    unit = np.sort(rng.integers(0, G, N))
    sex = rng.integers(0, 2, N)
    y1, y2 = rng.integers(0, 2, N), rng.integers(0, 6, N)
    base = {"assess_row_id": np.arange(N) * 3, "assess_unit": unit, "sex": sex, "y_income": y1, "y_occ": y2,
            "const_class": np.array([0, 1])}
    conf = conf or {}
    for k in I.SEEDS:
        keys = set()
        for rid in rids:
            c = conf.get(rid, 0.6)
            p1 = np.full((N, 2), (1 - c) / 1)
            p1[:, 0], p1[:, 1] = c, 1 - c
            p2 = np.full((N, 6), (1 - c) / 5)
            p2[:, 1] = c
            arr = {**base, "prob1": p1, "prob2": p2, "hard1": np.zeros(N, int), "hard2": np.ones(N, int)}
            if seed_shift is not None and k == 1 and rid == rids[-1]:
                arr["assess_row_id"] = arr["assess_row_id"] + 1
            _write(tmp_path, INF.prob_name(k, rid), arr)
            keys.add(INF.partition_key(rid))
        for key in keys:
            sig = 0.3 if key == "SRC|U" else (0.2 if key == DP else 0.25)
            att = {"assess_row_id": base["assess_row_id"]}
            for crit in ("auc", "ce"):
                for v in ("v1", "v2", "pair"):
                    s = np.clip(0.5 + sig * (sex - 0.5) + 0.2 * rng.standard_normal((3, N)), 0.01, 0.99)
                    att[f"P_{crit}_{v}"] = np.stack([1 - s, s], -1)
            _write(tmp_path, INF.att_name(k, key), att)
    return base


def _EL(rids, nominee=None, tref=None, nstate="NO_ELIGIBLE", tstate="ELIGIBLE"):
    return {"technical_validity": {"ok": True, "engineering_gate": "ENGINEERING_READY"}, "scored_releases": rids,
            "resolved": {"P*": nominee, "T*": tref, "Ucal*": I.U_ID, "U0": I.U_ID},
            "statuses": {"P*": {"state": nstate}, "T*": {"state": tstate}}}


DIAG = [I.release_id(DP, d) for d in ("T-TOKEN32", "H-TOKEN32", "H-GLOBAL-TEMP")]


def test_no_nominee_keeps_23_slots_and_scores_diagnostics(tmp_path):
    rids = [I.U_ID, "U|FINE-TASK|i8o64"] + DIAG
    _store(tmp_path, rids, conf={DIAG[0]: 0.55, DIAG[1]: 0.6, DIAG[2]: 0.65})
    el = tmp_path / "EL.json"
    el.write_text(json.dumps(_EL(rids, tref="U|FINE-TASK|i8o64")))
    out = INF.main(["--evaluation-lock", str(el)], units=tmp_path, run_dir=tmp_path, pkg_dir=tmp_path)
    assert len(out["primary"]) == 23
    assert all(e["decision"] == "NOT_SCORED_NO_NOMINEE" for e in out["primary"] if e["id"].startswith("P"))
    d = {e["id"]: e for e in out["primary"] if e["id"].startswith("D")}
    assert all(np.isfinite(d[i]["point"]) for i in d)
    assert out["label"] == "NO_ELIGIBLE_COMPETITIVE_NOMINEE"
    assert out["n_assessment"] == N and out["n_groups"] == len(np.unique(np.sort(np.random.default_rng(0)
                                                                                .integers(0, G, N))))
    assert out["OriginalCriterion"] == "NOT_TESTED_NO_NOMINEE"


def test_nominee_scores_all_p_slots(tmp_path):
    nom = "U|JOINT|i8o64|l0.1|H-GLOBAL-TEMP"
    rids = [I.U_ID, "U|FINE-TASK|i8o64", nom] + DIAG
    _store(tmp_path, rids)
    el = tmp_path / "EL.json"
    el.write_text(json.dumps(_EL(rids, nominee=nom, tref="U|FINE-TASK|i8o64", nstate="ELIGIBLE")))
    out = INF.main(["--evaluation-lock", str(el)], units=tmp_path, run_dir=tmp_path, pkg_dir=tmp_path)
    p = {e["id"]: e for e in out["primary"]}
    assert all(p[f"P{j:02d}"]["decision"] != "NOT_SCORED_NO_NOMINEE" for j in range(1, 16))
    assert p["P04"]["point"] == pytest.approx(0.0)          # identical decisions -> zero accuracy difference
    assert p["P01"]["point"] > 0                            # weaker leak in the diagnostic partition
    assert out["label"] in FAM.LABELS


def test_seed_alignment_refused(tmp_path):
    rids = [I.U_ID, "U|FINE-TASK|i8o64"] + DIAG
    _store(tmp_path, rids, seed_shift=True)
    el = tmp_path / "EL.json"
    el.write_text(json.dumps(_EL(rids, tref="U|FINE-TASK|i8o64")))
    with pytest.raises(AssertionError):
        INF.main(["--evaluation-lock", str(el)], units=tmp_path, run_dir=tmp_path, pkg_dir=tmp_path)


def test_missing_comparator_is_invalid(tmp_path):
    rids = [I.U_ID] + DIAG
    _store(tmp_path, rids)
    el = tmp_path / "EL.json"
    el.write_text(json.dumps(_EL(rids, tref=None, tstate="NO_ELIGIBLE")))
    out = INF.main(["--evaluation-lock", str(el)], units=tmp_path, run_dir=tmp_path, pkg_dir=tmp_path)
    assert out["label"] == "INCOMPLETE_OR_INVALID"


def test_refuses_without_engineering_gate(tmp_path):
    el = tmp_path / "EL.json"
    e = _EL([I.U_ID])
    e["technical_validity"]["engineering_gate"] = "NOT_READY"
    el.write_text(json.dumps(e))
    with pytest.raises(SystemExit):
        INF.main(["--evaluation-lock", str(el)], units=tmp_path, run_dir=tmp_path, pkg_dir=tmp_path)
