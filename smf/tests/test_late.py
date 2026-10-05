"""Inference end-to-end on synthetic assessment predictions (nothing touches real units or the results folder)."""
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rgj import finalize as FN
from smf import data as DA
from smf import eval_lock as EL
from smf import family as FAM
from smf import infer as INF
from smf import run as R


def fake_preds(rng, n, sex, y_inc, y_occ, strength):
    out = {"assess_row_id": np.arange(n), "assess_unit": np.arange(n) // 2, "sex": sex, "race": rng.integers(0, 3, n),
           "y_income": y_inc, "y_occ": y_occ, "hard1": np.where(rng.random(n) < 0.85, y_inc, 1 - y_inc),
           "hard2": np.where(rng.random(n) < 0.5, y_occ, (y_occ + 1) % 6), "p1": np.full((n, 2), 0.5),
           "p2": np.full((n, 6), 1 / 6)}
    for w in ("v1", "v2", "pair", "p1", "p2", "ppair", "h1", "h2", "hpair"):
        s = strength.get(w, 0.3)
        P = np.empty((3, n, 2))
        for a in range(3):
            z = s * (2 * sex - 1) + rng.normal(size=n)
            p = 1 / (1 + np.exp(-z))
            P[a] = np.stack([1 - p, p], 1)
        out[f"P_auc_{w}"] = P
        out[f"P_ce_{w}"] = P
    out["race_pos"] = np.arange(0, n, 3)
    out["race_y"] = out["race"][out["race_pos"]] % 2
    for w in ("v1", "v2", "pair"):
        out[f"Prace_auc_{w}"] = np.full((3, len(out["race_pos"]), 2), 0.5)
    return out


def test_inference_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 400
    sex = rng.integers(0, 2, n)
    y_inc, y_occ = rng.integers(0, 2, n), rng.integers(0, 6, n)
    monkeypatch.setattr(R, "UNITS", tmp_path / "units")
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path / "pkg")
    monkeypatch.setattr(FAM, "B", 60)
    labels = FAM.SCORED
    strength = {lab: {"pair": (0.6 if lab == "J-F" else 1.2), "v1": 0.5, "v2": 0.5} for lab in labels}
    for k in (0, 1, 2):
        for lab in labels:
            p = fake_preds(np.random.default_rng([k, labels.index(lab)]), n, sex, y_inc, y_occ, strength[lab])
            FN.save_unit(R.U(f"outer__s{k}__{INF.safe(lab)}"), {"preds.npz": lambda q, p=p: np.savez(q, **p)}, {"x": 1})
    D = DA.load()
    lock = {"sex_prior_defense_fit_sha256": EL.prior_hash(D),
            "seeds": {str(k): {"valid_reference": True, "score": {lab: {} for lab in labels},
                               "status": {"J-F": "NOMINEE", "L-F": "NOMINEE"}, "comparator": {"arm": "RAW-J"}}
                      for k in (0, 1, 2)}}
    lp = tmp_path / "EL.json"
    lp.write_text(json.dumps(lock))
    out = INF.main(["--evaluation-lock", str(lp)])
    z = np.load(R.U("outer__s0__L-F") / "preds.npz")
    a = np.mean([roc_auc_score(sex, z["P_auc_pair"][s][:, 1]) for s in range(3)])
    assert out["levels"]["R|0|L-F|prim|pair"]["point"] == pytest.approx(a, abs=1e-12)
    p10 = next(e for e in out["primary"] if e["id"] == "P10")
    assert p10["point"] > 0.05                        # C* = RAW-J resolved per seed
    p13 = next(e for e in out["primary"] if e["id"] == "P13")
    p04 = next(e for e in out["primary"] if e["id"] == "P04")
    assert p13["point"] == p04["point"]
    assert len(out["secondary"]) == FAM.SECONDARY_SIZE
    assert (tmp_path / "pkg" / "PRIMARY_ENDPOINTS.csv").exists()
    bad = dict(lock, sex_prior_defense_fit_sha256="0" * 64)
    lp.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        INF.main(["--evaluation-lock", str(lp)])


def test_conjunction():
    ids = [e["id"] for e in FAM.PRIMARY if e["claim"] == "A"]
    ok = {k: {"valid_reference": True, "J-F": "NOMINEE", "L-F": "NOMINEE", "C*": "RAW-J"} for k in (0, 1, 2)}
    assert FAM.claim_decision("A", {i: "PASS" for i in ids}, ok)["decision"] == "PASS"
    assert FAM.claim_decision("A", {**{i: "PASS" for i in ids}, "P02": "NOT_ESTABLISHED"}, ok)["decision"] == "NOT_ESTABLISHED"
    bad = {**ok, 1: {**ok[1], "J-F": "NO_FEASIBLE_NOMINEE"}}
    assert FAM.claim_decision("A", {i: "PASS" for i in ids}, bad)["decision"] == "NOT_ESTABLISHED"
