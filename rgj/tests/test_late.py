"""Tests of the later-locked lead modules (inference) on synthetic assessment predictions; nothing touches real units."""
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rgj import family as FAM
from rgj import finalize as FN
from rgj import infer as INF
from rgj import run as R

LABELS = FAM.SCORED


def fake_preds(rng, n, sex, y_inc, y_occ, strength):
    out = {"assess_row_id": np.arange(n), "assess_unit": np.arange(n) // 2, "sex": sex, "race": rng.integers(0, 3, n),
           "y_income": y_inc, "y_occ": y_occ, "hard1": np.where(rng.random(n) < 0.85, y_inc, 1 - y_inc),
           "hard2": np.where(rng.random(n) < 0.5, y_occ, (y_occ + 1) % 6), "p1": np.full((n, 2), 0.5),
           "p2": np.full((n, 6), 1 / 6)}
    for w in ("v1", "v2", "pair", "p1", "p2", "ppair", "h1", "h2", "hpair"):
        s = strength[w] if w in strength else 0.3
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
    strength = {lab: {"pair": (0.6 if lab == "J-G" else 1.2), "v1": 0.5, "v2": 0.5} for lab in LABELS}
    for k in (0, 1, 2):
        for lab in LABELS:
            p = fake_preds(np.random.default_rng([k, LABELS.index(lab)]), n, sex, y_inc, y_occ, strength[lab])
            FN.save_unit(R.U(f"outer__s{k}__{INF.safe(lab)}"), {"preds.npz": lambda q, p=p: np.savez(q, **p)}, {"x": 1})
    EL = {"sex_prior_defense_fit": [0.32, 0.68],
          "seeds": {str(k): {"valid_reference": True, "score": {lab: {} for lab in LABELS},
                             "status": {"J-G": "NOMINEE", "L-G": "NOMINEE"}, "comparator": {"arm": "L-G"}}
                    for k in (0, 1, 2)}}
    lp = tmp_path / "EL.json"
    lp.write_text(json.dumps(EL))
    out = INF.main(["--evaluation-lock", str(lp)])
    p01 = next(e for e in out["primary"] if e["id"] == "P01")
    z = np.load(R.U("outer__s0__L-G") / "preds.npz")
    zj = np.load(R.U("outer__s0__J-G") / "preds.npz")
    a_lg = np.mean([roc_auc_score(sex, z["P_auc_pair"][a][:, 1]) for a in range(3)])
    a_jg = np.mean([roc_auc_score(sex, zj["P_auc_pair"][a][:, 1]) for a in range(3)])
    assert out["levels"]["R|0|L-G|prim|pair"]["point"] == pytest.approx(a_lg, abs=1e-12)
    assert out["levels"]["R|0|J-G|prim|pair"]["point"] == pytest.approx(a_jg, abs=1e-12)
    assert p01["point"] > 0.05 and p01["decision"] in ("PASS", "NOT_ESTABLISHED")
    p13 = next(e for e in out["primary"] if e["id"] == "P13")
    p04 = next(e for e in out["primary"] if e["id"] == "P04")
    assert p13["point"] == p04["point"] and p13["alias_of"] == "P04"
    assert out["claimA"]["decision"] == FAM.claim_decision("A", {e["id"]: e["decision"] for e in out["primary"]
                                                                  if e["claim"] == "A"}, out["claimA"]["seed_status"])["decision"]
    assert (tmp_path / "pkg" / "PRIMARY_ENDPOINTS.csv").exists() and (tmp_path / "pkg" / "RAW_LEVELS.csv").exists()
    assert len(out["secondary"]) == FAM.SECONDARY_SIZE
