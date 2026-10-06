"""Inference end to end on synthetic outer predictions (temporary directories only)."""
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dpc import data as DA
from dpc import eval_lock as EL
from dpc import family as FAM
from dpc import infer as INF
from dpc import run as R
from rgj import finalize as FN


def fake(rng, n, sex, yI, yO, s_pair, prob_shift=0.0):
    out = {"assess_row_id": np.arange(n), "assess_unit": np.arange(n) // 2, "sex": sex, "race": np.zeros(n, int),
           "y_income": yI, "y_occ": yO, "const_class": np.array([0, 4])}
    for w, s in (("v1", 0.5), ("v2", 0.5), ("pair", s_pair)):
        P = np.empty((3, n, 2))
        for a in range(3):
            z = s * (2 * sex - 1) + rng.normal(size=n)
            p = 1 / (1 + np.exp(-z))
            P[a] = np.stack([1 - p, p], 1)
        out[f"P_auc_{w}"] = P
        out[f"P_ce_{w}"] = P
        out[f"P_auc_scores_{w}"] = P
    p1 = np.clip(rng.dirichlet([2, 2], n) + prob_shift, 1e-9, None)
    p1 /= p1.sum(1, keepdims=True)
    p2 = rng.dirichlet([1] * 6, n)
    out.update(prob1=p1, prob2=p2, hard1=p1.argmax(1), hard2=p2.argmax(1))
    return out


def test_inference_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 400
    sex = rng.integers(0, 2, n)
    yI, yO = rng.integers(0, 2, n), rng.integers(0, 6, n)
    monkeypatch.setattr(R, "UNITS", tmp_path / "units")
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path / "pkg")
    (tmp_path / "pkg").mkdir()
    monkeypatch.setattr(FAM, "B", 50)
    labels = ["SRC|U", "U|JOINT|m4|l1", "U|FINE-TASK|m4", "U|LOCAL|m4|l1", "U|CLASS|m1"]
    strength = {"SRC|U": 1.2, "U|JOINT|m4|l1": 0.6, "U|FINE-TASK|m4": 1.0, "U|LOCAL|m4|l1": 0.8, "U|CLASS|m1": 0.9}
    for k in (0, 1, 2):
        for lab in labels:
            p = fake(np.random.default_rng([k, labels.index(lab)]), n, sex, yI, yO, strength[lab])
            FN.save_unit(R.U(f"outer__s{k}__{INF.safe(lab)}"), {"preds.npz": lambda q, p=p: np.savez(q, **p)}, {})
    D = DA.load()
    st = {"J*": {"status": "NOMINEE", "config": "U|JOINT|m4|l1"},
          "C_match": {"status": "NOMINEE", "config": "U|FINE-TASK|m4"},
          "C_global": {"status": "NOMINEE", "config": "U|FINE-TASK|m4"},
          "T*": {"status": "NOMINEE", "config": "U|FINE-TASK|m4"},
          "P*": {"status": "NO_FEASIBLE_NOMINEE", "config": None, "descriptive_config": "U|LOCAL|m4|l1"}}
    lock = {"sex_prior_defense_fit_sha256": EL.prior_hash(D), "statuses": st, "U_valid": True,
            "resolved": {x: s.get("config") or s.get("descriptive_config") for x, s in st.items()},
            "seeds": {str(k): {"score": {lab: {} for lab in labels}} for k in (0, 1, 2)}}
    lp = tmp_path / "EL.json"
    lp.write_text(json.dumps(lock))
    out = INF.main(["--evaluation-lock", str(lp)])
    z = np.load(R.U("outer__s0__U_JOINT_m4_l1") / "preds.npz")
    a = np.mean([roc_auc_score(sex, z["P_auc_pair"][s][:, 1]) for s in range(3)])
    assert out["levels"]["R#0#U|JOINT|m4|l1#primary#pair"]["point"] == pytest.approx(a, abs=1e-12)
    ll = -np.log(np.clip(z["prob1"][np.arange(n), yI], 1e-12, 1))
    assert out["levels"]["ll#0#U|JOINT|m4|l1#0"]["point"] == pytest.approx(ll.mean(), abs=1e-12)
    P01 = next(e for e in out["primary"] if e["id"] == "P01")
    assert P01["point"] > 0.05 and P01["z"] == FAM.Z_PRIMARY
    P12 = next(e for e in out["primary"] if e["id"] == "P12")          # C_global coalition == C_match: same value
    assert P12["point"] == P01["point"]
    assert all(e["decision"] == "DESCRIPTIVE_ONLY" for e in out["primary"] if e["claim"] == "C")
    assert out["claimC"]["decision"] == "NOT_ESTABLISHED" and out["complete"] is True
    assert "R#0#SRC|U#scores#pair" in out["levels"]
    assert len(out["primary"]) == 33


def test_amendment_a1_plant_bits_ignore_sealed_rows():
    """Regression (AMENDMENT_A1): sealed -1 labels must not create bits outside {0,1} or colliding split tokens."""
    from dpc import audit as AU
    n = 50
    rng = np.random.default_rng(1)
    tok = rng.integers(0, 4, n)
    q = np.array([[0.7, 0.3], [0.6, 0.4], [0.2, 0.8], [0.4, 0.6]])[tok]
    z = {"tok1": tok, "q1": q, "hard1": q.argmax(1), "alpha1": np.int64(4)}
    bit = rng.integers(0, 2, n)
    bit[:10] = -1                                      # sealed rows
    bit[10:12] = 2                                     # flipped sealed rows
    for collide in (True, False):
        zp = AU.split_tokens(z, 1, bit, collide=collide)
        assert zp["tok1"].min() >= 0 and zp["tok1"].max() < 8
        assert AU.token_decoder_check(zp["tok1"], zp["q1"], zp["hard1"])["ok"]
        assert np.array_equal(zp["tok1"][:12], 2 * tok[:12])
