"""Lead modules on synthetic records only (temporary directories): capacity gate rate selection, inner selection
statuses (including technical failures and blocked guards) and the 37-slot inference end to end."""
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from qpc import family as FAM
from qpc import gate as GT
from qpc import run as R
from qpc import select as SEL

PROTO = {"stage_b": {"rates": [[8, 32]], "lams": [0.1]}}


# ------------------------------------------------------------------ gate
def _seed(elig, head, ll_occ, ll_inc=0.33, a1=16, a2=200, ne=0.5):
    g = {"eligible": elig, "headroom": head, "income": {"norm_excess": ne / 2, "ll_excess": 0.0, "brier_excess": 0.0},
         "occupation": {"norm_excess": ne, "ll_excess": 0.0, "brier_excess": 0.0}}
    return {"gate": g, "util": {"income": {"logloss": ll_inc}, "occupation": {"logloss": ll_occ}},
            "alpha1": a1, "alpha2": a2}


def test_gate_rate_selection_rule():
    S = {}
    S["U|DIRECT-TASK|i8o64"] = GT.summarize({k: _seed(True, True, 1.270, a2=321) for k in (0, 1, 2)})
    S["U|DIRECT-TASK|i8o32"] = GT.summarize({k: _seed(True, True, 1.272, a2=161) for k in (0, 1, 2)})
    S["U|DIRECT-TASK|i4o32"] = GT.summarize({k: _seed(True, False, 1.272, a1=8, a2=161) for k in (0, 1, 2)})
    S["U|DIRECT-TASK|i4o16"] = GT.summarize({k: _seed(k != 2, True, 1.28, a1=8, a2=81, ne=1.4) for k in (0, 1, 2)})
    d = GT.select_rates(S)
    assert d["gate"] == "CAPACITY_GATE_MET"
    assert d["selected_rates"] == ["U|DIRECT-TASK|i8o32", "U|DIRECT-TASK|i8o64"]   # headroom first, fewer states
    assert "U|DIRECT-TASK|i4o16" not in d["eligible"]                               # one seed fails -> ineligible
    S2 = {"U|DIRECT-TASK|i4o32": S["U|DIRECT-TASK|i4o32"], "U|DIRECT-TASK|i8o64": S["U|DIRECT-TASK|i8o64"]}
    assert GT.select_rates(S2)["selected_rates"] == ["U|DIRECT-TASK|i8o64", "U|DIRECT-TASK|i4o32"]  # H then E
    S3 = {"U|DIRECT-TASK|i4o16": S["U|DIRECT-TASK|i4o16"]}
    d3 = GT.select_rates(S3)
    assert d3["gate"] == "CAPACITY_GATE_NOT_MET" and d3["selected_rates"] == []
    assert d3["best_shortfall_config"] == "U|DIRECT-TASK|i4o16"


# ------------------------------------------------------------------ selection
def _util(acc_i=0.85, acc_o=0.48, ll_i=0.33, ll_o=1.27, br_i=0.21, br_o=0.65):
    return {"income": {"acc": acc_i, "logloss": ll_i, "brier": br_i, "const_acc": 0.76, "gain": acc_i - 0.76},
            "occupation": {"acc": acc_o, "logloss": ll_o, "brier": br_o, "const_acc": 0.30, "gain": acc_o - 0.30}}


def _write_inner(k, cid, pair, v1=0.69, v2=0.75, ll_o=1.275, br_o=0.653, states=180, pres=True, drop=None):
    rec = {"recovery": {"auc": {"v1": v1, "v2": v2, "pair": pair}}, "utility": _util(ll_o=ll_o, br_o=br_o),
           "preserved": {"1": pres, "2": pres}, "token_states": None if R.parse_id(cid)["kind"] != "policy" else states}
    if cid == "SRC|U":
        rec["utility"] = _util()
        rec["composed"] = {"auc": rec["recovery"]["auc"], "winner": {"pair": "source"}}
    if drop:
        rec.pop(drop)
    R.save(f"inner__{R.unit_for(k, cid)}", {}, rec)


@pytest.fixture
def tmp_study(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "UNITS", tmp_path / "units")
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path / "pkg")
    (tmp_path / "pkg").mkdir()
    monkeypatch.setattr(R, "lock_protocol", lambda: PROTO)
    return tmp_path


def _populate(joint_pair=0.78, joint_ll=1.276, missing=None, fine_ll=1.276):
    for k in R.SEEDS:
        for cid in R.scored_ids(PROTO):
            if cid == missing and k == 1:
                continue
            fam = R.parse_id(cid).get("family")
            if cid == "SRC|U":
                _write_inner(k, cid, 0.86, 0.70, 0.86, 1.27, 0.65)
            elif cid.startswith("SRC|"):
                _write_inner(k, cid, 0.79, ll_o=1.282)
            elif cid.startswith("REF|"):
                _write_inner(k, cid, 0.70, ll_o=1.315, br_o=0.676)
            elif fam == "DIRECT-TASK":
                m2 = R.parse_id(cid)["m2"]
                ll = {8: 1.29, 16: 1.282, 32: 1.276, 64: 1.273}[m2]
                _write_inner(k, cid, 0.80 + 0.005 * np.log2(m2 / 8), v2=0.75 + 0.01 * np.log2(m2 / 8), ll_o=ll,
                             br_o=0.65 + (ll - 1.27) / 3, states=m2 * 5 + 8)
            elif fam == "JOINT":
                _write_inner(k, cid, joint_pair, v2=0.74, ll_o=joint_ll, states=168)
            elif fam == "CLASS":
                _write_inner(k, cid, 0.74, ll_o=1.38, br_o=0.69, states=8)
            elif fam == "FINE-TASK":
                _write_inner(k, cid, 0.815, v2=0.77, ll_o=fine_ll, states=168)
            else:
                _write_inner(k, cid, 0.80, v2=0.76, ll_o=1.276, states=168)


def test_select_statuses_nominal(tmp_study):
    _populate()
    out = SEL.select_all()
    st = out["statuses"]
    assert st["Q*"]["status"] == "NOMINEE" and st["Q*"]["config"] == "U|DIRECT-TASK|i4o64"   # least excess, fewer states
    assert st["T*"]["status"] == "NOMINEE" and st["C_global"]["status"] == "NOMINEE"
    assert st["J*"]["status"] == "NOMINEE" and st["J*"]["config"] == "U|JOINT|i8o32|l0.1"
    assert st["C_rate"]["status"] == "NOMINEE" and st["C_rate"]["cell"] == "i8o32"
    assert st["P*"]["status"] == "NOMINEE" and st["P*"]["winning_family"] == "JOINT"
    assert out["claim_role_states"]["A"] == {"nominee": "ELIGIBLE", "comparator": "ELIGIBLE"}
    assert (tmp_study / "pkg" / "INNER_SELECTION_TABLE.csv").exists()


def test_select_no_eligible_joint_gives_fallback(tmp_study):
    _populate(joint_ll=1.29)                                   # joint fails the log-loss gate on every seed
    st = SEL.select_all()["statuses"]
    assert st["J*"]["status"] == "NO_ELIGIBLE_NOMINEE" and st["J*"]["descriptive_config"] == "U|JOINT|i8o32|l0.1"
    assert FAM.role_state(st["J*"]) == "NO_ELIGIBLE"
    assert st["C_rate"]["cell"] == "i8o32"                     # comparator resolved at the fallback's cell


def test_select_missing_unit_is_technical_failure(tmp_study):
    _populate(missing="U|LOCAL|i8o32|l0.1")
    st = SEL.select_all()["statuses"]
    assert st["C_global"]["status"] == "INVALID_COMPARATOR"
    assert st["C_rate"]["status"] == "INVALID_COMPARATOR"
    assert st["P*"]["status"] == "INVALID_NOMINEE"
    assert FAM.claim_status(FAM.role_state(st["J*"]), FAM.role_state(st["C_global"]), "ALL_PASS") == \
        "INVALID_COMPARATOR"


def test_select_missing_field_is_technical_failure(tmp_study):
    _populate()
    _write_inner(2, "REF|F", 0.70, drop="preserved")
    st = SEL.select_all()["statuses"]
    assert st["C_global"]["status"] == "INVALID_COMPARATOR"
    assert st["T*"]["status"] == "NOMINEE"                    # REF|F is not a T* candidate


# ------------------------------------------------------------------ inference end to end (synthetic preds)
def _fake(rng, n, sex, yI, yO, s_pair, shift=0.0):
    out = {"assess_row_id": np.arange(n), "assess_unit": np.arange(n) // 2, "sex": sex,
           "y_income": yI, "y_occ": yO, "const_class": np.array([0, 4])}
    for w, s in (("v1", 0.5), ("v2", 0.5), ("pair", s_pair)):
        P = np.empty((3, n, 2))
        for a in range(3):
            z = s * (2 * sex - 1) + rng.normal(size=n)
            p = 1 / (1 + np.exp(-z))
            P[a] = np.stack([1 - p, p], 1)
        out[f"P_auc_{w}"] = P
    p1 = np.clip(rng.dirichlet([2, 2], n) + shift, 1e-9, None)
    p1 /= p1.sum(1, keepdims=True)
    p2 = rng.dirichlet([1] * 6, n)
    out.update(prob1=p1, prob2=p2, hard1=p1.argmax(1), hard2=p2.argmax(1))
    return out


def test_inference_end_to_end(tmp_study, monkeypatch):
    from jcv.finalize import save_unit
    from qpc import infer as INF
    monkeypatch.setattr(FAM, "B", 50)
    rng = np.random.default_rng(0)
    n = 400
    sex = rng.integers(0, 2, n)
    yI, yO = rng.integers(0, 2, n), rng.integers(0, 6, n)
    labels = ["SRC|U", "U|JOINT|i8o32|l0.1", "U|FINE-TASK|i8o32", "U|DIRECT-TASK|i8o64", "U|LOCAL|i8o32|l0.1"]
    strength = {"SRC|U": 1.2, "U|JOINT|i8o32|l0.1": 0.6, "U|FINE-TASK|i8o32": 1.0, "U|DIRECT-TASK|i8o64": 1.1,
                "U|LOCAL|i8o32|l0.1": 0.8}
    for k in (0, 1, 2):
        for lab in labels:
            p = _fake(np.random.default_rng([k, labels.index(lab)]), n, sex, yI, yO, strength[lab])
            save_unit(R.U(f"outer__s{k}__{INF.safe(lab)}"), {"preds.npz": lambda q, p=p: np.savez(q, **p)}, {})
    st = {"J*": {"status": "NOMINEE", "config": "U|JOINT|i8o32|l0.1"},
          "C_rate": {"status": "NOMINEE", "config": "U|FINE-TASK|i8o32"},
          "C_global": {"status": "NOMINEE", "config": "SRC|U"},
          "T*": {"status": "NOMINEE", "config": "SRC|U"},
          "P*": {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": "U|LOCAL|i8o32|l0.1"},
          "Q*": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}}
    lock = {"statuses": st, "resolved": {x: s.get("config") or s.get("descriptive_config") for x, s in st.items()},
            "seeds": {str(k): {"score": {lab: {} for lab in labels}} for k in (0, 1, 2)},
            "stage_a_valid": True, "gate_met": True, "technical_validity": {"ok": True}}
    lp = tmp_study / "EL.json"
    lp.write_text(json.dumps(lock))
    out = INF.main(["--evaluation-lock", str(lp)], check_prior=False)
    assert len(out["primary"]) == 37 and all(e["z"] == FAM.Z_PRIMARY for e in out["primary"])
    z = np.load(R.U("outer__s0__U_JOINT_i8o32_l0.1") / "preds.npz")
    a = np.mean([roc_auc_score(sex, z["P_auc_pair"][s][:, 1]) for s in range(3)])
    assert out["levels"]["R#0#U|JOINT|i8o32|l0.1#primary#pair"]["point"] == pytest.approx(a, abs=1e-12)
    P01 = next(e for e in out["primary"] if e["id"] == "P01")
    assert P01["point"] > 0.05 and P01["decision"] in ("PASS", "NOT_ESTABLISHED")
    assert all(e["decision"] == "DESCRIPTIVE_ONLY" for e in out["primary"] if e["claim"] == "C")
    assert out["claim_status"]["C"] == "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
    assert {e["id"] for e in out["primary"] if e["claim"] == "Q"} == {"P34", "P35", "P36", "P37"}
    assert out["q_status"] in ("PASS", "NOT_ESTABLISHED")
    assert out["label"] in ("CONFIDENCE_FEASIBILITY_ESTABLISHED", "EXPERIMENTAL_NO_ADVANTAGE",
                            "JOINT_DEVELOPMENT_CRITERION_MET")
    assert (tmp_study / "pkg" / "PRIMARY_ENDPOINTS.csv").exists()
