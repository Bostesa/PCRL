"""cbp.infer and cbp.eval_lock helpers end to end on synthetic predictions (temporary directories; no real data)."""
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from cbp import eval_lock as EL
from cbp import family as FAM
from cbp import infer as INF
from cbp import run as R


def _fake(rng, n, sex, yI, yO, s_pair, s_local=0.5, noise=0.0):
    out = {"assess_row_id": np.arange(n), "assess_unit": np.arange(n) // 2, "sex": sex,
           "y_income": yI, "y_occ": yO, "const_class": np.array([0, 4])}
    for w, s in (("v1", s_local), ("v2", s_local), ("pair", s_pair)):
        P = np.empty((3, n, 2))
        for a in range(3):
            z = s * (2 * sex - 1) + rng.normal(size=n)
            p = 1 / (1 + np.exp(-z))
            P[a] = np.stack([1 - p, p], 1)
        out[f"P_auc_{w}"] = P
    base = np.random.default_rng(99)                              # same task outputs for every arm (+ small noise)
    p1 = base.dirichlet([2, 2], n)
    p2 = base.dirichlet([1] * 6, n)
    if noise:
        p1 = np.clip(p1 + noise * rng.normal(size=p1.shape), 1e-6, None)
        p1 /= p1.sum(1, keepdims=True)
    out.update(prob1=p1, prob2=p2, hard1=p1.argmax(1), hard2=p2.argmax(1))
    return out


@pytest.fixture
def tmp_study(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path)
    monkeypatch.setattr(R, "UNITS", tmp_path / "units")
    monkeypatch.setattr(FAM, "B", 60)
    return tmp_path


def _write(tmp_study, labels, strength, n=400):
    from jcv.finalize import save_unit
    rng = np.random.default_rng(0)
    sex = rng.integers(0, 2, n)
    yI, yO = rng.integers(0, 2, n), rng.integers(0, 6, n)
    for k in (0, 1, 2):
        for lab in labels:
            p = _fake(np.random.default_rng([k, labels.index(lab)]), n, sex, yI, yO, *strength[lab])
            save_unit(R.UNITS / f"outer__s{k}__{INF.safe(lab)}", {"preds.npz": lambda q, p=p: np.savez(q, **p)}, {})
    return sex


def _lock(tmp_study, st, labels, tv=True):
    res = {x: s.get("config") or s.get("descriptive_config") for x, s in st.items()}
    al = EL.role_aliases(res)
    lock = {"statuses": st, "resolved": res, "role_aliases": al, "alias_of_by_role": EL.alias_of_by_role(al),
            "seeds": {str(k): {"score": {lab: {} for lab in labels}} for k in (0, 1, 2)},
            "technical_validity": {"ok": tv}}
    lp = tmp_study / "EL.json"
    lp.write_text(json.dumps(lock))
    return lp


LABELS = ["SRC|U", "U|SEQ-21|i8o64|l0.04", "U|JOINT|i8o64|l0.04", "U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64",
          "U|LOCAL|i8o64|l0.1"]
STRENGTH = {"SRC|U": (1.2,), "U|SEQ-21|i8o64|l0.04": (0.5,), "U|JOINT|i8o64|l0.04": (0.55,),
            "U|DIRECT-TASK|i8o64": (1.1,), "U|FINE-TASK|i8o64": (1.0,), "U|LOCAL|i8o64|l0.1": (0.8,)}


def test_inference_end_to_end_and_finite(tmp_study):
    sex = _write(tmp_study, LABELS, STRENGTH)
    st = {"P*": {"status": "NOMINEE", "config": "U|SEQ-21|i8o64|l0.04", "winning_family": "SEQ-21"},
          "J*": {"status": "NOMINEE", "config": "U|JOINT|i8o64|l0.04"},
          "T*": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"},
          "C_rate": {"status": "NOMINEE", "config": "U|FINE-TASK|i8o64"},
          "C_global": {"status": "NOMINEE", "config": "U|FINE-TASK|i8o64"},
          "Q": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}}
    out = INF.main(["--evaluation-lock", str(_lock(tmp_study, st, LABELS))], check_prior=False)
    assert len(out["primary"]) == 37 and all(e["z"] == FAM.Z_PRIMARY for e in out["primary"])
    assert out["B"] == 60 and out["seed"] == 20261008
    z = np.load(R.UNITS / "outer__s0__U_SEQ-21_i8o64_l0.04" / "preds.npz")
    a = np.mean([roc_auc_score(sex, z["P_auc_pair"][s][:, 1]) for s in range(3)])
    assert out["levels"]["R#0#U|SEQ-21|i8o64|l0.04#primary#pair"]["point"] == pytest.approx(a, abs=1e-12)
    for e in out["primary"]:
        assert e["outcome"] in FAM.CLAUSE_OUTCOMES
    P23 = next(e for e in out["primary"] if e["id"] == "P23")             # C coalition: T* pair - P* pair
    assert P23["point"] > 0.1 and P23["outcome"] == "PASS"
    # identical task outputs: utility differences are exactly 0 -> logloss/brier PASS, accuracy identity PASS
    assert all(e["outcome"] == "PASS" for e in out["primary"] if e["kind"] in ("acc", "logloss", "brier"))
    assert out["q_status"]["status"] == "PASS"
    # C_rate == C_global -> B's coalition/local slots alias A's
    assert out["role_aliases"] == {"C_rate==C_global": "U|FINE-TASK|i8o64", "T*==Q": "U|DIRECT-TASK|i8o64"}
    assert next(e for e in out["primary"] if e["id"] == "P12")["alias_of_by_role"] == "P01"
    assert out["claim_status"]["C"]["status"] in ("PASS", "NOT_ESTABLISHED")
    if out["claim_status"]["C"]["status"] == "PASS":
        assert out["label"].startswith("PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (SEQ-21)")
    txt = (tmp_study / "inference.json").read_text()
    json.loads(txt, parse_constant=lambda c: pytest.fail(c))
    assert (tmp_study / "PRIMARY_ENDPOINTS.csv").exists() and (tmp_study / "ALL_LEVELS.csv").exists()


def test_fallback_is_descriptive_and_failures_are_scoped(tmp_study):
    _write(tmp_study, LABELS, STRENGTH)
    st = {"P*": {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": "U|LOCAL|i8o64|l0.1",
                 "reason": "HEADROOM_SELECTION_FAILURE"},
          "J*": {"status": "NOMINEE", "config": "U|JOINT|i8o64|l0.04"},
          "T*": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"},
          "C_rate": {"status": "NOMINEE", "config": "U|SEQ-21|i8o64|l0.04"},
          "C_global": {"status": "NOMINEE", "config": "U|FINE-TASK|i8o64"},
          "Q": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}}
    lp = _lock(tmp_study, st, LABELS)
    fp = tmp_study / "fail.json"
    fp.write_text(json.dumps({"global": [], "per_claim": {"A": ["verification FAIL on P01"]}}))
    out = INF.main(["--evaluation-lock", str(lp), "--failures", str(fp)], check_prior=False)
    assert all(e["decision"] == "DESCRIPTIVE_ONLY" for e in out["primary"] if e["claim"] == "C")
    c = out["claim_status"]["C"]
    assert (c["status"], c["root_cause"]) == ("NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", "HEADROOM_SELECTION_FAILURE")
    assert out["claim_status"]["A"]["status"] == "INCOMPLETE_OR_INVALID"
    assert out["claim_status"]["A"]["root_cause"] == "FAILED_REQUIRED_CONTROL"
    assert out["label"] == "INCOMPLETE_OR_INVALID"                     # no favourable claim beside an incomplete one
    fp.write_text(json.dumps({"global": ["teacher parity"], "per_claim": {}}))
    out = INF.main(["--evaluation-lock", str(lp), "--failures", str(fp)], check_prior=False)
    assert out["technical_valid"] is False and out["label"] == "INCOMPLETE_OR_INVALID"


def test_scored_labels_follow_section_12():
    S = {"statuses": {"P*": {"status": "NOMINEE", "config": "U|SEQ-21|i8o64|l0.04"},
                      "J*": {"status": "NO_ELIGIBLE_NOMINEE", "descriptive_config": "U|JOINT|i8o64|l0.025"},
                      "T*": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"},
                      "C_rate": {"status": "NOMINEE", "config": "U|SEQ-21|i8o64|l0.1"},
                      "C_global": {"status": "NOMINEE", "config": "REF|F"},
                      "Q": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}},
         "diagnostics": {"ordinary_privacy_winner_no_headroom": {"config": "U|SEQ-21|i8o64|l0.1"},
                         "family_headroom_winners": {"LOCAL": {"descriptive_config": "U|LOCAL|i8o64|l0.01"},
                                                     "SEQ-12": {"config": "U|SEQ-12|i8o64|l0.04"},
                                                     "SEQ-21": {"config": "U|SEQ-21|i8o64|l0.04"},
                                                     "JOINT": {"config": "U|JOINT|i8o64|l0.04"}}}}
    L = EL.scored_labels(S)
    assert len(L) == len(set(L))
    need = {"U|SEQ-21|i8o64|l0.04", "U|JOINT|i8o64|l0.025", "U|DIRECT-TASK|i8o64", "U|SEQ-21|i8o64|l0.1", "REF|F",
            "SRC|U", "U|CLASS|i1o1", "U|JOINT|i8o64|l0.1", "U|SEQ-12|i8o64|l0.1", "U|LOCAL|i8o64|l0.01",
            "U|SEQ-12|i8o64|l0.04", "U|JOINT|i8o64|l0.04", "SRC|RAW-J_b0.3", "REF|F0", "REF|E"}
    assert set(L) == need


def test_nonfinite_attacker_score_is_invalid_never_ranked(tmp_study):
    from jcv.finalize import save_unit
    _write(tmp_study, LABELS, STRENGTH)
    d = R.UNITS / "outer__s1__U_JOINT_i8o64_l0.04"
    z = dict(np.load(d / "preds.npz"))
    z["P_auc_pair"][1, 0, 1] = np.nan
    import shutil
    shutil.rmtree(d)
    save_unit(d, {"preds.npz": lambda q: np.savez(q, **z)}, {})
    st = {"P*": {"status": "NOMINEE", "config": "U|SEQ-21|i8o64|l0.04", "winning_family": "SEQ-21"},
          "J*": {"status": "NOMINEE", "config": "U|JOINT|i8o64|l0.04"},
          "T*": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"},
          "C_rate": {"status": "NOMINEE", "config": "U|FINE-TASK|i8o64"},
          "C_global": {"status": "NOMINEE", "config": "SRC|U"},
          "Q": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}}
    out = INF.main(["--evaluation-lock", str(_lock(tmp_study, st, LABELS))], check_prior=False)
    o = {e["id"]: e["outcome"] for e in out["primary"]}
    assert o["P01"] == "INVALID" and o["P12"] == "INVALID" and o["P23"] != "INVALID"
    assert out["claim_status"]["A"]["status"] == "INCOMPLETE_OR_INVALID"
    assert out["finiteness_receipt"]["all_finite"] is False
    assert out["finiteness_receipt"]["nonfinite_counts"]["s1|U|JOINT|i8o64|l0.04"]["P_auc_pair"] == 1


def test_outer_report_tables_and_figures(tmp_study):
    from cbp import report as RP
    _write(tmp_study, LABELS, STRENGTH)
    st = {"P*": {"status": "NOMINEE", "config": "U|SEQ-21|i8o64|l0.04", "winning_family": "SEQ-21"},
          "J*": {"status": "NOMINEE", "config": "U|JOINT|i8o64|l0.04"},
          "T*": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"},
          "C_rate": {"status": "NOMINEE", "config": "U|FINE-TASK|i8o64"},
          "C_global": {"status": "NOMINEE", "config": "SRC|U"},
          "Q": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}}
    lp = _lock(tmp_study, st, LABELS)
    INF.main(["--evaluation-lock", str(lp)], check_prior=False)
    L = json.loads(lp.read_text())
    L["scored_labels"] = LABELS
    (tmp_study / "EVALUATION_LOCK.json").write_text(json.dumps(L))
    rows = {c: {"family": R.parse_id(c).get("family", "SRC"), "ordinary": True, "headroom": True} for c in LABELS}
    (tmp_study / "selection.json").write_text(json.dumps({"rows": rows}))
    assert RP.outer_tables() == len(LABELS)
    for f in ("fig2_pair_vs_occ_ll", "fig3_individual_vs_pair", "fig4_states_vs_recovery"):
        assert (tmp_study / "figures" / f"{f}.png").exists()
    import csv
    r = list(csv.DictReader(open(tmp_study / "ASSESSMENT_COMPARISON.csv")))
    assert {x["release"] for x in r} == set(LABELS) and "auc_pair" in r[0] and "ll_excess_occupation" in r[1]
