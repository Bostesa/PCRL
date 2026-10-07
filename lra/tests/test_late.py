"""[lra port of lcr/tests/test_late.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
lra.infer and lra.eval_lock helpers end to end on synthetic predictions (temporary directories; no real data)."""
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from lra import eval_lock as EL
from lra import family as FAM
from lra import infer as INF
from lra import run as R


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


def _lock(tmp_study, st, labels, tv=True, abl=None):
    res = {x: s.get("config") or s.get("descriptive_config") for x, s in st.items()}
    al = EL.role_aliases(res)
    lock = {"statuses": st, "resolved": res, "role_aliases": al, "alias_of_by_role": EL.alias_of_by_role(al),
            "decoder_ablation_pair": abl,
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
    st = {"P*": {"status": "NOMINEE", "config": "U|SEQ-21|i8o64|l0.04", "winning": "SEQ-21; existing"},
          "N*": {"status": "NOMINEE", "config": "U|JOINT|i8o64|l0.04"},
          "J*": {"status": "NOMINEE", "config": "U|JOINT|i8o64|l0.04"},
          "T*": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"},
          "C*": {"status": "NOMINEE", "config": "U|FINE-TASK|i8o64"},
          "C_pair*": {"status": "NOMINEE", "config": "U|FINE-TASK|i8o64"},
          "Q": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}}
    out = INF.main(["--evaluation-lock", str(_lock(tmp_study, st, LABELS))], check_prior=False)
    assert len(out["primary"]) == 37 and all(e["z"] == FAM.Z_PRIMARY for e in out["primary"])
    assert out["B"] == 60 and out["seed"] == 20261009
    z = np.load(R.UNITS / "outer__s0__U_SEQ-21_i8o64_l0.04" / "preds.npz")
    a = np.mean([roc_auc_score(sex, z["P_auc_pair"][s][:, 1]) for s in range(3)])
    assert out["levels"]["R#0#U|SEQ-21|i8o64|l0.04#primary#pair"]["point"] == pytest.approx(a, abs=1e-12)
    for e in out["primary"]:
        assert e["outcome"] in FAM.CLAUSE_OUTCOMES
    P01 = next(e for e in out["primary"] if e["id"] == "P01")             # A coalition: T* pair - P* pair
    assert P01["point"] > 0.1 and P01["outcome"] == "PASS"
    # identical task outputs: utility differences are exactly 0 -> logloss/brier PASS, accuracy identity PASS
    assert all(e["outcome"] == "PASS" for e in out["primary"] if e["kind"] in ("acc", "logloss", "brier"))
    assert out["q_status"]["status"] == "PASS"
    # N* == J* and C* == C_pair* -> claim C's slots alias claim B's
    assert out["role_aliases"]["N*==J*"] == "U|JOINT|i8o64|l0.04" and out["role_aliases"]["C*==C_pair*"]
    assert next(e for e in out["primary"] if e["id"] == "P23")["alias_of_by_role"] == "P12"
    assert out["claim_status"]["A"]["status"] in ("PASS", "NOT_ESTABLISHED")
    if out["claim_status"]["A"]["status"] == "PASS":
        assert out["label"].startswith("PRIVACY_RELEASE_DEVELOPMENT_CRITERION_MET (SEQ-21; existing)")
    txt = (tmp_study / "inference.json").read_text()
    json.loads(txt, parse_constant=lambda c: pytest.fail(c))
    assert (tmp_study / "PRIMARY_ENDPOINTS.csv").exists() and (tmp_study / "ALL_LEVELS.csv").exists()


ROLES_OK = {"P*": {"status": "NOMINEE", "config": "U|SEQ-21|i8o64|l0.04", "winning": "SEQ-21; existing"},
            "N*": {"status": "NOMINEE", "config": "U|JOINT|i8o64|l0.04"},
            "J*": {"status": "NOMINEE", "config": "U|LOCAL|i8o64|l0.1"},
            "T*": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"},
            "C*": {"status": "NOMINEE", "config": "U|FINE-TASK|i8o64"},
            "C_pair*": {"status": "NOMINEE", "config": "SRC|U"},
            "Q": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}}


def test_fallback_is_descriptive_and_failures_are_scoped(tmp_study):
    _write(tmp_study, LABELS, STRENGTH)
    st = json.loads(json.dumps(ROLES_OK))
    st["P*"] = {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": "U|LOCAL|i8o64|l0.1",
                "reason": "LOCAL_GUARD_FAILURE"}
    lp = _lock(tmp_study, st, LABELS)
    fp = tmp_study / "fail.json"
    fp.write_text(json.dumps({"global": [], "per_claim": {"B": ["verification FAIL on P12"]}}))
    out = INF.main(["--evaluation-lock", str(lp), "--failures", str(fp)], check_prior=False)
    assert all(e["decision"] == "DESCRIPTIVE_ONLY" for e in out["primary"] if e["claim"] == "A")
    a = out["claim_status"]["A"]
    assert (a["status"], a["root_cause"]) == ("NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", "LOCAL_GUARD_FAILURE")
    assert out["claim_status"]["B"]["status"] == "INCOMPLETE_OR_INVALID"
    assert out["claim_status"]["B"]["root_cause"] == "FAILED_REQUIRED_CONTROL"
    # prompt sec. 12 literal: no method claim passes and Q passes -> feasibility label; the incomplete B stays displayed
    assert out["label"] == "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION"
    assert out["displayed_statuses"]["B"] == "INCOMPLETE_OR_INVALID"
    fp.write_text(json.dumps({"global": ["teacher parity"], "per_claim": {}}))
    out = INF.main(["--evaluation-lock", str(lp), "--failures", str(fp)], check_prior=False)
    assert out["technical_valid"] is False and out["label"] == "INCOMPLETE_OR_INVALID"


def test_scored_labels_follow_section_13():
    S = {"statuses": {"P*": {"status": "NOMINEE", "config": "U|W-SEQ-21|i8o64|l0.06|D1"},
                      "N*": {"status": "NOMINEE", "config": "U|K-SEQ-12|i8o64|D1"},
                      "J*": {"status": "NO_ELIGIBLE_NOMINEE", "descriptive_config": "U|K-JOINT-PAIR|i8o64|D1"},
                      "T*": {"status": "NOMINEE", "config": "U|C-TASK|i8o64|D1"},
                      "C*": {"status": "NOMINEE", "config": "U|W-SEQ-21|i8o64|l0.06|D1"},
                      "C_pair*": {"status": "NOMINEE", "config": "U|W-SEQ-21|i8o64|l0.06|D1"},
                      "Q": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}},
         "diagnostics": {"best_d1_fixed_privacy": {"config": "U|SEQ-12|i8o64|l0.06|D1", "paired_d0": "U|SEQ-12|i8o64|l0.06"},
                         "best_weighted_privacy": {"config": "U|W-SEQ-21|i8o64|l0.06|D1"}}}
    L = EL.scored_labels(S)
    assert len(L) == len(set(L))
    need = {"U|W-SEQ-21|i8o64|l0.06|D1", "U|K-SEQ-12|i8o64|D1", "U|K-JOINT-PAIR|i8o64|D1", "U|C-TASK|i8o64|D1",
            "U|DIRECT-TASK|i8o64", "SRC|U", "U|CLASS|i1o1", "U|SEQ-12|i8o64|l0.06|D1", "U|SEQ-12|i8o64|l0.06",
            "U|K-LOCAL|i8o64|D1", "U|K-SEQ-21|i8o64|D1", "U|K-JOINT-SINGLE|i8o64|D1", "SRC|RAW-J_b0.3", "REF|F",
            "REF|F0", "REF|E"}
    assert set(L) == need


def test_nonfinite_attacker_score_is_invalid_never_ranked(tmp_study):
    from jcv.finalize import save_unit
    _write(tmp_study, LABELS, STRENGTH)
    d = R.UNITS / "outer__s1__U_SEQ-21_i8o64_l0.04"
    z = dict(np.load(d / "preds.npz"))
    z["P_auc_pair"][1, 0, 1] = np.nan
    import shutil
    shutil.rmtree(d)
    save_unit(d, {"preds.npz": lambda q: np.savez(q, **z)}, {})
    out = INF.main(["--evaluation-lock", str(_lock(tmp_study, ROLES_OK, LABELS))], check_prior=False)
    o = {e["id"]: e["outcome"] for e in out["primary"]}
    assert o["P01"] == "INVALID" and o["P12"] != "INVALID"
    assert out["claim_status"]["A"]["status"] == "INCOMPLETE_OR_INVALID"
    assert out["finiteness_receipt"]["all_finite"] is False
    assert out["finiteness_receipt"]["nonfinite_counts"]["s1|U|SEQ-21|i8o64|l0.04"]["P_auc_pair"] == 1


def test_decoder_ablation_contrast_is_computed(tmp_study):
    _write(tmp_study, LABELS, STRENGTH)
    out = INF.main(["--evaluation-lock", str(_lock(tmp_study, ROLES_OK, LABELS,
                                                   abl=["U|SEQ-21|i8o64|l0.04", "U|JOINT|i8o64|l0.04"]))],
                   check_prior=False)
    da = out["decoder_ablation"]
    assert set(da["contrasts"]) == {"logloss_income", "logloss_occupation", "brier_income", "brier_occupation"}
    assert da["verdict"] in ("did", "did not clearly")
    # identical task outputs in the fixture -> zero differences -> not 'did'
    assert da["verdict"] == "did not clearly" and all(abs(c["point"]) < 1e-12 for c in da["contrasts"].values())
