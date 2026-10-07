"""[lra port of lcr/tests/test_late.py at 091afc2: lcr->lra renames; later edits: REVIEW_FINDINGS_DISPOSITION.json
(F04/F07/F10/F13/F14)]
lra.infer, lra.eval_lock and the lra.assess refusal path end to end on synthetic predictions (temporary directories;
no real data, no labels)."""
import inspect
import json
import subprocess

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


READY = {"verdict": "ENGINEERING_READY", "ready": True, "sha256": "0" * 64}


def _lock(tmp_study, st, labels, tv=True, abl=None, pairs=None, gate=READY):
    res = {x: s.get("config") or s.get("descriptive_config") for x, s in st.items()}
    al = EL.role_aliases(res, st)
    lock = {"statuses": st, "resolved": res, "role_aliases": al, "alias_of_by_role": EL.alias_of_by_role(al),
            "decoder_ablation_pair": abl,
            "seeds": {str(k): {"score": {lab: {} for lab in labels}} for k in (0, 1, 2)},
            "technical_validity": {"ok": tv, "engineering_gate": gate}}
    if pairs is not None:
        lock["same_map_decoder_pairs"] = pairs
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
    assert out["B"] == 60 and out["seed"] == 20261010
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
    assert FAM.label_headline(out["label"]) == "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION"
    assert out["displayed_statuses"]["B"] == "INCOMPLETE_OR_INVALID" and "B=INCOMPLETE_OR_INVALID" in out["label"]
    fp.write_text(json.dumps({"global": ["teacher parity"], "per_claim": {}}))
    out = INF.main(["--evaluation-lock", str(lp), "--failures", str(fp)], check_prior=False)
    assert out["technical_valid"] is False and FAM.label_headline(out["label"]) == "INCOMPLETE_OR_INVALID"
    assert "Q=PASS" in out["label"]                                     # Q shown, never hiding the incomplete work


def test_scored_labels_follow_section_13():
    S = {"statuses": {"P*": {"status": "NOMINEE", "config": "U|W-SEQ-21|i8o64|l0.06|D1"},
                      "N*": {"status": "NOMINEE", "config": "U|K-SEQ-12|i8o64|D1"},
                      "J*": {"status": "NO_ELIGIBLE_NOMINEE", "descriptive_config": "U|K-JOINT-PAIR|i8o64|D1"},
                      "T*": {"status": "NOMINEE", "config": "U|C-TASK|i8o64|D1"},
                      "C*": {"status": "NOMINEE", "config": "U|W-SEQ-21|i8o64|l0.06|D1"},
                      "C_pair*": {"status": "NOMINEE", "config": "U|W-SEQ-21|i8o64|l0.06|D1"},
                      "Q": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}},
         "diagnostics": {"best_d1_fixed_privacy": {"config": "U|SEQ-12|i8o64|l0.06|D1", "paired_d0": "U|SEQ-12|i8o64|l0.06"},
                         "best_weighted_privacy": {"config": "U|W-SEQ-21|i8o64|l0.06|D1"},
                         "same_map_decoder_pairs": [
                             {"name": "best_d1_fixed_privacy", "d1": "U|SEQ-12|i8o64|l0.06|D1", "d0": "U|SEQ-12|i8o64|l0.06"},
                             {"name": "P*", "d1": "U|W-SEQ-21|i8o64|l0.06|D1", "d0": "U|W-SEQ-21|i8o64|l0.06|D1|D0SAME"},
                             {"name": "C-TASK", "d1": "U|C-TASK|i8o64|D1", "d0": "U|C-TASK|i8o64|D1|D0SAME"}]}}
    L = EL.scored_labels(S)
    assert len(L) == len(set(L))
    need = {"U|W-SEQ-21|i8o64|l0.06|D1", "U|K-SEQ-12|i8o64|D1", "U|K-JOINT-PAIR|i8o64|D1", "U|C-TASK|i8o64|D1",
            "U|DIRECT-TASK|i8o64", "SRC|U", "U|CLASS|i1o1", "U|SEQ-12|i8o64|l0.06|D1", "U|SEQ-12|i8o64|l0.06",
            "U|K-LOCAL|i8o64|D1", "U|K-SEQ-21|i8o64|D1", "U|K-JOINT-SINGLE|i8o64|D1", "SRC|RAW-J_b0.3", "REF|F",
            "REF|F0", "REF|E", "U|W-SEQ-21|i8o64|l0.06|D1|D0SAME", "U|C-TASK|i8o64|D1|D0SAME", "U|CLASS|i1o1|D1"}
    assert set(L) == need
    assert EL.decoder_pair_labels(S) == {"U|SEQ-12|i8o64|l0.06|D1", "U|SEQ-12|i8o64|l0.06", "U|W-SEQ-21|i8o64|l0.06|D1",
                                         "U|W-SEQ-21|i8o64|l0.06|D1|D0SAME", "U|C-TASK|i8o64|D1",
                                         "U|C-TASK|i8o64|D1|D0SAME"}
    for x in ("U|W-SEQ-21|i8o64|l0.06|D1|D0SAME", "U|C-TASK|i8o64|D1|D0SAME"):
        p = R.parse_id(x)
        assert p["kind"] == "policy" and p["diagnostic_only"] and R.unit_for(0, x).startswith("d0s__s0__")


def test_f04_missing_guard_fallback_stays_on_the_scored_list():
    S = {"statuses": {"P*": {"status": "INVALID_NOMINEE", "reason": "MISSING_GUARD_COMPARATOR", "config": None,
                             "descriptive_config": "U|W-JOINT|i8o64|l0.1|D1"},
                      "T*": {"status": "NO_ELIGIBLE_COMPARATOR", "descriptive_config": "U|CLASS|i1o1"}},
         "diagnostics": {"best_d1_fixed_privacy": {"descriptive_config": "U|JOINT|i8o64|l0.1|D1",
                                                   "paired_d0": "U|JOINT|i8o64|l0.1"},
                         "best_weighted_privacy": {"descriptive_config": "U|W-JOINT|i8o64|l0.1|D1"}}}
    L = EL.scored_labels(S)
    assert {"U|W-JOINT|i8o64|l0.1|D1", "U|CLASS|i1o1", "U|JOINT|i8o64|l0.1|D1", "U|JOINT|i8o64|l0.1"} <= set(L)
    assert EL._selection_technical(S) == []           # a missing guard comparator is not a technical failure


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


def test_same_map_contrasts_use_the_supplementary_z_on_the_same_draws(tmp_study):
    _write(tmp_study, LABELS, STRENGTH)
    pairs = [{"name": "best_d1_fixed_privacy", "d1": "U|SEQ-21|i8o64|l0.04", "d0": "U|JOINT|i8o64|l0.04",
              "registered_sentence": True},
             {"name": "P*", "d1": "U|SEQ-21|i8o64|l0.04", "d0": "U|FINE-TASK|i8o64", "registered_sentence": False},
             {"name": "C-TASK", "d1": "U|DIRECT-TASK|i8o64", "d0": "U|LOCAL|i8o64|l0.1", "registered_sentence": False}]
    out = INF.main(["--evaluation-lock", str(_lock(tmp_study, ROLES_OK, LABELS, pairs=pairs))], check_prior=False)
    sm = out["same_map_decoder_contrasts"]
    assert sm["z"] == FAM.Z_SUPPLEMENTARY and abs(sm["z"] - 1.959963984540054) < 1e-12 and sm["seed"] == 20261010
    assert [p["name"] for p in sm["pairs"]] == ["best_d1_fixed_privacy", "P*", "C-TASK"]
    for p in sm["pairs"]:
        assert set(p["contrasts"]) == {"logloss_income", "logloss_occupation", "brier_income", "brier_occupation"}
        for c in p["contrasts"].values():
            assert c["upper"] - c["point"] == pytest.approx(FAM.Z_SUPPLEMENTARY * c["se"], abs=1e-12)
        assert p["verdict"] in ("did", "did not clearly")
    assert out["decoder_ablation"]["pair"] == ["U|SEQ-21|i8o64|l0.04", "U|JOINT|i8o64|l0.04"]
    assert all(e["z"] == FAM.Z_PRIMARY for e in out["primary"])               # primary slots keep the family z


# ------------------------------------------------------------------ F10: inference requires ENGINEERING_READY
@pytest.mark.parametrize("gate", [None, {"verdict": "ENGINEERING_BLOCKED", "ready": False},
                                  {"verdict": "GATE_MET", "ready": True}, {"verdict": "ENGINEERING_READY", "ready": False}])
def test_f10_infer_refuses_without_engineering_ready(tmp_study, gate):
    _write(tmp_study, LABELS, STRENGTH)
    lp = _lock(tmp_study, ROLES_OK, LABELS, gate=gate)
    with pytest.raises(SystemExit, match="REFUSED: the evaluation lock does not bind ENGINEERING_READY"):
        INF.main(["--evaluation-lock", str(lp)], check_prior=False)
    assert not (tmp_study / "inference.json").exists()
    src = inspect.getsource(INF)
    assert "gate_met" not in src and "engineering_gate=gate" in src


def test_f10_infer_label_carries_the_ready_gate_and_untrained_disclosure(tmp_study):
    _write(tmp_study, LABELS, STRENGTH)
    st = json.loads(json.dumps(ROLES_OK))
    st["N*"]["identical_to_untrained"] = ["U|FINE-TASK|i8o64"]
    out = INF.main(["--evaluation-lock", str(_lock(tmp_study, st, LABELS))], check_prior=False)
    assert out["engineering_gate"] == "ENGINEERING_READY"
    assert out["label"].endswith(FAM.status_suffix(out["displayed_statuses"]))
    if out["claim_status"]["B"]["status"] == "PASS":
        assert "N* release identical to privacy-untrained U|FINE-TASK|i8o64" in out["label"]


# ------------------------------------------------------------------ F10 / F13: the evaluation-lock builder refuses
def _valid_env(tmp_path, monkeypatch, gate_ok=True):
    from lra import lock as LK
    pkg, priv = tmp_path / "pkg", tmp_path / "priv"
    (priv / "admitted").mkdir(parents=True)
    pkg.mkdir()
    monkeypatch.setattr(R, "PKG", pkg)
    monkeypatch.setattr(R, "PRIV", priv)
    monkeypatch.setattr(R, "RUN", priv / "run")
    monkeypatch.setattr(LK, "PKG", pkg)
    monkeypatch.setattr(LK, "verify_lock", lambda p, stage=None, require_pushed=True: {"ok": True, "mismatches": []})
    monkeypatch.setattr(R, "engineering_ready", lambda: (gate_ok, "ENGINEERING_READY" if gate_ok else
                                                         "verdict is 'ENGINEERING_BLOCKED', not ENGINEERING_READY"))
    (pkg / "AUDIT_PRELOCK_CHECKS.json").write_text(json.dumps({"verdict": {"all_ok": True}}))
    (pkg / R.GATE_RESULT).write_text(json.dumps({"verdict": "ENGINEERING_READY" if gate_ok else "ENGINEERING_BLOCKED"}))
    for n in ("CORRECTNESS_LOCK", "SCIENCE_LOCK"):
        (pkg / f"{n}.json").write_text(json.dumps({"name": n}))
    (priv / "admitted" / "ADMISSION_RECEIPT.json").write_text(json.dumps({"verdict": "ADMITTED"}))
    S = {"inner_validation": {"ok": True}, "technical_failures": {},
         "statuses": {x: {"status": "NOMINEE", "config": "U|CLASS|i1o1"} for x in EL.ROLES}}
    return pkg, priv, S


def test_f10_f13_technical_validity_ok_only_with_ready_gate_controls_locks_admission_and_clean_selection(
        tmp_path, monkeypatch):
    pkg, priv, S = _valid_env(tmp_path, monkeypatch)
    tv = EL.technical_validity(S)
    assert tv["ok"] is True and tv["engineering_gate"]["verdict"] == "ENGINEERING_READY"
    assert tv["engineering_gate"]["sha256"] and set(tv["locks_sha256"]) == {"CORRECTNESS_LOCK", "SCIENCE_LOCK"}
    # the old source gate is irrelevant: a GATE_MET FIXTURE_GATE.json neither helps nor is required
    (pkg / "FIXTURE_GATE.json").write_text(json.dumps({"verdict": "GATE_NOT_MET"}))
    assert EL.technical_validity(S)["ok"] is True
    monkeypatch.setattr(R, "engineering_ready", lambda: (False, "ENGINEERING_GATE_RESULT.json missing"))
    (pkg / "FIXTURE_GATE.json").write_text(json.dumps({"verdict": "GATE_MET"}))
    tv = EL.technical_validity(S)
    assert tv["ok"] is False and {"engineering_gate": "ENGINEERING_GATE_RESULT.json missing"} in tv["failures"]
    _valid_env(tmp_path / "b", monkeypatch)
    bad = {**S, "technical_failures": {"U|K-LOCAL|i8o64|D1": [{"code": "FIT_RECORD_TECHNICAL_FAILURE"}]}}
    assert EL.technical_validity(bad)["ok"] is False
    bad = {**S, "statuses": {**S["statuses"], "C*": {"status": "INVALID_COMPARATOR", "reason": "NON_ESTIMABLE_INNER_METRIC"}}}
    assert EL.technical_validity(bad)["ok"] is False
    sci = {**S, "statuses": {**S["statuses"], "P*": {"status": "INVALID_NOMINEE", "reason": "MISSING_GUARD_COMPARATOR",
                                                     "descriptive_config": "U|W-LOCAL|i8o64|l0.1|D1"}}}
    assert EL.technical_validity(sci)["ok"] is True            # scientific ineligibility is locked with its fallback
    assert EL.technical_validity({**S, "inner_validation": {"ok": False}})["ok"] is False


@pytest.mark.parametrize("defect", ["controls_failed", "controls_missing", "admission", "science_lock", "correctness_lock",
                                    "gate"])
def test_f13_technical_validity_each_failure_alone(tmp_path, monkeypatch, defect):
    pkg, priv, S = _valid_env(tmp_path, monkeypatch, gate_ok=defect != "gate")
    if defect == "controls_failed":
        (pkg / "AUDIT_PRELOCK_CHECKS.json").write_text(json.dumps({"verdict": {"all_ok": False, "failures": ["XOR"]}}))
    elif defect == "controls_missing":
        (pkg / "AUDIT_PRELOCK_CHECKS.json").unlink()
    elif defect == "admission":
        (priv / "admitted" / "ADMISSION_RECEIPT.json").write_text(json.dumps({"verdict": "PARITY_FAILED"}))
    elif defect == "science_lock":
        (pkg / "SCIENCE_LOCK.json").unlink()
    elif defect == "correctness_lock":
        (pkg / "CORRECTNESS_LOCK.json").unlink()
    tv = EL.technical_validity(S)
    assert tv["ok"] is False and len(tv["failures"]) == 1, tv["failures"]


def test_f13_eval_lock_main_refuses_technical_failure_and_has_no_escape_flag(tmp_path, monkeypatch):
    out = tmp_path / "EVALUATION_LOCK.json"
    monkeypatch.setattr(EL, "build", lambda: {"technical_validity": {"ok": False, "failures": ["controls"]}})
    with pytest.raises(SystemExit, match="REFUSED: technical validity failed"):
        EL.main([str(out)])
    assert not out.exists()
    for argv in ([str(out), "--accept-technical-failure", "reviewed"], ["--accept-technical-failure", "x", str(out)],
                 [str(out), "--force"], []):
        with pytest.raises(SystemExit, match="usage"):
            EL.main(argv)
    assert not out.exists()
    monkeypatch.setattr(EL, "build", lambda: {"technical_validity": {"ok": True, "failures": []}, "statuses": {}})
    EL.main([str(out)])
    assert json.loads(out.read_text())["technical_validity"]["ok"] is True
    assert "accept" not in inspect.getsource(EL).replace("no --accept", "").lower().replace("accept-technical", "")


# ------------------------------------------------------------------ F10 / F13: the assessment opening refuses
def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True)


def _valid_lock(pkg):
    from lra import assess as AS
    return {"technical_validity": {"ok": True, "engineering_gate": {
        "verdict": "ENGINEERING_READY", "ready": True, "sha256": AS._sha(pkg / R.GATE_RESULT)},
        "controls_verdict_sha256": AS._sha(pkg / "AUDIT_PRELOCK_CHECKS.json")},
        "locks_sha256": {n: AS._sha(pkg / n) for n in ("CORRECTNESS_LOCK.json", "SCIENCE_LOCK.json")}}


@pytest.mark.parametrize("defect", ["tv_false", "lock_gate_blocked", "lock_gate_missing", "gate_not_ready_now",
                                    "gate_result_changed", "science_lock_changed", "correctness_lock_missing_in_lock",
                                    "controls_changed", "controls_failed", "admission_missing"])
def test_f10_f13_assess_verify_validity_refuses(tmp_path, monkeypatch, defect):
    from lra import assess as AS
    pkg, priv, _ = _valid_env(tmp_path, monkeypatch)
    L = _valid_lock(pkg)
    assert AS.verify_validity(L)["engineering_gate"] == "ENGINEERING_READY"
    if defect == "tv_false":
        L["technical_validity"]["ok"] = False
    elif defect == "lock_gate_blocked":
        L["technical_validity"]["engineering_gate"]["verdict"] = "ENGINEERING_BLOCKED"
    elif defect == "lock_gate_missing":
        del L["technical_validity"]["engineering_gate"]
    elif defect == "gate_not_ready_now":
        monkeypatch.setattr(R, "engineering_ready", lambda: (False, "not on origin"))
    elif defect == "gate_result_changed":
        (pkg / R.GATE_RESULT).write_text(json.dumps({"verdict": "ENGINEERING_READY", "rerun": 2}))
    elif defect == "science_lock_changed":
        (pkg / "SCIENCE_LOCK.json").write_text(json.dumps({"name": "SCIENCE_LOCK", "amended": True}))
    elif defect == "correctness_lock_missing_in_lock":
        del L["locks_sha256"]["CORRECTNESS_LOCK.json"]
    elif defect == "controls_changed":
        (pkg / "AUDIT_PRELOCK_CHECKS.json").write_text(json.dumps({"verdict": {"all_ok": True}, "rerun": 1}))
    elif defect == "controls_failed":
        (pkg / "AUDIT_PRELOCK_CHECKS.json").write_text(json.dumps({"verdict": {"all_ok": False}}))
        L["technical_validity"]["controls_verdict_sha256"] = AS._sha(pkg / "AUDIT_PRELOCK_CHECKS.json")
    elif defect == "admission_missing":
        (priv / "admitted" / "ADMISSION_RECEIPT.json").unlink()
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.verify_validity(L)


def test_f13_assess_open_calls_the_validity_check_with_no_override(tmp_path, monkeypatch):
    from lra import assess as AS
    pkg, priv, _ = _valid_env(tmp_path, monkeypatch)
    origin, repo = tmp_path / "origin.git", tmp_path / "wt"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    repo.mkdir()
    _git(repo, "init", "-q", "-b", AS.STUDY_BRANCH)
    _git(repo, "config", "user.email", "t@example.org")
    _git(repo, "config", "user.name", "t")
    _git(repo, "remote", "add", "origin", str(origin))
    lp = repo / AS.LOCK_REL
    lp.parent.mkdir(parents=True)
    bad = {**_valid_lock(pkg), "seeds": {}, "locked_code_files": {}}
    bad["technical_validity"] = {**bad["technical_validity"], "ok": False, "failures": ["controls"]}
    lp.write_text(json.dumps(bad))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "lock")
    _git(repo, "push", "-q", "origin", AS.STUDY_BRANCH)
    AS.close_assessment()
    with pytest.raises(SystemExit, match="REFUSED: the evaluation lock records a technical failure"):
        AS.open_assessment(lp, repo, check_code=False, fetch=False)
    with pytest.raises(SystemExit, match="sealed"):
        AS.load_unsealed()                                            # nothing was opened
    params = set(inspect.signature(AS.open_assessment).parameters)
    assert params == {"lock_path", "repo", "branch", "rel_required", "check_code", "chain", "fetch"}
    assert "accept" not in " ".join(params) and "v[\"validity\"] = verify_validity(lock)" in \
        inspect.getsource(AS.open_assessment)
