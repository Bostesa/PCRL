"""Exhaustive truth table (LABEL_TRUTH_TABLE.json <-> cbp.family), fixed 37-slot family and clause classifier."""
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist

from cbp import family as FAM

PKG = Path(__file__).resolve().parents[2] / "results" / "pcrl_confidence_budgeted_privacy_v1"


def test_family_fixed_and_z_verified():
    assert FAM.PRIMARY_SIZE == 37 and len(FAM.PRIMARY) == 37
    assert FAM.Z_PRIMARY == NormalDist().inv_cdf(1 - 0.05 / 74)
    assert abs(FAM.Z_PRIMARY - 3.2048452050105634) < 1e-15
    assert FAM.B == 1999 and FAM.BOOT_SEED == 20261008
    assert [e["id"] for e in FAM.PRIMARY if e["claim"] == "Q"] == ["P34", "P35", "P36", "P37"]
    for c in "ABC":
        cl = [e for e in FAM.PRIMARY if e["claim"] == c]
        assert len(cl) == 11 and cl[0]["kind"] == "coalition" and cl[0]["target"] == 0.02 and cl[0]["side"] == "lower>"
        assert [e["target"] for e in cl if e["kind"] == "logloss"] == [0.01, 0.01]
        assert [e["target"] for e in cl if e["kind"] == "brier"] == [0.005, 0.005]
        assert {e["kind"] for e in cl} == {"coalition", "local", "acc", "logloss", "brier", "retain"}


def test_clause_outcomes():
    up, lo = "upper<", "lower>"
    assert FAM.clause_outcome(up, 0.01, 0.008, 0.004, 0.0121) == "NOT_ESTABLISHED_PRECISION"   # qpc P29 shape
    assert FAM.clause_outcome(up, 0.01, 0.005, 0.002, 0.009) == "PASS"
    assert FAM.clause_outcome(up, 0.01, 0.015, 0.011, 0.019) == "MEASURED_VIOLATION"
    assert FAM.clause_outcome(up, 0.01, 0.012, 0.008, 0.016) == "NOT_ESTABLISHED_POINT"
    assert FAM.clause_outcome(lo, 0.02, 0.0336, 0.0286, 0.0387) == "PASS"
    assert FAM.clause_outcome(lo, 0.02, 0.025, 0.015, 0.035) == "NOT_ESTABLISHED_PRECISION"
    assert FAM.clause_outcome(lo, 0.02, 0.003, 0.0001, 0.006) == "MEASURED_VIOLATION"
    assert FAM.clause_outcome(lo, 0.02, 0.015, 0.001, 0.029) == "NOT_ESTABLISHED_POINT"
    assert FAM.clause_outcome(lo, 0.0, 0.0, 0.0, 0.0) == "NOT_ESTABLISHED_POINT"        # zero-variance identity at target
    assert FAM.clause_outcome(lo, -0.01, 0.0, 0.0, 0.0) == "PASS"                       # accuracy identity (class preserving)
    for bad in (math.nan, math.inf, None):
        assert FAM.clause_outcome(up, 0.01, bad, 0.0, 0.0) == "INVALID"


def _outs(n_fail=0, kind="NOT_ESTABLISHED_PRECISION"):
    o = {f"c{i}": "PASS" for i in range(11)}
    for i in range(n_fail):
        o[f"c{i}"] = kind
    return o


def test_claim_status_precedence_and_every_single_failed_clause():
    E, N, T = "ELIGIBLE", "NO_ELIGIBLE", "TECHNICAL_FAILURE"
    assert FAM.claim_status(E, E, _outs()) == ("PASS", "COMPLETE_PASSING_CONJUNCTION", [])
    for i in range(11):                                         # every single failed clause blocks the pass
        o = _outs()
        o[f"c{i}"] = "NOT_ESTABLISHED_PRECISION"
        st, cause, failing = FAM.claim_status(E, E, o)
        assert st == "NOT_ESTABLISHED" and cause == "ASSESSMENT_PRECISION_FAILURE" and failing == [f"c{i}"]
    o = _outs(1, "MEASURED_VIOLATION")
    assert FAM.claim_status(E, E, o)[1] == "MEASURED_VIOLATION_SUPPORTED_BY_BOUND"
    o = _outs(1, "NOT_ESTABLISHED_POINT")
    assert FAM.claim_status(E, E, o)[1] == "CLAUSE_NOT_ESTABLISHED"
    o = _outs(1, "INVALID")
    assert FAM.claim_status(E, E, o)[:2] == ("INCOMPLETE_OR_INVALID", "NONFINITE_PRIMARY_QUANTITY")
    # absent nominee: completed negative with its selection reason; a fallback can never pass
    assert FAM.claim_status(N, E, _outs(), nominee_reason="HEADROOM_SELECTION_FAILURE") == \
        ("NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", "HEADROOM_SELECTION_FAILURE", [])
    # absent / invalid comparator is never a completed negative
    assert FAM.claim_status(E, N, _outs(3))[:2] == ("INCOMPLETE_OR_INVALID", "NO_ELIGIBLE_COMPARATOR")
    assert FAM.claim_status(E, T, _outs())[:2] == ("INCOMPLETE_OR_INVALID", "INVALID_OR_MISSING_COMPARATOR")
    assert FAM.claim_status(T, E, _outs(), nominee_reason="MISSING_GUARD_COMPARATOR")[:2] == \
        ("INCOMPLETE_OR_INVALID", "MISSING_GUARD_COMPARATOR")
    assert FAM.claim_status(E, E, _outs(), required_control_ok=False)[:2] == ("INCOMPLETE_OR_INVALID",
                                                                             "FAILED_REQUIRED_CONTROL")
    # exhaustive over role states: technical comparator failure dominates
    for nom, comp in itertools.product(FAM.ROLE_STATES, FAM.ROLE_STATES):
        st = FAM.claim_status(nom, comp, _outs())[0]
        if comp == T or nom == T or (comp == N and nom == E):
            assert st == "INCOMPLETE_OR_INVALID"
        elif nom == N:
            assert st == "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
        else:
            assert st == "PASS"


def test_overall_label_exhaustive_and_mixed():
    S = FAM.CLAIM_STATUSES
    for a, b, c in itertools.product(S, repeat=3):
        for q in ("PASS", "NOT_ESTABLISHED", "INCOMPLETE_OR_INVALID"):
            for tv in (True, False):
                lab, shown = FAM.overall_label({"A": a, "B": b, "C": c}, q, tv, "SEQ-21")
                assert shown == {"A": a, "B": b, "C": c, "Q": q}             # every status always displayed
                if not tv:
                    assert lab == "INCOMPLETE_OR_INVALID"
                    continue
                fav_c, fav_j = c == "PASS", a == "PASS" and b == "PASS"
                if fav_c or fav_j:
                    assert lab.startswith("PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (SEQ-21)") == fav_c
                    assert ("JOINT_DEVELOPMENT_CRITERION_MET" in lab) == fav_j
                elif "INCOMPLETE_OR_INVALID" in (a, b, c, q):
                    assert lab == "INCOMPLETE_OR_INVALID"
                elif q == "PASS":
                    assert lab == "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION"
                else:
                    assert lab == "EXPERIMENTAL_NO_ADVANTAGE"
    # a legitimately passing C next to an incomplete A: reported, both displayed
    lab, shown = FAM.overall_label({"A": "INCOMPLETE_OR_INVALID", "B": "NOT_ESTABLISHED", "C": "PASS"}, "PASS",
                                   True, "LOCAL")
    assert lab == "PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (LOCAL)" and shown["A"] == "INCOMPLETE_OR_INVALID"
    # joint needs both A and B
    assert FAM.overall_label({"A": "NOT_ESTABLISHED", "B": "PASS", "C": "NOT_ESTABLISHED"}, "NOT_ESTABLISHED")[0] == \
        "EXPERIMENTAL_NO_ADVANTAGE"


def test_q_status():
    Q = {"P34": "PASS", "P35": "PASS", "P36": "PASS", "P37": "PASS"}
    assert FAM.q_status(True, Q) == ("PASS", "COMPLETE_PASSING_CONJUNCTION", [])
    assert FAM.q_status("ELIGIBLE", {**Q, "P35": "NOT_ESTABLISHED_PRECISION"})[1:] == \
        ("ASSESSMENT_PRECISION_FAILURE", ["P35"])
    assert FAM.q_status(True, {**Q, "P35": "MEASURED_VIOLATION"})[1] == "MEASURED_VIOLATION_SUPPORTED_BY_BOUND"
    assert FAM.q_status(False, {})[:2] == ("INCOMPLETE_OR_INVALID", "Q_NOT_RESOLVED")
    assert FAM.q_status(True, {})[:2] == ("INCOMPLETE_OR_INVALID", "NO_CLAUSES")          # unscored, not "precision"
    assert FAM.q_status(True, {"P34": "PASS"})[:2] == ("INCOMPLETE_OR_INVALID", "MISSING_SLOTS")
    assert FAM.q_status(True, {**Q, "P34": "INVALID"})[0] == "INCOMPLETE_OR_INVALID"
    assert FAM.q_status("NO_ELIGIBLE", Q) == ("NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", "ORDINARY_UTILITY_FAILURE", [])


def test_slot_set_and_cause_by_kind_and_simplest_family():
    E = "ELIGIBLE"
    c = {i: "PASS" for i in FAM.claim_ids("C")}
    assert FAM.claim_status(E, E, c, claim="C")[0] == "PASS"
    assert FAM.claim_status(E, E, {"P23": "PASS"}, claim="C")[:2] == ("INCOMPLETE_OR_INVALID", "MISSING_SLOTS")
    o = {**c, "P23": "MEASURED_VIOLATION", "P28": "NOT_ESTABLISHED_PRECISION"}
    by, detail = FAM.cause_by_kind(o)
    assert by == {"coalition": ["MEASURED_VIOLATION"], "logloss": ["NOT_ESTABLISHED_PRECISION"]}
    assert detail == "MEASURED_VIOLATION_SUPPORTED_BY_BOUND[coalition]; ASSESSMENT_PRECISION_FAILURE[logloss]"
    assert FAM.simplest_family(["JOINT", "SEQ-21"]) == "SEQ-21"
    assert FAM.simplest_family(["JOINT", "SEQ-12", "SEQ-21"]) == "SEQ-12=SEQ-21"
    assert FAM.simplest_family(["JOINT", "LOCAL", "SEQ-12"]) == "LOCAL"
    assert FAM.simplest_family(["JOINT"]) == "JOINT"


def test_truth_table_document_matches_executable():
    T = json.loads((PKG / "LABEL_TRUTH_TABLE.json").read_text())
    assert set(T["clause_outcomes"]) == set(FAM.CLAUSE_OUTCOMES)
    assert set(T["selection_reason_classes"]) == {r for r in FAM.SELECTION_REASONS if r}
    txt = json.dumps(T)
    for lab in ("PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET", "JOINT_DEVELOPMENT_CRITERION_MET",
                "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION", "EXPERIMENTAL_NO_ADVANTAGE",
                "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", "INCOMPLETE_OR_INVALID"):
        assert lab in txt
