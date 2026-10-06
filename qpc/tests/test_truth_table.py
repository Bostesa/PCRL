"""Exhaustive label truth table (LABEL_TRUTH_TABLE.json <-> qpc.family) and the fixed 37-slot family."""
import itertools
import json
from pathlib import Path

from qpc import family as FAM

PKG = Path(__file__).resolve().parents[2] / "results" / "pcrl_confidence_capacity_v1"


def test_family_fixed_37_slots_and_z():
    assert FAM.PRIMARY_SIZE == 37 and len(FAM.PRIMARY) == 37
    assert repr(FAM.Z_PRIMARY) == "3.2048452050105634"
    assert [e["id"] for e in FAM.PRIMARY if e["claim"] == "Q"] == ["P34", "P35", "P36", "P37"]
    assert sum(e["claim"] == c for e in FAM.PRIMARY for c in "ABC") == 33
    assert all(e["target"] == 0.02 and e["side"] == "lower>" for e in FAM.PRIMARY if e["kind"] == "coalition")


def test_claim_status_exhaustive():
    seen = {}
    for nom, comp, cl in itertools.product(FAM.ROLE_STATES, FAM.ROLE_STATES, FAM.CLAUSE_STATES):
        s = FAM.claim_status(nom, comp, cl)
        seen[(nom, comp, cl)] = s
        if comp == "TECHNICAL_FAILURE":
            assert s == "INVALID_COMPARATOR"
        elif nom == "TECHNICAL_FAILURE":
            assert s == "INVALID_NOMINEE"
        elif nom == "NO_ELIGIBLE":
            assert s == "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
        elif comp == "NO_ELIGIBLE":
            assert s == "NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR"
        else:
            assert s == {"ANY_INVALID": "INVALID", "ALL_PASS": "PASS", "SOME_NOT_PASS": "NOT_ESTABLISHED"}[cl]
    # a fallback (no eligible nominee) can never pass, whatever its clauses say
    assert all(seen[("NO_ELIGIBLE", c, "ALL_PASS")] != "PASS" for c in ("ELIGIBLE", "NO_ELIGIBLE"))
    # a missing eligible comparator is never a completed negative
    assert seen[("ELIGIBLE", "NO_ELIGIBLE", "SOME_NOT_PASS")] in FAM.COVERAGE_MISSING


def test_overall_label_exhaustive():
    statuses = sorted(set(FAM.COMPLETED_NEGATIVE) | set(FAM.COVERAGE_MISSING) | {"PASS"})
    qs = ("PASS", "NOT_ESTABLISHED", "INVALID", "NOT_APPLICABLE_NO_Q")
    for a, b, c in itertools.product(statuses, repeat=3):
        for q in qs:
            for tv in (True, False):
                lab, miss = FAM.overall_label(True, True, tv, {"A": a, "B": b, "C": c}, q, "LOCAL")
                cov = any(x in FAM.COVERAGE_MISSING for x in (a, b, c)) or q in ("INVALID", "NOT_APPLICABLE_NO_Q")
                if not tv:
                    assert lab == "INCOMPLETE_OR_INVALID" and miss
                    continue
                assert bool(miss) == cov                      # missing coverage is always listed, never hidden
                if a == "PASS" and b == "PASS":
                    assert lab.startswith("JOINT_DEVELOPMENT_CRITERION_MET")
                    assert ("PRIVACY_COMPRESSION" in lab) == (c == "PASS")
                elif c == "PASS":
                    assert lab == "PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (LOCAL)"
                elif cov:
                    assert lab == "INCOMPLETE_OR_INVALID"     # never a completed negative with missing coverage
                elif q == "PASS":
                    assert lab == "CONFIDENCE_FEASIBILITY_ESTABLISHED"
                else:
                    assert lab == "EXPERIMENTAL_NO_ADVANTAGE"
    assert FAM.overall_label(True, False, True, {}, "NOT_APPLICABLE_NO_Q")[0] == "CAPACITY_GATE_NOT_MET"
    assert FAM.overall_label(False, True, True, {}, "PASS")[0] == "INCOMPLETE_OR_INVALID"
    # joint needs BOTH A and B: passing only against continuous U (claim B) is not a joint contribution
    assert FAM.overall_label(True, True, True, {"A": "NOT_ESTABLISHED", "B": "PASS", "C": "NOT_ESTABLISHED"},
                             "NOT_ESTABLISHED")[0] == "EXPERIMENTAL_NO_ADVANTAGE"


def test_role_state_mapping():
    assert FAM.role_state({"status": "NOMINEE", "config": "U|JOINT|i8o32|l0.1"}) == "ELIGIBLE"
    assert FAM.role_state({"status": "NOMINEE", "config": None}) == "TECHNICAL_FAILURE"
    assert FAM.role_state({"status": "NO_ELIGIBLE_NOMINEE"}) == "NO_ELIGIBLE"
    assert FAM.role_state({"status": "NO_ELIGIBLE_COMPARATOR"}) == "NO_ELIGIBLE"
    assert FAM.role_state({"status": "INVALID_COMPARATOR"}) == "TECHNICAL_FAILURE"
    assert FAM.role_state(None) == "TECHNICAL_FAILURE"


def test_truth_table_document_matches_executable():
    T = json.loads((PKG / "LABEL_TRUTH_TABLE.json").read_text())
    cls = T["claim_status_classes"]
    assert tuple(cls["completed_negative"]) == FAM.COMPLETED_NEGATIVE
    assert set(cls["coverage_missing_or_invalid"]) == set(FAM.COVERAGE_MISSING)
    assert cls["favourable"] == ["PASS"]
