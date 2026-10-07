"""[lra port of lcr/tests/test_truth_table.py at 091afc2: lcr->lra renames; later edits: REVIEW_FINDINGS_DISPOSITION.json
(F08/F10/F12/F14)]
Exhaustive truth table (LABEL_TRUTH_TABLE.json <-> lra.family), fixed 37-slot family and clause classifier."""
import re
import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist

from lra import family as FAM

PKG = Path(__file__).resolve().parents[2] / "results" / "pcrl_adult_learned_decoder_release_v1"


def test_family_fixed_and_z_verified():
    assert FAM.PRIMARY_SIZE == 37 and len(FAM.PRIMARY) == 37
    assert FAM.Z_PRIMARY == NormalDist().inv_cdf(1 - 0.05 / 74)
    assert abs(FAM.Z_PRIMARY - 3.2048452050105634) < 1e-15
    assert FAM.B == 1999 and FAM.BOOT_SEED == 20261010
    assert FAM.Z_SUPPLEMENTARY == NormalDist().inv_cdf(0.975)
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
    for reason in ("ORDINARY_UTILITY_FAILURE", "LOCAL_GUARD_FAILURE", "CONSTRAINED_FIT_INFEASIBLE"):   # F12
        assert FAM.claim_status(N, E, _outs(), nominee_reason=reason) == \
            ("NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", reason, [])
    for reason in FAM.TECHNICAL_SELECTION_REASONS + ("MISSING_GUARD_COMPARATOR",):
        assert FAM.claim_status(T, E, _outs(), nominee_reason=reason)[:2] == ("INCOMPLETE_OR_INVALID", reason)
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


GATES = ("ENGINEERING_READY", "ENGINEERING_BLOCKED", None, "GATE_MET", "GATE_NOT_MET")


def _expected(a, b, c, q, tv, gate, prefit, winning="SEQ-21; constrained", disc=None):
    """Independent restatement of PROTOCOL.md section 12 / LABEL_TRUTH_TABLE.json overall_label_rules_in_precedence_order."""
    disc = disc or {}
    if gate != "ENGINEERING_READY" or prefit:
        shown = {"A": "NOT_RUN", "B": "NOT_RUN", "C": "NOT_RUN", "Q": "NOT_RUN"}
        head = "ENGINEERING_BLOCKED_NOT_RUN" if gate == "ENGINEERING_BLOCKED" else "INCOMPLETE_NOT_RUN"
        return head, shown
    shown = {"A": a, "B": b, "C": c, "Q": q}
    if not tv:
        return "INCOMPLETE_OR_INVALID", shown
    parts = []
    if a == "PASS":
        parts.append(f"PRIVACY_RELEASE_DEVELOPMENT_CRITERION_MET ({winning})")
    if b == "PASS":
        parts.append("CONSTRAINED_SEARCH_INCREMENT_ESTABLISHED" + (f" ({disc['B']})" if "B" in disc else ""))
    if c == "PASS":
        parts.append("PAIRED_JOINT_INCREMENT_ESTABLISHED" + (f" ({disc['C']})" if "C" in disc else ""))
    if parts:
        return " + ".join(parts), shown
    if q == "PASS":
        return "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION", shown
    if "INCOMPLETE_OR_INVALID" in (a, b, c, q):
        return "INCOMPLETE_OR_INVALID", shown
    return "EXPERIMENTAL_NO_ADVANTAGE", shown


def test_overall_label_whole_truth_table_with_engineering_gate():
    """F10/F14: every combination of claim statuses, Q, global technical validity, engineering-gate verdict and pre-fit
    blocker; the label always carries A/B/C/Q separately; readiness is never assumed; MECHANISM_GATE_NOT_MET never
    appears."""
    S = FAM.CLAIM_STATUSES
    QS = ("PASS", "NOT_ESTABLISHED", "INCOMPLETE_OR_INVALID", "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE")
    n = 0
    for a, b, c in itertools.product(S, repeat=3):
        for q, tv, gate, prefit in itertools.product(QS, (True, False), GATES, (None, "admission parity failed")):
            lab, shown = FAM.overall_label({"A": a, "B": b, "C": c}, q, tv, "SEQ-21; constrained", gate, prefit)
            head, want_shown = _expected(a, b, c, q, tv, gate, prefit)
            assert shown == want_shown
            assert lab.endswith(FAM.status_suffix(want_shown)), lab
            assert FAM.label_headline(lab).startswith(head), (lab, head)
            assert all(f"{k}={want_shown[k]}" in lab for k in "ABCQ")           # A/B/C/Q always displayed
            assert "MECHANISM_GATE_NOT_MET" not in lab
            assert FAM.label_headline(lab).split(" (")[0].split(" + ")[0] in FAM.OVERALL_LABELS
            if head == "INCOMPLETE_NOT_RUN" and prefit:
                assert "pre-fit blocker: admission parity failed" in lab      # its concrete reason
            n += 1
    assert n == 4 ** 3 * 4 * 2 * len(GATES) * 2


def test_overall_label_required_gate_argument_and_q_never_hides_incomplete_work():
    import inspect
    sig = inspect.signature(FAM.overall_label)
    assert sig.parameters["engineering_gate"].default is inspect.Parameter.empty      # never defaulted to ready
    assert "gate_met" not in sig.parameters
    # a Q-only result next to incomplete method work: Q headline, incomplete claims in the label itself
    lab, _ = FAM.overall_label({"A": "INCOMPLETE_OR_INVALID", "B": "NOT_ESTABLISHED", "C": "INCOMPLETE_OR_INVALID"},
                               "PASS", True, None, "ENGINEERING_READY")
    assert lab == ("CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION [A=INCOMPLETE_OR_INVALID; B=NOT_ESTABLISHED; "
                   "C=INCOMPLETE_OR_INVALID; Q=PASS]")
    # a global technical failure after science: INCOMPLETE_OR_INVALID even with a valid Q and a passing A
    lab, _ = FAM.overall_label({"A": "PASS", "B": "PASS", "C": "PASS"}, "PASS", False, "X; y", "ENGINEERING_READY")
    assert FAM.label_headline(lab) == "INCOMPLETE_OR_INVALID" and "Q=PASS" in lab and "A=PASS" in lab
    # the gate failure is the NEW verdict, never the historical source label
    lab, shown = FAM.overall_label({"A": "PASS", "B": "PASS", "C": "PASS"}, "PASS", True, "X", "ENGINEERING_BLOCKED")
    assert lab == "ENGINEERING_BLOCKED_NOT_RUN [A=NOT_RUN; B=NOT_RUN; C=NOT_RUN; Q=NOT_RUN]"
    assert set(shown.values()) == {"NOT_RUN"}
    for legacy in ("GATE_MET", "GATE_NOT_MET", True, None):
        assert FAM.label_headline(FAM.overall_label({"A": "PASS"}, "PASS", True, "X", legacy)[0]).startswith(
            "INCOMPLETE_NOT_RUN")
    lab = FAM.overall_label({"A": "NOT_ESTABLISHED"}, "NOT_ESTABLISHED", True, None, "ENGINEERING_READY",
                            prefit_blocker="TIMING: mandatory bank exceeds the CPU ceiling")[0]
    assert lab.startswith("INCOMPLETE_NOT_RUN (pre-fit blocker: TIMING: mandatory bank exceeds the CPU ceiling)")
    # a passing A next to an incomplete B: reported, both displayed; B can pass without A
    lab, shown = FAM.overall_label({"A": "PASS", "B": "INCOMPLETE_OR_INVALID", "C": "NOT_ESTABLISHED"}, "PASS", True,
                                   "LOCAL; calibrated", "ENGINEERING_READY")
    assert FAM.label_headline(lab) == "PRIVACY_RELEASE_DEVELOPMENT_CRITERION_MET (LOCAL; calibrated)"
    assert "B=INCOMPLETE_OR_INVALID" in lab
    lab = FAM.overall_label({"A": "NOT_ESTABLISHED", "B": "PASS", "C": "NOT_ESTABLISHED"}, "PASS", True, None,
                            "ENGINEERING_READY", disclosures={"B": "N* release identical to privacy-untrained X"})[0]
    assert FAM.label_headline(lab) == ("CONSTRAINED_SEARCH_INCREMENT_ESTABLISHED (N* release identical to "
                                       "privacy-untrained X)")


def test_claims_and_family_layout():
    assert FAM.CLAIMS == {"A": ("P*", "T*"), "B": ("N*", "C*"), "C": ("J*", "C_pair*")}
    for c, (n, m) in FAM.CLAIMS.items():
        cl = [e for e in FAM.PRIMARY if e["claim"] == c]
        assert all(e["nominee"] == n for e in cl) and cl[0]["ref"] == m and all(e.get("alias_of") is None for e in cl)
    assert FAM.simplest_family(["JOINT-PAIR", "SEQ-12"]) == "SEQ-12"
    assert FAM.simplest_family(["JOINT-SINGLE", "JOINT-PAIR"]) == "JOINT-SINGLE"


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
    assert FAM.simplest_family(["JOINT", "SEQ-12", "SEQ-21"]) == "SEQ-12=SEQ-21"     # reporting helper only
    assert FAM.simplest_family(["JOINT", "LOCAL", "SEQ-12"]) == "LOCAL"
    assert FAM.simplest_family(["JOINT"]) == "JOINT"


def test_truth_table_document_matches_executable():
    T = json.loads((PKG / "LABEL_TRUTH_TABLE.json").read_text())
    assert set(T["clause_outcomes"]) == set(FAM.CLAUSE_OUTCOMES)
    assert set(T["selection_reason_classes"]) == {r for r in FAM.SELECTION_REASONS if r}
    assert set(T["overall_labels"]) == set(FAM.OVERALL_LABELS)
    assert T["engineering_gate_verdicts"] == list(FAM.ENGINEERING_VERDICTS)
    txt = json.dumps(T)
    for lab in FAM.OVERALL_LABELS + ("NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE",):
        assert lab in txt
    assert T["bootstrap_seed"] == FAM.BOOT_SEED and T["z_primary"] == FAM.Z_PRIMARY
    assert T["z_supplementary"] == FAM.Z_SUPPLEMENTARY


STALE = ("C_rate", "C_global", "HEADROOM_SELECTION_FAILURE", "cbp.family", "cbp/", "lcr.family", "lcr/", "gate_met",
         "GATE_MET")


def _strip_historical(o):
    if isinstance(o, dict):
        return {k: _strip_historical(v) for k, v in o.items() if k != "historical"}
    if isinstance(o, list):
        return [_strip_historical(v) for v in o]
    return o


def test_f08_f12_no_stale_cbp_or_lcr_names_and_every_reason_code_mapped():
    """F08/F12: the registered documents carry no cbp role or function names (C_rate, C_global, HEADROOM, cbp.family)
    outside an explicit 'historical' key, and every reason code emitted by lra/select.py is a registered class in
    both LABEL_TRUTH_TABLE.json and SELECTION_RULES.json (and vice versa)."""
    from lra import select as SEL
    T = json.loads((PKG / "LABEL_TRUTH_TABLE.json").read_text())
    SR = json.loads((PKG / "SELECTION_RULES.json").read_text())
    for name, doc in (("LABEL_TRUTH_TABLE", T), ("SELECTION_RULES", SR)):
        txt = json.dumps(_strip_historical(doc))
        for bad in STALE:
            assert bad not in txt, (name, bad)
        assert "MECHANISM_GATE_NOT_MET" not in txt.replace(json.dumps(FAM.HISTORICAL_SOURCE_LABEL), ""), name
    assert T["historical"]["source_label"] == FAM.HISTORICAL_SOURCE_LABEL
    src = (Path(SEL.__file__)).read_text()
    emitted = set(re.findall(r'"code": "([A-Z_]+)"', src)) | set(re.findall(r'"reason": "([A-Z_]+)"', src))
    emitted |= {r for r in SEL.FALLBACK_CLASS}
    registered = {r for r in FAM.SELECTION_REASONS if r}
    assert emitted <= registered, emitted - registered
    assert set(SEL.TECHNICAL_REASONS) == set(FAM.TECHNICAL_SELECTION_REASONS)
    assert set(SR["fallback_ordering"]["reason_codes"]) == registered
    assert set(SR["fallback_ordering"]["fallback_classes"]) == set(SEL.FALLBACK_CLASS.values())
    for r, cls in SEL.FALLBACK_CLASS.items():
        assert SR["fallback_ordering"]["fallback_class_of_reason"][r] == cls
    assert SR["roles"]["T*"]["candidates"] == list(SEL.T_STAR_CLOSED)
    assert SR["schema"].startswith("lra-") and T["schema"].startswith("lra-")


def test_primary_family_document_matches_executable():
    d = json.loads((PKG / "PRIMARY_FAMILY.json").read_text())
    assert d["size"] == 37 and d["B"] == FAM.B == 1999 and d["bootstrap_seed"] == FAM.BOOT_SEED == 20261010
    assert d["z"] == FAM.Z_PRIMARY == NormalDist().inv_cdf(1 - 0.05 / 74)
    assert d["slots"] == json.loads(json.dumps(FAM.PRIMARY))
    assert d["supplementary"]["z"] == FAM.Z_SUPPLEMENTARY == NormalDist().inv_cdf(0.975)
