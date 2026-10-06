"""Registered primary family, clause classifier and label truth table (LABEL_TRUTH_TABLE.json; frozen before any fit).

Adapted from qpc/family.py at d0c8a45 (same 37 slots and margins); new bootstrap seed 20261008; new labels.
  Claim A: J* vs C_rate      11 clauses       Claim B: J* vs C_global   11 clauses
  Claim C: P* vs T*          11 clauses       Q: DIRECT-TASK i8o64 confidence feasibility, 4 clauses
z = NormalDist().inv_cdf(1 - 0.05/(2*37)) (verified, not copied); B = 1999 paired exact-record-group bootstrap replicates.
Method-claim clauses for nominee N and comparator C (recovery = SEX AUC of the inner-AUC-selected attacker, mean over
attacker seeds 0-2; U = continuous teacher; const = OSF_DEFENSE_FIT majority class):
  1      AUC_pair(C) - AUC_pair(N)                      lower bound > 0.02
  2-3    AUC_vi(N) - AUC_vi(C), i = 1, 2                upper bound < 0.01
  4-5    Acc_t(N) - Acc_t(U)                            lower bound > -0.01
  6-7    LogLoss_t(N) - LogLoss_t(U)                    upper bound < 0.01 nats
  8-9    Brier_t(N) - Brier_t(U)                        upper bound < 0.005
  10-11  Acc_t(N) - 0.8 Acc_t(U) - 0.2 Acc_t(const)     lower bound > 0
Q: LogLoss_t(Q) - LogLoss_t(U) upper bound < 0.01; Brier_t(Q) - Brier_t(U) upper bound < 0.005 (t = income, occupation).
"""
from statistics import NormalDist

ALPHA = 0.05
B, BOOT_SEED = 1999, 20261008
TASKS = ("income", "occ")
CLAIMS = {"A": ("J*", "C_rate"), "B": ("J*", "C_global"), "C": ("P*", "T*")}
PRIMARY_SIZE = 37
Z_PRIMARY = NormalDist().inv_cdf(1 - ALPHA / (2 * PRIMARY_SIZE))
assert abs(Z_PRIMARY - 3.2048452050105634) < 1e-15

PRIMARY = []
for ci, (claim, (nom, ref)) in enumerate(CLAIMS.items()):
    base = 11 * ci
    alias = claim == "B"
    PRIMARY += [
        {"id": f"P{base + 1:02d}", "claim": claim, "kind": "coalition", "nominee": nom, "ref": ref,
         "stat": f"AUC_pair({ref}) - AUC_pair({nom})", "target": 0.02, "side": "lower>"},
        {"id": f"P{base + 2:02d}", "claim": claim, "kind": "local", "view": "v1", "nominee": nom, "ref": ref,
         "stat": f"AUC_v1({nom}) - AUC_v1({ref})", "target": 0.01, "side": "upper<"},
        {"id": f"P{base + 3:02d}", "claim": claim, "kind": "local", "view": "v2", "nominee": nom, "ref": ref,
         "stat": f"AUC_v2({nom}) - AUC_v2({ref})", "target": 0.01, "side": "upper<"}]
    for off, kind, target, side, fmt in ((4, "acc", -0.01, "lower>", "Acc_{t}({n}) - Acc_{t}(U)"),
                                         (6, "logloss", 0.01, "upper<", "LogLoss_{t}({n}) - LogLoss_{t}(U)"),
                                         (8, "brier", 0.005, "upper<", "Brier_{t}({n}) - Brier_{t}(U)"),
                                         (10, "retain", 0.0, "lower>", "Acc_{t}({n}) - 0.8 Acc_{t}(U) - 0.2 const_{t}")):
        for j, t in enumerate(TASKS):
            PRIMARY.append({"id": f"P{base + off + j:02d}", "claim": claim, "kind": kind, "task": j, "nominee": nom,
                            "stat": fmt.format(t=t, n=nom), "target": target, "side": side,
                            "alias_of": f"P{off + j:02d}" if alias else None})
for off, kind, target in ((34, "logloss", 0.01), (36, "brier", 0.005)):
    for j, t in enumerate(TASKS):
        nm = "LogLoss" if kind == "logloss" else "Brier"
        PRIMARY.append({"id": f"P{off + j:02d}", "claim": "Q", "kind": kind, "task": j, "nominee": "Q",
                        "stat": f"{nm}_{t}(Q) - {nm}_{t}(U)", "target": target, "side": "upper<", "alias_of": None})
assert len(PRIMARY) == PRIMARY_SIZE == len({e["id"] for e in PRIMARY})
assert [e["id"] for e in PRIMARY] == [f"P{i:02d}" for i in range(1, 38)]

# ------------------------------------------------------------------ clause classifier
CLAUSE_OUTCOMES = ("PASS", "NOT_ESTABLISHED_PRECISION", "NOT_ESTABLISHED_POINT", "MEASURED_VIOLATION", "INVALID")


def clause_outcome(side, target, point, lower, upper):
    """PASS: the registered bound clears the target. NOT_ESTABLISHED_PRECISION: the point estimate satisfies the
    target but the bound does not (assessment precision failure). MEASURED_VIOLATION: the OTHER bound excludes the target
    on the failing side (a violation supported by the interval). NOT_ESTABLISHED_POINT: the point fails but no violation
    is established. INVALID: any nonfinite input."""
    import math
    if any(x is None or not math.isfinite(float(x)) for x in (point, lower, upper)):
        return "INVALID"
    if side == "lower>":
        if lower > target:
            return "PASS"
        if upper < target:
            return "MEASURED_VIOLATION"
        return "NOT_ESTABLISHED_PRECISION" if point > target else "NOT_ESTABLISHED_POINT"
    if upper < target:
        return "PASS"
    if lower > target:
        return "MEASURED_VIOLATION"
    return "NOT_ESTABLISHED_PRECISION" if point < target else "NOT_ESTABLISHED_POINT"


# ------------------------------------------------------------------ truth table
ROLE_STATES = ("ELIGIBLE", "NO_ELIGIBLE", "TECHNICAL_FAILURE")
SELECTION_REASONS = ("FIT_OR_ADMISSION_FAILURE", "ORDINARY_UTILITY_FAILURE", "HEADROOM_SELECTION_FAILURE",
                     "LOCAL_GUARD_FAILURE", "MISSING_GUARD_COMPARATOR", None)
CLAIM_STATUSES = ("PASS", "NOT_ESTABLISHED", "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", "INCOMPLETE_OR_INVALID")


def role_state(status):
    s = (status or {}).get("status")
    if s == "NOMINEE" and (status or {}).get("config"):
        return "ELIGIBLE"
    if s in ("NO_ELIGIBLE_NOMINEE", "NO_ELIGIBLE_COMPARATOR"):
        return "NO_ELIGIBLE"
    return "TECHNICAL_FAILURE"


def claim_ids(claim):
    return [e["id"] for e in PRIMARY if e["claim"] == claim]


def _kind_of(i):
    return next(e["kind"] for e in PRIMARY if e["id"] == i)


def _cause(bad):
    kinds = set(bad.values())
    return ("MEASURED_VIOLATION_SUPPORTED_BY_BOUND" if "MEASURED_VIOLATION" in kinds else
            "ASSESSMENT_PRECISION_FAILURE" if kinds == {"NOT_ESTABLISHED_PRECISION"} else "CLAUSE_NOT_ESTABLISHED")


def claim_status(nominee, comparator, outcomes, required_control_ok=True, nominee_reason=None, claim=None):
    """Returns (status, root_cause, failing). outcomes: {clause id: clause_outcome}.
    Precedence: technical/coverage failure -> no eligible nominee -> missing eligible comparator -> registered slot set
    incomplete -> any INVALID clause -> all PASS -> NOT_ESTABLISHED with the failing clauses classified.
    claim: when given, outcomes must cover exactly that claim's registered slots (never a shrunken conjunction)."""
    assert nominee in ROLE_STATES and comparator in ROLE_STATES
    if not required_control_ok:
        return "INCOMPLETE_OR_INVALID", "FAILED_REQUIRED_CONTROL", []
    if comparator == "TECHNICAL_FAILURE":
        return "INCOMPLETE_OR_INVALID", "INVALID_OR_MISSING_COMPARATOR", []
    if nominee == "TECHNICAL_FAILURE":
        return "INCOMPLETE_OR_INVALID", nominee_reason or "FIT_OR_ADMISSION_FAILURE", []
    if nominee == "NO_ELIGIBLE":
        return "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", nominee_reason, []
    if comparator == "NO_ELIGIBLE":
        return "INCOMPLETE_OR_INVALID", "NO_ELIGIBLE_COMPARATOR", []
    if not outcomes:
        return "INCOMPLETE_OR_INVALID", "NO_CLAUSES", []
    if claim is not None and sorted(outcomes) != claim_ids(claim):
        return "INCOMPLETE_OR_INVALID", "MISSING_SLOTS", sorted(set(claim_ids(claim)) ^ set(outcomes))
    bad = {i: o for i, o in outcomes.items() if o != "PASS"}
    if any(o == "INVALID" for o in bad.values()):
        return "INCOMPLETE_OR_INVALID", "NONFINITE_PRIMARY_QUANTITY", sorted(bad)
    if not bad:
        return "PASS", "COMPLETE_PASSING_CONJUNCTION", []
    return "NOT_ESTABLISHED", _cause(bad), sorted(bad)


def cause_by_kind(outcomes):
    """Reporting only (never changes a status): the failing clauses grouped by clause kind, e.g.
    'MEASURED_VIOLATION_SUPPORTED_BY_BOUND[coalition]; ASSESSMENT_PRECISION_FAILURE[logloss]'."""
    by = {}
    for i, o in sorted(outcomes.items()):
        if o != "PASS":
            by.setdefault(_kind_of(i), {})[i] = o
    detail = "; ".join(f"{_cause(b)}[{k}]" for k, b in by.items())
    return {k: sorted(set(b.values())) for k, b in by.items()}, detail


def q_status(q_state, outcomes):
    """q_state: a ROLE_STATE (True / False accepted as ELIGIBLE / TECHNICAL_FAILURE). Returns (status, cause, failing).
    Q not ordinarily eligible -> NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE (ORDINARY_UTILITY_FAILURE); its 4 slots are then
    scored DESCRIPTIVE_ONLY and can never pass."""
    q_state = {True: "ELIGIBLE", False: "TECHNICAL_FAILURE"}.get(q_state, q_state)
    assert q_state in ROLE_STATES
    if q_state == "TECHNICAL_FAILURE":
        return "INCOMPLETE_OR_INVALID", "Q_NOT_RESOLVED", []
    if q_state == "NO_ELIGIBLE":
        return "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", "ORDINARY_UTILITY_FAILURE", []
    if not outcomes:
        return "INCOMPLETE_OR_INVALID", "NO_CLAUSES", []
    if sorted(outcomes) != claim_ids("Q"):
        return "INCOMPLETE_OR_INVALID", "MISSING_SLOTS", sorted(set(claim_ids("Q")) ^ set(outcomes))
    bad = {i: o for i, o in outcomes.items() if o != "PASS"}
    if any(o == "INVALID" for o in bad.values()):
        return "INCOMPLETE_OR_INVALID", "NONFINITE_PRIMARY_QUANTITY", sorted(bad)
    if not bad:
        return "PASS", "COMPLETE_PASSING_CONJUNCTION", []
    return "NOT_ESTABLISHED", _cause(bad), sorted(bad)


# simplest-story family order for aliased winners (prompt sec. 9 / 17: no reward for being called JOINT)
FAMILY_SIMPLICITY = {"LOCAL": 0, "SEQ-12": 1, "SEQ-21": 1, "JOINT": 2}


def simplest_family(families):
    """The family named in the label when P*'s deployed maps are identical (all three seeds) to other families' maps:
    LOCAL < SEQ-12 = SEQ-21 < JOINT; equally simple families are joined with '='."""
    fs = sorted(set(families), key=lambda f: (FAMILY_SIMPLICITY[f], f))
    lo = FAMILY_SIMPLICITY[fs[0]]
    return "=".join(f for f in fs if FAMILY_SIMPLICITY[f] == lo)


def overall_label(claims, q, technical_valid=True, winning_family=None):
    """claims: {"A": status, "B": status, "C": status}; q: Q status. Returns (label, displayed_statuses).
    Favourable labels for validly passing conjunctions are reported even if another claim is incomplete; the
    displayed statuses always show every claim (prompt sec. 11). technical_valid=False is reserved for failures that
    touch EVERY claim (LABEL_TRUTH_TABLE.json failure_scope); a failure that touches only some claims is routed through
    claim_status(required_control_ok=False) of those claims, so a valid passing claim beside it is still reported."""
    shown = {**claims, "Q": q}
    if not technical_valid:
        return "INCOMPLETE_OR_INVALID", shown
    labels = []
    if claims.get("C") == "PASS":
        labels.append("PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET" + (f" ({winning_family})" if winning_family else ""))
    if claims.get("A") == "PASS" and claims.get("B") == "PASS":
        labels.append("JOINT_DEVELOPMENT_CRITERION_MET")
    if labels:
        return " + ".join(labels), shown
    if any(s == "INCOMPLETE_OR_INVALID" for s in shown.values()):
        return "INCOMPLETE_OR_INVALID", shown
    if q == "PASS":
        return "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION", shown
    return "EXPERIMENTAL_NO_ADVANTAGE", shown
