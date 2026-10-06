"""Registered primary family and label truth table (frozen in AUDIT_AND_SELECTION_LOCK and PRIMARY_FAMILY.json).

37 slots, always kept (aliases, missing nominees and stage gates never shrink them):
  Claim A: J* (JOINT nominee) vs C_rate (strongest eligible nonjoint control at J*'s U rate caps)      11 clauses
  Claim B: J* vs C_global (strongest eligible nonjoint release overall; JOINT excluded)                 11 clauses
  Claim C: P* (privacy-trained nominee) vs T* (strongest eligible privacy-untrained release, F0 included) 11 clauses
  Q*:      confidence feasibility of the Stage A DIRECT-TASK nominee                                    4 clauses
z = NormalDist().inv_cdf(1 - 0.05/(2*37)) = 3.2048452050105634 (conservative two-sided Bonferroni; nominal; does not
repair adaptive data reuse). B = 1999 paired exact-record-group bootstrap replicates, seed 20261007.

Method-claim clauses for nominee N and comparator C (AUC = SEX AUC of the inner-AUC-selected final attacker, mean over
attacker seeds 0-2; U = the task-only teacher's continuous output; const = OSF_DEFENSE_FIT majority class):
  1      AUC_pair(C) - AUC_pair(N)                      lower bound > 0.02
  2-3    AUC_vi(N) - AUC_vi(C), i = 1, 2                upper bound < 0.01
  4-5    Acc_t(N) - Acc_t(U)                            lower bound > -0.01
  6-7    LogLoss_t(N) - LogLoss_t(U)                    upper bound < 0.01 nats
  8-9    Brier_t(N) - Brier_t(U)                        upper bound < 0.005
  10-11  Acc_t(N) - 0.8 Acc_t(U) - 0.2 Acc_t(const)     lower bound > 0
Q* clauses: LogLoss_t(Q*) - LogLoss_t(U) upper bound < 0.01 (t = income, occupation); Brier_t(Q*) - Brier_t(U) upper
bound < 0.005. Q*'s decision preservation and inner eligibility are prerequisite invariants.
"""
from statistics import NormalDist

ALPHA = 0.05
B, BOOT_SEED = 1999, 20261007
TASKS = ("income", "occ")
CLAIMS = {"A": ("J*", "C_rate"), "B": ("J*", "C_global"), "C": ("P*", "T*")}
PRIMARY_SIZE = 37
Z_PRIMARY = NormalDist().inv_cdf(1 - ALPHA / (2 * PRIMARY_SIZE))
assert repr(Z_PRIMARY) == "3.2048452050105634"

PRIMARY = []
for ci, (claim, (nom, ref)) in enumerate(CLAIMS.items()):
    base = 11 * ci
    alias = claim == "B"                       # B shares J* with A: its utility clauses are structural aliases
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
        PRIMARY.append({"id": f"P{off + j:02d}", "claim": "Q", "kind": kind, "task": j, "nominee": "Q*",
                        "stat": f"{'LogLoss' if kind == 'logloss' else 'Brier'}_{t}(Q*) - "
                                f"{'LogLoss' if kind == 'logloss' else 'Brier'}_{t}(U)",
                        "target": target, "side": "upper<", "alias_of": None})
assert len(PRIMARY) == PRIMARY_SIZE == len({e["id"] for e in PRIMARY})
assert [e["id"] for e in PRIMARY] == [f"P{i:02d}" for i in range(1, 38)]

# ------------------------------------------------------------------ truth table (LABEL_TRUTH_TABLE.json)
ROLE_STATES = ("ELIGIBLE", "NO_ELIGIBLE", "TECHNICAL_FAILURE")
CLAUSE_STATES = ("ALL_PASS", "SOME_NOT_PASS", "ANY_INVALID")
COMPLETED_NEGATIVE = ("NOT_ESTABLISHED", "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE")
COVERAGE_MISSING = ("NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR", "INVALID_COMPARATOR", "INVALID_NOMINEE", "INVALID")


def role_state(status: dict | None) -> str:
    """Map a selection status record to ELIGIBLE / NO_ELIGIBLE / TECHNICAL_FAILURE."""
    s = (status or {}).get("status")
    if s == "NOMINEE" and (status or {}).get("config"):
        return "ELIGIBLE"
    if s in ("NO_ELIGIBLE_NOMINEE", "NO_ELIGIBLE_COMPARATOR"):
        return "NO_ELIGIBLE"
    return "TECHNICAL_FAILURE"


def clause_state(decisions) -> str:
    d = list(decisions)
    if any(x == "INVALID" for x in d):
        return "ANY_INVALID"
    return "ALL_PASS" if d and all(x == "PASS" for x in d) else "SOME_NOT_PASS"


def claim_status(nominee: str, comparator: str, clauses: str) -> str:
    assert nominee in ROLE_STATES and comparator in ROLE_STATES and clauses in CLAUSE_STATES
    if comparator == "TECHNICAL_FAILURE":
        return "INVALID_COMPARATOR"
    if nominee == "TECHNICAL_FAILURE":
        return "INVALID_NOMINEE"
    if nominee == "NO_ELIGIBLE":
        return "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
    if comparator == "NO_ELIGIBLE":
        return "NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR"
    if clauses == "ANY_INVALID":
        return "INVALID"
    return "PASS" if clauses == "ALL_PASS" else "NOT_ESTABLISHED"


def q_status(q_exists: bool, clauses: str) -> str:
    if not q_exists:
        return "NOT_APPLICABLE_NO_Q"
    if clauses == "ANY_INVALID":
        return "INVALID"
    return "PASS" if clauses == "ALL_PASS" else "NOT_ESTABLISHED"


def overall_label(stage_a_valid: bool, gate_met: bool, technical_valid: bool, claims: dict, q: str,
                  winning_family: str | None = None):
    """Returns (label, missing_items). claims: {"A": status, "B": status, "C": status}."""
    if not stage_a_valid:
        return "INCOMPLETE_OR_INVALID", ["Stage A technically incomplete or invalid"]
    if not gate_met:
        return "CAPACITY_GATE_NOT_MET", []
    missing = [] if technical_valid else ["required technical validity failed"]
    missing += [f"claim {c}: {s}" for c, s in sorted(claims.items()) if s in COVERAGE_MISSING]
    if q == "INVALID":
        missing.append("Q*: INVALID")
    if q == "NOT_APPLICABLE_NO_Q":
        missing.append("Q*: no eligible Stage A configuration despite a met gate")
    if missing:
        return "INCOMPLETE_OR_INVALID", missing
    labels = []
    if claims["A"] == "PASS" and claims["B"] == "PASS":
        labels.append("JOINT_DEVELOPMENT_CRITERION_MET")
    if claims["C"] == "PASS":
        labels.append("PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET"
                      + (f" ({winning_family})" if winning_family else ""))
    if labels:
        return " + ".join(labels), []
    if q == "PASS":
        return "CONFIDENCE_FEASIBILITY_ESTABLISHED", []
    return "EXPERIMENTAL_NO_ADVANTAGE", []
