"""Registered primary family (frozen in SELECTION_AND_AUDIT_LOCK and PRIMARY_FAMILY.json before the assessment).

33 slots = three 11-clause conjunctions; deliberately conservative two-sided Bonferroni critical value
z = NormalDist().inv_cdf(1 - 0.05/(2*33)) = 3.1717657833516224 (never shrunk after selection, failures or aliases; a
nominal simultaneous benchmark calculation that does not repair adaptive data reuse).
  Claim A: J* (JOINT nominee) vs C_match (its matched teacher/rate nonjoint control)
  Claim B: J* vs C_global (the strongest eligible nonjoint control overall)
  Claim C: P* (privacy-trained nominee) vs T* (strongest eligible privacy-untrained score/compression release)
Clauses for nominee N and comparator C (recovery = SEX AUC of the inner-AUC-selected final attacker, mean over attacker
seeds 0-2; U = the task-only teacher's continuous deployed output; const = OSF_DEFENSE_FIT majority class):
  1      AUC_pair(C) - AUC_pair(N)                lower bound > 0.02
  2-3    AUC_vi(N) - AUC_vi(C), i = 1, 2          upper bound < 0.01
  4-5    Acc_t(N) - Acc_t(U)                      lower bound > -0.01
  6-7    LogLoss_t(N) - LogLoss_t(U)              upper bound < 0.01 nats
  8-9    Brier_t(N) - Brier_t(U)                  upper bound < 0.005
  10-11  Acc_t(N) - 0.8 Acc_t(U) - 0.2 Acc_t(const)  lower bound > 0
A claim passes only if all 11 clauses PASS (strict) and its nominee and comparator are valid NOMINEEs; a descriptive
fallback is DESCRIPTIVE_ONLY on every row and cannot pass. B=1999 paired exact-record-group bootstrap, seed 20261007.
"""
from statistics import NormalDist

ALPHA = 0.05
B, BOOT_SEED = 1999, 20261007
TASKS = ("income", "occ")
CLAIMS = {"A": ("J*", "C_match"), "B": ("J*", "C_global"), "C": ("P*", "T*")}
PRIMARY_SIZE = 33
Z_PRIMARY = NormalDist().inv_cdf(1 - ALPHA / (2 * PRIMARY_SIZE))
assert repr(Z_PRIMARY) == "3.1717657833516224"

PRIMARY = []
for ci, (claim, (nom, ref)) in enumerate(CLAIMS.items()):
    base = 11 * ci
    alias = claim == "B"                       # B shares the nominee J* with A: utility clauses are structural aliases
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
assert len(PRIMARY) == PRIMARY_SIZE == len({e["id"] for e in PRIMARY})


def claim_decision(claim, clause_decisions, status):
    """All 11 clauses PASS and the claim's nominee and comparator are valid NOMINEEs with a resolved configuration."""
    nom, ref = CLAIMS[claim]
    ids = [e["id"] for e in PRIMARY if e["claim"] == claim]
    all_pass = all(clause_decisions.get(i) == "PASS" for i in ids)
    req = all(status.get(x, {}).get("status") == "NOMINEE" and status.get(x, {}).get("config") for x in (nom, ref))
    return {"all_clauses_pass": all_pass, "status_requirements_met": bool(req),
            "clauses_passing": sum(clause_decisions.get(i) == "PASS" for i in ids),
            "decision": "PASS" if (all_pass and req) else "NOT_ESTABLISHED"}


def overall_label(dec, complete=None):
    """A favourable label needs its claim(s) PASS and valid; otherwise INCOMPLETE_OR_INVALID if any claim lacks
    validity/coverage (per-claim "valid", review R3), else EXPERIMENTAL_NO_ADVANTAGE (a complete valid negative)."""
    valid = {c: dec[c].get("valid", True if complete is None else complete) for c in dec}
    labels = []
    if all(dec[c]["decision"] == "PASS" and valid[c] for c in ("A", "B")):
        labels.append("JOINT_DEVELOPMENT_CRITERION_MET")
    if dec["C"]["decision"] == "PASS" and valid["C"]:
        labels.append("PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET")
    if labels:
        return " + ".join(labels)
    return "INCOMPLETE_OR_INVALID" if not all(valid.values()) else "EXPERIMENTAL_NO_ADVANTAGE"
