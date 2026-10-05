"""Registered endpoint families (frozen in SELECTION_AND_AUDIT_LOCK, PRIMARY_FAMILY.json and SECONDARY_FAMILY.json
before the assessment is opened).

PRIMARY: 27 slots = three nine-clause conjunctions; common two-sided Bonferroni z = Phi^-1(1 - 0.05/54) = 3.113017.
  Recovery R = SEX AUC on OSF_DEVELOPMENT_ASSESSMENT of the inner-AUC-selected final attacker (mean over attacker seeds
  0, 1, 2) on the complete recipient views [r_i, centred logits_i] and the coalition; accuracy = deployed hard
  decisions; U = the admitted task-only model; const = OSF_DEFENSE_FIT majority class per task. Per-seed paired
  statistics are averaged over encoder seeds 0, 1, 2. Nominees and comparators are GLOBAL configurations (one per family
  across seeds), resolved from EVALUATION_LOCK.json.
  Claim A (N* vs L*): P01 R_pair(L*) - R_pair(N*) > 0.02 (lower bound); P02/P03 R_vi(N*) - R_vi(L*) < 0.01 (upper);
      P04/P05 Acc_j(N*) - Acc_j(U) > -0.01; P06/P07 Acc_j(N*) - 0.8 Acc_j(U) - 0.2 const_j > 0; P08/P09 gain > 0.03.
  Claim B (N* vs C*): P10-P12 as P01-P03 with C*; P13-P18 structural aliases of P04-P09 (same nominee).
  Claim C (R* vs L*): P19-P21 as P01-P03 with R* and L*; P22-P27 the utility clauses for R*.
  A claim passes only if all nine clauses PASS and its nominee (and comparator) are valid NOMINEEs. Descriptive
  fallbacks never pass. Strict thresholds; a tie cannot pass. Aliases are retained; the family never shrinks.
SECONDARY: the frozen ledger below (two-sided, target 0; ABOVE / BELOW / NOT_RESOLVED), z = Phi^-1(1 - 0.05/(2 m)).
  It supports individual declared contrasts only and is never an alternate route to primary success. A contrast whose
  configuration is not in the locked bank is NOT_ESTIMABLE (kept in the count).
"""
from scipy.stats import norm

ALPHA = 0.05
B, BOOT_SEED = 1999, 20261006
TASKS = ("income", "occ")
INCUMBENT = "RAW-J|b0.3"
CLAIMS = {"A": ("N*", "L*"), "B": ("N*", "C*"), "C": ("R*", "L*")}

PRIMARY = []
for ci, (claim, (nom, ref)) in enumerate(CLAIMS.items()):
    base = 9 * ci
    PRIMARY += [
        {"id": f"P{base + 1:02d}", "claim": claim, "kind": "coalition", "nominee": nom, "ref": ref,
         "stat": f"R_pair({ref}) - R_pair({nom})", "target": 0.02, "side": "lower>"},
        {"id": f"P{base + 2:02d}", "claim": claim, "kind": "local", "view": "v1", "nominee": nom, "ref": ref,
         "stat": f"R_v1({nom}) - R_v1({ref})", "target": 0.01, "side": "upper<"},
        {"id": f"P{base + 3:02d}", "claim": claim, "kind": "local", "view": "v2", "nominee": nom, "ref": ref,
         "stat": f"R_v2({nom}) - R_v2({ref})", "target": 0.01, "side": "upper<"}]
    alias = (claim == "B")
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 4 + j:02d}", "claim": claim, "kind": "acc", "task": j, "nominee": nom,
                        "stat": f"Acc_{t}({nom}) - Acc_{t}(U)", "target": -0.01, "side": "lower>",
                        "alias_of": f"P{4 + j:02d}" if alias else None})
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 6 + j:02d}", "claim": claim, "kind": "retain", "task": j, "nominee": nom,
                        "stat": f"Acc_{t}({nom}) - 0.8 Acc_{t}(U) - 0.2 const_{t}", "target": 0.0, "side": "lower>",
                        "alias_of": f"P{6 + j:02d}" if alias else None})
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 8 + j:02d}", "claim": claim, "kind": "useful", "task": j, "nominee": nom,
                        "stat": f"Acc_{t}({nom}) - const_{t}", "target": 0.03, "side": "lower>",
                        "alias_of": f"P{8 + j:02d}" if alias else None})
PRIMARY_SIZE = 27
assert len(PRIMARY) == PRIMARY_SIZE == len({e["id"] for e in PRIMARY})


def _rec(id_, a, b, view):
    return {"id": id_, "kind": "rec", "view": view, "a": a, "b": b, "stat": f"R_{view}({a}) - R_{view}({b})",
            "target": 0.0, "side": "two_sided"}


def _acc(id_, a, b, j):
    return {"id": id_, "kind": "accdiff", "task": j, "a": a, "b": b,
            "stat": f"Acc_{TASKS[j]}({a}) - Acc_{TASKS[j]}({b})", "target": 0.0, "side": "two_sided"}


def _block(tag, a, b, views=("pair", "v1", "v2"), tasks=(0, 1)):
    out = [_rec(f"S-{tag}-{w}", a, b, w) for w in views]
    out += [_acc(f"S-{tag}-acc-{TASKS[j]}", a, b, j) for j in tasks]
    return out


SECONDARY = []
SECONDARY += _block("inc-vs-U", "U", INCUMBENT)                       # recovery removed by the incumbent; task cost
SECONDARY += _block("inc-vs-RAWL", "RAW-L|b0.3", INCUMBENT)           # incumbent vs its local twin
SECONDARY += _block("norm3-vs-inc", "NORM-J|r3|a1", INCUMBENT)        # symmetric rho 3 vs the incumbent
for hi, lo in (("3", "1.5"), ("5", "3")):                              # strength along NORM-J at fixed length
    SECONDARY.append(_rec(f"S-normJ-r{hi}-vs-r{lo}-pair", f"NORM-J|r{lo}|a1", f"NORM-J|r{hi}|a1", "pair"))
    SECONDARY.append(_acc(f"S-normJ-r{hi}-vs-r{lo}-acc-occ", f"NORM-J|r{hi}|a1", f"NORM-J|r{lo}|a1", 1))
SECONDARY += _block("coupling-r3", "NORM-L|r3|a1", "NORM-J|r3|a1")    # joint vs local at identical budgets
for t in ("J", "L"):                                                   # fixed allocation vs a = 1 at rho 3
    for a in ("0.5", "2"):
        SECONDARY += _block(f"alloc-{t}-a{a}", f"NORM-{t}|r3|a1", f"NORM-{t}|r3|a{a}")
SECONDARY += _block("Nstar-vs-inc", INCUMBENT, "N*")                  # each nominee vs the fixed incumbent
SECONDARY += _block("Rstar-vs-inc", INCUMBENT, "R*")
for x in (INCUMBENT, "N*", "R*", "L*", "C*"):                          # coalition minus the better local
    SECONDARY.append({"id": f"S-syn-{x}", "kind": "synergy", "arm": x,
                      "stat": f"R_pair({x}) - max(R_v1({x}), R_v2({x}))", "target": 0.0, "side": "two_sided"})
for x in ("N*", "R*"):                                                 # proper-loss recovery vs L*
    SECONDARY.append({"id": f"S-ll-{x}-vs-Lstar", "kind": "logloss", "view": "pair", "a": "L*", "b": x,
                      "stat": f"LLR_pair(L*) - LLR_pair({x})", "target": 0.0, "side": "two_sided"})
SECONDARY_SIZE = len(SECONDARY)
assert SECONDARY_SIZE == 61 and len({e["id"] for e in SECONDARY}) == SECONDARY_SIZE


def z(m):
    return float(norm.ppf(1 - ALPHA / (2 * m)))


Z_PRIMARY, Z_SECONDARY = z(PRIMARY_SIZE), z(SECONDARY_SIZE)
assert abs(Z_PRIMARY - norm.ppf(1 - 0.05 / 54)) < 1e-15 and round(Z_PRIMARY, 6) == 3.113017


def claim_decision(claim, clause_decisions, status):
    """Full conjunction: all nine clauses PASS and the claim's nominee and comparator are valid NOMINEEs (global,
    every seed). status: {"N*": ..., "R*": ..., "L*": ..., "C*": ...} -> {"status": NOMINEE | ..., "config": id}."""
    nom, ref = CLAIMS[claim]
    ids = [e["id"] for e in PRIMARY if e["claim"] == claim]
    all_nine = all(clause_decisions.get(i) == "PASS" for i in ids)
    req = all(status.get(x, {}).get("status") == "NOMINEE" and status.get(x, {}).get("config") for x in (nom, ref))
    return {"all_nine_pass": all_nine, "status_requirements_met": bool(req),
            "clauses_passing": sum(clause_decisions.get(i) == "PASS" for i in ids),
            "decision": "PASS" if (all_nine and req) else "NOT_ESTABLISHED"}


def overall_label(dec, complete=True):
    if not complete:
        return "INCOMPLETE_OR_INVALID"
    labels = []
    if dec["A"]["decision"] == "PASS" and dec["B"]["decision"] == "PASS":
        labels.append("NORM_DEVELOPMENT_CRITERION_MET")
    if dec["C"]["decision"] == "PASS":
        labels.append("RAW_JOINT_DEVELOPMENT_CRITERION_MET")
    return " + ".join(labels) if labels else "EXPERIMENTAL_NO_ADVANTAGE"
