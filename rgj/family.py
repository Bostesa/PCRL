"""Registered endpoint families (frozen in CODE_LOCK.json and PRIMARY_FAMILY.json before any nonzero fit).

PRIMARY: 18 slots, two nine-clause conjunctions, common two-sided Bonferroni z = Phi^-1(1 - 0.05/(2*18)) = 2.991316.
  Recovery R = SEX AUC on DEVELOPMENT_ASSESSMENT of the inner-AUC-selected final attacker (mean over attacker seeds
  0, 1, 2), primary views [r_i, centred logits_i] and the coalition [v1, v2]. Accuracy = deployed hard decisions.
  Claim A (J-G vs L-G)  P01 R_pair(L-G) - R_pair(J-G) > 0.02 (lower bound);  P02/P03 R_vi(J-G) - R_vi(L-G) < 0.01
                        (upper bound);  P04/P05 Acc_j(J-G) - Acc_j(U) > -0.01;  P06/P07 Acc_j(J-G) - 0.8 Acc_j(U) -
                        0.2 const_j > 0;  P08/P09 Acc_j(J-G) - const_j > 0.03.
  Claim B (J-G vs C*)   P10-P12 as P01-P03 with the frozen per-seed C*;  P13-P18 declared aliases of P04-P09.
  A passes only if all nine clauses pass, L-R is a valid reference and J-G and L-G are nontrivial NOMINEEs on all three
  seeds. B passes only if all nine pass, L-R is valid, J-G is a NOMINEE and C* exists on all three seeds. Aliases and
  missing comparators never shrink the family.
SECONDARY: 30 slots, separate Bonferroni family z = Phi^-1(1 - 0.05/60) = 3.143980. 'lower>' rows PASS iff lower >
  target; 'two_sided' rows are ABOVE / BELOW / NOT_RESOLVED. Unenumerated comparisons are descriptive only.
"""
from scipy.stats import norm

ALPHA = 0.05
B, BOOT_SEED = 1999, 20261004
TASKS = ("income", "occ")
CAND = "J-G"

PRIMARY = []
for claim, ref in (("A", "L-G"), ("B", "C*")):
    base = 0 if claim == "A" else 9
    PRIMARY += [
        {"id": f"P{base + 1:02d}", "claim": claim, "kind": "coalition", "ref": ref,
         "stat": f"R_pair({ref}) - R_pair(J-G)", "target": 0.02, "side": "lower>"},
        {"id": f"P{base + 2:02d}", "claim": claim, "kind": "local", "view": "v1", "ref": ref,
         "stat": f"R_v1(J-G) - R_v1({ref})", "target": 0.01, "side": "upper<"},
        {"id": f"P{base + 3:02d}", "claim": claim, "kind": "local", "view": "v2", "ref": ref,
         "stat": f"R_v2(J-G) - R_v2({ref})", "target": 0.01, "side": "upper<"},
    ]
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 4 + j:02d}", "claim": claim, "kind": "acc", "task": j,
                        "stat": f"Acc_{t}(J-G) - Acc_{t}(U)", "target": -0.01, "side": "lower>",
                        "alias_of": None if claim == "A" else f"P{4 + j:02d}"})
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 6 + j:02d}", "claim": claim, "kind": "retain", "task": j,
                        "stat": f"Acc_{t}(J-G) - 0.8 Acc_{t}(U) - 0.2 const_{t}", "target": 0.0, "side": "lower>",
                        "alias_of": None if claim == "A" else f"P{6 + j:02d}"})
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 8 + j:02d}", "claim": claim, "kind": "useful", "task": j,
                        "stat": f"Acc_{t}(J-G) - const_{t}", "target": 0.03, "side": "lower>",
                        "alias_of": None if claim == "A" else f"P{8 + j:02d}"})
PRIMARY_SIZE = 18
assert len(PRIMARY) == PRIMARY_SIZE == len({e["id"] for e in PRIMARY})

SECONDARY = []
for fmt in ("prob", "hard"):                       # output-only formats, J-G vs L-G
    for w in ("pair", "v1", "v2"):
        SECONDARY.append({"id": f"S-out-{fmt}-{w}", "kind": "out", "fmt": fmt, "view": w, "a": "L-G", "b": CAND,
                          "stat": f"R_{fmt}_{w}(L-G) - R_{fmt}_{w}(J-G)", "target": 0.02 if w == "pair" else -0.01,
                          "side": "lower>"})
for w in ("pair", "v1", "v2"):                     # supported-race stress audit, J-G vs L-G
    SECONDARY.append({"id": f"S-race-{w}", "kind": "race", "view": w, "a": "L-G", "b": CAND,
                      "stat": f"Rrace_{w}(L-G) - Rrace_{w}(J-G)", "target": 0.02 if w == "pair" else -0.01,
                      "side": "lower>"})
for w in ("pair", "v1", "v2"):                     # proper-loss recovery (CE-selected attacker), J-G vs L-G
    SECONDARY.append({"id": f"S-ll-{w}", "kind": "logloss", "view": w, "a": "L-G", "b": CAND,
                      "stat": f"LLR_{w}(L-G) - LLR_{w}(J-G)", "target": 0.0, "side": "two_sided"})
for w in ("pair", "v1", "v2"):                     # fixed-beta local-feedback component (beta 0.1, epoch 20)
    SECONDARY.append({"id": f"S-fb-{w}", "kind": "fixed", "view": w, "a": "J-R@fx", "b": "J-G@fx",
                      "stat": f"R_{w}(J-R_b0.1_e20) - R_{w}(J-G_b0.1_e20)", "target": 0.0, "side": "two_sided"})
for w in ("pair", "v1", "v2"):                     # fixed-beta refresh component (beta 0.1, epoch 20)
    SECONDARY.append({"id": f"S-rf-{w}", "kind": "fixed", "view": w, "a": "J-O@fx", "b": "J-R@fx",
                      "stat": f"R_{w}(J-O_b0.1_e20) - R_{w}(J-R_b0.1_e20)", "target": 0.0, "side": "two_sided"})
for x in ("L-R", "L-O", "J-R", "J-O", "U", "E", "F", "F0"):   # coalition, J-G vs each control (selected/reference)
    SECONDARY.append({"id": f"S-ctrl-{x}", "kind": "rec", "view": "pair", "a": x, "b": CAND,
                      "stat": f"R_pair({x}) - R_pair(J-G)", "target": 0.02, "side": "lower>"})
for x in ("J-G", "L-G", "J-R", "U"):              # coalition minus best local recovery
    SECONDARY.append({"id": f"S-syn-{x}", "kind": "synergy", "arm": x,
                      "stat": f"R_pair({x}) - max(R_v1({x}), R_v2({x}))", "target": 0.0, "side": "two_sided"})
SECONDARY_SIZE = 30
assert len(SECONDARY) == SECONDARY_SIZE == len({e["id"] for e in SECONDARY})

# scored labels per seed (EVALUATION_LOCK resolves each to a frozen unit); C* resolves to one of them
SCORED = ["J-G", "L-G", "J-R", "J-O", "L-R", "L-O", "U", "E", "F", "F0", "J-G@fx", "J-R@fx", "J-O@fx"]


def z(m):
    return float(norm.ppf(1 - ALPHA / (2 * m)))


Z_PRIMARY, Z_SECONDARY = z(PRIMARY_SIZE), z(SECONDARY_SIZE)
assert abs(Z_PRIMARY - norm.ppf(1 - 0.05 / 36)) < 1e-15 and abs(Z_SECONDARY - norm.ppf(1 - 0.05 / 60)) < 1e-15
assert round(Z_PRIMARY, 6) == 2.991316 and round(Z_SECONDARY, 6) == 3.143980


def claim_decision(claim, clause_decisions, seed_status):
    """Full-conjunction rule. clause_decisions: {slot id: 'PASS' | other} for the claim's nine slots (a missing slot or
    None counts as failed). seed_status: {seed: {"valid_reference": bool, "J-G": status, "L-G": status,
    "C*": status or None}} for seeds 0, 1, 2. Statuses: NOMINEE (nontrivial, feasible) is the only accepted value for
    J-G and L-G; C* must exist (any feasible control, including a task-only one, which is disclosed)."""
    ids = [e["id"] for e in PRIMARY if e["claim"] == claim]
    all_nine = all(clause_decisions.get(i) == "PASS" for i in ids)
    seeds_ok = set(seed_status) == {0, 1, 2}
    req = seeds_ok
    for k in (0, 1, 2):
        s = seed_status.get(k, {})
        req &= bool(s.get("valid_reference")) and s.get("J-G") == "NOMINEE"
        if claim == "A":
            req &= s.get("L-G") == "NOMINEE"
        else:
            req &= s.get("C*") is not None
    return {"all_nine_pass": all_nine, "status_requirements_met": bool(req),
            "clauses_passing": sum(clause_decisions.get(i) == "PASS" for i in ids),
            "decision": "PASS" if (all_nine and req) else "NOT_ESTABLISHED"}
