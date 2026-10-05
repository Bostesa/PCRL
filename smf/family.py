"""Registered endpoint families (frozen in PHASE_B_PROTOCOL_LOCK and PRIMARY_FAMILY.json before any assessment).

PRIMARY: 18 slots, two nine-clause conjunctions, common two-sided Bonferroni z = Phi^-1(1 - 0.05/36) = 2.991316.
  Recovery R = SEX AUC on NEW_DEVELOPMENT_ASSESSMENT of the inner-AUC-selected final attacker (mean over attacker seeds
  0, 1, 2) on the primary views [r_i, centred logits_i] and the coalition; accuracy = deployed hard decisions; U = U e40.
  Claim A (J-F vs L-F): P01 R_pair(L-F) - R_pair(J-F) > 0.02 (lower); P02/P03 R_vi(J-F) - R_vi(L-F) < 0.01 (upper);
      P04/P05 Acc_j(J-F) - Acc_j(U) > -0.01; P06/P07 Acc_j(J-F) - 0.8 Acc_j(U) - 0.2 const_j > 0; P08/P09 gain > 0.03.
  Claim B (J-F vs C*): P10-P12 as P01-P03 with the frozen per-seed C*; P13-P18 aliases of P04-P09.
  A passes only if all nine pass, the reference is valid and J-F and L-F are NOMINEEs on all seeds; B only if all nine
  pass, the reference is valid, J-F is a NOMINEE and C* exists on all seeds. The family never shrinks.
SECONDARY: 34 slots, z = Phi^-1(1 - 0.05/68). 'lower>' rows PASS iff lower > target; 'two_sided' rows ABOVE / BELOW /
  NOT_RESOLVED. Labels '@r0.75' are the registered fixed-rho component points (Phase A NJ units at epoch 20; Phase B
  units at rho 0.75).
"""
from scipy.stats import norm

ALPHA = 0.05
B, BOOT_SEED = 1999, 20261005
TASKS = ("income", "occ")
CAND = "J-F"

PRIMARY = []
for claim, ref in (("A", "L-F"), ("B", "C*")):
    base = 0 if claim == "A" else 9
    PRIMARY += [
        {"id": f"P{base + 1:02d}", "claim": claim, "kind": "coalition", "ref": ref, "stat": f"R_pair({ref}) - R_pair(J-F)",
         "target": 0.02, "side": "lower>"},
        {"id": f"P{base + 2:02d}", "claim": claim, "kind": "local", "view": "v1", "ref": ref,
         "stat": f"R_v1(J-F) - R_v1({ref})", "target": 0.01, "side": "upper<"},
        {"id": f"P{base + 3:02d}", "claim": claim, "kind": "local", "view": "v2", "ref": ref,
         "stat": f"R_v2(J-F) - R_v2({ref})", "target": 0.01, "side": "upper<"}]
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 4 + j:02d}", "claim": claim, "kind": "acc", "task": j,
                        "stat": f"Acc_{t}(J-F) - Acc_{t}(U)", "target": -0.01, "side": "lower>",
                        "alias_of": None if claim == "A" else f"P{4 + j:02d}"})
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 6 + j:02d}", "claim": claim, "kind": "retain", "task": j,
                        "stat": f"Acc_{t}(J-F) - 0.8 Acc_{t}(U) - 0.2 const_{t}", "target": 0.0, "side": "lower>",
                        "alias_of": None if claim == "A" else f"P{6 + j:02d}"})
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 8 + j:02d}", "claim": claim, "kind": "useful", "task": j,
                        "stat": f"Acc_{t}(J-F) - const_{t}", "target": 0.03, "side": "lower>",
                        "alias_of": None if claim == "A" else f"P{8 + j:02d}"})
PRIMARY_SIZE = 18
assert len(PRIMARY) == PRIMARY_SIZE == len({e["id"] for e in PRIMARY})

SECONDARY = []
for a, b in (("A-REFRESHED@r0.75", "A-ONLINE@r0.75"), ("A-MATCHED@r0.75", "A-ONLINE@r0.75"),
             ("A-REFRESHED@r0.75", "A-MATCHED@r0.75")):                         # fixed-rho schedule contrasts
    SECONDARY.append({"id": f"S-sched-{a.split('@')[0][2:]}-vs-{b.split('@')[0][2:]}", "kind": "rec", "view": "pair",
                      "a": a, "b": b, "stat": f"R_pair({a}) - R_pair({b})", "target": 0.0, "side": "two_sided"})
SECONDARY.append({"id": "S-raw-vs-norm-pair", "kind": "rec", "view": "pair", "a": "RAW-J", "b": "J-N",
                  "stat": "R_pair(RAW-J) - R_pair(J-N)", "target": 0.0, "side": "two_sided"})
SECONDARY.append({"id": "S-raw-vs-norm-acc-occ", "kind": "accdiff", "task": 1, "a": "RAW-J", "b": "J-N",
                  "stat": "Acc_occ(RAW-J) - Acc_occ(J-N)", "target": 0.0, "side": "two_sided"})
for w in ("pair", "v1", "v2"):                                                   # feedback effect, joint, fixed rho
    SECONDARY.append({"id": f"S-fb-joint-{w}", "kind": "rec", "view": w, "a": "J-N@r0.75", "b": "J-F@r0.75",
                      "stat": f"R_{w}(J-N@0.75) - R_{w}(J-F@0.75)", "target": 0.0, "side": "two_sided"})
for w in ("v1", "v2"):                                                           # feedback effect, local, fixed rho
    SECONDARY.append({"id": f"S-fb-local-{w}", "kind": "rec", "view": w, "a": "L-N@r0.75", "b": "L-F@r0.75",
                      "stat": f"R_{w}(L-N@0.75) - R_{w}(L-F@0.75)", "target": 0.0, "side": "two_sided"})
for x in ("J-F", "L-F", "J-N", "C*"):                                            # coalition minus best local
    SECONDARY.append({"id": f"S-syn-{x}", "kind": "synergy", "arm": x,
                      "stat": f"R_pair({x}) - max(R_v1({x}), R_v2({x}))", "target": 0.0, "side": "two_sided"})
for w in ("pair", "v1", "v2"):                                                   # supported race, J-F vs L-F
    SECONDARY.append({"id": f"S-race-{w}", "kind": "race", "view": w, "a": "L-F", "b": CAND,
                      "stat": f"Rrace_{w}(L-F) - Rrace_{w}(J-F)", "target": 0.02 if w == "pair" else -0.01, "side": "lower>"})
for fmt in ("prob", "hard"):                                                     # output-only formats, J-F vs L-F
    for w in ("pair", "v1", "v2"):
        SECONDARY.append({"id": f"S-out-{fmt}-{w}", "kind": "out", "fmt": fmt, "view": w, "a": "L-F", "b": CAND,
                          "stat": f"R_{fmt}_{w}(L-F) - R_{fmt}_{w}(J-F)", "target": 0.02 if w == "pair" else -0.01,
                          "side": "lower>"})
for w in ("pair", "v1", "v2"):                                                   # proper-loss recovery, J-F vs L-F
    SECONDARY.append({"id": f"S-ll-{w}", "kind": "logloss", "view": w, "a": "L-F", "b": CAND,
                      "stat": f"LLR_{w}(L-F) - LLR_{w}(J-F)", "target": 0.0, "side": "two_sided"})
for x in ("RAW-J", "RAW-L", "J-N", "L-N", "U", "E", "F", "F0"):                  # J-F vs each control (coalition)
    SECONDARY.append({"id": f"S-ctrl-{x}", "kind": "rec", "view": "pair", "a": x, "b": CAND,
                      "stat": f"R_pair({x}) - R_pair(J-F)", "target": 0.02, "side": "lower>"})
SECONDARY_SIZE = 34
assert len(SECONDARY) == SECONDARY_SIZE == len({e["id"] for e in SECONDARY})

SCORED = ["J-F", "L-F", "J-N", "L-N", "RAW-J", "RAW-L", "U", "E", "F", "F0",
          "A-ONLINE@r0.75", "A-REFRESHED@r0.75", "A-MATCHED@r0.75", "J-F@r0.75", "L-F@r0.75", "J-N@r0.75", "L-N@r0.75"]


def z(m):
    return float(norm.ppf(1 - ALPHA / (2 * m)))


Z_PRIMARY, Z_SECONDARY = z(PRIMARY_SIZE), z(SECONDARY_SIZE)
assert abs(Z_PRIMARY - norm.ppf(1 - 0.05 / 36)) < 1e-15 and round(Z_PRIMARY, 6) == 2.991316


def claim_decision(claim, clause_decisions, seed_status):
    """Full conjunction: all nine clauses PASS and, on seeds 0-2, a valid reference and J-F NOMINEE (+ L-F NOMINEE
    for A; C* present for B). A missing slot counts as failed."""
    ids = [e["id"] for e in PRIMARY if e["claim"] == claim]
    all_nine = all(clause_decisions.get(i) == "PASS" for i in ids)
    req = set(seed_status) == {0, 1, 2}
    for k in (0, 1, 2):
        s = seed_status.get(k, {})
        req &= bool(s.get("valid_reference")) and s.get("J-F") == "NOMINEE"
        req &= (s.get("L-F") == "NOMINEE") if claim == "A" else (s.get("C*") is not None)
    return {"all_nine_pass": all_nine, "status_requirements_met": bool(req),
            "clauses_passing": sum(clause_decisions.get(i) == "PASS" for i in ids),
            "decision": "PASS" if (all_nine and req) else "NOT_ESTABLISHED"}
