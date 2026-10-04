"""Registered endpoint families for the focused no-erasure study (frozen in LOCK.json before any fit).

PRIMARY: 18 slots, two nine-clause conjunctions, common two-sided Bonferroni z = Phi^-1(1 - 0.05/36).
  Claim A (PN vs LN)   P01 R_pair(LN) - R_pair(PN) > 0.02;  P02/P03 R_vi(PN) - R_vi(LN) < 0.01 (upper bound);
                       P04/P05 Acc_j(PN) - Acc_j(U) > -0.01;  P06/P07 Acc_j(PN) - 0.8 Acc_j(U) - 0.2 const_j > 0;
                       P08/P09 Acc_j(PN) - const_j > 0.03.
  Claim B (PN vs C*)   P10-P12 as P01-P03 with C*;  P13-P18 declared aliases of P04-P09.
  A PASSES only if all nine clauses pass AND on every seed the selected PN and the selected LN are nonzero-beta
  NOMINEEs (a TASK_ONLY_ALIAS on either side cannot support the coalition-coupling component claim).
  B PASSES only if all nine clauses pass AND on every seed PN is a nonzero NOMINEE and C* exists.
SECONDARY: 30 slots, separate Bonferroni family, z = Phi^-1(1 - 0.05/60). Two-sided classification:
  'lower>' rows PASS iff lower > target; 'two_sided' rows report ABOVE (lower > target), BELOW (upper < target) or
  NOT_RESOLVED.
"""
from scipy.stats import norm

ALPHA = 0.05
B, BOOT_SEED = 1999, 20261015
TASKS = ("income", "occ")

PRIMARY = []
for claim, ref in (("A", "LN"), ("B", "C*")):
    base = 0 if claim == "A" else 9
    PRIMARY += [
        {"id": f"P{base + 1:02d}", "claim": claim, "kind": "coalition", "ref": ref, "stat": f"R_pair({ref}) - R_pair(PN)",
         "target": 0.02, "side": "lower>"},
        {"id": f"P{base + 2:02d}", "claim": claim, "kind": "local", "view": "v1", "ref": ref,
         "stat": f"R_v1(PN) - R_v1({ref})", "target": 0.01, "side": "upper<"},
        {"id": f"P{base + 3:02d}", "claim": claim, "kind": "local", "view": "v2", "ref": ref,
         "stat": f"R_v2(PN) - R_v2({ref})", "target": 0.01, "side": "upper<"},
    ]
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 4 + j:02d}", "claim": claim, "kind": "acc", "task": j,
                        "stat": f"Acc_{t}(PN) - Acc_{t}(U)", "target": -0.01, "side": "lower>",
                        "alias_of": None if claim == "A" else f"P{4 + j:02d}"})
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 6 + j:02d}", "claim": claim, "kind": "retain", "task": j,
                        "stat": f"Acc_{t}(PN) - 0.8 Acc_{t}(U) - 0.2 const_{t}", "target": 0.0, "side": "lower>",
                        "alias_of": None if claim == "A" else f"P{6 + j:02d}"})
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 8 + j:02d}", "claim": claim, "kind": "useful", "task": j,
                        "stat": f"Acc_{t}(PN) - const_{t}", "target": 0.03, "side": "lower>",
                        "alias_of": None if claim == "A" else f"P{8 + j:02d}"})
PRIMARY_SIZE = 18
assert len(PRIMARY) == PRIMARY_SIZE == len({e["id"] for e in PRIMARY})

SECONDARY = []
# erasure on/off: PN vs erased JP, beta-matched, mean over beta in {0.1, 1, 10} and seeds
for j, t in enumerate(TASKS):
    SECONDARY.append({"id": f"S-erase-acc-{t}", "kind": "erase_acc", "task": j, "stat": f"Acc_{t}(PN_b) - Acc_{t}(JP_b), mean over b",
                      "target": 0.0, "side": "two_sided"})
for w in ("pair", "v1", "v2"):
    SECONDARY.append({"id": f"S-erase-rec-{w}", "kind": "erase_rec", "view": w, "stat": f"R_{w}(PN_b) - R_{w}(JP_b), mean over b",
                      "target": 0.0, "side": "two_sided"})
# output-only formats, selected PN vs selected LN
for fmt in ("prob", "hard"):
    for w in ("pair", "v1", "v2"):
        SECONDARY.append({"id": f"S-out-{fmt}-{w}", "kind": "out", "fmt": fmt, "view": w, "a": "LN", "b": "PN",
                          "stat": f"R_{fmt}_{w}(LN) - R_{fmt}_{w}(PN)", "target": 0.02 if w == "pair" else -0.01, "side": "lower>"})
# supported-race stress audit, selected PN vs selected LN
for w in ("pair", "v1", "v2"):
    SECONDARY.append({"id": f"S-race-{w}", "kind": "race", "view": w, "a": "LN", "b": "PN",
                      "stat": f"Rrace_{w}(LN) - Rrace_{w}(PN)", "target": 0.02 if w == "pair" else -0.01, "side": "lower>"})
# selected PN vs every individual control (coalition)
for x in ("LN", "U", "E", "JP", "F", "F0"):
    SECONDARY.append({"id": f"S-ctrl-{x}", "kind": "rec", "view": "pair", "a": x, "b": "PN",
                      "stat": f"R_pair({x}) - R_pair(PN)", "target": 0.02, "side": "lower>"})
# coalition minus best local, per arm (selected / reference units)
for x in ("PN", "LN", "U", "E", "JP", "F", "F0"):
    SECONDARY.append({"id": f"S-syn-{x}", "kind": "synergy", "arm": x, "stat": f"R_pair({x}) - max(R_v1({x}), R_v2({x}))",
                      "target": 0.02, "side": "lower>"})
# beta-matched PN vs LN coalition
for b in ("0.1", "1", "10"):
    SECONDARY.append({"id": f"S-bm-b{b}", "kind": "beta_matched", "beta": b, "stat": f"R_pair(LN_b{b}) - R_pair(PN_b{b})",
                      "target": 0.02, "side": "lower>"})
SECONDARY_SIZE = 30
assert len(SECONDARY) == SECONDARY_SIZE == len({e["id"] for e in SECONDARY})


def z(m):
    return float(norm.ppf(1 - ALPHA / (2 * m)))


Z_PRIMARY, Z_SECONDARY = z(PRIMARY_SIZE), z(SECONDARY_SIZE)
assert abs(Z_PRIMARY - norm.ppf(1 - 0.05 / 36)) < 1e-15 and abs(Z_SECONDARY - norm.ppf(1 - 0.05 / 60)) < 1e-15
assert round(Z_PRIMARY, 6) == 2.991316 and round(Z_SECONDARY, 6) == 3.143980
