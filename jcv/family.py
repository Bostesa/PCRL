"""Registered endpoint families (frozen in LOCK.json before any real fit; counts and z asserted).

PRIMARY (18 slots, two nine-clause conjunctions; common two-sided Bonferroni z = Phi^-1(1 - 0.05/36)):
  claim 1, J vs L                                   claim 2, J vs C* (validation-selected strongest feasible control)
  P01 R_pair(L) - R_pair(J) > 0.02                   P10 R_pair(C*) - R_pair(J) > 0.02
  P02 R_v1(J) - R_v1(L) < 0.01  (upper bound)        P11 R_v1(J) - R_v1(C*) < 0.01
  P03 R_v2(J) - R_v2(L) < 0.01                       P12 R_v2(J) - R_v2(C*) < 0.01
  P04/P05 Acc_j(J) - Acc_j(U) > -0.01                P13-P18: declared aliases of P04-P09 (identical utility clauses)
  P06/P07 Acc_j(J) - 0.8 Acc_j(U) - 0.2 const_j > 0
  P08/P09 Acc_j(J) - const_j > 0.03
A claim PASSES only if all nine of its clauses pass AND every seed has nominees for the arms involved.
SECONDARY (30 slots, separate Bonferroni family): enumerated below.
"""
from scipy.stats import norm

ALPHA = 0.05
B, BOOT_SEED = 1999, 20261013
TASKS = ("income", "occ")

PRIMARY = []
for claim, ref in ((1, "L"), (2, "C*")):
    base = 0 if claim == 1 else 9
    PRIMARY += [
        {"id": f"P{base + 1:02d}", "claim": claim, "kind": "coalition", "stat": f"R_pair({ref}) - R_pair(J)", "ref": ref,
         "target": 0.02, "side": "lower>"},
        {"id": f"P{base + 2:02d}", "claim": claim, "kind": "local", "view": "v1", "stat": f"R_v1(J) - R_v1({ref})",
         "ref": ref, "target": 0.01, "side": "upper<"},
        {"id": f"P{base + 3:02d}", "claim": claim, "kind": "local", "view": "v2", "stat": f"R_v2(J) - R_v2({ref})",
         "ref": ref, "target": 0.01, "side": "upper<"},
    ]
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 4 + j:02d}", "claim": claim, "kind": "acc", "task": j,
                        "stat": f"Acc_{t}(J) - Acc_{t}(U)", "target": -0.01, "side": "lower>",
                        "alias_of": None if claim == 1 else f"P{4 + j:02d}"})
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 6 + j:02d}", "claim": claim, "kind": "retain", "task": j,
                        "stat": f"Acc_{t}(J) - 0.8 Acc_{t}(U) - 0.2 const_{t}", "target": 0.0, "side": "lower>",
                        "alias_of": None if claim == 1 else f"P{6 + j:02d}"})
    for j, t in enumerate(TASKS):
        PRIMARY.append({"id": f"P{base + 8 + j:02d}", "claim": claim, "kind": "useful", "task": j,
                        "stat": f"Acc_{t}(J) - const_{t}", "target": 0.03, "side": "lower>",
                        "alias_of": None if claim == 1 else f"P{8 + j:02d}"})
PRIMARY_SIZE = 18
assert len(PRIMARY) == PRIMARY_SIZE == len({e["id"] for e in PRIMARY})

SECONDARY = []
for fmt in ("prob", "hard"):          # output-only formats, J vs L
    for w in ("pair", "v1", "v2"):
        SECONDARY.append({"id": f"S-out-{fmt}-{w}", "kind": "out", "fmt": fmt, "view": w, "a": "L", "b": "J",
                          "stat": f"R_{fmt},{w}(L) - R_{fmt},{w}(J)", "target": 0.02 if w == "pair" else -0.01,
                          "side": "lower>"})
for w in ("pair", "v1", "v2"):        # race stress audit (supported classes, macro AUC), J vs L
    SECONDARY.append({"id": f"S-race-{w}", "kind": "race", "view": w, "a": "L", "b": "J",
                      "stat": f"Rrace_{w}(L) - Rrace_{w}(J)", "target": 0.02 if w == "pair" else -0.01, "side": "lower>"})
for w in ("pair", "v1", "v2"):        # projection vs penalty ablation
    SECONDARY.append({"id": f"S-proj-{w}", "kind": "rec", "view": w, "a": "JP", "b": "J",
                      "stat": f"R_{w}(JP) - R_{w}(J)", "target": 0.02 if w == "pair" else -0.01, "side": "lower>"})
for j, t in enumerate(TASKS):
    SECONDARY.append({"id": f"S-proj-acc-{t}", "kind": "accdiff", "task": j, "a": "J", "b": "JP",
                      "stat": f"Acc_{t}(J) - Acc_{t}(JP)", "target": -0.01, "side": "lower>"})
for x in ("U", "E", "JP", "S12", "S21", "F", "F0"):   # J vs every individual mandatory control (coalition)
    SECONDARY.append({"id": f"S-ctrl-{x}", "kind": "rec", "view": "pair", "a": x, "b": "J",
                      "stat": f"R_pair({x}) - R_pair(J)", "target": 0.02, "side": "lower>"})
for x in ("U", "E", "L", "J", "JP", "S12", "S21", "F", "F0"):   # does the coalition exceed both recipients?
    SECONDARY.append({"id": f"S-syn-{x}", "kind": "synergy", "arm": x,
                      "stat": f"R_pair({x}) - max(R_v1({x}), R_v2({x}))", "target": 0.02, "side": "lower>"})
SECONDARY_SIZE = 30
assert len(SECONDARY) == SECONDARY_SIZE == len({e["id"] for e in SECONDARY})


def z(m):
    return float(norm.ppf(1 - ALPHA / (2 * m)))


Z_PRIMARY, Z_SECONDARY = z(PRIMARY_SIZE), z(SECONDARY_SIZE)
assert abs(Z_PRIMARY - norm.ppf(1 - 0.05 / 36)) < 1e-15 and abs(Z_SECONDARY - norm.ppf(1 - 0.05 / 60)) < 1e-15
assert round(Z_PRIMARY, 6) == 2.991316 and round(Z_SECONDARY, 6) == 3.143980   # Phi^-1(1-0.05/36), Phi^-1(1-0.05/60)
