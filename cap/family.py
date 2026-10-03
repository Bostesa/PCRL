"""Literal endpoint families (frozen in LOCK.json before any new fit), with executable count assertions."""
from scipy.stats import norm

ARMS = ("B", "F", "F0")
ALPHA = 0.05
B_SE, SEED_SE = 1999, 20261041

PRIMARY = []
for arm in ARMS:
    PRIMARY += [
        {"id": f"U-acc-{arm}-vs-A", "kind": "utility", "arm": arm, "stat": f"Acc({arm}) - Acc(A)", "target": -0.01},
        {"id": f"U-gain-{arm}", "kind": "utility", "arm": arm, "stat": f"Acc({arm}) - Acc(const)", "target": 0.03},
        {"id": f"U-retain-{arm}", "kind": "utility", "arm": arm, "stat": f"Acc({arm}) - 0.8 Acc(A) - 0.2 Acc(const)", "target": 0.0},
    ]
CONTRASTS = [("A", "B"), ("A", "F"), ("A", "F0"), ("B", "F"), ("F0", "F")]
for contract in ("centred", "hard"):
    for hi, lo in CONTRASTS:
        PRIMARY.append({"id": f"R-out-{contract}-{hi}-minus-{lo}", "kind": "recovery", "contract": f"output-only/{contract}",
                        "stat": f"R({hi}) - R({lo})", "hi": hi, "lo": lo, "target": 0.02})
PRIMARY_SIZE = 19
assert len(PRIMARY) == PRIMARY_SIZE == 9 + 2 * 5 == len({e["id"] for e in PRIMARY})

SECONDARY = []
for hi, lo in CONTRASTS:
    SECONDARY.append({"id": f"S-out-full-{hi}-minus-{lo}", "contract": "output-only/full", "hi": hi, "lo": lo, "target": 0.02})
for c in ("full", "centred", "hard"):
    for hi, lo in CONTRASTS:
        SECONDARY.append({"id": f"S-feat+out-{c}-{hi}-minus-{lo}", "contract": f"features+own-output/{c}", "hi": hi, "lo": lo, "target": 0.02})
for hi, lo in CONTRASTS:
    SECONDARY.append({"id": f"S-feat-{hi}-minus-{lo}", "contract": "features-only", "hi": hi, "lo": lo, "target": 0.02})
for arm in ("A",) + ARMS:
    SECONDARY.append({"id": f"S-offset-out-{arm}", "contract": "output-only", "arm": arm, "stat": "R(full bank) - R(ignore-offset bank)", "target": 0.02})
for arm in ("A",) + ARMS:
    SECONDARY.append({"id": f"S-bypass-{arm}", "contract": "features+historical clean output vs features+own output (adverse control)",
                      "arm": arm, "stat": "R(feat+clean) - R(feat+own full)", "target": 0.02})
SECONDARY_SIZE = 33
assert len(SECONDARY) == SECONDARY_SIZE == 5 + 15 + 5 + 4 + 4 == len({e["id"] for e in SECONDARY})


def z(m):
    return float(norm.ppf(1 - ALPHA / (2 * m)))


Z_PRIMARY, Z_SECONDARY = z(PRIMARY_SIZE), z(SECONDARY_SIZE)
assert round(Z_PRIMARY, 6) == 3.007787 and round(Z_SECONDARY, 6) == 3.171766   # Phi^-1(1-0.05/38), Phi^-1(1-0.05/66)
