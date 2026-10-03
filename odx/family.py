"""Literal endpoint families with executable count assertions (frozen in LOCK.json before any new fit)."""
from math import comb

from scipy.stats import norm

HMDA_RACE_GROUPS = (0, 1, 2, 3, 4)
HMDA_PAIRS = [(i, j) for i in HMDA_RACE_GROUPS for j in HMDA_RACE_GROUPS if i < j]
assert len(HMDA_PAIRS) == comb(5, 2) == 10

PRIMARY = []
for ds in ("adult", "hmda"):
    PRIMARY.append({"id": f"U-frozen-{ds}", "block": "usefulness", "dataset": ds, "stat": "Acc(frozen head) - Acc(const)", "target": 0.01})
    PRIMARY.append({"id": f"U-refit-{ds}", "block": "usefulness", "dataset": ds, "stat": "Acc(U2__A probe) - Acc(const)", "target": 0.01})
for ds in ("adult", "hmda"):
    PRIMARY.append({"id": f"FC-{ds}", "block": "macro", "dataset": ds, "stat": "R(full bank) - R(ignore-offset bank)", "target": 0.02})
    PRIMARY.append({"id": f"CH-{ds}", "block": "macro", "dataset": ds, "stat": "R(ignore-offset bank) - R(hard)", "target": 0.02})
for contrast in ("FC", "CH"):
    PRIMARY.append({"id": f"{contrast}-adult-pair0-1", "block": "pair", "dataset": "adult", "pair": [0, 1],
                    "stat": f"{contrast} pair AUC", "target": 0.02})
    for i, j in HMDA_PAIRS:
        PRIMARY.append({"id": f"{contrast}-hmda-pair{i}-{j}", "block": "pair", "dataset": "hmda", "pair": [i, j],
                        "stat": f"{contrast} pair AUC", "target": 0.02})
PRIMARY_SIZE = 30
assert len(PRIMARY) == PRIMARY_SIZE == 4 + 4 + 2 * (1 + 10) == len({e["id"] for e in PRIMARY})
# declared before any fit: unsupported HMDA race groups 3 (95 attacker_fit rows) and 4 (53) -> 14 NOT_ESTIMABLE slots
EXPECTED_NOT_ESTIMABLE = sorted(f"{c}-hmda-pair{i}-{j}" for c in ("FC", "CH") for (i, j) in HMDA_PAIRS if 3 in (i, j) or 4 in (i, j))
assert len(EXPECTED_NOT_ESTIMABLE) == 14
# declared exact aliases (binary attribute: pair AUC (0,1) == macro AUC); kept in the count, verified equal, not extra support
DECLARED_ALIASES = {"FC-adult-pair0-1": "FC-adult", "CH-adult-pair0-1": "CH-adult"}

S3_PAIRS = 14
S3 = [{"id": f"S3-{c}-{i}", "block": "replication"} for i in range(S3_PAIRS) for c in ("FC", "CH")] + \
     [{"id": f"S3-U-{p}", "block": "usefulness"} for p in range(6)]
S3_SIZE = 34
assert len(S3) == S3_SIZE == 2 * 14 + 6
S4 = [{"id": f"S4-{c}-pair-minus-{who}", "contract": c, "vs": who} for c in ("full", "centred", "hard")
      for who in ("income_prediction", "employment_analysis")]
S4_SIZE = 6
assert len(S4) == S4_SIZE
S5 = ["S5-LEACE-minus-FARE-rep+head", "S5-FZ-minus-FARE-rep+head", "S5-Acc(FARE)-Acc(A)", "S5-retained-gain-share"]
S5_SIZE = 4
assert len(S5) == S5_SIZE

ALPHA = 0.05
B_SE, SEED_SE = 1999, 20261021


def z_two_sided(m: int) -> float:
    """Bonferroni normal critical value for m two-sided 95 % intervals."""
    return float(norm.ppf(1 - ALPHA / (2 * m)))


Z_PRIMARY = z_two_sided(PRIMARY_SIZE)      # 3.14398 = Phi^-1(1 - 0.05/60)
assert round(Z_PRIMARY, 6) == 3.143980
assert round(z_two_sided(S3_SIZE), 6) == 3.180426 and round(z_two_sided(S4_SIZE), 6) == 2.638257 and round(z_two_sided(S5_SIZE), 6) == 2.497705
