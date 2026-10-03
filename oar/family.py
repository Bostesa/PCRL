"""The primary endpoint family, as a literal table (12 rows) with an executable count assertion.

R(arm, view)  = supported-class macro one-vs-rest AUC of the base nonlinear attacker (benchmark rule: GBT vs MLP on
                attacker_val log loss, refit at attacker seeds 0,1,2; plus-surfaces include ignore-rep / ignore-outputs
                candidates), averaged over attacker seeds, then encoder seeds.
Acc(arm)      = U2 probe accuracy on assessment (benchmark U2: LR grid on attacker_fit, selected on attacker_val).
F*            = FARE nominee (one per encoder seed, chosen on validation only).
B             = official target-only LEACE (benchmark maps).
Views: rep = protected features alone; rep+clean = + historical clean logits; rep+head = + logits of a head fitted
       only on the protected features (defense_fit). O_full / O_hard = outputs-only recipient (full logit vector /
       frozen-head argmax).
Each test: one-sided, simultaneous over the 12 rows (Bonferroni alpha_each = 0.05/12), group bootstrap B = 20,000.
PASS iff the simultaneous lower bound of `delta` exceeds `margin` (strict). Otherwise NOT_ESTABLISHED.
Missing / non-estimable rows are UNRESOLVED and count toward the family size.
"""

FAMILY = [
    # id, cell, question, delta (as computed), margin, kind
    ("P1-adult", "adult", "Q1 hard predictions reduce outputs-only recovery", "R(O_full) - R(O_hard)", 0.02, "superiority"),
    ("P2-adult", "adult", "Q2 FARE reduces feature recovery vs target LEACE", "R(B,rep) - R(F*,rep)", 0.02, "superiority"),
    ("P3-adult", "adult", "Q3 FARE task accuracy within 0.01 of untreated", "Acc(F*) - Acc(A)", -0.01, "non-inferiority"),
    ("P4-adult", "adult", "Q4 FARE task accuracy within 0.01 of target LEACE", "Acc(F*) - Acc(B)", -0.01, "non-inferiority"),
    ("P5-adult", "adult", "Q5 FARE+own head reduces recovery vs FARE+clean outputs", "R(F*,rep+clean) - R(F*,rep+head)", 0.02, "superiority"),
    ("P6-adult", "adult", "Q6 FARE complete release beats LEACE complete release (same contract)", "R(B,rep+head) - R(F*,rep+head)", 0.02, "superiority"),
    ("P1-hmda", "hmda", "Q1 hard predictions reduce outputs-only recovery", "R(O_full) - R(O_hard)", 0.02, "superiority"),
    ("P2-hmda", "hmda", "Q2 FARE reduces feature recovery vs target LEACE", "R(B,rep) - R(F*,rep)", 0.02, "superiority"),
    ("P3-hmda", "hmda", "Q3 FARE task accuracy within 0.01 of untreated", "Acc(F*) - Acc(A)", -0.01, "non-inferiority"),
    ("P4-hmda", "hmda", "Q4 FARE task accuracy within 0.01 of target LEACE", "Acc(F*) - Acc(B)", -0.01, "non-inferiority"),
    ("P5-hmda", "hmda", "Q5 FARE+own head reduces recovery vs FARE+clean outputs", "R(F*,rep+clean) - R(F*,rep+head)", 0.02, "superiority"),
    ("P6-hmda", "hmda", "Q6 FARE complete release beats LEACE complete release (same contract)", "R(B,rep+head) - R(F*,rep+head)", 0.02, "superiority"),
]
FAMILY_SIZE = 12
ALPHA_FAMILY = 0.05
ALPHA_EACH = ALPHA_FAMILY / FAMILY_SIZE
B_PRIMARY, SEED_PRIMARY = 20000, 20261011
B_EXPLORATORY, SEED_EXPLORATORY, LEVEL_EXPLORATORY = 2000, 20261013, 0.90
COMPETITIVE_CONJUNCTION = {"adult": ("P2-adult", "P3-adult", "P4-adult"), "hmda": ("P2-hmda", "P3-hmda", "P4-hmda")}

assert len(FAMILY) == FAMILY_SIZE == len({r[0] for r in FAMILY}), "primary family size accounting error"
assert {r[1] for r in FAMILY} == {"adult", "hmda"} and all(sum(r[1] == c for r in FAMILY) == 6 for c in ("adult", "hmda"))
