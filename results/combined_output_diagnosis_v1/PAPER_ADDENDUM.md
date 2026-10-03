# Paper addendum: verified tables and scoped candidate paragraphs

**Status.** For the manuscript owner. Every number below is in PRIMARY_ENDPOINTS.csv, S3_ENDPOINTS.csv,
S4_ENDPOINTS.csv, S5_ENDPOINTS.csv or FROZEN_HEAD_UTILITY.csv, and the independent replay reproduced it
(INDEPENDENT_VERIFICATION.json). All of it is development evidence on repeatedly used Adult and HMDA CA-2023 data.

## Table A: usefulness and recovery by released output

Frozen PCRL Round-4 heads, mean over 3 encoder seeds, assessment rows. Recovery is the supported-class macro AUC of the
validation-selected attacker.

| | Adult income / sex | HMDA underwriting / race |
|---|---|---|
| Accuracy gain over constant (frozen head) | 0.039 [0.031, 0.048] | 0.0156 [0.0098, 0.0214]: just below the 0.01 target |
| Balanced accuracy, by seed | 0.60 / 0.57 / 0.60 | 0.61 / 0.53 / 0.60 |
| Recovery: full logits (bank) | 0.779 | 0.769 |
| Recovery: offset removed (centred logits / probabilities) | 0.708 | 0.702 |
| Recovery: offset alone | 0.704 | 0.737 |
| Recovery: decision | 0.513 | 0.505 |
| Recovery: true task label (reference) | 0.601 | 0.522 |
| Full − offset-removed (simultaneous 95 % over 30) | +0.072 [0.062, 0.081] | +0.067 [0.060, 0.075] |
| Offset-removed − decision | +0.195 [0.177, 0.212] | +0.196 [0.185, 0.208] |

## Candidate paragraphs (scoped)

**On the decision.**

> "The decisions of the frozen purpose heads revealed little about sex or race (AUC 0.51), but they were also of
> limited use: 3–5 accuracy points above a constant guess on Adult income, and not established above 1 point on HMDA
> underwriting, where the head recovered 7–22 % of denials. Low leakage from a decision should be read together with
> what the decision conveys."

**On the logit offset.**

> "A two-class logit vector carries, besides the margin that determines every probability and decision, a common
> offset with no effect on any task quantity. For the frozen PCRL heads, an attacker recovered sex or race from the
> offset alone about as well as from the margin. Withholding it, which leaves probabilities exactly unchanged, lowered
> measured recovery by 0.07 AUC in both primary cells (pre-registered; offset-attributable on every seed). The effect
> did not appear for multiclass employment and education heads, and a refitted head with higher utility leaked as much
> through its margin alone. We therefore report it as a property of these trained heads, not as a privacy method."

**On recipients combining outputs.**

> "Two purpose recipients that combine their outputs recovered race better than either alone, under full logits,
> centred logits and hard decisions (e.g. 0.61 vs 0.55 / 0.58 for decisions). Low recovery from one recipient does
> not bound recovery from a coalition."

**On FARE** (scope it).

> "On a cell where the task is easy to keep, official FARE preserved accuracy and lowered complete-release recovery
> well below target LEACE, but a matched zero-fairness tree of the same budget achieved nearly the same reduction.
> FARE's demographic-parity certificate was unavailable or vacuous at our certification sample sizes."

**Do not write:**
- "Hard labels protect privacy."
- "Removing the offset is a privacy guarantee."
- "FARE outperforms LEACE" (without the compression control).
- Any population or mutual-information claim from attacker AUC.

## Corrections to carry over

These are listed in ORIGINAL_VS_REPAIRED.csv and are reporting-only:
- R1: the role-label truncation; exposure exclusions are 17 Adult / 42 HMDA rows.
- R2: feature-equality counts give rows and distinct vectors separately (HMDA s1: 1,138 / 1,385 cert rows share one
  vector).
- R3: the HMDA P3 / P4 lower bounds are −0.0062 / −0.0057.
