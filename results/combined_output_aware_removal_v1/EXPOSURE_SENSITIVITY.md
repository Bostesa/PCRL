# Exposure sensitivity of the matched removal benchmark

**Post hoc.** This is a sensitivity analysis on saved predictors. It does not replace the original analysis or serve
as a confirmation.

**Rule.** `exposure/EXPOSURE_RULE.md`, committed at 7a4a086 before any sensitivity result was computed.

**Machine-readable table.** `EXPOSURE_ENDPOINTS.csv`, 164 rows.

## Overlap inventory, all roles and seeds

The record-matching rule is the benchmark's `canon_key`: Adult is the raw-row hash with the adult.test trailing "."
removed; HMDA is a hash over 17 LAR columns. One seeded split is shared by all encoder seeds, so the overlap is
identical for seeds 0, 1 and 2.

| Overlap | Adult | HMDA |
|---|---|---|
| defense_fit vs any attacker role or assessment | 0 | 0 |
| attacker_fit vs attacker_val vs assessment | 0 | 0 |
| Test-split rows with a record equal to an encoder-training record (attacker_fit / attacker_val / assessment) | 17 (6 / 4 / 7) | 42 (21 / 7 / 14) |
| Train rows already excluded by the benchmark (`excluded_dup`) | 18 | 43 |

Repeated records are not proven identical people. HMDA has no applicant identifier.

**Limit.** The encoder's validation-split rows are not in the admitted arrays, so overlap of test roles with the
validation split is not measured. The Round-4 `final.pt` was not selected on validation.

**Additional inventory, found after the rule was fixed and not part of it.** Some rows have a feature vector, and
therefore a representation, byte-identical to a defense_fit row, but a *different* record key: they differ in at least
one other column, such as the label. Counts in the new study's roles, from the seed-0 representation:

| Role | Adult | HMDA |
|---|---|---|
| cert | 8 | 6 |
| attacker_fit | 33 | 18 |
| attacker_val | 13 | 12 |
| assessment | 34 | 25 |

These are reported, not removed: removing them now would be a post hoc rule chosen after a first sensitivity was
seen. They are also why the FARE wrapper's feature-hash guard refused certificates (amendment A1).

## Sensitivity

**Setup.**
- Removed: the training-overlapping **assessment** groups, 7 Adult rows and 14 HMDA rows.
- Kept fixed: every encoder, LEACE map, noise draw, attacker, probe and reference. Nothing was refitted.
- Recomputed: all 435 benchmark units, from saved per-row predictions on the retained assessment rows. The locked
  inference was used unchanged, with a row filter applied at load time.
- Same B and seeds as the original. Identical retained rows on both sides of every comparison.
- Support was recomputed per class on the retained rows. **No class or pair lost support.** For example,
  HMDA race class 2 goes from 1,247 to 1,240 assessment rows, and Adult sex from 1,744 / 3,506 to 1,742 / 3,501.

| Item | Result |
|---|---|
| **24 original primary endpoints** | **24/24 decisions STABLE.** The largest point shift is 0.00044. Examples: P-adult-Rrep-A 0.8154 → 0.8153; P-hmda-G1-A 0.0626 → 0.0625 (still UNRESOLVED); P-adult-U2NI-C −0.0208 → −0.0208 (still INFERIOR). |
| **Native N0 check, 42 untreated pair × seed** | Recomputed with all 17 / 42 overlapping test rows removed from every role. **42/42 pass/fail categories STABLE**, including the 12 failures. |
| **Exploratory headline rows (90 %)** | 97/98 STABLE. These cover untreated nonlinear recovery, B−A and C−A nonlinear AUC, B−A U2, outputs-only minus label-only, and ρ₁² under A and B, on all 14 pairs. |
| **The one change** | Adult income/sex B−A nonlinear AUC: +0.0020 [−0.00002, 0.0039] → +0.0020 [+0.00001, 0.0039]. The lower bound moved from just below 0 to just above. Labelled UNRESOLVED-sensitive: the substantive reading, that LEACE barely changes nonlinear recovery, is unchanged. |

**Cost.** 2,524 s wall, 2,506 CPU-s, 2.7 GB peak.

**Verdict.** Every original headline that this sensitivity can test is stable under removing the training-overlapping
assessment groups.
