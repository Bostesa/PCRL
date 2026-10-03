# Correction addendum to the matched removal benchmark

**Dated 2026-10-03.**

**Corrected study.** `results/combined_matched_removal_benchmark_v1/` at
70f978ffc0afff55ecdf49cf1a74908cc3db5f49. Its original lock is 3274fe11a45f4e29f458ef7c7eff1e97df133456.

**Status of the original files.** The original files are **not modified**. They remain the historical record at that
commit, and on this branch as well. The corrected values live only in this package:
- `ORIGINAL_VS_CORRECTED.csv`, a 70-row before/after ledger;
- `MATCHED_COMPARISONS_CORRECTED.csv` and `RECOVERY_CORRECTED.csv`. These are full copies of the original tables with
  a `correction_status` column; the original values are kept beside every corrected value.

**Custody.** The 24 original primary endpoints are outside every correction below. The custody check re-ran the locked
inference from the saved predictions. See CUSTODY below for the result.

**No refits.** No defense, attacker or probe was refitted to make any correction. Every corrected number is recomputed
from saved per-row predictions with the locked bootstrap (B = 2,000, seed 20261003, resampling unit = assessment
group).

## Implementation errors (deterministic arithmetic)

### C1. Unpaired encoder seeds in a budget-truncated noise group

**Where.** Adult employment_analysis / marital_status, σ ∈ {0.5, 1, 2, 4, 8}. These are Tier-2 E3 rows.

**What went wrong.** The budget stop left encoder seed 2 unrun at these σ. The runner's paired contrasts (D − A, and
rep+outputs − outputs) subtracted an untreated mean over seeds {0, 1, 2} from a noise mean over seeds {0, 1}.

**Correction.**
- Both sides now use only the encoder seeds present on both sides: {0, 1}.
  - D side: 6 units (2 encoder seeds × 3 release draws).
  - A side: 2 units.
- Release-draw aggregation is unchanged.
- σ = 0.25 is unaffected: all 3 seeds are present on both sides, and it reproduces exactly.
- 50 rows are corrected.

| Statistic | Original | Corrected |
|---|---|---|
| D−A rep NL AUC at σ = 2 | −0.283 | −0.302 [−0.310, −0.294] |
| D−A rep+outputs NL AUC (every σ ≥ 0.5) | −0.010 | −0.028 [−0.032, −0.024] |
| D−A U2 accuracy at σ = 2 | −0.338 | −0.362 |
| "rep+outputs minus outputs-only" (every σ ≥ 0.5) | +0.014 | **0.000** |

The last row matters most: the spurious +0.014 came entirely from mismatched seeds. Once the noise destroys the
representation, the selected attacker is the outputs-only attacker, exactly as in every other cell.

The same pairing error would affect any future truncated group. The corrected rule pairs on the intersection of
encoder seeds, and the new study's verifier checks membership row by row.

### C2. ρ₁² precision loss from uncentred moments

**Where.** Adult education_assessment, arm C, seeds 1–2. That is 6 units and 9 table rows (unit level and group
level).

**What went wrong.** The saved canonical scores `RHO_u` have |mean|/sd up to 6.5 × 10⁶. Weighted moments formed before
centring cancel catastrophically. The point estimates were off by up to 1.2 % relative. Worse, the bootstrap
intervals were wrong: for s1 income the original interval [0.0010, 0.0035] excluded its own point estimate, 0.0048.

**Correction.**
- The same statistic, the weighted squared correlation of the saved (u, v), is computed in float64 after a fixed
  centring shift. The squared correlation is shift-invariant, so the definition is unchanged; no tolerance was part
  of this statistic.
- Point estimates agree with an independently written SVD calculation to 1.3 × 10⁻¹⁵ across all 435 units.
- The corrected values are tiny either way, all below 0.015. For example s1 income:

  | | Point | 90 % interval |
  |---|---|---|
  | Original | 0.004828 | [0.0010, 0.0035] |
  | Corrected | 0.004769 | [0.0024, 0.0081] |

  No conclusion changes.

## Not an error: Monte Carlo simulation spread (C3)

The original independent replay flagged 8 rows, which are 5 distinct exploratory bounds. They exceeded its frozen
6.5-MC-SE tolerance by 0.3–3.5 %.
- Against a B = 20,000 reference quantile, the runner's B = 2,000 bounds sit 1.3–3.4 SD away. That is ordinary spread
  over about 37k bound comparisons.
- They are listed in the ledger as simulation error. They are **not** "corrected", and bootstrap endpoints are not
  forced to match bitwise.
- The original primary family used B = 20,000 and reproduced exactly.

## Reporting and wording corrections (no number changed)

| Item | Original | Corrected |
|---|---|---|
| W4 | HANDOFF: "noise at σ\* costs 40 % (Adult) … of probe lift" | Adult **retains** about 40 % of the lift (0.782 vs 0.831, constant 0.749), so about 60 % is lost. HMDA retains 0 %. |
| W5 | "the linear guarantee transfers to unseen rows"; "generalises" | Measured held-out ρ₁² under target LEACE: 0.0006 (Adult; 90 % upper 0.013) and 0.0001 (HMDA; upper 0.033), against 0.025 / 0.21 untreated. This is one already-used held-out split, not a distribution-free guarantee about future people. Rank-deficient seeds leave out-of-support directions untouched. |
| W6 | "outputs carry 0.18–0.24 AUC more information than the label" | "0.18–0.24 higher measured AUC than the label-only reference" |
| W7 | "more pairs or seeds … would not change the conclusion"; "the 14 pairs agree, do not scale" | Removed. More seeds, pairs or datasets could differ. Choosing another comparison next is a prioritisation decision. |
| W8 | "few distinct values … an affine eraser cannot change a lookup" | The claim needs an injectivity argument, and a projection can merge distinct rows. Reported separately as observations: (i) assessment rows have 447 / 315 / 3,690 distinct representations of 4,778 for seeds 0 / 1 / 2; (ii) every LEACE map was **observed** to be collision-free on those rows, so the partition is identical. The earlier "6,703 / 4,742" counted all 77,408 rows. |
| W9 | "17 Adult and 42 HMDA **assessment** rows duplicate training records" | 17 / 42 **test-split** rows across all three roles. Adult 6 / 4 / 7 (attacker_fit / attacker_val / assessment); HMDA 21 / 7 / 14. PROTOCOL.md had it right. |

## Effect on the original conclusions

None of C1–C3 or W4–W9 touches a primary endpoint or decision. Specifically:
- the C1 correction strengthens the "outputs-only attacker" interpretation;
- C2 changes values below 0.015;
- W4–W9 correct how results were described, not what was measured.

The exposure sensitivity (`EXPOSURE_SENSITIVITY.md`) asks separately whether the headlines survive removing the
training-overlapping assessment groups.

## CUSTODY

The custody check re-ran the locked benchmark inference (`stored_model_eval.bench_infer.infer_bench`, Tier 1, code
unchanged) from the saved per-row predictions, with the original seeds.
- **24/24** original primary endpoints reproduce with point, lower and upper bound identical to 1e-12.
- **24/24** decisions are identical.
- Cost: 865 CPU-s, 1.9 GB peak.
- Script: `scripts/s1_custody_primary.py`.

