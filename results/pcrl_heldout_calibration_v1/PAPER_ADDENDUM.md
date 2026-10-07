# Paper addendum — hcal (for manuscript owners)

**Status:** NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED
[OriginalCriterion NOT_ESTABLISHED; CalibrationMatchedCriterion NOT_ESTABLISHED; FittingRole MIXED;
ParameterSharing MIXED]. This is exploratory development evidence on repeatedly used Adult rows; the intervals condition
on fitted artifacts and the selection history (EXPOSURE_LEDGER.md).

## Allowed summary (drop-in)

> On the Adult development benchmark, we held the teachers, partitions and token assignments fixed and refitted only the
> released probability tables, using 2,000 calibration rows held out from teacher fitting and head selection. We compared
> a per-token proper-loss calibrator (κ = 32 teacher prior) with shared temperature scaling (Guo et al., 2017). The
> controls were calibrated equally, including the continuous teacher. Every decoder variant of a code faced the same
> attack predictions.
>
> Held-out per-token fitting improved income log loss over matched teacher-row fitting (0.010 nats [0.004, 0.015];
> family-wise interval), but the occupation contrasts were unresolved. With 7–9 calibration rows per occupation token it
> remained worse than the original mean decoder. A one-parameter temperature beat per-token fitting on occupation log
> loss (0.011 [0.008, 0.015]) and Brier, but only tied the original decoder.
>
> The inner-nominated privacy release was an existing privacy-trained map with a held-out class-temperature decoder. It
> reduced pair attribute recovery by 0.032 AUC [0.027, 0.037] below the strongest calibrated task-only code, with
> identical decisions. Its occupation log-loss preservation remained unresolved against both the original teacher
> (excess 0.0075, upper bound 0.0120 against 0.01) and the calibrated teacher (0.0092, upper bound 0.0124), so neither
> release criterion was established.

**Scope.**
- A decoder-only change is not information removal.
- The privacy benefit is the same frozen map's.
- Temperature scaling is established prior work, not a new algorithm.
- This is a development result, not confirmation.
- Fewer leaking finite attackers is not a population privacy guarantee.

## Supporting facts

| Fact | Value | Source |
|---|---|---|
| Launch gate | ENGINEERING_READY (130/130 synthetic checks), reviewed by B, D, E | ENGINEERING_RESULT.json; MATH_REVIEW.md; verification/E_ENGINEERING_REVIEW.md |
| Nominee / comparator / Ucal\* | JOINT λ0.1 H-CLASS-TEMP / FINE-TASK H-GLOBAL-TEMP / U H-GLOBAL-TEMP | SELECTION.json |
| Pair benefit (P01) | 0.0319 [0.0272, 0.0366] PASS | PRIMARY_ENDPOINTS.csv |
| Occupation LL vs U0 (P07) | 0.0075 [0.0031, 0.0120] NOT_ESTABLISHED_PRECISION | PRIMARY_ENDPOINTS.csv |
| Occupation LL vs calibrated U (P13) | 0.0092 [0.0060, 0.0124] NOT_ESTABLISHED_PRECISION | PRIMARY_ENDPOINTS.csv |
| Fitting role, JOINT λ0.1 (D01–D04) | income LL +0.0099 [0.0044, 0.0154]; three unresolved | PRIMARY_ENDPOINTS.csv |
| Parameter sharing, JOINT λ0.1 (D05–D08) | occupation LL +0.0113 [0.0075, 0.0150], occupation Brier +0.0039 [0.0024, 0.0054]; income unresolved | PRIMARY_ENDPOINTS.csv |
| Shared temperature vs original decoder | 0.0013 [−0.0002, 0.0027] (nominal 95%, supplementary) | run inference; ALL_LEVELS.csv |
| Independent reproduction | inner replay PASS; phase 3 PASS (23/23 slots exact) | verification/INDEPENDENT_VERIFICATION.json |

## May be said only with its qualifier

- **"A shared temperature beat per-token calibration."**
  - Qualifier: on occupation log loss and Brier on this development assessment.
  - It did not beat the original mean decoder.
- **"Held-out fitting helped."**
  - Qualifier: on income log loss only.
  - The occupation contrasts were unresolved.

## Do not write

- "calibration fixes confidence", "the method works", or "a release meeting the criterion";
- "calibration reduces information" or "calibration improves privacy";
- "held-out fitting solves overfitting", "pooling alone cured overfitting", or "calibration can never help";
- "new calibration algorithm" (temperature scaling is Guo et al., 2017);
- "label-blind" for the nominee: its partition was privacy-trained with SEX;
- "joint beats sequential";
- "confirmed", or "population privacy guarantee";
- any R²-to-accuracy guarantee (refuted; never used).
