# Advisor brief — hcal (held-out, shared calibration of frozen releases)

**Bottom line.** The controlled test ran end to end and was independently reproduced. No competitive release criterion
was established (**NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED**).
- The best calibrated privacy release keeps the familiar privacy benefit.
- It still cannot show occupation log-loss preservation. The interval upper bound is 0.012, against a 0.01 limit,
  measured against both the original and the calibrated continuous teacher.
- A shared temperature beats per-token calibration but only ties the original mean decoder.
- Recommendation: close this frozen-decoder rescue line.

## Plain answers

1. **Did separate (held-out) fitting help identical codes?** Only partly.
   - On the fixed JOINT λ0.1 map, held-out per-token fitting improved income log loss by 0.0099 nats
     [0.0044, 0.0154] over teacher-row fitting.
   - Occupation log loss and both Brier scores were unresolved (FittingRole: MIXED).
   - Held-out per-token fitting is still worse than the original mean decoder on occupation log loss: 1.2866 vs 1.2766.
     It is indistinguishable from the old in-sample learned decoder.
   - Both per-token fits overfit their own fitting rows by a similar amount. With about 7–9 calibration rows per
     occupation token, the fitting role was not the binding problem.
2. **Did pooling parameters help?** Yes against per-token fitting; no against the original decoder.
   - The one-parameter temperature beat the per-token calibrator on occupation log loss by 0.0113 [0.0075, 0.0150] and
     on occupation Brier by 0.0039 [0.0024, 0.0054]. Income was unresolved (ParameterSharing: MIXED).
   - Against the original mean decoder the temperature gain is 0.0013 [−0.0002, 0.0027]: a tie.
   - This supports the shared-calibrator recipe over per-token fitting. It is not a causal claim that pooling alone
     cured overfitting: the families also differ in objective and prior.
3. **Did any release meet the original requirements?** No.
   - The nominee passed 10 of 11 original clauses:
     - pair recovery 0.0319 AUC [0.0272, 0.0366] below the strongest calibrated task-only code;
     - identical decisions;
     - every accuracy, Brier and retention clause.
   - Occupation log-loss preservation stayed unresolved: excess 0.0075, upper bound 0.0120, limit 0.01. This is
     precision, not a measured violation.
4. **Did it also hold against calibrated U?** No.
   - Calibrating U improved its occupation log loss by about 0.0017 nats, while calibrating the nominee improved it by
     about 0.0005.
   - The calibrated-reference excess was 0.0092 [0.0060, 0.0124]: unresolved. The other three calibrated clauses passed.
5. **Which privacy map and calibrator won, and what earned evidence?**
   - The winner is the existing JOINT λ0.1 map, whose partition was privacy-trained with SEX historically, with the new
     held-out class-temperature decoder. It is the same frozen partition as lra's nominee, so its privacy outcome
     repeats the earlier result on the same rows. A decoder change removes no information.
   - The new evidence is about calibration: shared temperatures generalise better than per-token tables, and none of
     them closes the occupation-confidence gap.
6. **What remains unresolved, and should this line continue?**
   - Occupation confidence preservation for privacy-trained codes is unresolved.
   - The frozen-decoder line should not continue: no κ, temperature-bound, λ, sample-split or capacity sweep, and no
     rescoring of these rows.
   - Next is an evidence-based decision about a different mechanism.

## Strongest favourable and adverse comparisons (assessment, mean over seeds)

| Comparison | Recovery | Utility |
|---|---|---|
| Favourable: P\* vs T\* (calibrated FINE-TASK) | pair AUC 0.8137 vs 0.8456; benefit 0.0319 [0.0272, 0.0366] | identical decisions; occupation LL 1.2761 vs 1.2704 |
| Favourable: shared vs per-token calibration (JOINT λ0.1) | identical (same tokens, common attack bank) | occupation LL 1.2754 vs 1.2866 (−0.0113) |
| Adverse: P\* vs calibrated U (P13) | U pair AUC 0.8585 | occupation LL 1.2761 vs 1.2669: +0.0092 [0.0060, 0.0124] |
| Adverse: held-out per-token vs original mean (JOINT λ0.1) | identical | occupation LL 1.2866 vs 1.2766 (+0.0100) |

## Losses together (JOINT λ0.1, occupation LL)

| Decoder | Own fitting rows | Calibration rows | Inner | Assessment | Parameters (seed 0) |
|---|---|---|---|---|---|
| Original mean (D0) | 1.2216 (teacher rows) | 1.3063 | 1.2747 | 1.2766 | 0 new |
| In-sample learned (D1) | 1.1977 | 1.3194 | 1.2879 | 1.2866 | per token |
| H-TOKEN32 | — | 1.2212 (its own) | 1.2832 | 1.2866 | 1,060 + 14 |
| T-TOKEN32 | 1.1570 (its own matched rows) | 1.3136 | 1.2839 | 1.2882 | 1,055 + 14 |
| H-GLOBAL-TEMP | — | 1.3022 | 1.2736 | 1.2754 | 2 |
| H-CLASS-TEMP (P\*) | — | 1.3022 | 1.2740 | 1.2761 | 6 |

- Token counts: occupation 270 tokens at seed 0, of which 212 have calibration rows; income 14.
- The calibration rows have a higher baseline loss than the inner rows for every decoder. That is a sample difference,
  not the calibrators.

## Integrity

- Registered roles, calibrators, controls, gates, the 23-slot family and the forecasts were all fixed before any fit.
- One post-lock amendment was made (A1): a NaN-blind serialisation receipt in the controls, fixed before any nomination.
- Every required real-data control passed on the common attack bank.
- Role E reproduced every inner number and all 23 slots exactly.
- Deployment and restores from a verified same-device copy passed.
- Off-device backup is pending (drive absent).
- 2.12 CPU-hours were captured by the semaphore, plus at most 0.43 uncaptured from the stopped first control attempt,
  of 20 budgeted; about 2.9 hours of wall time; $0 cloud (COST_AND_CLOSEOUT.md).
