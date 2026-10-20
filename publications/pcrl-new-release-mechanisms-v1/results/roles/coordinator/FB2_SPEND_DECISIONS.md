# FB2 spend-only decisions

Recorded about 15:52Z on 2026-10-20 (times from `date -u`). These decisions are mechanical, from MODEL_SELECT paired costs only.

## Inputs

MODEL_SELECT paired costs from `nrm.evaluate`, pooled over versions by `nrm.rules._primary` (total NLL; loss(candidate) − loss(point)):

| Candidate | vs FB2_BASE | vs FB2_MINF_S3 | vs FB2_MINF_S5 |
|---|---|---|---|
| FB2_SEX_S3 | +0.00740 (SE 0.00068) | −0.00025 (SE 0.00085) | −0.01596 (SE 0.00143) |
| FB2_SEX_S5 | +0.02263 (SE 0.00111) | +0.01498 (SE 0.00122) | −0.00073 (SE 0.00168) |

## Decision 1: no MINF_S6

The contingent MINF_S6 is NOT triggered. Each FB2_SEX point has a MINF path point that spends at least as much:
- FB2_MINF_S3 brackets FB2_SEX_S3;
- FB2_MINF_S5 brackets FB2_SEX_S5.

So `frontier_comparison` cannot return CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT on the MINF path. The trigger rule
(DECLARATIONS_V1; reviewer C2/C3) is spend only. No MINF strength is added.

## Decision 2: CTL_REM bracketing targets (reviewer C7)

The remainder-flattening levels on FB2_BASE are to bracket the realized pooled total-NLL spends of FB2_SEX_S3 (0.0074) and FB2_SEX_S5
(0.0226) over FB2_BASE. The challenger chooses them by spend only, predicted label-only on CALIBRATION, under the CTL v2 card committed
before the CTL fit.

## Utility status (G1 inputs; adaptive)

| Configuration | Original limits | Reserve |
|---|---|---|
| FB2_BASE | pass both versions | MET 4.58 / 3.30 |
| FB2_SEX_S3 | pass both | MET 3.90 / 2.82 |
| FB2_SEX_S5 | pass both | ONE_SEED_ONLY 2.38 / 1.53 |
| FB2_MINF_S3 | pass both | MET |
| FB2_MINF_S5 | v1 FAILS occupation NLL vs Ucal (1.005 × cap) | NOT_MET |
