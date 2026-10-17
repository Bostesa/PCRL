# Family C: SEX-mixture marginalization (MIX), investigator report

Written by investigator C (a separate agent), 2026-10-17, 17:04Z–17:25Z. The harness did not let the subagent write report files, so the coordinator saved this text verbatim from its hand-back.

This is exploratory development on previously used public-use Florida ACS 2019 development records, not confirmation.
- The service predicts income class and occupation group from AGEP, WKHP, COW, SCHL, MAR and POBP (the 145-column admitted encoding).
- Readers try to infer the recorded SEX column (0 = male, 1 = female) from the released probability vectors.
- All numbers are MODEL_SELECT development proxies from `amd.evaluate` (household halves). They are not fresh confirmation and not population-optimal.

## What I built (`amd/fam_c.py`, sha256 4502ec4c0e5f49da4744d51b0de032c4c62e22067b9af20c54395968bf0fca27)

- **C1_MIX_FULL (initial version).**
  - The T_TREE recipe from `acs_replication/stage_a.py`, with SEX appended as a 146th input column at training time only.
  - Settings: depth 4, learning rate 0.05, at most 800 trees, l2 3, `thread_count=1`, the T_TREE CatBoost seeds, the same 15% household-disjoint internal early-stopping split, then a refit on all 20,000 MODEL_FIT people.
  - Release on all six roles: w0·P(y|x,SEX=0) + w1·P(y|x,SEX=1), with w = the MODEL_FIT SEX shares 0.5103 / 0.4897. SEX is never a runtime input. MODEL_FIT outputs are in-sample.
  - Check: the identical recipe without SEX reproduced the stored T_TREE bitwise on both seeds and every role, with equal teacher hashes. T_TREE is therefore the exact SEX-blind control.
  - Check: the SEX-aware numpy teachers match CatBoost bitwise on routing and to ≤ 1e-12 on probabilities, on every role, with SEX set to 0 and to 1.
- **C1_CTRL_TEMP (the card's control).** The SEX-blind T_TREE with one temperature per task. The temperatures were set, without labels, so that the release's average divergence from T_TREE on MODEL_FIT matches C1's: income 0.0013 nats, occupation 0.0064 nats. The temperatures are income 1.1237 / 1.1208 and occupation 1.1820 / 1.1835.
- **C2_MIXINC_JOCC (revision 1).** Income is C1's marginalized income. Occupation is the frozen JOINT_TREE E20 output, which is bitwise identical to H_LJ's occupation output, so H_LJ is the matched component control.
- **CDIAG_TTINC_JOCC (diagnostic, counted 0).** T_TREE income paired with the same JOINT occupation output, used for attribution.

## Results (MODEL_SELECT, seed 0 / seed 1)

Promotion bars:

| Bar | Seed 0 | Seed 1 |
|---|---|---|
| Pair proxy | < 0.6224 | < 0.6233 |
| Income proxy | ≤ 0.5500 | ≤ 0.5476 |
| Occupation proxy | ≤ 0.5824 | ≤ 0.5953 |
| Guard (each SEX group) | ≤ 0.005 | ≤ 0.005 |

The proxy is the maximum over OUT_LR, OUT_HGB and LOG_MLP, plus LOOKUP for the pair.

| Configuration (seed) | Clauses | Worst normalized | Guard, men | Guard, women | Pair | Income | Occupation |
|---|---|---|---|---|---|---|---|
| C1_MIX_FULL (0) | 12/12 | −0.189 | 0.00439 | −0.00797 | 0.6263 | 0.5479 | 0.6211 |
| C1_MIX_FULL (1) | 12/12 | 0.097 | **0.00640 (fails)** | −0.00723 | 0.6368 | 0.5461 | 0.6286 |
| C1_CTRL_TEMP (0) | 12/12 | −0.071 | −0.00191 | −0.00035 | 0.6489 | 0.5573 | 0.6412 |
| C1_CTRL_TEMP (1) | 12/12 | 0.225 | −0.00029 | 0.00103 | 0.6496 | 0.5589 | 0.6366 |
| C2_MIXINC_JOCC (0) | 12/12 | 0.341 | 0.00442 | −0.00494 | 0.6277 | 0.5479 | 0.5724 |
| CDIAG_TTINC_JOCC (0), diagnostic | 12/12 | 0.341 | 0.00442 | −0.00494 | 0.6351 | 0.5573 | 0.5724 |
| T_TREE reference (0 / 1) | 12/12 | — | −0.00103 / 0.00059 | −0.00931 / −0.00788 | 0.6487 / 0.6493 | 0.5573 / 0.5589 | 0.6426 / 0.6391 |
| NEURAL_RAW reference (0 / 1) | 12/12 | — | 0.00147 / 0.00198 | −0.00099 / −0.00133 | 0.6238 / 0.6269 | 0.5400 / 0.5376 | 0.6211 / 0.6163 |
| H_LJ reference (0 / 1) | 12/12 | — | 0.00442 / 0.00394 | −0.00494 / −0.00304 | 0.6224 / 0.6233 | 0.5423 / 0.5439 | 0.5724 / 0.5853 |

Utility vector, excess against U0 / against Ucal. The caps are accuracy 0.010, NLL 0.010 and Brier 0.005.

| Configuration (seed) | Income accuracy | Income NLL | Income Brier | Occupation accuracy | Occupation NLL | Occupation Brier |
|---|---|---|---|---|---|---|
| C1 (0) | −.00325 / −.00325 | −.00644 / −.00534 | −.00403 / −.00359 | −.00225 / −.00225 | −.00277 / −.00189 | −.00172 / −.00139 |
| C1 (1) | −.00075 / −.00075 | −.00594 / −.00567 | −.00381 / −.00376 | +.00075 / +.00075 | +.00092 / +.00097 | −.00034 / −.00033 |
| CTRL (0) | −.00125 / −.00125 | −.00460 / −.00350 | −.00291 / −.00247 | −.00637 / −.00637 | −.00159 / −.00071 | −.00114 / −.00080 |
| CTRL (1) | −.00062 / −.00062 | −.00457 / −.00429 | −.00315 / −.00310 | −.00612 / −.00612 | +.00221 / +.00225 | +.00037 / +.00037 |
| C2 (0) | −.00325 / −.00325 | −.00644 / −.00534 | −.00403 / −.00359 | −.00200 / −.00200 | +.00253 / +.00341 | −.00021 / +.00012 |

For comparison, T_TREE's NLL excess against U0 is income −0.00663 / −0.00682 and occupation −0.01213 / −0.00825.

## Diagnosis

1. **C1 removes real SEX structure; it is not just weaker readers.**
   - Reader-by-reader change against T_TREE (OUT_LR / OUT_HGB / LOG_MLP):

     | View | Seed 0 | Seed 1 |
     |---|---|---|
     | Pair | −.022 / −.022 / −.028 | −.023 / −.012 / −.013 |
     | Occupation | −.033 / −.020 / −.022 | −.033 / −.012 / −.011 |
     | Income | −.009 / −.016 / −.009 | −.013 / **+.011** / −.013 |

     The one increase, seed-1 income OUT_HGB, is a reader that already under-reads T_TREE (0.526, against OUT_LR's 0.559).
   - The matched-cost temperature control changes every reader by at most ±0.0025 on both seeds. Its MODEL_SELECT occupation NLL cost is no smaller than C1's: +0.0105 against +0.0094 on seed 0, +0.0105 against +0.0092 on seed 1.
   - It is not information removal in the strict sense. The release stays essentially one-to-one on the observed inputs:

     | Distinct rows | C1 (seed 0 / 1) | T_TREE (seed 0 / 1) |
     |---|---|---|
     | Pair | 7,500 / 7,500 | 7,500 / 7,500 |
     | Occupation | 7,499 / 7,494 | 7,500 / 7,498 |
     | Income | 7,496 / 7,470 | 7,453 / 7,335 |

     What it removes is the smooth SEX-propensity term (π(x) − w)·[P(y|x,F) − P(y|x,M)] that finite-sample readers exploit.
2. **The removal is small relative to the bar.**
   - Occupation recovery falls 0.022 / 0.011, to 0.621 / 0.629. That is NEURAL_RAW's level (0.621 / 0.616), 0.039 / 0.033 above the bar.
   - The residual comes from within-SEX input–occupation links, which marginalization keeps by design.
   - SEX is the strongest occupation predictor given the inputs: CatBoost importance rank 1, used in 163 / 153 trees, lowering the internal-fold loss by 0.057 / 0.065 nats. For income it ranks 6th, with a gain of 0.0045 / 0.0057 nats.
3. **The burden falls on men.**
   - Occupation class-wise NLL against T_TREE: NATRES_CONSTRUCTION_MAINTENANCE +0.099 / +0.097 nats, PRODUCTION_TRANSPORT_MATERIAL +0.035 / +0.031; SERVICE and SALES_OFFICE unchanged or better.
   - Men's occupation Brier excess rises +0.0054 / +0.0058, against +0.0013 / +0.0007 for women. The men's guard is at 0.88× its cap on seed 0 and fails on seed 1.
   - The cause is that the SEX-specific patterns are asymmetric. Mean P(female) − P(male) is −0.142 for NATRES_CONSTRUCTION and −0.087 for PRODUCTION, against +0.145 for SALES_OFFICE. Averaging with population shares pulls men's true-class probabilities down most.
   - The temperature control places a similar total cost on women instead: women +0.009, men −0.001.
4. **Income marginalization is nearly free** (+0.0002 / +0.0009 nats against T_TREE). It meets the income bar on both seeds, tightly on seed 1 (0.5479 ≤ 0.5500; 0.5461 ≤ 0.5476).
5. **C2.** With occupation fixed at JOINT_TREE's output, the pair reads 0.6351 with T_TREE income, 0.6277 with marginalized income and 0.6224 with LOCAL income. Marginalization removes about 0.007 of the income output's contribution to the pair, about 58% of what LOCAL's adversarial training removes. C2 was abandoned on seed 0 under its own rule.
6. **Why there is no C3.** Full marginalization with constant weights is the family's strongest removal of the SEX pathway. Nothing left in the family can close a 0.03–0.04 occupation gap at a feasible men's guard:
   - pooling the SEX propensity to a few levels puts propensity structure back in;
   - smooth partial shrinkage keeps the propensity direction;
   - a different constant weight moves the burden between men and women;
   - separate per-SEX models widen the very SEX gaps that create the men's burden.

## Best configuration and promotion

The best configuration is C1_MIX_FULL; its figures are in the results table above.

**Promotion rule (PROTOCOL §6): not met.**
- C1 is not eligible: the seed-1 men's guard fails, and the occupation recipient margin fails on both seeds.
- C1 is not promotable either: its pair proxy is above both references on both seeds.
- C2 is not promotable: its seed-0 pair proxy is 0.6277.

Family C has no finalist candidate.

## Resources and access

- **Configurations.** 5 of 10 counted: C1 on both seeds, the control on both seeds, and C2 on seed 0. One further diagnostic was counted 0. Ledger entries: 25, 29, 30, 33, 34, 36.
- **Compute.** 12 `AMD_C_*` queue units, all done, 0 failed, 118.0 CPU-s. A further 60 CPU-s of light scripts was charged as `AMD_C_investigator_light_scripts`. Total about 0.05 CPU-h.
- **Data access.**
  - MODEL_SELECT labels were read 6 times, all through `amd.devrun score` and logged.
  - MODEL_FIT features, labels and SEX were used for fitting; every other role was read as features only.
  - No CALIBRATION labels were used, and no ATTACK SEX or DEV_REPORT labels were read.
- **Files.**
  - `amd/fam_c.py`.
  - `results/.../roles/investigator_c/CARD_REVISIONS.md`.
  - Private runs in `.private/pcrl_autonomous_method_development_v1/runs/C/<config>/S<seed>/`: RELEASE.npz and .json, FIT_DIAG.json, EVAL_MS.json, and the SEX-aware teacher JSONs.
- Nothing was committed, and no shared module needs changing.
- The ledger's code hashes differ between rows because configurations were appended to the module between runs. The C1 fitting code did not change after the first fit.
