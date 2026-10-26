# Independent verifier, wave-2 hand-back (verbatim; delivered to the coordinator at about 22:51Z; harness indentation removed)

# RFA1 verifier hand-back 2 (wave 2; verify access through queue units only)

**Verdict.** Model selection, usefulness, the reader grid, the refits and the decision labels all reproduce exactly with my own code. Every D1-D4 label and the outcome class are confirmed.

| Label | Value |
|---|---|
| D1 | TASK_FEASIBLE |
| D2 | PARTIAL_START_ONLY |
| D3 | NOT_SUPERIOR |
| D4 | NOT_MET |
| Outcome class | BOUNDED_NEGATIVE_EXPERIMENT (partial PARTIAL_START_ONLY) |

There is one new substantive finding (F1 below), and it does not change any label. RF_FULL's released income is not bitwise the frozen bank's NEURAL_RAW income (|Δp| ≤ 6.8e-8). The card claims "released income vector stay[s] START's bitwise", so that claim does not hold against the frozen START release that the grid and the evaluator use as START.

The R1 derivative check was delivered early, before the R1 fits, and is saved as INTERIM_R1_DERIVATIVE.md. Its result is repeated in section 5.

Two sets of failed checks are kept, not loosened, with their explanations in section 7.

## 0. Code verified equals code that ran

- **v1 fits:** both fit units ran at git HEAD 43830ae19. Its blobs equal the files I verified in wave 1: `rfa/fam/rfit.py 04b22c66…`, `rfa/fam/fit.py 9c5577b8…`, `rfa/fam/net.py 1f08e9c7…`, `rfa/devreaders.py 4046bf1d…` (equal to DEVGRID_CODE_LOCK).
- **Current `fit.py` `d1c9efb5…`:** this adds only the `rf_module` / `alias_check` / `revision` plumbing for R1 (diff read); with `rf_module=None` the v1 path is unchanged.
- **R1 module:** `rfa/fam/rfit_r1.py 24c2bd99…`; R1 card `f…` committed in 4c746de74 at 22:23:27Z.
- **Decision record:** DECISIONS.json cites plan `aced3463…` (equal to the file) and RULES.json `4e03270c…` (equal to the registration receipt). The plan pins `FIT_RF_V0 aff6fda0…` and `FIT_RF_V1 ed752752…`, both equal to the files.

## 1. Model selection, from the records (label-free; 59/59 pass; out/V2_RECORDS.json)

**k\* of RF_FULL.** I recomputed it with the card §5 rule: estimate + 2 SE ≤ budget on all 5 occupation components, for every step from S10 to S_k. The recorded budget equals the card's.

| Version | k\* | First failing step | What fails there | Tightest margin at k\* |
|---|---|---|---|---|
| v0 | S20 | S30 | women's Brier 0.00299 > 0.0028 | women's Brier, 0.00037 |
| v1 | S70 | S80 | women's Brier 0.00293, occupation NLL 0.00695 > 0.006 | occupation NLL, 9e-6 |

The CAND record is consistent with these, and its spend over START equals the selection path to within 1e-12.

**Brackets.** I re-implemented the SAR1 rule from `sar/fb/pfd.py:251-301`. LO, HI, flags and fineness are reproduced for all three paths in both versions, and the grid spends equal the selection path to within 1e-12.

| Path | v0 (LO, HI) | v1 (LO, HI) | LO saved? |
|---|---|---|---|
| STOPFIT | S10, S20 | S60, S70 | yes |
| ALTCRIT | S10, S20 | S60, S70 | no (LO_NOT_SAVED) |
| ALT | S100, S120 | S170, S200 | no (LO_NOT_SAVED) |

My RULES 4.4 fineness reading: in v0, STOPFIT and ALTCRIT are COARSE_BELOW and ALT is COARSE_BOTH; in v1 all three are FINE. This is consistent with the CONTROL_GRID_COARSE_* flags in DECISIONS.json.

**F0 gate.** I recomputed the derived statistics from the recorded cross-group estimates (diff ≤ 9e-16). n = 64 episodes and 8 groups. The rule equals the card's.

| Version | Orthogonal share | cos(tot, dir) | Label |
|---|---|---|---|
| v0 | 0.1024 | 0.99507 | MECHANISM_ACTIVE |
| v1 | 0.1119 | 0.99404 | MECHANISM_ACTIVE |

F1: no orthogonal share fell below 0.05 before S_k\*, in the deployed fit or any fold.

**ALTCRIT lag.** The median relative lag is 0.109 (v0) and 0.210 (v1). Both are ≥ 0.01, so ALTCRIT is not a NEAR_ALIAS and counts as an independent control.

**Saved configurations.** There are 6 per version, at k\* and at the bracket points. λ is shared by FULL, STOPFIT and ALTCRIT (v0 3.972), and TASK has λ = 0. Every RELEASE.json names the committed card (`f484c258…`, commit 43830ae19) and records MODEL_FIT only.

**Configuration ledger.** At 22:0xZ it held exactly 12 entries, numbered 1 to 12, matching the release records' SHA-256 and kinds. In the usefulness unit, all 12 RELEASE.npz files matched their records, and all 12 arrays in each file matched their per-array hashes.

## 2. Usefulness (unit RFA_VER_USEFULNESS_v1; out/V2_USEFULNESS.json)

The 28 conditions were recomputed from the RELEASE.npz files and the frozen bank. The bank was hash-checked against both the source admission record and the manifest, and its row order against the export. The MODEL_SELECT labels came through the verify operation.

The values, SEs and pass flags equal EVAL_MS.json and REFERENCES/EVAL_MS.json with max abs diff 0.0 (14/14 conditions per version, every configuration).

| Release | G1 | Worst value/cap v0 | Worst value/cap v1 |
|---|---|---|---|
| RF_FULL | yes | 0.502 | 0.885 |
| RF_TASK | yes | 0.360 | 0.694 |
| RF_STOP_LO | yes | 0.406 | 0.767 |
| RF_STOP_HI | yes | 0.482 | 0.824 |
| RF_ALTC_HI | yes | 0.452 | 0.784 |
| RF_RAWJ_HI | **no** (v1 fails income Brier excess vs U0 and Ucal; worst 1.059) | 0.543 | 1.059 |
| NEURAL_RAW | yes | 0.340 | 0.570 |
| H_LJ | yes | 0.883 | 0.788 |

The paired costs (Q\* input) match every pair in EVAL_MS with max abs diff 0.0.

The D1 confidence conjunction matches DECISIONS.json:

| Release | Label | Conditions passing | Unresolved |
|---|---|---|---|
| CAND | CONFIDENCE_UNRESOLVED | 26 | v1 occupation NLL vs Ucal (0.00737); v1 men's guard (0.00442) |
| START | ALL_PASS | 28 | none |
| CONT | ALL_PASS | 28 | none |

The five occupation-only configurations share a bitwise-identical income vector in each version. The one failed check in this unit is the income-identity check against the bank (F1 below).

## 3. Data boundaries (out/V2_RECORDS.json and V2_RECORDS_access_final.json; 29 label reads to 22:46Z)

All pass, with no problems:
- **Timing:** every read came after the 21:53:27Z receipt.
- **Queue units:** every read sits inside a queue unit window.
- **Closed roles:** no CALIBRATION, ATTACK or DEV_REPORT label was read.
- **No fitting entry point read MODEL_SELECT:** all MODEL_SELECT reads came from benchmark, readers or my verify units.
- **Fit units:** all four (RFA_FIT_RF_V0/V1 and RFA_FIT_RF_R1_V0/V1) read MODEL_FIT only. Each fit read names a committed RF card, with matching SHA-256 and commit, committed before the read.
- **Plan timing:** the plan and fits commit ff741a72c (21:59:32Z) precedes the first MODEL_SELECT read (21:59:53Z).
- **My reads:** 4 verify reads, all MODEL_SELECT.

Observation: the v1 fit units recorded `uncommitted_code_paths` containing two of my verifier files (v1_graph_check.py, vcore.py), because I was editing them at launch. The fit never imports them, so this has no effect.

## 4. Reader grid, refits and decisions

**Grid recomputation (unit RFA_VER_GRID_v1; 20/20; out/V2_GRID.json).** RFA1_MAIN S0 and S1, 8 cells × 17 members × 3 views, all recomputed with my own code. The following equal GRID.json and BOOT.npz with max abs diff **0.0**:
- the winners, including the near-tie flags and gaps;
- the VALIDATION and SCORE AUCs;
- the B = 400 household-bootstrap draws;
- all 16 per-recipe SCORE AUCs;
- the non-table winners and their AUCs;
- the paired differences and SEs of RF_FULL against every comparator.

The member wires match the grid's provenance. Seed-dependent recipes differ between k0 and k1 in 102/102 checks for MLP and 98/102 for ET (4 identical per version).

**Bounded refit (unit RFA_VER_REFIT_v1; out/V2_REFIT.json).** I refit the pair winners of RF_FULL, RF_STOP_HI and NEURAL_RAW in S0R1k0, S0R2k1, S1R1k0 and S1R2k1 (MLP64, ET25 and CATI1 recipes), using my own sklearn code. **12/12 reproduce the cached oriented SCORE scores bitwise**, and the SCORE and VALIDATION AUC differences are 0.0. Library versions equal the grid lock.

**Decisive comparisons.** All computed with my own aggregate and frontier code (out/V2_GRID.json), which I first checked against frozen NRM1 on invented data: 0 mismatches over 400 screen trials and 300 frontier trials.

| Comparison | Q\* | Mean | se_agg | Label | Pass? |
|---|---|---|---|---|---|
| START (NEURAL_RAW) | — | 0.012724 | 0.005939 | AGG_POSITIVE | **yes** |
| CONT (RF_TASK) | — | 0.008608 | 0.009176 | AGG_POSITIVE_POINT_ONLY | no |
| STOPFIT | RF_STOP_HI | 0.002795 | 0.004462 | NOT_SEPARATED_FROM_TASK_SPENDING | no |
| ALTCRIT | RF_ALTC_HI | 0.000548 | 0.001 (floor) | NOT_SEPARATED_FROM_TASK_SPENDING | no |
| RAWJ | RF_RAWJ_HI | −0.001626 | 0.008443 | NOT_SEPARATED_FROM_TASK_SPENDING | no |

- START: lower bound 0.00085; version means 0.01004 and 0.01541; both recipient views RESOLVED.
- STOPFIT: secondary metric men's Brier resolved higher; c2 passes; c3 empty.
- ALTCRIT: secondary metric men's Brier resolved higher; c2 passes.
- RAWJ: c2 fails; secondary metrics occupation Brier, men's Brier and NLL resolved higher.
- Reader targeting (non-table winner) fails only against RF_RAWJ_HI (v0 −0.0031).

**D3.** Passed: NEURAL_RAW, W, TR100, TT100. Failing: B2TWIN, H_LJ, RETIRED_B3CTL, SUCC, TWJ. All reader-budget flags are COMPARABLE, so the credit rule is inactive.

**D4.** NEURAL_RAW: mean 0.0127, lower bound 0.0008. H_LJ: mean 0.0168, lower bound −0.0061. Both TARGET_AGG_NOT_MET, so D4 is NOT_MET.

**Decision check (unit RFA_VER_DECISIONS_v1; 21/21; out/V2_DECISIONS.json).**
- My D1-D4, class, D1 confidence, every D2 comparison, the targeting, the D3 per-reference values and the D4 screens all equal DECISIONS.json.
- The frozen-rules replay on the campaign's record reproduces every label (D1-D6, LC LEARNABILITY_LIMITATION, class).
- The frozen rules applied to my substituted grid entries and non-table benefits give identical labels.
- The substituted inputs equal the campaign's (max abs diff 0.0).

## 5. Revision R1 (interim result, saved as INTERIM_R1_DERIVATIVE.md; out/V2_R1_CHECK.json)

11/11 pass. Setup: my own numpy softbin implementation, centres from scipy `norm.ppf`, moments and decision scales frozen at θ0.

| Check | Result |
|---|---|
| FULL vs central FD of the re-solved J | rel 6.3e-10 |
| softbin family alone vs FD | rel 1.4e-9 |
| STOPFIT vs FD at fixed w | rel 3.3e-10 |
| Sign (step along −gradient) | actual/predicted 1.000004 |
| R1 with families [simple, rff] vs v1 | bitwise (FULL, STOPFIT, ALTCRIT, 2 calls) |

**R1 plumbing (out/V2_FIT_WIRING_R1.json; all core checks pass).** On `fit.py d1c9efb5` with `rf_module=R1`:
- every objective call is the R1 class with the softbin family;
- λ is shared across the episode arms and reproduced exactly;
- the pre-clip gradient equals ∇CE + λ∇S_R1 bitwise in all four arms.

## 6. Findings and defects

- **F1 — income wire not bitwise START's (substantive disclosure; no label change).** RF_FULL's MODEL_SELECT income differs from the frozen bank's NEURAL_RAW income by up to 6.47e-8 (v0) and 6.84e-8 (v1).
  - The decisions are identical, and the RF configurations agree with each other bitwise.
  - The RF income is START's network evaluated in float64 (`rfa/fam/net.py:13-15`, np_forward). The bank's vectors are float64 but are not float32-rounded copies (\|bank − float32(RF)\| up to 8.6e-8), so the cause is an upstream forward-pass difference. I have not verified this, because doing so needs the model files.
  - The claim at `cards/RF_RFIT_v1.md:12` ("released income vector stay[s] START's bitwise") and the "frozen bitwise" wording at `rfa/fam/fit.py:6,58` therefore hold against START's weights, but not against the frozen START release used on the grid and in the evaluator.
  - Effect on the grid: NEURAL_RAW's income is a separate wire, not an exact alias. Its income-view differences vs RF_FULL are non-zero in 3 of 8 cells: −0.00074 (S0R2k0), and −7.1e-7 in S1R1k0 and S1R1k1.
  - Bounded sensitivity (unit RFA_VER_INCOME_v1; out/V2_INCOME_SENSITIVITY.json). I refit START's pair winner per cell on RF_FULL's income, with the winner fixed. The controls reproduce the grid in 8/8 cells. The swap changes START's AUC by −0.00081 to +0.00003 (mean \|Δ\| 0.00017). The START comparison stays AGG_POSITIVE: mean 0.01256 vs 0.01272, lower bound 0.000845 vs 0.000846. This is descriptive only.
- **D4-new (minor, records).** `rfa/fam/fit.py:538` (`9c5577b8:517`) sets `n_episodes_deployed_fit` to the whole arm's episode count. RF_STOP_LO records 20 (v0) and 70 (v1), but its checkpoint is S10 / S60 (10 / 60 episodes).
- **Earlier defects still apply.** D1-D3 from wave 1 are logged in PROTOCOL_DEVIATIONS.md. D1, the ALTCRIT inner-fit undercount, applies to R1 as well: the probe showed 18 recorded vs 24 actual exact fits.
- **Notes for the coordinator:**
  - The specification ledger is now full, at 24/24 (the R1 entries RF1_* are 13-24).
  - RF1_TASK and RF1_RAWJ_HI were saved without the alias hook. They may be bitwise duplicates of RF_TASK and RF_RAWJ_HI; I have not checked this.
  - An R1-recipe OccTrainer without `rf_module` fails loudly; it does not fail silently.

## 7. Failed checks kept (not loosened)

- **(a) RFA_VER_USEFULNESS_v1, check "RF_FULL_income_equals_NEURAL_RAW_bank_income (abs < 1e-12)".** Measured 6.5e-8 and 6.8e-8, so the queue records the unit as failed. This is finding F1; every other check in the unit passed.
- **(b) V2_R1_CHECK run 1, feature-map check (9.7e-9).** My harness passed float32 scales into the helper. I fixed only the test input, and rerun 2 passes (6.1e-16). Run 1 is kept as out/V2_R1_CHECK_run1_HARNESS_FLOAT32_SCALE_FAILED.json.
- **(c) V2_RECORDS_access_final.json, five v1-scoped ledger checks fail after the R1 saves.** The checks are: exactly 12 entries, numbered 1-12, v1 card SHA, match to the release records, and 6 per version. All five failures are explained by the ledger growing to 24 entries. I confirmed entries 1-12 are unchanged from the passing 22:0xZ audit, and entries 13-24 carry the R1 card's SHA.

## 8. CPU

- **In the queue (charged by the queue): about 27.9 CPU-s.**

  | Unit | CPU-s |
  |---|---|
  | RFA_VER_USEFULNESS_v1 | 0.57 |
  | RFA_VER_REFIT_v1 | 7.55 |
  | RFA_VER_GRID_v1 | 10.34 |
  | RFA_VER_INCOME_v1 | 9.26 |
  | RFA_VER_DECISIONS_v1 | 0.19 |

- **Outside the queue: about 12 CPU-s** (please charge an upper estimate of 15 s). This covers the R1 derivative checks (2 runs), the R1 wiring probe, the vagg test, the record and access audits (3 runs), compiles and JSON inspection. All of it was label-free.

## 9. Files

All in `results/pcrl_reader_fit_aware_release_v1/roles/verifier/`:
- **Code** (`code/`): v2_records.py, v2_usefulness.py, v2_grid.py, v2_refit.py, v2_decisions.py, v2_income_sensitivity.py, v2_r1_check.py, v2_fit_wiring_r1.py, v2_test_vagg.py, vagg.py
- **Outputs** (`out/`):
  - V2_RECORDS.json, V2_RECORDS_access_2230Z.json, V2_RECORDS_access_final.json
  - V2_USEFULNESS.json, V2_GRID.json, V2_REFIT.json, V2_DECISIONS.json, V2_INCOME_SENSITIVITY.json
  - V2_R1_CHECK.json, V2_R1_CHECK_run1_HARNESS_FLOAT32_SCALE_FAILED.json
  - V2_FIT_WIRING_R1.json, V2_TEST_VAGG.json

I changed no campaign code and no git state.
