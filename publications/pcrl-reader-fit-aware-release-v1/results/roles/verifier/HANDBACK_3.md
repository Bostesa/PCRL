# Independent verifier, hand-back 3 (verbatim; delivered to the coordinator at about 23:12Z; harness indentation removed)

# RFA1 verifier hand-back 3 (wave 3)

**Verdict.** Every check in tasks 1-3 passes. I independently reproduce, with my own code:
- the R1 decision record: ADAPTIVE_R1, with D1 TASK_FEASIBLE, D2 NOT_SEPARATED, D3 NOT_SUPERIOR, D4 NOT_MET, LC LEARNABILITY_LIMITATION and class BOUNDED_NEGATIVE_EXPERIMENT;
- the half-budget grid validity;
- the corrected family-change reading FC_R1;
- the rebuilt v1 record;
- the v1 second split D2R, which is ROBUSTNESS_NOT_CONFIRMED;
- the deployment packages.

Task 4 (the publication scan) is pending your message after about 00:15Z.

I found no new defect. There is one disclosure (D-W3 below): `rfa/rfclaims.py` was modified, still uncommitted, while my main unit was running.

## Task 1: the R1 decision and its inputs

### 1a. R1 fit records (label-free; `out/V3_RECORDS.json`; 61/61 pass)

All of the following are recomputed from `fam/FIT_RF_R1_V<v>.json` with the card rule and the SAR1 bracket rule, as in wave 2.

| | v0 | v1 |
|---|---|---|
| k\* | S30 | S60 |
| First failing step | S40 (women's Brier 0.00294 > 0.0028) | S70 (occupation NLL 0.00628 > 0.006) |
| STOPFIT bracket | S20 / S30 | S50 / S60 |
| ALTCRIT bracket (LO not saved) | S20 / S30 | S50 / S60 |
| ALT bracket | S110 / S140 | S150 / S180 |
| F0 gate (orthogonal share, cos) | MECHANISM_ACTIVE (0.098, 0.99564) | MECHANISM_ACTIVE (0.1015, 0.99531) |
| ALTCRIT median relative lag | 0.247 | 0.257 |
| λ (shared by FULL / STOPFIT / ALTCRIT; TASK 0) | 4.100 | 3.729 |

- All brackets reproduce exactly, and all read FINE under RULES 4.4.
- F1: no orthogonal share fell below 0.05 before k\*.
- The lag is not a near-alias in either version. The plan uses the larger version's lag (0.2565).
- The RELEASE.json records name the R1 card (MODEL_FIT only, unit RFA_FIT_RF_R1_V<v>).
- The configuration ledger has 24/24 entries. Entries 1-12 are unchanged since wave 2. Entries 13-24 are RF1_* under the R1 card, and they match the release records.

### 1b. R1 usefulness (unit RFA_VER_W3_MAIN_v1; `out/V3_MAIN.json`)

The 28 conditions of all six RF1 configurations equal EVAL_MS exactly (max abs diff 0.0, flags identical). The paired costs, including those against FAM/RF_FULL, are also exact (0.0, none skipped). Every release file and all 12 of its arrays were hash-checked.

| Release | G1 | Worst value/cap v0 | Worst value/cap v1 |
|---|---|---|---|
| RF1_FULL | yes | 0.634 | 0.915 |
| RF1_TASK | yes | — | — |
| RF1_STOP_LO | yes | — | — |
| RF1_STOP_HI | yes | — | — |
| RF1_ALTC_HI | yes | — | — |
| RF1_RAWJ_HI | **no** | — | 1.137 |

Income vectors: the occupation-only RF1 configurations share one income vector, and RF1_FULL's income equals RF_FULL's bitwise. The plan's disclosure is confirmed: RF1_TASK ≠ RF_TASK and RF1_RAWJ_HI ≠ RF_RAWJ_HI in both versions.

### 1c. Grids

The following grids are recomputed from GRID.json, BOOT.npz and the cache with my code:

| Grid | Cells per version | Notes |
|---|---|---|
| RFA1_R1_MAIN | 4 | |
| RFA1_R1_LC | 8 | includes the @0.5 half-budget cells |
| RFA1_LC | 8 | includes the @0.5 half-budget cells |
| RFA1_ROB | 16 | R1-R4 × k0-k3 |

For every grid and both versions, the winners, SCORE AUCs, B = 400 draws, per-recipe AUCs, non-table AUCs and paired differences/SEs reproduce with max abs diff **0.0**. There are no mismatches, and the member wires match the grid provenance.

### 1d. My own R1 labels (vagg, checked against frozen NRM1)

**D1** is TASK_FEASIBLE.

**D2** is NOT_SEPARATED: no gating comparison passes.

| Comparison | Q\* | Mean | se_agg | Label |
|---|---|---|---|---|
| START | — | 0.009818 | 0.005399 | AGG_POSITIVE_POINT_ONLY |
| CONT | — | 0.008969 | 0.005134 | POINT_ONLY |
| STOPFIT | RF1_STOP_HI | −0.00144 | — | NOT_SEPARATED_FROM_TASK_SPENDING |
| ALTCRIT | RF1_ALTC_HI | −0.00304 | — | NOT_SEPARATED_FROM_TASK_SPENDING |
| RAWJ | RF1_RAWJ_HI | −0.00462 | — | NOT_SEPARATED_FROM_TASK_SPENDING |

The START, CONT and path-control values all equal the record within 1e-12.

**D3** is NOT_SUPERIOR. The failing references are NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC and TWJ; the per-reference values equal the record.

**D4** is NOT_MET.

The ADAPTIVE_R1-prefixed labels equal the record, and so does the class.

### 1e. Frozen-rules legs (`rfa/rules.py 76baa04a…`, unchanged since registration)

- The R1 record and the rebuilt v1 record were rebuilt with `rfclaims.build`. In each case, the replay equals the record, and the labels computed from my substituted inputs equal the record (D1-D4, LC and class).
- My substituted inputs were the grid entries, the non-table entries and the half-budget LC entries.
- `rfa.rules.d2r` applied to my merged MAIN + ROB entries gives ROBUSTNESS_NOT_CONFIRMED, equal to `claims/RFA_v1_D2R/D2R.json`.

### 1f. FC_R1 (RULES_ADDENDUM_1_R1 §3)

My computation gives FAMILY_CHANGE_NOT_POSITIVE.

| Quantity | Value |
|---|---|
| Pair mean | −0.002906 |
| se_agg | 0.00459 |
| Version means | −0.0016 / −0.0042 |
| Spend state | R1_SPENDS_MORE_UNRESOLVED (d = +0.000195, SE 0.000134) |
| Secondary metric resolved higher | men's Brier |

This equals `FAMILY_CHANGE_vs_RF_FULL_corrected.json`, and `rfclaims.family_change` gives the same result on my entry. The first file has `benefit_mean` None; this is deviation 9, and the label is unchanged.

### 1g. Half-budget grid validity

The recorded R1 LC validity is READER_SEEDS_EFFECTIVE (1,728 checks). My own count from the cache entries finds 0 k0/k1 equalities among the 6 seed-dependent recipes, at both fractions, in both versions. That holds for 54 checks per recipe and fraction in R1_LC, and 48 in v1's LC.

The rebuilt v1 record's LC validity is READER_SEEDS_EFFECTIVE (1,536 checks), and its labels equal the first v1 record. My zero count for v1's LC agrees with this.

### 1h. v1 D2R, per comparator

All 8 per-comparator labels equal the record:
- every comparator reads ROBUSTNESS_NOT_CONFIRMED;
- the robustness means equal the record exactly;
- the refit reads REFIT_GENUINELY_EFFECTIVE.

The sign changes (RF_ALTC_HI, RF_STOP_HI, SUCC) and the recipient failures (H_LJ, RETIRED_B3CTL, RF_RAWJ_HI, SUCC) equal the record.

START is v1's only D2 pass on the registered cells, and it does not survive the second split. Its robustness-cell mean is 0.0120 with se_agg 0.0075, which gives AGG_POSITIVE_POINT_ONLY.

## Task 2: data boundaries (`out/V3_RECORDS.json`; all pass)

- **Label reads:** 33 to 23:00Z, every one inside a queue unit and after the receipt. No CALIBRATION, ATTACK or DEV_REPORT label was read.
- **No fitting entry point read MODEL_SELECT.** MODEL_SELECT was read only by benchmark, readers or my verify units.
- **R1 fits:** RFA_FIT_RF_R1_V0 and V1 read MODEL_FIT only, at 22:40:05Z and 22:40:07Z. Both name the committed R1 card (`RF_RFIT_R1.md`, commit 4c746de74 at 22:23:27Z); its SHA-256 matches, and the commit precedes the reads.
- **R1 plan before R1 outcomes:** the first R1 MODEL_SELECT read is RFA_EVAL_RFA_R1 at 22:46:10Z. It follows the R1 plan commit 2b830dc5e (22:46:00Z), by 10 s.

## Task 3: deployment packages (unit RFA_VER_W3_DEPLOY_v1; `out/V3_DEPLOY.json`; 14/14 pass per package)

I re-ran RF_FULL_V0 and RF1_FULL_V1 under the venv (Python 3.13.7, numpy 2.4.2). The coordinator's check had run under Homebrew Python 3.14.3 with numpy 2.4.4.

**Package integrity:**
- I re-hashed all 20 pinned dependencies of each package; all match.
- MODEL.pkl matches both MODEL.json and the package pin.
- RELEASE.npz is the evaluated release.

**My own encoder** was written from the package's encoder spec. On 50 raw MODEL_FIT rows it equals the admitted features bitwise.

**My own float64 forward pass**, from the pinned weights:
- on the 50 MODEL_FIT rows it matches the evaluated release within 2.2e-16 (a batch-size effect at the ulp level);
- on the attacked MODEL_SELECT full batch it is bitwise identical;
- decisions are identical in both cases.

**The package entry points, used as black boxes:**
- `release` on the 50 raw rows is valid (finite, non-negative, rows sum to 1 within 1e-12, decision = argmax as int64). It exports exactly {probability, decision} and matches the release within 2.2e-16.
- `release_csv` equals `release` bitwise on invented rows.
- `underlying` on the MODEL_SELECT full batch is bitwise identical to the release.
- Request sizes 1, 7, 50 and 333 give identical decisions, with differences of at most 4.4e-16.

**Refusals:** all 24 of my own probes are refused.
- PermissionError for SEX, PINCP, OCCP, SERIALNO, SPORDER, ROW_ID, HOUSEHOLD and PUMA fields; hidden outputs (logits, raw_score, embedding); draw keys on a deterministic package; the CSV path with SEX or PINCP; and the encoded path with a SEX column.
- ValueError for an unknown extra field, reordered fields, a missing POBP, a non-finite AGEP, a CSV with an extra column, and encoded inputs with extra, missing or reordered columns or NaN.

The temporary CSV files held invented rows only, were written under `roles/verifier/out/invented_deploy/`, and have been deleted.

## Disclosure D-W3 (no effect on my results)

The queue recorded `code_changed_during_run: ['rfa/rfclaims.py']` for RFA_VER_W3_MAIN_v1. The coordinator edited the file, without committing, at 23:08:27Z, inside my unit's run (23:08:05-23:08:52Z).

The diff only adds an optional `d2r_dir` argument to `rfclaims.run` and its CLI. My unit used `rfclaims.build`, `rfclaims.family_change` and the `rfa.claims` helpers, and these are identical in both versions. The change is still uncommitted (` M rfa/rfclaims.py`; current SHA-256 `a960483d…`), so please commit it.

Separately, the registration-to-launch digest change for my unit is due to my own v3_deploy.py and v3_records.py, which v3_main does not import.

## CPU

- **In the queue (charged by the queue):** RFA_VER_W3_MAIN_v1 44.96 s + RFA_VER_W3_DEPLOY_v1 0.94 s = **45.9 CPU-s**.
- **Outside the queue:** about **3 CPU-s**, all label-free. This covers the v3_test_vagg2 invented-data check (900 trials, 0 mismatches against NRM1), v3_records, compiles and JSON inspection. Please charge 5 s.

## Hashes and files

Checked records (SHA-256 prefixes):
- `claims/RFA_R1/DECISIONS.json` `1bd80ff7`
- `claims/RFA_R1/FAMILY_CHANGE_vs_RF_FULL_corrected.json` `dbb209ff`
- `claims/RFA_v1_rebuilt/DECISIONS.json` `c0c2c452`
- `claims/RFA_v1_D2R/D2R.json` `eea6d600`
- `deploy/DEPLOYMENT_CHECK_MAIN.json` `c9df4475`
- `rfa/deploy.py` `cbe3c252`

All files below are in `results/pcrl_reader_fit_aware_release_v1/roles/verifier/`:
- **Code** (`code/`): v3_main.py `bb232416`, v3_deploy.py `4f6279ac`, v3_records.py `0c2db194`, vagg2.py `660252ea`, v3_test_vagg2.py `1c984f85`
- **Outputs** (`out/`): V3_MAIN.json, V3_DEPLOY.json, V3_RECORDS.json, V3_TEST_VAGG2.json

I changed no campaign code and no git state.
