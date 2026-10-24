# Independent verification, wave 1 (strong-anchor release campaign, SAR1)

Verifier, launched 05:53Z; this report written 2026-10-24T06:18:53Z (`date -u`). Inputs: the brief
`roles/coordinator/VERIFIER_BRIEF.md` (Launch details), PROTOCOL.md sections 5-8, `roles/statistical_reviewer/RULES.md`, the four
cards, the plans and decision records in `claims/`, the fit records in `fa/` and `fb/`, the saved releases, the grid records and the
reader cache under `.private/pcrl_strong_anchor_release_v1/runs/`, and the access and queue logs. Machine-readable results:
`CHECKS.json` (SHA-256 of every input per check; 0b5e2c20...747d at writing). Every elementary value is in `work/`.

## How the numbers were recomputed

- My own code in `code/` (`vlib.py`: task metrics, clustered and linearized SEs, the weighted AUC, the household bootstrap, the split
  rule, the cache key, the aggregate screen, the recipient and frontier rules). I did not import or call `sar.rules`, `sar.claims`,
  `nrm.rules` or `sar.devreaders`. I used these only as loaders: `sar.common.load(..., 'verify', ...)` for the labels,
  `sar.release.load` (hash-checked releases), `ohn.common.frozen_bank` (U0 / Ucal), `ohn.devrun.reference_release` and
  `tsz.evaluate` (the frozen reference arrays). The refits used the locked recipe modules (`scm_bench.readers`,
  `task_conditioned_representation.attacks`, `ffv.readers`), called through my own dispatch.
- Label reads ran in two queue units, one at a time:
  - **SAR_VERIFY_1** (python3; 14.2 CPU-s). MODEL_SELECT income, occupation and SEX: utility, paired costs, and every grid winner,
    SCORE AUC and bootstrap draw.
  - **SAR_VERIFY_2** (torch venv, the grid's locked interpreter; 3.8 CPU-s). MODEL_SELECT SEX, plus MODEL_FIT SEX for p_men: the
    6 refits and the seed checks.
- The label-free scripts `v_decide.py` and `v_assemble.py` ran outside the queue, under 15 CPU-s in total, together with the record
  inspection. That makes about 33 CPU-s of the 2,400 cap.
- Both units' reads are in the access log with entry points `results/.../roles/verifier/code/v_ms.py:main` and `v_refit.py:main`.

## Totals

**132 checks: 129 PASS, 3 FAIL.** The 56 grouped MODEL_SELECT entries cover **8,808 elementary comparisons, 0 failed.**

| Check | Result |
|---|---|
| 1 Utility | **All PASS.** 28 conditions and their SEs per release: FB_PFO, FB_PFO_TASK, FB_R1_CTRL_HI, FB_R1_ALT_HI, FA_TFRO_R1, FA_TFRO_R1_CONT and FA_R1_CTRL_LO, plus all 11 entries of `runs/REFERENCES/EVAL_MS.json` (NEURAL_RAW, H_LJ, SUCC, ...). Also every recorded paired cost, and the evaluated wire hashes. Largest difference 2.3e-16 (tolerance 1e-9). |
| 2 Selections | **All PASS.** FA_REG: rho 0.3 fails (a) and (b) in both versions. FB_REG: v0 k* = E1; v1 E1 fails income Brier (bound 0.00383 > 0.0025). FB_R1: k* = S50 in both versions (S60 fails the women's guard); CTRL HI = DQ 0.417725 / 0.37975 and ALT HI = S160 / S130, with LO 0.285312 / 0.313843 and S140 / S120 recorded, not saved. FA_R1: rho* = 1.0 / 0.6, the guard estimate Delta = L +/- w C re-derived; CTRL LO s = 0.2 / 0.3, HI none (CONTROL_PATH_MAY_NOT_BRACKET) / the origin (alias START). Margins equal SUCC's recomputed MODEL_SELECT margins (1e-16); p_men = 0.5103 recomputed on MODEL_FIT. |
| 3 Grid | **All PASS, computed on every cell, member and view of SAR1_FB_R1 and SAR1_FA_R1 rather than a sample.** It covers the winner re-pick from the stored VALIDATION AUCs, the SCORE AUCs and the 400 bootstrap draws (registered and non-table winners; all bitwise, difference 0.0), and the paired cells and bootstrap SEs (difference 0.0). Every decisive aggregate and label is recomputed: FB_R1 D2 START / CONT / CTRL / ALT, D3 against the 9 references, D4, and the UM profile. The reader and bootstrap seeds match. No seed-dependent recipe is equal under k0 and k1 on any minimum-set wire. |
| 4 Refits | **6 of 6 bitwise** (VALIDATION AUC, orientation, SCORE score vector and SCORE AUC). The sample was drawn with seed 20261024 (indices 12, 202, 293, 329, 451, 478) from the 600 registered-cell winners of the two banks
(excluding U0 / Ucal): FB_R1 CTRL_HI income LOG_LR, FB_PFO_TASK occupation CATI1, NEURAL_RAW pair CATI01, FA bank NEURAL_RAW pair MLP64, TWJ occupation MLP64, SUCC occupation ET5. |
| 5 Outcomes | **All PASS.** FA_REG / FB_REG: NO_FEASIBLE_STRENGTH, class TASK_INFEASIBLE, registered. FB_R1 (ADAPTIVE_R1): D1 TASK_FEASIBLE (CAND confidence UNRESOLVED on v1 men's guard), UM MARGIN_NOT_RESOLVED, D2 PARTIAL_START+CONT_ONLY (CTRL NO_BRACKET; ALT NOT_SEPARATED_FROM_TASK_SPENDING), D3 NOT_SUPERIOR (passed NEURAL_RAW only), D4 NOT_MET, class FEASIBLE_NO_RELIABLE_GAIN. FA_R1 (ADAPTIVE_R1): D1 CAND_INFEASIBLE, v1 men's guard +0.009943 (SE 0.002016) against 0.005, a VIOLATION at 2 SE; v0 meets all 14 conditions; class TASK_INFEASIBLE. |
| 6 Access | **PASS, except the 2 plan-hash entries below.** The first label read (05:11:51Z) comes after the receipt (verified 05:11:39Z; registration commit 05:10:48Z). Every fitting read follows its card's commit, and each card is unchanged. No MODEL_SELECT read comes from a fitting module. No closed role was read, and no SEX was read on CALIBRATION. Every read is inside its queue unit. PLAN_FB_R1 / PLAN_FA_R1 were committed at 05:38:20Z / 05:45:00Z, before the first reading reads at 05:38:26Z / 05:45:05Z. Registration status is as recorded. |

Key FB_R1 numbers I recomputed match the record to 1e-12:
- pair benefit vs NEURAL_RAW +0.01252 (se_agg 0.00555) and vs FB_PFO_TASK +0.01425 (0.00546);
- MODEL_SELECT total-NLL cost CAND - Q: NEURAL_RAW +0.003836 and CTRL_HI +0.001416, so no bracket; ALT_HI -0.000754, which makes it
  Q*;
- benefit vs ALT_HI +0.00115 (0.00744): AGG_POSITIVE_POINT_ONLY;
- income recipient gate vs ALT_HI: POINT_FAILED;
- the non-table winner's benefit vs ALT_HI by version: -0.00044 / +0.00275.

## Failed checks (preserved; none changes a label)

1. **S6_FB_R1_plan_hash_recorded** and **S6_FA_R1_plan_hash_recorded.** PROTOCOL 6 requires the plan's hash in the reading's
   decision record. `claims/FB_R1/DECISIONS.json` and `claims/FA_R1/DECISIONS.json` contain only `"plan": "FB_R1"` / `"FA_R1"`;
   the only SHA-256 in each is the rules hash.
   - The missing plan hashes are 349a6f6e...5a45 (PLAN_FB_R1.json) and 5a78be99...51ab (PLAN_FA_R1.json).
   - The plans themselves were committed before the first read (check 6).
2. **S1_FB_R1_income_bitwise_START.** Card FB_PFO_R1 item 1 says CAND's income equals START's bitwise. On MODEL_SELECT, FB_PFO's
   income differs from the frozen NEURAL_RAW reference release used as START by the grid and the evaluator:
   - max |diff| 6.47e-8 (v0) / 6.84e-8 (v1); decisions identical; the wire hashes differ (plan: 53b756a3... vs f4de1c9f...);
   - the cause is that the family recomputes NEURAL_RAW's E2 income network under its own float64 forward law;
   - effect: income paired utility differences below 1e-9, and every income-view grid cell vs NEURAL_RAW exactly 0 (ALIAS_ZERO).
   - D2 / UM treat this view as ALIAS_ZERO by the numbers, although the wires are not exact aliases.

## Reported, not failures

- D2's CONTROL_POINT_RULE_NOT_FOLLOWED_CTRL / _ALT flags are reproduced. With only HI saved, the origin is the lower point, while the
  full grid holds a larger qualifying LO (0.00182 / 0.00192 in v0), so the frozen fineness rule reads SAVED_POINT_RULE_NOT_FOLLOWED.
  PROTOCOL 10 had anticipated COARSE_BELOW for this case. The flag is never gating.
- The session's starting git snapshot showed `claims/FA_REG/` and `claims/FB_REG/` untracked at 05:53Z. I did not check whether they
  have been committed since.

## Not checked in this wave

- No git was run.
  - Commit times come from the worktree reflog file (`logs/HEAD`, read as a file), and the plan commits are identified by the hash
    prefixes in the coordinator's commit log.
  - I did not read commit trees, so I did not confirm that the committed plan and card bytes equal the current files. The cards'
    current SHA-256 equals the hashes logged at each fitting read.
- The MODEL_FIT / CALIBRATION spends, their SEs and the lambda = 0 path spends were not recomputed. I re-applied the rules to the
  recorded values; only p_men was recomputed.
- From the records, not recomputed:
  - the enhanced (TABLE_CV) winner AUCs (TABLE_CV scores are not cached; used only for the reported READER_SENSITIVE flag);
  - the reader-budget flags (all READER_BUDGET_COMPARABLE).
- The full grid-validity pass (2,880 seed checks; 16 explained ET5 / ET25 equalities off the minimum set) was not re-derived beyond
  the minimum-set seed check.
- D2R reads NOT_RUN in the FB_R1 record (the SAR1_FB_R1_ROB bank was still running); it is left for a later wave.
- Deployment and packaging equivalence and descriptive claims (for example "lowest pair so far") are not checked.

# Wave 2 (written 2026-10-24T06:49:03Z, `date -u`)

Scope: the coordinator's wave-2 request, all five items: FA_R2, FS_CPI, FB_R1 D2R, FA_R2 D2R and housekeeping. The lane for item 4
had finished at 06:38:00Z, before I read any of its cells. The rules are as in wave 1.

The wave-2 entries are appended to `CHECKS.json` (field `"wave": 2`). The wave-1 file was hash-checked first (0b5e2c20...747d), and
its entries and top-level fields are unchanged; per-wave totals are under `waves`.

**Units and CPU.** All queue units, one at a time:

| Unit | Interpreter | What | CPU-s |
|---|---|---|---|
| SAR_VERIFY_3 | python3 | FA_R2 utility, paired costs, the SAR1_FA_R2 grid and the full SAR1_FB_R1_ROB grid | 20.3 |
| SAR_VERIFY_4 | torch venv | 8 refits and the FA_R2 minimum-set seed check | 2.3 |
| SAR_VERIFY_5 | python3 | the full SAR1_FA_R2_ROB grid | 12.7 |
| SAR_VERIFY_6 | torch venv | the FA_R2 guard contrast from the saved bag; a MODEL_FIT label read | 4.7 |

The label-free `v2_decide.py` / `v2_assemble.py` ran outside the queue, under 5 CPU-s. All waves together: queue 57.9 CPU-s, plus at
most about 30 CPU-s outside the queue, of the 2,400 cap.

I made no code edit while one of my own units ran; each verifier unit's record shows an empty `code_changed_during_run`. Two
coordinator units did record my new files as drift:
- SAR_GRID_SAR1_FA_R2_ROB_V1 records `v2_ms.py` and `v2_refit.py`, created at about 06:36-06:37Z;
- SAR_CUSTODY_REHEARSAL_V1 records `v2_assemble.py` (06:47Z).

Neither unit imports the verifier folder.

## Totals

**Wave 2: 124 checks, 124 PASS, 0 FAIL.** The grouped entries cover 17,404 elementary MODEL_SELECT comparisons, 0 failed.

**All waves: 256 checks, 253 PASS, 3 FAIL.** The three FAILs are wave 1's and are preserved:
- S6_FB_R1_plan_hash_recorded
- S6_FA_R1_plan_hash_recorded
- S1_FB_R1_income_bitwise_START

Wave 2 adds separate adjudication entries for them; the original FAILs stand:
- W2_S6_FB_R1_plan_sidecar / W2_S6_FA_R1_plan_sidecar: the sidecar hashes equal the plan and decision files.
- W2_ADJ_S1_FB_R1_income_bitwise_START: deviation 10's corrected bound of 7e-8 holds against the 6.84e-8 measured.

## Results by item

1. **FA_R2.** Checks 1-6 all PASS.
   - **Check 1, utility.** Recomputed for FA_TFRB_R2, FA_R2_CTRL_LO, FA_R2_CTRL_HI (v1 = the frozen SUCC, bitwise) and the reused
     FA_TFRO_R1_CONT record, with every recorded paired cost; largest difference 2.8e-16. CAND's income equals SUCC's bitwise.
   - **Check 2, selections.** rho* = 0.6 / 1.0, re-derived as follows:
     - SE_C = max(household SE, SD of the five per-fold contrasts (ddof 1) / sqrt 5); the between-fold SE binds at rho*: 0.00339
       vs 0.00137 (v0) and 0.00250 vs 0.00134 (v1);
     - the budgets equal SUCC's MODEL_SELECT cap margins as I recomputed them;
     - the scan stops at rho 1.0 in v0 (men's bound 0.000636 > 0.000583) and at rho 2.0 in v1 (occupation NLL and the men's guard,
       bound 0.00904 > 0.00106).
   - **The guard contrast at rho\*.** I recomputed it from the saved bag (five fold networks and their feature maps in MODEL.pkl,
     SUCC via `ihs.fam_h.predict`, the teachers' stored out-of-fold MODEL_FIT outputs, my own folds and forward pass) without
     importing sar.fa. It is **bitwise equal** to the record: C = -0.000735 (v0) / +0.000289 (v1), together with the five per-fold
     contrasts and SE_C_between.
     - The fold-level MODEL_FIT inputs (men / women spends, C, p, the household SE) equal FA_R1's at every common rho, bitwise.
     - Not recomputed: the household SE of C, and the contrasts at the stopping rho (those fold models are not saved).
   - **CTRL.** v0: LO s 0.4 and HI s 0.1 (both new points); v1: LO s 0.25 (new) and HI s 0 (alias START). Roles and reuse match.
     Fineness reads NOT_APPLICABLE_CAND_NOT_ABOVE_START, because CAND's CALIBRATION spend is below 0.
   - **Check 3, grid.** Every cell, member and view, plus paired cells and SEs, matches with difference 0.0.
   - **Check 4, refits.** 6 of 6 bitwise (seed 20261025, frame of 312, indices 44, 49, 54, 97, 193, 195). Two further refits from
     the SAR1_FB_R1_ROB robustness cells (frame of 504, indices 26 and 275) are also bitwise.
   - **Check 5, outcomes.** All labels reproduce:

     | Decision | Label | Key numbers |
     |---|---|---|
     | D1 | TASK_FEASIBLE | CAND confidence UNRESOLVED on the v1 men's guard |
     | UM | MARGIN_GAIN_PROFILE_NOT_RETAINED | resolved men's-guard improvement (pooled -0.00285, se 0.00133); occupation view +0.0252 in v0 |
     | D2 | PARTIAL_CONT_ONLY | START +0.00658 (se 0.0141), occupation recipient point failed; CONT +0.0432 (0.0183); CTRL Q* = FA_R2_CTRL_HI at -0.00421, benefit +0.00663 (0.0140) |
     | D3 | NOT_SUPERIOR | passed W, TR100, TT100 |
     | D4 | MATERIAL_0020_POINT_ONLY | +0.0241 vs NEURAL_RAW, +0.0208 vs H_LJ |
     | Class | FEASIBLE_NO_RELIABLE_GAIN | |

   - **Check 6, access.** The plan_sha256 in the record matches (054a8b4b...). The plan was written at 06:13:49Z and committed
     (a85bd035a) at 06:14:04Z, before the first reading read at 06:14:17Z. The card (d3b453d1d, 06:08:14Z) was committed before
     the fits.
2. **FS_CPI.**
   - Selection re-applied to the recorded spends: v0 lambda 0.025 fails income accuracy (bound 0.00519 > 0.0040), so
     NO_FEASIBLE_STRENGTH; v1 lambda* = 0.05 (lambda 0.1 fails income accuracy at 0.00470).
   - The reading has no candidate, and every label reproduces (ADAPTIVE_FS__..., class TASK_INFEASIBLE).
   - Version 0 saved nothing. No MODEL_SELECT record, grid bank, grid output, role or MODEL_SELECT label read concerns any FS_CPI
     release; the only FS units are the two fit units.
   - Could not recompute: the CALIBRATION spends themselves. That would need refitting the residual and START_CF; only the rule
     was re-applied.
3. **FB_R1 D2R.** ROBUSTNESS_NOT_CONFIRMED, re-derived from my own recomputation of all 32 cells (24 robustness cells: R3 / R4,
   and k2 / k3), registered vs robustness-cell means:

   | Comparator | Registered | Robustness (se) | Result |
   |---|---|---|---|
   | NEURAL_RAW | +0.01252 | +0.01187 (0.00674) | not resolved |
   | FB_PFO_TASK | +0.01425 | +0.01405 (0.00643) | CONFIRMED |
   | FB_R1_ALT_HI | +0.00115 | -0.00118 (0.00701) | sign change; SIGN_CHANGE_FB_R1_ALT_HI reproduced |

   The n_eff values and the reader-seed qualifiers match. The robustness bank's registered cells equal the main bank's, and the
   bootstrap and reader seeds match.
4. **FA_R2 D2R.** ROBUSTNESS_NOT_CONFIRMED, with no sign change, re-derived the same way:

   | Comparator | Robustness (se) | Result |
   |---|---|---|
   | SUCC | +0.00341 (0.01393) | not confirmed |
   | FA_TFRO_R1_CONT | +0.03741 (0.01556) | CONFIRMED |
   | FA_R2_CTRL_HI | +0.00404 (0.01340) | not confirmed |

5. **Housekeeping.**
   - All five PLAN_SHA256.json sidecars (FA_REG, FB_REG, FB_R1, FA_R1, FS_CPI) equal the SHA-256 of the plan and decision files.
   - claims/FA_REG and claims/FB_REG are tracked in this worktree's git index with identical content (index v2, read as a file;
     git not run). The reflog shows commits after 06:21Z, but HEAD's tree was not read, so "committed" is supported, not proven.
   - Over all 52 label reads logged when the check ran, the access rules hold, including the FA_R2 and FS card-commit and plan
     ordering.

## Not checked in wave 2

- SE_C_household for FA_R2. The R2 fold models at the stopping rho are not saved, so their contrasts were not recomputed.
- The CALIBRATION / MODEL_FIT spends behind any selection, other than the FA_R2 guard contrast at rho*.
- FS_CPI v1's saved releases, which are not evaluated by any reading.
- The full grid-validity pass beyond the minimum-set seed checks.
- The deployment and custody units (SAR_DEPLOY_*, SAR_CUSTODY_*).

# Wave 3 (written 2026-10-24T07:08:25Z, `date -u`)

Scope: the 13 deployment packages and the publication dry build (237 files). Everything was label-free; no queue unit and no git were
used. The work ran outside the queue: `v3_deploy.py` and `v3_deploy_diag.py` under the torch venv (Python 3.13.7, numpy 2.4.2,
sklearn 1.8.0), `v3_pubscan.py` under `python3 -I`, and `v3_assemble.py`, about 5 CPU-s in total. New code files only. The wave-3
entries are appended to `CHECKS.json`; the file was hash-checked first (97681e1a...).

**Totals.** Wave 3: **109 checks, 102 PASS, 7 FAIL.** All waves: **365 checks, 355 PASS, 10 FAIL.** This includes 2 addendum
entries appended at 07:09:28Z.

## Deployment packages: (a) to (f) per package

Five of the six parts pass for all 13 packages:
- **(a) Dependencies.** Every dependency hash re-hashes: 21 to 31 per package, with my own key resolution.
- **(b) Stored wire.** The stored wire equals the evaluated RELEASE.npz bitwise for both tasks on all six roles, and the release
  hash chain holds.
- **(c) MODEL_SELECT wires.** They equal `wire_sha256_eval` of the record that evaluated each release (for the role points, the
  ROLES records).
- **(d) Validity.** All arrays are finite and within [0, 1]; the worst row-sum error is 4.4e-16 against my 1e-9 tolerance.
- **(f) Refusals.** All three probes are refused in every package:
  - an extra SEX field (PermissionError);
  - a missing POBP field (ValueError);
  - a raw_score request (PermissionError).

**(e) Black-box equivalence** on 50 DEV_REPORT raw rows (seed 20261026, the same rows for every package) splits by family:
- **FA packages (7):** bitwise.
- **FB packages (6): FAIL on bitwise, kept.** The occupation probabilities differ from the stored DEV_REPORT wire by at most:

  | Package | V0 | V1 |
  |---|---|---|
  | FB_PFO | 5.6e-16 | 2.2e-16 |
  | FB_PFO_DQ_HI | 1.1e-16 | 1.1e-16 |
  | FB_PFO_RAWJ_HI | 3.3e-16 | 2.2e-16 |

  Income is bitwise, and every decision equals the stored argmax.
- **Separate entries (the FAILs stand):**
  - The 50-row outputs are within 1e-15, with identical decisions.
  - A diagnostic shows that `sar.deploy.release` on **all 5,943 DEV_REPORT rows**, the batch the stored wire was computed on,
    reproduces the stored wire bitwise for all six FB packages and for FA_TFRB_R2_V0.
  - 50-row and 1-row batches differ at the ULP level; even FA_TFRB_R2_V0's single-row occupation differs by 5.6e-17.

  The deployed output therefore depends on batch composition at the floating-point level (float64 matrix products). No decision is
  affected.
- **Comparison with your DEPLOYMENT_CHECK / VALIDITY**, read only after my checks: your all-rows bitwise parity, validity and
  refusal verdicts agree with mine. Your `all_bitwise: true` holds for the full DEV_REPORT batch only, not for arbitrary request
  batches.

## Publication leak scan (dry build, 237 files)

No hits in any of these categories:
- absolute `/Users/` paths;
- e-mail addresses;
- tokens or keys (GitHub, AWS, sk- keys, private-key blocks, secret / password / api-key / token words);
- session temp paths;
- binary or array files (by extension, NUL bytes or non-UTF-8 content);
- record-level data: no long numeric arrays and no person-level keys with long lists. The only CSV is the aggregate
  `results/dev/UTILITY_LEDGER.csv` (44 rows of per-release aggregates).

**One finding, FAIL (W3_PUB_user_name).** The user's name appears inside the drive-relocation folder name
"<owner>-Mac-relocated-2026-10-22":
- `results/PROTOCOL.md:59`
- `results/STORAGE_ADMISSION.json:6`
- `results/STORAGE_ADMISSION.json:47`

It is a personal identifier in an anonymized publication; redact it (for example, `<relocation run>`) before the final build.

**Hits judged harmless:**

| Pattern | Hits | Locations | Why harmless |
|---|---|---|---|
| `/Volumes/` | 3 | `code/sar/admit.py:366`; `code/sar/storage.py:24`, `:33` | the backup-drive label YOTUO and the system path /System/Volumes/Data; no user name |
| `~/storage-relocation-20261022/` | 5 | `code/sar/storage.py:7`, `:64`; `results/PROTOCOL.md:55`; `results/STORAGE_ADMISSION.json:39`; `results/roles/coordinator/COORDINATOR_LOG.md:7` | home-relative, no user name |
| worktree paths | 295 | mostly my published CHECKS.json and work files, plus PROTOCOL.md, SOURCE_ADMISSION.json and hand-backs | all sanitized `<repository>/...` or relative project worktree names |
| `Bostesa` | 6 | `code/sar/common.py:95`, `code/sar/receipt.py:7`, `results/HANDOFF.md:5`, `results/PROTOCOL.md:47`, `results/RECEIPTS.json:33`, `results/SOURCE_ADMISSION.json:2893` | the public remote URL; the owner keeps this repository public by decision |
| "scratchpad" | 3 | statistical reviewer hand-backs | a word, not a path |

The worktree-path sanitization also rewrote the reflog path literal in my published `code/v_decide.py:656`, so that copy no longer
runs as-is. This is cosmetic.

Every hit's file and line is in `work/pubscan3.json`.

## Not checked in wave 3

- The final publication build; this was the dry build.
- The final backup.
- Request batches other than the 50-row sample, single rows and the full DEV_REPORT batch.
- Deployment on other roles' raw inputs; only DEV_REPORT has raw inputs.
- Randomized-release key handling; no package is randomized.
