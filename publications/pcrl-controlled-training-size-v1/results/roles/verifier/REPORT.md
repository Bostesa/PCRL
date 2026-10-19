<!-- Attribution: written by the independent verifier (an agent assignment of this campaign) and returned as text in its final
hand-back, because the harness refuses report files from a subagent. Saved verbatim by the coordinator at about 01:58Z (2026-10-19,
date -u); not coordinator analysis. Its numbers come from its own code (roles/verifier/code/); its checks are in CHECKS.json and
CHECK_CHANGES.json (adjudications A1, A4, A6). -->

# Independent verification report: TSZ1 controlled training-size campaign (2026-10-19, 00:11–01:55Z)

**Bottom line.** All numbers I replayed with my own code agree with the campaign records: the size stage, the trigger, model identities, nested selection, releases, readers, claims, tables and deployment.
- I ran 121 checks. There are 5 FAILs: 3 are bugs in my own checks (corrected and adjudicated), and 2 are one genuine procedural finding recorded at two times.
- That finding is five pre-registration benchmark units that ran uncommitted code. It is disclosed as PROTOCOL_DEVIATIONS item 7.
- The registered outcomes stand as recorded:
  - TRIGGER_FIRES.
  - SIZE_GAIN_RESOLVED, FULL_ONLY / FULL_ONLY_70_RETAINS, SUPPORTED_ON_THIS_POOL.
  - REPRODUCED_BITWISE.
  - s* = 0.3 for LOC and PAIR in both versions.
  - CORRECTION_RECIPE_CLOSED.
- All 11 final packages deploy bitwise from the six raw fields, with every refusal, both live and from the restored backup.

| Kind | PASS | FAIL | ATTENTION | NOT_RUN |
|---|---|---|---|---|
| Numerical replay | 87 | 1 | 2 | 0 |
| Procedural compliance | 23 | 4 | 4 | 0 |

## FAILs and causes
1. **V0.PINS.PUBLICATION_TREE__V1: my bug.** I hashed CSVs with CRLF line endings. The source build reads text with universal newlines (CRLF → LF), so 21 published CSVs looked different. V2 passes: code 36/35/1 and results 342/319/23, as SOURCE_ADMISSION records. This is change 1, adjudicated A1 (accepted).
2. **V6.TABLES__V1: my bug.** I compared the 0.020 target against all six references. It is defined only against NEURAL_RAW and H_LJ. V2 passes; change 3, A4 (accepted).
3. **V6.TEXT_SCOPE_SCAN__V1: my bug.** My automated scan matched negated mentions of forbidden terms and missed a line-wrapped phrase. V2 gives ATTENTION (below). Change 4: **adjudication pending**.
4. **V6.PROCESS.CODE_PROVENANCE__V1 and __V2: genuine.** TSZ_BENCH_CORR_{ROUTING, CORRECT_LOC, CORRECT_PAIR, WITNESS, DIRECTION_PAIR}_V1 ran from 00:14:46 to 00:17:47Z with tsz/ entirely uncommitted. That is before the registration commit at 00:19:41Z.
   - They are label-free compute benchmarks on invented labels, and no outcome depends on them.
   - This is still a literal departure from PROTOCOL section 12. Disclosed as deviation 7.
   - Every post-registration production unit ran committed code.

## Numerical reproduction (own code; labels only through tsz.common.load, operation 'verify', always through the queue)

**V0 structure (label-free).**
- **Subsets:** all 9 subsets rebuild exactly from SHA-256('TSZ1_order_<o>|<household>') with ceil(0.5G) = 6,669 and ceil(0.7G) = 9,336. Household counts, people, household-set hashes and row-set hashes all match. Nesting, the 12 pairwise overlaps, the TSZ1_ncv10 folds and the inner-fold counts also match.
- **Reader splits and seeds:** the R1/R2 reader splits, their overlaps and the reader seeds (2026101705/06/15/16, none an OHN1/CTS1 seed) match READER_SPLITS.json.
- **Pins:** the four pinned commits exist, and 507024649 is an ancestor of HEAD. Every pinned artifact hash matches: A1_PO_16 OCC, the frozen banks, H3 RELEASE and the T_TREE teachers.
- **Rules constants:** the floor SEs are copied unchanged from OHN1 RULES.json (e18f1c68…), and the allowance table recomputes exactly.

**V1 gating deployment (DEPLOY_W100.json, sha f5de9259…).** For W100_V0_B2 and W100_V1_B2, `bitwise` and `refusals_all_pass` are both true.
- **Outputs:** all 16 occupation comparisons are bitwise. They cover the wrapper api, my own CSV and underlying from my raw encoding on DEV_REPORT and MODEL_FIT, underlying on the admitted encoding for the other four roles, and stored wire = RELEASE.npz.
- **Refusals:** 24 of 24 refusals raised, each with a positive control.
- **Other parity:** income equals the H3 wire, my encoder equals the admitted encoding, and my own pipeline is bitwise.
- **Superseded packages:** W100_V0/V1 fail their pinned code hashes, as you reported.
- **Agreement:** my record agrees with DEPLOYMENT_CHECKS_W100.json, and it was committed (00:27:11Z) before the trigger record (00:27:20Z).

**V2 trigger.**
- **Conditions:** all 14 conditions per version match, values within 1.7e-16 and SEs within 4.3e-19.
- **Ratios:** the occupation-side minimum ratios are identical: 4.8814643846635155 and 4.299044124207418, binding guard|sex0.
- **H3 wire:** the H3 wire equals the admitted RELEASE.json wire.
- **Decision:** T0–T3 pass, giving TRIGGER_FIRES.
- **Wording point:** RULES.md T0 says "28 conditions per version"; the correct count is 14 per version. This is ERRATA E1.

**V3 model identities.**
- **Size fits:** all 22 A1_PO_16 fits (18 + 4 ADM) refit bitwise on all six roles from my own subset rows, dictionary, folds, T* and HGB built from the recipe constants. T* is equal, the inner NLL curve differs by 0.0, and DICT.json is equal. The smallest best/second-best T* gap is 3.1e-7.
- **Source reproduction:** W100_v is also bitwise equal to the source A1_PO_16 S<v> on every role.
- **Alias:** there is one S100 fit per version, aliased to O0..O3.
- **S100 controls:** the teachers are file-identical to the admitted ones, TASK_REFIT θ at E20 is bitwise, and the MODEL_SELECT outputs equal the frozen bank.
- **T_TREE refit:** my independent T_TREE refit for S50_O1 v0 gives 759 trees, identical routing and raw difference 0.0.

**V4 size table and contrasts.**
- **Fit table (58 fits, VER_SIZE_V1):** values agree within 3.8e-16, SEs within 8.7e-19 and ratios within 1.6e-13.
- **Contrasts and labels:** every paired contrast (50→70, 70→100, 50→100), ordering summary and label agrees.
- **Primary endpoint (50→100 men's Brier):**
  - v0: +4.32e-3 ± 1.13e-3
  - v1: +4.22e-3 ± 1.12e-3
  - No sign reversal, giving SIZE_GAIN_RESOLVED.
- **Margin pattern:** 4 of 8 S50 cells and 8 of 8 S70 cells meet the reserve, giving FULL_ONLY with FULL_ONLY_70_RETAINS and SUPPORTED_ON_THIS_POOL.
- **Control-explains (TASK_REFIT):** CONTROL_EXPLAINS_GAIN for men's Brier, Brier and NLL.
- **CONTROL_ALSO_MEETS_RESERVE:** False at 50, False at 70, True at 100.
- **Other labels:** reproduction is REPRODUCED_BITWISE, and the encoding effect is ENCODING_EFFECT_UNRESOLVED.
- **Records:** the training records (22 size fits, 18 controls, fixed effort and seeds), class support (my counts equal FIT.json for 22 fits) and Stage B update accounting (equal to deviation 3) all agree.

**V5 selection, nested estimator, releases and readers.**
- **Selection:** I rebuilt each nested table from the stored out-of-fold arrays, matching within 1.7e-18 (SE within 4.3e-19). My own fixed-sequence rule gives s* = 0.3 for LOC and PAIR in both versions, first failure at 0.5, no PASS_AFTER_FAIL. The CONT tables (descriptive) also replay.
- **One nested outer fold end to end (V0, fold 3, declared beforehand):**
  - the 90% witness (T* 144) and its cross-fitted base are bitwise;
  - the CatBoost routing has 794 trees and identical routing;
  - the source P2 recipe on my own inputs gives identical θ for both LOC and PAIR, and the held-out Δ is bitwise.
- **Full-pool v0:** my own cross-fitted base is bitwise equal to BASE.npz and to the source CROSSFIT.npz. The deployed CONT/LOC/PAIR θ hashes are identical.
- **Releases:**
  - Δ matches exactly, and r_s matches within 5.6e-16 on every arm, strength and role.
  - The reader bank members are exactly the selected releases at 0.3, with W = W100 and income = H3.
- **Invariance (RULES 8.1):** θ is bitwise invariant to `targets` under guard 'none'.
- **Readers:**
  - All 73 pairs' cells and SEs equal GRID.json.
  - Across 336 cache replays, winners, SCORE AUCs and all 400 bootstrap draws match within 1e-12.
  - My OUT_LR/LOG_LR refits are bitwise.
  - My seed-check equalities are exactly the production's 16 (ET5/ET25, income view, B2TWIN/RETIRED_B3CTL).
  - The validity labels are literal NOT_EFFECTIVE and A-SC1 EFFECTIVE, and my distinct-input counts equal WIRE_CARDINALITY.json.
  - My labels equal CLAIMS.json on all 39 compared pairs.

**V6 utility and claims.**
- **COND_UTILITY:** point status, reserve, ratios and realized costs match.
- **Claims:** they match exactly.
  - C1 is MARGIN_RESTORED_BY_TRAINING_SIZE_ON_THIS_POOL.
  - Both arms are C2/C2p UTILITY_PRESERVED_BOTH_VERSIONS.
  - C3, C3w, C4, C5 and C6 are NOT_MET; C7 is NOT_CLAIMABLE and C8 NOT_CLAIMED.
  - The outcome is CORRECTION_RECIPE_CLOSED.
- **Tables:** the equal-reserve tables and frontiers, the original-benchmark rows and the COARSE flags all match.

## Procedural compliance
- **Exposure (V4.ACCESS, final run at 01:53Z): PASS.**
  - Every MODEL_FIT read equals my reconstruction of its subset.
  - No closed role was read, no read precedes the receipt, and no fitting unit read MODEL_SELECT.
  - A1 fits read occupation only, and every read falls inside its unit interval.
  - No MODEL_SELECT read of a corrected release precedes the selections: the first, at 01:30:12Z, is after the commit at 01:18:34Z.
- **Order (V6.PROCESS.ORDER): PASS.** Registration, then receipt, then reads. DEPLOY_W100, then the trigger, then the correction stage. Code is committed before its units. Selections come before the grid (01:21:11Z), and the reader code still equals its lock.
- **Restored deployment: PASS, 11 of 11.** I re-hashed backup B1's manifest myself and extracted it into my own empty directory. My checker then imported the restored tsz and data and ran all packages: bitwise, 24/24 refusals each, and my own pipeline matches (bitwise for A1/CTRL, ≤1e-12 for CORR). This is a warm, same-device restore. I deleted the extraction afterwards.

**ATTENTION items.**
- **Null-unit reads:** the two selftest reads with no unit id match deviation 4.
- **V5.READERS.SEED_EFFECTIVENESS__V1:** this used the minimum-set scope and is superseded by the A-SC1 agreement check.
- **V5.CORRECTION.TARGETS_INVARIANCE__V1:** my sensitivity control was badly designed; change 2, A4.
- **V6.TEXT_SCOPE_SCAN__V2.** No forbidden claim was found, the scope sentences and the A-SC1 disclosure are present in RESEARCH_DECISION.md, and every quoted number I could replay matches. Six wording points remain:
  - **Causal phrasing:** "explained by training size" and "is a training-size effect" (START_HERE) read causally beyond the C1 scope.
  - **Recipe-specific claim:** "not a recipe-specific one" overstates an unresolved DiD.
  - **"not resolved for any metric":** the women's-Brier DiD against TASK_REFIT is VERSION_DEPENDENT; it is resolved in v1.
  - **Binding reserve:** "binds the whole-release reserve of every H3-anchored release at 2.86" is false for TT100 (2.60, occupation side) and for the 50% fits.
  - **Reference ratios:** "0.32–1.95" is a per-version range; the table range is 0.32–1.56.
  - **Disclosure sentence:** the A-SC1 sentence is only paraphrased in ADVISOR_BRIEF.md and absent from START_HERE.md.

**Disclosure about my reader labels.** I stored my own stability, target and gate labels (V5.READERS.LABELS__V1) at 01:32:47–01:33:09Z.
- That was after READER_SEED_CHECK.json was written (01:31:29Z) and before A-SC1 (01:34:35–01:36Z).
- My first message about this wrongly said "before the seed-check record existed"; it is corrected as item 12.
- The labels were computed with no validity gate, which means they correspond to the A-SC1 status. Nobody read them before the ruling. They equal CLAIMS.json exactly.

## Not run or not covered
- Stage B TASK_REFIT on subsets was not refitted (torch training). Its exposure rests on the access log, the records and the S100 identity.
- Only nested fold 3 of v0 was recomputed end to end. The full-pool directions were recomputed for v0 only.
- Only the OUT_LR and LOG_LR readers were refitted; the other recipes were replayed from the cache.
- Not verified:
  - the CALIBRATION diagnostic temperature;
  - REPRO's model-file identity (I compared arrays);
  - the CTS1 42–53% men's-excess values quoted in RESEARCH_DECISION;
  - the correction investigator's 88 selftests;
  - any cold or off-device restore.

## Resources and files
I used 23 queue units for 1,302.8 CPU-s, plus 150 CPU-s charged for light work (about 0.40 CPU-h). One unit failed: VER_REFIT_A1_B2_V1, a record-name collision in my code; it was rerun as B3.

Files are in `results/pcrl_controlled_training_size_v1/roles/verifier/`:
- CHECKS.json (121 checks)
- CHECK_CHANGES.json (changes 1–3 accepted, 4 pending)
- DEPLOY_W100.json
- records/ (V*_*.json, NOTES.jsonl, history/)
- code/ (vlib.py and 28 scripts)
