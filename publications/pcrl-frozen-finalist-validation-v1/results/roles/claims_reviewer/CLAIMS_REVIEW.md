# Claims and deployment review: PCRL frozen-finalist validation, before publication

**Role.** Deployment and claims reviewer (a separate agent), 2026-10-17, about 22:35–23:00Z. I am adversarial but fair. I did not edit any campaign file.

**Context.** This is statistical machine-learning research on public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, using the existing development roles only.
- The service predicts an income class (2 classes) and an occupation group (5 classes) from AGEP, WKHP, COW, SCHL, MAR and POBP.
- Each recipient receives only its own float64 probability vector and argmax decision.
- SEX is the dataset's recorded demographic column. Separate evaluation readers try to infer it from the released predictions. This is defensive evaluation of a privacy-protecting release.

**Versions reviewed.**
- The coordinator's revised versions (about 22:44Z) of `START_HERE.md`, `RESEARCH_DECISION.md` and `CONFIRMATION_FEASIBILITY.md`, checked against `CONFIRMATION_FEASIBILITY_RESULTS_V2.json`.
- `PROTOCOL_DEVIATIONS.md` (items 1–10).
- The unchanged `QUICKSTART.md` and `OUTPUT_MAPPING_CHECK.md`.

The reviewed documents were initially read in their earlier versions and re-read after the coordinator's note. Every quotation below is from the current files.

**Sources checked.**
- `report/DEV_REPORT_RESULTS.json` and `PRIMARY_ENDPOINTS.csv` (both scorecards);
- `ORIGINAL_VS_ENHANCED_READERS.csv`, `READER_MATCHED_BENEFITS.csv` and `FORECAST_SCORES.json`;
- `CONFIRMATION_FEASIBILITY_RESULTS.json` (V1) and `_V2.json`, including the saved V2 curves;
- `OUTPUT_MAPPING_RESULTS.json` and `ffv/mapping.py`;
- `DEPLOYMENT_CHECKS.json`, `BACKUP_DEPLOYMENT_CHECK.json` and `FROZEN_CANDIDATE_MANIFEST.json`;
- `PROTOCOL.md` §§8, 10, 12 and 13;
- `roles/statistical_reviewer/PROTOCOL_REVIEW.md` and `FEASIBILITY_REVIEW.md`;
- skims of `EXPOSURE_AND_SELECTION_AUDIT.md` and `POPULATION_INVENTORY.md`;
- code: `ffv/deploy_check.py`, `ffv/backup_deploy_check.sh`, `ffv/feasibility.py`, `amd/deploy.py` (`check_rows`, `refusals`), `amd/fam_b.py` (`package_release`, `coarse_bins`), `amd/restore_check.py`, `amd/common.py`, `scm_bench/common.py` and `acs_replication/deploy.py`.

**Not done.**
- No SEX or task label, no DEV_REPORT truth and no sealed-assessment content was opened.
- Private files opened:
  - the admitted FINALIST deployment packages, only through `amd.fam_b.package_release` and `amd.deploy.release` on invented rows;
  - the label-free, identifier-free per-person threshold-tie detail (`.private/.../runs/DEPLOY/DEPLOYMENT_CHECKS_DETAIL.json`), to check one sentence;
  - the queue ledger;
  - two coordinator scratchpad scripts named in it.

**My spot check.** Queue unit `CR_DEPLOY_SPOT_V1` (stage `review`, 3.4 CPU-s charged by the queue) used invented inputs only. Its script and results are reproduced in Appendix A, because the scratchpad copy is not durable.

**Verification status at review time.**
- `INDEPENDENT_VERIFICATION.json` does **not** exist yet; `roles/verifier/` holds only `code/`.
- `roles/statistical_reviewer/FEASIBILITY_REVIEW.md` exists. The V2 documents adopt its corrections E1–E5 (consistency notes in §5).

## Summary

**Most numbers match their sources**, in every document reviewed. That includes:
- the reproduction (70 endpoints, maximum difference 0.0, 8 of 8 bitwise);
- the claim labels;
- the pair benefits and bounds;
- the utility slack and 25 of 28 unresolved;
- the reader-matched ranges;
- the output-mapping per-role, union and domain counts;
- the threshold ties;
- almost every V2 feasibility cell.

**14 must-fix items:**
- 1–4: feasibility arithmetic and wording, including one number that is wrong against V2 (M1) and one claim that contradicts the endpoints (M2);
- 5–7: missing qualifiers and the NEURAL_RAW seed-1 coverage gap;
- 8–10: QUICKSTART deployment statements;
- 11: a publication gate (the referenced verification and closeout files do not exist yet);
- 12–13: output-mapping sourcing, including one count that is wrong (M12);
- 14: an information-claim wording.

**Recommendation.** Option 3 (retire as a competitive candidate; retain the evidence and the runnable artifact) is the best-supported of the four options. It should be stated as conditional on the independent verification (§4).

---

## 1. MUST-FIX

### M1. Joint power at the ~17,500 available households is misstated (wrong number)

- **Files and exact text.**
  - `RESEARCH_DECISION.md`, item 4:
    > At the size of the only Florida 2019 people who could test this release (about 17,500 households), joint power is at most about 0.2 for D1 and about 0 for D2.
  - `CONFIRMATION_FEASIBILITY.md`, Conclusion 1:
    > At the size of the only Florida 2019 people who could test this release (about 17,500 households), joint power is at most about 0.2 for the source scorecard and about 0 for the confirmatory version.
- **Source of truth.** `CONFIRMATION_FEASIBILITY_RESULTS_V2.json`, `scenarios["S1_plugin|D1"].curve`, interpolated at 17,510 households:
  - ENHANCED: **0.013**. This is the governing (less favourable) scorecard.
  - ORIGINAL: **0.252** (0.248 at the 16,587 grid point, 0.256 at 18,376).
  - The quoted "0.2" is the value at 10,000 households (ORIGINAL 0.215; `power_at_interpolated["10000"]`), which is what the "Study sizes that exist" table correctly shows.
- **Correction.**
  > At about 17,500 households, D1 joint power is about 0.01 under the governing ENHANCED scorecard (about 0.25 under ORIGINAL); D2 is about 0.

  Alternatively, keep the 10,000-household figures and say so. Optionally add the scenario context: at an assumed true slack of 0.002, ORIGINAL D1 at 17,500 households would be about 0.57, against ENHANCED 0.03 (same curves).

### M2. "Recovery is not binding" contradicts the ENHANCED recipient clause

- **Files and exact text.**
  - `RESEARCH_DECISION.md`, item 4:
    > **What binds.** The binding constraint is the utility slack, not recovery: the recovery clauses alone pass at DEV's size.
  - `START_HERE.md`, feasibility row:
    > The binding constraint is utility, not recovery.
  - `CONFIRMATION_FEASIBILITY.md`:
    > The recovery clauses are not the bottleneck.
- **Source of truth.**
  - `PRIMARY_ENDPOINTS.csv`: ENHANCED `COMPETITIVE|income_worsening_vs_H_LJ|s1` is **UNRESOLVED** at DEV's size (point 0.0076, clause bound 0.0235 > 0.010). It is the clause that broke claim B, so not every recovery clause passes at DEV's size.
  - `FEASIBILITY_REVIEW.md` §2: its single-clause n80 is 317,958 households.
  - `CONFIRMATION_FEASIBILITY.md` itself: "With a true slack of 0.002 or more, the ENHANCED D1 requirement (323,000) is set by its recipient clause against H_LJ".
  - V2 S5 D1 ENHANCED n80 is 323,219 at both 0.001 and 0.002.
  - Joint power cannot exceed that clause's marginal power, and the ENHANCED requirement governs, so D1 (and D2) cannot fall below about 3.2×10^5 households under any utility headroom, with that clause at its plug-in.
- **Correction.**
  > At the plug-in utility slack the binding constraint is income utility. The four pair-benefit clauses pass at DEV's size, but the ENHANCED seed-1 income recipient clause against H_LJ does not: alone it needs about 318,000 households, and it sets the ENHANCED requirement whenever the true utility slack is 0.002 or more.

  This slack-independent floor also strengthens the option-1 rejection (§4).

### M3. The item-4 table in RESEARCH_DECISION omits D2 and the governing value at slack 0.002

- **File and exact text.** `RESEARCH_DECISION.md`, item-4 table:
  > | 0.002 or more | about 4×10^4 (ORIGINAL) or 3.2×10^5 (ENHANCED) households | — |
- **Source of truth.** `CONFIRMATION_FEASIBILITY_RESULTS_V2.json`:
  - S5 slack 0.002, D2 n80: ENHANCED 396,683; ORIGINAL 323,219. The "—" hides an existing value.
  - `governing_rule`: the less favourable governs, so for D1 the ENHANCED 323,219 governs.
  - `FEASIBILITY_REVIEW.md` §3 item 1 requires "(ENHANCED, **limited by its recipient clause**)", which is dropped here.
  - The registered slack grid stops at 0.002. Above it, ORIGINAL requirements were not computed and would be lower, while ENHANCED cannot fall below about 318,000 (M2).
- **Correction.**
  > | 0.002 (largest value on the registered grid) | about 3.2×10^5, set by the ENHANCED recipient clause against H_LJ (governs) [ORIGINAL about 4×10^4] | about 4.0×10^5 [ORIGINAL 3.2×10^5] |

  Apply the same change to `CONFIRMATION_FEASIBILITY.md` Conclusion 1: "...if that slack is at least 0.002..." becomes "...at an assumed slack of 0.002, the largest on the registered grid...".

### M4. The CONFIRMATION_FEASIBILITY results header misstates which scorecard governs

- **File and exact text.** `CONFIRMATION_FEASIBILITY.md`:
  > ## Results (ENHANCED governs; ORIGINAL in brackets where it differs)

  and the slack-0 row:
  > | 0 | does not reach 80% at any n: limit 0.266 [0.268] | limit 0.0004 | limit 0.0004 |
- **Source of truth.** V2 `governing_scorecard_by_scenario` gives **ORIGINAL** for:
  - `S1_plugin|D3`;
  - `S5_binding_true_slack_{0.0005, 0.001, 0.002}|D3`;
  - `S5_binding_true_slack_0.0|D2` and `|D3`;
  - `S3_binding_at_cap|D2` and `|D3`;
  - `S3_binding_halfway_to_cap|D2` and `|D3`;
  - `S4_plugin_tau_reader_refit_rough|D3`;
  - `S4_..._plus_release_retraining_HYPOTHETICAL|D1` and `|D3`.

  The document's own "Scorecards" paragraph states the rule correctly, so the header contradicts it. In the slack-0 row, ORIGINAL differs and governs but is not bracketed: D2 limit 0.0003 (`limit_n_to_infinity_mc` 0.00032); D3 limit 0.
- **Correction.**
  - Header: "Results (ENHANCED shown; ORIGINAL in brackets where it differs; the less favourable governs, which is ORIGINAL in the D3 column, the D2 boundary cells and the combined-term D1 cell)".
  - Slack-0 row: "limit 0.0004 [0.0003]" and "limit 0.0004 [0]".

### M5. START_HERE lacks the status qualifiers

- **File and exact text.** `START_HERE.md`; the results header is the only status statement:
  > ## Results (one logged follow-up DEV_REPORT evaluation, 22:27:16Z; seed 0 / seed 1)

  Nowhere in START_HERE does it say that these are development robustness results on already-used records, that this is not confirmation, that the source's adaptive selection history is not erased, that DEV_REPORT was used before, or that the seeds share people.
- **Source of truth.**
  - `report/DEV_REPORT_RESULTS.json` `status`: "DEVELOPMENT ROBUSTNESS EVIDENCE (follow-up re-read of an already-exposed role; not confirmation; the source adaptive selection history is not covered)".
  - `EXPOSURE_AND_SELECTION_AUDIT.md` §4: "10 DEV_REPORT truth reads before this campaign: 5 coordinator evaluations and 5 verifier replays"; "This campaign adds two logged follow-up re-reads".
  - `CONFIRMATION_FEASIBILITY.md`: "The two seeds share people and are not replications".
- **Correction.** Add a status line directly under the results header:
  > Status: development robustness evidence on already-used records. DEV_REPORT truth had been read 10 times across the lineage before this campaign (5 evaluations, 5 replays); this campaign adds one logged coordinator re-read and one verifier replay. It is not confirmation and does not erase the source's adaptive selection history. Seed 0 and seed 1 are two fits of each release evaluated on the same people, not replications.

### M6. RESEARCH_DECISION does not say that the seeds share people

- **File and exact text.** `RESEARCH_DECISION.md`:
  - it reports "seed 0 / seed 1" throughout;
  - "on each seed and against each reference" (Descriptive support);
  - "Status of the evidence" has no seed qualifier.
- **Source of truth.**
  - `PROTOCOL.md` §8: bootstrap weights "shared across ... both seeds, which share people".
  - `PROTOCOL_REVIEW.md` finding 4: "'both seeds' is robustness to the release seed, not replication".
- **Correction.** Append to "Status of the evidence":
  > The two seeds are two fits of each release evaluated on the same DEV_REPORT people with shared bootstrap weights; agreement across seeds is robustness to the release seed, not replication.

### M7. The NEURAL_RAW seed-1 deployment gap is missing where deployment is summarized

- **Files and exact text.**
  - `START_HERE.md`:
    > All 7 packages redeploy bitwise, including from the source backup alone.
  - `RESEARCH_DECISION.md`, item 1:
    > All seven deployment packages reproduce their evaluated DEV_REPORT outputs bitwise, through the wrapper API, the CSV path and the underlying function. They also do so from the source's same-device backup alone.
  - `RESEARCH_DECISION.md`, "What remains useful":
    > The frozen packages deploy reproducibly, including from backup.
- **Source of truth.**
  - `DEPLOYMENT_CHECKS.json` `unavailable.NEURAL_RAW_S1`: "no raw-input package exists in the source (deployment coverage gap); its output is admitted by hash".
  - `FROZEN_CANDIDATE_MANIFEST.json` `bank.neural_raw_seed1`.
  - `BACKUP_DEPLOYMENT_CHECK.json` uses the same-device backup only.
- **Correction.**
  > All 7 existing packages (FINALIST and TASKONLY, both seeds; H_LJ, both seeds; NEURAL_RAW seed 0) redeploy bitwise as full-role batches, including from the source's same-device backup. No NEURAL_RAW seed-1 package exists (a coverage gap; that output is admitted by hash only).

  Use "same-device backup" in the "What remains useful" bullet too.

### M8. QUICKSTART overgeneralizes batch reproduction

- **File and exact text.** `QUICKSTART.md` §3:
  > Batch requests reproduce the evaluated outputs bitwise.
- **Source of truth.**
  - `DEPLOYMENT_CHECKS.json` `FINALIST_S{0,1}.batch_size_characterization`: **8-row batches** differ bitwise from the evaluated outputs on 60 of 200 (seed 0) and 47 of 200 (seed 1) rows.
  - `FROZEN_CANDIDATE_MANIFEST.json` `small_batch_last_bit_observation`: a 6-row batch differs on levels 2 and 4 (seed 0) and 3 and 4 (seed 1).
  - Spot check `CR_DEPLOY_SPOT_V1` (Appendix A):
    - 1-, 6- and 8-row requests differ on every tested row of income levels 2 and 4 (seed 0) and 3 and 4 (seed 1), by at most 1.1e-16;
    - all other levels are bitwise equal, and decisions are equal;
    - a 600-row invented batch is bitwise equal to the evaluated per-level vectors.

  The "2 of the 6 levels" and "decisions are unaffected" statements are therefore confirmed. Only the last sentence is too broad.
- **Correction.**
  > Full-role batches (and a 600-row batch in the claims review) reproduce the evaluated outputs bitwise; 1-, 6- and 8-row requests do not (income levels 2 and 4 on seed 0, 3 and 4 on seed 1). The batch size at which the evaluated bits return was not determined.

### M9. QUICKSTART's "strict" wrapper validates keys, not values

- **File and exact text.** `QUICKSTART.md` §3:
  > **The documented wrapper** is strict. Rows must have exactly AGEP, WKHP, COW, SCHL, MAR and POBP, in that order. SEX, extra or reordered keys, and requests for hidden quantities are refused.
- **Source of truth.**
  - `amd/deploy.py` `check_rows` checks only key names and order.
  - `acs_replication/deploy.py` `encode_raw` maps unknown category codes to `__OTHER__` and z-scores numbers with no range check; only non-finite values raise.
  - Spot check `CR_DEPLOY_SPOT_V1`: the wrapper **accepted** AGEP 17, AGEP 200, WKHP 0, COW "X" and POBP "999", and refused only AGEP "nan" (ValueError).
- **Correction.** Append:
  > Strictness covers field names and order only. Values are not checked against the cohort or the encoder vocabulary: out-of-cohort ages and hours are accepted and extrapolated, unknown category codes are encoded as `__OTHER__`, and only non-numeric or non-finite AGEP/WKHP fail. The valid input domain of `OUTPUT_MAPPING_CHECK.md` is not enforced at deployment.

### M10. QUICKSTART public-package consistency and Census context

- **File and exact text.**
  - `QUICKSTART.md`:
    > | Deployment from the source backup alone | `FFV_PYTHON=$PY FFV_PYLIB=<private CatBoost lib> bash ffv/backup_deploy_check.sh <backup dir> <out.json>` | |
  - The title, "# Quickstart: frozen-finalist validation (PCRL, Florida ACS 2019 development roles)", is the only data context in the file.
- **Source of truth.** `PROTOCOL_DEVIATIONS.md` item 7: the shell helper "is not in the public package ... It stays on the research branch". The brief requires explicit Census context in every published document. QUICKSTART mentions SEX only as a refused key, with no explanation.
- **Correction.**
  - Mark the backup row "(research branch only; not in the public package)".
  - Add one line under the title:
    > Data: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only. The service predicts income class (2) and occupation group (5) from the six fields; SEX is the dataset's recorded demographic column, which separate evaluation readers try to infer from the released predictions; it is never a service input.

### M11. Publication gate: the option-4 rejection and feasibility text rest on files that do not exist yet

- **Files and exact text.**
  - `RESEARCH_DECISION.md`:
    > **Why validation is not incomplete (option 4).** Every registered unit ran, every lock was committed and published before the stage it governed, and the independent verifier's result is recorded in `INDEPENDENT_VERIFICATION.json`.
  - `CONFIRMATION_FEASIBILITY.md`:
    > The verifier also checks the calculation (`INDEPENDENT_VERIFICATION.json`).
  - `START_HERE.md` and `QUICKSTART.md` §4 reference `RESOURCE_ACCOUNTING.json`, `INDEPENDENT_VERIFICATION.json`, `roles/verifier/` (`VERIFICATION.md`), `BACKUP_VERIFICATION.json`, `ACCESS_RECEIPT.json` and `HANDOFF.json`.
- **Source of truth.** At about 22:50Z none of these files exists. `roles/verifier/` contains only `code/`, so the verifier is still working. `roles/coordinator/VERIFIER_BRIEF.md` item 8 points the verifier at `CONFIRMATION_FEASIBILITY_RESULTS.json` (V1), not V2.
- **Correction.**
  1. Publish only after `INDEPENDENT_VERIFICATION.json` exists and records no unresolved discrepancy in a primary endpoint, label or feasibility number.
  2. State which feasibility version the verifier checked. V1 and V2 share every n80 and power value; only the limits, assurance limits and the τ labels differ, and those were checked only by the statistical reviewer. Either extend the verifier's check to V2 or say so.
  3. Make the recommendation explicitly conditional:
     > Option 3, provided `INDEPENDENT_VERIFICATION.json` records no unresolved primary discrepancy; otherwise option 4 with the exact missing or failing units.
  4. In the option-4 paragraph, mention the post-lock feasibility correction (`PROTOCOL_DEVIATIONS.md` item 9). "Every lock was committed ... before the stage it governed" stays true, but the feasibility numbers were corrected after the lock.
  5. Confirm `BACKUP_VERIFICATION.json` before QUICKSTART §4 is published.

### M12. OUTPUT_MAPPING_CHECK: the union count of shared occupation vectors is wrong on seed 1 and unsourced

- **File and exact text.** `OUTPUT_MAPPING_CHECK.md`, "Only repeated vectors can test it":
  > - seed 0: 6 such vectors in the union, with no conflict;
  > - seed 1: 49 such vectors in the union, of which 2 carry different income codes.
- **Source of truth.**
  - `OUTPUT_MAPPING_RESULTS.json` has **no** union-level count of occupation vectors shared by distinct inputs. `ffv/mapping.py` `empirical()` computes it per role only.
  - 6 and 49 equal distinct encoded inputs minus distinct occupation vectors in the union (39,585 − 39,579 and 39,585 − 39,536). That difference is the sum of (k − 1) over shared vectors, which equals the number of shared vectors only if each one is shared by exactly two inputs.
  - On seed 1 that fails: within ATTACK_FIT alone there are 7,531 inputs, 7,526 vectors and 4 shared vectors (`occupation_vectors_shared_by_distinct_encoded_inputs` = 4, excess 5). One vector therefore carries three distinct inputs, and the seed-1 union has **at most 48** such vectors.
- **Correction.** Either write "distinct encoded inputs exceed distinct occupation vectors by 6 (seed 0) and 49 (seed 1) in the union", or recompute the exact count (label-free) and record it in a versioned results file. The verifier's check 3 recomputes union counts; reconcile with it.

### M13. OUTPUT_MAPPING_CHECK: the seed-1 cross-role conflict details have no preserved source

- **File and exact text.** `OUTPUT_MAPPING_CHECK.md`:
  > All numbers come from `OUTPUT_MAPPING_RESULTS.json` (unit FFV_MAPPING_V2)

  The section "Across roles, seed 1" then states which roles share each conflicting vector (DEV_REPORT/MODEL_FIT and ATTACK_FIT/MODEL_SELECT) and:
  > In both pairs the encoded inputs differ only in POBP. The occupation vectors are bitwise identical, but the income levels are 2 and 3
- **Source of truth.**
  - `OUTPUT_MAPPING_RESULTS.json` records only the count (`cross_role_occupation_vectors_with_more_than_one_income_code` = 2).
  - The details come from unit `FFV_MAPPING_CONFLICTS_V1` (`PROTOCOL_DEVIATIONS.md` item 1).
  - The queue ledger shows that unit ran a session-scratchpad script (`.../scratchpad/ffv_conflicts.py`), printed to stdout, and registered no output (`output_sha256: {}`). Neither the script nor its result is in the repository.
- **Correction.**
  - Copy the script into the repository (for example `ffv/mapping_conflicts.py`).
  - Save its identifier-free printed result (roles, levels, differing fields) as a results file.
  - Cite `FFV_MAPPING_CONFLICTS_V1` in the sourcing sentence.

### M14. RESEARCH_DECISION partly endorses an information claim

- **File and exact text.** `RESEARCH_DECISION.md`, Output structure:
  > **Consequence:** the source's "no added information" reading holds only as a fact about the observed records within each role. It cannot be extended to future records.
- **Source of truth.** `OUTPUT_MAPPING_CHECK.md` says the within-role relation is "close to vacuous" (92.8–97.1% of occupation vectors occur once). `OUTPUT_MAPPING_RESULTS.json` per role: only 2 (seed 0) and 13 (seed 1) within-role occupation vectors are shared by distinct inputs. The decision's own "Not claimed" excludes information claims.
- **Correction.**
  > ...holds only as a near-vacuous fact about the observed records within each role (93–97% of occupation vectors occur once there; only 2 and 13 within-role vectors on seeds 0 and 1 are shared by distinct inputs). It is not evidence about information content and cannot be extended to future records.

---

## 2. OPTIONAL

- **O1. Qualify "from backup alone".**
  - It is accurate for code and data:
    - `amd.restore_check` ran from the backup's `code.tar`;
    - `load_role`, `admitted` and `frozen_bank` resolve inside the extracted `private.tar`;
    - `HYB_RESULTS` resolves inside the extracted code tree.
  - The interpreter, numpy and scikit-learn, and the CatBoost library (`FFV_PYLIB`) came from the local environment, and the git bundle was verified against this repository. Say so once.
  - Also say:
    - the backup tip is `32e9719f0`, two record-only commits before the cited `9e2b6c567` (`git diff --stat 32e9719f0 9e2b6c567 -- amd acs_replication acs_hybrids scm_bench task_headroom` is empty);
    - the restore check compares against the backup's own stored outputs. Record that these equal the admitted copies by hash.
- **O2. Clarify COMPARATOR_READ_LOWER without weakening it.**
  - "Not added finalist exposure" describes the change between banks: the finalist's seed-1 income reader is unchanged.
  - The failing clause itself is **UNRESOLVED**. Under the ENHANCED readers the finalist's seed-1 income view reads +0.0076 above H_LJ's, and the data allow up to +0.0235.
  - Selection instability is not one-sided:
    - the finalist's own seed-1 income winner sits on an **exact** ATTACK_VALIDATION tie (gap to runner-up 0.0, broken by the source-first rule; `ORIGINAL_VS_ENHANCED_READERS.csv`);
    - H_LJ's seed-0 income winner also changed (−0.0085; marginal 95% −0.0146 to −0.0025);
    - the seed-1 H_LJ change of −0.011 has a marginal 95% interval of −0.024 to +0.001.
  - One sentence in RESEARCH_DECISION item 2 would prevent "not added exposure" being read as "no more exposed than H_LJ".
- **O3. Put the refit evidence next to "survived".** START_HERE ("The pair benefits do") and RESEARCH_DECISION item 2 ("The pair benefits survived this larger bank") are accurate for the frozen fitted readers. With household-bootstrap-trained readers (`DEV_REPORT_RESULTS.json` `per_seed.*.variation.benefit_matched_refits`):
  - individual benefits ranged from **−0.0011** (NEURAL_RAW, seed 0) to +0.031;
  - the seed-1 H_LJ refits gave 0.006–0.019 against the frozen +0.026.
  - Add "against these frozen fitted readers; with bootstrap-trained readers the benefit shrank in 3 of 4 cells (`CONFIRMATION_FEASIBILITY.md`)".
- **O4. Report M's frozen interpretation.** It is `READER_SENSITIVE` (via M_SS, side `COMPARATOR_READ_LOWER`), with ENHANCED label `TASK_ONLY_ATTRIBUTION_ONLY` (`DEV_REPORT_RESULTS.json` `interpretation.M`). RESEARCH_DECISION gives only the ENHANCED label and the step results.
- **O5. Threshold-tie precision** (QUICKSTART §3 and OUTPUT_MAPPING_CHECK).
  - "No one in any other role is within 4.5e-6" is technically true, but the closest person is 4.51e-6 away (ATTACK_FIT, seed 0). Write "the closest is 4.5e-6 (ATTACK_FIT, seed 0)".
  - Add that neither seed-1 move crosses the income decision: the levels change 0→1 and 1→2, with package values 4.4e-16 and 2.2e-16 below the exact sum (private detail; the "2–4e-16 below" wording is confirmed).
- **O6. Batch-size coverage.**
  - OUTPUT_MAPPING_CHECK says "the occupation output and H_LJ are unaffected". Only H_LJ seed 0 was tested.
  - TASKONLY uses the same family-B arithmetic (`package_release` → standardized one-hot × w) and was not tested, so it probably shares the dependence. Say both.
- **O7. QUICKSTART wording.**
  - "only a missing field fails" should read "among key problems, only a missing field fails (KeyError); non-numeric or non-finite AGEP/WKHP also raise".
  - The refusal suite's "extra key/column" case uses PUMA, which is a forbidden field, so the wrapper refuses it as forbidden (PermissionError). The generic extra-key path (keys ≠ PERMITTED → ValueError) is established by code reading, not exercised.
- **O8. Label collisions.** "the D2 extraction disclosure" and "D2, confirmatory" appear in the same document, as do "source D3" (threshold ties) and "D3" (the 0.020 definition). Prefix them, for example "replication disclosure D2" and "source deviation D3".
- **O9. Census context wording.** `CONFIRMATION_FEASIBILITY.md` and `OUTPUT_MAPPING_CHECK.md` say "Public-use ACS 2019 Florida person records". Write "public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida".
- **O10. Conflict wording.** RESEARCH_DECISION has "two occupation vectors carry different income codes, one on each side of the income decision". Write "two occupation vectors each carry two income codes (levels 2 and 3), which straddle the income decision".
- **O11. Stale condition string.** In the frozen `DEV_REPORT_RESULTS.json`, the ENHANCED claim-B CV condition string reads "source-definition OUT_HGB planted and shuffled controls". The ENHANCED CV value follows the A5 rule (`control_validity`: all 5 recipes × 3 views × 2 seeds in band). Disclose this in PROTOCOL_DEVIATIONS; the file cannot be regenerated.
- **O12. State the τ values used in S4.** The documents quote component SDs (0.003–0.011). S4 adds τ_r = 0.0036–0.0134 (quadrature of the seed and bootstrap terms; V2 `tau_reader_refit_rough`). The HYPOTHETICAL τ_h is 0.0017 / 0.0037 for ENHANCED, but 0.0024 / 0.0050 for ORIGINAL.
- **O13. Avoid near-"impossible" wording.** CONFIRMATION_FEASIBILITY says "It is not achievable in any study of this population". Prefer "it exceeds the number of households in Florida" (factual and conditional on the plug-in), in line with `FEASIBILITY_REVIEW.md` §3.
- **O14. Count the DEV_REPORT reads.** RESEARCH_DECISION says "DEV_REPORT was read once more". Add the lineage count (10 prior truth reads) and the verifier's replay, as in M5.
- **O15. Custody of the tie aggregation.** `DEPLOYMENT_CHECKS.json` as published was rewritten in place by a scratchpad script (`ffv_aggregate_ties.py`) that dropped the per-person `detail`. The detail is preserved privately (`runs/DEPLOY/DEPLOYMENT_CHECKS_DETAIL.json`, matching PROTOCOL_DEVIATIONS item 7). Commit the aggregation script with the campaign code.

---

## 3. Deployment review

**(a) API versus wrapper: stated correctly.**
- QUICKSTART §3 and `DEPLOYMENT_CHECKS.json` (`refusals_enforced_only_by_wrapper`: `api_extra_key`, `api_reordered_keys`, `api_sex_key`) attribute strict key rejection to the wrapper.
- The spot check confirms the underlying `amd.fam_b.package_release`:
  - **accepts** a SEX key (value 1 or 2), an extra key and reordered keys;
  - its outputs are **bitwise identical** to the request without them, so "silently accepted and ignored" is shown, not just inferred;
  - a missing key raises KeyError.
- The wrapper refuses:
  - SEX and PUMA as forbidden fields (PermissionError);
  - reordered or missing keys (ValueError).
- Hidden-output requests are refused by both layers. The CSV path is strict in both, through `read_raw_csv`.
- Gaps: value validation (M9) and the refusal-suite nuance (O7).

**(b) Batch size and threshold ties: characterized, not repaired; accurate except one sentence.**
- "At most 1.1e-16", "2 of the 6 levels per seed" and "decisions are unaffected" are all confirmed for 1-, 6- and 8-row requests (Appendix A).
- "Batch requests reproduce the evaluated outputs bitwise" is too broad (M8).
- Threshold ties match the public aggregate and the private detail: 2 (seed 0) and 4 (seed 1) MODEL_FIT people sit exactly on a frozen threshold, and 2 seed-1 people move one level under either alternative summation.
- No document claims a repair. Thresholds, calibration and summation are unchanged, and the midpoint variant is correctly called a different release.

**(c) NEURAL_RAW seed-1 coverage gap.**
- Stated in QUICKSTART, `DEPLOYMENT_CHECKS.json` and the manifest.
- Missing from START_HERE and RESEARCH_DECISION (M7).

**(d) "From the source backup alone": accurate for code and data.**
- `ffv/backup_deploy_check.sh` re-hashes the backup manifest, verifies the bundle, extracts `code.tar` and `private.tar` into a fresh directory, and runs the backup's own `amd.restore_check`.
- All data paths that check uses resolve inside the extracted copy (`amd/common.py`: `SRCP`, `admitted`, `frozen_bank`, `HYB_RESULTS`).
- The environment came from the local machine (O1).
- `BACKUP_DEPLOYMENT_CHECK.json`: 7 of 7 bitwise; `manifest_rehash_ok`, `bundle_verify` and `all_ok` all true.

**Custody and the off-device state.**
- QUICKSTART §4 says the external drive was not mounted, so no off-device copy was made and no cold check was possible. It does not describe a warm read-back as cold.
- This is consistent with `POPULATION_INVENTORY.md` ("/Volumes lists no backup drive") and with `/Volumes` at review time (no external drive).
- `BACKUP_VERIFICATION.json` is not written yet (M11).

---

## 4. Does the recommendation follow? (options of `PROTOCOL.md` §13)

**Option 3 (retire as a competitive candidate; retain the evidence and the artifact) is the best supported.**
1. The registered claim that made the candidate competitive (B: superior to all eligible references) is `READER_SENSITIVE` under the frozen interpretation rule. M degrades to `TASK_ONLY_ATTRIBUTION_ONLY` (M itself is `READER_SENSITIVE`). J and the 0.020 target are not established in either bank.
2. Its income utility sits 0.03–0.11 SE from the caps, so even the source scorecard needs millions of households at the plug-in.
3. More decisively, under the governing ENHANCED scorecard, the seed-1 recipient clause against H_LJ alone needs about 3.2×10^5 households **whatever the utility headroom** (M2). The only untouched-model Florida 2019 people number about 17,500 households.
4. Bootstrap-trained readers shrank the benefit in 3 of 4 cells, so the advantage is reader-relative.
5. The output-mapping check removes the "no added information" reading beyond the observed records.

**Option 2 (preserve as a reader-sensitive engineering result, without confirmation yet).**
- Its strongest case: the frozen label is literally `READER_SENSITIVE`, and the four pair-benefit bounds held in both banks.
- But "without confirmation yet" presumes a path to confirmation that the arithmetic does not support (points 2–3).
- Option 3 already retains the evidence and the runnable artifact, so nothing of value is lost.

**Option 1 (preserve and prepare a prospective study)** is excluded by points 2–3. `PROTOCOL.md` §13 also bars treating a fresh population as a cure.

**Option 4 (validation incomplete) is the live alternative today.**
- The independent verification is not recorded, and the closeout files are absent (M11).
- If the verifier reports an unresolved discrepancy in a primary endpoint, label or feasibility number, option 4 is the correct call.
- The recommendation should therefore read "option 3, conditional on `INDEPENDENT_VERIFICATION.json`".

**Overreach check.** No document claims any of the following:
- arbitrary-reader protection;
- a privacy guarantee;
- a population-optimal statement;
- information removal (apart from M14's wording);
- confirmation;
- that a hypothetical variance was measured (the HYPOTHETICAL label is kept everywhere);
- equivalence from UNRESOLVED;
- that confirmation is impossible under every true effect;
- that COMPARATOR_READ_LOWER is finalist exposure.

---

## 5. Consistency with the statistical reviewer and the verifier

**`FEASIBILITY_REVIEW.md`.** The V2 documents adopt E1–E5:
- correlated limits 0.266 [0.268];
- assurance limits 0.102 / 0.161 and 0.053 / 0.064;
- the "rough" τ;
- saved curves;
- the governing tie-break;
- the reviewer's single-clause n80s (318,000; 522,000).

Every V2 number I checked matches, except M1 and M4. Two residual differences:
- RESEARCH_DECISION drops the reviewer's mandated qualifier "limited by its recipient clause" (M3).
- The statements that recovery does not bind ("The binding constraint is the utility slack, not recovery: the recovery clauses alone pass at DEV's size") come from neither reviewer file. They contradict the statistical reviewer's §2 table, which gives the ENHANCED recipient clause its own n80 of 317,958 (M2).

**`INDEPENDENT_VERIFICATION.json`.** Absent at review time; the verifier is in progress. When it appears, check:
- that it covers V2 (M11);
- its union and conflict recomputation against M12 and M13;
- its check-9 reading of the documents against this list.

---

## 6. Checked and consistent (no change needed)

- **START_HERE** results rows on reproduction, the pair benefits and claim B, utility, the 0.020 target, the mapping and the populations. M5 and M7 apply to the first row.
- **RESEARCH_DECISION items 1–3.** In particular:
  - 414 copies; 70 endpoints with difference 0.0; labels E, B, M and J; 8 of 8 bitwise; controls bitwise;
  - the benefit table (+0.027 / +0.022 with bounds 0.0135 / 0.0069; +0.028 / +0.026 with bounds 0.0191 / 0.0140; ORIGINAL +0.0199 with bound 0.0059, +0.025 with bound 0.0129);
  - +0.0076 / 0.0235;
  - HGB15 0.5523 against LOG_MLP 0.5517, and 0.549 → 0.538;
  - M_TO +0.051 / +0.048;
  - reader-matched: 14 of 15, +0.005 to +0.034, CONF_HGB −0.015 to −0.002, LOOKUP about 0;
  - utility 0.00492 / 0.00494 and 0.00959, slack 0.04 / 0.03 / 0.11 SE, 25 of 28 unresolved;
  - guards 0.0032 / 0.0040;
  - populations 9,853 and 7,657.
- **CONFIRMATION_FEASIBILITY (V2).**
  - binding slack table, including posterior 0.51 / 0.52 / 0.54;
  - slack-grid cells for 0.002, 0.001, 0.0005, plug-in and −0.001;
  - 0.75 and 0.5 shrink rows, halfway row, both S4 rows (with the [5.7 M] and [0.964] brackets);
  - assurance limits; same-size replication 0.003 / 0.115;
  - 10,000-household power 0.007 [0.215];
  - refit spread 0.003–0.011; all 24 bootstrap refits lower; 3 of 4 cells shrank; τ_h 0.0017 / 0.0037.
- **OUTPUT_MAPPING_CHECK.**
  - every per-role, union and domain count; the 43% / 48% shares;
  - 9,000 / 8,400 pairs, 689 / 1,610 counterexamples, 600 base rows;
  - the WKHP 72 / 74 example (0.431 / 0.551);
  - the WKHP 3 / 4 and POBP 046 / 217 examples;
  - Part 3 accuracies and the about-10% overlap;
  - batch counts 60 / 47.
  - Exceptions: M12 and M13.
- **QUICKSTART.** Package list, the API-versus-wrapper text, tie counts and the custody statement, except M8–M10.
- **FORECAST_SCORES.json.** Outcomes match the results; mean Brier 0.0674.

---

## Appendix A. Spot check `CR_DEPLOY_SPOT_V1` (label-free; invented inputs only)

**Command.**
```
python3 -m ffv.jobqueue run --unit CR_DEPLOY_SPOT_V1 --stage review --rss-gib 2 -- \
  /Users/<you>/PCRL/.venv/bin/python <scratchpad>/cr_deploy_spot.py <scratchpad>/cr_deploy_spot.json
```
The ledger records return code 0, 3.4 CPU-s and wall 3.9 s at 2026-10-17T22:49:33Z.

**Method.**
- 600 invented rows per seed: AGEP 25–64, WKHP 1–99, COW/SCHL/MAR/POBP drawn from the package encoder vocabulary, rng seed 20261017 + s.
- Run through the admitted `FINALIST_S{s}` package.
- Each row's level is identified by matching the manifest's per-level income vectors, which equal the evaluated release bitwise.
- Up to 10 rows per level are re-requested one at a time and in 8-row batches; one row per level is requested as a 6-row batch.
- Key and value handling are checked on one invented row (seed 0).

**Results.**

| Check | Seed 0 | Seed 1 |
|---|---|---|
| 600-row batch: rows not bitwise equal to a manifest level vector | 0 | 0 |
| Levels whose rows differ in 1-row and 8-row requests (10 of 10 rows each) | 2, 4 (max 5.6e-17, 1.1e-16) | 3, 4 (max 1.1e-16, 5.6e-17) |
| Levels differing in a 6-row request (one row per level) | 2, 4 | 3, 4 |
| Income decisions equal everywhere | yes | yes |

Seed 0 key handling (underlying `package_release` versus wrapper `amd.deploy.release`):

| Request | Underlying function | Wrapper |
|---|---|---|
| + SEX = 1 or 2 | accepted, output bitwise equal | PermissionError |
| + PUMA (extra) | accepted, output bitwise equal | PermissionError (forbidden field) |
| reordered keys | accepted, output bitwise equal | ValueError |
| POBP missing | KeyError | ValueError |

Seed 0 value handling, wrapper:
- **accepted:** AGEP 17, AGEP 200, WKHP 0, COW "X", POBP "999";
- **refused:** AGEP "nan" (ValueError).

**Script.** This is a condensed copy, logically equivalent to the file that ran (`cr_deploy_spot.py` in the session scratchpad, which is not durable). It has the same RNG call order, row selection and requests; the run's output also recorded `invented_rows` = 600 and printed the JSON. It reads no label and writes nothing in the repository.
```python
import json, sys
import numpy as np
ROOT = '/Users/<you>/PCRL/.claude/worktrees/pcrl-frozen-finalist-validation-v1'
sys.path.insert(0, ROOT)
from ffv import common as C
from amd import fam_b, deploy as AD
OUT = sys.argv[1]
man = json.load(open(ROOT + '/results/pcrl_frozen_finalist_validation_v1/FROZEN_CANDIDATE_MANIFEST.json'))
PERM = list(AD.PERMITTED)
res = {'labels_read': False, 'inputs': 'invented only', 'seeds': {}}
def vocab(meta):
    out = {}
    for n in meta['encoder']['feature_names']:
        if '=' in n:
            c, v = n.split('=', 1)
            if v != '__OTHER__':
                out.setdefault(c, []).append(v)
    return out
def invented_rows(meta, n, rng):
    voc = vocab(meta); rows = []
    for _ in range(n):
        r = {}
        for c in PERM:
            r[c] = (str(int(rng.integers(25, 65))) if c == 'AGEP' else str(int(rng.integers(1, 100))) if c == 'WKHP'
                    else str(rng.choice(voc[c])))
        rows.append(r)
    return rows
def try_call(fn):
    try:
        return 'ACCEPTED', fn()
    except Exception as e:
        return type(e).__name__, None
for s in C.SEEDS:
    pkg = C.ADM / f'deploy/FINALIST_S{s}'; meta = json.loads((pkg / 'PACKAGE.json').read_text())
    lv = np.array([[float.fromhex(x) for x in row] for row in man['per_seed'][f'S{s}']['per_level_income_probability_hex']])
    rows = invented_rows(meta, 600, np.random.default_rng(20261017 + s))
    pinc = fam_b.package_release(pkg, rows)['income']['probability']
    lev = np.array([int(np.argmin(np.abs(lv - p).max(1))) for p in pinc])
    big_bitwise = np.array([bool((lv[l] == p).all()) for l, p in zip(lev, pinc)])
    chosen = [int(i) for L in range(6) for i in np.flatnonzero(lev == L)[:10]]
    per_level = {L: {'rows': 0, 'single_differ': 0, 'single_max_abs': 0.0, 'eight_differ': 0, 'eight_max_abs': 0.0} for L in range(6)}
    dec_ok = True
    for i in chosen:                                   # one row per request
        out = fam_b.package_release(pkg, [rows[i]]); p = out['income']['probability'][0]; L = int(lev[i])
        per_level[L]['rows'] += 1; per_level[L]['single_differ'] += int(not (p == lv[L]).all())
        per_level[L]['single_max_abs'] = max(per_level[L]['single_max_abs'], float(np.abs(p - lv[L]).max()))
        dec_ok &= int(out['income']['decision'][0]) == int(lv[L].argmax())
    for k in range(0, len(chosen) - 7, 8):             # 8-row requests
        blk = chosen[k:k + 8]; out = fam_b.package_release(pkg, [rows[i] for i in blk])
        for j, i in enumerate(blk):
            L = int(lev[i]); p = out['income']['probability'][j]
            per_level[L]['eight_differ'] += int(not (p == lv[L]).all())
            per_level[L]['eight_max_abs'] = max(per_level[L]['eight_max_abs'], float(np.abs(p - lv[L]).max()))
            dec_ok &= int(out['income']['decision'][j]) == int(lv[L].argmax())
    one = [int(np.flatnonzero(lev == L)[0]) for L in range(6) if (lev == L).any()]
    six = fam_b.package_release(pkg, [rows[i] for i in one])  # 6-row request, one row per level
    rec = {'levels_hit_in_600': {int(L): int((lev == L).sum()) for L in range(6)},
           'rows_600_batch_not_bitwise_equal_to_manifest_level': int((~big_bitwise).sum()),
           'per_level_single_and_8row': {str(L): v for L, v in per_level.items()},
           'six_row_one_per_level_differing_levels': [int(lev[i]) for j, i in enumerate(one) if not (six['income']['probability'][j] == lv[lev[i]]).all()],
           'income_decisions_equal_everywhere': bool(dec_ok)}
    if s == 0:                                         # key and value handling on one invented row
        r0 = rows[chosen[0]]; base = fam_b.package_release(pkg, [r0])
        same = lambda o: all(np.array_equal(o[t]['probability'], base[t]['probability']) for t in ('income', 'occupation'))
        cases = {'sex_key_1': [dict(r0, SEX='1')], 'sex_key_2': [dict(r0, SEX='2')], 'extra_key': [dict(r0, PUMA='00101')],
                 'reordered_keys': [{c: r0[c] for c in reversed(PERM)}], 'missing_key_POBP': [{c: r0[c] for c in PERM if c != 'POBP'}]}
        rec['key_handling_underlying'], rec['key_handling_wrapper'] = {}, {}
        for name, rr in cases.items():
            st, o = try_call(lambda rr=rr: fam_b.package_release(pkg, rr))
            rec['key_handling_underlying'][name] = {'status': st, 'output_bitwise_equal_to_base': same(o) if o is not None else None}
            rec['key_handling_wrapper'][name] = try_call(lambda rr=rr: AD.release(pkg, rr))[0]
        vals = {'COW_unknown_code_X': dict(r0, COW='X'), 'AGEP_200': dict(r0, AGEP='200'), 'AGEP_17_out_of_cohort': dict(r0, AGEP='17'),
                'WKHP_0': dict(r0, WKHP='0'), 'POBP_unknown_999': dict(r0, POBP='999'), 'AGEP_nan': dict(r0, AGEP='nan')}
        rec['value_handling_wrapper'] = {n: try_call(lambda row=row: AD.release(pkg, [{c: row[c] for c in PERM}]))[0] for n, row in vals.items()}
    res['seeds'][f'S{s}'] = rec
open(OUT, 'w').write(json.dumps(res, indent=1) + '\n')
```
