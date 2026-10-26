# Protocol deviations and process defects (append-only; campaign `pcrl_reader_fit_aware_release_v1`)

Each entry states what happened, when, its consequence and what was done. Nothing here changes a registered rule or an earlier
campaign's record.

1. **Fit-record accounting (verifier wave 1, D1).** `rfa/fam/fit.py` records `inner_reader_fits = episodes x views x families`. That is
   exact for FULL and STOPFIT but understates ALTCRIT: its first call adds an exact initialization fit, every call adds one exact
   diagnostic fit for the lag, and its 10 Adam rounds per view and family are described only in text. The calibration's 64 exact fits
   and the gate's fits are not counted in any arm record. Consequence: accounting only. The CPU of every fit is charged by the queue.
   Not changed during the running fits (registered code).
2. **Reader refusals abort a unit (verifier wave 1, D2).** `ReaderRefused` is not caught in `calibrate`, `mechanism_gate` or `run_arm`,
   so a refusal would abort the whole unit without a FIT record, whereas `rfa/fam/rfit.py` says the training loop "records it and
   stops". The queue would still record the failed unit. No refusal occurred: both v1 fit units ended `done`.
3. **Log label (verifier wave 1, D3).** The fold records' `FULL_logs_every_10` key holds every fourth 10-step log, i.e. one log
   every 40 steps. Cosmetic.
4. **Coordinator time stamps.** Two document times written ahead of the clock were corrected before the registration commit (FAILURE_MAP,
   PROTOCOL). Three coordinator-log times were inexact: one written ahead of the clock (21:44-21:50Z) was corrected in place to
   21:47Z before that log was committed; two launch / end times were corrected from the queue ledger by appended lines. No decision
   depends on them.
5. **Interrupted card draft.** At about 21:36Z a coordinator response was stopped by the platform's safety classifier while it was
   writing the first draft of the card. The partial file it left was deleted, and a new card was written and reviewed before the
   registration commit. Nothing had been committed or fitted.
6. **The first v1 decision record (statistical reviewer, claims review wave 1, items 2-3).** `claims/RFA_v1/DECISIONS.json` (22:17Z)
   passed the main grid's validity as the learning curve's validity and did not save the falsifier inputs. `rfa/rfclaims.py` was
   corrected: the half-budget cells get their own grid validity (`lc_seed_checks`), and FALSIFIER_DETAIL.json is saved beside every
   decision record. The record was rebuilt from the same plan and grids (`claims/RFA_v1_rebuilt/`, 22:29:52Z, label-free, outside
   the queue): the half-budget validity is READER_SEEDS_EFFECTIVE (1,536 checks), so LC and F4 stand, and every label is unchanged.
   The first record is kept; the rebuilt one governs.
7. **The released income vector is START's network, not the frozen START wire bitwise (verifier wave 2, F1).** Every FAM
   configuration emits START's income network evaluated in float64 by `rfa.fam.net` (bitwise identical across the FAM
   configurations), but that vector differs from the frozen bank's NEURAL_RAW income wire by at most 6.8e-8. The card's sentence
   "the released income vector stay[s] START's bitwise" therefore holds against START's weights, not against the frozen START release
   used by the grid and the evaluator. Effect: NEURAL_RAW's income view is a separate wire on the grid (non-zero income-view
   differences in 3 of 8 cells, the largest -0.00074). The verifier's bounded sensitivity refit (winner fixed, START's pair winner
   refitted on RF_FULL's income) moved START's pair AUC by -0.00081 to +0.00003 and left the START comparison AGG_POSITIVE (mean
   0.01256 vs 0.01272). No label changes.
8. **Episode count on saved records (verifier wave 2, D4-new).** `n_episodes_deployed_fit` records the whole deployed arm's episode
   count, not the count up to the saved checkpoint (RF_STOP_LO: 20 / 70 recorded for checkpoints S10 / S60). Cosmetic.
9. **Family-change reading field (coordinator).** The first `claims/RFA_R1/FAMILY_CHANGE_vs_RF_FULL.json` read the pair mean from the
   wrong field of the benefit screen (it is kept under `aggregate`), so it reported a mean of 0. The label was the same
   (NOT_POSITIVE: the true mean is -0.0029). The function was fixed and the reading recomputed label-free
   (`FAMILY_CHANGE_vs_RF_FULL_corrected.json`); the first file is kept.
10. **Final decision records carry the second split (coordinator, after reviewer claims review wave 2, R2).** `claims/RFA_v1_rebuilt/`
    and `claims/RFA_R1/` were built before their second-split records existed, so their D2R field read NOT_RUN. Both were rebuilt
    label-free from the same saved inputs with the D2R record attached (`python3 -m rfa.rfclaims run ... --d2r-dir
    claims/<reading>_D2R`): `claims/RFA_v1_final/` and
    `claims/RFA_R1_final/`, which now govern. Every label and both outcome classes are unchanged; the outcome flags now include
    D2R_SIGN_CHANGE_CARRIED_INTO_HEADLINE and D2R_RECIPIENT_FAILURE_CARRIED_INTO_HEADLINE. The earlier records are kept. The `d2r_dir`
    argument was added to `rfa/rfclaims.py` at about 23:08:27Z, while the verifier's unit RFA_VER_W3_MAIN_v1 was running; the queue
    recorded the change (`code_changed_during_run`) and the verifier found that its unit used none of the changed code (verifier
    hand-back 3, D-W3).
11. **Record details found by the verifier's wave-4 conformance check (hand-back 4).** None changes a label or a quoted 4-dp value.
    - The R1 second-split label is written unprefixed (ROBUSTNESS_NOT_CONFIRMED in `claims/RFA_R1_D2R/D2R.json`, the final R1 record,
      ENDPOINTS, TABLES and the campaign summary): `rfclaims.d2r` does not apply the ADAPTIVE_R1 prefix. The record is left as
      written; the documents say that the label is unprefixed and that R1 is adaptive as a whole (PROTOCOL section 9).
    - `claims/PER_RECIPE_PAIR_MEANS.json` computed its falls against START as differences of 5-dp means, so 8 of its 15 fall entries
      are off by 1e-5. The script now subtracts unrounded means and writes `claims/PER_RECIPE_PAIR_MEANS_corrected.json`, which the
      documents cite. The first record is kept, and the prediction outcome (not met) is unchanged.
    - The decision drafts called the same-checkpoint spend differences unresolved point estimates without recorded uncertainty. The
      fit records hold their paired SEs (3.1-6.2 SE above zero), and the compared STOPFIT and ALTCRIT points sit at that checkpoint.
      The documents were corrected before publication.
12. **Publication-stage findings (verifier hand-back 5).** None changes a label.
    - **The SE range.** Item 11 and verifier hand-back 4 give the same-checkpoint spend differences as "3.1-6.2 SE above zero". The
      unrounded ratios run from 3.137 to 6.148, so the range is 3.1-6.1. RESEARCH_DECISION.md and START_HERE.md now say 3.1-6.1;
      item 11 and hand-back 4 stay as written.
    - **CSV line endings.** The publication builder reads text with newline normalization, so `results/dev/UTILITY_LEDGER.csv`
      (CRLF in the research copy, from csv.writer) is published with LF line endings. The content is otherwise identical. This is
      recorded in PUBLICATION_RECEIPT.json, and the builder is unchanged.
    - **A custody record annotated after its unit ran.** `custody/OFFDEVICE_COPY.json` gained its `size_note` (and moved it to the top
      level) after its unit ran, in research commit c4bc191e1. The change is visible in the history.
    - **The backup gap.** It also covers private-tree changes made after 23:27:18Z: the verifier's two wave-4 units (their label-access
      entries and ledger lines), the off-device unit's folder and the queue close. HANDOFF.md says so.
13. **A reviewed leak-scan hit (coordinator, 00:41Z).** The publication build of research commit 60acaf352 refused on one hit:
    "session scratch path" in `results/roles/verifier/code/v5_pubscan.py`, line 93. That line holds the bare session-scratch folder
    prefixes (the strings the rule matches) as the verifier's own classification patterns; no session identifier or path follows them.
    The first wording of this item quoted those prefixes literally, and the next build refused on that quotation too. It was reworded
    before any publication. The pair (file,
    problem) was added to `rfa.publish_build.REVIEWED_SCAN_HITS`, the mechanism SAR1 used for its verifier's search patterns. The scan
    itself is unchanged, and every other file is still refused on any hit. The build then passed; PUBLICATION_RECEIPT.json lists the
    reviewed hit.
