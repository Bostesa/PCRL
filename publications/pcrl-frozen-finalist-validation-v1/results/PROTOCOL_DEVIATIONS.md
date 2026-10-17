# Protocol deviations and disclosures

All times are UTC on 2026-10-17. Unit times are from the queue ledger (`RESOURCE_ACCOUNTING.json`). None of the items below changed the candidate, a reader, a winner or an endpoint after DEV_REPORT was read.

## Order of the governing steps (as run)

| Step | Time |
|---|---|
| Registration committed | 22:19:55 (`c499e0135`) |
| Registration receipt published | by 22:21:03 (`55b017918`) |
| First reader fit | 22:21:20 (units FFV_READERS_S0_V1 and FFV_READERS_S1_V1) |
| Reporting lock committed | 22:26:46 (`41088d8ac`) |
| Reporting receipt published | by 22:27:08 (`8993a4bdd`) |
| Coordinator DEV_REPORT re-read | 22:27:16 (event `DEV_REPORT_TRUTH_REREAD_BY_FFV_COORDINATOR`, unit FFV_REPORT_V1) |

## Disclosures

1. **Label-free checks ran before the registration commit.** They were registered in the protocol draft at the time, and none can inform reader fitting or claims.
   - The output-mapping check: FFV_MAPPING_V2 at 22:04, plus the conflict check FFV_MAPPING_CONFLICTS_V1 at 22:05.
   - The deployment checks: FFV_DEPLOY_V1 at 22:11 and FFV_BACKUP_DEPLOY_V1 at 22:12.
   - All are committed in the registration commit.
2. **Failed attempts preserved:**
   - FFV_MANIFEST_V1: wrong key name for the output contract;
   - FFV_MAPPING_V1: wrong key name for the feature matrix;
   - EXR_OVERLAP_V1: the exposure reviewer's code-hash check was too strict; its counts were identical to V2.
3. **A superseded manifest output.** FFV_MANIFEST_V2 wrote a manifest that reported mismatches between the per-level income vectors and the released vectors.
   - The cause was the manifest's own recomputation: it evaluated the 6 levels in a 6-row batch, and the package arithmetic's last bit depends on batch size.
   - That file was never committed. It was replaced by FFV_MANIFEST_V3, which records the released bit patterns and characterizes the small-batch difference as an observation (`FROZEN_CANDIDATE_MANIFEST.json`, `small_batch_last_bit_observation`).
4. **A coordinator draft rule was superseded before the lock.** The draft limited the ENHANCED control-validity conjunct to OUT_HGB. The statistical reviewer required all five control recipes (item A5). The stricter rule was adopted before the reader lock (PROTOCOL.md §7.5).
5. **The protocol was revised before the reader lock to adopt the statistical review** (§16). Revised:
   - the candidate order (A1);
   - estimator construction (A2);
   - CATI details (C1–C6);
   - the seed classification and auxiliary-sample refits (A4, O5);
   - the controls (A5);
   - the interpretation table (B1);
   - the feasibility specification (B4).

   All were committed and published together before any reader fit.
6. **One unconverged reader fit.** `PAIR__NEURAL_RAW__MLP6432` on seed 1 hit max_iter. It was used as fitted and flagged (rule C6). It was not a selected winner.
7. **Publication.**
   - The shell helper `ffv/backup_deploy_check.sh` is not in the public package, because the publication leak scan accepts only listed text file types. It stays on the research branch, and the scan was not bypassed.
   - Per-person threshold-tie details (6 MODEL_FIT people, no identifiers) were aggregated in the public `DEPLOYMENT_CHECKS.json`; the detail stays private.
8. **Platform event.** A platform safety classifier stopped the coordinator's first response to the brief before any of its tool calls ran. The campaign restarted at 21:03:32Z from state inspection. No variable or task description was renamed or obscured.
9. **Feasibility V2, a post-lock correction made after the statistical reviewer's feasibility review.**
   - No label was read. No claim label and no n80 or n90 value changed.
   - Feasibility status labels did change: the 6 assurance cells, from NOT_REACHED_BY_1E8 to LIMIT_BELOW_TARGET, and the governing-scorecard assignment in 8 cells (verifier item V-D4).
   - V1 (`CONFIRMATION_FEASIBILITY_RESULTS.json`, from the locked `ffv/feasibility.py`) is kept unchanged.
   - V2 (`CONFIRMATION_FEASIBILITY_RESULTS_V2.json`, from the new `ffv/feasibility_v2.py`, unit FFV_FEASIBILITY_V2) corrects the following:
     - **E1.** The limits are now Monte Carlo limits that respect the correlated boundary clauses. V1 multiplied per-clause limits: 0.0625 against the correct 0.266.
     - **E5.** The assurance limits are computed.
     - **E4.** The full curves are saved.
     - **E2 and E3.** The measured refit term is labelled "rough", and its scope is stated.
   - The documents use V2 and the reviewer's required wording.
10. **Withdrawn wording.** PROTOCOL.md §7.6 calls the measured reader-refit component "partial, lower-bound". That wording is withdrawn (review item E2): all 24 household-bootstrap refits read lower than the full-sample fits, so the spread mixes a resampling shift with variance, and the direction of the net bias is unknown. The locked protocol text is left as registered; this note supersedes it.
11. **A stale condition string in the frozen results** (claims review item O11). In `report/DEV_REPORT_RESULTS.json`, the ENHANCED scorecard's CV condition text still reads "source-definition OUT_HGB planted and shuffled controls". The value it reports follows the adopted rule (review item A5): `control_validity.CV.ENHANCED` covers all 5 control recipes × 3 views × 2 seeds, and all are in band. Only the description string is stale. The file is the one logged evaluation's output and cannot be regenerated.
12. **Output-mapping sourcing corrected after the claims review** (items M12 and M13).
    - The union counts first given as "6 / 49 shared vectors" were the excess of distinct inputs over distinct occupation vectors. The exact counts are 5 (seed 0) and 46 (seed 1) shared vectors (`MAPPING_CONFLICTS.json`, unit FFV_MAPPING_CONFLICTS_V2, `ffv/mapping_conflicts.py`).
    - The seed-1 conflict details had come from a session-scratchpad script (unit FFV_MAPPING_CONFLICTS_V1) that saved no output. They are now reproduced and saved, identifier-free, in the repository.
13. **Tie aggregation script committed** (item O15). The script that aggregated the public threshold-tie record is now in the repository as `ffv/aggregate_ties.py`.
14. **Independent verification (checks 1–8 PASS; check 9 PARTIAL, documentation only).** Disposition of the verifier's items (`INDEPENDENT_VERIFICATION.json`):

    | Item | Disposition |
    |---|---|
    | V-D1 | Union counts: corrected to 5 and 46, as in item 12. |
    | V-D2 | Tree counts per seed: corrected in OUTPUT_MAPPING_CHECK. |
    | V-D3 | Power at about 17,500 households: corrected to about 0.01 (ENHANCED, governing) and 0.25 (ORIGINAL). |
    | V-D4 | Which feasibility labels changed: now stated, in item 9 and CONFIRMATION_FEASIBILITY. |
    | V-D5 | Header for the governing scorecard: corrected. |
    | V-D6 | "What binds": corrected; the recipient-clause floor is stated. |
    | V-D7 | Conflict wording: corrected. |
    | V-D8 | The verifier's replay at 22:50:37Z: now stated. |
    | V-D9 | Cosmetic: two IDENTICAL_RELEASE family-S keys carry `clause_pass = false`, where other family-S keys carry null. They enter no claim. The file is the logged evaluation's output and is not regenerated. |
    | V-D10 | The extension of the side rule to a recipient clause: now stated in RESEARCH_DECISION. |
    | V-D11 | Near-tie governing assignments that depend on Monte Carlo noise: noted. |
    | V-D12 | V1 limits: resolved in V2. |
    | V-D13 | The residual "changed no label" sentence in RESEARCH_DECISION (option 4) was corrected after the verifier's final re-check: V2 changed no claim label and no n80 or n90 value, but did change feasibility status labels. |
