# Claims review (independent read-only role) and resolutions

The reviewer checked every figure in the closeout documents against the full-precision results, both locks, the pre-lock review and the source study's public records. They opened no `.private` directory and no DEV_REPORT truth. All 15 must-fix and 23 should-fix findings were applied. The receipted `PROTOCOL.md` was left unedited, and its one accounting gap is recorded in `LOCK_NOTES.md`.

## Must-fix

| id | finding | resolution |
|---|---|---|
| MF-01 | The seed-0 increment's lower bound was double-rounded (0.007). | Now 0.015 [0.006, 0.024] in START_HERE and ADVISOR_BRIEF. |
| MF-02 | H_JJ coalition AUCs were double-rounded (0.604 / 0.607). | Now 0.603 / 0.606. |
| MF-03 | START_HERE said the reduction was "too little to establish 0.020", though both points exceed 0.020; the limiting comparison and the closure were not stated. | The Answer now says the margin fails on precision (clause bounds 0.012 / 0.014; unresolved, not a shortfall), names the neural control as the limiting comparison, and closes the test. |
| MF-04 | The brief's closing section had the same misstatement, did not name the limit, implied equivalence ("matches the neural control") and omitted the subgroup and privacy caveats. | Rewritten as suggested. |
| MF-05 | "They are tied" implied equivalence. | Now "Not established either way": the point slightly favours NEURAL_RAW, and equivalence was not tested. |
| MF-06 | JOINT income's utility failure gave only the MODEL_SELECT pattern. | The DEV_REPORT failures are added: seed 0 Brier (0.0054 / 0.0070); seed 1 accuracy loss 0.0140 and Brier (0.0067 / 0.0077). |
| MF-07 | H_JL and H_JJ were not called a post-hoc observation, not a nominee; "clear the bars" overstated the NEURAL_RAW comparison. | Labels added in every document. The NEURAL_RAW comparison now notes H_JL's seed-1 interval includes zero and neither reaches 0.020. |
| MF-08 | The handoff forbade saying "any hybrid" beats the neural control or meets 0.020, which contradicts the descriptive results. | It now forbids the claim for H_LJ or any utility-eligible combination. |
| MF-09 | The "not a population privacy guarantee" caveat was missing. | Added to ADVISOR Q4 and to the handoff (prohibition and caveat). |
| MF-10 | The handoff credited the reduction to joint training. | It is now described as a decomposition of fixed outputs, not evidence about the training objective. |
| MF-11 | The suggested sentence was causal, omitted the separate statuses, implied equivalence and left out the post-hoc nature. | Replaced. |
| MF-12 | The sign of "(differences −0.001 and −0.005)" was ambiguous. | Now explicit: H_LJ is 0.001 / 0.005 higher; the benefit intervals are given. |
| MF-13 | The new-information accounting covered only 36 of 38 endpoints. | The 30 / 2 / 6 split is in the handoff; a clarification row is in LOCK_NOTES (PROTOCOL.md is receipted, so it is not edited). |
| MF-14 | The rehearsals were misdescribed. | Now V1 halted by design, V2 failed on a script defect, V3 completed; the ledger charges are recorded in LOCK_NOTES. |
| MF-15 | START_HERE lacked "not an independent replication, not a rescue" and the named source labels. | A Registration paragraph was added. |

## Should-fix

All 23 were applied as proposed:
- **Plain answers and decomposition:** SF-01 decomposition caption; SF-02 plain "not both" answer; SF-03 interaction wording; SF-04 recipient-level balance.
- **Definitions and scope:** SF-05 attacker definition (chosen on ATTACK_VALIDATION); SF-06 timing of the fix; SF-07 238 attackers plus 12 controls.
- **Deployment and custody:** SF-08 teacher files in the deployment description (5,943 rows); SF-09 restore evidence cited; SF-21 closeout files written.
- **Status fields:** SF-10 clause bounds and sex coding in Q5, with the three status fields separated; SF-11 order of the tightest clauses; SF-12 published recipient outcomes marked as not findings.
- **Labels and conventions:** SF-13 claim B nested label and the FAGTB/FARE note; SF-14 new-value note, sign convention and scope placement; SF-23 ρ1.0 naming.
- **Source records and registration:** SF-15 complete D1 and D2 wording; SF-16 source records in the brief; SF-17 separate status fields, source labels and the scope sentence in the handoff.
- **Reproduction and figure:** SF-18 QUICKSTART steps (queue init, DEPLOY_V1, closeout); SF-19 accurate write-once statement; SF-20 figure caption; SF-22 the lock's extra rules cited.

## Confirmed without change

- **Labels and decisions:** all labels and statuses, and the scope sentence (verbatim).
- **Figures:** every other number, including the ADVISOR Q4 decomposition cells as differences of full-precision AUCs.
- **Conventions:** the sex naming (sex0 = male, ACS SEX 1; sex1 = female, SEX 2), and D1/D2 core wording (10/11; "never accessed" nowhere).
- **Reproduction and figure:** the QUICKSTART flags, and every plotted point in the figure.
