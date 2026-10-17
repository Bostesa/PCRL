# Independent numerical verification: SCM utility-matched follow-up

**Role.** Independent numerical verifier, 2026-10-17 (05:33Z to 06:05Z). All eight checks of `roles/coordinator/VERIFIER_BRIEF.md` are done.

**How the work was done.**
- My own code is in this folder: `vlib.py`, `vengine.py`, the scripts `check1` to `check8`, `check7b`, and `make_summary.py`. Each script writes a `checkN_*.json` output.
- Every script ran through the shared queue: one thread, one job at a time, a new unit id per attempt, and no failed attempt.
- No person-level array was written to the repository.
- The one DEV_REPORT truth replay ran through `scm_um.common.dev_report_truth(…, event_name='DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER')`, in unit `VERIFY_CHECK5_V1` at 05:54:14Z. It came after an end-to-end dry run on invented labels (`VERIFY_CHECK5_DRYRUN_V1`, no access event; its output was deleted). The crash copy (scratchpad, mode 600) was deleted after the run.
- Nothing was committed.

**Verdict.**
- **All eight checks PASS.**
- Every quantity I recomputed equals the campaign's recorded value bitwise. This covers:
  - merge sequences, spending, labels and tables;
  - 14,499 MODEL_SELECT values and 3,198 proxy AUCs;
  - roles;
  - the 82 DEV_REPORT endpoints (point, SE, family bounds, clause bounds);
  - every status, availability and label;
  - reader refits, package outputs and access order.
- I found ten discrepancies, D1 to D10. They concern procedure, wording or records. None changes a release, endpoint, status, label or the research decision.

## Checks

| # | Check | Status | Evidence (own code unless stated) |
|---|---|---|---|
| 1 | Source admission and household-role integrity | PASS | See notes below the table |
| 2 | Trajectory checkpoints and aliases | PASS | See notes below the table |
| 3 | MODEL_SELECT utility, guards, expenditures, proxy, nomination | PASS | See notes below the table |
| 4 | Matching and comparator selection | PASS | See notes below the table |
| 5 | Primary results, intervals and labels (one logged replay) | PASS | See notes below the table |
| 6 | Raw-field packages and input validation | PASS (qualified, D6) | See notes below the table |
| 7 | Predefined reader refits | PASS | See notes below the table |
| 8 | Access | PASS | See notes below the table |

**Check 1: source admission and household-role integrity.**
- **Admitted copies.** All 51 admitted entries were re-hashed against their own manifests: 18 exports, 2 frozen banks, 28 fit files, 1 encoder record and 2 computed routing files. The manifests are EXTRACTION_SUMMARY, the source TRAINING_COMPLETE_LOCK with FIT_SEED `files_sha256`, FROZEN_BANK_MANIFEST and MODEL_MANIFEST.
- **Source checks.** All 12 manifest checks were redone on the source tree. There are no unexpected or missing files, and the source tip is 2738cc30b.
- **Roles.** The six roles are pairwise disjoint in households and in rows. Their counts equal the public summary; DEV_REPORT has G = 3955 households and 5,943 persons.
- **Routing.** I evaluated the oblivious-tree teachers myself (the fold average), so I could route by KL to the nearest centroid. My routing agrees with the admitted routing on all four roles and both seeds, with 0 mismatches. My teacher probabilities differ from the source evaluator's by at most 5.6e-16, and the smallest gap between the best and second-best score is 3.5e-08, so no near-tie risk.
- **Historical arms.** All 31 historical arms are rebuilt bitwise on all four roles for both seeds. The centroids equal the pooled out-of-fold means bitwise.
- **Sealed assessment.** No SEALED file or export exists. `load_role` refuses SEALED_ASSESSMENT and every truth field I tried; the refusal tests import `scm_um.common` as the object under test.

**Check 2: trajectory checkpoints and aliases.**
- **Merge path.** I wrote an independent merge-cost implementation (`vengine.py`, from source PROTOCOL sections 3–5) and used it to issue a greedy certificate along each recorded path. All 8,194 recorded merges (4,116 + 4,078 over 20 trajectories) are the first admissible pair in priority order, with 0 near-ties and 0 divergences. Each capped trajectory ends with no admissible pair, and each IBSI trajectory ends at exactly 32 cells.
- **Running costs.** My F_m equals the recorded value bitwise. The absolute-cost route agrees to within 3.4e-16. The guard running sums agree to within 3.3e-17; I compute the shrunk label sums in the protocol's literal order, which explains that difference.
- **Labels.** All checkpoint labels agree: N{t} with the 1e-15 slack, C{c}, TERM, S{t} and HIST, together with the reached flags. Every per-merge change in F is at least 1.9e-07.
- **Releases.** I found 272 / 261 effective releases. Their alias partition equals the recorded one, and 325 / 324 label tables equal RELEASE_TABLES bitwise. Capacity and wire hashes all match.
- **History.** The 10 historical arms equal ARM_TABLES and ARM_OUTPUTS bitwise at 261 / 266 cells. The recorded spending at K_hist equals the source FIT_SEED bitwise.

**Check 3: MODEL_SELECT utility, guards, expenditures, proxy and nomination.**
- **Coverage.** All 533 effective releases plus the frozen NEURAL_RAW and H_LJ were measured on both seeds, with no sampling.
- **Values.** The 7,398 + 7,101 clause, guard, expenditure and summary values are exact.
- **Proxy.** I reproduced the proxy with my own LR, HGB and lookup readers, my own household split and my own AUC. All 1,632 + 1,566 direction AUCs are exact.
- **Nomination.** t* = 0.008 is NOMINEE_SELECTED. The eligible thresholds are 0 to 0.008, K* = 298 / 300, and e* = 0.0082480 / 0.0084263. The fallback grid is identical.

**Check 4: matching and comparator selection.**
- These checks use my check-3 measurements.
- All ten roles per seed are identical, with the same statuses and trajectories (IBSI γ 2.0, SHUF r2, IBSI_CNT γ 1.0).
- The candidate lists, match and clause flags, effort counts, coverage gaps, services and aliases are identical. So are the SCM − role differences and the income qualifiers.

**Check 5: primary results, intervals and labels (one logged replay).**
- **Bootstrap.** My own household bootstrap is bitwise equal to `task_headroom.inference.group_weights`.
- **Critical values.** They equal the registered values to 7 dp.
- **Endpoints.** All 58 + 24 endpoints are identical (maximum absolute difference 0 in every field), with the same statuses, availabilities and IDENTICAL_RELEASE flags.
- **Claims.** Claims M, MC, K, B and C are identical, together with every condition and failure category. The separate fields, the research decision with its qualifiers, and the bets (Brier 0.10375) are identical too.
- **Supporting results.** The per-service results, the descriptive companions and both CSVs are identical.
- **Label-free replay (S2).** For HIST_SCM, C0, NEURAL_RAW and H_LJ, the winners, AUCs and SEs equal the source DEV_REPORT_RESULTS exactly.

**Check 6: raw-field packages and input validation.**
- **Packages and paths.** All nine packages reproduce the evaluated DEV_REPORT wire bitwise through both the API and CSV paths.
- **Own path.** My own encoder reproduces the DEV_REPORT feature export bitwise. My own teacher, routing and lookup path reproduces the SCM-format packages bitwise.
- **Integrity.** The file hashes, teachers, encoder, centroids and tables equal the locked records.
- **Refusals.** All 23 refusal cases hold for `scm_um.deploy`: 8 CSV, 6 API and 9 hidden-output requests. For the frozen references, see D6.

**Check 7: predefined reader refits.**
- I refitted the eight predefined pair winners (SCM, ORD_NLLM, IBSI_NLLM and NEURAL_RAW on both seeds) with my own scaler and sklearn estimators. Their DEV_REPORT scores are bitwise equal to the stored ones, and their ATTACK_VALIDATION AUCs are exact.
- I also refitted 16 more recipient winners, all bitwise. The wires rebuilt from the tables equal the bank. The winner rule, re-applied, picks the same readers.
- **7b.** The six control readers per seed (shuffled null and planted bit) are bitwise equal to the stored ones.

**Check 8: access.**
- **Order.** Every unit started after its lock was committed: TRAJ after ANALYSIS_LOCK, SELECT after TRAJECTORY_COMPLETE_LOCK, BANK after SELECTION_LOCK, READERS after the bank manifest, REPORT after REPORTING_LOCK, and DEPLOY after REPORTING_LOCK.
- **DEV_REPORT events.** There are exactly two DEV_REPORT events: the coordinator's at 05:30:14Z in UM_REPORT_V1, and the verifier's at 05:54:14Z in VERIFY_CHECK5_V1. Both lie inside their unit windows and after the REPORTING_LOCK commit (05:29:47Z). Both cite the committed lock hash and the public truth hashes, and the two access logs are identical.
- **Sealed access.** There was none.
- **Receipt.** Receipt 3c7d69317 (05:30:02Z) is the remote publication ref.
- **Ledger.** Each lock file was committed once. No queue unit failed.

## The verified outcome

**Research decision.** COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE. The four qualifiers are:
- DEV utility confidence is CONFIDENCE_UNRESOLVED: 3 PASS, 21 UNRESOLVED, 0 violations.
- Claim C is SUBGROUP_UNRESOLVED.
- The ORD_NLLM DEV match is MISMATCH_ESTABLISHED (s0) and EQUIVALENCE_UNRESOLVED (s1). Both are SCM_SPENT_MORE.
- Claim B is NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD.

| Item | Seed 0 | Seed 1 |
|---|---|---|
| DEV occupation-NLL SCM − ORD_NLLM (match gate, \|·\| ≤ 0.001) | +0.00697 (family [+0.0012, +0.0128]) | +0.00175 (family [−0.0044, +0.0079]) |
| Pair benefit vs ORD_NLLM (clause L1) | +0.0336 (L1 +0.0225) | +0.0252 (L1 +0.0144) |
| Same-count benefit vs ORD_CNT (clause L1) | +0.0361 (L1 +0.0243) | +0.0275 (L1 +0.0169) |
| Income worsening vs ORD_CNT (clause U1, limit 0.010) | −0.0015 (U1 +0.0136) | +0.0039 (U1 +0.0174) |
| Men's within-SEX occupation Brier excess (point, limit 0.005) | 0.00536 | 0.00637 |

**Claim labels.**
- M is MATCH_DID_NOT_CARRY_OVER. Its comparison-only clauses all pass, but they are not cost-matched on DEV_REPORT.
- The components are NOT_TESTED_M_NOT_ESTABLISHED.
- K is NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD.
- B is NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD.
- C is SUBGROUP_UNRESOLVED.

**Bets.**
- HELD: P1, P2, P3, P5, P6.
- NOT_HELD: P4, P7, P8.

## Discrepancies

| Id | Check | Type | Discrepancy |
|---|---|---|---|
| D1 | 1 | Record wording | See below |
| D2 | 1, 8 | Procedure | See below |
| D3 | 1 | Procedure | See below |
| D4 | 1, 8 | Procedure | See below |
| D5 | 2 | Registered text | See below |
| D6 | 6 | Deployment qualification | See below |
| D7 | 1 | Trivial wording | See below |
| D8 | 4 | Trivial, commit message | See below |
| D9 | 4 | Reporting completeness | See below |
| D10 | 2 | Trivial, commit message | See below |

**D1. The admission record gives the wrong reason for three CSV differences.**
- SOURCE_ADMISSION.json says the source report CSVs PRIMARY, SECONDARY and RECOVERY_AND_UTILITY "changed on the research branch after the publication build: cc7f8a168".
- In fact cc7f8a168 (23:22-04:00) predates the publication commit 6fc06021d (23:48-04:00).
- The files are identical to the published ones after CRLF→LF normalisation, as PROTOCOL §1 correctly states.
- **Cause.** `admit.py` takes the last research commit that touched the file, and the publication branch is an orphan.

**D2. A locked brief was edited after the lock.**
- VERIFIER_BRIEF.md is hashed in ANALYSIS_LOCK. It was edited in cf647d650 (05:24:38Z), which expanded the check-5 wording.
- This conflicts with PROTOCOL §16: "locked files are never edited".
- The edit preceded every MODEL_SELECT read and changes no rule.

**D3. Locked dry-run code was edited.**
- `scm_um/dryrun.py` is hashed in ANALYSIS_LOCK. It was edited in cf647d650, which added a stand-in proxy for the selection dry run.
- The new hash is in REPORTING_LOCK. The commit message does not mention the change.
- The edit came before any MODEL_SELECT label was read.

**D4. Locked table code was edited after the DEV_REPORT evaluation.**
- `scm_um/tables.py` is hashed in ANALYSIS_LOCK and REPORTING_LOCK. It was edited after the DEV_REPORT evaluation, in 57f50a5ff (05:34:26Z).
- The edit changes only figure annotation offsets and the K* string format.
- No endpoint is computed there. The commit message does not mention the change.

**D5. UNIT_INVENTORY.json still states the old checkpoint rule.**
- The inventory's `nll_rule` gives the "last prefix with F_m ≤ t + 1e-15" rule.
- The protocol and the code use the first-crossing rule.
- There is no numerical consequence: F is strictly increasing on all 20 trajectories, so the two rules agree on all 220 N{t} labels.

**D6. The frozen packages' direct entry points do not refuse bad keys.**
- The frozen references NEURAL_RAW_S0, H_LJ_S0 and H_LJ_S1 have direct source API entry points: `acs_replication.deploy.release` and `acs_hybrids.hybrid_deploy.release`.
- These entry points silently accept a SEX key, a lower-case sex key, an extra PUMA key, or reordered keys. The keys are ignored, so outputs are unaffected, but nothing is refused.
- The refusals hold only behind `scm_um.deploy.check_rows` (as DEPLOYMENT_RESULTS.json states) and on the CSV path.
- PROTOCOL §14 ("refuses extra or reordered inputs, SEX as an input") therefore holds for the frozen references on those two paths only.
- By code reading, `scm_bench.deploy_scm.release` used directly accepts reordered keys. The documented `scm_um.deploy.release` is strict, which I verified.

**D7. PROTOCOL §1 counts 51 copies but lists 49.**
- PROTOCOL §1 says "51 hash-checked copies" but lists 49 items.
- The 51 entries include the 2 computed routing files.

**D8. A commit message understates a matching gap.**
- The SELECTION_LOCK commit message says "\|delta\| <= 0.00015".
- Seed 1's value is 0.000154299. The locked JSON is correct.

**D9. Selection effort is not listed for every role.**
- PROTOCOL §7 says selection effort is reported "for every role".
- SELECTION_LOCK `selection_effort` omits SCM (11 distinct releases per seed, visible in the nomination grid) and ORD_CNT (1).

**D10. A commit message counts the summary flag as a gate check.**
- The TRAJECTORY_COMPLETE_LOCK commit message says the gate passed "52/52".
- That count includes the ALL_PASS flag; the gate has 51 checks.

## Readings

**Favourable to the campaign's integrity.**
- Every lock, ordering and access rule holds. Every registered computation reproduces bitwise from independent code, and the trajectories are certified greedy step by step.
- The source label-free replay reproduces AUCs and SEs exactly.
- The controls are valid: planted 1.0, nulls 0.487 to 0.517.

**Favourable to SCM, descriptive only and not cost-matched.**
- Against ORD_NLLM, the pair benefit's clause bound L1 is > 0 on both seeds, and every worsening clause passes.
- The same-count benefit against ORD_CNT is established on the family interval in both seeds, with L1 > 0.020 in seed 0.
- The SHUF component's benefit (+0.020 in both seeds) is established on the family interval.
- SCM lowers the pair AUC relative to C0 by 0.039 / 0.030.

**Unfavourable to SCM.**
- **Spending.** On DEV, SCM spent more occupation NLL than the matched ordinary release in both seeds; seed 0 is MISMATCH_ESTABLISHED. It also spent more income NLL (+0.0065 / +0.0035).
- **Subgroup guard.** SCM fails the men's within-SEX occupation Brier point guard in both seeds (0.0054 / 0.0064 > 0.005). That gates claims K and B; claim C is unresolved, not harm.
- **Same-count worsening.** Against ORD_CNT, SCM's income-worsening clause bound exceeds 0.010 in both seeds (U1 +0.0136 / +0.0174). So K would not be established even without the guard.
- **H_LJ.** Against H_LJ, SCM's occupation worsening is ESTABLISHED_VIOLATION in seed 0.
- **IBSI.** The matched IBSI comparator reads lower than SCM in seed 1 (benefit −0.0101).
- **Utility confidence.** DEV utility confidence is unresolved on 21 of the 24 clauses.

## Open issues

1. **Bet scoring for P4 and P5.** The rule says these bets are "NOT_HELD unless ORD_NLLM is MATCHED on both seeds".
   - The frozen code reads MATCHED as the MODEL_SELECT role status, consistent with P2. That gives P5 HELD and a mean Brier of 0.10375.
   - If MATCHED meant the DEV point match (the wording rule of MF-2), P5 would be NOT_HELD and the mean Brier would be 0.11625.
2. **Thin ORD_NLLM margins.** The ORD_NLLM choice rests on thin proxy margins (0.00014 and 0.00029).
   - The seed-0 runner-up matched expenditure more closely (\|Δ\| 3.2e-06 against 2.8e-05).
   - This is the registered strongest-proxy rule, applied correctly.
3. **Equivalence was out of reach.** Equivalence at ±0.001 could not be reached with an SE of 0.0019 to 0.0020, as the statistical review anticipated.

## CPU

- **Queue units.** Eleven verifier units were logged in the ledger, including the summary assembly, totalling 523.3 s. None failed.
  - Check 2: 304.0 s.
  - Check 3: 121.7 s.
  - The two check-5 runs, dry and real: 52.6 s.
  - The rest: 45.1 s.
- **Light commands outside the queue** (git and JSON reads): at most 60 s.
- **Agent analysis:** an upper estimate of 0.25 CPU-h.
- **Total upper estimate:** 0.41 CPU-h. The registered verifier line was 0.5.
- The agent allowance is not yet logged with `jobqueue charge`; I left that to the coordinator to avoid double counting.
