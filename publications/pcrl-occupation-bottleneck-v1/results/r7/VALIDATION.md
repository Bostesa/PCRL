# Independent validation closeout

## Review scope

This document records completed ROOT-owned queue receipts and existing public JSON reports reviewed on 2026-10-13. This closeout performed metadata and code review only: no numerical imports, jobs, array/checkpoint loads, numerical artifact hashes, new model fits, or shared lock edits. ROOT owns numerical execution and custody.

Independent arithmetic is frozen in `occupation_bottleneck_independent/replay.py`, SHA256 `5a845905f1f7d29c950b66cf1fb05f49d4b08e66b8b4f3db852e3c5086d9a242`. The R7-owned release worker is `release_replay_worker_r7.py`, SHA256 `db2c416b2e51a2eb0626b0f0323231385f532d57e2c9d813f2b54c52ba67ee95`. Scientific runner methods are not imported by these implementations; the shared queue authenticator is their permitted campaign dependency.

## Completed checks

Every queue unit below has status `done` and a completed receipt with return code 0. Links refer to the retained public outputs.

| Queue unit | Evidence and coverage | Result |
|---|---|---|
| `OB_INDEPENDENT_AUDIT_R7` | [Scientific audit](INDEPENDENT_AUDIT.json): all 84 nominal source packages, 66 canonical models, 20 exact source tensor aliases; 72 banks; 66 FIT trajectories; 18 coarse saved path sequences; 18 anchor-role geometry records | Complete; no source package missing; trajectory/geometry checks passed |
| `OB_INDEPENDENT_C3_R7` | [C3 FIT audit](INDEPENDENT_C3.json): 72 bank records, FIT folds 0/1 label tables scored on inherited FIT fold 2 | Completed; independently reproduced label association tables and supported task strata |
| `OB_INDEPENDENT_HEADLINES_R7` | [FIT2/INNER headlines](INDEPENDENT_HEADLINES.json): all 72 packages in both roles; 36 paired point gates; 144 package-role strata records; 12 projected/full decision-reader alias checks | All package records and aliases passed; compared to the public report |
| `OB_INDEPENDENT_UTILITY_R7` | [Utility inference](INDEPENDENT_UTILITY.json): all 864 frozen endpoints | All numeric fields agreed at the frozen verifier tolerance; all literal gates and inference dimensions agreed exactly |
| `OB_INDEPENDENT_PRIVACY_R7` | [Primary privacy inference](INDEPENDENT_PRIVACY.json): all 108 frozen endpoints | All compared endpoint fields matched exactly in the public JSON |
| `OB_INDEPENDENT_COMPLETE_VIEW_ANCHORS` | [Complete-view inference](INDEPENDENT_COMPLETE_VIEW.json): all 18 frozen supplementary anchor endpoints | All compared endpoint fields matched exactly in the public JSON |
| `OB_INDEPENDENT_R7_ALL_RELEASE_FORWARDS` | [Owned actual-checkpoint replay](INDEPENDENT_R7_ALL_RELEASE_FORWARDS.json): 72 canonical packages, 36 paired configurations, 3 roles, 216 package-role checks, 432 recipient forwards | PASS; global representation, centered-logit, and probability differences all 0.0; every decision identical |
| `OB_INDEPENDENT_CACHE_EQUIVALENCE_V2` | [Cache fixture](CACHE_EQUIVALENCE_V2.json): synthetic data, identical group draws and endpoint arithmetic | Exact result JSON equality; 12 cached versus 96 uncached weighted-AUC calls; no real data or scientific fits |
| `OB_INDEPENDENT_ANCHOR_REFITS` | [Four independent reader refits](../r4/INDEPENDENT_ANCHOR_REFITS.json): both seeds for U0 and RAW_JOINT_R03_E5 | Fixed target orientations matched; INNER labels remained unopened |
| `OB_INDEPENDENT_BACKUP_ONLY_REPLAY` | [Independent backup replay](INDEPENDENT_BACKUP_ONLY_REPLAY.json): U0 seed 0 full and RAW_JOINT_R03_E5 seed 0 full, with saved independent pair readers | PASS; restored complete wires and reader predictions match exactly, with all reported differences 0.0 |
| Twelve `R6_RESTORE_LOCAL_*_V1` units | [R6 restore evidence](../r6/): three anchor families, both seeds, full and Pz releases; 72 saved reader checks | All twelve PASS; zero wire/reader differences; runtime code from snapshot, no original sources used |

The R7-owned actual-checkpoint replay independently validates the canonical roster binding, raw83 feature names and order, source row/group counts and role disjointness, group-hash FIT fold 2, all 16 encoder/head tensor shapes and precisions, and the 21,288 parameter count. It reconstructs all full/Pz releases using independent functional forward and factorized SVD projection math. The observed differences were zero, stronger than the permitted full-head tolerance of 2e-12 and projected tolerance of 1e-10. The queue receipt completed at `2026-10-13T15:21:39+00:00`; its registered public output SHA256 is `58e94aabbc22666d6d32a035ed9f5924f23da9a2d73b540c9301d3b2090d617b`.

### Interval comparison precision

All three families use 2,235 rows, 2,234 duplicate groups, 2,048 shared group bootstrap replicates, bootstrap seed 2026101301, and df 2233. Endpoint IDs, status, point status, thresholds, K, df, and replicate counts match their runner reports. Privacy and complete-view point, standard error, bounds, and gap fields also match exactly as parsed JSON values.

Utility comparison passes the frozen absolute tolerance of 2e-12 with zero relative tolerance. It is not bit-exact JSON equality: 560 of 864 utility endpoint records have at least one differing last floating-point digit among point, standard error, bounds, or gap. For example, one independent NLL point is `-0.00047230135906932284`, while its runner value is `-0.0004723013590693264`. This distinction does not change any endpoint gate. No numerical job was launched to obtain this closeout comparison; it compares existing report fields.

### Gradients, controls, and C3

The scientific audit contains four stored gradient-summary records covering two minibatches each, plus four independently regenerated batch-0 gradient records. Stored actual parameter tensor norms and conflict summaries replayed successfully. Regenerated gradient vectors match their saved vectors exactly; regeneration performed no model updates or new critic fits and did not open INNER labels.

The headline replay independently derives utility/class metrics and reference contrasts from original frozen wires, checks original-to-derived per-row utility and score inputs, regenerates FIT folds 0/1 label tables and exact class counts, and compares FIT2/INNER task association and supported strata. Missing-class recall availability is explicit; fixed-list balanced accuracy retains zero contributions for absent classes. FIT2 has 5,211 rows/5,209 groups; the training label tables have 10,223 rows/10,219 groups. These are interpretation diagnostics, not a universal privacy floor. Source model training had seen all FIT.

Real ATT validation and INNER null/shuffled/planted recovery controls were independently replayed and compared. All six required paired baseline control dependencies are valid: U0, RAW_JOINT_R03_E5, and JOINT_CONDITIONAL_R03_E5 at seeds 0 and 1. Privacy inference uses the frozen dependency maps and retains the family size if a required control is missing or fails. The supplementary complete-view family evaluates the frozen all-applicable matched reader bank; primary K108 and supplementary K18 remain separate registered families.

The four independent representative refits are U0 seed 0 REP_LR_C1, U0 seed 1 HGB_15x200, and RAW_JOINT_R03_E5 seeds 0/1 REP_MLP_64x32. They fit only on the inherited ATT training people and orient only on inherited ATT validation people. Their source INNER release is label-free.

## Preserved attempt history and label access

The first cache fixture failed with a Python class lexical-scope NameError before scientific arithmetic. Its original source and failed attempt remain preserved; V2 corrects fixture plumbing and passes exact cached/uncached result equality. See [fixture correction](../INDEPENDENT_FIXTURE_CORRECTION_F.json).

ROOT's original all-release forward adapter V1 completed successfully and remains retained in [V1 output](INDEPENDENT_ALL_RELEASE_FORWARDS.json). Its generic saved-ATT archive loader materialized the permitted ATT SEX member, along with split members. This was authorized ATT data; it did not read or hash INNER labels. V1's broad `labels_opened:false` field must not be interpreted as no ATT-label materialization.

ROOT preserved V1 and ran [V2](INDEPENDENT_ALL_RELEASE_FORWARDS_V2.json), which opens only row_id/unit/r1/r2 saved-wire members and records `true_label_arrays_materialized:false`. The separate R7-owned implementation independently repeats that restriction, validates exact archive member schemas, and records `ATT_label_or_split_members_opened:false` and `INNER_label_bytes_read_or_hashed:false`. Whole authorized ATT archive bytes, including unopened SEX/split member bytes, are hashed for registered-file custody in V2 and R7; the reports explicitly acknowledge this.

These forward/source/gradient/refit label restrictions are scoped to those operations. Later headline and inference replay uses explicitly registered INNER truth under the committed reporting lock; it does not claim to be label-free. No blanket no-label-access claim is made for the full campaign.

## Backup custody and restore closure

The independent backup-only numerical verification is complete. [Its public report](INDEPENDENT_BACKUP_ONLY_REPLAY.json) verifies U0 seed 0 full with REP_LR_C1 and RAW_JOINT_R03_E5 seed 0 full with REP_MLP_64x32. Each uses 2,235 source input rows; both recipients' complete wires and each frozen reader score vector match exactly. Code and all numerical sources resolve inside the restored tree. No original model/reader source fallback, label reads/hashes, or new fits occurred. The queue receipt is `done`, return code 0, completed at `2026-10-13T16:07:45+00:00`; its public output SHA256 is `5399c10166a98cb08df45d93cbdb58d3bcfaabda726d4d527695854f9c6d3fda`. The [restore API](RESTORE_VERIFIER_API.md) documents `restore_worker.py`, SHA256 `3529a71eeb036898e8db5010f7ae58c731069a0ce222d35bde9037f1db8a6d1d`.

All twelve R6 snapshot-only numerical restores also have completed successful receipts and public PASS outputs. They cover U0, RAW_JOINT_R03_E5, and JOINT_CONDITIONAL_R03_E5, both seeds, full and head-row-space releases. All 24 recipient wire comparisons and 72 saved reader checks report zero differences. Every report records runtime code from the snapshot and no original sources used. These R6 checks provide complementary deployment evidence; the two representative R7 checks use independently owned scientific replay.

The [versioned snapshot](../r6/VERSIONED_SNAPSHOT.json) and [versioned restore](../r6/VERSIONED_RESTORE.json) both passed uncached full readback for 4,984 files totaling 1,372,495,673 bytes (approximately 1.3 GiB). Originals remain intact. Both receipts explicitly record `same_device:true` and `separate_filesystem_verified:false`. The [custody inventory](../r6/CUSTODY_INVENTORY.json) passed under V2.

The first custody inventory attempt was interrupted after avoidable metadata CPU cost from parsing the 7.6 MiB queue state per file; it materialized no numerical arrays. That attempt remains preserved. V2 checks queue authorization every 128 files and hashes every file successfully. A metadata index naming collision is also preserved in the original RESTORE_PREPARED spec; all twelve deployment restores were rebound to the distinct DEPLOYMENT_RESTORE_PREPARED index. These are custody/metadata corrections, not additional scientific failures.

## Remaining material gap and limits

Off-device or separate-filesystem backup verification remains pending. Same-device snapshot and restore plus actual restored model/reader replay are verified, but they do not establish survival of loss of the source device. No other material discrepancy was identified in the completed independent checks reviewed here.

Utility interval equality is tolerance-based as explained above; it must not be described as bit-exact JSON agreement. Passing replay establishes agreement for frozen models, readers, people, metrics, and registered exploratory interval procedures; it does not establish broader training-seed or population coverage.
