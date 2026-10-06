# Exposure ledger: confidence capacity and privacy study (qpc)

## Registered exposure statement

Registered before any fitting, verbatim (prompt section 5):

> "The design is motivated by previously opened Adult development results, including the completed decision-preserving compression study. Every assessment row has been used historically. This is an exploratory, locked development comparison. Its nominal intervals condition on fitted artifacts and do not correct for the adaptive research history. It is not fresh confirmation or a population privacy guarantee."

## What this means

- **Every row has been used before.** Every role and every assessment pool is a UCI Adult record that earlier PCRL studies used. They trained the encoders, fitted the heads, LEACE maps, FARE trees, critics, attackers and compression policies, selected models, and scored and reported results.
- **The assessment has now been scored by two immediately preceding studies.** OSF_DEVELOPMENT_ASSESSMENT was scored once by the online-strength frontier study (osf), and once more by the decision-preserving compression study (dpc). The dpc outcomes shaped this study's design. Scoring the rows again is exploratory development scoring: not fresh data, not confirmation and not an independent replication.
- **Masking labels is procedural.** This study masks the assessment labels until its own pushed EVALUATION_LOCK. That keeps qpc's fitting and selection separate from those rows. It does not make the rows unseen, and it cannot undo the adaptive history.
- **No new population.** No California ACS, Texas, New York or other population is acquired or opened. The analysis unit is the Adult exact-record group.

Owner: data/custody role (E). Written 2026-10-06 (UTC). Pinned source evidence: `research/pcrl-decision-preserving-compression-v1` @ `0a7b05a52746544213742f50efd0a48167efffb1` (dpc). Teacher provenance: `research/pcrl-online-strength-frontier-v1` @ `925e0fddfcb666116c6179575339728a324ed78e` (osf). Both remote heads equal their pins (`SOURCE_INDEX.json`).

The role and pool history up to and including osf is in the dpc ledger, `results/pcrl_decision_preserving_compression_v1/EXPOSURE_LEDGER.md` at the source commit, and through it the osf ledger. Those files are the authority for that history. This file adds dpc's uses and this study's uses. It publishes no label statistic of any role.

## 1. Roles (reused exactly; counts and hashes in `ROLE_MANIFEST.json`)

`qpc.data.load()` imports the pinned `dpc.data` and `osf.data` unchanged (both byte-identical to their blobs at the pins) and adds no new rule. It verifies every role's rows, exact-record groups, sorted-row-id hash and group-set hash against both the osf and the dpc `ROLE_MANIFEST.json` at their pins.

| Role | Rows | Exact-record groups | Use in this study (qpc) |
|---|---:|---:|---|
| OSF_DEFENSE_FIT | 15,434 | 15,428 | Stage A k-means, Stage B fine partitions, prototypes, compression policies and privacy objectives (fit here only) |
| HEAD_VALIDATION | 1,500 | 1,499 | Historical deployed-head C selection only. No head is refitted, recalibrated or selected; no label is read |
| AUDIT_FIT | 6,065 | 6,061 | Attackers fitted here only |
| INNER_SELECTION | 2,235 | 2,234 | Stage A capacity gate, rate selection, attackers and nominees chosen here only |
| OSF_DEVELOPMENT_ASSESSMENT | 13,936 | 13,929 | Single locked exploratory assessment after the pushed qpc EVALUATION_LOCK |

- **Group isolation.** No OSF_DEFENSE_FIT exact-record group occurs in AUDIT_FIT, INNER_SELECTION or OSF_DEVELOPMENT_ASSESSMENT. The full role-by-role shared-group matrix is zero off the diagonal (`ROLE_MANIFEST.json`, `group_isolation`).
- **Assessment pools.** ORIG_ASSESSMENT 5,243, RGJ_DEV 3,397, SMF_DEV 3,796 and CERT 1,500 rows, as in dpc. No group or pool is removed or pooled differently.
- **Dropped rows.** The osf rule drops the same 35 rows (17 excluded_exposure, 18 excluded_dup).
- **Label sealing.** SEX, race and both task labels are −1 on every assessment row at load. Only `qpc.assess` may unseal, and only when the committed qpc `EVALUATION_LOCK.json` is byte-identical on `origin/research/pcrl-confidence-capacity-v1`. dpc's own unseal path cannot open these rows for qpc.

## 2. History per role, including the two source studies

| Rows | Up to smf (dpc ledger) | osf | dpc | qpc (this study) |
|---|---|---|---|---|
| OSF_DEFENSE_FIT | T | **T**: encoders, heads, LEACE, FARE | **T**: fine KL partitions, prototypes, 258 policies, privacy objectives | **T**: k-means codebooks, fine partitions, policies, privacy objectives |
| HEAD_VALIDATION | S | **S**: deployed-head C | Used only through the frozen heads | Used only through the frozen heads |
| AUDIT_FIT | T | **T**: attackers | **T**: attackers | **T**: attackers |
| INNER_SELECTION | S | **S**: L\*, C\* | **S**: configurations, attackers, nominees (T\* = C_global = U continuous) | **S**: capacity gate, rates, attackers, nominees |
| SMF_DEV, RGJ_DEV, ORIG_ASSESSMENT, CERT | see dpc ledger §2 | **Sc once** | **Sc once** | **Sc once**, after the qpc EVALUATION_LOCK |

Key: **T** fitted on, **S** selected on, **Sc** scored or reported on.

**dpc facts** (source `RESEARCH_DECISION.md` and `EVALUATION_LOCK.json` at the source commit):

- **Lock and scoring.** The dpc EVALUATION_LOCK (written 2026-10-06T01:18:31Z, commit `3140d4c`) was pushed before its single assessment, which scored every OSF_DEVELOPMENT_ASSESSMENT row.
- **Label.** EXPERIMENTAL_NO_ADVANTAGE. J\* and P\* had no feasible nominee; T\* = C_global = U continuous; the matched-control slot recorded NO_FEASIBLE_CONTROL.

## 3. Reused artifacts (admitted, frozen)

`SOURCE_ADMISSION.json` gives the plan and every pinned hash; `provenance/ADMISSION_RESULT.json` records the admission run.

- **Teachers.** U and RAW-J β 0.3 (`rel__s{k}__U`, `rel__s{k}__RAW-J_b0.3`, epoch 40, seeds 0–2), with frozen encoders and deployed StandardScaler/LogisticRegression heads. They are restored from the hash-pinned dpc admitted copies (the osf store is the fallback source). Their outputs are rebuilt by qpc's forward application and must be bitwise equal to the dpc teacher units before any code is fitted.
- **References.** Official LEACE (E), official FARE (F) and the no-fairness FARE twin (F0), seeds 0–2. They are admitted by a verified copy of the dpc reference units, with array parity against the dpc admitted artifacts. FARE/F0 seed units include recorded aliases of the seed-0 tree, which are the same reference, not an independent replication.
- **Fit roles.** All were fitted on OSF_DEFENSE_FIT, with heads selected on HEAD_VALIDATION. No assessment group entered their fitting or selection (osf `ADMISSION.json` and dpc `ADMISSION.json` receipts at the pins).
- **Prior encoding.** These artifacts already encoded every assessment row. Their scores were computed and opened by osf, and compressed and scored again by dpc.

## 4. Previously opened results that motivated this design

These were seen before this study was designed. They are design motivation, not evidence produced by this study.

- dpc found that every tested code preserved its teacher's predicted class pointwise.
- dpc assessment means: U continuous pair SEX AUC about 0.858. U task-only m8 about 0.825, with occupation log-loss cost about 0.020 nats and Brier cost about 0.007. U JOINT m8 λ 0.1 about 0.807, with costs about 0.0214 and 0.0079.
- Joint improved over matched task-only compression by about 0.0176 AUC, below the 0.02 target. Both sequential orders performed almost the same as joint.
- No finite code in the registered m = 2, 4, 8 bank qualified. Most fine clustering runs hit the 20-round cap without converging.
- U's decision-only pair AUC was about 0.74 under the declared attackers.

Because these numbers were opened on the same assessment rows, this study's assessment is adaptive reuse. Its nominal intervals and Bonferroni bounds condition on the fitted artifacts and do not correct for this history.

## 5. Input contract and proxies

- **Input.** `<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz`, sha256 `e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12`, present and re-hashed. Nothing was downloaded.
- **Columns.** 83 permitted columns in the pinned order (feature-name hash `7101f2fb…d975`; full order in `ROLE_MANIFEST.json`).
- **Excluded.** SEX, race, the income and occupation labels, fnlwgt and identifiers.
- **Proxies kept.** Relationship categories (Husband/Wife) are retained, as in the source. Recovery therefore includes signal carried by permitted inputs.
- **Removed scratch input.** The dpc closeout removed its session-scratch deploy inputs (an 83-column X plus feature names, and a schema file; never in git). qpc reconstructs only that derived input, from the declared private source through the pinned loader, into its private store.

## 6. Historical identifying drive-folder name (recorded separately; not rewritten)

- **What.** Four earlier commits on the rgj research branch contain the name of a folder on the external drive that identifies its owner. The osf ledger (§9) and the osf and dpc closeouts recorded this.
- **Status.** Unchanged. No history rewrite or force-push is authorised (prompt section 1), so this study does not rewrite those commits and does not claim the exposure is removed.
- **This study.** Public qpc files carry placeholders only (`<PRIVATE_CACHE>`, `<WORKTREE>`, `<DRIVE_ROOT>`). The writers refuse any home folder, user name or volume path. The drive is detected by the content of known backups, never by its name.

## 7. Publication cautions

- **No new label aggregates.** Before the EVALUATION_LOCK, do not publish any new OSF_DEFENSE_FIT or pool label aggregate in a form that, combined with older public aggregates, would let a sealed pool's share be derived by subtraction.
- **Placeholders.** Public files contain no home folder, user name, volume label, drive path, row id or person-level value.
- **History.** Earlier commits are unchanged. No history is rewritten.
