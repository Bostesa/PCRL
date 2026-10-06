# Exposure ledger: decision-preserving joint compression study

## Registered exposure statement

Registered before any fitting, verbatim (prompt section 4):

> "The design is motivated by previously opened Adult development results. Every assessment row has been used historically. This is an exploratory, locked benchmark comparison. Its nominal intervals condition on the fitted artifacts and do not correct for the adaptive research history. It is not fresh confirmation or a prospective population guarantee."

## What this means

- **Every row has been used before.** Every role and every assessment pool is a UCI Adult record that earlier PCRL studies used. They trained encoders, fitted heads, LEACE maps, FARE trees, critics and attackers, selected models, and scored and reported results.
- **The assessment is a reused benchmark.** OSF_DEVELOPMENT_ASSESSMENT was scored once by the previous study. That was the online-strength frontier study, osf, scored after its pushed EVALUATION_LOCK. Those outcomes shaped this study's design (§4). Scoring it again is exploratory benchmark scoring. It is not fresh data, not confirmation and not an independent replication.
- **Masking labels is procedural.** This study masks the assessment labels until its own pushed EVALUATION_LOCK. That keeps the new procedure's fitting and selection separate from those rows. It does not make the rows unseen, and it cannot undo the adaptive history.
- **No new population.** No California ACS, Texas, New York or other confirmation population is acquired or opened. No survey-weight or household procedure from the ACS line is used: this is Adult, and the analysis unit is the exact-record group.

Owner: data/custody role. Written 2026-10-05 (UTC). Pinned source: `research/pcrl-online-strength-frontier-v1` @ `925e0fddfcb666116c6179575339728a324ed78e`. The remote head equals the pin (`SOURCE_INDEX.json`).

The role and pool history up to and including smf is copied from the source ledger, `results/pcrl_online_strength_frontier_v1/EXPOSURE_LEDGER.md` (sha256 at the pin `77dde2d3…c7c5`). That file is the authority for those facts, and its §2–§4 give per-study detail. This file adds the osf uses and this study's uses. It publishes no label statistic of any role.

## 1. Roles (reused exactly; counts and hashes in `ROLE_MANIFEST.json`)

`dpc.data.load()` imports the pinned `osf.data` unchanged and adds no new rule. Its counts, groups and sorted-row-id hashes equal the pinned osf `ROLE_MANIFEST.json` for every role, subrole and pool. `provenance/role_check.py` recomputes the roles independently.

| Role | Rows | Exact-record groups | Use in this study (dpc) |
|---|---:|---:|---|
| OSF_DEFENSE_FIT | 15,434 | 15,428 | Fit score partitions, prototypes, compression policies and privacy objectives here only |
| HEAD_VALIDATION | 1,500 | 1,499 | Historical deployed-head C selection. No head is refitted or selected. An optional secondary temperature control may be chosen here only. |
| AUDIT_FIT | 6,065 | 6,061 | Fresh attackers fitted here only |
| INNER_SELECTION | 2,235 | 2,234 | Configurations and attackers chosen here only |
| OSF_DEVELOPMENT_ASSESSMENT | 13,936 | 13,929 | Single locked exploratory assessment after the pushed EVALUATION_LOCK |

**Assessment pools.** The assessment combines four pools, by `osf.data`:

| Pool | Rows |
|---|---:|
| ORIG_ASSESSMENT | 5,243 |
| RGJ_DEV | 3,397 |
| SMF_DEV | 3,796 |
| CERT | 1,500 |

- **Groups.** No assessment group has a row in any fitting or selection role (0 shared groups for each of the four roles).
- **Dropped rows.** 35 rows are dropped: 17 excluded_exposure and 18 excluded_dup.
- **CERT.** The CERT pool is included under the source custody verdict ESTABLISHED (source ledger §6). That verdict concerns fit provenance only; the pool remains historically exposed.

## 2. History per role, including the source study (osf)

| Rows | PCRL v2 … smf (source ledger §3–§4) | osf (source study) | dpc (this study) |
|---|---|---|---|
| OSF_DEFENSE_FIT | T: trained every predecessor model; smf refitted everything on exactly these rows | **T**: all 21 bank configurations × 3 seeds (U, RAW-J/L, NORM-J/L), deployed heads, LEACE maps, FARE trees | **T**: score partitions, prototypes, policies, privacy objectives (on teacher scores; no encoder retraining) |
| HEAD_VALIDATION | **S**: deployed-head C in oar/odx/cap, jcv/pnx, rgj, smf | **S**: deployed-head C of every osf release, including the admitted U and RAW-J β 0.3 heads | Used only through the frozen historical heads |
| AUDIT_FIT | T: every attacker slate, pilot through smf | **T**: inner and final attackers | **T**: fresh attackers |
| INNER_SELECTION | **S**: attackers, gates, nominees in jcv … smf | **S**: L\*, C\* = RAW-J β 0.3; N\*/R\* NO_FEASIBLE_NOMINEE | **S**: configurations, attackers, C_match / C_global / T\* / J\* / P\* |
| SMF_DEV | T up to rgj; **Sc once** by smf | **Sc once** (pool of OSF_DEVELOPMENT_ASSESSMENT) | **Sc once** after the dpc EVALUATION_LOCK |
| RGJ_DEV | **S** (head C) oar…pnx; **Sc once** by rgj | **Sc once** | **Sc once** |
| ORIG_ASSESSMENT | **Sc**: outer assessment of pilot, bench, oar, odx, cap, jcv, pnx | **Sc once** | **Sc once** |
| CERT | T (attackers) pilot/bench; **C** (FARE certificates) oar/odx/jcv/pnx | **Sc once** | **Sc once** |

Key: **T** fitted on, **S** selected on, **Sc** scored or reported on, **C** certificate statistic.

**osf facts** (source `RESEARCH_DECISION.md` and `COST_AND_CLOSEOUT.md` at the pin):

- **Lock and scoring.** The osf EVALUATION_LOCK (`3554235`) was pushed 2026-10-05 04:19:24Z. The single assessment scored 72 outer units over all 13,936 OSF_DEVELOPMENT_ASSESSMENT rows (about 04:19–04:51Z).
- **Published results.** Its per-seed and mean results on these rows are published in `ALL_LEVELS.csv`, `PRIMARY_ENDPOINTS.csv`, `SECONDARY_ENDPOINTS.csv` and `ACTUAL_TASK_UTILITY.csv`. Their sha256 values at the pin are in `SOURCE_INDEX.json`.

## 3. Reused artifacts (admitted, frozen)

`ADMISSION.json` lists every admitted unit, with its hashes and label-free checks. The admitted artifacts are:

- **Teachers.** U (`rel__s{k}__U`) and RAW-J β 0.3 (`rel__s{k}__RAW-J_b0.3`) at epoch 40, seeds 0, 1 and 2, each with frozen encoders and deployed StandardScaler/LogisticRegression heads.
- **References.** Official LEACE (`lc__s{k}__E`), official FARE F (`fare__s{k}__p{0,1}__c1`) and its no-fairness twin F0 (`fare__s{k}__p{0,1}__Z1`), with the official trees.

**Fit roles.** All of these artifacts were fitted on OSF_DEFENSE_FIT, with heads selected on HEAD_VALIDATION; no assessment group entered their fitting or selection. The receipts are the osf `ADMISSION.json` and the `ROLE_MANIFEST.json` section `assessment_exclusion_from_eligible_models` at the pin. This study re-verifies them label-free: the heads' scaler moments equal those of the OSF_DEFENSE_FIT features, the LEACE fit-row and feature hashes match, and the FARE fit-row fingerprints match.

**Prior encoding.** These artifacts already encoded every assessment row: the osf releases cover all 39,170 rows. The teacher scores used here were therefore computed, and opened, by osf.

## 4. Previously opened results that motivated this design

These were seen before this study was designed. They are design motivation, not evidence produced by this study.

- **Strength study outcome.** The osf strength study was a complete negative (EXPERIMENTAL_NO_ADVANTAGE). Its assessment means were:

  | Model | Income accuracy | Occupation accuracy | Pair SEX AUC |
  |---|---:|---:|---:|
  | RAW-J β 0.3 (C\*, deployable best) | 0.845 | 0.467 | 0.802 |
  | U | 0.844 | 0.475 | 0.883 |

- **Output-only audits.** osf's descriptive output-only audits used the smaller SECONDARY slate. Averaged over the three RAW-J β 0.3 seeds, pair AUC was:

  | View | Pair AUC |
  |---|---:|
  | Features + scores | 0.801533 |
  | Centred scores alone | 0.782677 |
  | Decisions alone | 0.692915 |

  U's decision-only pair AUC was about 0.739. These are leads, not results. The slates differ, so their differences are never cited as a matched advantage.
- **Task quality.** U's mean task log losses were about 0.338 (income) and 1.269 (occupation). The rarest occupation class had recall of about 0.022, and RAW-J's occupation log loss was about 0.0134 worse than U's.

Because these numbers were opened on the same assessment rows, this study's assessment is adaptive reuse. Its nominal intervals and Bonferroni bounds condition on the fitted artifacts and do not correct for this history.

## 5. Input contract and proxies

- **Columns.** The input has 83 permitted columns in the pinned order (feature-name hash `7101f2fb…d975`; the full order is in `ROLE_MANIFEST.json`). Five numeric columns are re-standardised on OSF_DEFENSE_FIT; 78 one-hot columns use the loader's fixed vocabulary.
- **Excluded.** SEX, RAC1P/race, the income and occupation labels, fnlwgt and identifiers are excluded.
- **Proxies kept.** Relationship categories (Husband/Wife) are kept as permitted proxies. Recovery therefore includes signal carried by permitted inputs.
- **Shared input vectors.** 5,191 assessment rows share an 83-column input vector with an OSF_DEFENSE_FIT row (source ledger §5). They are different exact-record groups and are never pooled as one person.

## 6. Publication cautions

- **No new label aggregates.** Do not publish any new OSF_DEFENSE_FIT or pool label aggregate before the EVALUATION_LOCK in a form that, combined with older public aggregates, lets a sealed pool's share be derived by subtraction (source ledger §9).
- **Placeholders.** Public files use only `<PRIVATE_CACHE>`, `<WORKTREE>`, `<SOURCE_WORKTREE>` and `<DRIVE_ROOT>`. They contain no home folder, user name, volume label, drive path, row id or person-level value.
- **History.** Historical exposures in earlier commits are unchanged. No history is rewritten.
