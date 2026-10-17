# Exposure and selection audit: frozen finalist `B3CTL_UNCOUPLED_MIX50_C6A` and its bank

**Role.** Output-mapping and exposure reviewer (separate agent), PCRL frozen-finalist validation and confirmation-readiness study, 2026-10-17 (written about 22:10–22:40Z). PROTOCOL.md §4.

**Data.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, in the existing development roles of the protected-tree replication. The service predicts an income class (2 classes) and an occupation group (5 classes) from AGEP, WKHP, COW, SCHL, MAR and POBP. Each recipient receives only its own probability vector and argmax decision. SEX is the recorded demographic column that separate evaluation readers try to infer.

**What this reviewer read.** Public campaign records, the code, admitted metadata (release records and access-log snapshots listed in this campaign's `SOURCE_ADMISSION.json`), and the lineage DEV_REPORT access log (events only). The only person-level data opened were the `row_id` and `unit` fields of the six label-free features exports (§5). No SEX, task label, DEV_REPORT truth or sealed-assessment content was read.

**Plain statement.**
- Numerical reproduction does not erase the adaptive selection history recorded below.
- It does not make the source DEV_REPORT evaluation fresh confirmation.
- Anything not documented in a record is marked UNKNOWN. Missing evidence is not proof of independence.

Abbreviations for record paths:
- `AMD/` = `results/pcrl_autonomous_method_development_v1/` (the source campaign);
- `REP/` = `results/pcrl_acs_protected_tree_replication_v1/`;
- `HYB/` = `results/pcrl_acs_frozen_hybrids_v1/`;
- `SCMB/` = `results/pcrl_scm_engineering_benchmark_v1/`;
- `SCMUM/` = `results/pcrl_scm_utility_matched_v1/`.

Times are UTC on 2026-10-17 unless dated otherwise.

---

## 1. Which components each bank release contains

| Release (role here) | Occupation output | Income output | Post-processing | Source MODEL_SELECT eligibility (`AMD/ANALYSIS_LOCK.json` → `availability_ms_point_eligible`) |
|---|---|---|---|---|
| FINALIST (frozen candidate) | T_TREE occupation routing + JOINT_TREE E20 corrections (bitwise H_LJ's) | 6 levels of d = 0.5 × LOCAL_TREE + 0.5 × JOINT_TREE E20 income log-odds | MODEL_FIT thresholds; CALIBRATION-fitted per-level probabilities (h) | — (candidate) |
| NEURAL_RAW (eligible reference) | U0 + RAW-J fine-tuning (ρ 1.0, checkpoint E2) | same network family | none | true |
| H_LJ (eligible reference) | JOINT_TREE E20 | LOCAL_TREE E20 (continuous) | none | true |
| TASKONLY (claim M, step 1) | TASK_REFIT E20 | 6 levels of TASK_REFIT E20 income log-odds | MODEL_FIT thresholds; CALIBRATION h | true ("TO needs only presence") |
| B2TWIN (claim J) | JOINT_TREE E20 | 6 levels of LOCAL_TREE E20 income log-odds | MODEL_FIT thresholds; CALIBRATION h | true |
| PUB_IBSI (diagnostic; cannot be counted as beaten) | Cell table from a merged 512-cell partition (IBSI objective, γ = 2.0, 473 / 475 cells) | 32-group income table | checkpoint and γ chosen on MODEL_SELECT | **false** |

Sources:
- `AMD/FINALIST_LOCK.json` → `bank`, `release_records[...].base_occupation`, `.income_statistic_definition`;
- this campaign's `FROZEN_CANDIDATE_MANIFEST.json` → `candidate.definition`, `bank.releases`;
- `HYB/FROZEN_BANK_MANIFEST.json` → `definition.H_LJ`;
- `SCMUM/SELECTION_AND_MATCHING.md` (role table, row `IBSI_NLLM`).

---

## 2. Fitting ledger: which records and labels fitted each component

| # | Component (used by) | Records and labels that fitted it | Fixed or selected? | Record (file → field) |
|---|---|---|---|---|
| F1 | 145-column input encoder (all six releases) | MODEL_FIT features only: categorical levels kept if they occur ≥ 10 times among MODEL_FIT people; AGEP and WKHP z-scored with MODEL_FIT moments. Label-free. | Fixed by the registered schema | `REP/POPULATION_AND_SCHEMA.md` → "Inputs"; `acs_replication/schema.py` → `CATEGORY_MIN_COUNT`, `NUMERIC_SCALING`; `acs_replication/extract.py` → `fit_encoding(people['MODEL_FIT'])` |
| F2 | T_TREE CatBoost teachers, income and occupation, per seed (FINALIST, H_LJ, B2TWIN, TASKONLY: routing) | MODEL_FIT, 20,000 people in 13,337 households: income and occupation labels; no SEX. Tree count from a 15% household internal split of MODEL_FIT (a label-free draw followed by a whole-household class-support repair that uses the labels; `acs_replication/data.py` → `internal_split`); then refit on all MODEL_FIT. | Fixed registry (no search). A usefulness gate read MODEL_SELECT labels (`SUPPORTED_CONTINUE`). | `REP/SCIENCE_LOCK.json` → `models.T_TREE.rule`, `access.every_method`; `acs_replication/stage_a.py` (docstring); `REP/STAGE_A_HEADROOM.json` → `role` = MODEL_SELECT, `gate` |
| F3 | Stage B LOCAL_TREE E20 corrections, per seed (H_LJ income; B2TWIN income statistic; FINALIST income statistic, weight 0.5) | MODEL_FIT income and occupation labels (task pool) and MODEL_FIT SEX (privacy pool, online critics); local privacy objective; λ calibrated on MODEL_FIT only. | Fixed: ρ = 1, checkpoint E20; "fixed and reported, never selected" | `REP/SCIENCE_LOCK.json` → `models.LOCAL_TREE.recipe.task_pool`, `.privacy_pool`, `.calibration`, `models.LOCAL_TREE.checkpoint`; `REP/SELECTION_LOCK.json` → `rule`; `acs_replication/stage_b.py` (docstring; `load_role('MODEL_FIT', (..., 'sex'))`) |
| F4 | Stage B JOINT_TREE E20 corrections, per seed (occupation output of FINALIST, H_LJ and B2TWIN, bitwise identical; FINALIST income statistic, weight 0.5) | The same MODEL_FIT labels and SEX, plus the pair term. | Fixed: ρ = 1, E20; registered as an "OPTIONAL diagnostic ... no strength search, no nomination". It is utility-infeasible as a complete release on MODEL_SELECT. | `REP/SCIENCE_LOCK.json` → `models.JOINT_TREE.status`; `REP/SELECTION_LOCK.json` → `fixed_services.JOINT_TREE.failed_clauses` |
| F5 | Stage B TASK_REFIT E20 corrections, per seed (TASKONLY, both outputs) | MODEL_FIT income and occupation labels; privacy terms disabled (λ = 0). The shared Stage B trainer also reads MODEL_FIT SEX for its critics. | Fixed: E20 | `REP/SCIENCE_LOCK.json` → `models.TASK_REFIT.note`; `acs_replication/stage_b.py` (docstring) |
| F6 | NEURAL_RAW ρ 1.0 E2, per seed | Start: U0, fitted on MODEL_FIT task labels (15% household-disjoint internal fold, then refit on all MODEL_FIT). Then RAW-J fine-tuning: MODEL_FIT task labels and MODEL_FIT SEX critics; 12 configurations (ρ ∈ {1, 3, 10} × {E2, E5, E10, E20}). | **Selected on MODEL_SELECT**: 24 point clauses with MODEL_SELECT labels, then the minimum OUT_LR pair proxy. That reader is fitted on the MODEL_FIT pair wire with MODEL_FIT SEX and scored on MODEL_SELECT SEX. Selection happened before any DEV_REPORT read. | `REP/SCIENCE_LOCK.json` → `models.U0.note`, `models.NEURAL_RAW`, `selection.rule`; `REP/SELECTION_LOCK.json` → `frozen.NEURAL_RAW`, `DEV_REPORT_labels_opened` = false, `utc` 2026-10-16T00:18:37Z; `acs_replication/neural_raw_worker.py` |
| F7 | PUB_IBSI (`IBSI_NLLM`), per seed | Five fold CatBoost task teachers trained on MODEL_FIT household folds (income and occupation labels). An out-of-fold SEX propensity fitted on MODEL_FIT SEX (training only). A 512-cell KL k-means partition of the MODEL_FIT out-of-fold pairs. Greedy merging minimising ΔNLL_income + ΔNLL_occupation − γ ΔI(T; SEX). The released tables are pooled MODEL_FIT out-of-fold teacher probabilities, with a 32-group income rule by MODEL_FIT mass. CALIBRATION is not used. | **Selected on MODEL_SELECT** (SCM utility-matched follow-up): an occupation-NLL match within 0.001 nats of SCM's MODEL_SELECT spend, 12 point clauses, then the minimum proxy; one common γ by mean proxy. Result: γ = 2.0, checkpoint N0.007. | `SCMB/PROTOCOL.md` §1 (roles), §2 (teachers, propensity), §3 (partition), §4 (grouping), §5 (row `IBSI_b_g`); `SCMUM/PROTOCOL.md` §2, §7 (`IBSI_NLLM`); `SCMUM/SELECTION_LOCK.json` → `utc` 05:27:06Z |
| F8 | Income thresholds (FINALIST: the mixed statistic d; B2TWIN: LOCAL_TREE income log-odds; TASKONLY: TASK_REFIT income log-odds), per seed | **MODEL_FIT, label-free**: 0 by construction (decision-aligned), plus the terciles of each side of 0. Seed-0 FINALIST bins hold 4,114 / 4,113 / 4,114 / 2,553 / 2,554 / 2,552 MODEL_FIT people. The statistic itself was fitted in-sample on these same MODEL_FIT people with their labels and SEX (F3, F4, F5). That is why a few MODEL_FIT people sit exactly on a threshold (D3). | Fixed rule (m = 6 aligned) | `AMD/FINALIST_LOCK.json` → `release_records["B/B3CTL_UNCOUPLED_MIX50_C6A\|S0"].income_statistic_definition`, `.labels_used_at_training.MODEL_FIT`; admitted `runs/B/<config>/S<seed>/RELEASE.json` → `coarse_cell_counts.MODEL_FIT`; `AMD/PROTOCOL_POSTRUN_CLARIFICATIONS.md` D3 |
| F9 | Income calibration h (FINALIST, B2TWIN, TASKONLY), per seed | **All 2,982 CALIBRATION people (1,974 households), income labels only, no SEX.** L2 logistic regression (C = 1) on the standardized one-hot level: 5 coefficients and an intercept. The same CALIBRATION income labels were also the data for the m and α screens (S2, S3). CALIBRATION task labels also fitted Ucal's temperature, the utility reference of claim E. | Fixed construction | `AMD/FINALIST_LOCK.json` → `release_records[...].h_fit_role`, `.labels_used_at_training.CALIBRATION`; `FROZEN_CANDIDATE_MANIFEST.json` → `per_seed.S*.calibration_h`; `AMD/roles/investigator_b/REPORT.md` → "Release", "Diagnostics"; `REP/SCIENCE_LOCK.json` → `models.Ucal.rule` |
| F10 | Final source readers (all releases) | ATTACK_FIT (8,001 people; SEX) for fitting; ATTACK_VALIDATION (3,000; SEX) for orientation and winner selection; 174 fits per seed. DEV_REPORT was scored from released outputs only. Every lineage campaign that fitted readers used these same two roles: the replication, the hybrids, the SCM benchmark, the SCM utility-matched follow-up and the AMD campaign. | Frozen in the source REPORTING_LOCK | `AMD/HANDOFF.json` → `commits[4]` (721d4163d); `REP/SCIENCE_LOCK.json` → `adversary_and_readers` |

**Summary of label use by role.**
- **MODEL_FIT labels and SEX.** They trained every privacy correction and teacher (F2–F7).
- **CALIBRATION income labels.** They fitted the finalist's calibration and selected m and α.
- **MODEL_SELECT labels and SEX.** They selected NEURAL_RAW, PUB_IBSI, the finalist and the eligibility labels.

---

## 3. Selection and designation ledger

| # | Decision | Decided on (role / evidence) | When | Record (file → field) |
|---|---|---|---|---|
| S1 | Family B card: an income code beside the JOINT_TREE occupation output | Design. The card quotes published DEV_REPORT values: LOCAL_TREE income 0.558 / 0.549, occupation 0.582 / 0.584, pair 0.657 / 0.649. The family A card quotes JOINT_TREE pair 0.604 / 0.607. The family B card also quotes the published MODEL_SELECT teacher headroom. | ~17:05Z | `AMD/MECHANISM_CARDS.md` → Family A and Family B, "Bottleneck addressed" and "Why utility might survive" |
| S2 | **m = 6 levels** (chosen for B2, inherited by B3, B3CTL, B2TWIN and TASKONLY) | **CALIBRATION income labels**: seed-0 screen `DIAG2_S0` (m ∈ {4, 6, 8, 12, 16}, aligned; the smallest m inside the caps). The ledger says "no MODEL_SELECT use before this score". The card revision's reasoning nevertheless cites MODEL_SELECT being "more favourable to LOCAL_TREE income than CALIBRATION by about 0.002 nats"; how that figure was derived is **UNKNOWN**. The MODEL_SELECT baseline table of the 22 frozen references (ledger 1–22) already existed. | screen ended 17:13:21Z | `AMD/EXPERIMENT_LEDGER.csv` entry 31 → `selection_effort`; `AMD/roles/investigator_b/CARD_REVISIONS.md` → B2 "Why m = 6"; `AMD/RESOURCE_ACCOUNTING.json` → `units[AMD_B_diag2_S0_V1].utc` |
| S3 | **α = 0.5** (chosen for the coupled B3, inherited by B3CTL) | **CALIBRATION income labels**: seed-0 screen `DIAG3_S0` (α ∈ {0, .25, .5, .75, 1} × m ∈ {6, 8} × coupled / uncoupled). The rule was "largest alpha whose screen point is inside the caps **with the MS offset**", which is edge-seeking. The screen ran after the B2-family MODEL_SELECT results (17:17–17:20Z). The offset's exact value and source are **UNKNOWN**: `DIAG3_S0.json` is private and not admitted. The JOINT-only code (α = 1) was not run because the screen put it outside the Brier cap. | screen ended 17:21:01Z | `AMD/EXPERIMENT_LEDGER.csv` entry 40 → `selection_effort`; `AMD/roles/statistical_reviewer/DEVELOPMENT_REVIEW.md` §0 item 3; `AMD/RESEARCH_DECISION.md` (opening exclusions list); `AMD/RESOURCE_ACCOUNTING.json` → `units[AMD_B_diag3_S0_V1].utc` |
| S4 | **Run the finalist at all** (B3CTL, the coarse-only twin of B3) | Adaptive, on **MODEL_SELECT feedback**: it was run after the coupled B3 failed the seed-0 MODEL_SELECT income Brier clause (entry 40, 17:23:36Z). Seed 1 was run after the seed-0 result. It is **control-designated**. | 17:24:21–25Z (S0); 17:24:46–52Z (S1) | `AMD/EXPERIMENT_LEDGER.csv` entries 41–42 → `selection_effort`, `notes`; `AMD/PROMOTION.json` → `trials[B3CTL_UNCOUPLED_MIX50_C6A].designation` = "control" |
| S5 | TASKONLY control (`B3CTL_TASKONLY_TR_C6A`) constructed and scored | Coordinator-run matched control, fitted after the finalist's MODEL_SELECT results; its own MODEL_SELECT results were known before promotion. The reviewer's files that call for it (DEVELOPMENT_REVIEW F3; the draft's "to be fitted") carry file times after the fit (about 17:34Z and 17:37Z), so the order of recommendation and fit is **UNKNOWN**. | 17:29:39–52Z | `AMD/EXPERIMENT_LEDGER.csv` entries 43–44; `AMD/RESOURCE_ACCOUNTING.json` → `units[AMD_CTL_TASKONLY_TR_*]`; `DEVELOPMENT_REVIEW.md` F3; `CLAIMS_AND_ENDPOINTS_DRAFT.md` §0 (slot TO) |
| S6 | B2TWIN (`B2CTL_UNCOUPLED_C6A`) | The card control of B2, fitted before B3 existed. It was designated the claim-J comparator (LOC) only in the post-promotion claims draft. | fit 17:19:16Z / 17:20:02Z; LOC designation ≈17:37Z | `AMD/EXPERIMENT_LEDGER.csv` entries 35, 37; `AMD/ANALYSIS_LOCK.json` → `roles.LOC` |
| S7 | **Finalist promotion** | **MODEL_SELECT**, under PROTOCOL §6. Utility and guard point clauses use MODEL_SELECT labels; the recipient screens and pair proxies use MODEL_SELECT SEX (one fixed household split `SCM1proxysplit`, fixed reader seeds, the maximum over OUT_LR, OUT_HGB and LOG_MLP, plus LOOKUP on the pair). 17 trials were considered; 3 were promotable, all EDGE. B3CTL has the lowest mean pair proxy: 0.60002, against B2's 0.61314 and B2CTL's 0.61533. Had the shared seed-0 men's guard been below 0.8 × cap, the rule would have selected B2. | `PROMOTION.json` 17:35:23Z; ledger row 48 17:35:40Z | `AMD/PROMOTION.json` → `rule`, `candidate_set`, `proxy_definition`, `n_trials_considered`, `n_promotable`, `n_promotable_robust`, `finalist`, `utc`; `AMD/RESEARCH_DECISION.md` §2 "Selection"; `DEVELOPMENT_REVIEW.md` F6 (winner's curse on a fixed split) |
| S8 | **Superseded interim exclusion and the adjudication** | The coordinator excluded ids containing CTL / CTRL / TASKREFIT, writing that rule at 17:23:49Z, after B2's controls were scored. The interim record (17:23:54Z) named B2 on an incomplete trial set; it was never committed and was deleted at 17:33Z. The statistical reviewer then ruled, with B3CTL's MODEL_SELECT results known (ledger 1–42), that the registered §6 text covers every fitted configuration. That ruling changed the finalist from B2 to B3CTL. | exclusion 17:23:49Z; interim 17:23:54Z; adjudication ≈17:34Z | `AMD/PROMOTION.json` → `superseded_records[0]`, `adjudication`; `DEVELOPMENT_REVIEW.md` §0 and F1; `AMD/RESEARCH_DECISION.md` §2 "Disclosure"; this campaign's `FROZEN_CANDIDATE_MANIFEST.json` → `candidate.source_designation`, `candidate.source_promotion_adjudication` |
| S9 | Reference eligibility labels (NEURAL_RAW, H_LJ, B2TWIN, TASKONLY eligible; PUB_IBSI ineligible) | **MODEL_SELECT** development table: 12 clauses and 2 guards on both seeds. PUB_IBSI's seed-1 men's guard is 0.00501 against 0.005; it also fails the occupation recipient screen against H_LJ. | ANALYSIS_LOCK 17:42:54Z | `AMD/ANALYSIS_LOCK.json` → `availability_ms_point_eligible`; `AMD/EXPERIMENT_LEDGER.csv` entry 22 → `ms_guard_pass` = False, `ms_guard_men` = 0.00501 |
| S10 | Hybrid H_LJ fixed as the frozen-hybrids primary | Design choice, made **after** the replication published its DEV_REPORT results for LOCAL_TREE and JOINT_TREE (all three views, 12 utility clauses per seed, both guards). | HYB `ANALYSIS_LOCK` 2026-10-16T02:46:48Z | `HYB/ANALYSIS_LOCK.json` → `scope_and_selection[0]`, `primary`, `utc` |

---

## 4. What was already known at each selection point

| Selection point | Time | DEV_REPORT results already public | Other information already known |
|---|---|---|---|
| Replication fixed services (T_TREE, LOCAL_TREE, TASK_REFIT, JOINT_TREE) | SCIENCE_LOCK 2026-10-16T00:09:08Z | None on Florida 2019 | The recipe (L_TREE ρ 1, E20) was transported from the Adult line, where it was developed and evaluated (`REP/SCIENCE_LOCK.json` → `source_publications.source_recipe`). |
| Replication NEURAL_RAW selection | 2026-10-16T00:18Z | None: DEV_REPORT was first opened at 00:20:20Z (lineage access log) | MODEL_SELECT results of 12 configurations |
| H_LJ fixed (S10) | 2026-10-16T02:46:48Z | Replication DEV results for LOCAL_TREE and JOINT_TREE | — |
| PUB_IBSI selected (F7) | 2026-10-17T05:27:06Z | Replication, hybrids and SCM-benchmark DEV results. These include the SCM benchmark's IBSI γ = 1 pair, 0.632 / 0.636 (`SCMB/DECISION.md`). | SCM-benchmark MODEL_SELECT results |
| AMD designs, m, α, B3CTL, TASKONLY, promotion (S1–S8) | 17:05–17:35Z | Everything above, plus the SCM-utility-matched results (public `8cb70b290`). This includes, for the finalist's own occupation output (= H_LJ's), its DEV recovery 0.560 / 0.583, its utility and its guards (men 0.0032 / 0.0040). It also includes H_LJ's pair 0.641 / 0.643, NEURAL_RAW's 0.640 / 0.638, TASK_REFIT's 0.664 / 0.666 and the JOINT income output's DEV utility failure. The investigator also cites H_JJ's published DEV income Brier, 0.0054 / 0.0067. | MODEL_SELECT feedback: 24 logged MODEL_SELECT label reads before promotion (2 baseline events covering the 22 frozen references, then 22 development evaluations); a 25th, by the verifier, came at 17:52:58Z. The published teacher headroom on MODEL_SELECT was also known. |
| AMD claim definitions (§6) | ≈17:37–17:43Z | Same; no AMD DEV result yet | The finalist's identity and its MODEL_SELECT results; the TASKONLY and B2TWIN MODEL_SELECT results |
| This campaign's registration (enhanced reader recipes, PROTOCOL §7) | campaign start 21:03:32Z | The AMD DEV_REPORT results for the whole bank: pair 0.613 / 0.618 against 0.640 / 0.638 and 0.641 / 0.643, with labels E, B, M and J (publication `5901f7c73`) | The source reader winners. Enhanced winners are selected on ATTACK_VALIDATION only, but the recipe menu was written after the source DEV results were public. |

Sources for row 5:
- `AMD/PROTOCOL.md` → "Disclosure";
- `DEVELOPMENT_REVIEW.md` F4;
- `AMD/ANALYSIS_LOCK.json` → `endpoints[].flags` (`REPLAY_OF_PUBLISHED`, `IDENTICAL_RELEASE`);
- `AMD/roles/investigator_b/REPORT.md` → "Caveats: Drift risk on DEV_REPORT";
- the admitted snapshot `access/MODEL_SELECT_USE.jsonl` (this campaign's `SOURCE_ADMISSION.json` → `copies`).

**DEV_REPORT exposure across the lineage.** The lineage access log records **10 DEV_REPORT truth reads before this campaign**: 5 coordinator evaluations and 5 verifier replays, by the replication, the hybrids, the SCM benchmark, the SCM utility-matched follow-up and the AMD campaign.
- The first opening was at 2026-10-16T00:20:20Z.
- The AMD coordinator read at 17:45:30Z and the AMD verifier at 17:56:47Z.
- Record: `<worktrees>/pcrl-acs-protected-tree-replication-v1/.private/acs_replication/access/DEV_REPORT_ACCESS.jsonl`, fields `event` and `utc`.

Each later campaign's design was informed by the earlier published results. This campaign adds two logged follow-up re-reads (PROTOCOL.md §3).

---

## 5. Household and person overlap between the admitted roles

**Method.** Unit `EXR_OVERLAP_V2` ran through `ffv.jobqueue` (stage `exposure`, 1 GiB, 0.145 CPU-s). The code is `roles/exposure_reviewer/code/household_overlap.py`; the output is `roles/exposure_reviewer/HOUSEHOLD_OVERLAP_EXR_OVERLAP_V2.json` (sha256 `dafe8d19…`).
- **Input.** `ffv.common.load_role(role, ('features',))` for the six roles, keeping only `row_id` and `unit`. The feature matrix was discarded unused, and no SEX or label file was requested.
- **Ids.** `unit` is the first 60 bits of sha256(`ACS|2019|1-Year|FL|<SERIALNO>`), and `row_id` is the same hash of the household key plus SPORDER (`acs_replication/extract.py`). Equal units therefore mean the same housing-unit serial.
- **Collision bound.** The chance that two distinct serials share a unit is at most 4.4 × 10⁻¹⁰ for the 31,865 households.

**Earlier attempt.** `EXR_OVERLAP_V1` (preserved as `HOUSEHOLD_OVERLAP_EXR_OVERLAP_V1.json`, sha256 `82980aef…`; renamed from the path in its ledger entry, bytes unchanged, and charged) produced identical counts.
- Its only failing check compared `acs_replication/extract.py` with its SCIENCE_LOCK hash.
- The file differs from the locked version by exactly one line, the documented engineering amendment A1 (`REP/LOCK_AMENDMENTS.md`: the occupation-mapping lock-key path).
- V2 verifies this from the locked commit `4d50dea13`. `schema.py`, which holds `role_of`, `unit_interval` and the salts, equals its SCIENCE_LOCK hash exactly.

| Role | Persons | Households | Largest household (persons) |
|---|---|---|---|
| MODEL_FIT | 20,000 | 13,337 | 7 |
| CALIBRATION | 2,982 | 1,974 | 5 |
| MODEL_SELECT | 8,001 | 5,306 | 8 |
| ATTACK_FIT | 8,001 | 5,319 | 6 |
| ATTACK_VALIDATION | 3,000 | 1,974 | 5 |
| DEV_REPORT | 5,943 | 3,955 | 11 |
| **Union** | **47,927 (all distinct)** | **31,865 (all distinct)** | |

**Results.**
- **All 15 role pairs share 0 households and 0 persons.**
- Every role's counts equal `REP/EXTRACTION_SUMMARY.json` → `roles.<role>.selected_persons` and `.selected_households`.
- For ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT and MODEL_SELECT, the row-id arrays equal the frozen-hybrids bank's on both seeds (`HYB/FROZEN_BANK_MANIFEST.json` → `per_seed.S*.row_id_sha256`).
- Count check: the eligible development households summed over the six roles (39,522) plus the sealed households (9,853) equal all eligible households (49,375). The persons add up the same way: 59,398 + 14,693 = 74,091. This is consistent with a household partition; it is not proof of one.

**Sealed assessment: empirical overlap UNKNOWN.** Its ids are not admitted and were not opened; only a key-set hash was ever written (`REP/EXTRACTION_SUMMARY.json` → `sealed_assessment.household_key_set_sha256`).

**What the code implies.**
- `schema.role_of(serialno)` draws one unit-interval value per SERIALNO. It returns `SEALED_ASSESSMENT` iff u < 0.20, and exactly one development role otherwise.
- `extract.py` groups eligible persons by SERIALNO before assigning roles. Each household therefore has exactly one role, and no household can be both sealed and in a development role.
- Capped-out development households (7,657 households, 11,471 persons) keep their development-role draw but were not exported.

**Limits.**
- Disjointness here is by housing-unit serial within one survey file. All roles are random household subsets of the same Florida 2019 sample, so they are not independent populations.
- Person-level uniqueness across different serials (for example movers) is not checkable.

---

## 6. Claim definitions finalized after promotion (source PROTOCOL §9 timing deviation)

**Registered before any fit** (AMD `PROTOCOL.md`, commit `d215c00b3`, 17:01:23Z; public receipt `7e306dcfd`, 17:01:39Z):
- the acceptance requirements U, G, R and P (§4), including the 0.020 target;
- the promotion rule (§6);
- the list of required controls (§7);
- the requirement that claim conjunctions be written "before a finalist is selected" (§9).

**Written after promotion:**
- the explicit conjunctions of claims E, B, M (fixed sequence: TO, then SS = H_LJ alias) and J;
- N (not claimed) and the descriptive IBSI entry;
- the instantiation of TO = TASKONLY, LOC = B2TWIN and SS = H_LJ;
- the 52 + 20 endpoint roster, flags, margins, critical values and mandatory qualifiers.

| Time | Event | Record |
|---|---|---|
| ≈17:34Z | Adjudication (DEVELOPMENT_REVIEW §0) | `AMD/PROTOCOL_POSTRUN_CLARIFICATIONS.md` D1 table |
| 17:35:23Z | Promotion record | `AMD/PROMOTION.json` → `utc` |
| 17:37:02Z / 17:37:13Z | FINALIST_LOCK written / committed (`ad37c120c`) | `AMD/FINALIST_LOCK.json` → `utc`; D1 table |
| ≈17:37Z | CLAIMS_AND_ENDPOINTS_DRAFT written. Its self-reported "about 17:50Z" is impossible, and its worked instance names the finalist. | D1 table; `CLAIMS_AND_ENDPOINTS_DRAFT.md` header |
| 17:42:54Z / 17:43:02Z | ANALYSIS_LOCK written (carries the draft's sha256 `8389e591…`) / committed (`345e0adb0`) | `AMD/ANALYSIS_LOCK.json` → `utc`, `frozen_design_sha256`, `frozen_design_source` |
| 17:43:14Z onward | First final reader unit | D1 table |
| 17:45:01Z | REPORTING_LOCK committed (`721d4163d`) | `AMD/roles/verifier/VERIFICATION.md` §1 |
| 17:45:17–17:45:22Z | Reporting receipt `614c9f36b` pushed (the verifier gives 17:45:17Z; START_HERE gives 17:45:22Z) | VERIFICATION.md §1; `AMD/START_HERE.md` |
| 17:45:30Z | The one coordinator DEV_REPORT read | admitted `access/DEV_REPORT_REREAD.jsonl` |

**D1 is clarified, not resolved.** The conjunctions were fixed after the finalist and its MODEL_SELECT results were known. They were fixed before any final reader fit and before the DEV_REPORT read. The record cannot show whether the draft predates the finalist-lock commit, because both fall in the same minute.

---

## 7. Open items (UNKNOWN in the searched records)

1. The value and derivation of the MODEL_SELECT-derived offset in the α screen (S3), and of the "≈0.002 nats" MODEL_SELECT-versus-CALIBRATION figure in the m reasoning (S2).
2. Whether the TASKONLY fit followed or preceded the reviewer's recommendation (S5).
3. Whether the claims draft predates the finalist-lock commit (§6).
4. Empirical household overlap with SEALED_ASSESSMENT (§5). The code implies there is none.
5. Agent reasoning that was not logged. The records show only the DEV_REPORT values that investigators quoted; any other published value they consulted is unknown.

None of these is evidence of independence.
