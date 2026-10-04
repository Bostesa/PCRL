# Confirmation plan (prospective draft)

**Outcome note (lead, 2026-10-04, after the development assessment): the development label is EXPERIMENTAL_NO_ADVANTAGE** (`RESEARCH_DECISION.md`). Under section 1 of this plan, that means **no confirmation**: every candidate population below stays unspent. The audit of their histories is kept for any future, differently specified method.

**Status: DRAFT. Nothing has been acquired, downloaded or opened.**
- No confirmation label was read for this plan.
- No new-population file was requested, not even an HTTP HEAD.
- This study's development result does not exist yet, and nothing here depends on it.

The plan is contingent on that result. It names candidate populations only together with their audited history. It
names none of them "untouched" without evidence.

Owner: data and provenance role. Usage audit run read-only on 2026-10-04 across Bostesa/PCRL (all 53 local and
remote ref tips, the full history for the TX/NY file names, local disks and the drive inventories) and
Bostesa/durable-guarantees @ `956f5c883f515646aa457db55ecbd74b913768b2` (full history).

## 1. When a confirmation is justified

| Development label (prompt section 16) | Confirmation action |
|---|---|
| DEVELOPMENT_ADVANTAGE_ESTABLISHED | Draft a confirmation protocol from this plan and ask the user to choose the population. Nothing is acquired automatically (prompt section 5). |
| EXPERIMENTAL_NO_ADVANTAGE | No confirmation. Spending a fresh population on a recipe that failed in development would waste it. Keep every candidate below unspent. |
| INCOMPLETE_OR_INVALID | No confirmation. Repair the development comparison first. |

An outcome-driven change after the development assessment opens (grid, thresholds, arms) makes the recipe a new
development candidate. It does not create a confirmation candidate.

## 2. What a confirmation could confirm

No fresh rows exist in Adult's 83-column schema (section 3, rows A1–A4). The deployed Adult checkpoint therefore
cannot be confirmed on new people. Only the **frozen procedure** can be tested on a new population:
- the training recipe, schedules and grid;
- the selection rule and feasibility gates;
- the endpoints, allowances and decision rule.

The procedure would be refit end to end under a newly registered input contract. That is a replication of a method
under transport, not confirmation of a released model. It must be described that way.

Schema facts that any ACS mapping must face, registered before acquisition:
- **Capital gain and loss.** Adult's capital-gain and capital-loss have no direct ACS PUMS equivalent. Dropping them or substituting another variable (for example interest or dividend income) changes the input contract.
- **Relationship proxy.** Adult's relationship = Husband/Wife nearly determines SEX for about 46% of rows (`EXPOSURE_LEDGER.md` section 4). In ACS 2018, `RELP` uses codes 00–17, and the spouse code is expected to be sex-neutral (check against the official dictionary at admission). The main SEX proxy of the development data would then be absent or weaker, so protection difficulty would differ by construction.
- **Occupation.** occupation_group (6 classes, a many-to-one map of Adult occupation) needs a registered crosswalk from `OCCP`.
- **Income threshold.** The income label is a nominal >50K threshold in both sources, but in 1994 dollars versus 2018 dollars. The threshold rule must be fixed in advance.
- **2019 and later.** `RELP` is replaced by `RELSHIPP` (20–38). There is no exact map back (shared-context `CONFIRMATION_PLAN.md` section 2), so a temporal design needs its own mapping.
- **2020.** Only an experimental 1-year PUMS was released; it is ineligible.

## 3. Audited history of candidate populations

| # | Candidate | Status | Evidence |
|---|---|---|---|
| A1 | Adult rows in this study (all five roles) | **Exposed** (all rows) | `EXPOSURE_LEDGER.md` |
| A2 | Adult old `assessment` (5,243) | **Spent.** Outer pool scored by pilot, bench, oar, odx, cap, jcv and pnx; part of `adult.test`, the PCRL v2 test set. | jcv `DATA_ADMISSION.json` role semantics; pnx `PROTOCOL.md` |
| A3 | Adult old `cert` (1,500) | **Spent.** FARE certificates (oar; jcv amendment A3). Part of `adult.test`. | `results/pcrl_joint_complete_view_method_v1/FARE_CERTIFICATES_A3.json` |
| A4 | Adult PCRL validation split (about 20% of `adult.data`; not in `adult_jcv.npz`) | **Spent as a selection set.** PCRL v2 checkpoint choice (best.pt / Cotter fallback). | `results/combined_empirical_preparation_v1/DATA_EXPOSURE_LEDGER.csv` (Adult val row, source `run_v2_dataset.py:319,337-339`) |
| A5 | Adult rows removed by the loader's missing-value filter | Never used as scored rows, but values are missing (workclass, occupation, native-country). Unusable under the 83-column contract and the occupation label without an imputation choice. Not a confirmation population. | `jcv/data.py` (post-dropna rows only) |
| C18 | ACS CA 2018 (all cohorts) | **Spent.** dg used CA 2018 1-Year for all three Folktables cells, including its 80% splits; the 20% holdouts are unread by dg but overlap people fully used by PCRL ACS studies. PCRL v2 ACS encoders. The AGEP 19–34 cohort is EXHAUSTED: the locked evaluator found no fresh rows. | dg `utils/folktables_io.py` (STATE="CA", YEAR="2018"; the only state/year in dg history); prep ledger rows "ACS CA 2018"; `ablations-facct@ad2c08872:results/redesign_20260910_acs_locked_evaluation_v1/DATA_INDEPENDENCE.md` |
| C17 | ACS CA 2017 | **Spent.** Final partition opened 2026-09-17 (spectral transport); reanalysed by the evidence paper. | prep ledger row "CA 2017" (`origin/ablations-facct-2026-07-24@349efa454:results/redesign_20260917_acs_spectral_transport_v1/`) |
| C16 | ACS CA 2016 | **Spent (scored).** Fitting and validation pools used by final-prospective. Final partition (23,684 people / 15,928 households) scored once. Project memory's "2016 admitted but unscored" is **out of date**. | Verified here: `git show 5e154e5c4:results/pcrl_final_prospective_v1/EVALUATION_LOCK.json` (created 2026-09-23T00:49:24Z, dataset acs2016, 156 units) and `INFERENCE_2016.json` (2026-09-23T00:54:06Z); also `SCORE_INTEGRITY_2016.json`, `VERIFICATION_2016.json` |
| T18 | ACS TX 2018 1-Year (`csv_ptx.zip` → `psam_p48.csv`) | **Not acquired, but already named.** It is the primary confirmation cohort of the shared-context release programme (AGEP 19–34; same-residence and service tasks). Privacy-first never triggered its Texas power table ("No Texas file was downloaded or opened"). Only HTTP HEAD requests were made (2026-09-24). Choosing it here would spend the same population for that programme. | PCRL full-history pickaxe for `csv_ptx` finds only `66e8186f8` (2026-09-24, shared-context plan) and `b67f04b5e` (2026-10-01, preparation ledger). `537e74a44:results/pcrl_shared_context_release_v1/CONFIRMATION_PLAN.md` lines 52–53 and 66; `8fdc61e39:.../RESEARCH_DECISION.md` "Texas 2018 power table". No TX file on local disks or in the drive inventories. No TX reference in dg history. |
| N18 | ACS NY 2018 1-Year (`csv_pny.zip`) | **No use found. Named only as a registered fallback** (TX+NY pooled) in the shared-context plan; HTTP HEAD only. NY alone is smaller than TX (same plan). | Pickaxe: `psam_p36` absent from PCRL history; `csv_pny` only in `66e8186f8`. No NY file locally or in drive inventories. No NY data reference in dg. Prep ledger: "NOT USED in either repository". |
| O | Other ACS states (any year) | **No use found in the audited scopes.** All PCRL ref tips reference ACS person files only for CA (`psam_p06`, `ss16pca`, `csv_pca*`) and the TX names above. dg history references only CA 2018. The shared-context audit (2026-09-24, `git log --all -G`) covered the 12 largest states and 2019–2023 with no data-use hit. | This audit (ref-tip `git grep`; drive inventories `*.json.gz` contain only CA file names); shared-context plan lines 13–17 |
| Y | ACS CA 2019, 2021–2023 | **No use found**, but the schema changed (`RELSHIPP`). This would be a temporal claim, with CA people overlapping earlier waves only as a population, not as records. | shared-context plan section 2; prep ledger |
| H | HMDA other states or years | No Adult-schema mapping exists, so it is irrelevant to this Adult income/occupation method. HMDA CA 2023 is spent (PCRL train/val/test and dg). | prep ledger HMDA rows |

"No use found" means no reference in the searched repositories, history, local disks and drive inventories on
2026-10-04. It is not proof of non-use elsewhere, for example in cloud logs or other machines. Re-run the audit
immediately before any decision.

## 4. Draft recommendation (for the user to decide; not acquired)

1. **Do not use T18 (Texas 2018) by default.** It is already the named confirmation population of another programme, and spending it here would also spend it there. The user should decide explicitly whether one population can serve both, or whether this study needs a different one.
2. **If an Adult-like ACS replication is wanted,** the least-entangled documented option is a 2018 1-Year state that no study has named: no use found, same 2018 national dictionary as CA, no schema change. N18 (New York) has the cleanest audit record, but it appears in another programme's fallback; any other state needs a fresh name-level audit first. Pick the state by predeclared design criteria (sample size after the registered filter, household grouping, support for the six occupation groups and the SEX classes), never by outcomes.
3. **What is registered and pushed before any download:**
   - population and filter;
   - the Adult→ACS input contract (including the capital-gain and relationship decisions above) and the label definitions (income threshold, occupation crosswalk);
   - household-grouped role allocation with a fixed salt;
   - the frozen recipe, grid, selection rule, endpoints, allowances, multiplicity and power or precision target;
   - a one-shot evaluation rule;
   - which development result triggered the confirmation.
4. **Admission is label-blind.** Schema and dictionary checks and label-free counts only. Labels of the final partition are read once, after the evaluation lock is pushed.
5. **Claims a confirmation could support:** the frozen procedure's advantage, if any, under transport to that population and contract. It would not support population privacy, DP or MI, protection of other attributes, or properties of the Adult checkpoint.

## 5. Re-audit checklist before any acquisition

- `git fetch --all`; `git log --all -S'<file name>'` and `git grep` across all ref tips of both repositories for the chosen state's file names (`psam_pNN`, `csv_pXX`, `ss1Yp..`) and its postal code, Texas/New York/state name strings, and `survey_year` values.
- Search local disks and `<drive>/*/inventories/*.json.gz` for those file names.
- Check open `CONFIRMATION_PLAN.md` files and handoffs under `pcrl_week_finish_v2/evaluation/` for claims on the population.
- Record the result and the date in the confirmation protocol before the lock.
