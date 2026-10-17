# Population inventory (read-only): candidate ACS state-years for a possible future, separately authorized study

**Role.** Output-mapping and exposure reviewer, PCRL frozen-finalist validation and confirmation-readiness study, 2026-10-17 (about 22:00–22:45Z). PROTOCOL.md §11.

**Context.** The frozen release `B3CTL_UNCOUPLED_MIX50_C6A` predicts an income class (2 classes) and an occupation group (5 classes) from AGEP, WKHP, COW, SCHL, MAR and POBP. Separate evaluation readers try to infer the recorded SEX column from the released predictions.
- Every fitted part of the release comes from public-use ACS 2019 1-Year PUMS records for **Florida**: the 145-column encoder, the T_TREE teachers, the Stage B corrections, the income thresholds and the calibration.
- The thresholds were fitted on MODEL_FIT and the calibration on CALIBRATION (`EXPOSURE_AND_SELECTION_AUDIT.md` §2).

**Rules followed.**
- Nothing was downloaded and no network was used.
- For this inventory no person-level file, task label or SEX value was opened for any population. The only person-level fields this role read anywhere were the `row_id` and `unit` of the Florida 2019 development exports (`EXPOSURE_AND_SELECTION_AUDIT.md` §5).
- No population is ranked, preferred or chosen. Nothing here uses task or recovery outcomes, and the row order is lineage order only.
- **Opening any population, including the Florida 2019 sealed assessment, requires a separate authorization and a frozen protocol. That protocol must re-run a bounded prior-use audit immediately before any acquisition.**

**Status vocabulary.**

| Status | Meaning |
|---|---|
| KNOWN_USED | A record documents use. |
| RESERVED | A record names it for another programme or as a sealed set. |
| NO_USE_FOUND_IN_SEARCHED_RECORDS | Bounded absence in the scope below. It is not certified non-use and not person-level independence. |
| EXPOSURE_UNKNOWN | The records cannot settle it. |

## Searched scope (2026-10-17)

- **S1.** Every `results/*/` entry of this worktree (156). All lineage campaigns since the replication are ancestors of this branch. The text search covered PUMS file names (`psam_[ph]NN`, `csv_[ph]xx`, `ssYY[ph]xx`), state names, "ACS <year>", and the CONFIRMATION, POPULATION, EXPOSURE, RESERVATION, SEALED and LEDGER records.
- **S2.** Code at HEAD (`git grep` for PUMS file names, `survey_year`, `STATE_POSTAL`). Also the two later branches that are not ancestors: `research/pcrl-mechanism-design-correction-v1` and `research/combined-paper-advisor-revision-v1`. Their hits are only records already covered by S1 and S5 (CA 2016–2018, FL 2019, the TX/NY plan names); they add no new state-year.
- **S3.** The replication's provenance and admission records:
  - `acs_replication/provenance.py`;
  - `REP/POPULATION_DECISION.json` and `REP/PROTOCOL_SEAL.json`;
  - the private `PROVENANCE_AUDIT.json` (audited Florida 2019 only) and `admission_provenance.log`;
  - `REP/EXTRACTION_SUMMARY.json`, `REP/SEALED_ASSESSMENT_STATUS.md` and `REP/LOCK_AMENDMENTS.md` (D2).
- **S4.** Usage logs:
  - the lineage DEV_REPORT access log (10 read events plus 1 correction entry, all Florida 2019);
  - the AMD MODEL_SELECT and DEV_REPORT access-log snapshots admitted here;
  - this campaign's queue ledger.
- **S5.** Prior inventories, carried forward and not re-run:
  - the frozen-tree completion `POPULATION_AUDIT.md` and `POPULATION_COMPATIBILITY_AND_USAGE.csv` (2026-10-15). These covered all ref tips of both git stores, durable-guarantees @ `956f5c8` through the API, the relocation inventories, and S3 object names as listed on 2026-09-24;
  - `combined_empirical_preparation_v1/DATA_EXPOSURE_LEDGER.csv` (2026-10-01);
  - the rgj and smf `CONFIRMATION_PLAN.md`;
  - the shared-context `CONFIRMATION_PLAN.md` @ `537e74a44`.
- **S6.** Local file names under the home directory, to depth 9 (Library, .Trash, node_modules, .venv, site-packages and .git excluded), matching PUMS file patterns. Names and sizes only; no file was opened.
- **S7.** The user's project memory notes, read-only.

**Not searched (UNKNOWN):**
- S3/GCS listings, since no network was allowed;
- other machines and deleted EC2 disks;
- the YOTUO drive, which was not mounted (`/Volumes` lists no backup drive);
- GitHub branches not present locally;
- the method-portfolio git store's refs after 2026-10-15 (not re-scanned);
- the contents of archive tars.

Path abbreviations: `REP/` = `results/pcrl_acs_protected_tree_replication_v1/`; `FTC/` = `results/pcrl_frozen_tree_control_completion_v1/`.

## Inventory

| Candidate | Status | Documented sample availability | Historical extraction or opening disclosure (carried forward) | Use for this frozen release | Evidence |
|---|---|---|---|---|---|
| **FL 2019 1-Year: the six development roles** | **KNOWN_USED** (this lineage, repeatedly) | 47,927 persons in 31,865 households; 0 shared households between roles (`HOUSEHOLD_OVERLAP_EXR_OVERLAP_V2.json`) | DEV_REPORT truth read 10 times before this campaign. MODEL_SELECT was used for selection or development evaluation by the replication, the SCM benchmark, the SCM utility-matched follow-up and the AMD campaign. The ATTACK roles were used for readers by every campaign that fitted readers. | Not fresh. Any further use is development robustness evidence only, as in this campaign. | `REP/EXTRACTION_SUMMARY.json` → `roles`; lineage access log; `EXPOSURE_AND_SELECTION_AUDIT.md` §4 |
| **FL 2019 1-Year: SEALED_ASSESSMENT** (20% of eligible households) | **RESERVED** (the lineage's sealed set; closed; every loader refuses it) | **14,693 persons in 9,853 households**; key-set sha256 `b3dca8eb…` | **D2.** The extraction's single eligibility pass parsed every sealed record. For each sealed person it built an in-memory record (six inputs, SEX code, PWGTP, RELSHIPP and **both task labels**), then discarded it unwritten. Only counts, the key-set hash and one data-quality count were kept. The replication verifier read the raw SERIALNO column, including sealed serials. **Not untouched.** | **Would test the frozen release**: same state-year population, people no model or reader has used. Disjoint from the development roles by construction (empirical check UNKNOWN). Reuse also requires re-running the prior-use audit and recording D2 (`REP/SEALED_ASSESSMENT_STATUS.md` → "Reuse"). | `REP/EXTRACTION_SUMMARY.json` → `sealed_assessment`; `REP/LOCK_AMENDMENTS.md` D2; `REP/SEALED_ASSESSMENT_STATUS.md`; memory note on the replication |
| **FL 2019 1-Year: capped-out development households** (never exported) | **NO_USE_FOUND_IN_SEARCHED_RECORDS** for any analysis (no export file exists) | **11,471 persons in 7,657 households**. By role: MODEL_FIT 6,694 / 4,483; MODEL_SELECT 857 / 561; ATTACK_FIT 916 / 601; ATTACK_VALIDATION 3,004 / 2,012. | **D2 applies as for the sealed set.** The receipt's field name `capped_out_*_unopened` is contradicted by D2: these records were parsed in memory, with both task labels and SEX, and discarded. **Not untouched.** | Would test the frozen release (same population, new people). They are not a registered holdout: they were carved out by a cap order (salted hash), so any use must be designated in a frozen protocol before opening. | `REP/EXTRACTION_SUMMARY.json` → `roles.*.capped_out_*`; `REP/POPULATION_AND_SCHEMA.md` → "Household roles"; `REP/LOCK_AMENDMENTS.md` D2 |
| FL 2019 1-Year: records outside the service cohort | NO_USE_FOUND_IN_SEARCHED_RECORDS for analysis (but every record was read by the extraction's eligibility pass) | 131,203 person records (= 205,294 − 74,091): age outside 25–64, 98,305; group quarters, 8,202; not civilian employed, 24,696 | The single eligibility pass read the `NEEDED` columns of every record (RT, SERIALNO, SPORDER, ST, ADJINC, PWGTP, AGEP, COW, SCHL, MAR, POBP, WKHP, ESR, OCCP, SEX, PINCP, RELSHIPP), counting ineligible records by first failing reason | **Different benchmark**: outside the cohort the release was built for (housing-unit residents aged 25–64, civilian employed, WKHP 1–99) | `REP/EXTRACTION_SUMMARY.json` → `exclusion_counts_first_failing_reason`; `acs_replication/extract.py` → `NEEDED` |
| PA, IL, OH, GA and NC 2019 1-Year (the rest of the replication's frozen admission order) | NO_USE_FOUND_IN_SEARCHED_RECORDS | UNKNOWN (no receipt gives counts) | Pennsylvania: the first automated provenance run wrongly auto-advanced to it; it was stopped and "no Pennsylvania result was written or used". IL, OH, GA and NC were never audited. No download is recorded. | **Transport** (geographic, same year and dictionary). The frozen package can run on the six raw fields (Florida-specific encoder levels; unseen categories go to OTHER), but that tests transfer under shift, not the Florida development result. An end-to-end refit is a recipe transport. | `REP/POPULATION_DECISION.json` → `admission_order`, `disclosures[0]`; `REP/PROTOCOL_SEAL.json`; S1, S2, S6 |
| Other states, 2019 1-Year | NO_USE_FOUND_IN_SEARCHED_RECORDS | UNKNOWN | none | Transport (geographic) | `FTC/POPULATION_COMPATIBILITY_AND_USAGE.csv` (row "any state 2019 and 2021–2023"); S1, S2, S6 |
| FL and other states, 2021–2023 1-Year | NO_USE_FOUND_IN_SEARCHED_RECORDS | UNKNOWN | none | Transport (temporal, or temporal plus geographic). Each year needs its own ADJINC and a dictionary check. 2023 renames ST to STATE. | as above; shared-context `CONFIRMATION_PLAN.md` @ `537e74a44` §2 |
| FL and other states except CA, TX and NY, 2018 1-Year | NO_USE_FOUND_IN_SEARCHED_RECORDS | UNKNOWN | none | Transport. The 2018 dictionary has RELP rather than RELSHIPP, a 2018 SERIALNO prefix and a different ADJINC. | `FTC/POPULATION_COMPATIBILITY_AND_USAGE.csv` (row "other 2018 1-Year states"); S1, S2, S6 |
| **TX 2018 1-Year** | **RESERVED**: the primary confirmation cohort of the shared-context release programme. No CONFIRMATION_LOCK exists. Whether the reservation is still live after that programme closed is an open owner decision (FTC U6). | `csv_ptx.zip` 47,257,733 B (HTTP HEAD, 2026-09-24); persons and households UNKNOWN; "≈0.7 × CA cohort" is "an unverified expectation, not a count" | Never downloaded or opened; HTTP HEAD only | Transport (geographic and temporal). Using it would also spend another programme's reservation. | `FTC/POPULATION_AUDIT.md` §3.2; shared-context `CONFIRMATION_PLAN.md` §3; `REP/SCIENCE_LOCK.json` → `population.excluded_by_rule` |
| NY 2018 1-Year | NO_USE_FOUND_IN_SEARCHED_RECORDS; entangled as the registered conditional fallback (TX + NY pooled) of the TX 2018 reservation | `csv_pny.zip` 34,958,982 B (HTTP HEAD); counts UNKNOWN | Never downloaded | Transport, and only with the owner's decision on the TX reservation | `FTC/POPULATION_AUDIT.md` §3.2; `REP/PROTOCOL_SEAL.json` → `never_touched` |
| CA 2018 1-Year | **KNOWN_USED** (spent; the AGEP 19–34 cohort is exhausted) | 378,817 raw person rows; AGEP 19–34: 80,329 persons in 53,907 households (a different cohort definition); this benchmark's cohort count UNKNOWN | PCRL v2 Folktables lineage; the September 2026 release studies; durable-guarantees cells (its 20% holdouts were unread by durable-guarantees but overlap persons PCRL used) | Not fresh | `FTC/POPULATION_COMPATIBILITY_AND_USAGE.csv`; `results/combined_empirical_preparation_v1/DATA_EXPOSURE_LEDGER.csv` data rows 8–11, 26–30, 37, 40 (FTC's numbering) |
| CA 2017 1-Year | **KNOWN_USED** (spent 2026-09-17) | AGEP 19–34: 79,869 persons in 53,504 households; final partition 15,924 opened | Final partition opened by the locked spectral-transport study | Not fresh | as above, data row 31; memory note on the 2017 transport |
| CA 2016 1-Year | **KNOWN_USED** (spent 2026-09-23) | 376,035 raw rows; AGEP 19–34: 79,298 persons in 53,093 households; final 23,684 persons in 15,928 households scored once | The admission read the whole file, including label columns | Not fresh | as above, data rows 32–33; memory note on the evidence audit |
| CA 2016 and 2017 persons outside AGEP 19–34 | EXPOSURE_UNKNOWN | UNKNOWN | Files opened; these people share households with used persons; programme ledgers call the years spent at population level | Not recommended; needs an owner ruling (FTC U7) | `FTC/POPULATION_AUDIT.md` §3.1 |
| **CA 2014 1-Year** (new finding; not in the 2026-10-15 inventory) | **KNOWN_USED** | A local person file `ss14pca.csv` (354,200,643 B), written 2026-10-03T05:08Z inside the output-aware-removal study's vendored FARE checkout; counts UNKNOWN (not opened) | On 2026-10-03 the official FARE entry point ran on "public ACSIncome-CA-2014" as an installation reproduction gate: a FARE tree was fitted and evaluated on that task, labels included. No record shows a PCRL release or reader using it. This corrects FTC's "2010–2015: no use found". | Not fresh for any purpose. A different benchmark in any case: the occupation coding and dictionary differences from 2018/2019 are unchecked (UNKNOWN). | `results/combined_output_aware_removal_v1/METHOD_ADMISSION.md` ("Reproduction gate"); `.../notes/fare/FARE_METHOD_NOTES.md` §6; `.../scripts/fare_admission_checks.py` (docstring, `GATE_REL`); S6 |
| Any other state-year 2010–2017 1-Year | NO_USE_FOUND_IN_SEARCHED_RECORDS | UNKNOWN | none | Different benchmark or a registered crosswalk. The frozen occupation mapping was built from the official 2018 ACS PUMS occupation code list and checked only against the 2019 dictionary; earlier dictionaries are unchecked. | `REP/POPULATION_AND_SCHEMA.md` → "Tasks"; `FTC/POPULATION_AUDIT.md` §3.3 |
| Any state, 2020 1-Year | NO_USE_FOUND_IN_SEARCHED_RECORDS | n/a | none | Different benchmark: only an experimental PUMS with experimental weights exists, and Census advises against comparing it with standard releases | shared-context `CONFIRMATION_PLAN.md` §2; `FTC/POPULATION_COMPATIBILITY_AND_USAGE.csv` |
| ACS 5-Year files that include FL 2019 or CA 2016–2018 1-Year records | EXPOSURE_UNKNOWN at record level; treated as overlapping used people | UNKNOWN | none recorded | Not new people. The replication seal excludes 5-Year files that overlap 1-Year people. | `REP/PROTOCOL_SEAL.json` → `never_touched`; `FTC/POPULATION_AUDIT.md` §3.3 |

## Notes for any future authorization

1. **Only Florida 2019 people can test the frozen release itself.** Only they share the population on which its encoder, teachers, corrections, thresholds and calibration were fitted.
   - The candidates are the sealed 20% (9,853 households) and the capped-out development households (7,657).
   - Both carry the D2 in-memory extraction disclosure. They are new people, not untouched people, and not a new population.
   - Whether either is large enough is for the confirmation-feasibility calculation (PROTOCOL.md §10), not this inventory.
2. **Custody.** The Florida 2019 raw file exists in six private local copies: the replication's raw folder and its snapshot and restore-test folders (S6, names only). The private snapshot was also copied to the off-device backup drive (`REP/LOCK_NOTES.md`, row "Custody V2 (final)").
   - The sealed closure is therefore operational (loaders refuse the role), not cryptographic (`REP/SEALED_ASSESSMENT_STATUS.md`).
   - Any opening is an action to authorize and log, not a technical impossibility.
3. **Every other state-year is a transport.** Running the frozen package unchanged tests transfer under distribution shift; refitting tests the recipe. Neither confirms the Florida 2019 development result.
   - Each transport needs a registered schema check (ADJINC, RELP or RELSHIPP, the SERIALNO format, the occupation code list, and ST or STATE in 2023).
   - Each needs a label-blind admission count, because no candidate's cohort size is documented.
   - Each needs a fresh bounded prior-use audit. That audit must also cover the scopes left UNKNOWN above (S3 listings, the YOTUO drive, other machines).
4. **TX 2018 and its NY fallback are entangled with another programme's reservation.** Releasing or reusing them is the owner's decision.
5. **No population was chosen here,** and no candidate was compared on task or recovery outcomes.
