# Source, population, schema and household roles

## Source packages (pinned)

| Package | Pin | Verification |
|---|---|---|
| Task-headroom publication | `c8ae518190100decf39e605f191747d0f10723e2` | transported source files byte-identical to the publication blobs (`SOURCE_ADMISSION.json`) |
| Frozen-tree completion publication | `8975a23d3d57cbab8c3e7fb27c723d8c4dc827ad` | same check |
| FAGTB (Grari et al., ICDM 2019) | `vincent-grari/FAGTB` @ `0cd0bb1` | audit in `baselines/FAGTB_AUDIT.md` |
| FARE (Jovanović et al., ICML 2023) | `eth-sri/fare` @ `89cb1b6` | used through `oar.fare_official` (pins verified at run time) |

- The transported modules `acs_replication/{trees,leafcorr,neural,neural_raw}.py` change their sources only in declared ways: input width, the five occupation classes, critic widths 3/6/9, role names, the privacy-pool access (MODEL_FIT with SEX replaces the source's UPDATE/GUARD split), schema labels and the appended Ucal procedure fixed in the SCIENCE_LOCK.
  - Most changed lines carry an `ACS-CHANGE` marker, but not all. Two width edits carry their marker on the adjacent line, the Ucal block is headed `ACS addition` instead, and some docstring lines are unmarked.
  - The independent verifier found this (check 1), and it is disclosed as D1 in `LOCK_AMENDMENTS.md`. Behaviour is unaffected, and the files are byte-identical to their SCIENCE_LOCK versions.
- Neither FAGTB nor FARE has a licence file, so both are run and cited only.

## Population choice (before any outcome)

**Admission order** (frozen in the protocol seal): Florida, Pennsylvania, Illinois, Ohio, Georgia, North Carolina, all 2019 1-Year.

**Never touched:**
- California 2016–2018 (spent);
- Texas 2018 (reserved by another programme);
- New York 2018 (Texas's conditional fallback);
- ACS 5-Year files.

**Florida 2019 was admitted** as NO_USE_FOUND_IN_SEARCHED_RECORDS (`POPULATION_DECISION.json`). The searched scope covered:
- every ref tip and the full pickaxe history of both PCRL git stores;
- the durable-guarantees repository on GitHub (195 files);
- the storage-relocation inventories;
- local file names;
- other local research repositories.

Every automated hit was reviewed by hand. They were self-references (this campaign's own seal commit and raw copy) and natural-language text mentions in unrelated corpora. No hit was ACS use or a reservation.

- **Not searched (UNKNOWN):** S3/GCS listings, deleted EC2 disks, other machines and archive tar members.
- **Wording:** this is bounded, documented absence. It is not certified non-use, and not person-level independence.
- **Disclosures** (all three are in `POPULATION_DECISION.json`):
  1. A script defect first auto-advanced to Pennsylvania. It was stopped, and nothing was written or used.
  2. Case-insensitive and ERE searches were abandoned for speed and replaced by case-sensitive PCRE searches with explicit state-name variants. Earlier partial runs wrote nothing.
  3. The metadata admission (2026-10-15T23:55Z, SERIALNO only) ran after the manual review of the hits. It ran before the corrected automated receipt finished (00:04Z) and before the decision was recorded (2026-10-16T00:06:00Z).

**Official source** (hashes in `POPULATION_DECISION.json`):
- **File:** `https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/csv_pfl.zip`, Last-Modified 2020-10-15, 36,712,920 B, member `psam_p12.csv`.
- **Contents:** 205,294 person records; 86,036 housing-unit serials; 8,202 group-quarters serials.
- **Documentation:** the 2019 PUMS data dictionary (CSV and TXT), the 2019 code lists, the 2019 README, accuracy and design factors, and the official 2018 ACS PUMS occupation code list with crosswalk.

**Admission stages.**
- **Metadata admission:** hashes, header, raw counts and serials only.
- **Automated eligibility extraction:** reads the eligibility and target-validity variables. It ran only after the public science lock. Its single pass parsed every eligible record in memory, sealed and capped-out people included, before roles were assigned. It wrote only the development exports, counts and the sealed key-set hash (disclosure D2).

## Benchmark (custom shared-worker, two-task; not unmodified ACSIncome)

**Cohort** (person rows, Florida): all of the following must hold.
- **Residence:** housing-unit residents only; every 2019GQ serial is excluded.
- **Age:** 25–64.
- **Employment:** ESR ∈ {1, 2}, i.e. civilian employed.
- **Hours:** WKHP 1–99.
- **Occupation:** OCCP present and mapped to a civilian group.
- **Income:** PINCP present and ADJINC = 1010145.
- **SEX:** valid.
- **Inputs:** COW, SCHL, MAR and POBP all present.

Exclusions by first failing reason: age 98,305; group quarters 8,202; not civilian employed 24,696. That leaves **74,091 eligible** people in 49,375 households.

**Tasks.**
- **Income:** `PINCP × ADJINC > 50,000 × 10^6` in exact integer arithmetic, i.e. adjusted personal income above $50,000 in 2019 dollars. This is not Adult's 1994 threshold.
- **Occupation:** five official broad civilian groups (`OCCUPATION_MAPPING.json`): management/business/science/arts; service; sales/office; natural resources/construction/maintenance; production/transportation/material moving.
  - The mapping is taken code by code from the official 2018 ACS PUMS Occupation Code List: 525 codes, 0 disagreements with the 2019 dictionary label prefixes.
  - Military 9800–9830, 9920 and blank are excluded.

**Inputs.**
- One common whitelist: AGEP and WKHP (z-scored with MODEL_FIT moments), plus one-hot COW, SCHL, MAR and POBP.
- Categorical levels are kept if they occur at least 10 times among MODEL_FIT people. Everything else goes to a per-variable OTHER column.
- Width: 145.

**Excluded from inference inputs:** SEX, every race variable, OCCP, SOCP, industry codes, all income amounts, ESR, RELP/RELSHIPP, identifiers, weights and ADJINC.

**Weights and protected attributes.**
- PWGTP is used only for weighted point summaries.
- SEX is the protected attribute.
- Race is not an input, and no race claim is made.

## Household roles

**Assignment.**
- **Key:** `ACS|2019|1-Year|FL|<SERIALNO>`, hashed with the published salt `PCRL-ACS-protected-tree-replication-v1|household-roles|FL-2019|2026-10-15|`. Every eligible member of a household is in the same role.
- **Split:** 20% of households go to SEALED_ASSESSMENT. The remaining DEVELOPMENT households split 45 / 5 / 15 / 15 / 10 / 10 % into MODEL_FIT / CALIBRATION / MODEL_SELECT / ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT.
- **Caps:** deterministic whole-household caps (salt `…|cap-order|…`), with overshoot recorded. Capped-out households are not exported. Like the sealed households, their records were parsed in memory by the extraction's single eligibility pass (disclosure D2).

| Role | People (households) | Overshoot (people) | Capped out, not exported: people (households) |
|---|---|---|---|
| MODEL_FIT | 20,000 (13,337) | 0 | 6,694 (4,483) |
| CALIBRATION | 2,982 (1,974) | 0 | 0 (0) |
| MODEL_SELECT | 8,001 (5,306) | 1 | 857 (561) |
| ATTACK_FIT | 8,001 (5,319) | 1 | 916 (601) |
| ATTACK_VALIDATION | 3,000 (1,974) | 0 | 3,004 (2,012) |
| DEV_REPORT | 5,943 (3,955) | 0 | 0 (0) |
| SEALED_ASSESSMENT | 14,693 (9,853) | — | **closed**: nothing written or used beyond counts and the key-set hash; records parsed in memory only (D2) |

**SEX coding in the tables:** `sex0` = male (SEX 1), `sex1` = female (SEX 2) (`schema.SEX_CODE`).

**Support** (pre-registered thresholds):
- **MODEL_FIT task classes:** income 11,580 / 8,420; occupation groups 8,691 / 3,152 / 4,469 / 1,750 / 1,938. All are ≥ 100.
- **Attacker roles, SEX classes:** ATTACK_FIT 4,021 / 3,980; ATTACK_VALIDATION 1,534 / 1,466. All are ≥ 100.
- **DEV_REPORT households per SEX group** (checked after the reporting lock): 2,815 and 2,728. Both are ≥ 200.
