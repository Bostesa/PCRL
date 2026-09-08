# ACS 2018 schema notes for the transfer pilot

Verified September 7, 2026 against the [official 2018 Census documentation](https://www.census.gov/programs-surveys/acs/microdata/documentation/2018.html), its [2018 one-year PUMS dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt) (November 14, 2019), and [official Folktables task code](https://github.com/socialfoundations/folktables/blob/main/folktables/acs.py). The dictionary is specifically 2018, not the 2018–2022 five-year release. Installed Folktables is 0.0.12; its `acs.py` SHA256 is `5946ac913930c464e35cbf6288b96746f232cded2966925be79b04e2d082907d`.

## Labels and eligibility

Blank CSV entries represent inapplicable/missing values. Zero is not a universal missing code. Invalid values stay masked and must not become negative-class observations. Use the following fixed schemas, retaining absent classes in support reports.

| Column | Valid values and interpretation | Pilot label |
|---|---|---|
| `PINCP` | Signed integer income, −19998 through 4209995; zero and losses are valid. Blank: under age 15. | Raw `PINCP > 50000`, plus source-fitting quantile bins. |
| `ESR` | 1: civilian at work; 2: civilian with job, absent; 3: unemployed; 4: military at work; 5: military with job, absent; 6: outside labor force. Blank: under 16. | Six classes; `ESR == 1` when a binary at-work label is needed. |
| `PUBCOV` | 1: public coverage; 2: no public coverage. | Two classes, with code 1 meaning coverage. |
| `MIG` | 1: same house; 2: previously outside US/Puerto Rico; 3: different US/Puerto Rico house. Blank: under age 1. | `MIG == 1`, named **same residence**, not “moved.” |
| `JWMNP` | Integer 1–200 minutes; blank for nonworkers or home workers. | `JWMNP > 20` only where valid. |
| `SEX` | 1: male; 2: female. | Fixed two-class audit, using the dataset's recorded variable. |
| `RAC1P` | Codes 1–9; detailed meanings below. | Fixed nine-class audit; no binary collapse. |

The race classes are White alone; Black/African American alone; American Indian alone; Alaska Native alone; the dictionary's combined/unspecified American Indian–Alaska Native category; Asian alone; Native Hawaiian/other Pacific Islander alone; other race alone; and multiple races. These are recorded Census categories, not inferred identities. [Census dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt).

The common cohort is `19 <= AGEP <= 34` and `PWGTP > 0`. The commuting task includes **all** Census-valid `JWMNP` observations in this cohort, including military workers. Its eligibility mask is used only to select fitting/evaluation rows and is never a representation input. All release arms use the same eligible rows. This is an explicitly adapted benchmark population.

Raw income thresholds match Folktables' direct `PINCP` comparison. Do not silently multiply by `ADJINC`: the dictionary gives the 2018 factor as 1.013097, so inflation-adjusted income would define a different threshold. `ESR == 1` is specifically civilian employment **at work**, not all civilian employment.

## Folktables differences that matter

The executable definitions, rather than README paraphrases, establish the filters:

- Income: age above 16, income above 100, positive usual weekly hours, and weight at least 1.
- Public coverage: age below 65 and income at most 30000.
- Mobility: age strictly between 18 and 35, matching integer ages 19–34.
- Travel time: age above 16, weight at least 1, and `ESR == 1`. Its filter does not separately exclude missing travel time before applying `> 20`.
- `ACSEmployment` applies no filter; `ACSEmploymentFiltered` separately applies ages strictly between 16 and 90 and positive weight.

The pilot deliberately uses its common cohort and explicit validity masks instead of calling these task wrappers. Their feature lists also mix targets: mobility includes `ESR`, `JWMNP`, and `PINCP`; employment/public coverage include `MIG`; public coverage includes `PINCP` and `ESR`. They cannot be combined unchanged for task-identity holdout. [Folktables source](https://github.com/socialfoundations/folktables/blob/main/folktables/acs.py).

## Input allowlist and duplicate-answer exclusions

The agreed ten existing covariates are numerical `AGEP`, `WKHP` and categorical `SCHL`, `MAR`, `RELP`, `CIT`, `DIS`, `DEAR`, `DEYE`, `DREM`. Fit imputation, scaling, and categorical discovery only on representation-fitting rows; reserve distinct encodings for missing/invalid and unseen-valid categorical values.

This allowlist excludes full target copies and obvious direct-answer families. It does **not** assert independence from tasks or audit attributes. `WKHP` describes usual hours over the preceding year, whereas `ESR` concerns current work status; its value and missingness can still convey partial information. Disability variables and other retained covariates can correlate with protected labels.

Use an allowlist, not an incomplete denylist. In particular omit:

- All five source/held-out columns and their bins, thresholds, aliases, validity masks, or allocation flags; `SEX`, every `RAC*` recode/indicator, and their derived encodings.
- Income components and related totals (`WAGP`, `SEMP`, `INTP`, `OIP`, `PAP`, `RETP`, `SSIP`, `SSP`, `PERNP`, `POVPIP`); insurance answers/recode families (`HINS*`, `HICOV`, `PRIVCOV`). These include components or related direct answers, not necessarily individually exact copies.
- `MIGPUMA` and `MIGSP`: in this cohort, their missingness exactly identifies `MIG == 1`.
- `FER`: in this cohort, its nonmissingness exactly identifies `SEX == 2`.
- Current-work questions/recodes (`NW*`, `WRK`, `WKL`, `WKW`, `MIL`, `ESP`), and `COW`. Specifically, missing `COW` identifies `ESR == 6`, and `COW == 9` identifies `ESR == 3` here. This is partial deterministic disclosure, even though `COW` does not reconstruct every `ESR` class.
- Journey-to-work answers (`JW*`, `POW*`, `DRIVESP`), including arrival/departure times; identifiers/geography (`SERIALNO`, `SPORDER`, `PUMA`, `ST`, etc.), weights and all allocation flags. Keep grouping keys outside the release for separation checks.

Also omit redundant `NATIVITY`: observed `CIT` codes 1–3 map to native and 4–5 to foreign-born. `ENG` is outside the existing repository input set and is unnecessary for this compact first pass. These exclusion judgments combine dictionary semantics with exact local schema checks; they are not privacy guarantees. [Census definitions](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt).

## Local provenance and checks

Read-only input: `data/folktables/2018/1-Year/psam_p06.csv`, 267297811 bytes, SHA256 `dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`. It contains 378817 rows and 286 columns. The declared cohort has 80329 distinct person keys and 53907 `SERIALNO` groups. This is whole-cohort schema inventory, not model selection or held-out evaluation.

All required labels except commute are nonmissing and within their fixed schemas in that cohort. There are 53302 valid commuting observations: 52172 civilian-at-work and 1130 military-at-work; 27027 observations have inapplicable/missing commute. `WKHP` has 16482 missing observations. Exact `FER`, migration-location, citizenship/nativity, and partial `COW` relations above were checked directly without fitting a predictor.

The existing [repository loader](../pcrl/data/folktables.py) and processed parquet are unsuitable unchanged: they retain `MIG`, `SEX`, binary race, and `FER`, use a different population, and fill missing task fields before forming negatives. The new pilot reads raw columns explicitly and keeps source-label, held-out-label, audit-label, and feature-preprocessing APIs separate. Household grouping uses `SERIALNO`; person identity uses `(SERIALNO, SPORDER)`. Neither identifier enters the feature matrix.
