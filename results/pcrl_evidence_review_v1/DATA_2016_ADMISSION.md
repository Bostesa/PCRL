# Data admission: California ACS 2016 one-year person file

**Decision: admissible on provenance and schema grounds, with one documented change of estimand.**

Prepared by Terminal B, 2026-09-17, branch `research/pcrl-evidence-paper-v1`. This is *admission
preparation only*. **No model was fitted, no row was transformed through any existing model, no
prediction was made, no loss was scored, and no label of the prospective final partition was read.**
The only label access anywhere was class-support counting inside the fitting and validation pools,
which is planning information and is narrowly recorded in §6.

Machine-readable record: [`ACS_2016_ADMISSION_MANIFEST.json`](ACS_2016_ADMISSION_MANIFEST.json).
Code: `experiments/pcrl_evidence_review_v1/admit_acs_2016.py`.

## 1. Provenance

| Object | Source | SHA256 |
|---|---|---|
| `csv_pca_2016.zip` | `https://www2.census.gov/programs-surveys/acs/data/pums/2016/1-Year/csv_pca.zip` | `a434bf9316f174d896766488cd5fd4ae7356a9512258321110b62110c6ded818` |
| `ss16pca.csv` (extracted) | the above | `9d8b7eb28baa74a88537d3f429b0a361cc36e8c60f67c25d52cac1fb0a7d5c1a` |
| `PUMSDataDict16.txt` | `https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict16.txt` | `b21a4dd6500b03fe61de92f5433a3b25b635b07492a9806b51db2d8a49e2b89b` |
| `PUMS_Data_Dictionary_2017.txt` | `https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2017.txt` | `ea5bcde2a07b0c76ac145fac108e9013480b686a9d70afc4e3f4a6512ccbfd69` |

Only the California person CSV is used. Raw records stay local under the ignored path
`data/acs_2016_admission/`; only aggregate provenance and counts are published.

*Note on the dictionary URL.* The 2016 one-year dictionary is `PUMSDataDict16.txt`, not
`PUMS_Data_Dictionary_2016.txt` — the latter returns a Census 404 page, which a naïve download would
save as a valid-looking file. The downloaded file was checked to begin
`2016 ACS PUMS DATA DICTIONARY / October 19, 2017`.

## 2. Prior-use audit

* **Repository history.** All 177 reachable commits were searched for year-tokens
  (`acs[-_ ]?2016`, `ss16p`, `pums.*2016`). The only pre-existing hits are the 2017 admission's own
  prior-use scanner, which searches *for* the string, and a bibliographic mention in
  `PRIOR_WORK_AND_NOVELTY.md`. No 2016 data path, cache, manifest or result exists.
* **Overlapping five-year products.** None. The only cached ACS artifact is
  `data/folktables/acs_2018_CA_v2_ffb_binarized.parquet`; the Folktables loader's `DEFAULT_YEAR` is
  `"2018"` with `horizon="1-Year"`; and `docs/ACS_2018_SCHEMA_NOTES.md` explicitly records that the
  2018 dictionary used is the one-year, "not the 2018–2022 five-year release". No 2012–2016 or other
  five-year file has been downloaded or referenced.
* **What cannot be verified.** A repository inventory cannot rule out undocumented external use of
  this public file, and a filename search returning nothing is not proof of non-use.

## 3. Schema validation against the frozen definitions

The 2016 frame was validated against exactly the code ranges the 2017 admission used. **Zero
violations.**

| Check | Result |
|---|---|
| Record type / state | all rows `RT = P`, `ST = 06` |
| Out-of-dictionary values, all 17 feature/target/sensitive columns | 0 |
| `SPORDER` in 1–20, integral | pass |
| `PWGTP` in 0–9999, integral | pass |
| Raw rows | 376,035 |
| Eligible (19 ≤ AGEP ≤ 34, PWGTP > 0) | 79,298 people in 53,093 published groups |
| Duplicate person keys removed | 0 |
| Conflicting duplicate keys | 0 |
| Distinct `2016\|SERIALNO\|SPORDER` keys | 79,298 (= rows) |

For scale, the 2017 file gave 79,869 eligible people in 53,504 groups, so the cohorts are closely
comparable in size.

## 4. Feature and target semantics: 2016 against 2017

Dictionary blocks were parsed from both files — they use different layouts, so the comparison is on
extracted (code → label) maps and declared widths, which is what the semantics depend on.

**16 of 17 variables have identical code sets:** `AGEP`, `WKHP`, `SCHL`, `MAR`, `RELP`, `CIT`, `DIS`,
`DEAR`, `DEYE`, `DREM`, `ESR`, `PUBCOV`, `MIG`, `JWMNP`, `SEX`, `RAC1P`. In particular `RELP` keeps
the same 00–17 codes and `RAC1P` the same nine classes, so no category is merged or remapped.

Two differences, both recorded rather than repaired:

| Variable | 2016 | 2017 | Assessment |
|---|---|---|---|
| `AGEP` | label reads `1 to 99 years (Top-coded***)` | `1 to 99 years (Top-coded)` | cosmetic footnote marker; identical code set and range |
| `PINCP` | declared positive range `2..9999999`; description `Total person's income (signed)` | declared `2..4209995`; description adds `use ADJINC to adjust to constant dollars` | **Declared top-code ceiling differs.** No observed 2016 California value falls outside the 2017 range (0 out-of-dictionary rows), and the frozen task threshold is `PINCP > 50,000`, far below either ceiling, so the binary target is unaffected in this data. |

## 5. Change of estimand — documented, not repaired

`ADJINC` is **1.007588** for 2016, against 1.011189 for 2017 and 1.013097 for 2018. The frozen task
`income_binary = PINCP > 50,000` is **nominal and unadjusted**, as in the original code. Carrying it
to 2016 therefore moves the threshold further in real terms than the 2017 transport already did: the
same nominal cut-off corresponds to a higher real income in an earlier year.

This is a change in what the income task *means* across years. It is recorded here and must be
restated in any report that uses 2016. **It is not deflated**, because adjusting it would redefine a
target that was frozen before any of this evidence existed, and silently changing a frozen task to
obtain compatibility is exactly the manoeuvre an admission record exists to prevent.

## 6. Prospective partition

Deterministic and label-blind. Groups are ordered by
`SHA256("PCRL-evidence-review-2016-admission-v1|2016|06|SERIALNO")`, ties broken by `SERIALNO`, then
cut at rounded cumulative boundaries.

**A new salt is used, not the 2017 one.** Two reasons, both fixed before any outcome: reusing the
2017 salt would make the two years' partitions correlated through the identifier hash; and 2016
`SERIALNO` values are 9 digits with no year prefix (e.g. `000000033`), while 2017 prefixes the year,
so identifier spaces are not comparable and year-qualified keys `2016|SERIALNO|SPORDER` are required
throughout.

The instruction's **50/20/30** fitting / validation / final split is used at the top level. Inside
the 50% and 20% blocks the 2017 study's convention is retained — a further even split into disjoint
attacker and task pools — so that utility probes and attackers never share a household, as they did
not in 2017. Retaining that convention is what makes a 2016 result comparable to the 2017 design; it
is recorded here rather than assumed.

| Pool | People | Households |
|---|---:|---:|
| fitting | 39,697 | 26,546 |
| — attacker_fit | 19,914 | 13,273 |
| — task_fit | 19,783 | 13,273 |
| validation | 15,917 | 10,619 |
| — attacker_validation | 7,911 | 5,310 |
| — task_validation | 8,006 | 5,309 |
| **final_evaluation** | **23,684** | **15,928** |

Household overlap between the three top-level pools: **0, 0, 0**. Array hashes for the identifier,
weight and partition-index arrays are in the manifest.

The prospective final partition is about 49% larger than 2017's (23,684 against 15,924 people).

### Protected-class support, fitting and validation pools only

`RAC1P` counts by class index 0–8 (code − 1):

| Pool | counts |
|---|---|
| fitting | [22268, 2123, 288, 7, 98, 6227, 156, 6346, 2184] |
| — attacker_fit | [11158, 1053, 145, 4, 46, 3180, 81, 3158, 1089] |
| — task_fit | [11110, 1070, 143, 3, 52, 3047, 75, 3188, 1095] |
| validation | [9009, 844, 114, 4, 40, 2531, 66, 2461, 848] |
| — attacker_validation | [4446, 406, 51, **2**, 22, 1290, 33, 1222, 439] |
| — task_validation | [4563, 438, 63, 2, 18, 1241, 33, 1239, 409] |
| **final_evaluation** | **not read** |

**Every one of the nine classes has support in every fitting and validation pool.** The bolded cell
matters: index 3 is `RAC1P = 4`, Alaska Native alone, which had **zero** support in the 2017 attacker
validation pool and is the reason 2017 class-level race statements were withheld. 2016 has two such
people there. That is a planning improvement, not a result, and two people is still far too few for a
class-level claim — but it removes an outright impossibility rather than reducing an error bar.

**The final partition contributes 23,684 rows and 15,928 households, and nothing else.** No label,
outcome distribution or protected-class summary of the final partition was computed, printed or
stored; the manifest records `labels_read: false` for that pool and the code path is structurally
incapable of producing one.

## 7. The evaluator interface that would enforce the future lock

`experiments/pcrl_evidence_review_v1/sealed_year_loader.py`. It records the SHA256 of every declared
input and the partition manifest, refuses to re-seal, refuses to open the final partition if any
hashed input is missing or changed, logs every access with the lock digest, requires the first access
to be strictly after the lock's creation, and accepts amendments that name **code files already in
the lock** and nothing else. `open_final` returns row counts and hashes — never a label array — so a
score cannot be computed before the lock verifies.

Verified by **21 tests** (`tests/pcrl_evidence_review_v1/test_sealed_year_loader.py`) on synthetic
fixtures plus one round-trip over published files of the **already spent** 2017 study. It has **not**
been run on 2016 outputs, because there are none: no 2016 lock exists, and creating one now would be
premature — the lock belongs after the future study's fitting and selection, not before it.

## 8. Limitations

* **Not an independent sample.** This is transport between public survey years of the same state and
  age band, not a fresh draw from the 2018 distribution. Public identifiers **cannot** establish that
  2016 respondents are different people from 2017 or 2018 respondents; ACS housing-unit sampling
  generally avoids re-sampling an address within five years, but a mover can appear at a new sampled
  address. Verified disjointness is limited to published `SERIALNO` groups *within* 2016. Nothing
  here may be described as verified distinct people or as same-distribution confirmation.
* **The income estimand shifts** (§5).
* **Category support in a fitting *subset*.** Attackers are fitted on subsets of the attacker-fit
  pool, so a rare class present in the pool may still be absent from a given fit; that must be
  reported per seed in any future study, as 2017 did.
* **Two people is not support for a class-level claim.** Index 3 remains flagged.
* **Admission is not a licence.** 2016 is admissible; it is not yet locked, and a future study needs
  its own prospective protocol, its own comparison specification and its own lock before any 2016
  prediction exists.
