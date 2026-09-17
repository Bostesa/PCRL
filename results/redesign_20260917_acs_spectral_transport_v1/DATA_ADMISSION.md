# Data admission: California ACS 2017 one-year (temporal transport source)

**Decision: admitted (2017).** The 2016 fallback was not accessed. Admission rested on provenance and schema only. It was recorded at commit `ad2c088` (2026-09-10), before any 2017 prediction existed. The complete original record is the development study's [`TRANSPORT_ADMISSION.md`](../redesign_20260910_acs_residual_spectral_v1/TRANSPORT_ADMISSION.md), with hashes in [`DATA_ADMISSION.json`](../redesign_20260910_acs_residual_spectral_v1/DATA_ADMISSION.json). This file summarizes that record and adds the checks made in this study before the lock.

## Provenance

* Source: U.S. Census Bureau, [2017 ACS 1-year PUMS, California person file](https://www2.census.gov/programs-surveys/acs/data/pums/2017/1-Year/csv_pca.zip) (SHA256 `d28c43da…d6617`), from the [official release directory](https://www2.census.gov/programs-surveys/acs/data/pums/2017/1-Year/).
* Dictionaries: [2017](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2017.txt) and [2018](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt). Every categorical feature and target entry is identical across the two years.
* Only the person CSV is used. Raw files stay local under the ignored `data/acs_spectral_transport/`.

## Definitions carried over unchanged

| Role | Definition |
|---|---|
| Cohort | 19 <= AGEP <= 34 and PWGTP > 0; all eligible rows; no cap |
| Features | AGEP, WKHP (numeric; invalid or missing -> frozen median plus missing flag); SCHL, MAR, RELP, CIT, DIS, DEAR, DEYE, DREM (categorical) |
| Source tasks | PINCP > 50,000; ESR == 1; PUBCOV == 1 |
| Reserved tasks | MIG == 1 (same residence); JWMNP > 20 |
| Sensitive | SEX (2 classes), RAC1P (9 classes; no collapsing) |
| Weights | PWGTP; scoring only |

Income is nominal and not inflation-adjusted, as in the original code; ADJINC differs between years (1.011189 vs 1.013097). This limits interpretation of income-service quality across years. It is not repaired, because repairing it would redefine a frozen target.

## Compatibility with the frozen 2018 objects (checked in this study before the lock)

* **Pipeline identity:** the transport code applied to the 2018 cohort reproduces bitwise every saved 2018 PCA array, anchor vector, rich/tree bank, historical wire and derived release, and spectral release, for every pool and seed (57–59 array checks per pool; `seed_*/REPLAY_2018.json`).
* **Category handling:** the frozen `CovariatePreprocessor` has explicit `missing` and `unseen` columns for every categorical feature; these existed before 2017 was touched. Across the four fitting/validation partitions, valid categories unseen in 2018 representation fitting are rare: at most 5 SCHL and 3 RELP persons per partition (`seed_*/feature_support_fit_partitions.json`). No merging or remapping occurs. Final-partition counts are recorded after the lock in `seed_*/context/feature_support_final.json`.
* **RELP:** the 2017 and 2018 dictionaries define the same 0–17 codes, so no relationship-category schema change needs repair.
* **Keys:** 2017 SERIALNO is 13 digits with a year prefix; 2018 uses HU/GQ prefixes. SPORDER changes storage type only. Year-qualified keys `2017|SERIALNO|SPORDER` are used.

## Protected-class support

| Partition | People | RAC1P=4 (Alaska Native) count |
|---|---:|---:|
| attacker fit | 19,988 | 2 |
| attacker validation | 11,894 | 0 |
| task fit | 20,017 | 4 |
| task validation | 12,046 | 1 |
| final evaluation | 15,924 | 1 |

Fresh 2017 race attackers are fitted on a 4,096-row subset of attacker fit and may or may not contain the rare classes. The per-seed fitting support of both modes is reported in `TRANSPORT_DECISION.json` (`race_support`). Class-level recall and AUROC are undefined where support is absent. Classes with fewer than 10 final-partition people are flagged as insufficient for class-level claims. Full nine-class log loss is always scored with the fixed 1e-12 floor.

## Prior-use audit and its limits

The admission scan (all available text, compressed JSON/CSV, configurations, manifests and 166 reachable commits in the original checkout) found no documented 2016 or 2017 ACS use. A bounded re-check on 2026-09-17 listed every file modified after the admission commit (2026-09-10 23:24) in both the original checkout and the worktree, plus all commits since. The only changes were Finder `.DS_Store` metadata and this study's own protocol and code, and the only new commit is this study's protocol commit `9be345a`. A repository inventory cannot rule out undocumented external use, and a filename search with no hit is not proof of nonuse.

## Independence limitation

This is transport between public survey years, not a fresh sample from the 2018 distribution. ACS housing-unit sampling generally avoids re-sampling an address within five years, but a person who moves can appear at a new sampled address. Public identifiers therefore cannot establish that 2017 respondents are different people from 2018 respondents. Verified disjointness is limited to published SERIALNO groups within 2017.
