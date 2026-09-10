# California ACS 2017 one-year: future transport admission

The California 2017 ACS one-year person file is admitted for **future file-level temporal transport**, subject to the exact limitations below. No model was fitted, applied, selected, or scored on this release. The 2016 fallback was not accessed because the first candidate has compatible schema and no documented prior use in the audited checkout. This admission does not repair the exhausted 2018 evaluation population or constitute a new scientific result.

## Source and schema checks

The source is the Census Bureau's [2017 one-year California person archive](https://www2.census.gov/programs-surveys/acs/data/pums/2017/1-Year/csv_pca.zip), selected from its [official release directory](https://www2.census.gov/programs-surveys/acs/data/pums/2017/1-Year/). Archive CRC integrity and extracted-member SHA256 equality are verified. Only the person CSV is used. `DATA_ADMISSION.json` records archive, raw CSV, dictionary, code, per-array, and per-file hashes.

The [2017 dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2017.txt) and [2018 dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.txt) were downloaded and compared with `experiments/acs_transfer_data.py`, the transfer configuration, and the locked evaluator's task definitions. The entire dictionary entries for every categorical feature and categorical target are identical across these annual releases. The raw CSV passes explicit range, missingness, state, record-type, integer-value, person-key, and release-identifier checks. Invalid nonmissing categories fail admission rather than being mapped to another category.

| Role | Exact columns and interpretation |
|---|---|
| Numeric features | AGEP and WKHP, in that order; original age 0–99 and work-hours 1–99 validity conventions |
| Categorical features | SCHL 1–24; MAR 1–5; RELP 0–17; CIT 1–5; DIS, DEAR, DEYE, DREM 1–2 |
| Source tasks | PINCP > 50,000 within original valid range −19,998 to 4,209,995; ESR == 1 among ESR 1–6; PUBCOV == 1 among PUBCOV 1–2 |
| Reserved tasks | MIG == 1 among MIG 1–3; JWMNP > 20 among integer JWMNP 1–200 |
| Audit targets | SEX 1–2 and RAC1P 1–9, converted to original zero-based class indices; no race collapsing |
| Keys and weights | SERIALNO retained as a string, SPORDER normalized only as its exact integer person number 1–20, PWGTP retained as person weight 1–9999 |
| Eligibility | 19 ≤ AGEP ≤ 34 and PWGTP > 0, all eligible rows, no additional GQ or task-validity filter |

2017 SERIALNO has 13 decimal characters beginning with 2017; 2018 uses 2018HU/2018GQ prefixes. SPORDER's dictionary type changes from character to numeric, with the same person-number domain. AGEP, WKHP, JWMNP and PWGTP have numeric zero-padding differences. PINCP retains the same bounds, with revised presentation of its values. These differences do not require feature or target category remapping. The [2017 readme](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/ACS2017_PUMS_README.pdf) documents the annual serial-number format.

**Income stays nominal and unadjusted**, matching the original code. ADJINC differs (1.011189 in 2017; 1.013097 in 2018), and annual price levels and rolling income reference periods differ. This limits interpretation of cross-year income utility. No inflation adjustment or refitting of income-bin edges occurs. This package contains the five specified binary tasks and two audit targets; it does not construct a newly fitted income-bin or joint-label anchor bank. Frozen 2018 preprocessors, PCA, anchors and learned representations must remain frozen in any future transport analysis.

## Prior use and independence scope

`DATA_ADMISSION_PRIOR_USE.json` records the historical checkout commit/tree, scan scope, every exact 2016/2017 token match, configuration/manifest hashes, available annual data-cache names, and direct pipeline/exhaustion evidence. The audit covers tracked and available ignored text artifacts, including compressed JSON/CSV evidence, under the read-only original checkout. It searches code, configurations, manifests, source snapshots, recorded research use and documentation; raw datasets are inventoried by filename rather than scanned. Numerical support/step-count matches and unrelated bibliography years are retained for review. No documented ACS 2017/2016 research exposure was found.

The reachable git-history search covers 166 commits and all data loaders, experiments, configurations, and config/manifest JSON paths (`TRANSPORT_HISTORY_AUDIT.json`). Its only matched change is an unrelated 2016 covariance-barycenter citation. A supplemental case-insensitive scan also checks embedded names (such as `acs2017`, `acs_2017`, and year-delimited paths) in available text, compressed JSON, and the same reachable git history; it returns zero matches.

The historical transfer configuration points to California 2018 one-year data; earlier Folktables and dependent redesign studies also use 2018. Commit `482f14d` and the preserved locked evaluation report establish that all 80,329 eligible 2018 people in 53,907 published household/GQ identifiers were previously exposed. Those records remain unchanged and provide no untouched 2018 remainder. A different public annual release is a transport population, not a continuation of that same-population test.

**Public identifiers cannot prove longitudinal individual independence.** ACS housing-unit sampling generally avoids resampling an address within five years, but a moving individual can appear at a new sampled address; the Census Bureau explicitly describes this qualification in its [survey-methodology training](https://www2.census.gov/about/training-workshops/2025/2025-05-21-survey-sample-and-developing-estimates-transcript.pdf). Annual SERIALNO equality or inequality cannot establish cross-year human identity. In addition, SERIALNO represents a housing unit or GQ person, and the public file does not expose a shared physical facility identifier for all GQ residents. The verified disjointness claim is therefore limited to the published groups within 2017. Undocumented external research use cannot be ruled out from a repository inventory.

## Reproducible partitions and sealed future evaluation

All eligible published SERIALNO groups are sorted by SHA256 of the UTF-8 string `PCRL-residual-spectral-transport-v1-20260910|2017|06|SERIALNO`, breaking any hash tie by SERIALNO. The ordered groups are allocated 25%, 15%, 25%, 15%, and 20% to attacker fit, attacker validation, task fit, task validation, and final evaluation, using rounded cumulative group-count boundaries. This rule reads no label, feature, or weight value and is shared across future system seeds. Groups are complete, disjoint, and collectively exhaustive; row ordering is the original CSV order. There is no cap, resampling for rare labels, outcome-based admission, or final-score inspection.

Local arrays under ignored `data/acs_spectral_transport/` contain raw features (not fitted feature transformations), exact task/audit labels with −1 missingness, PWGTP, identifiers, and original row indices. Each partition has its own NPZ file, with all hashes and aggregate unweighted/weighted class support published in `DATA_ADMISSION.json`. Final support was inspected only for admission documentation. Future final prediction outcomes remain uncomputed. A future evaluation must lock representation/system choices and attacker/task selection before scoring the final partition; admission alone does not authorize any new tuning on it.

The admitted cohort contains **79,869 eligible people in 53,504 published groups**, from 377,575 raw person records; no duplicate person rows were removed.

| Partition | People | Published groups |
|---|---:|---:|
| Attacker fit | 19,988 | 13,376 |
| Attacker validation | 11,894 | 8,026 |
| Task fit | 20,017 | 13,376 |
| Task validation | 12,046 | 8,025 |
| Final evaluation | 15,924 | 10,701 |

**Race support is limited.** RAC1P code 4 (Alaska Native alone, class index 3) has 2, 0, 4, 1, and 1 people in the five partitions respectively. Thus the attacker-validation partition lacks this class. Full-schema macro AUROC and class-specific recall/AUROC must remain undefined wherever their required support is absent; one-person final support cannot sustain a robust subgroup claim. Log loss still uses the full nine-class schema and must not be substituted for evidence about unsupported class discrimination. The split is not changed to improve support.

## Reproduction

From the new worktree, with the existing local Python environment:

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m scripts.audit_acs_spectral_transport --download
/Users/nathansamson/PCRL/.venv/bin/python -m pytest tests/test_acs_spectral_transport.py -q
/Users/nathansamson/PCRL/.venv/bin/python -m scripts.audit_acs_spectral_transport --verify
```

**Historical admission is frozen at `482f14d648ced588949ff23faeb8d0cb616a781f`.** The `--verify` command requires a historical checkout at that exact commit with the same originally inventoried ignored/local evidence (35,896 scanned text files and 76 config/manifest hashes); a fresh git-only clone is insufficient because it lacks local evidence. The default root `/Users/nathansamson/PCRL` works only while that snapshot is preserved. After publication or any fast-forward, pass `--historical-root /path/to/preserved-482f14d-checkout`. The preserved snapshot must retain its original available-file inventory and annual cache filenames. Verification against a later research tree is expected to fail; it is neither retrospective proof of fresh admission nor permission to ignore subsequently recorded 2017 use. The publication preserves the original admission report, prior-use inventory, and successful contemporaneous verification even when that exact local snapshot is unavailable. A new admission requires a new full provenance review rather than replacing or bypassing the frozen match hash. Verification independently reconstructs cohort, partitions, labels, support and hashes, and compares all arrays without overwriting them. No paid or remote compute is used. PWGTP permits a future weighted sensitivity analysis; no survey-design variance or population-representativeness claim follows from these preparation checks.

Verification completed successfully: eight admission tests passed, followed by a full `--verify` reconstruction with unchanged local array, raw source, dictionary, prior-use and published report hashes. See `TRANSPORT_TESTS.txt` and `TRANSPORT_VERIFICATION.json`.
