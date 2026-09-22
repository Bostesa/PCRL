# Data independence and freshness of ACS 2016

**Verdict: admissible as a previously unused survey year.** Before this study, no 2016 record had been used
for model fitting, transformation, prediction, scoring, tuning or performance-guided selection. The prior
use was label-blind admission preparation only.

## Audit of prior use

The audit ran after protocol commit `3f9aaee5`. Its raw record is `private/PROVENANCE_AUDIT_RAW.json`.

- **Git, all refs.**
  - Pickaxe searches covered `ss16p`, `acs_2016`, `PUMSDataDict16`, `csv_pca_2016`, `acs2016`, `ACS_2016`
    and `2016_admission`.
  - The only commits that touch data are the admission commits `20a04617` and `299d9a91`
    (evidence-paper branch, 2026-09-17).
  - Every other hit is one of three kinds:
    - an exclusion guard that forbids 2016 paths in bundles (`908093a1`, `adcb5496`, `1ea0e0b0`);
    - a storage note recording that the sealed files were excluded (`df84dfd7`);
    - a manuscript reference to the admission report (`e507e8cd`, `adb8e7ce`, `83b00d59`).
  - No tracked path holds 2016 records or derived arrays.
- **Filesystem, all PCRL worktrees.** The only 2016 files are the admission's raw download
  (`PCRL-terminal-b/data/acs_2016_admission/`) and copies of the admission script and report. No
  derived feature array, prediction or score exists.
- **Private archive** (`pcrl-ux-archive-ed9d21fd`, all prefixes): no 2016 object.

## What the admission itself accessed

`admit_acs_2016.py` at `0d8f4b67`:

- It read the whole state file.
- It reported observed code sets, missing counts and value ranges for the covariates, keys and label
  columns. This was a schema check over all rows, including the people who now form the final partition.
- It computed coded label arrays in memory for the whole eligible cohort. It stored class-support counts
  for the fitting and validation pools only.
- For the final partition it computed and stored row and household counts only.
- No model was fitted, no row was transformed through a model, and no loss was computed.

This is not performance-guided use. The disclosed exposure is the schema/code-set check and the
fitting/validation class-support counts. Those counts were available to us, as planning information, before
this protocol was written.

## Official documentation

- Files were verified by SHA-256 against the admission manifest: `ss16pca.csv`, `csv_pca_2016.zip`,
  `PUMSDataDict16.txt` and the 2017 dictionary.
- The 2018 one-year dictionary was downloaded for this study from the official Census URL
  (sha256 `843a18e1…d5fd`) and parsed with the admission's own parser (`DICTIONARY_2016_VS_2018.json`).
- The ten covariates, the keys and the labels (SEX, RAC1P, MIG, ESR, PUBCOV, JWMNP, WKHP) have identical
  code sets and widths in 2016 and 2018. Two differences exist:
  - AGEP: a footnote marker only.
  - PINCP: a different declared top-code range. No valid 2016 value falls outside the frozen range.
- **Income estimand shift.** ADJINC 2016 = 1.007588, against 1.013097 for 2018. The frozen H_A income
  service uses nominal `PINCP > 50000`. It is neither recalibrated nor deflated. Its 2016 accuracy is
  reported separately.

## Partition

The label-free replay reproduces every committed admission hash: serialno, sporder, pwgtp, raw_row and all
seven partition index arrays (`prepare.py`, `transport.load_cohort_2016`). The partition is one global
household split with zero household overlap between fitting, validation and final.

## Limits

- Transport across public survey years is not a sample of different people. A mover can reappear at a new
  address. Verified disjointness is within 2016 only.
- This is not a forecast of future years.
- PWGTP-weighted household bootstrap intervals are not official ACS design-based intervals.
