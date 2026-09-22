# 2016 data admission plan (committed before any 2016 transform, fit or outcome)

1. **Provenance audit (Tier B, after this commit).**
   - Search all refs, worktrees and archives for 2016 paths and derived outputs, including `ss16`,
     `acs_2016`, `2016|`, `PUMSDataDict16` and `csv_pca_2016`.
   - Check prior code paths, manifests and logs for any fitting, transformation, scoring, label inspection
     or tuning that used 2016.
   - Keep label-blind schema/support checks separate from performance-guided use. The known prior use is
     the 2026-09-17 admission (evidence-paper `20a04617`/`299d9a91`). It computed coded label arrays
     in memory for the whole cohort. It stored class-support counts for the fitting and validation pools
     only, and row/household counts for the final pool. Its schema validation reported observed code sets
     and missing counts over the full state file.
   - If performance-guided use turns up, freshness is downgraded and the scope reported. Nothing is
     relabeled and the year is not switched.
2. **Official documentation.**
   - The admitted files are verified by SHA-256: `csv_pca_2016.zip`, `ss16pca.csv`, `PUMSDataDict16.txt`
     and the 2017 dictionary.
   - Fields and coding are re-checked against `PUMSDataDict16.txt` for the ten covariates, the keys and
     the labels (SEX, RAC1P, MIG, plus the service targets).
   - Income estimand shift: ADJINC 2016 = 1.007588. The frozen H_A income service targets nominal
     `PINCP > 50000`. It is not recalibrated or deflated; the shift is documented.
   - The frozen input convention is preserved: invalid or missing values and unseen valid categories map to
     the historical separate columns.
   - An unsupported schema is a technical failure to report. No conversion will be invented.
3. **Cohort and partition.**
   - Cohort: 19 ≤ AGEP ≤ 34, PWGTP > 0, whole SERIALNO groups, and year-qualified keys `2016|SERIALNO|SPORDER`.
   - The committed partition is reproduced by a label-free replay and must match every committed array hash
     (serialno, sporder, pwgtp, raw_row, all partition indices) before use.
   - Pools: fitting 50% (attacker_fit ∪ task_fit), validation 20% (attacker_validation, task_validation),
     final 30%.
4. **Label gating.**
   - Features are transformed with no label column read.
   - Fitting and validation labels are read by row index.
   - Final labels are read only after `EVALUATION_LOCK.json` exists, verifies, and has been committed and
     pushed.
   - Complete-case masks are per role (the target's valid-label mask). They are identical across interfaces
     by construction; the scorer aborts if two interfaces' masks differ.
5. **Records.**
   - Households and people are counted per pool.
   - Missing-label masks, weights and class support are reported for fitting and validation. For the final
     pool they are reported after the lock.
   - Every excluded row is listed with its reason (label missing for the role).
   - The full fixed RAC1P schema is kept. A class absent from a fitting set receives the declared 1e-9
     full-schema floor, and the limitation is reported.
6. **Language.** The evaluation is described as transport to a previously unused survey year with a
   prospectively locked final evaluation. It does not guarantee different underlying people, it is not a
   forecast of future years, and PWGTP weighting does not make it a design-based population interval.
