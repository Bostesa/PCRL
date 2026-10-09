# Privacy-score access ordering correction — 2026-10-09

Status: engineering correction before any `r2_privacy_score` seal or unit.
The sealed `r2_pair_select` stage does not import or govern this worker.

The initial committed privacy-score worker enumerated the INNER SEX role file
by hashing its bytes before full-input F gate admission. That was a transient
protected-label read even though no role array was materialized. No privacy
score unit had been registered, sealed, or run, so this caused no real-data
read. The corrected worker first compares the declared role-file SHA in the
sealed unit specification to the already verified manifest JSON. It then
checks the F gate and all selected reader/release identities. Only afterward
does it hash the INNER SEX file and load its role-only array for scoring.

The role, row split, reader slate, control probes, scores, and scientific
criteria are unchanged. A data-free test denies opening the synthetic SEX
file before gate admission; it failed on the original helper and passes on
the corrected helper. This is not a new experiment or result.
