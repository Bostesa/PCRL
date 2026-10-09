# Predeclared simultaneous development intervals for selected full-input F

This file is prospective to the interval worker's first Adult run. It fixes
the method, data role and complete endpoint family. R0 must commit a new
stage lock and R3 must review the worker and inputs before admission.

Prospective correction, 2026-10-09, before any Adult interval run: a synthetic
test showed that an unbounded signed max-t quantile could be negative and put
a one-sided bound inside its point estimate. Commit `219e40eec` clamped each
critical value below at zero. The endpoint family, draws, roles and caps were
unchanged; this correction was committed before the interval science seal.

- Input: the already frozen selected F `GATE_PREDICTIONS.npz` in seeds 0 and 1,
  the matching gate receipts, `FULL_INPUT_GATE.json`, and role-only
  INNER_SELECTION income/occupation task labels. No new fit, calibrator,
  learner choice, mask, SEX label or old assessment array is read.
- Unit: the 2,234 INNER_SELECTION groups, containing 2,235 people. Sample
  exactly 2,234 groups with replacement for each replicate, carrying every
  member of a sampled group. Keep the ordered people and the same sampled
  groups across both seeds, tasks, references and metrics. Paired seeds are
  models on the same people, not independent populations.
- Replicates and RNG: 2,048 draws from NumPy `default_rng(20261009)` (PCG64),
  in batches of 64, using `integers(0, group_count, size=(batch, group_count),
  dtype=int32)`. Order groups by ascending numeric ID. Record a SHA-256 of
  the concatenated C-contiguous sampled `int32` index bytes.
- Fixed family: 24 oriented costs: each seed (0,1) × task (income,
  occupation_group) × reference (U0,Ucal) × endpoint (accuracy drop,
  natural-log NLL excess, summed multiclass Brier excess). Order those axes
  as written. Per-person accuracy drop is correct(reference) minus
  correct(F); NLL excess is `-log(max(F_true,1e-12)) +
  log(max(reference_true,1e-12))`; Brier excess is summed squared-error(F)
  minus summed squared-error(reference). The sample estimate is the mean over
  people, not the unweighted mean of group means. Validate every point cost
  against the frozen R1 gate receipt to `1e-12` before resampling.
- Bounds: compute the standard deviation of each cost's 2,048 bootstrap
  estimates with `ddof=1`. A zero-SD cost has its lower/upper bound equal to
  its point estimate; otherwise define the upper pivot in each draw as
  `(point-bootstrap)/SD` and the lower pivot as `(bootstrap-point)/SD`. Across
  all nonzero-SD costs, take the maximum pivot per draw. Use the 0.95
  quantile with NumPy `method="higher"` for each maximum, then clamp each
  critical value below at zero so neither bound falls inside its point
  estimate. Set simultaneous upper=`point + upper_critical*SD`,
  lower=`point - lower_critical*SD`.
  This yields separate familywise one-sided upper and lower descriptive
  bounds. The two sets together are not claimed to have 95% two-sided
  simultaneous coverage.
- Boundaries: caps are 0.010 for accuracy drop, 0.010 nats for NLL excess,
  and 0.005 for Brier excess. PASS when upper is at or below the cap;
  VIOLATION when lower is strictly above the cap; otherwise UNRESOLVED.
  No endpoint or seed is dropped from the family after results are seen.

These are conditional-on-fitted-model, post-selection development intervals
on reused Adult people. They do not account for the search that chose A,
historical Ucal family selection, FIT/CAL model uncertainty or a fresh
confirmation population. The selected F point gate remains unchanged.
