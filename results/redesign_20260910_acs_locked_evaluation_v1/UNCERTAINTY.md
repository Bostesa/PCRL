# Household uncertainty: unavailable

All six declared paired contrasts are **unassessable** because there are0 selected household groups. No point estimate, seed SD, percentile interval or simultaneous band exists. Planned replicates:2,000; executed:0. No draws were made, replaced or discarded. [UNCERTAINTY.json](UNCERTAINTY.json) retains every unavailable contrast.

The implemented supplement uses household-level loss-difference sums and eligible count/PWGTP denominators, common multiplicities across all conditions/targets/three fitted seeds, and per-replicate ratios. It averages three per-seed metrics, never probabilities. RNG seed20260910 and linear/type7 quantiles are fixed. Ordinary95% percentile intervals and the exact six-member max-absolute-standardized centered-bootstrap family are implemented; zero variance is degenerate and undefined replicates make the affected interval unavailable. The full family is not silently reduced.

Synthetic heterogeneous-size household tests match explicit repeated-household arithmetic with missing masks, different weights and shared seed resampling. Tampered deterministic draw files are rejected. No neural inference or fitting occurs in bootstrap. These are implementation checks, not population intervals.

If usable data existed, intervals would describe household variability conditional on the fitted systems and benchmark cohort under a bootstrap approximation. They would not be official ACS survey-design intervals, exact guarantees, privacy certificates or new training-cohort replication. Favorable bands could not override failed source/utility matching, individual race regressions or missing category support. Here even that conditional assessment is unavailable.
