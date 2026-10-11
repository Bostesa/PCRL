# Independent arithmetic derivation before fitting

R6 owns `conditional_release_independent`. It imports standard Python, NumPy
and SciPy only. No release-runner optimizer, deployment, metric or selection
implementation is imported. Sharing formats and locked written definitions
does not share arithmetic.

## Exact law AUC

For each positive row and negative row, independently draw a single release
from that row's actual finite law. Credit is one for positive score greater
than negative score and one half for an exact equality. Sum masses for each
exact reader-score value first. In ascending score order, each positive mass
is multiplied by all negative mass below plus half the tied negative mass.
Divide by total positive times total negative mass. This computes the stated
conditional single-query estimand and has no release-realization variance.
It does not remove uncertainty from fitted readers, fitted policies, reused
people or sampling a population. Reader orientation is frozen from permitted
attack validation; no reporting-label flip occurs.

Duplicate groups have constant SEX. Thus two records representing one person
never appear on opposite sides of the AUC positive-negative comparison.
Shared cached coins among that person's duplicates are compatible with this
pair expectation. Different people require independent deployment coins.
Bootstrap counts resample groups, multiply every row in a group by the same
count, and remain paired across arms and source seeds. The declared slate
maximum is recomputed within each replicate from the fixed reader slate,
without refitting or changing orientation; two seed endpoints are then
averaged. Fixed slate maxima are not a newly trained reader.

At least 2048 bootstrap values provide a standard deviation with ddof 1.
Interval half-width is that SD times the two-sided Bonferroni t quantile
at 1 - .05/(2K), with group-count-minus-one degrees of freedom. This is
an approximate conditional development interval, not exact simultaneous
coverage, confirmation or a population privacy certificate. The old df=2
three-release uncertainty component is not carried into this estimand.

## Costs, objectives and LP certificate

Each person's loss is the weighted average of losses of emitted decoded
probability vectors. It is not loss of their average probability vector.
Brier sums all class coordinates; argmax uses first index. Actual change
mass excludes all actions that alias the original emitted code tuple.
Bayes recovery sums the larger SEX mass at each actual emitted tuple and
its two marginal IDs. Constant metadata and deterministic probability
functions add no information beyond an exactly equivalent code view, but
wire equality must be verified, not presumed.

For a minimization LP, SciPy/HiGHS inequality marginals are nonpositive,
lower-bound marginals nonnegative and upper-bound marginals nonpositive.
The independent certificate checks primal constraints and bounds,
stationarity, marginal signs, complementarity, primal/dual objective and
gap. A certificate binds the explicit empirical constraint matrix only.
An optimal restricted master is not a full-support certificate unless every
omitted allowed column is priced. A status string is insufficient.

ORIGINAL remaining budgets must subtract the source P0-reference loss
excess on the same FIT rows/folds; negative remaining budgets are retained.
For accuracy, use reference-minus-P0 accuracy as the corresponding loss.
Every fold and both references remain present. Infeasible families imply
untested protection, not a privacy failure.

## Execution status

The eight fixed tests in `test_arithmetic.py` are written but not run pending
engineering-law lock and shared queue permission. No numerical optimizer,
private array, INNER sensitive label or new reader is used by this document.
