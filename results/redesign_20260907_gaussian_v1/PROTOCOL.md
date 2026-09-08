# Fixed Gaussian purpose-conflict pilot

The configuration and seed list (0, 1, 2) were fixed in the runner before seed 0.
There is one configuration per method, no hyperparameter search, and no method
or checkpoint selection using validation or final-test outcomes. All runs are
valid; no pilot run was superseded. `PILOT_BUDGET.md` records measurement before
continuing beyond the first seed.

## Population and recipients

Each observation has independent U,V,S,E1,...,E5 ~ N(0,1). E1,...,E5 are nuisance
variation. Let L contain these eight columns. Observed features are
X = L Q diag(linspace(0.7,1.3,8)) + linspace(-0.4,0.4,8), with a fixed Q from QR
of an 8×8 standard Gaussian matrix generated with NumPy seed 20260907 (diagonal
sign correction applied). `mixing.json` saves the exact matrix and offset.
The transformation is invertible and identical for every seed and method.

P1 predicts U and prohibits V,S. P2 predicts V and prohibits U,S. Both use
squared-error tasks. The combined recipient is authorized to recover U and V;
only S is prohibited. The dataset therefore contains a real purpose conflict:
the union of individual prohibitions includes both required task signals.

For seed s, split RNG seeds are 100000+100*s+i, with i=0,1,2,3 for representation
fit, attacker fit, validation, test. Sizes are 2048,2048,2048,4096. Rows are
independent draws, not reused. Features are centered only using representation
fit. Task-head slopes and intercepts are fit only on representation fit;
attacker slopes and intercepts are fit only on attacker fit. No preprocessing
is fit on validation or test. Numeric target means on evaluated data appear
only in descriptive variance/covariance and the conventional R² denominator.

## Methods and comparability

- No erasure: identical starting features supplied to both purposes.
- Shared union: one affine LEACE map fit against continuous columns U,V,S.
- Per-purpose LEACE: independently fit maps against V,S and U,S, respectively.
- PCRL post-erasure LoRA: the same per-purpose maps followed by the existing
  `PerPurposeLoRAEncoder` (identity frozen linear backbone; rank 2, alpha 2) and
  `ProxyLagrangianOptimizer`. Twenty full-fit-data Adam updates jointly optimize
  the ordinary affine task heads and adapters with continuous ridge R² privacy
  constraints, including combined S. Final task heads are refit with the same
  OLS rule as every baseline. The task head starts at OLS; no random-feature
  backbone or task pretraining is involved. Step-0 and step-20 checkpoints,
  parameter changes, fit objective history, and duals are saved for each seed.

LEACE receives float64 continuous concept matrices through `LeaceEraser.fit`,
as supported by the [official implementation](https://github.com/EleutherAI/concept-erasure)
and [paper §4.3](https://arxiv.org/abs/2306.03819). Installed implementation was
also inspected: it directly accumulates cross-covariance with the supplied
numeric matrix. Shrinkage and covariance-trace constraint are disabled; the
SVD threshold is 1e-10. All methods output 8 coordinates per view, or 16 when
concatenated. Effective ranks are 8/8/8 (raw), 5/5/5 (union), and 6/6/7 (separate
and PCRL), for P1/P2/combined. These rank differences are the intended erasure.

The PCRL arm is a continuous-task adaptation wiring control using existing
primitives, not an execution of the complete categorical V2 trainer. No
VICReg/vCLUB or upstream adaptation is used. A nonsingular linear post-map with
refitted optimal affine heads cannot improve linear predictive information.
The observed tie is expected and supplies no evidence of a PCRL advantage.

## Scores and evidence

`score` reports conventional out-of-sample R² = 1−SSE/SST and MSE, without
clipping negative scores; an intercept predicting the fitting mean can be
worse than the evaluation mean. Constant evaluation targets are undefined.
OLS uses a relative singular-value cutoff of 1e-10, avoiding exploitation of
roundoff remnants of erased directions. `empirical_statistics` separately
saves empirical cross-covariance, target variances, effective ranks, and an
in-sample OLS projection score. Those statistics are not held-out prediction.
Class support is inapplicable to these continuous targets; sample counts and
variances are saved, and categorical support is covered by regression tests.

`seed_*/metrics.json` records all per-view, per-signal validation and test
scores, including authorized combined U,V. `TABLE.md` shows mean ± sample SD
over three seeds; the individual leakage summary takes the maximum of the four
prohibited-pair scores within each seed, then averages. `fitted_arrays.npz`
stores erasers, preprocessing, task heads and attackers. Split RNG seeds plus
the saved transform reproduce data without storing duplicate raw arrays.

`composition_stress.json` is a separate analytical population and Gaussian
fit/test check of h1=N+δS, h2=N−δS. Single-view squared-linear R² is
δ²/(1+δ²); concatenation recovers S exactly whenever δ≠0. Delta values
0, .01, .1, 1 were fixed. An exact four-outcome independent unit-variance
sample also verifies this formula in `tests/test_redesign_conflict.py`.
This is approximate-leakage amplification, not a failure of exact
zero-cross-covariance composition.

Measured process wall time: seed 0 = 1.28 s; seeds 1+2 = 1.03 s; total = 2.31 s.
Measured seed computation time totals 0.8264 s. Hardware: Apple M4 Pro CPU,
14 cores, 24 GiB RAM, macOS 15.6 arm64; one numerical-library thread used.
No downloads, GPU allocation, paid resources, or interference with other runs.

Next experiment: repeat this fixed four-split protocol with a small fixed
nonlinear observation map and task-pretrained features, comparing independent
per-purpose LEACE against PCRL adaptation of upstream features. This is the
first setting here where adaptation can change available task information;
post-erasure linear adaptation with optimal linear heads cannot.
