# Retired classification-accuracy certificate

The universal claim that zero affine least-squares R² forces every linear
threshold classifier to majority accuracy is false. The former
`certified_accuracy_bound` API now raises `NotImplementedError`. Its derived
`NonlinearComplianceCertificate.check` also raises, because randomized
smoothing does not establish the missing R²-to-accuracy implication.

Use this exact finite sample, with each row indicating repeated observations:

| A | h | Count |
|---|---:|---:|
| 1 | 1 | 9 |
| 1 | -9 | 1 |
| 0 | -1 | 9 |
| 0 | 9 | 1 |

Both conditional feature means are zero. Therefore Cov(h, A) = 0, the
optimal affine least-squares predictor is the constant 1/2, and its R² is
zero. Nevertheless, predicting A = 1 exactly when h > 0 classifies 18 of
20 observations correctly (90%), versus a majority baseline of 50%.
`tests/test_accuracy_bound.py` verifies these values directly without
training a classifier or invoking asymptotic approximations.

The valid narrower statement is about squared loss: zero cross-covariance
means affine least-squares prediction cannot improve on the optimal
constant predictor on that distribution (or the supplied empirical
sample). It does not imply statistical independence, a bound on arbitrary
threshold decisions, or out-of-sample privacy. Equal class means also do
not imply equal class-conditional distributions. Empirical, regularized
R² scores should be described with their fitting data and numerical
regularization; refitting on an evaluation sample is a descriptive
least-squares statistic, not a held-out attacker's predictive score.

For two views with exactly zero cross-covariance with the same prohibited
signal, their concatenation has exactly zero cross-covariance too. The
approximate-leakage example h1 = N + delta S and h2 = N - delta S does not
contradict this statement: for nonzero delta each view already has nonzero
squared-linear R² = delta²/(1 + delta²), and their difference recovers S.

Active compliance reports no longer generate or display accuracy bounds.
Legacy result fields remain readable, with new report values set to
`None` and `accuracy_guarantee_status = "retired_invalid"`. Historical
results, archived plans, and paper result tables remain unchanged. The
available `paper-body/` tree contains result tables only; a manuscript
theorem source is not present in this checkout, so this correction is the
active statement governing interpretation of those artifacts.
