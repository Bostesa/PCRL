# Locked 2018 inference review

Reviewed 2026-09-24, read-only against `SELECTION_LOCK.json` SHA-256
`4f894f98af67fd2a79e35ad3697971041aa53451b292b8bdf9691c8dd761f484`.
This is repeatedly used 2018 development evidence, conditional on frozen fitted
releases, predictors, and selection. No outer fit or outcome-driven change was made.

An independent NumPy routine on the Linux host read the three hash-checked
`OUTER_CONTRIBUTIONS.npz` archives and recomputed all 70 locked primary contrasts
and four separately registered H-capability contrasts. Maximum absolute error
in a reported point or anchor estimate was `6.17e-16` (SSM command
`a1f94448-0708-4214-ad1c-91c93b2bbe52`). All releases and five scored roles
within an anchor used identical person IDs, household IDs, and weights.
Overlapping person IDs across anchors had zero household or weight mismatches.
There were 2,968 distinct households in the union and 1,532 present in all
three anchors; the anchors are not independent populations. Twenty-seven
stratified U/P/D17 saved prediction-route and H-route metadata checks matched
the frozen inner records. This last check verified route selection metadata,
not a fresh model fit.

A separate dense-household bootstrap routine regrouped each original-person
loss difference by household, formed each anchor's U or PWGTP numerator and
denominator, and used one multinomial draw on the same 2,968-household union
for every endpoint and anchor. It recomputed 10,000 draws with the locked seed
`20260923`, then used sample standard deviation (`ddof=1`) and equal averaging
of the three anchor-specific ratios. The primary 70-endpoint two-sided
Bonferroni critical value is `3.38403625186456`; its largest absolute SE and
interval-bound differences from `INFERENCE.json` were `6.08e-18` and
`5.99e-17`. The separate four-endpoint capability family has critical value
`2.49770547441237`; its largest SE and bound differences were `9.55e-18` and
`2.09e-17`. All 74 interval decisions agreed (SSM command
`30d1b7c5-fb6b-4b40-a8da-e3e85b91cf91`). The earlier replay command
`451668ac-741d-468b-83a7-accb406b19a4` computed the checks but failed
only while JSON-encoding a NumPy integer; the same read-only calculation was
rerun with scalar conversion. No scientific output was written by either.

The locked U slot is `A_selected` and the P slot is `PrivacyFirst_selected`;
both were marked `DIAGNOSTIC_ONLY` by inner selection. Machine-readable
inference reports U passing 2/30 and P passing 5/40 primary clauses, so
neither full route claim passes. U-versus-D17 task contrasts are +0.00145046
and +0.00091409 nats (U/PWGTP), with intervals that do not establish the
required −0.003 improvement. P-versus-D17 task contrasts are +0.00953851
and +0.00983506 nats; both lower bounds exceed the +0.001 guard. Its
AB/SEX recovery contrast is negative at the point estimate in both weights
but both upper bounds exceed the −0.002 protection target, so superiority is
unproved. All four separately corrected H-capability bounds support at least
0.01 nats of residence benefit; that does not rescue either primary
conjunction. Sensitive contrast is `CE_comparator − CE_candidate`; negative
favors the candidate. A failed upper-bound test is not equivalence, and a
failed non-inferiority test alone is not proof of excess disclosure.

No numerical or sign discrepancy was found in the locked inference or
aggregate endpoint table. One reporting-only comment in `render_results.py`
calls the intervals percentile intervals, while the registered calculation
uses normal bounds centered on the point estimate with a household-bootstrap
SE. The current numeric tables are consistent with the registered method.
These intervals do not cover adaptive model/design selection or full survey
design uncertainty.
