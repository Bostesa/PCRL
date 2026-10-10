# R2 utility and sampled-channel audit

Status: data-free reconciliation of admitted public aggregate receipts and old
implementation code. This is exploratory development on reused Adult people.
The old assessment, HEAD_VALIDATION labels, private person arrays and channel
checkpoints were not opened for this reconciliation. The script
`build_channel_identity_audit.py` writes the complete 48-row identity
cost/cap/margin table, sixteen-fit disposition table, and source-hash receipt.

## Utility conventions and exposure

The common task convention is `income` with two classes and `occupation` with
six classes. Probabilities are checked for finite, nonnegative rows summing to
one. Decisions use NumPy first-argmax. Accuracy is an unweighted person mean.
NLL is the mean negative natural log of the true-class probability after a
`1e-12` floor. Brier is the person mean of the **sum** of squared errors over
all classes. Cost is reference minus candidate accuracy, or candidate minus
reference NLL/Brier; each cap is inclusive. The current common comparison
requires U0 **and** Ucal on the same ordered role rows. A point cap is not a
simultaneous interval pass.

The repaired role-only export has 15,434 OSF_DEFENSE_FIT people (15,428
groups), 2,000 CALIBRATION_HELDOUT people (2,000 groups), and 2,235
INNER_SELECTION people (2,234 groups), with zero direct row/group overlap.
U0's source encoder and deployed heads were fitted on all FIT rows; historical
head selection used HEAD_VALIDATION task labels. Ucal applies one global
temperature per task/seed fitted on CALIBRATION_HELDOUT; its family had been
chosen historically using INNER. The predecessor reference repair replayed
48 U0/Ucal role/task/seed outputs with bitwise parity, and the latest campaign
reused those role-only exports. An internal validation fold for a new head was
not unseen by the frozen U0 encoder. The earlier full-input F and U0 had
unequal task-fitting exposure in the predecessor diagnostic, so their old
loss contrast is descriptive.

Any new comparable-role utility worker must use the role-only store, exact
83-feature order, frozen source/head/calibrator manifests and queue slot.
FIT task labels are admitted for task learning/checks; CAL labels only for
fixed global-temperature fitting; INNER task labels only after releases and
selection are frozen. HEAD_VALIDATION labels, old assessment arrays, and
historical mixed-role scientific loaders remain embargoed. SEX is unnecessary
for utility calculations.

## Exact bank identity versus old channel

The old `build_actions` lists each fine cell's source output first, then two
nearest same-class outputs and a same-class medoid. Exact duplicate public
probability/decision values are assigned one global code ID. Each seed has 16
income bank states and 321 occupation bank states. The prior source replay
found exact fine-cell/source-code/decoder parity on FIT, ATT and INNER; this
audit cross-checks all 36 later fixed INNER identity/U0/Ucal task points
against that independent earlier aggregate receipt within `1e-12`.

`I1` has two distinct answers. The fixed `A_IDENTITY` diagnostic is an exact
same-bank deployable action, and old deterministic controls can emit its first
action exactly through hard argmax. The old stochastic joint/product/local
and sequential factors used a finite softmax initialized with a `+4` identity
bias. They had no explicit exact identity checkpoint; that bias gives a
near-identity distribution, not exact identity. An explicit `eta=0` boundary
mixture could repair this representation issue for the appropriate family,
but cannot fix the current identity's fitting loss.

The old FIT constraints were **four** mean costs: NLL and Brier for each
recipient, at the tighter of U0 and Ucal fitting-role references plus 0.010
and 0.005 respectively. Accuracy was not an old FIT constraint. The protected
local Bayes ceilings equaled identity recovery; exact identity would meet
them at equality. Pair recovery was the optimization objective, not a hard
FIT benefit requirement.

| Identity result | Seed 0 | Seed 1 |
| --- | ---: | ---: |
| `I2` FIT occupation NLL excess vs U0 (cap .010) | .014312011, FAIL | .015420556, FAIL |
| `I2` FIT occupation Brier excess vs U0 (cap .005) | .004977071, PASS | .005348077, FAIL |
| `I3` common INNER, both tasks and references | point PASS | point PASS |

The exact identity thus fails the unchanged fitting program in both seeds,
even though its INNER point utility passes. The source's finite bank may have
other feasible policies; the optimistic label-aware bank minima do not
construct one. No infeasibility theorem follows.

Fourteen prior arm/seed searches completed without a selected feasible
checkpoint. The two reverse-sequential searches stopped after infeasible
first-stage runs, with second stages unrun; they are incomplete searches, not
completed proofs. The full sixteen-row disposition is in
`CHANNEL_SEARCH_DISPOSITION.csv`. The prior joint stochastic seed-0
independent checkpoint replay found zero feasible among 26 saved checkpoints,
with the smallest maximum residual 0.004342792 in occupation NLL. That is a
bounded heuristic observation.

The prompt's A continuation requires I1, I2 and I3 all to pass under the
current fitting semantics. I2 fails, so no new channel optimization is
admitted for this study. The code-capacity and conditional kernel lane has a
separate prerequisite and may continue. A future alternative FIT formulation
would be a new prospective design, not an amendment inferred from these
outcomes.

Sources: prior `r04/A_PREFIT_INDEPENDENT_REPLAY.json`,
`r01/BANK_IDENTITY_AUDIT.json`, `r02/A_FIXED_POINT_ROWS_R2.csv`,
`r03/A_PRODUCER_FIT_CLOSEOUT.json`, `r04/A_FIT_16_STATUS_INDEPENDENT_AUDIT.json`,
reference repair `REFERENCE_AND_ROLE_AUDIT.md`, and the source files hashed in
`CHANNEL_IDENTITY_RECEIPT.json`.
