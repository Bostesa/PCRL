# D seven-arm independent replay and interval plan

R03 completed both R04 all-arm complete-view verification units. The seed-0
receipt is `FOLLOWUP_S0_ATTACK_INDEPENDENT_REPLAY.json` (SHA-256
`e7b2a5ca7a28466008e9a26dab8441437462b23299f9ffd8faee2f08a3b1024f`);
the seed-1 receipt is `FOLLOWUP_S1_ATTACK_INDEPENDENT_REPLAY.json` (SHA-256
`7654b780f1db676c173b1d625f4da7ca18f18a77e619fd340b0cf31cb1eae086`).
Both have `FOLLOWUP_COMPLETE_VIEW_REPLAY_PASS` status. R04 rebuilt each hard
or continuous ATT/INNER view, matched ordered saved matrices, forwarded the
frozen readers, checked ATT selection, fixed-slate INNER AUCs, pair ignore
aliases and V2 controls. These are technical and point-estimate results;
simultaneous intervals have not yet been run.

| Arm | Seed 0 coalition SEX AUC | Seed 1 coalition SEX AUC | Both-seed INNER utility point gate |
| --- | ---: | ---: | --- |
| T | 0.832582 | 0.835960 | Fail |
| D-K | 0.805905 | 0.793778 | Fail |
| L | 0.817431 | 0.796456 | Fail |
| S12 | 0.802288 | 0.797881 | Fail |
| S21 | 0.811717 | 0.797484 | Fail |
| C | 0.743901 | 0.764677 | Fail |
| F, continuous | 0.869335 | 0.862196 | Pass |

D-K's point coalition recovery falls below T's by `0.026676127` in seed 0
and `0.042181434` in seed 1. Its utility fails in both seeds. Relative to
the lowest-AUC hard control C, D-K has *higher* coalition recovery by
`0.062004738` and `0.029101479` in seeds 0 and 1. C fails utility, so it is
only a descriptive privacy comparison, not an eligible strongest matched
control. None of L, S12, S21 or C passes the two-seed utility gate; the
eligible matched-control set is empty. No D ten-bit arm qualifies for a
competitive claim. The prescribed D-K task-only fallback remains an
ineligible diagnostic.

F passes every U0/Ucal task point cap in both seeds, while T and D-K fail.
F's coalition recovery is higher than T's by `0.036753592` and `0.026236861`,
and higher than D-K's by `0.063429718` and `0.068418295`, in seeds 0 and 1.
Its 16-value continuous wire has different capacity; these are utility/privacy
tradeoff diagnostics, not matched ten-bit protection comparisons. The largest
of those four adverse F coalition gaps is `0.068418295` (seed 1 against
D-K). The strongest favorable D-K versus T point coalition reduction is
`0.042181434` (seed 1). Neither point difference is a confidence conclusion.

## Fixed D interval replay

The sealed `r02/D_FAMILY_ORDER.json` contains exactly 567 ordered raw signed
contrasts: 252 two-task utility differences, 189 D-arm pair AUC differences
and 126 D-arm versus fixed U0/Ucal AUC differences. R04 independently
enumerated the identical IDs without importing R02 scoring. The seven D-arm
INNER probabilities and frozen reader predictions exist and have been
independently replayed. U0/Ucal INNER task probabilities exist, but their
prospective ATT-trained, INNER-frozen complete-view reader prediction banks
have not yet been produced. R02 is preparing those score/decision anchor
attacks. The 126 anchor clauses will remain registered; no reduced-family
critical value will replace the fixed 567-clause family.

After anchor custody and independent wire/reader verification, assemble for
both seeds the same ordered INNER row/group identities, truth labels, nine
release task probabilities, and r1/r2/pair fixed reader score banks. Recompute
every arm's task accuracy, natural-log NLL, summed Brier and every fixed
reader's tie-correct oriented AUC on each resample; recompute the reader
envelope *inside* that resample. Generate 2,048 paired duplicate-group
resamples over the 2,234 INNER groups using PCG64DXSM seed 20261010 and the
sealed 32-byte seed construction, applying identical sampled group positions
to every arm and both seeds. Form all 567 clauses for seed 0, seed 1 and their
equal mean on every draw. Studentize with the sealed `1e-12` floor and
numerically-constant rule; the 95th percentile of the maximum absolute
studentized deviation sets one two-sided simultaneous width. R04's numeric
path will be an R03 verification unit, with all source bytes and public
receipts hash-pinned. It will not import R02 or inherited scoring code.

Apply the sealed nonzero caps *after* forming raw intervals: accuracy
candidate-minus-reference lower bound at least `-0.010`; NLL and Brier
candidate-minus-reference upper bounds at most `0.010` and `0.005`;
coalition control-minus-candidate lower bound at least `0.020`; local
candidate-minus-control upper bound at most `0.010`. Preserve unresolved
crossings and the full family even where an arm has already failed the point
utility screen. No previous assessment role is an input.
