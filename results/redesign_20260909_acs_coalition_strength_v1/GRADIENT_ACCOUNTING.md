# Gradient and continuation accounting

All 36 new continuations used the fixed ordinary individual coefficient .1 and one of the three new added strengths .025, .05 or .2. The 18 beta-zero/.1 systems remain historical reuse. This note describes objectives, update histories and fixed-minibatch gradients; it does not select a strength or infer the cause of a utility/audit outcome.

The source objective keeps weights **.25 income, .25 employment, .5 coverage**. Iplus minimizes `L_source − .1 L_ind − beta*(M_A+M_B)`; J minimizes `L_source − .1 L_ind − beta*M_AB`. The local extra is a **sum**, so the nominal added sensitive-target coefficient is beta/2 on each branch. The opposing-source coefficients remain unchanged. All nine observers minimize their ordinary masked CE recipe; only the reversed forward losses use fixed fitting-prior entropy normalization. No preservation/reconstruction term or new warmup is present.

## Actual optimizer history

Counts below apply to every one of the 12 new conditions within each seed. Inherited states were loaded from the same historical interface fork, including complete Adam moments and counters, schedule descriptors/cursor and torch RNG. “New” counts are the saved final counters minus the saved first-curve fork counters, checked against 80 epochs and three observer updates per forward update.

| Seed | Fitting rows | Forward Adam: inherited + new | Observer Adam: inherited + new, per role |
| --- | ---: | ---: | ---: |
| 0 | 10,513 | 2,520 + 3,360 | 840 + 10,080 |
| 1 | 10,428 | 2,460 + 3,280 | 820 + 9,840 |
| 2 | 10,551 | 2,520 + 3,360 | 840 + 10,080 |

Across all 36 new systems, the saved counts give **120,000 new forward optimizer steps and 360,000 new observer optimizer steps**. Every observer role participates in each observer step. New source exposure is 80 passes per row/task and new observer exposure is 240 passes per row/role; with inherited prefixes, each final history contains 140 source and 260 observer passes. All five training labels are known on these rows. This does not repair absent census code 4 support in independent attacker fitting and validation; observer representation fitting contains respectively 1, 1 and 0 code-4 examples for seeds 0, 1 and 2. [TRAINING_COUNTS.csv](TRAINING_COUNTS.csv) retains the per-condition counts; original prefixes are not counted as new fitting.

## Fixed-state gradients and realized pressure

[GRADIENT_ACCOUNTING.csv](GRADIENT_ACCOUNTING.csv) contains 360 records: 36 systems × fork/final × A mapper, B mapper, A heads, B heads and both mappers together. Each row includes raw losses, actual coefficients, raw/applied component norms, dots/cosines, combined protection and full-objective norms, and the source training-JSON hash. The point is always the original first source-warmup minibatch. No gradient vectors or person-level inputs are published.

The raw fork diagnostics are identical across all six new continuations of a given interface/seed. The table shows three-seed mean **unit-coefficient extra mapper-gradient norms** at those six immutable forks. An added coefficient beta scales this signed component; it does not scale the complete gradient, which retains source and ordinary-individual terms.

| Interface / branch | Iplus unit extra-local norm | J unit coalition norm | Seeds with local norm greater |
| --- | ---: | ---: | --- |
| F / A | .278406 | .192551 | 0, 1, 2 |
| F / B | .358501 | .188512 | 0, 1, 2 |
| P / A | .050418 | .047267 | 1 |
| P / B | .061943 | .058836 | 1, 2 |

Thus matching nominal coefficients does not match realized gradient strength. The F local extra has larger norm at this common state in every seed; the P ordering varies. The full gradient is affine in beta: `grad L(beta) − grad L(0) = beta * grad(−M)`. It is not proportional to beta. The independent replay checked this identity at all five fixed strengths and both regimes on all six unique forks, with maximum float32 gradient-difference error **2.02126103e-8** (declared absolute tolerance 2e-7 and relative tolerance 3e-5). Bitwise miniature terminal-state tests separately reproduce historical I at beta0 and historical Iplus/J at beta.1; no historical scientific continuation was rerun for those tests.

For completeness, final combined applied protection-gradient norms are below, as means over the same three seeds. These are norms of vector sums, not sums of norms; they are evaluated at different trained states and cannot establish a causal account of final utility or recovery.

| Interface / beta | A Iplus | A J | B Iplus | B J |
| --- | ---: | ---: | ---: | ---: |
| F / .025 | .021773 | .017824 | .044713 | .040579 |
| F / .05 | .026291 | .020264 | .044913 | .044088 |
| F / .2 | .038747 | .037233 | .062503 | .060484 |
| P / .025 | .008238 | .008383 | .030275 | .029703 |
| P / .05 | .010700 | .010779 | .031419 | .032035 |
| P / .2 | .022463 | .022885 | .044178 | .047110 |

Across all 72 new diagnostic points, local/extra A-to-B and B-to-A gradients are exactly zero, F protection-to-head gradients are zero, P raw protection-to-head gradients are nonzero, and the coalition raw gradient reaches both mappers. A zero applied coefficient is distinguished from an available nonzero hypothetical component. All diagnostics preserve model/observer tensors, existing gradients and RNG without optimizer steps.

[TRAINING_REPLAY.json](TRAINING_REPLAY.json) passed all 36 new systems, exact full fork/Adam/RNG/schedule and exposure checks, and **72 literal gradient replays with maximum recorded-scalar discrepancy 0**. It reads source/attribute labels only and does not refit historical models or use reserved labels. [Focused tests](FOCUSED_TRAINING_TESTS.json), [protocol](PROTOCOL.md), and [global 54-system release freeze](RELEASE_MANIFEST.json) keep the distinction between immutable fitting controls, later utility/audit measurements and separate deployment alternatives explicit.
