# Prospective protocol — nonlinear moment refinement and channel dimension

Study: `pcrl_nonlinear_rank_v1` (Terminal A)
Baseline commit: `349efa454afd907389760fd1f59fd8806a215efd` (branch `ablations-facct-2026-07-24` at origin)
Worktree: `/Users/nathansamson/PCRL-terminal-a`, branch `research/pcrl-nonlinear-rank-v1`
Written: 2026-09-17, **before any new interface was fitted, audited or scored.**

## 0. Status of this study in the evidence chain

This is a **development study**. It spends no confirmation set.

* The 2018 ACS pools are the original development resource of the residual spectral study and have been used repeatedly.
* The 2017 California partitions — including the `final_evaluation` partition — were spent by the locked transport study (`results/redesign_20260917_acs_spectral_transport_v1`). Every 2017 number produced here is **EXPLORATORY CROSS-YEAR DEVELOPMENT** evidence. It is not a second confirmation. The original frozen 2017 transport result keeps its historical status and is not restated, reinterpreted or overwritten.
* **2016 is not touched.** No 2016 model output, score, attack, outcome plot or final-label read occurs in Terminal A, in either the go or the no-go branch.
* Development uncertainty here is descriptive. A paired household bootstrap quantifies variability for fixed fitted systems; it cannot undo repeated use of these partitions and does not manufacture a confirmation set.

## 1. Scientific question

Can an objective sensitive to **nonlinear functions of the released channel**, a **smaller channel**, or their combination improve the residence/disclosure tradeoff against **both** local protection (L1, equal-total-mass L2) **and** the frozen neural channel J?

The completed study's diagnosis was surrogate mismatch: its penalty controls a finite set of *first* moments, linear in `Z`, while the attacks that defeat it are nonlinear (MLPs and boosted trees are selected in most roles). This study changes the release-side moment family and the channel dimension. It changes nothing else.

## 2. Registered predictions (recorded before any outcome)

The completed study explicitly recorded that **no directional prediction** had been registered before its locked evaluation, and had to disclose its pass as a prespecified-rule confirmation rather than a called shot. This study registers directional predictions now, so that whatever happens is reported against a standing commitment.

| # | Prediction | Registered call |
|---|---|---|
| P1 | `r_plus = min(16, #positive eig of U-(P_L+P_AB))` equals 16 in **all three seeds**, so eigenvalue-sign rank selection changes nothing and the reduced-rank recipe is an exact alias of rank 16. | **Confirmed before fitting** from already-saved 2018 eigenvalues (`RANK_DIAGNOSTIC.md`). Recorded here as a *structural* prediction that was checkable without a new fit, not as an experimental result. |
| P2 | The nonlinear refinement **will** reduce its own training objective below the original-moment solution at rank 16 (the initialisation is not a stationary point of the new objective). | Open |
| P3 | The nonlinear refinement will **not** produce a strict Pareto improvement over J on the sensitive-recovery endpoints. Specifically, additional A/RAC1P recovery for the best new candidate will remain **above** J's in at least 2 of 3 seeds. | Open |
| P4 | Training-objective improvement will **exceed** independent-attack improvement: the reduction in fitted nonlinear moments will not transfer proportionally to selected fresh-attacker log loss. This is the surrogate-failure prediction. | Open |
| P5 | Rank-8 compression will reduce residence capability (gain over H) relative to rank 16 by more than 0.002 nats in at least 2 of 3 seeds, i.e. the smaller channel costs real utility. | Open |
| P6 | Neither the nonlinear penalty nor rank-8 compression will satisfy the primary coordination rule of §7 against both L1 and L2 while holding residence within 0.001. | Open |

P3, P4 and P6 are predictions of **failure**. They are registered as such deliberately: this study is authorised to publish a negative result, and the scope is bounded by the specified work, not by an obligation to find a positive one.

## 3. Frozen inputs — nothing historical is refitted

Reused bitwise, resolved read-only and hashed in `EVIDENCE_MANIFEST.json`:

* the three frozen 2018 `SpectralModel` objects (`maps.joblib`), including their whitening, residual coefficients, RFF `omega`/`phase`, both polynomial bases, priors and **all fifteen fitted three-fold household OOF nuisance models**;
* historical PCA32 teacher coordinates `T`, released anchors `hA` (4 columns) and `hB` (2 columns), pool row assignments and the representation-fit household IDs;
* the six historical interfaces H, E, A0, L025, L20, J and the eight historical spectral arms, with their 2018 development scores;
* the 2017 partitions and the transport study's per-interface Mode B artifacts;
* the utility- and attacker-subset index arrays (verified against the historical `indices.json` hashes).

**Nuisances are held fixed across old and new arms.** This study changes the release-side moment family and the dimension; it does not simultaneously tune the sensitive nuisance model. Nuisance misspecification is reported as a limitation, and a held-out nuisance calibration diagnostic on already-used data is permitted. **No new nuisance sweep is part of this run.**

## 4. Interface construction — roles and parity

Unchanged from the frozen interface:

* `H_A` keeps its four original class-probability coordinates **bitwise**; `H_B` keeps its two.
* A receives `(H_A, Z)`; B receives `H_B` only; AB receives the union.
* A's inference transform uses `T` and `H_A` and **never** `H_B`. `H_B` appears only in fitting coalition penalties and in auditing.
* Every role, eligibility mask and output dtype (`float64`) is preserved. Anchor parity is asserted on every released view of every pool.

Consequence that must not be misreported: because B receives only `H_B`, every B-view quantity — including `commute_over20` and `public_coverage` capability — is **identical across all interfaces by construction**. There is no commute-capability endpoint to win or lose. B-view numbers are reported as the structural constants they are.

## 5. The experiment matrix

Main matrix: 2 penalty families × 2 rank recipes × 3 policies × 3 seeds = **36 nominal interfaces**.

* Penalty families: `original` (historical projected linear moments) and `nonlinear` (the refinement of `METHOD.md`).
* Rank recipes: `16` and `r_plus`.
* Policies: `L1`, `L2`, `C1` (definitions in `METHOD.md` §5; identical aggregation to the historical arms).

Because §2/P1 holds — `r_plus = 16` for every seed — the `r_plus` column is an **exact alias** of the rank-16 column and is neither fitted nor audited twice. The predeclared branch therefore fires:

* **rank-8 sensitivity**, both penalty families, all three policies, three seeds = **18 nominal interfaces**, labelled **compression, not eigenvalue-sign selection**.

Total nominal interface records: **54**. Unique new fits: **27** (`nonlinear@16`, `original@8`, `nonlinear@8`, each 3 policies × 3 seeds). The nine `original@16` conditions **are** the historical `spectral_L1`/`spectral_L2`/`spectral_C1` objects and are reused, not refitted; parity is verified bitwise (§8). This grid is frozen. It will not be expanded in response to a disappointing result.

A ledger of reused objects, new fits, starts, aliases and audit units is maintained in `MATRIX.json` and `RUN_STATUS.md`. A cache read is never counted as a new fit.

## 6. Predeclared comparisons

1. `nonlinear` vs `original` at rank 16, each policy.
2. `nonlinear` vs `original` at the common reduced rank — **void by alias**; recorded as void, and the rank-8 sensitivity is reported in its place under its own label.
3. Reduced rank vs rank 16 within each penalty family and policy (i.e. 8 vs 16).
4. `C1` vs `L1` and `C1` vs `L2` within each penalty family and rank.
5. Each candidate vs frozen `J` and vs the simple interfaces `H`, `E`, `A0`.
6. Rank-8 comparisons **only** within the prescribed no-rank-change branch.

Comparisons are made at **equal realised rank**. A rank-8 channel is a 19-column A-view wire, not a zero-padded 27-column one: no zero padding is used and no claim of retained representational capacity is made for the smaller channel. Rank change is reported as a separate factor from penalty family — that two-factor structure is the point of the design.

## 7. Endpoints and decision rules

Reused **verbatim** from the completed transport study so that numbers remain comparable, and labelled as reused rather than newly justified. Definitions, code references and constants are restated in `METHOD.md` §7.

* Losses: natural-log log loss, probability floor `1e-12` with row renormalisation. Selection is always on **unweighted** validation log loss; both weightings score the identical selected predictions.
* Absolute recovery = prior loss − attack loss, same mode's prior, never clipped, negatives retained.
* **Additional (incremental) recovery** = interface absolute recovery − H's absolute recovery, matched on seed, mode, scope, budget, view, target and weighting. Reported alongside absolute recovery, never in place of it. H ancestor attack scores stay visible.
* Residence gain over H = H residence loss − interface residence loss.
* `0.01` residence reference: seed-mean gain ≥ .01 **and** every seed's gain ≥ .01.
* Original half-headroom criterion: residence loss ≤ midpoint of the PCA32 reference probe and the better of the rich/tree bank, per seed, never averaged.
* Legacy source allowance: each selected source-probe loss ≤ PCA32 reference + .01, per seed.
* Two weightings: unweighted and PWGTP person-weighted (ratio estimator, recomputed in each bootstrap replicate).
* **Advantage** of X over Y (per weighting): at least one of the four family sensitive endpoints has adjusted upper bound < 0 **and** no sensitive endpoint has adjusted lower bound > 0.
* **Primary coordination rule**: advantage over both comparators under both weightings, **and** seed-mean residence difference ≤ `.001 + 1e-12` for both comparators and both weightings, **and** `A/RAC1P` not significantly worse anywhere.
* Simultaneous 95% intervals: paired household-cluster bootstrap, 2,000 replicates, clustering on the evaluation partition's `SERIALNO`, single-step studentized max-|t| with `c = quantile(T, .95, method='higher')`, `se ≤ 1e-15` degenerate and excluded. Multiplicity corrected within family across contrasts, endpoints and both weightings jointly.

Rules that are **reused for comparability** are labelled as original point-match rules. If an equivalence or noninferiority interval is computed it is reported **separately**, with the interval-inside-margin condition implemented explicitly. The `0.001` residence condition is a point-difference rule: **a pass under it is not an equivalence result**, and the simultaneous intervals are not claimed to establish equivalence within ±.001. No old decision rule is silently replaced; any amendment is timestamped in `RUN_STATUS.md`.

No combined utility/privacy score, leakage budget or cross-target nat exchange rate is used.

## 8. Order of operations (the seal this study can still offer)

The 2017 and 2018 evaluation partitions are already exposed, so a hash lock cannot restore a confirmation. What is still meaningful is that **model selection never sees an outcome**, and that is enforced:

1. Write this protocol, `METHOD.md` and `RANK_DIAGNOSTIC.md`. Hash them into `PROTOCOL_FREEZE.json`.
2. Synthetic and parity checks (`VALIDATION.md`), including the falsification fixtures of §9. These run before any new ACS score.
3. Fit the 27 new maps. **Checkpoint selection uses the training objective only** — never attackers, residence, commute, development outcomes or the transport table. Record initial/final objectives, gradient diagnostics, restart variability, stopping reason and feasibility residuals.
4. **Freeze the complete representation slate**, then fit reserved-task utility probes and attackers.
5. 2018 development comparison on the 2018 test pool.
6. 2017 exploratory cross-year comparison: probes fitted on the old 2017 fitting partition, selected on the old 2017 validation partition, evaluated on the old 2017 final partition **as now-exposed development evaluation**. Years are reported separately; rows are never pooled into a larger training set.
7. Report, decide, publish — positive or negative.

## 9. Falsification fixtures (must pass before any new ACS score)

1. Original covariance, reconstruction-trace identity, fixed-rank eigenvalue optimum and exact `H` parity reproduced.
2. **Magnitude fixture**: `(Z,S)` equally likely `(-1,0),(1,0),(-2,1),(2,1)`; the old first moment is zero and the quadratic block detects recovery from `|Z|`.
3. **XOR fixture**: independent fair signs `S,H`, `Z = S·H`; marginal protection misses the disclosure and an `H`-interaction moment detects it.
4. **Conditional-null fixture**: `S` and `Z` correlated marginally only through `H`, independent residual randomness given `H`; with oracle nuisance conditionals the population conditional moments vanish and finite-sample moments concentrate — they are not required to equal zero exactly.
5. **Nuisance-error fixture**: deliberately misspecified `m(H)`; nonzero residual moments need not mean incremental leakage. The penalty is not presented as an infallible conditional test.
6. **Dimension fixture**: fixed-rank top eigenvectors distinguished from a variable-rank optimum, including zero rank and repeated eigenvalues.
7. Analytic-gradient finite-difference and retraction/orthogonality checks with the nonlinear term active.
8. Two repeated CPU evaluations with fixed objects and **different chunk sizes**: stable losses, transforms and predictions at stated tolerance.

These establish that the code measures the stated finite objective. They do **not** certify privacy. No fixture parameter is tuned until a broken measure looks successful.

## 10. Claim boundaries

* The global-optimality statement of the original baseline belongs to its **fixed linear-moment matrix** and is not inherited. The refined objective is generally nonconvex; its solver is iterative projected gradient on the Stiefel manifold and is at best locally optimal. The word "spectral" is not used for the new solver.
* Vanishing fitted finite moments imply neither conditional independence `Z ⊥ S | H` nor any bound on `I(S;Z|H)`.
* No calibration is inherited from KCI or RCoT: a penalty at its optimum has no p-value and no Type-I rate.
* Equal penalty scale at the reference projection is **not** equal privacy strength away from it.
* Same channel width is **not** matched utility. Same-dimensional systems are never called utility matched.
* Fresh and frozen attacks stay distinct. A negative validation-selected test increment is retained as a measurement; it does not erase information reachable by an H-only attack, and it is never read as negative information.
* Larger sample size is not asserted to be the only reason an effect resolves in one year and not another.
* The 2017 numerical corruption observed under severe memory pressure remains a **suspected** contributing condition, not a demonstrated root cause.

## 11. Resource discipline

Single worker and one BLAS/OpenMP thread to start; peak RSS, memory pressure, swap and per-unit runtime measured; parallelism raised only if free memory actually permits. Resumable per-condition outputs, written atomically. Deterministic CPU float64 for matrix algebra, eigensolutions, finite moments and verification; a CPU reference is established for prediction paths. Every new prediction is computed twice and must agree; a surviving mismatch quarantines that unit, retains both versions and logs, and invalidates the unit rather than being averaged or majority-voted into a report. Unrelated user workloads are never stopped or reconfigured.
