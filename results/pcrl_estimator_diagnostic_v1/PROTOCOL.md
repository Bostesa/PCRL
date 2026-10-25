# EDG1 protocol: synthetic diagnostic of PCRL's recovery estimators

Campaign `pcrl_estimator_diagnostic_v1` (EDG1). Registered before any simulation outcome; amendments are append-only (section 9).

## 1. Purpose and scope

EDG1 asks whether the recovery measurements used to guide DRF1's method development separate real recovery differences from
estimation bias and reader limitations. It is a measurement diagnostic, not a release method. It searches no protection model.

- **Source.** The published DRF1 decision and code at `publication/pcrl-direct-recovery-frontier-v1`, final commit c4896479e
  (`publications/pcrl-direct-recovery-frontier-v1/`; this worktree is branched from that commit).
  - Exact AUC: `drf/finite_law.py` and `drf/solver.py` (`auc_star`).
  - Nested estimator: `drf/dr.py` `nested`.
  - Reader grid: `drf/devreaders.py`.
- **Code not in the publication.** The nested estimator's helpers `lookup_scores` and `auc` live in `nrm/fa/prc.py`. The 16
  reader recipes live in `scm_bench/readers.py`, `task_conditioned_representation/{attacks,inference}.py` and
  `ffv/{readers,features}.py`. EDG1 copies them from the DRF1 research tip 913eb5c91.
  - The five lock-listed recipe files have the SHA-256 recorded in the publication's `DEVGRID_CODE_LOCK.json`.
  - The copied prc functions are text-equal to the source (`SOURCE_PROVENANCE.json`, `edg.selftest`).
- **Data.** Only synthetic data from fully specified laws, plus published aggregate counts:
  - MODEL_FIT 20,000 people in 13,337 households;
  - nested held-out folds of about 4,000;
  - MODEL_SELECT 8,001 people (FIT 50%, VALIDATION 20%, SCORE 30% of households);
  - income 16 atoms, occupation 64 atoms, pair up to 16 x 64.
- **Not read.** No census record, label, model, private array, row-level release or cache. No MODEL_FIT, MODEL_SELECT,
  CALIBRATION, ATTACK, DEV_REPORT, sealed or capped-out data.
- **Unchanged.** Every DRF1 release, claim label and protocol.
- **Motivation, not a target.** DRF1's twin had nested held-out exact pair AUC* 0.7475 against held-out LOOKUP 0.6303 (V0).
  EDG1 does not assume that difference is all bias, and does not tune laws to reproduce it.

## 2. Quantities (computed and labelled separately)

All AUCs score ties as 1/2. An AUC whose scoring sample lacks a class is UNDEFINED. It is recorded and counted, never set to 1/2,
and the sample is never redrawn.

- **A, population optimal-reader AUC.** The likelihood-ratio ranking of the known finite law over merged observable wires
  (`drf.finite_law.auc_star`).
- **B_indep.** Sample AUC of the known population-optimal score (true posterior ranking) on an independently generated test sample
  of the same size.
  - Diagnostic **B_same**: the same score on the study sample. Since C_same is the maximum sample AUC over all rankings,
    `C_same - B_same >= 0` is pure ranking-selection optimism.
- **C, empirical optimal AUC.** Cell posteriors estimated on a sample, ranked, and evaluated on the same sample. This is
  `drf.solver.auc_star` on the sample's per-wire SEX masses, as in DRF1.
  - **C_same**: on the whole study sample (DRF1's in-sample analogue at n = 20,000).
  - **C_nested**: DRF1's nested held-out exact AUC* (`heldout_exact_auc_mean`) for a fixed release. The sample is split into 5
    random folds (households in the clustered cell); each fold's own empirical optimal AUC; people-weighted mean.
- **D, independently evaluated LOOKUP.**
  - **D_nested**: DRF1's nested held-out LOOKUP. Five-fold cross-fit of `lookup_scores` with Beta(a·rbar, a·(1 − rbar))
    shrinkage, a = 1 (`drf.dr.LOOKUP_A`). An unseen wire gets the training rate rbar. AUC is pooled over all people (`prc.auc`).
  - **D_bank**: the bank's LOOKUP recipe. Keys are exact float64 bytes of the emitted vectors; pseudo-count 2 toward the FIT rate;
    an unseen key gets the FIT rate. Fitted on FIT, oriented on VALIDATION, scored once on SCORE.
  - For each fitted LOOKUP ranking, its exact population AUC under the known law:
    - D_nested: per fold, plus a pooled analogue (law over wire × fold, with fold shares);
    - D_bank: its oriented table.
    - Estimation loss = A − pop. Test-sample variation = sample value − pop.
- **E, the declared reader bank.** The 16 DRF1 recipes, fitted on FIT, oriented and selected on VALIDATION, scored once on SCORE,
  on the bounded subset of section 5.

Interpretation rules:
- Empirical optimal AUC is never called a population guarantee.
- A separately fitted reader's lower AUC is never treated as proof that less information exists.
- No test label chooses a reader, smoothing parameter, orientation or tie rule. Orientation and selection use VALIDATION only;
  LOOKUP smoothing is the source's fixed value.

## 3. Known-law families (frozen; `edg/laws.py`, values in `LAWS.json`)

Every law is a finite joint over internal cells c with P(C = c) = p[c] and P(S = 1 | c) = q[c]. Each view is the deterministic
emitted wire of c:
- income-like: a 2-probability vector with argmax decision;
- occupation-like: a 5-probability vector;
- linked pair: both.

Identical emitted wires are merged (`merge_wires`) before any recovery computation; internal ids are never extra inputs.

**Frequencies and occupancy.**
- Class frequency pi ∈ {0.5, 0.2}.
- Occupancy is uniform, or skewed:
  - Zipf weights 1/k over a fixed permutation of the atoms;
  - skewed child shares (0.55, 0.25, 0.13, 0.07), in a fixed per-parent permuted order.
- The parameters below were fixed from population values only, before any estimator was run on a grid seed.

**Moderate signal (REF and RED).** 16 coarse atoms g with q_g = sigmoid(alpha + beta z_g):
- beta = 1;
- z on 8 equally spaced levels in [−1, 1], each shared by two atoms, which gives TIED likelihood ratios with distinct wires;
- alpha is solved so that the marginal equals pi exactly.

| family | views (observable support) | population relation (verified in `LAWS.json`, tolerance 1e-12) |
|---|---|---|
| IND | income 16, occupation 64, pair 1024; S independent of both | A = 1/2 for every view and every fitted score |
| REF | one occupation-like view at levels L16, L64, L256, L1024 of the same people; each atom split into 4 children independently of S given the parent | every child inherits its parent's posterior, so A is identical at every level (0.6708–0.6814) |
| RED | income = g (16); occupation = (g // 4, z), z 16 noise levels independent of (S, g) (64); pair (256) | S ⊥ occupation given income, so A(pair) = A(income) (0.642–0.677) > A(occupation) |
| COMP | income i (16), occupation j (16 internal, 15 observable), product occupancy a_i b_j; q = pi + delta u_i v_j with weighted-centred u, v, delta = 0.24 sqrt(pi(1 − pi)) | A(income) = A(occupation) = 1/2 < A(pair) = 0.563–0.604 |

- **COMP merging check.** Occupation cells 14 and 15 emit the identical vector, giving pair support 240. Scoring unmerged internal
  ids would overstate A(pair).
- **Emitted vectors.** Seeded and deterministic per law; all valid probability vectors.
  - Income: (1 − sigmoid(tau), sigmoid(tau)).
  - Occupation: softmax of 5 coordinates.
  - For the signal families the first coordinate tracks the logit posterior, plus noise.
  - Refinement children perturb their parent's coordinates by N(0, 0.15²).
- **Comparisons.** Each family's comparisons are written finer-or-linked minus coarser-or-single where that applies:
  - IND: pair − income, pair − occupation, occupation − income (all dA = 0);
  - REF: L64 − L16, L256 − L16, L1024 − L16 (dA = 0);
  - RED: pair − income (dA = 0), income − occupation (dA > 0), pair − occupation (dA > 0);
  - COMP: pair − income (dA > 0), pair − occupation (dA > 0), occupation − income (dA = 0).

## 4. Simulation grid (frozen; `edg/simulate.py`, `GRID.json`)

- **Primary cells.** 4 families × 2 pi × 2 occupancies × n ∈ {2,000, 8,000, 20,000} = 48 cells of independent people.
  - n = 20,000 matches MODEL_FIT; its nested folds have about 4,000 people.
  - n = 8,000 matches MODEL_SELECT's reader split; n = 2,000 is sparse.
- **Clustered sensitivity cell (cell 48).** REF, pi = 0.5, skewed, about 20,000 people in explicit synthetic households:
  - household sizes 1–4 with probabilities 0.62 / 0.27 / 0.08 / 0.03 (mean 1.52 against DRF1's 1.50);
  - the second member has the opposite S with probability 0.8;
  - g | S and each refinement level are drawn by Gaussian copulas with household latents, rho = 0.5.
  - The person-level marginal law equals the independent REF law exactly. The truth is that person-weighted marginal AUC.
  - Whole households are split: folds and FIT / VALIDATION / SCORE are drawn by household.
  - Its matched independent cell is REF | 0.5 | skewed | 20,000.
- **Repetitions.** 500 per cell (prospective minimum 200 if compute ran short).
- **Seeds and samples.** Repetition seed `SeedSequence(20261025, spawn_key=(cell, rep))`. Each repetition draws:
  - a study sample;
  - an independent test sample (for B_indep);
  - 5 nested folds;
  - an independent 10-part split: parts 0–4 FIT, 5–6 VALIDATION, 7–9 SCORE.
  - All views of a repetition use the same people (paired coarse-versus-refined comparisons). Training, validation and test
    samples never overlap.
- **Source checks.**
  - D_bank's vectorized table is checked bitwise against `scm_bench.readers.fit_lookup / predict_lookup` on repetition 0 of every
    cell.
  - C_same's influence AUC is checked against `drf.solver.auc_star`, and D_nested's against `prc.auc`, on every repetition.
- **Pre-registration smoke tests.** These used repetition indices 900000+ (disjoint from the grid) and printed timing and shapes
  only.

## 5. Reader-bank subset (estimator E; `edg/bank.py`)

Six scenario cells at n = 8,000 (FIT 4,000 / VALIDATION 1,600 / SCORE 2,400), using the same people and split as the grid's D_bank:
1. IND | 0.5 | skewed: income, occupation, pair;
2. REF | 0.5 | skewed: L16;
3. REF | 0.5 | skewed: L256 (same people as cell 2: paired);
4. RED | 0.5 | skewed: income, occupation, pair;
5. COMP | 0.5 | uniform: income, occupation, pair;
6. RED | 0.2 | skewed: income, occupation, pair.

**Recipes and selection.**
- The 16 recipes and DRF1's selection rule:
  - maximum oriented VALIDATION AUC, exact ties to the earlier candidate;
  - pair candidates are the pair readers, then ignore-occupation and ignore-income readers, source recipes first.
- Reader seeds are MODEL_SEED_BASE + 7100 + rep.
- Support wires are appended as rows that enter no fit, orientation or selection. This gives each fitted recipe's exact population
  AUC.

**Decomposition.**

| component | definition |
|---|---|
| finite training + model class | A − best pop in bank |
| validation selection loss | best pop − winner pop |
| winner's curse | winner VALIDATION − winner pop |
| test-sample variation | winner SCORE − winner pop |

- **Class limits.** The LR-type recipes (OUT_LR, LOG_LR, DEC_LR, CATI01, CATI1) and LOOKUP are refitted once per bank view on
  30,000 separate people, oriented on 10,000 more. This separates class limitation from finite training for those recipes.
- **Not separated.** For trees and MLPs, the class component is not separated (cost).
- **Repetitions (declared rule).** A timing-only microbenchmark on a separate law and seed (`BENCH.json`) projects the CPU of one
  repetition of the whole subset. Then reps = min(10, max(3, floor(1500 CPU-s / projected))).
- **Coverage.** No benchmark AUC is computed. The subset is not expanded after results, and actual coverage is reported.

## 6. Analysis plan (frozen `edg/analyze.py`)

**Per cell × view × estimator.**
- Mean, replicate SD and MCSE (SD / sqrt R).
- Bias and RMSE against A; for D also against the fitted reader's population AUC.
- Undefined count.
- `C_same − B_same`; LOOKUP estimation loss and test variation; unseen-cell shares.

**Per comparison × estimator.**
- dA; mean, SD and MCSE of the estimated difference d; P(d > 0), P(d < 0), P(d = 0).
- **Declared directional rule R2SE.**
  - 'X > Y' iff d − 2·se > 0; 'X < Y' iff d + 2·se < 0.
  - se is the linearized paired SE with the scored rankings held fixed: DeLong placement values with ties 1/2, aggregated by unit,
    G / (G − 1) factor. In the clustered cell the unit is the household; the person-unit SE is also reported.
  - For C, holding the ranking fixed ignores its same-sample selection; that is the naive practice under test.
- **Classification.**
  - dA = 0: any resolved conclusion is a false positive, reported in total and by direction. A point estimate's sign is only
    descriptive.
  - dA ≠ 0: a resolved conclusion of the opposite sign is a wrong directional conclusion (power is reported too); a descriptive
    sign error is sign(d) ≠ sign(dA).
  - Secondary margin rule M20: |d| ≥ 0.020 (DRF1's D4 target scale) when dA = 0.
- **Sensitivity.** Class frequency, sample size, occupancy and support.
- **Clustered cell.** Compared with its matched independent cell, separating clustered sampling variance from estimator bias.
- **Outputs.** Two figures: bias against support; and the decomposition of the nested-exact-to-LOOKUP gap at n = 20,000.

## 7. Interpretation limits (fixed in advance)

- Synthetic findings describe these specified laws and sizes only.
- They do not quantify the bias in DRF1's actual results, whose population law is unknown.
- DRF1's no-method conclusion stands unless an actual numerical error in its recorded decision is independently established.
- Its higher task cost, unmatched comparisons and failed strong-reference conditions remain separate evidence.
- DRF1's objective used the pi-estimated law (propensity weights), not the true-label empirical law. EDG1 does not simulate that
  estimator; it is a stated gap.
- No unbiased estimator or universal guarantee is proposed.

## 8. Roles, budget, integrity

| item | limit |
|---|---|
| live agents | at most 3, including the coordinator (implementation): an independent verifier and a statistical reviewer; no recursive spawning |
| elapsed time | 2 hours (start 2026-10-25T16:55:33Z) |
| CPU | 2 CPU-hours |
| paid compute | none |
| heavy processes | at most 2, one numerical thread each |
| interpreter | the PCRL venv (numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0 = DRF1's lock; matplotlib 3.10.8) |

- **Untouched.** main, manuscripts, earlier studies and public history. Nothing is force-pushed; SAR1's receipt is not rewritten.
- **Corrections.** If code is corrected after results, the original run is preserved and the recomputed results are named.

## 9. Amendments (append-only)

(none at registration)

- **A1 (2026-10-25, about 17:22Z; before any outcome was viewed).** Added `code/edg/custody.py`, which backs up the branch and
  checks a restore from the bundle alone, and its `custody` dispatch in `edg_run.py`. No registered command, estimator, grid, seed
  or analysis changed. `edg_run.py`'s SHA-256 therefore differs from the GRID.json code lock; every other locked file is unchanged.
- **Bank repetitions.** BENCH.json (17:18:50Z, timing only) projected 25.4 CPU-s per subset repetition. The declared rule therefore
  gives 10 repetitions per bank cell.
- **A2 (about 17:23Z; before any outcome was viewed).** Added `code/edg/publish.py` and its `publish` dispatch in `edg_run.py`. It
  builds a sanitized package with a private, counts-only scan for owner identifiers. No registered element changed.
- **A3 (17:39Z; after outcomes; tooling only).** Custody rehearsal 1 failed (`custody/RESTORE_REHEARSAL_1_FAILED.json`, kept).
  - Selftest defect: in a clone restored from the bundle, the source-text check indexed the unavailable DRF1 blob. It now skips that
    comparison and records it as unavailable.
  - Custody comparison: SUMMARY.json is now compared by NaN-aware value equality. Its byte order differs between runs only in the
    order of bank 'winners' entries with tied counts, which follows Python's string hash seed. TABLES.md was byte-equal.
  - No estimator, analysis value or output was recomputed or changed.
- **A4 (17:45Z; after outcomes; POST HOC).** Added `code/edg/posthoc.py` and `POST_HOC.json`, requested by the statistical
  reviewer's design review (`roles/statistical_reviewer/REVIEW_DESIGN.md`). They contain:
  - integrity counts;
  - D_bank R2SE judged against its own fitted-reader target;
  - B coverage in the non-null comparisons;
  - the cross-fit term per Figure-2 view;
  - the directional 0.020 margin;
  - bank MCSEs, the class / finite-training split, paired bank differences, and a bank-LOOKUP versus grid-D_bank equality check;
  - a DRF1-type agreement analogue.

  These are post hoc and labelled so in the report. The registered analysis, SUMMARY.json and TABLES.md are unchanged.
