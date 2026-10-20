# FA_PRC_v1: propensity-routed finite cells (family A, Designer A)

Campaign `pcrl_new_release_mechanisms_v1`. This is exploratory development on previously used public-use ACS 2019 1-Year PUMS
Florida records, not confirmation.

- **Governs:** every fit of family FA.
- **Code:** `nrm/fa/prc.py`, with the deployment interface in `nrm/fa/release.py`.
  - Commits: 47916952c (wave 1), 7308e45fa (stages), 300d017a6 (card decisions V1).
  - Invented-data selftest: 25/25 (`roles/designer_a/SELFTEST_WAVE3.json`).
- **Sources:**
  - proposal: `roles/designer_a/PROPOSAL_WAVE1.md`;
  - changes from `roles/coordinator/CARD_DECISIONS_V1.md` and `roles/challenger/REVIEW_PROPOSALS.md`.
- **Readers:** "Recovery" means the declared readers' AUC for the recorded SEX column from the released vectors. It is never
  protection against every reader.

## 1. Operation (raw fields -> income vector, occupation vector; decisions = argmax)

**Coding.** Z = `ohn.fam_a.compact_po(X)`, the A1_PO_16 coding of the admitted 145 columns.

**Statistics of x.** All are fitted on MODEL_FIT and computed from X alone at runtime. The learner settings are A1_PO_16's: sklearn
HGB, lr 0.025, 16 leaves, min leaf 100, l2 3, COW and MAR as native categories. T* is the first argmin of the pooled inner-CV NLL
over household folds `OHN_A_cv_S<v>`, with T* ≤ 1000. Deployed models are full-pool refits at T*.

| Statistic | What it is | Label | Used by |
|---|---|---|---|
| qI(x) | income posterior | income | all |
| qO(x) | occupation posterior (the A1_PO_16 recipe) | occupation | all |
| π(x) = P(SEX = 1 \| x) | SEX propensity | SEX | protected configurations only |
| qO0(x), qO1(x) | occupation posterior given [Z, s], evaluated at s = 0 and 1 | occupation, SEX | guard term only |

**Rule.** (k, l) = argmin over income cell k and occupation cell l of

- CE(qI, cI_k) + CE(qO, cO_l)
- \+ t · [ D(π; rP_kl) + D(π; rI_k) + D(π; rO_l) ]
- \+ t · κ · [ (1 − π)/p0 · |qO0 − cO_l|² + π/p1 · |qO1 − cO_l|² ]

where D(π; r) = π log(r/r̄) + (1 − π) log((1 − r)/(1 − r̄)): the person's expected contribution to the plug-in I(SEX; cell), that is,
to the log-likelihood gain of the Bayes LOOKUP reader. The pair weight and the local weights are both 1.

The fitted constants are:
- codewords cI (16 × 2) and cO (64 × 5);
- cell SEX-rate tables rI, rO and rP, shrunk with 20 pseudo-counts (rP toward an additive-logit fit);
- the rule's own tilt t ≤ λ;
- κ = 1.

**Output.**
- income = dI_k, occupation = dO_l. These are label-refit decoders: MODEL_FIT label counts of the out-of-fold-routed people in the
  cell, shrunk toward the codeword with 20 pseudo-counts.
- No calibrator is applied, and CALIBRATION is not read.
- Deterministic: no randomness, keys or draws.

**Disclosure.** The released vectors identify the cell.
- The linked pair sees the joint value (k, l). Each recipient sees only its own cell.
- The dependence objective and every check are defined on exactly this recoverable identity. No other field is transmitted.

**Fitting the rule (MODEL_FIT out-of-fold embeddings, 5 household folds `NRM_FA_xf_S<v>`).**
- **Twin:** KL k-means++ (seed v) and Lloyd steps at λ = 0.
- **Protected:** a deterministic continuation up the λ grid {0.25, 0.5, 1, 2, 4, 8}, warm-started at the twin. At each λ:
  - Lloyd steps refresh the codewords (cell means) and the tables (shrunk cell means of π);
  - they then try tilts λ, λ/2, …, λ/64 and accept the first whose partition lowers the exact objective: mean CE + λ·[I_π(pair) +
    I_π(income) + I_π(occupation)] + λκ·guard;
  - every accepted state is rule-consistent (the deployed rule reproduces the fitted partition), and the result is never worse than
    the twin at the same λ.

## 2. Closest prior attempt; what differs; bottleneck addressed

**Closest prior attempts.**
- SCM (`results/pcrl_scm_engineering_benchmark_v1/DECISION.md`, `results/pcrl_scm_utility_matched_v1/RESEARCH_DECISION.md`, code
  `scm_bench/engine.py`): a privacy funnel / IBSI quantizer of a task-only teacher, h(q(x)).
- The Adult JOINT finite codes (decision_preserving_compression, confidence_capacity, confidence_budgeted_privacy).

**One real new operation.** The partition is g(q, π, qO0, qO1), not h(q): people with equal task posteriors can be split by their
own SEX propensity.

- A cell's NLL depends on X only through E[q | cell].
- The Bayes cell reader depends only on E[π | cell].
- The per-SEX guard needs π, qO0 and qO1.
- So e(x) is sufficient for the objective, and h(q) cannot reach partitions that separate equal-q people by π.

**Knobs and procedures, not mechanisms** (challenger's classification, accepted):
- Lloyd boundary moves instead of greedy merges;
- the guard as a penalty;
- the A1_PO_16 teacher, which is shared with the twin and ORD;
- refit decoders;
- nested selection.

**Bottlenecks addressed.**
- SCM's fitted spend did not carry over: it spent +0.0070 / +0.0017 nats more on DEV than the matched ORD.
- The occupation NLL clause bound (0.0106 versus Ucal on DEV).
- The men's guard bound (0.0054 / 0.0064).

## 3. Why it could improve protection without repeating the observed utility cost

- **Headroom.** A1_PO_16's occupation NLL is −0.0161 / −0.0128 versus U0, and its men's Brier is −0.0039 / −0.0023. That is room
  SCM's teacher lacked.
- **Boundary cost.** At a task-optimal partition, re-routing near-boundary people costs second-order task loss but changes the cell
  tables at first order. h(q) merges cannot do this for people with equal posteriors.
- **Earlier evidence.** In the SCM study, SEX-blind coarsening stayed flat on the MODEL_SELECT proxy (0.645–0.649 up to 0.0145
  nats), while SEX-aware cells reached about 0.61. The open issue was honest spend measurement.
- **Not novel.** This is the deterministic privacy funnel / IBSI family. FARE is the published finite-cell input-routing method.
- **Prediction, before any real fit:**
  - PRC_S2's nested held-out pair LOOKUP AUC is below the twin's by 0.010–0.030;
  - P(PRC beats ORD at equal measured MODEL_SELECT spend, both versions) ≈ 0.5;
  - P(HQ within 0.005 of PRC) ≈ 0.55;
  - P(0.020 target against NEURAL_RAW and H_LJ) ≈ 0.2.
- **Invented-data warning (selftest, not evidence):** at λ = 2, SHUF lowered the cross-fitted pair AUC more than PRC did (0.671
  against 0.712; twin 0.719), presumably by spending utility.

## 4. Task-only twin

PRC_TWIN: λ = 0.
- Two independent KL k-means quantizers (16 income, 64 occupation cells) of the same task posteriors, with the same label-refit
  decoders.
- π and qOs are never fitted. SEX is never read: the twin units load income and occupation only, and the selftest shows bitwise
  identical releases with and without SEX.

Internal capacity gate (stage 1b, no SEX):
- the twin's nested held-out spend over the continuous task teachers (estimate + 2 household-clustered SE) must be ≤ 0.004 nats NLL
  per task and ≤ 0.003 accuracy loss per task;
- otherwise R0 applies (capacity ladder (24, 128), +2 configurations).

The coordinator then evaluates the twin on MODEL_SELECT (24 clauses + 4 guards) before any protected fit.

## 5. Simplest controls and the published objective

- **Twin:** PRC_TWIN. PRC versus TWIN at the same cells is the clean contrast.
- **ORD:** the G3 SEX-blind control path. The twin's quantizers at coarser codebooks: ladder (12,48), (8,32), (6,24), (4,16),
  (3,12), (2,8).
  - ORD_S1–S3 take the most coarsened entry that meets each budget.
  - The release never uses SEX. Its operating point is chosen by the stage-2 rule, whose subgroup rows read MODEL_FIT SEX only as
    an evaluation of SEX-blind candidates.
- **SHUF_S2:** the SEX-blind control at the same 16 × 64 cells and the same optimizer, λ grid and tilts. π, qO0 and qO1 are
  replaced by a donor copy.
  - Donor of a row = SHA-256(`NRM_FA_shuf_S<v>` | the row's compact features) mod 20,000 into the MODEL_FIT out-of-fold bank. The
    objective, tables and acceptance all use the copy.
  - **Deviation from "household-permuted copy":** a household permutation is not a function of the six fields, so `predict` could
    not reproduce it. The keyed-hash donor copy is its deployable equivalent.
  - On MODEL_FIT the self-donor and same-household-donor counts are recorded in SELECTION_V<v>.json (expected about 1 to 2 of
    20,000).
  - The bank (20,000 × 11 out-of-fold statistics) is stored in SHUF's private model file only.
- **HQ_S2:** post-hoc SEX-aware grouping on q only, the closed SCM/IBSI line and the relevant published objective.
  - Same optimizer and grid, with π, qO0 and qO1 replaced by their MODEL_FIT means within the twin's joint task cells. The routing is
    then a function of the task posteriors.
- **LOC_S2:** dropped. No joint-design claim is made and G6 is not assessed.

## 6. Falsifying outcome

The rationale (SEX-propensity routing buys protection at lower utility cost than SEX-blind alternatives) is falsified if any of the
following holds.

1. **ORD.** On MODEL_SELECT (`nrm.evaluate` spend, `nrm.screen` and grid readers), the PRC path does not lie below the ORD path in
   linked-pair recovery at equal or lower measured occupation-NLL and men's-Brier spend.
   - Read over the matched points and both versions.
   - A PRC-vs-ORD advantage is credited only if it also holds for smooth readers and for the MODEL_FIT cross-fitted LOOKUP.
2. **SHUF.** SHUF_S2 reaches PRC_S2's recovery at equal or lower measured spend. The gain would then be untargeted re-routing.
3. **HQ.** HQ_S2 reaches PRC_S2's recovery at equal spend. Then π-routing adds nothing and PRC is "SCM with a better teacher".
4. **MODEL_FIT gate, before any MODEL_SELECT read of a protected release.** PRC_S2's nested held-out pair LOOKUP AUC is not below
   the twin's by more than 2 SE.

**Reader targeting.** The objective is the π-weighted plug-in LOOKUP statistic itself, so an advantage seen only by LOOKUP-type
readers is labelled reader-targeted.

**Verdicts** use spend measured on MODEL_SELECT by `nrm.evaluate`, never the nested MODEL_FIT estimates.

## 7. Fitting plan

**Roles and labels.**
- MODEL_FIT (20,000 people, 13,337 households), operation `fit`, this card.
- Stage 1 and the capacity gate read income and occupation.
- Stage 2 reads income, occupation and SEX; SEX is a training label of π and qOs.
- Stage 3 reads income and occupation; SEX reaches PRC, HQ and SHUF only through the cached stage-2 statistics.
- Check 1 reads SEX as an evaluation label.
- CALIBRATION is not read. MODEL_SELECT is used only through the coordinator's `nrm.evaluate`, `nrm.screen` and grid outputs; it is
  never a fitting input.

**Full pool and folds.**
- Deployed teachers are full-pool refits at T*: 20,000 people / 13,337 households.
- Out-of-fold embeddings: 5 household folds, about 16,000 people per fold model.
- T* is not re-selected inside folds (non-nested; disclosed).
- Routing and decoders use all 20,000 out-of-fold-routed people.
- Nested estimator: 5 outer household folds `NRM_FA_ncv_S<v>`, refitting the whole pipeline at the full-pool T* on about 16,000
  people. Each outer-training part has its own 5 out-of-fold folds.

**Seeds per version.**
- Version v uses inner folds `OHN_A_cv_S<v>`, cross-fitting folds `NRM_FA_xf_S<v>`, outer folds `NRM_FA_ncv_S<v>`, the k-means++
  seed SeedSequence([v, KI, KO]), HGB random_state v and the SHUF key `NRM_FA_shuf_S<v>`.

**Strength path.** Budgets are spend over the twin; the rule is nested estimate + 2 clustered SE ≤ budget on every metric.

| Budget | occ NLL | inc NLL | men's / women's occ Brier | occ acc loss | inc acc loss |
|---|---|---|---|---|---|
| S1 | 0.004 | 0.002 | 0.0015 | 0.003 | 0.002 |
| S2 | 0.008 | 0.004 | 0.0030 | 0.006 | 0.004 |
| S3 | 0.012 | 0.006 | 0.0045 | 0.008 | 0.006 |

- PRC: the largest grid λ such that it and every smaller λ meet the budget.
- ORD: the most coarsened ladder entry meeting it.
- HQ and SHUF: at S2 only.
- A budget no candidate meets is NOT_FEASIBLE and is not fitted.
- A budget whose selection equals an earlier budget's is recorded as an alias and never saved twice.

**Fit order and queue units.** All on one thread, through the shared queue. Unit ids are append-only; a retry gets a new suffix.

| Step | Unit | Command |
|---|---|---|
| 1 | FA_TWIN_V0_V1 | `fit --config PRC_TWIN --version 0` |
| 1 | FA_TWIN_V1_V1 | `fit --config PRC_TWIN --version 1` |
| 1b | FA_GATE_V0_V1, FA_GATE_V1_V1 | `nested --phase gate --version v` |
| 2 | — | Message the coordinator; the coordinator runs `nrm.evaluate` on the twins. |
| 3 | FA_PATHS_V0_V1, FA_PATHS_V1_V1 | `nested --phase paths --version v` |
| 4 | FA_CFG_V0_V1, FA_CFG_V1_V1 | `fit --config PRC_S1,PRC_S2,PRC_S3,ORD_S1,ORD_S2,ORD_S3,HQ_S2,SHUF_S2 --version v` |
| 5 | FA_CHECK_V0_V1, FA_CHECK_V1_V1 | `check --version v` |

- Step 3 runs only after the coordinator's OK.
- Every command is `python3 -m nrm.fa.prc … --card results/pcrl_new_release_mechanisms_v1/cards/FA_PRC_v1.md`.

**Required reporting.**
- **Per release:** people per released vector and per joint value for every role, in the record's `reader_budget` (label-free).
- **Fitted versus held-out divergence per strength:** the nested in-sample pair LOOKUP AUC minus the held-out AUC, and the fitted π-MI
  (SELECTION_V<v>.json).
- **Check 1:** the MODEL_FIT cross-fitted LOOKUP for every saved release (CHECK1_V<v>_*.json).

## 8. Readers needed

- **LOOKUP** frequency tables on the observed vector values, with shrinkage:
  - the pair key is the joint value of the two vectors, up to 1,024;
  - the income and occupation views are read separately.
- **Smooth readers** on the vectors, which pool across cells.
- **The MODEL_FIT cross-fitted LOOKUP** (check 1, 20,000 people) beside the grid.
- **Reader budget:** people per value is reported for every release. PRC versus TWIN and SHUF are matched on joint-value count; ORD
  is read more densely.

## 9. Budget

**Configurations: 18.** PRC_TWIN, PRC_S1, PRC_S2, PRC_S3, ORD_S1, ORD_S2, ORD_S3, HQ_S2 and SHUF_S2, each in both versions.

Contingent, each only with recorded evidence:
- R0 twin ladder (24, 128): +2;
- one ORD bracketing entry, chosen by measured MODEL_SELECT spend only, if `nrm.ctl.frontier` returns EXTEND_CONTROL or
  AHEAD_VS_SPARSE_PATH: +2.

**CPU estimate (one thread):**

| Step | Each | Units |
|---|---|---|
| Twin | about 5 min | 2 |
| Gate | about 6 min | 2 |
| Paths: SEX-side T* fits plus 5 outer folds × (4 teachers × 6 refits + 3 chains × 6 λ + 6 ORD) | about 40 min | 2 |
| Configurations | about 5 min | 2 |
| Check 1 | about 1 min | 2 |

- Total about 2 CPU-h; I request ≤ 2.5 CPU-h.
- Out-of-queue agent work so far: about 0.25 CPU-h, all charged, including a disclosed 740 CPU-s multi-threaded selftest.

## 10. Revision plan

- **R0, capacity.** The twin fails the internal gate, or the coordinator's MODEL_SELECT evaluation shows no utility margin → twin
  ladder (24, 128) under new configuration ids, +2.
- **R1, income.** The income HGB fails income clauses or keeps no margin → a stronger task-only income statistic, used identically by
  twin, PRC, ORD, HQ and SHUF, under a new card.
- **R2, optimizer.** The fitted π objective falls but the held-out label LOOKUP does not → diagnose π calibration and table shrinkage
  first. The only remedy is a stronger search under the same objective (exact single-person moves with rule refit), not a new λ grid.
- **R3, guard.** The men's guard binds first along the nested path → κ = 4, once, under new ids.
- **Stop.** Falsifying outcome 4 at the MODEL_FIT stage, or outcome 1, 2 or 3 on MODEL_SELECT → the family is closed as
  "propensity routing adds nothing measurable" and no further PRC configurations are requested. Any successor needs its own card.
