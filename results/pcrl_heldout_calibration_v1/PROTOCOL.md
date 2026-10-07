# Protocol — hcal (held-out, shared calibration of frozen releases)

Study `pcrl_heldout_calibration_v1`, branch `research/pcrl-heldout-calibration-v1`, package `hcal/`, tests
`tests/pcrl_heldout_calibration_v1/`, results here. It starts from `research/pcrl-adult-learned-decoder-release-v1` at
976202546a100f2b89a863cb3785c3465cf76c6a. The lra evidence commit 1dee332 resolves to
1dee33253af03bb5efcaa652774cb222df642aa3. Start: 2026-10-07T12:22:22Z (worktree creation). Private store:
`<PRIVATE_CACHE>/hcal_v1`, named by the environment variable `PCRL_HCAL_PRIVATE_CACHE`.

This file is the integrated protocol. Only role A edits it. Each named lock binds its bytes. Sections marked
**[frozen at SCIENCE_LOCK]** may be refined before that lock and never after it.

## 0. Questions

1. **Fitting role.** Does fitting on held-out rows improve held-out confidence? The comparison is matched-sample with a
   fixed prior: H-TOKEN32 against T-TOKEN32, same objective, same 2,000 observations, same prior, same κ.
2. **Parameter sharing.** Does sharing a few parameters (H-GLOBAL-TEMP) beat fitting a probability vector for every
   token (H-TOKEN32), on the same held-out sample?
3. **Release criterion.** With the controls calibrated equally, does any existing privacy-trained release meet the
   original privacy/utility criterion, and stay within the same utility allowances of calibrated continuous U?

This is a controlled follow-up, not another λ, κ, capacity, neighbourhood or critic-schedule search. The registered
design runs whatever the forecasts (PREDICTIONS.json). Favourable performance is never a correctness gate. Temperature
scaling (Guo, Pleiss, Sun and Weinberger, ICML 2017) is established prior work: it is fitted on separate
validation/calibration data and preserves predictions for positive temperature. A success here is a pipeline
improvement, not a new calibration algorithm and not a privacy mechanism.

## 1. Registered exposure statement (verbatim, prompt §3)

"This study is motivated by observed outcomes on repeatedly used Adult development data. Frozen maps, teachers and
earlier decoder outcomes are known. New calibration roles, fitting rules and selection rules are chosen after those
outcomes. All new assessment results are exploratory development evidence, not fresh confirmation. Their intervals do
not account for the full history of research selection. Assessment masking controls this execution's selection process;
it cannot undo previous exposure. No new population is opened."

**Threat model.**
- Each code recipient receives its token identity, its public probability vector and its predicted class. The
  coalition receives both aligned interfaces. Both recipients and the coalition are audited, and the coalition may
  ignore either recipient.
- A deterministic decoder change that keeps the token identity changes no information. Public alternative decoder
  tables can be evaluated from the same token. A finite attacker scoring lower after a decoder change is not
  information removal.
- Engineering properties: frozen assignments, deterministic decoding, schema enforcement, verified decision
  preservation.
- Privacy means recovery measured by the declared attacker slate. There is no population-MI, Bayes-optimal,
  training-data, repeated-query or unseen-attacker guarantee.
- The historical R²-to-accuracy guarantee was refuted and is never used.

## 2. Roles and the new calibration split (hcal/data.py; ROLE_MANIFEST.json)

**Inherited roles (unchanged).** These come from lra.data.load → qpc → dpc → osf, with their pinned blob and role
checks.

| Role | Rows | Groups |
|---|---|---|
| OSF_DEFENSE_FIT | 15,434 | 15,428 |
| HEAD_VALIDATION | 1,500 | 1,499 |
| AUDIT_FIT | 6,065 | 6,061 |
| INNER_SELECTION | 2,235 | 2,234 |
| OSF_DEVELOPMENT_ASSESSMENT | 13,936 | 13,929 |

- The original exclusions, normalisation, vocabulary, row order and the 83 permitted columns are retained.
- No excluded group is recovered.

**Split rule.** The split was assigned from group identities only, before any label, SEX frequency or calibration
performance was inspected.
- The canonical group ID is `D["unit"]`, the osf exact-record group integer, serialised as a decimal string.
- Groups are ranked by sha256(UTF-8(salt + ID)), with ties broken by the integer ID.
- Each selected group has one representative, its smallest row ID.

| Role | Definition | Realised |
|---|---|---|
| CALIBRATION_HELDOUT | first 2,000 AUDIT_FIT groups, salt `hcal-v1\|heldout\|20261011\|` | 2,000 groups, 2,001 rows, 2,000 representatives |
| ATTACK_FIT_NEW | every row of the remaining AUDIT_FIT groups | 4,061 groups, 4,064 rows (expected 4,061 groups ✓) |
| CALIBRATION_TRAIN_MATCHED | first 2,000 OSF_DEFENSE_FIT groups, salt `hcal-v1\|train-matched\|20261011\|` | 2,000 groups, 2,002 rows, 2,000 representatives (diagnostic only) |

**Disjointness proofs** (ROLE_MANIFEST.json). Zero shared rows and zero shared groups for each of:
- CALIBRATION_HELDOUT vs OSF_DEFENSE_FIT (teacher parameter fitting), HEAD_VALIDATION, ATTACK_FIT_NEW, INNER_SELECTION
  and the assessment;
- ATTACK_FIT_NEW vs INNER_SELECTION, the assessment, OSF_DEFENSE_FIT and HEAD_VALIDATION;
- TRAIN_MATCHED vs INNER_SELECTION, the assessment, AUDIT_FIT and HEAD_VALIDATION.

TRAIN_MATCHED lies inside OSF_DEFENSE_FIT. CALIBRATION_HELDOUT ∪ ATTACK_FIT_NEW = AUDIT_FIT.

**Historical use.** These rows are held out from the current teachers' fitting, not unseen by the project. AUDIT_FIT
rows were used historically to fit attackers. They are not called fresh, and this design does not isolate a causal
mechanism. Teacher-held-out provenance (teacher, heads, head selection, preprocessing) is verified from the actually
called code in PROVENANCE_REPORT.md.

**Label allowlist** (hcal.data.labels_for). HEAD_VALIDATION labels are unreadable by every hcal procedure. Assessment
labels stay masked (−1) until this study's pushed EVALUATION_LOCK; only `hcal.assess` may unseal, through a new gate,
with no lra or older gate. No monkeypatching.

| Procedure | Labels | Rows |
|---|---|---|
| calibration | task | CALIBRATION_HELDOUT and TRAIN_MATCHED representatives |
| selection | task | INNER_SELECTION |
| fitting | task | OSF_DEFENSE_FIT: the source constant predictor, and descriptive fitting-row losses for CALIBRATION_GENERALIZATION (never an objective, gate or selection input; registered before SCIENCE_LOCK, role E finding E-S2) |
| attack | SEX | ATTACK_FIT_NEW (fresh readers), AUDIT_FIT (admitted legacy readers, refit for replay and assessment), INNER_SELECTION |
| attack_diagnostic | SEX | OSF_DEFENSE_FIT (MI diagnostic only) |
| assessment | — | only after the hcal unseal |

## 3. Admission and the frozen bank (hcal/admit.py; SOURCE_ADMISSION.json)

**Copies.** Each copy is verified against (i) the lra unit's COMPLETE.json, (ii) the lra same-device copy SHA256SUMS,
and (iii) for scored units, lra's EVALUATION_LOCK `unit_file_sha256`, read at the evidence commit. 540 units are
copied:
- 2 teachers, 3 references, the fine partitions and the C-TASK D0SAME witness;
- the 84 original releases (27 legacy D0, 27 legacy D1, 30 lra fits);
- 89 lra inner audits;

all per seed. Also admitted: the 6 deployed teacher directories and the deploy input.

**Not admitted.**
- `outer__*`: lra assessment predictions; they carry assessment labels and are not opened before this study's
  EVALUATION_LOCK.
- `inner__*` (cbp custody) and `cor__*`.

**Parity (bitwise).**
- Teacher forward passes on the 83 permitted columns.
- Every legacy D0 release re-encoded.
- Every D1 release re-encoded from policy + decoder, with every token re-solved.
- D1 tokens equal the partition tokens.

**Frozen bank.** There are 57 partition pairs per seed (171 partition/seed units):
- 27 legacy partitions: DIRECT-TASK, FINE-TASK, CLASS, and LOCAL/SEQ-12/SEQ-21/JOINT at λ ∈ {0.01, 0.025, 0.04, 0.06,
  0.08, 0.1};
- 30 lra partitions: C-TASK, 24 W-family, and K-LOCAL, K-SEQ-12, K-SEQ-21, K-JOINT-SINGLE, K-JOINT-PAIR.

The 84 original release IDs are D0/D1 variants of these 57 partitions. Logical IDs are kept, and similar metrics are
never treated as aliases. All seeds use one logical configuration, with seed-specific frozen maps and fitted
parameters.

For every partition and seed the bank table stores:
- tokens and decisions on all rows, from the admitted release;
- token classes and OSF_DEFENSE_FIT token counts;
- canonical teacher sums;
- **mu_t**, the unsmoothed teacher mean, = token_S / token_n;
- **q0_t**, the frozen source-smoothed mean vector, = policy.token_proto;
- the admitted D1 table.

**TEACHER-MEAN decoder.**
- It is q0. Its row checks reproduce token_n exactly and token_S within 1e-9·n on OSF_DEFENSE_FIT.
- For the 27 legacy partitions it equals the admitted D0 release on every row, bitwise.
- For the 30 lra partitions it is a newly evaluated mean-decoded control, not a new assignment. For C-TASK it equals
  lra's D0SAME release, bitwise.
- **Reserved empty tokens** (no fitting rows) keep the original fallback smooth(uniform, class).
- mu_t and q0_t are frozen before any calibration. They are never derived from calibration labels or held-out teacher
  means.

**References.** Continuous U, RAW-J β0.3, LEACE (E), FARE (F) and F0 are admitted with their exact interfaces and lra
inner records, with no refits. RAW-J, E, F and F0 are descriptive only.

## 4. Calibrators (hcal/calib.py; CALIBRATION_RULES.json) [frozen at SCIENCE_LOCK]

New nominee-capable calibrators use only CALIBRATION_HELDOUT task labels. SEX never enters a calibration objective.
There is no grid search, κ sweep, temperature-bound sweep or capacity sweep.

### H-TOKEN32

For each token t of recipient i, minimise over the class-dominant simplex:

  Σ_i [−log q(Y_i) + 0.5‖q − onehot(Y_i)‖²] + 32 KL(mu_t ‖ q),  with q = smooth(u, d) and ε = 1e-12.

- **Rows.** The calibration representatives routed to token t.
- **Prior and solver.** mu_t is the fixed original teacher mean. The certified lra solver is wrapped unchanged, with an
  explicit prior input (A = y + 32·mu_t). It is bitwise identical to lra.decoder.solve_batch when mu_t = S/n.
- **Fallbacks.** If n_t = 0, the result is q0_t exactly, marked NO_CALIBRATION_OBSERVATIONS. A reserved empty token
  keeps q0_t, marked RESERVED_EMPTY_ORIGINAL_FALLBACK.
- **Certificates.** Every fitted token carries the lra certificate (stationarity, dual feasibility, projection, strict
  margin).

### T-TOKEN32

- The identical objective, prior, κ and constraints, fitted on the 2,000 CALIBRATION_TRAIN_MATCHED representatives.
- Applied only to DIRECT-TASK, FINE-TASK, CLASS and legacy JOINT λ0.1: 12 partition-pair fits.
- Occupancy differences are recorded. It is diagnostic only: never a nominee and never T*.

### H-GLOBAL-TEMP

  q_α(t)_k = softmax_k(α · log q0_tk),  α ∈ [0.25, 4].

- **Scope.** One α per task, map and source seed.
- **Objective.** Unclipped calibration NLL, evaluated with a stable log-sum-exp in float64.
- **Convexity.** The NLL is convex in α. Its derivative is mean_i[Σ_k q_α,k log q0_k − log q0_{Y_i}], and its curvature
  is the mean q_α-weighted variance of log q0.
- **Solver.** Deterministic bounded bisection on the derivative, with boundary KKT checks.
- **Identity path.** α = 1 returns q0 exactly.
- **Scoring.** Assessment scoring applies the pinned 1e-12 clip for every arm. The clipped-probability count and both
  objectives are recorded.
- **Convexity claim scope.** NLL-plus-Brier is not claimed convex in temperature.

### H-CLASS-TEMP

- The same scalar, fitted within each predicted class and shared over that class's tokens: at most 2 parameters for
  income and 6 for occupation.
- A class with fewer than 50 calibration representatives (or none) uses α = 1, predetermined.
- There are no per-token temperatures.

### Continuous U

- Both temperature families apply to U, with the same rows, bounds, solver and count rule.
- **Log-input rule.**
  - For α ≠ 1: p' = max(p, 1e-12), renormalised, then softmax(α log p').
  - α = 1 returns p exactly.
- U's original source interface stays visible. Only the deterministic calibrated vector is appended, and utility
  scores that vector. Protected codes never expose continuous scores.
- There is no per-token calibration of U.

### Preservation and records

- Positive α preserves class order. The argmax is checked against the token class (or U's decision) on every token and
  row. A mismatch is never hidden.
- **Recorded:** parameter counts, token and class counts, identity fallbacks, boundary solutions, objectives and solver
  residuals.
- **Evaluated:** NLL and Brier on calibration, TRAIN_MATCHED and inner rows, plus later assessment. ECE is
  supplementary.

**Decoder variants per partition** (hcal/ids.py).

| Partition | Variants |
|---|---|
| Legacy | D0 (= TEACHER-MEAN), D1 (admitted), H-TOKEN32, H-GLOBAL-TEMP, H-CLASS-TEMP; + T-TOKEN32 for the 4 diagnostic partitions |
| lra | D1 (admitted), MEAN (TEACHER-MEAN control), H-TOKEN32, H-GLOBAL-TEMP, H-CLASS-TEMP |

## 5. Attacker equivalence and strong controls (hcal/bank.py, hcal/controls.py; CONTROL_PLAN.json) [frozen at SCIENCE_LOCK]

No weaker reader is refit and called privacy. Each partition and seed has one common attack bank, shared by all public
decoder variants. Every variant receives the same selected attack predictions, hence identical primary AUC and CE
(COMPLETE_INTERFACE_EQUIVALENCE.json).

**Bank order** (an ordered union):

1. **Admitted legacy banks.** These are the lra complete-interface inner audits of the partition's original releases:
   legacy D0 then D1 for legacy partitions; D1 for lra partitions. They contain:
   - the pinned FINAL slate (LR×5, MLP×4, HGB×4, DA_LR, DA_MLP);
   - cell readers CC_α{0.5, 1, 5} and CCpair_α{0.5, 1, 5};
   - both ignore-the-other-recipient banks.

   They were fitted on all historical AUDIT_FIT rows, which include the rows now in CALIBRATION_HELDOUT (disclosed), and
   never on inner or assessment labels. Their original transforms, one-hot vocabularies and decoder tables are kept.
   They prevent the smaller fresh fitting pool from creating artificial protection.
2. **One fresh bank.**
   - Fitted only on ATTACK_FIT_NEW and selected on INNER_SELECTION.
   - Uses the same slate and cell readers, the inherited unseen-token prior and the INNER-selected unseen-tuple fallback.
   - Its view is the complete fresh interface: [one-hot token over the full alphabet (occurrence-ordered columns), every
     registered public lookup vector of the partition (all decoder variants' tables, in registered order), one-hot
     decision]. The pair view is [v1, v2]. Both ignore-recipient banks are included.

**Fresh-view hygiene.** Registered before SCIENCE_LOCK, after a deterministic LAPACK SVD non-convergence inside the
pinned DA readers on a synthetic planted fresh view.
- The rule applies only to fresh views (audit, controls and every refit), on each design matrix (v1, v2, then the pair
  concatenation).
- It drops every column that is constant on the attacker fit rows (ATTACK_FIT_NEW; role membership only, no labels),
  then every exact duplicate on those rows of an earlier kept column.
- No slate member can learn a weight for a dropped column from the fit rows, so every reader is kept. Cell readers key
  on token identities and are unaffected.
- Legacy views are the admitted lra views, unchanged.
- The kept-column lists and their hashes are recorded.

**Selection.**
- AUC and CE readers are selected separately on INNER_SELECTION, with fixed orientation P(SEX=1) and first-in-bank-order
  ties (1e-12).
- No assessment-based orientation reversal, no AUC clipping, and no choosing the smaller recovery estimate.
- The selected reader is refit at attacker seeds 0–2. Recovery is the seed mean.
- Union winner: per view and criterion, the first bank whose seed-0 selected value beats the running best by more than
  1e-12. The reported value is that bank's seed 0–2 mean.
- Token identities are categorical; integer IDs are never an ordered feature.

**Continuous U** has the complete source interface. Its bank is:
- the admitted lra SRC|U composed record (U's own interface bank ∪ the 84 lra code banks);
- followed by the fresh bank of every audited partition, in registered order.

An attacker holding U's interface can compute every code token and every public decoder table, so every code reader
composes. Stored predictions are reused, not refit. U's bank contains every audited code's bank, so its seed-0
selection value is never below an audited code's; the reported attacker-seed mean of the selected reader can differ
(MATH_REVIEW note 4). U's three decoder variants share this one bank: a calibrated continuous vector is a utility change, not
information removal.

**Audit pruning.** This is the only scientific pruning rule, fixed from utility before any new attack result is seen.
- A partition is audited (fresh bank and common record) iff at least one of its non-diagnostic variants passes **all
  §6 inner utility gates on every task and every seed**: the original U0 gates and the calibration-matched Ucal*
  gates.
- The 4 task-only partitions (DIRECT-TASK, FINE-TASK, CLASS, C-TASK) and the 4 fixed diagnostic partitions
  (DIRECT-TASK, FINE-TASK, CLASS, JOINT λ0.1) are always audited.
- Skipped partitions are recorded PREDECLARED_UTILITY_INELIGIBLE, not as fitted, failed or budget-truncated.
- Utility is evaluated for the whole decoder bank.
- If every partition is ineligible, the fixed diagnostics still run.

**Real-data controls** (CONTROL_PLAN.json). These use the source rules and limits unchanged: NULL_Z 3.5; PLANT_MIN 0.75
(strict); the realised null threshold must equal 0.5654143765984265 within 1e-12; CONF η 0.05; ROT amplitude 1e-6.
- **Registered 2026-10-07, before any real control ran and before SCIENCE_LOCK.** The required controls test the
  pipeline that produces the primary recovery: the COMMON bank, meaning the admitted legacy complete-code pipeline
  (fitted on AUDIT_FIT, on each original release view of the partition) plus the fresh pipeline (fitted on
  ATTACK_FIT_NEW, on the fresh complete view).
- **One S\* for both pipelines.** S\* is the lra/smf permutation: within AUDIT_FIT, INNER half A and half B. The fresh
  pipeline uses S\* restricted to ATTACK_FIT_NEW. Plants use the same noisy S\* bits for both pipelines.
- **Common verdict.** Each check takes the union winner, chosen on half A over the banks in bank order, and evaluates
  it on half B.
- **Diagnostics only.** Fresh-only and legacy-only verdicts are reported but are never required.
- **Why.** A synthetic run at real role sizes showed that the smaller fresh fitting pool alone may miss the XOR plant
  (0.716). The primary audit never relies on the fresh bank alone.
- **Checks and plan.**
  - NULL, NULL_CALIBRATION ×5 (common bank), CONF_r1/r2, COLL_r1/r2 and XOR on U|DIRECT-TASK, U|JOINT λ0.1, U|CLASS
    and U|K-JOINT-PAIR (seed 0).
  - ROT_r1/r2 on SRC|U through the unchanged lra machinery.
- **Timing and failures.**
  - Controls are fitted after SCIENCE_LOCK and must pass before nomination.
  - A failed required control is a technical validity issue, not a negative method outcome.
  - Synthetic tests are used for engineering only and never replace the real-data controls.

**Supplementary.** Token-only recovery (cell readers on ATTACK_FIT_NEW) is supplementary only. Probability-only
recovery is not computed. Primary protection always uses the complete interface.

## 6. Nomination and calibration fairness (hcal/select.py; SELECTION_RULES.json) [frozen at SCIENCE_LOCK]

**Original inner utility rules** (qpc.utility.gate_record, unchanged). They apply on each task and each source seed,
on INNER_SELECTION, against U0 (the original frozen U of that seed):
- acc ≥ U0 − 0.01;
- LL ≤ U0 + 0.01;
- Brier ≤ U0 + 0.005;
- gain ≥ 0.8 · U0 gain;
- gain ≥ 0.03;
- exact U0 decision preservation on every admitted row.

The source constant predictor, class vocabulary, 1e-12 clip and per-row metrics are retained. There is no 0.006
buffer and no relaxation.

**Ucal\*.**
- Chosen among U identity, H-GLOBAL-TEMP and H-CLASS-TEMP by the smallest mean INNER_SELECTION sum of income and
  occupation NLL, averaged equally over the 3 seeds.
- Vectors must be finite and decision preservation verified.
- Ties within 1e-12 prefer identity, then global, then class.
- NLL here, and in the "mean summed task NLL" tie key, is the source scoring log loss (true-class probability
  clipped at 1e-12). The unclipped NLL is only the calibrators' fitting objective.
- One family across seeds; parameters differ by seed. Locked in SELECTION.json before any privacy selection.

**Calibration-matched gate (added).** Each candidate and comparator must have LL ≤ Ucal\* + 0.01 and Brier ≤ Ucal\* +
0.005 on each task and seed, in addition to the U0 rules. This is a new, stronger comparison, separate from the
original eleven clauses. No predecessor verdict changes.

**T\*** is the eligible task-only release with the lowest mean inner pair AUC.
- **Pool:** DIRECT-TASK, FINE-TASK, CLASS and C-TASK with every admitted original, mean, held-out and shared variant
  (T-TOKEN32 excluded); plus U identity and its two shared calibrations.
- No candidate's partition used SEX. C-TASK is label-supervised task-only, not label-free.
- Candidates must pass both reference gates.
- **Ties:** lowest worst-seed normalised LL/Brier excess, then lowest mean summed task NLL, then the fixed ID order.
- If none is eligible, that is comparator coverage failure, not a method win.

**P\*** is drawn from every eligible privacy-trained partition: 24 legacy, 24 W-family and 5 K-family. Its decoder is
an admitted original decoder (legacy D0, any D1) or a held-out decoder (H-TOKEN32, H-GLOBAL-TEMP, H-CLASS-TEMP). MEAN of
an lra partition is a control only; T-TOKEN32 is diagnostic only. P\* must satisfy:
- every utility condition above, including the calibration-matched gate;
- inner AUC_i(P\*) ≤ AUC_i(T\*) + 0.005 for each recipient on each seed;
- mean inner pair benefit AUC(T\*) − AUC(P\*) ≥ 0.02. This point gate is fixed before fitting and never reduced.

**Ordering.** Lowest mean pair AUC; then lowest worst-seed normalised LL/Brier excess, which is max over tasks and seeds
of max((LL − LL_ref)/0.01, (B − B_ref)/0.005), taken against both U0 and Ucal\*; then lowest mean summed task NLL; then
the fixed source/map/calibrator ID order. One configuration applies across seeds. Every rejection and margin is
recorded.

**Disclosures.**
- An exact deployed alias of a task-only map gets no privacy-training credit.
- If an original decoder wins: "the incumbent won; no calibration credit".
- A LOCAL or sequential map gets no joint-design credit.
- The source P\* (lra: D0 JOINT λ0.1) and this study's P\* carry distinct IDs and provenance.
- With no P\*: NO_ELIGIBLE_COMPETITIVE_NOMINEE. There is no fallback nominee; the fixed diagnostics still run.
- A missing comparator, artifact or failed control makes the dependent claim INCOMPLETE_OR_INVALID.

## 7. Primary family and the one locked assessment (hcal/family.py, hcal/infer.py) [frozen at SCIENCE_LOCK]

There are exactly **23 primary slots**. Aliases and absent nominees never reduce the size.

**P01–P11: original criterion, P\* vs T\* with the U0 reference.**

| Slot | Statistic | Clause |
|---|---|---|
| P01 | AUC(T\*,pair) − AUC(P\*,pair) | LB > 0.02 |
| P02, P03 | AUC(P\*) − AUC(T\*), income and occupation local | UB < 0.01 |
| P04, P05 | acc(P\*) − acc(U0) | LB > −0.01 |
| P06, P07 | LL(P\*) − LL(U0) | UB < 0.01 |
| P08, P09 | Brier(P\*) − Brier(U0) | UB < 0.005 |
| P10, P11 | acc(P\*) − 0.8 acc(U0) − 0.2 acc(const) | LB > 0 |

**P12–P15: calibrated reference.**

| Slot | Statistic | Clause |
|---|---|---|
| P12, P13 | LL(P\*) − LL(Ucal\*) | UB < 0.01 |
| P14, P15 | Brier(P\*) − Brier(Ucal\*) | UB < 0.005 |

**D01–D04: fitting role** (fixed legacy JOINT λ0.1). T-TOKEN32 minus H-TOKEN32 for D01 income LL, D02 occupation LL,
D03 income Brier and D04 occupation Brier. Positive means held-out fitting helped.

**D05–D08: parameter sharing** (same partition). H-TOKEN32 minus H-GLOBAL-TEMP, same order. Positive means the shared
calibrator helped.

**Reading the diagnostics.**
- No effect size is required. Each D slot reports a signed estimate and interval: SUPPORTS_POSITIVE if LB > 0,
  SUPPORTS_NEGATIVE if UB < 0, UNRESOLVED otherwise.
- The families differ in objective, prior regularisation and parameterisation. A favourable contrast supports the
  shared-calibrator recipe; it is not a causal claim that pooling alone solved overfitting.
- Mixed outcomes are reported as mixed.
- The partition is never substituted.
- Supplementary, nominal 95%: the same contrasts on DIRECT-TASK, FINE-TASK and CLASS, plus H-CLASS-TEMP comparisons.

**Inference.**
- Per model seed, recovery is the SEX AUC (P(SEX=1), fixed orientation) of the inner-AUC-selected common-bank attacker,
  refit at attacker seeds 0–2 on its own fit rows and averaged.
- Utility uses the released probabilities and decisions with the 1e-12 clip.
- Endpoint = the equal-weight mean over model seeds of the per-seed paired statistic.
- SE = sd (ddof 1) over B = 1,999 paired multinomial bootstrap replicates of exact-record groups (seed 20261011,
  identical draws across arms and seeds).
- Interval = point ± z·SE, with z = NormalDist().inv_cdf(1 − 0.05/(2·23)) = 3.0653831516447343, independently
  recomputed.
- These are conditional exploratory intervals; the seeds are not independent populations.
- An interval crossing a cap is unresolved preservation. A violation is labelled only when the interval supports it.

**Labels** (LABEL_TRUTH_TABLE.json).
- OriginalCriterion: PASS iff P01–P11 all PASS with a valid nomination.
- CalibrationMatchedCriterion: PASS iff P01–P15 all PASS.
- ExactDecisionPreservation is an engineering receipt.
- FittingRole and ParameterSharing: four contrasts each.

Overall precedence:
1. ENGINEERING_BLOCKED_NOT_RUN or INPUTS_UNAVAILABLE_NOT_RUN;
2. INCOMPLETE_OR_INVALID;
3. NO_ELIGIBLE_COMPETITIVE_NOMINEE;
4. CALIBRATED_PRIVATE_RELEASE_DEVELOPMENT_CRITERION_ESTABLISHED;
5. ORIGINAL_REQUIREMENTS_MET_CALIBRATED_REFERENCE_NOT_ESTABLISHED;
6. NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED.

## 8. Stages and locks

| Lock | Governs |
|---|---|
| SOURCE_ADMISSION_LOCK | `admit`: hashes, verified copies, parity, role split, frozen bank. No fit. |
| ENGINEERING_LOCK | `engineering`: synthetic correctness checks only (ENGINEERING_CHECKS.json), reviewed by B, D and E. Verdict ENGINEERING_READY on correctness alone. |
| SCIENCE_LOCK | `calibrate`, `utility`, `audit`, `compose`, `controls`, `select`, `replay` |
| EVALUATION_LOCK | `hcal.assess` (single opening), then `hcal.infer` |

**ENGINEERING_LOCK.** No gate requires a favourable effect, reducing information below the decision floor, or beating a
task-only optimum. A known-law zero-leakage CLASS is not hidden.

**SCIENCE_LOCK** freezes roles, representatives, model bank, calibrators, attack plan, controls, gates, selection,
inference, budgets, source dependencies and loaded-module hashes. Each lock is committed, pushed and remote-verified
before its stage runs (`hcal.lock verify`). Every stage process refuses unlocked or changed loaded modules.

**EVALUATION_LOCK** binds the nominee, comparator, Ucal\*, diagnostic bindings, decoder tables, attack models, view
fingerprints, software and role hashes, thresholds and endpoint definitions. No science code changes after it is
pushed.

**Amendments.** At most two substantive post-lock engineering amendments, each dated and pushed before any rerun. Failed
attempts are preserved. Report-only corrections live in separately versioned report code. A split, objective, cap, pool
or primary definition never changes to rescue an outcome.

## 9. Counts, budget, deployment

**Logical units** (QUEUE_MANIFEST.json).
- 171 partition/seed units.
- 513 new held-out decoder-pair units.
- 12 TRAIN_MATCHED diagnostic units.
- 6 U calibration-pair units.
- 252 admitted logical release controls.
- 90 mean-decoded controls for the lra partitions; the legacy means are aliases of D0.
- Privacy-bank fits are counted per audited partition/seed, not per decoder variant.

**Budget.**
- $0 cloud.
- At most 20 CPU-hours and 10 elapsed hours for the whole study, including agents, verification and restores.
- At most two heavy processes (shared semaphore), one BLAS/torch thread each, 8 GiB memory, at least 5 GiB free disk.
- Resumable atomic units.

**Deployment.** `python -m hcal.deploy`:
- consumes exactly the 83 permitted columns, in order;
- binds the teacher, partition policy and public decoder table by hash;
- returns only token, calibrated probability and decision per recipient;
- verifies that argmax equals the preserved decision;
- reproduces the stored releases;
- refuses extra or reordered columns, an incompatible teacher, schema or table, raw-score or fine-ID export, and unknown
  flags.

Research deployment is not authorisation to deploy externally.

## 10. Agents and file ownership (TEAM_PLAN)

| Role | Owns |
|---|---|
| A (coordinator) | hcal/ids.py, data.py, admit.py, run.py, lock.py, sema.py, stages.py, select.py, family.py, engineering.py, registry.py, eval_lock.py, assess.py, infer.py, deploy.py, report.py; tests/…/test_data_select.py, test_infer.py, test_deploy.py; PROTOCOL.md, SELECTION_RULES.json and every integrated document |
| B (math review) | MATH_REVIEW.md (findings only) |
| C (calibration) | hcal/calib.py, tests/…/test_calib.py |
| D (attack/control) | hcal/bank.py, hcal/controls.py, tests/…/test_bank.py, test_controls.py |
| E (verifier) | results/…/verification/ (separately written replay; no import of hcal calibration, runner, metrics, selection or inference) |
| F (custody, resources, reporting) | hcal/closeout.py, tests/…/test_closeout.py, PRIOR_ART_AND_CLAIM_SCOPE.md, TEAM_PLAN.md; BACKUP_VERIFICATION.json, RESTORE_INDEX.json, COST_AND_CLOSEOUT.md (draft), consistency and identity scans |

Reviewers write findings separately and never edit another role's locked file.

**Pre-run claims.** No Adult task or SEX label is read before SCIENCE_LOCK is pushed. Admission and engineering read none.
