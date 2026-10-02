# Pilot protocol: frozen-artifact re-evaluation

> **Superseded (2026-10-02, branch research/combined-stored-model-pilot-v1):** this preparation protocol was executed in amended form; see `results/combined_stored_model_pilot_v1/EXECUTED_PROTOCOL.md` and `AMENDMENT_1.md`. The original bytes are preserved at commit 031860f.

**Version.** v1, 2026-10-02, methodology role (draft label superseded by the executed protocol). Machine-readable twin:
`notes/methodology/protocol_config.json`. Where the two disagree, the JSON governs the evaluator and this
text must be amended.

**Status.**
- Nothing in this protocol has been run.
- Any fitting of attackers or utility probes on real rows needs approval before it happens.
- All data used here is development data that has already been spent (§15).

**Companion files:**

| File | Contents |
|---|---|
| `ATTACKER_ACCESS_TABLE.csv` | attacker recipes R00–R15 |
| `GUARANTEE_CARDS.md` | what each method actually guarantees |
| `SCOPE_CORRECTIONS.md` | corrections and the C1–C5 classification |
| `CHECKPOINT_INVENTORY.csv` and `ARTIFACT_TO_EVALUATION_MAP.csv` | artifact agent |

## 1. Empirical question

When a release passes a method's own stated protection check, what can recipients still recover under
explicitly specified access, and what task performance does the release keep?

Every result is attributed to exactly one of five separate questions:

1. **Fitting-sample compliance** (the native check, R00).
2. **Held-out generalisation of the same quantity** (R02 against the native τ). This is the only route
   to C2.
3. **Recovery outside the guarantee's scope:** other function classes, metrics or surfaces (R01, R03,
   R04, R06–R08, R11–R13).
4. **Prediction-output leakage**, relative to the label-only reference (R05) and the clean-output
   reference (R08).
5. **Recovery from combined releases** (R14; §13 only).

A successful out-of-scope attack is reported as C3, never as a broken guarantee.

## 2. Pilot panel

**Eligibility rule** (fixed before cell selection). A cell is eligible only if all of the following hold:

1. A frozen model is available, and its sha256 matches `CHECKPOINT_INVENTORY.csv`.
2. Row indices are deterministic for every role (§3).
3. There are at least 100 rows per sensitive class in both the assessment role and the attacker_fit role.
4. The same encoder and cell has all of:
   - the untreated reference;
   - at least one documented projection defense;
   - at least one documented noise defense.

   Fitted parameters must be stored, or regenerable by a forward pass without refitting.
5. Task labels are available for the assessment rows.

**Ranking.** Eligible cells are ranked by artifact completeness, using the
`ARTIFACT_TO_EVALUATION_MAP.csv` columns: frozen head, stored outputs, stored defense parameters,
grouping unit. Ties go to the cell with fewer rows.

**Slots** (the coordinator fills these in; a placeholder must not be read as a selection):

| Slot | Cell (dataset / task / sensitive attribute) | Encoder + sha256 | Arms (reference / projection / noise / other) | Grouping unit | n per role | Notes |
|---|---|---|---|---|---|---|
| CELL-A | Adult (PCRL v2 purposes): 8 disallowed (purpose, attribute) pairs, incl. income_prediction / sex = the AAAI "hard" cell | PCRL Round-4 seed-0 `final.pt`, sha256 1cfc2fef…c061 (= durable-guarantees' audited encoder; drive inventory match) | **New-fit track:** untreated (all 8 pairs); Gaussian noise σ_abs ∈ {0.25, 0.5, 1, 2, 4, 8} × seeds {0,1,2} for income_prediction / sex (fit-free regeneration). **Historical-recount track only:** 10 projection releases (MMD/HSIC r1–r16) via stored per-row attacker scores on dg's original rows — no projection regeneration because Q was never saved (refit not authorized). | record (Adult has no linkage key; 5 exact-duplicate records collapse to 15,055 units) | pool = PCRL Adult **test** split, 15,060 rows: attacker_fit 7,571 / attacker_val 2,239 / assessment 5,250 | Inputs written, frozen forward done, 152 manifests admitted (2026-10-02). Race classes 61 and 40 in assessment → NE for race per-class/pairs (3/5 classes, 3/10 pairs supported). |
| CELL-B | HMDA race / loan_decision (AAAI "easy" cell) | PCRL Round-4 seed-0 `final.pt`, sha256 e29d0367…2846 | untreated; noise σ = 8 (only documented noise case on this encoder); no projection | record | pool = PCRL HMDA test split (row counts to be written at input preparation) | **Not eligible** (fails rule 4: no documented projection arm on this encoder); **not part of the CELL-A run**; inputs not prepared (Adult-specific prep script; extension step E1 in QUICKSTART). Smallest race class 126 in dg's assessment role; test-split counts to be checked against n_min = 100 before lock. |
| CELL-C (optional) | — | — | — | — | — | No further cell qualifies (see notes/artifacts/pilot_cell_selection.csv). |

**Coordinator note on rule 4 (2026-10-02).** CELL-A satisfies rule 4 only through the historical-recount track: the
projection arms exist as stored per-row attacker probabilities on durable-guarantees' original rows (75/25 split of
the PCRL train partition, `defense_fit_overlap=true`), not as regenerable releases. The new-fit track therefore
compares untreated vs noise; projection arms are evaluated with the repaired metrics (support, worst-class,
bars, intervals) by recount only. A projection-refit arm needs a separate, explicit defense-refit authorization.

A cell that fails a rule is listed in `PANEL_EXCLUSIONS` with the rule number. It is never silently
dropped.

## 3. Roles and rows

| Role | Content | Source |
|---|---|---|
| `defense_fit` | rows that entered any fitted parameter (encoder, eraser, projection, head) | historical; fixed by the artifact |
| `attacker_fit` | the attacker's labelled population | rows **outside** `defense_fit`; 50% |
| `attacker_val` | selection and early stopping only | rows outside `defense_fit`; 15% |
| `assessment` | scoring only, used once per locked recipe | rows outside `defense_fit`; 35% |

**How the attacker and assessment rows are split:**
- Stratified by (s, y).
- Grouped by unit:
  - Diabetes: `patient_nbr`;
  - ACS (if used): SERIALNO;
  - CelebA: identity;
  - Adult and HMDA: row, because no linkage key exists (stated).
- Seed and row-index manifests (sha256) are written to the lock file.

**PCRL v2 cells.** The pool is the test split. dg cells on PCRL R4 encoders use the PCRL val and test
rows, which dg never read. Both pools are spent through PCRL selection, and this is stated in every
table.

**Secondary arm, for historical comparability only:** `attacker_fit` = `defense_fit` rows.
- Every result from it is labelled `defense_fit_overlap=true`.
- It is never pooled with the primary arm.

**All surfaces and recipes are scored on the *same aligned* assessment rows**, joined by row or unit ID
with an assertion.

## 4. Release surfaces

| Surface | Content |
|---|---|
| `rep` | the released representation, exactly as the method releases it (deterministic, or one fresh draw per release) |
| `out` | the declared task-output object of the frozen head: hard label, calibrated score, or logit vector. Declared per cell in the lock file. |
| `rep+out` | only where the release contract gives both to the same recipient |

Linear and nonlinear attackers are compared **within** each surface under the same metric.

## 5. Attackers

Recipes come from `ATTACKER_ACCESS_TABLE.csv`.

**Pilot slate:**

| Recipe | Role in the pilot |
|---|---|
| R00 | native check, reported separately and never pooled |
| R01, R02 | held-out linear |
| R03, R04 | nonlinear |
| R05 | label-only |
| R06L, R06N, R07L, R07N | output and combined |
| R08 | clean-output reference |
| R09, R10 | noise arms only |
| R11 | general adaptive |
| R12 | noise arms whose contract issues fresh draws; N ∈ {1, 4, 16, 64, 256}; persistent token ⇒ N = 1 |
| R13 | subspace arm only |

R14 is used only for the multi-purpose scenario. R15 is not run.

**Selection rules:**
- Hyperparameters are chosen on `attacker_val` by log-loss.
- The selected configuration is refit on `attacker_fit` only.
- Each locked configuration is scored once on `assessment`.
- Identical releases receive identical candidate sets.
- Budgets are in the JSON.

**Knowing the defense is not knowing the target's clean input.** R11 simulates releases of its *own*
population with the published fitted parameters. Where those parameters are not stored, R11 is reported
as C5 ("not estimable without refit").

## 6. Native check, kept separate

R00 reproduces each method's own test as the method defines it: rows, penalty, precision and threshold.
- For PCRL, the historical float32 value is reported beside the float64 value.
- It appears in its own column and is never averaged with attacker results.
- Its pass or fail is the conditioning event for C2 and C3 labels.

## 7. Decomposing the historical R² → AUC jump

Every attacker is scored with **two metric types**:
- an R²-type held-out explained variance of the one-hot attribute: the pooled Brier skill,
  1 − SSE/SSE_prior, with the prior taken from `attacker_fit` frequencies;
- AUC.

This lets each step change one factor at a time:

| Step | Rows | Attacker family | Metric | Surface | Isolates |
|---|---|---|---|---|---|
| F0 | in-sample (historical) | native linear | R² | rep | the method's own check |
| F1 | held-out | linear (R02) | R² | rep | **generalisation** (F1 vs F0 → C2) |
| F2 | held-out | linear (R01) | AUC | rep | **metric change** at a fixed linear family (F2 vs F1) |
| F3 | held-out | nonlinear (R03/R04) | R² (Brier skill) | rep | **attacker family** at a fixed R²-type metric (F3 vs F1) |
| F4 | held-out | nonlinear (R03/R04) | AUC | rep | the historical audit's combination (F4 vs F0 = metric + family + rows) |
| F5 | held-out | linear / nonlinear (R06L/N) | R² and AUC | out | **surface change** (F5 vs F2/F4) against R05 and R08 |
| F6 | held-out | linear / nonlinear (R07L/N) | R² and AUC | rep+out | combined surface |
| F7 | held-out | adaptive (R11–R13) | R² and AUC | rep (and out if released) | **defense knowledge** (F7 vs F3/F4) |

**Reporting the decomposition:**
- Each contrast is reported as **verdict changes at each bar**, because R² and AUC are not on one scale.
- Numeric differences are reported only within one metric.
- The historical 59/67 corresponds to F0 → F4. The AI review's "surface change" premise does not apply,
  since the audit scored the representation only (SCOPE_CORRECTIONS A1).

## 8. Label-only reference for outputs

R05 predicts s from the *true* task label on the same assessment rows. It is an **empirical** reference
under this recipe, not a mutual-information bound or floor.

**Output leakage** is reported three ways, each paired over assessment units:
- R06 − R05: beyond what the label carries;
- R06 − R08: what the defense changed relative to the clean model;
- R06 itself, in absolute terms.

## 9. Metrics

All metrics are computed on assessment rows unless they are marked as in-sample. Precision is float64.

| Metric | Definition |
|---|---|
| native | the method's own statistic (R00) |
| held-out linear R² | 1 − Σ‖z − ẑ‖² / Σ‖z − z̄_fit‖² over one-hot columns. ẑ is from R02 fit on `attacker_fit`. **Intercepts and means come from `attacker_fit`**, never from the scored rows (this repairs the test-mean centring of HL-05). Uses the numerical floor from the JSON. |
| Brier skill (R²-type, any attacker) | the same formula with ẑ the predicted class probabilities |
| log-loss reduction | prior log-loss minus model log-loss, with the prior from `attacker_fit` |
| macro OvR AUC | mean of one-vs-rest AUC over **supported** classes, with the coverage reported |
| supported worst-category | maximum over supported classes of OvR AUC |
| supported worst-pair | maximum over pairs (j, k), both supported, of the pairwise AUC computed on rows of classes j and k only, using the score p_j/(p_j + p_k). Reported with **denominator** = number of supported pairs, and **coverage** = supported pairs / all pairs and rows in supported classes / all rows. |
| ρ₁² (contrast diagnostic) | top squared canonical correlation between the release Z and the centred one-hot attribute with one class dropped: ρ₁² = λ_max(Σ_YY^{-1/2} Σ_YZ (Σ_ZZ + εI)^{-1} Σ_ZY Σ_YY^{-1/2}), with ε = 1e-6·tr(Σ_ZZ)/d. Equals the maximum R² over all linear contrasts of the classes. Reported in-sample (definitional) **and** held-out: directions fit on `attacker_fit`, correlation on assessment. |

**Max pair AUC and contrast R² are different quantities:**
- Pair AUC measures separability of two classes by *any* monotone score of one model.
- ρ₁² measures the linear explained variance of the best class contrast.
- Neither implies the other's bar.

## 10. Minimum support and "not estimable"

**Thresholds:**
- A class is **supported** if it has n_k ≥ 100 rows in **both** `attacker_fit` and assessment, and
  n_k ≥ 30 in `attacker_val`.
- A pair is supported if both of its classes are supported.

**What is not estimable:**
- unsupported classes or pairs;
- an attribute with fewer than 2 supported classes;
- a recipe whose inputs do not exist (e.g. R11 without stored defense parameters).

These are recorded as `NE` with a reason code and the counts. They are never imputed, never scored as 0,
and never scored as "protected". Macro metrics average over supported classes only, and report coverage.

## 11. Bars and the decision rule

**Bars:**
- AUC bars: 0.52, 0.55 and 0.60. 0.55 is the historical bar; the other two are sensitivity bars, not
  separate hypotheses.
- R²-type τ grid: 0.01, 0.02, 0.05 (historical) and 0.10.

**Decision rule** (per metric × bar), using one-sided 95% bounds from §12:

| Condition | Verdict |
|---|---|
| UCB < bar | **established below**: recovery is below the bar for this recipe and access |
| LCB > bar | **established above**: recovery is above the bar |
| otherwise | **unresolved** |

Failing to establish recovery is not evidence that recovery is absent. "Established below" is always
qualified by its recipe and access, and is never phrased as "removed".

**Mapping to outcome categories:**

| Category | Condition |
|---|---|
| C1 | the native check fails as the method defines it |
| C2 | native passes, and held-out linear R² (R02) is established above τ |
| C3 | native passes, and any out-of-scope recipe is established above its bar |
| C4 | a population guarantee's assumptions are verified, and its bound is established exceeded (e.g. an AUC LCB above Φ(μ/√2) on a correctly clipped channel) |
| C5 | NE, or every relevant bound is unresolved |

## 12. Uncertainty

**Sampling unit:** the grouping unit of §3.

**Interval:** a cluster bootstrap over assessment units, B = 2000, percentile method. The 90%
two-sided interval gives the one-sided 95% UCB and LCB.

**Separate variance components** (each reported on its own, never folded into the bootstrap interval):

| Component | How it is computed |
|---|---|
| attacker refit variance | attackers are held fixed inside the bootstrap; 3 attacker seeds per selected configuration (and K draws for stochastic channels) give the refit SD |
| defense-seed variation | the spread over training seeds, as a per-seed table |

Seeds share assessment units, so they are refit replicates and not independent samples.

**Max statistics** (worst-category and worst-pair) are re-maximised inside each bootstrap replicate. A
permutation null at the real class sizes uses the same max-over rule (200 permutations). There is no
maximum over seeds.

**Survey weights:** ACS PWGTP if ACS is used, with weighted and unweighted estimates both reported. Not
applicable to Adult, HMDA, Diabetes or CelebA.

## 13. Multiplicity

The **primary family** is declared in the lock file, has at most 3 endpoints, and is evaluated on all
native-pass (cell, arm) combinations of the panel at τ = 0.05 and AUC 0.55:

| Endpoint | Test |
|---|---|
| P1 (C2) | R02 held-out linear R² > 0.05 |
| P2 (C3, representation) | best of R03/R04 macro OvR AUC > 0.55 |
| P3 (output leakage) | R06N − R05 AUC difference > 0 |

**Adjustment rules:**
- Holm adjustment across all primary tests (endpoint × cell × arm). p-values come from the bootstrap
  (the share of replicates on the wrong side of the bar).
- Worst-pair and worst-class selection is handled by bootstrapping the max (§12), not by Holm.
- Everything else is secondary, reported with unadjusted intervals and labelled exploratory.
- A conclusion that combines several clauses is reported as an intersection-union decision. "k of m
  clauses passed" is not partial confirmation.

## 14. Utility

**Rows:** all utility is measured on the **same assessment rows** within each cell.

**Two kinds, reported separately** (SCOPE_CORRECTIONS B):
- **U1, frozen head:** the deployed head, scored on assessment rows. Rows that entered head or defense
  fitting are labelled.
- **U2, independent probe:** a common family (multinomial LR plus the R04-style MLP) fit on
  `attacker_fit` and selected on `attacker_val`.

**What is reported:**
- Absolute utility: accuracy, log-loss and AUC (macro-F1 for multiclass).
- The untreated-reference value.
- The **paired difference** to the reference, bootstrapped over the same units.

**Normalised utility:**
- Reported only when clean lift over the constant predictor has an LCB of at least 0.03 and a CI
  half-width below 0.25 × lift.
- Otherwise it is "lift too small".

**Frontiers:**
- Drawn only among arms whose utility CIs overlap, or which are within a declared 1 pp margin.
- Collapsed or constant releases are flagged and never counted as passes.

**Retrospective selection:** any operating point chosen after inspecting results (U3) is labelled as
such.

## 15. Multi-purpose scenario (secondary; not the headline)

**Inputs:**
- A declared sensitive set S.
- A purpose-permission table, deny by default.
- For frozen PCRL artifacts, the table is the one each model was trained under (`pcrl/data/*.py`
  disallowed lists). It is copied into the lock file.
- Explicit coalition membership.

**What is reported:**
- **Separate view:** each recipient's own release.
- **Combined view:** a declared coalition's concatenation (R14).
- **Combination gain:** best single view minus union, on a log-loss scale, with refit replicates.
  Negative values are reported unclipped.

**Scope rules:**
- Targets are only the attributes disallowed to **every** coalition member.
- The protocol does not promise to hide an attribute from a coalition that contains a recipient entitled
  to it.

**Per-recipient adaptations** of existing methods (e.g. per-recipient LEACE) are scenario arms, and are
run rather than assumed to fail. They need new defense fits, so they are **deferred** until approved.
The pilot uses existing frozen per-purpose and union-erased models only.

## 16. Exposure, locks and amendments

**Exposure:**
- All pilot data is development data already spent by earlier tuning, selection or narrative.
- Results are development evidence only.
- No new state, year or cohort is opened.

**Lock.** Before any scientific attacker or probe fit, a lock commit records:
- `protocol_config.json` (sha256);
- the panel slots;
- the per-role row manifests (sha256);
- the declared output objects and the permission table;
- the primary family.

**Amendments** go to `AMENDMENTS.md`, each with:
- UTC time;
- what changed;
- why;
- whether any assessment-row result had been seen.

Post-lock changes are reported as amendments, and any result they affect is labelled post hoc.

**Controls before any method row is read:**
- **Synthetic fixtures:** positive, XOR, planted contrast, null, constant extension and replay.
- **Per cell:** a positive control (the untreated reference recovers above R05) and a null control
  (permuted s: every bar established below, or unresolved).
- If a control fails, the cell stops.

## 17. Outputs

For every assessment unit, per-unit scores are saved as compressed arrays with sha256 manifests.

Per-cell tables are kept separate:
- native;
- F0–F7;
- utility U1 and U2;
- NE counts.

Each row carries all of:
- recipe_id;
- access tags;
- rows role;
- `defense_fit_overlap`;
- checkpoint rule (final or best);
- the historical criterion name (ABS or INCR) where relevant.

Archives go to the non-expiring bucket only.


## 18. Coordinator decisions recorded before any scientific fit (2026-10-02)

These settle the points on which the methodology and evaluator drafts differed. They become binding when the
lock file is written (§16); until then they are the recorded intention, and any change is logged as an
amendment with its timing and reason.

| Item | Decision | Reason |
|---|---|---|
| R02 denominator | SS_tot around **attacker_fit** means (primary). The score-row-mean variant is computed and reported as a sensitivity only. | The fit-mean predictor is the prior-only reference the protocol requires; a recipient does not know the assessment-row mean. |
| R02 clamping | Never clamp; negative held-out R² is retained. | A held-out fit worse than the prior is information, not zero. |
| Macro AUC | Mean over **supported** classes, with coverage reported; NE when fewer than 2 classes are supported. | Methodology §10; prevents silent averaging over absent classes while keeping a usable statistic. |
| Interval | One-sided 95 % bounds = two-sided 90 % cluster-bootstrap interval (B = 2,000) over assessment units. | Methodology §12; evaluator translation verified. |
| Role split | Deterministic hash of the record key (not stratified by (s, y)). | One shared split is required so every pair, attribute, surface and arm is scored on the **same** aligned rows; per-(s, y) stratification would differ by pair. Class support is checked after the split and reported. |
| n_min | 100 per class in attacker_fit and assessment, 30 in attacker_val; **no post-hoc pooling** of small classes. | Adult race is NE for 2 of 5 classes; pooling after seeing counts would be a data-dependent choice. A pooled-minority race analysis may be added only as a labelled amendment. |
| Noise arms | Restricted to income_prediction / sex on CELL-A in the pilot; all 8 pairs × 6 σ × 3 seeds are prepared (144 manifests) as the first extension. | Matches the documented AAAI noise cases; keeps the pilot near 1 CPU-hour. |
| Noise outputs surface | Noise arms release the noisy representation; the outputs surface is the clean model's output (historical channel). rep+outputs for a noise arm combines both and is labelled so. | Release contract of the historical noise channel. |
| Seeds | Release seeds {0,1,2} are separate release draws (one release per row each), reported as seed variation, not as extra sampling units. | Methodology §12. |
| Closed-form endpoints | Native check (fit = score on assessment rows), R02, held-out ρ₁² and descriptive health are produced by `fit-attackers` behind the same `--execute-scientific-fits` flag. | R02 is a least-squares fit on real rows. |
