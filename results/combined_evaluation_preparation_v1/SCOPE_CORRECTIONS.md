# Scope corrections: addendum dated 2026-10-02

**What this file is.** A dated addendum to the 2026-10-01 reconciliation package
(`results/combined_evidence_reconciliation_v1/`, commit 07a9ca3). Those files are left unchanged. This
addendum corrects statements in them where new tracing contradicts them.

**What was done.**
- Code was read at the pinned refs (PCRL `origin/main`@55e4cb1d plus named branches; durable-guarantees
  @956f5c88).
- Existing committed outputs were recounted.
- Synthetic-only checks were run.
- Nothing was trained, refit or fitted on real rows.

**Where the supporting files are.** Under `notes/methodology/`:

| File | Contents |
|---|---|
| `heldout_linear_inventory.csv` | every linear evaluation found, with fit, selection and score rows (HL-*) |
| `failure_classification.csv` | historical "failures" classified C1–C5 (FC-*) |
| `heldout_linear_recount.py` + `_output.json` | recount of stored held-out linear outputs |
| `scale_composition_check.py` + `_output.json` | ridge scale dependence, health labels, composition, Prop 6 conventions |
| `leace_tolerance_check.py` + `_output.json` | concept-erasure 0.2.4 defaults, float32 |
| `batch_r2_bias_check.py` + `_output.json` | null level of the training-time / Cotter R² |

**Outcome labels.** As in `notes/PREP_CONTEXT.md`:

| Label | Meaning |
|---|---|
| C1 | the implementation fails its own fitting-sample check |
| C2 | the fitting-sample check passes but does not generalise to held-out rows |
| C3 | an attacker or surface outside the stated guarantee recovers the attribute |
| C4 | a population guarantee is contradicted under its own assumptions |
| C5 | insufficient evidence, or the quantity is not estimable |

**Status words:**
- **corrected**: the previous statement is replaced.
- **qualified**: the previous statement is kept with a stated scope.
- **pending**: needs an operation not done here. The exact operation is named.

---

## A. Guarantee scope

### A1. "No held-out linear probe was ever run"

**Previous statement:**
- `combined_evidence_reconciliation_v1/UPDATED_MEETING_BRIEF.md:63`: "No held-out linear probe was ever
  run".
- `UPDATED_MEETING_BRIEF.md:159`: "Category (1) has **never been measured** in either repository".
- `REBUTTAL_WORK_RECONCILIATION.md:281-282`: "None was ever tested with a held-out linear probe".

**Corrected statement.** Held-out linear evaluations of defended representations do exist. None of them
is the in-scope test for the guarantees in question: a held-out squared-loss linear R² of the same
quantity that the certificate bounds. Each one is limited in a specific way:

**PCRL:**
- **Logistic regression, train→test** (HL-04), stored per purpose for the submitted R5/R7 final.pt.
  - 4 of 60 disallowed pair-seeds exceed majority by more than 1 pp; 2 by more than 2 pp.
  - 3 of those 4 pass the native check. The fourth (Diabetes R7 s2 quality_research/age_bucket, native
    R² 0.0686) fails it.
  - This is linear, but the metric is 0–1 loss, so it is C3, not C2 (FC-09, FC-10).
- **SPLINCE LR auditor, train→test** (HL-08): 0 of 60 above 1 pp.
- **INLP's own stopping rule** (HL-07): a held-out linear classifier on the validation rows. Those rows
  also drive selection.
- **Ridge one-hot R² fit on train and scored on test** (HL-05), on the May 2026 Adult cross-purpose-
  constraint retrain only.
  - 5 of 24 pair-seeds exceed 0.05 (0.082–0.128).
  - The native check on the same checkpoints is not stored, so the C1-vs-C2 split is **pending** (FC-11).
  - The same function on the rebuttal branch (HL-06) has not been recounted.
- **Best-of-{LR, RF, SVM, XGB} auditor, train→test** (HL-03). Only the maximum is stored, so the LR part
  cannot be separated.

**durable-guarantees:**
- **One held-out linear attribute probe**, in `experiments/vfae_calibration.py` (HL-11).
  - It is VFAE accuracy, fit on va and scored on te.
  - te also selected β.
  - It is not one of the 67 audited configurations.
- **No linear attacker is in any battery used for the 67.**
  - The battery is XGB, MLP and LoRA32 (`diagnostic.py:192-202`).
  - `stronger_attackers.py:75-77` defines LR, but it is never used (`:137`).
- **Every "certificate" R² is in-sample on the rows the defense was fit on** (HL-12).

**Not held-out, though labelled "test" or "val":**
- **PCRL native R²** (HL-01, HL-02): fit and scored on the same test rows.
- **CelebA PCRL-V "Val-partition R²" 0.156–0.171** (HL-09): in-sample on 4,096 rows with d = 512.
  - Its null bias is about 0.125.
  - The degrees-of-freedom-adjusted value is about 0.035–0.053.
- **The training-time and Cotter-selection R²** (HL-10): in-sample per minibatch of 256 (see D4).

**Status:** corrected. "Within-scope (held-out linear R²) failure has not been tested on the submitted
grids" replaces "never run".

### A2. Historical "failures", classified

Full table: `notes/methodology/failure_classification.csv` (FC-01 to FC-18).

| Category | Cases | Count |
|---|---|---|
| C1 | PCRL native strict failures (FC-08); Diabetes LR cell that already fails natively (FC-10); Obliviator not reaching its own stop (FC-07); LEACE-on-raw in float32 without a numerical floor (FC-14, numerical) | 4 |
| C2 | none established. The only candidate (FC-11) is pending | 0 |
| C3 | the 59/67 audit; attacked-R²; Tier-2 LRT survivors; published-method gauntlet; FARE multiclass; FNF; held-out LR accuracy on native passes; adjusted-criterion auditor; concatenation flags; averaging and knows-Q (FC-01–06, 09, 12, 13, 17) | 10 |
| C4 | the refuted NeurIPS R²→accuracy bound (FC-15) | 1 |
| C5 | CROSSPURP held-out R² without a native counterpart (FC-11); CelebA in-sample val R² (FC-16); clipped DP channel, never attacked (FC-18); LR-vs-nonlinear attribution in the stored auditor maximum (part of FC-12) | 4 |

C2 is therefore *untested* on the submitted grids (no held-out linear R² was paired with a passing native check), not shown absent.

The earlier description "only failures outside scope" is right for the AAAI audit. It needs two
exceptions:
- FC-15 is C4.
- FC-14 is a numerical C1. A ridge or eigenvalue floor hides it.

**Status:** corrected (classification added).

### A3. Composition wording and Proposition 6 under the code's regularisation

**Previous statement.** `UPDATED_MEETING_BRIEF.md:55`: "Per-purpose linear bounds do not compose."

**Corrected statement:**
- **Exact zero cross-covariance composes.** If each release has exactly zero cross-covariance with A
  under some law, the concatenation also has zero cross-covariance under that law. Every affine
  predictor of A from the concatenation is then no better than constant in squared loss.
  - Synthetic check: concatenation OLS R² = 0 to rounding (`scale_composition_check_output.json`, A1a).
  - This holds *on the law where it is exact*: LEACE's fit sample. It need not hold on held-out rows.
  - It says nothing about nonlinear recipients. In the same check, adding squared features recovers
    R² 0.235.
- **Small nonzero per-release R² need not give a small concatenated R².**
  - Synthetic check (A1b): per-release R² of 5e-8 and 6e-5, concatenation R² 0.998.
  - Prop 6 is not violated there. Its bound, Σ_p R²_p / λ_min(R) = 1.89, is simply vacuous.
- **"Do not compose" is therefore about approximate bounds, not about exact erasure.**

**Prop 6(III), Tikhonov form** (NeurIPS PDF, b.txt l.1793–1903):
- The printed proof whitens with the *unregularised* Σ_p. Its premise is therefore the unregularised
  per-purpose R², while the conclusion is about the regularised concatenation.
- The fully regularised statement also holds. Write R_τ = S R S + (I − S²) with
  S = blockdiag(Σ_p^{1/2}(Σ_p+τI)^{-1/2}). For unit x, xᵀR_τx ≥ 1 − (1 − λ_min(R))‖Sx‖² ≥ λ_min(R).
  Hence R²_τ(H) ≤ Σ_p R²_τ(h_p) / λ_min(R).
- PCRL's audit quantity is not this plug-in form. It is in-sample ridge R² = tr(Cᵀ(G+λ)⁻¹(G+2λ)(G+λ)⁻¹C)/TSS.
  - This lies between the plug-in form and OLS.
  - Provably it is at most 2 × Σ_p R²_τ(h_p)/λ_min(R).
  - In 400 random trials, including λ comparable to the Gram eigenvalues, the bound held with ratio
    ≤ 0.76 (A2).
- Prop 6 is an in-sample algebraic statement when applied to one row set. It does not cover held-out R².

**Status:** qualified.

### A4. LEACE as actually called

**Previous statement.** The reconciliation treats concept-erasure LEACE as exact on the fit sample.

**Qualified statement:**
- **How it is called.** Every call in both repositories is `LeaceEraser.fit(x, z)` with package defaults
  on float32 tensors: `svd_tol=0.01`, `constrain_cov_trace=True`, `shrinkage=True`.
  - Call sites: PCRL `v2_trainer.py:584`, `laftr_proxy_trainer.py:482`; dg `baseline_gauntlet.py:317`,
    `smart_erasure.py:205`.
- **What the defaults could do.** In principle, `svd_tol` truncation and trace-constraint mixing can
  leave cross-covariance.
- **What the synthetic check found** (`leace_tolerance_check_output.json`):
  - Fit-row cross-covariance is about 1e-7 in all three regimes, including the regime where the
    whitened signal is below `svd_tol`.
  - The erased directions keep variance of about 1e-14 (float32 rounding).
  - An *unregularised* OLS attacker recovers R² 0.13 from them, and this survives on held-out rows
    (0.136).
  - The PCRL ridge (1e-6) reads 0.0003, and standardised logistic regression reads AUC 0.50.
- **Consequence.** Any held-out *linear* recipe must declare a numerical floor. The default proposed in
  `notes/methodology/protocol_config.json` is: discard covariance directions with eigenvalue below 1e-6
  of the largest, and report the unfloored value separately.

**Status:** qualified.

## B. Held-out utility

**Previous statement.** `UPDATED_MEETING_BRIEF.md:196-197`: the pilot "cannot show held-out utility
rankings, because the defenses were fit on spent splits".

**Corrected statement.** "Spent" (used to tune or select) and "fit on" (entered a fitted parameter) are
different. Four kinds of utility number exist; label every utility number with exactly one:

| Kind | Meaning |
|---|---|
| U1 | **Frozen head, rows excluded from fitting:** an existing deployed task head evaluated on rows that entered no fitted parameter (encoder, defense or head). The rows may still be spent through selection. |
| U2 | **New probe on a frozen representation:** a utility head fitted on probe-fit rows and scored on disjoint rows. Fitting it on real data needs approval. |
| U3 | **Retrospective selection:** any utility read at an operating point chosen after inspecting results on the same rows. |
| U4 | **Fresh confirmation:** new, unopened data. Not authorised. |

**Per lineage** (interfaces in `ARTIFACT_TO_EVALUATION_MAP.csv`, built by the artifact agent):

| Lineage | Defense / head fit rows | Existing outputs | Forward pass only | Needs new probe fit |
|---|---|---|---|---|
| PCRL v2 R4/R5/R7 (submitted grid) | encoder, LEACE and heads on train; test spent through tuning | **U1** best.pt test task accuracy (`per_seed_results.json`); **U2-existing** LR utility probe fit on train, scored on test for final.pt (`splince_benchmark/*/metrics.json task_acc_pre_splince`) | U1 for final.pt heads on test or val (stored checkpoints) | a common probe family on a common role split |
| PCRL erase pilot / VICReg / rank-8 / CROSSPURP retrains | as above | U1 test task accuracy per seed (preserved per-seed files) | only where checkpoints survive (see C) | as above |
| dg on PCRL R4 s0 encoders (frozen-rep cells) | defenses, heads and attackers on the PCRL **train** partition. **The PCRL val and test rows of these cells were never read by dg**, but they were spent by PCRL R4 selection. | in-sample utility (rows entered defense fitting); `lift_lr` refit on 75/25 of the same rows (head held-out, defense not) | U1 on PCRL test rows **only if** the fitted defense parameters (Q, LEACE eraser, σ) were saved. Otherwise a closed-form refit is needed, which is not authorised. | yes for a common probe |
| dg-trained channels (easy/middle/hard) | channel trained on all PCRL-train rows; **weights never saved** (map, DG-TRAINED-*) | stored release arrays and in-sample utility; fresh-partition run for 9 points (encoder on one half, utility on the other) | none | the defense must be retrained (not authorised) |
| Published-method gauntlet (dg) | method-specific, mostly all cell rows | max(own head in-sample, LR refit) | only for methods with saved models (artifact agent) | yes |

**Rows that entered defense fitting** are labelled `defense_fit_overlap=true` in every utility and
attacker table.

**Status:** corrected.

## C. Checkpoint inventory

**Previous statement.** `MISSING_FILES_EXACT.md:62`: "All PCRL checkpoints. On the drive".

**Corrected statement.** Availability is model-by-model. Use `CHECKPOINT_INVENTORY.csv` and
`ARTIFACT_TO_EVALUATION_MAP.csv` (package root, artifact agent, in progress).

The same reconciliation lists several checkpoint sets as referenced only in the 7-day-lifecycle bucket
(`MISSING_FILES_EXACT.md` items 10–11):
- erase pilot;
- VICReg;
- rank-8;
- cross-purpose retrain;
- LAFTR-hard.

So "all" was never supported.

**Principles:**
- Metrics surviving does not mean weights survive.
- Weights surviving does not mean the fitted defense parameters survive.
- S3 not inspected does not mean lost.
- Drive not mounted does not mean missing.

**Status:** corrected.

## D. Scale and historical counts

### D1. "R² … invariant to isotropic rescaling"

**Previous statements:**
- `REBUTTAL_WORK_RECONCILIATION.md:126` says R² is invariant to isotropic rescaling.
- `:129-130` says scaling by 1.15–1.41 would turn 0/60 into 60/60 clean "with no change in leakage".

**Qualified statement:**
- **OLS is invariant.** Unrestricted OLS R² is invariant to any invertible linear map, in exact
  arithmetic.
- **PCRL's native R² is not.** It adds a fixed penalty of 1e-6·I to the *unnormalised* centred Gram, so
  its effective penalty is 1e-6/(n·c²) under rescaling by c.
  - Well-conditioned H (D1 in the output file): identical to 1e-12 for c ≥ 0.01.
  - Attribute signal in a near-singular direction (D2):
    - ridge R² goes 0.0996 → 0.120 → 0.156 at c = 1, 1.15, 1.41;
    - it reads a *pass* (0.039) at c = 0.5;
    - OLS reads 0.357 throughout.
- **Training and auditing penalties.** The training constraint (ridge 1e-4 per batch of 256) and the
  CROSSPURP auditor (ridge 1e-4, float32) are more scale-sensitive.
- **Logistic regression.** L2-penalised LR (C = 1, unscaled) is also not scale-invariant.
- **So "no change in leakage" holds only if** the smallest Gram eigenvalues carrying signal are ≫ the
  penalty. The pending check is a frozen forward pass on stored pilot or round checkpoints, reporting
  the eigen-spectrum of the test Gram against 1e-6.

**Status:** qualified; check pending.

### D2. Health labels

- **per_dim_std_mean is linear in scale.** The ≥ 0.5 rule flips under rescaling (D1: clean label false
  at c ≤ 1.41, true at c = 2, with R² unchanged).
- **effective_rank (spectral entropy)** is invariant to isotropic rescaling and changes under anisotropic
  rescaling (D3: 2.86 → 7.995 when the two dominant dimensions are shrunk).
- The reconciliation's statement here is confirmed.
- **New reporting criterion (proposed, explained).** A scale-free health metric replaces the absolute
  threshold:
  - effective rank of the correlation (not covariance) matrix;
  - the per-dimension std ratio, released over the untreated reference on the same rows.
- The historical rule is still reported under its own name.

**Status:** confirmed and extended.

### D3. Historical counts

Final iterate and best validation are not interchangeable. Keep them separate:
- 56/60 and 54/60 strict;
- 6/60 and 5/60 clean;
- 7/60 under the mixed rule.

ABS and INCR are likewise different criteria. Keep them separate:
- submitted 26/33 and 22/33;
- rebuttal 19/33 and 8/33.

No new historical count is introduced here. The only additions are recounts labelled by rule:
- HL-04: 4/60 and 2/60 LR held-out above 1 pp and 2 pp;
- HL-05: 5/24 held-out ridge R².

**Status:** unchanged; recounts added.

### D4. New: training-time and Cotter R² has a small-sample null level of about r/255

**Finding:**
- **Training constraint.** It is the in-sample ridge R² on each minibatch of 256
  (`v2_trainer.py:880`).
- **Cotter feasibility.** It averages the same statistic over validation minibatches (`:1090`).
- **Null level.** With no attribute signal, a representation with r non-degenerate directions reads
  R² ≈ r/255 (`batch_r2_bias_check_output.json`):

  | r (non-degenerate directions) | Null R² |
  |---|---|
  | 64 | 0.255 |
  | 16 | 0.063 |
  | 8 | 0.029 |

- **Consequence.** Meeting τ = 0.05 per batch requires r ≲ 12 *regardless of leakage*.

**Hypothesis, not established.** The constraint rewards rank reduction. This is consistent with:
- the low stored effective ranks (1.06–8.82 on R5 Adult, best.pt health in `per_seed_results.json`);
- 0 feasible epochs on most Cotter selections.

Testing it needs a frozen forward pass computing the same batch statistic with permuted attribute labels
on stored checkpoints, plus the val eigen-spectrum. Pending approval.

**Status:** pending.

### D5. CelebA validation R²

See A1/FC-16. "Val-partition R²" is an in-sample statistic, not a held-out one.

**Status:** corrected.

---

## Coordinator addendum (2026-10-02, after integrating all three roles)

These items were checked by the coordinator against primary artifacts. Earlier records are left unchanged.

| # | Earlier statement | Corrected statement | Evidence | Status |
|---|---|---|---|---|
| O1 | "The AAAI audit: 64 / 59 / 51 of 67 fail at 0.52 / 0.55 / 0.60" (reconciliation, VERIFICATION_REPORT F1) | **Both forms must be reported.** The stored count is 64/59/51 of 67, but the 67 rows include repeated measurements: E2 and E4S1 at σ = 1, 2, 8 point to the same score files, giving identical AUCs. Counting each distinct measurement once gives **62 / 57 / 51 of 64**. | `notes/evaluator/recount_aaai67_summary.json` (input sha256 match the drive inventory; max AUC deviation 1.7e-5) | independently recomputed (evaluator); duplicate mapping read from the spec builder |
| O2 | "No held-out linear probe was ever run" (UPDATED_MEETING_BRIEF.md, REBUTTAL_WORK_RECONCILIATION.md) | Held-out **linear classifiers** exist: logistic regression train→test on the submitted R5/R7 final.pt — 4/60 disallowed pairs beat majority by > 1 pp, 3 of which pass the native check (C3). The missing measurement is a held-out version of the native quantity itself (held-out R², R02). | `notes/methodology/heldout_linear_inventory.csv` HL-04; recount rerun by the coordinator, byte-identical output | independently recomputed |
| O3 | "Native R² rescaling leaves leakage unchanged" (reconciliation C4) | Holds for unrestricted least squares and AUC. PCRL's native score uses a fixed absolute ridge penalty (1e-6) on the unnormalised Gram, so it **does** move under rescaling when signal lies in directions with eigenvalue comparable to the penalty (coordinator synthetic: 0.065 → 0.194 for scale 0.5 → 2, OLS fixed at 0.208). Whether the real representations have such directions needs a frozen forward pass with an eigen-spectrum (pending, P-3). | `stored_model_eval/tests/test_14_releases_and_ridge_scale.py` | synthetic, verified |
| O4 | "Cross-purpose retrain checkpoints: referenced S3 archive only, probably lost" (MISSING_FILES_EXACT #10) | 18 checkpoints (best.pt + final.pt × 9 runs) were on the laptop in git-ignored `PCRL/checkpoints_archive/`, as the only copy. They are now backed up to the drive and hash-verified. | coordinator check: 18 `.pt` present; no match in any drive inventory | checked |
| O5 | "Erase-pilot / VICReg / rank-8 checkpoints: referenced S3 archive only" | **Never saved**: the launchers synced only `results/` (`d3921121:infra/cross_purpose/user_data.sh:237-240`). The metrics survive; the weights do not. | CHECKPOINT_INVENTORY.csv class d | checked against code |
| O6 | "S3 objects in the lifecycle bucket are probably lost" | Not established. S3 was not inspected (credentials expired); entries are "not verified — S3 not inspected". | CHECKPOINT_INVENTORY.csv class c | wording corrected |
| O7 | Reconciliation recount helpers treated as correct | Defects found, not fixed in the historical branch: `F_common.worst_pair` returns 0.0 when no pair is supported (reads as "no leakage") and folds AUC to max(a, 1−a) by default (upward bias under the null); `F_common.macro_ovr` silently averages present classes; `D_common` computes effective rank from s/Σs while PCRL uses s²/Σs². None of these changes a reported count in VERIFICATION_REPORT.json. They matter for any reuse. `stored_model_eval` implements the corrected forms and tests them. | `notes/evaluator/evaluator_notes.md` | checked against code |
| O8 | "AAAI pilot can use the NeurIPS headline model" | The cached and pilot checkpoints are **Round 4** seed 0, the encoders durable-guarantees audited. The NeurIPS headline Round 5/7 checkpoints are on the drive (class a) and are not part of this pilot. | CHECKPOINT_INVENTORY.csv; `notes/artifacts/round4_checkpoint_metadata.json` | checked |
| O9 | Backup of laptop-only launch configs "pending" | 102 files (26.5 MB) were preserved to a private drive directory. The coordinator re-hashed every copy with uncached reads (F_NOCACHE) against the originals: 102/102 match. A cold-disk read after a remount was attempted but the unmount was refused by a system process (not forced). | BACKUP_STATUS.md; private manifest | checked |
