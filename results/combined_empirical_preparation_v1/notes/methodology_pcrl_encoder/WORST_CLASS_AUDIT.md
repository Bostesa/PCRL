# Worst-class / Dominant-Axis Audit and threshold definitions — PCRL encoder lineage

Role: methodology (PCRL encoder), sub-audit B. Date 2026-10-01. Read-only inspection of
`Bostesa/PCRL` refs; no training, no cloud. Status labels: **[src]** source inspected,
**[stored]** stored results recomputed, **[fixture]** synthetic fixture run this phase,
**[hyp]** hypothesis not verified on checkpoints.

Manuscript cited as **PDF(23) l.N** = printed line number in
`~/Downloads/Formatting_Instructions_For_NeurIPS_2026 (23).pdf` (text-identical to `(24).pdf`;
latest dated version, treated as the submitted one; the `(21)` source tree is an earlier
state — see parent notes). Fixture: `worst_class_fixture.py` (this directory, numpy only;
output reproduced below). Recount scripts: scratchpad `mpe/forks/recount_B.py`, `recount_B2.py`.

---

## 1. Numbering history of the identity

| Draft | Label in PDF text |
|---|---|
| (9) (May 4) | Theorem 5 (Convex-Combination Identity) |
| (13) (May 5) | **Theorem 4** (Convex-Combination Identity) — the "Theorem 4" the meeting summary refers to |
| (17)–(20) (May 6) | Proposition 3 (20 has both 3 and 4 during renumbering) |
| (21)/(22) (May 7 03:25) | Proposition 4 (`\label{prop:darb}`) |
| (23)/(24) final | **Proposition 5** (PDF(23) l.267–271); corollaries Appendix E (l.552–556) |

Same statement throughout: R²_onehot = Σ_k w_k R²_OvR,k, w_k = π_k(1−π_k)/Σ_j π_j(1−π_j).

## 2. Implementations found

### 2.1 Evaluation-time (produced every number in Tables 1, 5, 7–9)

| Quantity | Location (origin/main 55e4cb1d1) | What it computes |
|---|---|---|
| One-hot "linear R²" certificate | `pcrl/purposes/verification.py:80-103` (`LinearComplianceCertificate.check`) | K = `max(label)+1` one-hot columns (all K, not K−1); centre H and Z; ridge W = (HᶜᵀHᶜ + 1e-6·I)⁻¹HᶜᵀZᶜ; **pooled** R² = 1 − Σ_k SSE_k / Σ_k SST_k; clamp at 0. Fit **and scored on the same rows**. |
| Which rows | `pcrl/evaluation/certificates.py:511-512` (`generate_report`); `scripts/eval_round4_dominant_axis.py:158-173` | the **test split** representations (in-sample on test; no train→test generalisation). |
| Per-class OvR R² and R²_DA | `pcrl/evaluation/certificates.py:47-116` (`compute_dominant_axis_r2`) | for each k, z_k = 1[y=k], same ridge 1e-6, in-sample on test; per-class R² clamped at 0; **R²_DA = max_k R²_OvR,k**, `argmax_class`; priors = empirical test-split π_k. |
| Identity "check" | `scripts/eval_round4_dominant_axis.py:111-119,187-199,271-278` | predicted = Σ w_k R²_OvR,k with w from test priors; residual = predicted − observed; "match" if |residual| < 0.01. |
| Checkpoint | `scripts/eval_round4_dominant_axis.py:137-138` | `checkpoints/v2_<ds>_<tag>_s<seed>/final.pt` (epoch 199). |
| MLP-DA (Tables 7–9 ∆MLP-DA) | `pcrl/evaluation/certificates.py:119-250` | per-class 2-layer MLP trained on train reps, accuracy on test minus **test-set** majority max(π_k,1−π_k) (pre-fix). Post-fix uses a train-chosen majority predictor (`research/pcrl-submission-finish-v1:pcrl/evaluation/certificates.py:115-192`). |
| Strict / combined pass | `experiments/run_v2_dataset.py:372-373` | strict: `linear_r2 < 0.05`; combined: also `best_auditor_acc − majority < 0.02`, majority from **train+test pooled labels** (`certificates.py:515-519`), best over auditors on test. |

**Precise definitions (as run).**
- *Dominant axis* is **not** an eigen-direction. It is the class index k* maximising the
  in-sample ridge R² of the single indicator 1[A=k] on the test-split representation:
  R²_DA = max_k R²(H, 1[A=k]). No normalisation beyond the standard per-column R²
  (SSE_k/SST_k, SST_k = n π_k(1−π_k)). It is not a leading generalized eigenvalue / canonical
  correlation; it does not search over contrasts.
- *Worst class* in the paper = this argmax class (Tables 7–9 column `arg max_k`). The paper
  text calls the HMDA case "the rarest training-population race class" (PDF(23) l.261) but the
  priors used are **test-split** priors.
- *One-hot R²* is the **variance-weighted pooled** R² (sum of SSE over sum of SST), i.e.
  sklearn `multioutput='variance_weighted'`, not sklearn's default `uniform_average`.

### 2.2 Training-time (what the proxy-Lagrangian constrains)

| Quantity | Location | What it computes |
|---|---|---|
| Joint constraint | `pcrl/training/losses.py:496-541` (`VerificationRegularizer.forward`) | same pooled formula, **float32**, ridge **1e-4**, on each **minibatch of 256**, in-sample. |
| Per-class OvR constraints (K ≥ 6 only) | `losses.py:543-565`; `v2_trainer.py:384-413,854-877` | per-column version; K independent constraints with own duals. |
| Degenerate / missing-class handling (pre-fix) | `v2_trainer.py:844-852` (batch with one observed class → all 0), `v2_trainer.py:859-867` (missing **highest** classes padded with 0) | a missing **middle** class is not padded — see §5. |
| Dual update | `v2_trainer.py:952-954`; `pcrl/training/proxy_lagrangian.py:93-101` | per minibatch, λ += 0.02·(R²_batch − 0.05), projected to [λ_min=5, 1000]. |
| Validation R² (Cotter selection) | `v2_trainer.py:1068-1095` | high-K pairs report **max per-class** batch R², low-K pairs pooled, averaged over val batches. |

Other implementations (same pooled convention, not used for the tabular headline):
`pcrl/vision/r2_helper.py:12` (`linear_r2`, ridge 1e-6), `pcrl/language/leace_warmstart.py:150`,
`pcrl/training/laftr_proxy_trainer.py:67,611` (LAFTR-hard-R²: max per-class OvR for high-K),
`pcrl/training/v2_trainer.py:282-301` (`_linear_r2_train`, warm-start diagnostics),
`pcrl/language/per_class_ovr_constraints.py` (BIOS only; separate module per its docstring l.1-11).

## 3. Proposition 5 (convex-combination identity): correctness

**Statement is correct for the quantity the code computes**, with weights
w_k = π_k(1−π_k)/Σ_j π_j(1−π_j) where π_k are the class proportions *of the rows on which both
scores are computed*. Reason: multi-output ridge/OLS with a shared design and shared scalar
penalty decouples column-wise, and SST_k = n π_k(1−π_k); so
1 − ΣSSE_k/ΣSST_k = Σ_k (SST_k/ΣSST)(1 − SSE_k/SST_k). It is an algebraic identity, not an
empirical finding.

Conditions (fixture §A, **[fixture]**):

| Variant | Identity residual |
|---|---|
| in-sample pooled, ridge 1e-6 (as run) | −2.6e-17 (exact) |
| in-sample, ridge α = n (any shared α) | −1.7e-16 (exact) |
| held-out (fit train, score test, SST from test mean, unclamped) | −1.9e-16 (exact, with **test** priors) |
| held-out with per-class scores clamped at 0 (as the code clamps) | **0.0695** (breaks) |
| sklearn default `uniform_average` (equal weights) | not the pooled score (0.020 vs 0.0044) |
| K−1 one-hot columns (drop rarest class) | different pooled score (0.0025 vs 0.0044) |

Consequences.
1. "We verified the identity within tolerance 0.01 in 33/33 multi-class cases (max residual
   0.0021)" (PDF(23) l.278, l.590-591) is a **code-consistency check, not evidence**. Recount
   **[stored]**: 33/33, max |residual| = 0.0021078 on Adult employment_analysis/age_group seed 0,
   where R²_onehot = 0.00538 — i.e. the residual is **39 % of the quantity**. Because the
   identity is exact, a residual of that size means the two code paths
   (`verification.py:94-95` matrix solve vs `certificates.py:96-97` solve-then-multiply) disagree
   numerically on that representation (likely an ill-conditioned Gram matrix); the 0.01 tolerance
   is 2–10× larger than most audited R² values and cannot detect such errors.
2. "with equality only when every class leaks equally" (l.273): correct only over classes with
   w_k > 0 (classes present in the scored rows).
3. The Appendix-A use of the identity inside the Proposition 3 accuracy proof does not rescue
   that proof (Prop 3 is retired; see `research/pcrl-submission-finish-v1:docs/ACCURACY_CERTIFICATE_RETIREMENT.md`,
   `tests/test_accuracy_bound.py:22-46`). τ = 0.05 is justified in the paper by Prop 3
   (PDF(23) l.202-205), so the threshold's stated rationale is retired with it.

## 4. Contrast gap: what the worst-class score misses

R²_DA is a max over the K class indicators. Linear leakage of A is fully described by the
canonical correlations between H and the (K−1)-dim centred one-hot Z; the largest linear
leakage over all contrasts c is ρ₁² = max_c R²(H, cᵀz) ≥ R²_DA ≥ R²_onehot. Leakage of a
*grouping* of classes (e.g. "age ≥ 50", "minority vs White") can be large while every single
class indicator is weakly predicted. **[fixture §B]** (n = 200,000):

| Case | R²_onehot | R²_DA (max class) | ρ₁² (best contrast) | binary group recovery (best threshold) vs majority |
|---|---|---|---|---|
| K=10 balanced, h = 1[A ≤ 4] exactly | 0.111 | 0.112 | **1.000** | 1.000 vs 0.501 |
| K=10 balanced, noise calibrated to R²_DA = 0.045 (both metrics **pass τ=0.05**) | 0.045 | 0.046 | **0.407** | **0.796** vs 0.501 |
| Diabetes age_bucket priors (71,518 deduped rows), group age<60, R²_DA = 0.045 | 0.021 | 0.045 | 0.109 | 0.698 vs 0.666 |

So the "dominant-axis" repair still certifies a representation from which a linear reader
recovers a balanced binary function of the attribute at ~80 % (vs 50 %). The general
population bound is R²_OvR,k = ρ²·π_k(1−q)/(q(1−π_k)) for a group of mass q containing k, so
with many small classes the per-class scores fall like π_k while the contrast stays at ρ².
**Repair**: report ρ₁² (top squared canonical correlation, = leading generalized eigenvalue of
Σ_ZZ⁻¹Σ_ZHΣ_HH⁻¹Σ_HZ) alongside R²_onehot and R²_DA, plus held-out logistic/linear probes on
pre-registered groupings; the paper's "maximum one-vs-rest leakage" (l.199) must not be
described as the worst linear leakage.

## 5. Unsupported / missing / constant classes

**Pre-fix (origin/main, the code that produced the paper):**
- Eval `compute_dominant_axis_r2`: K = max(label)+1 (`certificates.py:88`). A class absent from
  the scored split but below the max label gets z ≡ 0 ⇒ SST = 0, SSE = 0 ⇒ R² = 1 − 0/1e-12 =
  **1.0** (fixture §D reproduces 1.0); a missing top class silently disappears from K.
  Stored R5/R7 audits contain **no** zero-prior classes and no per-class value ≥ 0.999
  (**[stored]**), so the published tables are not affected by this path.
- One-hot certificate: an all-zero column contributes 0 to both sums — the class is silently
  dropped; a constant attribute gives R² = 0 (would "pass").
- **Training per-class constraints (affects Diabetes R6/R7, the rank-16→24 decision):** with
  batch 256, a middle class absent from the batch scores **R² = 1.0** in float32
  (`losses.py:562-565`; fixture §D), while a missing top class is padded with 0
  (`v2_trainer.py:863-867`). Diabetes age_bucket k=0 ([0-10)) has π = 0.00215 after
  first-encounter dedup (computed from `data/diabetes/diabetic_data.csv`), so it is absent from
  57.6 % of batches. Expected dual from absences alone:
  1 + 200 epochs × 196 steps × 0.02 × 0.576 × 0.95 ≈ **430**. Stored final duals for k=0:
  **R7 420.2 / 419.4 / 420.2; R6 421.1 / 421.5 / 422.6** (`results/v2_diabetes_ROUND{6,7}/per_seed_results.json`
  → `per_seed[i].lambdas_final.quality_research__age_bucket__class_0`), *including R6 seed 1
  where every other class sat at the floor 5 and the cell passed*. The audit-time R² for k=0 is
  0.000 on every seed (`results/V2_DIABETES_ROUND7_VERDICT.md`). The "λ saturated at 420/421"
  that motivated "encoder could not comply on classes k=5,6,7 at LoRA rank 16 … bumping LoRA rank
  to 24" (PDF(23) l.663-666; `V2_DIABETES_ROUND7_VERDICT.md` header) is therefore the
  absent-class artifact on k=0, not k=5–7 (whose duals were 50–96 in R6). Cotter selection is
  also affected: all three R7 seeds report `cotter_selection.kind = "fallback"`,
  `n_feasible_post_warmup = 0` — the constraint set can never be satisfied while k=0's batch
  score is 1.0 in ~58 % of batches. **[stored + fixture]**; the downstream effect on
  representation collapse in quality_research (lowest per_dim_std 0.150/0.258, eff_rank
  1.75/1.48 on s0/s1) is **[hyp]**.

**Post-fix (`research/pcrl-submission-finish-v1`, contains 5f162ab37 "Repair PCRL evaluation"):**
`compute_dominant_axis_r2(..., num_classes=K)` (`certificates.py:45-97` on that ref) requires a
fixed schema; classes with zero support, full support or zero variance get NaN and
`valid_mask=False`; `r2_da` is NaN unless `coverage_complete`; `observed_r2_da` reports valid
classes only. `tests/test_scoring_support.py:15-27` (absent middle and tail classes → NaN,
gradients finite), `:30-46` (constant class cannot pass any score), `:49-59` (partial schema fails
aggregate and DA), `:62-74` (NaN/inf constraint values do not move duals, `all_satisfied` false),
`:120-160` (MLP-OvR missing class in either split → NaN; majority chosen on train labels).
The fix is **not merged to origin/main** (main still ships the pre-fix path) and **no headline
result was re-run under it**.

## 6. Precision and estimator floors

- Eval casts reps to float64 (`certificates.py:84`); reps are extracted as float32 from torch
  (`run_v2_dataset.py:191-198`). Fixture §C: float32 input changes eval R² by < 1e-3 at all
  tested scales — **not material** for normal representations.
- **Ridge is not scale-invariant.** Eval ridge 1e-6 shrinks R² of a perfectly informative
  representation only when per-dim scale ≲ 1e-6 (R² = 0.38 at scale 1e-6, n = 13,661); training
  ridge 1e-4 on B = 256 shrinks it at scale ≲ 1e-4 (R² = 0.51 at 1e-4, 0.08 at 3e-5). Stored
  example: HMDA underwriting seed 0 at the Cotter-selected checkpoint has per-dim std mean
  1.2e-5, min 1.7e-8, l2-norm std 0.0, eff-rank 1.00 (`results/v2_hmda_ROUND5/per_seed_results.json`
  `per_seed[0].per_purpose_health.underwriting`) and `linear_r2 = 0.0`: a numerically constant
  representation that "passes". Loan-decision accuracy 0.892614 is identical on seeds 0 and 1
  (both rank-1 underwriting representations) — consistent with a constant predictor.
- **Minibatch in-sample null floor.** In-sample OLS R² of any target on r informative-free
  regressors has expectation ≈ r/(B−1). Fixture §C (50 batches, B = 256, d = 64, H independent
  of A): rank 64 → 0.253; 32 → 0.127; 12 → 0.046; 4 → 0.016; 1 → 0.0035 (float32 = float64).
  The training constraint R²_batch ≤ 0.05 is therefore **unsatisfiable by any full-rank 64-d
  representation, even one carrying zero information about A**; it can only be met by reducing
  the number of non-negligible directions to ≲ 12 (or shrinking them below the ridge scale).
  **[hyp]** This is a candidate mechanism for the paper's "compliance via collapse" (49/56)
  that the paper attributes to λ_min = 5 and the LoRA erasure floor (PDF(23) l.283-291). It is
  consistent with stored eff-ranks of 1.0–8.8 on R5/R7 and with the BIOS d/N saturation already
  recorded in memory, but has not been checked on the tabular checkpoints.

## 7. Thresholds: τ, strict, cleanly compliant, combined

| Criterion | Definition as run | Location | First appears |
|---|---|---|---|
| strict | test-split in-sample pooled ridge R²_onehot **< 0.05** (final.pt for the paper tables) | `run_v2_dataset.py:373`; `eval_round4_dominant_axis.py:158,172` | `a96ee0e10` 2026-03-25 (initial pipeline, `run_adult.py:234-235`) |
| combined | strict **and** best-of-auditors test acc − pooled train+test majority **< 0.02** | `run_v2_dataset.py:372-373` | same, 2026-03-25 |
| training τ | batch R² ≤ 0.05 per pair (or per class) | `v2_trainer.py:114`, `run_v2_dataset.py:278` | `7cd893f7d` 2026-04-27 |
| health ("cleanly compliant") | per-purpose **per_dim_std_mean ≥ 0.5** and **effective rank ≥ 2** on test reps; eff-rank = exp(entropy of normalised squared singular values) | `run_v2_dataset.py:81-82,168-188,437-450` | `53feba0f2` 2026-04-27 |
| K ≥ 6 per-class gate | attributes with ≥ 6 classes get per-class constraints | `v2_trainer.py:126-137` | `f542c064c` 2026-04-30 (after R5 Diabetes failure) |
| DA pass | R²_DA < τ | `scripts/tier1_tau_sensitivity.py:70-72` | `a3875c618` 2026-05-07 06:49 (1 min after PDF(23) creation) |

Registered vs chosen. No pre-registration exists for the encoder lineage (searched `*.md` on
origin/main; only a BIOS gate mentions "pre-registered"). τ = 0.05 and ∆ < 0.02 predate all V2
results (2026-03-25), so they were not tuned to the headline; the paper's *justification* via
Prop 3 was added later and is retired. Health thresholds were fixed 2026-04-27, before Round 4/5;
the "three nested bars" reporting (Table 1) was introduced in the May 6–7 drafts after results.
The K ≥ 6 gate was chosen after seeing Round 5 and is documented as chosen so that it "leaves
Adult/HMDA (max K=5) unchanged" (`v2_trainer.py:134-136`; PDF(23) l.660-662) — it therefore
**excludes the 5-class HMDA race attribute on which the paper's own DA pathology (0.288) occurs**.
`per_dim_std ≥ 0.5` is not scale-invariant (rescaling h with a compensating head changes it
without changing information); eff-rank is.

τ sweep. `results/tier1_analyses/tau_sensitivity.json` re-thresholds the *same* models trained
at τ = 0.05; it is not a retraining sweep. Recount **[stored]** from the three
`dominant_axis_audit.json` files (final.pt, 60 rows): strict / DA pass =
τ 0.01: 37/31; 0.025: 52/49; 0.05: 56/55; 0.1: 59/56; 0.2: 60/59; mean passing R²_onehot /
R²_DA 0.0037/0.0041, 0.0070/0.0085, 0.0089/0.0113, 0.0118/0.0122, 0.0136/0.0171 — **all match
PDF(23) Table 5**. `<` vs `≤` makes no difference at 0.05.

**Cross-checkpoint join in "cleanly compliant" (recount [stored]).** The 7/60 cleanly-compliant
count joins **R² from final.pt** (`dominant_axis_audit.json`) with **health from the
Cotter-selected best.pt** (`summary.json` health_notes / `per_seed_results.json`, produced after
`run_v2_dataset.py:336-339` reloads best.pt): `scripts/identify_collapse_cells.py:58-62,78-88,104-115`.
Recount: final-R² + best-health = **7** (matches); same-checkpoint best.pt R² + best.pt health =
**5**; strict at best.pt = **54/60** vs 56/60 at final.pt (disagreeing cells: Adult
employment_analysis/age_group s0 0.0619→0.0054 and /marital_status s0 0.0806→0.0033 — two of the
three Adult "cleanly compliant" cells fail R² at the checkpoint whose health was measured).
Of the 7 cleanly-compliant cells, **4 have post-hoc auditor ∆ ≥ 0.02** (Adult employment
age_group s0 +0.130, marital_status s0 +0.264; HMDA underwriting race s2 **+0.283**, ethnicity s2
+0.195, all measured at best.pt); only **3/60** satisfy R² + health + ∆aud (the 3/60 appears in
draft (22) Table 14 and in final PDF(23) App. R Table 14, but not in the main text — parent-role correction). Combined pass (R² and ∆ < 0.02, best.pt): 32/60.

## 8. Dominant-axis headline numbers (recount [stored])

Sources: `results/v2_{adult,hmda}_ROUND5/dominant_axis_audit.json`,
`results/v2_diabetes_ROUND7/dominant_axis_audit.json` (sha256 0d9746fd…a9f, 5c854f2f…077,
785dc5bd…00e), all rows epoch 199 (final.pt).

| Claim (PDF(23)) | Recomputed | Match |
|---|---|---|
| 1 hidden case (l.296-298, Table 1) | HMDA underwriting/race s1: 0.0268 vs 0.2880 | yes |
| median amplification 1.43× over 33 multi-class pair-seeds (l.279) | 1.4328 | yes |
| worst case 12.7× (l.60, l.280) | 12.65 = 0.03323/0.002627, **HMDA underwriting/race seed 0** | yes (rounding) |
| 12.7×, 10.7×, 3.2× on seeds 0,1,2 (l.280) | 12.65, 10.74, 3.22 | yes |

Notes. (i) The 12.7× "worst case" is the ratio of two sub-threshold numbers (0.0026 vs 0.033,
both < τ); amplification ratios are unstable when the denominator is near the estimator floor
and should not be headlined as leakage. (ii) All three seeds' argmax class is k = 4,
π_k = 0.00915 on the 13,661-row test split (≈125 positives), w_k = 0.0179. In
`experiments/prepare_hmda.py:109-118`, class 4 is **"Joint"** (applications with co-applicants of
different races), not a single race; leakage of "Joint" is plausibly explained by co-applicant
input features, which the paper does not discuss. (iii) The DA metric is measured on final.pt,
while the representation-health and auditor columns are from best.pt (§7); for seeds 0 and 1
the best.pt underwriting representation is rank-1 / constant. (iv) Over all 60 cells the median
ratio is 1.22 (binary cells are 1.0 by construction).

## 9. Fixture output (verbatim summary of `python3 worst_class_fixture.py`)

```
A: pooled 0.004391; Σw·R²_OvR 0.004391 (resid −2.6e-17); uniform_average 0.0203;
   ridge α=n resid −1.7e-16; K−1 columns 0.00252; held-out unclamped resid −1.9e-16;
   small held-out with clamp: resid 0.0695 (per-class [0.19,−0.26,0.21,−1.33,−inf]: class 4 absent in test → −inf)
B: see §4 table
C: null floor (B=256,d=64): rank 64 0.253 | 32 0.127 | 12 0.046 | 4 0.016 | 1 0.0035
   eval ridge 1e-6 informative rep: scale 1→1.0, 1e-5→0.985, 1e-6→0.380
   train ridge 1e-4 batch: scale 1e-3→0.994, 3e-4→0.911, 1e-4→0.514, 3e-5→0.084
D: absent middle class R² = 1.0 (pre-fix); absent tail padded 0.0; P(k=0 absent | B=256) = 0.576;
   implied λ_k0 ≈ 430 vs stored 420.2/419.4/420.2 (R7)
```
