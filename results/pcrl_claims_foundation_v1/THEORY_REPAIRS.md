# Theory repairs

Terminal 3, 2026-09-22. Scope: the six propositions of the original NeurIPS submission ("One Encoder, Many
Purposes", local PDF, text extracted with `pdftotext`; the line numbers below are lines of that extraction)
and the mathematical statements in the current manuscript (`research/pcrl-submission-finish-v1` @
`30a6fd19e17453bc8c421a65491b5b482ab9291f`, `papers/pcrl_satml_final_v1/main.tex`). Exact fixtures are in
`analysis/pcrl_claims_foundation_v1/exact_fixtures.py`, with outputs in `FIXTURE_OUTPUTS.json`; each fixture
is named below as F*n*. A fixture instantiates a statement or refutes one. It is not a proof of a universal
claim. LaTeX fragments are in `tex/`.

Labels: **valid** (correct under the stated assumptions); **valid, misapplied** (correct, but used for an
object it does not describe); **refuted**; **elementary / prior** (true, but not a contribution).

## Summary

| Item | Source | Verdict | Replacement |
|---|---|---|---|
| P1 Zhao–Gordon "near-optimality" | NeurIPS §3 | valid (derivation below), prior | "lower bound on summed group error given the DP gap"; not a certificate of method optimality |
| P2 Joint LEACE composition/optimality | NeurIPS §4, App. D | valid: LEACE Thms 4.1–4.3 applied to a concatenated target; the edit rank is ≤ Σ(c_i−1), **equal to rank Σ_hZ** | the rank *upper* bound is not a LoRA rank *requirement* (see P4) |
| P3 Linear Compliance Bound (R² → accuracy) | NeurIPS §4, App. A | **refuted** (F1) | the convex-loss statement (LEACE Thm 3.1), which excludes 0–1 accuracy |
| P4 LoRA Erasure Floor | NeurIPS §5.2 | valid for h=(I+BA′)f0; **misapplied** to the trained multi-layer LoRA (F12) | restrict it to a single multiplicative edit of a fixed representation (the warm start) |
| P5 Convex-combination identity + DA corollary | NeurIPS §5.3, App. E | valid, elementary (variance-weighted R²) | state the conventions; DA = max OvR, a lower bound on the best direction (F3) |
| P6 Cross-purpose linear-leakage bound | NeurIPS App. C | valid, tight (F11), elementary (Rayleigh–Ritz); form (III) needs its premise stated precisely | restore it in the current manuscript as a lemma under stated assumptions |
| Exact-zero composition | main.tex:153-156 | valid; nonlinear recovery can still be exact (F4) | keep; add the λ_min lemma |
| "floor on what is recoverable" | main.tex:110-111 | **wrong** as written (F9) | absolute recovery is a floor on I(S;view); the increment bounds nothing |
| Convexity of the finite programme | main.tex:195-200 | valid for a fixed table, cost and conditioning partition | add the solver-status and coarse-cell qualifications |
| Coarsened conditioning | main.tex:210-214 | valid qualification (F5) | give the partition size (2 cells for A, 4 for AB) |
| Replacement vs extension fixture | main.tex:216-221 | valid (F6) | keep, labelled as a design example |
| Refinement embedding | study METHOD.md:11 | valid (F7) under identical rows, costs and decoder | feasible-set ordering of the fitted objective only |
| Expected-loss scoring | main.tex:105-107 | valid (F8); the code matches (audits.py:167-180, finite.py:85-109 @ f4bdf4cd) | keep |
| Marginal SEX, RAC1P constraints ⇒ joint protection | study DATA_USE.md | **not implied** (F5 with S1,S2 in place of S,H) | already corrected in INTERPRETATION_ADDENDUM item 4 |

---

## T1. The retired accuracy guarantee (original P3)

**Original statement** (NeurIPS App. A, proof at extraction line 1322; code `certified_accuracy_bound` on
public `origin/main` 55e4cb1): if the one-hot affine least-squares R² < ε, then every linear classifier's
accuracy is at most max(π_maj, π_maj + sqrt(ε k π_maj(1−π_maj))).

**Refutation (F1, exact).** Take a balanced binary A and 20 rows: A=1 has h=1 on 9 rows and h=−9 on 1; A=0
has h=−1 on 9 rows and h=9 on 1. Both conditional means are 0, so Cov(h,A)=0 and the affine least-squares R²
is exactly 0. The threshold h>0 classifies 18/20 correctly, against a bound of 50%. The proof fails at its
Step 2→3: a thresholded classifier applies a nonlinear function to w′h, and zero correlation constrains only
the affine score.

**Valid replacement** (Belrose et al. 2023, LEACE, Thm 3.1 with Thm 3.4). If Cov(h,A)=0 (equivalently, equal
class-conditional means), then no affine predictor η(h)=b+Wh attains lower expected loss than the best
constant, for **every loss convex in the prediction**: squared loss, and log loss applied to affine logits.
The 0–1 loss of a thresholded score is not convex, so accuracy is not covered. The current manuscript's
"about squared loss only" (main.tex:150) is too narrow in one direction and correctly excludes accuracy.
Replacement: "zero cross-covariance means no affine predictor improves on the best constant for any loss
convex in its output [LEACE Thm 3.1]; thresholded accuracy is not such a loss, and the counterexample shows
it can reach 90%."

**Code status.** Retired (raises) from `5f162ab3` onward on the research branches. It is still **active on
public `origin/main` 55e4cb1 and local `main` 0eee48f**: the docstring states the theorem, and
`generate_report` calls `NonlinearComplianceCertificate.check`. Separable fix:
`patches/0001-retire-accuracy-guarantee-on-main.patch`. Its three API tests fail on unpatched main and pass
after the patch.

## T2. Aggregate one-hot versus per-class and dominant-axis scores (original P5)

**Statement.** Let H be n×d and Y ∈ {1..K}. Let z_k be the indicator columns, centred by their means over the
*same rows*. Fit a common affine ridge map with penalty λ≥0 (λ=0 gives OLS) on the rows *being scored*. Write
RSS_k and TSS_k = n·π̂_k(1−π̂_k) for column k's in-sample residual and total sums of squares. Then

  R²_onehot := 1 − Σ_k RSS_k / Σ_k TSS_k = Σ_k w_k R²_k, with w_k = π̂_k(1−π̂_k)/Σ_j π̂_j(1−π̂_j) and R²_k = 1 − RSS_k/TSS_k.

*Proof.* Multi-output ridge decouples by column, so RSS_k is also the per-class fit's residual. Then
Σ_k RSS_k/Σ_j TSS_j = Σ_k (TSS_k/ΣTSS)(RSS_k/TSS_k). ∎ This is the "variance-weighted" multi-output R²
(e.g. scikit-learn `multioutput='variance_weighted'`), so it is **elementary**, not a new result. F2
verifies it exactly for OLS and for ridge λ=1/10, and shows that uniform weights break it.

**Conditions under which the identity fails or needs restating:**
- *Clipping.* Held-out R² can be negative, and max(0,·) applied per class or to the aggregate breaks the
  identity. In-sample ridge R² is ≥0, so clipping never binds there.
- *Held-out scoring.* The identity holds with the **test-row** priors and TSS when both scores use the same
  fitted W on the same test rows. Weights from training priors break it.
- *Dropping a column (K−1 coding).* This changes the aggregate.
- *Absent classes.* w_k=0 and R²_k is undefined. The current code returns NaN and fails closed (tested in
  `tests/pcrl_claims_foundation_v1`).
- *Different λ, centering or rows.* The identity holds only when both scores share all three.

**Consequence for rare classes.** With R²_k ≥ 0, R²_c ≤ R²_onehot / w_c, a bound that is vacuous when w_c is
small. F2b: priors (0.9, 0.09, 0.01), with one feature equal to the rare indicator. The rare class has
R²=1 while the aggregate is 111/1111 ≈ 0.0999, and aggregate/w_rare = 1.83 > 1.

**What the dominant-axis statistic is.** In the code (`compute_dominant_axis_r2`,
`pcrl/evaluation/certificates.py` @ 0176f149), R²_DA = max_k R²_k: the in-sample ridge (λ=1e-6)
one-versus-rest score, computed on the held-out test representations. It is **not** the maximum over all
linear combinations of the class indicators. That maximum is the largest squared canonical correlation
ρ₁² between H and the centred one-hot (pseudo-inverse on its (K−1)-dimensional range), and
R²_onehot ≤ R²_DA ≤ ρ₁². F3: three balanced classes with h = z₀ − z₁ give per-class (3/4, 3/4, 0),
aggregate 1/2, DA 3/4, and exact recovery of the contrast (ρ₁²=1). **DA can detect a large single-class
signal. It cannot certify the absence of a signal along a class contrast, a nonlinear signal, or an
out-of-sample one.**

**Numerical replay of the original Table I** (`DOMINANT_AXIS_REPLAY.json`, from the committed per-cell JSON
at 0176f149; SHA-256 values in SOURCE_MAP). This replays stored numbers; no model was re-run. The means
reproduce: Adult 0.0080/0.0114, HMDA 0.0192/0.0609, Diabetes 0.0125/0.0179. The 0.027/0.288 cell is HMDA
underwriting/race seed 1 (epoch 199). Its argmax class 4 is the **rarest** class (π̂=0.0092, w=0.018), so
aggregate/w = 1.50 and the class bound is vacuous there. The identity reproduces within 2e-5 on 31/33 cells.
Two Adult seed-0 cells miss by 4.5e-4 and 2.1e-3. At the generating commit 135e440e,
`LinearComplianceCertificate` solved on float32 representations, while `compute_dominant_axis_r2` cast to
float64. A synthetic reproduction of the two paths (`HISTORICAL_PRECISION_DIAGNOSTIC.json`) gives float32
residuals up to 1.6e-2 and float64 residuals of 1e-15. With the float64 reconstruction the Adult mean
aggregate becomes 0.0082. Interpretation only: the numbers are unchanged in the source, and the conclusion
is unchanged.

## T3. Composition

**Exact zero (valid).** If Cov(Z₁,S)=0 and Cov(Z₂,S)=0 on one distribution, then Cov([Z₁;Z₂],S)=0,
because the stacked cross-covariance is the stacked blocks. By LEACE Thm 3.4 the concatenation is linearly
guarded. This does **not** give independence: in F4, with independent fair signs V and S, the views Z₁=SV
and Z₂=V each have zero covariance with S, yet S = Z₁Z₂ exactly.

**Approximate leakage (valid bound, original P6).** For views h_p with Σ_p ≻ 0, let R be the block-whitened
correlation matrix (R_pq = Σ_p^{−1/2} Σ_pq Σ_q^{−1/2}). Then

  R²(concat; A) ≤ Σ_p R²(h_p; A) / λ_min(R),

with λ₊(R) (the smallest positive eigenvalue) in the singular case. The proof in NeurIPS App. C (Steps 1–5)
is correct. It is Rayleigh–Ritz after block whitening, so label it an **elementary lemma**, not a theorem of
the method. F4/F11: with ε=1/10, Z₁=V+εS and Z₂=V−εS give individual R² = 1/101 each, R_offdiag = 99/101,
λ_min = 2/101, and a bound of **exactly 1** = the joint R². The bound is tight, and the ill-conditioning is
visible: the condition number of Cov(Z) is 100 = 1/ε².

**Precise premise for the ridge form (III).** Form (III) bounds the Tikhonov-regularised concatenated score,
but its premise Σ_p R²(h_p;A) ≤ kε uses the **unregularised** per-purpose scores. A per-purpose *ridge*
audit ≤ ε does not imply that premise, because the ridge score is ≤ the OLS score. The lemma also needs
Σ_p ≻ 0, or pseudo-inverses on the support. Collapsed representations (low effective rank) are where this
matters. The empirical Table 3 of the NeurIPS paper used sample ridge statistics with λ=1e-6; describe it
as a numerical check of the sample form, not as the population lemma.

**Replacement for main.tex:153-156** (fragment `tex/composition.tex`): "Two views with exactly zero
cross-covariance with the same signal retain exactly zero under concatenation, but this gives neither
independence nor protection against nonlinear recovery (two such views can determine the signal exactly).
Approximate leakage composes with a conditioning penalty: R²(h₁,…,h_k) ≤ Σ_p R²(h_p)/λ_min(R), where R is
the block-whitened cross-view correlation. This is tight: h₁=N+δS and h₂=N−δS each have
R²=δ²/(1+δ²), λ_min(R)=2δ²/(1+δ²), and their concatenation recovers S."

## T4. Rank statements (original P2 and P4)

**P2** is LEACE Thms 4.1–4.3 applied to the concatenated target [Z₁|Z₂]: the eraser is feasible and
optimal for every positive-definite M simultaneously among affine erasers with zero cross-covariance. Its
edit rank is **rank(Σ_{h,[Z₁|Z₂]})**, which is ≤ min(d, Σ_i(c_i−1)) after centring. The actual rank can be
smaller. Examples: a class absent from the fitting rows; attributes with deterministic dependence (their
centred one-hots then share directions); a representation whose covariance has lower rank. "Joint
cardinality 13" for Diabetes quality_research is the upper bound Σ(c_i−1) = 4+9. The actual
cross-covariance rank was not recorded. Report it with the class support before calling 13 a "floor".

**P4** is correct for its own model, h = (I+BA′)f₀ with rank(BA′) ≤ r: the whitened Eckart–Young argument
gives R² ≥ Σ_{i>r} σ_i²/tr Cov(A_oh). But the trained encoder applies a rank-r LoRA to **every** Linear
layer of the backbone (`pcrl/models/lora.py` @ dbe0fdc: `self._linear_modules = [m for m in
backbone.modules() if isinstance(m, nn.Linear)]`), with BN/ReLU between them, and it trains all of them.
F12: a rank-1 edit of the *first* layer removes a rank-2 sensitive cross-covariance entirely (pooled R² goes
1 → 0) while preserving the task feature, which is below the rank-1 floor that P4 would assign to the output
layer. So P4 does not "explain" collapse or set a rank requirement for the trained architecture. It bounds a
single multiplicative edit of a fixed representation, such as the LEACE warm start written into one layer.
The Diabetes rank-8 erase-layer ablation (18/18 strict pass, commit 98c4d3c) is consistent with this
narrowing (there the erasure is in the 128-d backbone space).

Replacement: "For a single rank-r multiplicative edit of a fixed representation, erasure below
Σ_{i>r}σ_i²/tr Cov(A) is impossible. This does not bound multi-layer adapters with intervening
nonlinearities, which is what we train."

## T5. Original P1 (Zhao–Gordon restatement)

*Derivation.* For any classifier ŷ, write q_a for the law of ŷ given A=a. Then Err_a ≥ d_TV(p_a, q_a).
Also d_TV(p_a, q₁) ≤ d_TV(p_a, q_a) + d_TV(q_a, q₁) ≤ Err_a + Δ_DP. Summing over a, and using the term
a=1, gives Σ_a Err_a ≥ Σ_a d_TV(p_a,q₁) − (n−1)Δ_DP ≥ B_c − (n−1)Δ_DP. ∎ Valid, short and prior (Zhao &
Gordon). It lower-bounds the summed group error of a classifier *given its demographic-parity gap*. It does
not certify that the method is near-optimal for the paper's R² constraint, and the paper itself reports it
vacuous on 9 of 20 cells. The separate Sadeghi–Boddeti certificate was never implemented
(`results/reviewer_dropins/PAPER_PASTE_theory.md` lines 93–127 @ 0176f149). Current Correction 4 is accurate
for that item and should name it.

## T6. Purpose conflict and retention

A shared union eraser and separate per-purpose erasers solve different problems. The pinned Gaussian pilot
(`results/redesign_20260907_gaussian_v1/TABLE.md` @ 0176f149) shows the shared union destroying both
permitted tasks (R² −0.0005/−0.0004), while per-purpose LEACE retains them (0.9996/0.9997). The restricted
PCRL arm **ties** per-purpose LEACE to six decimals; its own note says a tie is expected. A retention claim
needs a permitted task that actually requires the attribute, measured by useful recovery. Omitting the
attribute from a prohibition list is not a retention test. Keep Correction 2 as written.

## T7. Fixed-service preservation and information accounting

*Architectural property.* If the release is the pair (H, Z), where H is produced by the same frozen code and
weights and Z is appended, then every H byte is unchanged. Terminal 1's Tier-A status reports byte parity
(PCA32, H_A, H_B and J recomputed byte-identical on every anchor and pool). That comes from its status
file and was not re-run by Terminal 3. **Any** baseline that
leaves H untouched (J, LEACE on a side channel, a constant) inherits this property, so it is not evidence
for Q. Identical outputs also do not preserve source accuracy under distribution shift, and do not imply
the extra channel is useful.

*Deployments.* Three are possible: append to H (the recipient holds H, then H plus Z); append to J (the
recipient already holds J, so its knowledge includes J whatever Z is); replace J (only coherent if J was
never released). Correction C9 records that no deployment of J exists, so J is an internal comparator. The
contract is therefore "append to H", and the J comparison is "which of two alternative additions is better",
not "Q repairs a released J".

*Nested views.* In the population, I(S; H,Z) ≥ I(S; H), and the Bayes log loss with (H,Z) is ≤ that with H.
A *selected finite* predictor need not respect this ordering (F9 shows the reverse failure: the increment
can be positive when I(S;Z|H)=0). Validation-stage ancestor inclusion (H-only candidates in every augmented
slate, `evaluation.py:316-333` @ f4bdf4cd) makes the selected augmented attacker at least as good as the
H-only one **on validation**. It does not force test-loss monotonicity, and choosing the best test candidate
afterwards would be selection on test.

*Measurement semantics (replaces main.tex:109-113; fragment `tex/measurement.tex`).* Absolute recovery is
the entropy baseline minus the held-out loss of an attacker fixed before evaluation. It lower-bounds
I(S; view) up to sampling error, because cross-entropy ≥ conditional entropy. The increment over the
H-restricted fit is a difference of two such lower bounds and bounds I(S;Z|H) in neither direction. F9: H
uniform on {−1,0,1}, S=1[H=0], Z=f(H). Then I(S;Z|H)=0, yet the H-only linear-logistic loss is
H_b(1/3)=0.6365 nats, and it falls to 0 with Z.

## T8. The finite programme: statement, convexity, what must be fixed

*Programme, in the code's notation* (METHOD.md:9, finite.py:273-431, mechanisms.py:186-237 @ f4bdf4cd).
Fix the fitted table p_w(s,c,t) for each weighting w ∈ {U, PWGTP} and role r ∈ {A/SEX, A/RAC1P, and in
policy C also AB/SEX, AB/RAC1P}; the cost table D[t,z]; the action dictionary (offsets a_z); and the
conditioning partition c = C_r (2 cells for A, 4 for AB). Then solve

  min_Q Σ_t p(t) Σ_z Q[t,z] D[t,z]  s.t.  Q ≥ 0, Q1 = 1,  I_Q(S; Z | C_r) ≤ δ  for every constrained r and w.

*Convexity.* a(s,c,z) = Σ_t p(s,c,t)Q[t,z] and b(s,c,z) = p(s|c)Σ_t p(c,t)Q[t,z] are affine in Q. So
I_Q(S;Z|C) = Σ a log(a/b) = D(a‖b) is jointly convex in (a,b), and hence convex in Q (F10 spot-checks
midpoint convexity on 200 random tables). The objective is linear, so the programme is convex. This relies
on the Markov structure Z–T–(S,C) and on everything listed above staying fixed. Refitting the teacher,
bins, decoder offsets or partition jointly with Q breaks the argument. A convex downstream Q problem does
not make the nonconvex teacher, or the selected configuration, globally optimal.

*Prior art for this form.* Calmon & Fawaz 2012 (convex finite-alphabet leakage–distortion programme);
Salamatian et al. 2015 (quantise, then solve over the channel); Erdogdu & Fawaz 2015 (a new release beside
fixed prior releases, incremental leakage, convex, Thm III.3). Because I(S;Z|C) = I(S;Z,C) − I(S;C) and C
is fixed, a conditional budget is a joint-view budget with a shifted constant. Rassouli & Gündüz give the
perfect-privacy nullspace criterion and an LP for *mutual-information* utility, not for linear cost (see
PRIOR_ART_MATRIX).

*Zero budget and zero mass.* A constant row law makes Z independent of (S,C), so the zero-budget problem is
always feasible (F7). A row for a code with p(t)=0 does not affect the modelled law (F7). The study's exact
integer nullspace verification (ZERO_CERTIFICATE_METHOD.md) is a finite empirical feasibility certificate.

*Solver evidence for the selected release* T0_L_0.01_a17: CVXPY/CLARABEL status `optimal` on anchors 0 and
2 and `optimal_inaccurate` on anchor 1. The replay finds maximum CMI excess 0, a simplex error of 2e-16 and
an objective error of 0. **No dual bound or duality gap was recorded**, so optimality of the fitted
programme is solver-reported, not certified. The A-role constraints bind at 0.01 (within 1e-9). The AB
fitted CMIs are 0.0119–0.0177 in all 12 cells, because policy L imposes no AB constraint.

*Local vs coalition.* The coalition feasible set is a subset of the local one, so the coalition's fitted
optimal objective is ≥ the local one's. This is a feasible-set ordering of fitted quantities. It does not
order held-out privacy.

*Marginal ≠ joint.* Constraints on I(SEX;Z|C) and I(RAC1P;Z|C) do not bound I((SEX,RAC1P);Z|C). With
independent fair bits S₁, S₂ and Z = S₁ xor S₂, each marginal information is 0 and the joint is log 2 (F5,
relabelled). INTERPRETATION_ADDENDUM item 4 already makes this correction.

## T9. Coarsening, refinement, tokens

- *Coarsening (F5).* With independent fair bits S, H and Z = S xor H: I(S;Z)=0 but I(S;Z|H)=log 2, and
  conditioning on a constant partition of H reports 0. Every guarantee must read "fitted, coarsened
  finite-model constraint", and auditors must see continuous H. They do (`evaluation.py`, one-hot Z beside
  continuous H).
- *Refinement (F7).* Copying each parent row to its children reproduces the coarse joint law and objective
  exactly, provided costs pool with p(t) weights, on the same rows, with the same decoder and baseline. So a
  correctly nested refined problem has fitted optimum ≤ the coarse one at the same constraints. The replay
  confirms the actual aggregation (refinement_aggregation_max_error 6.9e-17, 0 parent mismatches). This
  orders fitted objectives only; it gives no generalisation or measured-attacker guarantee.
- *Stochastic separation fixture* (`experiments/pcrl_task_directed_release_v1/fixtures/`). Replayed
  byte-identical JSON (script SHA-256 06800bb9…, results SHA-256 4316fa41…). All 5 deterministic partitions
  of the three-state input are checked, and only the constant one is exactly private. The randomised kernel
  (1/2, 0, 1) is exactly independent of S and retains 0.0863 nats of task information, while releasing the
  kernel row itself reveals X. This is the perfect-privacy phenomenon of Rassouli–Gündüz (and of
  randomised mappings generally): a design example, not an ACS result.
- *Tokens.* Expected single-token loss Σ_z Q(z|t) loss(f(H,z),y) differs from the loss of the averaged
  prediction (F8: 1.204 vs 0.693 nats). The code scores the former (audits.py:167-180, finite.py:85-109). The
  public mechanism may be known to the attacker without the person's row or coin; releasing the row is a
  different mechanism.

## T10. Fixed-offset action decoders

`fit_dictionary` (encoding.py @ f4bdf4cd) solves the stationarity condition Σ w(σ(logit b + a) − p) = 0 per
residual group for the offset a, with a bracketed root in [−12, 12] (boundary statuses are recorded). This
is the first-order condition of a weighted soft-label cross-entropy in a scalar offset with b fixed. It is
convex in a, so a root is the minimiser within its group. It says nothing about global optimality of the
dictionary, of the cut points, or of partially projected candidates. The earlier competitive-method study
records the relevant caveat (`results/pcrl_competitive_method_v1/METHOD.md:216`: no free per-class
intercept, whose gradient cross-fitting does not zero). Do not extend stationarity claims to those
candidates without a proof.
