# Guarantee cards

2026-10-02, methodology role. There is one card per method planned for the pilot and its extension.

**How to read a card.** Each card states:
- what quantity is bounded, and against which function class;
- on which law or rows the bound holds, and under which assumptions;
- which release surface it covers, and the numerical tolerance;
- the primary source;
- what falls outside its scope;
- which outcome categories it can produce.

A card is a *scope statement*. It is not a verdict.

**Pins and sources:**
- `dg` = durable-guarantees@956f5c88.
- `PCRL` = Bostesa/PCRL `origin/main`@55e4cb1d unless stated otherwise.
- Primary-source citations follow `results/combined_empirical_preparation_v1/METHOD_CATALOG.csv`, where the
  pins were verified by `ls-remote` on 2026-10-01.
- **[derived]** marks a statement derived in this role and not taken from the cited paper. Each one
  carries a check or a tight example.

**Outcome categories** (`notes/PREP_CONTEXT.md`):

| Label | Meaning |
|---|---|
| C1 | the method fails its own fitting-sample check |
| C2 | the check passes but does not generalise to held-out rows |
| C3 | out-of-scope recovery |
| C4 | a population guarantee is contradicted under its own assumptions |
| C5 | not estimable |

**General rules:**
- C4 is possible only where a *population* guarantee is claimed, and only when that guarantee's
  assumptions are verified to hold.
- C3 is always possible.
- C5 is always possible.

---

### 1. Untreated reference (clean encoder output; same encoder and cell)

**Bounded quantity:** none.

**Role:** positive control and denominator for "recovery removed".

**Release surface:**
- The clean representation.
- The clean task output (the clean-output reference).

**Applies to:** the assessment rows of the cell.

**Outside scope:** not applicable.

**Categories:** none. It is a reference. If it shows no recovery above the label-only reference, the
cell is uninformative for removal and is reported as C5.

### 2. PCRL native one-hot linear-R² check

**What it is:** a measurement on given rows, not a theorem.

**Bounded quantity:** in-sample pooled one-hot ridge R².
- R² = 1 − Σ‖Z_c − H_cW‖² / Σ‖Z_c‖², with W = (H_cᵀH_c + 1e-6·I)⁻¹H_cᵀZ_c.
- Centred per variable; clamped at 0.
- Pass is R² < 0.05.

**Function class:** affine predictors with a fixed ridge penalty, under squared loss on the one-hot
indicator.

**Rows:** the same rows are used to fit and to score.

| Model | Rows |
|---|---|
| PCRL v2 | the test split |
| durable-guarantees | all cell rows, which are also the defense-fit rows |

**Assumptions:** none. The scale convention matters: the penalty is not normalised (see
SCOPE_CORRECTIONS D1).

**Release surface:** the representation.

**Numerical tolerance:**
- On origin/main the Gram is computed in float32. This understates R² by at most 0.0033 absolute on
  stored audits (METHODOLOGY_IMPROVEMENTS §4).
- Use float64.
- Some aggregators use `<=` rather than `<`.

**Source:** `pcrl/purposes/verification.py:62-109` and `pcrl/evaluation/certificates.py:426-583`.
- The dominant-axis variant (max per-class OvR R²) is at `certificates.py:47-116`.

**Outside scope:**
- held-out rows;
- nonlinear predictors;
- 0–1 or log loss;
- class contrasts (the dominant axis misses them by up to (K−1)×);
- task outputs;
- concatenations (Prop 6 covers concatenation only in-sample and through λ_min(R)).

**Categories:** C1 (native fail); C2 (held-out linear R² > τ); C3; C5.
- The retired R²→accuracy bound is a C4 case on its own (FC-15).

### 3. LEACE: concept-erasure 0.2.4, `LeaceEraser.fit(x, z)` with defaults

**Bounded quantity:** Cov(r(X), Z) = 0 under the law the moments were estimated on, with Z the one-hot
attribute.
- Equivalently, every affine predictor of Z from r(X) is no better than the constant predictor, for
  every convex loss: squared loss and logistic log-loss included, 0–1 accuracy excluded.
- Source: Belrose et al., NeurIPS 2023, Thm 3.1 / "linear guardedness".

**Rows:** the fitting sample.
- PCRL v2: the train split.
- durable-guarantees: all cell rows.
- Held-out rows are covered only approximately, by sampling.

**Assumptions:** finite second moments. The guarantee is for the exact eraser, in exact arithmetic.

**Release surface:**
- r(X).
- Any *affine* post-processing of r(X), such as a linear head's logits, by linearity.
- A nonlinear head's output is not covered.

**Numerical tolerance:** these are the package defaults as called (SCOPE_CORRECTIONS A4):

| Setting | Effect |
|---|---|
| `svd_tol=0.01` | truncates whitened cross-covariance singular values ≤ 0.01 |
| `constrain_cov_trace=True` | may mix in an orthogonal projection |
| `shrinkage=True` | estimates Σ_xx with shrinkage |
| float32 inputs | erased directions keep rounding-level variance (~1e-14) |

In synthetic data:
- fit-row cross-covariance is about 1e-7;
- an *unregularised* OLS recovers R² 0.13 from the rounding residue, also held-out;
- ridge 1e-6 or a 1e-6 relative eigenvalue floor reads ≈ 0;
- standardised logistic regression reads AUC 0.50.

The protocol therefore applies the declared floor and reports the unfloored value separately.

**Implementation pins:**
- PyPI `concept-erasure==0.2.4`.
- Call sites: PCRL `pcrl/training/v2_trainer.py:584`; dg `experiments/baseline_gauntlet.py:317` and
  `smart_erasure.py:205`.

**Outside scope:**
- nonlinear predictors (including quadratic; see QLEACE);
- held-out generalisation;
- task outputs from nonlinear heads;
- attributes not in Z;
- coalitions where the attribute is erased from only some members.

**Categories:** C1 (cross-covariance on fit rows above the declared tolerance); C2 (held-out affine
predictor beats constant under a convex loss beyond sampling error); C3; C5.
- C4 would need an exact-arithmetic violation, which is not expected because the result is a theorem.

### 4. durable-guarantees learned projections: MMD ("LEOPARD-style"), HSIC ("Obliviator-style"), "targeted projection"

**Mechanism:** h′ = h − (hQ)Qᵀ.
- Q is an orthonormal d×r basis learned by Adam on balanced minibatches drawn from **all** cell rows.
- Q minimises class-conditional Gaussian-kernel MMD or HSIC in the projected space.
- Source: `dg:experiments/smart_erasure.py:105-145`.

**Bounded quantity:** none.
- Approval is the in-sample PCRL R² ≤ 0.05 on all rows (card 2).
- These are dg constructions inspired by LEOPARD (Saillenfest & Lemberger, arXiv 2507.12341) and
  Obliviator (Akbari et al., NeurIPS 2025). They are not the official methods.

**Assumptions:** none stated.

**Release surface:** the representation.

**Tolerance:** stochastic fit (seeded).

**Outside scope:** everything beyond the approval check, including Johansson-type inversion when the
projection is fit on the released rows.

**Categories:** C1 (approval fails); C2 (held-out linear R² > τ); C3; C5.
- C4 is not applicable.

### 5. Isotropic Gaussian noise channel

**Mechanism:** h′ = h + σ_abs·ε, with ε ~ N(0, I) drawn fresh per release.
- σ_abs = σ_rel·σ_h, where σ_h is the cell's representation scale.
- Source: `dg:experiments/noise_channel_test.py:85-92, :229`.

**Bounded quantity:** none formal. The representation is **unclipped**, so its sensitivity is unbounded
and Proposition 3 does not apply.

**Evidence base:** empirical approval through in-sample R² averaged over 5 draws.

**Assumptions** (deployment): one release per input; a fresh draw per release.

**Release surface:** the representation, and any function of the noisy representation.
- A task output computed from the *clean* h is not covered.

**Tolerance:** Monte Carlo over draws.

**Outside scope:**
- repeated releases (averaging N draws reduces σ by √N);
- population white-box LRT, which is a recovery test rather than a contradiction.

**Categories:** C1 (approval fails); C2; C3; C5.
- C4 is not applicable.

### 6. Subspace channel (isolate-then-noise)

**Mechanism:** h′ = h + σ·(zQᵀ), with fresh z ~ N(0, I_r).
- Q is an HSIC-learned rank-r basis.
- Variants: post hoc on a fixed representation, or trained end to end with an HSIC penalty pushing the
  attribute out of h_⊥.
- Source: `dg:experiments/end_to_end_surgical.py:131-165`, `targeted_noise.py`.

**Bounded quantity:** none formal.

**Declared assumptions** (from the paper's abstract):
1. representation-only release;
2. one release per input;
3. a private basis Q.

**Release surface:** the representation only.

**Outside scope:** each assumption, when violated:
- averaging at N = 16 breaches it;
- knowing Q gives AUC 0.98;
- releasing outputs falls outside it.

**Categories:** C1, C2 and C3 (including access beyond the declared assumptions); C5.
- C4 is not applicable.

### 7. Clipped full-rank DP channel (AAAI Proposition 3)

**Mechanism:** M(x) = Π_C(φ(x)) + σZ, with Z ~ N(0, I_d) drawn fresh per release.
- φ is fixed (non-private with respect to its own training data).
- Π_C is projection onto the ℓ2 ball of radius C.

**Bounded quantity:** μ-GDP per release, with μ = 2C/σ under replace-one adjacency.
- Equivalently (ε, δ) by the Balle–Wang conversion.
- Corollary: the AUC of *any* attacker separating the attribute classes from the release alone is at
  most Φ(μ/√2), by the mixture argument.
- k adaptive releases give √k·μ.
- Sources: Dong, Roth & Su 2022; Balle & Wang 2018; `dg:docs/proposition3_dp_guarantee.md`.

**Function class:** all measurable attackers, with any knowledge of φ, C, σ and the population.

**Assumptions:**
- clipping is applied before noise;
- the noise is fresh and independent;
- one release, or a declared k;
- φ is fixed.

**Release surface:**
- M(x), and anything computed from M(x) alone, such as a head on the noisy representation.
- Outputs from clean φ(x), or other releases of x, are **not** covered.

**Tolerance:** floating-point Gaussian sampling is not certified.

**Outside scope:**
- side information, since the ceiling bounds the release's contribution rather than the attacker's
  total accuracy;
- the unclipped full-rank points in AAAI Table 1 and Fig. 4.

**Categories:**
- **C1:** the implementation does not clip, does not use fresh noise, or uses a wrong C or σ in its
  accounting.
- **C4:** an attacker exceeds Φ(μ/√2) beyond sampling error on a correctly implemented clipped channel.
  This is not expected.
- **C5:** currently, because no attack on clipped points is stored.

### 8. FARE (Jovanović et al., ICML 2023)

**Bounded quantity:** with probability ≥ 1 − ε over an i.i.d. certification sample,
sup_g ΔDP(g) ≤ T* for **every** downstream classifier g of the finite-cell (fair-tree) representation.
- Uses Clopper–Pearson and Hoeffding bounds.
- Requires a binary sensitive attribute.

**[derived]** For binary s, sup_g ΔDP = TV(P₀, P₁) over cells. TV ≤ t implies attacker AUC
≤ 1/2 + t − t²/2.
- Tight example: P₀ = (1−t)δ_a + tδ_b, P₁ = (1−t)δ_a + tδ_c, with b < a < c.
- Checked numerically: no violation in 20,000 random discrete pairs, scored by the likelihood-ratio
  ordering (methodology role, 2026-10-02).

**Rows:** the population. The certificate is computed on the held-out certification split. In dg the
tree split is 60/20/20.

**Assumptions:** i.i.d. sampling; binary s; no distribution shift.

**Release surface:** the cell index, and any function of it, task outputs included.

**Tolerance:** ε (confidence); the minimum cell size (min_ni = 100 in dg).

**Implementation pin:** `eth-sri/fare@89cb1b66`, via `dg:experiments/fare_stage_*.py`.

**Outside scope:**
- multiclass s, so the dg 5-class cells are out of scope;
- other attributes;
- shift.

**Categories:** C1 (T* computation wrong); C2 (empirical ΔDP on held-out rows exceeds T* more often
than ε); C3 (multiclass); C4 (ΔDP or AUC above the bound under i.i.d., beyond the 1 − ε allowance);
C5.

### 9. VFAE (Louizos et al., ICLR 2016)

**Bounded quantity:** none. MMD is a training penalty.

**Implementation:** the dg reimplementation, `dg:experiments/baseline_gauntlet.py:238-303`. An official
implementation was not located.

**Release surface:** the representation, released either as sampled z (stochastic) or as mean z. The
release contract must declare which.

**Outside scope:** everything.

**Categories:** C3; C5.
- If MMD on held-out rows is used as "its own test", C2 is also possible.

### 10. INLP (Ravfogel et al., ACL 2020)

**Bounded quantity:** none formal. The method's own test is an empirical stopping rule: a linear
classifier's validation accuracy within 1 pp of majority.

**Rows:** fit on train; stop on val.

**Implementation:**
- PCRL `pcrl/baselines/inlp.py:88-141`. The docstrings say "test", but the code uses val.
- Official: `shauli-ravfogel/nullspace_projection@e1edcc19`.

**Release surface:** the projected representation.

**Outside scope:**
- exact linear guardedness (INLP is suboptimal; RLACE Prop 4.1);
- nonlinear predictors;
- the R² criterion, which INLP was not trained to.

**Categories:** C1 (stopping not reached); C2 (held-out LR accuracy on assessment rows > majority +
margin); C3; C5.

### 11. LAFTR: official (Madras et al., ICML 2018) and reimplementations

**Bounded quantity:** ΔDP (or ΔEO) of a binary classifier on z is bounded by the *optimal* adversary's
group-normalised L1 loss (Madras et al., Thm 2).

**Assumptions:** an optimal adversary (not certifiable in practice); binary s.

**Implementations:**

| Implementation | Pin or location |
|---|---|
| Official | `VectorInstitute/laftr@a166ba3c` (dg TF1 shim) |
| dg multiclass reimplementation | `baseline_gauntlet.py:127-190` (outside the theory) |
| PCRL LAFTR benchmark | `scripts/run_laftr_benchmark.py` |

**Release surface:** the representation.

**Outside scope:**
- finite-sample certification;
- non-optimal adversaries;
- multiclass s;
- AUC recovery.

**Categories:** C3; C5.
- C4 is not testable, because the optimality premise is unverifiable.
- The original LAFTR HMDA and Diabetes rows (36 of 60) are unsupported (C5).

### 12. LAFTR-hard (PCRL stack + adversary + hard R² constraint)

**Bounded quantity:** none. Its own test is the PCRL native R² (card 2).

**Source:** `experiments/run_laftr_hard_r2.py:317-325`.

**Implementation pin:** the branch `laftr-hard-r2-2026-05-17`@2037ad45. The outputs are on the drive;
the checkpoints are probably lost (see CHECKPOINT_INVENTORY.csv).

**Categories:** C1 (Adult 0/24 native strict); C2; C3; C5.

### 13. SPLINCE (Holstege, Ravfogel & Wouters, NeurIPS 2025)

**Bounded quantity:** the oblique projection P with PΣ_xz = 0 and PΣ_xy = Σ_xy.
- Linear guardedness for Z, as in LEACE.
- Covariance with the task Y is preserved.
- Thm 1, under population (fit-sample) moments.

**Assumptions:** colsp(WΣ_xz) ∩ colsp(WΣ_xy) = {0}.

**Rows:** fit on train representations.

**Implementation:**
- PCRL in-house `scripts/run_splince_benchmark.py`, which includes a `fallback_to_leace` flag. It
  post-processes PCRL final.pt.
- Its fidelity to the official `fced0d3` is not verified.

**Release surface:** the representation, and affine post-processing of it.

**Outside scope:** the same as LEACE; plus it is not an independent pipeline.

**Categories:** C1, C2, C3 and C5, as for LEACE.

### 14. PCRL erase layer (erase-layer pilot; CelebA PCRL-V)

**Mechanism:** a frozen union-LEACE eraser, fit on train against the union of all purposes' disallowed
attributes. It sits upstream of maps that are **affine** in eval mode (LoRA on `repr_proj` only).

**Bounded quantity:** zero cross-covariance with the union attribute set on the fit law. This carries
through every downstream affine map (Cov(Ar + b, Z) = A·Cov(r, Z) = 0).

**Assumptions:** the downstream map is affine in eval mode; LEACE as in card 3.

**Release surface:** the per-purpose representation.

**Outside scope:**
- nonlinear recovery, which *rose* in the pilot on 48 of 60 cells;
- held-out rows;
- allowed attributes that are over-erased (for Adult, the income task label itself).

**Categories:** C1, C2, C3 and C5, as for LEACE.
- CelebA "val R²" is in-sample (C5; SCOPE_CORRECTIONS D5).

### 15. PCRL per-purpose LoRA + LEACE warm start (submitted R5/R7; Round 4 used by dg)

**Bounded quantity:** at **initialisation only**, joint LEACE gives zero cross-covariance per purpose
on the train split (Prop 2).

**After training:** all-layer LoRA moves the features, so no structural guarantee remains. The trained
model's claim is empirical: native check card 2 on the test split, in-sample.

**Training constraint:** per-batch in-sample R² (the D4 null level is about r/255).

**Release surface:**
- the per-purpose representation;
- task heads on it.

**Outside scope:**
- held-out rows;
- nonlinear predictors;
- concatenation (Prop 6, a linear in-sample bound);
- task outputs.

**Categories:** C1 (4/60 final-iterate, 6/60 best); C2 (pending: no held-out R² on the submitted
grid); C3; C5.

### 16. Multi-purpose adaptations (scenario arms)

**Per-recipient LEACE** (catalog BAS-01): one eraser per recipient over its declared disallowed set.
- Covers zero cross-covariance per release.
- For an attribute disallowed to **every** coalition member, the concatenation also has zero
  cross-covariance (block identity, exact on the fit law).
- Not covered: attributes allowed to some coalition member, which the protocol does not promise to hide;
  nonlinear combination; held-out rows.

**Joint union LEACE** (BAS-02): one shared eraser over the union of disallowed sets.
- Every coalition sees a linearly guarded release with respect to the union.
- It over-erases attributes some recipients are allowed.

**Categories:** C1, C2, C3 and C5, as for LEACE.

**Reporting rule:** an entitled recipient's coalition is never scored as a failure.
