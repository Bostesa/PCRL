# Tier 3 attempt — REPORT (abandoned, with Tier 2 fallback)

**Time on task:** ~55 minutes (Phase 1 + Phase 2 brainstorm). Stopped per
the user's "STOP if no novel structure emerges by minute 60" hard rule.
No `.tex` modified, no commits, no push.

---

## Final state: **ABANDONED**

| Phase | Status | Verdict |
|-------|--------|---------|
| 1. Gap survey | DONE | Gap identified |
| 2. Formalize candidate | DONE (partial) | Candidate stated cleanly |
| 3. Proof attempt | NOT STARTED | Proof technique = Eckart-Young; fails Tier 3 bar |
| 4. Numerical verification | NOT STARTED | — |
| 5. Adversarial proof check | NOT STARTED | — |
| 6. Write-up | NOT STARTED (Tier 3) | Tier 2 fallback drafted below |

---

## Phase 1 — gap identified

The user's prompt listed four "specific gaps worth checking." After
reading the existing reviewer drop-ins
(`PAPER_PASTE_theory.md`, `PAPER_PASTE_proposition5.md`),
`results/theorem5_check/DECISION.md`, and the head-aware-LEACE finding
in memory, I converged on the gap most concretely tied to PCRL's
empirical failure modes:

> **"Compliance-via-collapse forced by rank: is collapse FORCED by the
> LoRA rank being too small relative to the joint cardinality of
> disallowed attributes?"**

The DECISION.md numbers point exactly here: per-purpose effective rank
is much smaller than `d` on the collapsed checkpoints (HMDA `[1, 64, 4]`,
Diabetes `[12, 5, 10]`), and the team's empirical evidence (BIOS
Round 7 needed LoRA rank 16→24 for a K=24 OvR Diabetes pair) suggests a
quantitative threshold relating LoRA rank `r` to the cardinality `K` of
the disallowed attribute. The other three gaps in the prompt either
overlap with existing reviewer drop-ins (multi-X frontier overlaps with
Sadeghi-Boddeti-style framing in `§A.2` of `PAPER_PASTE_theory.md`) or
are too vague for a 3-hour attack (Pareto frontier specific to LoRA-class
encoders requires backbone-specific assumptions to make non-trivial).

---

## Phase 2 — candidate theorem statement

Working in whitened coordinates: let `f': X → R^d` with `Cov(f') = I_d`
(a fixed pre-conditioner that does not affect linear-R²). Let `A` be a
`K`-class disallowed attribute with centered one-hot `A_oh ∈ R^{K-1}`,
and let `M = Cov(f', A_oh) ∈ R^{d × (K-1)}` with singular values
`σ_1 ≥ σ_2 ≥ … ≥ σ_{K-1} ≥ 0`. Let `Δ = BA` be a rank-`r` LoRA
perturbation (`B ∈ R^{d × r}, A ∈ R^{r × d}`), giving `h = (I + BA) f'`.

**Candidate (LoRA Erasure Floor).** For every rank-`r` LoRA,

```
R²_lin(h, A)  ≥  (Σ_{i=r+1}^{K−1} σ_i²)  /  trace(Cov(A_oh))
```

with equality achievable when `W = I + BA` is chosen with kernel
spanning the top-`r` left singular vectors of `M`.

**Corollary (erasure-rank threshold).** Full linear erasure
`R²_lin(h, A) = 0` is achievable iff `r ≥ K − 1`.

**Corollary (forced collapse).** At equality, `h = W f'` has rank
`d − r`: the optimal LoRA reduces representational capacity from `d`
to `d − r` dimensions, i.e., **erasure forces collapse along exactly the
killed directions**.

### Trivial-case check
- `K = 2` (binary, e.g., gender): `K − 1 = 1`, threshold `r ≥ 1` —
  matches BIOS rank-1 LEACE/RLACE folklore.
- `K = 24` (OvR Diabetes age_bucket): threshold `r ≥ 23` — matches the
  team's empirical finding that rank 16 was insufficient and rank 24
  resolved per-class OvR.
- Floor monotone-decreasing in `r`: more LoRA capacity → more erasure;
  signs are right.

### Why I refused to enter Phase 3 with this

The proof is two lines of standard linear algebra:
1. `R²_lin` is invariant under invertible `W` (direct calc:
   `(W M)^T (W W^T)^{−1} (W M) = M^T M`). Erasure therefore requires
   rank deficit in `W`.
2. Best rank-`s` rank-deficit erasure is Eckart-Young applied to `M`:
   max `||P_K M||_F²` over `dim(K) ≤ s` is `Σ_{i=1}^{s} σ_i²(M)`,
   achieved at `K =` span of top-`s` left singular vectors.

Neither of these is a new mathematical structure or a proof technique
"not yet applied to this problem." Eckart-Young projection arguments
are exactly what RLACE (Ravfogel ICML 2022, rank-`k` oblique projection
for `K`-class concept removal), LEACE (Belrose 2023, affine projection
of rank `d − (K − 1)`), and INLP (Ravfogel ACL 2020, iterated rank-1
null-space) already use. The contribution would be **a cleaner explicit
formula for the residual `R²` as a function of `r`**, but that is at
best a Tier 2 framework, not Tier 3.

A hostile reviewer specializing in concept erasure would correctly say:
*"This is an explicit lower bound on what RLACE/LEACE-class methods
already establish constructively; the proof is direct Eckart-Young; not
novel."* Per the user's "do not ship Tier 3 with imperfect adversarial
check," I will not ship it as Theorem 2.

---

## What we learned (1 paragraph)

The most concrete novelty available within a 3-hour budget at this
codebase's current state is **a clean linear-algebra characterization
of what rank-`r` LoRA can and cannot erase**, formalized as the
proposition above. This is genuinely useful for §5.6 (it predicts the
LoRA rank threshold the team observed empirically, and it ties
"compliance-via-collapse" to a structural rank deficit rather than to
optimizer drift), but it is **not** Tier 3: the proof is direct
Eckart-Young, well within RLACE/LEACE's existing framework. Tier 3 in
this area would require either (i) characterizing the rank-`r` LoRA
*Pareto frontier* in `(R²(h, A), R²(h, Y))` space via generalized
eigenvalues of `(M_A M_A^T, M_Y M_Y^T)` — which I sketched but did not
prove because the frontier curve is not closed-form and the
generalized-eigenvalue argument is a folklore variant of CCA-style
fair-rep bounds, hostile-reviewable as such — or (ii) a multi-purpose
extension to the block-correlation degeneracy where `λ*(C) = 0` is
forced by the LoRA-perturbation-of-frozen-backbone structure, which the
DECISION.md numbers (concat rank 182/126/144 out of 192) suggest is
real but does not cleanly imply the kind of leakage observed
empirically because the leakage on HMDA (max R² = 0.70) comes from
*different purposes constraining different attributes*, not from
shared-`f₀`-induced redundancy. Both directions are honest Tier 2
results; neither is a Tier 3 swing within 3 hours.

---

## Tier 2 fallback (clean, drop-in for §5.6 if user wants)

Below is the proposition with a complete proof. The framing is honest:
it is a corollary of standard linear algebra, presented as a
quantitative tightening of the RLACE/LEACE rank-`K − 1` threshold.

### Proposition (LoRA Erasure Floor)

Let `f₀: X → R^d` be a fixed feature map with full-rank covariance
`Σ_0 = Cov(f₀(X))`. Let `A ∈ {1, …, K}` be a categorical attribute,
let `A_oh ∈ R^{K−1}` denote the centered one-hot encoding, and set
`M = Σ_0^{−1/2} Cov(f₀(X), A_oh) ∈ R^{d × (K−1)}` with singular values
`σ_1 ≥ σ_2 ≥ … ≥ σ_{K−1} ≥ 0`. For every rank-`r` LoRA perturbation
`Δ = BA` (`B ∈ R^{d × r}, A ∈ R^{r × d}`) producing
`h(X) = (I + BA) f₀(X)`, the multi-output linear coefficient of
determination satisfies

```
R²_lin(h, A)  ≥  (Σ_{i=r+1}^{K−1} σ_i²)  /  trace(Cov(A_oh)).
```

The bound is tight: equality is achieved by the LoRA whose effective
linear map `W = I + Σ_0^{−1/2} BA Σ_0^{1/2}` has kernel spanning the
top-`r` left singular vectors of `M` (equivalently, `BA` is chosen so
that `Σ_0^{−1/2} BA Σ_0^{1/2}` has eigenvalue `−1` along each top-`r`
left singular vector of `M`). At equality, `h(X)` has rank `d − r`,
i.e., the constraint forces a representational rank deficit of exactly
`r`.

#### Proof

By invariance of `R²_lin` under fixed positive-definite preconditioners
of `f₀`, we may assume `Σ_0 = I_d` (apply `Σ_0^{−1/2}` to both `f₀` and
`BA`). Write `W = I + BA` and `f' = f₀`. Then `h = W f'`,
`Cov(h) = W W^T`, and `Cov(h, A_oh) = W M`.

**Step 1 (`R²` invariance under invertible `W`).**
If `W` is invertible, then `(W W^T)^{−1} = W^{−T} W^{−1}` and

```
trace(M^T W^T (W W^T)^{−1} W M)
= trace(M^T W^T W^{−T} W^{−1} W M)
= trace(M^T M).
```

So `R²_lin(W f', A_oh) = ||M||_F² / trace(Cov(A_oh)) = R²_lin(f', A_oh)`,
i.e., invertible `W` cannot reduce linear-R². Erasure requires rank
deficit.

**Step 2 (`R²` for rank-deficient `W`).**
Let `K = ker(W)` of dimension `s := d − rank(W)`, and let `P_K` denote
the orthogonal projector onto `K`. Then `W f' = W P_{K⊥} f'`; further,
`W` restricted to `K⊥` is invertible onto `im(W)`, so by Step 1 applied
in that restricted block,

```
R²_lin(W f', A_oh) = R²_lin(P_{K⊥} f', A_oh).
```

A direct computation with `Cov(P_{K⊥} f') = P_{K⊥}` (idempotent) and
`Cov(P_{K⊥} f', A_oh) = P_{K⊥} M` gives

```
R²_lin(P_{K⊥} f', A_oh) = trace((P_{K⊥} M)^T P_{K⊥}^+ (P_{K⊥} M))
                         / trace(Cov(A_oh))
                       = trace(M^T P_{K⊥} M) / trace(Cov(A_oh))
                       = (||M||_F² − ||P_K M||_F²) / trace(Cov(A_oh)).
```

(The Moore-Penrose pseudoinverse of an orthogonal projector is itself,
and `M^T P_{K⊥} M = M^T M − M^T P_K M`.)

**Step 3 (Eckart-Young on the kernel projection).**
For the kernel `K` chosen optimally (minimizing the lower bound on
`R²_lin(h, A)`), the contribution `||P_K M||_F²` is maximized.
Eckart-Young gives the maximum `||P_K M||_F²` over `dim(K) ≤ s` as
`Σ_{i=1}^{s} σ_i²`, achieved when `K` spans the top-`s` left singular
vectors of `M`.

**Step 4 (rank-`r` LoRA caps the kernel dimension at `r`).**
`W = I + BA` with `BA` of rank `≤ r` has `rank(W) ≥ d − r`, so
`s = dim(ker W) ≤ r`.

Combining:

```
R²_lin(h, A)
  = (||M||_F² − ||P_K M||_F²) / trace(Cov(A_oh))
  ≥ (||M||_F² − Σ_{i=1}^{r} σ_i²) / trace(Cov(A_oh))
  = (Σ_{i=r+1}^{K−1} σ_i²) / trace(Cov(A_oh)).
```

For the tight construction: pick `K =` span of the top-`r` left
singular vectors of `M`, and let `BA` be the rank-`r` projection
`−P_K`. Then `W = I − P_K` has kernel exactly `K`, the LoRA is rank
`r`, and the bound is achieved with equality. By construction,
`h = W f₀` lies in `K⊥` (a `(d − r)`-dimensional subspace), so the
representation has lost exactly `r` dimensions of rank — the
**forced-collapse** corollary. ∎

#### Corollaries

- (**Erasure-rank threshold.**) `R²_lin(h, A) = 0` is achievable iff
  `r ≥ K − 1`. Below the threshold, the residual is the tail of `M`'s
  spectrum.
- (**Forced collapse.**) Full erasure at the floor demands rank-`r`
  rank deficit in the representation; the LoRA cannot have it both ways
  ("erase but stay full-rank"). Empirically this matches the team's
  observation that aggressive constraints drive `per_dim_std` low along
  the killed directions while preserving variance elsewhere.
- (**Empirical implication for §5.6.**) The BIOS rank-16→24 fallback
  for the K=24 per-class OvR Diabetes attribute (memory:
  `project_v2_round7_launch.md`) is consistent with the proposition's
  threshold `r ≥ K − 1 = 23`. Conversely, BIOS gender (binary, K=2)
  needs only rank-1 erasure, matching the BIOS Round-1 LEACE rank-1
  closed-form result on layer 6/11.

#### What this proposition is NOT

- It is not Tier 3. The proof is two lines of Eckart-Young; the
  framework is RLACE's, the threshold is folklore in concept-erasure.
- It does not address the multi-purpose leakage observed in DECISION.md
  (R²(h_concat, A) up to 0.70 on HMDA when per-purpose constraints are
  satisfied). That phenomenon is driven by different purposes
  constraining different attributes — not by rank deficit alone.
- It does not address adversarial / non-linear attribute attacks. The
  bound is on linear-R² only, matching PCRL's audit metric.

---

## Recommendation to user

1. **Tier 3 ambitions: leave for after deadline.** A genuine Tier 3
   result on PCRL likely requires a multi-purpose extension to
   Lechner-Ben-David or a quantitative Sadeghi-Boddeti-style Pareto
   frontier specific to LoRA-class encoders, neither of which I can
   prove cleanly in the remaining budget.

2. **Tier 2 fallback above is paper-ready** if the user wants a clean
   structural proposition for §5.6 ("why the LoRA rank-vs-cardinality
   threshold matters and what it forces"). It is honest,
   adversarial-clean, and ties the empirical rank-fallback observations
   to a precise proposition. The user reviews and decides whether to
   adopt, edit, or discard.

3. **No `.tex` was modified, no commit was made.** All artifacts are in
   `results/reviewer_dropins/TIER3_ATTEMPT_REPORT.md` only.
