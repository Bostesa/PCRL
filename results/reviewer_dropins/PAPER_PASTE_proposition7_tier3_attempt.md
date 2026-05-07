# Proposition 7 (Tier 3 attempt) — VICReg–Erasure Trilemma

**Final state: SHIPPED as Tier 2.5 (synthesis-novelty, not technique-novelty).**
**Honest grade: this is stronger than Proposition 6 but does not clear the strict Tier-3 bar of "novel proof technique."**

The user requested a Tier 3 attempt with 2 hours. After 60 min of formalization + 30 min of numerical verification + 10 min of adversarial check, the cleanest result that survives is the proposition below. The proof uses two folklore ingredients (CCA invariance of linear-$R^2$ + Geršgorin's circle theorem); the contribution is their *synthesis* applied to PCRL's specific design. A hostile reviewer with concept-erasure / fairness-rep expertise would correctly call the proof techniques folklore — but no prior work names this particular interaction between linear-$R^2$ erasure and VICReg-style regularization.

If the user's Tier-3 bar is "novel proof technique," **this should be classified as Tier 2.5 and ship as a §5.6 strengthening of the LoRA Erasure Floor** (`TIER3_ATTEMPT_REPORT.md`), not as Theorem 2. If the bar is "novel structural insight specific to PCRL with a clean falsifiable prediction," the proposition is paper-ready.

---

## 1. Proposition 7 (PCRL VICReg–Erasure Trilemma)

### Setup

Let $f_0: \mathcal{X} \to \mathbb{R}^d$ be a frozen feature map with
covariance $\Sigma_0 := \mathrm{Cov}(f_0(X)) \succ 0$, and let
$A_{\mathrm{oh}} \in \mathbb{R}^c$ be the centered one-hot encoding of a
categorical attribute with $\mathrm{tr}(\Sigma_A) > 0$. Define the
backbone leakage $R^2_0 := R^2(f_0; A) > 0$.

Let $h = W f_0$ where $W \in \mathbb{R}^{d \times d}$ is *any* linear
map (the rank-$r$ LoRA family $W = I + B A^\top$ with
$\mathrm{rank}(BA^\top) \le r$ is a special case but is not used in the
proof — see R4). The multivariate linear-$R^2$ is defined as in
Proposition 6 (`pcrl/purposes/verification.py:90-103`).

Three constraints, idealizing PCRL's training objective
(`pcrl/training/independence/vicreg.py`):

- **(C1) Linear erasure:** $R^2(h; A) \le \varepsilon$ with
  $\varepsilon < R^2_0$ (proxy-Lagrangian objective).
- **(C2) Variance floor:** $\mathrm{Cov}(h)_{ii} \ge \tau^2$ for all $i$
  (VICReg variance hinge at target $\tau$, default $\tau = 1.0$).
- **(C3) Decorrelation:** $|\mathrm{Cov}(h)_{ij}| \le \gamma$ for all
  $i \ne j$ (VICReg covariance penalty in its hard-constraint limit).

### Theorem (Trilemma)

Let $\gamma_{\mathrm{crit}}(\tau, d) := \tau^2 / (d - 1)$.

**(I) Linear-$R^2$ invariance.** For any invertible $W$ and any $A$,
$$R^2(W f_0; A) = R^2(f_0; A).$$

**(II) VICReg-Erasure incompatibility.** If
$\gamma < \gamma_{\mathrm{crit}}$, then (C1), (C2) and (C3) cannot
hold simultaneously: any $W$ satisfying (C2) and (C3) has
$\mathrm{Cov}(h) \succ 0$, hence $W$ is invertible, hence $R^2(h; A) =
R^2_0 > \varepsilon$, contradicting (C1).

**(III) Soft-regime contrapositive.** Under PCRL's actual soft
regularization, any feasible point $h$ with $R^2(h; A) \le \varepsilon
< R^2_0$ satisfies at least one of:
$$\min_i \mathrm{Cov}(h)_{ii} < \tau^2 \quad\text{or}\quad \max_{i\ne j} |\mathrm{Cov}(h)_{ij}| > \tau^2 / (d-1).$$
In particular, the empirical compliance-via-collapse pattern
(`memory:project_v2_round5.md`, 49/56 cells) is not optimizer drift —
it is a structural consequence of the design.

### Proof

**(I)** For invertible $W$, $\mathrm{Cov}(h) = W \Sigma_0 W^\top$ and
$\mathrm{Cov}(h, A_{\mathrm{oh}}) = W M_A^{(0)}$ where
$M_A^{(0)} := \mathbb{E}[f_0 A_{\mathrm{oh}}^\top]$. Using
$(W \Sigma_0 W^\top)^{-1} = W^{-\top} \Sigma_0^{-1} W^{-1}$:
\begin{align*}
R^2(h; A) \cdot \mathrm{tr}(\Sigma_A)
&= \mathrm{tr}\!\left((W M_A^{(0)})^\top (W \Sigma_0 W^\top)^{-1} (W M_A^{(0)})\right) \\
&= \mathrm{tr}\!\left((M_A^{(0)})^\top W^\top W^{-\top} \Sigma_0^{-1} W^{-1} W\, M_A^{(0)}\right) \\
&= \mathrm{tr}\!\left((M_A^{(0)})^\top \Sigma_0^{-1} M_A^{(0)}\right)
= R^2(f_0; A) \cdot \mathrm{tr}(\Sigma_A).
\end{align*}
Dividing by $\mathrm{tr}(\Sigma_A) > 0$ yields (I). $\square$

**(II)** Suppose (C2), (C3) and $\gamma < \gamma_{\mathrm{crit}}$ all
hold. By Geršgorin's circle theorem applied to the real-symmetric
$\mathrm{Cov}(h)$, every eigenvalue $\lambda$ satisfies
$$\lambda \ge \min_i \!\left( \mathrm{Cov}(h)_{ii} - \sum_{j\ne i} |\mathrm{Cov}(h)_{ij}| \right) \ge \tau^2 - (d-1)\gamma > 0,$$
so $\mathrm{Cov}(h) \succ 0$. Since $\mathrm{Cov}(h) = W \Sigma_0 W^\top$
and $\Sigma_0 \succ 0$, the matrix $W \Sigma_0^{1/2}$ has rank $d$,
hence $W$ has rank $d$ and is invertible. Applying (I): $R^2(h; A) =
R^2_0 > \varepsilon$, contradicting (C1). $\square$

**(III)** is the contrapositive of (II): if $R^2(h; A) \le \varepsilon < R^2_0$,
then we cannot have *both* $\min_i \mathrm{Cov}(h)_{ii} \ge \tau^2$ *and*
$\max_{i\ne j} |\mathrm{Cov}(h)_{ij}| \le \gamma_{\mathrm{crit}}$. $\square$

### Remarks

**R1 — Honesty about novelty.** The two ingredients are folklore: (a) linear-$R^2$
invariance under invertible linear transformations is the standard CCA / LDA
identity (Mardia–Kent–Bibby, *Multivariate Analysis*, ch. 9); (b) Geršgorin
(1931) is undergraduate matrix analysis. The contribution is *not* a new
proof technique. It is the synthesis: identifying that PCRL's three
design choices — the proxy-Lagrangian linear-$R^2$ constraint, VICReg's
variance hinge, and VICReg's covariance penalty — *jointly* enforce a
fundamental tension that no prior fairness-representation paper has
named. RLACE / LEACE / INLP analyze erasure in isolation; the original
VICReg analyzes self-supervised collapse in isolation; this proposition
shows their interaction inside PCRL is incompatible at the threshold.

**R2 — The threshold is sufficient, not necessary.** The Geršgorin lower
bound $\lambda_{\min} \ge \tau^2 - (d-1)\gamma$ is one-sided. For
specific covariance structures, $\lambda_{\min}$ can exceed this bound,
so erasure may also be impossible at $\gamma$ slightly above
$\gamma_{\mathrm{crit}}$. The exact incompatibility threshold is
data-dependent and tighter than the Geršgorin form.

**R3 — Idealization vs. PCRL implementation.** PCRL implements (C2) as
a hinge loss and (C3) as a quadratic penalty (`vicreg.py:18-46`), not
hard constraints. The proposition applies to the hard-constraint
idealization. The *quantitative* refinement for the soft regime gives a
positive lower bound on VICReg loss at any feasible point: the next
section computes this lower bound and confirms PCRL's empirical VICReg
loss exceeds it. The full quantitative proof requires bounding
$\sum_{i\ne j} \mathrm{Cov}(h)_{ij}^2$ via a Cauchy–Schwarz refinement
of Geršgorin, sketched in §A below.

**R4 — Rank-$r$ structure is not used in (II).** The proof of (II)
uses only that $h = W f_0$ for *some* linear $W$. The rank-$r$ LoRA
ansatz $W = I + BA^\top$ is decorative for this proposition; the
incompatibility holds for any encoder family producing affine
representations. The rank-$r$ structure becomes relevant when combining
this with the Erasure Floor (`TIER3_ATTEMPT_REPORT.md`) to obtain a
joint feasibility characterization (R5).

**R5 — Relationship to Erasure Floor (Tier 2 fallback).**
The Erasure Floor states: rank-$r$ LoRA achieves
$R^2(h; A) \ge \sum_{i=r+1}^{K-1} \sigma_i^2(M_A) / \mathrm{tr}(\Sigma_A)$,
with equality iff $W$ has kernel spanning the top-$r$ left singular
vectors of $M_A$ — which forces rank deficit $r$ in $W$.
Proposition 7 strengthens this by giving a *VICReg-side sufficient
condition* for full-rank $W$, hence a sufficient condition for *no*
rank deficit, hence a sufficient condition for the Erasure Floor's
equality case to be unreachable. Combined: at any rank-$r$ LoRA
feasible point with $R^2(h;A) \le \varepsilon < R^2_0$, the Erasure
Floor mandates rank deficit $\ge s_{\min}(\varepsilon)$ in $W$, which
forces at least $s_{\min}(\varepsilon)$ near-zero eigenvalues of
$\mathrm{Cov}(h)$, which forces VICReg violation by Geršgorin.

**R6 — Limiting cases.**
- $\varepsilon \to R^2_0$: trilemma trivializes (no erasure required, any
  invertible $W$ works).
- $\tau \to 0$: $\gamma_{\mathrm{crit}} \to 0$, so (C3) imposes no
  meaningful constraint; impossibility vanishes.
- $d \to \infty$ with $\tau$ fixed: $\gamma_{\mathrm{crit}} \to 0$, so
  the threshold tightens — large-$d$ representations need *very* clean
  decorrelation to satisfy VICReg, making the trilemma easier to trigger.
- $d = 1$: $\gamma_{\mathrm{crit}}$ is undefined (division by zero), but
  the proposition is trivially vacuous (no off-diagonal entries).

---

## 2. Numerical verification (Adult R5, 27 cells)

Verification script: `/tmp/tier3_prop7/verify_prop7.py`. For each
$(\text{seed}, \text{purpose}, \text{attribute})$ cell on Adult R5 we
compute the backbone-only $R^2(f_0; A)$ and the LoRA-applied
$R^2(h_p; A)$ on the test split, plus diagonal and off-diagonal
statistics of $\mathrm{Cov}(h_p)$.

| | $d=64$, $\tau=1.0$, $\gamma_{\mathrm{crit}} = 1/63 = 0.01587$ |
|---|---|
| Total cells | 27 |
| Cells with genuine erasure ($R^2_h < R^2_{\mathrm{bb}}$) | 27 / 27 |
| Predicted: at least one violation | 27 / 27 |
| Falsifications | 0 |

**Per-purpose VICReg violations on Adult R5 (median over seeds):**

| Purpose | $R^2(\mathrm{bb};\mathrm{race})$ → $R^2(h;\mathrm{race})$ | min diag $\mathrm{Cov}(h)$ | max off-diag | eff. rank | VAR | DECORR |
|---|---|---:|---:|---:|---|---|
| income_prediction       | $0.59 \to 0.05$ | $0.018$ | $0.34$ | 21 / 64 | ✗ | ✗ |
| employment_analysis     | $0.59 \to 0.01$ | $0.013$ | $0.32$ |  6 / 64 | ✗ | ✗ |
| education_assessment    | $0.59 \to 0.05$ | $0.029$ | $0.34$ |  5 / 64 | ✗ | ✗ |

(`min diag` ≪ $\tau^2 = 1.0$: variance constraint violated. `max off-diag` ≫ $\gamma_{\mathrm{crit}} = 0.0159$: decorrelation constraint violated. Effective rank << 64: explicit rank deficit consistent with the Erasure Floor.)

The numerical verification is **one-sided**: every cell shows *both*
violations simultaneously, so we cannot falsify the OR-form prediction
with the current data. A stronger empirical test would require a
training regime where one of the two VICReg components is heavily
weighted and the other lightly weighted; we have not run such a
sweep. As a sanity check, we verified the prediction *is*
falsifiable in principle: a hypothetical cell with $R^2_h < R^2_{\mathrm{bb}}$
*and* $\min \mathrm{Cov}(h)_{ii} \ge 1.0$ *and* $\max |\mathrm{Cov}(h)_{ij}| \le
0.0159$ would refute Proposition 7. None of the 27 cells exhibits this.

**Backbone baseline:** the Adult backbone itself has $\min
\mathrm{Cov}(f_0)_{ii} = 0.020$ and $\max |\mathrm{Cov}(f_0)_{ij}| =
0.114$ — already violating both VICReg constraints. So the trilemma's
hypothesis (C2)+(C3) at the strict thresholds is not satisfied even at
$W = I$. The contrapositive (III) is what provides predictive content
in this regime.

---

## 3. What this drop-in claims and does not claim

**Claims.**
- A clean structural impossibility result connecting PCRL's three
  design choices, with an explicit threshold $\gamma_{\mathrm{crit}} = \tau^2/(d-1)$
  and an explicit prediction (III) that is satisfied in 27/27 cells.
- An explanation of compliance-via-collapse as a structural
  consequence of the design rather than optimizer drift.
- A complement to the Erasure Floor (Tier 2 fallback): together they
  characterize PCRL's feasible set as "rank deficit forced by erasure
  constraint, accompanied by VICReg violation forced by rank deficit."

**Does not claim.**
- A novel proof technique. The two ingredients are folklore; the
  contribution is the synthesis. By the strict "Tier 3 = novel proof
  technique" bar, this is Tier 2.5.
- A sharp threshold. Geršgorin is one-sided; the actual incompatibility
  threshold is data-dependent and tighter.
- A multi-layer-LoRA proof. The proof is for the single-layer affine
  setting $h = W f_0$. PCRL's multi-layer encoder is not formally
  covered, though the empirical numbers extend.
- An empirical falsification. The numerical verification confirms but
  does not falsify the OR-form, since every cell shows both violations.
- A quantitative VICReg-loss lower bound for the soft regime. R3
  sketches the path; the full proof needs a Cauchy–Schwarz refinement
  of Geršgorin and is left for future work.

---

## §A. Sketch of quantitative refinement (soft regime)

For any rank-$r$ LoRA achieving $R^2(h;A) \le \varepsilon < R^2_0$ with
$\min_i \mathrm{Cov}(h)_{ii} \ge \tau^2$ (variance hinge satisfied),
the Erasure Floor mandates rank deficit $s \ge s_{\min}(\varepsilon)$
in $W$, hence at least $s$ eigenvalues of $\mathrm{Cov}(h) = W\Sigma_0 W^\top$
are zero. Pick $i^*$ with $\mathrm{Cov}(h)_{i^*,i^*}$ on the
zero-eigenvalue Geršgorin disk. Then $\sum_{j\ne i^*}|\mathrm{Cov}(h)_{i^*,j}|
\ge \mathrm{Cov}(h)_{i^*,i^*} \ge \tau^2$. By Cauchy–Schwarz,
$\sum_{j\ne i^*} \mathrm{Cov}(h)_{i^*,j}^2 \ge \tau^4 / (d-1)$. By
symmetry of $\mathrm{Cov}$, $\sum_{i\ne j} \mathrm{Cov}(h)_{ij}^2 \ge
2\tau^4/(d-1)$, so the VICReg covariance loss is bounded below by
$L_{\mathrm{cov}}(h) = \frac{1}{d}\sum_{i\ne j}\mathrm{Cov}(h)_{ij}^2 \ge
\frac{2\tau^4}{d(d-1)}$.

For $d=64$, $\tau = 1$: $L_{\mathrm{cov}}(h) \ge 4.96 \times 10^{-4}$.
Empirically (single Adult R5 cell, income_prediction): $L_{\mathrm{cov}} \approx
\frac{1}{64}\cdot \sum \mathrm{Cov}^2 \approx 0.04$, two orders of
magnitude above the lower bound — consistent. The bound is loose because
it assumes only one row is Geršgorin-saturating; iterating over $s$ rows
sharpens it to $L_{\mathrm{cov}}(h) \ge \Omega(s\,\tau^4/[d(d-1)])$,
left as future work.

---

## 4. Verification artefacts

- Proof draft / scratch: `/tmp/tier3_prop7/proof_draft.md`
- Verification script: `/tmp/tier3_prop7/verify_prop7.py`
- Per-cell numerics: `/tmp/tier3_prop7/numerical.json`

---

## 5. Recommendation

If the user wants this in the paper:

1. **As Theorem 2 / Proposition 7 in §5.6** ("compliance-via-collapse
   is structurally forced"): paper-ready, **but** the writeup must
   honestly disclose R1 (folklore proof techniques) and R3 (idealization
   from soft to hard constraints). Reviewers familiar with concept
   erasure will recognize the components; the synthesis is what's new.
2. **As an appendix-only result** strengthening the Erasure Floor (per
   the Tier-3 ABANDONED-with-Tier-2-fallback structure of
   `TIER3_ATTEMPT_REPORT.md`): also paper-ready, less prominent
   placement, lower reviewer-criticism risk.
3. **Skip entirely:** if the user's bar is strict Tier 3 (novel proof
   technique). Proposition 6 (cross-purpose linear-leakage bound,
   `PAPER_PASTE_proposition6_tier2.md`) is the safer ship.

The user reviews the proof and decides which placement, if any.
