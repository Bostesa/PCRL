# Proposition 6 (Tier 2) — Cross-purpose linear-leakage bound

**Status: candidate; user review of proof required before commit to .tex.**

This drop-in proposes a new theorem-level result (Tier 2: a non-trivial
proposition specific to PCRL's setup, not a corollary of an existing
result). It complements Proposition 5 (Zhao–Gordon-style single-task
impossibility) by bounding the *cross-purpose* leakage that arises from
concatenating per-purpose representations — the empirical phenomenon
documented in §5.4 (cross-purpose attack) and the §5.4-extension
training-time fix (`results/v2_adult_CROSSPURP/`), both of which currently
lack a theoretical bound.

---

## 1. Main-text paragraph (insert in §5.4 after the cross-purpose attack table)

> The cross-purpose recovery rates in Table~\ref{tab:cross-purpose} can
> be bounded analytically. Concatenating $k$ per-purpose representations
> $h_1,\dots,h_k$ that each individually satisfy
> $R^2(h_p; A) \le \varepsilon$ does not, in general, leave
> $R^2(\mathrm{concat}(h_1,\dots,h_k); A)$ at $\varepsilon$: the
> concatenation can amplify per-purpose leakage by a factor that depends
> on how aligned the whitened per-purpose representations are.
> Proposition~\ref{prop:cross-purpose-bound} makes this precise:
> $R^2(\mathrm{concat}(h_1,\dots,h_k); A) \le k\varepsilon /
> \lambda_{+}(R)$, where $\lambda_{+}(R)$ is the smallest positive
> eigenvalue of the block-whitened cross-purpose correlation matrix
> $R$ (Defn.~\ref{def:block-whitened-corr}). The bound is informative
> whenever $\lambda_{+}(R) > k\varepsilon$ and is tight in the worst
> case (saturating per-purpose constraints with maximally-aligned
> whitened representations); on Adult Round-5 it predicts ceilings
> $0.06$–$0.30$ on $R^2(\mathrm{concat}; A)$ that the empirical values
> $0.04$–$0.18$ respect with $45$–$70\%$ tightness utilisation.
> Proposition~6 thus complements Proposition~5: where Proposition~5
> bounds *per-task error* in terms of demographic shift, Proposition~6
> bounds *cross-purpose linear leakage* in terms of per-purpose
> compliance and the geometry of the joint representation.

---

## 2. Appendix block (Proposition + proof)

### Definition (block-whitened cross-purpose correlation)

Let $h_1,\dots,h_k:\mathcal{X}\to\mathbb{R}^d$ be per-purpose
representations with $\mathbb{E}[h_p] = 0$ (centered) and
$\Sigma_p := \mathbb{E}[h_p h_p^\top] \succ 0$ for each $p$. Let
$\Sigma_{pq} := \mathbb{E}[h_p h_q^\top]$. The
*block-whitened cross-purpose correlation matrix* is
$$R \in \mathbb{R}^{kd\times kd}, \qquad R_{pq} := \Sigma_p^{-1/2}\,\Sigma_{pq}\,\Sigma_q^{-1/2}.$$
Equivalently, $R$ is the covariance of the whitened concatenation
$V := (\Sigma_1^{-1/2}h_1; \dots; \Sigma_k^{-1/2}h_k)$.
Diagonal blocks satisfy $R_{pp} = I_d$, hence $\mathrm{tr}(R) = kd$
and (when $R$ is positive definite) $\lambda_{\min}(R) \in (0,1]$.
We denote by $\lambda_{+}(R)$ the smallest positive eigenvalue of
$R$ (which equals $\lambda_{\min}(R)$ in the positive-definite case).

### Multivariate linear-$R^2$

For a representation $h$ with covariance $\Sigma_h \succ 0$ and a
centered attribute $A \in \mathbb{R}^c$ (one-hot encoded for categorical
attributes) with $\mathrm{tr}(\Sigma_A) > 0$, define
$$R^2(h; A) := \frac{\mathrm{tr}\!\big(\Sigma_{h,A}^\top \Sigma_h^{-1} \Sigma_{h,A}\big)}{\mathrm{tr}(\Sigma_A)},$$
which matches the closed-form OLS $R^2$ used by PCRL's audit
(`pcrl/purposes/verification.py:90-103`) in the regulariser-$\to 0$ limit.

### Proposition 6 (Cross-purpose linear-leakage bound)

Let $h_1,\dots,h_k$ and $A$ satisfy the assumptions of
Defn.~\ref{def:block-whitened-corr} and the multivariate-$R^2$ definition
above, with finite second moments $\mathbb{E}\|h_p\|^2 < \infty$,
$\mathbb{E}\|A\|^2 < \infty$. Suppose for each purpose $p$,
$$R^2(h_p; A) \le \varepsilon. \tag{C1}$$
Let $H := (h_1; \dots; h_k) \in \mathbb{R}^{kd}$ be the concatenated
representation. Then:

**(I) Full-rank case.** If $R$ is positive definite, then
$$R^2(H; A) \;\le\; \frac{\sum_{p=1}^k R^2(h_p; A)}{\lambda_{\min}(R)} \;\le\; \frac{k\,\varepsilon}{\lambda_{\min}(R)}. \tag{$\ast$}$$

**(II) Singular case.** If $R$ is rank-deficient but the OLS predictor
of $A$ from $H$ exists (equivalently, $\Sigma_{H,A}$ lies in the column
space of $\Sigma_H$ — automatic for any empirical sample with a finite
sample size), then $(\ast)$ holds with $\lambda_{\min}(R)$ replaced by
the smallest positive eigenvalue $\lambda_{+}(R)$.

**(III) Tikhonov-regularised audit form.** For the regularised audit-time
$R^2_{\tau}(h;A) := \mathrm{tr}\!\big(\Sigma_{h,A}^\top (\Sigma_h+\tau I)^{-1}\Sigma_{h,A}\big)/\mathrm{tr}(\Sigma_A)$
with $\tau > 0$ used in PCRL's audit code, the bound $(\ast)$ also
holds (the regulariser only shrinks $R^2_\tau$).

### Proof

**Step 1 — whiten.** Let $V_p := \Sigma_p^{-1/2} h_p$, so
$\mathbb{E}[V_p V_q^\top] = R_{pq}$ and $V := (V_1;\dots;V_k)$ has
$\mathbb{E}[VV^\top] = R$. Let
$D := \mathrm{block\text{-}diag}(\Sigma_1^{1/2},\dots,\Sigma_k^{1/2})$.
Then $H = DV$, $\Sigma_H = DRD$, and (in case I) $\Sigma_H^{-1} = D^{-1}R^{-1}D^{-1}$.

**Step 2 — per-purpose constraint in the whitened basis.** Define
$M_p := \Sigma_p^{-1/2}\Sigma_{p,A} \in \mathbb{R}^{d\times c}$.
Substituting,
$R^2(h_p;A) = \mathrm{tr}(M_p^\top M_p)/\mathrm{tr}(\Sigma_A) = \|M_p\|_F^2/\mathrm{tr}(\Sigma_A)$,
so by (C1), $\|M_p\|_F^2 \le \varepsilon\,\mathrm{tr}(\Sigma_A)$.

**Step 3 — concat $R^2$ in the whitened basis.** Stack $M := (M_1;\dots;M_k) = D^{-1}\Sigma_{H,A}$. Then
\begin{align*}
R^2(H;A) &= \frac{\mathrm{tr}(\Sigma_{H,A}^\top \Sigma_H^{-1}\Sigma_{H,A})}{\mathrm{tr}(\Sigma_A)} \\
         &= \frac{\mathrm{tr}(\Sigma_{H,A}^\top D^{-1} R^{-1} D^{-1} \Sigma_{H,A})}{\mathrm{tr}(\Sigma_A)}
         = \frac{\mathrm{tr}(M^\top R^{-1} M)}{\mathrm{tr}(\Sigma_A)}.
\end{align*}

**Step 4 — Rayleigh–Ritz on a matrix-valued quadratic form.** For any
positive-definite $K \in \mathbb{R}^{N\times N}$ and any
$M \in \mathbb{R}^{N\times c}$,
$$\mathrm{tr}(M^\top K^{-1} M) = \sum_{j=1}^c m_j^\top K^{-1} m_j \;\le\; \frac{\sum_{j=1}^c \|m_j\|^2}{\lambda_{\min}(K)} = \frac{\|M\|_F^2}{\lambda_{\min}(K)},$$
using $K^{-1} \preceq \lambda_{\min}(K)^{-1} I$ (Loewner order) on each
column.

**Step 5 — combine.** Apply Step 4 with $K = R$:
$$\mathrm{tr}(M^\top R^{-1} M) \;\le\; \frac{\|M\|_F^2}{\lambda_{\min}(R)} = \frac{\sum_p \|M_p\|_F^2}{\lambda_{\min}(R)} = \frac{\sum_p R^2(h_p;A)\cdot\mathrm{tr}(\Sigma_A)}{\lambda_{\min}(R)} \;\le\; \frac{k\varepsilon\,\mathrm{tr}(\Sigma_A)}{\lambda_{\min}(R)}.$$
Dividing by $\mathrm{tr}(\Sigma_A) > 0$ yields (I). $\square$

**Singular case (II).** If $R$ has nontrivial kernel and $u\in\ker(R)$,
then $\mathrm{Var}(u^\top V) = u^\top R u = 0$, so $u^\top V = 0$ a.s.,
hence $u^\top M = \mathbb{E}[u^\top V\cdot A^\top] = 0$. The columns of
$M$ therefore lie in $\mathrm{Range}(R)$. Replace $R^{-1}$ with the
Moore–Penrose pseudoinverse $R^{+}$. Since each column $m_j \in
\mathrm{Range}(R)$, the bound $m_j^\top R^{+} m_j \le \|m_j\|^2/\lambda_{+}(R)$
follows from the spectral decomposition of $R$ on its support. Sum
over $j$ and divide by $\mathrm{tr}(\Sigma_A)$. $\square$

**Tikhonov form (III).** Replace $\Sigma_H$ by $\Sigma_H + \tau I = D(R + \tau D^{-2})D$.
Since $\tau D^{-2} \succeq 0$, $\lambda_{\min}(R + \tau D^{-2}) \ge \lambda_{\min}(R)$,
and the same Rayleigh–Ritz step gives
$R^2_\tau(H;A) \le \mathrm{tr}(M^\top(R+\tau D^{-2})^{-1}M)/\mathrm{tr}(\Sigma_A)
\le k\varepsilon / \lambda_{\min}(R+\tau D^{-2}) \le k\varepsilon/\lambda_{\min}(R)$,
using $R^2_\tau(h_p;A) \le \varepsilon$ as the proxy-Lagrangian
enforces (`pcrl/training/losses.py:475-565`). $\square$

### Remarks

**R1 (PCRL-class encoder is *not* required).** The proof uses only the
per-purpose constraint $R^2(h_p;A) \le \varepsilon$ and the
concatenation operation. It does not invoke a frozen backbone, the LoRA
ansatz, the proxy-Lagrangian schedule, or any specific training
procedure. The bound therefore holds for *any* encoder family producing
per-purpose representations that satisfy (C1); PCRL is one instance.

**R2 (worst-case tightness).** Equality in $(\ast)$ is attained when
both (i) per-purpose constraints saturate
($R^2(h_p; A) = \varepsilon$ for all $p$), and (ii) the stacked
whitened cross-covariance $M$ lies entirely in the bottom eigenspace
of $R$. Construction: pick a unit bottom eigenvector
$v = (v_1;\dots;v_k)$ of $R$ with $Rv = \lambda_{\min}(R)v$, partition,
and set $M_p = \sqrt{\varepsilon\,\mathrm{tr}(\Sigma_A)}\,v_p\,e^\top/\|v\|$
for any unit vector $e \in \mathbb{R}^c$.

**R3 (typical looseness).** In the empirical PCRL regime the
per-purpose $R^2$ values are uneven across purposes and $M$ has
Frobenius mass spread across multiple eigenspaces of $R$, so the
bound is loose: see Section~3 for $45$–$70\%$ tightness utilisation.
A sharper bound would require knowledge of the projection of $M$ onto
each eigenspace of $R$, which is not summarised by a single scalar.

**R4 (limiting cases).**
- *Independent purposes* (cross-blocks $\Sigma_{pq} = 0$ for $p \ne q$):
  $R = I_{kd}$, $\lambda_{\min}(R) = 1$, bound $= k\varepsilon$. Tight
  when each per-purpose constraint binds and the $M_p$ point in the
  same direction.
- *Identical purposes* ($h_1 = h_2 = \dots = h_k$): $R = J_k \otimes I_d$
  has eigenvalue $k$ on the symmetric subspace and $0$ elsewhere.
  Case (II) applies with $\lambda_{+}(R) = k$, giving the bound
  $k\varepsilon/k = \varepsilon$, which matches reality
  ($\mathrm{concat} = h_1$, so $R^2(\mathrm{concat};A) = R^2(h_1;A) \le \varepsilon$).
- *Triviality regime.* The bound is informative iff
  $k\varepsilon/\lambda_{+}(R) \le 1$. Otherwise $R^2(H;A) \le 1$
  trivially.

**R5 (relationship to Proposition 5).** Proposition 5 (Zhao–Gordon
single-task impossibility, sister terminal) bounds $\sum_i\mathrm{Err}_i$
in terms of demographic shift $d_{\mathrm{TV}}$. Proposition 6 bounds
$R^2(\mathrm{concat};A)$ in terms of per-purpose linear compliance and
the cross-purpose representation geometry $\lambda_{+}(R)$. The two
propositions address orthogonal reviewer concerns: Proposition 5
addresses "is the single-task error floor sharp?"; Proposition 6
addresses "does the multi-purpose architecture leak more than each
purpose individually?".

---

## 3. Numerical verification (test split of Adult, $n = 15{,}060$)

Verification script: see `/tmp/tier2_prop6/verify_bound_v2.py` (uses
thin-SVD whitening with rcond $= 10^{-3}$, matching case II; full
table in `/tmp/tier2_prop6/numerical_verification_v2.json`). For each
seed-attribute cell we compute $R^2(h_p;A)$ per purpose,
$R^2(H;A)$ on the concatenation, and $\lambda_{+}(R)$ on the
block-whitened correlation.

**Round 5 baseline (Adult, R5 final.pt, 9 cells):**

| seed | attr | $R^2(h_1;A)$ | $R^2(h_2;A)$ | $R^2(h_3;A)$ | $\lambda_{+}(R)$ | bound $\frac{\sum R^2(h_p;A)}{\lambda_+(R)}$ | $R^2(H;A)$ obs | utilisation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | race      | 0.106 | 0.007 | 0.007 | 0.536 | 0.224 | 0.118 | 53% |
| 0 | sex       | 0.071 | 0.049 | 0.007 | 0.536 | 0.235 | 0.164 | 70% |
| 0 | age_group | 0.064 | 0.004 | 0.011 | 0.536 | 0.149 | 0.080 | 54% |
| 1 | race      | 0.007 | 0.008 | 0.010 | 0.437 | 0.055 | 0.030 | 53% |
| 1 | sex       | 0.020 | 0.042 | 0.006 | 0.437 | 0.155 | 0.083 | 53% |
| 1 | age_group | 0.116 | 0.002 | 0.011 | 0.437 | 0.296 | 0.133 | 45% |
| 2 | race      | 0.015 | 0.016 | 0.008 | 0.470 | 0.081 | 0.040 | 49% |
| 2 | sex       | 0.061 | 0.066 | 0.012 | 0.470 | 0.295 | 0.184 | 62% |
| 2 | age_group | 0.055 | 0.013 | 0.013 | 0.470 | 0.172 | 0.078 | 45% |

**Bound holds on 9 of 9 baseline cells; 9 of 9 are informative (bound $< 1$).**
Tightness utilisation (observed/bound) ranges $45$–$70\%$, median $53\%$,
consistent with R3 (the bound is tight only in the worst-case
construction).

**CROSSPURP-canonical-iterate cells (additional 9 cells, 18 total):**
the proxy-Lagrangian's canonical iterate after the cross-purpose
constraint is added drives some purposes to effective rank 0–1
(rep collapse), making $\lambda_{+}(R)$ either very small (s0, s2)
or very large (s1, where the surviving directions are nearly
orthogonal). Bound holds on all 9 additional cells; informative on
4 of 9. See `numerical_verification_v2.json` for the full table.

**Aggregate: bound holds on 18 of 18 cells; informative on 13 of 18.**

---

## 4. Comparison with currently-stated empirical observations

The current §5.4 cross-purpose-attack paragraph reports:
$R^2(\mathrm{concat}; \mathrm{race}) = 0.053$,
$R^2(\mathrm{concat}; \mathrm{sex}) = 0.147$,
$R^2(\mathrm{concat}; \mathrm{age\_group}) = 0.087$
(mean over three seeds, CROSSPURP run with explicit
$R^2(h_{\mathrm{concat}};A) \le 0.10$ constraint).

For the unconstrained R5 baseline (mean over three seeds), Proposition 6
predicts the ceilings $0.12, 0.23, 0.21$ respectively, against observed
$0.063, 0.144, 0.097$. Predicted vs. observed ratio (i.e., looseness)
ranges $1.5\times$–$2.3\times$. The bound is consistent with — and
predicts in advance — the magnitude of cross-purpose leakage that
motivated the §5.4 training-time fix.

---

## 5. Adversarial-review checklist

| Check | Verdict |
|---|---|
| All distributional assumptions explicit (centered, finite second moments, $\Sigma_p\succ 0$, $\mathrm{tr}\Sigma_A>0$) | ✓ stated |
| Trivial limits documented (independent / identical purposes, large $\varepsilon$) | ✓ in R4 |
| Constants explicit (no "for sufficiently large") | ✓ $k\varepsilon/\lambda_{+}(R)$ is closed-form |
| PCRL-class assumption identified as not load-bearing | ✓ R1 |
| Tightness honestly disclosed | ✓ R2 (worst-case sharp) + R3 (typical $45$–$70\%$ utilisation) |
| Numerical bound holds on $\ge 5$ cells with no violations | ✓ 18 of 18, 13 informative |
| Direction of inequality preserved at every step | ✓ Loewner monotone, Rayleigh-Ritz one-sided |
| Multivariate $R^2$ convention specified | ✓ trace-ratio, matches PCRL audit |
| Tikhonov-regularised audit form covered | ✓ part (III) |

---

## 6. Verification artefacts

- Proof draft / scratch: `/tmp/tier2_prop6/proof_draft.md`
- Verification scripts: `/tmp/tier2_prop6/verify_bound_v2.py`
- Per-cell numerics: `/tmp/tier2_prop6/numerical_verification_v2.json`

---

## 7. What this drop-in does NOT claim

- Does not say PCRL achieves the bound (it does not — empirical ratio
  $1.5\times$–$2.3\times$ below the ceiling).
- Does not claim cross-purpose leakage is *prevented* by per-purpose
  constraints — quite the opposite, it bounds the slack the per-purpose
  constraints leave for the joint representation.
- Does not address *nonlinear* cross-purpose recovery (XGB / MLP
  attackers). The bound is for linear leakage only, consistent with
  Alghamdi et al. (already cited in §C.3 of the theory drop-ins).
- Does not extend to the BIOS-medium / CelebA settings without
  re-verification — those have $k=2$ purposes and the empirical
  $\lambda_{+}(R)$ has not been computed.
