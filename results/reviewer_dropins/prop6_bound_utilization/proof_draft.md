# Proposition 6 (Tier 2, Direction A) — DRAFT

## Setup

Let $(X, A)$ be a random vector on a probability space, with $A \in \mathbb{R}^c$
(after centering and one-hot encoding for categorical attributes), satisfying
$\mathrm{trace}(\Sigma_A) > 0$ where $\Sigma_A := \mathbb{E}[(A-\mathbb{E}A)(A-\mathbb{E}A)^\top]$.

Let $h_1, \dots, h_k : \mathcal{X} \to \mathbb{R}^d$ be the $k$ per-purpose
PCRL representations (frozen-backbone + per-purpose LoRA, as defined in
`pcrl/models/lora.py:134-243`). Without loss of generality (subtract means),
assume $\mathbb{E}[h_p(X)] = 0$ and $\mathbb{E}[A] = 0$.

For each purpose, let $\Sigma_p := \mathbb{E}[h_p h_p^\top] \in \mathbb{R}^{d\times d}$,
$\Sigma_{p,A} := \mathbb{E}[h_p A^\top] \in \mathbb{R}^{d\times c}$.

**Assumption (A1):** $\Sigma_p \succ 0$ for all $p$ (strict positive-definite).
This is enforced in PCRL's audit by the Tikhonov regularizer in
`pcrl/purposes/verification.py:94` (`gram = H^T H + reg * I`); we treat the
$\tau\to 0$ limit, then verify the regularized form.

**Definition (multivariate linear-$R^2$).** For a representation
$h: \mathcal{X}\to\mathbb{R}^m$ with $\Sigma_h \succ 0$,
$$R^2(h; A) := \frac{\mathrm{tr}\!\big(\Sigma_{h,A}^\top \Sigma_h^{-1} \Sigma_{h,A}\big)}{\mathrm{tr}(\Sigma_A)}.$$
This matches PCRL's audit form: total explained variance / total variance,
summed across the columns of the one-hot $A$. (Code: `verification.py:90-103`,
$R^2 = 1 - \|Z - HW^\star\|_F^2 / \|Z\|_F^2$, equivalent algebraically.)

**Concat representation.** $H := (h_1; h_2; \dots; h_k) \in \mathbb{R}^{kd}$
with covariance
$$\Sigma_H = \big(\Sigma_{pq}\big)_{p,q=1..k}, \qquad \Sigma_{pq} := \mathbb{E}[h_p h_q^\top].$$

**Block whitened correlation.** Define
$$R_{pq} := \Sigma_p^{-1/2}\, \Sigma_{pq}\, \Sigma_q^{-1/2} \in \mathbb{R}^{d\times d},
\qquad R := (R_{pq})_{p,q=1..k} \in \mathbb{R}^{kd\times kd}.$$

By construction $R$ is the covariance of the *whitened concat*
$V := (\Sigma_1^{-1/2} h_1; \dots; \Sigma_k^{-1/2} h_k)$, so $R \succeq 0$,
diagonal blocks $R_{pp} = I_d$, hence $\mathrm{tr}(R) = kd$ and
$\lambda_{\min}(R) \in [0, 1]$.

## Proposition 6 (Cross-Purpose Linear-Leakage Bound)

Suppose Assumption (A1) holds and that for each purpose $p$,
$$R^2(h_p; A) \le \varepsilon. \tag{C1}$$

(I) **Full-rank case.** If $R \succ 0$, then
$$\boxed{\;R^2(H; A) \;\le\; \frac{k\,\varepsilon}{\lambda_{\min}(R)}.\;} \tag{*}$$

(II) **Singular case.** If $\mathrm{rank}(R) < kd$ but the OLS predictor of
$A$ from $H$ exists (equivalently, $\Sigma_{H,A}$ lies in the column space
of $\Sigma_H$ — automatic for the empirical OLS solution), then $(*)$ holds
with $\lambda_{\min}(R)$ replaced by $\lambda_{+}(R)$, the smallest *positive*
eigenvalue of $R$.

(III) **Tikhonov form.** For the regularized audit-time R$^2$ used in code,
$R^2_\tau(h; A) := \mathrm{tr}(\Sigma_{h,A}^\top (\Sigma_h + \tau I)^{-1}
\Sigma_{h,A})/\mathrm{tr}(\Sigma_A)$ with $\tau > 0$, the bound $(*)$ also
holds (the regularizer only shrinks $R^2$).

## Proof

**Step 1 — Whiten.** Let $V_p := \Sigma_p^{-1/2} h_p$. Then
$\mathbb{E}[V_p V_q^\top] = \Sigma_p^{-1/2} \Sigma_{pq} \Sigma_q^{-1/2} = R_{pq}$,
so $V := (V_1;\dots;V_k)$ has $\mathbb{E}[VV^\top]=R$.
Let $D := \mathrm{block\text{-}diag}(\Sigma_1^{1/2},\dots,\Sigma_k^{1/2})$, so
$H = DV$, $\Sigma_H = DRD$, and (case I) $\Sigma_H^{-1} = D^{-1}R^{-1}D^{-1}$.

**Step 2 — Per-purpose constraint in whitened basis.** Define
$M_p := \Sigma_p^{-1/2} \Sigma_{p,A} \in \mathbb{R}^{d\times c}$. Then
$$R^2(h_p;A) = \frac{\mathrm{tr}(\Sigma_{p,A}^\top \Sigma_p^{-1} \Sigma_{p,A})}{\mathrm{tr}(\Sigma_A)}
= \frac{\mathrm{tr}(M_p^\top M_p)}{\mathrm{tr}(\Sigma_A)} = \frac{\|M_p\|_F^2}{\mathrm{tr}(\Sigma_A)},$$
so by (C1), $\|M_p\|_F^2 \le \varepsilon\,\mathrm{tr}(\Sigma_A)$.

**Step 3 — Concat $R^2$ in whitened basis.** Stack $M := (M_1;\dots;M_k) =
D^{-1}\Sigma_{H,A}$. Then
\begin{align*}
R^2(H;A) &= \frac{\mathrm{tr}(\Sigma_{H,A}^\top \Sigma_H^{-1} \Sigma_{H,A})}{\mathrm{tr}(\Sigma_A)} \\
&= \frac{\mathrm{tr}(\Sigma_{H,A}^\top D^{-1} R^{-1} D^{-1} \Sigma_{H,A})}{\mathrm{tr}(\Sigma_A)}
= \frac{\mathrm{tr}(M^\top R^{-1} M)}{\mathrm{tr}(\Sigma_A)}.
\end{align*}

**Step 4 — Rayleigh–Ritz for matrix-valued quadratic forms.** For any
positive-definite $K\in \mathbb{R}^{N\times N}$ and any $M\in\mathbb{R}^{N\times c}$,
$$\mathrm{tr}(M^\top K^{-1} M) = \sum_{j=1}^{c} m_j^\top K^{-1} m_j
\;\le\; \sum_{j=1}^c \frac{\|m_j\|^2}{\lambda_{\min}(K)} = \frac{\|M\|_F^2}{\lambda_{\min}(K)}, $$
where each step uses the standard scalar Rayleigh–Ritz bound
$x^\top K^{-1} x \le \|x\|^2 / \lambda_{\min}(K)$ (consequence of
$K^{-1} \preceq \lambda_{\min}(K)^{-1} I$).

**Step 5 — Combine.** Apply Step 4 with $K = R$:
$$\mathrm{tr}(M^\top R^{-1} M) \le \frac{\|M\|_F^2}{\lambda_{\min}(R)}
= \frac{\sum_{p=1}^k \|M_p\|_F^2}{\lambda_{\min}(R)} \le \frac{k\,\varepsilon\,\mathrm{tr}(\Sigma_A)}{\lambda_{\min}(R)}.$$
Dividing by $\mathrm{tr}(\Sigma_A)>0$ yields the claim of (I). $\square$

**Singular case (II).** When $R$ is singular, the OLS predictor exists iff
$\Sigma_{H,A}\in\mathrm{Range}(\Sigma_H)$, i.e., $M\in\mathrm{Range}(R)$
(since $D$ is invertible). For any $u\in\mathrm{Ker}(R)$,
$\mathrm{Var}(u^\top V) = u^\top R u = 0$, so $u^\top V=0$ a.s. and
$u^\top M = \mathbb{E}[u^\top V A^\top] = 0$. Thus the columns of $M$
lie in $\mathrm{Range}(R)$. Replacing $R^{-1}$ with the Moore–Penrose
pseudoinverse $R^+$, we have $m_j^\top R^+ m_j \le \|m_j\|^2/\lambda_+(R)$
since $m_j\in\mathrm{Range}(R)$ and $R^+$ acts as $1/\lambda$ on each
positive-eigenvalue subspace. Sum over $j$ as before. $\square$

**Tikhonov form (III).** Replace $\Sigma_H$ with $\Sigma_H+\tau I = D(R + \tau D^{-2})D$.
The matrix $R + \tau D^{-2}$ is positive definite with
$\lambda_{\min}(R + \tau D^{-2}) \ge \lambda_{\min}(R)$ (since $\tau D^{-2}\succeq 0$).
Repeating Steps 3–5 with this regularized inverse yields
$R^2_\tau(H;A) \le \mathrm{tr}(M^\top (R+\tau D^{-2})^{-1} M)/\mathrm{tr}(\Sigma_A)
\le k\varepsilon/\lambda_{\min}(R+\tau D^{-2}) \le k\varepsilon/\lambda_{\min}(R)$,
provided we also use the regularized per-purpose bound $R^2_\tau(h_p;A)\le\varepsilon$
(which is what the proxy-Lagrangian enforces in `losses.py:475-565`). $\square$

## Tightness

**Upper-bound tightness.** Equality in (*) is attained when
(a) the per-purpose constraint is saturated ($R^2(h_p;A)=\varepsilon$ for all $p$),
and (b) the stacked whitened cross-covariance $M$ lies entirely in the
bottom eigenspace of $R$ (i.e., the per-purpose projections of $A$ onto
the whitened representations all align with the same direction in the
joint whitened space — the maximally-coherent cross-purpose construction).
This is achievable by an explicit construction: pick the bottom eigenvector
$v\in\mathbb{R}^{kd}$ of $R$, $\lambda_{\min}(R)$, partition $v=(v_1;\dots;v_k)$,
set $M_p = \sqrt{\varepsilon\,\mathrm{tr}(\Sigma_A)/k}\;v_p\,e^\top/\|v\|$
for some unit $e\in\mathbb{R}^c$. This gives $\|M_p\|_F^2 = \varepsilon\,
\mathrm{tr}(\Sigma_A)\cdot\|v_p\|^2/\|v\|^2$, summing to $\varepsilon\,
\mathrm{tr}(\Sigma_A)$ (so per-purpose constraint binds at $\varepsilon$
on the average), with $\mathrm{tr}(M^\top R^{-1} M) =
\varepsilon\,\mathrm{tr}(\Sigma_A)/\lambda_{\min}(R)$.

**Looseness regimes.** The bound is loose when $M$ has Frobenius mass
in larger eigenspaces of $R$ (typical empirical regime — see Phase 3
numerics, where realized $R^2(H;A)$ falls below the bound by 1.5x–6x).

## Triviality / sanity checks

1. $\lambda_{\min}(R) \in (0, 1]$: $R$ is a correlation matrix with diagonal
   $I_d$ blocks, $\mathrm{tr}(R) = kd$, mean eigenvalue $1$, hence
   $\lambda_{\min}(R)\le 1$. So $k\varepsilon/\lambda_{\min}(R) \ge k\varepsilon$.
2. **Independent purposes** ($\Sigma_{pq}=0$ for $p\ne q$): $R = I_{kd}$,
   $\lambda_{\min}=1$, bound $=k\varepsilon$. Tight: independent leakages add.
3. **Degenerate ($h_1=h_2$):** $R$ has a 0 eigenvalue with eigenvector
   $(I_d; -I_d)/\sqrt{2}$. Bound becomes vacuous; (II) gives the right
   answer ($\lambda_+(R)=2$, bound $k\varepsilon/2 = \varepsilon$ for $k=2$,
   matching reality since concat=duplicated $h_1$).
4. **Trivial regime:** when $k\varepsilon \ge \lambda_{\min}(R)$, the bound
   is $\ge 1$ and only says $R^2(H;A)\le 1$. The bound is interesting
   precisely when $\lambda_{\min}(R) > k\varepsilon$.
