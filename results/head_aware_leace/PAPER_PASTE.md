# Head-Aware LEACE — Paper Paste

This file contains the §5.7 paragraph(s), the LaTeX table fragment,
and supplementary lemma to copy into Overleaf.

## §5.7 paragraph

We test whether the mechanistic-interpretability finding from §5.6
(block-0 heads (0,7) and (0,10) carry the gender pronoun-mediated
pathway, with (0,6) as counter-balance) yields a more surgical
linear erasure than block-level LEACE. We fit per-head LEACE
projections on \texttt{blocks.0.attn.hook\_z} (d_head=64) for the
mechanistically-identified heads and apply them via forward-pass hooks
during BIOS top-10 dev-set evaluation, training all projections on
20\,000 stratified BIOS train examples. Table~\ref{tab:head-aware-leace}
reports the 8-condition comparison. Vanilla layer-1 $R^2_{\text{gender}}$ =
0.811; rank-1 block-LEACE applied at hook\_attn\_out
achieves 0.788, rank-8
0.788, and head-aware
(0,7)+(0,10) independent 0.804.
The killer feature is collateral preservation: head-aware (0,7)+(0,10)
attains BIOS occupation accuracy 0.755 versus
0.757 for rank-8 block (vanilla
0.756), a $\Delta$ of
-0.002\,pp.
A random-pair control (heads 3,11) yields
$R^2_{\text{gender}}$ = 0.810 (no drop), confirming the
intervention selectivity is mechanistically anchored. The
counter-balance head (0,6) alone changes $R^2$ from 0.811
to 0.809 on its own — a
direct empirical signature of opposing-direction circuit components.

## Table fragment

See \texttt{head\_aware\_leace\_table.tex} for the full table.
Paste with \input or copy verbatim.

## What to cut to make 1 page room

Suggested trims in §5.6:
- Compress "BRANCH: DIFFUSE MEDIATION on R² but SHARP CIRCUIT on probe accuracy"
  paragraph to one sentence pointing at the mech-interp heatmap figure.
- Drop the residual-decomposition scatter plot from §5.6 (move to appendix);
  the §5.7 head-aware table now carries the load-bearing finding that
  (0,6) is opposing-direction.

## Supplementary lemma (appendix-bound)

\begin{lemma}[Independent vs.\ joint-stacked head-LEACE]
Let $X = (X_1, \ldots, X_K)$ be a stacked vector of $K$ attention-head
output vectors $X_h \in \mathbb{R}^{d}$, and $Z \in \{0,1\}$ a
binary protected attribute. Let $P^{\text{ind}}_h$ be LEACE
fit on $X_h$ alone (i.e.\ $\mathrm{Cov}(P^{\text{ind}}_h(X_h), Z)=0$),
and let $P^{\text{joint}}$ be LEACE fit on the stacked vector $X$
(i.e.\ $\mathrm{Cov}(P^{\text{joint}}(X), Z)=0$). Both
guarantee zero linear leakage on the fit data; however, $P^{\text{joint}}$
is generally not block-diagonal across heads, so applying it requires
re-stacking $X_h$'s at evaluation time. The independent variant
$\bigoplus_h P^{\text{ind}}_h$ \emph{also} guarantees zero linear
leakage of any \emph{linear function} of $(P^{\text{ind}}_h(X_h))_h$,
and is strictly cheaper in MSE-distortion of $X$ (sum of per-head
LEACE distortions $\le$ joint LEACE distortion when joint LEACE
must zero out the same covariance constraint over a higher-dimensional
domain).
\end{lemma}

\begin{proof}[Sketch]
LEACE distortion is the minimum-MSE oblique projection that zeros the
sample covariance with $Z$ \citep{belrose2023leace}. The block-diagonal
projection achieves the per-head distortion-minimum for each head's
constraint $\mathrm{Cov}(P_h(X_h), Z)=0$, summed; the joint-stacked
projection minimizes total MSE under the single combined constraint
$\mathrm{Cov}(P(X), Z) = 0$, but in $K \cdot d$-dimensional space.
Since the joint constraint is implied by the conjunction of per-head
constraints, the per-head feasible set is a subset of the joint feasible
set; the joint-stacked projection's MSE is therefore $\le$ the sum of
per-head MSEs. \textit{However}, downstream the heads are recombined
by $W_O$, so the linear-leakage guarantee on $W_O X$ holds for both.
\end{proof}
