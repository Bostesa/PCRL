# Paper-paste responses to reviewer critiques 7, 8, 12, 13, 14, 15

Single-source paste-ready prose and LaTeX for the deadline-weighted critique
fixes. Each section is independently liftable. Terminal 1 owns critiques
#1, #2, #6, #9, #10, #11 — those edits are not duplicated here.

---

## §A · Critique 8 — Per-(dataset, purpose) compliance decomposition

### A.1  Three-way breakdown table (paste-ready LaTeX)

```latex
% Per-(dataset, purpose) decomposition of compliance into cleanly | collapse | failed.
% Counts derived from results/v2_pcrl_variance_constrained/target_cells.json.
\begin{table}[t]
\centering
\small
\begin{tabular}{l l c c c c}
\toprule
Dataset & Purpose & Pairs & Cleanly & Collapse & Failed \\
\midrule
\multirow{3}{*}{Adult}
  & income\_prediction      & 6 & 0 & 5 & 1 \\
  & employment\_analysis    & 9 & 3 & 6 & 0 \\
  & education\_assessment   & 9 & 0 & 9 & 0 \\
\midrule
\multirow{3}{*}{HMDA}
  & underwriting            & 6 & 2 & 4 & 0 \\
  & pricing\_analysis       & 6 & 0 & 4 & 2 \\
  & fair\_lending\_audit    & 6 & 0 & 6 & 0 \\
\midrule
\multirow{3}{*}{Diabetes}
  & billing\_audit              & 6 & 2 & 4 & 0 \\
  & quality\_research           & 6 & 0 & 5 & 1 \\
  & clinical\_decision\_support & 6 & 0 & 6 & 0 \\
\midrule
\multicolumn{2}{l}{\textbf{Total}}
  & 60 & 7 & 49 & 4 \\
\bottomrule
\end{tabular}
\caption{Per-(dataset, purpose) decomposition of the 60-cell compliance grid.
Cleanly-compliant requires $R^2_{\text{1H}} < 0.05$ \emph{and} per-dimension
standard deviation $\bar\sigma_{\text{dim}} \geq 0.5$ \emph{and} effective rank
$\geq 2$. Collapse-compliant satisfies $R^2_{\text{1H}} < 0.05$ but violates at
least one health threshold. Failed has $R^2_{\text{1H}} \geq 0.05$. Of the
seven cleanly-compliant cells, three are Adult/employment\_analysis on seed~0,
two are HMDA/underwriting on seed~2, and two are Diabetes/billing\_audit on
seed~2 — every other (dataset, purpose) row is exclusively collapse-or-failed.
The decomposition is computed from the same artifacts that feed Table~1.}
\label{tab:compliance_decomposition}
\end{table}
```

### A.2  Disclosure paragraph (paste into §5)

> Table~\ref{tab:compliance_decomposition} stratifies the 60 cells of
> Table~1 by *how* compliance is achieved. Of the 56 R²-passing cells, 49
> (88\%) satisfy the linear certificate at the cost of one or both
> representation-health thresholds — per-dimension standard deviation
> $\bar\sigma_{\text{dim}} < 0.5$ or effective rank $< 2$ — and only seven
> are cleanly-compliant. The decomposition is dataset-asymmetric: every
> Adult `education_assessment` cell collapses (9/9), every Diabetes
> `clinical_decision_support` cell collapses (6/6), and the only
> cleanly-compliant cells outside Adult `employment_analysis` seed~0 are
> two HMDA `underwriting` pairs and two Diabetes `billing_audit` pairs,
> all on seed~2. We previously criticised the per-purpose LAFTR baseline
> for achieving compliance on Diabetes through representation collapse;
> the same critique applies to PCRL on Diabetes, where 16/18 of our
> R²-passing cells are collapse-compliant rather than cleanly-compliant.
> We therefore report Table~1 totals alongside this decomposition rather
> than in isolation, and we treat compliance-via-collapse as a known
> failure mode of the current method (see §\ref{sec:collapse_mitigation})
> rather than as a property uniquely diagnostic of an adversarial
> baseline.

### A.3  LaTeX edit to Table~1

The cleanest integration is to keep Table~1 as the headline R²-pass / Adj-pass
grid and add Table~\ref{tab:compliance_decomposition} as Table~1b directly
below it, paired in a single floating environment so reviewers see both at
once. Concretely, replace the existing `\end{table}` after Table~1 with:

```latex
\end{table}

% Table 1b — compliance decomposition (cleanly vs collapse vs failed).
% Same row schema as Table 1 but stratified by compliance type. See §A.1.
\input{tables/compliance_decomposition}
```

If page budget rules out two adjacent tables, replace Table~1's ``Hidden``
column with three side-by-side ``C / C\textsubscript{coll} / F'' columns:

```latex
% Replacement column header for Table 1
... & $R^2_{\text{1H}}$ & $R^2_{\text{DA}}$ & C & C$_{\text{coll}}$ & F \\
\midrule
% Replacement Adult row (concatenate purposes; counts come from §A.1)
Adult        & 24 & 0.0119 & --     & 3 & 20 & 1 \\
HMDA         & 18 & 0.0201 & --     & 2 & 14 & 2 \\
Diabetes     & 18 & 0.0092 & --     & 2 & 15 & 1 \\
\midrule
\textbf{Total} & 60 & --   & --     & 7 & 49 & 4 \\
```

with caption text amended:

> ``C'' is cleanly-compliant, ``C\textsubscript{coll}'' is
> collapse-compliant, ``F'' is R²-failed.

---

## §B · Critique 13 — §5.x subsection: "Compliance via collapse and unsuccessful mitigation"

### B.1  Subsection scaffold

```latex
\subsection{Compliance via collapse and unsuccessful mitigation}
\label{sec:collapse_mitigation}

The compliance decomposition in Table~\ref{tab:compliance_decomposition}
shows that 49 of the 56 R²-passing cells in our 60-cell grid satisfy the
linear-erasure certificate at the cost of representation health: their
per-dimension standard deviation $\bar\sigma_{\text{dim}}$ falls below
$0.5$, their effective rank below $2$, or both. We refer to this regime
as \emph{compliance-via-collapse}. It is the same regime under which we
critique the per-purpose LAFTR baseline on Diabetes (\S\ref{sec:laftr_compare}),
and the same regime that motivated the independent-encoder threat model
in \S\ref{sec:independent_encoders}. By Table~\ref{tab:compliance_decomposition}
this regime applies to most of PCRL's compliance evidence as well, and
in particular to all 9 \texttt{education\_assessment} cells on Adult, all
6 \texttt{clinical\_decision\_support} cells on Diabetes, and 5 of 6
\texttt{quality\_research} cells on Diabetes.

We interpret compliance-via-collapse as a known failure mode of the
proxy-Lagrangian primal optimiser at the $\lambda_{\min} = 5$ floor we
introduced in Round~5: the floor stabilises the R² constraint by pinning
the dual but trades a portion of representation entropy for that
stability. Removing the floor reintroduces the dual-saddle oscillation
documented in Appendix~\ref{app:r1_appendix}; keeping it produces
collapse on the cells reported in Appendix~\ref{app:per_cell}.

\paragraph{Tested mitigation 1 — Appendix~H mitigation, original VICReg
schedule.} \emph{[Existing Appendix~H text reproduced or summarised here:
40 originally-collapse-compliant pair-seeds were retrained for 150 epochs
with a 5$\times$ stronger VICReg variance term. Result: zero recoveries;
one pair lost R²-compliance.]} The negative outcome was reported but its
implications were not surfaced as a §5 result; we correct that here.

\paragraph{Tested mitigation 2 — variance-constrained retrain.}
The natural follow-up replaces VICReg's soft variance term with a hard
per-dimension variance Lagrangian (target $\bar\sigma_{\text{dim}} \geq
0.5$, dual ascent rate $\eta_\lambda = 0.05$, $\lambda \in [0.1, 100]$)
and adds a soft effective-rank penalty (sigmoid hinge on the
participation-ratio proxy at threshold~$2$). Warm-started from the
Round~5/Round~7 checkpoints, retrained for 150 epochs per
(dataset, seed). %% [APPENDIX-DETAIL: see infra/varconstraint/].

%% [APPENDIX-DETAIL: full hyperparameters and per-cell health curves in
%% Appendix~\ref{app:varconstraint}.]
```

### B.2  Closing paragraph **A — POSITIVE outcome** (variance-constrained run lifts ≥10 cells to cleanly-compliant)

> The variance-constrained retrain converts \emph{N\textsubscript{lift}}
> collapse-compliant cells to cleanly-compliant while preserving R² at or
> below the original $0.05$ threshold (mean $\bar\sigma_{\text{dim}}$
> rises from $X.XX$ to $Y.YY$; mean effective rank from $X.XX$ to
> $Y.YY$). Compliance-via-collapse remains the dominant regime for the
> remaining cells, but the result demonstrates that the failure mode is
> not intrinsic to the joint multi-purpose architecture — it is the
> consequence of the proxy-Lagrangian's neutrality on representation
> health, and an explicit dual on $\bar\sigma_{\text{dim}}$ closes the
> gap on a non-trivial fraction of the grid. We treat the variance-
> constrained recipe as the recommended training protocol going forward
> and the original recipe as the source of the data in
> Table~\ref{tab:compliance_decomposition} that this work supersedes.

### B.3  Closing paragraph **B — NEGATIVE outcome** (variance-constrained run fails)  [⬅ LIVE: use this one]

> The variance-constrained retrain does not produce a cleanly-compliant
> cell. Across the 7 (dataset, seed) units that completed within the
> 5-hour budget — three on Adult, three on HMDA, one on Diabetes,
> covering 48 pair-seeds — the effective-rank soft penalty does
> recover the rank ledger (mean post-training effective rank rises from
> below $2$ to between $2.5$ and $9$ depending on purpose), but the hard
> variance Lagrangian fails to push $\bar\sigma_{\text{dim}}$ above the
> $0.5$ threshold on any purpose: post-training values stay in the
> $[0.20, 0.40]$ band across every cell, even as the dual variable
> climbs to $\lambda_{\text{var}} \approx 11$ on Adult
> \texttt{education\_assessment}. The R² ledger is partially preserved
> (44/48 pair-seeds retain $R^2_{\text{1H}} < 0.05$, four lose
> compliance) and the auditor delta widens on Adult and HMDA. Combined
> with the Appendix~H result, two distinct mitigation strategies
> targeting collapse have failed: a stronger VICReg variance term, and
> a hard Lagrangian variance dual paired with an effective-rank penalty.
> We therefore report compliance-via-collapse as a property of PCRL's
> R5/R7 training recipe that we do not yet know how to remove without
> sacrificing the R² ledger, and we propose two refinements left to
> future work: (i) lifting the variance constraint into a Cotter-style
> proxy-Lagrangian primal so that primal updates respect the variance
> threshold the dual is targeting, rather than treating the threshold as
> an external dual signal the primal can ignore, and (ii) replacing the
> $\lambda_{\min} = 5$ floor with a sigmoid-modulated Cotter swap that
> activates the floor only when the proxy R² has stabilised. Neither
> refinement is tested here.

### B.4  Live numbers from the variance-constrained run (for paragraph B)

Per-cell summary (R²-pass / adj-pass / cleanly-compliant / per_dim_std
range / effective-rank range), 7 cells completed before the 5-hour cap
fired, diabetes_s1 and diabetes_s2 truncated:

```
adult_s0      r2=7/8  adj=2/8  clean=0/8  pds=[0.23, 0.34]  eff_rank=[1.16, 9.01]
adult_s1      r2=8/8  adj=1/8  clean=0/8  pds=[0.24, 0.33]  eff_rank=[1.83, 2.93]
adult_s2      r2=7/8  adj=3/8  clean=0/8  pds=[0.22, 0.30]  eff_rank=[1.27, 4.75]
diabetes_s0   r2=6/6  adj=6/6  clean=0/6  pds=[0.20, 0.24]  eff_rank=[1.28, 4.33]
hmda_s0       r2=4/6  adj=0/6  clean=0/6  pds=[0.23, 0.35]  eff_rank=[2.83, 9.06]
hmda_s1       r2=6/6  adj=0/6  clean=0/6  pds=[0.29, 0.31]  eff_rank=[2.47, 4.55]
hmda_s2       r2=6/6  adj=0/6  clean=0/6  pds=[0.35, 0.40]  eff_rank=[3.59, 8.54]
```

Final Lagrangians on adult_s0: variance dual climbed to
$\lambda_{\text{var}}^{\text{income}} = 2.90$,
$\lambda_{\text{var}}^{\text{employment}} = 4.31$,
$\lambda_{\text{var}}^{\text{education}} = 11.16$ — none saturated at the
upper bound $100$; R² duals pinned at the floor $\lambda_{\min} = 5$ for
six of eight pairs, climbed to $\sim 21$ for `income_prediction/{race,sex}`.

---

## §C · Critique 14 — Compute cost reframing (option b: "feasibility demonstration")

### C.1  One-paragraph rewrite (replace the existing "deployment" / "training cost" paragraph in §6 or wherever it currently sits)

> We position PCRL as a feasibility demonstration of joint
> multi-purpose representation learning under \emph{certifiable}
> compliance constraints, rather than as a deployment-ready alternative
> to per-purpose encoders. The training cost of joint optimisation under
> proxy-Lagrangian constraints exceeds that of $K$ independent
> per-purpose encoders by a factor of $7$--$18\times$ on the
> tabular-scale benchmarks reported here, with negligible inference-side
> savings. The methodological argument the paper makes — that a single
> backbone with per-purpose adapters can simultaneously satisfy a
> per-purpose \emph{linear leakage bound} ($R^2_{\text{1H}} < 0.05$) on
> every (purpose, attribute) pair under a single jointly-optimised
> training loss — does not rest on the training-cost ledger. Whether
> joint training amortises in regimes where $K$ is large enough that
> $K$ independent encoders are infeasible to maintain or to redeploy in
> step with regulatory updates, or where the joint backbone enables
> shared inference-time compute or transferable purpose embeddings, is a
> question we do not resolve in this work and that we leave to future
> evaluation. The reader should treat the experiments here as evidence
> that the joint-with-certificates regime is reachable, not as evidence
> that it is the cheapest route to per-purpose compliance.

### C.2  Term swaps elsewhere in the paper

Replace these phrasings wherever they appear (Abstract, Intro, Conclusion,
Discussion):

| Current language                                        | Replacement                                                   |
|---------------------------------------------------------|---------------------------------------------------------------|
| ``deployment-ready'' / ``production-ready''             | ``research demonstration'' / ``feasibility study''            |
| ``parameter-efficient deployment''                      | ``parameter-efficient \emph{architecture}'' (drop "deployment")|
| ``reduces serving cost''                                | strike the clause; the paper does not measure serving cost     |
| ``competitive with per-purpose encoders''               | ``methodologically comparable to per-purpose encoders''        |
| ``practical alternative''                               | ``architectural alternative''                                  |
| ``scales to many purposes''                             | ``demonstrates feasibility for $K \leq 4$ purposes''           |

### C.3  Abstract patch

Replace the abstract sentence claiming deployment efficiency with:

> We show that the joint-multi-purpose-with-certifiable-compliance
> regime is reachable on three tabular benchmarks (Adult, HMDA,
> Diabetes), and we report training-cost and representation-health
> trade-offs that delineate where the regime is currently competitive
> with per-purpose encoders and where it is not.

---

## §D · Critique 7 — LAFTR comparison presentation patch (no relaunch)

### D.1  Replacement paragraph for §\ref{sec:laftr_compare}

> We compare PCRL against per-purpose LAFTR \citep{madras2018laftr} on
> the 60-cell grid. The two methods optimise toward different
> compliance criteria: PCRL targets a hard linear-R² constraint
> ($R^2_{\text{1H}} < 0.05$) via a proxy-Lagrangian dual on every
> (purpose, attribute) pair, while LAFTR targets an adversarial loss in
> which a learned 2-layer MLP discriminator competes with the encoder.
> We therefore report LAFTR on \emph{both} metrics. Under its native
> adversarial-loss criterion (Table~\ref{tab:laftr_native}), LAFTR
> achieves discriminator accuracy at chance on
> $56/60$ cells (per-dataset: Adult $21/24$, HMDA $18/18$,
> Diabetes $17/18$; chance is defined as discriminator validation accuracy
> within $2$pp of the per-attribute majority baseline). Under our
> $R^2_{\text{1H}} < 0.05$ criterion (Table~1), LAFTR achieves $15/60$
> (Adult $0/24$, HMDA $0/18$, Diabetes $15/18$). The gap
> reflects the metric, not the method's intent: an adversarial encoder
> that satisfies its discriminator's no-recovery criterion can still
> leave linearly-recoverable signal on directions the discriminator did
> not learn to exploit. The methodologically clean apples-to-apples
> comparison would be to retrain LAFTR with a hard $R^2_{\text{1H}} <
> 0.05$ constraint via the same proxy-Lagrangian wrapper PCRL uses; we
> leave that experiment to future work and explicitly do \emph{not}
> claim that PCRL is superior to LAFTR on LAFTR's native objective.

### D.2  Numbers to fill in

Filled 2026-05-07: $X_{\text{native}} = 56$ and $X_{\text{r2}} = 15$
(both out of $60$). Computed from LAFTR
`results/laftr_benchmark/<ds>/<purpose>/seed_<s>/metrics.json` for Adult
and `/tmp/laftr_stage3/results/<ds>/<purpose>/seed_<s>/metrics.json` for
HMDA and Diabetes. Native-metric criterion: final-epoch
`val_disc_acc[<attr>]` $\le$ majority $+ 2$pp. R$^2$ criterion: same
files' `metrics.per_attr[<attr>].r2_onehot` $\le 0.05$. Per-cell CSV at
`/tmp/laftr_native_vs_r2.csv`.

### D.3  Caveat sentence (paste once into Limitations)

> The LAFTR comparison reported in Table~1 audits both methods under
> PCRL's native $R^2_{\text{1H}} < 0.05$ criterion, which LAFTR was not
> trained against; a hard-$R^2$ LAFTR variant via proxy-Lagrangian is
> the right future comparison and would settle whether the gap reflects
> method or metric.

---

## §E · Critique 15 — VICReg-below-floor sentence for Limitations

### E.1  Paste-ready sentence (single sentence, drop into Limitations § or Appendix)

> VICReg's variance term, configured at the canonical $\gamma = 1.0$
> floor, fails to keep $\bar\sigma_{\text{dim}} \geq 0.5$ on
> \textbf{49 of 56} R²-passing cells under our recipe — i.e., the
> auxiliary loss operates below its design point in 88\% of compliant
> cells — and we do not report a VICReg-vs-no-auxiliary ablation in
> this work; the right ablation, isolating the contribution of the
> variance term and the covariance term separately, is left to future
> evaluation.

### E.2  Optional second sentence if a footnote is allowed

> The variance-constrained retrain in §\ref{sec:collapse_mitigation}
> replaces the VICReg variance term with a hard Lagrangian dual and
> finds the dual cannot lift $\bar\sigma_{\text{dim}}$ either, which is
> consistent with — but does not establish — VICReg's variance term
> being a vestigial component of the current recipe.

---

## §F · Critique 12 — Principled K ≥ 6 threshold via convex-combination identity

### F.1  Justification paragraph (replace the current "we use K ≥ 6" sentence in §\ref{sec:per_class_constraints} or wherever the threshold is introduced)

> The per-class constraint regime is gated by a threshold on the
> number of classes $K$ of the disallowed attribute. The threshold is
> set from the Convex-Combination Identity of
> Proposition~\ref{prop:convex_combination}: the standard one-hot $R^2$
> equals
> $\sum_{k=1}^{K} w_k\, R^2_{\text{OvR}, k}$ with weights $w_k = \pi_k(1
> - \pi_k) / \sum_j \pi_j(1 - \pi_j)$. When the empirical class
> distribution $(\pi_k)$ is sufficiently imbalanced, the largest
> per-class $R^2_{\text{OvR}, k}$ — the dominant-axis quantity of
> §\ref{sec:dominant_axis} — can exceed the one-hot $R^2$ by a factor
> determined by the smallest non-trivial weight $\min_k w_k$. We define
> the threshold operationally as: apply per-class constraints when
> $\min_k \pi_k(1 - \pi_k)$ is small enough that one-hot $R^2$ can
> hide an $R^2_{\text{OvR}, k}$ above the certificate threshold,
> empirically when $\min_k \pi_k(1 - \pi_k) < 0.05 \cdot \overline{\pi
> (1 - \pi)}$, which on our three datasets corresponds to $K \geq 6$.
> The trigger condition fires on Diabetes \texttt{age\_bucket} (10
> classes, with the smallest two categories carrying $0.21\%$ and $0.76\%$ of
> mass — $\min_k \pi_k(1 - \pi_k) \approx 0.0021$ versus an average
> $\overline{\pi(1 - \pi)} \approx 0.0817$, ratio $0.026 < 0.05$), and does not fire on any
> attribute with $K < 6$ in our grid because no smaller-$K$ attribute
> exhibits the required imbalance. The threshold is therefore a
> property of the empirical $\pi_k$ distribution, not of the dataset
> identity; an attribute with $K = 4$ and $\min_k \pi_k(1 - \pi_k) <
> 0.05 \cdot \overline{\pi(1 - \pi)}$ would also trigger per-class
> constraints under this rule.

### F.2  Empirical numbers backing the paragraph

Verified 2026-05-07 against the training split loaded by
`pcrl/data/diabetes.py` (N=50,053). Full $\pi_k$ distribution:

| $k$ | count | $\pi_k$  | $\pi_k(1-\pi_k)$ |
|----:|------:|---------:|-----------------:|
| 0   |   107 | 0.00214  | 0.002133         |
| 1   |   379 | 0.00757  | 0.007515         |
| 2   |   805 | 0.01608  | 0.015824         |
| 3   |  1917 | 0.03830  | 0.036833         |
| 4   |  4814 | 0.09618  | 0.086928         |
| 5   |  8733 | 0.17448  | 0.144034         |
| 6   | 11202 | 0.22380  | 0.173715         |
| 7   | 12730 | 0.25433  | 0.189646         |
| 8   |  8054 | 0.16091  | 0.135018         |
| 9   |  1312 | 0.02622  | 0.025525         |

  - smallest two $\pi_k$: $0.21\%$ and $0.76\%$ (paper claimed $1.2\%$ and $1.7\%$ — corrected)
  - $\min_k \pi_k(1-\pi_k) = 0.002133$ (paper claimed $\approx 0.012$ — corrected)
  - $\overline{\pi(1-\pi)} = 0.0817$ (paper claimed $\approx 0.083$ — close, kept rounded)
  - ratio $\min/\overline{\;} = 0.026 < 0.05$ (trigger fires; threshold rule unchanged)

---

## §G · Action checklist for Terminal 0 vs Terminal 1

This document covers critiques **7, 8, 12, 13, 14, 15** as paste-ready
text. Terminal 1 is independently handling **1, 2, 6, 9, 10, 11**. No
overlap.

The variance-constrained run has terminated and
produced a **negative outcome**: 0/48 cells lifted to cleanly-compliant
across the 7 cells that completed before the 5-hour cap. The live
closing paragraph for §B is **paragraph B.3 (NEGATIVE)**. Paragraph B.2
(POSITIVE) is preserved here as a reference template only and should
not be pasted into the paper.
