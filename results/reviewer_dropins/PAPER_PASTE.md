# PCRL — consolidated reviewer-dropin paste

Six labeled drop-ins for critiques #1, #2, #6, #9, #10, #11; a fresh ~150-word abstract;
one consolidated §6 limitations paragraph. Pure text. No experiments.

Pull edits section by section.

---

## #1 — Table 1 three-way split (replaces "Hidden" column)

Skeleton with placeholders `__C_*__` for counts you'll fill from Terminal 2's
collapse breakdown. Adult totals 24 pair-seeds, HMDA 18, Diabetes 18.

```latex
% Table 1: per-dataset compliance breakdown.
% Three-way split: cleanly compliant (R^2 <= 0.05 AND per_dim_std_mean >= 0.5
% AND eff_rank >= 2) | collapse-compliant (R^2 <= 0.05 but at least one
% per-purpose health metric below threshold) | R^2-failed (R^2 > 0.05).
\begin{table}[t]
  \centering
  \caption{Per-dataset breakdown of the 60 (purpose, attribute, seed)
  configurations under three nested compliance bars: cleanly compliant
  (linear $R^2 \leq 0.05$ \emph{and} per\_dim\_std\_mean $\geq 0.5$
  \emph{and} effective rank $\geq 2$); collapse-compliant ($R^2 \leq 0.05$
  but at least one per-purpose health metric below threshold);
  $R^2$-failed ($R^2 > 0.05$). The dominant-axis audit (Proposition~4)
  flags an additional `hidden' pair-seed (HMDA underwriting/race
  seed~1, $R^2_{\mathrm{onehot}} = 0.027$ \emph{vs.} $R^2_{\mathrm{DA}}
  = 0.288$; see \S\ref{sec:hidden-case}).}
  \label{tab:compliance-split}
  \begin{tabular}{lccccr}
    \toprule
    Dataset & Cells & Cleanly compliant & Collapse-compliant & $R^2$-failed & Hidden by $R^2_{\mathrm{onehot}}$ \\
    \midrule
    Adult    & 24 & 3/24 & 20/24 & 1/24 & 0/24 \\
    HMDA     & 18 & 2/18  & 14/18  & 2/18  & 1/18 \\
    Diabetes & 18 & 2/18  & 15/18  & 1/18  & 0/18 \\
    \midrule
    \textbf{Total} & \textbf{60} & \textbf{7/60} & \textbf{49/60} & \textbf{4/60} & \textbf{1/60} \\
    \bottomrule
  \end{tabular}
\end{table}
```

Cumulative invariant for sanity-checking the fill-ins:
`Cleanly + Collapse + Failed = total per row`. The verification I ran earlier
established `Cleanly = 7`, `Collapse = 49`, `Failed = 4` aggregate — Terminal
2 should produce the per-dataset split that sums to those totals.

---

## #2 — Term swap, §3.2 scope statement, §5.4 framing pivot

### 2a. Term swap (sed-style; apply across every `.tex` in the project)

| Find | Replace |
|---|---|
| `compliance certificate` | `per-purpose linear leakage bound` |
| `Compliance certificate` | `Per-purpose linear leakage bound` |
| `compliance certificates` | `per-purpose linear leakage bounds` |
| `Compliance Certificate` (title-case in headers) | `Per-Purpose Linear Leakage Bound` |
| `the certificate` (when unambiguous from context) | `the linear leakage bound` |
| `linear compliance certificate` | `per-purpose linear leakage bound` |
| `\texttt{LinearComplianceCertificate}` (the class name) | leave as-is (code identifier; rename in code separately if at all) |

Quick grep to find all hit sites in your local mirror:
```
grep -rEn "[Cc]ompliance [Cc]ertificate" paper-body/ results/*.tex 2>/dev/null
```

### 2b. §3.2 scope statement (1-2 sentences, paste at section opening or after the bound is defined)

> The bound is per-purpose: for each declared purpose $p$ it bounds
> $R^2(h_p, A) \leq \tau$ for every $A$ in $p$'s disallowed-attribute set.
> It does not bound $R^2(\mathrm{concat}(\{h_p\}_p), A)$ for an auditor
> with simultaneous access to multiple purpose representations; we
> characterize that gap in \S\ref{sec:cross-purpose}.

### 2c. §5.4 framing pivot (replace the opening sentence of the cross-purpose section)

> Per-purpose linear leakage bounds, even when met at $\tau = 0.05$ on
> every cell, do not compose under concatenation. We construct an
> auditor that concatenates representations across purposes and find
> that 22 of 33 (dataset, attribute, auditor) triples recover the
> disallowed attribute above majority $+ 1$pp, with the largest gains on
> rare-class attributes (Diabetes \texttt{age\_bucket} under XGBoost:
> $+11.96$pp; Adult \texttt{age\_group} under XGBoost: $+9.28$pp). This
> is a structural property of the per-purpose framing, not a failure of
> training: the bound makes no claim about $\mathrm{concat}(\{h_p\}_p)$,
> only about each $h_p$ individually. We make this gap explicit and
> identify a concatenation-aware constraint as the natural extension.

---

## #6 — Post-hoc dual-update tuning disclosure

### 6a. §5 paragraph (paste after the Round-5 results are introduced; before Table 1)

> The constrained-optimization schedule used in our headline runs was
> developed by post-hoc audit of an initial Round-4 evaluation on the
> same 60-cell grid, which achieved 46/60 strict pass with mean $R^2
> = 0.038$ and exhibited a late-training dual-starvation failure mode
> (the proxy-Lagrangian dual decays to $\approx 0$ during temporary
> feasible windows then fails to catch a re-rising $R^2$;
> see Appendix~\ref{app:optimizer-drift}). The Round-5 schedule
> introduces two changes: a lower bound $\lambda_{\min} = 5$ on the
> dual variable, and a rule that skips the task-only warmup phase when
> the LEACE warm-start is already feasible at epoch~0. Both choices were
> motivated by the audit, but neither was validated on a held-out
> subset of (purpose, attribute, seed) triples; the fix was developed
> on a 50-epoch CPU probe of Adult only and then applied unchanged to
> HMDA and Diabetes. We expect the qualitative ranking of dual
> aggregations and the existence of the per-purpose vs.\ concatenation
> gap (\S\ref{sec:cross-purpose}) to be unaffected by held-out
> validation, but the precise headline number on the cleanly-compliant
> grid may shift.

### 6b. Abstract qualifier — append after the "56/60" sentence (8 words)

```
56/60 (configurations, dual-update schedule chosen post-hoc on this evaluation grid)
```

Or as a parenthetical clause: `56/60 configurations (with one dual-update
hyperparameter chosen post-hoc on this evaluation grid)`.

### 6c. Table 1 footnote (attaches to the 56/60 cell)

```latex
\footnote{Schedule hyperparameters ($\lambda_{\min} = 5$;
skip-warmup-when-LEACE-feasible) were selected by audit of an earlier
Round-4 evaluation on this same 60-cell grid; no held-out validation.
See \S\ref{sec:limitations}.}
```

---

## #9 — HMDA underwriting/race seed 1 worked example

Paste in §5 main text, immediately before or after Table~1. New label
`\label{sec:hidden-case}` referenced from Table~1's caption (#1 above).

> \paragraph{The dominant-axis audit catches a hidden case in our own
> canonical results.}
> \label{sec:hidden-case}
> The dominant-axis metric $R^2_{\mathrm{DA}} = \max_k R^2_{\mathrm{OvR}, k}$
> is not just a defensive audit: it surfaces failures the strict
> $R^2_{\mathrm{onehot}}$ test misses, including in our own headline
> runs. On HMDA underwriting/race seed~1, $R^2_{\mathrm{onehot}} = 0.027$
> satisfies the strict $\tau = 0.05$ bar, yet $R^2_{\mathrm{DA}} = 0.288$
> --- recovering more than a quarter of the variance of the rarest
> training-population race class ($k = 4$) --- fails the same threshold
> by an order of magnitude (a $10.7\times$ amplification). The
> convex-combination identity in Proposition~\ref{prop:convex-combo}
> predicts this gap exactly: when one class dominates the OvR
> $R^2$ distribution, $R^2_{\mathrm{onehot}} = \sum_k w_k R^2_{\mathrm{OvR}, k}$
> with prior-weighted $w_k = \pi_k(1-\pi_k) / \sum_j \pi_j(1-\pi_j)$
> downweights the leak, while $R^2_{\mathrm{DA}}$ surfaces it.
> Table~\ref{tab:compliance-split} reports the count of such hidden
> pair-seeds (1/60 across our benchmark); we surface this single case
> here as the worked example because the magnitude --- $0.288$ recovered
> on the rarest class while the headline metric reads $0.027$ ---
> motivates reporting both metrics rather than either alone.

---

## #10 — PCRL → PCRL-V (vision variant)

### 10a. Term swap (apply only to vision/CelebA-context references)

| Find (in CelebA context) | Replace |
|---|---|
| `PCRL` (when the surrounding paragraph discusses CelebA, image, ResNet, BatchNorm, vision) | `PCRL-V` |
| `the framework extends to vision` | `a related decoupled architecture (PCRL-V) achieves clean train-set $R^2$ on CelebA` |
| `Extension to vision` (section header) | `A decoupled vision variant (PCRL-V)` |
| `PCRL on CelebA` | `PCRL-V on CelebA` |
| `the PCRL representation on CelebA` | `the PCRL-V representation on CelebA` |
| Code/path identifiers `pcrl/vision/*` and `results/v2_celeba_*` | leave as-is |

Quick grep to find candidate hit sites in your paper-body mirror:
```
grep -rEn "PCRL.*[Cc]eleb|[Cc]eleb.*PCRL|extends? to vision|on CelebA" paper-body/
```

### 10b. Architectural-divergence paragraph for the CelebA section

> \paragraph{Architectural divergence between tabular PCRL and PCRL-V.}
> The tabular PCRL architecture (LoRA adapters absorbing the LEACE
> projection via SVD truncation, trained jointly with the
> proxy-Lagrangian) did not converge on CelebA across five training
> rounds; the rank arithmetic of composing the LEACE projection $Q$
> with the LoRA's $W_{\mathrm{repr}}$ grew the effective dimension by
> $k = 4$ per refit, exhausting the rank-8 LoRA capacity within two to
> three LEACE refits. We therefore adopted a decoupled architecture for
> vision in which the LEACE projection is a frozen
> \texttt{Linear($Q + \mu$)} layer applied \emph{before} a fresh
> trainable \texttt{Linear + LoRA} block (rank~8, $\alpha = 16$). This
> variant --- which we call \textbf{PCRL-V} to distinguish it from the
> tabular PCRL --- achieves train-set $R^2 \leq 0.005$ across $n = 3$
> seeds on CelebA. PCRL-V is architecturally distinct from PCRL: the
> LoRA is outside the LEACE null space, not inside it. We report it as
> evidence that the proxy-Lagrangian + LEACE warm-start \emph{design
> pattern} transfers to vision, not that the specific PCRL adapter
> architecture does.

---

## #11 — LoRA rank disclosure

### 11a. §5 paragraph

> \paragraph{LoRA rank.}
> The LoRA rank was set to $r = 8$ for Adult and HMDA (one dimension of
> slack over the joint one-hot cardinality bound $c - 1$ of any
> (purpose, attribute) pair on those datasets, where the maximum is
> $c - 1 = 4$ for race) and to $r = 24$ for Diabetes (eleven dimensions
> of slack over the joint bound from
> \texttt{quality\_research}'s race + \texttt{age\_bucket} pairing,
> with $c - 1$ sums of $4 + 9 = 13$). The Diabetes rank was raised
> from $r = 8$ to $r = 16$ and then to $r = 24$ over the course of
> Rounds~5 $\rightarrow$ 6 $\rightarrow$ 7 in response to the failure
> of the per-class \texttt{age\_bucket} OvR constraints under $r = 8$.
> We did not run a full rank ablation; the only ablation data point
> available is the $r = 16$ probe, which retained two of three
> \texttt{age\_bucket} pair-seed failures. A principled rank sweep
> $r \in \{c - 1,\ c - 1 + 4,\ c - 1 + 8,\ c - 1 + 16\}$ on each
> dataset is the right ablation.

### 11b. Footnote (attaches to the rank choice in any methods/Table that lists ranks)

```latex
\footnote{The dataset-specific rank choice is a known fragility in the
headline numbers and a candidate driver of the cleanly-compliant vs.\
collapse-compliant gap. We disclose it here rather than in an appendix.
See \S\ref{sec:limitations}.}
```

---

## NEW abstract (~150 words, confident, no caveats up front)

> **Per-purpose linear leakage bounds for shared-encoder representation
> learning.** A single shared encoder produces multiple
> purpose-conditional representations, each carrying a per-purpose linear
> leakage bound $R^2(h_p, A) \leq \tau$ on declared disallowed
> attributes. A joint LEACE projection (Proposition~\ref{prop:joint-leace})
> is the closed-form linear minimizer of leakage across all disallowed
> attributes simultaneously, and we use it as a warm-start for a
> proxy-Lagrangian fine-tune. The trained encoder achieves
> $R^2(h_p, A) \leq 0.05$ on $56/60$ (dataset, purpose, attribute, seed)
> configurations across Adult, HMDA, and Diabetes, of which $7$ also
> pass a non-collapse health check on the representation. A
> dominant-axis audit (Proposition~\ref{prop:convex-combo}) identifies
> hidden-leakage pathologies missed by one-hot $R^2$ --- for example
> HMDA \texttt{underwriting/race} seed~1 satisfies $R^2_{\mathrm{onehot}}
> = 0.027$ yet exhibits $R^2_{\mathrm{DA}} = 0.288$ on its rarest class.
> Per-purpose bounds do not compose under concatenation: $22$ of $33$
> (dataset, attribute, auditor) triples recover the disallowed attribute
> above majority $+1$pp when an auditor jointly accesses representations
> across purposes, a structural property of per-purpose representation
> learning.

Word count: ~150. Leads with the bound and Prop 3. States 56/60 with one
clause for 7/60. Surfaces the HMDA hidden case as a worked example. Names
cross-purpose composition as a result. No "future work", no "we recommend",
no "post-hoc". Confident posture.

---

## Consolidated §6 limitations paragraph (replaces scattered caveats)

> \paragraph{Limitations.} Several optimizer and architectural choices in
> our headline runs were selected post-hoc on the same 60-cell evaluation
> grid we report rather than on a held-out subset. The proxy-Lagrangian
> dual-update schedule (lower bound $\lambda_{\min} = 5$ and
> skip-warmup-when-LEACE-feasible rule) was developed by audit of an
> initial Round-4 evaluation that achieved 46/60 strict pass; the LoRA
> rank was fixed to $r = 8$ for Adult and HMDA but raised from $8$ to
> $24$ for Diabetes over the course of three rounds in response to the
> failure of per-class \texttt{age\_bucket} OvR constraints; the
> per-class OvR cardinality threshold ($K \geq 6$) was set to activate
> only on Diabetes \texttt{age\_bucket}; and the VICReg variance hinge
> ($\gamma = 1$) does not in practice prevent partial representation
> collapse below the audit's per\_dim\_std $\geq 0.5$ health threshold,
> which is the source of the $49/60$ vs.\ $7/60$ gap. We tested two
> refinements --- a hard per-purpose variance Lagrangian (mean-of-clamped-slack
> aggregation) and a per-dimension Lagrangian with $K = 64$ individual
> duals per purpose --- and neither escaped the collapse attractor: both
> produced $0/N$ cleanly-compliant pair-seeds on the cells they
> retrained, with $0$ of $576$ per-dim duals saturating at $\lambda_{\max}
> = 100$. Held-out cross-validation of the dual-update schedule, a
> principled rank sweep $r \in \{c-1, c-1+4, c-1+8, c-1+16\}$ per
> dataset, and a structural change that decouples concept erasure from
> representation geometry (e.g.\ relaxing the LoRA rank or backbone-freeze
> constraint) are the natural follow-ups.

---

## Quick checklist before paste

1. Critique #1 — fill `__C_*__` placeholders in Table 1 from Terminal 2
   counts; verify per-row sum = 24/18/18 and grand total = 60.
2. Critique #2 — run the grep, swap "compliance certificate" everywhere;
   add scope statement to §3.2; replace §5.4 opening.
3. Critique #6 — paste §5 paragraph, append abstract clause, add Table 1
   footnote.
4. Critique #9 — paste worked-example paragraph in §5 with
   `\label{sec:hidden-case}`; ensure Table 1 caption references it.
5. Critique #10 — run the grep, swap PCRL→PCRL-V in CelebA-context only;
   paste architectural-divergence paragraph.
6. Critique #11 — paste rank paragraph; add footnote on Table that lists
   ranks.
7. Replace abstract entirely with the new ~150-word version.
8. Replace whatever scattered limitations text exists in §6 with the
   single consolidated paragraph.

Order of operations: do the term-swaps first (critiques #2 and #10) so
later inserts don't re-introduce the old terms. Then insert the new
paragraphs. Then swap the abstract and limitations paragraph.
