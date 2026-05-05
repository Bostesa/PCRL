# PAPER_PASTE — d_TV pair-witness corollary, PCRL Round 5/7

## 1. Drop-in corollary for §3 (Multi-Purpose Impossibility)

See `corollary.tex` in this directory. Three-sentence proof, no factor
of ½, no Pinsker chain, attribute-pair witness convention.

## 2. Suggested integration sentence for §5.4 worked example

**Honest framing (recommended).** The pair-witness corollary's tight
lower bound $B_{\rm tight} = \sum_i \max_{a,a'}\min(p_a,p_{a'})(d_{\rm TV}(Y_i\mid A=a, Y_i\mid A=a')-d_{\rm TV}(\widehat Y_i\mid A=a,\widehat Y_i\mid A=a'))$ is
satisfied on every seed of all three datasets
(Adult: 0.010 bound vs.\ 0.221 observed; 
HMDA: 0.025 vs.\ 0.970; 
Diabetes: 0.010 vs.\ 0.802). 
Theorem~\ref{thm:multi-purpose-impossibility}'s d$_{\rm TV}$ form is
qualitatively confirmed---the bound is positive and respected---but
the gap (a factor of
22\,$\times$ on Adult, 41\,$\times$ on HMDA, 113\,$\times$ on Diabetes)
shows that the demographic-shift component captured by the corollary
is *not* the binding constraint on PCRL. The binding source of
$\sum_i {\rm Err}_i$ on these benchmarks is irreducible task noise
(majority-class baselines for the multi-class tasks: HMDA loan-amount
$\approx0.50$, Diabetes primary-diagnosis $\approx0.32$); the corollary
predicts the *demographic floor* on error, not its level.

## 3. Recommended placement

- **Theorem 2 (d$_{\rm TV}$ corollary):** add to §3 as Corollary~1, after
  the existing mutual-information form. Use the proof in `corollary.tex`.
- **§5.4 worked example:** insert the integration sentence above and
  cite `tightness_table.tex` (this directory) for numerics. Plot
  `bound_vs_observed_plot.pdf` is optional---it visualizes the gap on
  log-log axes, which can either underline the qualitative agreement or
  invite criticism, depending on framing.
- **Limitations §:** acknowledge that the d$_{\rm TV}$ floor is not
  the binding constraint at PCRL's operating point. The corollary is a
  *necessity* bound, not a *sufficiency* certificate of optimality.

## 4. Don't claim

Avoid "PCRL achieves the impossibility bound on Adult within 10\%".
The actual ratio is 22\,$\times$
on Adult; that's a clear gap, not a tight match. The honest claim is:
"PCRL's observed $\sum_i{\rm Err}_i$ respects Corollary 1 with
$B_{\rm tight}>0$ on every seed; the gap reflects task-noise dominance."

## 5. Numbers (mean over 3 seeds)

| Dataset | $\sum\gamma_i$ | $\sum\delta_i$ | $B_{\rm tight}$ | $\sum {\rm Err}_i$ | ratio |
|---|---|---|---|---|---|
| Adult (R5) | 0.0344 | 0.6104 | 0.0105 | 0.2212 | 22.1× |
| HMDA (R5) | 0.0580 | 0.3609 | 0.0254 | 0.9700 | 41.2× |
| Diabetes (R7) | 0.0210 | 0.1699 | 0.0096 | 0.8025 | 112.5× |

Per-seed, per-purpose witness data: `bound_numerical_results.json`.

## 6. Why the loose form is reported separately

The user's original draft of the corollary used
$\gamma_i := \max_{a,a'}\min(p_a,p_{a'})\,d_{\rm TV}(Y_i)$ and
$\delta_i := \max_{a,a'}d_{\rm TV}(\widehat Y_i)$ as *independent* maxima.
Subtracting them gives $\sum\gamma_i-\sum\delta_i$, which is
negative on every dataset (Adult: -0.576; HMDA: -0.303; Diabetes: -0.149). 
The proof actually delivers the same-pair ("tight") form via
$\max_{a,a'}\min(p_a,p_{a'})(d_{\rm TV}(Y_i)-d_{\rm TV}(\widehat Y_i))
\le {\rm Err}_i$, which is what `corollary.tex` states. Reporting
both makes the gap source obvious---it is the witness-pair coupling,
not the data---so reviewers don't have to take the tighter version on
faith.
