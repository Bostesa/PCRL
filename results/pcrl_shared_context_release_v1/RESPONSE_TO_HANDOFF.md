# Response to the shared-context handoff (pre-fit)

Written 2026-09-24, 17:26–17:48Z, before any new release candidate was fitted and before any new outcome was observed. Evidence: `agents/support/SUPPORT_CENSUS.{json,md}` (label-blind census), `agents/pipeline_audit/{CODE_CHECKS,AUDIT_SELECTION_MAP,ENDPOINT_UNCERTAINTY}.md` (code audit + power), `agents/pipeline_fit/PIPELINE_FIT_MAP.md` (implementation map), `agents/math/MATH_REVIEW.md` (independent mathematical review; see the addendum at the end of this file).

## 1. Is the support diagnosis reproduced from code and committed artifacts?

**Partly, and the reproduced part is stronger than the prose.**

- Reproduced from stored data (label-blind census, stored Linux T0 codes, hash-verified inputs): coefficient-role households per T32 parent are 131–198 (anchor 0, median 161), 127–195 (anchor 1), 142–192 (anchor 2); household-weight ESS per parent 61–137. With household-disjoint children, **0/32 parents can be split at a 100-household floor on any anchor**; at 50 or 25 all 32 can. The anchor-0 maximum of 198 matches the predecessor's `RUN_STATE.json`. Pooling nuisance+coefficient rows would allow a 100-household split in all 32 parents, but ESS >= 200 in only 7–8 parents per anchor.
- Confirmed from code: the floor was >= 100 unique and >= 100 Kish-effective households *per child* (`refinement.py:336-340`, `fit_b.py:776-777`).
- **Not reproducible from committed artifacts:** the predecessor's children were person-disjoint, not household-disjoint, so a household split across a threshold counted toward both children; "198 makes a split impossible" is therefore a heuristic for that code, not a proof. Anchors 1 and 2 have no per-parent receipt. And the terminal label `support_limited` is assigned whenever no positive-gain candidate survives and `min_households > 1` (`fit_b.py:566-567`), so it cannot distinguish support rejection from zero gain (demonstrated on a synthetic case with 400 households per child).
- Two further verified defects, with scope: `fixed_price_gain` is nonnegative by construction (`refinement.py:262-266`), and `rank_splits` re-optimizes tokens on checking rows (`:358-362`), so the checking score is a re-optimized opportunity, not held-out performance; on pure noise 73% of splits pass the 0.25 checking/fitting ratio, while a frozen-action version is negative 54% of the time. None of this changes the old verdict: zero splits were accepted, B was an exact A alias, and no endpoint involved B.

Conclusion: the richer-state hypothesis was not tested, and with this role allocation a per-child 17-token kernel design is support-starved (about 65–100 households per child), independent of the floor chosen. This supports a pooled design, not a lower floor.

## 2. Which part of the architecture changes available release decisions?

Only two ingredients can make two people with the same T32 state receive different token laws:

1. **Policy columns** d_m(x) that depend on legal inputs beyond T0 (X_A, H_A, stored teacher posterior p, residual r, historical risk). A policy that reproduces D17 on the admitted rows adds nothing; we measure disagreement with D17 within each T32 parent before any LP.
2. **Context-dependent mixing** (K=4): A[k,m] differs across contexts that cut through T32 parents.

B, eta and the D17 column do not: with only the D17 column, any (B, A, eta) gives a law that is a function of T0 alone, i.e. an old T32 kernel under a new name. "eta > 0" is therefore *not* evidence of a new release; the nonalias metric is the actual within-T32 total-variation spread of the fitted per-person law (weighted and unweighted mean, maximum, affected unique households), reported at every checkpoint separately from what selection picks.

## 3. Which proposal do I choose, and which do I reject?

**Chosen: the nested contextual policy mixture with HARD contexts, K in {1,4}, a five-policy bank, fixed delta = .001, and the predecessor Branch-A alternation.** Modifications to the handoff's proposal:

- **Policies are one step of approximate column generation.** Each non-D17 policy is `argmin_z g_hat(z|x)`, where `g_hat` is a regularized regression, fitted only on nuisance-construction rows, of the per-person Lagrangian cost `u_i(z) - sum_j lambda_j a_ij(z)` at the dual prices of the round-0 T32 LP (task-only: lambda=0; local-priced; coalition-priced; all-priced with doubled prices). This is the cost-sensitive reduction of Agarwal et al. applied to the release LP. It aligns every policy to the same frozen decoder's codebook by construction, which removes the token-permutation problem, and it answers "which within-state decisions would lower the priced objective" directly instead of hoping a generic policy lands there.
- **Hard contexts only** (2x2 at nuisance-role medians of the stored residual and the frozen local SEX risk). Soft contexts make the deterministic comparison ambiguous and add nothing this budget can test.
- **Candidate screening on inner_check, not on validation minima.** The predecessor screened candidates on the minimum validation losses that also chose the audit routes (winner's-curse optimism). Here routes are chosen on inner_selection and candidates screened on inner_check.

Mathematical and statistical reasons:

- *Statistical pooling.* The new block adds K(M-1)+1 mixing parameters (5 at K=1; 17 at K=4), each estimated from coefficient sums over roughly 1,000–3,500 households, instead of 16 free probabilities per child cell estimated from about 65–100 households. The policy learners are separate models (17 regularized boosted regressors each, fit on nuisance rows), so the full method is not a 17-parameter model; its policies are frozen before the coefficient-role LP sees them, so the LP cannot overfit them.
- *Exact baseline retention.* eta=0 embeds every T32 kernel, including D17 and the parent-restricted T32 optimum, so the fixed-bank LP optimum can never be worse than the T32 LP on the same bank (the empirical outcome after decoder/attacker refits can still be worse).
- *LP structure.* For fixed decoder and bank, the LP optimum is a vertex. With c binding cut constraints beyond the equalities, only a bounded number of rows/contexts can be fractional, so the solution is close to deterministic. Randomization can therefore matter only through the few binding cuts. This predicts that a strong deterministic policy (RD_PRIV) is a serious competitor, which is why it is mandatory.

**Rejected (for this run):**

- *Supported nested refinement (section 9).* The census shows per-child support of about 65–100 households at any floor that permits splitting, with 16 free token probabilities per child. Shrinkage would make it a QP, and its checking gain needs the frozen-action repair. That is a new study, not a rescue.
- *Policy-enrichment rounds (section 10).* This is unbounded search at the budget; the single pricing step above is the registered, bounded version.
- *Soft contexts and a KL penalty to D17.* KL to a one-hot reference is infinite for new tokens.
- *Changing any competitive margin.* Margins are kept exactly.

## 4. Strongest simple control that might explain away an apparent advantage

**RD_TASK**: a richer deterministic task-only policy (the lambda=0 policy with three Lloyd-style decoder/policy refits). If the mixture's task gain comes from coding residence better from X_A, RD_TASK captures that without any privacy machinery or randomness. On the privacy side the strongest simple control is **RD_PRIV**, a single-multiplier Lagrangian deterministic policy bisected to satisfy the same bank cuts. The **T32 parent-restricted LP** (eta fixed at 0, same bank and rounds) isolates "decisions beyond the old code". **NM1 vs NM4** isolates contextual allocation. **ADV** (PPAN-style adversarial categorical encoder with exact 17-token expectation) is the standard learned alternative.

A second, subtler explanation to guard against: **audit-slate noise**. An endpoint where both releases select the same H-only route is exactly zero, and some predecessor "passes" were of this kind. Such passes are counted but labelled.

## 5. Established prior work, implementation properties, and what needs this experiment

- **Established prior work:**
  - richer versus restricted observation and finite-law perfect privacy (Rassouli–Gündüz);
  - randomized mixtures over a policy bank under empirical constraints, via cost-sensitive reductions and LP/column generation (Agarwal et al. 2018);
  - learned randomized adversarial privatizers (Tripathy–Wang–Ishwar);
  - collusion and sequential releases (Taylor–Vippathalla–Coon 2026);
  - the privacy funnel (Calmon–Fawaz and follow-ups).

  Examples A and B illustrate known facts, not new theorems.
- **Implementation properties** (proved or tested here, finite and fixed-bank only):
  - row-stochasticity;
  - affinity in (B, A, eta);
  - exact embedding of every T32 kernel at eta=0;
  - linear exact-expectation losses for frozen predictors;
  - D17 is a constructive witness for delta >= 0 under same-row min-bank rebasing;
  - LP optimality relative to the fixed bank;
  - one-token wire with keyed persistence.
- **Needs this experiment:**
  - whether richer columns are instantiated (they differ from D17 within T32);
  - whether the alternating fits keep them;
  - whether the selected release beats D17 and the matched deterministic and adversarial controls under the independent audit;
  - which ingredient (richer inputs, context, randomization, privacy pricing) accounts for any difference.

Novelty is a separate judgement. The expected honest description, if it wins, is "a known reduction applied to this release problem with a new empirical result".

## 6. What counts as completing a real test even if D17 is selected

1. **Instantiation.** On all three anchors the four richer policies are trained, and their disagreement with D17 within T32 parents on coefficient rows is measured and exceeds the numerical tolerance for at least one policy. If every policy aliases D17, that is recorded as a failed instantiation and the structural fallback is used (compact policy-only model / K=2).
2. **Availability.** The richer columns are in the feasible set of every NM LP at every round. Their use (eta, A, within-T32 TV of the fitted law) is saved per checkpoint.
3. **Controls.** All mandatory controls are fitted and audited under the same slate on all three anchors, and positive controls run on all three anchors (AB/SEX and AB/RAC1P).
4. **Assessment.** Selection follows the locked rule, and the locked development assessment is run once on the full endpoint family.

If the LP declines the richer columns (eta about 0), or if selection prefers D17, that is a real negative about this programme: the columns were available and lost under the bank and audit. It is not an uninstantiated test. The distinction "trained" / "used by the LP" / "selected" is reported separately.

## 7. Power: the honest expectation (registered before fitting)

From the predecessor's assessment (same outer pool, 70-endpoint Bonferroni, z = 3.38), interval half-widths are about 0.0045–0.0060 nats for task and 0.005–0.008 for AB/SEX. That implies:

- **Utility route:** needs a true task improvement of roughly −0.008 to −0.009 nats versus D17 for 80% power.
- **Privacy route:** needs an AB/SEX difference of similar size.
- **Guards:** the +0.001 upper-bound guards pass with only about 0.1–3% probability for a candidate truly equal to D17 but audited by independently selected attackers.

A candidate that coincides with D17 for most people has smaller paired variance, so near-D17 mixtures are the most plausible passers. Even so, I expect the complete conjunctions to fail with high probability regardless of the true merit of richer inputs. The margins are not relaxed (section 16 of the handoff). All point estimates and intervals will be reported so the programme's actual effect sizes are visible. Registered subjective predictions are in `PROTOCOL.md` section 9.

## Addendum: independent mathematical review (received ~17:45Z, pre-fit)

`agents/math/MATH_REVIEW.md` reproduced Examples A and B exactly from the committed fixtures, confirmed the nested parameterization is algebraically sound, and **changed the design** (Amendment M1 in `DESIGN_SPEC.md`):

1. **A shared eta with a global mixture cannot choose where to use a richer policy.** There is an exact counterexample in which the nested LP value is 1/5 while a switched policy achieves 0. Every non-D17 policy is now D17-anchored: it deviates only where the predicted priced improvement exceeds a fixed .002 nats.
2. **Time-sharing is what the LP returns.** With r tight constraints the optimum mixes at most r+1 deterministic releases. With 0<eta<1 any deterministic output is a T32 function, so the exact same-context deterministic comparator is an exhaustive per-context policy choice (625 assignments at K=4). It is added as DET_SEL1/DET_SEL4, the randomization-attribution control.
3. **Frozen attackers overstate mixture privacy.** A mixture of D17 with a relabelled copy overstated recovery loss by .0105 nats, about 10x delta. Only refit attackers (the alternation bank and the independent audit) count as privacy evidence.
4. **eta is not identified.** Each T32-only column adds an alias direction, so the nonalias gate is within-T32 law variation plus its within-T32-averaged re-score.
5. **Task sufficiency.** If Y is independent of X_A given (T0,H_A), within-cell variation cannot improve task loss. The U route therefore needs X_A to carry task information beyond T0,H_A, and RD_TASK versus D17 is reported as that preflight.
6. **Convexity correction.** Attacker refits and the min over the bank inside rho do not break convexity for fixed rows. Decoder refits, policy learning and context learning do.
7. **Cross-fitting.** Policy task costs on the decoder's own training rows are optimistic, so they are now cross-fitted.
8. **Persistence.** Two fresh draws break Example A's privacy, so tokens are persistent per record.

The prior-work table and claims ledger are in the review. Only the Rassouli–Gündüz and Agarwal et al. texts were read in full; the other sources were verified from abstracts only.
