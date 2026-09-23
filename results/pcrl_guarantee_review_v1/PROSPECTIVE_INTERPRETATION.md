# Prospective 2016 evaluation — independent interpretation

**Source.** `research/pcrl-final-prospective-v1` @ `5e154e5c4fdaeb23d327a0ebefe838525f1a19cb`. Only the
published aggregates were read: INFERENCE_2016.json, PRIMARY_CLAIMS.json and SECONDARY_CONTRASTS.json
(SHA-256 values in PROSPECTIVE_RECOMPUTATION.json), plus PROTOCOL.md §8.

**Recomputation.** `analysis/pcrl_guarantee_review_v1/verify_prospective.py` recomputes all 20 one-sided
bounds (maximum error 3.5e-18), all 20 clause decisions, both conjunctions, the secondary critical value
z = 3.38404 and all 70 intervals (maximum error 6.9e-17). Every stored decision is reproduced.

**Scope.** No individual-level 2016 data was accessed. These results are now observed evidence. Nothing
here redesigns a registered test or changes a registered decision.

## Headline (both registered decisions reproduce)

| | Q (T0_L_0.01_a17) | D17 (T0_U_unconstrained_a17) |
|---|---|---|
| Ten-clause conjunction | **fails** (8/10) | **fails** (8/10) |
| Sensitive clauses (dG ≤ +0.001) | 8/8 pass; the largest upper bound is −0.00724 | 8/8 pass; the largest upper bound is −0.00723 |
| Task clauses (dU ≤ −0.003) | 0/2 pass; estimates −0.00327 / −0.00400; upper bounds −0.00151 / −0.00187 | 0/2 pass; estimates −0.00474 / −0.00514; upper bounds −0.00291 / −0.00295 |
| z needed for the task clause to pass (registered: 1.96) | 0.30 / 0.92 | 1.87 / 1.92 |

Signs: dU = CE_task(M) − CE_task(J) and dG = CE_J − CE_M for the fitted attackers. Negative favours M.

## The five distinctions

1. **All ten jointly versus some passing.** The registered primary claim is the conjunction of ten
   clauses (an intersection-union test), and it is **not established** for either release. "Eight of ten
   pass" is a count of component decisions, not a partial success of the claim. The registered claim did
   not include any sub-conjunction (such as "all eight sensitive clauses") as a hypothesis. Reporting one
   is a descriptive reading of the registered components.

2. **Componentwise versus simultaneous.** Each upper bound is a *pointwise* one-sided 97.5% bound, as
   PRIMARY_CLAIMS.json itself states. The ten bounds of one candidate do not form a simultaneous 97.5%
   confidence set: by Bonferroni the joint coverage is only guaranteed ≥ 75% for ten bounds, and ≥ 80% for
   the eight sensitive ones. The IUT's size guarantee concerns *declaring the conjunction*, not joint
   coverage of the bounds.
   - **Post hoc, descriptive, not a decision:** every sensitive clause still passes, and every task upper
     bound is still below zero, when the one-sided critical value is Bonferroni-corrected over 8, 10 or
     all 20 primary endpoints (z = 3.023 at m = 20).
   - So the statement "all eight sensitive upper bounds and both task signs hold simultaneously at nominal
     family-wise one-sided level 0.025 (Bonferroni over the 20 primary endpoints)" is supported. That is the
     strongest correct simultaneous reading. It is nominal and asymptotic, and conditional on the fitted
     releases, attackers and their selection.

3. **Measured sensitive non-inferiority versus an information guarantee.** dG compares the held-out
   cross-entropy of *fitted, validation-selected attackers* under M and under J. Passing means M's
   measured recovery is not more than 0.001 nats above J's (here it is 0.009–0.038 nats *below* J's). It
   is **not** an information-theoretic bound.
   - Relative to the published service H, Q's own measured recovery is still positive. The unadjusted
     descriptive intervals exclude zero for 6/8 endpoints (0.0034–0.0060 nats: A/SEX, A/RAC1P, AB/SEX),
     and the two AB/RAC1P intervals contain zero.
   - D17 is similar (8/8 unadjusted intervals exclude zero, 0.0020–0.0058).
   - "Passes the sensitive clauses" therefore means "leaks measurably less than J", not "adds nothing
     beyond H".

4. **A negative task upper bound versus the material −0.003 requirement.** All four task upper bounds are
   below zero, so a task improvement over J is supported (as a sign, even under Bonferroni-20), but the
   registered **minimum** of 0.003 nats is not.
   - Q's point estimates (−0.00327, −0.00400) are themselves beyond the margin. The clause fails on
     precision: 1.96 × SE = 0.0018–0.0021 exceeds the 0.0003–0.0010 distance to the margin.
   - D17 misses by 0.00009 / 0.00005.
   - Neither fact rescues the clause. Both mean "effect of about the margin's size, not resolved at the
     registered level". It does not mean "effect absent".

5. **D17 better on unweighted task loss versus full domination.** In the separate secondary family (70
   endpoints, two-sided simultaneous z = 3.384):
   - CE_Q − CE_D17 is +0.00148 [+0.00033, +0.00262] unweighted, which resolves in favour of D17.
   - It is +0.00115 [−0.00033, +0.00262] person-weighted, which is unresolved.
   - None of the eight sensitive differences resolves. Point estimates favour Q in 6/8.

   D17 is therefore **better on one of the ten comparisons and not shown different on the other nine**:
   not a statistical domination, and not an equivalence. D33 shows the same pattern.

## Against the other controls (all reproduced)

| Comparator | Task (CE_Q − CE_comp) | Q's recovery resolved higher / lower |
|---|---|---|
| RR75 | −0.01165 / −0.01117 (Q better, both resolved) | 6 / 0 |
| W75 | −0.00706 / −0.00688 (Q better, both resolved) | 6 / 0 |
| C | +0.00440 / +0.00379 (C better, both resolved) | 0 / 7 |
| E | +0.01012 / +0.00776 | 0 / 8 |
| S | +0.01107 / +0.00890 | 0 / 8 |

Against randomised response and withholding, Q is better on the task and worse on recovery: a trade-off,
not a dominance. Against C, E and S, Q leaks less and performs worse on the task.

## Check of the manuscript's integration

**Update (re-checked at 55c0c5a358eea193e6d2293a2c814a5435487d14):** the manuscript owner added an
inferential-scope paragraph (lines 386–392) stating that the sensitive clauses are per-clause results, not
a jointly certified family. That is correct and makes P-2 optional. P-1 is **still present** at line 365.
P-3 is not yet present. Fragment line numbers now target 55c0c5a3.

Checked against `research/pcrl-submission-finish-v1` @ `8df7527c7c9a49668860223cd6d445256c345ca4`,
`papers/pcrl_satml_final_v1/main.tex`. Every number in §`sec:prospective` and in the abstract matches the
recomputation. Three wording repairs are proposed. Numbers are unchanged.

- **P-1 (a real error).** Lines 355–356, "eight sensitive-recovery clauses capping added recovery at
  $+0.001$ nats", reads as recovery beyond H. The clause caps recovery **in excess of J's**. Replace with:
  "eight sensitive-recovery clauses capping each release's measured recovery at no more than $0.001$
  nats above that of $J$".
  Add after line 377: "Measured recovery beyond the published service remains positive for both releases
  ($0.0034$–$0.0060$ nats for $Q$ on six of eight endpoints, unadjusted), so passing these clauses means
  leaking less than $J$, not adding nothing."
- **P-2 (precision).** Table caption line 381, "Upper bounds are one-sided $97.5\%$", should add:
  "pointwise, not simultaneous; all eight sensitive bounds and both task signs also hold under a
  Bonferroni correction over the $20$ primary endpoints (post hoc, descriptive)".
- **P-3 (balance).** After line 362, add: "$Q$'s task point estimates ($-0.00327$, $-0.00400$) are
  themselves beyond the margin; the clauses fail on precision, and $D_{17}$'s by less than $10^{-4}$.
  This is an effect of about the margin's size that the registered test did not resolve, not an absent
  effect." This sharpens, and does not contradict, the existing sentence about development effect
  shrinkage: the development estimate −0.0163 did shrink to about −0.0033.
- The D17 sentence at lines 401–402 is correct. Optionally add "(person-weighted task difference
  unresolved)" so it cannot be read as domination.

Fragments: `tex/prospective_fragments.tex`.
