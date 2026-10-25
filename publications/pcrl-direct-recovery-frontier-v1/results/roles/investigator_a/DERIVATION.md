# Exact optimal-reader AUC of a specified finite law (DRF1, investigator A)

Code: `drf/finite_law.py`; check: `python3 -m drf.finite_law selftest` (JSON, exit 0 iff all checks pass). Synthetic laws only.

## Object

Each person emits one observable wire w (income vector, occupation vector, or the linked pair). A finite law is the pair of
nonnegative masses over the DISTINCT observable wires, m1[w] = P(W = w, S = 1), m0[w] = P(W = w, S = 0); only P1 = m1 / sum m1 and
P0 = m0 / sum m0 enter. Empirical law: per-person weights w1 = 1[S = 1], w0 = 1 - w1. Estimated law: w1 = pi, w0 = 1 - pi.
Identical emitted wires (exact float64 equality of the whole vector, -0.0 == +0.0; for a pair, of both components) are one
observable value, so `merge_wires` sums their masses before any recovery computation. The reader never sees an internal cell id.

## Bound and attainment

For a score f, AUC(f) = P(f(W1) > f(W0)) + 1/2 P(f(W1) = f(W0)) with W1 ~ P1 and W0 ~ P0 independent. Let
Delta(a, b) = P1(a) P0(b) - P1(b) P0(a). Then

    AUC(f) = 1/2 sum_a P1(a) P0(a) + sum_{a<b} c_f(a, b).

Here c_f(a, b) is P1(a) P0(b) when f(a) > f(b), P1(b) P0(a) when f(a) < f(b), and the mean of the two on a tie. Each term is at
most the larger product, which is mean + |Delta(a, b)| / 2. Since sum over all (a, b) of P1(a) P0(b) = 1,

    AUC(f) <= AUC* = 1/2 + 1/4 sum_{a != b} |P1(a) P0(b) - P1(b) P0(a)|.

Equality holds iff f orders every pair with Delta(a, b) != 0 the way the likelihood ratio m1/m0 does. Pairs with Delta = 0 (equal
ratios) contribute the same however f treats them, so ties among equal ratios can count 1/2 or be broken. The likelihood-ratio
ranking (equivalently the posterior P(S = 1 | w)) therefore attains the bound. Pure wires are handled exactly. A wire with m0 = 0
has ratio +inf: it sits above every mixed wire and ties with other pure S = 1 wires (Delta = 0). A wire with m1 = 0 has ratio 0.
A wire with zero total mass is dropped.

## Computation

- Closed form, O(M^2): `auc_star_closed_form` (float, blocked) and `auc_star_closed_form_exact` (Fraction).
- Sorted, O(M log M): sort by ratio and group equal ratios g. Then AUC* = sum_g P1_g (P0_below_g + P0_g / 2). This is `auc_star`
  (float) and `auc_star_exact` (Fraction, exact rational ratios). In float64, ratios that are equal in exact arithmetic but round
  differently get ordered arbitrarily. Their Delta is at rounding level, so the value error is too: the selftest measured about
  1e-16 against exact values.
- Brute force: `best_auc_by_enumeration` takes the maximum over all weak orders (ordered set partitions, Fubini number,
  47,293 for k = 7) of the k <= 7 wires, in exact integer arithmetic.

## Properties (each verified in the selftest)

- AUC* lies in [1/2, 1], and a one-wire law has AUC* = 1/2.
- AUC* does not change when wires are relabelled or permuted, when m1 and m0 are scaled by separate positive constants, or when
  the two classes are swapped.
- Data processing: a score of g(W) is also a score of W, so merging or coarsening wires never raises AUC*. Hence
  AUC*_pair >= max(AUC*_inc, AUC*_occ), with equality when the pair is a function of one view.
- With ties 1/2, AUC* agrees with `sklearn.metrics.roc_auc_score` on an expanded integer population scored by the posterior.
- AUC* is at least the AUC of any fixed score.

## Scope

These are exact values for the SPECIFIED finite law only. For fitted census models they are not a population privacy guarantee.
An empirical or estimated law is itself a fitted, in-sample object, which may be optimistic or pessimistic. A population reader
faces the population law.

Lower mutual information, covariance, critic loss or linear separability do NOT stand in for lower AUC*. Both counterexamples
below are exact and checked in the selftest.

**MI, covariance and critic loss.** Law A has m1 = (3, 2), m0 = (2, 3). Law B has m1 = (1, 9), m0 = (0, 10). Both have prior 1/2.
Law A has lower dependence on every one of these measures, yet the higher AUC*:

| Measure | Law A | Law B |
|---|---|---|
| Plug-in MI (nats) | 0.0201 | 0.0360 |
| Cov(S, P(S=1\|W)) = Var(P(S=1\|W)) | 1/100 | 1/76 |
| Optimal Brier critic loss | 6/25 | 9/38 (lower) |
| Optimal log-loss critic loss | 0.6730 | 0.6572 (lower) |
| AUC* | 3/5 | 11/20 |

**Linear separability.** An XOR law has vectors (0,0) and (1,1) pure S = 1, and (0,1) and (1,0) pure S = 0, with equal masses.
Every linear score a x + b y has AUC exactly 1/2: the four comparisons pair off to a total of 2. Yet AUC* = 1. By contrast,
law A placed on a line has best linear AUC 3/5 and AUC* 3/5.
