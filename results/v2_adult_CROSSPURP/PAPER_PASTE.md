# PAPER_PASTE — Cross-purpose constraint at training time (§5.4 extension)

## Drop-in §5.4 paragraph (flowing prose)

To test whether the cross-purpose leakage observed in §5.4 is addressable at training time rather than only at audit time, we extended the proxy-Lagrangian constraint set with three additional duals on $\text{linear-}R^2(h_{\rm concat}, A) \le 0.10$ for $A \in \{\text{race}, \text{sex}, \text{age\_group}\}$, where $h_{\rm concat} = [h_{p_1}; h_{p_2}; h_{p_3}]$ is the concatenation of the three per-purpose representations. We retrained Adult from scratch with the augmented constraint set (75 epochs, three seeds, all other hyperparameters identical to Round 5). On Adult, per-pair strict-pass count fell from 23/24 to 19/24 on the auditor R² metric; the Adult-only cross-purpose attack flag count under Criterion A (concat acc > majority + 1pp, mean over 3 seeds, 5 attributes × 3 auditor architectures = 15 cells) moved from 14/15 (baseline R5) to 12/15 (with cross-purpose constraint).

## Per-cell $h_{\rm concat}$ R² (test split, mean over 3 seeds)

| Attribute | $h_{concat}$ R² mean | $h_{concat}$ R² max | Pass $\le$ 0.10 |
|-----------|---------------------:|--------------------:|:----------------|
| race | 0.0528 | 0.0913 | 3/3 |
| sex | 0.1466 | 0.2302 | 1/3 |
| age_group | 0.0870 | 0.1311 | 1/3 |

## Honest framing

**Outcome (b/c): partial cross-purpose reduction at the cost of per-pair compliance.** On the apples-to-apples Adult-only attack comparison, flags fell only modestly (14/15 → 12/15) — XGB on race dropped below 1pp delta on 1 of 3 seeds, mean delta on race dropped from +6.21pp to +1.06pp under LR. Per-pair compliance dropped 4 cells (23/24 → 19/24). The h_concat R² constraint is partially achievable on race (mean 0.053, 3/3 pass at the loose 0.10 threshold) but fails on sex (mean 0.147, 1/3 pass) and age_group (mean 0.087, 1/3 pass). The training-time linear-R² constraint on $h_{\rm concat}$ is not strong enough to defeat the post-hoc nonlinear auditors (MLP/XGB) on the most leaky attributes. Adding the constraint *does* reduce mean per-attribute attack delta on race (the only attribute where the loose 0.10 threshold actually binds) but trades off cleanly with per-pair compliance: 4 per-pair cells lost is a real cost. Recommend either (i) reporting honestly as a partial result with the exact tradeoff above, or (ii) extending the constraint with kernel-HSIC on $h_{\rm concat}$ to address the nonlinear residual the linear-R² constraint can't reach.

## Files
- `results.json` — per-seed per-pair R², h_concat R², attack accuracies
- `comparison_table.tex` — paper-ready 3-column table
- `HEADLINE.txt` — 5-line summary
