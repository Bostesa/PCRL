# Paper addendum: integration text for the combined-paper plan (outcome-specific)

**Scope.**
- This is for the combined empirical-plus-method paper. The empirical manuscript (`papers/combined_empirical_v1/`) is unchanged.
- No method advantage may appear in the abstract: none was established.
- Whether to split the papers is not decided here.

## Suggested section: "A matched method prototype, and why it did not pass"

**Method paragraph.** We implemented one prototype aimed at the complete-recipient-view gap identified above:
- two purpose-specific encoders trained jointly;
- official LEACE as the final feature transform;
- affine task heads releasing centred logits;
- nonlinear critics on each recipient's view and on the coalition;
- a privacy update projected off active task-loss guards, an adaptation of GEM/A-GEM (Lopez-Paz & Ranzato 2017; Chaudhry et al. 2019).

Every ingredient is established. The coalition term adapts our earlier linear cross-purpose constraint to nonlinear critics on complete views.

**Comparison set.** We compared it against matched controls on corrected inputs, from which sex, race, income, the occupation label source and the census weight were removed:
- task-only (U); post-hoc LEACE (E);
- the same training without coalition coupling (L); a plain penalty (JP);
- two neural sequential designs (not Taylor et al.'s algorithm);
- official FARE and its zero-fairness twin.

**Procedure.** Protocol, selection and outer scoring were locked in sequence.

**Result paragraph.**
- No arm that applies LEACE kept both permitted tasks within one accuracy point of the task-only reference. LEACE alone cost 4.9 points on income and 3.7 on occupation group, because linear guardedness of the task representation pushes the income head toward equal predicted rates for men and women (gap 0.014, against a true 0.20).
- The registered method claims are therefore not established.
- Descriptively:
  - the coalition-coupled candidate lowered coalition recovery relative to the uncoupled one by 0.017 AUC (simultaneous interval 0.009–0.024, below the 0.02 target) without local cost;
  - a plain penalty lowered it further (by 0.040 relative to the candidate) with better income accuracy;
  - official FARE gave the best income trade-off (task-only accuracy, recovery 0.72) but lost 3.8 points on occupation group.

**Interpretation paragraph.**
- The empirical gap stands. For tasks whose labels depend on the protected attribute, a mandatory linear eraser cannot coexist with a one-point utility allowance on these data.
- Coalition coupling adds only a small measurable reduction when coalition-specific signal is small: coalition minus the best single recipient was ≤ 0.025 for every arm.
- LEACE fulfils its own linear guarantee here (fit-row cross-covariance about 1e-15 on all 48 erasure units) while nonlinear recovery remains 0.71–0.86. That is outside its theorem, not a refutation of it.

**Limitations sentence.** One development cell (Adult, SEX primary), three seeds, reused rows, intervals conditional on fitted models; race was audited but not protected.

## Table row for the comparison table (keep LEACE)

| Method | Native criterion fulfilled? | Broader goal (useful tasks + low complete-view recovery) met? |
|---|---|---|
| LEACE (E) | Yes (linear, fit rows) | No: −4.9 / −3.7 accuracy points; nonlinear recovery 0.85–0.86 |
| FARE (F) | Certificate vacuous or unavailable | Income yes (0.842, recovery 0.69); occupation no |
| JCV prototype (J) | No native guarantee (critic surrogate only) | No: utility gates failed; coalition 0.814 |
