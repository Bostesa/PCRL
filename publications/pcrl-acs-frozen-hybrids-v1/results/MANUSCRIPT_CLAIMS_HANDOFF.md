# Manuscript and claims handoff (additive; no manuscript branch changed)

## May be stated (exploratory development evidence, already-exposed people)

- **Utility.** A fixed composition of two existing tree releases keeps the point task utility of its components on all 24 inherited clauses. The composition (H_LJ) pairs LOCAL_TREE's income output with JOINT_TREE's occupation output. This follows by construction from published component values.
- **Coalition recovery vs continued task fitting.** H_LJ lowers linked-coalition SEX recovery compared with continued task fitting by 0.022 and 0.024 AUC: K = 38 intervals [0.008, 0.037] and [0.011, 0.036], established above zero. The pre-registered 0.020 margin is not established.
- **The neural control.** H_LJ is not superior to the eligible neural adversarial control.
  - Its coalition AUC is 0.001 and 0.005 higher than the control's at the point (benefit K = 38 intervals [−0.018, 0.015] and [−0.021, 0.011]).
  - The difference is not established in either direction, and equivalence was not tested.
  - The 0.020 superiority target is an established shortfall.
- **Where the reduction comes from** (descriptive, post-hoc). At the point, most of JOINT_TREE's larger coalition reduction relative to LOCAL_TREE goes with JOINT_TREE's income output: swapping it alone lowers coalition AUC by 0.032 and 0.024, out of a total of 0.053 and 0.042. That output fails income utility. This is a decomposition of fixed outputs, not evidence about the training objective.
- **Overall confidence (separate field).** `CONFIDENCE_UNRESOLVED`: 10 of 24 utility clauses PASS, 14 are UNRESOLVED, 0 violations.
- **Subgroup status (separate field).** `SUBGROUP_UNRESOLVED`; no harm is established.
- **Runnable artifact.** The composition is runnable from raw permitted fields. Each recipient is computed only from its own frozen component.

## Must not be stated

- That H_LJ beats the neural control or meets the 0.020 coalition margin, or that any utility-eligible combination does. H_JL and H_JJ show larger descriptive reductions, but they fail income utility and were not nominees.
- That the hybrid is a new algorithm. That the results show joint training is superior. That the four-way interaction is a theorem about information synergy, causal proof about the training objective, or a population privacy guarantee.
- That H_JL or H_JJ are useful releases or new nominees. They fail income utility; their larger descriptive reductions are a post-hoc observation.
- That this is confirmation, an independent replication or a rescue of the source claim. The source labels stand: `RECIPE_EFFECT_NOT_ESTABLISHED`, `NOT_SUPERIOR_TO_ALL_ELIGIBLE` (complete), `SUBGROUP_UNRESOLVED`.
- That the sealed assessment was never accessed. Source disclosure D2 applies: the source extractor computed sealed features, SEX, weights and both labels transiently in memory and discarded them unwritten. This campaign did not open, regenerate or read the sealed role, and nothing here justifies opening it.
- That the source verification passed 11/11. It passed 10/11; D1 was a transport-annotation failure, not a numerical failure.

## Required caveats

- **Post-hoc composition.** H_LJ was fixed after its components' DEV_REPORT results were public, so error rates are nominal.
- **What was new.** Of the 38 primary endpoints:
  - 30 recompute published source values exactly (24 utility clauses, 4 subgroup guards and 2 income-worsening clauses); their outcomes were settled before readers ran;
  - 2 more, the occupation-recipient worsening clauses, have published points (−0.089 and −0.061) and new standard errors;
  - only 6 carry new information: the two coalition benefits vs TASK_REFIT, the two comparisons with NEURAL_RAW and the two increments over H_LL.

  In the secondary family, the 10 endpoints involving the H_LJ or H_JL pair are new and descriptive.
- **Conditioning.** Intervals are conditional on the frozen defense models and the fitted readers. The two seeds share people and are not independent samples.
- **Linkage.** Perfect linkage of the two recipients is assumed for the coalition.
- **Not a privacy guarantee.** Recovery AUCs come from a fixed slate of seven fitted readers chosen on ATTACK_VALIDATION; other attackers could recover more. Nothing here is a population or formal privacy guarantee.
- **Variance.** The household bootstrap is not ACS survey-design variance. PWGTP-weighted values are point summaries only.
- **Scope sentence.** Attach it verbatim to any statement of the overall result: "Exploratory development result on already-opened DEV_REPORT data; the composition was fixed after its components' DEV_REPORT results were public, so error rates are nominal; conditional on the two fitted seed pairs (shared people, not independent samples), the frozen defense models and the fitted readers, in the unweighted DEV_REPORT household superpopulation; not confirmation; source labels unchanged."

## Suggested sentence

> "As an exploratory follow-up on already-exposed development data, we composed two existing tree releases (the locally trained income output with the jointly trained occupation output), choosing the composition after its components' results were public, so error rates are nominal. The composition kept its components' point utility (simultaneous utility confidence and subgroup suitability remained unresolved). It reduced linked-coalition SEX recovery relative to continued task fitting by 0.022 and 0.024 AUC on two model seeds, but did not establish the pre-specified 0.020 margin, and it was not superior to the neural adversarial control (point estimates favoured the control by 0.001 and 0.005 AUC). Larger reductions appeared only in combinations that used the jointly trained income output, which did not meet the utility requirements."
