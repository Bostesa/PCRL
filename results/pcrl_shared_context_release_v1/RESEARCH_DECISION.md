# Shared-context release: 2018 development decision

**Verdict: `EXPERIMENTAL_NO_ADVANTAGE`.** Neither registered competitive route passed.

- **Utility nominee.** It is an exact alias of the parent-restricted T32 control. It contains no decision beyond the old 32-state code, and its task difference from D17 is essentially zero.
- **Privacy nominee.** It lowered measured coalition SEX recovery in point estimate but did not establish the −.002 target. Its task cost of +.0078 nats is demonstrated in both weightings.
- **Deterministic control.** The same-context deterministic selector, built from the same richer policy bank, reached similar coalition-SEX recovery at about .008 nats lower task loss. Randomized mixing showed no benefit. The control chooses on task rather than privacy, so this is not a clean randomization ablation (see below).

This is development evidence on 2018 households that have been used repeatedly. It is not confirmation. The completed 2016 prospective result is unchanged.

**The richer-input hypothesis did receive a real test here** (see `ENCODER_CAPACITY.md`):

- **Trained.** Four richer policies were trained on all three anchors. They differ from D17 for 52–93% of coefficient people, with within-state variation in every one of the 32 T32 states.
- **Available.** The policies were in the LP's feasible set at every one of 7 fitting rounds, on every anchor.
- **Declined by the utility LP.** The utility-form LP never put mass on a richer column (η = 0 at every round).
- **Used by the privacy LP.** The privacy-first LP used the richer columns on 5 of 6 unit-anchors. After attackers were refit on each release, a richer law survived selection on three unit-anchors.

## Locked assessment

- **Lock.** `SELECTION_LOCK.json`, SHA-256 `d995484103cfbc9ca9f1411a2f89d6f5d9ee1f46e1bbdce1c8d0d82d20cadbfc`, committed at `e5d555a1438e14def094b1cdc1b1f5d2c2d52394`. It was remote-verified before the outer role was opened, once, at 19:56:26Z.
- **Nominees.** Both slots are `DIAGNOSTIC_ONLY`: no NM unit passed its inner_check point screen.
- **Comparator merging.** RD_PRIV and every ADV unit are exact D17 aliases on all three anchors, so their comparator group merged with D17. The primary family therefore has 40 endpoints: 2 slots × {D17-group, RD_TASK} × 5 roles × 2 weightings. The simultaneous two-sided 95% Bonferroni critical value is z = 3.227.
- **Bootstrap.** 10,000 common paired household draws over the union of households across the three overlapping anchors.
- **Scope of the intervals.** They condition on the fitted, selected objects. They do not cover the adaptive history or the ACS survey design.

Sign conventions:

- task = CE_Y(candidate) − CE_Y(comparator); negative is better.
- recovery = CE_S(comparator) − CE_S(candidate); negative means less recovery.

### U slot: NM1_U (≡ NM4_U ≡ T32_U), 8 of 20 primary clauses

| Contrast vs D17 | Estimate U / PWGTP | 95% simultaneous interval U | 95% simultaneous interval PWGTP | Clause |
|---|---|---|---|---|
| Task (target ≤ −.003) | −.00015 / −.00002 | [−.00332, +.00302] | [−.00362, +.00358] | fail |
| A/SEX (guard ≤ +.001) | +.00042 / +.00095 | [−.00236, +.00319] | [−.00201, +.00391] | fail |
| AB/SEX (guard) | +.00122 / +.00142 | [−.00139, +.00383] | [−.00162, +.00446] | fail |
| A/RAC1P, AB/RAC1P (guards) | 0 / 0 | exact zero | exact zero | pass |

- **The RAC1P passes are artefacts.** All eight passing clauses (four per comparator group) are `exact_zero_same_route`: on every anchor both releases chose the same H-only RAC1P route. They say nothing about protection.
- **Versus RD_TASK:** task −.00032 / −.00021, with the same guard pattern.
- **Capability versus H:** −.0378 / −.0398 nats. The separate capability family passes.

### P slot: NM4_P, 0 of 20 primary clauses

| Contrast vs D17 | Estimate U / PWGTP | 95% simultaneous interval U | 95% simultaneous interval PWGTP | Clause |
|---|---|---|---|---|
| Task (guard ≤ +.001) | **+.00781 / +.00757** | [+.00326, +.01236] | [+.00217, +.01296] | fail; demonstrated adverse |
| AB/SEX (target ≤ −.002) | −.00396 / −.00313 | [−.00946, +.00154] | [−.00933, +.00306] | fail (favorable point, unresolved) |
| A/SEX (guard) | −.00105 / +.00005 | [−.00557, +.00348] | [−.00486, +.00495] | fail |
| A/RAC1P (guard) | +.00046 / +.00052 | [−.00250, +.00343] | [−.00329, +.00433] | fail |
| AB/RAC1P (guard) | +.00078 / +.00077 | [−.00232, +.00388] | [−.00316, +.00471] | fail |

- **Versus RD_TASK:** task +.00764 / +.00737, also demonstrated adverse, with the same recovery pattern.
- **Capability versus H:** −.0298 / −.0322, which passes.
- **Selection optimism.** On inner_check the P nominee's AB/SEX contrast was −.0115 / −.0094. On the outer role it shrank to −.0040 / −.0031, as expected for a selected candidate.

### Secondary family (separately corrected, z = 3.529; descriptive)

| Comparison (NM4_P minus comparator) | Task U / W | AB/SEX recovery U / W | Reading |
|---|---|---|---|
| vs T32_P (parent-restricted, same LP form) | +.0029 / +.0031 | −.0012 / −.0010 | Beyond the old code: unresolved, worse task in point estimate |
| vs **DET_SEL4** (exact same-context deterministic selector, chosen on task in the NM4_U closing bank) | **+.0080 [+.0027, +.0133]** / +.0072 | −.0003 / +.0003 | **Strongest adverse comparison.** Same recovery, demonstrated worse task (U) |
| vs NM1_P (K=1 mixture) | +.0053 [+.0011, +.0094] / +.0044 | −.0038 / −.0031 | Context adds recovery reduction in point estimate at a demonstrated task cost (U) |
| vs Q_HIST (historical task-directed Q) | +.0088 / +.0079 (both demonstrated adverse) | −.0045 / −.0029 | Worse utility than the historical Q |
| vs J (continuous; different interface) | −.0091 / −.0088 | +.0004 / +.0034 | Better utility than J in point estimate, unresolved |

For the U nominee:

- vs Q_HIST: task +.0008 / +.0003.
- vs J: task −.0171 / −.0164. The U interval clears −.003; the PWGTP upper bound is −.0029.

The **RD_TASK preflight** measures whether X_A carries task information beyond (T0, H_A) that a 17-token deterministic release can deliver. Versus D17, RD_TASK's task difference is +.00017 / +.00019, with interval [−.0002, +.0005]: **no usable additional task information**. This is the direct reason the utility route could not succeed with richer inputs under this decoder family.

## Which ingredient did what

- **Richer inputs for utility: negative.** The task-only richer policy shows no gain over D17 on outer data. The utility LP never bought a richer column. The task-sufficiency condition flagged by the math review appears to hold empirically: within-state variation cannot improve the task.
- **Richer inputs for privacy: unresolved.** The P nominee's richer law exists only on anchor 0; on anchors 1–2 it is a T32 kernel. Against T32_P the AB/SEX point estimate favors it by about .001, well inside noise.
- **Randomization: no evidence of benefit.**
  - The deterministic same-context selector DET_SEL4 matched the stochastic P nominee's AB/SEX recovery at lower task loss.
  - Caveat: DET_SEL4 was registered as the randomization control for the NM4 family, but it chooses on *task* within the NM4_U closing bank. It is not the deterministic counterpart of the privacy-first NM4_P. The comparison therefore mixes "deterministic vs stochastic" with "task-first vs privacy-first selection".
  - It shows that a deterministic richer release reached similar measured AB/SEX recovery at lower task cost. It is not a clean randomization ablation.
- **Contextual allocation (K=4 vs K=1): mixed.** Recovery reduction in point estimate, at a demonstrated task cost.
- **Privacy training for deterministic and adversarial baselines: selection declined it.** RD_PRIV and all four ADV units selected D17 under the registered rules. ADV's task-selected checkpoints did not beat D17 on inner task among bank-feasible epochs. Its privacy-selected epochs never met the D17 + .001 task cap. The shared rule (M4) means no family was favored.

## Registered predictions: scorecard (subjective probabilities from PROTOCOL §9)

| # | Prediction | P | Outcome |
|---|---|---|---|
| 1 | A richer policy differs from D17 for > 10% of coefficient people on every anchor | .85 | **True** (52–93%) |
| 2 | NM4_U final within-T32 TV > .02 on ≥ 2 anchors | .55 | **False** (0/3) |
| 3 | Inner RD_TASK task vs D17 ≤ −.003 | .35 | Not scored: the registered selection code does not produce this inner contrast. The outer preflight (+.0002) points the same way as False. |
| 4 | U nominee passes full conjunction | .03 | **False** |
| 5 | P nominee passes full conjunction | .04 | **False** |
| 6 | Either nominee inner-screen eligible | .25 | **False** |
| 7 | Outer U task ≤ −.003 in both weightings | .15 | **False** (−.00015 / −.00002) |
| 8 | RD_TASK within .001 of, or better than, the U nominee on task | .55 | **True** (difference .0003 / .0002) |
| 9 | RD_PRIV matches or beats the P nominee on AB/SEX at task cost ≤ .001 | .45 | **False.** RD_PRIV is D17: equal task, AB/SEX worse by about .004 in point estimate. |
| 10 | ADV representative beats the P nominee on the same criterion | .25 | **False** (ADV = D17) |
| 11 | NM4 improves on NM1 of the same form by ≥ .001 on the route contrast (inner) | .30 | U form: **False** (identical). P form: **True** (inner AB/SEX −.0115 vs −.0022). |
| 12 | DET_SEL4 within .001 of NM4_U inner task, with comparable recovery | .60 | Not scored against the registered inner basis. On outer, NM4_U ≡ T32 kernel and DET_SEL4 is a different release. |

## What this does not show

- It does not show that richer information cannot help privacy. The retained richer laws are few, and the AB/SEX differences are unresolved.
- It does not establish a population privacy guarantee of any kind. All privacy numbers are fitted-attacker recovery statistics.
- It does not show that the deterministic selector is better than D17. That comparison was not registered; any number for it is post hoc (`POSTHOC_EXPLORATORY.*`).
- It does not show superiority to continuous erasers, which were not refitted as 17-token controls.

## Independent verification

See `INDEPENDENT_VERIFICATION.json` and `REVIEW.md`; the status is summarized in `RUN_STATUS.md`.
