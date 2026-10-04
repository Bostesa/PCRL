# Research decision: ordinary-penalty joint training without mandatory erasure (PN)

**Status: EXPLORATORY DEVELOPMENT evidence** on previously exposed Adult rows, motivated by the completed predecessor study. A pre-fit lock (`f1d1b76`) and a selection lock (`b0b9256`) were pushed before the respective fits and outer scoring. Nothing here is fresh confirmation or a population, DP or MI guarantee. Model label: **EXPERIMENTAL_NO_ADVANTAGE**.

## Verdict

**Neither registered claim is established.** Both conjunctions miss for two reasons.
1. **Clause P02 fails.** The income recipient's local guard requires R_v1(PN) − R_v1(LN) < 0.01 on the upper bound. The point is +0.010 and the upper bound 0.019. The difference is driven by seed 2, where PN's income view leaked 0.032 more than LN's (seeds 0 and 1: +0.004 and −0.005).
2. **The status requirement fails on seed 0.** PN β = 0.1 passed the utility gates there, but its inner occupation-view recovery (0.827) exceeded the frozen LN's allowance (0.815 + 0.01). PN is therefore NO_FEASIBLE_NOMINEE on seed 0; its β = 0.1 configuration was scored descriptively.

The other eight clauses pass, including a coalition improvement over LN **above** the 0.02 target. That does not make the conjunction a success.

## Primary family (18 slots, z = 2.9913; C\* = LN on every seed, so claim B equals claim A)

| Clause | Point | Simultaneous interval | Decision |
|---|---|---|---|
| P01: R_pair(LN) − R_pair(PN) > 0.02 | **+0.029** | [0.021, 0.037] | PASS |
| P02: R_v1(PN) − R_v1(LN) < 0.01 | +0.010 | [0.001, **0.019**] | **NOT_ESTABLISHED** |
| P03: R_v2(PN) − R_v2(LN) < 0.01 | −0.019 | [−0.028, −0.011] | PASS |
| P04, P05: Acc(PN) − Acc(U) > −0.01 | −0.001 / −0.003 | lower −0.005 / −0.009 | PASS / PASS |
| P06, P07: retention > 0 | +0.018 / +0.037 | lower 0.013 / 0.030 | PASS / PASS |
| P08, P09: Acc(PN) − const > 0.03 | +0.092 / +0.196 | lower 0.075 / 0.173 | PASS / PASS |
| **Claim A, Claim B** | 8 of 9 clauses | — | **NOT_ESTABLISHED** (P02; seed-0 PN not a nominee) |

## Levels (assessment, seed means)

| Arm | Income accuracy | Occupation accuracy | Income >50K recall | v1 AUC | v2 AUC | Coalition AUC |
|---|---|---|---|---|---|---|
| U | 0.843 | 0.476 | 0.61 | 0.861 | 0.877 | 0.873 |
| **PN β = 0.1** (selected seeds 1, 2; descriptive seed 0) | **0.842** | 0.473 | 0.60 | 0.757 | 0.823 | **0.840** |
| LN β = 0.1 (selected) | 0.842 | 0.475 | 0.60 | 0.747 | 0.842 | 0.869 |
| JP β = 0.1 (erased; infeasible) | 0.773 | 0.440 | 0.16 | 0.735 | 0.797 | 0.817 |
| PN β = 1 / LN β = 1 (infeasible) | 0.834 / 0.841 | 0.444 / 0.447 | — | — | — | 0.751 / 0.786 |
| E (infeasible) | 0.794 | 0.439 | 0.38 | 0.846 | 0.859 | 0.861 |
| F (infeasible on occupation) | 0.842 | 0.438 | 0.53 | 0.693 | 0.644 | 0.717 |
| F0 (own gates pass) | 0.849 | 0.465 | 0.59 | 0.798 | 0.857 | 0.874 |

**Decomposition of PN's coalition gain over LN.** The gain is +0.029 = Δ(worse local view) +0.019 + Δ(coalition synergy) +0.010. About two-thirds comes from PN's lower occupation-view recovery and one-third from reduced coalition-specific signal (synergy: PN 0.017, LN 0.026).

## Answers

1. **Did removing erasure restore accuracy, and how much recovery did it cost?**
   - **Accuracy: yes.** β-matched, PN − JP is +4.3 points on income [3.4, 5.2] and +1.7 on occupation [1.0, 2.4]. At β = 0.1, PN is within 0.1 / 0.3 points of U, with the same >50K recall (0.60 vs 0.61).
   - **Nonlinear recovery: no measurable change.** Coalition +0.003 [−0.004, 0.010], v1 +0.006, v2 −0.004, all unresolved.
   - **Linear recovery: real.** The coalition OLS R² of SEX goes from 0 (erased; by construction on the fitting rows) to 0.23 (PN β = 0.1; 0.22 held-out). The exact linear guarantee is lost.
2. **Did PN beat separate protection at useful task performance?** **Not established as registered.**
   - At equal, useful accuracy, PN lowered coalition recovery relative to LN by 0.029 [0.021, 0.037] and occupation-view recovery by 0.019.
   - It did so at the cost of income-view recovery (+0.010, upper bound 0.019, above the 0.01 guard).
   - It also missed LN's local allowance on seed 0 at selection.
   - The contrast also changes penalty mass and bank sizes, so even a pass would not be a pure coupling effect.
3. **Did it beat the strongest feasible matched control?** C\* = LN on every seed (no other control met the gates and LN's local allowance), so the answer is the same as (2).
   - Descriptively, PN's coalition recovery is lower than U's (−0.033), F0's (−0.034) and E's (−0.021).
   - It is higher than erased JP's (+0.067) and FARE's (+0.123). Both of those fail the utility gates. Here JP is the frozen per-seed JP reference, the closest nonzero-β configuration: β = 0.1 on seed 0 and β = 1 on seeds 1 and 2. At matched β = 0.1, PN's coalition AUC is 0.023 higher than JP's (0.840 vs 0.817).
4. **Were the training critics genuinely weaker on the same metric?** **Yes** (`CRITIC_GAP_DIAGNOSIS.md`).
   - On identical frozen views and rows, fresh critics of the same architecture beat the online critics on validation cross-entropy in 45 of 45 cells, by +0.047 nats on average.
   - At β ≥ 1 the gap is +0.05 to +0.09 nats, with the online critics sitting near the prior.
   - The encoders made SEX hard for the online critics to read rather than removing it.
5. **What guarantees were lost, and what evidence is supported?**
   - **Lost:** LEACE's fitted-row linear guardedness of each released feature vector, of the primary view (by affine closure) and of the coalition (by stacking). Nothing held-out, nonlinear or decision-level was guaranteed before.
   - **Supported:** attacker-based development evidence on reused rows, against the declared fitted slate, with intervals conditional on fitted models.

## Strongest comparisons

- **Favourable.** At U-level accuracy on both tasks, PN β = 0.1 lowered coalition SEX recovery by 0.029 relative to LN and by 0.033 relative to U, and occupation-view recovery by 0.019 relative to LN. Removing erasure restored 4.3 income points with no resolved change in nonlinear recovery.
- **Adverse.**
  - PN shifted leakage toward the income recipient (+0.010 vs LN; +0.032 on seed 2).
  - It lost the linear guarantee (coalition R² 0.23).
  - It still leaks heavily in absolute terms (coalition AUC 0.84).
  - FARE and erased JP leak far less, but at a utility cost the gates reject.
  - Stronger penalties (β ≥ 1) fail the utility gates.

## Prediction scorecard (PROTOCOL §8)

| Prediction | Outcome |
|---|---|
| (1) Accuracy restored (ABOVE 0) | **Hit**: +4.3 income, +1.7 occupation |
| (2) Recovery raised by +0.02 to +0.06 | **Miss**: NOT_RESOLVED, +0.003 |
| (3) LN feasible at β = 0.1; β ≥ 1 infeasible | **Hit** (all seeds) |
| (4) Claim A NOT_ESTABLISHED | Hit, but **the coalition margin itself passed**, contrary to the stated reasoning |
| (5) Claim B NOT_ESTABLISHED, C\* = LN or F0 | Hit (C\* = LN) |
| (6) Critic gap positive | **Hit** (45/45) |

## Counts and status

**Fits and units.**
- 18 new encoder fits, with no rescue triggered and no nonfinite gradients.
- 3 parity seeds: environment, β = 0 bitwise and JP replication all pass.
- 39 outer units; 18 critic-gap units; controls sound (nulls 0.495–0.513, planted leaks 0.94–0.97).
- Re-scored U, E, JP, F and F0 reproduce the predecessor's outer predictions **bitwise**.

**Incomplete units.** None.

**Infeasible arms.**
- PN on seed 0;
- JP (all seeds);
- E;
- F (occupation).

## Measured bottleneck and next step (prospective; not run)

The bottleneck has two parts:
- **income-recipient local leakage** under joint training;
- **critic tracking at β ≥ 1**: the online critics fall to the prior while fresh critics still read 0.05–0.09 nats.

A next protocol on new data could register:
- critic restarts or refits on frozen snapshots;
- a non-amplifying critic input map;
- a per-recipient local term weight, chosen in advance.

None of this may be rescored on these outer rows.
