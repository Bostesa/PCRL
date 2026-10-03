# Research decision: output-leak diagnosis and replication

**Status:** development evidence, 2026-10-03.

> **This is planned analysis of repeatedly used development data. Prior outputs motivated the study. Naming a primary
> endpoint or committing a lock does not turn the existing assessment pool into fresh confirmation. New comparisons
> are development evidence, and no population privacy guarantee is inferred from attacker performance.**

**Pins:**
- Source: f7425b15cb89bee841f5b1f9dd290d05a879ae1e.
- Lock v1: 01ef342f59a87432b22836b60a44b77423a3f6b6, pushed before any new fit.
- Amendment L1: c677ec2ae04c491d76690d901f3008503ac2d6ff, adding the stage-5 code before any stage-5 fit.
- Independent replay: INDEPENDENT_VERIFICATION.json; see VALIDATION.md.

## 1. Were the decisions useful, and by how much over a constant guess?

| Cell | Frozen-head gain over constant (s0 / s1 / s2) | Mean, simultaneous lower bound | Decision |
|---|---|---|---|
| Adult income | 3.7 / 3.3 / 4.8 points | 0.039, 0.031 | **PASS** |
| HMDA underwriting | 2.1 / **0.6** / 2.0 points | 0.0156, 0.0098 | **NOT_ESTABLISHED** (misses the 0.01 target by 0.0002) |

**Adult income.** The gain is real but modest. The frozen head recalls only 14–22 % of the >50K class, for a balanced
accuracy of 0.57–0.60. The common refitted probe does better: a gain of 4.9 / 10.1 / 9.4 points (PASS, lower bound
0.066).

**HMDA underwriting.** The frozen head recalls only 7–22 % of denials. On seed 1 it is essentially a constant
predictor. The refitted probe is no better (0.0164, NOT_ESTABLISHED).

**Other heads** (S3). Employment and education have huge gains (0.69 / 0.63), but these tasks are close to recodings
of input columns. HMDA pricing gains 0.37. The **HMDA fair_lending frozen head is exactly constant on every seed**: its
gain is 0.

**Consequence.** Part of the earlier "hard decisions leak little" result came from decisions that rarely depart from
the majority class. A decision that says little also reveals little.

## 2. Did removing the common score offset lower recovery while leaving the probabilities unchanged?

**Yes, on the two primary cells.** Removing the offset leaves the probabilities, the decisions and every task loss
exactly unchanged (`EXACTNESS.json`; `SCORE_DECOMPOSITION.md`).

| Contrast | Adult | HMDA |
|---|---|---|
| Recovery from the full-output bank | 0.779 | 0.769 |
| Recovery from the ignore-offset bank | 0.708 | 0.702 |
| FC (lower bound) | **+0.072 (0.062), PASS** | **+0.067 (0.060), PASS** |
| CH, decision instead of centred scores (lower bound) | +0.195 (0.177), PASS | +0.196 (0.185), PASS |

- **FC is offset-attributable:** an offset-using candidate was selected on every seed.
- **The offset alone** is as revealing as the margin: 0.704 / 0.737.
- **Seeds:** all three Adult seeds support FC. On HMDA, seeds 0 and 2 show it (drops of 0.06 and 0.14), while the
  collapsed seed 1 shows no change.
- **HMDA race pairs:** FC passes on all 3 estimable pairs. CH passes on (0,1) and (0,2), but is NOT_ESTABLISHED on
  (1,2) (Black vs Asian applicants): there the decision leaks about as much as the scores.
- **Not a general lever.** A refitted LR head, whose offset is a function of its margin, is twice as useful on Adult
  and leaks 0.774 through its offset-free margin alone. More useful decision information can mean more leakage.

## 3. Did the broader stored-model analysis reproduce the pattern, or reveal a counterexample?

**Both** (S3: 14 pairs × 3 seeds, Bonferroni over 34).

**Offset effect (FC):**
- **PASS on 5 pairs:** Adult income × {sex, race}; HMDA underwriting × {race, ethnicity}; HMDA pricing × race.
- **NOT_ESTABLISHED on 9.** Counterexamples are all 6 Adult employment and education pairs (multiclass heads, from
  −0.017 to +0.001) and HMDA pricing × sex. HMDA fair_lending is trivial, because its head is constant.

**Decision vs centred scores (CH):** PASS on 14/14, but on HMDA fair_lending only because its decision is a constant.

**Usefulness:** PASS for 4 of 6 purposes; NOT_ESTABLISHED for HMDA underwriting and fair_lending.

**So the offset effect is a property of particular trained heads, not of logits in general.**

## 4. Did combining two recipients' outputs add measurable recovery?

**Yes.** S4: Adult income + employment, race; all 6 endpoints PASS.

| Contract | Pair recovery | Best single recipient |
|---|---|---|
| Full logits | 0.917 | 0.845 |
| Centred logits | 0.883 | 0.792 |
| Hard decisions | 0.611 | 0.576 |

In every contract the pair exceeds **both** single recipients by more than 0.02. Low recovery from one recipient does
not imply low recovery from a coalition. This is a measured property of these fitted attackers, not a composition
theorem.

## 5. Did FARE contribute beyond compression while keeping a useful task?

**Not established** (stage 5; Adult employment/age_group, chosen by a validation-only screen; FARE_USEFUL_TASK_RESULTS.md).
- FARE kept the task: accuracy −0.0004, PASS; retention, PASS.
- It lowered complete-contract recovery far below target LEACE: 0.560 vs 0.713, PASS.
- But the zero-fairness tree with the same budget did almost as well: 0.567, a difference of +0.008 (NOT_ESTABLISHED).

**Replay of the existing cells.**
- The HMDA seed-1 nominee is a constant release (0 % of the gain kept).
- Adult s2 keeps only 79 %.
- The zero-fairness twin removes much of the recovery throughout.

**Certificate.** FARE's native certificate is UNAVAILABLE or vacuous (> 1) at the available certification sizes.

## Strongest comparisons

**Strongest favourable.** Withholding the common logit offset lowers measured sex/race recovery by about 0.07 AUC on
both primary cells, every seed where the head is not collapsed, with the probabilities and decisions exactly
unchanged. FC is offset-attributable, with lower bounds 0.062 and 0.060.

**Strongest adverse.** Three findings cut against the favourable story:
- The low recovery from hard decisions partly reflects heads that barely beat a constant (HMDA, gain not established).
- Combining two recipients' decisions raises race recovery (0.611 vs 0.576).
- On the useful-task cell, FARE's advantage over LEACE is matched by plain compression.

## Counts, runtime, storage

| Item | Value |
|---|---|
| Units | 559 unit directories: 24 reused by alias, 105 banks, 87 stage-5, the rest new attack, probe and head units |
| Model fits | 15,042 |
| Failed scientific units | 0 |
| Budget-unrun | 0 |
| Technically unavailable | FARE certificates as tabulated; 14 declared NOT_ESTIMABLE primary slots (HMDA race groups 3–4) |
| Wall time | about 1 h 05 m elapsed (custody to backup) |
| CPU | about 0.70 CPU-h for scientific runs, plus about 0.3 CPU-h for verification |
| Peak RSS | 0.98 GB |
| New private storage | 629 MB, 3,487 files |
| Drive copy | `private_odx_v1_20261003`, verified 3,487/3,487 uncached; restore of a FARE tree, head and attacker exact |

## Limits

- Repeatedly used development data.
- Intervals are conditional on the fitted attackers, with no retraining variance; between-seed spread is shown
  separately.
- Two primary cells; three encoder seeds.
- HMDA race groups 3 and 4 are not estimable.
- Encoder-validation exposure is unchecked.
- The stage-5 task is near an input recoding.
- FARE is applied to stored representations.
- No repeated-query attacks.

## One next decision

**Strengthen the output-contract empirical paper.**
1. Report usefulness and recovery together for every released output: full logits, centred, probabilities and decision.
2. Present the offset ablation as a measured property of the frozen PCRL heads, not a general privacy method.
3. Add the coalition result.
4. Repair or replace the near-constant HMDA heads before interpreting their decisions as "protective".

Any fresh confirmation of the offset effect needs its own prospective protocol and data-use review.
