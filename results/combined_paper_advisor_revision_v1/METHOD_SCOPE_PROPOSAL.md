# Proposed scope for the future method (for discussion with the advisor)

This is a proposed research scope, not a result. **No competitive new mechanism has been established**, and agreeing
on this scope does not supply one. This document authorises no experiment, pilot, sweep, new population or
sealed-assessment access, and it changes no benchmark definition.

## Objective

Develop purpose-specific releases that retain the usefulness required by each permitted task while reducing
sensitive-attribute recovery from everything recipients receive, both individually and when releases are linked.

## What the objective commits to

1. **The empirical endpoint is recovery under declared access.** A method is judged by what a declared set of attacker
   families, fitted on a declared number of labelled releases from a declared population, recovers about the sensitive
   attribute from each recipient's release and from linked releases. The access model, the attacker families and the
   auxiliary-data budget are part of every claim, and recovery is always reported beside the held-out utility of the
   outputs that are actually released.
2. **Removing information and measuring with finite attacks are compatible goals.** A mechanism may be designed to
   remove information that the permitted tasks do not need, for example by giving the same output to inputs whose
   sensitive posteriors differ, or by not releasing a degree of freedom that no task uses, such as a free logit offset.
   It is still evaluated through finite attacks. The design intent explains why recovery should fall; the attacks
   measure whether it does. Learning curves over the auxiliary budget can indicate, but not establish, which applies:
   an advantage that shrinks as the labelled budget grows is consistent with slower learning, and one that persists is
   consistent with information removal or with a limit of the attacker families.
3. **A formal claim needs formal evidence.** A statement about arbitrary attackers (every classifier, every budget)
   needs a valid theorem with stated assumptions, or other evidence appropriate to that statement. Finite-attack results
   support statements about the evaluated attackers and budgets only.
4. **Mathematical examples show their own laws.** An exact or synthetic example demonstrates what holds under its
   specified law, for instance that a deterministic many-to-one probability release can remove sensitive information
   without utility loss. It does not show privacy, feasibility or utility for a real population.
5. **A competitive method need not be a joint design.** A method is competitive if it beats the eligible comparators
   under the same utility requirements and the same access. It does not also have to show that designing several
   recipients' releases jointly adds something.
6. **Three questions stay separate.** *Joint benefit*: whether jointly designed releases lower linked recovery relative
   to the method's strongest per-recipient twin and both sequential orders, under equal access and budget.
   *Competitiveness*: whether the method beats every eligible comparator. *Novelty*: judged against prior work only;
   numerical success is not evidence of novelty, and neither is a constructed example. Any one can hold without the
   others.

## Benchmark definitions that stay unchanged

Where this scope refers to the current benchmarks, their definitions are used exactly as they stand.

- **ACS two-recipient output service** (Florida ACS 2019; definitions recovered from code in the mechanism-design
  package's `PROBLEM_SPECIFICATION.md`, with the access taken from its `ADVERSARY_AND_MOTIVATION.md` and
  `CORRECTIONS.md` §6, public commit `c9709a2`):
  - release contract: recipient *i* receives R_i = (f_i(X), argmax f_i(X)), its own probability vector and decision,
    deterministic in the six permitted fields; nothing else is released; the coalition view is the perfectly linked
    pair;
  - access: readers are fitted on 8,001 labelled people (ATTACK_FIT) and selected and oriented on 3,000
    (ATTACK_VALIDATION) from random household splits of the same cohort; one fixed deterministic release per person;
    the pair bank includes readers that ignore either recipient; household and person identifiers are never reader
    features;
  - utility: 24 clauses with inclusive caps (accuracy loss ≤ 0.010, NLL excess ≤ 0.010 nats, summed Brier excess
    ≤ 0.005) for 2 tasks × references U0 and Ucal × 2 seeds, plus the within-SEX occupation Brier guard (≤ 0.005);
  - claims reported separately: point feasibility, utility confidence, recovery reduction against the method's own
    task-only twin (lower bound > 0.020 on both seeds, each recipient's worsening upper bound ≤ 0.010), competitive method (lower
    bound > 0 against every eligible comparator, at minimum the neural adversarial control, with the nested 0.020
    margin), joint component, subgroup suitability and novelty.
- **Stored-model benchmark of the revised paper**: release contracts (features only, for interpretation; output only;
  features with own head output; the clean-output bypass as an adverse control), the four output formats (and, in one experiment, the offset alone and the margin
  with the offset), each experiment's access and labelled budget (the paper's access table), and the per-experiment
  targets: level bars of 0.55 AUC, 0.05 and one point in Experiment 1; the 0.02 AUC recovery target for the contrasts
  of Experiments 2–4; the gain target of 0.01 for the frozen heads in Experiment 3; and the non-inferiority margin of
  −0.01, the gain target of 0.03 and the retention contrast for the useful heads of Experiment 4 (retention also on
  Experiment 3's easier task).

A randomised release, shared randomness or any change to the released objects is a contract change. It would have to be
declared and every comparator evaluated under the same changed contract. This scope does not make such a change.

## Scientific choices that remain open

1. **What usefulness the service actually needs.** Do recipients need calibrated probabilities, scores for ranking, or
   only decisions? The current caps protect probabilistic quality (NLL and Brier) as well as accuracy. If decisions
   suffice, the revised paper's post-hoc gap (on one Adult setting, the zero-fairness tree's
   scores recover about 0.10 AUC more than its own, equally accurate, decisions) suggests a simpler release; if scores are needed, that gap is the problem to solve.
2. **Which attacker families and auxiliary budgets are declared,** and whether results are reported as learning curves
   over the budget rather than at one size.
3. **Which recipients and which linkage model:** perfect or error-prone linkage, and whether coalitions larger than two
   matter.
4. **Which comparators define "competitive"** beyond the neural adversarial control, for example LEACE, FARE where it is
   eligible, and a task-only release.
5. **Whether a formal guarantee is sought,** for which attacker class, and from what theorem.
6. **Whether a randomised contract would be acceptable** as a declared change.

## Where the evidence stands

- No competitive new mechanism has been established. The most recent design study found no defensible new mechanism
  for a real-data pilot, and its leading candidate (averaging a fitted model's predictions over the sensitive attribute)
  is an established technique applied to a new endpoint.
- Information removal inside the current deterministic probability contract is possible in exact examples. Whether any
  such release meets the 24 utility caps on the ACS population is unknown.
- The revised stored-model paper shows where existing methods' promises end and what an evaluation of a new method must
  cover. It does not supply a method.
