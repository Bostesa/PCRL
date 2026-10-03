# Advisor brief: combined paper and useful-prediction comparison

2026-10-03. This is development evidence on rows that earlier analyses already used; it is not fresh confirmation.

## What is ready

- **Manuscript.** A complete draft in `papers/combined_empirical_v1/`: 10-page PDF, LaTeX source, verified bibliography, one recipient diagram, two figures and two tables. Every page has been inspected.
- **Supporting files:**
  - a claim ledger (26 claims, each traced to a file and a commit);
  - a both-repositories evidence map;
  - a corrections file;
  - a closest-prior-work comparison.
- **New experiment.** One bounded experiment with useful task heads. It was locked and pushed before any fit, run in about 10 minutes on the laptop, and independently replayed.

## The new comparison, in plain terms

**Setup.** On Adult income/sex, each method released the output of a real, deployable task head: a logistic regression fitted on that method's own features. All four heads are useful, at about 82–83 % accuracy against a constant guess of 75 %.

| What the recipient gets | Untreated | LEACE | FARE | Same-size tree without the fairness term |
|---|---|---|---|---|
| Accuracy of the released head | 0.832 | 0.831 (non-inferior) | 0.822 (**not shown** within 1 point) | 0.828 (non-inferior) |
| Sex recovery from the head's score | 0.774 | 0.759 | **0.539** | 0.648 |
| Sex recovery from the head's decision | 0.544 | 0.544 | 0.536 | 0.546 |

**What the table shows:**
- **LEACE keeps the head's accuracy but barely changes what its score reveals.** The difference is 0.015, a smaller difference below the 0.02 target. Post hoc, a *linear* reader of that score sees almost nothing (0.515); the leak is nonlinear.
- **FARE's score reveals much less.** Here, unlike the earlier easy-task cell, the fairness term matters beyond compression: 0.109 AUC on every seed, PASS. But FARE's head costs about one accuracy point, 2 points on seed 2.
- **Decisions alone leak about 0.54 for every method.** The untreated decision leaks about as little as FARE's score, at one point higher accuracy (post hoc).

## Corrections (no registered decision changes)

1. **FARE on the useful task.** "Most of the reduction was reproduced by compression; FARE had a smaller additional improvement (+0.007, interval 0.002–0.013), below the 0.02 target." I previously wrote "just as well" and "not significant". Both were wrong wording.
2. **HMDA fair_lending** gives the *same decision* to everyone; its scores vary and still leak (0.65).
3. **The offset effect** appears on 5 of 6 encoder seeds (not HMDA seed 1) and on 5 of 14 pairs.
4. **The 0.0814 gain** belongs to the refitted probe; the deployed refitted head's gain is 0.0826. Its margin-only recovery is 0.773.
5. **Coalition from decisions:** 0.611 vs 0.576, lower bound 0.023.
6. **The backup index** lists 3,489 files, not 3,487.
7. **The 22→8 and 26→19 counts** each change the model, the checkpoint rule and the criterion at once. They are kept on separate rows and never merged.

## Strongest comparisons

- **Favourable.** FARE's fairness term removes 0.109 AUC of score-level leakage beyond a capacity-matched tree, on every seed.
- **Adverse:**
  - FARE's accuracy is not shown to be within 1 point;
  - LEACE does not protect the score;
  - any defense is bypassed by an unchanged output;
  - two recipients recover more than one.

## One measured gap for a future method

At matched accuracy, the best score release still leaks 0.648, while the untreated decision leaks 0.544. A method that released useful, calibrated scores with leakage near the decision's level would close about 0.10 AUC on this cell. We do not claim that such a method exists or is impossible.

## Decision needed

**Approve the manuscript's framing:** an empirical, utility-qualified evaluation, with no new method claim. If approved, the next step is a targeted novelty search before any "not previously reported" sentence. Then go to submission formatting.
