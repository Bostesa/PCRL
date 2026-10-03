# Title and abstract

## Title options

1. **What a Recipient Can Recover from a Useful Prediction: An Evaluation of Feature Defenses and Output Contracts on
   Stored Representation Models** (current manuscript title). This is descriptive and puts utility first.
2. **Protected Features, Leaky Predictions: Measuring Attribute Recovery from the Complete Release of Purpose-Specific
   Models.** This is punchier and foregrounds the output bypass.
3. **Beyond the Representation: Utility-Qualified Attribute Recovery under LEACE, FARE and Output Contracts.** This
   names the methods for discoverability.

**Recommendation: option 1.** It states the unit of analysis (the recipient), the qualifier (a *useful* prediction)
and the object (stored models) without implying a new method.

## Abstract (as in `papers/combined_empirical_v1/main.tex`)

A model that serves several purposes often releases two things to each recipient: protected features and a task
prediction. We ask, on stored Adult and HMDA models, what a recipient can recover about a declared protected attribute
from the *complete* release, and how useful that release is for its task.

**What we evaluate.**
- Purpose-specific encoders, with each method's own protection check.
- Official target LEACE and official FARE.
- A capacity-matched zero-fairness FARE tree.
- Four output formats: full logits, logits without their common offset, probabilities and hard decisions.

All use one attacker slate, held-out roles and paired group-bootstrap intervals.

**Four findings recur.**
1. **Methods can pass their own checks while nonlinear recipients still recover the attribute** (AUC about 0.81–0.87
   under official LEACE). The purpose-specific encoders fail their own stored linear check on 12 of 42 attribute–seed
   combinations.
2. **Releasing an unchanged task output bypasses any feature defense:** recovery returns to 0.77–0.79.
3. **Low recovery from hard decisions often reflects decisions that say little.** We compared arms that each release a
   genuinely useful logistic-regression head fitted on their own features.
   - The head's score under LEACE still recovers sex at 0.759 (untreated: 0.774).
   - FARE's recovers 0.539, at an accuracy cost; non-inferiority within one point is not established.
   - A same-size tree without the fairness term recovers 0.648. On this cell, the fairness term therefore removes
     0.109 AUC beyond compression. On an easier task, compression reproduced most of FARE's effect.
4. **Two recipients pooling their outputs recover more than either alone,** including from decisions alone.

**Scope.** All results are development evidence on repeatedly used data. We propose no new algorithm, no population
privacy certificate and no composition theorem.

**Word count:** about 230.
