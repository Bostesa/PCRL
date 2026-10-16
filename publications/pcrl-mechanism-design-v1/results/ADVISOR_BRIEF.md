# Advisor brief: what to implement next, and the adversary definition

## 1. Plain answers

- **Did we find a defensible new mechanism?** No.
  - **Final status:** `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` for a real-data pilot.
  - Two designers proposed independently, and three reviewers assessed both proposals. One was selected at the design stage: SEX-mixture marginalisation (MMP). It was the only proposal that removes a defined component without optimising against any attacker.
  - Its own pre-registered prototype, and an independent verifier probe, then contradicted the part of its rationale that a real-data advantage would need.
  - It is also not a new algorithm: the device is established in fairness and causal work.
- **What MMP does.** Task models are trained with SEX visible. At release, each model's "if male" and "if female" predictions are averaged with fixed population weights. The released prediction therefore drops the part that comes from guessing a person's SEX from their other fields. Its utility cost is known exactly in advance.
- **Why it might have beaten the neural control.** The neural control barely moves away from a model that estimates exactly that SEX-guessing component. If that component carried most of what attackers learn, MMP would remove it at a known price, with no adversarial training.
- **What the prototype showed.**
  - **Where it works.** MMP removes SEX information where it merges inputs: fields that predict SEX but not the tasks within each SEX (pure proxies), or SEX directions that the tasks do not use. There it works exactly as predicted.
  - **Where it does nothing.** In a law where every SEX-predictive field also matters for the tasks, MMP paid its utility price and attackers learned exactly as much as before.
  - **Where it can backfire.** In the verifier's 10-field probe, MMP lowered coalition recovery by 0.033 in one random draw but **raised** it by about 0.02 in two. When the SEX-guessing term happens to mask other SEX signal, deleting it unmasks that signal.
  - **Utility costs.** The price can exceed the caps, and the within-SEX burden can land on either group: up to 2.6× the subgroup guard.
- **Is the real-data pilot ready?**
  - The code is ready: written and rehearsed end to end on synthetic data.
  - Scientifically it is not, and I do not recommend running it as a competitive test.
  - If a data answer is wanted, run only Stage 0: model-fitting and model-selection roles, no attackers, no report labels, about 0.5 CPU-hour, and only after explicit authorisation. It would show whether the Florida data contain the pure-proxy structure that MMP's surviving part needs.

## 2. What we learned that we did not know

1. **Measured protection is learnability.** (The verifier's probes confirm this independently.) Every release evaluated so far is a deterministic continuous function of the six fields, and essentially one-to-one. A population-optimal attacker on the linked pair therefore recovers exactly what it could from the raw fields. This holds for the trees, the hybrids, MMP and the neural control.
   - Every measured difference, including the 0.020 targets, is about how much a fixed set of seven attackers learns from about 8,000 labelled examples.
   - Protection that does not depend on the attacker requires merging inputs. Earlier campaigns found that too costly at the occupation-confidence and income-Brier caps.
2. **Two plausible ideas, tested exactly.**
   - Marginalising over SEX removes information only along pure-proxy directions.
   - Cancelling SEX signal with a propensity-driven shift (RCCS) hides it from linear and quadratic attackers. It can fail for other attackers when the inputs are categorical, as they mostly are here, and it never lowers the population-optimal attacker's recovery.
3. **Concrete corrections to the original papers.**
   - **NeurIPS:**
     - The "linear certificate" is an in-sample ridge R² on the labelled test rows.
     - Its accuracy proposition was later shown false, and the retirement sits on an unmerged branch.
     - The aggregating attack keeps the best of three auditor seeds by test accuracy.
     - The motivating "auditor must see race" case is not implemented.
   - **AAAI:**
     - The 59/67 count uses the representation only.
     - "Knows the defense" holds only for noise channels.
   - **Combined manuscript:** its Stadler et al. sentence over-applies an all-attribute theorem to one declared attribute.

## 3. The adversary, and why this access

Our claims concern a release policy, so the adversary is drawn at the release boundary. It has:
- each recipient's own outputs, which test the per-purpose promise;
- the linked outputs of two recipients who serve the same person, which test whether that promise survives pooling, something a per-recipient policy cannot forbid;
- a labelled sample from the same population, which is the measuring instrument without which no attacker can be calibrated.

The labelled sample is granted by the benchmark. It is not shown to be obtainable in practice.

Inputs, model internals, repeated queries and insiders are excluded and listed as untested, because they would measure a breach of the provider rather than what the release adds. The adversary is not new: it is the auxiliary-data attribute-inference attacker plus the linked or colluding-recipient setting from prior work.

Manuscript-ready prose and a clean patch are in `ADVERSARY_AND_MOTIVATION.md`.

## 4. Recommended discussion

The missing ingredient is a decision about the target, not another training trick. Under the current contract no deterministic release can beat the raw-field leakage against an optimal attacker, so the 0.020 margin over the neural control is a contest in learnability. Three questions for the advisor:

1. **Is "what a fixed attacker learns from about 8,000 labelled releases" the right target?**
   - If yes, the paper should state the attacker budget explicitly.
   - It should report learning curves over auxiliary sample size, because a defense that only slows learning will look worse with more attacker data.
2. **Should the program allow a contract change** that makes attacker-independent protection possible: coarsened or randomised outputs, scored under the same caps? Earlier coarsening attempts cost too much occupation confidence, so this means accepting looser utility or a different utility definition.
3. **Should the paper drop the competitive-method aim** for now? It would report the empirical analysis with the corrected adversary, as the 2026-10-01 meeting suggested.

My recommendation is option 1 for the combined paper. No further method lane should open until the advisor chooses between options 2 and 3.

## 5. Scope

- Data-free campaign, about 2 hours elapsed by closeout.
- About 0.42 CPU-hours charged, including a 0.25-hour allowance for agent shell work outside the queue (`RESOURCE_INVENTORY.json`).
- $0 cloud spend.
- Everything is exploratory or design work. The Florida sealed assessment stayed closed; disclosure D2 still applies.
