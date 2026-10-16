# Advisor brief: what to implement next, and the adversary definition

> **Corrected on 2026-10-16 (see `CORRECTIONS.md`).**
> - The original brief (public `e141330`) said that every evaluated release is essentially one-to-one, so that no deterministic release could beat the raw-field leakage and the 0.020 margin was purely a contest in learnability. **That is withdrawn.**
> - §1 (the reason for the status and MMP's description), §2, §3, §4 and §5 are rewritten. The status is unchanged.

## 1. Plain answers

- **Did we find a defensible new mechanism?** No.
  - **Final status:** `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` for a real-data pilot.
  - Two designers proposed independently, and three reviewers assessed both proposals. One was selected at the design stage: SEX-mixture marginalisation (MMP).
  - Its own pre-registered prototype contradicted the learnability leg of its rationale in a synthetic law, and the verifier's probe showed its effect can go either way. A real-data benefit would need either merging structure in the Florida fields or a learnability effect; which applies, if either, is unknown. It is also not a new algorithm.
- **What MMP does.** Task models are fitted with SEX as an extra input. At release, each fitted model is evaluated twice per person, with its SEX input set to male and then to female, and the two predictions are averaged with fixed population weights.
  - The output drops the part of an ordinary (SEX-blind) prediction that comes from inferring SEX from the other fields.
  - These are evaluations of a fitted model, not individual counterfactuals.
  - Under a known law, its utility price relative to the SEX-blind prediction is an exact formula. For fitted models on real data the cost must be estimated against U0 and Ucal.
- **Why it might have beaten the neural control.** The neural control barely moves from U0, and U0 estimates the SEX-blind prediction, which contains that inferred-SEX part in full. If that part carried most of what attackers learn, MMP would remove it without adversarial training.
- **What the prototype showed.**
  - **Where it works:** MMP lowers the optimal attacker's recovery where its output merges people with different SEX posteriors. That happens along fields that predict SEX but not the tasks, along any SEX direction the task predictions do not use, and wherever its averaged outputs happen to coincide.
  - **Where it does nothing:** in a law where the release is one-to-one by construction, attackers learned essentially as much as before. The optimal values were equal, and fitted readers differed by 0.0004.
  - **Where it backfires:** in the verifier's 10-field probe, MMP lowered recovery by 0.033 in one draw and **raised** it by about 0.02 in two. Deleting the inferred-SEX part can unmask other SEX signal.
  - **Utility:** the price can exceed the caps, and the within-SEX burden reached 2.6× the subgroup guard.
- **Is the real-data pilot ready?** The code is: it is written and rehearsed on synthetic data. Scientifically it is not, and I do not recommend running it as a competitive test. Nothing in this correction authorises a run.

## 2. What we learned that we did not know (corrected)

1. **What the measurements are.**
   - The optimal attacker can never recover more from a release than from the inputs. It recovers the same amount only if the release keeps P(SEX | inputs), for example by being one-to-one on the population.
   - This is **not established for any of our ACS models**. Every model reads an encoding that pools rare category levels, so none is one-to-one in the raw fields on the declared domain. Whether pooling, or anything else, merges people with different SEX posteriors in the population is unknown.
   - Our recovery figures are what a fixed slate of seven readers learns from about 8,000 labelled releases; for the pair, the slate also includes readers that ignore one recipient. How much of each difference between defenses is information removed, and how much is learnability, is unknown. (The original brief said all of it was learnability; that does not follow.)
2. **Information can be removed inside the current contract.**
   - A deterministic many-to-one probability release can lower the optimal attacker's recovery: an exact example does it with no utility loss at all.
   - Whether any such release meets our 24 caps on the Florida data is an open question. Earlier coarsening attempts failed for the specific codes tried (mostly on Adult); that rules out those codes, not every merge.
3. **Two ideas, tested in exact and synthetic laws.**
   - **MMP:** removes information only where it merges inputs with different SEX posteriors, and can raise recovery elsewhere.
   - **RCCS:**
     - its objective directly controls only linear and quadratic critics;
     - in two settings of one smooth synthetic law every fitted reader, trees and neural nets included, fell by 0.018–0.045, while optimal recovery rose;
     - with categorical drivers an optimal reader recovers much of the cancelled signal.

     It stays rejected as presented, because its feasibility case failed the binding clauses. A repaired version is unresolved.
4. **Concrete corrections to the original papers** (unchanged).
   - **NeurIPS:**
     - the "linear certificate" is an in-sample ridge R² on the labelled test rows;
     - its accuracy proposition was later shown false, and the retirement sits on an unmerged branch;
     - the aggregating attack keeps the best of three auditor seeds by test accuracy;
     - the "auditor must see race" case is not implemented.
   - **AAAI:**
     - the 59/67 count uses the representation only;
     - "knows the defense" holds only for noise channels.
   - **Combined manuscript:** its Stadler et al. sentence over-applies an all-attribute theorem.

## 3. The adversary, and why this access (corrected)

Our claims concern a release policy, so we chose to test an adversary that holds what is released:
- each recipient's own outputs, testing the per-purpose promise;
- two recipients' outputs for the same person, perfectly linked, testing whether that promise holds when they pool. We do not assume governance prevents linkage; perfect linkage is the attacker-favourable limit;
- a same-population labelled sample (8,001 people to fit, 3,000 to select) for our supervised readers. The benchmark grants it; it is not shown to be obtainable in practice.

This isolates one narrow question: what the delivered outputs reveal under that data budget. The following are distinct threat models, untested rather than dismissed:
- recipients that hold a person's inputs;
- public parameters;
- permitted repeated or chosen queries;
- objective- or parameter-aware attacks;
- insiders.

The adversary is not new: it is the auxiliary-data attribute-inference attacker in the linked-recipient setting of prior work.

Manuscript-ready prose and a clean patch (v2) are in `ADVERSARY_AND_MOTIVATION.md`.

## 4. Recommended discussion (corrected)

The missing ingredient is still a decision about the target, not another training trick. The correction removes the false reason that seemed to force one answer. The options are:

1. **A declared attacker budget.** The target is what a stated reader bank learns from a stated number of labelled releases. The paper states the budget and reports learning curves over auxiliary sample size, since a defense that only slows learning looks worse with more attacker data. This is measurable now and reasonable, but it is a research choice, not a consequence of any impossibility.
2. **Information removal under the current contract.** The target is the optimal attacker's recovery. A method would have to merge inputs that differ in SEX posterior while keeping all 24 utility clauses. That is not ruled out, but nothing here shows it is feasible on Florida, and the specific merges tried so far failed on confidence.
3. **A declared contract change.** Randomised releases or shared randomness, scored under the same caps.
4. **Drop the competitive-method aim for now.** Publish the empirical analysis with the corrected adversary.

**The single useful next step toward a method is this decision.** It fixes which recovery a method must lower, and therefore what the next design problem is. No sweep, pilot or new lane should start before it.

## 5. Scope

- **Campaign:** data-free, about 2 hours elapsed, about 0.43 CPU-hours charged (0.4323 h, including a 0.25-hour agent allowance; the original brief rounded this to 0.42), $0 cloud spend.
- **Correction pass:** 5 small queue units, none failed. 0.10 CPU-hours charged, almost all of it a deliberately generous 0.1-hour shell allowance. About 1.0 hour elapsed at the inventory, within limits of 2 CPU-hours and 2 hours; $0 cloud (`RESOURCE_INVENTORY.json`, section `correction_pass_2026_10_16`).
- Everything is exploratory or design work. The Florida sealed assessment stayed closed, and disclosure D2 still applies.
