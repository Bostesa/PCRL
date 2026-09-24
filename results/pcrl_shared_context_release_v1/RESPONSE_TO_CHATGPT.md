# Response to the ChatGPT handoff: what the shared-context study found

The study ran on 2026-09-24 from 17:26 to about 21:00 UTC, on branch `research/pcrl-shared-context-release-v1`. The lock is at commit `e5d555a`, and the final commit is recorded in `COST_AND_CLOSEOUT.md`.

In short, the richer-input hypothesis finally received a real test, and no competitive operating point was established.

- **Utility side: task information was the problem, not support.** A 17-token deterministic policy using the richer inputs cannot improve residence prediction over D17: +.0002 nats. The utility LP never bought a richer column.
- **Privacy side: same trade-off as the predecessor.** The privacy nominee repeats the earlier trade-off: coalition SEX recovery about .004 lower (unresolved), at a demonstrated task cost of +.008.
- **Deterministic control did better.** A deterministic control built from the same richer policies matched that recovery at lower task cost.

## 1. Which ideas in the handoff were right, wrong, or modified?

**Right**

- **Support diagnosis.** A child-cell design is support-starved. The label-blind census reproduced 127–198 coefficient households per T32 parent, and 0/32 parents can be split at 100 households per child.
- **Pooled design.** A pooled design with D17 embedded exactly was the correct response.
- **Controls.** Matched deterministic, task-only, parent-restricted and adversarial controls were essential. The strongest adverse result came from one of them (DET_SEL4).
- **Warnings.** The warnings about "η>0 is not a new release", frozen attackers, persistence, and "a win over J is insufficient" were all borne out.

**Wrong or incomplete**

- *The "198 makes a split impossible" argument.* It is a heuristic for the predecessor code, which splits people rather than households. In addition, the terminal label `support_limited` cannot distinguish support rejection from zero gain. The old verdict of zero splits still stands.
- *A shared η with a global mixture.* This cannot put a richer policy only where it helps; there is an exact counterexample in MATH_REVIEW. It was repaired with D17-anchored switched policies.
- *Convexity.* The handoff said attacker refits and the min over the bank inside ρ break convexity. They do not, for fixed rows. Decoder refits and policy learning do.
- *Implicit reliance on the fixed-bank LP.* Once attackers were refit on each checkpoint (closing refit, M3), many fitted rounds violated their own constraints. The D17-witness rule (M4) then sent most U-form units and all RD_PRIV and ADV units back to D17.

**Modified before fitting.** Amendments M1–M5 are in `PROTOCOL_AMENDMENTS.md`:

- M1: switched policies, cross-fitted costs, and the exhaustive same-context deterministic selector.
- M2: paired-difference oracle.
- M3: closing refit and route-specific ADV selection.
- M4: identical witness rule for every family.
- M5: P-form privacy key, stronger privacy ADV, outer alias guard, positive-control gate, and remote-verified outer gate.

**Rejected**

- Supported refinement (section 9), because per-child support is about 65–100 households.
- Policy-enrichment rounds (section 10).
- Soft contexts.
- A KL penalty to D17.

## 2. What exact algorithm was implemented, and which part changes decisions beyond T32?

The release law is

    q(z|x) = B[T0,z] + Σ_m A[k(x),m]·1{d_m(x)=z}

- **Contexts:** hard contexts, K ∈ {1,4}.
- **Policy bank:** five frozen deterministic policies:
  - d_0 = D17;
  - task-only;
  - local-priced;
  - coalition-priced;
  - all-priced ×2.
- **How the policies are built:** each is a D17-anchored switched cost-sensitive policy. It is fitted by boosted regression of the paired improvement over D17 in the round-0 Lagrangian cost, on nuisance rows.
- **Fitting:** a fixed-bank LP (U form: task-minimizing; P form: AB/SEX-maximizing) inside Branch-A alternation, 6 rounds plus a closing refit, with D17-witness fallback selection.
- **What changes decisions beyond T32:** only the non-D17 policy columns and context-dependent mixing. That is measured by within-T32 total-variation spread (V), not by η.

See `METHOD.md` and `DESIGN_SPEC.md`.

## 3. Was the richer class instantiated on all three anchors? How many real fits and non-alias checkpoints were there?

**Yes.**

- **Policies:** 12 trained (4 per anchor). Each differs from D17 for 52–93% of coefficient people, with within-state variation in all 32 states.
- **Units:** 58 queue units completed with 0 technical failures, of which 42 are released-function fits. Distinct released functions per anchor audited after aliasing: 7 on anchor 0 (see `RUN_MANIFEST.json`).
- **Checkpoints with V>0:**
  - P-form NM checkpoints on 5 of 6 unit-anchors;
  - 0 of 6 U-form unit-anchors, at every one of 7 rounds.
- **Final releases with V>0:**
  - NM4_P a0 (V=.52);
  - NM1_P a0 (.22);
  - NM1_P a2 (.03);
  - DET_SEL4 a0 (.30) and a2 (.23);
  - RD_TASK a2 (.005).

## 4. Did selection choose a richer model or the baseline, and why, under the locked rule?

- **Status:** no NM unit passed the inner_check screen, so both slots are `DIAGNOSTIC_ONLY`, chosen as the rank-minimum under the registered lexicographic keys.
- **U nominee NM1_U:** an exact alias of NM4_U and T32_U. It is a deterministic T32 kernel on anchor 0 and D17 on anchors 1–2. It is not a richer model.
- **P nominee NM4_P:** richer on anchor 0 only; T32 kernels on anchors 1–2.

## 5. Did either complete competitive conjunction pass against the strong matched controls?

**No.**

- **U:** 8/20 primary clauses, all eight being exact-zero same-H-only-route RAC1P guards. Task versus D17 was −.00015 / −.00002.
- **P:** 0/20. Task was +.0078 / +.0076, demonstrated adverse. AB/SEX was −.0040 / −.0031 with upper bounds +.0015 / +.0031.
- **Comparator merge:** the RD_PRIV and ADV comparators merged with D17 because all of them selected D17.

## 6. If there was a gain, where did it come from? Which ablations establish it?

There was no established gain.

- **Richer inputs for utility: negative.** The RD_TASK preflight was +.0002, and the utility LP never used the richer columns.
- **Richer inputs for privacy: unresolved.** NM4_P versus T32_P gave AB/SEX −.0012 / −.0010 with task +.0029 / +.0031.
- **Context: mixed.** NM4_P versus NM1_P gave AB/SEX −.0038 / −.0031 at task +.0053 (demonstrated for U).
- **Randomization: no benefit.** NM4_P versus DET_SEL4 gave AB/SEX −.0003 / +.0003, with task +.0080, demonstrated. Caveat: DET_SEL4 selects on task, not privacy, so this is not a clean ablation.

## 7. What is the strongest counterexample or adverse comparison?

**NM4_P versus DET_SEL4.**

- The exact same-context deterministic selector from the same richer bank matches the stochastic nominee's coalition-SEX recovery.
- Its task loss is .008 nats lower (U interval [+.0027, +.0133] against NM4_P).

Post hoc and not registered, DET_SEL4 versus D17 gives:

| Contrast | U | PWGTP | U interval |
|---|---|---|---|
| Task | −.0002 | +.0004 | — |
| AB/SEX recovery | −.0036 | −.0034 | [−.0084, +.0012] |
| AB/RAC1P recovery | −.0020 | −.0020 | — |

That is zero task cost with unresolved recovery reductions, and it would still fail the P route.

## 8. What is guaranteed exactly, and what remains empirical or unverified?

**Exact, as implementation properties with tests:**

- row-stochastic laws;
- exact D17 embedding and witness feasibility for δ≥0;
- linear exact-expectation losses for frozen predictors;
- η=0 reproduces the predecessor T32 LP bit for bit;
- one persistent keyed token per record;
- legal-input enforcement.

**Empirical:** every privacy number. These are fitted-attacker recovery statistics under a finite validated slate with detected positive controls (6/6). They are not conditional mutual information bounds, not population guarantees, and not protection against unseen attackers or fresh redraws.

## 9. What did the programme fail to test, and why?

- **A clean randomization ablation for the privacy form.** DET_SEL was defined on the U-form bank.
- **Strong privacy-trained deterministic and adversarial competitors.** They collapsed to D17 under the shared feasibility rule, so matched privacy-side competition is weaker than planned.
- **Continuous erasers (LEACE, SPLINCE)** as 17-token controls.
- **Supported refinement and policy-enrichment rounds**, which were out of scope.
- **Any new population.** Nothing was confirmed on a fresh population. 2018 has been reused, and 2016 and 2017 are spent.
- **Inner-only predictions 3 and 12**, which were not scored.

## 10. What single next experiment would change the decision?

**A registered, deterministic, privacy-first same-context selector**, confirmed on a new cohort.

- **Design:** exhaustive per-context choice among the frozen switched policies, keyed on AB/SEX slack under a task cap, with no stochastic mixing. It should be fitted on 2018 and assessed once on the pre-declared **Texas 2018** geographic-transport cohort in `CONFIRMATION_PLAN.md`.
- **Target:** the P route (AB/SEX ≤ −.002 with task ≤ +.001).
- **Sample size:** with a post-hoc planning effect of about −.0035 and the observed SE of about .0015 on 2,968 outer households, the final-evaluation sample needs roughly 6× as many households (about 18,000). Any 2018 outcome used to plan it counts as post-hoc.
- **What would change the decision:** a pass would establish the first competitive PCRL operating point. A failure would close the richer-input line.
