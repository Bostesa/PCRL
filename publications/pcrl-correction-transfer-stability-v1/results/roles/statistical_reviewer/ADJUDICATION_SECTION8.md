# Adjudication A5: PROTOCOL §8 and the question-A bootstrap correction on R0 / R1

Statistical reviewer, 2026-10-18 about 20:50Z.
- Procedural decision only: no data read, nothing computed.
- It addresses the verifier's procedural FAIL in F3_FROZEN_RULES_AND_RECIPE__V2.
- Disclosure: A2 is my own adjudication, and the defect it corrected is in my code. I state this so it can be weighed.

## Decision: (b), with the conditions below

The registered-text labels (the corrected implementation, rerun 20:34:41-20:35:44Z at eece7dbdb) are the **primary** question-A
labels on R0 and R1. The frozen-code labels are kept beside them as **"frozen implementation output, superseded: departed from
RULES.md §6"**. Nothing is deleted.

## Reasons

1. **What §8 restricts.** "After a repair CHECK outcome is seen, revisions **using it** are adaptive and descriptive and cannot replace
   or relabel the primary repair result." The restriction applies to a revision that uses a repair outcome. This correction does not.
   - *Trigger.* The verifier's code-against-text audit, from D-stage replays (20:29Z).
   - *Content.* Fully determined by text registered before any fit: RULES.md §6 and RULES.json define exactly one scheme, with no free
     parameter or choice among alternatives.
   - *Scope.* Applied uniformly to all 34 units on D0, D1, R0 and R1, not to selected units.
   - *Adjudicator.* I specified A2 without having seen any R output.
   - *Direction.* It removes two method-favourable labels. This is supporting evidence, not the criterion (see reason 3).
2. **Which artefact governs.** REPAIR_FREEZE froze the rules and their implementation so that neither could change after outcomes. It
   did not turn an implementation defect into the rule. When the frozen code departs from the registered text, the registered text is
   the specification.
   - Making the frozen output primary would report, as the registered result, a procedure known to understate Var(D) (anti-conservative).
   - This mirrors how programming errors found after unblinding are handled under a registered analysis plan: the corrected analysis
     per the plan is primary, and the deviation is documented.
3. **Direction-neutral policy (recorded so it can be applied again).** A post-freeze change may replace a frozen output only if all of
   the following hold. The rule applies whichever direction the change moves a result.
   - its trigger, content and scope were fixed without any repair outcome;
   - it brings the implementation into line with text registered before any fit;
   - it applies to every unit;
   - it is recorded as a deviation with the full timeline.

   Every other post-outcome revision stays adaptive and descriptive, as §8 says.

## Conditions (all required; otherwise (a) applies)

1. **The verifier confirms the trigger used no R output.** Its 20:29Z evidence must have used only D-stage inputs (replay inputs listed
   by file and hash). The finding came after C0 had been read and while C1 was being read, so this must be shown, not assumed. **If it
   cannot be confirmed, decision (a) applies:** the frozen labels are primary on R, and the registered-text labels are a post-freeze
   correction (adaptive, descriptive).
2. **Record a deviation with the exact timeline:**
   - REPAIR_FREEZE committed 20:23:59Z;
   - C0 read 20:25:19-20:27:53Z; C1 read 20:28:02-20:29:05Z;
   - verifier finding 20:29Z;
   - corrected code committed 20:34:05Z (eece7dbdb);
   - rerun 20:34:41-20:35:44Z;
   - A2 saved about 20:35Z;
   - note that the commit preceded the saved adjudication by about a minute.
3. **Show both label sets for every R unit.** Flag `SCHEME_DEPENDENT` on:
   - R1 REFIT_LOC full OOF (frozen q05 +0.105 → registered -0.020);
   - R1 REFIT_PAIR_TR full OOF (+0.050 → -0.020).

   Any sentence about these units must state the dependence, for example: "an improvement appeared only under the superseded
   implementation; it is not established under the registered scheme."
4. **Record the variance for the two flagged units.** The verifier should report D, the bootstrap SE under both schemes and the
   influence of the largest households. Their q05 shifts (about 0.07-0.13) are far larger than the 7-25% SE change seen on D, which
   suggests heavy-tailed household increments, for example NLL near the 1e-12 floor. This is descriptive and changes no label.

## Consequences

- **Claims: none change.** Rung 1 is NOT_MET under both schemes: the outcome is NO_RECIPE, and no direction is improved on all four
  partitions under either scheme. Rungs 2-5 do not use these labels.
- **Recipe selection: unaffected.** Gates G1-G4 do not use question-A labels, and G1 failed.
- **The D-stage designation from A2 stands.** The D labels came from units run before REPAIR_FREEZE. D is diagnosis, and §8 concerns
  repair outcomes only. The D-stage primary labels are the registered-text labels, with the frozen-code labels shown as superseded.
