# PCRL: mechanism-design campaign (data-free), corrected 2026-10-16

> **This is the corrected version. Start with `results/CORRECTIONS.md`.**
> - **The original package** was published at commit `e141330`. It is pinned and unchanged at https://github.com/Bostesa/PCRL/tree/e141330fbb241c004dd154ec023e714bf56ce401/publications/pcrl-mechanism-design-v1.
> - **What was wrong.** That version said that every evaluated release is essentially injective. It concluded that the optimal attacker recovers exactly what the raw fields allow and that every measured difference between defenses is a learnability difference. **That claim does not follow and is withdrawn.** Whether the actual ACS models are injective on their population is unknown. They all read an encoding that pools rare category levels.
> - **Other corrections:**
>   - a deterministic many-to-one probability release can remove sensitive information inside the existing output contract;
>   - MMP's "exact price", "if male / if female" wording and "pure-proxy only" removal claim;
>   - RCCS's "linear and quadratic only" summary;
>   - the adversary's exclusion rationale (a v2 manuscript patch).
> - **Evidence.** Small pre-registered exact checks support the corrected statements, and an independent verifier recomputed them with separate code.
> - **The status is unchanged.** No number changed.

**Result: no mechanism is ready for a real-data pilot.**

Two independent mechanism designs for the ACS two-recipient output service were reviewed for mathematics, threat model and prior work.
- **Selected (conditionally): SEX-mixture marginalisation (MMP).** It fits the task models with SEX as an input and releases the fixed-weight average of the fitted model's predictions with SEX set to each value. It is critic-free and, under a known law, its price is an exact formula. It is an established technique applied to a new endpoint.
- **Rejected as presented: a propensity-driven conditional counter-shift (RCCS).** Its feasibility case failed the binding utility clauses and its supporting accounting was inconsistent.
  - The correction pass found that the other ground first given, "protection only against its own linear and quadratic critics", is law-dependent rather than established. In two smooth synthetic laws its fitted protection also held against tree and neural readers, while its optimal recovery rose.
  - A repaired RCCS is unresolved and was not promoted.

MMP's pre-registered data-free prototype passed 24 of 27 checks but partly contradicted its rationale:
- **Confirmed:** exact removal of SEX information where MMP merges inputs with different SEX posteriors, at the predicted price.
- **Contradicted:** the learnability leg. In a law whose release is one-to-one by construction, deleting the inferred-SEX term changed nothing a reader learns.
- **Also:** the verifier's probe showed MMP can raise recovery.

The pilot is prepared and rehearsed on synthetic stand-ins only. It is not run, not recommended as a competitive test, and not authorised by the correction.

**Methodological point (corrected).**
- Recovery from any release is bounded by recovery from the inputs. Equality needs the release to keep P(SEX | inputs) almost surely, which is not established for any actual ACS model.
- Measured differences between defenses are differences in what the declared readers recover from the declared auxiliary sample. Their split into removed information and learnability is unknown.

**The campaign also delivers:**
- a corrected adversary definition, with an access table for both original papers, the combined manuscript and the ACS service, plus a manuscript patch (v2);
- a closed-lane decision aid;
- a closest-prior-work review.

**Data access.** No real-person data were read, fitted or attacked, in the campaign or in the correction pass. The Florida sealed assessment stayed closed; source disclosure D2 still applies.

## Contents

- **`results/CORRECTIONS.md`:** the correction record. Read it first, then `results/START_HERE.md` and `results/ADVISOR_BRIEF.md`.
- **`results/corrections/`:** the pre-registered exact checks and their results.
- **`results/roles/correction_v1/`:** the correction pass's mathematical review, adversary review and independent verification.
- **`results/`:**
  - adversary and motivation, with the manuscript patch;
  - closed lanes and closest prior work;
  - the problem specification and evidence pack;
  - the selection decision and method specification;
  - the prototype pre-registration, results and report;
  - the pilot protocol draft and rehearsal results;
  - the decision log, agent roster, resources, backup verification and handoff;
  - `roles/`: the role work (proposals, cross-critiques and reviews).
- **`code/mechanism_design/`:** the campaign code, including `correction/` (the exact checks, queue, custody and closeout of the correction pass).
- **`code/mechanism_design_independent/`:** the verifiers' separate code, including `correction_v1/`.

## Not included

- private review sources, reviewer identifiers and the meeting transcript;
- person-level data, predictions and models;
- absolute paths and credentials.

Content scrubbing does not make the repository anonymous: the public account name and the commit identity remain visible.
