# PCRL: mechanism-design campaign (data-free)

**Result: no mechanism is ready for a real-data pilot.**

Two independent mechanism designs for the ACS two-recipient output service were reviewed for mathematics, threat model and prior work.
- **Selected (conditionally): SEX-mixture marginalisation (MMP).** It fits with SEX and releases the fixed-weight average over SEX. It is critic-free and its price is exact, but it is an established technique applied to a new endpoint.
- **Rejected: a propensity-driven conditional counter-shift (RCCS).** Its protection is defined against its own linear and quadratic critic class.

MMP's pre-registered data-free prototype passed 24 of 27 checks, but it partly contradicted MMP's rationale:
- **Confirmed:** exact removal of pure-proxy SEX information at the predicted price.
- **Contradicted:** the learnability leg. Without pure proxies, deleting the inferred-SEX term changed nothing a reader learns. A real-data benefit would have needed this leg.

The pilot is prepared and rehearsed on synthetic stand-ins only. It is not run and not recommended as a competitive test.

**Methodological finding.** Every evaluated deterministic release is essentially injective. The population-optimal coalition attacker therefore recovers as much as from the raw fields, and all measured differences between defenses, the neural control included, are differences in what a fixed reader slate learns.

**The campaign also delivers:**
- a corrected adversary definition, with an access table for both original papers, the combined manuscript and the ACS service, plus a manuscript patch;
- a closed-lane decision aid;
- a closest-prior-work review.

**Data access.** No real-person data were read, fitted or attacked. The Florida sealed assessment stayed closed; source disclosure D2 still applies.

## Contents

- **`results/START_HERE.md`:** start here, then `results/ADVISOR_BRIEF.md`.
- **`results/`:**
  - adversary and motivation, with the manuscript patch;
  - closed lanes and closest prior work;
  - the problem specification and evidence pack;
  - the selection decision and method specification;
  - the prototype pre-registration, results and report;
  - the pilot protocol draft and rehearsal results;
  - the decision log, agent roster, resources, backup verification and handoff;
  - `roles/`: the role work (proposals, cross-critiques and reviews).
- **`code/mechanism_design/`:** the campaign code (prototype, demonstrations, pilot, queue, data guard, custody, closeout).
- **`code/mechanism_design_independent/`:** the verifier's separate code.

## Not included

- private review sources, reviewer identifiers and the meeting transcript;
- person-level data, predictions and models;
- absolute paths and credentials.

Content scrubbing does not make the repository anonymous: the public account name and the commit identity remain visible.
