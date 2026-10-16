# Start here: PCRL mechanism-design campaign (data-free)

**Bottom line: `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` for a real-data pilot.**
- Two independent designs were reviewed and one was selected at the design stage: SEX-mixture marginalisation (MMP).
- Its own pre-registered prototype and the verifier's independent probes then contradicted the part of its rationale that a real-data benefit needs.
- Without merging structure, MMP's effect on coalition recovery has no definite sign: −0.033 in one draw, about +0.02 in two.

The campaign also:
- corrected the adversary definition and its motivation, with a manuscript patch;
- wrote a closed-lane decision aid;
- checked the closest prior work.

**Data access.** No real-person array was read in any role. There were no real-person fits, no attacks, no new population and no sealed-assessment access, and DEV_REPORT labels were not read.

## What happened

1. **Problem fixed first.** `PROBLEM_SPECIFICATION.md` sets out:
   - utility definitions recovered from code (24 clauses: accuracy ≤ 0.010, NLL ≤ 0.010, summed Brier ≤ 0.005, against U0 and Ucal);
   - the release contract and access by role;
   - the claims kept separate;
   - the mathematical facts every proposal must respect.

   `EVIDENCE_PACK.md` holds the published numbers.
2. **Closed lanes.** `CLOSED_LANES.md` covers about 40 campaigns. In every lane the bottleneck was task utility or held-out generalisation, never the knob that was turned. Neural adversarial joint training was unbeaten except by the Adult tree recipe, which did not transfer to Florida.
3. **Adversary corrected.** `ADVERSARY_AND_MOTIVATION.md` contains:
   - an access table for both original papers, the combined manuscript and the ACS service, with every paper/code mismatch flagged;
   - manuscript-ready prose and a short answer for the advisor;
   - prior adversary models, showing the adversary is not new;
   - `MANUSCRIPT_ADVERSARY_PATCH.diff`, which applies cleanly to `main.tex` at `3b5065805`.
4. **Two independent proposals,** each hashed before either was revealed:
   - designer 1's MMP: fit with SEX, release the fixed-weight average over SEX;
   - designer 2's RCCS: income frozen, propensity-driven occupation counter-shift.

   Three reviewers assessed both: mathematics, threat model and prior work.
5. **Selection** (`SELECTION_DECISION.md`, under criteria fixed before either proposal existed).
   - **MMP was selected conditionally.** It is critic-free, its price is exact, and it is distinct from the closed lanes. It is, however, an established technique (Žliobaitė & Custers; Pearl's adjustment formula) applied to a new endpoint.
   - **RCCS was rejected.** Its protection is defined against its own linear and quadratic critic class, and its feasible case violates the binding clauses.
6. **Prototype.** `prototype/PROTOTYPE_REPORT.md`, pre-registered and run once: **24 of 27 checks held.**
   - **The merging leg is confirmed.** MMP removes pure-proxy SEX information exactly, at the predicted price.
   - **The learnability leg failed.** Without pure proxies, deleting the inferred-SEX term changed nothing a reader learns: pair 0.7207 vs 0.7211.
   - **The verifier reproduced every number** with separate code. It also:
     - found my masking comparator mis-specified, so masking was never tested;
     - corrected its own subgroup expectation: men bore 2.2–2.6× the guard;
     - showed in a registered probe that MMP can raise recovery.
7. **Pilot.** `PILOT_PROTOCOL_DRAFT.md`, with code in `mechanism_design/pilot/`. It is prepared and rehearsed end to end on synthetic stand-ins, but it is not run and not recommended as a competitive test. At most, Stage 0 (MODEL_FIT and MODEL_SELECT only, about 0.5 CPU-h) could serve as a diagnostic, after explicit authorisation.
8. **Methodological finding.** Under the deterministic contract, every evaluated release is essentially injective.
   - The population-optimal coalition reader therefore recovers as much as it could from the raw fields: AUC*(S; V) = AUC*(S; X). `prototype/DEMONSTRATIONS.json` D3 shows this.
   - All measured differences between defenses, the neural control included, are differences in what a fixed slate of readers learns from about 8,000 labelled releases.
   - Reader-independent protection requires merging input cells, which the closed lanes found too costly at the caps.

## Read in this order

1. `ADVISOR_BRIEF.md` (at most two pages).
2. `ADVERSARY_AND_MOTIVATION.md`, with `MANUSCRIPT_ADVERSARY_PATCH.diff`.
3. `SELECTION_DECISION.md`, `METHOD_SPECIFICATION.md`, then `prototype/PROTOTYPE_REPORT.md`.
4. `CLOSED_LANES.md`, `CLOSEST_PRIOR_WORK.md`.
5. `PILOT_PROTOCOL_DRAFT.md`.
6. `DECISION_LOG.md`, which is the full chronology and includes the disclosures.
7. `HANDOFF.json`, which lists commits, artifacts, verified scope and missing dependencies.
8. Role work, in `roles/`:
   - `designer_1/` and `designer_2/`: proposals and cross-critiques;
   - `adversary_reviewer/`, `prior_work_reviewer/` and `math_verifier/`;
   - `coordinator/`: briefs, the selection criteria and the closed-lane helper report.

## Disclosures

These are also in `DECISION_LOG.md`.
- **Launch messages.** The two designers' launch messages differed by one sentence, a diversity nudge for designer 2. Their briefs were identical.
- **Late verifier result.** The verifier's smooth-law test of the RCCS map arrived after the selection. In that smooth Gaussian law, every reader, nonlinear ones included, fell 0.019–0.045. It does not change the decision, and it is recorded as a post-decision note.
- **Additions before the first run.** Prototype laws were added before the first run to satisfy the verifier's conditions. Designer 1 had strengthened one of their own laws after a first run, which they disclosed.
- **Failed expectations.** Three pre-registered prototype expectations failed and are reported as failures.
- **Dependency.** CatBoost 1.2.8 was installed into this campaign's own private library folder for the synthetic rehearsal. The source kept its copy in a private directory that this campaign did not open.
