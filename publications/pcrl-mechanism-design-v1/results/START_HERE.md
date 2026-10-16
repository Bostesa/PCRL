# Start here: PCRL mechanism-design campaign (data-free)

> **Corrected on 2026-10-16. Read `CORRECTIONS.md` first.**
> - The original version of this package (public commit `e141330`) said that every evaluated release is essentially injective. It concluded that the optimal attacker recovers exactly the raw-field level and that every measured difference is a learnability difference. **That claim does not follow and is withdrawn.**
> - Several other statements are also corrected: MMP's exact price, the "if male / if female" wording, the "pure-proxy only" removal claim, RCCS's "linear and quadratic only" summary, and the adversary's exclusion rationale.
> - The original files remain at `e141330` and in git history. Historical log entries, locks, numbers and failed checks are unchanged.
> - **The status below is unchanged; its rationale is corrected.**

**Bottom line: `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` for a real-data pilot.**
- Two independent designs were reviewed and one was selected at the design stage: SEX-mixture marginalisation (MMP).
- Its own pre-registered prototype contradicted the learnability leg of its rationale in a synthetic law, and the verifier's probe showed its effect can go either way. A real-data benefit would need either merging structure in the Florida fields or a learnability effect; which applies, if either, is unknown.
- In the verifier's probe without pure-proxy fields (all ten fields task-relevant), MMP's net effect relative to its SEX-blind twin had no definite sign: it lowered coalition recovery by 0.033 in one draw and raised it by about 0.02 in two.

The campaign also:
- corrected the adversary definition and its motivation, with a manuscript patch (patch v2 after the correction pass);
- wrote a closed-lane decision aid;
- checked the closest prior work.

**Data access.** No real-person array was read in any role, in the campaign or in the correction pass. There were no real-person fits, no attacks, no new population and no sealed-assessment access, and DEV_REPORT labels were not read.

## What happened

1. **Problem fixed first.** `PROBLEM_SPECIFICATION.md` sets out:
   - utility definitions recovered from code (24 clauses: accuracy ≤ 0.010, NLL ≤ 0.010, summed Brier ≤ 0.005, against U0 and Ucal);
   - the release contract and access by role;
   - the claims kept separate;
   - the mathematical facts every proposal must respect.

   Its §7 lists "coarsened outputs" as a contract change. That holds only for outputs outside the probability simplex: a deterministic many-to-one probability release is inside the primary contract (`CORRECTIONS.md` row 6). `EVIDENCE_PACK.md` holds the published numbers.
2. **Closed lanes.** `CLOSED_LANES.md` covers about 40 campaigns. Each lane reports task utility or held-out generalisation as the bottleneck, never the knob that was turned. Neural adversarial joint training was unbeaten except by the Adult tree recipe, which did not transfer to Florida. These are findings about those mechanisms, not a universal privacy–utility limit.
3. **Adversary corrected.** `ADVERSARY_AND_MOTIVATION.md` contains:
   - an access table for both original papers, the combined manuscript and the ACS service, with every paper/code mismatch flagged;
   - manuscript-ready prose and a short answer for the advisor;
   - prior adversary models, showing the adversary is not new;
   - `MANUSCRIPT_ADVERSARY_PATCH.diff`, which applies cleanly to `main.tex` at `3b5065805`.

   The correction pass replaced the exclusion rationale. Other access is now described as distinct threat models outside this evaluation's scope, not as a provider breach.
4. **Two independent proposals,** each hashed before either was revealed:
   - designer 1's MMP: fit with SEX as an input, then release the fixed-weight average of the fitted model's predictions with SEX set to each value;
   - designer 2's RCCS: income frozen, propensity-driven occupation counter-shift.

   Three reviewers assessed both: mathematics, threat model and prior work.
5. **Selection** (`SELECTION_DECISION.md`, under criteria fixed before either proposal existed).
   - **MMP was selected conditionally.** It is critic-free, its price under a known law is an exact formula, and it is distinct from the closed lanes. It is, however, an established technique (Žliobaitė & Custers; the adjustment formula) applied to a new endpoint.
   - **RCCS was rejected as presented.** Its feasibility case failed the binding clauses and its supporting accounting was inconsistent.
   - The correction pass found that the second ground originally given for rejecting RCCS ("protection only against its own linear and quadratic critic class") is law-dependent, not established. In the verifier's two smooth synthetic laws, its fitted protection also held against tree and neural readers, while its optimal recovery rose. A repaired RCCS is unresolved and was not promoted.
6. **Prototype.** `prototype/PROTOTYPE_REPORT.md`, pre-registered and run once: **24 of 27 checks held.**
   - **In the prototype laws, where MMP merged inputs with different SEX posteriors, it removed SEX information exactly, at the predicted price.** This covers pure-proxy fields and, as the verifier showed, SEX directions that the tasks do not use. Relative to its twin, its effect can still go either way (below).
   - **The learnability leg failed.** In a law where the release is injective by construction, deleting the inferred-SEX term changed nothing a reader learns: pair AUC twin 0.7211, MMP 0.7207.
   - **The verifier reproduced every number** with separate code. It also:
     - found my masking comparator mis-specified, so masking was never tested;
     - corrected its own subgroup expectation: men bore 2.2–2.6× the guard;
     - showed in a registered probe that MMP can raise recovery.
7. **Pilot.** `PILOT_PROTOCOL_DRAFT.md`, with code in `mechanism_design/pilot/`. It is prepared and rehearsed end to end on synthetic stand-ins, but it is not run and not recommended as a competitive test. Nothing in the correction pass authorises a run.
8. **Methodological point (corrected).**
   - By data processing, the optimal coalition attacker can recover no more from any release than from the inputs: AUC\*(S; V) ≤ AUC\*(S; X). Equality needs the release to reconstruct the inputs, or at least keep P(S | X), almost surely.
   - That is **not established for any actual ACS model**. Every model reads an encoding that pools rare category levels, so none is one-to-one in the six raw fields on the declared domain. Whether any keeps P(S | X) on the population support is unknown.
   - Measured differences between defenses are therefore differences in what the declared readers recover from the declared auxiliary sample. How much of each difference is information removed and how much is learnability is unknown.
   - A deterministic many-to-one release can remove SEX information inside the existing output contract: exact check B does it at full Bayes utility. Whether any such release meets this benchmark's caps on Florida is an open feasibility question.
   - **What backs these statements.** The finite-law instances behind them (checks A–E in `corrections/`) are pre-registered, exactly checked and independently recomputed (`roles/correction_v1/verifier/`). The general statements rest on the proofs in the mathematical review and the verification. The ACS-specific statements rest on code reading.

## Read in this order

1. `CORRECTIONS.md`: what was wrong, the replacement wording, and the evidence.
2. `ADVISOR_BRIEF.md` (at most two pages).
3. `ADVERSARY_AND_MOTIVATION.md`, with `MANUSCRIPT_ADVERSARY_PATCH.diff`.
4. `SELECTION_DECISION.md`, `METHOD_SPECIFICATION.md`, then `prototype/PROTOTYPE_REPORT.md`.
5. `CLOSED_LANES.md`, `CLOSEST_PRIOR_WORK.md`.
6. `PILOT_PROTOCOL_DRAFT.md`.
7. `DECISION_LOG.md`, which is the full chronology, including the correction entries and the disclosures.
8. `HANDOFF.json`, which lists commits, artifacts, verified scope and missing dependencies.
9. Role work, in `roles/`:
   - `designer_1/` and `designer_2/`: proposals and cross-critiques;
   - `adversary_reviewer/`, `prior_work_reviewer/` and `math_verifier/`;
   - `coordinator/`: briefs, the selection criteria and the closed-lane helper report;
   - `correction_v1/`: the correction pass's mathematical review, adversary review and independent verification.

## Disclosures

These are also in `DECISION_LOG.md`.
- **Launch messages.** The two designers' launch messages differed by one sentence, a diversity nudge for designer 2. Their briefs were identical.
- **Late verifier result.** The verifier's smooth-law test of the RCCS map arrived after the selection. In two settings of that smooth Gaussian law family, every fitted reader, nonlinear ones included, fell by 0.018–0.045. The source text says 0.019; its data give a smallest drop of 0.0183. It did not change the selection, and it is recorded as a post-decision note. The correction pass uses it to correct the RCCS summary.
- **Additions before the first run.** Prototype laws were added before the first run to satisfy the verifier's conditions. Designer 1 had strengthened one of their own laws after a first run, which they disclosed.
- **Failed expectations.** Three pre-registered prototype expectations failed and are reported as failures.
- **Dependency.** CatBoost 1.2.8 was installed into this campaign's own private library folder for the synthetic rehearsal. The source kept its copy in a private directory that this campaign did not open.
- **Coordinator errors corrected after publication.**
  - The unsupported injectivity claim and the conclusions drawn from it, including the advisor brief's impossibility sentence.
  - The overstatements listed in `CORRECTIONS.md`.

  These were found in the correction pass; none changed a number.
