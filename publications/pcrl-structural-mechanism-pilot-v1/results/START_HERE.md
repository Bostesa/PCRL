# Start here: PCRL structural-mechanism campaign

**Bottom line: `NO_CANDIDATE_SURVIVED` at candidate selection. No pilot was run.**

Two independent designers proposed the same family: a shared many-to-one partition of task-posterior space, merged greedily under exact per-clause utility caps. Under the selection criteria committed before either proposal was read:
- **Both fail S3, distinct ingredient, as committed.** The closest-work verdict for both is `ENGINEERING_COMBINATION_NO_NEW_INGREDIENT`: a privacy-funnel or IBSI-type constrained merge. That alone triggers rule 1.
- **A further ruling on S2, made after reading.** The coordinator rules that both fall outside the intent of S2: the release is a SEX-aware quantiser applied after the fact to a task-only teacher's outputs, which the reviewer places inside the output-channel family. Read literally, S2 is arguably met, and the outcome does not depend on this ruling.

This is a failed selection prerequisite, not a negative scientific result. Nothing was fitted:
- no defense model;
- no reader;
- no MODEL_SELECT label array loaded (the restore copied and byte-hashed the export files only);
- no DEV_REPORT truth read;
- no sealed-assessment access.

No competitive new mechanism has been established. That corrected status is unchanged.

## What was produced

1. `SERVICE_AND_THREAT_MODEL.md` (v1). The concrete two-recipient ACS service, called a controlled benchmark with an illustrative scenario; no documented deployment was found. It covers:
   - why the recipients' actions need probabilities, with primary sources;
   - the declared adversary, with linkage and auxiliary labels as explicit grants;
   - the motivation;
   - what is documented and what is assumed;
   - the closest methods versus this access;
   - the composition statement made precise;
   - **the lookup reader, frozen before any fit**.
2. **Two independent proposals**, hashed before review:
   - `roles/designer_1/PROPOSAL.md`, SCM, `6bc0617b…`;
   - `roles/designer_2/PROPOSAL.md`, SCAR, `74b42067…`.
3. **Reviews.**
   - Adversary reviewer: `roles/adversary_reviewer/SERVICE_REVIEW.md` and `PROPOSAL_THREAT_CHECK.md`.
   - Closest-work reviewer: `roles/prior_work_reviewer/FIELD_MATRIX.md` (18 sources, primary text or official code) and `PROPOSAL_COMPARISON.md`.
4. `SELECTION_DECISION.md`. The criteria applied, the rulings, what is and is not established, and the ready specification if the family is ever piloted as an engineering application (not authorised).
5. **Implementation.** Runnable code, tested where possible, never run on real data except the restore and label-free support units. It lives in `structural_pilot/`:
   - hash-checked restore of the development roles and the frozen comparator bank (38 files);
   - role loaders with access guards;
   - a job queue with a cumulative CPU ledger;
   - reader, report, selection, bank and fit stages;
   - the frozen lookup reader, with synthetic checks;
   - the shared-cell merge engine, whose exact merge identities match brute force to 6.4e-17;
   - task-only teacher utilities.
6. `roles/coordinator/SUPPORT_FACTS.json`, label-free: about 90% of people have a unique encoded input in their role, and about 17% of held-out inputs occur in MODEL_FIT.
7. `MANUSCRIPT_NOTE.md`, the secondary task:
   - the manuscript's public-code sentence re-checked against GitHub main `55e4cb1d1`, and accurate;
   - one delta-review item still open;
   - a motivation paragraph within the paper's evidence cutoff.
8. `DECISION_LOG.md` (chronology), `RESOURCE_INVENTORY.json`, `HANDOFF.json` and, from the verifier, `roles/verifier/VERIFICATION.md`.

## What remains untested

- Whether any shared deterministic partition of these releases can meet the 24 caps on MODEL_SELECT (gate G0 of both proposals).
- Whether SEX-aware merging lowers fresh-reader recovery relative to its task-only twin.
- Whether any structural mechanism beats NEURAL_RAW ρ1 E2 (pair 0.640 / 0.638).

## Roles (actual participation)

| Role | Who |
|---|---|
| Coordinator and implementation owner | main session |
| Designer 1 and designer 2 | one agent each, identical briefs |
| Recipient-use and adversary reviewer | one agent, two passes |
| Closest-work and baseline reviewer | one agent, two phases |
| Independent verifier | one agent |
