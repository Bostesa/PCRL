# Research decision: solver adequacy of the finite-output benchmark (campaign `pcrl_finite_solver_adequacy_v1`)

Florida ACS 2019 development cohort (admitted public-use PUMS records), 2026-10-21. Real-data roles: MODEL_FIT only. CALIBRATION was
not used, and MODEL_SELECT was not opened by any stage. The cached out-of-fold statistics are derived from MODEL_FIT training labels.
Everything here is development evidence on previously and adaptively used records.

This version supersedes the draft 5ad143f54. It applies the statistical reviewer's closing statement
(`roles/statistical_reviewer/CLOSING_STATEMENT.md`, 177e4eeb1 and addendum 1 af57badc8: corrections C1-C17) and the engineering
reviewer's claims review (`roles/engineering_review/REVIEW_CLAIMS.md`, 2312b801f). No label changed.

## Headline (the statistical reviewer's required wording)

> Development evidence on adaptively used MODEL_FIT records only; MODEL_SELECT was not opened. Under the frozen FSA1 rules the
> advancement decision is `OPTIMIZER_REPAIR_CLOSED`, and the conditional development-data pilot did not run, so no pilot outcome
> class (method candidate, component, engineering result or bounded negative result) was computed. No implementation defect was
> found: the four documented approximations are not defects under the rules (route D). The revised solver SV1 lowers PRC's stated
> objective at every registered strength in both versions, by 3e-6 to 0.0105 nats. At its S2 point, the MODEL_FIT nested held-out
> pair LOOKUP AUC against the spend-matched original point changed by -0.0018 (se 0.0020) and -0.0038 (se 0.0017), pooled -0.0028
> (mean se 0.0019; 2-SE interval -0.0065 to +0.0009). That is not resolved, and the task alternative also failed (spend -0.0006 /
> -0.0007 against the required -0.001). Route S therefore ended `S_PROXY_ONLY` and supersedes route E's advance, which rests on a
> single registered invented witness law built to contain a feasible gain. The previous campaign's P1 evidence was reproduced
> bitwise and independently recomputed (`P1_REPLAY_VERIFIED` under the reviewer's estimator-identity ruling; the strict text of
> RULES 8 gives `P1_ESTIMATOR_CHANGED`, because the replay re-implements the selection loop with renamed local variables). Better
> optimization is shown for the surrogate objective only. Whether it improves usefulness or recovery for the MODEL_SELECT readers,
> and whether propensity adds anything beyond the equally optimized HQ control, remain untested.

## Decision

**`OPTIMIZER_REPAIR_CLOSED`** under the frozen FSA1 rules (`decisions/ADVANCEMENT.json`). **The conditional development-data pilot
did not run.**
- Rules identity: `roles/statistical_reviewer/RULES.json` SHA-256 ec4af5e6e07f33ba…, `fsa/rules.py` bd8a4c5b9563d33b…, both
  unchanged since they were frozen. ADVANCEMENT.json itself records no rules hash.
- The statistical reviewer re-derived every label with the frozen functions from the saved inputs.

| Route | Label | In one line |
|---|---|---|
| D, implementation defect | `D_NO_MATERIAL_DEFECT` | The four documented approximations were examined and classified as approximations, not defects. Objective, routing, refresh, Lloyd and chain all reproduce in independent code. Route D classifies; it does not measure effects on decisions. |
| E, registered invented laws | `E_SUPERSEDED_BY_S` | Its own label is `E_ADVANCES`. On 60 registered cells (30 distinct: seed 1 equals seed 0) SV1 was never worse than the original. It gave an exact recovery gain with no task loss up by more than 0.0005 at all six strengths of law D (12 cells). Law D is the witness law built to contain a feasible gain, and no other law gave a useful cell. Route S supersedes route E for the same revision (RULES 5.2). |
| S, cached MODEL_FIT statistics and nested MODEL_FIT | `S_PROXY_ONLY` | S-a holds: SV1 lowers PRC's fitted objective at every registered strength in both versions. The per-version recovery point conditions held, but the pooled 2-SE condition failed (-0.0028 + 2 × 0.0019 > 0), and so did the task alternative. |

**Rule outcome `OPTIMIZER_REPAIR_CLOSED`; no pilot outcome class was computed.**
- Descriptively, the optimizer-repair hypothesis closes on MODEL_FIT, with the bound stated: pooled pair LOOKUP change -0.0028,
  2-SE interval -0.0065 to +0.0009, measured with the objective's own LOOKUP reader only.
- The campaign also produced verified engineering findings. They are not the rules' `ENGINEERING_RESULT` class, which needs a
  measured MODEL_SELECT gain over the original solver (RULES 7.6).
- There is no method candidate and no component, and no fitted method is claimed.
- The rules' closing condition applies: "the optimizer-repair hypothesis closes: … a revised solver improves only the surrogate
  objective without a usefulness / recovery rationale". It is not an impossibility result about other mechanisms.

**Verified engineering findings (descriptive):**
- The previous campaign's P1 evidence reproduces bitwise from a rerun of its nested estimator, and the verifier recomputed it
  independently (`P1_REPLAY_VERIFIED`; see "Rulings and disclosures").
- SV1 lowers PRC's fitted objective at every registered strength in both versions. It is not uniformly better (HQ, version 0,
  strength 8) and not optimal (question 1).
- Exact references show where the original seven-tilt search misses deployable solutions.

**The previous campaign is unchanged.** FRC1's labels stand: the replay rule never relabels them, and route D found no defect
affecting them. Both of its readings stand: `INCOMPLETE` under its frozen rule, `ENGINEERING_REPAIR` under its post hoc ER1.

## The brief's questions

**1. Did the seven-tilt proposal search miss demonstrably better deployable solutions?** Yes, for its own objective.
- **Exact laws.** The original tries tilts λ/2^j, j = 0..6. Improving windows that contain no such tilt are invisible:
  - law D: window (0.002717, 0.002866) at every strength (the smallest tilt tried, λ/64, is at least 0.0039); a feasible partition
    with exact pair AUC 0.5803 against the stalled 0.6882;
  - law E: window (0.287, 0.466) at strengths 0.5 and 1.
- **Real cached statistics.**
  - In version 1 at strength 8, the original's final refreshed-book tilt line holds a deployable partition with objective 2.5732, in
    a tilt window 4.8e-6 wide near 0.1875 (between the tried 0.125 and 0.25), against the stalled 2.5902.
  - In version 0 no partition on that line beats the current state. Improvement there needs the current book's line, which SV1
    adds.
- **SV1.**
  - It lowers PRC's fitted objective at all 6 registered strengths in both versions, by 3e-6 to 0.0105.
  - For HQ it is lower in 11 of 12 cells, and higher in version 0 at strength 8 (+1.4e-4).
  - It is not optimal. In version 1 at strength 8 its chain ends at 2.5797, above the 2.5732 partition the audit exhibited, which
    lies off SV1's own continuation path.
  - On the exact laws it stays above the best same-class reference in 10 of 30 distinct cells: law C at 0.25 and 1, law D at all six
    strengths (by 0.00006-0.00015), law E at 0.25 and 8. The largest gap is 0.0102.
  - That reference is the best certified realizable point, not a proven class optimum. In law E at strengths 0.5-4, SV1 ends 0.0003
    to 0.041 below it.
- "Better" here means the stated surrogate objective, not usefulness or protection.

**2. Did the objective / probability inconsistencies matter?** Under the frozen rule they are documented approximations, not defects
(`D_NO_MATERIAL_DEFECT`). Route D does not measure whether they change decisions. The seven-tilt search changes the solutions
(questions 1 and 3). The other three are quantified below (`tables/OBJECTIVE_AUDIT_TABLES.md`).
- **Shrunk, linearized routing score against the exact information.**
  - At strength 8, the routing's shrunk dependence score and the exact dependence term are anti-ranked (Spearman -0.995 / -0.996,
    PRC). This holds along the final refreshed book's tilt line, across its 82,052 / 91,490 distinct partitions.
  - This is consistent with the original chains accepting almost nothing (PRC: 1 of 7 Lloyd steps).
  - Acceptance by the exact objective keeps every accepted step valid.
- **Cell-mean codeword refresh against the prototype-dependent guard.** On the real chains the refresh never raised the objective
  (largest change -9.1e-7). In law B, stale codewords raised the objective without changing the release.
- **Emitted B1 tables against search prototypes.**
  - The largest absolute difference is 0.0095 / 0.0066 (PRC_S2) and 0.032 / 0.054 (HQ_S2).
  - The emitted tables' training cross-entropy is lower than or equal to the prototypes'. The objective's task term uses the
    prototypes, so it slightly overstates the emitted release's training task loss.
  - Partitions are accepted by the objective. Strengths are selected on the emitted release's nested held-out spend.
- **Monotonicity.** The original accepted objective decreases strictly. This is not an optimality certificate.
- **Rule consistency.**
  - Every accepted state, original or SV1, is reproduced by the deployed routing rule: 72 of 72 saved books, also from the backup.
  - No accepted SV1 tilt is on a knife edge: a 1e-9 relative tilt change moves no one.
  - Routed with the deployed full-pool teachers, 48-53% of the training rows land in the same joint cell as their out-of-fold
    routing, so about half move. This is because full-pool predictions on training rows differ from out-of-fold ones.
  - The B1 tables use the out-of-fold cells, which is what new people resemble. This was not examined further and is not a defect
    under the rules.

**3. Did a revised solver improve actual usefulness or recovery?** Not by the registered standard.
- **MODEL_FIT nested held-out evaluation.**
  - SV1's S2 point is strength 8 in both versions, with total NLL spend over the twin of 0.00153 / 0.00121. The original's S2 point
    (PRC@8.0) spends 0.00214 / 0.00194.
  - The bracket is the cheapest original path point whose nested spend is at least SV1's spend − 0.0005 (RULES 4.3). That is
    PRC@2.0 in both versions.
    - In version 0 it behaves as the same partition as PRC@8.0: identical nested spend and identical paired differences.
    - In version 1 it spends 0.00088, which is 0.00034 less than SV1's point, inside the 0.0005 no-worse band.
  - Against the bracket, SV1's pair LOOKUP AUC is lower by 0.0018 (se 0.0020) and 0.0038 (se 0.0017).
    - The per-version point conditions held in both versions. Only the pooled 2-SE condition failed: pooled -0.0028 (mean se
      0.0019), 2-SE interval -0.0065 to +0.0009.
    - "Not resolved" does not mean "no change".
  - Against the original S2 point:
    - pair LOOKUP AUC -0.0018 / -0.0003, at lower spend (-0.0006 / -0.0007);
    - in version 1 the single views move in opposite directions: occupation view +0.0023 (se 0.0022), within the 0.010 allowance,
      and income view -0.0062 (se 0.0026);
    - the task alternative fails: spend -0.0006 / -0.0007 against the required -0.001.
  - These use the objective's own reader type (LOOKUP) only, and were not carried to MODEL_SELECT.
- **Exact laws.**
  - Law D: pair AUC from 0.6882 to 0.5803 in all six distinct cells, with every task loss within +0.0005. Law D is the witness
    law, and no other law gave a useful cell.
  - Laws C and E at strength 2 and above: SV1 reaches partitions beyond the original limits (infeasible). In law C, occupation
    accuracy loss rises by 0.033.
  - Law E at strengths 0.5 and 1: SV1 lowers recovery (pair AUC 0.6182 to 0.5979) only by raising occupation accuracy loss by
    0.0060, so it is not "no worse" on the task.
  - Better optimization of this objective can therefore trade usefulness away, beyond the original limits (infeasible) at strength
    2 and above in laws C and E. The objective has no utility constraint.

**4. Did HQ improve equally?**
- SV1 lowered HQ's fitted objective in 11 of 12 strength-version cells, and was worse in version 0 at strength 8 (+1.4e-4).
- In the nested runs, revised HQ at strength 8 has pair LOOKUP AUC 0.6225 / 0.6251 at spends 0.0029 / 0.0063. Original HQ has
  0.6262 / 0.6276 at 0.0012 / 0.0041: revised HQ is lower in recovery at higher spend.
- Descriptively (no SEs, no registered comparison, MODEL_FIT nested LOOKUP only):
  - At strength 8, revised PRC's pair LOOKUP AUC is 0.0016 / 0.0048 below revised HQ's, at lower spend (0.0015 / 0.0012 against
    0.0029 / 0.0063).
  - Under the original solver the same gap was 0.0035 / 0.0070. In the previous campaign that larger nested gap did not carry to the
    MODEL_SELECT reader grid (benefit +0.0003, se 0.0051).
  - LOOKUP is the reader type the objective targets, so by card FA_PRC_v1 an advantage seen only by it is reader-targeted.
  - Question C, whether propensity adds anything beyond HQ, is unanswered.
- **Expressivity.** Task scores identify the six-field support states for 98.7-99.0% of people. An unrestricted task-score mapping
  could therefore reproduce any PRC partition, except on the 1.0-1.3% of people whose task scores are shared with another support
  state (MODEL_FIT support). Any PRC-HQ difference is about rule parametrization and search inside two restricted families, not
  the information propensity adds.

**5. Did the conditional pilot run, and what triggered or prevented it?** It did not run.
- Route S ended `S_PROXY_ONLY`. The per-version point conditions held, but the pooled 2-SE recovery condition at the S2 selection
  failed (-0.0028 + 2 × 0.0019 > 0), and so did the task alternative (spend -0.0006 / -0.0007 against the required -0.001).
- Route E's advance rests on the witness law, and route S supersedes it (RULES 5.2).
- Route D found no material defect.
- **The coordinator's prediction.** It was `S_PROXY_ONLY`, a probability of advancing of about 0.15, and a nested pair AUC
  difference against the bracket within ±0.003.
  - It was committed at 03:24:24Z (7600d6ae2), before any nested result existed.
  - It was written after the source-stage results, which already showed S-a and the fitted pair-information change of at most
    0.0002 nats that it cites, and after the two nested units had started (03:23:37Z / 03:23:49Z).
  - Its ±0.003 range held in version 0 (-0.0018) and missed in version 1 (-0.0038).

**6. Method candidate, component, engineering result or bounded negative result?**
- Rule outcome `OPTIMIZER_REPAIR_CLOSED`; no pilot outcome class was computed. METHOD_CANDIDATE, COMPONENT, ENGINEERING_RESULT and
  BOUNDED_NEGATIVE_RESULT are pilot classes, and no pilot ran.
- Descriptively, the optimizer-repair hypothesis closes on MODEL_FIT, with the bound stated (pooled pair LOOKUP change -0.0028, 2-SE
  interval -0.0065 to +0.0009, the objective's own LOOKUP reader only).
- The verified engineering findings listed under "Decision" are not the rules' `ENGINEERING_RESULT` class.
- There is no method candidate and no component.

**7. What remains unknown?**
- Nothing is known about the MODEL_SELECT readers. The reader-validity and pilot rules (RULES sections 6-7) were never exercised, no
  grid was built, and MODEL_SELECT was not read.
- Question C, whether propensity adds anything beyond the equally optimized HQ, stays unanswered.
- Whether a selection rule with the utility limits inside it would turn better optimization into usefulness. The exact laws show
  that the unconstrained objective's optimum can be infeasible.
- Other solver changes. SV2 (for example a guard-consistent codeword refresh, restarts, or exact single-person moves with rule
  refit) was not built. SV1 is path-dependent and not optimal, so better optima of the same objective exist.
- Exact references beyond 2 × 3 cells and 4-8 support states.
- The S3 budget region and the prospective capacities.
- Whether any finite-cell routing can close the 0.018-0.023 pair-AUC gap to the target references reported by the previous campaign.

## What was done

| Step | Evidence | Result |
|---|---|---|
| P1 replay | The previous campaign's nested P1 estimator rerun on the recorded inputs. `nrm/fa/prc.py` is byte-identical. The `frc/fa/route.py` functions are imported unchanged from the pinned 933d0d6f, which differs from the file that ran P1 (b538fa7a4) only in `emit`'s dispatch through `route_any`, a change that is behaviour-preserving for P1 books. The selection loop is re-implemented with three renamed local variables (AST-identical after renaming, verifier). | All 809 recorded values per version are equal (difference 0.0), including the 413 nested means, the selection, the brackets and F4 (0.00715 / 0.01113, se 0.00225 / 0.00231). `P1_REPLAY_VERIFIED` under the reviewer's ruling (strict text: `P1_ESTIMATOR_CHANGED`), re-derived with the frozen `p1_replay` from FRC1 `SELECTION_V<v>.json`, `replay/P1_REPLAY_V<v>.json` and `roles/verifier/P1_VERIFIER_V<v>.json`. From the per-person outputs, the verifier recomputed spends, SEs, brackets, LOOKUP AUCs, F4, the selection and 275 nested mean values per version (within 2.2e-19). It could not recompute the `nested_means` keys that need training-side partitions (`VERIFICATION.md`). |
| Objective audit | `audit/OBJECTIVE_AUDIT_V<v>.json` | No defect. The four approximations are quantified. Expressivity: 17,744 support states, and 1.0-1.3% of people share task scores with another state. |
| Exact references | 5 registered laws (null, collisions, no collisions, witness, small-sample) and 60 registered cells (30 distinct); a narrow tilt-breakpoint reference; enumeration relaxations of 1,296 to 1,679,616 partitions | The code reproduces exactly. The seven-tilt grid misses windows in D and E. SV1 finds the missed windows and is never worse, but it stays above the best same-class reference in 10 of 30 distinct cells (largest 0.0102). |
| Solver comparison | Cached full-pool statistics: original, SV1 and SV1_first for PRC and HQ at 6 strengths | SV1 lowers PRC's objective at every strength, and HQ's in 11 of 12 cells. SV1_first, the same candidates with first improvement, stopped on the 30-step path cap, still improving, in 23 of 24 strength chains (all 12 PRC, 11 of 12 HQ). It used 436.5-510.8 CPU-s per chain against 24.5-32.9 for SV1, and 13-18× SV1's Lloyd steps, so it matches the candidate set per step, not the effort per chain. It ended above the original and above SV1 in every cell. That attribution is confounded by the cap, and it does not show that finer resolution cannot help. On the exact laws SV1_first equals SV1. |
| Nested comparison | Both solvers in the same 5 outer folds and teacher fits; the original entries equal the replay (117 arrays, 0.0) | Route S input; `S_PROXY_ONLY` |

Tables: `tables/OBJECTIVE_AUDIT_TABLES.md`, `tables/EXACT_REFERENCE_TABLES.md` and `tables/SOLVER_COMPARISON_TABLES.md`, generated by
`fsa.report` from the records. Narratives: `OBJECTIVE_AUDIT.md`, `EXACT_REFERENCE_CASES.md` and `SOLVER_COMPARISONS.md`.

## Limits of this evidence

- **Multiplicity.** No family-wise error control is claimed. Every registered law cell, strength and variant is reported.
- **Adaptivity.**
  - SV1 was designed knowing the previous campaign's saturation. It was revised once (card FA_FSA_v1 to v2) after an invented
    trace, before any real-data or registered-law run.
  - The prediction was written after the source results (question 5).
- **Surrogate.** The fitted objective is a log-likelihood surrogate of a LOOKUP reader on label-derived training statistics. A lower
  objective is not protection.
- **One reader type.** All nested evidence uses the objective's own LOOKUP reader on label-derived MODEL_FIT statistics.
- **The decision rests on the pooled condition.** The pooled 2-SE condition failed while the per-version point conditions were met.
- **Uncertainty.** Household-clustered SEs and the LOOKUP bootstrap treat refitted pipelines as fixed, which understates
  uncertainty. The two versions share training rows, so they are procedure repeats, not replications.
- **Exact laws.**
  - They are small and team-written.
  - Route E's advance rests on the single witness law D, built to contain a feasible gain. It shows that SV1 can find such a gain
    when one exists, not that such gains are common.
  - Seed 1 duplicates seed 0.
  - The exact Bayes AUC is reader-optimal only within functions of the release.
- **SV1 is short of the best known point.** It is above the best same-class reference in 10 of 30 distinct exact cells. On the
  source a lower-objective deployable partition exists at version 1, strength 8.
- **HQ was not uniformly improved** (version 0, strength 8).
- **The equal-effort control ran under a cap.** SV1_first ran into the 30-step cap in 23 of 24 chains, so the resolution-versus-
  selection attribution is confounded.

## Rulings and disclosures

- **RULES 3.1, laws run before registration.**
  - The statistical reviewer adopted the coordinator's reading as the intended meaning, not a rule change. The exclusion covers laws
    that a solver ran on before the law's registration commit; runs after registration are the route-E evaluation itself.
  - The records give this order: laws committed at 02:42:16Z (c27a91477), the SV1 card at 02:46:39Z (91ba9a27e), and the first
    registered-law run at 02:49:12Z (NARROW_V1). RESULTS_V1, ENUM_V1 and STEP5_V1 carry no run times. The queue ledger gives
    02:49:13Z for ENUM_V1, 02:54:59Z for the summary and 03:05:44Z for STEP5.
  - Two disclosures go with the ruling. The laws were committed before the rules were frozen (02:59:51Z): the rules did not shape
    the laws, and the reviewer did not see them. Law D is a constructed witness (see "Limits").
- **P1 estimator identity (CLOSING_STATEMENT addendum 1).**
  - The reviewer ruled that `estimator_unchanged = True` stands. The label is `P1_REPLAY_VERIFIED`, a disclosed reading made after
    the result; the strict text of RULES 8 gives `P1_ESTIMATOR_CHANGED`.
  - The reviewer's reasons:
    - a consistent renaming of local variables cannot change the function;
    - the per-person outputs do not pass through the loop;
    - all values reproduce bitwise;
    - no decision reads this label.
  - The reviewer records an erratum to RULES 8. Its pin named 933d0d6f, not the b538fa7a4 file that produced the record. The one
    change on the P1 path is behaviour-preserving for every book that is not a UNT book, and P1 has none.
- **The equal-effort record.**
  - RULES 9 lists "the SV1_first control records" for the advancement record. ADVANCEMENT.json instead carries
    `equal_effort_attribution: NOT_RECORDED` and no controls.
  - The attribution was computed separately (`decisions/EQUAL_EFFORT_ATTRIBUTION.json`). SV1_first's route E input exists
    (`roles/exact_reference/route_e_input_SV1_first.json`) but was not passed as a control.
  - No label changes, because controls never advance (RULES 5.3). See `PROTOCOL_DEVIATIONS.md`.
- **Uncommitted files under hashed code paths at launch.** The queue ledger records them for 17 units. No stage imported them. See
  `PROTOCOL_DEVIATIONS.md`.

Related records: `START_HERE.md`, `ADVISOR_BRIEF.md`, `VERIFICATION.md`, `PROTOCOL.md` (sections 8 and 13),
`roles/statistical_reviewer/RULES.md` and `CLOSING_STATEMENT.md`, `roles/engineering_review/REVIEW_CLAIMS.md`,
`PROTOCOL_DEVIATIONS.md`, `roles/coordinator/COORDINATOR_LOG.md`.
