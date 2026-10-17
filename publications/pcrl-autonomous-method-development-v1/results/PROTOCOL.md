# PCRL autonomous method-development campaign: registration of boundaries, data access and acceptance rules

**Status.** Registered before any new fit in this campaign (the commit that adds this file is the registration). This is an exploratory development campaign on previously used Florida ACS 2019 development records. It is not confirmation. Mechanism designs and their revisions are exploratory decisions made during the campaign inside the boundaries below; they are recorded as they occur in `MECHANISM_CARDS.md` and `EXPERIMENT_LEDGER.csv` and were not individually pre-registered.

**Disclosure.** The design of this campaign was informed by previously published results on these same records, in particular the SCM utility-matched follow-up (public `8cb70b290`), the SCM engineering benchmark (`6fc06021d`), the frozen-hybrids follow-up (`3a1f509`) and the protected-tree replication (`f4ea199`). DEV_REPORT is not historically unseen. MODEL_SELECT has been used for selection by earlier campaigns.

## 1. Census context and service (unchanged)

- Public-use Florida ACS 2019 person records. Household grouping, eligibility and role assignment are those of the protected-tree replication.
- Tasks: income class (2 classes) and occupation group (5 classes), with the replication's definitions.
- Permitted runtime inputs: the six raw fields AGEP, WKHP, COW, SCHL, MAR, POBP, with the admitted input encoding (145 columns).
- Service contract: each recipient receives only its own float64 probability vector and argmax decision. The evaluation considers each output alone and the two outputs aligned to the same record (the linked pair).
- SEX is the dataset's recorded demographic column. Readers are separate evaluation classifiers that infer SEX from the released outputs.
- References U0 and Ucal: the replication's neural task model and its CALIBRATION-temperature version (frozen outputs in the frozen-hybrids bank).

A method may change how releases are learned and computed, train new task models within MODEL_FIT, change its internal representation or grouping, and revise its optimization. It must compute releases from the permitted runtime inputs only. Training-time labels (including SEX) and internal quantities are never recipient outputs.

## 2. Boundaries and resources

| Item | Limit |
|---|---|
| Elapsed time | 6 hours from 2026-10-17T16:45:41Z (first action); hard stop 22:45:41Z |
| CPU | 10 CPU-hours in total, counting failed and stopped units, verification, reader fits and agent work (upper estimates) |
| Concurrency | at most 2 heavy numerical jobs at once (two queue slots), one numerical thread per job, through the single shared queue `amd.jobqueue` |
| New configurations | at most 36 new seed-specific defense or task-model configurations, including newly fitted controls and ablations; counted in `EXPERIMENT_LEDGER.csv` (`counts_toward_36`) |
| Reader fits | separate units; they count toward CPU and elapsed time but not toward the 36 |
| Compute | no paid or cloud compute; no new population |
| Families | at most 3 materially different mechanism families; each with an initial implementation and at most 2 evidence-driven revisions |

Branch `research/pcrl-autonomous-method-development-v1` in its own worktree, based on `482808212`. Main, manuscript branches, historical locks, protocols and results are not modified. Person-level arrays, fitted models, reader artifacts, ledgers of private runs and private paths stay under `.private/` and are never published.

## 3. Data roles and access

| Role | Fields | Use in this campaign |
|---|---|---|
| MODEL_FIT (20,000) | features, raw inputs, income, occupation, SEX | fitting release mechanisms and task models; household-separated internal folds for development diagnosis |
| CALIBRATION (2,982) | features, income, occupation | calibrators only |
| MODEL_SELECT (8,000) | features, income, occupation, SEX | development evaluation, reader proxies, adaptive selection and revision feedback. Every label read is logged (`MODEL_SELECT_USE.jsonl`). Never used as gradient-fitting labels or to fit release parameters |
| ATTACK_FIT (8,001), ATTACK_VALIDATION (3,000) | features, SEX | final readers only, after a finalist freeze; never used to train a release mechanism |
| DEV_REPORT (5,943 people, 3,955 households) | features, raw inputs (label-free) | deployment parity. Truth only through `amd.common.dev_report_truth`, after this campaign's committed and published REPORTING_LOCK, for one logged coordinator evaluation and one logged independent-verifier replay of a frozen finalist. Not read during exploration. If no finalist is promoted, DEV_REPORT labels are not read |
| Sealed assessment | none | closed; every loader refuses it. Other projects' reserved records are never read |

Internal cross-fitting within MODEL_FIT and new splits of used people are development diagnostics, never fresh confirmation.

## 4. Acceptance requirements (fixed; never relaxed)

- **U (utility, 24 clauses).** For both tasks, both references (U0, Ucal) and both seeds: accuracy loss ≤ 0.010, NLL excess ≤ 0.010 nats, summed Brier excess ≤ 0.005.
- **G (subgroup guard).** Within-SEX occupation summed-Brier excess against U0 ≤ 0.005 for each SEX group and seed.
- **R (recipient non-worsening, binding source margin).** For each eligible reference (NEURAL_RAW, H_LJ) and seed: the candidate's single-output recovery minus the reference's ≤ 0.010 AUC, for the income output and for the occupation output.
- **P (pair benefit).** Reference pair AUC minus candidate pair AUC > 0. The original 0.020 material-improvement target is reported separately: a positive measured improvement is distinguished from an improvement established beyond 0.020.
- Point feasibility is distinguished from simultaneous confidence. An unresolved interval is not a demonstrated violation, and a point pass is not established population preservation.
- The target column (SEX), the recipient outputs and these limits are not changed to manufacture eligibility. No equivalence tolerance is introduced or widened after outcomes are seen; the 0.001-nat matching requirement of the previous campaign is not used as a success gate.

## 5. Development rules

- Each family starts with a mechanism card (`MECHANISM_CARDS.md`): the bottleneck addressed, the operation, why utility and subgroups might survive, the strongest simple control, the revise/abandon observation and the closest prior method.
- Staging per family: software and real-update checks (invented data where useful; a toy result is not evidence on ACS); a first-seed development fit; the full MODEL_SELECT utility vector and subgroup burdens; household-separated MODEL_SELECT reader proxies on the actual transmitted outputs; diagnosis; a justified revision or the next family; promising configurations then on the second seed.
- Development readers are the household-split MODEL_SELECT proxies of `amd.evaluate` (OUT_LR, OUT_HGB, LOG_MLP on each view; exact-wire LOOKUP on the pair). A candidate is never selected because its own training critic became weak. Collapse, numerical scale, aliases and output changes are inspected.
- Every fit, evaluation and decision is a ledger row with family, version, code hash, hypothesis, change, roles read, training and selection effort, utility, subgroup and proxy outcomes, resources and the decision.

## 6. Finalist promotion (MODEL_SELECT and the development record only)

1. **Eligible:** U and G hold at the point on MODEL_SELECT for both seeds, and R is screened on MODEL_SELECT proxies (each single-output proxy ≤ the reference's MODEL_SELECT proxy + 0.010, for both references and seeds).
2. **Promotable:** eligible, and the MODEL_SELECT pair proxy is below both references' MODEL_SELECT pair proxies on both seeds (a positive development improvement on both seeds; one-seed gains do not qualify).
3. **Choice:** among promotable configurations, prefer those whose U and G values are all ≤ 0.8 × their caps on MODEL_SELECT (robustness margin against role drift; a promotion without it is labelled `EDGE_SELECTED`); then the lowest mean MODEL_SELECT pair proxy over the two seeds; ties to the simpler mechanism.
4. At most one finalist. If none is promotable, no finalist is frozen, DEV_REPORT labels are not read, and the campaign delivers the development results, runnable diagnostics and the exact reasons.

## 7. Final evaluation (only for a promoted finalist)

- Frozen before any final reader fit: the algorithm and both seed-specific packages; the selection rule and all trials considered; the strongest eligible controls (NEURAL_RAW, H_LJ); the matched task-only control with the same input access, task-training exposure, capacity and optimization effort; the published IBSI objective as adapted in the SCM follow-up; any matched component control that supports an intended claim; the exact output contract; the reader bank, selection rules and reporting code; the endpoint roster, signs, margins and multiplicity (frozen by the statistical reviewer).
- A compact bank of at most 8 effective releases per seed, exact aliases deduplicated.
- Fresh final readers on ATTACK_FIT and selected and oriented on ATTACK_VALIDATION under the source's common budget and slate (8 readers, including the corrected LOOKUP reader), with single-output alternatives for pair readers; real-data planted-SEX and shuffled-label controls; a frozen independent refit subset.
- A sanitized reporting receipt committed and pushed before one logged coordinator evaluation of DEV_REPORT, followed by one separately logged independent-verifier replay. After that evaluation the campaign does not return to exploration; a final failure stays a failure.
- Inference: household bootstrap (B = 2048) with Wald intervals, conditional on the fitted releases and readers. Intervals do not cover the adaptive search history. The two seeds are the same people.

## 8. Roles

Coordinator and integration owner (main session); up to three mechanism investigators (separate agents, run in waves); baseline and statistical reviewer; independent verifier with separate code; engineering and claims reviewer. Actual assignments are recorded in `roles/` and `HANDOFF.json`.

## 9. Claims

Before a finalist is selected, the reviewer writes an explicit conjunction for every intended claim. Novelty, mechanism attribution, joint-design advantage and overall competitiveness are separate conclusions. Reader results are evidence against the declared readers and the auxiliary sample budget only; they are not guarantees against every reader. Deterministic outputs are not assumed one-to-one, an unresolved contrast is not equivalence, and the lookup reader is not population-optimal.
