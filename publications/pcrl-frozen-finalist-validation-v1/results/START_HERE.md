# Start here: PCRL frozen-finalist validation and confirmation-readiness study

**What this is.** A bounded validation of an existing, frozen candidate, run on 2026-10-17 from 21:03:32Z. The limits were 6 elapsed hours, 10 CPU-hours, 2 heavy jobs and no paid or cloud compute; actual use is in `RESOURCE_ACCOUNTING.json`. Nothing in the candidate was refitted, retuned or reselected.

**Data.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, in the existing development roles of the protected-tree replication.
- Cohort: housing-unit residents aged 25–64, civilian employed, WKHP 1–99.
- No new population, download or label was opened. The sealed assessment stayed closed.

**The service.** From AGEP, WKHP, COW, SCHL, MAR and POBP it predicts:
- an income class (2 classes: PINCP × ADJINC / 10^6 above $50,000);
- an occupation group (5 classes).

Each recipient receives its own float64 probability vector and argmax decision.

**What is measured.** SEX is the dataset's recorded demographic column. Evaluation readers try to infer it from each released output alone and from the two outputs aligned to the same record (the linked pair). Reader recovery is ROC AUC for SEX. It is not a privacy guarantee.

**The candidate.** `B3CTL_UNCOUPLED_MIX50_C6A` from the source campaign: branch `research/pcrl-autonomous-method-development-v1` at `9e2b6c567`, publication `5901f7c73`.
- **Occupation output:** the stored JOINT_TREE output, bitwise identical to H_LJ's.
- **Income output:** 6 decision-aligned levels of the average of the stored LOCAL_TREE and JOINT_TREE income log-odds, with CALIBRATION-fitted per-level probabilities.

## Results (one logged follow-up DEV_REPORT evaluation, 22:27:16Z; seed 0 / seed 1)

**Status of these results.** Development robustness evidence on already-used records.
- DEV_REPORT truth had been read 10 times across the lineage before this campaign: 5 evaluations and 5 replays.
- This campaign adds one logged coordinator re-read (22:27:16Z) and one separately logged verifier replay (22:50:37Z).
- It is not confirmation, and it does not erase the source's adaptive selection history.
- Seed 0 and seed 1 are two fits of each release evaluated on the same people, not replications.

| Question | Answer |
|---|---|
| Does the source result reproduce? | **Yes, exactly.** All 70 endpoints and every claim label; maximum point and SE difference 0.0. The source winner refits are bitwise identical (8/8), and so are the source-definition controls. All 7 existing packages redeploy bitwise as full-role batches, including from the source's same-device backup alone. No NEURAL_RAW seed-1 package exists (a coverage gap; that output is admitted by hash). |
| Does the pair benefit survive a larger, access-matched reader bank? | **The pair benefits do, against these frozen fitted readers; the registered claim-B conjunction does not.** With household-bootstrap-trained readers the benefit shrank in 3 of 4 cells. ENHANCED benefits: +0.027 / +0.022 over NEURAL_RAW and +0.028 / +0.026 over H_LJ, every clause bound above 0, none above 0.020. Claim B fails on the seed-1 income recipient clause against H_LJ (point +0.0076, bound 0.0235 > 0.010). The cause is H_LJ's new income winner, chosen on an ATTACK_VALIDATION near tie, reading 0.011 lower on DEV. The frozen label is `READER_SENSITIVE` (side `COMPARATOR_READ_LOWER`: selection instability, not added finalist exposure). |
| Utility, subgroup and recipient requirements? | All hold at the point in both scorecards. Simultaneous confidence is unresolved (25 of 28). The binding income Brier clauses are 0.03–0.04 SE below the cap. |
| 0.020 material target? | Not established under either reader bank. |
| Can a fresh study confirm the release? | **No realistic scenario found.** The required size is not identified, because the binding income utility slack sits 0.03–0.11 SE from its caps. At the plug-in it is about 4.6 million households (source scorecard) or 60 million (confirmatory) for 80% joint power; it is never reached if the true slack is ≤ 0. Whatever the utility headroom, the ENHANCED seed-1 income recipient clause against H_LJ alone needs about 318,000 households. At the about 17,500 households that could test the release, D1 power is about 0.01 (ENHANCED; about 0.25 ORIGINAL). Assurance is at most about 0.10–0.16 at every size. |
| Is the occupation output one-to-one in the inputs? | **No.** It is many-to-one over the valid domain. The income code is not a function of the occupation vector there (constructive counterexamples). Within each observed role the relation holds empirically; across roles on seed 1 it fails twice. |
| Which populations could test the frozen release? | Only Florida 2019 people never used by any model or reader: the sealed 20% (9,853 households) and the capped-out households (7,657). Both carry a historical in-memory extraction disclosure. Everything else is a transport. None was opened. |

**Recommendation:** retire the candidate as a competitive candidate, and retain the evidence and the runnable artifact (`RESEARCH_DECISION.md`).

## Read in this order

1. `RESEARCH_DECISION.md`: the five distinctions and the recommendation.
2. Registration and locks:
   - `PROTOCOL.md`, registered before any reader fit (receipt `55b017918`, pushed 22:21Z);
   - `roles/statistical_reviewer/PROTOCOL_REVIEW.md`;
   - `READER_BANK_LOCK.json`;
   - `REPORTING_LOCK.json` (receipt `8993a4bdd`, pushed 22:27:08Z, before the DEV_REPORT re-read at 22:27:16Z).
3. Source admission: `FROZEN_CANDIDATE_MANIFEST.json` and `SOURCE_ADMISSION.json`.
4. Exposure: `EXPOSURE_AND_SELECTION_AUDIT.md`, which includes the household-overlap check (0 shared households between any two admitted roles).
5. Output mapping: `OUTPUT_MAPPING_CHECK.md` and `OUTPUT_MAPPING_RESULTS.json`.
6. Results:
   - `report/DEV_REPORT_RESULTS.json`;
   - `ORIGINAL_VS_ENHANCED_READERS.csv`;
   - `PRIMARY_ENDPOINTS.csv`, both scorecards;
   - `READER_MATCHED_BENEFITS.csv`, descriptive;
   - `FORECAST_SCORES.json`.
7. Feasibility:
   - `CONFIRMATION_FEASIBILITY.md` and `CONFIRMATION_FEASIBILITY_RESULTS_V2.json`;
   - V1, `CONFIRMATION_FEASIBILITY_RESULTS.json`, is kept; the correction is listed in `PROTOCOL_DEVIATIONS.md` item 9;
   - `roles/statistical_reviewer/FEASIBILITY_REVIEW.md`.
8. Populations: `POPULATION_INVENTORY.md`.
9. Verification and engineering:
   - `INDEPENDENT_VERIFICATION.json` and `roles/verifier/`;
   - `DEPLOYMENT_CHECKS.json` and `BACKUP_DEPLOYMENT_CHECK.json`;
   - `QUICKSTART.md`.
10. Custody and closeout: `BACKUP_VERIFICATION.json`, `RESOURCE_ACCOUNTING.json`, `ACCESS_RECEIPT.json`, `HANDOFF.json` and `roles/claims_reviewer/`.

## Roles

| Responsibility | Done by |
|---|---|
| Coordination and implementation | the main session |
| Output-mapping and exposure review, including the population inventory | a separate agent (`roles/exposure_reviewer/`) |
| Statistical and readiness review: the protocol before fitting, and feasibility | a separate agent (`roles/statistical_reviewer/`) |
| Independent numerical verification, with one separately logged DEV_REPORT replay | a separate agent (`roles/verifier/`) |
| Deployment and claims review | a separate agent (`roles/claims_reviewer/`) |

**Platform event.** A platform safety classifier stopped the coordinator's first response to this brief before any tool call ran. The campaign restarted from state inspection. No task description or variable was renamed or obscured.
