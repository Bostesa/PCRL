# Start here: quality-matched comparison of grouped prediction tables (SCM utility-matched follow-up)

**What this is.** A separate, exploratory follow-up to the SCM engineering benchmark (research tip `2738cc30b`, public package `6fc06021d`). Its earlier results and labels are unchanged.
- **Question.** Does Shared-Cell Merging (SCM) lower fresh-reader recovery of the recorded SEX column relative to ordinary task-only grouping, when both spend similar held-out occupation prediction loss? Ordinary grouping here is a task-only merge objective that still uses SEX through the shared within-SEX guard; its comparator was selected with the SEX-labelled MODEL_SELECT reader proxy.
- **Two further questions.** Is the answer different at the same cell count? How does the selected release compare with the eligible frozen neural (NEURAL_RAW) and tree (H_LJ) references?
- **Records.** Florida ACS 2019 development records that earlier work has already used. The sealed assessment stayed closed.
- **What it compares.** Existing methods only: SCM, ordinary merging, the published IBSI objective (adapted) and shuffled-signal controls.
- **Claims not made.** No novelty, new-algorithm or joint-design claim. The source's published results informed this design; the new registration fixes execution order only.

## Bottom line

**The fixed comparison closed as inconclusive at the registered tolerance: `COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE`** (from claim M = `MATCH_DID_NOT_CARRY_OVER`). Its four mandatory qualifiers:
- DEV utility confidence: `CONFIDENCE_UNRESOLVED` (3 PASS, 21 UNRESOLVED, 0 violations).
- Claim C: `SUBGROUP_UNRESOLVED`.
- ORD_NLLM match on DEV_REPORT:
  - seed 0: `POINT_MISMATCH` and `MISMATCH_ESTABLISHED`;
  - seed 1: `POINT_MISMATCH` and `EQUIVALENCE_UNRESOLVED`;
  - direction `SCM_SPENT_MORE` on both seeds.
- Claim B: `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`.

No release is frozen as an engineering reference.

**Scope of every number below.**
- Pairs are seed 0 / seed 1 on DEV_REPORT unless marked MODEL_SELECT.
- The two seeds evaluate the same 5,943 people in 3,955 households. They are not separate populations.
- Intervals are conditional on the fitted releases and readers. They exclude the development and selection history on these records, and they are not ACS survey-design variance.
- The service is a controlled benchmark. Its readers rely on three explicit assumptions:
  - SEX-labelled auxiliary records (ATTACK_FIT, 8,001 people);
  - accurate linkage of the two outputs;
  - distributional similarity between the auxiliary and the released people.

1. **Conservative SCM endpoint.** SCM at threshold 0.008 (298 / 300 cells) kept all 24 original utility point clauses on DEV_REPORT. Unlike the historical nominee, it did not miss the occupation NLL clause. It did not keep the men's within-SEX occupation Brier guard at the point: 0.0054 / 0.0064 against 0.005. That is UNRESOLVED, not established harm.
2. **MODEL_SELECT-matched ordinary control (ORD_NLLM; the match did not carry over).**
   - On MODEL_SELECT, a strongest-proxy ordinary control was within 0.0002 nats of SCM's held-out occupation NLL on both seeds (SCM − control: −0.00003 / +0.00015).
   - On DEV_REPORT, SCM's linked-pair AUC read 0.034 / 0.025 below that control (clause bounds 0.0225 / 0.0144 > 0), and all four recipient non-worsening clauses pass. This is the comparison-only status `BENEFIT_AND_NON_WORSENING_CLAUSES_PASS (not cost-matched on DEV_REPORT)`, not a matched result.
   - **The match did not carry over.** On DEV_REPORT, SCM spent +0.0070 / +0.0017 nats more occupation NLL. The direction sub-status is `SCM_SPENT_MORE` on both seeds; it states only who spent more. Seed 0's mismatch is established (`MISMATCH_ESTABLISHED`) and seed 1's is `EQUIVALENCE_UNRESOLVED`. Because SCM spent more, its lower pair AUC is not separated from the extra loss.
   - SCM also spent more income NLL: +0.0064 / +0.0035, ABOVE_ZERO on seed 0 and UNRESOLVED on seed 1.
3. **Same capacity.** Claim K is `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD` (men's guard, POINT_FAILURE).
   - The benefit over the same-count ordinary control is 0.036 / 0.028 (clause bounds 0.024 / 0.017).
   - But SCM spent +0.010 / +0.006 nats more occupation NLL (above zero on both seeds) and +0.007 / +0.006 more income NLL.
   - The income non-worsening clauses are UNRESOLVED (comparison-only status `COMPARISON_CLAUSES_NOT_ALL_PASS`).
4. **The earlier difference.** This item is descriptive: the components are `NOT_TESTED_M_NOT_ESTABLISHED`.
   - Ordinary grouping barely changed recovery at the losses measured (MODEL_SELECT proxies up to 0.0145 / 0.0125 nats; fresh DEV_REPORT readers only up to 0.0043 / 0.0054 nats, below SCM's DEV spend).
   - The published IBSI objective (PUB_IBSI, γ = 2.0), selected at a MODEL_SELECT match, read within about 0.01 AUC of SCM on DEV_REPORT. SCM's pair benefit was +0.006 / −0.010 (positive means SCM read lower), UNRESOLVED on both seeds. This is not equivalence.
   - PUB_IBSI's match also drifted (+0.0011 / −0.0039 nats), and on seed 1 it fails three DEV utility point clauses.
   - Its finer partition (median 13 ATTACK_FIT people per vector) can be under-read even more than SCM's.
   - SCM's specific ordering showed no measured edge over the published objective.
   - The SEX-guided releases (SCM, PUB_IBSI) read lower than the ordinary and shuffled controls, but they spent more DEV occupation and income loss, so the lower recovery is not separated from that extra loss.
5. **References.** Not competitive (claim B `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`).
   - Pair benefits against NEURAL_RAW (+0.017 / +0.005) and H_LJ (+0.018 / +0.010) are UNRESOLVED on the family interval except H_LJ seed 0 (ESTABLISHED). All four are below the 0.020 target at the point (POINT_SHORTFALL).
   - SCM's income recipient reads higher than both references (POINT_EXCESS on three of four clauses, UNRESOLVED on H_LJ seed 0).
   - Its occupation recipient reads higher than H_LJ's (ESTABLISHED_VIOLATION on seed 0, POINT_EXCESS on seed 1).
6. **Measured limitation.** A ±0.001-nat occupation-NLL match cannot be certified on samples of this size.
   - Paired DEV standard errors are 0.0016–0.0020, so a MODEL_SELECT match selects partly on noise. Both sides moved:
   - The MODEL_SELECT-matched ordinary control's held-out spend went from 0.0083 / 0.0083 on MODEL_SELECT to 0.0043 / 0.0054 on DEV_REPORT. Over the same step, SCM's went from 0.0082 / 0.0084 to 0.0113 / 0.0072.
   - No MODEL_SELECT-matched comparator (ordinary, PUB_IBSI or shuffled) carried over to DEV_REPORT.
7. **Reading the recovery numbers.**
   - Every AUC above comes from a fixed 8,001-person reader budget.
   - SCM's finer partition can be under-read, which favours SCM. Its pair vectors hold a median of 16 / 17 ATTACK_FIT people, against 53 / 64 for the MODEL_SELECT-matched ordinary control.
   - PUB_IBSI's partition is finer still (median 13).
   - For every grouped release, the occupation output identifies the shared cell and income is a function of it. The linked pair therefore carries no information beyond the occupation output for these people, and differences between pair readers and occupation-only readers are differences between fitted readers.
   - The lookup reader is not population-optimal, and no AUC here is a privacy guarantee.

Details, with every label as an explicit conjunction and every failure category, are in `RESEARCH_DECISION.md`.

## Read in this order

1. `RESEARCH_DECISION.md`: the answers, the measured limitation, and what is established, a point result, unresolved, unavailable or diagnostic.
2. `SELECTION_AND_MATCHING.md`: the nomination grid, every role, the matching coverage and the selection effort (MODEL_SELECT only).
3. `UTILITY_RECOVERY_FRONTIER.png`. The top row shows MODEL_SELECT cross-fitted proxies for every registered checkpoint; the bottom row shows fresh DEV_REPORT readers for the final bank only. These are two measurements, not one frontier. Lines are illustrations only.
4. `PRIMARY_ENDPOINTS.csv` (58 keys), `report/SECONDARY_ENDPOINTS.csv` (24 keys) and `report/DEV_REPORT_RESULTS.json`. The JSON holds all labels and condition vectors, the separate status fields, the descriptive companions and the diagnostics.
5. `UTILITY_TABLE.csv` and `RECOVERY_TABLE.csv`:
   - the full utility vector of every effective release on MODEL_SELECT and of the final bank on DEV_REPORT;
   - recovery for the final bank.
6. Registration:
   - `PROTOCOL.md`, `UNIT_INVENTORY.json`, `ANALYSIS_LOCK.json`;
   - the pre-fit reviews and their disposition in `roles/`; the statistical reviewer's `FROZEN DESIGN` block is authoritative.
7. Stage locks, in order: `SOURCE_ADMISSION.json` → `ANALYSIS_LOCK.json` → `TRAJECTORY_COMPLETE_LOCK.json` (with `TRAJECTORY_SUMMARY.json`) → `SELECTION_LOCK.json` → `SERVICE_BANK_MANIFEST.json` → `REPORTING_LOCK.json` (with `READERS_SUMMARY.json`).
8. `INDEPENDENT_VERIFICATION.json` and `roles/verifier/VERIFICATION.md` (8/8 PASS; discrepancies D1–D10), and `PROTOCOL_POSTRUN_CLARIFICATIONS.md` (their resolution; the locked files keep their bytes).
9. `DEPLOYMENT_RESULTS.json`, `QUICKSTART.md`, `BACKUP_VERIFICATION.json`, `RESOURCE_ACCOUNTING.json`, `HANDOFF.json`, and `MANUSCRIPT_CLAIMS_HANDOFF.md`.

## How it ran

**Admission.**
- The source tip and the public commit were verified, and 12 manifests checked.
- 51 artifacts were copied by hash: the teachers, out-of-fold predictions, the 512-cell partition, κ = 20 shrinkage, routing and frozen comparators. Nothing was retrained.
- The admitted routing reproduces all 31 historical arm outputs bitwise.

**Trajectories.** Ten fixed trajectories per seed: SCM β = 1.0; ordinary (source twin rule, continuing to 60 / 58 cells); PUB_IBSI γ 0.5, 1.0 and 2.0 to 32 cells; five shuffled.
- They passed the historical reproduction gate per seed: 51 checks plus the summary flag. Every source endpoint reproduced bitwise.
- They gave 272 / 261 effective releases at the registered checkpoints.

**Selection and matching.** MODEL_SELECT only.
- One common SCM threshold.
- Matching tolerance 0.001 nats per seed.
- Strongest-proxy comparators.
- PUB_IBSI and shuffled controls: one common γ / r.

**Final bank.** Ten distinct services per seed.

**Readers.**
- The source 8-reader slate on ATTACK_FIT (8,001) and ATTACK_VALIDATION (3,000): 246 fits per seed.
- The label-free replay of the historical SCM, C0, NEURAL_RAW and H_LJ matches the source bitwise (32 / 32 per seed).

**DEV_REPORT.**
- One logged coordinator evaluation (05:30:14Z), after the committed and published reporting lock.
- One separately logged verifier replay (`DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER`, 05:54:14Z). The independent verifier passed all eight checks with its own code: admission and roles, trajectories, MODEL_SELECT selection, matching, DEV_REPORT endpoints and labels, deployment, reader refits and access. Every recomputed number equals the record bitwise. Its ten discrepancies are procedural or wording and change no release, endpoint or label; they are resolved in `PROTOCOL_POSTRUN_CLARIFICATIONS.md`.
- Selection and report code were first dry-run on invented labels.

**Receipts.** Both are on `publication/pcrl-scm-utility-matched-v1`; push times are from the coordinator's push log, recorded in `HANDOFF.json`.
- Pre-fit receipt `738dcbbbe`: committed 05:19:18Z, pushed 05:19:22Z, before any fit.
- Reporting receipt `3c7d69317`: committed 05:30:02Z, pushed 05:30:06Z, before the DEV_REPORT re-read.

## Roles (who did what)

| Responsibility | Done by |
|---|---|
| Implementation and coordination | the main session (coordinator) |
| Baseline and mathematical review (pre-fit) | a separate reviewer agent (`roles/math_reviewer/`) |
| Statistical-design review (froze signs, counts, α) | a separate reviewer agent (`roles/statistical_reviewer/`) |
| Independent numerical verification | a separate verifier agent with its own code (`roles/verifier/`) |
| Deployment, reporting and closeout review | a separate claims reviewer agent (`roles/claims_reviewer/`); deployment and custody runs by the coordinator |
