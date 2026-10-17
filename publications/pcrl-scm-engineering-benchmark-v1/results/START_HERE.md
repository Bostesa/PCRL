# Start here: SCM engineering benchmark

**What this is.** A bounded engineering benchmark of Shared-Cell Merging (SCM), an application of established merging methods (the privacy funnel, agglomerative information bottleneck, IBSI). **No novelty claim is made.** It follows the stopped structural campaign (`research/pcrl-structural-mechanism-pilot-v1` at `ec20c8bdf`), whose decision and locks are unchanged.

**Data and scope.** Statistical machine-learning research on public ACS Census tabular records (Florida 2019 development roles). The question is whether releasing income and occupation probabilities from shared prediction cells keeps them useful while reducing recovery of the recorded field SEX, by one recipient or by two linked recipients. Everything is exploratory development evidence on already-exposed people; the sealed assessment stayed closed.

## Bottom line

**Recovery (descriptive).** At the point, the selected SCM setting (β = 1.0, 261 / 266 cells) lowered fresh-reader recovery of SEX from the linked pair against every comparator, on DEV_REPORT, seed 0 / seed 1. These are not claims: the nominee fails one DEV utility point clause (below), and the seed-1 family interval against NEURAL_RAW includes 0.

| Comparator | Comparator pair AUC | SCM lower by |
|---|---|---|
| Matched task-only twin | 0.658 / 0.660 | 0.041 / 0.038 (beyond the 0.020 target at the clause bound) |
| Published IBSI objective | 0.632 / 0.636 | 0.014 / 0.014 |
| Selected shuffled-signal control (r2; most protective eligible on the MODEL_SELECT proxy) | 0.656 / 0.658 | 0.038 / 0.036 |
| Eligible neural adversarial comparator | 0.640 / 0.638 | 0.022 / 0.016 |
| Most protective eligible frozen tree hybrid (H_LJ) | 0.641 / 0.643 | 0.024 / 0.021 |

SCM's own pair AUC was 0.618 / 0.622.

Per recipient the picture is mixed. SCM's income recipient read higher than NEURAL_RAW's and H_LJ's on both seeds, and its occupation recipient read higher than H_LJ's (`DECISION.md` §4).

**Why no claim follows.** The nominee **missed one DEV_REPORT utility point clause**: occupation NLL excess against Ucal on seed 0 was 0.0106, against a 0.010 cap. All 24 utility confidence statuses are unresolved, and men's within-SEX occupation Brier excess (0.0062 / 0.0078) is above the 0.005 guard at the point. Under the frozen rules:
- the recipe effect is not established;
- attribution was not tested;
- the competitive claim fails the utility gate;
- the subgroup status is unresolved.

**What the evidence supports (descriptive).** Task-only merging to the same cell count lowered pair AUC by only 0.0035 / 0.0030 relative to C0 (unresolved). This is the matched twin, which uses SEX only through the within-SEX guard. SCM read 0.041 / 0.038 below the twin.

That comparison is matched on cell count, not on utility. SCM spent its whole plug-in occupation-NLL budget (0.0100 nats, against the twin's 0.0027 / 0.0026) and its whole grouped income-Brier budget (0.005). On held-out people its occupation-NLL increase over C0 was 0.0106 / 0.0111 on MODEL_SELECT and 0.0134 / 0.0092 on DEV_REPORT.

Occupation NLL is the only clause any SCM setting breached: β = 1.5 on MODEL_SELECT and β = 1.0 on DEV_REPORT seed 0. It is therefore the measured held-out bottleneck at these caps. Whether the reduction is due to the SEX-guided merge order was not tested.

## Compute, custody and receipts

- **Compute.** 0.46 CPU-hours charged of the 8 CPU-hour cap, and about 1 h 10 min of the 6-hour elapsed limit counted from 02:38:58Z (job-queue ledger). The ledger counts 11 units done and 2 failed attempts, the CatBoost-path environment error. The two fit units used 856 CPU-s, and agent and light-script work is charged as upper estimates. No cloud or paid compute was used.
- **Receipts.** The pre-fit receipt `f0a3b94e9` was pushed to GitHub at 03:08:29Z, before the first fit attempt. The reporting-lock receipt `bd31bcde0` was pushed at 03:20:10Z, before the DEV_REPORT re-read at 03:20:19Z. The final publication and its verified remote SHA are recorded in `PUBLICATION_RECORD.json`.
- **Custody.** The on-device backup and an off-device copy on the external drive both passed the restore-and-deploy check from the backup alone (`BACKUP_VERIFICATION.json`). The off-device read-back was warm (the drive stayed mounted), so a cold re-check after remounting is pending.

## Read in this order

1. `DECISION.md`: the answers, the favourable and adverse comparisons, the bottleneck, and what remains unresolved.
2. `RESULTS_TABLES.md`: the utility bank on MODEL_SELECT, final-bank recovery and utility on DEV_REPORT, all 42 primary endpoints, claims, controls and coverage.
3. `PROTOCOL.md` and `ANALYSIS_LOCK.json`: frozen before any real fit (PROTOCOL sha256 `d19d364d…`, as locked). The public pre-fit receipt is `f0a3b94e9` and the reporting-lock receipt is `bd31bcde0`. `PROTOCOL_POSTRUN_CLARIFICATIONS.md` holds the verifier's post-run clarifications; no number or decision changed.
4. `SELECTION_LOCK.json`, `SERVICE_BANK_MANIFEST.json`, `REPORTING_LOCK.json`, `FIT_SUMMARY.json` and `READERS_SUMMARY.json`.
5. `report/`: `DEV_REPORT_RESULTS.json`, `PRIMARY_ENDPOINTS.csv`, `SECONDARY_ENDPOINTS.csv` and `RECOVERY_AND_UTILITY_TABLE.csv`.
6. `DEPLOYMENT_RESULTS.json`, `DEPLOYMENT_RECHECK.json` and `QUICKSTART.md`: raw-field packages for SCM and TWIN (both seeds) and the seed-0 neural reference, each reproducing the attacked DEV_REPORT outputs bitwise.
   - After the verifier's D1, the SCM package path gained its own schema refusals.
   - The re-check refused all 12 cases on all four packages, and the CSV path reproduces their wires bitwise. The packages were not rebuilt.
   - No seed-1 neural package exists, a coverage gap.
7. `ENGINEERING_REPAIRS.md`:
   - one environment failure before any fit, preserved and rerun;
   - one post-lock logged-access addition for the verifier (`common.py`);
   - one post-report deployment schema repair (`deploy_scm.py`, re-checked);
   - a docstring note in `selection.py`;
   - the verifier's coverage notes.

   Three locked modules therefore differ from their ANALYSIS_LOCK hashes: `common.py`, changed before the report, and `deploy_scm.py` and `selection.py`, changed after it. None changed a reported number.
8. Roles:
   - `roles/math_reviewer/`: the pre-fit review, 5 must-fix items applied;
   - `roles/verifier/`: independent recomputation, 8/8 checks PASS. Discrepancies D1–D8 are documentation and protocol-text gaps; no reported number changed;
   - `roles/claims_reviewer/`.
9. `ADMISSION.json`: the stopped campaign's artifacts, identified by hash.
10. `HANDOFF.json`.

## Roles

There were four roles:
- the coordinator and implementation owner (the main session);
- the mathematical and baseline reviewer;
- the independent verifier;
- the claims and closeout reviewer.

No new mechanism designers were used.
