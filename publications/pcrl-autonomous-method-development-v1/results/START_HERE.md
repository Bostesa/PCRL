# Start here: PCRL autonomous method-development campaign

**What this is.** A bounded, exploratory method-development campaign, run on 2026-10-17 from 16:45:41Z. The limits were 6 elapsed hours and 10 CPU-hours; actual use is in `RESOURCE_ACCOUNTING.json`.

**Data.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, using only the existing development roles, which earlier campaigns already used.
- Cohort: housing-unit residents aged 25–64, civilian employed.
- The sealed assessment stayed closed.

**The service.** From six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP), it predicts:
- an income class (PINCP × ADJINC / 10^6 above $50,000);
- an occupation group (OCCP mapped to five civilian broad groups).

Each recipient receives only its own probability vector and argmax decision.

**What is measured.** Evaluation readers infer the recorded SEX column from each output alone and from the two outputs aligned to the same record (the linked pair).

**What was built and tested.** Three mechanism families, each with an initial version and up to two revisions. The coordinator implemented A; separate investigator agents implemented B and C.

| Family | Mechanism | Outcome |
|---|---|---|
| A | Constrained coalition-aware leaf correction, with a new holdout utility controller | Not promotable |
| B | Coarse or coupled income code beside the protected JOINT_TREE occupation output | Promotable |
| C | SEX-mixture marginalization | Not promotable |

**The finalist.** `B3CTL_UNCOUPLED_MIX50_C6A`, labelled `EDGE_SELECTED` under the registered promotion rule. It recombines the protected-tree replication's stored components:
- **Occupation output:** the stored JOINT_TREE output, bitwise H_LJ's.
- **Income output:** a 6-level code, calibrated on CALIBRATION, of the average of the stored LOCAL_TREE and JOINT_TREE income scores.

**DEV_REPORT results** (one logged evaluation; seed 0 / seed 1):

- **Linked-pair recovery:**

  | Release | Seed 0 | Seed 1 |
  |---|---|---|
  | finalist | 0.613 | 0.618 |
  | NEURAL_RAW | 0.640 | 0.638 |
  | H_LJ | 0.641 | 0.643 |

- **Claim B, `SUPERIOR_TO_ALL_ELIGIBLE`.** The linked-pair benefit (reference AUC − finalist AUC) has a clause bound above 0 against both eligible references on both seeds.

  | Reference | Benefit (point) | Clause bound |
  |---|---|---|
  | NEURAL_RAW | +0.027 / +0.0199 | 0.0135 / 0.0059 |
  | H_LJ | +0.028 / +0.025 | 0.0191 / 0.0129 |

  - The improvement is **positive, below the original 0.020 target**: no clause bound exceeds 0.020, so the material improvement is not established.
  - The 2 occupation recipient clauses against H_LJ are IDENTICAL_RELEASE, and the 2 against NEURAL_RAW are REPLAY_OF_PUBLISHED.
- **Claim E, `DEV_POINT_ELIGIBLE`.** Every utility clause (24/24), within-SEX guard (4/4) and recipient non-worsening condition (8/8) holds at the point.
  - This is at the edge: income summed-Brier excess against Ucal is 0.0049 / 0.0049 (cap 0.005), and seed-1 income accuracy loss is 0.0096 (cap 0.010).
  - Simultaneous confidence is `CONFIDENCE_UNRESOLVED` (3 PASS, 25 UNRESOLVED of 28). A point pass is not established population preservation.
  - The occupation-side utility and guards replay H_LJ's published DEV values.
- **Claim M, `MECHANISM_ATTRIBUTION_ESTABLISHED`.**
  - Step 1 compares against the matched task-only control: +0.052 / +0.047. That control differs in both outputs, so this step credits the inherited privacy-trained components as a whole.
  - Step 2 compares against H_LJ: +0.028 / +0.025, using the same keys as claim B.
  - The finalist spends +0.019 / +0.015 nats more occupation NLL than the task-only control.
- **Claim J, `JOINT_ADVANTAGE_NOT_ESTABLISHED`.** Against the LOCAL-only coarse twin the benefit is +0.016 / +0.010; seed 1 is UNRESOLVED (clause bound −0.0025). The finalist spends +0.0027 / +0.0027 more income Brier than the twin.
- Novelty is not claimed. The IBSI comparison is descriptive only.

**Caveats.**
- The selection was adaptive and edge-seeking. The finalist was a control-designated configuration, promoted under the statistical reviewer's adjudication, which superseded an unregistered interim exclusion (`PROMOTION.json`).
- The claim conjunctions were written after the finalist was selected, but before any final reader fit (verifier D1; a PROTOCOL §9 timing deviation).
- The two seeds are the same 5,943 people with shared bootstrap draws. DEV_REPORT intervals exclude the adaptive search.
- On every observed role, the income output adds no distinct split of people beyond the occupation output. The differences are in what the declared readers extract, not in the information carried.
- Reader recovery is not a privacy guarantee.

**Verification and custody.**
- The independent verifier, using its own code, passed 8/8 checks. Every recomputed endpoint equals the record.
- Same-device and off-device restore-and-deploy checks pass from the backup alone. The off-device read-back was warm (it may have been served from cache); a cold re-check after an ordinary remount is pending. See `BACKUP_VERIFICATION.json`.

## Read in this order

1. `RESEARCH_DECISION.md`: the claims with their explicit conjunctions, the answers to the campaign questions, the bottleneck that remains, and what is not claimed.
2. `FAILURE_MAP.md` and `MECHANISM_CARDS.md`: the starting record, each family's card, and every revision with its reason.
3. `EXPERIMENT_LEDGER.csv` and `COMPARISON_TABLE.csv`: every configuration, control and reference on MODEL_SELECT (development proxies), with decisions.
4. `PROMOTION.json`: the registered rule applied to all fitted configurations, with the disclosed superseded interim record. Then `roles/statistical_reviewer/DEVELOPMENT_REVIEW.md` §0 for the adjudication.
5. Locks, in order:
   - `PROTOCOL.md` (registration; public receipt `7e306dcfd`, pushed 17:01:39Z, before any new fit);
   - `SOURCE_ADMISSION.json` and `SOURCE_ADMISSION_2.json`;
   - `FINALIST_LOCK.json`;
   - `FORECASTS.json` and `FORECASTS_2.json`;
   - `ANALYSIS_LOCK.json`, with the reviewer's draft and encoding check;
   - `SERVICE_BANK_MANIFEST.json`;
   - `REPORTING_LOCK.json` and `READERS_SUMMARY.json` (public receipt `614c9f36b`, pushed 17:45:22Z, before the DEV_REPORT read at 17:45:30Z).
6. Results:
   - `report/DEV_REPORT_RESULTS.json`;
   - `PRIMARY_ENDPOINTS.csv` (52 keys) and `SECONDARY_ENDPOINTS.csv` (20 keys);
   - `UTILITY_TABLE.csv` and `RECOVERY_TABLE.csv`;
   - `FORECAST_SCORES.json`;
   - `UTILITY_RECOVERY.png`. The left panel shows MODEL_SELECT development proxies; the right panel shows DEV_REPORT fresh readers. These are two measurements, not one frontier.
     - The right panel's bars are marginal 95% intervals of each AUC. They are conditional on the fitted releases and readers and do not cover the adaptive search. The claims rest on paired differences, not on whether bars overlap.
     - Open markers flag a failed DEV utility point clause only. PUB_IBSI also fails the men's guard (0.0051 / 0.0091).
7. Verification and engineering:
   - `INDEPENDENT_VERIFICATION.json` and `roles/verifier/`;
   - `PROTOCOL_POSTRUN_CLARIFICATIONS.md`;
   - `DEPLOYMENT_RESULTS.json` and `QUICKSTART.md`;
   - `BACKUP_VERIFICATION.json`.
8. Closeout:
   - `roles/claims_reviewer/`;
   - `RESOURCE_ACCOUNTING.json` and `HANDOFF.json`;
   - `MANUSCRIPT_CLAIMS_HANDOFF.md`;
   - `roles/`, which records who did what.

## Roles (who did what)

| Responsibility | Done by |
|---|---|
| Coordination and integration; family A; registration, final stage, documents, custody, publication | the main session (coordinator) |
| Mechanism investigator, family B | a separate agent (`roles/investigator_b/`) |
| Mechanism investigator, family C | a separate agent (`roles/investigator_c/`) |
| Baseline and statistical review: development review, promotion adjudication, frozen claims and endpoints, lock-encoding check | a separate agent (`roles/statistical_reviewer/`) |
| Independent verification with separate code | a separate agent (`roles/verifier/`) |
| Engineering and claims review | a separate agent (`roles/claims_reviewer/`) |
