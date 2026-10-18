# Start here: PCRL income-headroom successor development campaign

**What this is.** A bounded, exploratory method-development campaign (2026-10-18 from 01:51:35Z; limits 6 elapsed hours, 10
CPU-hours, 2 heavy jobs, 36 configurations, 3 families; actual use in `RESOURCE_ACCOUNTING.json`). It looked for a successor to
the retired `B3CTL_UNCOUPLED_MIX50_C6A` whose measured demographic recovery is lower while its income predictions keep real margin
below the original quality caps.

**Data.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed), existing development roles only. No new population, download or sealed record was opened.

**The service.** From six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP) it predicts an income class (2) and an occupation group
(5); each recipient receives only its own probability vector and argmax decision. SEX is the dataset's recorded demographic column;
evaluation readers estimate it from each output alone and from the two outputs aligned to the same record (the linked pair).

**Result in brief** (`RESEARCH_DECISION.md`):
- **Successor:** `H3_STRONG_LJDIR_PB` (family H), chosen by the rule registered before any fit (registration committed 14 s
  before the first fit; its public receipt followed 24 s after it, `PROTOCOL_DEVIATIONS.md` item 1). Its income output is a
  stronger task learner (a gradient-boosting stack) plus a budgeted transplant of the lineage's stored critic-trained (LOCAL_TREE
  / JOINT_TREE) income corrections. Its occupation output is the inherited JOINT_TREE output.
- **Income headroom kept on DEV_REPORT for Brier and NLL** (3.8-7.0 SE of margin; the retired candidate's binding Brier clause
  had 0.03-0.04 SE). Seed-1 income accuracy erodes the registered reserve (0.83 SE): claim R2 is `RESERVE_ERODED`, R3
  `INCOME_CONFIDENCE_UNRESOLVED (4 of 12)`. "Headroom kept" is descriptive, not a registered label.
- **Lower recovery, against the declared readers and auxiliary budget only:** pair benefit +0.014 / +0.017 over H_LJ (both
  benefit clauses pass at c1, but the ENHANCED seed-1 income recipient clause against H_LJ is unresolved and 3 of 4 matched
  bootstrap reader refits reverse the seed-1 benefit) and +0.013 / +0.012 over NEURAL_RAW (seed-1 clause unresolved); positive at
  the point, but the 0.020 target is not established; about half the retired candidate's benefit.
- **Claims:** E point-eligible; R reserve eroded; P `NOT_SUPERIOR_TO_ALL_ELIGIBLE [READER_SENSITIVE]`; N unresolved under
  ENHANCED (established under ORIGINAL); M task-only attribution; C not established; T not established; composite
  `NOT_COMPETITIVE`; beside it, O: inherited occupation constraints bind.
- **Feasibility:** at the plug-in, claim P's conjunction needs about 129,000 (D1) to 292,000 (D2) households (retired: 4.6-60
  million), against about 17,510 that exist for a same-population test; the requirement is not identified by these data (binding
  margins 0.40-0.87 SE; 80% power is never reached at zero true slack).
- **Decision:** a useful piece (income Brier and NLL headroom kept on DEV_REPORT), not a competitive successor; no confirmation
  population is requested.

## Read in this order

1. `RESEARCH_DECISION.md`: the answers, the claims with their conjunctions, the recommendation.
2. `MECHANISM_CARDS.md` (index) and `roles/investigator_{h,p,o}/CARD.md` and `REPORT.md`: every family, revision and control.
3. `EXPERIMENT_LEDGER.csv`, `COMPARISON_TABLE.csv`, `FRONTIER.csv`, `NOMINATION.json` / `NOMINATION_TABLE.csv`: every
   configuration and reference on MODEL_SELECT (development proxies) and the registered selection.
4. Registration and locks, in order:
   - registration commit `74376912c` (02:35:11Z): `PROTOCOL.md`, `SOURCE_ADMISSION.json`, `INHERITED_OCCUPATION_MARGINS.json`,
     `REFERENCE_MS_TABLE.csv`, `DEV_EVALUATOR_FINGERPRINT.json` and `roles/statistical_reviewer/` (`UTILITY_RESERVE_RULE.md`,
     `RESERVE_RULE.json`, `NOMINATION_RULE.md`, `DEV_GRID_REVIEW.md`, `CLAIMS_DRAFT.md`); public receipt `77dd6710f` (02:35:49Z);
   - `roles/statistical_reviewer/LOCK_ENCODING_CHECK.md`;
   - freeze commit `930235e9e` (03:03:35Z): `FINALIST_LOCK.json`, `ANALYSIS_LOCK.json`, `SERVICE_BANK_MANIFEST.json`,
     `READER_PLAN.json`; `roles/statistical_reviewer/FREEZE_CHECK.md`;
   - reporting-lock commit `8e559efe8` (03:07:29Z): `REPORTING_LOCK.json`; public receipt `9ad0fb25d` (03:07:55Z, before the
     DEV_REPORT read at 03:08:07Z).
5. Results: `report/DEV_REPORT_RESULTS.json`, `report/PRIMARY_ENDPOINTS.csv`, `UTILITY_RESERVE_RULE.md` (how the rule applied),
   `CONFIRMATION_FEASIBILITY.md` and its JSON.
6. Verification, engineering and custody: `INDEPENDENT_VERIFICATION.json` and `roles/verifier/`, `roles/claims_reviewer/`,
   `DEPLOYMENT_CHECKS.json`, `QUICKSTART.md`, `BACKUP_VERIFICATION.json`, `PROTOCOL_DEVIATIONS.md`.
7. Closeout (written last): `RESOURCE_ACCOUNTING.json`, `ACCESS_RECEIPT.json`, `HANDOFF.json`, `MANUSCRIPT_CLAIMS_HANDOFF.md`,
   `PUBLICATION_RECORD.json`.

## Roles (who did what)

| Responsibility | Done by |
|---|---|
| Coordination and integration; registration, evaluator, combinations, final stage, documents, custody, publication | the main session (coordinator) |
| Family H (headroom-first income learner) | a separate investigator agent (`roles/investigator_h/`) |
| Family P (pair-aware learned income output) | a separate investigator agent (`roles/investigator_p/`) |
| Family O (occupation-side margin repair) | a separate investigator agent (`roles/investigator_o/`) |
| Baseline and statistical review: reserve rule, nomination rule, grid review, claims draft, lock-encoding check, freeze check | a separate agent (`roles/statistical_reviewer/`) |
| Independent verification with separate code | a separate agent (`roles/verifier/`) |
| Engineering and claims review | a separate agent after the DEV_REPORT read (`roles/claims_reviewer/CLAIMS_REVIEW.md`, saved by the coordinator), plus the statistical reviewer's lock-encoding and freeze checks before it and the verifier's check 10 |

The investigators' and the claims reviewer's report files were saved by the coordinator because the platform refused, or the
agents' instructions forbade, subagent report-file writes (`PROTOCOL_DEVIATIONS.md` item 4).
