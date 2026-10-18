# Start here: overnight occupation-headroom and robust-release campaign (PCRL, 2026-10-18)

This folder is the record of one exploratory development campaign. It asked whether a stronger occupation predictor could create
enough task-quality margin to lower SEX recovery from both released outputs (income and occupation probability vectors with their
argmax decisions) while keeping H3's income improvement. The data are public-use ACS 2019 Florida PUMS development roles, records
already used by earlier campaigns.

**Outcome in one line.** A repaired occupation learner gives occupation headroom on both seeds. Both SEX-group guards are met with
the development reserve, and the binding men's guard improves. At the B80 level, a coalition-aware occupation correction beats matched
controls for the declared readers, but it does not beat the inherited JOINT_TREE releases on both seeds. No configuration passed the
frozen nomination rule, so DEV_REPORT stayed closed (only its label-free raw inputs were used, for deployment checks) and no final
evaluation ran.

## Read in this order

1. `ADVISOR_BRIEF.md`: one page.
2. `RESEARCH_DECISION.md`: the decision. It answers the eight campaign questions with both seeds, the strongest favourable and adverse
   comparisons, and the distance to each requirement.
3. `MECHANISM_CARDS.md`: index of the five mechanism cards and their reports under `roles/`.
4. `PROTOCOL_DEVIATIONS.md`: 21 numbered items (timing, provenance, access, errata and a platform restriction).
5. `MANUSCRIPT_CLAIMS_HANDOFF.md`: which statements the evidence supports, with qualifiers, and which must not be made.
6. `roles/claims_review/REPORT.md`: the claims review of these documents and how each item was handled.

## Records

- **Registration and rules (frozen before fitting):**
  - `PROTOCOL.md`, `EXPOSURE_TABLE.md`, `DEV_GRID.json`, `DEVGRID_CODE_LOCK.json`;
  - `roles/statistical_reviewer/RULES.json`, `SELECTION_RULE.md` and `CLAIMS_DRAFT.md`;
  - public receipt in `RECEIPTS.json`.
- **Admission:** `SOURCE_ADMISSION.json`: 201 copies of the source campaign's admitted artefacts. 140 are checked against
  committed expected hashes; the other 61 were compared bitwise with the source's private files.
- **Experiments:**
  - `EXPERIMENT_LEDGER.csv`: every configuration and grid run, with hypothesis, change, roles read, effort and results;
  - `WITNESS_SELECTION.json`;
  - `NOMINATION.json` and `NOMINATION_TABLE.csv`;
  - `NOMINATION_FRONTIER.csv`: score against minimum whole-release ratio.
- **Tables:**
  - `UTILITY_FRONTIER.csv`: utility per configuration and seed;
  - `READER_STABILITY.csv`: 8-cell pair benefits, stability labels and recipient maxima;
  - `GRID_RECOVERY.csv`: SCORE AUC per release, seed and bank;
  - `decomposition/`: paired cost decompositions on both seeds;
  - `roles/reader_robustness/RR_FINAL__*.csv`: fixed-recipe and probe readers.
- **Verification and engineering:**
  - `roles/verifier/` (independent code, `CHECKS.json`, `REPORT.md`);
  - `roles/engineering_review/` (`CHECKS.json`, `REPORT.md`, applied patch);
  - `checks/` (self-tests).
- **Closest work:** `roles/closest_work/EVIDENCE_TABLE.md`.
- **Deployment:** `QUICKSTART.md` (interpreters, raw-field deployment, stage order and gates) and `deploy/DEPLOYMENT_CHECKS_BATCH3.json`
  (final package checks; batches 1 and 2 are earlier records).
- **Closeout:**
  - `RESOURCE_ACCOUNTING.json` (elapsed time, CPU, configurations by budget class, fits, reader fits);
  - `ACCESS_RECEIPT.json` (every MODEL_SELECT label read by entry point; no ATTACK or DEV_REPORT read);
  - `BACKUP_VERIFICATION.json`, `HANDOFF.json`, `PUBLICATION_RECORD.json`.
- **Coordinator notes:** `roles/coordinator/` (`COORDINATOR_LOG.md`, `FINDINGS_NOTES.md`, briefs given to the agents).

## What is not here

Person-level records, fitted private models and reader scores stay in the private tree (`.private/`, never published; restorable
from the verified backup). The research branch with its full history stays local. The public copy is the sanitized publication
branch `publication/pcrl-occupation-headroom-overnight-v1`.
