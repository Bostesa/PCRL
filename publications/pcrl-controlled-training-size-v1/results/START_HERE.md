# Start here: controlled training-size test with one conditional protection experiment (PCRL, 2026-10-19)

This folder records one exploratory development campaign on previously used public-use ACS 2019 Florida PUMS development roles
(MODEL_FIT, CALIBRATION, MODEL_SELECT). It asked whether a fixed occupation recipe (A1_PO_16) keeps its task and subgroup margin
when trained on 50%, 70% and 100% of MODEL_FIT, against an equally trained ordinary control; and, because the full-data refit kept
it, whether one fixed correction recipe uses that margin to lower the declared development readers' recovery of the recorded SEX
column while keeping usefulness. Every earlier verdict stands unchanged.

**Outcome in brief.**
- **Training size.** On this pool the registered size explanation is supported. SUPPORTED_ON_THIS_POOL means larger nested training sets
  had resolved lower loss on MODEL_SELECT and the margin pattern matched; it is not a population learning curve or causal proof. The
  ordinary control shows a resolved size gain of its own.
- **Reproduction.** The full-data refit reproduces the source package bitwise.
- **Correction.** The correction recipe is closed. At its selected strength 0.3 it keeps point utility (and, separately, the reserve
  heuristic), but it does not lower the readers' recovery stably beyond the witness or the matched continuation.
- **Reader-label disclosure (amendment A-SC1).** The registered effective-reader-seed check, read literally over all 14 checked bank
  members, returned READER_SEEDS_NOT_EFFECTIVE: ExtraTrees readers (ET5, ET25) are seed-invariant on the income views of B2TWIN and
  RETIRED_B3CTL, which take 6 distinct values. On the registered minimum set (W100_v and every selected release) there was no
  equality. Amendment A-SC1, made after this check's outcome and before any reader-stability label was computed or read, limited grid
  invalidation to unexplained equalities. The grid passed the amended check. No stability formula changed.
- **Qualification** (PROTOCOL_DEVIATIONS items 11-13). "Before any reader-stability label was computed or read" holds for the
  coordinator and the statistical reviewer. The independent verifier had already computed its own labels (01:32:59-01:33:09Z, after the
  seed-check record and before the ruling at 01:36:00Z) and stored them unread.

## Read in this order

1. `ADVISOR_BRIEF.md`: one page.
2. `RESEARCH_DECISION.md`: the decision and plain answers to the eight campaign questions.
3. The registration, published on GitHub (receipt verified at 00:20:07Z) before any real-label fit or label read:
   - `PROTOCOL.md`, `SUBSETS.json`, `EXPOSURE_MAP.md`, `ACCESS_POLICY.json`, `SOURCE_ADMISSION.json`, `READER_SPLITS.json`,
     `DEVGRID_CODE_LOCK.json`;
   - `roles/statistical_reviewer/RULES.md` and `RULES.json`.
4. `PROTOCOL_DEVIATIONS.md` (every departure, failed attempt and timing disclosure), `ERRATA.md`, `roles/coordinator/ADJUDICATIONS.md`
   (A1-A4), `roles/statistical_reviewer/ADJUDICATION_SEEDCHECK.md` (A-SC1), `ACCESS_AMENDMENTS.json`.

## Records

- **Training size** (`size/`):
  - `SIZE_TABLE.csv`: every fit, with its metrics, clauses, guards, margins, reserve and income conditions.
  - `PAIRED_CONTRASTS.csv`: 50->70, 70->100 and 50->100, per ordering and version, with their labels.
  - `CONTROL_COMPARISON.csv`, `ENCODING.csv`, `SUBSET_VARIATION.csv`.
  - `SIZE_RESULTS.json`: everything, including the margin pattern, the readings, the reproduction and the evaluations.
  - `figures/SIZE_CURVE.{svg,png}` with its table view `figures/SIZE_CURVE.csv`: men's Brier excess and occupation reserve ratio
    against training size, for the recipe and both control stages.
- **Identity and exposure:** `IDENTITY_CHECKS.json` (reproduction, aliases, control identity, correction-base gates, deployment, subset
  membership of every label read).
- **Trigger:** `trigger/TRIGGER.json`, `roles/verifier/DEPLOY_W100.json`, `deploy/DEPLOYMENT_CHECKS_W100.json`.
- **Conditional stage** (`conditional/`):
  - `SELECTION_V0.json`, `SELECTION_V1.json`: the selected strengths and the full audit trail.
  - `COND_UTILITY.json`: utility, reserve, realized against predicted costs, and the references.
  - `CLAIMS.json`: comparisons, claim labels, the original benchmark and the equal reserve.
  - `MATCHED_CONTROLS.csv`, `ORIGINAL_BENCHMARK.csv`, `EQUAL_RESERVE.csv`.
- **Readers:** `grid_specs/COND.json` (the bank), `readers/READER_SEED_CHECK.json`, `readers/WIRE_CARDINALITY.json`.
- **Deployment:** `QUICKSTART.md`; `deploy/DEPLOYMENT_CHECKS.json` (11 packages: the two full-data witnesses, the two full-pool TASK_REFIT
  controls, a 50% fit, and the selected LOC, PAIR and matched CONT releases of both versions).
- **Verification and review:**
  - `roles/verifier/`: the verifier's separate code, append-only `CHECKS.json`, `CHECK_CHANGES.json` and REPORT.md.
  - `roles/engineering_review/`: REVIEW_WAVE1.md and the patches; `roles/claims_review/REPORT.md`: the claims review of these documents.
- **Investigators:** `roles/task_refit_investigator/REPORT.md` and `roles/correction_investigator/REPORT.md` (saved with
  attribution); the compute benchmark.
- **Closeout:** `EXPERIMENT_LEDGER.csv`, `RESOURCE_ACCOUNTING.json`, `ACCESS_RECEIPT.json`, `BACKUP_VERIFICATION.json`, `HANDOFF.json`.
  `PUBLICATION_RECORD.json` is written after the final push, so it is not part of the commit it describes.
- **Coordinator:** `roles/coordinator/` (DESIGN_BRIEF.md, COORDINATOR_LOG.md, ADJUDICATIONS.md).

## What is not here

- Person-level records, fitted private models, reader scores and caches, stored wires and deployment packages stay in the private
  tree `.private/`. They are never published, and are restorable from the verified backup.
- The research branch with its full history stays local.
- The public copy is the sanitized publication branch `publication/pcrl-controlled-training-size-v1`.
