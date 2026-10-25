# Handoff: direct-recovery frontier campaign (`pcrl_direct_recovery_frontier_v1`, DRF1)

First written 2026-10-25 from 04:46Z (`date -u`). Campaign start 02:17:51Z.
- **Branch.** `research/pcrl-direct-recovery-frontier-v1`, based on 73931682e, the strong-anchor (SAR1) research tip. The research
  branch is not pushed.
- **Registration receipt.** Commit dae5914f4, the root commit of `publication/pcrl-direct-recovery-frontier-v1` at
  github.com/Bostesa/PCRL. It was pushed at 03:15:40Z and verified on the remote at 03:15:46Z, before the first label read
  (03:16:17Z).
- **Decision.** No method claim; failure stage 3 (`RESEARCH_DECISION.md`).

## State

- **Research is closed.** Both evidence-driven revisions are used:
  - R1 was triggered by MODEL_FIT evidence before any MODEL_SELECT read, so it is registered.
  - R2 was triggered by the R1 reading, so it is adaptive. It changed the search budget only.
  - Family DR has no revision left (PROTOCOL 9).
- **Specifications.** 28 of 32 version-specific release specifications were used: 6 candidates (DR_AUC, DR_AUC_R1, DR_AUC_R2 x 2
  versions), 18 matched controls (DR_MI and two CTRL points per revision and version) and 4 optimistic diagnostics (OPT_L, OPT_S).
  The remaining 4 were not spent. Internal optimizer counters (sweeps, moves, merges, candidate evaluations) and nested-fold fits are
  recorded separately in `dr/FIT_*.json` (`traces`) and `RESOURCE_ACCOUNTING.json`.
- **Readings.** Three readings plus two robustness records:
  - DRF_V1, registered: stopped at a usefulness failure, releases kept.
  - DRF_R1, registered: the headline.
  - DRF_R2, adaptive.
  - `claims/DRF_R1/D2R.json` and `claims/DRF_R2/D2R.json`.
  - Two D2 looks in total, none met (`claims/CAMPAIGN_SUMMARY.json`). No multiplicity correction is applied.
- **Verification.** The independent verifier recomputed the following with separate code, in queue units for every label read:
  - the finite-law objective (v1, R1 and R2);
  - the 28 conditions and the paired costs;
  - every reader comparison and label, before each decision record existed;
  - both D2R records;
  - the bootstrap SEs, plus a bounded refit sample of winning readers;
  - the deployment packages.
  - the closeout and custody records against the logs and the files they summarize;
  - the numbers in the decision documents.
  `CHECKS.json` holds 196 records after the final hand-back (4). The only non-PASS records are verifier-side defects, each kept with
  its corrected re-run beside it:
  - the integrity run that failed on the verifier's own revision parser;
  - two closeout cross-check runs that misread the RESTORE_CHECK.json layout.
  None changes a label. Details are in `roles/verifier/` (`VERIFICATION_REPORT.md`, `CHECKS.json`, hand-backs 1-4 verbatim).
- **Claims review.** The statistical reviewer's review is `roles/statistical_reviewer/CLAIMS_REVIEW.md`. Its 11 required corrections
  are applied and its 11 recommended changes adopted in `RESEARCH_DECISION.md` and `START_HERE.md`.
- **Deviations.** Five, append-only, in `PROTOCOL_DEVIATIONS.md`:
  1. a selftest ran without the thread limits (charged);
  2. and 3. and 5. files edited or uncommitted while units ran, none imported by them;
  4. a cosmetic SUCC duplicate in the frozen D2R rules output.
- **Agents.** Four live agents at most, including the coordinator: statistical reviewer, investigator A (exact finite-law AUC and tiny
  exact instances) and the independent verifier. None spawned another agent.
- **Resources** (`RESOURCE_ACCOUNTING.json`, written by `python3 -m drf.closeout` after the queue closed at 05:53:36Z):
  - 3.60 h elapsed of the 8 h ceiling;
  - 3.99 CPU-hours charged of 12, counting queue units and every out-of-queue and agent estimate, including the 1,085 CPU-s of
    deviation 1. The verifier charged 20 more CPU-s after the closeout (VERIFIER_OUT_OF_QUEUE_5), for 4.00 CPU-hours in all;
  - by stage: fits 3,731 CPU-s, readers 3,635, custody 3,638, deployment checks 1,226, verification 137, evaluation 12, claims 9;
  - 58 queue units done and none failed. The verifier's DRF_VER_C1_V2 was withdrawn before it started and read nothing;
  - at most two heavy jobs at once, one numerical thread each;
  - zero paid compute, and no campaign process left running.
  - Internal fits, recorded separately from the 28 specifications: 96 solver runs, 30 nested outer fits and 120 cross-fit fits on
    MODEL_FIT, and 17,280 reader fits (1,224 cache hits).
- **Publication.** The sanitized publication is a single commit on `publication/pcrl-direct-recovery-frontier-v1`, a child of the
  receipt dae5914f4, built from the research tip that holds this handoff.
  - Before the push, the proposed tree, its commit message and the branch history were scanned privately for the owner's names,
    username and host names. Matched strings are never written anywhere.
  - Its commit, push time and remote verification are in `PUBLICATION_RECEIPT.json` on the research branch. That file was written
    after the push, so it is not in the published tree.
  - The research branch itself is not pushed. Public account and commit metadata may still identify the owner; no anonymity is
    claimed.
- **Data access** (`ACCESS_RECEIPT.json`).
  - 38 logged label reads, none before the public registration receipt (verified 03:15:46Z).
  - MODEL_FIT fit reads: 8, each under its committed card (v1 4, R1 2, R2 2).
  - MODEL_SELECT reads:
    - utility evaluation, 14;
    - reader grids, 8;
    - reader custody, 2;
    - verifier, 4.
  - Verifier MODEL_FIT reads: 2.
  - No ATTACK, DEV_REPORT or CALIBRATION label was read, and no sealed or capped-out record. No MODEL_SELECT read came from a fitting
    entry point, and no fitting read preceded its card's commit.
  - The lineage DEV_REPORT access log is unchanged against its campaign-start snapshot.
  - Release wires of every role were computed from features only, as in earlier campaigns. No population was added and no new
    personal record was downloaded.

## Where things are

| What | Where |
|---|---|
| Protocol, rules, cards, plans | `PROTOCOL.md`, `roles/statistical_reviewer/RULES.md` / `RULES.json`, `cards/`, `claims/PLAN_*.json` |
| Decisions and robustness | `claims/<reading>/DECISIONS.json`, `claims/<reading>/D2R.json`, `claims/CAMPAIGN_SUMMARY.json` |
| Endpoint tables | `ENDPOINTS.json`, `FRONTIER.json` / `FRONTIER.csv`, `TABLES.md` / `TABLES.json` |
| MODEL_FIT records | `dr/BUDGETS.json` (frozen), `dr/FIT_V*.json`, `dr/FIT_R1_V*.json`, `dr/FIT_R2_V*.json`, `dr/OPTIMISTIC_V*.json` |
| Team records (verbatim) | `roles/` (coordinator log and briefs; investigator A's derivation, tiny exact instances and hand-backs; reviewer's rules, pre-registration and claims reviews; verifier's code, outputs and hand-backs) |
| Deployment checks | `deploy/DEPLOYMENT_CHECK.json` (R1 packages), `deploy/DEPLOYMENT_CHECK_R2.json`, `deploy/DEPLOYMENT_VALIDITY.json` |
| Custody records | `custody/` (reader custody, backup and off-device records) |
| Private artefacts (never published) | `.private/pcrl_direct_recovery_frontier_v1/`: runs (models, releases, evaluations, grids, reader cache, custody readers), packages under `deploy/`, ledgers under `compute/`, the label access log under `access/` |

## Packages (six raw fields in, each recipient's probability vector and decision out; `QUICKSTART.md` 3)

The 8 packages are the four releases that keep the original usefulness conditions at the point, in both versions: DR_AUC_R1 and
DR_MI_R1 (registered), and DR_AUC_R2 and DR_MI_R2 (adaptive).
- **Build.** They are built under python3 3.14.3, where the releases were fitted.
- **Raw-input parity.** Each package reproduces its stored release bitwise on the raw fields of all 20,000 MODEL_FIT rows, through
  the API, the CSV wrapper and the underlying function.
- **Attacked-output parity.** It reproduces the evaluated MODEL_SELECT wire bitwise on the whole batch and in requests of 1, 7, 50
  and 333 rows, with identical decisions. The maximum absolute difference is 0.0 everywhere (`deploy/DEPLOYMENT_CHECK.json`,
  `deploy/DEPLOYMENT_CHECK_R2.json`).
- **Validity.** Every stored probability is finite and in [0.005, 0.971], and rows sum to one within 3.3e-16. The encoder has
  exactly the six permitted inputs and no feature named after a forbidden field, and no dependency is label-like
  (`deploy/DEPLOYMENT_VALIDITY.json`).
- **Refusals.** Requests carrying SEX, a task target or extra, missing or reordered fields are refused. So are per-request draw keys
  (the releases are deterministic) and hidden-output requests.
- **Staleness.** Each package binds the 23 code modules and model files its deployment function reads; a changed module makes it
  stale.
- **Independent check.** The verifier checked all 8 packages with its own emission (244 of 244 items each).

## Custody

- **Scope** (`drf.custody`). The final code (the archive of the tip: `drf/` and every imported source package, plus the admitted
  results trees the code reads) and the private files the code reads:
  - this campaign's private tree: models, releases, evaluations, grids and reader cache, custody readers, packages, ledgers and the
    access log;
  - the income-headroom tree: the admitted mirror and the frozen reference bank;
  - A1_PO_16 (W);
  - the NRM1 task cache;
  - the FRC1 SEX-side cache and the FRC1 twin TWB_C64.
  No code reads data by an absolute path, so a restore check runs from the backup alone.
- **Rehearsal backup** (`custody/BACKUP_REHEARSAL.json`). Taken at 04:50:17Z at tip d939c2dd8, 566.5 MB, restored from the backup
  alone with all checks passing. It is kept beside the final backup and superseded.
- **Final backup** (`custody/BACKUP_VERIFICATION.json`). Taken at 05:11:38Z at research tip 5a791d7f1: 568.2 MB, in
  `<worktrees>/pcrl-direct-recovery-frontier-v1-backup/pcrl_direct_recovery_frontier_v1_20261025T051138Z`. It was restored from the
  backup alone (checked 05:32:00Z):
  - the manifest re-hashes and the branch bundle verifies;
  - the 8 packaged releases re-run bitwise from features;
  - the 8 deployment packages redeploy bitwise at every request size, with every refusal;
  - the 6 custody readers re-score bitwise;
  - all 96 bank-member wires reload with their recorded hashes;
  - 24 models and 24 releases re-hash;
  - 1,080 reader-cache entries pass their checksums;
  - the frozen DR inputs match their pins.
  Both interpreter passes passed.
- **Off-device copy** (`custody/OFFDEVICE_COPY.json`). The final backup was copied at 05:32-05:33Z to YOTUO, at
  `PCRL_private_backups/pcrl_direct_recovery_frontier_v1/`.
  - Identity was checked at 04:25:55Z, 05:32:16Z and 05:53:02Z: disk6s1, ExFAT, volume UUID 3D74932B-33DD-3896-8CC6-0D2FDDAD6B8D,
    578.2 GB free after the copy.
  - The copy's manifest equals the source's, and the restore check run from the copy on the drive passed in both passes (05:52:52Z).
  - It is **warm**: the drive stayed mounted, so the read-back may have been served from cache. **A cold re-check after an ordinary
    remount is the owner's to run** (QUICKSTART 4).
  - The 22 existing campaign folders under `PCRL_private_backups/` and every other item on the drive were not touched, and nothing
    was mounted, unmounted or forced. macOS added its usual `._` metadata files beside the copied files on the ExFAT volume.
- **Storage.**
  - Internal free space was 121 GiB at the start and about 123 GiB later in the run, never near the 20 GiB floor.
  - The two backups take about 1.1 GB. Each restore check extracted about 1.1 GB temporarily and removed it afterwards.
- **Known custody gaps.**
  - Records written after the final backup are not in it: the backup and off-device records, the closeout receipts, the verifier's
    final check, this handoff's final text and the publication receipt, plus the matching tail of the private ledgers. The public
    ones are on GitHub.
  - The working cache of inherited private trees is not backed up as a whole; only the files the code reads are.

## Owner actions, not done by this campaign

- **The SAR1 public receipt.** ede0755fb carries identifying strings (SAR1 deviation 13). It was not force-pushed or rewritten.
  Removing them would need a force push to a public branch and would break the timestamped registration evidence; that is the
  owner's decision. This campaign's receipt and publication were scanned privately for the owner's names, username and host names
  before pushing (no match in the receipt; see `PUBLICATION_RECEIPT.json` for the final publication).
- **The 64 archived worktrees** from the storage relocation are untouched. The owner runs `finish_worktrees.py`.
- **The working cache.** The `.private/pcrl_*_v1` clones inside this worktree can be removed once the final backup is accepted.
- **A cold re-check** of the YOTUO copy after an ordinary remount (`QUICKSTART.md` 4).

## If the work continues

- **Family DR is closed here.** Both revisions are used, and its readings tie the strong start, SUCC, at a higher total NLL.
- **An open question, not a recommendation.** Whether another mapping in the same finite family could coarsen the occupation
  recipient within the budgets was not tested. The local search ended with occupation NLL, the men's guard and income Brier at
  budget.
- **Objective alignment** would need a design that resolves differences near 0.005-0.010 in pair AUC. At the present se_agg
  (0.008-0.012) a true 0.010 benefit is resolved 13-22% of the time. Optimizer-seed variation (R1 to R2) is of the same order and is
  not covered by any SE.
- **Confirmation** of anything here would need households not used by these campaigns.
