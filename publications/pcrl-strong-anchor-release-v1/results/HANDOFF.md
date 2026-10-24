# Handoff: strong-anchor release campaign (`pcrl_strong_anchor_release_v1`, SAR1)

First written 2026-10-24 from 06:58Z (`date -u`). Revised from 07:30Z after the claims review and verifier wave 3. Campaign start
04:02:04Z.
- **Branch.** `research/pcrl-strong-anchor-release-v1`, based on 9c9f95385, the input-and-pair research tip.
- **Registration receipt.** Commit ede0755fb on `publication/pcrl-strong-anchor-release-v1` at github.com/Bostesa/PCRL, verified on
  the remote at 05:11:39Z before the first label read (05:11:51Z).
- **Decision.** No competitive method; see `RESEARCH_DECISION.md`.

## State

- **Research is closed.**
  - FA used both of its revisions.
  - FB was stopped after revision 1 by the coordinator's decision, with one revision unused. Its gain over NEURAL_RAW is not separated
    from NEURAL_RAW's own recipe continued to matched spend, and that difference changes sign on the robustness split. No PROTOCOL 9
    stop rule required the stop.
  - FS, the one replacement family, was closed on investigator B's stop memo.
- **Specifications.** 28 of 36 seed-specific release specifications were used: 8 candidates, 3 task twins and 17 matched controls. The
  remaining 8 were not spent.
- **Readings.** Six readings (FA_REG, FB_REG, FB_R1, FA_R1, FS_CPI, FA_R2) and two robustness records (FB_R1_D2R, FA_R2_D2R).
  - Both D2 looks were adaptive revisions; the two registered readings found no feasible strength.
  - The campaign summary is `claims/CAMPAIGN_SUMMARY.json`.
- **Verification.** Three waves: 365 checks, 355 PASS and 10 FAIL. The failures are kept and adjudicated in `PROTOCOL_DEVIATIONS.md`
  items 9-10, 13 and 14; none changes a label. The hand-backs are saved verbatim in `roles/verifier/`.
- **Claims review.** The statistical reviewer's review is `roles/statistical_reviewer/CLAIMS_REVIEW.md`. Its 12 required corrections
  are applied and its 11 recommended changes adopted.
- **Deviations.** Fourteen, append-only, in `PROTOCOL_DEVIATIONS.md`.
- **Failed attempts.** All are charged.
  - SAR_EVAL_FA_R2_V1 stopped on a reused CONT's write-once record (deviation 8).
  - SAR_REPORT_TABLES_V1 failed on the no-candidate readings' missing grid field; it was fixed and rerun.
  - A first deployment batch-dependence sweep, outside the queue, was stopped after about 10 minutes because it was too slow, and was
    rerun bounded.
- **Resources.** Final figures from `RESOURCE_ACCOUNTING.json`, written by `python3 -m sar.closeout` after the queue was closed
  (07:41:25Z):
  - 3.66 h elapsed of the 8 h ceiling;
  - 2.28 CPU-hours charged of 12, counting queue units and every out-of-queue and agent estimate;
  - 77 queue units done and 2 failed;
  - 28 of 36 configurations;
  - zero paid compute and no process left running.
- **Publication.** The sanitized publication is commit bca6e5b28, a child of the receipt. It was pushed to GitHub at 07:40:32Z and its
  remote tip verified at 07:40:40Z (`PUBLICATION_RECEIPT.json`): 260 files, no scan problem, and every gated file equal to its
  receipt blob apart from PROTOCOL.md's one redacted line. The research branch itself is not pushed.
- **Data access** (`ACCESS_RECEIPT.json`).
  - 53 logged label reads, none before the public registration receipt.
  - No ATTACK or DEV_REPORT label, no CALIBRATION SEX, no sealed or capped-out record.
  - No MODEL_SELECT read from a fitting entry point, and no fitting read before its card's commit.
  - The lineage DEV_REPORT access log is unchanged against its snapshot.

## Where things are

| What | Where |
|---|---|
| Protocol, rules, cards, plans | `PROTOCOL.md`, `roles/statistical_reviewer/RULES.md` / `RULES.json`, `cards/`, `claims/PLAN_*.json` |
| Decisions and robustness | `claims/<reading>/DECISIONS.json`, `claims/<reading>_D2R/D2R.json` |
| Endpoint tables | `ENDPOINTS.json` (compact), `TABLES.md` / `TABLES.json` (all members and comparisons) |
| Anchors | `anchors/ANCHOR_TABLE.md` / `.json`, `anchors/ANCHOR_SELECTION.md` (frozen before any fit) |
| Fit records | `fa/FIT_*.json`, `fb/FIT_*.json`, `fs/FIT_CPI_V*.json` |
| Team records (verbatim) | `roles/` (coordinator log, investigators, statistical reviewer, verifier) |
| Deployment checks | `deploy/DEPLOYMENT_CHECK.json`, `deploy/DEPLOYMENT_VALIDITY.json`, `deploy/DEPLOYMENT_TOLERANCE.json` |
| Custody records | `custody/` (scope trace, external inputs, reader custody, backup and off-device records) |
| Private artefacts (never published) | `.private/pcrl_strong_anchor_release_v1/`: runs, reader cache, packages under `deploy/`, ledgers under `compute/`, the label access log under `access/` |

## Packages (raw input to both recipients' outputs; `QUICKSTART.md` 2)

The 13 packages are the point-eligible releases FA_TFRB_R2 and FB_PFO, both versions, plus their informative controls.
- **Build.** They are built under the torch venv (python 3.13.7), where all 13 releases were fitted.
- **Equivalence to the evaluated output.**
  - Bitwise for the full evaluated DEV_REPORT batch.
  - For other request batch sizes, within 6e-16 per probability with identical decisions (`deploy/DEPLOYMENT_TOLERANCE.json`;
    deviation 14).
- **Checks.** Every refusal probe is refused, no forbidden encoded input is present, and every probability is valid.
- **Staleness.** Each package binds every code module its deployment function loads. After a change to any of those modules the
  package is refused as stale; rebuild it with `sar.deploy store` / `build`.

## Custody

- **Scope.** It comes from a label-free open-file trace (`custody/SCOPE_TRACE.json`):
  - this campaign's private tree;
  - the income-headroom tree (the admitted mirror, the frozen bank, H3, the HGB teachers and SUCC's frozen trees);
  - the DDR_PTC_R1 and A1_PO_16 (W) reference releases;
  - nrm `fa_cache` (TASK.pkl, a fit-time input);
  - the input-and-pair anchor-table inputs;
  - a hash-checked snapshot of FB's fit-time replication inputs (`custody/EXTERNAL_INPUTS.json`; the originals live only in an archived
    worktree outside this repository).
- **Rehearsal backup.** Taken at 06:46:54Z: 796 MB, restored from the backup alone. Both interpreter passes passed:
  - 13 of 13 packaged releases re-ran bitwise on the full MODEL_SELECT batch;
  - 5 of 5 custody readers re-scored bitwise;
  - 140 bank-member wires re-loaded with their recorded hashes;
  - deployment from the backup was bitwise, with every refusal.
- **Final backup** (`custody/BACKUP_VERIFICATION.json`). Taken at 07:33:46Z at research tip 4ae4809bf: 803 MB, in
  `<worktrees>/pcrl-strong-anchor-release-v1-backup/pcrl_strong_anchor_release_v1_20261024T073346Z`.
  - It was restored from the backup alone, and both interpreter passes passed (the same checks as the rehearsal).
  - The rehearsal backup (tip d0755eec1) is kept beside it and superseded.
- **Off-device copy** (`custody/OFFDEVICE_COPY.json`). The final backup was copied at 07:35-07:36Z to YOTUO, at
  `PCRL_private_backups/pcrl_strong_anchor_release_v1/`.
  - Identity was re-checked at 07:34:56Z: disk6s1, ExFAT, volume UUID 3D74932B-33DD-3896-8CC6-0D2FDDAD6B8D, 582.5 GB free.
  - The restore check was run from the copy on the drive, and both passes passed.
  - It is **warm**: the drive stayed mounted, so the read-back may have been served from cache.
  - **A cold re-check after an ordinary remount is the owner's to run** (QUICKSTART 4).
  - The 21 existing campaign folders under `PCRL_private_backups/` and every other item on the drive were not touched, and nothing
    was mounted, unmounted or forced. macOS added its usual `._`
    metadata files beside the copied files on the ExFAT volume.
- **Storage.**
  - Internal free space: 135.4 GB at 04:39Z, 136.4 GB at 05:21Z, 133.9 GB at 07:02Z and 133.0 GB after the final backup
    (07:34:41Z). It never approached the 20 GiB floor.
  - Peak temporary storage, an upper estimate: about 1.9 GB while a backup and its restore extraction coexist. That is the 0.8 GB
    backup plus its extraction of about 1.1 GB (0.86 GiB of private scope and 0.16 GB of code), removed after the check.
  - Beyond that peak, the working cache of APFS clones (6.3 GB logical, sharing blocks with its sources) and the kept backups
    (about 0.8 GB each).
- **Known custody gaps.**
  - Records written after the final backup are not in that backup. These are the backup and off-device records, the final resource
    accounting and the publication receipt, plus the matching tail of the private ledgers. The public ones are on GitHub.
  - The 6.3 GB working cache of inherited private trees is not backed up as a whole. Only the traced files are.

## Owner actions, not done by this campaign

- **The public registration receipt (deviation 13).** Receipt commit ede0755fb, public since 05:11Z, carries the owner's name in a
  drive folder name and the machine's host name, in four lines of three files.
  - The final publication redacts them in every published file.
  - Removing them from the public history would need a force push to a public branch and would break the timestamped registration
    evidence. That is the owner's decision.
- **The 64 archived worktrees** from the storage relocation are untouched. The owner runs `finish_worktrees.py`.
- **The working cache** (`.private/pcrl_*_v1` clones inside this worktree) can be removed once the final backup is accepted. Removal
  frees only blocks the clones no longer share with their sources.
- **The cold re-check** of the YOTUO copy after a remount.

## If the work continues

- **A post hoc single-version observation, not evidence of a method.** FA_TFRB_R2's version 1 has pair 0.5975 against SUCC's 0.6155.
  But version 0 is worse than SUCC, the pooled benefit is unresolved, and version 1's men's guard is unresolved in the confidence
  conjunction.
  - Version 0's strength was bounded by the guard estimate's between-fold uncertainty, and at that strength its occupation view rose by
    +0.025.
  - A new campaign would have to address that heterogeneity prospectively, without tuning on these MODEL_SELECT people.
  - A confirmation would need households not used here.
- **FB.** Its operation should not be resumed without a construction that NEURAL_RAW's own recipe cannot match at equal spend.
- **The material target.** It needs a pair AUC near 0.58 or lower in both versions, against references at 0.631-0.634.
