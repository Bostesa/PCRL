# Independent verifier, hand-back 5 (verbatim; delivered to the coordinator at about 00:37Z on 2026-10-26; harness indentation removed)

# Independent verifier, wave-5 hand-back (RFA1)

Written 2026-10-26, about 00:13Z to 00:36Z.

- **Git.** Read-only throughout: no commit, push or ref change. The refs are unchanged: the publication branch is still at a9e535345 and the research branch at c4bc191e1.
- **Writes.** Only under roles/verifier/. Every wave-5 file is untracked; the list is at the end.
- **Data.** Nothing read a label. No private array, model or person-level table was opened.
- **Queue.** Everything ran outside it, since the queue is closed.
- **Paths.** Paths are given relative to `<worktree>`, the campaign worktree; personal absolute paths are withheld.

**Summary**
- **(4) Publication scan of a9e535345.**
  - The 211 tree blobs and both commit messages contain zero owner identifiers, user names, host names, owner e-mail or e-mail-like strings, by both run-time derivations.
  - There are 35 absolute paths, none of them personal.
  - No private content is in the tree.
  - The 9 gated files are byte-identical to the research copies.
  - The package equals research c4bc191e1 apart from the sanitizer's substitutions, with one unrecorded exception: the CSV's line endings were converted from CRLF to LF.
  - One observation is pre-existing and already public: the commit identity lines carry the owner's e-mail.
- **(5) Closeout records.** RESOURCE_ACCOUNTING.json, ACCESS_RECEIPT.json and custody/BACKUP_VERIFICATION.json all agree with the ledger, the access log and the backup on disk.
  - **The drive is absent**, so I checked the local backup only. OFFDEVICE_COPY.json is consistent with it.
  - My first run failed 3 checks because of my own check design. That run is kept, and the corrected run passes 33 of 33.
- **(6) Conformance at c4bc191e1.** Every wave-4 defect and nit is fixed.
  - **One mismatch remains, and it is my error from hand-back 4:** "3.1-6.2 SE" should read "3.1-6.1".
  - It appears at RESEARCH_DECISION.md:101, START_HERE.md:60, PROTOCOL_DEVIATIONS.md:60, and my HANDBACK_4.md:36 and :50.

## (4) Publication scan

Code: code/v5_pubscan.py and code/vsafe.py. Output: out/V5_PUBSCAN.json. 17 of 18 checks pass.

### Method

- **Inputs.** Plain read-only git commands (ls-tree, archive, cat-file, rev-list, config) wrote their output into a private scratch folder. Every archive member was checked against its tree blob id, so the scan covers exactly the tree blobs of:
  - the publication commit;
  - the receipt 34dc42b51;
  - the research commit's results/<campaign> and rfa/.
- **History.** The publication commit has tree 5276df3a and parent 34dc42b51. The receipt is a root commit, and rev-list shows exactly these two commits.
- **Identifiers, derived at run time two ways.**
  - rfa.publish_build.owner_patterns, which gives 11 patterns.
  - My own derivation, 15 strings in 5 categories:
    - the full name in five forms, plus its parts;
    - the account and home-folder names;
    - the git e-mail, and its local part;
    - the host names, from gethostname and the ComputerName, LocalHostName and HostName keys.
  - The git user name, which is the public repository handle, is kept as a separate category.
- **Counts only.** The output files refuse to write if any identifying string would appear in them.

### Counts

| Scope | Owner ids (union) | owner_patterns | User | Host | Owner e-mail | E-mail-like | Absolute paths | Credential-like |
|---|---|---|---|---|---|---|---|---|
| Publication tree (211 blobs) | 0 | 0 | 0 | 0 | 0 | 0 | 35, none personal | 0 |
| Receipt tree (79 blobs) | 0 | 0 | 0 | 0 | 0 | 0 | 8, none personal | 0 |
| Commit messages (2) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Commit identity lines (author and committer, 2 commits) | 2 per commit | 2 per commit | 0 | 0 | 2 per commit | 2 per commit | 0 | 0 |

File names have 0 hits. There are 0 Windows user paths and 0 file:// URLs.

### The 35 absolute paths, by category (line numbers only)

- **Drive mount point, 2.** Its label is the drive name recorded in custody: code/rfa/admit.py:366 and code/rfa/storage.py:24.
- **System paths, 26.**
  - code/rfa/storage.py:33 (/System/Volumes/Data).
  - code/rfa/release.py:6 (/run).
  - results/deploy/DEPLOYMENT_CHECK_MAIN.json:7 (an interpreter under /opt).
  - results/roles/challenger/HANDBACK_1.md:6 and HANDBACK_MESSAGE.md:22.
  - The `/usr/bin/env` shebang on line 1 of 21 verifier scripts.
- **Bare "/Volumes" or "/Volumes/..." in prose, 5.** HANDOFF.md:87, custody/OFFDEVICE_COPY.json:13, COORDINATOR_LOG.md:46 and :131, and verifier/code/v4_conformance.py:431.
- **Generic /tmp paths in example commands, 2.** QUICKSTART.md:17 and :38.

There are no personal home paths, no other-user homes, no session-scratch paths and no unknown volumes. The 4 home-relative `~/` paths (rfa/storage.py ×2, PROTOCOL.md, STORAGE_ADMISSION.json) name the relocation folder and contain no user name.

Your 4 hits are a subset of these 35. Your pattern counts only /Users/, /Volumes/ and /home/ followed by a name, so it does not see bare "/Volumes" or the system paths.

The public handle appears 5 times, all in the registered remote URL: rfa/common.py, rfa/receipt.py, RECEIPTS.json, SOURCE_ADMISSION.json and roles/coordinator/admit_stdout.json. That is by design.

### Observation O1 (for the owner; not new and not introduced by this build)

The author and committer lines of both commits carry the configured git identity:
- the name is the public handle;
- the e-mail is the owner's own address, not a no-reply alias. It is the same string the shared sanitizer replaces in file content.

The receipt commit 34dc42b51 is already on the remote (RECEIPTS.json, verified at 21:53:27Z), as is every earlier pushed commit with this identity. Pushing a9e535345 adds one more.

scripts/rfa1_private_scan.py scans commit messages but not identity lines. It also skips non-UTF-8 blobs, though there are none here.

### Private content: none in the tree

- **File types.** 211 regular 100644 text blobs: 111 .json, 66 .py, 32 .md, 1 .csv and 1 .txt.
  - No binary blob.
  - No array, model, reader, archive, log or image suffix.
  - No .private/ or runs/ path.
- **Arrays.** No JSON numeric list of 500 or more entries.
- **The one CSV.** results/dev/UTILITY_LEDGER.csv is the aggregate utility ledger: 46 rows by 29 columns (configuration, version, clause values). It has no person-level column.
- **Census keywords.** PUMA, SERIALNO and SPORDER appear 11 times, only as column names in code/rfa/deploy.py and in my refusal probes (v3_deploy.py, HANDBACK_3.md). No record-like rows.
- **Verbatim copies.** No blob equals a private record.
  - I saw 32,544 private files.
  - Only the 76 aggregate text records whose size matches a blob were hashed, and none matches.
  - Arrays, models, readers and tables were never opened.

### Gated registration files

All 9 are byte-identical across:
- the publication;
- research c4bc191e1;
- the receipt tree.

Their SHA-256 equals RECEIPTS.json files_sha256, and their blob ids equal published_blobs.

### The package against research c4bc191e1

The publication has 211 files. Every one has its research source, and every in-scope research file is published.

- **175 are identical.**
- **35 differ only by the recorded sanitizer.** Applying mechanism_design.publish_build.sanitise and then rfa.publish_build.redact_owner (with the run-time patterns) to each research copy reproduces the publication bytes exactly.
  - In total 123 spans are replaced. Each is an absolute path containing the home-folder name, replaced by `<repository>` (35) or `<worktrees>` (88).
  - The files:
    - the four fit records, with 21 spans each;
    - 21 of my verifier scripts, plus V1_GRAPH_CHECK.json and V3_DEPLOY.json;
    - rfa/common.py, rfa/custody.py, and rfa/fam/fit.py, fit_r1.py, rfit.py and rfit_r1.py;
    - roles/challenger/fitpath_check.py, the investigator's HANDBACK_1.md and the reviewer's HANDBACK_1.md.
- **1 differs in another way. This is defect P1 (minor, record fidelity).**
  - results/dev/UTILITY_LEDGER.csv has 47 CRLF line endings in the research copy (csv.writer's default) and LF in the publication. The two are equal after converting CRLF to LF.
  - Path.read_text() in rfa.publish_build.build normalizes newlines, so this change is not among the recorded substitutions. The content is identical.
  - I kept this as a failed check.
  - Fix: copy .csv files as bytes, or record the normalization.

## (5) Closeout cross-checks

Code: code/v5_closeout.py. Output: out/V5_CLOSEOUT.json, with all 33 checks passing.

**First run kept.** The first run failed 3 checks because of my check design:
- it counted macOS AppleDouble `._` tar entries as files;
- it compared the records' descriptive `unit` field ("<unit id> (queue; ...)") as if it were a unit id.

That run is kept as out/V5_CLOSEOUT_run1_CHECK_DESIGN_APPLEDOUBLE_AND_UNIT_FIELD.json, and the corrected run sits beside it.

### RESOURCE_ACCOUNTING.json against the ledger (13 of 13 pass)

The ledger has 108 events: init, 48 unit starts, 48 unit ends, 10 charges, and queue_closed at 23:34:19Z, which equals the record's time. No event comes after it, so the outside-queue CPU spent since the close is not in the ledger yet.

- 46 units done and 2 failed: RFA_VER_USEFULNESS_v1 (0.574 s) and RFA_VER_W4_R1ROB_v1 (19.375 s).
- One start per unit, and no unit without an end.
- CPU by stage and by unit prefix equal the ledger at 1 dp.
- The 10 agent charges equal the ledger's charge events.
- Charged CPU is 8,988.97 s, or 2.497 h. Elapsed time is 2.594 h.
- The limits are as initialized.
- The internal fit counters equal the sums over the units.
- The 24 configurations equal the fit records' saved configurations: 4 candidates, 4 twins and 16 controls, each attributed to the right card.
- No campaign process is running now.

### ACCESS_RECEIPT.json against the access log (7 of 7 pass)

The log has 40 entries: 4 MODEL_FIT and 36 MODEL_SELECT.

- None of the 40 reads comes after the receipt's time.
- The counts per operation, role, fields, entry point and card are equal.
- The registration time agrees across ACCESS_RECEIPT.json, RECEIPTS.json and the private verification record, and no read precedes it.
- There is no CALIBRATION, ATTACK or DEV_REPORT read, and no MODEL_SELECT read outside the benchmark, readers and verify entry points.
- Every fitting read happened under a card committed before the read, and the card's SHA-256 matches the card file.
- Every read falls inside its unit's start-end window, in a unit of the matching stage.
- The lineage DEV_REPORT log's SHA-256 now equals the campaign-start snapshot and the record.

### custody/BACKUP_VERIFICATION.json against the backup (all pass)

The backup is at `<worktree>/../pcrl-reader-fit-aware-release-v1-backup/..._20261025T232718Z`.

- **Folder.** It holds the four payload files, MANIFEST.sha256 and RESTORE_CHECK.json.
- **Manifest and tip.** The manifest equals the record and a fresh re-hash. TIP_COMMIT is 42ef29832.
- **Size.** size_bytes, 846,815,076, is the total of all six files. The payload alone is 846,807,864; adding the manifest gives 846,808,188, which is the copy's recorded size.
- **Bundle.** `git bundle verify` reports it is okay. It holds the research branch at 42ef29832 and requires 913eb5c91.
- **Restore check.** RESTORE_CHECK.json on disk equals the record.
- **code.tar.** 3,262 members, each equal to its blob at the tip.
- **private.tar.gz.** Only names, sizes and hashes were read.
  - It holds 2,065 files (1,817 campaign, 248 inherited), plus 2,269 AppleDouble entries. Among them are 12 GRID.json, 12 BOOT.npz and 45 RELEASE.npz.
  - No campaign file is missing now.
  - 4 files changed and 9 appeared after the backup time of 23:27:18Z: the access log, the ledger, interpreters.json, and the unit folders of RFA_CUSTODY_OFFDEVICE_V1 and of my two W4 units.
- **Timing.** The record was written inside its unit's window, 23:27:18-23:29:29Z.

### custody/OFFDEVICE_COPY.json

The drive is absent. It is not in /Volumes, and no external physical disk was listed at 00:25Z or 00:33Z. I mounted, unmounted and forced nothing, and I did not check the drive copy.

The record is consistent with the backup:
- tip;
- size = payload + manifest;
- WARM;
- all_ok and the manifest and bundle flags;
- drive UUID equal to PROTOCOL's;
- source path.

It was written inside its unit's window (23:31:03-23:33:14Z), and its stated 106.8 CPU-s matches the ledger's 106.752.

### Notes

- **N-C1.** OFFDEVICE_COPY.json was annotated after its unit ran: a `size_note` key was added in c4bc191e1. The content is correct and the change is visible in the history.
- **N-C2.** The HANDOFF Gap bullet (HO:90-93) lists only git commits. The private-tree changes made after 23:27:18Z are also missing from the backup: my two W4 units' label-access entries and ledger lines, the off-device unit's folder, and the queue close.

## (6) Conformance at c4bc191e1

Code: code/v5_conformance.py. It is derived from v4_conformance.py:
- Wave-4 line specifications carry over through an exact alignment with the wave-4 document versions.
- Every changed line got a new specification.
- Statement checks that tested the wave-4 defects now test the corrected text.

Output: out/V5_CONFORMANCE.json.

### Numbers

216 lines carrying numbers, 495 numbers; 371 values derived from records were checked against the records.

Two mismatches, both from my own hand-back 4:
- RESEARCH_DECISION.md:101 and START_HERE.md:60 say "each 3.1-6.2 SE". The unrounded ratios run from 3.137 to 6.148, so the range is **3.1-6.1**.
- I rounded 6.148 first to 6.15 and then to 6.2.
- The same slip is in PROTOCOL_DEVIATIONS.md:60 (deviation 11, which is append-only, so it needs an appended note) and in my HANDBACK_4.md:36 and :50, which is verbatim. This hand-back corrects it.

### Statements

112 of 114 pass. The two that fail are the 3.1-6.2 range above and the pending PUBLICATION_RECEIPT.json (HO:5, HO:12), which does not exist yet.

### Wave-4 items, all fixed

- **D-A1.** Same-checkpoint spend:
  - RD:100-116 has the table of 16 values at 6 dp, the 8 MODEL_SELECT residuals (each at least 3.3 SE), "post hoc", and the note that HI = S_k* in every fit.
  - SH:59-62 carries the matching sentences.
- **D-A2.** RD:72 now reads +0.0100, citing D2.targeting.
- **D-A3.** RD:17-22 states the decision per reading.
- **D-A4.** Bounds per reading at HO:27-29 and SH:109-110.
- **D-A5.** RD:28 discloses the unprefixed second-split label.
- **N1.** RD:182-183 gives the minimum share, 0.08595.
- **N2.** SH:53 now reads 1.4e-14.
- **N3.** HO:40-51 names all 11 deviations.
- **N4.** SH:83-87 gives the pair winners per reading, and ignore-one readers do win robustness cells in both readings.
- **N5.** The rule text at RD:55-56.
- **N6.** RD:91 reads "ALT (RAWJ)".
- **N7.** RD:68 states that the HI points sit at S_k* in every fit.

### Also confirmed

- claims/PER_RECIPE_PAIR_MEANS_corrected.json: its 15 falls equal my exact values at 5 dp; the means and flags are unchanged; prediction_met is false.
- The gap arithmetic at HO:92: 846,808,188 = 846,815,076 − 6,888.
- TABLES.md and ENDPOINTS.json are unchanged and still equal the final records.
- The reviewer's R1-R7 are still addressed.

## CPU

- **This wave:** about 15 CPU-s, all outside the queue, an upper estimate.
  - Publication scan: 2 runs and inspections, about 5.5 s.
  - Closeout: 2 runs, 1 failed start and one listing of private.tar.gz, about 7.5 s.
  - Conformance: 2 runs, about 0.3 s.
  - git archive, show and bundle-verify: about 1 s.
- **Previous wave:** wave 4 outside the queue was about 3 CPU-s, as reported.

## Files to commit (untracked; all under roles/verifier/)

**Code**
- vsafe.py 7cfec1b26f16b77cba52275574cbd7ddfdb59f8ec90d7d39cafdf629ed04c809
- v5_pubscan.py b3ed97230cc1ff9509c51285973ed1f25e346fd200c0c799e5eed72a163335c3
- v5_closeout.py d84d40cfedd87eed957019df7a3bf32cc878128cb255d71f37bdf49128c6d2cc
- v5_conformance.py f4ce53a78dbabc511fdbacdd02beb40e6692d34433cbe86d41dba8f5304637cf

**Outputs**
- V5_PUBSCAN.json f03c1872fcf9f3b42ed9b6c1fc350b5c33d0a85ee975f6c40b158fdc8312a786
- V5_CLOSEOUT.json b13ed1971b5e0f1d5973c9fc93ddc1098dfcf34b2b1e9e481b1794dcbb8a04c1
- V5_CLOSEOUT_run1_CHECK_DESIGN_APPLEDOUBLE_AND_UNIT_FIELD.json 326b038140d1e36a0bf699e94c1e7d6a5899464420fd18e540047edf1db05cff
- V5_CONFORMANCE.json f2bded4c353ea824357648d58a64ecfe3188e4ba7bc7cd728adee8d13925c76f

A counts-only scan of these eight files finds no owner identifier and no personal path. v5_pubscan.py does contain the generic prefixes '/Users/' and '/home/' as patterns.

**Documents checked at c4bc191e1**
- RESEARCH_DECISION.md 933fcde7aaeb2333d73281e87e6110bb2c710f52bdecadd0a072a1bd5f7f77b5
- START_HERE.md f703cd7e48f3a2c4aee78422c34fad4fcf27bcacd14aa1b3f7cce452fd96ac96
- HANDOFF.md 3482252a0fea82513ac1e058342f1ec5edc18c6d8e74faf6a248d92a321532e6

**If you correct "3.1-6.2" (RD:101, SH:60)**
- The publication would need rebuilding, since RD and SH are in the tree.
- PROTOCOL_DEVIATIONS.md:60 and HANDBACK_4.md stay as written.
- The fix is recorded through this hand-back and an appended deviation line.
