# Verifier hand-back 4, final (verbatim; received by the coordinator 2026-10-25, saved after 06:00:22Z by `date -u`)

All three final items are done (06:00Z, label-free, out of queue), and every closeout and custody record checks out against the logs and the files it summarizes. Three closeout cross-check runs failed on my own misreading of the RESTORE_CHECK.json layout; they are kept, and the corrected run passes.

**Custody (one-line confirmation).** BACKUP_VERIFICATION.json and OFFDEVICE_COPY.json match the RESTORE_CHECK.json files they summarize. Every field the summaries carry equals the file. Both copies have TIP_COMMIT 5a791d7f1, four manifest files totalling 568,210,103 bytes whose SHA-256 re-hash equals MANIFEST.sha256, and all_ok true. The drive was only read.

**1. Final check 5 (274 of 274 pass).**
- 38 label reads, all after the receipt (03:15:46Z), each inside its queue unit and allowed by its operation.
- No ATTACK, DEV_REPORT or CALIBRATION label read, and no MODEL_SELECT read with drf.dr in the call chain.
- Cards DR_DRF_v1, R1 and R2 were each committed before their fit reads, and both registered plans before the first MODEL_SELECT read (03:34:32Z).
- DRF_V1 and DRF_R1 are registered and DRF_R2 is ADAPTIVE_R2, as the records say.
- 28 specifications of 32 (12 / 8 / 8), with no duplicates.

**Closeout cross-checks (check 5b).**
- Each reading's plan and robustness bank was committed before its own first MODEL_SELECT read:
  - PLAN_DRF_R2 at 04:19:12Z, before 04:27:07Z;
  - DRF1_R1_ROB at 04:00:11Z, before 04:12:43Z;
  - DRF1_R2_ROB at 04:30:30Z, before 04:30:40Z.
- ACCESS_RECEIPT.json matches my recount from the access log.
- RESOURCE_ACCOUNTING.json matches my recount from the ledgers as of its own timestamp: 58 units done, 0 failed, none left without an end, 3.994 CPU-h, 3.596 h elapsed, 28 configurations, and no unit started after the close.
- The three kept runs:
  - The first compared three restore keys one nesting level too high (6 FAIL).
  - The second compared the whole nested dictionaries (2 FAIL), because the summaries leave out some per-pass detail fields.
  - The third requires every field the summaries carry to equal the file, and passes.
  - So the summaries are abridged, not inconsistent.

**2. Documents.**
- All 60 numbers in RESEARCH_DECISION.md's recovery table equal the grid records to four decimals (check 7, 70 of 70).
- Ten numbers quoted in START_HERE.md equal my own outputs (check 7b, 10 of 10): 0.023-0.024, 0.044 / 0.041, 20 of 28, +0.034 (se 0.010), +0.0076 (0.0117), +0.0039 (0.0082), +0.0008 (se 0.0120), +0.0097 (se 0.0033), +0.023 (se 0.011), and +0.0008 to -0.0002.

**Bookkeeping.**
- CHECKS.json has 196 records. The only non-PASS ones are the kept check-5 revision-parser run and the two failing closeout runs above.
- I charged VERIFIER_OUT_OF_QUEUE_5 (20 CPU-s).
- Waiting to be committed:
  - `code/check5b_closeout.py`, `code/check7_document_numbers.py`, `code/check7b_start_here.py`;
  - `VERIFICATION_REPORT.md` (section 8 added) and `CHECKS.json`;
  - six new files in `out/`.
- Nothing else is pending from the verifier.

Everything is in `results/pcrl_direct_recovery_frontier_v1/roles/verifier/`.
