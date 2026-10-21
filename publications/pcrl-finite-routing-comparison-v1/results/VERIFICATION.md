# Independent verification (campaign `pcrl_finite_routing_comparison_v1`)

The independent verifier wrote its own replay code under `roles/verifier/code/`. That code imports no frc / nrm numerical function;
it uses only the admitted loaders and the saved teachers' decision functions. Every script has an invented-data selftest. Checks are
append-only in `roles/verifier/CHECKS.json`, with per-check records under `roles/verifier/records/`. The verifier read labels only
inside queue units, after the registration receipt (4 reads, all with operation `verify`). Its reports are saved verbatim:
`roles/verifier/INTERIM_REPORT_0030.md` (00:30Z), `roles/verifier/REPORT_0058.md` (00:58Z) and `roles/verifier/REPORT_FINAL_0125.md`
(01:25Z, the final report).

## What was replayed, and the result

| Block | What | Result |
|---|---|---|
| V1 | Receipt on the remote (own ls-remote), registration files byte-equal; source admission (pins, hashes, reference wires, library versions); access log (every read after the receipt, no closed role, entry points, row sets, unit intervals); registration-file history (deviation 9) | PASS. Procedural FAIL R6 three times, one cause: card FA_FRC_v1 amended append-only after the T1 reads (deviation 7). ATTENTION R11: units launched with uncommitted files outside their import path (deviation 8) |
| V2 | Out-of-fold routing, B1 tables, A decoders, partition identity on all six roles, deployment-function replay | PASS (bitwise) |
| V3 | Every number of the four nested records, GATE reproduction, Q-PE | PASS (largest difference 4.3e-19) |
| V4 | MODEL_SELECT utility, paired costs, reserve and QUAL for every evaluation record and 11 references; D1 | PASS |
| V5a-d | P1 selection rule, aliases, brackets, F4 decision; protected / HQ / SHUF / ORD / UNT routing and B1 tables on all 18 configuration-versions; exact aliases; R1 UNT bracket | PASS; P1 person-level numbers NOT_RUN (no per-person output saved; would need a refit) |
| V6 | Registered grid FRC1_MAIN: 4,800 paired cell values and bootstrap SEs, ER1 grid validity from score fingerprints, D1-D6 and the outcome class for both readings, reader targeting, references | PASS (difference 0.0); outcome `ENGINEERING_REPAIR` reproduced |
| V6b | Reader-budget contrasts of both plans | PASS (0 mismatches) |
| V7 | Robustness grid FRC1_ROB: 2,592 paired cell values; CLASSIFIER_STABILITY.json labels; split identities; refit effectiveness | PASS |
| V8 | Same-device backup at tip 628937bb7: manifest re-hash, bundle verification, the verifier's own fresh restore (6,505 files), its own deployment replay of PRC_S2 / TWB_C64 / HQ_S2 (both versions, 5,943 DEV_REPORT people each), the 10 custody readers re-scored under the torch venv, the probe's strength-8 partition against its own PRC_S2 routing; restore directory deleted afterwards | PASS (bitwise throughout; the restored predict refused per-request keys, draw 1 and reordered columns) |

Final totals: 115 append-only checks.

| | PASS | FAIL | ATTENTION | NOT_APPLICABLE | NOT_RUN |
|---|---|---|---|---|---|
| Numerical | 96 | 0 | 0 | 17 | 2 |
| Procedural | 66 | 3 | 4 | 42 | 0 |

The three procedural FAILs share one cause (deviation 7). The two numerical NOT_RUN checks are the P1 person-level replay (no
per-person output saved) and a first stability check run before its record existed (superseded by a passing rerun).

## Not covered by the verifier

- The readers' own SCORE AUCs and bootstrap draws (only score fingerprints are saved; the paired values were recomputed from the saved
  AUCs and draws).
- Winner-recipe selection, and `nrm.rules.reader_stability`.
- The FIT-fraction grid FRC1_LC (descriptive).
- The P1 person-level numbers.
- Off-device custody (the drive was not attached).

## Final report

`roles/verifier/REPORT_FINAL_0125.md` (saved verbatim). Bottom line, in the verifier's words: "Every decisive record I could
replay agrees with my own code and data paths ... There is no numerical FAIL. The three procedural FAILs are one cause: the
append-only card amendment, your deviation 7." Verifier computation: 26 queue units (one failed, its own bug, disclosed) using
45.04 CPU-s, plus 18 CPU-s charged for out-of-queue work (the last 1 CPU-s after the queue closed, for the selftest below).

After the final report, at the coordinator's request, the verifier made one publication-only change (commit e048210a8): the
publication leak scan refused a hardcoded local scratch path in `code/v8_backup.py`. The scratch root is now a required argument
or environment variable with no default, and the behaviour is identical when the same root is passed. A new invented-data
selftest passes 9 of 9. V8 was not re-run, and no check result changed.
