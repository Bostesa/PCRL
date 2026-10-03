# Run status

2026-10-03, final.

| Stage | Status | Evidence |
|---|---|---|
| 0 Admission, plan, predictions | DONE | INPUT_ADMISSION.json, RUN_PLAN.md, PREDICTIONS.md (f70bad3) |
| 1 Corrections + custody | DONE | CORRECTION_ADDENDUM.md, ORIGINAL_VS_CORRECTED.csv, *_CORRECTED.csv, corrections/ (70dbdf2) |
| 2 Exposure sensitivity | DONE | EXPOSURE_SENSITIVITY.md, EXPOSURE_ENDPOINTS.csv (rule 7a4a086; results 614846f) |
| 3 Protocol and lock | DONE | PROTOCOL.md, EXECUTION_LOCK.json (v1 3130c8d; amendment L1 7244537), PRIMARY_FAMILY.csv |
| 4 Output surfaces | DONE | OUTPUT_SURFACES.md |
| 5 FARE with controls | DONE | METHOD_ADMISSION.md, FARE_FRONTIER.csv, run_records/ |
| 6 Defense-aware attackers and real-data controls | DONE | EXPLORATORY.csv (CC rows), run_records/controls/ |
| 7 Verification, backup, handoff | DONE | VALIDATION.md (12/12 primary and 24/24 exposure agree; 4 reporting FAILs), ARCHIVE_INDEX.json (30,624/30,624; restore PASS), HANDOFF.json |

**Units.**
- Planned 378; completed 372.
- 6 not fitted because of 3 FARE aliases.
- 0 failed scientific units.
- 0 budget-unrun.
- Plus 14 real-data control runs (7 interface types × 2 datasets, seed 0).

**Primary family.** 12 rows, all decided: 10 PASS and 2 NOT_ESTABLISHED (Adult P3, P4).

**CPU (process user + sys, single-threaded workers):**

| Stage | CPU-s |
|---|---|
| Custody | 865 |
| Corrections | 38 |
| Exposure | 2,506 |
| Adult run | 1,066 |
| HMDA run | 1,820 |
| HMDA attempt 1 | 28 |
| Inference | 169 |
| **Total science** | **≈ 6,500 CPU-s ≈ 1.8 CPU-h** |
| Independent verification (4 replay runs) | ≈ 4,400 CPU-s ≈ 1.2 CPU-h |
| **Aggregate** | **≈ 3.0 CPU-h** |

FARE setup and compilation is recorded separately (notes/fare). The ceiling was 12 CPU-h.

**Memory peaks:**

| Job | Peak |
|---|---|
| Custody | 1.9 GB |
| Exposure | 2.7 GB |
| Adult run | 0.57 GB |
| HMDA run | 0.98 GB |
| Inference | 0.85 GB |

The ceiling was 6 GiB.

**Workers.** At most 2 scientific workers at a time, each with 1 BLAS thread.

**Storage.**
- Laptop: 1.7 GB new private data, against an 8 GiB cap; about 128 GB free.
- Drive copy: 30,624 files, verified.

**Unfinished.** None of the mandatory work. The optional kernelized adversarial erasure was not run.
