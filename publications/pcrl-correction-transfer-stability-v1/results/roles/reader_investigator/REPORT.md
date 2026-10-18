# Reader investigator: report (tasks 1-2)

Saved by the coordinator at 19:55Z from the reader investigator's returned final text (the subagent harness blocks hand-written
report files). Findings in detail: `SCREEN_FINDINGS.md`; generated tables: `../../screen/`.

## Task 1: development reader grid (`cts/devreaders.py`, committed)
- Unchanged from the source grid: cells (R1 / R2 x reader seeds k0 / k1; FIT 50% / VALIDATION 20% / SCORE 30% of MODEL_SELECT
  households), the 16 recipes, winner selection on VALIDATION, bootstrap (B 400, seed 2026101806 + 100 x release seed + repetition
  index). MODEL_SELECT SEX is read only through `cts.common.load(operation='readers')`.
- Cache key verbatim from ohn/devreaders.py; own cache first, then the source cache read-only; new entries only in this campaign's
  cache with checksum and read-back; source hits, own hits and new recipe fits counted separately.
- R3: the source's extra split (tag OHN1_devgrid_R3_diag, bootstrap index 2); the source's R3 readers reused at full FIT where they
  exist (seed 0 RR_SPLIT3 members only), others are new fits. Nested learning curves (1/4, 1/2, 1 of reader FIT) as in the source.
- Label-free manifest (`DEV_GRID.json`, R1 / R2 household sets byte-identical to the source) and write-once code lock
  (`DEVGRID_CODE_LOCK.json`); real runs use the project .venv interpreter (numpy 2.4.2 = the source lock).
- Selftest 30 / 30 (invented data; source code writes cache entries that the new module reads as source hits, bitwise).

## Task 2: screen study (`cts/screen_study.py`, committed; selftest 24 / 24)

| Unit | What | CPU-s | Fits |
|---|---|---|---|
| CTS_SCREEN_SIM_V1 | L1 simulations, no data | 845 | - |
| CTS_SCREEN_L0B_V1 | identical releases, refitted readers | 920 | 3,072 new; 120 source + 48 own cache hits |
| CTS_SCREEN_L2_V1 | semi-synthetic SEX shift | 460 | 1,536 grid + 682 calibration |
| out of queue (charged) | part A, report, selftests, label-free checks | 200 | - |

Findings: see `SCREEN_FINDINGS.md` (items 1-6). In addition: the income view is the bottleneck of the recipient screen (its
between-split SD, 0.021, is 2.4 times its bootstrap SE).

## Open issues reported
- L2 at the boundary would need a newly registered law (slate-calibrated plants or a smaller eta); nothing post hoc was run. The
  explanation for why the slate reads the plants so strongly (flexible local readers decoding each person's SEX-determined
  displacement) is untested.
- L0b measures reader-fitting noise only; information-equivalent but non-identical wires were not studied.
- The code lock is write-once; any change to cts/devreaders.py needs a new versioned lock.
- No commits, pushes or stashes by the investigator; no ATTACK_FIT, ATTACK_VALIDATION or DEV_REPORT access; no person-level arrays
  printed.
