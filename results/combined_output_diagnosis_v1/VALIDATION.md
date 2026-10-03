# Validation

## Before any new fit

| Check | Result |
|---|---|
| Custody (`CUSTODY.json`) | 18/18 admitted inputs and checkpoints match their hashes; 372 reused units (2,286 files) intact. Encoder forward pass replayed from the checkpoint is bitwise equal to the cache; a release head and an attacker replay exactly (both datasets). |
| Repairs R1–R3 | Regression tests pass. Role row sets are unchanged (hashes asserted). Ledger: `ORIGINAL_VS_REPAIRED.csv`; `AMENDMENT_R_2026-10-03.md`. |
| Protocol review | The read-only reviewer's 14 required changes were incorporated before the lock (`notes/review/STATS_REVIEW.md`). Among them: the ignore-offset bank as FC's comparator; the zero-by-construction rule; declared aliases and NOT_ESTIMABLE slots; the saturation thresholds; log-softmax log loss; the S5 logic. |
| Lock | v1 at 01ef342f was pushed before any fit. It pins the code, dependencies, admitted inputs, reused-unit hashes, the 30/34/6/4 families (asserted) and z values to 6 decimals. Amendment L1 (c677ec2) adds only the stage-5 code and the canonical utility, before any stage-5 fit. |
| Fixtures | Synthetic tests: an offset carrying the sensitive bit while the margin carries the task; a margin carrying everything; a constant head; exact invertible recoding; multiclass centring; saturation; the canonical utility (doctest); family counts and z. All pass. |

## After the runs

| Check | Result |
|---|---|
| Units | 559 unit directories complete and hash-verified. 0 failed scientific units, 0 budget-unrun. All 333 timestamped fitted units were completed after the lock. |
| Real-data controls (fit / validation only) | Shuffled-label nulls 0.47–0.52 (none flagged); planted leaks detected at 0.91–0.97. Adult and HMDA, full and centred surfaces. |
| Exactness (`EXACTNESS.json`) | Round trip bit-exact; argmax = 1[d > 0]; σ(d) matches softmax to ≤ 1.7e-16; no float64 saturation on the primary binary heads. |

## Independent replay

`INDEPENDENT_VERIFICATION.json` and `verification/replay_odx.py`. An import guard refuses `odx`, `oar`,
`stored_model_eval` and `report`. The replay uses its own roles, support, surfaces, banks, weighted Mann–Whitney AUC,
bootstrap, SE, bounds, decisions and flags.

**Final result: 81 PASS, 6 INFO, 0 WARN, 0 FAIL.**

| Family | PASS | NOT_ESTABLISHED | NOT_ESTIMABLE | Agreement with the runner |
|---|---|---|---|---|
| Primary (30) | 13 | 3 | 14 | exact (difference 0.0 on points, SE and bounds) |
| S3 (34) | 23 | 11 | — | exact, including flags |
| S4 (6) | 6 | 0 | — | exact |
| S5 (4) | 3 | 1 | — | within 1e-9 after amendment S1 |

**What else the replay checked:**
- Aliases: FC/CH-adult-pair0-1 equal FC/CH-adult on every replicate.
- All 105 bank selections recomputed from validation log losses.
- The stage-5 screen, grid, feasibility and nominees.
- 18 sampled attacker units and all 51 cell-conditional attackers reproduced from surfaces rebuilt from the logits
  alone.
- Two per-person replays per dataset (frozen RNG), plus every constant or collapsed case. The HMDA fair_lending head is
  constant on all seeds, and its hard-surface attackers score exactly 0.5.
- Deployed inputs depend on logits only.
- Identical rows across the two S4 purposes.
- Exposure rebuilt as 17 / 42.

## Defects found during verification, and how they were handled

1. **S5 scoring omitted the registered plus-surface alias rule** (a FAIL in the verifier's first S5 pass). Repaired as
   amendment S1. Values moved slightly; **no decision changed**: FZ − FARE stays NOT_ESTABLISHED, with its lower bound
   going from 0.0001 to 0.0019.
2. **S3 table reporting gap.** The flags and percentile columns were missing. Added: IDENTICAL_BY_CONSTRUCTION for the
   constant fair_lending head; NEAR_BOUND for the employment and education usefulness rows, with Bonferroni-tail
   percentile columns.
3. **Wording.**
   - The FARE certificate refusal counts distinct vectors, not rows (418 / 266 / 360 rows).
   - The A1 premise text.
   - "Recoding of the occupation column": no single input column determines the task.
   - The descriptive 90 % z is now exact.
   - The reviewer's float32 saturation count used 16.6, not 24 ln 2.

## Backup and restore

- The drive copy `<drive>/private_odx_v1_20261003` holds 3,487 files. All were re-read uncached (F_NOCACHE; a cold
  unmount was not performed) and all match.
- Restored from the drive copy alone, a stage-5 FARE tree encodes to identical cells, and the head on FARE features and
  an attacker reproduce their saved outputs exactly (`PRIVATE_BACKUP_INDEX.json`).
- Small files written after the backup (logs, the pre-S1 S5 table, inference metadata) were re-synced and re-verified
  before handoff.
- Nothing was deleted.
