# Validation and verification

## Engineering, run before the science lock and only through the queue
- `SPG_ENGINEERING_TESTS_V1`: 16 pytest checks on synthetic inputs.
  - The slack QP matches SLSQP on 400 random problems; zero stays feasible; unused slack is spent.
  - A curved loss fails its first fraction and backtracks to 0.125.
  - Rejected steps restore the exact float32/float64 weights.
  - Budgets are absolute and never reset.
  - Chunked whole-pool passes are bitwise repeatable and agree with a single pass to float32 rounding; a sampled pool is detectably not the whole pool.
  - Population task weights are correct.
  - The old closed queue and the old trial lock cannot authorize this trial.
  - The compressed-income wire contract holds.
  - A synthetic end-to-end run of all five arms works (matched warm critics, budgets constant, checkpoints).
- `SPG_PREFLIGHT_V1` (real roles, no updates, no loss values reported).
  - Repeat-forward noise was 0, so the wire tolerance is 1e-10.
  - Chunked GUARD/UPDATE gradients are bitwise repeatable; chunk-versus-single relative difference is at most 3e-6 (float32 encoder).
  - Whole-pool passes cost 2–13 ms each.
- One code change was made after the engineering lock and before any fit: the trainer now passes the science-lock path in the queue's path+sha form. It is recorded in SCIENCE_LOCK.json, and the recipe did not change.

## Exposure controls
- The trainer's sources are UPDATE and GUARD exports only. It refuses any VALIDATION path.
- VALIDATION was opened once (`SPG_VALIDATION_UTILITY_V1`), after TRAINING_COMPLETE_LOCK was committed at HEAD and registered with the queue. The worker checks the committed bytes.
- INNER, ATT and old assessment arrays were never opened.

## Independent replay (`SPG_INDEPENDENT_STAGE_A_V2`, separate numpy code): PASS on all six checks
1. **Role partition and exposure:** recomputed from inherited arrays, with byte-identical exports.
2. **Frozen income:** 40/40 checkpoints bitwise equal to the source; the rank-1 release is decision-preserving.
3. **Task weights and proposal/critic accounting:** match.
4. **Whole-pool anchors, slacks and budgets:** recomputed at all 161 states per guarded arm per seed. No violation. The tightest margins were seed-1 SEPARATE_GUARD Brier at 3.4e-9 and its correct count exactly at the floor.
5. **Counterfactual scans:** a stored-vector check (the QP was not re-solved), with new probe forwards confirming wire changes.
6. **VALIDATION metrics:** reproduced within 5e-9. Selections and the FAIL gate classification are identical.

The first attempt, V1, had bugs in the verifier's own key names and was superseded. It found no runner discrepancy.

## Deployment and custody (runner runtime against stored canonical wires; custody, not independent arithmetic)
- 8 packages, with standalone and CLI output equal to the canonical wires (difference 0). Every rejection check fired. 8 bitwise epoch-0 alias receipts.
- A same-device snapshot of 171 files. A backup-alone restore re-executed 6 packages exactly and verified the role map and income basis.
- Off-device copy is pending: no writable external drive was found.

Stage B did not run, so there is no reader, resampling or criteria replay.
