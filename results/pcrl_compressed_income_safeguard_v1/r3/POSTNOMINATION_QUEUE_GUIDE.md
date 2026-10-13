# Independent metadata preparation after nomination

`compressed_income_safeguard_independent.prepare_postnomination` reads JSON receipts and code only. It carries NPZ, tensor, model and update-log hashes from queued receipts. ROOT alone enqueues and runs the generated units. Rebuild not-yet-enqueued specs with `--replace` after a guard amendment changes source pins; queued identities are immutable.

Run the metadata helper with the admitted compatible Python executable:

```text
python -m compressed_income_safeguard_independent.prepare_postnomination nomination
python -m compressed_income_safeguard_independent.prepare_postnomination freeze_targets
python -m compressed_income_safeguard_independent.prepare_postnomination refit
python -m compressed_income_safeguard_independent.prepare_postnomination release_replay
python -m compressed_income_safeguard_independent.prepare_postnomination final
python -m compressed_income_safeguard_independent.prepare_postnomination K240
```

The helper's inputs default to R4's actual metadata files. Override them with `--package-index`, `--nomination-input`, `--nomination-lock`, `--target-registry`, `--freeze-receipt`, `--report-index` or `--runner-summary` if ROOT documents another identity. `--amendment` binds a specific already registered verification amendment; otherwise the latest verification amendment is selected.

| Phase | Required completed inputs | Generated units and evidence |
| --- | --- | --- |
| nomination | R4 `CIS_NOMINATION_V2` config, 32 trajectory/bank receipts and two exact RAW_HYBRID starts | One independent selection replay, using FIT2 `p1/p2` and all 54 ATT pair curves. |
| freeze_targets | R4 `FINAL_PACKAGE_INDEX_V2.json`, 18 selected package banks and the ROOT nomination lock | One independently recomputed ATT target freeze, producing exactly 54 immutable target attempts. No new freeze after the reporting lock. |
| refit | Queued target-freeze receipt and its exact `TARGETS.json` hash | Eighteen service/seed shards, three fresh registered attempts each. Their target objects are copied unchanged. All 54 must finish before ROOT opens INNER truth. |
| release_replay | R4 final package index, admitted features and original source checkpoint pins | All 18 models forwarded independently on FIT, ATT and label-free INNER. Income aliases, unchanged occupation aliases and selected epoch-zero occupation aliases are compared literally. |
| final | Committed reporting lock and completed runner report input index | All 216 task clauses, complete 18/54 reader envelopes, ATT/INNER real controls and ROOT gate vectors. Independent row metrics and control vectors are emitted as private queue outputs. |
| K240 | Completed independent final worker and completed runner inference | Shared paired duplicate-group bootstrap using the exact frozen combined registry, independent row metrics and independent gate vectors. Every point, bound and status is compared with the runner. |

The package index carries `manifest`, `nomination_lock`, `reporting_registry` and 18 entries. Entries carry `service`, `arm_id`, `seed`, `epoch`, `checkpoint`, `geometry`, `trajectory`, `freeze`, `release`, `scores`, `att_release`, `att_scores`, `fit_release`, `fit_scores` and model-file pins. No numeric artifact is opened by preparation.

Phase plans are written privately under `r3/GENERATED_<PHASE>_V1.json`, with configs and queue specs in the adjacent `configs/` and `specs/` directories. The refit target registry pin is carried from the queued freeze receipt. K240 independent NPZ pins are carried from the queue's `output_sha256` records, rather than recomputed by this helper.

The selected source epochs remain the ROOT nomination: CONSTRAINED, ORDINARY and LOCAL_TWIN at epoch 0; TASK_ONLY at epoch 5. Actual training completion and nonzero accepted proposals are separate from selected service epochs. The metadata helper never changes that selection.
