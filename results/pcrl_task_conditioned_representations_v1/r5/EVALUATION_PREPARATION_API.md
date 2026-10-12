# Evaluation metadata preparation

`prepare_evaluation_units.py` creates private configurations and queue specifications using metadata only. It reads JSON and hashes code/JSON, copies the already admitted archive/checkpoint pins, and never deserializes arrays/checkpoints or executes jobs. R00 owns every lock, nomination, point-gate decision, fallback choice and queue operation.

Utility preparation accepts R00's operational index with `manifest`, `science_lock`, `nomination_lock` pins and all 40 canonical `instances` with private checkpoint paths. The public lock must be FROZEN, contain nonempty nominees and exactly match every arm/seed/checkpoint association. The 22 nominee instances include the two additive H_RAW_CSTAR references; 18 FIT-preselected fallback instances retain distinct `*_DIAGNOSTIC_FALLBACK` IDs. Every arm ID has source seeds 0 and 1.

Executed metadata command:

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m task_conditioned_representation.prepare_evaluation_units --phase utility --operational-index .private/task_conditioned_representation/root/NOMINATION_OPERATIONAL_V1.json --attacks-amendment results/pcrl_task_conditioned_representations_v1/ATTACKS_UTILITY_LOCK.json --python /Users/nathansamson/PCRL/.venv/bin/python --output .private/task_conditioned_representation/r5/PREPARED_UTILITY_ORACLE_V1.json
```

The prepared index lists `utility` and `oracle` records, each with exact config/spec pins, private output path and public receipt path. It does not launch either unit. Utility reserves/caps 600 seconds; the label-only oracle reference reserves/caps 120 seconds. Both use stage attacks, explicit ATTACKS_UTILITY_LOCK amendment and 2 GiB maximum RSS. Every scheduler source record has only `path` and `sha256`.

Later reader preparation accepts a separate R00 operational index with `manifest`, `science_lock`, `attack_lock`, `instances` for exactly 22 actual feature arm/seed banks, and optional `legacy_source_index` pointing to the admitted four historical P0/S12 metadata records. The final release/endpoint lock must have status RELEASES_AND_ENDPOINTS_FROZEN, exact checkpoint associations in `instances`, and four P0/S12 identities in `legacy_instances`.

Reader-index instance metadata must preserve `point_eligible` from the original nominee's common two-seed gate, `diagnostic_only`, `is_nominee_or_exact_alias`, `exact_source_alias`, `selection_kind` and `nominee_checkpoint_sha256`. Choosing a FIT-preselected fallback cannot promote an ineligible nominee. The helper copies R00's decisions and performs no gate-based selection. H_RAW_CSTAR remains an additive historical descriptive reference outside the original primary pool.

Future metadata command, to execute only after R00's final reader lock:

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m task_conditioned_representation.prepare_evaluation_units --phase readers --operational-index ROOT_READER_OPERATIONAL_INDEX --attacks-amendment ROOT_FINAL_ATTACKS_LOCK --python /Users/nathansamson/PCRL/.venv/bin/python --output .private/task_conditioned_representation/r5/PREPARED_ALL_READERS_V1.json
```

This prepares 22 continuous banks, at most 726 fresh scientific/control reader fits, plus four separate literal historical score-wire banks, at most 120 fresh fits. Each bank reserves/caps 1200 seconds. Exact source aliases remain explicit. No reader is reused from previous studies. INNER SEX is absent from these configurations and source lists; its known custody hash remains metadata until R00 freezes all readers and opens the separate reporting stage.
