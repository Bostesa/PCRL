# Reader and reporting unit preparation

The coordinator executes queued jobs. This helper creates immutable private
configuration and queue-spec files; it does not start numerical workers.
Run it with the project Python environment so prepared jobs use that environment.

## Frozen source index

`ARM_SOURCE_INDEX.json` is private and contains:

```json
{
  "source_manifest": {"path": "...", "sha256": "..."},
  "science_lock": {"path": "...", "sha256": "..."},
  "attack_lock": {"path": "...", "sha256": "..."},
  "instances": [
    {
      "arm_id": "G_STRICT",
      "seed": 0,
      "point_eligible": true,
      "attack_law": {"path": "...", "sha256": "..."},
      "inner_law": {"path": "...", "sha256": "..."},
      "utility_rows": {"path": "...", "sha256": "..."},
      "policy": {
        "type": "allocation",
        "directory": "...",
        "receipt_sha256": "..."
      },
      "policy_pins": [{"path": "...", "sha256": "..."}],
      "deps": ["completed-policy-unit-id"]
    }
  ],
  "additional_utility_inputs": [
    {"arm_id": "ineligible-arm", "seed": 0,
     "pin": {"path": "...", "sha256": "..."}}
  ]
}
```

Every arm needs both source-seed instances. Fresh reader fitting is prepared
only if both pass the frozen original-reference point gate. All unavailable
comparisons remain registered. Additional utility inputs preserve inference
for arms whose reader stage did not trigger; do not duplicate eligible inputs.

Policy types are `P0`, `historical`, `allocation`, `template`, and `control`.
Historical policies also specify `seed` and `source_arm` (`V_J` or `V_S21`).
All new policy receipts and packages belong in `policy_pins`.
Default fit IDs are `R5_FIT_{arm_id}_S{seed}_V1`; explicit `reader_unit_id`
overrides are supported and must be admitted by the coordinator's stage lock.

## Commands

```sh
python -m conditional_release.build_attack_units --phase fit \
  --index ARM_SOURCE_INDEX.json --output PREPARED_FIT_INDEX.json
python -m conditional_release.build_attack_units --phase report \
  --index PREPARED_FIT_INDEX.json --reporting-lock REPORTING_LOCK.json \
  --output PREPARED_REPORT_INDEX.json
python -m conditional_release.build_attack_units --phase inference \
  --index PREPARED_REPORT_INDEX.json --endpoint-family PRIMARY_ENDPOINT_FAMILY.json \
  --output PREPARED_INFERENCE_INDEX.json
```

Each prepared entry identifies a private queue spec. Enqueue that spec with
`conditional_release.scheduler enqueue`; the coordinator starts queue hosts.
Fit units reserve 600 CPU seconds and 600 wall seconds. Reporting units reserve
240/240; paired inference reserves 1800/1800. These are ceilings, and actual
CPU use is recorded by the common scheduler.

## Label boundary and custody

The reporting lock must contain `inner_sex_pin` as a path/hash pair, the same
`inner_sex_sha256`, and every global `reader_freezes` receipt. Preparation
validates all frozen receipts before hashing or copying the one-role INNER
SEX archive. The exact archive is then copied into campaign-private custody,
with its original hash and reporting-lock binding. Reporting configurations
use this verified copy. The role-safe source manifest continues to exclude
INNER SEX permanently.

Reporting itself verifies every global receipt again before opening labels.
The 552 prospective endpoint IDs remain fixed, including missing dependencies.
No reader fitting, orientation, or nominee selection uses INNER SEX.
