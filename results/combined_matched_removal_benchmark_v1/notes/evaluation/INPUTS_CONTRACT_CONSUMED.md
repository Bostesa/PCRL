# INPUTS_INDEX fields consumed by the evaluator (`bench.Inputs`)

Schema: `bench_v1.inputs_index/v1`, from role 1. Paths may be absolute or relative to the index. Every file listed below
is sha256-verified on load, and pinned in `LOCK.json`.

## Per dataset

**File pairs**

| Path field | Hash field |
|---|---|
| `labels_npz` | `labels_sha256` |
| `roles_npz` | `roles_sha256` |
| `rows_npz` | `rows_sha256` |
| `features_npz` | `features_sha256` |

`rows_npz` and `features_npz` are pinned but not read.

**`labels_npz` keys read**

- `row_id`, `split`, `unit`, `role`.
- `canon_key`, used as the record key for role disjointness and the HMDA hash rule. The fallback is `record_key`.
- The sensitive attributes by name.
- The task labels at `purposes[p].labels_task_key` (for example `task_income`).

Rows whose `role` is not one of {defense_fit, attacker_fit, attacker_val, assessment} are dropped first. In practice these
are `excluded_dup` and `unused_train`. The dropped counts are recorded.

**`roles_npz`**

- `<role>__row_id` and `<role>__unit` are cross-checked against the per-row role and unit arrays.
- Any disagreement is a refusal.

**`purposes[p]`**

- `index` must equal the frozen PCRL b96c412 table.
- `rep_key`, `logits_key` and `labels_task_key` are also read.

**`defense_route`**

`PREFERRED` maps to `historical_train`; `FALLBACK` maps to `fallback_30pct`. The route is checked against the registered
hash rules by `--dry-run`.

**`encoders[k]`**

- `checkpoint` and `checkpoint_sha256`, used by U1: a `weights_only` load with a hash check.
- `lineage_status`: a unit runs only if this is ADMITTED, VERIFIED or OK. Otherwise the seed's units are reported missing,
  with the exact file and reason.
- `forward_npz` and `forward_sha256`, read through `row_id`, `split`, `rep_p<i>` and `logits_<purpose>`. The arrays are
  joined to the labels by `row_id`, never by position. The forward-cache split must agree with the labels split.

**Synthetic only**

`historical_native_json` and `historical_native_sha256`. Real runs read the pinned git blobs instead.

## Real-data dry-run (2026-10-02, no fits)

| Check | Adult | HMDA |
|---|---|---|
| Hash mismatches | 0 | 0 |
| Tier-1 units admitted | of 126 total: all 126 runnable | |
| Role counts (defense_fit / attacker_fit / attacker_val / assessment) | 24,127 / 7,571 / 2,239 / 5,250 (pilot counts match) | 63,704 / 6,794 / 2,089 / 4,778 |
| HMDA role hash-rule agreement | — | 1.0 |
| Supported sensitive classes | sex {0, 1} | race {0, 1, 2} (3 and 4 NE) |
