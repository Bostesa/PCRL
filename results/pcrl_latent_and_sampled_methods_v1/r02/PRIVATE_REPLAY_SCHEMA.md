# R02 private replay contract, frozen before ATT fits

One distinct arm/seed/release uses a new `.private/<unit>/` directory. R09 or
R14 supplies admitted ordered role data and R03's active queue guard. R02 never
loads a role. The outer worker binds `arm_id`, `model_seed`, `release_id`, exact
release/source hashes, the R00 lock SHA and the U0/Ucal source package paths and
file SHAs in `SOURCE_BINDING.json`. All paths in this file point to the actual
local artifacts; a branch name alone is insufficient.

## Primary reader custody

For each `r1`, `r2`, `pair`, inherited `save_fitted_slate` writes
`<view>/readers/READERS_RECEIPT.json`, `view_att.npz`, `view_inner.npz`,
`predictions_tune.npz`, `predictions_inner.npz`, and `model__<reader>.joblib`
or `.bin`. `READERS_RECEIPT.json.files` holds SHA-256 of each file; it also
binds the ordered row/group hash, full view value/schema hashes, frozen ATT
group split, reader order, TUNE validation/orientation, and zero-fit aliases.

`view_att.npz` has 4064 rows; `view_inner.npz` has 2235 rows under the current
admission. Required keys: `row_id` and `unit` as int64 vectors `(n,)`, `x` as
float64 `(n,d)`. Applicable keys: `x_defense_aware`, `x_metadata`,
`x_code_bits`, `x_raw_only` as finite float64 `(n,d_other)` and `cell_key` as
Unicode `(n,)`. `x` binds every actual transmitted field via the R02 schema's
`component_columns`. The categorical D code ID is computed from the actual
little-endian transmitted bit vector and occupies extra encoded columns bound
to the **bit wire component**; it is not another wire field.

`predictions_tune.npz` and `predictions_inner.npz` contain one float64
probability vector per frozen reader key, shape `(n_tune,)` and `(2235,)`.
`n_tune` is the actual row count from the inherited 1015-group TUNE split,
not a hard-coded 1015 rows. Primary D and A keys per view are
`LR_C1`, `MLP_64x32`, `HGB_15x200`, `DA_MLP_64x32`, `CELL_A1`,
`CODE_BIT_LR_C1`, `PROB_LR_C1`, and `PROB_HGB_15x200`. The pair file also
contains `IGNORE_R2__<local r1 reader>` and `IGNORE_R1__<local r2 reader>`;
each is an exact prediction alias of the local file, with zero new fit.
Family-specific specialists must be declared in R00's lock and included in
all three view files if applicable. `PROB_*` private model artifacts are also
written under `<view>/probability/` and copied into reader custody.

After all three reader receipts exist, R02 writes and fsyncs
`FROZEN_COMPLETE.json`, including their exact SHAs, the effective lock SHA,
and ATT/INNER complete-view hashes. Its SHA appears in the public attack unit.
The lazy `load_inner_sex()` callback is invoked **only after** this receipt.
The `INNER_SELECTION` SEX labels then score fixed predictions; they never fit,
orient, add or remove a primary reader.

## Endpoint and control replay inputs

The outer worker must privately preserve a role-aligned `ENDPOINT_INPUTS.npz`
for each arm/seed before publishing a task or interval result. Exact required
keys and dtypes: `row_id`, `unit`, `income_y`, `occupation_y`, `sex_y`,
`candidate_decision_1`, `candidate_decision_2` as int64 `(2235,)`;
`candidate_prob_1`, `U0_prob_1`, `Ucal_prob_1` as finite float64 `(2235,2)`;
`candidate_prob_2`, `U0_prob_2`, `Ucal_prob_2` as finite float64 `(2235,6)`.
Decision vectors must equal argmax of their actual candidate probability rows.
The reference arrays are exact restored U0/Ucal predictions on the same ordered
INNER people, not newly refitted convenience heads. A `ENDPOINT_INPUTS.json`
sidecar binds the NPZ SHA, arm/seed/release IDs, R00 lock SHA, class order,
source/reference package paths and SHAs, and the three reader-receipt SHAs.
R04 may consume equivalent R01 private role/reference custody if it verifies
identical ordered array hashes and documents that transport.

For each view, `<view>/controls/CUSTODY_RECEIPT.json` and
`probe_replay.npz` preserve the V2 real-data null/planted engineering copies,
labels, fit indices, predictions, source hashes, and two model artifacts.
The null uses independently group-shuffled INNER labels; its gate is AUC <=
0.60. The planted copy carries an artificial SEX column only inside the
engineering probe; its gate is AUC >= 0.99. Both fits follow primary freeze.

R04 replays all task metrics, every fixed-reader AUC and coalition alias,
control gates, and group-bootstrap margins without importing R02 or the
runner scorer. Any missing required key/hash is a technical incomplete unit.
