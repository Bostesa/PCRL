# M1 admission and capacity handoff

## Admitted inputs

The focused study reuses the portfolio's 25 S4-admitted role-only NPZ files in the prior coordinator worktree. `admission.py` checks their byte hashes, exact ZIP member schemas, ordered row-to-group identity and 83-column feature-name identity. It opens only `row_id`, `unit` and `feature_names` members; it does not open X, task labels, SEX or the mixed-role source. The public [M1 data manifest](DATA_ACCESS_MANIFEST.json) records counts and hashes. It reports zero row and group overlap among FIT, CAL, ATT and INNER. S4's independent source-ID audit reports zero excluded-ID overlap. Prior mechanical export did transiently deserialize full mixed-source X and selected labels twice before permitted-row slicing; no excluded values entered scientific files.

Scientific workers must use the prior `shared/safe_data_guard.py` `SafeRoleStore` with the original portfolio `DATA_ACCESS_MANIFEST.json` and an M0 effective study lock. The M1 report is an admission receipt, not fitting permission. The old assessment and HEAD_VALIDATION labels remain barred.

## Capacity and mask API

`capacity.capacity_geometry(admitted_feature_names)` returns five feasible mask records and all 25 ordered pair records. Each mask has:

- `mask_id`, `mask_sha256`, `source_schema_sha256`;
- canonical `atoms` and source-order `column_indices`;
- numeric, categorical and total source-group counts;
- actual `encoded_columns`, `source_fields`, `capacity_signature`;
- `historical_exposure`, distinguishing FIT proxy search from prior final-head exposure.

The primary profile retains all four standalone numeric atoms and four of five categorical atoms, with at most 77 encoded columns. The `education_fact` atom couples `education-num` with the 16 education one-hot columns. Each categorical block is indivisible. Five masks are feasible, at widths 42, 66, 75, 76 and 77. This is a **maximum-width budget**. It does not make those five actual releases equal in transmitted columns. Exact numeric/categorical group counts plus exact encoded width yield one mask per width and would make selection comparisons degenerate.

Every pair record includes ordered recipient mask hashes, source-order union columns, union group/column counts, overlap counts and `union_sha256`. Twenty distinct-deletion pairs have the full 83-column union. The five shared-deletion pairs have reduced unions of unique widths; an equal-union-width control can legitimately select the identical masks and tie. A same-union raw-data equivalence is distinct from a complete-release alias because head outputs may differ.

The profile was chosen from grouping geometry and prior **task-only** C evidence. `C-SIMPLE-TASK-SHARED1` selected native-country deletion using OSF_DEFENSE_FIT task labels without SEX and its INNER_SELECTION task metrics were point-feasible against U0/Ucal. Those are reused development rows, and this rationale does not use historical SEX recovery. No fallback profile is registered by M1.

`validate_release_input` checks exact selected float32 values against the source-order 83-column matrix and refuses partial groups, reordered columns or mask hash changes. `assert_role_alignment` checks private row/group order for both recipients. `reconstruct_union` checks equal disclosed values on overlapping columns and can compare the reconstructed union to an admitted full input. Row and group IDs are evaluation custody fields and are never transmitted.

M2 and M3 should consume the same five mask records and 25 ordered pairs. M4 must independently bind the actual emitted raw values, calibrated head outputs, decisions, masks and public model metadata in its complete views. The full-input F head and its calibration are a shared protocol/evaluation identity to be fixed by M0/M2/M4, not inferred from M1's role hashes.

## Reproduction

From `m1/`, with the prior portfolio package path supplied explicitly:

```sh
python3 -m unittest test_capacity.py
python3 build_evidence.py --portfolio-package /path/to/prior/results/pcrl_method_portfolio_v1
```

The builder writes the two public JSON files once and refuses different bytes at the same names. No science fit or attacker fit is performed.
