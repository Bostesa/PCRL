# R2 raw-union accounting correction — 2026-10-09

Status: prospective engineering correction before any `r2_raw_union`,
`r2_attack_fit`, or `r2_privacy_score` real-data unit. The source five-mask bank,
selected pair rules, reader types, data roles, utility and privacy criteria,
and maximum of six canonical unions per seed are unchanged.

The earlier R2 proposal described `U` as the number of distinct canonical raw
unions among selected pair IDs. A complete attack fits RAW_ONLY readers on each
recipient's selected raw feature view as well as the coalition union. An
off-diagonal selected pair can require two local diagonal mask unions even
when neither diagonal pair is selected by a control rule. Thus the required
union set is the union of:

1. each selected pair's coalition `union_sha256`;
2. the diagonal union SHA for its income mask; and
3. the diagonal union SHA for its occupation mask.

Each exact union receives one two-reader ATT fit per seed and is transported
to all matching local or coalition views only after row, column, source-value,
model and saved-prediction checks. Across this frozen bank, `U <= 6`, so the
previous upper bound `2*U` reader fits per seed remains valid. The output
inventory and selected-view lineage are explicit in `raw_union_worker.py`.
