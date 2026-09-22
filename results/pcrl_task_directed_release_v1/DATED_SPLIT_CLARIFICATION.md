# Dated split clarification

Written 2026-09-22T02:41:25.531193+00:00.

This is a late public index of the existing owned, inherited split artifacts promised in `DATA_USE.md`. It records their present file hashes and original saved pool assignments; it neither creates a new split nor changes any fit-time assignment. It must not be represented as a contemporaneous preregistration.

Only the seven numeric pool row-index members of each anchor’s `split_rows.npz` were read. No features, labels, predictions, losses, models, raw CSV, or 2016/2017 artifacts were opened. `SPLITS.json` contains the file hashes, canonical ordered and sorted row-index hashes, every within-anchor pairwise check, and the complete cross-anchor overlap tables.

All three anchors contain the same 30,000 distinct person-row indexes. Within each anchor, its seven pools are pairwise disjoint, and no pool contains duplicate row indexes. These checks establish person-row disjointness only. No household overlap or household count is inferred from person indexes.

| Pool | Anchor 0 | Anchor 1 | Anchor 2 |
|---|---:|---:|---:|
| representation_fit | 10513 | 10428 | 10551 |
| source_validation | 3004 | 3010 | 2979 |
| downstream_fit | 4539 | 4521 | 4464 |
| downstream_validation | 2984 | 3041 | 3024 |
| attacker_fit | 2993 | 3021 | 2941 |
| attacker_validation | 2985 | 2964 | 3062 |
| test | 2982 | 3015 | 2979 |

Anchor-specific pool assignments differ. The following counts compare a test pool with the union of all six non-test pools in the other anchor; that union includes inherited source validation. These are person-row role overlaps, not measurements of which specific labels a fitted model consumed.

| Test anchor | Other anchor | Test rows | Assigned to other non-test pools | Assigned to other test pool |
|---:|---:|---:|---:|---:|
| 0 | 1 | 2982 | 2696 | 286 |
| 0 | 2 | 2982 | 2705 | 277 |
| 1 | 0 | 3015 | 2729 | 286 |
| 1 | 2 | 3015 | 2717 | 298 |
| 2 | 0 | 2979 | 2702 | 277 |
| 2 | 1 | 2979 | 2681 | 298 |

| Test anchor | Test rows | In any other anchor’s non-test pool | In test for all three anchors |
|---:|---:|---:|---:|
| 0 | 2982 | 2949 | 33 |
| 1 | 3015 | 2982 | 33 |
| 2 | 2979 | 2946 | 33 |

The registered test seal concerns loader and scoring access for the requested anchor: its fitting loader excludes its test members, and its test scoring requires the frozen-selection permit. It does not mean that the test people are globally unseen by the other anchors. A person assigned to one anchor’s test pool can be assigned to another anchor’s fitting or validation pool. Exact label access also depends on the declared role splits and missing-label masks, which were not opened in this metadata audit. The study therefore cannot describe these splits as globally unseen people or globally untouched labels.

The three anchors remain dependent development analyses of the same inherited 2018 cohort. This disclosure does not change the frozen estimands, role assignments, fitted artifacts, or protocol. Shared-household inference remains a separate analysis using actual household identifiers; the row-index overlaps below cannot substitute for it.

| Anchor | Owned source relative to the study directory | File SHA-256 |
|---:|---|---|
| 0 | `private/inputs/results/redesign_20260909_acs_fixed_predictions_v1/seed_0/split_rows.npz` | `669fcff80bf61b3d8f17b7cb07d676a2b38030fe638942bfa4bae90f12311f8a` |
| 1 | `private/inputs/results/redesign_20260909_acs_fixed_predictions_v1/seed_1/split_rows.npz` | `d8543a6857e68adcbb2345781eba135a850463285a77a76b42b5c3e701233d1e` |
| 2 | `private/inputs/results/redesign_20260909_acs_fixed_predictions_v1/seed_2/split_rows.npz` | `07d9b48e99a05f30396c8dd46d2a3e7b8bd182ca9e8ef55baa55604530e65038` |

Ordered row hashes use SHA-256 over the saved row sequence converted to C-contiguous little-endian signed 64-bit integers, without a header or separator. Sorted hashes use the same encoding after ascending sort. The source files were hashed before and after their numeric members were read and were unchanged. No row-index values or person identifiers are published.

Per-anchor household counts and new RF teacher/mechanism subrole hashes are outside this artifact-only index and require separate disclosure from accepted `PREPARED.json` receipts.
