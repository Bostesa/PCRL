# Local objects and compact evidence

Repository root: `/Users/nathansamson/PCRL`. This study's local root is `results/redesign_20260909_acs_coalition_strength_v1/`. Raw people, PCA/release arrays, fitted weights, optimizer states, candidate predictions and caches are deliberately excluded from Git. They remain available for exact replay on this machine.

[REUSE_MANIFEST.json](REUSE_MANIFEST.json) binds historical path/hash identities and original exposure records. Its `systems` entries distinguish 18 historical final paired systems from 36 new systems, without copying old archives. Shared I anchors count once. Direct E, prior and exposed-target controls reference their original evidence and fitted objects. The manifest also binds original source/score-replay records; old fits do not acquire newly edited source hashes. No required artifact was regenerated during preparation.

Each new training unit has `seed_N/training/CONDITION/fork.pt`, `final.pt`, `training.json`, `frozen_releases.npz`, `release_freeze.json`, and an immutable `complete.json`. The full historical fork/Adam/schedule/RNG is verified on load and again in read-only training replay. The global `RELEASE_MANIFEST.json` binds all 54 functions before new reserved-task fitting.

Each new evaluation unit has `seed_N/CONDITION/` containing fitted utility candidates, `audits/` with all candidate lineage and actual terminal checkpoints, selected and candidate `predictions.npz`, frozen `releases.npz`, native source probabilities, compact metric/selection/parity records, local file hashes and immutable completion. The `local_artifacts.json` and completion hashes identify precisely which omitted files are needed for exact prediction replay. Original `split_rows.npz` identities are preserved per seed.

Large score/lineage exports may be published using deterministic compression, with plain/compressed hashes, while duplicate plain exports remain local. These exports contain derived metrics, not raw person records. Historical archives, raw ACS data, fitted `.pt`/`.joblib` objects and `.npz` arrays are not included in the commit. Reproduction instructions distinguish compact report regeneration from exact local model replay.

The new per-unit `metrics.json` and `audits/audit_selection.json` are published as exact deterministic `.json.gz` siblings; their unmodified original JSON bytes remain local. [COMPACT_UNIT_EVIDENCE.json](COMPACT_UNIT_EVIDENCE.json) records both hashes and byte sizes. The documented restore command reconstructs only these derived score/selection files, allowing report regeneration from the public evidence without person-level artifacts.

The ACS source is `data/folktables/2018/1-Year/psam_p06.csv`, SHA256 `dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`. Initial free storage was about 20 GiB; the projected new condition evidence was 2.44 GiB, with a separate 2 GiB allowance for training/report artifacts. No unrelated files were deleted for space.
