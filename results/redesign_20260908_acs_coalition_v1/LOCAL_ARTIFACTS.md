# Local objects and public evidence

The fitted experiment lives at `/Users/nathansamson/PCRL/results/redesign_20260908_acs_coalition_v1/`. Person-level arrays and fitted weights are intentionally local. Public reproduction instructions are in [REPRODUCTION.md](REPRODUCTION.md); the exact execution dependencies and historical starting identities are in [protocol_freeze.json](protocol_freeze.json), [PREFIT_IDENTITY.json](PREFIT_IDENTITY.json), and the explicit [operational amendment](EXECUTION_AMENDMENTS.json).

Each completed seed has three complementary records:

| Record | Meaning |
| --- | --- |
| `seed_N/local_artifacts.json` | Local paths and SHA256 hashes for new model weights, optimizer states, release arrays, selected/candidate predictions and fitted preprocessing |
| `seed_N/parent_provenance.json` | Exact reused historical inputs and fitted objects, retaining their original source identities |
| `seed_N/complete.json` | Completion record binding the compact seed evidence and local-artifact manifest |

Completed local manifests: [seed 0](seed_0/local_artifacts.json), [seed 1](seed_1/local_artifacts.json), [seed 2](seed_2/local_artifacts.json). Reused-reference provenance: [seed 0](seed_0/parent_provenance.json), [seed 1](seed_1/parent_provenance.json), [seed 2](seed_2/parent_provenance.json).

Training states are under `seed_N/training/`: genuine initialization, common source warmup, interface-specific observer warmups and each condition's exact fork/final state. The unused historical objects remain read-only. Candidate models and their actual terminal optimizer/RNG states are under each condition's `fitted/` and `audits/fitted/` directories. Nested validation-selected checkpoints are separate from actual last-iterate checkpoints. `releases.npz`, `predictions.npz`, `native_predictions.npz` and `split_rows.npz` stay local; they are not part of the public package.

The original local ACS file is `data/folktables/2018/1-Year/psam_p06.csv`, SHA256 `dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`. Historical PCA maps, E teachers and source-head tensors are loaded through verified parent manifests; no new teacher, cohort, shuffle or historical model was fitted for packaging.

Public per-seed `metrics.json`, selection records, support counts, training curves, gradient diagnostics and projection/composition parity records retain compact numerical evidence. Large derived CSV/JSON exports use deterministic gzip with plain-content hashes in [COMPACT_EXPORTS.json](COMPACT_EXPORTS.json). These are score/lineage tables, not archives of raw records or fitted objects. Full local model replay needs the hashed local artifacts; compact report regeneration needs the public evidence and compatible historical score records.

The preserved first operational failure and recovery records identify exactly what was loaded again. Completed learned models and auditors were not retrained, and their original files were not overwritten. This distinction matters for both provenance and the measured compute total.
