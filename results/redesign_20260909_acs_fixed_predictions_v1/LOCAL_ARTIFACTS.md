# Local fitted objects and omitted person-level arrays

All exact local paths, sizes and SHA256 hashes are in [LOCAL_ARTIFACTS.json](LOCAL_ARTIFACTS.json.gz). Original reused dependencies retain their own identities in [REUSE_MANIFEST.json](REUSE_MANIFEST.json); they are referenced, never recopied into a historical-looking archive.

The local repository root is `/Users/nathansamson/PCRL`. Required omitted objects are:

- Existing ACS person data `data/folktables/2018/1-Year/psam_p06.csv`, SHA256 `dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`; original sampled cohort/PCA/household caches remain at the reuse-manifest paths.
- Nine original selected source model directories under `results/redesign_20260908_acs_protection_v1/seed_s/fitted/transfer/E_pca/task/candidate`, including their fitted standardizers/class order and saved selected weights. The original selections and exact predictor-specific paths are in ANCHOR_PARITY and PREFIT_IDENTITY. Eight are MLPs; seed0 employment is logistic. No new choice or fit replaced them.
- Original initialized source-head/mapper tensors and the three selective-study E map objects, referenced by exact hash. No teacher refit or shuffle redraw occurred.
- New per-seed `anchors.npz`, `teacher.npz`, `pca.npz`, `split_rows.npz`; these contain person-level values and stay local.
- New `seed_s/training/initial.pt`, `observer_initial.pt`, `fork.pt`, each condition's checkpoints/releases and `B_trajectory.local.pt`. These preserve model/Adam/buffer/RNG histories, public auxiliary heads and the shared B witness.
- New per-system `fitted/` and `audits/fitted/` candidate directories, including actual last120/360 model/optimizer/RNG checkpoints and separate validation-best candidates; `predictions.npz` contains selected and candidate person-level probability arrays.
- Local process stdout and startup working-tree details. Compact process status/failure records are published separately.

None of those binary fitted objects or person-level arrays is committed. Compact candidate metadata, losses, confusion/support tables, curves, selections, exposure and prediction/model hashes are published. [EXPORT_MANIFEST.json](EXPORT_MANIFEST.json) binds lossless compressed exports to their original bytes. It is not a substitute for the omitted fitted objects. Restore commands recover only compressed aggregate evidence; exact inference replay requires the local objects above. Missing original anchors cannot be replaced by a new best predictor while retaining the exact-service claim.
