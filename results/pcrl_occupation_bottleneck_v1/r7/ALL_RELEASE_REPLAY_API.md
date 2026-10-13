# R7 all-release replay worker

Code: `occupation_bottleneck_independent/release_replay_worker_r7.py`

SHA256: `db2c416b2e51a2eb0626b0f0323231385f532d57e2c9d813f2b54c52ba67ee95`. Static AST validation completed. Numerical execution belongs exclusively to the ROOT queue. Frozen independent arithmetic remains `replay.py`, SHA256 `5a845905f1f7d29c950b66cf1fb05f49d4b08e66b8b4f3db852e3c5086d9a242`.

## Queue API

Stage `verification`; command `python -m occupation_bottleneck_independent.release_replay_worker_r7 --spec CONFIG.json`.

Use the existing V2 release replay config with a fresh `unit_id`, fresh `public_output`, and required `canonical_packages` file pin to ROOT's `CANONICAL_PACKAGES.json`. Its existing `operation` field is ignored. Source registrations must include config, both independent code files, manifest, canonical roster, all three feature exports, and every checkpoint and role wire pin. No label file is registered or needed. Pins have `{path,sha256}`; all numeric pins come from existing receipts.

Each of exactly 72 package entries has `arm_id`, `seed`, `release_variant` (`full` or `head_row_space`), `checkpoint`, `att_release`, `fit_release`, and `inner_release`. The canonical JSON has its existing `entries` schema. Exactly 36 arms have seeds 0 and 1; 66 packages are full and 6 projected. Optional existing `rule` object must preserve the three independent frozen rank/tie values.

## Checks and output

The worker reconstructs all 216 package-role wires, equivalent to 432 recipient forwards, from actual checkpoint encoders and heads. It validates raw83 feature names/order, role counts, disjoint rows/groups, inherited group-hash FIT fold 2, canonical package bindings, all 16 parameter shapes and precisions, and the 21,288 parameter count. Projected releases independently reconstruct the factorized SVD row-space projection under the frozen rank and sign rule.

Full representations require zero absolute difference; full centered logits/probabilities require absolute difference at most 2e-12. Projected components require at most 1e-10. All decisions must be identical. Output includes per-role component maxima and global component maxima; it contains no individual arrays or private paths.

Saved ATT archive schema is exactly row_id/unit/r1/r2/sex/train/validation. Only the first four members are opened. The entire authorized ATT archive is hashed for registered-file custody, including unopened label/split archive bytes. FIT2 and INNER wire archives contain only those four members. INNER labels are neither read nor hashed. No new fit occurs, and scientific methods are imported only from the independent namespace.

The ROOT fallback worker remains separately preserved and pinned; this worker does not modify it.
