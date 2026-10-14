# Source and exposure ledger

## Exposure statement (registered in PROTOCOL_SEAL.json before science)
"This campaign is exploratory development on repeatedly used Adult people. Source models already trained on all FIT. The previous FIT2 validation outcomes and study-level results are exposed. This successor was designed in light of those failures. Excluding FIT2 from this continuation's optimization does not make it fresh confirmation. Intervals are conditional on the fitted models and readers and do not account for the full adaptive research history."

The internal fitting budgets (+0.004 NLL, +0.002 summed Brier, 0.5% accuracy, KL 0.001) were chosen knowing that the previous guard overfit its fitting pool. They are outcome-informed development budgets, not the original criterion or a bound.

## Observed source facts (verified in queue unit MTA_ADMISSION_V2, `ADMISSION.json`, status PASS)
| item | observed |
|---|---|
| Start checkpoints, RAW_JOINT strength 0.3 epoch 5 | seed 0 `a1a0dfba…4a4c`, seed 1 `b10a389a…7bdf` (hash-verified) |
| Folded own heads | checkpoint head tensors equal the bound geometry head hashes. The predecessor's recorded RAW_PROJ and RAW_HYBRID wires reproduce **bit-exactly** on all FIT (15,434), ATT (4,064) and INNER (2,235) rows, both seeds. Those wires contain own probabilities, and the predecessor admission checked folded-head-vs-sklearn parity to ≤2e-12. |
| Previous campaign's stored RAW_PROJ FIT2 release | reproduced bit-exactly, both seeds |
| Class orders / raw83 schema | [[0,1],[0..5]]; 83 named float32 columns, no forbidden names |
| h64 | `enc.1` modules 0–3 output (input of the 64→16 map); `enc.1.4(h64)` equals `encode(1,x)` bitwise |
| Occupation head rank | 5 both seeds; ‖A Bᵀ B − A‖∞ = 2.2e-16 (S0) and 1.0e-15 (S1); A Bᵀ has full column rank 5 |
| Trainable residual | 5×64 + 5 = **325** parameters |
| Hidden scale s | mean‖h64‖² = 21.38 (S0) and 16.12 (S1), so the per-coordinate RMS is 0.58 / 0.50 < 1 and **s = 1** in both seeds (the rule only scales down) |
| Zero-residual witness | the raw83 adapter path equals inherited RAW_PROJ **bitwise** on every admitted role, both seeds; the cached path equals the raw path |
| Income | factorized RAW income projection, rank 1, byte-identical in every service and checkpoint |
| Roles | inherited UPDATE 8,180 / GUARD 2,043 / FIT2 5,211 (exact subsets of FIT with identical features and units); ATT 4,064 rows / 4,061 groups; INNER 2,235 / 2,234; groups disjoint across roles; duplicate groups (FIT 6, ATT 3, INNER 1) have identical features |
| Prior guard checkpoints | SEPARATE_GUARD and GLOBAL_UPDATE_GUARD epoch 5, both seeds: hashes from the previous TRAINING_COMPLETE_LOCK. Income tensors are bitwise equal to the start; occupation tensors and head changed |
| GLOBAL_UPDATE_GUARD own-head projection | geometry rebuilt from that checkpoint's own current head (rank 5). It preserves the checkpoint's centered logits within 1.3e-14, probabilities within 2e-12 and decisions exactly, on all roles |
| Repeat-forward noise | 0, so the frozen wire tolerance is 1e-10 per block |

## Reconstructed facts
- The INNER label paths are the inherited role exports. Their hashes were checked only after REPORTING_LOCK was committed, and they equal the committed public hashes: income `7c4d6eb9…`, occupation `b631efaf…`, sex `ed5f69bc…`.

## Missing dependencies
- None for this campaign's fits. The previous campaign had no GLOBAL_UPDATE_GUARD deployment package, so this campaign built one from the hash-verified checkpoint.

## Class 5 limitation
Class 5 has 0 GUARD rows, 2 UPDATE rows and 2 FIT2 rows. The six-class task and order are unchanged. Class-5-specific statistics are not estimable, and nothing here claims protection for unsupported classes.
