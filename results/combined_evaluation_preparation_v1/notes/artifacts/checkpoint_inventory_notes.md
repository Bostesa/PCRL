# CHECKPOINT_INVENTORY.csv: build notes (2026-10-02)

Builder: session scratch `build_ckpt_inventory.py`. It reads the drive inventories, re-hashes local copies and stats laptop files. Nothing was trained and nothing was read from S3. The drive /Volumes/YOTUO was briefly disconnected and then remounted on 2026-10-02. After the remount I confirmed with ls that the archive tars exist (fl-PCRL-main-checkpoints.tar, fl-PCRL-main-results-ignored.tar, wt-PCRL-superpowers-laftr-hard-r2-2026-05-17.tar). I did not extract anything from them. Class (a) stays inventory-based.

## Rows: 246

| class | rows | meaning |
|---|---|---|
| a | 138 | On drive, present in the inventory. The sha256 comes from the inventory. |
| b | 57 | Local copy, hashed today. See below. |
| c | 30 | Referenced only by an S3 object path. Status: "not verified — S3 not inspected". |
| d | 21 | Never persisted, or never run. |

The 56 class-b rows are:
- **30 extracted drive members, each re-hashed and equal to the inventory.**
  - celeba_v2/final.pt (b0df3fb7…, 35,605,887 bytes), extracted by the CelebA role after the remount.
  - R4 adult/hmda s0 final.pt: 2 copies each, in `~/PCRL_eval_cache_private` and in scratch.
  - 9 LEACE erased files.
  - 27 INLP eval_reps.npz.
- **18 cross-purpose retrain checkpoints from the laptop-only `checkpoints_archive/`.**
  - Hashed today. My hashes match the backup role's 18/18.
  - No independent reference hash exists for them.

No hash mismatches. Every expected member is present.

## Corrections and surprises
1. **The cross-purpose constrained retrain checkpoints are on the laptop, not only in S3.**
   - Location: `/Users/nathansamson/PCRL/checkpoints_archive/{cross_purpose_ab,cross_purpose_diabetes}/v2_*_s{0,1,2}/{best,final}.pt` (18 files, mtime 2026-05-31).
   - They are git-ignored (`.gitignore:17 *.pt`) and appear in no drive inventory, so this is a single copy.
   - The reconciliation's "referenced archive only" status for these is wrong.
2. **The erase-pilot, VICReg and rank-8 checkpoints were never persisted (class d, not c).** Each launcher (`infra/{erase_pilot,erase_pilot_diabetes,erase_vicreg_sweep,erase_rank8_diabetes_cpu}/user_data.sh`) syncs only `results/` dirs. `d3921121:infra/cross_purpose/user_data.sh:237-240` states that the erase pilot "lost its Adult checkpoints because this block synced only results/". The `s3://…/erase_*/checkpoints/` path in missing_files.csv is a guess, and no launcher wrote it.
3. **The LEACE baseline files hold erased input features, not representations.** `results/{ds}_LEACE/erased_*.pt` are dicts `{train, test}`; Adult is (24145,105)/(15060,105). There is no encoder behind them.
   - Adult train n = 24145 here, while durable-guarantees uses n = 26,048 (an 80% split of adult.data). This is a row/split question for the mapping role.
4. **Folktables R2 seed 2 was never saved.**
5. **Two metric gaps:**
   - Diabetes R5 and R6 have no final.pt dominant-axis audit (metrics "partial").
   - The celeba_v2 metrics cannot be attributed. `results/celeba/baseline_v2*.csv` (5b000e624) may come from this run but are not tied to the checkpoint.
6. **The per-dim Lagrangian full-run checkpoints have only an inferred S3 path.** The drive holds only `_smoke`.
7. **LAFTR-hard checkpoints (class c):** the S3 path is taken from the launch-#3 user_data. The Adult rows come from launch #2, whose archive destination was not re-verified.

## Excluded
- 258 `epoch_*.pt` files in fl-PCRL-main-checkpoints (including 12 in celeba_v2) are counted here but not listed.
- Pre-v2 directories (adult/, har*, celeba*, marstat, cond_*, sweep*, synthetic, v2_smoke, probes: R1R2/OVR/RANK24/FOLKTABLES probes, ERASE/CROSSPURP_ERASE smokes) are not listed, because no evaluated lineage uses them.
- The September ACS redesign fitted models are out of scope.
