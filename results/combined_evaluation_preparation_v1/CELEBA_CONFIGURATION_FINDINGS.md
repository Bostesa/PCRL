# CelebA encoder used by durable-guarantees: configuration findings (2026-10-02)

**Subject:** `fl-PCRL-main-checkpoints.tar::checkpoints/celeba_v2/final.pt`. sha256 `b0df3fb7fb3f5fa36e7a847d1fd2d16481f08389e76acdd81300648be4fd08fb`, 35,605,887 bytes, member mtime 2026-04-16T07:28:10Z.

**Inspection status: DONE (metadata only).**
- The drive `/Volumes/YOTUO` was reported not mounted earlier on 2026-10-02. It was then briefly disconnected and remounted that same day, according to the coordinator update.
- Only this one member was extracted, read-only, into session scratch, and the sha256 matched. Nothing on the drive was modified, and no copy was made in the repo.
- The checkpoint was loaded with `torch.load(weights_only=True)`, which succeeded. The restricted-unpickler path gave identical tensor groups. No forward pass was run and no data was touched.
- Machine-readable output: `notes/artifacts/celeba_v2_checkpoint_metadata.json`.
- Tool: `notes/artifacts/inspect_celeba_checkpoint.py`. It was tested on both cached Round-4 checkpoints (`notes/artifacts/round4_checkpoint_metadata.json`).

Sections:
- A: facts read from the checkpoint.
- B: facts from recorded sources, cited as `ref:path:line`.
- C: derived conclusions, with the reasoning shown.
- D: hypotheses.
- E: what remains undetermined.

## A. Recovered from the checkpoint itself

| Field | Value |
|---|---|
| Top-level keys | `encoder, task_heads, auditors, encoder_optimizer, auditor_optimizer, state, history, config` |
| Encoder class (by state-dict keys) | **FiLM `CNNEncoder`**: `purpose_embedding` [5,32]; `conv_layers.{0,1,2}` 3→32→64→128 (3×3); `film_gamma/film_beta.{0,1,2}`; `bn_layers.{0,1,2}`; `repr_proj.0` [256,8192]; `repr_proj.3` [128,256]. **No `purpose_projections.*` keys, so it is not `CNNPurposeProjectionEncoder`.** |
| repr_dim / purposes / input size | 128; 5; flat dim 8192 = 128×8×8, which implies 64×64 input |
| Task heads | smile_detection; age_estimation; expression_analysis (two sub-heads: Smiling and Mouth_Slightly_Open); attractiveness_prediction; gender_analysis. Each head is 128→64→2. |
| Auditors | One `MultiAttributeAuditor` group per purpose (5) |
| Attribute names present | Smiling, Young, Mouth_Slightly_Open, Male, Attractive (all in history keys) |
| Embedded `config` | `{"lambda_adv": 1.0, "auditor_steps": 10}`. That is the only config saved. |
| `state` | epoch 39 (0-indexed), so **40 epochs completed**; global_step 8000; best_val_loss 26.5595; patience_counter 5 |
| History | 40 entries per key. Best val_loss is at epoch index 34 (consistent with best.pt mtime 07:12 < final.pt 07:28). Final val_loss is 555.5, against a minimum of 26.6. |
| Final logged accuracies (val partition, eval mode) | task Smiling 0.520; Young 0.749; Mouth 0.528; Attractive 0.482; Male 0.559 |
| Final logged accuracies (train-time) | task Smiling 0.935; Young 0.791; Mouth 0.908; Attractive 0.563; Male 0.898 |
| Final logged auditor accuracies (val) | Male 0.533; Young 0.743; Attractive 0.548; Smiling 0.500 |
| Optimizers | AdamW; initial_lr 1e-3, **final lr 0.0**; weight_decay 1e-4 |
| Encoder optimizer state | **Present only for** purpose_embedding and FiLM γ/β (8000 steps each) and the task heads (1600 steps each). **There is no optimizer state for conv_layers, bn_layers affine or repr_proj (16 tensors).** |
| Auditor optimizer | All 104 tensors at step 16000 |
| BN `num_batches_tracked` | 42000 (all three layers) |
| Row indices / split arrays / data seed / normalisation statistics | **None.** There are no integer arrays (other than BN counters) and no split, seed or preprocessing fields. |

## B. Facts from recorded sources

1. **durable-guarantees loads the checkpoint as a FiLM encoder.**
   - `dg@956f5c8:experiments/celeba_extract.py:54` sets `CKPT = PCRL_ROOT/"checkpoints"/"celeba_v2"/"final.pt"`.
   - Lines 63-69 build `CNNEncoder(repr_dim=128, num_purposes=5, purpose_emb_dim=32, conv_channels=(32,64,128), dropout=0.3)`, call `load_state_dict(ck["encoder"])` (strict by default) and `eval()`.
   - It uses purposes `smile_detection`=0 and `attractiveness_prediction`=3 (`:55`), and labels Smiling, Attractive, Young and Male (`:56`).
2. **dg extracts every image of each split, without the random flip.**
   - dg uses all of the train, val and test splits (`celeba_extract.py:57,78-79`, with no `max_samples`).
   - It uses the deterministic test transform (`:80`).
   - Downstream, only the train partition is used: `dg@956f5c8:experiments/diagnostic.py:103-111` reads `data_cache/celeba/train.npz`.
   - Attackers use 75/25 stratified re-splits by seed (`diagnostic.py:134-136`; `utils/battery.py:53-56`), with sweep seeds [0,1] (`experiments/celeba_pipeline.py:69`).
   - The train-row count is 162,770 (`dg@956f5c8:results/celeba_coupling_scan.json:3`).
3. **The dg extraction finished.** The drive inventory lists `tree-durable-guarantees.tar::durable-guarantees/data_cache/celeba/train.npz` at 171,886,730 bytes (2026-07-03T18:14Z), plus val.npz (20,981,162) and test.npz (21,081,482).
4. **PCRL data module.** `a96ee0e10:pcrl/data/celeba.py` is unchanged through origin/main.
   - Attributes: `TASK_ATTRS = [Smiling, Young, Mouth_Slightly_Open, Male, Attractive]` and `SENSITIVE_ATTRS = [Male, Young, Attractive, Smiling]` (`:32-33`).
   - Transforms: Resize to 64×64 and Normalize with mean and std 0.5 on each channel; train adds RandomHorizontalFlip (`:37-51`).
   - Split: official `list_eval_partition` codes 0/1/2, and `max_samples` takes the **first N rows** of the partition (`:82-87`).
   - Five purposes, in the same order as the checkpoint's task heads (`:137-186`). Purpose index 0 is smile_detection and 3 is attractiveness_prediction, which matches dg's indices.
5. **The committed `run_celeba_v2.py` does not match the checkpoint.** It was first committed at `5b000e624` (2026-04-20), four days after the checkpoint mtime. It specifies:
   - `CNNPurposeProjectionEncoder` (`5b000e624:experiments/run_celeba_v2.py:67,147-153`);
   - `LAMBDA_ADV=0.5`, `AUDITOR_STEPS=5`, `EPOCHS=50`, `WARMUP_EPOCHS=5` and `MAX_TRAIN/VAL/TEST=10000/3000/3000` (`:80-97`);
   - `sequential_purposes=False` (`:197`) and `checkpoint_dir=checkpoints/celeba_v2` (`:198`).

   Its docstring says FiLM "fails on CelebA… the encoder collapses to constant output" (`:4-8`).
6. **The PCRL tables name two different encoders.** `5b000e624:results/celeba/celeba_main_table.csv:4-5` lists `PCRL_v1_FiLM` (4/13) and `PCRL_v2_projection` (6/13). In PCRL's own tables, "v2" means the projection encoder.
7. **The trainer saves almost no configuration.** `5b000e624:pcrl/training/trainer.py:1314-1331` saves only `lambda_adv` and `auditor_steps` as config, with no data, split or seed.
   - best.pt is saved on val loss (`:308-311`); epoch files every 10 epochs and final at the end (`:323-326`).
   - In sequential mode, `global_step` is incremented once per batch per purpose (`:481-485,525`), and `steps_per_epoch = len(train_loader) * n_purposes` feeds the cosine schedule (`:265-270,214-240`).
   - `trainer.py` is identical at `ab78f3217` (HEAD on 2026-04-16) and `5b000e624`.
8. **The run predates the committed CelebA code.** The history is `ab78f3217` (2026-04-14), then `5b000e624` (2026-04-20). `CNNPurposeProjectionEncoder` and `run_celeba_v2.py` first appear in 5b000e624 (`git log --all -S`), so the 2026-04-16 run executed uncommitted code.
9. **BUGREPORT.** `5b000e624:BUGREPORT.md:59` records "CelebA v2 | run_celeba_v2.py | generate_report() post-bug | April 16 | Affected".
   - The bug misaligned the *train-side* representations and labels in the post-hoc audit (`:23-46`). It is an evaluation-only bug and does not alter weights.
   - Test-side R² was unaffected (`:45-46`).
10. **The drive directory mixes several runs** (`fl-PCRL-main-checkpoints.json.gz`):
    - `celeba_v2/epoch_{50..120}.pt`: about 50.4 MB each, 2026-04-15 08:23–11:42Z (a different, ≥120-epoch run).
    - `celeba_v2/epoch_{10..40}.pt`, best.pt and final.pt: about 35.6 MB, 2026-04-16 04:35–07:28Z.
    - `celeba_v2_pretrain/{best,epoch_10..30,final}.pt`: 53.4 MB, 2026-04-16 02:22–04:08Z.
    - `celeba_smoke*`: 2026-04-16 02:00–02:16Z.
    - No log, argv or config member for any of these exists in any drive inventory or git ref. The search covered every PCRL/dg inventory for `celeba` log/txt members and every file with an mtime of 2026-04-15/17 outside checkpoints/.

## C. Derived conclusions

1. **The encoder is a FiLM `CNNEncoder` and not the committed `run_celeba_v2.py` model** (from A, B1 and B5).
   - Size cross-check (parameter arithmetic at the committed config):
     - `CNNPurposeProjectionEncoder` would hold 10.75M parameters, which is 43 MB of float32 for encoder weights alone. That is more than the 35.6 MB file.
     - The FiLM encoder holds 2.24M parameters. With its stepped AdamW moments plus heads, auditors and auditor optimizer, the observed tensor numel totals about 8.86M, which is about 35.4 MB.
   - The dg strict `load_state_dict` into `CNNEncoder` could only succeed because of this, and the extraction outputs exist (B3).
2. **The training configuration differs from the committed script.** It used `lambda_adv=1.0`, `auditor_steps=10` and **40 configured epochs**.
   - Epochs: the cosine LR reached exactly 0.0 at global_step 8000, so T_max (plus warmup) = 8000 = steps_per_epoch × epochs. With 200 steps per epoch, that gives 40 epochs.
   - The committed values are 0.5, 5 and 50.
3. **Purposes were trained sequentially.**
   - Each task head has 1600 optimizer steps, which is 1/5 of 8000. Under the trainer's all-purposes-per-batch mode every head would step every batch.
   - With sequential mode, 200 steps per epoch = 5 purposes × **40 batches**. At batch size 256 that means **9,985–10,240 training rows**, consistent with `max_samples=10000`.
   - Hence: **the encoder-fit rows are probably the first 10,000 rows of the official train partition** (B4, `head(N)`). They are a subset of the 162,770 rows that dg used for channel fitting and attacker splits.
   - The alternative reading is non-sequential training with about 51k rows. It is disfavoured because the heads then could not have 1/5 of the steps.
4. **The conv backbone, BN affine parameters and repr_proj were not updated in this run.**
   - These tensors have no AdamW state, so they never received a gradient.
   - Only the purpose embedding, FiLM and task heads were trained. The `celeba_v2_pretrain` run that finished at 04:08Z, about 27 min before the first `celeba_v2` epoch file, is the likely source of the frozen backbone (see D1).
5. **Eval-mode validation task accuracy at the final epoch is near chance for Smiling (0.52), Attractive (0.48) and Male (0.56).** Train-time accuracy is much higher, and val loss diverged (26.6 → 555.5).
   - dg extracted its representations in `eval()` mode. Whether eval-mode representations of the *train* partition carry the task signal was not checked here.
   - The dg "clean_lift" values (Smiling 0.116, Attractive 0.081; `dg@956f5c8:results/celeba_pipeline.json` `cells[*].cell`) are the only recorded indication.
6. **The checkpoint contains no information about rows, splits, seeds or normalisation.** All row roles must be reconstructed from `pcrl/data/celeba.py` (official partition, first-N rows) and the dg code (full train partition; 75/25 stratified re-splits; seeds 0/1, plus 0/1/2 for re-audit). No identity grouping is used anywhere: `identity_CelebA.txt` is not referenced by either loader.

## D. Hypotheses (not established)

1. **Pretraining.** `celeba_v2_pretrain/final.pt` (sha256 02d30bf1…, 53.4 MB) supplied the frozen conv backbone and repr_proj, and BN running statistics accumulated across both runs (42000 batches tracked compared with 8000 steps here).
2. **The script that ran was an earlier FiLM-based version of `run_celeba_v2.py`.** It would have had a pretrain stage, a frozen backbone, sequential purposes, λ_adv 1.0, K=10 and 40 epochs. Its failure (near-chance val accuracy, C5) motivated the projection-encoder rewrite committed on 04-20, whose docstring describes FiLM collapse.
3. **The 04-15 epoch_50–120 files (50.4 MB) belong to an earlier attempt that wrote into the same directory.** Their size (50.4 MB) differs from both the 35.6 MB files here and the 53.4 MB pretrain files, which suggests a different trainable-parameter or optimizer layout. This is unverified.
4. **The 16000 auditor-optimizer steps (2 per batch, against `auditor_steps=10`) reflect uncommitted trainer or script logic.** It is unexplained.
5. **`results/celeba/baseline_v2.csv` (train_time 1120.8 s) was not produced by this run**, which took about 3 h 20 min (04:08 → 07:28Z). It is probably a later projection-encoder run.

## E. Still to determine

1. **Pretrain link.** Do the conv, bn and repr_proj tensors equal those in `celeba_v2_pretrain/final.pt` or `best.pt`? This needs drive members `fl-PCRL-main-checkpoints.tar::checkpoints/celeba_v2_pretrain/{final,best}.pt` and a frozen tensor comparison, with no forward pass.
2. **Exact row count.** Is the training row count exactly 10,000 (first N of partition 0), and what data did the pretrain stage see? No log or argv survives, so this is limited to consistency checks: pretrain `state.global_step` and history length, and `celeba_v2/epoch_10.pt` step counts.
3. **Eval-mode task utility.** Do the frozen heads retain task utility in eval mode on the dg train/val/test representations?
   - This needs a frozen forward pass over CelebA images (`fl-PCRL-main-data-celeba.tar`, not local) or the dg rep caches on the drive (`tree-durable-guarantees.tar::durable-guarantees/data_cache/celeba/{train,val,test}.npz`).
   - Applying the *stored* heads is a frozen forward pass. Fitting new probes is not authorised.
4. **Earlier run.** Which run produced the 04-15 epoch_50–120 files (inspect one member's keys and config)? This matters only for provenance.
5. **Report outputs.** Neither the BUGREPORT-affected `generate_report` output of this run nor its stdout survives. Its train-side empirical auditor accuracies are unrecoverable, but they are not needed, because dg never used them.
6. **Paper statement.** For any paper statement: the AAAI CelebA encoder is an April FiLM CNN whose backbone was frozen in the final stage, trained on about 10k images. It is not the NeurIPS PCRL-V model (`results/v2_celeba_R5_*`), and not the committed `run_celeba_v2.py` projection model.
