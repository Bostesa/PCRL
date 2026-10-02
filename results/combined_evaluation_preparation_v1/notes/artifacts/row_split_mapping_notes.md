# Row/split mapping notes (artifact role, 2026-10-02)

Companion data: `row_index_verification.json` (`regenerate_row_indices.py`), `forward_pass_version_check.json`
(`forward_pass_version_check.py`), `round4_checkpoint_metadata.json`, `../../ARTIFACT_TO_EVALUATION_MAP.csv`,
`pilot_cell_selection.csv`. Nothing was fit. Only counts, index-array sha256 values and match flags were written; no row-level values.
The YOTUO drive was briefly disconnected and then remounted on 2026-10-02. The dg score arrays used here had been extracted the
day before and were re-verified against the inventory sha256.

## Verified facts
- **The PCRL loaders are identical on every ref.** The blobs of `pcrl/data/{adult,hmda,diabetes,base}.py` (ee8218d, 4474317,
  8cadb2e, 74c6ab1) are the same at be88a825e, a3875c618, 55e4cb1d1, HEAD and the working tree. `git log --all` shows no change
  after 2026-04-26. So durable-guarantees' unpinned PCRL_ROOT cannot have changed the rows.
- **Adult PCRL train.** The 32,561 rows of adult.data are permuted with `np.random.seed(42)`, and the first 26,048 go to train.
  `dropna` runs after the split, which leaves **24,145 rows** (val 6,017; test 15,060). The "n=26,048" in the reconciliation
  D-02/D-04/D-12 text is the pre-dropna count. durable-guarantees actually used 24,145 rows: the tpr59 assessment vectors have
  6,037 = 25% rows and match sequence-for-sequence. The Round-4 checkpoint's global_step gives 95 steps/epoch at batch 256,
  which is consistent with this.
- **HMDA.** `data/hmda_processed/{train,val,test}.npz` (63,747 / 13,660 / 13,661) regenerate with equal features from the
  local `hmda_raw/hmda_2023_ca.csv` using the committed `experiments/prepare_hmda.py` (blob d26adaf; filters, then
  RandomState(42) permutation, 70/15/15). Row identity is the position in the filtered frame. The AWS Round-4 run re-downloaded
  the CSV on 2026-04-29, and whether its rows equal this snapshot cannot be verified. The checkpoint's step count (250/epoch) is
  consistent.
- **Diabetes (PCRL).** One encounter per patient (`drop_duplicates(patient_nbr)`) is kept before a per-class RandomState(42)
  70/15/15 split. 71,509 patients after dedup; 71,506 after dropping unknown gender. The grouping unit is patient, and no
  patient crosses splits.
- **durable-guarantees attacker split.** `train_test_split(test_size=0.25, random_state=s, stratify=attr)` for s in 0, 1, 2,
  on rows in loader order (shuffle=False). The regenerated assessment labels equal the stored `y_ps{s}` sequence in
  `tpr59_scores/adult_noise_s0.npz` and `hmda_noise_s8.npz` for all 3 seeds. **The stored score files carry no row-index key**,
  only aligned vectors. Their alignment is now established by sequence equality, not by equal length alone.
- **Roles in the frozen-encoder dg cells.** These cover the encoder training rows, defense fit, defense validation, the
  attacker split, the LoRA attacker and the utility probe.
  - Encoder training rows = the PCRL train split, which is the same rows durable-guarantees audits.
  - Defense-fit = all PCRL-train rows (projections, fit_projection seed 0). Noise has no fit.
  - Defense-validation = none: every operating point was chosen on the same rows.
  - Attacker-fit = 75%, with an internal 10% early-stopping validation for MLP only. Assessment = 25%. Both are subsets of
    the defense-fit and encoder-training rows, so they are not held out from either.
  - The Exp 2-4 LoRA attacker is trained and scored in-sample on all rows.
  - The utility probe is a fitted LogisticRegression on a 75/25 split. Frozen task heads exist in the checkpoint but were
    never used.
- **PCRL val/test splits.** These are held out from encoder training and are regenerable, but durable-guarantees never used
  them.
- **Frozen forward pass.** It is bit-identical on CPU across PCRL b96c412, a3875c618, 39c5a84, 71d6a0f9d and 5f162ab37
  (H sha256 22846f04… Adult, fe3b33f0… HMDA). The Round-4 checkpoints carry no LEACE buffers. The original durable-guarantees
  runs used MPS, so bitwise equality with the original releases is not established.
- **Grouping units.** Adult and HMDA have none recorded. The durable-guarantees folktables split is person-level stratified,
  with SERIALNO not used. diabetes_hospital is encounter-level, and patient_nbr is not used (its presence in the fairlearn frame
  was not verified). CelebA has identities available but they are not used.

## Pilot rule result
Two cells are eligible: (1) DG-R4A-SEX-INCOME and (2) DG-R4H-RACE-LOANDEC. They are ranked by artifact completeness; see
`pilot_cell_selection.csv`. Both are chosen, with Adult first: it has more stored per-row outputs and more regenerable releases,
and it is also the smaller cell. In both cells only the untreated and noise releases can be produced without a defense refit.
The documented projection cases exist as stored per-row scores (Adult) or as aggregates (HMDA), but their projection matrices
were never saved.
