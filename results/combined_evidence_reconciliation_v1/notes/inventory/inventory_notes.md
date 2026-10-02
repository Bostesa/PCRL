# Role A — evidence inventory notes (2026-10-01)

Scope: both repositories (Bostesa/PCRL, Bostesa/durable-guarantees), every remote and local branch, git bundles on the YOTUO drive, the drive archive inventories (2026-09-25 and 2026-09-30 relocations), and the laptop. Nothing was trained, downloaded from S3, deleted or merged. Members were read with `tar -xOf` into scratch only. Companion files in this directory:
- `evidence_inventory.csv` (81 records; record_id prefixes P- = PCRL core, R- = PCRL May rebuttal-era, D- = durable-guarantees)
- `branch_inventory.csv`, `missing_files.csv`, `chronology.csv`
- `checkpoint_inventory.csv`: every .pt checkpoint member with sha256
- `s3_references.csv`
- `durable_drive_raw_members.csv`: 739 untracked durable-guarantees raw-array members with sha256
- `recount_support/`: the recount script and its JSON output. These are reconstructed evidence: our own counts from stored outputs, not original artifacts.

## EARLY FLAGS (consolidated)

1. **The AAAI paper's encoders are not the NeurIPS headline encoders.** durable-guarantees loads `PCRL_ROOT/checkpoints/v2_{adult,hmda}_s0/final.pt` and `celeba_v2/final.pt` (dg@956f5c8:utils/pcrl_io.py:55,167,170; experiments/celeba_extract.py:54) and does not pin a PCRL commit.
   - sha256 1cfc2fef… and e29d0367… are `fl-PCRL-main-checkpoints.tar::checkpoints/v2_adult_s0/final.pt` and `::checkpoints/v2_hmda_s0/final.pt`. Both are **PCRL v2 Round 4 seed 0**, not Round 5. Verified by extracting and re-hashing: epoch 204, and the saved lambdas equal `results/v2_{adult,hmda}_ROUND4/per_seed_results.json` seed 0.
   - b0df3fb7… is `::checkpoints/celeba_v2/final.pt` (2026-04-16), an April CNN run from `experiments/run_celeba_v2.py`. That script was committed later, at 5b000e624; the same commit's BUGREPORT lists CelebA v2 as affected. This is not the NeurIPS PCRL-V model.
2. **The NeurIPS headline numbers recount exactly from stored per-pair rows** (status: independently recomputed): 56/60 strict, 7/60 cleanly compliant, τ sweep 56/52/37, Round 4 46/60, held-out seed 3 6/8, INLP 43/60, SPLINCE 60/60→3/60.
   - The headline uses `dominant_axis_audit.json` r2_onehot at **final.pt**.
   - Other circulating counts use different metrics: the a00d5749b verdict (R5 47/60), `per_seed_results.json` linear_r2 at Cotter **best.pt** (R5 Adult 21/24), and the rebuttal aggregators' "paper headline" baseline of **54/60 strict / 5 clean**. The manuscript says 56/60 / 7.
   - Per-seed task accuracies come from best.pt, while R²/health come from final.pt. The paper mixes checkpoint rules.
3. **The LAFTR hard-R² 60-cell head-to-head finished, and its full per-seed outputs exist, but only on the drive.** Location: `wt-PCRL-superpowers-laftr-hard-r2-2026-05-17.tar::laftr-hard-r2-2026-05-17/results/laftr_hard_r2_{adult,hmda,diabetes}_LAFTR_HARD_R2/` (untracked in the relocated worktree).
   - Recount: Adult 0/24, HMDA 17/18, Diabetes 18/18 = 35/60; every summary.json says COLLAPSED. GitHub holds only the code and the Adult smoke.
   - In that comparison, the LAFTR-Q HMDA/Diabetes rows are copied from the submitted paper's Appendix Q Table 13, and the PCRL row is frozen. Neither is measured.
   - Checkpoints existed only in the 7-day-lifecycle bucket and are presumed gone.
4. **The original LAFTR baseline (Table 1 15/60; App. P/Q) is only partly supported.**
   - Adult per-seed results are on GitHub and recompute to 0/24.
   - The HMDA and Diabetes rows survive only as joined rows in `results/inlp_benchmark/inlp_results.json`. Their source, `s3://pcrl-bios-overnight-20260504/laftr_benchmark/FINAL_BENCHMARK.csv`, was in the lifecycle bucket.
   - The "discriminator at chance on 56/60" claim has no located artifact.
5. **Single-copy, laptop-only evidence.** These files are untracked (not git-ignored), so they are in no branch and no drive archive:
   - `results/rebuttal/erase_layer_pilot_aws/` per-seed files (erase pilot 60/60; also the input to the per_dim_std diagnostic and the VICReg comparison)
   - the cross-purpose constrained retrain `results/v2_*_CROSS_PURPOSE_*/results.json`
   - `results/rebuttal/cross_purpose/`

   The git-ignored `infra/*` launchers for the May runs are likewise laptop-only. The v2 Round 4–7 user-data (`/tmp/round5/userdata_*.sh`) was never preserved; each final.pt embeds its config dict instead.
6. **The rebuttal "22/33 → 8/33" compares different models.** The 8/33 comes from new constrained erase-layer retrains at best.pt; it is not a re-evaluation of the submitted encoder (which used final.pt).
   - Under the absolute criterion the result is 26/33 → 19/33.
   - On Diabetes, best_epoch = 0, so the deployed model is the LEACE-projected initialisation.
   - The executed wrapper `/tmp/cp_analysis/unified_protocol.py` is gone; a committed mirror exists.
   - The cross-purpose constraint run (paper 19/24, 12/15) used 75 epochs and `canonical_iterate.pt`, with seed 0 at 0/75 feasible epochs. The paper does not disclose this.
7. **"Held-out" means different things, and selection shares its data in both lineages.**
   - PCRL "held-out seed 3" holds out the seed only: same split, and the R² audit is fit and scored on the test representations.
   - durable-guarantees picks every operating point on the same PCRL-train rows its attackers are split from. The only held-out check is the fresh-partition run: 0 flips over 9 arms, max |Δ| 0.0133, recounted.
   - CelebA PCRL-V reports train-set R² only. The same run logs val R² of 0.156–0.171 with 0 feasible epochs; inconclusive given a d/n null bias of about 0.125.
8. **durable-guarantees 67/59/8.** The count is recounted from stored flags. However, 4 of the 5 new-cell "survivors" (noise σ=8) have stored Tier-2 max of 0.552–0.577, above the 0.55 bar. The fail rule uses Tier 1 only (`experiments/expansion_reaudit_paperframing.py:114`).
   - TPR@1%FPR and AUC for all 59 failing configs are independently recomputed from the drive-only per-person score arrays (`analysis/tpr59_scores/*.npz`, 57 files, sha256-verified). They match exactly.
9. **The durable-guarantees "blind-reviewer" objections answered on Jul 23–24 are simulated persona reviews,** not official ones: `6021b4c:analysis/yus_extras/simulated_reviews_v28.md`.
   - The PCRL May "rebuttal" branches (May 18 – Jun 5) all PREDATE the NeurIPS reviews (Jun 22–26).
   - The FAccT ablations were registered on 2026-07-24 (ecaeba46f), the day the reviews were released, but **never executed**: only CPU smoke checkpoints exist, and 15dbcc3 says "nothing launched".
10. **Not completed vs not on laptop.**
    - **Not completed:** FAccT ablations 1–3; the VICReg Diabetes arm (10 h cap); the final.pt dominant-axis audit for Diabetes R5/R6 (checkpoints exist on the drive).
    - **Lost to the S3 lifecycle (outcome unknown):** LAFTR-hard launch #1 and the first VICReg launch.
    - **Unsupported by located artifacts:** the LAFTR-official raw run dirs (`../laftr/experiments`).
    - **Not on laptop but on the drive:** all checkpoints, durable-guarantees raw arrays (≈964 MB), INLP encoders and representations, LAFTR-hard per-seed outputs, the CelebA data/rep caches.
11. **Branches.** All 106 heads in the drive bundles exist as local ref tips and are reachable from GitHub; **no bundle-only branch exists.**
    - Local `crosspurp-constraint` (+10) and `laftr-benchmark` (+2) look unpushed, but their commits are on GitHub via origin/main (anonymized supplementary cleanup, 2026-05-07).
    - The only local-only object is `refs/stash` 9e0b333 (WIP NEXT_STEPS.md, 2026-05-19).
    - origin/main@55e4cb1 deleted paper claims/data on 2026-08-19, so cite earlier commits such as a3875c618 or 17ef7d449. Likewise, the durable-guarantees files deleted on 2026-08-02 remain at 6021b4c.
12. **Durable S3 coverage.** The durable bucket `pcrl-ux-archive-ed9d21fd` covers only the September ACS lineage; no May-2026 experiment appears in its manifests.
    - Open item: the `pcrl_utility_extension_v1` "exec_main" chunk (2,545 files from the main checkout @ad2c0887) might contain a second copy of the laptop-only rebuttal results. Its per-file ledger is private and was not checked.
    - The submitted AAAI LaTeX source is not in the repo or on the drive. The only source is the Aug-30 Overleaf zip in ~/Downloads, which postdates the Jul-29 build.

## Branch inventory summary
- **PCRL:** 39 origin branches plus origin/HEAD.
  - Primary NeurIPS lineage: main, laftr-benchmark, overnight-bios-rank-mech, bios-pcrl-layer12, crosspurp-constraint, erase-layer-pilot, erase-layer-vicreg-sweep, diabetes-rank8-ablation, laftr-hard-r2, rebuttal-evidence, cross-purpose-rebuttal, fix/retire-accuracy-guarantee, and ablations-facct (registration only).
  - The ACS-release lineage is 26 `research/pcrl-*` branches from Sep 2026, all based on the vicreg-sweep tip 39c5a84.
  - Meta: `research/combined-*`.
  - Unrelated: `claude/frosty-khayyam`.
- **durable-guarantees:** main 956f5c8 and archive-cleanup e702c4b (fully merged). The relocated laptop repo's refs on the drive equal GitHub.


---

## Slice P — PCRL core runs and checkpoints (fork notes)


Scope: v2 Rounds 4-7, held-out seed 3, CelebA PCRL-V, INLP, SPLINCE, BIOS layer-12 / head-aware LEACE, LAFTR per-purpose baseline, CROSSPURP constraint, FAccT ablations, checkpoints. Git cited at origin/main@55e4cb1d1 unless stated (files still present there; introducing commits listed in core_evidence.csv). Drive = /Volumes/YOTUO/NathanSamson-Mac-relocated-2026-09-30/archives/.

### EARLY FLAGS

1. **The AAAI (durable-guarantees) Adult and HMDA encoders are the PCRL Round 4 seed-0 checkpoints, not the NeurIPS headline Round 5 ones.**
   - sha 1cfc2fef… = `fl-PCRL-main-checkpoints.tar::checkpoints/v2_adult_s0/final.pt` (mtime 2026-04-28T17:52Z).
   - sha e29d0367… = `::checkpoints/v2_hmda_s0/final.pt` (2026-04-29T01:48Z).
   - I extracted both and re-hashed them; the hashes match.
   - The untagged directory is Round 4. Each checkpoint has `state.epoch=204`, and its saved `lambdas` dict matches `results/v2_{adult,hmda}_ROUND4/per_seed_results.json` seed 0 `lambdas_final` digit for digit. Round 5 ends at epoch 199.
   - durable-guarantees loads these paths: dg@956f5c8:utils/pcrl_io.py:55,167,170 and results/falsification_attack.json:3.
2. **The AAAI CelebA encoder is not the NeurIPS PCRL-V model either.**
   - sha b0df3fb7… = `::checkpoints/celeba_v2/final.pt` (2026-04-16T07:28Z), from the April CNN script `experiments/run_celeba_v2.py`.
   - That script was committed four days later, at 5b000e624 (2026-04-20). BUGREPORT.md in the same commit lists "CelebA v2 … April 16 … Affected" (an evaluation bug).
   - The directory mixes epoch files from 04-15 and 04-16.
   - The NeurIPS PCRL-V model is `results/v2_celeba_R5_*` (final.pt in results-ignored.tar).
3. **The NeurIPS headline numbers recount exactly from stored per-pair rows (independently recomputed).**
   - Paper metric: `dominant_axis_audit.json` r2_onehot at final.pt.
   - R5 Adult 23/24, R5 HMDA 16/18, R7 Diabetes 17/18 = 56/60.
   - Cleanly compliant: 3+2+2 = 7/60.
   - τ sweep 56/52/37. Mean R² 0.0136.
   - Round 4: 20+11+15 = 46/60.
   - Caveats:
     - Commit a00d5749b reports different "strict" counts for the same rounds (R4 32/60, R5 47/60), so more than one metric version was in circulation.
     - `per_seed_results.json` linear_r2 is a different quantity again: Cotter best.pt / training-time. R5 Adult gives 21/24 on it.
4. **The reported checkpoints are mixed.**
   - The paper says it reports the final iterate. R² and health come from final.pt (epoch 199).
   - Per-seed task accuracies (`task_accuracies`, reused in task_acc_vs_unconstrained.json) are training-time values at the Cotter-selected best.pt. R5 Adult best_epoch is 158/188/147. This is stated explicitly for R4 in summary.json `selection_note`.
5. **"Held-out" seed 3 holds out the seed, not the data.**
   - Same train/test split as the grid.
   - The linear-R² audit is fit and scored on the test representations (`scripts/eval_round4_dominant_axis.py:172`), and the schedule was tuned on the same grid.
   - The numbers themselves recompute exactly: 6/8, clean 1/8, Δ mean +0.0111, sd 0.0372, range [−0.0395, +0.0824].
6. **CelebA PCRL-V reports only train-set R².**
   - The train-set claim checks out: max 0.0036, and Smiling accuracy 0.7517 ± 0.0103 matches the paper.
   - The same run logs CelebA val-partition linear R² every epoch (`training_log.json`): Male/Young 0.156–0.171 on all 3 seeds, with `n_feasible_epochs=0`.
   - With d=512 and n=4096 in-sample OLS, the null bias is about 0.125, so this is inconclusive. It is still the only held-out measurement, and the paper does not mention it.
7. **LAFTR (Table 1: 15/60, mean 0.270; App. Q) is only partly supported.**
   - Adult per-seed metrics are on GitHub and recompute (0/24).
   - HMDA and Diabetes (36 rows) survive only as joined rows in `results/inlp_benchmark/inlp_results.json`, built from S3 `pcrl-bios-overnight-20260504/laftr_benchmark/FINAL_BENCHMARK.csv`. That bucket has a 7-day lifecycle; I did not try to read it.
   - The "discriminator at chance on 56/60" claim (paper L631) has no located artifact.
8. **The CROSSPURP constraint (paper: 23/24→19/24, 14/15→12/15) used a different protocol from the comparator, and the paper does not say so.**
   - 75 epochs instead of 200.
   - `canonical_iterate.pt` selection instead of final.pt.
   - Seed 0 had 0/75 feasible epochs.
   - 19/24 recounts.
9. **FAccT ablations 1–3 were registered but not executed.**
   - Registration ecaeba46f; CPU smokes only (`checkpoints/v2_diabetes_ABL*_SMOKE*`); launch plan 15dbcc3c3 says "nothing launched".
   - No full outputs exist in any branch, drive inventory or laptop.
10. **Corrections on availability.**
    - The crosspurp-constraint local commits (a3875c618 etc.) and laftr-benchmark e2f81ee are ancestors of origin/main (merge 17ef7d449), so they are on GitHub, not local-only. This was a parent correction, and I verified it.
    - checkpoints/ is no longer on the laptop; it exists only on the drive.
    - The git-ignored infra launch scripts for May 5–30 runs (inlp, splince, varconstraint, perdim, erase_*) still exist on the laptop only.
    - The v2 Round 4–7 user-data was never committed: `/tmp/round5/userdata_*.sh` per `experiments/round5_orchestrator.py@135e440e6^`.

### Details

#### v2 rounds (Adult/HMDA/Diabetes)
- Entry point: `experiments/run_v2_dataset.py --dataset D --out-tag _ROUNDk --seeds 0 1 2`. Audit: `scripts/eval_round4_dominant_axis.py --tag ROUNDk`. Commands from docs/REPRODUCIBILITY.md@origin/main.
- Launches: AWS g4dn with a local-Mac orchestrator. The user-data files lived in /tmp and are not preserved.
- Config: a full config dict is embedded in every final.pt. For example, R4 Adult s0 has lora_rank 8, epochs 200, warmup 5, leace_init True, lr_lambda 0.02, r2_lambda_max 1000.
- Result dirs on GitHub:
  - `results/v2_{adult,hmda,diabetes}_ROUND4/`
  - `v2_{adult,hmda}_ROUND5/`
  - `v2_diabetes_ROUND{5,6,7}/`
  - Diabetes R5 and R6 have no dominant_axis_audit.json.
- Checkpoints on the drive, best.pt and final.pt for every seed (~1–1.2 MB each):
  - R4: `v2_{ds}_s{0,1,2}` (untagged)
  - `v2_{adult,hmda}_ROUND5_s*`
  - `v2_diabetes_ROUND{5,6,7}_s*`
  - Full list with sha256 in core_checkpoints.csv: 440 checkpoint-archive members plus results-ignored .pt files.
- Not reported in the NeurIPS PDF: Folktables R2 (only s0 and s1 checkpoints exist), BIOS v2 rounds, BIOS layer-12, head-aware LEACE.

#### Benchmarks
- **INLP:** 27 metrics.json on GitHub (a3875c618). Encoders, eval_reps.npz and test_labels.npz are in the drive's results-ignored.tar. Recount: 60 rows, mean 0.0382, 43/60 pass, task acc 0.815, matching the paper.
- **SPLINCE:** 9 metrics.json (a3875c618), warm-started from the R5/R7 final.pt. Recount: 60/60, health 3/60, +Δ 2/60, 16 fallbacks, 6.16 pp drop, matching the paper.
- **baselines_singlepurpose (May 2):**
  - This is a separate LAFTR/INLP run. Adult income/race LAFTR here is 0.300, versus 0.397 in laftr_benchmark.
  - I could not tie it to any paper number, and its producing script was not identified.

#### Checkpoint → run of origin
| sha prefix | member | run |
|---|---|---|
| 1cfc2fef | checkpoints/v2_adult_s0/final.pt | PCRL v2 Round 4 Adult seed 0 (verified via lambdas/epoch) |
| e29d0367 | checkpoints/v2_hmda_s0/final.pt | PCRL v2 Round 4 HMDA seed 0 (verified via lambdas/epoch) |
| b0df3fb7 | checkpoints/celeba_v2/final.pt | April CelebA CNN run (run_celeba_v2.py), not PCRL-V R5 |

#### Not on laptop vs not completed
- **Not on laptop, but on the drive:** all checkpoints; INLP encoders and reps; LAFTR Adult encoders; CelebA R5 final.pt; variance-constrained final.pt.
- **Referenced archive only:** LAFTR HMDA/Diabetes per-seed outputs; BIOS layer-12 per-launch outputs (S3).
- **Not completed:** FAccT ablations; final.pt DA audit for Diabetes R5/R6.


---

## Slice R — PCRL May-2026 rebuttal-era runs, LAFTR-hard, S3 (fork notes)


### EARLY FLAGS

1. **The full LAFTR hard-R² outputs exist, but only on the external drive.** These are the per-seed JSONs for Adult, HMDA and Diabetes, plus the generated HEADLINE, PAPER_PASTE and comparison files. They are in `wt-PCRL-superpowers-laftr-hard-r2-2026-05-17.tar::laftr-hard-r2-2026-05-17/results/laftr_hard_r2_{adult,hmda,diabetes}_LAFTR_HARD_R2/` (Adult per-seed sha256 8a784c75…, HMDA 4686406d…, Diabetes 5457db69…). They were untracked files in the relocated worktree; `tree-storage-cleanup-20260919::storage-cleanup-20260919/laftr-hard-r2-2026-05-17-status.txt` lists them as `??`. GitHub has only code and the Adult 5-epoch smoke.
   - My recount of the drive files gives Adult 0/24, HMDA 17/18 and Diabetes 18/18, so **35/60** (R² < 0.05). This matches HEADLINE.txt.
   - All three summary.json files say STATUS=COLLAPSED.
2. **Two LAFTR-hard comparison rows are not measurements.** The LAFTR-Q HMDA and Diabetes rows come from the submitted paper's Appendix Q Table 13. The PCRL row is frozen in `scripts/paper_baseline_numbers.json`.
3. **The erase-layer pilot per-seed outputs (60/60) exist only on the laptop.** The files are `/Users/nathansamson/PCRL/results/rebuttal/erase_layer_pilot_aws/` (untracked, not ignored).
   - They are not in any branch. They are not in `fl-PCRL-main-results-ignored.tar` (that archive covers ignored files only) or in any drive inventory.
   - This is a single point of failure. The per_dim_std diagnostic (row e) and the VICReg comparison also read these files.
   - My recount gives strict 60/60 and cleanly-compliant 0/60.
4. **Inconsistent baselines: 56/60 in the paper versus 54/60 in the rebuttal tables.**
   - The manuscript (b.txt l.18, l.131) says 56/60 strict and 7 clean.
   - The erase-pilot and LAFTR-hard aggregators label "PCRL (Round 5/7, paper headline)" as 54/60 strict and 5/60 clean, computed from the R5/R7 per-seed `linear_r2`.
   - The CROSSPURP HEADLINE uses R5 Adult 23/24, while the erase aggregator uses 21/24 for the same R5 Adult.
   - The other slice (v2 Rounds) should resolve this.
5. **Two checkpoint rules for the same encoders.**
   - The per-pair R² audit (`run_v2_dataset.py:330-361`) reloads best.pt, the Cotter best-iterate selected on val.
   - The original §5.5 cross-purpose attack (26/33, 22/33) used **final.pt** of R5/R7 (the `checkpoint` field in `origin/main:results/v2_cross_purpose/per_seed_results.json`).
6. **The rebuttal "22/33 → 8/33" compares different models.**
   - The 8/33 comes from new cross-purpose-constrained, erase-layer retrains evaluated at best.pt. It is not a re-evaluation of the submitted encoder.
   - On the same data, the absolute (majority) criterion gives 26/33 → **19/33**. `PAPER_PASTE.md` explicitly advises against reporting that number.
   - On Diabetes, best_epoch=0 (Cotter fallback, n_feasible 0/200). The deployed model is the LEACE-projected initialisation.
7. **Cross-purpose retrain results.json files exist only on the laptop.** They are `results/v2_{adult,hmda}_CROSS_PURPOSE_AB`, `results/v2_diabetes_CROSS_PURPOSE_DIABETES` (untracked).
   - The 9 constrained checkpoints exist only at `s3://pcrl-bios-overnight-20260504/archive/cross_purpose_*`.
   - The executed wrapper `/tmp/cp_analysis/unified_protocol.py` is gone. Only the committed "mirror" remains.
8. **No May checkpoints or outputs are in the durable S3 bucket.** Checkpoints and outputs for the erase pilot, VICReg, rank-8, LAFTR-hard and cross-purpose runs are referenced only under `pcrl-bios-overnight-20260504`. That bucket has a whole-bucket 7-day expiry, and its "archive/" prefix is in the same bucket.
   - No May experiment appears in the `pcrl-ux-archive-ed9d21fd` manifests, which cover the ACS lineage only.
   - These checkpoints are not on the drive either. `fl-PCRL-main-checkpoints` holds only smoke checkpoints for erase and CROSSPURP_ERASE.
9. **Two "not completed" items and one lost run.**
   - The VICReg Diabetes arm was cut by the 10h cap before syncing.
   - LAFTR-hard launch #1 (i-04547a540150e6b28) was lost to the lifecycle rule, with its outcome unknown.
   - The first VICReg launch (i-0180537b761f180f9) was also lost.
   - These items were not completed. They are different from the laptop-only files in flags 3 and 7, which were completed but are not on GitHub.
10. **Launch configurations for the erase pilot, VICReg and rank-8 runs are uncommitted.** They are git-ignored `infra/*` files, present on the laptop only (sha256 values in R-04). The LAFTR-hard and cross-purpose launchers are force-added on their branches.
11. **Timing.** All work in this slice was done between May 18 and June 5. That is before the NeurIPS reviews were written (Jun 22–26).
    - It addresses anticipated concerns; it was not done in response to the reviews.
    - The laptop copies all have mtime Jul 24 14:37, which looks like a bulk copy or restore on the day the reviews were released. Their content dates from May.
12. **Branch-ref note (affects the branch inventory).** The local tips of `crosspurp-constraint-2026-05-06` (ae63692c, 10 ahead of its origin ref) and `laftr-benchmark-2026-05-05` (e2f81eed, 2 ahead) are both reachable from origin/main and other origin branches. They are on GitHub; only their branch refs are stale.

### Details

#### Erase-layer pilot (R-01..R-04)
- **Code:** `erase-layer-pilot-2026-05-17` (96c03d01 architecture; 04cb7efa smoke; 65dd5c05 aggregator). The branch is based on the cleanup commits ae63692c/17ef7d44.
- **Run:** `experiments/run_v2_dataset.py --out-tag _ERASE_PILOT --seeds 0 1 2 --use-erase-layer --lora-target repr_proj_only`. The argv is taken from laptop-only `infra/erase_pilot*/user_data.sh`.
- **Data use:** model fit on train, Cotter best-iterate selection on val (best.pt), and `generate_report` probes fit on train and scored on test.
- **Recount from laptop files:**
  - Adult 24/24, R² mean 0.0078
  - HMDA 18/18, R² mean 0.0057
  - Diabetes 18/18, R² mean 0.0069
  - Clean 0/60
- **Adult summary.json** says STATUS COLLAPSED (adjusted-criterion pass counts 1/0/0). Adult task accuracy fell: occupation_group 0.75 and education_level 0.83, against roughly 0.99 at baseline.
- **Comparison output:** `results/rebuttal/erase_layer_pilot_aws/rebuttal/{HEADLINE.txt,comparison.json}` (laptop only). The baseline side of 54/60 was not recomputed here.

#### VICReg sweep (R-05..R-07)
- **Per-seed files:** Adult and HMDA are on GitHub (rebuttal-evidence, erase-layer-vicreg-sweep, diabetes-rank8-ablation branches). There are identical copies in `wt-PCRL-claims-guarantee-review.tar` (sha256 equal to the git blob content).
- **Recount:** 42/42 strict, 0/42 clean.
- **Launch:** from 67427e7b (i-09bdc3b150df928a5). The Diabetes arm was not completed.

#### per_dim_std diagnostic (R-08)
- `diagnostic_perdim_std.json` has 9 rows (dataset × seed).
- The script rebuilds the backbone from the seed and refits LEACE. It does not load the pilot checkpoints. The claim of bit-identity relies on the backbone being frozen with BN frozen at its defaults.
- Ranges match HEADLINE: backbone 0.18–0.22; no-LEACE repr_proj 0.23–0.29.

#### Rank-8 (R-09)
- On GitHub at rebuttal-evidence@5739d3bb, with a drive copy.
- Recount: 18/18 strict, R² max 0.0087, clean 0/18.
- Launched from diabetes-rank8-ablation 98c4d3ca (i-0979baafe7ffa533b) using `--lora-rank 8 --device cpu`.

#### Cross-purpose (R-10..R-13, R-22)
- **Original, LR/MLP/XGB** (`origin/main:results/v2_cross_purpose/per_seed_results.json`, sha 25e38e4a…):
  - incremental 22/33
  - absolute 26/33
  - final.pt of R5/R7
- **The `_extra` JSON** holds DeepMLP/RF/SVM: 20/33 incremental, 27/33 absolute.
- **Unified protocol** (d3921121, sha 2682f3f3…):
  - gain_pp recomputed from the stored accuracies, 99/99 rows match
  - incremental 8/33; absolute 19/33
  - concat_acc matches the laptop CROSS_PURPOSE results.json, 99/99
  - per-pair R² 60/60 (max 1e-5)
- **Adult cross-purpose constraint, 75 epochs** (origin/main `results/v2_adult_CROSSPURP/results.json`, canonical_iterate.pt):
  - recount 19/24 per-pair and 12/15 attack flags
  - checkpoints for s0–s2 (best, final, canonical) are on the drive in `fl-PCRL-main-checkpoints`

#### LAFTR hard-R² (R-14..R-20)
- **Branch** `laftr-hard-r2-2026-05-17`@2037ad45, merged into local main 0eee48f9 (also in origin/main history).
- **Runs:**
  - Adult from launch #2 (i-0f1513dadc4753710, mtime 2026-05-29T18:57Z)
  - HMDA and Diabetes from launch #3 (i-0e0c84a2943f3f237 @1c5e1aff, mtime 2026-05-30T09:23Z)
- **Selection:** best.pt (Cotter). Adult best_epoch was 11/15/17, so selection happened very early.
- **Extraction check:** I extracted members to scratch, `parts/_x_laftrhard/`, and their hashes match the drive inventory.
- **Not located:** the checkpoints exist only under the S3 lifecycle prefix. `tree-laftr_cells.tar` is unrelated: it holds the durable-guarantees (July) LAFTR-official input cells (`adult_sex_income`, `hmda_race_*` .npz with meta.json).

#### Archive indexes
- **GitHub:** ARCHIVE_MANIFEST.md, RESTORE.md, ARCHIVE_INDEX.json, artifact_manifest.json and MANIFEST.json exist on 197 branch/path combinations. All belong to the September ACS-release or manuscript lineage.
- The `pcrl_utility_extension_v1` exec_main chunk (main checkout @ad2c0887, 2,545 files) is described as "execution inputs". It may or may not contain the laptop-only rebuttal results. Its per-file manifest is in a private ledger, not in git, so I could not check.
- **Drive:** the only RESTORE.md files are the relocation RESTORE.md and `storage-cleanup-20260919/RESTORE.md`.


---

## Slice D — durable-guarantees (fork notes)


Sources: clone `dg` @ main 956f5c88 (= GitHub main; origin/archive-cleanup e702c4b7 is an ancestor-merged branch, 0 unique commits); drive `tree-durable-guarantees.tar` (inventory `tree-durable-guarantees.json.gz`); prior crosswalk `combined-empirical-preparation-v1/.../EVIDENCE_CROSSWALK.csv` DG-01..28.

### EARLY FLAGS

1. **Raw per-person arrays exist, but only on the external drive.** `analysis/{tpr_scores (14), tpr_ext_scores (26), tpr59_scores (57)}` attacker-probability npz, `analysis/gate_reps` (10 × ~31 MB representations, 63,747×64), `analysis/fare_cells/shards`, `analysis/fnf_cells/*.npz`, and `data_cache/{celeba,folktables,expansion}` were never tracked in git (`6021b4c:.gitignore`, `6021b4c:analysis/yus_extras/.gitignore`). After the 2026-09-30 relocation, the drive (`tree-durable-guarantees.tar`) is the only copy. Total ≈ 964 MB across 739 members (listing with sha256 in `dg_drive_raw_members.csv`). Not on laptop does not mean lost.
2. **Independently recomputed: TPR@1%FPR and attacker AUCs for all 59 failing configurations.** I extracted the 57 `tpr59_scores/*.npz` files and checked them against the inventory sha256 (all 57 match). From the stored per-person probabilities, TPR@1%FPR per probe seed matches `results/tpr_failing59.json` per_seed exactly for all 59 configs. Two Adult noise σ=1/2 configs are E2/E4S1 duplicates that share one npz. All 118 stored XGB/MLP suite AUC means equal the recomputed values (max diff 0). See `dg_tpr59_recompute.json`. These reruns passed the ≤0.01 reproduction gate against the original `honest_reaudit`/`expansion` AUCs, so the AUC basis of "59 fail" is reproducible from stored scores.
3. **The survivor rule ignores Tier 2.** 67/59/8 recounted from stored per-config flags: new cells give 46 approved, 41 failing and 5 surviving out of 81 rows; the existing set gives 21/18/3. However, 4 of the 5 new-cell survivors (all noise σ=8) have stored `tier2_max` 0.552–0.577 > 0.55 (`main@956f5c8:results/expansion_reaudit_paperframing.json` rows). The fail rule is `approved_at_rest and not honest_stopped` (`experiments/expansion_reaudit_paperframing.py:114`), which uses Tier 1 only.
4. **"Blind-reviewer R1/R2/A" objections are simulated, not official.** The registered predictions (fresh partition, FARE, isolate-vs-fullrank) cite them. They come from `6021b4c:analysis/yus_extras/simulated_reviews_v28.md`, persona reviews generated 2026-07-23 (commit c540391) and later removed from main at c173cfe on 2026-08-02. The official AAAI AI review is dated Jul 27 and the human reviews Aug 23–30. The rebuttal bundle `6021b4c:REBUTTAL_EVIDENCE_2026-07-24.md` answers the simulated reviews.
5. **Selection and evaluation share one partition.** Except for the fresh-partition run, every operating point (σ or knob) is chosen by a sweep on the same PCRL-train rows that the certifying attackers are split from (75/25 re-splits; sweep probe seeds 0,1 and certification seeds 0,1,2). PCRL val and test splits are never used. The only held-out evidence is `results/fresh_partition_generalization.json`: 50/50 split, seed 20260724. Recounted from its stored per-arm means: 0 flips across 9 arms, max |Δ| 0.0133.
6. **dg reads PCRL `final.pt` checkpoints and does not pin a PCRL commit.** It uses `PCRL_ROOT/checkpoints/v2_adult_s0/final.pt` (sha256 1cfc2fef…, mtime 2026-04-28T17:52Z), `v2_hmda_s0/final.pt` (e29d0367…, mtime 2026-04-29T01:48Z) and `celeba_v2/final.pt` (b0df3fb7…, mtime 2026-04-16T07:28Z). All three are in `fl-PCRL-main-checkpoints.tar` (`utils/pcrl_io.py:58,170`; `experiments/celeba_extract.py:54`). Both tabular mtimes predate the Round-5 launch (2026-04-30), and the HMDA mtime is about 70 min after the Round-4 HMDA relaunch at 00:38Z. Matching the run of origin is left to the PCRL fork. From Exp 8 onward, most experiments do not use the PCRL encoder at all: X is the raw PCRL train-loader features and dg trains its own channel (`experiments/diagnostic.py:96-126`). The exceptions are Exps 1–5, the 21 existing re-audit configs, tpr59 and CelebA.
7. **The LAFTR-official raw runs were not located.** The default location `../laftr/experiments` is not on the laptop, not in any drive inventory and not in any relocation job. Only the aggregates (`results/laftr_official.json`) and the exported input cells (`tree-laftr_cells.tar`, 3 cells, re-exported 2026-07-24T01:56Z for tpr_extension) exist. The FARE clone is also absent, but its repo and commit are pinned (89cb1b66). Clones of Obliviator (0f2233f, matching the pin) and fair-pca are on the laptop.
8. **The submitted AAAI tex is not in the repo.** `paper/*.tex` was removed at 83739fd (2026-07-18). The only source found is the Overleaf export `~/Downloads/AAAI27___Outputs_Leak_What_They_Use.zip`, dated 2026-08-30, which is later than the submitted Jul-29 build.
9. **Some results landed late in, or after, the AAAI window.** On Jul 28–29 (after the AI review on Jul 27) the repo gained the worst-pair and supported sweeps, the 5-seed recertification, knows-Q high-σ, the fleet shards and the "rebuttal diagnostics" (3ac75bd). `baseline_dual_score.json` was produced Jul 29 according to its logs but first committed in the 2026-08-02 restructure (07df5ee).

### Repository state facts
- Drive tree vs GitHub main: all 198 tracked files under results/experiments/utils/docs/figures are byte-identical (sha256) between `main@956f5c8` and `tree-durable-guarantees.tar`. The drive additionally holds untracked `analysis/` (≈487 non-venv members), `data_cache/`, `.venv`.
- Cleanup on 2026-08-02 (c173cfe..8bb3a7b): five results JSONs were deleted (`continuous_cost_{hmda,diabetes}`, `generalization_{hmda,diabetes}`, `surgical_vs_blunt_lora_extra`). The working notes and mock reviews were deleted, and the analysis intermediates were untracked. All of these remain on GitHub at `6021b4c`, an ancestor of main. That includes the fare_cells manifests and embeddings, fleet/gate/knowsq shards, fnf stage-a JSON and tpr_ext partials. They are "on GitHub (history)", not missing. `tpr_extension.json` differs between 6021b4c and main only in the script path inside a header string.
- Registration pattern: each `docs/<run>_prediction.md` commit precedes its result commit. Intervals: averaging 3 min (run 1.5 min), dp_bet 6 min (4.0 min), fresh 14 min, fare 35 min, isolate 14 min, knows-Q 13 min, tpr59 77 min, worst-pair 64 min, supported sweep 60 min, gate_5seed 6 min. The drive log mtimes are consistent with the runs finishing after registration. Unregistered (disclosed): tpr_results (`docs/README.md`). `knows_q_highsigma.json` has no prediction_commit.
- Corrections to the prior crosswalk:
  - DG-11/DG-27 called the raw FARE manifests and the CelebA rep cache "local-only". They are now on the external drive only, and the FARE manifests and embeddings are also on GitHub history at 6021b4c.
  - DG-01/06 are confirmed.
  - DG-12/14/15: the held-out utility numbers are in `fresh_partition_generalization.json` per-arm `utility_kept_eval_pct`.

### Per-experiment index
See `dg_evidence.csv` D-01..D-28. Its columns record the entry point, configuration, splits, seeds, checkpoint rule, artifact paths (GitHub `main@956f5c8:` or drive member) and sha256 (from `git show main:<path> | shasum -a 256`, identical to the drive inventory).

Status distribution:
- independently recomputed: D-02 and D-04 (counts from stored flags), D-12 (TPR/AUC from per-person scores).
- checked against code and recorded aggregates: most others.
- reported but not independently reproduced: early experiments, DP, hybrids, Fair PCA/FNF, fleet.
- unsupported by located artifacts: D-26, the LAFTR raw runs.

