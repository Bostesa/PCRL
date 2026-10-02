# Missing files: the exact items that would resolve each remaining uncertainty

Compiled 2026-10-01.

**Where we searched:**
- All PCRL remote branches (51 refs) and the git bundles on the drive. All 106 bundle heads are reachable
  from GitHub.
- durable-guarantees @956f5c8, plus history at 6021b4c.
- The external drive inventories for both relocation runs (2026-09-30, 2026-09-25).
- The laptop.

**Where we did not look:** S3 was not read (AWS credentials expired). The S3 objects below are manifest
references only.

Full detail is in `notes/inventory/missing_files.csv`.

## Priority 1: affects a headline number or a reviewer answer

| # | What is missing | Needed for | Exact file / manifest entry that would resolve it | Status |
|---|---|---|---|---|
| 1 | Original LAFTR per-purpose HMDA and Diabetes per-seed metrics (36 of 60 rows in NeurIPS App. Q, Table 13) | Baseline column of the submitted comparison; N-R2 baseline concern | `s3://pcrl-bios-overnight-20260504/laftr_benchmark/FINAL_BENCHMARK.csv` and `laftr_benchmark/{hmda,diabetes}/<purpose>/seed_<n>/metrics.json` | Referenced archive only. The bucket has a 7-day lifecycle, so these are **probably lost**. Treat the rows as unsupported unless you hold another copy. |
| 2 | Held-out hyperparameter selection for HMDA and Diabetes | First and second NeurIPS reviewers' central request | No artifact exists. Executing the registered FAccT ablation 3 (or an equivalent registered run) would produce `results/v2_{hmda,diabetes}_ABL3_*` | **Registered but not executed.** Needs new training, so it is out of scope here. |
| 3 | FAccT ablations 1–2 full outputs (linear adapter vs LoRA; no-erase-layer / no-warm-start) | Third NeurIPS reviewer's ablation requests | `results/v2_diabetes_ABL{1,2,3}_*` non-SMOKE runs per `results/rebuttal/ablations_facct/LAUNCH_PLAN.md` | Registered but not executed. Only `*_SMOKE` exist, and they target the erase-layer variant, not the submitted architecture. |
| 4 | Final.pt dominant-axis audit for Diabetes Round 5 and Round 6 | Consistent final-iterate counts across rounds | `results/v2_diabetes_ROUND{5,6}/dominant_axis_audit.json`. Recomputable from drive checkpoints `fl-PCRL-main-checkpoints.tar::checkpoints/v2_diabetes_ROUND{5,6}_s*/final.pt` (forward pass only, no training) | Not completed. Computing it would be an approved recount, not new training. |
| 5 | Training configuration / run log of `checkpoints/celeba_v2/final.pt` (sha256 b0df3fb7…, mtime 2026-04-16), used as the AAAI vision encoder | First AAAI human review's CelebA configuration request; whether "Young" was disallowed during training | The run log or argv of the 2026-04-16 `experiments/run_celeba_v2.py` run, or a config dict embedded in the checkpoint (inspect `fl-PCRL-main-checkpoints.tar::checkpoints/celeba_v2/final.pt` keys) | Not located. **Inspect the checkpoint dict first**: it is on the drive. |
| 6 | Pinned PCRL commit used by durable-guarantees (`PCRL_ROOT`) | AAAI reproducibility | A recorded commit or environment dump from run time. The best substitute is the checkpoint hashes already recorded (Round 4 seed 0). | Not pinned. |

## Priority 2: provenance and reproducibility

| # | What is missing | Exact resolution | Status |
|---|---|---|---|
| 7 | AWS user-data / argv for v2 Rounds 4–7 | `/tmp/round5/userdata_{adult,hmda,diabetes}.sh` (orchestrator `experiments/round5_orchestrator.py`@135e440e6^), and equivalents for R4/R6/R7 | Not located. Each final.pt embeds its config dict, which is a partial substitute. |
| 8 | Launch configs for erase pilot / VICReg / rank-8 | `infra/erase_pilot*/user_data.sh`, `infra/erase_vicreg_sweep/`, `infra/erase_rank8_diabetes_cpu/` (git-ignored, laptop only) | **Single copy on the laptop. Please preserve privately** (they may contain cloud configuration, so they were not committed). |
| 9 | VICReg ×5 Diabetes per-seed output | `results/rebuttal/erase_layer_vicreg_sweep_aws/v2_diabetes_ERASE_VICREG5/per_seed_results.json` | Not completed (the 10-hour cap cut seed 2). |
| 10 | Erase-pilot / VICReg / rank-8 / cross-purpose-retrain checkpoints | `s3://pcrl-bios-overnight-20260504/archive/erase_*/checkpoints/`, `…/archive/cross_purpose_{ab,diabetes}/checkpoints/` | Referenced archive only, in the lifecycle bucket. Probably lost. The per-seed metrics survive. |
| 11 | LAFTR-hard checkpoints | `s3://pcrl-bios-overnight-20260504/archive/laftr_hard_r2/checkpoints/` | Probably lost. The per-seed outputs are on the drive. |
| 12 | LAFTR-hard first launch outputs | `s3://pcrl-bios-overnight-20260504/laftr_hard_r2/` (2026-05-20) | Lost to the lifecycle; outcome unknown. |
| 13 | Executed cross-purpose wrapper | `/tmp/cp_analysis/unified_protocol.py` | Missing. The committed mirror is `scripts/crosspurp/unified_protocol_reeval.py`. |
| 14 | Official LAFTR run directories | a `laftr` clone with `experiments/<run>/` outputs | Unsupported by located artifacts. Only aggregates exist. |
| 15 | BIOS layer-12 per-launch outputs | `s3://pcrl-bios-layer12-20260505/bios_pcrl_layer12/results/` | Referenced archive only. |
| 16 | CelebA PCRL-V held-out compliance with an unbiased R² | a held-out R² on ≥60k CelebA val/test images, or a saved `holdout_r2` training log | Not located. |
| 17 | AAAI submitted LaTeX source as of the Jul 29 build | Overleaf history snapshot at 2026-07-29 04:57 EDT | Local Aug-30 export only. Main text matches the submitted PDF; the appendix corresponds to the Aug 2 revision. |
| 18 | Achieved CelebA output-coupling values 0.536/0.540 | a results JSON holding these values | Prose only. |
| 19 | Contents of S3 chunk `pcrl_utility_extension_v1` "exec_main" (2,545 files) | its per-file manifest in the private ledger | Not checked. It might hold a second copy of the laptop-only rebuttal results. |

## Review-source gaps (cannot be resolved from artifacts)

| # | Gap | What would resolve it |
|---|---|---|
| 20 | First AAAI human review cites "[1,2]" with no bibliography | The references as displayed on the original review page, or a note from the venue. **Not guessed.** |
| 21 | Second NeurIPS review: "Zhang et al. (2024)", information-theoretic robust and privacy-preserving representations; the link was dropped | The hyperlink from the original OpenReview page. One candidate is listed in the matrix as **unconfirmed**. |
| 22 | Second NeurIPS review mentions "NCSGD" and "NDR" | These appear nowhere in the paper or any branch. Probably carry-over from another review; recorded as unmappable. |
| 23 | Dropped inline math in review exports | Recovered from the manuscripts where the surrounding text matched (`notes/reviews/source_gaps.md`). One derived accuracy value is not printed in the manuscript and is left blank. |

## Resolved since the previous assessment (no longer missing)

- **Official review text and meeting transcript.** Supplied. Stored privately at
  `~/Documents/PCRL_private_review_sources_20261001/` and not committed.
- **Reviewer's threshold range.** 0.52 / 0.55 / 0.60 or CI-based.
- **LAFTR-hard per-seed outputs.** On the drive (sha256 recorded in the inventory, R-14 to R-16).
- **durable-guarantees analysis arrays.** On the drive (`notes/inventory/durable_drive_raw_members.csv`).
- **All PCRL checkpoints.** On the drive (`notes/inventory/checkpoint_inventory.csv`).
- **Erase-pilot and cross-purpose retrain per-seed files.** These were laptop-only single copies; they are
  now preserved in this branch (`preserved_laptop_only_results/`).
