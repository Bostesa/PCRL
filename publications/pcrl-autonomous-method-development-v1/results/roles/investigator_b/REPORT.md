# Investigator B report: Family B, coupled shared-statistic release (CSR)

Investigator B (a separate agent), 2026-10-17, about 16:58–17:30Z. The harness refused the subagent's Write of this file, so the coordinator saved the text verbatim from its hand-back.

This is exploratory development on the Florida ACS 2019 public-use development records, not confirmation.
- **Task.** The service predicts income (2 classes) and occupation group (5 classes) from AGEP, WKHP, COW, SCHL, MAR and POBP (the admitted 145-column encoding). Readers infer the dataset's recorded SEX column from the released probability vectors.
- **Numbers.** Every proxy is a MODEL_SELECT development reader from `amd.evaluate`, read only through `amd.devrun score` (ledger entries 28–42, family B). Values are seed 0 / seed 1.

## What I built (`amd/fam_b.py`, sha256 5661e524…6bf96)

**Base outputs.**
- Recomputed for every role from the frozen T_TREE routing plus the stored E20 corrections (T_TREE, LOCAL_TREE, JOINT_TREE).
- Each was checked against the stored replication outputs; the maximum absolute difference is 0.

**Release.**
- Occupation is a base occupation output.
- Income is softmax(−s/2, s/2), with s = h(design).
  - h is an L2 logistic map (sklearn logistic regression, C = 1, standardized design), fitted on all 2,982 CALIBRATION rows using income labels only.
  - The design can include the occupation output's 4 log-ratios (the coupling) and a one-hot m-level income code c.
  - c bins a stored income log-odds into decision-aligned bins, m/2 on each side of log-odds 0, with label-free MODEL_FIT quantile thresholds.
- SEX is never read at runtime or by h. MODEL_FIT SEX enters only through the stored replication corrections.

**Diagnostics.** `diag`, `diag2` and `diag3` are income screens on CALIBRATION only: household 5-fold cross-fitting, no SEX, no MODEL_SELECT. Saved in `runs/B/DIAG/`.

**Deployment hooks.**
- `build_package(name, seed, config_id, out_root)` and `package_release(pkg_dir, rows, requested)`, package kind FAMB.
- Each package holds:
  - the T_TREE package encoder;
  - the admitted teacher JSONs;
  - the stored E20 theta files that are needed;
  - an `H.npz` with h's coefficients and the thresholds;
  - a sha256 map.
- Thresholds are recomputed label-free and cross-checked against the release record.
- Test on DEV_REPORT raw rows (units AMD_B_pkgtest_V1 and AMD_B_pkgtest_csv_refusals_V1):
  - bitwise parity for B3CTL_UNCOUPLED_MIX50_C6A, B2_CSR_JT_C6A, B2CTL_UNCOUPLED_C6A and B2CTL_TASKONLY_TT_C6A on both seeds;
  - for the best configuration, the CSV path is bitwise and all 14 refusal cases hold.

## Results (MODEL_SELECT; income utility excesses against U0)

**Bars.** Pair: H_LJ 0.6224 / 0.6233; NEURAL_RAW 0.6238 / 0.6269. Recipients: income ≤ 0.5500 / 0.5476; occupation ≤ 0.5824 / 0.5953.

**Occupation side.** Every JOINT_TREE-based row has the same occupation clauses and guards as H_LJ:
- occupation NLL +0.0025 / +0.0029 (Ucal +0.0034 / +0.0030);
- occupation Brier −0.0002 / +0.0005;
- occupation accuracy loss −0.002 / −0.004;
- guard: men +0.00442 / +0.00394, women −0.00494 / −0.00304.

| Configuration (ledger) | Clauses | Worst normalized | Income NLL / Brier / accuracy loss | Pair proxy | Income proxy | Occupation proxy |
|---|---|---|---|---|---|---|
| B1 `B1_CSR_JT_LIN`, seed 0 only (28) | 6/12 | 8.82 | +0.0530 / +0.0437 / +0.0379 | 0.5666 | 0.5199 | 0.5724 |
| B2 `B2_CSR_JT_C6A` (31, 32) | 12/12 and 12/12 | 0.55 / 0.31 | s0 +0.0033 / +0.0023 / −0.0043; s1 +0.0026 / +0.0015 / −0.0009 | 0.6131 / 0.6132 | 0.5404 / 0.5394 | 0.5724 / 0.5853 |
| B2 coarse-only twin `B2CTL_UNCOUPLED_C6A` (35, 37) | 12/12 and 12/12 | 0.34 / 0.30 | s0 +0.0013 / +0.0005 / −0.0056; s1 +0.0006 / −0.0002 / −0.0020 | 0.6153 / 0.6154 | 0.5392 / 0.5399 | 0.5724 / 0.5853 |
| B2 matched task-only `B2CTL_TASKONLY_TT_C6A` (38, 39) | 12/12 and 12/12 | 0.41 / 0.25 | s0 +0.0020 / +0.0016 / +0.0016; s1 +0.0016 / +0.0010 / +0.0025 | 0.6454 / 0.6463 | 0.5524 / 0.5525 (fails bar) | 0.6426 / 0.6391 (fails bar) |
| B3 `B3_CSR_JT_MIX50_C6A`, seed 0 only (40) | 10/12 (income Brier fails) | 1.17 | +0.0073 / +0.0054 / −0.0008 | 0.6018 | 0.5307 | 0.5724 |
| B3 coarse-only twin `B3CTL_UNCOUPLED_MIX50_C6A` (41, 42) | 12/12 and 12/12 | 0.873 / 0.883 | s0 +0.0055 / +0.0039 / −0.0024; s1 +0.0064 / +0.0044 / +0.0035 | **0.6000 / 0.6001** | 0.5277 / 0.5286 | 0.5724 / 0.5853 |

## Versions, hypotheses and diagnoses

**B1 (initial version): pure coupling.**
- The pair became a function of the occupation output: pair proxy 0.5666, at or below the occupation proxy of 0.5724.
- Income missed every cap by 4–9×. The occupation vector does not carry the income information: on CALIBRATION, the teacher base costs +0.021 nats and the JOINT_TREE base +0.044.
- Tree occupation outputs are essentially one-to-one in the person's inputs. The information identity is therefore vacuous for an unrestricted h, and a simple h cannot carry income.

**B2 (revision 1): add a 6-level LOCAL_TREE income code.**
- m = 6 is the smallest number of levels inside the caps on the CALIBRATION screen. The pair carries at most log 6 of information beyond the occupation output.
- Eligible on both seeds, with the pair below both references.
- The coarse-only twin carries identical pair information. It reads 0.002 higher and has better income utility, so coupling is a finite-reader effect of about 0.002 that costs about 0.002 nats of income NLL.
- The matched task-only control barely moves below the teacher. The gain is income coarsening next to the privacy-trained JOINT_TREE occupation output.

**B3 (revision 2): a code from 0.5 LOCAL_TREE + 0.5 JOINT_TREE income log-odds.**
- JOINT_TREE's income is pair-aware. alpha = 0.5 was chosen on the CALIBRATION screen `DIAG3_S0`.
- Coupled B3 reached pair 0.6018 but failed the income Brier clause (+0.0054 vs U0, +0.0058 vs Ucal).
- Its coarse-only twin was run afterwards, an adaptive step. It is eligible on both seeds with pair 0.6000 / 0.6001.
- alpha = 1 (a JOINT-only code) was not run: on CALIBRATION it lies further outside the Brier cap.

## Best configuration and the promotion rule

**`B3CTL_UNCOUPLED_MIX50_C6A`.**
- U 24/24 at the point, G for both groups on both seeds, R for both recipients.
- Pair below both references on both seeds: by 0.0224 / 0.0232 against H_LJ and 0.0238 / 0.0268 against NEURAL_RAW.
- It misses the 0.8 × cap margin. Income Brier vs Ucal is at 0.873 / 0.883 of the cap and the seed-0 men's guard at 0.88, so a promotion would be EDGE_SELECTED.
- Every promotable family-B configuration is EDGE, because each inherits the seed-0 men's guard. By mean pair proxy: B3 twin 0.6000, B2 0.6131, B2 twin 0.6153.

**Caveats.**
- The best configuration is uncoupled. Its mechanism is a coarse, partly pair-aware income code next to the JOINT_TREE occupation output: a recombination of stored components on the line between H_LJ and H_JJ.
- These controls were not run because the budget was used up:
  - an uncoupled matched task-only control;
  - a continuous-mixture control;
  - coupled B3 on seed 1.
- There were 10 adaptive MODEL_SELECT evaluations. The 0.020 target is exceeded only at the point, on development proxies.
- Drift risk on DEV_REPORT:
  - Income Brier is at 0.87–0.88 of the cap.
  - In the published record, H_JJ's income Brier vs U0 was 0.0054 / 0.0067 on DEV, against 0.0053 / 0.0049 on MODEL_SELECT.
- Subgroup burden (not a protocol clause). On seed 1, women bear the income cost: NLL +0.0110 and Brier +0.0070 vs U0, against +0.0018 / +0.0018 for men. Seed 0 is balanced, with NLL +0.0054 / +0.0057.
- The income output has 6 distinct values per seed.
- The ledger's `roles_read` text for family-B rows is devrun's generic MODEL_FIT wording. The true label use is in each RELEASE.json: CALIBRATION income labels for h, and MODEL_FIT labels and SEX only inside the stored corrections.
- `runs/B/ZZ_B2_REPRO_CHECK/S0` is a software check, bitwise equal to B2 and never scored.

## Resources and files

- **Compute.** 26 queue units, all done, 150.9 CPU-s, plus a 60 CPU-s agent charge (`AMD_B_agent_light_scripts_V1`). About 0.06 CPU-h and about 45 minutes elapsed. 10 of 10 configurations used.
- **Files.**
  - `amd/fam_b.py`
  - `roles/investigator_b/CARD_REVISIONS.md`
  - private runs under `.private/pcrl_autonomous_method_development_v1/runs/B/`: `<config>/S<seed>/{RELEASE.npz, RELEASE.json, EVAL_MS.json}`, `DIAG/DIAG_S0.json`, `DIAG2_S0.json`, `DIAG3_S0.json` and `PKG_TEST_V1/`
- Nothing was committed and no shared module was edited.
