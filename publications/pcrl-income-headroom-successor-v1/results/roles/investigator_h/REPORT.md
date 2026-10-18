## Family H report: headroom-first income learner (investigator H)

*Saved by the coordinator, verbatim from investigator H's final hand-back: the harness refused the investigator's own write of
this file with the message "Subagents should return findings as text, not write report files. Include this content in your
final response instead." (PROTOCOL_DEVIATIONS.md item 4).*

**15-line summary (from the hand-back)**
1. Family H is done: initial version plus 2 revisions, 9 of 10 configurations, 0.153 CPU-h (489.1 s across 29 queue units, one failed selftest kept, plus a 60 s agent allowance).
2. Best configuration is **H3_STRONG_LJDIR_PB**. Its income base, STRONG, is a CALIBRATION stack of a cross-validated HGB income learner and the T_TREE income score. On top come the stored LOCAL_TREE income correction (full) and β × the stored JOINT_TREE income correction, with β = 0.25 (seed 0) / 0.20 (seed 1) set by a CALIBRATION budget that accounts for estimate noise, then a CALIBRATION Platt map. The occupation output is the frozen JOINT_TREE output.
3. Seed 0, MODEL_SELECT: income Brier vs Ucal −0.002342 (SE 0.001179), NLL −0.003592 (SE 0.001452), accuracy loss +0.000125 (SE 0.002637); RESERVE_MET, min ratio 3.511.
4. Seed 1, MODEL_SELECT: Brier −0.003254 (SE 0.001284), NLL −0.004879 (SE 0.001595), accuracy loss +0.001750 (SE 0.002861); RESERVE_MET, min ratio 2.862. Two-seed status RESERVE_MET; the binding condition is seed-1 accuracy.
5. Pair proxy cells: 0.6098–0.6103 (seed 0) and 0.6127–0.6144 (seed 1); two-seed mean 0.61185. Income view 0.5307–0.5343 / 0.5324–0.5359. Occupation view is inherited.
6. Pair benefit, min..max per seed: vs NEURAL_RAW +0.0179..+0.0208 / +0.0211..+0.0230; vs H_LJ +0.0134..+0.0146 / +0.0091..+0.0118; vs B2TWIN +0.0065..+0.0106 / +0.0012..+0.0064. All 24 cells are positive; the narrowest are seed-1 cells B0/B1 against B2TWIN, at +0.0012. Income-view recipient maximum vs NEURAL_RAW: −0.0065 / −0.0017.
7. Matched task-only twin **HC_STRONG**: Brier −0.005271 / −0.005783 (T_TREE's registered values are −0.00367 / −0.00446). Pair 0.6363–0.6388 / 0.6359–0.6365, income view +0.020 / +0.023 above NEURAL_RAW. On its own it is UTILITY_HEADROOM_ONLY.
8. Component control **HC_TTREE_LJDIR_PB** (same step, same β, old T_TREE base): Brier −0.000901 / −0.002149, pair 0.6106–0.6121 / 0.6156–0.6183, RESERVE_MET (3.902 / 2.400). It ties B2TWIN on seed 1 (−0.0000..+0.0012), so it fails the B2TWIN gate.
9. What STRONG is worth at the matched step: −0.00144 / −0.00111 income Brier and −0.0016 / −0.0033 pair mean. The headroom over T_TREE (0.0016 / 0.0013) shows on MODEL_SELECT only; on CALIBRATION STRONG is +0.0003 worse on both seeds.
10. H1 (the registered budget rule) released β = 0 because the noisy realized-accuracy term stopped it; pair ≈ H_LJ. H2 (decision-preserving, β 0.85) reached pair 0.6006–0.6079, but Brier was +0.00497 (RESERVE_NOT_MET) and the held mass at p ≈ 0.5 pushed the income view to +0.0113 vs NEURAL_RAW, so that idea was abandoned.
11. Main caveat: the CALIBRATION budget under-predicted the MODEL_SELECT Brier cost (H2 +0.0042 predicted vs +0.0102 realized; H3 +0.0005 / +0.0003 vs +0.0029 / +0.0025). H3 kept the reserve only because the expected-accuracy term stopped β early.
12. Configurations used: HC_TTREE_LJDIR S0; HC_STRONG S0, S1; H1_STRONG_LJDIR S0; H2_STRONG_LJDIR_DP S0; H3_STRONG_LJDIR_PB S0, S1; HC_TTREE_LJDIR_PB S0, S1. Ledger entries 34–36, 48, 58, 61–64; decisions 43, 57, 66.
13. Fits: 74 large fits (70 HGB cross-validation fits, 2 HGB refits, 2 ExtraTrees diagnostics) and 3,208 small CALIBRATION logistic maps. Per-record n_model_fits: HC_STRONG 23, H1/H2/H3 443, controls 442. Reader fits: 168 requested per score, 9 scores.
14. `ihs.fam_h.predict(seed, config, X, names)` recomputes all 9 releases bitwise, every role and both outputs, from features alone (unit IHS_H_VERIFYALL_V1; final fam_h.py sha256 c890263d56c9d0c0b8a384557d9fd71f8b1943e551f86081f0462e8236e5e176). `matched_twin` is in RELEASE.json. The component control (HC_TTREE_LJDIR_PB) is recorded in the card, this report and the code registry, not added to RELEASE.json after scoring.
15. One configuration slot is unused. No further revision.

---

This is exploratory development on previously used records (PCRL income-headroom successor development campaign), not confirmation. The data are public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida. The service predicts an income class (2 classes) and an occupation group (5 classes) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding `X`). Each recipient receives only its own float64 probability vector and argmax decision. SEX is the dataset's recorded demographic column. Readers are separate evaluation classifiers that estimate SEX from the released probability vectors.

Data handling:
- MODEL_SELECT values come only from `python3 -m ihs.devrun score` (EVAL_MS.json, ledger rows).
- No new fit used SEX; SEX entered only through the replication's stored LOCAL_TREE / JOINT_TREE corrections.
- MODEL_FIT income labels were used for the HGB; CALIBRATION income labels for the maps and budgets.
- Card: `CARD.md`. Code: `ihs/fam_h.py`.

### 1. What was tried

- **Step 1 (task learning, no privacy).** scikit-learn HistGradientBoosting on a compact re-coding of `X` (COW / MAR / POBP as categories, SCHL ordinal) or on `X` itself. Seven settings (learning rate 0.05, up to 600 trees) and the tree count were chosen by household-separated 5-fold cross-validation on MODEL_FIT. The pre-declared STRONG base is a 3-coefficient CALIBRATION stack of the HGB score and T_TREE's income score.
- **Step 2 (privacy).** Transplant the replication's stored critic-trained income corrections onto the base: `d = a·(s_B + D_L + β·D_J) + b`. Here D_L and D_J are the LOCAL_TREE and JOINT_TREE E20 income corrections relative to T_TREE. β is set by a CALIBRATION cost budget, B1 (summed Brier 0.0045, NLL 0.0080, accuracy 0.0035), fixed before any fit. There is no code: the output is continuous.
- **Learner-isolation controls** apply the identical step to the old T_TREE base, which is ≈ `a·(s_L + β·D_J) + b`.

### 2. Is there task headroom (Step 1)?

- On MODEL_FIT, out-of-fold summed Brier is flat across the grid: 0.34914–0.34969 (seed 0) and 0.34797–0.34979 (seed 1). T_TREE has no out-of-fold value there because it is in-sample.
- On CALIBRATION, with household cross-fitted maps, the difference vs raw T_TREE (seed 0 / seed 1; positive = worse):

| Score | Seed 0 | Seed 1 |
|---|---|---|
| HGB raw | +0.00191 (SE 0.00186) | +0.00279 (SE 0.00185) |
| STRONG stack | +0.00028 (SE 0.00082) | +0.00033 (SE 0.0007) |
| TASK_REFIT | −0.00070 (SE 0.00035) | −0.00059 (SE 0.00036) |
| Platt-mapped T_TREE | +0.00083 | +0.00057 |

- On MODEL_SELECT (one pre-declared score per seed), HC_STRONG income Brier vs Ucal is −0.005271 / −0.005783, against T_TREE −0.00367 / −0.00446 and TASK_REFIT −0.00396 / −0.00488. That is a gain of 0.0016 / 0.0013 over T_TREE.
- **Verdict.** On CALIBRATION the gain is below the 0.0005 threshold, so there is no headroom there. On MODEL_SELECT it is 0.0013–0.0016. The two held-out roles disagree by about 2 SE. A rough pooling is about +0.0009; the MODEL_SELECT paired SE vs T_TREE is approximately 0.0005 and was not computed.
- Alone, HC_STRONG is UTILITY_HEADROOM_ONLY: point-eligible and RESERVE_MET on both seeds (min ratio 4.578 / 3.296), but its pair is above both references and its income view is +0.020 / +0.023 above NEURAL_RAW.

### 3. Versions and results (MODEL_SELECT; values vs Ucal, SE)

| Config | Seed | β | Income Brier | Accuracy loss | Reserve | Pair cells | Benefit vs NEURAL_RAW / H_LJ / B2TWIN (min..max) | Income view |
|---|---|---|---|---|---|---|---|---|
| H1_STRONG_LJDIR | 0 | 0 (outside B1) | −0.004542 (0.001097) | −0.002875 | MET 4.578 | 0.6229–0.6241 | +0.0042..+0.0082 / −0.0008..+0.0020 / −0.0061..−0.0036 | 0.5396–0.5441 |
| HC_TTREE_LJDIR (control) | 0 | 0.10 | −0.002341 (0.001216) | −0.0030 | MET 4.607 | 0.6180–0.6195 | +0.0094..+0.0123 / +0.0053..+0.0054 / −0.0027..+0.0025 | 0.5374–0.5408 |
| H2_STRONG_LJDIR_DP | 0 | 0.85 | +0.004970 (0.001497) | −0.002875 | NOT MET | 0.6006–0.6079 | +0.0206..+0.0300 / +0.0154..+0.0243 / +0.0126..+0.0162 | 0.5507–0.5513 (+0.0113 vs NEURAL_RAW, fails) |
| **H3_STRONG_LJDIR_PB** | 0 | 0.25 | −0.002342 (0.001179) | +0.000125 | MET 3.511 | 0.6098–0.6103 | +0.0179..+0.0208 / +0.0134..+0.0146 / +0.0065..+0.0106 | 0.5307–0.5343 |
| **H3_STRONG_LJDIR_PB** | 1 | 0.20 | −0.003254 (0.001284) | +0.001750 | MET 2.862 | 0.6127–0.6144 | +0.0211..+0.0230 / +0.0091..+0.0118 / +0.0012..+0.0064 | 0.5324–0.5359 |
| HC_TTREE_LJDIR_PB (component control) | 0 | 0.25 | −0.000901 (0.001270) | −0.00125 | MET 3.902 | 0.6106–0.6121 | +0.0168..+0.0190 / +0.0114..+0.0128 / +0.0047..+0.0092 | 0.5321–0.5356 |
| HC_TTREE_LJDIR_PB (component control) | 1 | 0.20 | −0.002149 (0.001389) | +0.00275 | MET 2.400 | 0.6156–0.6183 | +0.0157..+0.0218 / +0.0056..+0.0079 / −0.0000..+0.0012 | 0.5362–0.5365 |

Further values:
- H3 NLL vs Ucal: −0.003592 / −0.004879.
- H3 Brier vs U0: −0.002790 (SE 0.001319) / −0.003303 (SE 0.001384).
- All rows are point-eligible except that H2 fails the reserve.
- Guards and occupation clauses are those of the inherited JOINT_TREE output.

### 4. What each revision taught

1. **H1 (registered B1 rule).** The realized-accuracy term on 2,982 CALIBRATION people is noise.
   - D_L alone "costs" +0.0044 (SE 0.0030) accuracy against the STRONG base there.
   - On MODEL_SELECT, H1's accuracy equals HC_STRONG's exactly (−0.002875).
   - So H1 released β = 0 (STRONG + D_L): pair ≈ H_LJ, Brier 0.0015 better than H_LJ.
   - The old-base control at β = 0.10 showed the JOINT direction is very efficient: −0.0053 pair vs H_LJ in every cell for about 0.0007 of Brier.
2. **H2 (decision-preserving residual).** It gave a large pair benefit (0.6006–0.6079), but two things went wrong:
   - The CALIBRATION Brier estimate did not transfer (+0.0042 predicted vs +0.0102 realized against HC_STRONG). Choosing the largest in-budget β also picks up noise (winner's curse), because the CALIBRATION cost SE grows with β (0.0023 at β 0.85).
   - Holding the 8.5% of people whose decision would cross at p = 0.5 ± 0.0025 made that set readable: the income view rose to 0.551.
3. **H3 (precision-aware rule).** The budget gate is the Brier/NLL cost plus one household-clustered SE, and accuracy is gated by a model-based expected-accuracy cost.
   - The expected-accuracy term bound at β 0.25 / 0.20.
   - The reserve held on both seeds. The CALIBRATION Brier estimate still under-predicted the MODEL_SELECT cost about 6–9-fold.

### 5. Attribution

- **Task learning (STRONG vs T_TREE).** At the matched step, H3 minus HC_TTREE_LJDIR_PB, same β: income Brier −0.00144 / −0.00111; NLL −0.00153 / −0.00123; pair mean −0.0016 / −0.0033; accuracy +0.0014 / −0.0010; min income ratio 2.862 vs 2.400. Without STRONG, seed 1 only ties B2TWIN (B cells 0.0000). STRONG itself does not leak less: HC_STRONG's pair is 0.636–0.639.
- **Calibration.** The CALIBRATION maps add estimation noise: a cross-fitted Platt map on T_TREE alone costs +0.0006–0.0008 on CALIBRATION. The stack gain is relative to Platt-mapped T_TREE.
- **Ordinary compression.** None; the output is continuous.
- **Privacy component.** D_L plus 0.20–0.25·D_J, against the task-only twin HC_STRONG: pair −0.0274 / −0.0226; Brier +0.00293 / +0.00253; accuracy +0.0030 / +0.0013; NLL +0.00367 / +0.00320. The privacy work is done by the replication's stored critic-trained corrections. Family H's own contributions are the stronger base, the transplant, and the budget rule. It is a close relative of the old LOCAL/JOINT mixtures (≈ s_L + 0.2–0.25·D_J on a better base). Nothing about it is novel, and none is claimed.

### 6. Best configuration and nomination profile (my reading of the frozen rule)

- **H3_STRONG_LJDIR_PB** passes all gates: (a) point eligibility on both seeds; (b) RESERVE_MET on both seeds, min_income_ratio 2.862 (binding: seed-1 accuracy); c1: 16/16 cells below both eligible references; c2: income view at most −0.0017 above NEURAL_RAW; occupation output inherited; c3: 8/8 cells below B2TWIN, narrowest +0.0012 on seed-1 cells B0/B1.
- Flags and context: READER_UNSTABLE (the winning pair reader varies across cells: seed 0 ET25 / CATI1 / OUT_LR / OUT_LR; seed 1 ET25 / ET25 / OUT_LR / OUT_LR); no PAIR_NEAR_TIE against the eligible references (mean benefits 0.0103–0.0221); the B2TWIN margin on seed 1 is within the validation's reader-refit spread (0.003–0.011); UTILITY_CONFIDENCE_UNRESOLVED: seed-1 income accuracy bound 0.00175 + 3.124 × 0.00286 = 0.0107 > 0.010; the inherited men's occupation guard is also unresolved.
- Matched task-only twin: HC_STRONG (both seeds). Matched component control: HC_TTREE_LJDIR_PB (both seeds).
- Compared with family P's P3_JR_B60 (as relayed by the coordinator): H3's two-seed mean pair is 0.61185 vs 0.60925 (within the 0.005 near-tie band), and H3 has about 0.002 more income Brier room.

### 7. Could STRONG plus a P-style or JOINT-direction step use the headroom?

- **JOINT direction: yes, demonstrated.** At an identical step, STRONG converts its MODEL_SELECT headroom into about 0.0011–0.0014 more Brier room and slightly lower pair. That is what lets seed 1 clear B2TWIN.
- **P-style step: plausible but untested.** If the effect is additive, adding P3's income correction (relative to T_TREE) to STRONG as a fixed residual (the `fixed` mechanism in `ihs/fam_h.py`), or training P's correction with STRONG as the offset, should gain similar room. That would be a new configuration; I did not run it.
- **Caution:** the headroom is not visible on CALIBRATION.

### 8. Caveats

- H3 is the third version. It was chosen after MODEL_SELECT feedback on H1 and H2 (logged decisions 43 and 57). β itself was set on CALIBRATION, and the one-SE margin was fixed, not tuned on MODEL_SELECT.
- Do not rely on the CALIBRATION budget's Brier term for transfer.
- The occupation output is inherited; the men's guard (0.0044 / 0.0039) binds on its own.
- The outputs are deterministic. No information or injectivity claim is made.

### 9. Recommendation

Treat H3_STRONG_LJDIR_PB as a qualifying development candidate, with these flags: the narrow seed-1 margin against B2TWIN; reader instability across cells; unresolved simultaneous utility confidence; the CALIBRATION→MODEL_SELECT transfer failure. For mechanism attribution, its controls are HC_STRONG (task-only) and HC_TTREE_LJDIR_PB (component). If the coordinator prefers P3_JR_B60, the STRONG base is a cheap add-on worth testing with P's correction.

### 10. Resources, provenance, files

- CPU: 489.1 s queue-charged plus a 60 s agent allowance ≈ 0.153 CPU-h of the 1.2 allotted. 29 units; IHS_H_SELFTEST_V4 failed (invented test budget) and is kept.
- Configurations: 9 of 10.
- Fits: 74 large and 3,208 small CALIBRATION maps (Step 1: 72 per seed).
- Invented-data checks: selftest V6 PASS (20 checks); dry runs IHS_H_DRYRUN_V1 / V2 PASS.
- Deployment: IHS_H_VERIFYALL_V1 shows all 9 releases bitwise-equal under `predict` from features. Packaging needs python3 (Homebrew 3.14) with scikit-learn 1.8.0, because the HGB is pickled at `runs/H/STEP1/S<s>/HGB.pkl` with its sha256 in STEP1.json.
- Files: `ihs/fam_h.py`; `results/pcrl_income_headroom_successor_v1/roles/investigator_h/CARD.md`; private `.private/pcrl_income_headroom_successor_v1/runs/H/<config>/S<s>/` (RELEASE.npz, RELEASE.json, STATE.json, EVAL_MS.json, VERIFY.json) and `runs/H/STEP1/S<s>/` (STEP1.json, HGB.pkl).
