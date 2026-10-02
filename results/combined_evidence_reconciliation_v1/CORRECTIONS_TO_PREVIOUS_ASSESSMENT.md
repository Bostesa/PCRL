# Corrections to the previous assessment

**Previous assessment:** PCRL `research/combined-empirical-preparation-v1` @ f381bc2,
`results/combined_empirical_preparation_v1/`. That package is left unchanged as a historical record; this
file supersedes it where they conflict.

**What was missing then:**
- the official review exports;
- the meeting transcript;
- the external drive, which was not mounted.

**Reference keys used below:**
- `V:` items in VERIFICATION_REPORT.json.
- `INV:` notes/inventory.
- `RM:` REVIEW_RESPONSE_MATRIX.csv.

## A. Withdrawn or reframed (the previous statement was wrong or misleading)

1. **"Official review text not available."**
   - Now available: NeurIPS (3 human reviews); AAAI (2 human reviews + 1 separately labelled AI review).
   - The supplied PDFs are identified: NeurIPS = local build (23) plus venue stamps; AAAI =
     byte-identical to the Jul 29 build.
   - Every "known only from the meeting summary" attribution is replaced by its review source (RM).
2. **"Threshold range requested by the reviewer is unknown."**
   - The second AAAI human review asks for 0.52 / 0.55 / 0.60 or CI-derived thresholds.
   - The previous grid of {0.52, 0.55, 0.60} matched; CI-based bars must be added.
   - The 0.52/0.60 sensitivity **already existed** in the AAAI supplement (64/59/51 of 67), and the
     previous assessment missed it (V: F1).
3. **"Two references from the first AAAI human reviewer are missing from a pasted excerpt."**
   - The first AAAI human review itself cites "[1,2]" with no bibliography. The limitation is in the
     review, not in a transcript.
   - The four named prior works (Ravfogel 2022, Elazar 2021, Song & Shmatikov 2020, Stadler 2024) come
     from the **second** AAAI human review.
   - Gitiaux & Rangwala and FairNVT come from the **AI review**, which cites FairNVT as ICLR. The
     primary-source check found TMLR 2026 plus an ICLR 2026 workshop; the discrepancy is recorded.
   - A separate NeurIPS review cites an unidentified "Zhang et al. (2024)". One candidate is listed in
     RM as unconfirmed, not adopted.
4. **"Corrected: PCRL strict compliance 56/60 → 54/60."**
   - Withdrawn. 56/60 is the final-iterate count, which is the rule the paper states. 54/60 is the
     best-validation count.
   - Both are correct under their own rule (V: A1, A2, A4).
5. **"7/60 cleanly compliant → 5/60 on a single checkpoint."**
   - Narrowed. It is final-iterate 6/60, best-validation 5/60, and 7/60 for the paper's mixed rule. The
     earlier claim that no final-checkpoint health was stored was wrong: the SPLINCE `pre_health` field
     records it (V: A3).
6. **"Rebuttal cross-purpose 8/33 → corrected to 19/33."**
   - Withdrawn as a "correction". 8/33 is the incremental criterion and 19/33 the absolute criterion,
     both on the rebuttal model. The submitted model reads 22/33 incremental and 26/33 absolute.
   - The criteria differ, and neither was registered (V: B1–B3).
7. **"durable-guarantees Adult/HMDA cells are the same underlying fits as the PCRL encoder results;
   count once."**
   - Corrected. They are PCRL **Round 4** seed-0 checkpoints, not the Round 5 checkpoints behind the
     NeurIPS headline (INV flag 1; recomputed hashes).
   - They are related, not identical. The CelebA encoder is an April 16 `celeba_v2` CNN, not PCRL-V.
8. **"Middle full-rank Tier-2 point passes as 0.5496 under mean-then-max but fails at 0.553 under
   max-then-mean."**
   - Corrected. It is 0.5496 under both orders; 0.553 is the worst single seed (V: F9).
9. **"Isolate-then-noise advantage shrinks to about 46 pp on the easy cell (frontier matching)."**
   - Corrected values: the frontier advantage at 0.55 is +4.4 / +44.1 / +87.9 pp (easy / middle / hard).
   - Out of partition, the easy-cell advantage reverses to −6.8 pp (V: F6).
10. **"LAFTR-hard Adult 0/24 unverifiable locally (S3 only)."**
    - The full run exists on the external drive: Adult 0/24, HMDA 17/18, Diabetes 18/18, total 35/60,
      all fallback checkpoints, all collapsed (V: D; INV).
    - It was "not on the laptop", not "not completed".
11. **"No published collusion-aware method exists."**
    - Too broad. Taylor, Vippathalla & Coon (2026) is collusion-aware for sequential releases, under a
      finite alphabet and a known pmf, protecting the whole of X.
    - Correct statement: no learned-representation method with a declared sensitive set and a coalition
      guarantee was located.
12. **"No official code exists for VFAE / FFVAE / Gitiaux & Rangwala / FairNVT."**
    - Corrected to: "an official implementation was not located" (searches recorded in the methods
      catalog).
13. **"One-cell pilot; reconsider the direction if the cost relation disappears."**
    - A single cell cannot test a cross-cell correlation. A one-cell pilot validates the evaluation
      pipeline only.
    - Testing the coupling–cost association needs several registered cells under held-out utility.
14. **"Dominant-axis audit is defective."**
    - Rebalanced. Two NeurIPS reviewers named the diagnostic as a strength, and it does expose rare-class
      leakage.
    - The verified limitation stands: it is the maximum single-class OvR R², so it misses multi-class
      contrasts by up to a factor of (K−1). This is a refinement to add, not a refutation.

## B. Narrowed (the earlier statement was directionally right but too broad)

15. **The 59/67 audit.** The AI review's premise that the audit "changes the surface" is factually wrong:
    the audit scored only the representation (0/67 output scores; V: F2). The metric (R² → AUC) and the
    attacker family do change. The paper's "either exposed surface" wording should be fixed. The earlier
    "output surface not gated" finding stands for the gauntlet.
16. **"Tier 2 = white-box population LRT, mis-described."**
    - Confirmed, and refined: Tier 2 knows the noise *distribution* (and Q), not the noise realisation.
    - It sees clean vectors of a labelled population sample, never the target's.
    - No general adaptive or white-box attacker exists (V: E2).
17. **Erasers fitted on attacked rows (Johansson-type risk).** Still true for durable-guarantees. In the
    PCRL v2 / erase pilot, the eraser is fit on the train split and audits use the test split (V: C6).
    Scope it to durable-guarantees.
18. **"PCRL tuned on its test splits."**
    - Confirmed for the 60-cell grid; the paper itself admits that schedule choices used it.
    - The remedy is not "unavailable data". It is a registered held-out selection run, which does not
      exist for HMDA or Diabetes.
19. **"0.5 cleanly-compliant floor is architectural"** (rebuttal-era claim; the previous package did not
    contest it). Narrowed to a scale-dependent observation about this frozen backbone (V: C4, C5).

## C. Restored (completed work the previous assessment omitted or understated)

20. **LAFTR hard-R² criterion-matched comparison.** Completed, with full per-seed outputs on the drive.
    It is the only comparator trained against the R² criterion.
21. **Supported-pair worst-pair rescoring** of FARE and the authors' arm existed before the reviews.
    VFAE's per-class arrays also exist on the drive, despite a report saying "not_available" (V: F5).
22. **Threshold sensitivity (0.52 / 0.60)** for the audit and the published methods was already in the
    supplement.
23. **Fresh-partition held-out check** for the authors' 9 operating points: 0 flips, max |Δ| 0.013.
    Limited, but real.
24. **Final-iterate health was stored** (via the SPLINCE `pre_health` field), so a consistent
    final-checkpoint clean count can be given.
25. **Erase-layer pilot, VICReg sweep, rank-8 and cross-purpose retrain** all have per-seed outputs.
    Several were laptop-only single copies; they are now preserved in this branch.
26. **durable-guarantees raw score arrays** (739 files, ~964 MB) exist on the drive. They allowed
    TPR@1%FPR and AUC for all 59 failing configurations to be recomputed exactly.

## D. New problems found in this reconciliation (not in the previous assessment)

27. **The erase-layer pilot raised nonlinear leakage** on 48 of 60 cells. Adjusted passes fell from 32/60
    to 16/60. This overturns the "auditor channel unchanged" note (V: C2).
28. **Rank-8 HEADLINE uses a mixed comparator.** Against the matched comparator, the utility drop is
    1.1 pp, not 24.7 pp (V: C3).
29. **Cross-purpose criteria were switched twice post hoc** (submission and rebuttal). Table 10's caption
    mislabels incremental values as "gain over majority" (V: B1, B3).
30. **INLP stored R² come from a float32 solve.** A float64 recompute gives 42/60, not 43/60. INLP utility
    exceeds PCRL by more than 1 pp on 14 of 27 cells (V: D1, D2).
31. **Original LAFTR HMDA/Diabetes rows (36 of 60) are unsupported** by any located per-seed file
    (V: D1).
32. **The LEACE-on-raw "0/20" linear failures are numerically fragile.** They sit in near-singular
    directions. App. P's "majority accuracy on all datasets" is contradicted by the stored per-task JSON
    (V: D).
33. **LAFTR-hard stored-file defects:** the HMDA `pass_count` field reads 0, and PAPER_PASTE wrongly says
    PCRL R7 Diabetes passes "without collapse" (RM N-R2-07).

## E. Confirmed unchanged

These items were checked again against primary artifacts:
- Random frozen backbone, not pretrained.
- "Within 1pp" utility claim false (2 of 7 tasks).
- Refuted R²→accuracy bound live on origin/main; fix branch unmerged (V: G1).
- AAAI utility in-sample; Table 1 conventions mixed (V: E1).
- Proposition 3 does not cover the unclipped points.
- CelebA image-level splits; the `identity_CelebA.txt` identity file never used.
- CA ACS 2016–2018 spent.
- PCRL tabular test splits spent through tuning.
- FAccT ablations registered but never run.
