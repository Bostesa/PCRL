# E/F/G-AAAI early flags (verification fork, 2026-10-01)

Pins: durable-guarantees@956f5c8 (all 202 files present in both the git clone and the drive tree are byte-identical);
drive `tree-durable-guarantees.tar::durable-guarantees/analysis/*` (score arrays, gate/fleet shards) were extracted
to scratch and re-scored. No attacker was fit. Item ids refer to `EFG_aaai.json`.

- **59/67 independently recomputed (F1).** All 67 configurations were re-scored from the stored held-out
  probabilities. The failure counts are 64/59/51 at bars 0.52/0.55/0.60, and the largest AUC difference from the
  stored values is 1.7e-5. Three of the failures sit within 0.004 of the bar.
- **The audit scores one surface only (F2).** The output surface was scored for 0 of the 67 configurations, although
  the audit question is "either exposed surface" (paper.tex:306-308). No per-config output readings exist anywhere.
- **The one multiclass audit row is criterion-dependent (F3).** HMDA/race noise σ=8 reads 0.551 under macro OvR and
  fails by 0.0011. Under the paper's own supported-class worst-pair rule it reads 0.524–0.531 and passes, so 59
  becomes 58. Under worst-class or all-pairs scoring it reads 0.64–0.65, driven by class 4 with 126 held-out rows.
- **Table 1 depends on the bar (F4).**
  - At 0.52 no combination has a 3-seed certified pass (4 FARE combinations are undetermined).
  - At 0.60 there are 8 certified passes and 2 single-seed rows.
  - The appendix's "4 of 36 pass at 0.60" counts the two single-seed VFAE rows as passes.
- **Worst-pair results (F5).**
  - The only non-FARE pass in Table 1, VFAE on the easy cell at Tier 1, fails every worst-case criterion. Its
    supported-pair reading is 0.566, above the supported null of 0.522. Its arrays exist on the drive, although
    `multiclass_dual_report.json` says "not_available".
  - The "Ours, full-rank" checkmarks fail the supported-pair criterion on both 5-class cells at both tiers.
  - Footnote 7's 0.014/0.060 use a different aggregation order from the macro verdict. Under the macro verdict's own
    order, the subspace easy cell passes on the representation surface (0.547).
- **The isolate-then-noise advantage is fragile on the easy cell (F6).**
  - The frontier advantage at 0.55 is +4.4 pp on the easy cell, against the paper's matched ~100 pp.
  - Out of partition, at the paper's own selected points, it reverses to −6.8 pp: full-rank 70.8% against subspace
    64.0%.
  - Middle and hard hold: +39 to +100 pp.
- **"Five of eight survivors leak at Tier 2" depends on the bar (F7).** The count is 8, 5 and 0 at 0.52, 0.55 and
  0.60. Two of the five exceed the bar by at most 0.0051.
- **Correction to the previous assessment (F9).** At the middle full-rank Tier-2 point, 0.5496 is obtained under both
  aggregation orders. The 0.5533 reading is the worst single seed, not "max-then-mean".
- **Tier 2 is a population-access attacker (E2).** It is a Gaussian LRT fit on clean pre-noise vectors of labelled
  attacker-train rows, with the noise covariance (and Q) known, scoring one release. It has no target clean vector, no
  noise realisation, no repeated queries and no model weights. No neural or white-box adaptive attacker exists.
  Averaging appears only as a separate run (one point, N ≤ 16).
- **Utility is in-sample (E1).** Baseline utility is max(in-sample own head, LR held out from the head only). Under
  held-out conventions the method orderings change on the easy and hard cells. Table 1's LAFTR "3–17%" is not
  reproducible from the stored operating points.
- **Scope of the certificate failures (G-A).**
  - The linear R² certificate, LEACE and Fair PCA fail only to nonlinear attackers, which is outside their scope. None
    was ever tested with a held-out linear probe. The 41/67 "attacked R²" readings come from a learned adapter with a
    ReLU, which is also outside linear scope; paper.tex:429-430 mislabels the attacked readings as "the certificate".
  - FNF's failure is within attacker scope but outside its density-estimate precondition.
  - FARE is coherent on the binary cell. On the multiclass cell it is outside its binary scope: 0.603–0.610 are
    macro-OvR AUCs of non-certified rows, not race pairs. The repo doc's "DP is not recovery" argument is invalid for
    an all-classifier DP bound.
  - Prop 3 covers only the clipped channel. Table 1 and Fig. 4 full-rank points are unclipped (mi_ceiling.py:230-256),
    so their Tier-2 passes are empirical, not certified.
