# Verification notes (role C, 2026-10-01)

Machine-readable: `verification_report.json` (48 items: A1–A4, B1–B4, C1–C6, G1–G2, D1–D9, E1–E2, F1–F10,
G-A1–G-A11). Scripts: `recount_scripts/` (standalone, system python3, read stored outputs via `git show` or file
reads; no repo imports; no training, attacker fitting or cloud). Outputs: `recount_outputs/`. Synthetic fixtures:
`../../fixtures/` (`C_scale_invariance.py`, `G_accuracy_guarantee.py`, `D_inhouse_leace_vs_leace.py`).
Items D*, E*, F*, G-A* were produced by two forks of this role and merged unchanged; one D finding (final.pt
health) was re-derived by the parent in `A_checkpoint_selection.py`. Rebuild: run each script, then
`build_report_ABCG.py`, then `merge_report.py`.

## EARLY FLAGS

1. **Clean compliance has three counts, one per rule (A3, corrected).** Final-iterate gives 6/60 (final.pt R² ×
   final.pt health). Best-validation gives 5/60 (best.pt × best.pt). The paper's 7/60 mixes final.pt R² with best.pt
   health. The final.pt health is stored: the SPLINCE benchmark's `pre_health` loads final.pt in eval mode on the
   test split. The previous assessment said no final.pt health was stored. Its three Adult s0
   employment_analysis "clean" cells are clean under neither single-checkpoint rule. Two of them fail R² (0.062,
   0.081) at the checkpoint whose health was used.
2. **Strict R² counts by rule (A1, A2, A4), all confirmed.** Final-iterate 56/60 is the rule the paper states.
   Best-validation is 54/60. The erase-pilot "54→60/60" is a best.pt-vs-best.pt comparison and must not be set
   against 56/60.
3. **The erase-layer pilot makes nonlinear leakage worse (C2, overturned).** Project notes say "auditor channel
   unchanged". The stored per-cell auditor delta in fact rose on 48/60 paired cells:
   - mean delta: Adult 0.068→0.160, HMDA 0.128→0.298, Diabetes 0.007→0.059;
   - cells with delta ≥ 0.02: 27→44 of 60;
   - adjusted passes: 32→16 of 60 (VICReg×5: 0/42).

   Strict 60/60 is structural. A frozen union-LEACE layer sits upstream of every trainable map, and the pilot's test
   R² is within 1.2–1.8× the in-sample OLS null floor.
4. **The rank-8 headline uses a mixed comparator (C3, corrected).** Its "rank-24" column combines erase-pilot R²,
   std and rank with the original R7 task accuracies. Against the matched rank-24 erase pilot, medication_change
   goes 76.4%→75.3% (−1.1 pp), not −24.7 pp. The 25 pp loss comes from the erase-layer architecture, not from LoRA
   rank. "Decoupled from LoRA rank" holds by construction for that architecture (C6). It does not test the R7
   rank-floor rationale.
5. **The "architectural 0.5 floor" depends on scale (C4, C5).** The diagnostic measured a frozen random
   [128,128]→64 map at initialisation scale, at 9 (dataset, seed) points. The fixture shows R², held-out linear AUC
   and kNN AUC are unchanged under isotropic rescaling, while per_dim_std scales linearly. eff_rank is scale-invariant
   but not anisotropy-invariant, and std even moves under rotation. Scaling the stored pilot h_p by 1.15/1.28/1.41
   would turn 0/60 into 60/60 "clean" with leakage unchanged.
6. **Cross-purpose: two criteria and two models (B1–B3).** Each count belongs to a fixed (model, rule, criterion):

   | Model, rule | Absolute | Incremental |
   |---|---|---|
   | Original PCRL, final.pt | 26/33 | 22/33 |
   | Erase-layer union-LEACE + h_concat constraint, best.pt | 19/33 | 8/33 |

   - On the second model, all 11 LR cells go to exactly 0 pp. On nonlinear cells absolute leakage rose 20/22
     (+19.4→+24.2 pp).
   - Neither criterion was registered. The submission switched incremental→absolute hours before the deadline: the
     DUAL_CRITERIA/HEADLINE "PAPER INTEGRATION DECISION" committed in a3875c618 at 06:49, and the merge 17ef7d449
     calls it the "22→26 criterion fix". The rebuttal switched back to incremental after the results
     (PAPER_PASTE: "do NOT use as headline").
   - Table 10's caption says "over majority", but its values are incremental gains.
7. **Baselines (D).** No baseline in the submitted tables was trained against R² ≤ 0.05.
   - LAFTR-hard-R² is the only R²-trained comparator. Its full outputs exist only on the drive: 35/60 strict, Adult
     0/24 at Cotter-fallback epochs 11–17.
   - SPLINCE post-processes PCRL's own final.pt h_p, one attribute per cell.
   - Recomputing INLP in float64 gives 42/60, not 43/60.
   - LAFTR's HMDA/Diabetes rows (36/60 cells) are unreproducible (they were joined from an S3 lifecycle CSV).
   - The in-house `LEACEEraser` is not LEACE.
8. **AAAI (E, F, G-A).**
   - 59/67 is independently recomputed. It reads 64/59/51 at bars 0.52/0.55/0.60, and it was scored on the
     representation surface only.
   - The one multiclass row flips under the paper's own supported-pair rule, making the count 58.
   - The isolate-then-noise advantage at 0.55 is +4.4/+44/+88 pp. The easy cell reverses (−6.8 pp) out of
     partition.
   - Tier 2 is a population-access Gaussian LRT on one release. No adaptive or white-box attacker exists.
   - The linear-certificate "failures" against MLP/XGB are outside linear scope. The full-rank points used are
     unclipped, so Prop 3 does not cover them.
   - One correction to the previous assessment: middle full-rank T2 reads 0.5496 under both aggregation orders.
9. **The R²→accuracy "guarantee" is still live on origin/main@55e4cb1 (G1).** At R² = 0 it returns 0.5, while
   the exact 20-row counterexample reaches 0.9. `fix/retire-accuracy-guarantee` (5d4eda0) raises first, leaving its
   numeric return as dead code, and is not merged. 27 other remote branches carry a retired version.

## A. Checkpoint selection (NeurIPS R5/R7 headline grid; Adult R5 + HMDA R5 + Diabetes R7)

| Rule | Strict R² ≤ 0.05 | Clean / collapse / failed | Source |
|---|---|---|---|
| Final iterate (final.pt) | 56/60 (23+16+17) | 6/50/4 | `dominant_axis_audit.json` + `splince_benchmark/*/metrics.json` `pre_health` |
| Best validation (best.pt, Cotter) | 54/60 (21+16+17) | 5/49/6 | `per_seed_results.json` |
| Paper mixed (final R² × best health) | — | 7/49/4 | `scripts/identify_collapse_cells.py` |

- **Adult flips between rules.** s0 employment age_group passes at final.pt (0.005) and fails at best.pt (0.062);
  marital_status likewise (0.003 vs 0.081).
- **Cotter selector status.** It was "feasible", not "fallback", on Adult s1 and HMDA s1/s2. Diabetes R7 fell
  back on all three seeds.
- **Diabetes R5,** the run the λ-floor fix applies to, is 16/18 at final.pt and 18/18 at best.pt.

## B. Cross-purpose

Definitions:
- **ABS:** the mean over seeds of (concat − majority) > 1 pp.
- **INCR:** the mean over seeds of (concat − best single purpose) > 1 pp.

Each accuracy is the maximum over auditor seeds {11, 22, 33} on the test split, with auditors fit on the encoder's
train split. Per-seed raw files reproduce the aggregate to within 5e-4 pp. The single-recipient count (best single −
majority > 1 pp) is 25/33 on original PCRL and 19/33 on the rebuttal model. LAFTR is 29/33 ABS from aggregates.
LAFTR's 16/33 INCR is recounted from the aggregate's gain field (D8; the previous assessment said that field was absent).

## C. Erase layer and VICReg

- **Architecture.** A frozen `nn.Linear(128,128)` erase layer (concept_erasure `LeaceEraser` on the union one-hot,
  fit on train in eval mode) sits between `backbone.network` and `repr_proj`. LoRA is attached only to `repr_proj`
  (`--lora-target repr_proj_only`).
- **Utility and checkpoint.** Task utility is test-split accuracy (`trainer.evaluate(test_loader)`). That split is
  also the tuning split, so it is not untouched held-out data. Every pilot seed is best.pt with Cotter fallback and
  0 feasible epochs.
- **Missing run.** No VICReg×5 Diabetes per-seed file was located.

## G. Guarantees (PCRL)

| Guarantee | Classification |
|---|---|
| Prop 2 (joint LEACE) | Linear covariance guarantee on the fit law. For R5/R7 it holds at initialisation only; for the erase pilot it holds throughout training. |
| Prop 3 | Invalid. |
| Prop 4 | Fixed-f0, last-layer population statement; it does not describe the R5/R7 trained model. |
| Prop 6 | Linear R² bound with 1/λ_min amplification. MLP/XGB concatenation recovery is outside its scope; LR accuracy flags do not test it directly. |
| Empirical nonlinear resistance | Test-split and auditor-suite specific (C2). |

## Limits

- No forward passes. torch is not in system python, and repo code was not imported.
- Results that depend on local untracked files: the erase-pilot per-seed files and the rebuttal `results.json` files
  were read in place and hashed, but are not on GitHub or the drive.
- Two items are reported, not reproduced: the rebuttal Diabetes "best_epoch = 0" and the EC2 launch argv.
- `sha256: null` marks inputs given as ref + line ranges or commit messages; those inputs are cited, not hashed.
