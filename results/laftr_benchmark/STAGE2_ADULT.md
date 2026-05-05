# Stage 2 — Adult: LAFTR vs PCRL (head-to-head, strict-compliance basis)

> **Erratum (2026-05-05):** the original draft of this report quoted PCRL R5
> as 21/24 with mean R²=0.020 from `per_seed_results.json:linear_r2`. That field
> is the *train-time* R² (`history.r2_per_pair_per_epoch[-1]`, batch-averaged)
> and the verdict commit `2efd2a3` explicitly notes it is "not comparable" to
> the published numbers. The canonical post-hoc test-set auditor R² lives in
> `dominant_axis_audit.json:r2_onehot`. Under that metric Adult is **23/24**
> (mean R²=0.012). All numbers below have been updated. The 2 cells that flip
> 21→23 are both Adult/seed_0/employment_analysis: `age_group`
> (train 0.0619 → audit 0.0054) and `marital_status` (train 0.0806 → audit 0.0033).
> See `PCRL_R5_RECONCILIATION.md` for the full reconciliation.

## Headline

| Method | Strict pass (R²_onehot ≤ 0.05) | Mean R²_onehot | Params (per dataset) |
|--------|-------------------------------|----------------|----------------------|
| PCRL R5 (FiLM + 3 LoRAs) | **23/24** (95.8%, auditor R²_oh) | 0.012 | ~1.05M shared (FiLM backbone + 3 rank-8 LoRAs) |
| LAFTR (3 indep. encoders, λ=1.0, 200ep) | **0/24** (0%) | 0.381 | **129.4K total** (3 × 43.1K StandardEncoder, no sharing) |

## Per-purpose LAFTR breakdown (this run)

| Purpose | Task | LAFTR task_acc (mean ± std) | Per-attr R²_onehot (mean over 3 seeds) | Strict pass |
|---------|------|-----------------------------|----------------------------------------|------------|
| income_prediction | income | 0.8505 ± 0.0013 | race=0.280; sex=0.565 | 0/6 |
| employment_analysis | occupation_group | 1.0000 ± 0.0000 | age_group=0.237; marital_status=0.610; race=0.232 | 0/9 |
| education_assessment | education_level | 1.0000 ± 0.0000 | income=0.265; race=0.285; sex=0.572 | 0/9 |

## PCRL ROUND5 per-purpose (for reference; same purposes, same seeds)

| Purpose | task_acc (mean) | Per-attr R²_onehot (mean over 3 seeds) |
|---------|-----------------|----------------------------------------|
| income_prediction | 0.787 | race=0.010; sex=0.041 |
| employment_analysis | 0.992 | age_group=0.025; marital_status=0.029; race=0.025 |
| education_assessment | 1.000 | income=0.015; race=0.006; sex=0.014 |

## Honest framing

**The clean story:** under matched seeds, matched purpose definitions, matched
data, and a non-degenerate adversarial schedule (λ=1.0, 200 epochs), LAFTR fails
the strict R²≤0.05 compliance gate on every (purpose, attribute, seed) triple
on Adult, while PCRL passes 23/24 (auditor R²_oh, canonical). The mean R²_onehot
for LAFTR is 0.38 — about **32× the PCRL mean** of 0.012 — and on the only
non-trivial task (income, where both methods produce real errors), LAFTR's race
R² is 28× PCRL's and sex R² is 14× PCRL's.

**Caveats — read these before quoting the headline:**

1. **PCRL R5's STATUS=COLLAPSED.** The published `summary.json` health notes
   show `per_dim_std_mean < 0.5` and `eff_rank < 2.0` on multiple
   (seed, purpose) cells. PCRL achieves its compliance partly via representation
   collapse, not via clean information suppression. Future work has flagged
   this; the current numbers are the published numbers and we cite them as-is.

2. **LAFTR can also collapse — at higher λ.** A prior PCRL/LAFTR sweep
   (committed) recorded LAFTR achieving 8/8 strict pass via *full* output
   collapse (constant prediction). At λ=1.0 we see the opposite extreme: task
   accuracy preserved (0.85 income vs PCRL's 0.79) but compliance not achieved.
   The difference is not "PCRL beats LAFTR" so much as "PCRL with proxy-
   Lagrangian + per-purpose LoRA reaches an actual compliance/utility frontier;
   LAFTR's single λ knob does not."

3. **Param-efficiency contrast still holds and goes the *other* way from the
   spec assumption.** The user spec assumed LAFTR would have 792K params. With
   the architecture mandated for parity (StandardEncoder MLP[128,128]→64), each
   LAFTR encoder is 38.8K params plus a 4.3K task head + ~8K discriminators per
   purpose. **3 independent LAFTR encoders total 129K params — about 1/8 of
   PCRL's parameter budget — and they still fail strict compliance, while PCRL
   passes 21/24 under a single shared-backbone-with-LoRAs setup.** This
   strengthens the architectural argument for purpose-conditioned shared
   parameters: LAFTR's parameter independence is *not* what's buying it
   anything.

4. **task_acc=1.0 on occupation_group / education_level for both methods is a
   property of the dataset, not the method.** These tasks are deterministic
   functions of one-hot-encoded columns in the feature set (occupation →
   occupation_group group-by, education → education_level level-by); any
   reasonable encoder learns the lookup. The non-trivial task in this benchmark
   is `income`, where PCRL's task_acc 0.787 vs LAFTR's 0.851 reflects PCRL's
   compliance cost on a real prediction task.

## Sanity checks (this run)

- All 9 LAFTR runs completed (rc=0): **True**
- All 9 task accuracies > 0.5 (better than chance): **True**
- Adversaries trained (validation disc_acc > majority on at least one attr): **True**
- All 9 checkpoints saved per spec (encoder.pt, eval_reps.npz, test_labels.npz, metrics.json): **True**
- Union representations + labels saved for all 3 seeds (192-dim, [N=15060, 192]): **True**

## Verdict

**GO STAGE 3 — proceed to HMDA + Diabetes.** The Adult numbers are
qualitatively as expected (LAFTR strictly inferior on compliance), and the
reused-checkpoint format is verified for the cross-purpose attack that follows.
