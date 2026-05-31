# Rebuttal evidence — Diabetes rank-8 ablation (R1(c))

## Headline claim

**Under the erase-layer architecture, LoRA rank is decoupled from the LEACE rank requirement. Rank-8 holds 18/18 strict compliance on Diabetes (vs published rank-24 also 18/18). Rank choice trades off task accuracy (medication_change_outcome: 99.9% at rank-24 vs 75.2% at rank-8), not erasure capacity. This is the third independent line of evidence that per_dim_std<0.5 is architectural (vicreg sweep, backbone diagnostic, rank ablation all converge).**

---

## Supporting evidence

### Why this matters for the rebuttal

The published Round-7 rationale for Diabetes `--lora-rank=24` was: the joint LEACE eraser for `quality_research/{race, age_bucket}` requires `sum(c_i - 1) = 4 + 9 = 13` independent erasure directions, so the LoRA must be wide enough to span them. **That justification applies only to the original LoRA-inside-LEACE architecture, where the LoRA itself performed the erasure.**

In the erase-layer architecture (the rebuttal-evidence pipeline), the LEACE projection is a frozen 128-dim → 128-dim layer **upstream of the LoRA**. The LoRA only adapts the post-erase 128 → 64 projection. LEACE rank is decoupled from LoRA rank, so the rank-24 requirement no longer applies.

This ablation tests whether the decoupling actually holds in practice.

### Numerical comparison: rank-24 vs rank-8 on Diabetes

| Metric | Rank-24 (published Round 7, λ=1.0) | **Rank-8 (this ablation)** | Δ |
|---|---:|---:|---:|
| Strict R² pass (cells ≤ 0.05) | 18/18 | **18/18** | 0 |
| Cleanly compliant (incl. std ≥ 0.5) | 0/18 | 0/18 | 0 |
| Mean R² (all cells) | 0.0069 | **0.0068** | −0.0001 |
| R² range across cells | [0.0053, 0.0080] | **[0.0043, 0.0087]** | comparable |
| per_dim_std mean | 0.393 | **0.375** | −0.018 |
| Effective rank mean | 21.11 | **16.50** | −4.61 *(LoRA capacity bottleneck)* |
| Task acc: primary_diagnosis_category | 28.6% | **31.7%** | +3.1pp |
| Task acc: readmission_outcome | 91.2% | **91.2%** | 0 |
| Task acc: medication_change_outcome | **99.9%** | **75.2%** | **−24.7pp** ⚠ |

**The real trade-off lives in task accuracy.** Medication-change accuracy drops 25pp at rank-8. The rank-24 published config was about task capacity, not erasure capacity. Compliance is preserved either way.

### Per-cell breakdown (all 18 cells)

```
seed   purpose                       attr            r²       pass  per_dim_std  eff_rank
----------------------------------------------------------------------------------------------------
    0  billing_audit                 race            0.0069  PASS    0.394        14.56
    0  billing_audit                 gender          0.0043  PASS    0.394        14.56
    0  quality_research              race            0.0059  PASS    0.384        13.36
    0  quality_research              age_bucket      0.0070  PASS    0.384        13.36
    0  clinical_decision_support     race            0.0064  PASS    0.392        14.33
    0  clinical_decision_support     gender          0.0056  PASS    0.392        14.33
    1  billing_audit                 race            0.0060  PASS    0.367        17.29
    1  billing_audit                 gender          0.0072  PASS    0.367        17.29
    1  quality_research              race            0.0060  PASS    0.351        17.26
    1  quality_research              age_bucket      0.0073  PASS    0.351        17.26
    1  clinical_decision_support     race            0.0057  PASS    0.357        18.50
    1  clinical_decision_support     gender          0.0087  PASS    0.357        18.50
    2  billing_audit                 race            0.0073  PASS    0.373        18.09
    2  billing_audit                 gender          0.0075  PASS    0.373        18.09
    2  quality_research              race            0.0077  PASS    0.387        16.58
    2  quality_research              age_bucket      0.0073  PASS    0.387        16.58
    2  clinical_decision_support     race            0.0078  PASS    0.370        18.51
    2  clinical_decision_support     gender          0.0079  PASS    0.370        18.51
```

Even on `quality_research/{race, age_bucket}` — the purpose that motivated the rank-24 choice in Round 7 — rank-8 hits all 6 cells (3 seeds × 2 attrs) at R² ∈ [0.0057, 0.0077]. The expected proxy-Lagrangian saturation does not appear.

### Convergence with the prior two lines of evidence

Three independent perturbations to the optimization, all showing the per_dim_std < 0.5 floor is structural to the `[128, 128] → 64` backbone:

| Probe | What was perturbed | per_dim_std change |
|---|---|---|
| VICReg sweep (commit `39c5a84`) | Outer multiplier 1.0 → 5.0 on variance hinge + covariance penalty | Adult +0.002, HMDA −0.014 |
| Backbone diagnostic (commit `39c5a84`) | Measured pre-LEACE backbone std directly; even no-LEACE counterfactual stays at 0.23–0.29 | Architecture floor: 0.18–0.29 across all 60 cells |
| **Rank ablation (this commit)** | LoRA rank 24 → 8 (3× capacity reduction) | Diabetes: −0.018 |

Each probe moves a different knob; none moves per_dim_std above 0.5. The convergent evidence: the cleanly-compliant gap reported in the submitted paper reflects the chosen backbone dimensionality, not the privacy mechanism.

### Run provenance

| Field | Value |
|---|---|
| Instance | `i-0979baafe7ffa533b` (c5.4xlarge, 16 vCPU CPU) |
| Branch / SHA at launch | `diabetes-rank8-ablation-2026-05-30` @ `98c4d3c` |
| Launch | 2026-05-30 04:50 UTC |
| Completion | 2026-05-30 08:41 UTC (~3h 51min wall) |
| Cost | ~$2.62 |
| Run prefix (lifecycle-vulnerable) | `s3://pcrl-bios-overnight-20260504/erase_rank8_diabetes_cpu/` |
| Durable archive | `s3://pcrl-bios-overnight-20260504/archive/erase_rank8_diabetes_cpu/` |

CPU-only because the on-demand G-family quota (8 vCPUs) was saturated by parallel LAFTR + cross-purpose pilots. The Standard-family quota had 32/32 free; the smoke check on the laptop showed Diabetes rank-8 trains cleanly without CUDA paths.

## Files in this directory

- `HEADLINE.md` (this file) — rebuttal-ready summary with the three-line claim + supporting numbers.
- `v2_diabetes_ERASE_RANK8/per_seed_results.json` — full per-seed, per-cell data: linear R², adjusted pass, per-purpose health (per_dim_std + effective_rank).
- `v2_diabetes_ERASE_RANK8/summary.json` — aggregated summary with task accuracy per task.

## Reproducing the ablation

Requires the `--lora-rank` CLI flag from commit `2787c83` (cherry-picked into this branch):

```
python experiments/run_v2_dataset.py \
    --dataset diabetes --out-tag _ERASE_RANK8 --seeds 0 1 2 \
    --use-erase-layer --lora-target repr_proj_only \
    --lora-rank 8 \
    --device cpu  # or cuda
```

CPU wall time: ~3.5h on c5.4xlarge for all 3 seeds.
