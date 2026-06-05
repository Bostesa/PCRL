# PAPER_PASTE — cross-purpose constraint at training time (§5.5 extension to 3 datasets)

## Canonical R3 number (unified protocol)

Δpp = `(concat_acc − best_single_purpose_acc) × 100`, where each `acc` is `max over auditor seeds {11,22,33}` of attack accuracy, mean over 3 PCRL seeds (std in parens). **This is the protocol the submission used in `paper-body/tables/cross_purpose_attack_extra.tex` — same code path as `experiments/run_cross_purpose_attack_v2.py:436-447`.** All numbers below are on this protocol; both the submission column and the cross-purpose column are apples-to-apples.

**Attack flags (Δ > +1pp): 22/33 → 8/33.** No cell crosses from ok → FLAG. 14 cells flip FLAG → ok.

## Per-pair compliance (eval-side independent linear refit, fit on train → score on test)

| Dataset | per-pair R² ≤ 0.05 | h_concat per-attribute R² ≤ 0.10 |
|---|---|---|
| Adult | 24/24 | 9/9 |
| HMDA | 18/18 | 9/9 |
| Diabetes | 18/18 | 9/9 |
| **Total** | **60/60** | **27/27** |

## Per-cell attack table — submission vs cross-purpose (same protocol)

| Cell | Submission Δpp | Cross-purpose Δpp | Δ | Verdict |
|---|---|---|---|---|
| adult/age_group/LR | +1.62±0.59 | +0.00±0.00 | −1.62 | IMPROVED |
| adult/age_group/MLP | +7.66±3.48 | +1.39±0.30 | −6.27 | IMPROVED |
| adult/age_group/XGB | +9.28±2.62 | +3.42±0.76 | −5.86 | IMPROVED |
| adult/income/LR | −0.05±0.22 | +0.00±0.00 | +0.05 | flat |
| adult/income/MLP | +0.14±0.13 | +0.18±0.16 | +0.04 | flat |
| adult/income/XGB | +0.30±0.54 | +0.24±0.09 | −0.06 | flat |
| adult/marital_status/LR | +4.08±1.98 | +0.00±0.00 | −4.08 | IMPROVED |
| adult/marital_status/MLP | +6.87±1.77 | +0.60±0.12 | −6.27 | IMPROVED |
| adult/marital_status/XGB | +10.87±1.12 | +2.04±0.37 | −8.83 | IMPROVED |
| adult/race/LR | −0.09±0.10 | +0.00±0.00 | +0.09 | flat |
| adult/race/MLP | +1.30±1.52 | +1.12±0.21 | −0.18 | flat |
| adult/race/XGB | +2.79±0.87 | +1.21±0.16 | −1.58 | IMPROVED |
| adult/sex/LR | +0.38±0.62 | +0.00±0.00 | −0.38 | flat |
| adult/sex/MLP | +4.09±1.87 | +0.83±0.03 | −3.26 | IMPROVED |
| adult/sex/XGB | +6.70±1.57 | +2.62±0.57 | −4.08 | IMPROVED |
| hmda/ethnicity/LR | +2.51±1.65 | +0.00±0.00 | −2.51 | IMPROVED |
| hmda/ethnicity/MLP | +6.04±6.29 | +0.05±0.05 | −5.99 | flat (sub-std=6.29) |
| hmda/ethnicity/XGB | +6.45±5.34 | +0.77±0.18 | −5.68 | IMPROVED |
| hmda/race/LR | +3.79±5.04 | +0.00±0.00 | −3.79 | flat (sub-std=5.04) |
| hmda/race/MLP | +5.55±0.90 | +0.19±0.08 | −5.36 | IMPROVED |
| hmda/race/XGB | +5.14±1.32 | +1.59±0.50 | −3.55 | IMPROVED |
| hmda/sex/LR | +4.78±7.21 | +0.00±0.00 | −4.78 | flat (sub-std=7.21) |
| hmda/sex/MLP | +5.24±2.46 | +0.05±0.04 | −5.19 | IMPROVED |
| hmda/sex/XGB | +5.07±2.47 | +1.29±0.24 | −3.78 | IMPROVED |
| diabetes/age_bucket/LR | +9.94±5.88 | +0.00±0.00 | −9.94 | IMPROVED |
| diabetes/age_bucket/MLP | +9.53±2.87 | **−0.11±0.21** | −9.64 | IMPROVED |
| diabetes/age_bucket/XGB | +11.96±2.60 | +0.83±0.36 | −11.13 | IMPROVED |
| diabetes/gender/LR | +0.20±0.34 | +0.00±0.00 | −0.20 | flat |
| diabetes/gender/MLP | +0.31±0.08 | +0.00±0.05 | −0.31 | IMPROVED |
| diabetes/gender/XGB | −0.50±0.26 | −0.75±0.32 | −0.25 | flat |
| diabetes/race/LR | −0.01±0.02 | +0.00±0.00 | +0.01 | flat |
| diabetes/race/MLP | +0.00±0.00 | −0.04±0.01 | −0.04 | IMPROVED |
| diabetes/race/XGB | −0.25±0.08 | −0.11±0.07 | +0.14 | within-noise (both < 0) |

**Aggregate breakdown:** 20 IMPROVED, 12 flat (within max-std), 1 within-noise positive shift (diabetes/race/XGB, both values negative — concatenation hurts the attacker on this cell under both runs).

**Flag-count cells (Δ > +1pp under unified protocol):**

- *Submission (22/33)*: adult/{age_group/LR,MLP,XGB; marital_status/LR,MLP,XGB; race/MLP,XGB; sex/MLP,XGB}; hmda/{ethnicity/LR,MLP,XGB; race/LR,MLP,XGB; sex/LR,MLP,XGB}; diabetes/age_bucket/{LR,MLP,XGB}.
- *Cross-purpose (8/33)*: adult/{age_group/MLP, age_group/XGB, marital_status/XGB, race/MLP, race/XGB, sex/XGB}; hmda/{race/XGB, sex/XGB}. Diabetes flags zero.

## Honest-framing caveats (rebuttal text must address)

### (a) Diabetes best_epoch=0 — structural, not behavioral, compliance

On all three Diabetes seeds, `best_epoch=0` and `cotter=fallback` with `n_feasible_post_warmup=0/200`. The deployed `best.pt` is therefore the **LEACE-projected pre-training backbone**, not a model that improved over training. Independent verification: a fresh sklearn Ridge fit on h_concat → {race, gender, age_bucket} on the FULL Diabetes train split (n=50053) and scored on the FULL test split (n=10728) yields R² ∈ [−0.000015, +0.000057] — i.e., zero linear leakage on the held-out attacker. The Diabetes linear-leakage control is genuine, but it is a property of the **LEACE projection at the architecture level** rather than a property of the constrained-training dynamics. Frame as consistent with the structural-over-behavioral thesis (the erase-layer §5.5 architecture provides linear compliance for free, and the cross-purpose dual then reduces the *concatenation-specific* attacker lift on top — which is the quantity these tables measure). Do not claim that cross-purpose-constrained training improved task loss on Diabetes; it did not.

### (b) In-loop n_feasible=0/200 vs held-out compliance

The trainer's own Cotter best-iterate selector reports `n_feasible_post_warmup = 0/200` on every seed of every dataset — the in-loop per-batch closed-form ridge R² (heavy in-sample overfit on 512-sample batches; n/d ≈ 2.7 for h_concat) never declares feasibility. The held-out linear certificate (independent ridge fit on full train, scored on full test) is nevertheless satisfied at R² < 0.001 on every cell. These two measurements are not the same quantity: the in-loop number is per-batch in-sample-fit, the held-out is generalization. One honest sentence is owed in the rebuttal — the **linear certificate is held-out compliance, not in-loop feasibility**.

### (c) Erase-baseline already at floor

The erase-layer baseline (Adult/HMDA/Diabetes ERASE_PILOT) was already 60/60 strict R²<0.05. Cross-purpose adds 0 cells of strict-R² compliance and 1 borderline adj_pass cell lost on Adult (employment_analysis/race seed 0: Δacc 0.018 → 0.021, both below the 0.05 strict threshold). The work the cross-purpose dual does is on the **gain over best-single-purpose** column (the quantity that moved 22/33 → 8/33), not on the per-pair linear R².

## Provenance

- Checkpoints (9, all `best.pt`): `s3://pcrl-bios-overnight-20260504/archive/cross_purpose_{ab,diabetes}/checkpoints/`
- Per-cell aggregate + per-seed: `unified_protocol_results.json` (this directory)
- Submission reference: `paper-body/tables/cross_purpose_attack_extra.tex`
- Re-eval code path: same auditor defs as `experiments/run_cross_purpose_attack_v2.py`; wrapper in `/tmp/cp_analysis/unified_protocol.py` mirrored into `scripts/crosspurp/unified_protocol_reeval.py`.

## Appendix: prior majority-baseline number (do NOT use as headline)

An earlier draft of this artifact reported "19/33 flags" against the majority-class baseline (Δ = concat_acc − majority). That is a different protocol from the submission's table; the magnitudes are not comparable and reporting that number invites a reviewer to notice the protocol swap. Use **22/33 → 8/33** (the unified protocol above) as the canonical R3 number. The majority-baseline numbers remain in `results/v2_*_CROSS_PURPOSE_*/results.json` for completeness.
