# V2 Fix 1 Audit — Joint Multivariate LEACE Warm-Start

## Verdict

**Step 1 audit: ALREADY JOINT.** Commit `910f4fe` (the original "LEACE
warm-start of LoRA adapters (Fix 1)") implements joint multivariate
LEACE. `pcrl/training/v2_trainer.py:368-381` per purpose:

```python
oh_blocks = []
for a_name in disallowed:
    a_int = attrs[a_name]
    n_classes = int(a_int.max().item()) + 1
    oh = torch.eye(n_classes)[a_int]                # (N, c_a)
    oh_blocks.append(oh)
A_oh = torch.cat(oh_blocks, dim=1).float()          # joint concat
eraser = LeaceEraser.fit(Z.float(), A_oh)           # ONE eraser
```

This is exactly the construction the prompt describes. **Round 4 results
already reflect joint LEACE warm-start.** No code change to commit.

## Step 3 — rank sufficiency

Required rank for joint LEACE: `sum(c_i - 1)` over disallowed attrs in a
purpose. LoRA rank = 8.

| Dataset | Purpose | Disallowed (cardinalities) | Req | OK? |
|---|---|---|---|---|
| Adult | income_prediction | race(5)+sex(2) | 5 | ✓ headroom 3 |
| Adult | employment_analysis | age_group(4)+marital_status(2)+race(5) | 8 | ✓ exactly at limit |
| Adult | education_assessment | sex(2)+race(5)+income(2) | 6 | ✓ headroom 2 |
| HMDA | underwriting | race(5)+ethnicity(2) | 5 | ✓ |
| HMDA | pricing_analysis | race(5)+sex(2) | 5 | ✓ |
| HMDA | fair_lending_audit | race(5)+sex(2) | 5 | ✓ |
| Diabetes | billing_audit | race(5)+gender(2) | 5 | ✓ |
| **Diabetes** | **quality_research** | **race(5)+age_bucket(10)** | **13** | **✗ DEFICIT 5** |
| Diabetes | clinical_decision_support | race(5)+gender(2) | 5 | ✓ |

## Step 4 — empirical post-init R² (auditor metric)

Closed-form post-erasure R² (eraser applied to 64-dim backbone output)
is ~0 on every pair, confirming LEACE machinery is correct. The
real test is **post-LoRA-rank-8-truncation R²** — what the trainer
actually sees at epoch 0:

| Dataset / Purpose / Attr | post_closed | post_lora |
|---|---|---|
| Adult / income_prediction / race | 0.0000 | **0.0000** ✓ |
| Adult / income_prediction / sex | 0.0000 | **0.0000** ✓ |
| Adult / employment_analysis / age_group | 0.0000 | **0.0000** ✓ |
| Adult / employment_analysis / marital_status | 0.0000 | **0.0000** ✓ |
| Adult / employment_analysis / race | 0.0000 | **0.0000** ✓ |
| Adult / education_assessment / income | 0.0000 | **0.0000** ✓ |
| Adult / education_assessment / race | 0.0000 | **0.0000** ✓ |
| Adult / education_assessment / sex | 0.0000 | **0.0000** ✓ |
| HMDA / underwriting / ethnicity | 0.0000 | **0.0000** ✓ |
| HMDA / underwriting / race | 0.0000 | **0.0000** ✓ |
| HMDA / pricing_analysis / race | 0.0000 | **0.0000** ✓ |
| HMDA / pricing_analysis / sex | 0.0000 | **0.0000** ✓ |
| HMDA / fair_lending_audit / race | 0.0000 | **0.0000** ✓ |
| HMDA / fair_lending_audit / sex | 0.0000 | **0.0000** ✓ |
| Diabetes / billing_audit / gender | 0.0000 | **0.0000** ✓ |
| Diabetes / billing_audit / race | 0.0001 | **0.0001** ✓ |
| Diabetes / quality_research / race | 0.0000 | **0.0072** ✓ |
| **Diabetes / quality_research / age_bucket** | **0.0000** | **0.2707** ✗ |
| Diabetes / clinical_decision_support / gender | 0.0000 | **0.0000** ✓ |
| Diabetes / clinical_decision_support / race | 0.0001 | **0.0001** ✓ |

## Implications for Round 5 plan

The prompt's premise — "joint LEACE may need to be implemented or
hardened" — is wrong. Joint LEACE has been in place since Round 1's
Fix 1 and the Round 4 numbers are joint-LEACE numbers. Two distinct
failure modes show up in Round 4 outliers, only one of which is a LEACE
issue:

1. **Adult income/race outliers (s1 R²=0.20, s2=0.36) and HMDA
   underwriting/race + pricing/race outliers** — *not* a LEACE-init
   issue. LEACE achieves R²=0 at construction AND survives the rank-8
   SVD truncation (`post_lora=0.0000`). Failure happens during the 200
   constrained epochs: optimizer drift away from the LEACE feasible
   set. **Re-running Adult Round 5 with no code change is a noise
   re-roll — the same code path that produced Round 4.**

2. **Diabetes quality_research/age_bucket outlier (4/6 fails)** — this
   one *is* a LEACE rank-deficiency issue. Required joint rank is 13
   (race contributes 4, age_bucket contributes 9). LoRA rank 8 can only
   capture the top-8 singular components of `(Q − I) W`, leaving
   residual R² = 0.27 on age_bucket at epoch 0. Every other Diabetes
   pair is < 0.01. **This is fixable: bump LoRA rank to 16 (fits all
   datasets and purposes comfortably).**

## Recommended next action (vs. prompt's plan)

Prompt's STEP 5/6 (commit Fix 1, launch Adult Round 5) assume there's a
code change to validate. There isn't — joint LEACE shipped in 910f4fe.
A noise re-roll on Adult costs AWS time without information.

Concrete actionable fix is on Diabetes only:
- Raise `lora_rank` from 8 → 16 (or per-purpose: only quality_research
  needs ≥13). Re-run probe to confirm `post_lora=0` on age_bucket.
- Launch Diabetes Round 5 with bumped rank.

For Adult/HMDA: the problem is constraint-optimizer dynamics during
training, not init. That's a separate investigation (different λ
schedule, longer warmup, tighter dual cap, or auditing whether
constrained training reverses the LEACE init).

## Files

- `experiments/probe_leace_rank.py` — diagnostic script (joint LEACE +
  post-LoRA-truncation R²).
- `results/leace_rank_probe.json` — raw per-pair pre / post_closed /
  post_lora.
- `results/v2_fix1_audit.md` — this file.
