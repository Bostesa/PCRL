# BIOS Phase 1 Round 1 — VERDICT

**Status:** BAIL @ 1-hr (dev R² = 0.862 ≫ 0.20 threshold)
**Outcome:** Constraint enforcement FAILED. Task accuracy reached 0.874 (above the 0.80 success target), but gender info was effectively unconstrained throughout training.

## Run summary

- Instance: `[redacted-instance-id]` @ `[redacted-ip]` (g4dn.xlarge)
- Wall: 22:16:46Z → 23:46:37Z (90 min total, 67 min training)
- Commit: `cc3c1ec` (LEACE-on-full + invariant + 30-min hard cap)
- Diagnostics: PASSED (R² = 0.0453, same as Mac/AWS-1&2)
- LEACE invariant: PASSED (LEACE-fit=50000, primal=45904, holdout=4096)

## Per-epoch metrics

| epoch | dev_acc | dev_R² | task_loss | mode | wall |
|---|---|---|---|---|---|
| 0 | 0.854 | 0.880 | 0.711 | plan_b (swap @ step 20) | 14.6m |
| 1 | 0.864 | 0.880 | 0.454 | plan_b | 27.7m |
| 2 | 0.870 | 0.881 | 0.411 | plan_b | 40.8m |
| 3 | 0.869 | 0.875 | 0.387 | plan_b | 53.9m |
| 4 | 0.874 | **0.862** | 0.365 | plan_b | 67.0m → BAIL |

## Plan-B swap

Fired at `global_step=20` in epoch 0. Trip reason:
```
PLAN_B_TRIP[mid_window_jump]: |Δholdout_r2| = 0.2277 > 0.20 between consecutive
dual refreshes
```
Held-out R² in the first 30 primal steps: 0.183 → 0.334 → 0.562. The post-projection collapsed almost immediately under task-loss training, exactly the failure mode the drift diagnostic predicted at higher resolution.

## Why Plan-B couldn't recover the constraint

1. **In-batch primal R² is structurally saturated at 1.0** (d=768, batch=32 → d/N≈24, OLS noise floor ≈ 1.0). The verifier ridge of 1e-4 doesn't move the noise floor below saturation.
2. **The proxy-Lagrangian gradient `∂(λ·R²_inbatch)/∂θ` from a saturated R² is dominated by noise** — its direction is essentially random.
3. **λ correctly drove to `lambda_max=100`** (saturation streak 7,153 primal steps over ep 0-4), but a noisy gradient at large magnitude is still noise.
4. **Task loss provides a clean gradient** that the LoRA happily follows toward 0.874 accuracy. Gender info comes along for the ride because gender and occupation are correlated in BIOS (nurse/teacher female-skewed; surgeon/dentist/attorney male-skewed).
5. **Plan-B's r2_ema (PI controller's view of the constraint) is hidden from history.jsonl** — only logged via in-batch verifier on epoch end. The saturated holdout R² (0.562) shown in history is the cached pre-swap value, since post-swap there are no option_d refreshes.

## Mitigation paths (for a future Round 2)

The bail message suggested "escalate to rank=48 or abandon" — but rank wasn't the bottleneck. The diagnosis is that the **primal differentiable signal is structurally insufficient at batch=32**. Real options:

1. **Larger batch (256+)** to reduce d/N noise on the primal R². Memory may be tight on g4dn.xlarge — can use grad accumulation.
2. **Recompute differentiable R² on the held-out 4096 every K steps**, with grad through the full forward pass. Expensive (≈2× wall) but the signal is real.
3. **Adversarial primal signal** (DANN-style gradient reversal on a gender classifier head) instead of OLS R². Gradient-driven, no d/N issue. Diverges from the LEACE/PCRL theory but works empirically on BIOS.
4. **Drop proxy-Lagrangian, fix λ ahead of time, run a sweep**. With the constraint structurally unenforceable at batch=32, the dual variable is doing nothing useful.

## What worked

- All wrapper/plumbing fixes from launches 1-3 held: invariant check fired, LEACE on full N=50K, no pre-training crash.
- 5-signal monitor caught the trip correctly.
- Plan-B auto-swap fired at the right moment with the correct trip reason.
- Bail conditions fired at the right time (1-hr R²>0.20).
- Task accuracy continued to improve cleanly (0.854 → 0.874).

## Cost

- Launch #1 (key error crash): ~$0.30
- Launch #2 (LEACE-on-subset crash): ~$0.30
- Launch #3 (this run): ~$0.85 (90 min wall)
- Total: ~$1.45 of the $2.65 approved.

## Files

- `summary.json` — final config, leace_warmstart, history
- `history.jsonl` — per-epoch metrics (5 lines)
- `monitor.jsonl` — per-dual-refresh signals + PLAN_B_SWAP event (4 lines, all in epoch 0)
- `diagnostics.json` — pre-launch 4 diagnostics (all PASSED)
- `checkpoint_final.pt` — model + task head at end of epoch 4 (425 MB)
- `config.json` — full args dump
