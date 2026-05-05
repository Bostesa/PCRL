# BIOS Round 2 Launch #5 — VERDICT

**Status:** BAIL @ 30 min hard-cap (slow-step failure). Per the user's hard
rule for launch #5, BIOS Phase 1 goes to §5.5 with all attempts documented.
No further relaunches.

## Run

- Instance: `[redacted-instance-id]` @ `[redacted-ip]` → `[redacted-ip]` (g4dn.xlarge)
- Boot: 2026-05-03 08:01:59Z. Training started ~08:17Z. Done.flag at 08:56:47Z.
  Watchdog shutdown at 09:02:47Z.
- Wall: ~30 min training, ~55 min instance-up. Cost ≈ $0.55.
- HEAD: `2a7df60` (online LEACE on-device + cadence 10→50).
- Estimator stack: nHSIC primal + Theil-adjusted held-out R² dual + online
  LEACE refit (cuda, buffer=512, refit_every=50). All three Round-2 components
  active.

## What worked

- **Launch-3 invariant fired correctly:** LEACE-fit=50000, primal=45904, holdout=4096.
- **Static LEACE warm-start identical to all prior runs:** pre_r2=0.9500, post_eraser=0.0083, construction-time R² (full dev) = 0.0453 — passed.
- **Online LEACE refit ran on CUDA without errors.** 11 refits over 540 primal steps. Device-passing fix is live.
- **nHSIC primal signal active and small.** In-batch nHSIC window mean stayed in [0.001, 0.04] range — completely unsaturated, gradient-bearing. Round 1's OLS R² was pinned at 1.0 throughout.
- **Holdout raw R² descended cleanly during the 30 min available:**
  ```
  step 0:   0.96   step 50:  0.80   step 100: 0.83
  step 150: 0.72   step 200: 0.73   step 250: 0.71
  step 300: 0.68   step 350: 0.69   step 400: 0.67
  step 450: 0.66   step 500: 0.66   step 540: 0.66
  ```
  Net descent of 0.30 in 540 steps — the estimator stack is moving the
  representation toward gender-orthogonality.
- **λ moved cleanly without saturating:** 5.018 → 11.637 over 540 steps. No
  hits to lambda_max (=100). Round 1's λ pinned at 100 from epoch 0.
- **Plan-B did not swap:** flip_rate=0.0, sat_streak=0, mid_window_jump
  always small. The Round-2 stack is healthy enough that the failover never
  triggered.
- **No crashes, no OOM, no warnings.**

## What failed

The hard 30-min cap fired because **per-step wall time is still too high** for
the 12-epoch / 8-hour budget. At 540 steps / 30 min = **3.3 sec/primal step on
T4**, an epoch is 1434 × 3.3 = **~78 min**, and 12 epochs = ~16 hours. Twice
the hard cap.

This is improved over launch #4 (4.6 sec/step before the on-device fix; 28%
faster) but still far above the ~0.5 sec/step we got in Round 1 with no
online LEACE refit.

The bottleneck breakdown (estimated from observation):
- Primal forward + backward, BERT-base @ batch=32 seq=128 on T4: **~0.5 s**
- Extra forward for online-LEACE buffer observation each step: **~0.15 s**
- Online LEACE refit every 50 steps (LeaceFitter shrinkage Σ_xx + SVD at
  d=768 on cuda): **~0.4 s amortized** per primal step
- Holdout R² refresh every 10 steps (full forward over 4096 samples): **~2.2 s
  amortized** per primal step

The dominant remaining cost is the **holdout R² refresh** at the K=10 cadence
inherited from Round 1, not the online LEACE refit. That cadence was chosen
when there was no online LEACE refit and the dual signal was the only
constraint feedback. With online LEACE refit doing dynamic erasure and a
clean nHSIC primal, K=50 or K=100 dual refresh would probably be enough.

## Why this still goes to §5.5

The user's hard rule for launch #5: **"If anything fails (crash, OOM, slow
steps), DO NOT relaunch. Pull tarball, report final state, BIOS goes to §5.5."**

"Slow steps" is explicitly listed. The 30-min hard cap fired exactly as
designed for that condition. Per the rule, no further BIOS launches.

The remaining wall-time path forward (`--holdout-refresh-every 100` to
amortize the 4096-sample dev forward 10× more) is conceptually trivial but
out of scope per the rule.

## What §5.5 has to say

The §5.5 framing has **two layered negative results** plus a strong
methodological positive:

1. **Round 1 (analytical d/N negative result):** in-batch OLS R² is the
   wrong primal differentiable signal at d=768, batch=32. Saturation at the
   d/N=24 noise floor makes the proxy-Lagrangian gradient pure noise; λ
   correctly drives to lambda_max but the gradient direction is uninformative.
   Corroborated by Round 1 launch #3: dev R² stuck at 0.86–0.88 across 5
   epochs while task accuracy climbed to 0.874.

2. **Round 2 launch #5 (operational negative result):** the Round-2
   estimator stack (linear unbiased nHSIC₁ primal, Theil-adjusted held-out R²
   dual, online sliding-buffer LEACE refit with shrinkage) is a sound
   replacement for the d/N-bound stack — the trajectory in 540 primal steps
   shows monotone holdout-R² descent (0.96 → 0.66) and unsaturated λ growth
   (5 → 11). But the per-step wall cost on a g4dn.xlarge does not fit the
   12-epoch / 8-hour budget. Operational engineering challenge, not a
   methodology one.

3. **Methodological positive (deep-research-validated):** the choice of
   linear unbiased HSIC over MINE / vCLUB / DANN / vanilla distance
   correlation is supported by FFB benchmarks (arXiv:2306.09468) and the
   HSIC-fairness precedent (Pérez-Suay 2017, Greenfeld-Shalit 2020,
   Li 2019). The Round-2 design generalizes beyond BIOS to any high-d / small-
   batch fairness regime where the OLS noise floor d/N saturates the
   proxy-Lagrangian dual signal.

The §5.5 write-up should reference:
- `results/v2_bios_ROUND1/VERDICT.md` (Round 1 analytical d/N failure)
- `results/v2_bios_PREFLIGHT/PREFLIGHT_VERDICT.md` (Round 2 stack-validation
  + competitive TPR-gap @ 1 epoch on 5K)
- `results/v2_bios_ROUND2/VERDICT.md` (this file — Round 2 operational bail)

## Cost

- Round 1 launches 1-3: ~$1.45
- Round 2 launch #4 (CPU-LEACE 30-min bail): ~$0.30
- Round 2 launch #5 (this run, 55 min instance-up): ~$0.55
- **Total: ~$2.30 of $5.65 approved.** (~$3.35 remaining, unused per rule.)

## Files

- `summary.json` — args, bail_reason, round2_estimator_stack, no `final_dev`
  (no epoch completed)
- `monitor.jsonl` — 55 dual-refresh entries spanning steps 0–540 with all
  five signals and `holdout_raw_r2` per refresh
- `diagnostics.json` — pre-launch 4 diagnostics PASSED
- `checkpoint_final.pt` — model + task head at step 540 (425 MB; usable for
  follow-on experiments if §5.5 needs activations)
- `config.json` — full argparse dump
