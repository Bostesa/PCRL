# Mathematics and design review: refreshed guarded joint study, before the lock

**Role:** mathematics and design reviewer, 2026-10-04.

**Scope:**
- the study prompt (sections 6 to 9, 11 and 12);
- `rgj/train.py`, `rgj/run.py`, `rgj/select.py`, `rgj/family.py`, `rgj/infer.py` and `rgj/critic_track.py`;
- the predecessor's locked engine (`jcv/train.py`, `pnx/train.py`, `pnx/critic_gap.py`, `pnx/select.py`, `pnx/family.py`, `pnx/infer.py`), its `review/MATH_REVIEW.md` and its `RESEARCH_DECISION.md`.

**Checks run.**
- Synthetic CPU fixtures only. Inputs are DEFENSE_FIT-shaped synthetic arrays: 19,230 rows, 83 inputs, a SEX prior of about 0.67, and 76 encoder steps per epoch. No study row was read and there was no scientific fit.
- Fixtures: `rgj/tests/test_math_review.py` (34 tests, about 5 s). Each test was written so that it fails on the defect it guards; failing receipts are listed at the end.

**Verdict.**
- Gradient signs, the surrogate, the multiplier rule, coefficient matching, transport algebra, β = 0 parity, final-snapshot alignment, the 18-slot family and the grouped bootstrap are correct.
- Two REQUIRED defects were found:
  - R1, in the training engine, is now repaired and verified.
  - R2, a one-line selection fix, is now repaired and verified (its failing receipt now passes).
- η_dual = β is **weak but not inconsistent** (A1). The prompt allows no repair on that ground.

## REQUIRED

### R1. Block-fixed floored transform plus a moving training head made the refreshed critics fail (repaired; verified)

**Defect (first `rgj/train.py`).**
- The critic view was v_i = [r_i, C(W_t r_i + b_t)], where C = I − 11ᵀ/K and (W_t, b_t) is the *current* training head.
- Write v − v̄ = M(r − r̄) with M = [I; C W]. The covariance M Σ_r Mᵀ then has K exact null directions u = [−Wᵀ C u_l; u_l].
- The floored ZCA amplifies these directions by 1/√(10⁻⁸ λ_max), about 3 × 10³ to 4 × 10³ here.
- The refreshed arms hold the transform T_k, fitted at the refit snapshot, for a whole block of about 304 encoder steps. The training head takes an SGD step on every step, so

  u_nullᵀ (v_t − μ_k) = u_lᵀ C [ (W_t − W_k) r_t + (b_t − b_k) ] ≠ 0.

- Encoder drift alone never excites these directions: with the head fixed, v_t − μ_k stays in col(M).
- The same mechanism produced the predecessor's 73.9-nat artefact (predecessor review R1). There it was a one-step lag at diagnosis time. Here it was built into training for 304 steps.

**Measured (synthetic, the engine's own `Transform`, `train_run` and `refit_banks`).**

| Quantity | Snapshot | 1 step | 76 steps | 304 steps |
|---|---|---|---|---|
| max \|T_k(V_t)\|, v1 (stale floored) | 5.7 | 56 | 425 | 1522 |
| same, encoder moving, head frozen | 5.8 | | | 5.8 |
| same, head moving, encoder frozen | | | | 997 |

- **L-R at β = 0, online critics on CRITIC_VAL** (constant CE 0.633):
  - best of bank at step 1: v1 0.515, v2 0.549, pair 0.470;
  - by step 20: 0.552, 0.650 and 0.594, and around or above the constant for the rest of the block;
  - the old per-step schedule (L-O) held 0.52, 0.56 and 0.48.
- So the refreshed arms' *online* critics, which supply every encoder gradient between refits, were worse than the inherited schedule's. That inverts the intended repair.
- At refits 4 and 8, the transported continued critic started at CE 0.55 to 0.71 at β = 0 (up to 1.57 at β = 0.3) and lost 12 of 12 times at β = 0.
- **L-R at β = 0.3 over 8 epochs:** median penalty-gradient norm 0.111 (moving head) vs 0.274 (fixed head). Calib R at epoch 8:

  | View | Moving head | Fixed head |
  |---|---|---|
  | v1 | 0.058 | 0.032 |
  | v2 | 0.038 | 0.006 |
  | pair | 0.090 | 0.040 |

  The defect roughly halved protection at a given β.

**Repair adopted (R1-a, verified).**
- Every critic view (online steps, refits, penalty, calibration, captures, tracking and the whitening diagnostic) uses one fixed per-seed head, the warm-start training head (W_w, b_w): v_i = [r_i, C(W_w r_i + b_w)].
- This holds for every critic-bearing arm, refreshed and online, so that J-O vs J-R remains a schedule-only contrast.
- Why it is sound:
  - The view carries the same information as r_i, because the logits are affine in r_i.
  - The null space is constant, so it is never excited.
  - Transport in view space is now exact *and* meaningful, because V_old(r) = V_new(r).
  - L-R → Stage C inheritance needs no extra map.
  - β = 0 parity is untouched: critics never enter the encoder at β = 0. The lead reports 21 of 21 parity units bitwise equal on the repaired code.
- Rejected alternative: freezing the head per block alone. At each refit the head snapshot changes, view-space transport then evaluates the old critic off its manifold, and the start CE reached 25 to 94.
- The capped transform reduces the blow-up (max |z| about 17 after 304 steps) but does not remove it.

**Fixture:** `test_block_transform_stays_valid_within_refit_block`.
- It passes on the repaired code.
- With `critic_views` patched to use the moving head it **fails**: v1 max |z| goes from 5.6 to 226 by step 20.
- The lead's `test_fixed_head_keeps_block_inputs_bounded` covers the same property.

### R2. C\* drops a feasible TASK_ONLY_ALIAS control (repaired; verified)

**Defect.**
- In `rgj/select.py` (`select_C`, the C\* filter), candidates must have status `NOMINEE`.
- A feasible L-R or L-O whose Stage B selection was the U-B alias carries status `TASK_ONLY_ALIAS` and is therefore excluded.

**Why it is wrong.**
- Prompt §7 lists L-R and L-O among the C\* candidates.
- U, itself a task-only release, is eligible.
- `family.claim_decision` already says C\* may be task-only (disclosed), and `comparator.is_task_only_release` anticipates an alias C\*.
- Excluding a feasible control can only raise C\*'s coalition AUC, which makes claim B anti-conservative.

**Failing case** (`test_cstar_considers_every_feasible_control_including_task_only_alias`):
- L-R is the U-B alias: feasible, coalition AUC 0.750.
- The other feasible controls are J-R 0.785, L-G 0.790, J-O 0.790 and E 0.800.
- Protocol C\* = L-R; code C\* = J-R.

**Fix.**
- Accept status in {NOMINEE, TASK_ONLY_ALIAS} with `feasible = True`.
- Carry the status into `comparator`.
- Keep the J-G nomination guard as the minimum over C\*'s local AUCs.
- Check that `write_table` labels an alias winner "selected".

**Status.** The lead changed the filter to status in {NOMINEE, TASK_ONLY_ALIAS} with `feasible = True`. `test_cstar_considers_every_feasible_control_including_task_only_alias` now passes, and the full `rgj/tests` suite passes (71 tests).

## 1. Gradient and constraint signs

- **Critics** minimise CE(S | critic(T(v))) with Adam on detached views of CRITIC_FIT rows. `opt.zero_grad()` runs before every critic `backward`, and the encoder step uses `autograd.grad`, so stray critic gradients cannot carry over.
- **Encoder objective:** L_task + Σ_v w_v R_v with w_v ≥ 0. R_v = (CE_const − min(CE_const, CE_A, CE_B)) / H gives ∂R_v/∂θ = −(1/H) ∂CE_sel/∂θ. Descending on it raises the selected critic's CE.
- **Multipliers:** λ_i ≥ 0 rises iff R_i > c_i, so the weight on R_i increases when recipient i violates its budget.
- **Fixtures:**
  - `test_privacy_step_reduces_readable_signal_and_critic_learns_sex`: the critic beats the constant by more than 0.02 nats, and one step along −∇R lowers R for that critic.
  - `test_dual_update_sign_clip_and_cap`.
  - `test_guarded_multiplier_rises_only_for_the_violating_recipient`, an integration test through `train_run`.

## 2. The surrogate: zero, clamp, and the difference from 1 − CE/H

**Gradient: identical to the old surrogate.** Selection uses the same argmin over the same three losses, and CE_const does not depend on θ. So the encoder gradient is identical to the predecessor's on every batch where the same critic is selected.

**Value: two differences.**
1. **An additive batch offset.** When a critic wins, R_new − R_old = (CE_const,b − H)/H.
   - For a 0.67/0.33 prior, H = 0.635 and the per-row SD of −log p(S) is 0.331.
   - The offset therefore has SD 0.033 on a 256-row batch and about 0.0097 on the ~2.9K CALIB rows.
   - The old per-batch R values (around 0.02 to 0.05 in the predecessor) were dominated by label-composition noise.
   - On fixed CALIB rows the offset is a constant, so under the old convention it would cancel in R − c, but only while a critic wins in both terms.
2. **Clamping.**
   - When the constant wins, R_new = 0 exactly, with no gradient path. R_old = 1 − CE_const,b/H, which can be negative.
   - Consequence: a failed critic reports R = 0. That reads as "no recovery", not "protected". The dual is then silent (λ is not raised), which is why selection relies on independent attackers.

**Float detail.**
- A critic whose logits equal log p gives CE equal to CE_const only to about 1e-8 in float32. It can tie or win and give R of about 5e-8, with an exactly zero input gradient.
- The claim "exactly 0" holds only up to rounding. Advisory: prefer the constant on ties, i.e. require a strict improvement larger than about 1e-7.

**Fixtures:**
- `test_constant_critic_gives_zero_recovery_and_no_gradient`, with batch SEX shares of 0.50, 0.67 and 0.90. The old convention would give |R| > 1e-3 there.
- `test_worse_than_constant_critics_clamp_to_zero_without_gradient`, which catches a min that omits the constant.
- `test_recovery_value_and_gradient_come_from_selected_critic_only`.

## 3. Multipliers: sign, clip and reachable scale (A1)

**Rule.** λ_i ← clip(λ_i + β(R_i − c_i), 0, 3β).
- With η = β, λ/β follows the same trajectory for every β given the same R path. This is dimensionally consistent.
- Updates that can act on training happen at refits 0, 4, 8, 12 and 16. The epoch-20 update is correctly skipped.
- At epoch 0 the model *is* the L-R reference, so R_0 − c ≈ −0.005 ± the refit noise of A2, and λ stays about 0.
- Hence λ ≤ β Σ_{k=4,8,12,16} (R_k − c)₊, from four informative updates.

| Violation per update | λ after 4 updates | Effective weight on R_i (J-G) | Ratio to base β/3 |
|---|---|---|---|
| 0.01 | 0.04β | 0.373β | 1.12 |
| 0.03 | 0.12β | 0.453β | 1.36 |
| 0.10 | 0.40β | 0.733β | 2.20 |

- **The cap is unreachable.** Reaching 3β requires a mean violation ≥ 0.6 per update in a surrogate bounded by about 1.
- Checkpoint epoch 5 has seen only the epoch-0 and epoch-4 updates, epoch 10 three updates, and epoch 15 four.
- **Decision: weak, not a concrete inconsistency.** The cap is a safety bound, not a target. A rule that can act only four times per run is a bounded heuristic, as the prompt says.
- No repair is authorised on this ground. Strengthening η would be a design change.
- **Register before fitting:**
  - λ_i/(β/3) ≲ 1 in all runs;
  - J-G and J-R are expected to differ little.
- Report the λ trajectories and effective weights (logged in `lambda_trace`).

## 4. Base-coefficient matching, and what it does not match

| Arm | Weight on R_1 | Weight on R_2 | Weight on R_pair | Sum |
|---|---|---|---|---|
| Joint | β/3 + λ₁ | β/3 + λ₂ | β/3 | β (+λ) |
| Local | β/2 + λ₁ | β/2 + λ₂ | 0 | β (+λ) |

The code matches this table exactly. The pair view is not even formed in a local arm's penalty, because the `active` list excludes zero weights. The equal sums do **not** equalise the following:
- **Direct per-recipient pressure:** 1.5× higher in the local arm (β/2 vs β/3).
- **Coalition pressure:** in the joint arm the coalition term's gradient flows into both encoders. At best response R_pair ≥ max(R_1, R_2), so the joint arm's base penalty *value* is larger, and the pair term can act as extra local pressure (the training pair bank has no ignore-recipient critics).
- **Gradient norms and the clip:**
  - the global norm clip at 5 changes the task share differently by arm;
  - report ‖t‖, ‖penalty‖ and clip hits, which are logged every 20 steps.
- **λ trajectories:** these differ by arm by design. In the local arm R_i is pushed harder, so violations, and hence λ, are smaller.
- **Grid strength vs the predecessor (A7):**
  - the /3 normalisation makes the new joint β = 0.3 equal to the old PN β = 0.1 per term;
  - the new local β = 0.3 equals the old LN β = 0.15 per term;
  - the new grid {0.03, 0.1, 0.3} therefore sits at or below the predecessor's weakest PN setting.

**Fixtures:**
- `test_base_coefficient_sums_match_and_local_pair_is_zero`.
- `test_local_arm_coalition_bank_never_reaches_the_encoder`. Replacing L-G's pair (shadow) bank leaves its encoder bitwise unchanged, both with refits and with refits disabled. J-G's encoder changes, which shows the fixture is sensitive.

## 5. Continued-critic transport

**Convention.** z = (V − μ)W with W symmetric, and the layer computes y = z Aᵀ + b. Requiring (V − μ_n)W_n A'ᵀ + b' = (V − μ_o)W_o Aᵀ + b for all V gives

  A' = A W_o W_n⁻¹,  b' = b + A W_o (μ_n − μ_o).

`transport_first_layer` computes this from the float64 factors (`solve`, not an inverse of the float32 W) and stores the result in float32.

**Exactness.**
- On non-commuting W_o and W_n with a mean shift, on and off the data manifold, the pre-activation error is about 5e-8 relative to the pre-activation's absolute scale, i.e. float32 storage level.
- The swapped-order formula A W_n⁻¹ W_o is detectably wrong on the same case.

**Amplified case.**
- W_sv_max / W_sv_min > 10³ (floored).
- The float32 training path reproduces the old logits to about 7e-3 on logits up to 24, roughly 3e-4 of logit scale. The untransported float32 path has an error of about 1e-4.
- Under R1-a the floored null directions are structural and the same at every snapshot, so W_o W_n⁻¹ is well conditioned on them.

**Fixtures:**
- `test_transport_row_column_convention_exact[floored|capped|raw]`;
- `test_transport_identity_when_transform_unchanged`;
- `test_transport_amplified_direction_float32_path`.

## 6. Snapshot and transform alignment

**Captures `captures[s]`.** These are taken after the step-s critic updates and before the step-s encoder update, so they hold θ_{s−1}, the critics trained at θ_{s−1}, the transform in use, and `critic_head`.
- `final["theta_T_minus_1"]` is this capture at the last step.
- `final["refit_critics"]` holds θ_T, the epoch-20 refit critics and T_20.
- `rgj/critic_track.py` pairs each model with its own critics, transform and head: start = step 1, mid = step 761, final = θ_{T−1}, and refit-at-θ_T.

**Checkpoint caveat (A5).** `checkpoints[e]` (and `critics.pt` in release units) stores the model *after* the last encoder step (θ_T, the release) together with the online critics of θ_{T−1}.
- That is correct for Stage C inheritance, where the epoch-0 refit retrains on frozen θ.
- It must never be used for a gap diagnosis. Label it in the record, e.g. `"critics_aligned_with": "theta_{step-1}"`.

**Fixture:** `test_final_snapshot_alignment_labels`.

## 7. Selection logic (`rgj/select.py`)

**Stage B.**
- Candidates are 12 checkpoints plus the U-B alias. Key: (worse local AUC, mean local AUC, β, epoch).
- If U-B fails its own G3, the result is NO_VALID_REFERENCE. If the alias wins, TASK_ONLY_ALIAS.
- This matches the prompt.

**Stage C, trained controls (L-G, J-R, J-O).**
- Rule: gates vs U, local AUC ≤ L-R per recipient (zero buffer), then minimise (coalition, β, epoch). This is J-G's rule without the extra comparators.
- It removes the predecessor's asymmetry, in which LN was selected for low *local* AUC while PN was selected for low *coalition* AUC.
- Fixture: `test_trained_controls_use_zero_buffer_coalition_rule`.

**Zero buffer vs the +0.01 clause.**
- Nomination uses inner point estimates with a zero buffer. The final clause (upper bound < 0.01) uses assessment intervals.
- The buffer is prospective, the final allowance is unchanged, and they are consistent.

**C\* and the J-G nomination guard.**
- C\* is chosen before J-G.
- J-G's guard per recipient is min(L-R, L-G if a nominee, C\* if present).
- Fixture: `test_jg_nomination_guard_is_min_over_available_comparators`.
- R2 covers the alias exclusion.

**Feasibility outlook (A6, heuristic).**
- Inner AUC on 2,235 rows has a level SE of about 0.01, and paired differences are smaller.
- Each comparator's inner AUC is a selected minimum, which biases it low (winner's curse).
- If J-G's local AUCs truly equalled the comparators', a single J-G checkpoint would pass both recipients' "≤ min of three" with probability of order (1/4)².
- With 12 correlated checkpoints and three seeds, NO_FEASIBLE_NOMINEE on at least one seed is the likely outcome under no true local advantage.
- The rule is conservative for J-G, as intended. Predict this outcome so that it is not read as a surprise.

## 8. Family and bootstrap

**Family.**
- z = Φ⁻¹(1 − 0.05/(2·18)) = 2.991316115 (`norm.isf(0.05/36)` agrees to 1e-12).
- 18 slots: claim A has 9 (coalition, two local, and six utility clauses); claim B has 3 comparator slots plus 6 declared aliases (P13 to P18 ≡ P04 to P09).
- If C\* = L-G, P10 to P12 equal P01 to P03 numerically but remain separate slots, so the family never shrinks.
- Two-sided Bonferroni per slot is conservative for the one-sided rows, and each claim is an intersection–union test, so the conjunction needs no further adjustment.
- `claim_decision` requires all nine PASS plus a valid reference, a J-G NOMINEE on all three seeds, and an L-G NOMINEE (claim A) or an existing C\* (claim B).

**Bootstrap.**
- `rgj/infer.py` resamples exact-record groups multinomially, with the same weight matrix for every quantity, arm and seed. Per-seed statistics are averaged and assessment rows are asserted identical across labels.
- Replicated seed outputs therefore add no records.

**Fixtures:**
- `test_z_value_for_18_slot_two_sided_bonferroni`.
- `test_bootstrap_resamples_record_groups_and_seeds_add_no_records`. Rows in a group share their weight. Identical seeds averaged under paired weights give exactly the one-seed SE. Treating seeds as extra people shrinks the SE by about √3, and the test detects that.

## 9. The planted 1e-6 clue (§8)

**Construction.** A SEX clue of amplitude 1e-6 is planted along a *rotated* low-variance direction: x₁₂ = x₁₁ + 10⁻⁶(2S − 1) + 10⁻⁷ε. Its variance is about 10⁻¹², far below the floor of 10⁻⁸·λ_max.

**Results.**

| Transform | W_sv_max | Clue scale in z | Re-whitened LR AUC | Plain LR(C=1) on z | Bounded critic B |
|---|---|---|---|---|---|
| floored | 7.1e3 | 7e-3 | 1.000 | 0.976 | 0.476 |
| capped | 71 | 7e-5 | 1.000 | 0.488 | 0.494 |
| raw | 1 | 1e-6 | 1.000 | 0.488 | 0.494 |

- **Information is preserved by every kind.** The clue is recovered after the float32 critic-input path by an exact float64 re-whitening (the suitably scaled classifier). T is invertible on the clue direction to below 1e-9 (the equivalently transported classifier).
- **The finite bounded critic is blind under all three kinds.** Blindness is an optimisation property. It never shows that less information is present, so `rgj/whiten_diag.py`'s BLIND label must be read that way.
- The audit's planted check uses appended, coordinate-aligned columns, which scale-invariant attackers can see. A rotated signal below about 1e-9 relative variance is invisible to the whole final slate, including DA, which drops directions at ≤ 1e-9·max. Disclose this as a limitation of the audit.

**Fixtures:**
- `test_planted_1e6_clue_survives_each_transform[floored|capped|raw]`.
- `test_planted_clue_fixture_rejects_a_truncating_transform`: a whitening truncated at 1e-9·max gives AUC < 0.6, so the planted test can fail.

## ADVISORY

- **A1.** Multiplier reach (section 3). Register the prediction λ/(β/3) ≲ 1 and J-G ≈ J-R. No repair. Registered in PROTOCOL §10 (with the A6 prediction).
- **A2. Calibration estimator vs the in-run estimator.** Kept as registered (the prompt specifies fresh critics); the gap will be reported.
  - Current state: c_i comes from best-of-two *fresh* critics (seeds "calib" and "calib-restart-999") at the L-R state. The in-run R comes from best-of(transported continued, restart "refit-restart-k").
  - At the same state (synthetic, L-R at epoch 5), R_epoch0 − c was −0.0020 (v1 and v2, β = 0.1), −0.0051 (v1, β = 0.3) and −0.0036 (v2, β = 0.3), instead of the design value −0.005. Fresh-vs-fresh seed noise in calib R was 0.001 to 0.010 across v1, v2 and pair. This is the same order as the 0.005 margin, so dual steps are partly noise-driven.
  - Option: compute c_i with the identical epoch-0 refit (inherited L-R critics and transforms, tag "refit", ep 0, same seeds). Stage C epoch 0 then gives R_0 − c = −0.005 exactly, a testable identity.
  - Otherwise, disclose the noise-to-margin ratio.
- **A3. Online kind-B overfitting within a block.** Adopted: `rgj/critic_track.py` adds a β = 0 J-R tracking baseline from the same L-R start.
  - Even at β = 0 with the fixed head, kind B's online CE on CRITIC_VAL rises over a block (v1 0.519 → 0.610, pair 0.485 → 0.717 by step 304) while kind A holds. The cause is about 1,520 Adam updates on CRITIC_FIT with no early stopping.
  - The registered gap (the mean over kinds) therefore mixes overfitting with encoder evasion.
  - Report gaps per kind, plus a β = 0 refreshed baseline gap, so that staleness without an adversary is separated from tracking failure.
- **A4. Low-variance growth at β > 0.** Adopted: the block-end maximum input scale is logged at each refit. Under the fixed head, max |T_k(v)| still grows within a block when the encoder is penalised (J-R at β = 0.1: 5.8 → 17 by step 304). The encoder moves r into directions that had low variance at the snapshot. Log the per-block maximum input scale and the energy in the bottom eigendirections.
- **A5.** Checkpoint critics are θ_{step−1}-aligned (section 6). Adopted: each checkpoint record carries a critics-alignment label.
- **A6.** Zero buffer vs min(L-R, L-G, C\*) on noisy inner AUC (section 7). Predict NO_FEASIBLE_NOMINEE as likely.
- **A7.** Grid strength relative to the predecessor (section 4).
- **A8.** METHOD_CARD wording on coefficient matching should say what is *not* matched (section 4).
- **A9.** Encoders train on all DEFENSE_FIT rows, including CRITIC_VAL and CALIB. Calibration and refit-choice recovery are therefore measured on rows where the encoder optimised its obfuscation. This is equal across arms and the reference; disclose it (as in the predecessor's fresh_att vs fresh_def contrast).
- **A10.** Surrogate float ties (section 2) and the rotated-clue limitation of the audit (section 9).

## Fixture status

Command: `env OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m pytest rgj/tests/test_math_review.py -q`

On the repaired code all 34 pass (the full `rgj/tests` suite: 71 pass). Before the R2 repair, 33 passed and R2's receipt failed.

| Area | Tests |
|---|---|
| Family z | `test_z_value_for_18_slot_two_sided_bonferroni` |
| Transport | `test_transport_row_column_convention_exact` ×3, `test_transport_identity_when_transform_unchanged`, `test_transport_amplified_direction_float32_path` |
| Surrogate | `test_constant_critic_gives_zero_recovery_and_no_gradient`, `test_worse_than_constant_critics_clamp_to_zero_without_gradient`, `test_recovery_value_and_gradient_come_from_selected_critic_only` |
| Signs | `test_privacy_step_reduces_readable_signal_and_critic_learns_sex`, `test_guarded_multiplier_rises_only_for_the_violating_recipient` |
| Multipliers | `test_dual_update_sign_clip_and_cap` ×3, `test_reachable_multiplier_range_with_registered_schedule` |
| Coefficients and coalition | `test_base_coefficient_sums_match_and_local_pair_is_zero` ×4, `test_local_arm_coalition_bank_never_reaches_the_encoder` |
| β = 0 identity | `test_beta0_identity_against_task_line` ×4 (L-R, L-G, J-G, J-O) |
| R1 and alignment | `test_block_transform_stays_valid_within_refit_block`, `test_final_snapshot_alignment_labels` |
| Planted clue | `test_planted_1e6_clue_survives_each_transform` ×3, `test_planted_clue_fixture_rejects_a_truncating_transform` |
| Bootstrap | `test_bootstrap_resamples_record_groups_and_seeds_add_no_records` |
| Selection | `test_trained_controls_use_zero_buffer_coalition_rule`, `test_cstar_considers_every_feasible_control_including_task_only_alias` (R2), `test_jg_nomination_guard_is_min_over_available_comparators` |

**Preserved failing receipts.**
- **R1.** Moving-head critic views fail `test_block_transform_stays_valid_within_refit_block` (v1 max |z| 5.6 → 226 by step 20).
- **R2.** Before the repair: protocol C\* = L-R (TASK_ONLY_ALIAS, coalition AUC 0.750); code C\* = J-R (0.785).
