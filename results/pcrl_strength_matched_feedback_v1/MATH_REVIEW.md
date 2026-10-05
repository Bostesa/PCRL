# Mathematics and design review: matched update strength and active local feedback

**Role:** design reviewer, 2026-10-04. Owns this file and `smf/tests/test_math_review.py`. I edited no lead file.

**Scope:**
- the study prompt, sections 7 to 11 and 14 in full, and sections 3, 5, 12, 13 and 15 where they constrain the method;
- `smf/train.py`, `smf/select.py`, `smf/run.py`, `smf/preflight.py` and `smf/family.py`;
- the pinned predecessor `rgj/train.py`, `rgj/select.py` and `rgj/family.py`, and the predecessor's `MATH_REVIEW.md`, `RESEARCH_DECISION.md` and `GRADIENT_AND_WEIGHT_DIAGNOSTICS.csv` (`results/pcrl_refreshed_guarded_joint_v1/`).

**Checks run.**
- Only synthetic CPU fixtures were run. No study row was read, there was no scientific fit, and no real-data preflight was run.
- The fixtures are `smf/tests/test_math_review.py`: 63 tests, about 4 s.
  - Command: `env OMP_NUM_THREADS=1 PYTHONPATH=. <venv>/python -m pytest smf/tests/test_math_review.py -q`.
  - Synthetic data has DEFENSE_FIT shape: 768 rows, 83 inputs, SEX prior about 0.67, 3 encoder steps per epoch.
- Every test is built to fail on a concrete defect. The mutation receipts at the end show this: each of 14 injected defects is caught.
- The selection fixtures redirect both `R.RUN` and `R.PKG` to `tmp_path`. A guard fixture asserts that the real results folder is byte-for-byte unchanged afterwards.

**Verdict.**
- Four REQUIRED defects were found in the first `smf/train.py`: R1 to R4. Each had a failing fixture. The lead repaired all four, and the fixtures now pass. All 63 tests pass on the current code.
- The following are correct and covered by fixtures:
  - the per-encoder norm-matching rule, with float64 norms, the zero rule, the cap and stop-gradient;
  - the Phase B RMS identity and the exact common-weight symmetry;
  - the controller's sign, clip, floor, selectivity and epoch-0 receipt reuse;
  - ONLINE_MATCHED bookkeeping, checked per critic;
  - snapshot and controller pairing;
  - ρ = 0 parity;
  - the absence of a coalition gradient in local arms;
  - the Phase A and Phase B selection rules;
  - the 18-slot family, z = 2.991316 and the bootstrap seed 20261005.
- **Proposed repair P1, before PHASE_B_PROTOCOL_LOCK:**
  - **Defect.** The registered additive controller is logically valid but measurably weakened by its own designed epoch-0 step. The same asymmetric evidence moves the local allocation 8:1 from w = (1, 1), but only 3:1 after the designed common-mode step.
  - **Repair.** Apply the same unit step multiplicatively. This keeps the first update identical and changes nothing else that is registered.
  - The evidence is arithmetic, with no fit, and applies to every unit where both targets are above the 0.5 floor.

## REQUIRED (all repaired and verified)

### R1. Zero-direction steps were not counted, so the achieved ratio was overstated (repaired; verified)

**Defect (first `smf/train.py`, protection block).**
- When the constant beat both critics on every active view of a batch, P had no gradient. The `if ... P.requires_grad` branch was skipped, q = 0, and nothing was logged.
- When p_i was exactly zero (t_i ≠ 0), `normalized_direction` returned `ratio=None`, and the step was left out of `ratio_mean_i`.
- So a unit in which, say, 30% of steps carried no protection still reported `ratio_mean_i = ρ`, i.e. "exactly matched".
- The lead's early "no zero events" receipt could not detect the first path at all.

**Failing case** (`test_zero_direction_events_are_logged_when_the_constant_wins_on_every_view`).
- Setup: every critic's last layer is zero, with a wrong-way constant bias that is worse than the prior, and critic_lr = 0.
- Result: q = 0 on all 3 steps, but `zero_events = [0, 0]` instead of `[3, 3]`.

**Repair (lead).**
- With ρ > 0 and no gradient on P, both encoders now count a zero event. A nonfinite p gets a separate tag.
- p-zero steps contribute a realized ratio of 0.
- Each epoch reports n_steps, zero/cap/nonfinite counts, `realized_ratio_mean_i` (zeros counted) and the conditional `ratio_mean_i`.
- **GRADIENT_MATCHING.csv must report both means and the zero fraction.**

### R2. The controller's "scale-aware" linear reader was blind to a rotated 1e-6 clue (repaired; verified)

**Defect.**
- `probe_auc` whitened the full view [r, C(W_w r + b_w)], dropping singular directions below CTRL_REL_TOL = 1e-5 × s_max.
- §12 names the old 1e-9-variance PCA cut as a known blind spot. That cut is about 3e-5 in singular value. The new tolerance sat only about 3× lower, still far above a 1e-6 clue.

**Failing case** (`test_probe_detects_rotated_1e6_clue`).
- Data: 16-dim r with 15 nuisance directions (scale 0.5 to 3) and one rotated direction carrying 1e-6·(2S − 1) + 1e-7 noise. The fixed-head centred logits are appended in float32.
- Measured relative singular values: clue 1.25e-7; logit-block null directions 2.7e-8 and 1.7e-8.
- On 3 seeds, `probe_auc` gave calib AUC 0.505, 0.477 and 0.514.
- A whitened LR on r only, at tolerance 1e-9, gave 1.000 on all three seeds. Shuffled-label nulls gave 0.52, 0.53 and 0.45.
- Real views put the float32 logit-null floor at about 2e-8 relative: the predecessor's covariance eig_min is about 1e-14 against eig_max of about 18 (`WHITENING_DIAGNOSTIC.json`).
- A full-view tolerance therefore cannot separate the clue from float32 noise safely, because they are only about 5× apart.

**Repair (lead).**
- The whitened LR now reads the r block only (R_DIM = 16) with tolerance 1e-9.
- A linear reader on [r, C(Wr + b)] has the same function class as one on r, because the logits are affine in r, so nothing is lost.
- The float64 SVD of exact float32 inputs has error of about n·ε₆₄ ≈ 1e-12 relative, so 1e-9 sits about 10³ above it.
- The MLP stays on the full standardised view.
- Residual: see A13.

### R3. The gradient archive lacked the §7 quantities (repaired; verified)

**Defect.**
- The epoch summary kept the ratio, cos, the t-norm mean and the *global* post-clip norm.
- It dropped q norms (collected but never summarised), p norms, the a_i extremes, per-encoder post-clip norms or the clip factor, and per-epoch cap/zero counts.
- §7 asks to "archive pre- and post-clipping task and protection norms, actual update norms, scale caps".

**Failing case.** `test_gradient_archive_has_protection_proxy_and_post_clip_norms`.

**Repair.** The summary now logs:
- the means of q, p and t norms;
- `a_min_i` and `a_max_i`;
- the mean and minimum of κ;
- `post_clip_enc_norm_mean_i`;
- cap and zero counts.

### R4. Checkpoints lacked critic optimizer state and the hypothetical weights (repaired; verified)

**Defect.**
- §8 says: "save theta, critic state, critic-view head, transform, and optimizer state together". `rgj.snapshot_state` saves no Adam state.
- No-feedback twins also need `w_hyp` to resume their hypothetical trajectory.

**Failing case.** `test_checkpoints_save_critic_optimizer_state_and_hypothetical_weights`.

**Repair.**
- `snap_full` adds the critic Adam states and `w_hyp` to checkpoints, captures and refit snapshots.
- Inheritance into Phase B still resets Adam for every arm alike. METHOD_CARD should state this (A14).

## Proposed repair P1: multiplicative controller step (recommended before PHASE_B_PROTOCOL_LOCK)

**Registered rule.** w_i ← clip(w_i + c_i, 0.25, 8), with c_i = clip((AUC_i − b_i)/0.01, −1, 1).

**What is wrong with it.** The local allocation s_i = w_i / RMS(w) depends only on w_1/w_2. An additive step changes that ratio by an amount that depends on the weights' current level. So purely common-mode evidence changes how much later asymmetric evidence can do, even though in a local arm common-mode evidence is meant to be inert.

The fixture `test_additive_rule_allocation_depends_on_common_mode_history` (arithmetic only) shows the effect of the same single asymmetric update (+1, −1):

| History before the asymmetric update | w after | local ratio s_1/s_2 |
|---|---|---|
| none, w = (1, 1) | (2, 0.25) | 8 |
| the registered epoch-0 step, w = (2, 2) | (3, 1) | 3 |
| three common-mode steps, w = (4, 4) | (5, 3) | 1.67 |
| saturated, w = (8, 8), three asymmetric updates | (8, 5) | 1.6 |

The registered epoch-0 receipt reuse produces exactly v_i = +0.01, hence c = (+1, +1), wherever both targets are above the floor. Checked bitwise on a grid of 0.51 to 0.999. So every such unit enters training with its asymmetric authority cut from 8:1 to 3:1 per update. That is logically valid but ineffective-by-construction, which is the case prompt §§3 and 14 allow the reviewer to repair before the dependent queue.

**Repair P1.** w_i ← clip(w_i · 2^{c_i}, 0.25, 8). This is a gain of ln 2 per unit c in log w.

What stays the same:
- the target b_i, the 0.01 scale, the inner clip, the bounds [0.25, 8] and the update schedule (epoch 0 plus after epochs 1 to 19);
- the first step from w = 1 (1 → 2), so the registered epoch-0 behaviour is identical;
- selectivity: c_2 = 0 leaves w_2 unchanged.

Joint arms keep the registered local/pair shift: a common rise still lowers the pair share 1/(2w + 1).

Properties (fixture `test_log_rule_is_history_independent_inside_bounds_and_keeps_registered_behaviour`):
- While no bound binds, the local allocation depends only on the accumulated asymmetric evidence. One (+1, −1) update gives 4:1 after any unsaturated common-mode history.
- From (8, 8), three asymmetric updates give 8:1, against 1.6:1 under the additive rule. From (1, 1) they give 32:1, against 16:1.

**Disclosed side effect.** Under persistent common-mode violation, J-F reaches w = 8 (pair share 1/17) after 3 updates instead of 7. This is faster local emphasis in the joint candidate.

**Engine contract.** `smf.train.HP["ctrl_mode"]` is `"additive"` (the default) or `"log"`. `controller_update` dispatches on it.

Fixtures written for this contract:
- `test_engine_controller_matches_the_registered_mode_spec` checks the engine against the declared mode on a grid of 8 × 3 × 9 points. It fails if the rule is switched without the flag; mutation M14 confirms this.
- All controller and integration tests read the mode and work under either rule. The integration tests use first steps from w = 1, which are identical under both.
- Verified by running the controller subset against an in-memory log-mode engine: 13 of 13 pass.

**Rejected alternatives.**
- *Subtracting the common mode* (c_i − mean_j c_j). This deletes the registered joint local/pair adaptation: J-F would alias J-N under common-mode evidence exactly as L-F aliases L-N. That changes the candidate's mechanism, not just its gain.
- *A lower gain.* It slows windup and response equally and does not remove the level dependence.
- *More frequent updates.* Same as a lower gain.
- *A smaller target offset.* It shrinks the designed initial violation, which then still acts common-mode.

**Registration.**
- P1 is a reviewer repair under §§3 and 14, not literally one of the §10 preflight knobs (frequency, offset, gain). It can be read as a re-parameterisation of the gain.
- It counts as one of the at most two repairs. It needs a dated pushed amendment before the Phase B queue, with the additive receipts preserved.
- The real-data preflight should confirm the precondition, namely that both targets are above the floor on each seed.
- If the lead prefers to wait for the preflight, these fixtures already cover both modes.

## 1. Per-encoder norm matching (§7)

**Rule.**
- q_i = a_i p_i, with a_i = ρ s_i ‖t_i‖ / ‖p_i‖ in float64 under no-grad.
- The cap is a_i ≤ 100.
- q_i = 0 when ‖p_i‖ ≤ 1e-12 or ‖t_i‖ = 0.
- t_i is the encoder-i slice of ∇(L_1 + L_2) over (enc_1, enc_2, heads). Because the encoders are separate, only L_i reaches encoder i.

**Scale invariance.**
- For c = 2^k the result is bitwise q(cp) = q(p), since scaling by a power of two is exact in IEEE arithmetic, as is sqrt(4^k x) = 2^k sqrt(x).
- Fixtures: k ∈ {−20, −3, 1, 5, 20, 70}, plus general c ∈ {0.37, 3.7, 1234.5, 1e-4} at rtol 2e-6.
- k = 70 makes Σp² about 1e39, which overflows float32. A float32 norm returns inf and fails this test (mutation M1).
- Invariance holds only while neither the cap nor the zero threshold is active. Both thresholds are absolute, so they are not scale-free by design (A7).

**Denominator.**
- `test_denominator_is_per_encoder_task_gradient_not_head_or_both_encoders` spies on the `t` passed inside `train_run`.
- Its length is the number of encoder-i parameters, and its norm equals an independent ∇_{enc_i} L_i to 1e-5.
- The head-including and both-encoder alternatives differ by more than 0.1%, so the check discriminates. Mutations M2 and M2b are caught.

**Zero rule.**
- p = 0 → q = 0, event "p", realized ratio 0.
- t = 0 → q = 0, event "t", ratio undefined.
- ‖p‖ below 1e-12 → zero.
- Residual: the summary also counts t-zero steps as realized 0 (A15).

**Cap.**
- With ‖p‖/‖t‖ = 1e-4 and ρ = 1.5, the cap binds (a = 100), the achieved ratio 0.01 is reported, and the cap flag is set. Just above the boundary the declared ratio is restored.
- When it binds in practice: A7.

**Stop-gradient.** q carries no graph even when the inputs require gradients.

**Clipping.**
- The inherited global clip at 5 scales [t_1 + q_1, t_2 + q_2, head grads] by one factor κ. So ‖κ q_i‖ / ‖κ t_i‖ = ‖q_i‖ / ‖t_i‖ exactly: **the achieved ratio after clipping equals the pre-clip ratio**.
- What κ < 1 changes is the absolute step for both encoders *and both heads*. Heads therefore take a protection-dependent effective learning rate whenever the clip binds.
- `test_global_clip_scales_encoders_and_heads_together` patches clip to 1e-3 and gets an update norm (encoders + heads) of lr × clip, with ratio_mean still ρ.
- `test_heads_receive_task_gradient_only` checks that, unclipped, the head update at step 1 is bitwise the task-only one.
- Expected binding rate: A8.

**Directions.**
- Norm matching does not match directions. The first-order change in task loss is −lr ‖t_i‖² (1 + ρ s_i cos(t_i, p_i)).
- With the lead's engineering receipt of cos ≈ −0.01 to −0.03, first-order task opposition is at most 6% at ρ s_i = 2.1. The task cost therefore comes mainly from second-order drift and from moving the representation, not from direct opposition.

## 2. Phase B allocation (§10)

**RMS identity.**
- s_i = w_i / sqrt((w_1² + w_2²)/2), so (s_1² + s_2²)/2 = 1.
- Before caps and clipping, sqrt(mean_i (‖q_i‖/‖t_i‖)²) = ρ, verified to 1e-14 on s and 2e-6 on q for 50 random weight pairs.
- Under a zero event the recipient's share is lost, not reassigned, which matches "do not invent extra strength". `realized_ratio_mean_i` shows it.

**Range.**
- On [0.25, 8], s_max = 8/sqrt(32.03) = 1.4137 < √2 and s_min = 0.0442.
- **Feedback can raise one recipient's relative strength by at most 41%, and only by taking strength from the other.**

**Asymmetric weights.**
- In a unit test, w = (3, 1) gives ‖q_i‖/‖t_i‖ = ρ·(1.342, 0.447).
- Through `train_run` (`test_asymmetric_violation_changes_allocation_and_actual_updates`): a scripted v = (0.02, 0) at epoch 0 gives L-F w = (2, 1), epoch-0 ratio means ρ·(1.265, 0.632), RMS = ρ, and parameters different from L-N.
- At step 1, the L-F and L-N p_1 directions coincide (cos = 1) while their strengths differ. In the joint arms, J-F and J-N p_1 directions differ (local/pair mix 2:1 vs 1:1).
- L-N records w_hyp = (2, 1) but applies (1, 1).

**Common-weight symmetry.**
- `coefficients("local", (w, w))` is exactly (½, ½, 0) and `allocation(w, w)` is exactly (1, 1), because sqrt(fl(w²)) = w. So L-F is **bitwise** L-N under common-mode evidence (`test_common_mode_feedback_is_an_exact_alias_in_local_but_not_joint_arms`).
- In joint arms, (w, w, 1)/(2w + 1) changes the local/pair mix at an unchanged norm. In the unit fixture, cos(q(1,1), q(3,3)) is below 0.95 while the ratio stays exactly ρ.
- J-F therefore differs from J-N under common-mode evidence through **direction only**. This is the registered effect, not per-recipient management.

**No coalition gradient in local arms.**
- Perturbing the shadow pair bank's initial weights leaves the local arm's θ bitwise unchanged and changes the joint arm's.
- The shadow bank is trained: its parameters move from their initial values. Mutation M3 is caught.

## 3. Controller (§10)

**Sign, gain, clip, floor.**
- v = AUC − b. A violation (v > 0) raises w; slack lowers it; exactly on target leaves it unchanged.
- The inner clip is ±1 unit at 0.01 AUC, and the bounds are [0.25, 8].
- Floor: b = max(0.5, AUC_ref − 0.01). A reference of 0.505 gives a partial initial step (additive w = 1.5). A below-chance reference (0.47) relaxes protection (additive w = 0.25).
- The engine is checked on a grid against the declared mode (mutation M4 is caught).

**Epoch-0 receipt reuse.**
- (a − max(0.5, a − 0.01))/0.01 rounds to ≥ 1.0000000000000009 for every a in [0.51, 1) on a 1e-6 grid. So the clip returns exactly 1 and w = 2 exactly, bitwise in both modes.
- This matters: a 1e-14 difference between w_1 and w_2 would break the exact L-F/L-N alias.
- Epoch 0 never refits a probe; mutation M13 is caught.

**Schedule and pairing** (`test_controller_schedule_snapshot_pairing_and_diagnostic_final`).
- Measurements occur at epochs 0..n; updates at 0..n−1 are applied; the epoch-n entry is diagnostic only and leaves w unchanged (M8 is caught).
- The probe at epoch e sees exactly θ after e epochs (hash equal to `checkpoints[e]`).
- The weights used in epoch e are those after update e.
- For n = 20 this gives the registered 20 applied updates.

**Selectivity.**
- A recipient exactly on target keeps w = 1 while the other rises. Under the RMS allocation its *share* s_2 falls; that is the declared budget shift, not extra weight.

**Activation (synthetic binding fixture).**
- With persistent violation of 0.015 on recipient 1 and slack on recipient 2: w_1 = 2 after one update, and s_1/s_2 ≥ 2 with s_1 > 1.3 after three, under both modes.
- Common-mode persistent violation moves the weights but not the local allocation.

**Can 20 updates at gain 1/0.01 redistribute?**
- Yes. From (1, 1) one asymmetric update reaches 8:1 (additive) or 4:1 (log), so the controller is close to bang-bang.
- Against it are common-mode windup (P1) and noise (A3).

**Probe design** (beyond R2).
- Orientation is chosen on CRITIC_VAL and never flipped on CONTROLLER_CALIB: a calibration set whose signal is reversed reports AUC below 0.3 (M7 is caught).
- Probe fits are deterministic (seeded tags).
- The null control stays within 0.5 ± 0.08.

## 4. ONLINE_MATCHED (§8)

**Matched.**
- Per critic (view × kind), the paired REFRESHED unit's refit optimizer updates at epochs 0, 4, 8, 12 and 16 are added. The count includes the continued fit *and* the discarded restart attempt, and excludes the diagnostic epoch-20 refit.
- They are spread with floor(E·s/S) − floor(E·(s−1)/S): the sum is exactly E, the per-step spread is at most 1, and the result is deterministic.
- Fixtures:
  - a zero template is bitwise ONLINE;
  - per-critic Adam step counts equal 5·S + c_vk exactly for a template with six different counts (M6, a pooled template, is caught);
  - `refit_counts` equals `critic_refit_updates`, with the diagnostic refit excluded;
  - the extra minibatches come from their own RNG stream ([seed, salt, ep, 13]), so the base critic and task streams are untouched.

**Still different (METHOD_CARD should list these).**
1. *Views.* REFRESHED refits on a frozen snapshot, with full CRITIC_FIT epochs without replacement. ONLINE_MATCHED extras see a moving encoder and use per-step sampled minibatches.
2. *Validation.* REFRESHED early-stops on CRITIC_VAL (patience 3), keeps the best state and chooses continued versus restart on CRITIC_VAL. ONLINE_MATCHED uses no validation, and its validation forward passes are not counted.
3. *Transform.* REFRESHED holds a full-CRITIC_FIT floored ZCA for a block. ONLINE_MATCHED recomputes it every step on 4,096 reference rows, and that cost is not counted as "compute".
4. *Initialisation.* REFRESHED restarts provide fresh initialisations, and a discarded restart costs updates that change nothing. ONLINE_MATCHED spends those updates on the critics in use, so it gets more *effective* training than REFRESHED for the same count.
5. *Adam state.* REFRESHED uses a new Adam per bounded fit and resets after each refit. ONLINE_MATCHED keeps one Adam per critic for the whole run, so bias correction and moment history differ.
6. *Timing.* REFRESHED concentrates its effort at block starts; ONLINE_MATCHED spreads it uniformly.

So the schedule contrast at matched counts still mixes freshness, input-coordinate changes, validation selection, restarts and Adam history (§8 already says this).

## 5. Snapshot, transform and head pairing

- Every critic view uses the fixed per-seed warm-start head, in Phase A, Phase B, the controller probes and the reference receipt (`stage_calibrate` and `do_norm_run` both pass `head_of(load_warm(k))`). So inherited critics stay on their manifold, and transport between floored transforms is exact in view space (predecessor R1).
- Checkpoints now hold θ, critic states, the critic head, transforms, Adam states, w and w_hyp (R4).
- `checkpoints[e]` is taken at the top of iteration e, before that epoch's refit or controller step. So it pairs θ_e with critics last updated on θ at step (e·steps − 1) and the w used during epoch e − 1, as labelled.
- The controller-to-checkpoint hash identity is in §3.
- The controller reads views built with the fixed head, not the release head (A16).

## 6. Selection and nomination (`smf/select.py`)

**Phase A** (`test_phase_a_*`). The checks cover:
- the per-schedule joint pick (lowest pair AUC, ties to lower ρ);
- exclusion of a schedule missing a feasible point on any seed, even one with a lower AUC elsewhere;
- ties in compute, then in ONLINE < ONLINE_MATCHED < REFRESHED order;
- FALLBACK by the summed per-seed smallest gate shortfall;
- the local reference by worse-local AUC, then mean local AUC, then ρ;
- U e20 as a disclosed TASK_ONLY_ALIAS candidate in the pool;
- NO_VALID_REFERENCE when U lacks a 3-point gain.

**Phase B** (`test_phase_b_*`). The checks cover:
- C\* with no local guard, over {L-F, J-N, L-N, RAW-J, RAW-L, E, F, F0, U};
- a raw arm or U can be C\*, and U is flagged task-only (M12 is caught);
- the J-F nominee requires task feasibility and AUC_i ≤ L-F_i + 0.005 and ≤ C\*_i + 0.005 (M11, a zero buffer, is caught);
- the descriptive fallback by summed shortfall, then pair AUC, then ρ.

The current `smf/baselines.select_fare` sets the F/F0 status on task gates only, with no local guard, so the predecessor's R2 pattern (a feasible control silently dropped from C\*) does not recur. The fixture uses a stub of that interface; the real module is the audit owner's.

## 7. Family and inference

- 18 slots: two nine-clause conjunctions, with six aliases in claim B.
- z = Φ⁻¹(1 − 0.05/36) = 2.991316, recomputed independently.
- B = 1999 and BOOT_SEED = 20261005 (`smf/family.py`; `smf/infer.py` reads both from the family).
- `claim_decision` fails a conjunction for a missing L-F nominee (claim A) or for any non-PASS clause.
- The grouped, paired bootstrap is the inherited `stored_model_eval.pilot_infer.UnitBootstrap`, tested by the predecessor review. The independent verifier owns its reconstruction.

## ADVISORY

- **A1. Activation versus aliasing in the preflight.**
  - The designed epoch-0 step is common-mode. In L-F it is an exact no-op; in J-F it lowers the pair share from 1/3 to 1/5.
  - The preflight checks `weights_change_after_epoch0` and `feedback_changes_parameters_joint` pass through this path, so they are not evidence of per-recipient activation.
  - Register now, per unit: the number of applied updates with w_1 ≠ w_2, the maximum |s_1 − s_2|, and the L-F versus L-N parameter difference.
  - Mark L-F `INACTIVE_OR_ALIAS` when w_1 = w_2 after every applied update. Its parameters are then bitwise L-N's, which can be checked.
  - Report a near-alias (max |s_1 − s_2| < 0.05) separately.
- **A2. Windup and level dependence.** Motivates P1; see the table there.
- **A3. Gain versus probe noise.**
  - A unit step equals 0.01 AUC. CONTROLLER_CALIB has 2,327 rows; the Hanley–McNeil SE of an AUC near 0.8 there is about 0.010.
  - That sampling error is shared by the reference and later snapshots (same rows), so it largely cancels in v. Probe-refit noise does not cancel.
  - Add to the preflight: refit the reference probes under two more tags per seed (6 fits, about 15 s) and report |ΔAUC|.
  - If |ΔAUC| ≥ 0.005, controller steps near the target are noise-driven (±0.5 units). Report this; it is not a reason to tune.
- **A4. In-sample calibration.**
  - Encoders train on all NEW_DEFENSE_FIT rows, including CONTROLLER_CALIB, whose SEX labels enter the privacy gradient on those minibatches.
  - The controller's AUC is therefore in-sample for the encoder, and so optimistic. Reference and candidate share the bias, which partly cancels in v.
  - It is not a population constraint; §10 already says so.
- **A5. Local/pair balance inside a joint p_i.**
  - Normalisation acts on the *sum* c_v1 ∇R_1 + c_pair ∇R_pair. Within it, the mix is set by the raw gradient magnitudes of the two terms, and those include each transform's amplification.
  - So at equal ρ, ONLINE and REFRESHED joint arms can still differ in their effective local/pair mix (part of the scale confound survives inside the direction).
  - Log every 20th step: ‖c_v ∇_i R_v‖ for the local and pair terms and their cosine. This costs two extra backward passes on those steps.
- **A6. Bang-bang strength.**
  - Protection is full strength ρ s_i ‖t_i‖ whenever any critic beats the constant on the batch, however slightly, and zero otherwise. Its absolute size follows ‖t_i‖, including minibatch noise and task convergence.
  - Equal relative strength is therefore not equal absolute strength across recipients. Report `t_norm_mean_i` (logged) and the realized ratio.
- **A7. The cap is practically inactive and not scale-free.**
  - Predecessor unit-weight ‖p‖/‖t‖ means (penalty/task ratio ÷ β, summed over both encoders) run from 1.3 to 19 across 57 units.
  - The cap binds only when ‖p_i‖/‖t_i‖ < ρ s_i/100 ≤ 0.021, at least 60× below the smallest mean. The lead's 4-epoch real receipt had no cap events.
  - The cap's meaning depends on the coefficient normalisation (½ local, ⅓ joint) and on the 1/H scale of R. Report `a_max_i`.
- **A8. Clip activity.**
  - The predecessor hit the clip at most 12 times in 1,520 steps even at a combined ratio of about 3 (L-O, β = 0.3).
  - With ρ s_i ≤ 2.12 and task-norm means of 0.65 to 0.90, it should stay rare. Report κ.
- **A9. Raw versus normalized strength is only comparable over both encoders combined.**
  - The raw arms (`rgj` J-O/L-O, unchanged) log norms over both encoders every 20 steps, so per-encoder raw ratios are not available.
  - Reference points: J-O at β = 0.3 has a combined ratio of 1.4 to 1.7, comparable to ρ = 1.5. J-R at β = 0.3 has 0.45 to 0.58.
  - The raw ratio is heavy-tailed (maximum about 3 to 5× the mean), unlike the constant normalized one. The match is in mean only.
- **A10. ONLINE_MATCHED residual differences.** List §4 items 1 to 6 in METHOD_CARD.
- **A11. "Actual compute" tie-break.** It counts critic optimizer updates only, excluding the per-step 4,096-row ZCA, validation passes and probe fits. Disclose this. Ties at 12 decimals in AUC are practically impossible.
- **A12. Nomination details.**
  - When C\* = L-F, the two J-F guards coincide and the descriptive `nomination_shortfall` counts that guard twice. This affects the descriptive fallback only.
  - The +0.005 inner buffer sits against a final upper-bound requirement below 0.01. Predecessor local-clause half-widths were 0.007 to 0.012, so a nominee that uses the buffer will usually fail its local clause.
  - The buffer turns nomination failures into clause failures. That is legitimate but should be expected.
- **A13. Controller reader residuals.**
  - With tolerance 1e-9 on r, a structurally collapsed r direction (fewer than 16 live units, or adversarial rank loss) is kept at float32 rounding scale, about 6e-8 relative, and whitened. That feature is noise but carries no SEX information: the null fixture shows no false detection.
  - The controller bank (whitened LR plus a 64-unit MLP) is weaker than the final slate, which has HGB and a CE-selected attacker. A met budget is not a final-audit result.
- **A14. Adam reset on inheritance.** Phase B resets critic Adam states on inheritance, for every arm alike. Document it.
- **A15. t-zero steps** are counted as realized 0 rather than undefined. This is negligible because ‖t_i‖ = 0 needs a dead encoder; log it if it ever occurs.
- **A16. Controller views use the fixed warm-start head.**
  - The final audits use release views with HEAD_VALIDATION-selected heads.
  - The linear reader is invariant to this, because the logits are affine in r. The MLP's conditioning differs.
  - Disclose this as a controller-construction choice.

## Fixture inventory

The 63 tests, all passing on the current code:

| Area | Tests |
|---|---|
| Norm matching | power-of-two scale invariance (6 parametrisations, including float32 overflow); general c (4); achieved ratio, direction and float64 norm; stop-gradient; zero p/t/threshold; cap and boundary; ρ = 0; per-encoder denominator (spy inside `train_run`) |
| Allocation | RMS identity (50 pairs × 3 ρ); √2 range; asymmetric norms; exact local symmetry (6 values of w); joint direction change |
| Controller | engine against the registered-mode spec on a grid; sign, bounds and on-target; floor and epoch-0 unit step on a grid; binding fixture (1 / 3 updates); selectivity; additive history dependence; log-repair properties; windup in both modes |
| Probes | rotated 1e-6 clue (2 seeds); null and determinism; orientation not flipped on calibration |
| Bookkeeping | `spread`; `refit_counts` |
| Integration (`train_run`) | ρ = 0 parity (4 schedule/base cases, plus J-F with real probes); shadow pair bank; heads get task gradient only; global clip; zero events (R1); archive fields (R3); checkpoint contents (R4); asymmetric feedback; common-mode alias; controller schedule and snapshot pairing; zero template = ONLINE; per-bank matched counts; REFRESHED template |
| Selection | Phase A rule; Phase A fallback; U alias and NO_VALID_REFERENCE; Phase B C\* and nomination; Phase B U as C\* and the descriptive fallback |
| Family | z, size, aliases, B, seed, `claim_decision` |

**Mutation receipts.** Each defect was injected in memory into the current code, and the suite rerun:

| # | Injected defect | Newly failing tests |
|---|---|---|
| M1 | float32 norms in `normalized_direction` | scale invariance at 2^70, ratio/float64 norm, cap |
| M2 | denominator = both-encoder task norm | per-encoder denominator |
| M2b | denominator includes the head gradient | per-encoder denominator |
| M3 | local arm puts weight on the pair | 10 tests: local symmetry, shadow pair, asymmetric/common-mode integration |
| M4 | controller sign flipped | 8 tests: spec grid, sign, binding, selectivity, integration |
| M5 | allocation without RMS normalisation | 13 tests: RMS identity, range, symmetry, binding, integration |
| M6 | matched template pooled across banks | per-bank matched counts |
| M7 | probe orientation re-chosen on calibration rows | orientation fixture |
| M8 | final (epoch-n) measurement acts | schedule/pairing fixture |
| M9 | task order depends on the arm | ρ = 0 parity (5), denominator, heads |
| M10 | no cap | cap fixture |
| M11 | J-F buffer 0 instead of 0.005 | Phase B nomination |
| M12 | C\* excludes the task-only U | Phase B U as C\* |
| M13 | epoch-0 probe refitted instead of reusing the receipt | asymmetric, common-mode, schedule |
| M14 | log rule switched in without the `ctrl_mode` flag | spec grid, epoch-0/floor, windup |

Against the first `smf/train.py`, before R1 to R4 were repaired, the REQUIRED fixtures failed with these receipts:
- `zero_events [0, 0] ≠ [3, 3]`;
- probe AUC 0.505, 0.477 and 0.514 on the planted clue (3 seeds);
- `q_norm` missing from the summary keys;
- no `opt` key in the checkpoints.

## Prior-art check (bounded, primary sources only)

I read the three PMLR proceedings PDFs, not secondary summaries, in about 25 minutes. This makes no novelty claim.

### Madras, Creager, Pitassi and Zemel, "Learning Adversarially Fair and Transferable Representations", ICML 2018, PMLR 80

**Method.**
- An encoder f, a classifier g, an adversary h and an optional decoder k are trained by min over (f, g, k) and max over h of αL_C + βL_Dec + γL_Adv.
- Training alternates single gradient steps: the encoder/classifier/decoder group descends, then the adversary ascends.
- For demographic parity the adversary's objective is a group-normalised ℓ1 loss: L_Adv^DP(h) = 1 − Σ_i |D_i|⁻¹ Σ_{(x,a) ∈ D_i} |h(f(x,a)) − a|. Equalised odds and equal opportunity use group × label normalisation.

**Theory.**
- For *binary* g and h, the optimal adversary's objective upper-bounds the demographic-parity distance (and, with the label as input, the equalised-odds distance) of any classifier g learnable from Z: L_Adv(h\*) ≥ Δ(g).
- This needs an optimal adversary over a class that contains the relevant test, and it bounds group-fairness gaps, not attribute recoverability in general.

**Bearing here.**
- Our critics are bounded CE adversaries, and the recovery surrogate is a fitted CE gap, not that objective. None of these bounds transfers.
- LAFTR's scalar γ is exactly the kind of fixed coefficient whose *effective* gradient strength varies with the adversary's scale. That is the confound §7 targets.
- LAFTR does not normalise gradient norms.

### Agarwal, Beygelzimer, Dudík, Langford and Wallach, "A Reductions Approach to Fair Classification", ICML 2018, PMLR 80

**Method.**
- Fairness constraints (demographic parity, equalised odds) are written as linear constraints on conditional moments, Mμ(h) ≤ ĉ.
- The method solves min over Q in Δ and max over λ ≥ 0 with ‖λ‖₁ ≤ B of L(Q, λ) = err̂(Q) + λᵀ(Mμ̂(Q) − ĉ), over *randomized* classifiers Q.
- In Algorithm 1 the λ-player runs exponentiated gradient: θ_{t+1} = θ_t + η(Mμ̂(h_t) − ĉ) and λ_k = B exp θ_k / (1 + Σ exp θ). The Q-player best-responds through a cost-sensitive classification oracle.

**Guarantees.**
- Theorem 1: the *average* play is a ν-approximate saddle point after O(ρ²B² log(|K| + 1)/ν²) iterations.
- An error analysis then bounds the randomized classifier's error and constraint violation.
- The guarantees rely on L being linear in the distribution Q (convexity through randomisation), on oracle best responses, and on the output being a mixture of iterates.

**Bearing here.**
- Our weights resemble a multiplier responding to a measured violation, and P1 resembles an exponentiated-gradient update in form.
- But the study has none of the following:
  - a randomized classifier;
  - averaging of iterates;
  - a best-response oracle;
  - convexity;
  - a moment constraint.

  It deploys the last iterate of a non-convex encoder.
- None of the paper's guarantees apply.

### Cotter, Jiang and Sridharan, "Two-Player Games for Efficient Non-Convex Constrained Optimization", ALT 2019, PMLR 98

**Method.**
- The proxy-Lagrangian is a *non-zero-sum* game: L_θ = λ₁g₀ + Σ λ_{i+1} g̃_i uses differentiable proxy constraints g̃_i ≥ g_i, while L_λ = Σ λ_{i+1} g_i uses the *original* (possibly non-differentiable) constraints, with λ in the (m + 1)-simplex.
- Algorithm 2: θ takes projected SGD, minimising external regret. λ is the stationary distribution of a left-stochastic matrix M, updated multiplicatively (M ⊙ exp(η Δ_λ λᵀ)) and column-normalised, minimising *swap* regret.

**Guarantees.**
- Theorem 8 / Lemma 9: in *expectation over a randomised iterate* θ̄ (weighted by λ₁), the result is nearly optimal and nearly feasible with respect to the original constraints, at rate O(1/√T).
- This requires convex objective and proxy constraints for the SGD version, or an approximate optimisation oracle in the non-convex case.
- A shrinking LP gives a distribution over at most m + 1 iterates.

**Bearing here.**
- The idea of "multipliers respond to the actual metric while the model descends a differentiable proxy" is Cotter et al.'s. Our split mirrors it: the controller reads calibration AUC (the original constraint), and the encoder descends the CE recovery surrogate (the proxy). It must be cited as such.
- **The hand-built controller is not the proxy-Lagrangian algorithm and inherits none of its guarantees.** It has:
  - no swap-regret λ-player and no stationary-distribution step;
  - no simplex weights, and no λ₁ on the objective;
  - a hand-set gain and box bounds;
  - 20 updates rather than T → ∞;
  - a non-convex encoder with no optimisation oracle;
  - a deployed last iterate rather than a randomised mixture;
  - a measured constraint that is itself a refitted, nonstationary probe on in-sample rows;
  - an RMS budget normalisation that has no counterpart in that game.

**Norm-controlled updates.**
- The rule q_i = ρ s_i ‖t_i‖ p_i/‖p_i‖ fixes a relative step size. It has **no optimality, convergence or fairness theorem here.** It removes one scale confound by construction and nothing more.

### Related gradient-balancing work (from memory; not re-read, confidence marked)

- **High confidence.**
  - *GradNorm* (Chen, Badrinarayanan, Lee and Rabinovich, ICML 2018) learns multi-task loss weights so that per-task gradient norms at a shared layer track their relative training rates. It is adaptive weighting through gradient norms, aimed at balancing task training rates, not at matching a declared ratio.
  - *MGDA* as multi-task learning (Sener and Koltun, NeurIPS 2018) takes a minimum-norm convex combination of task gradients, converging toward Pareto-stationary points.
  - *PCGrad* (Yu et al., NeurIPS 2020) projects away conflicting gradient components.
  - *Gradient reversal* (Ganin and Lempitsky, ICML 2015; Ganin et al., JMLR 2016) scales the adversary's gradient by a scheduled λ.
  - *Adversarial debiasing* (Zhang, Lemoine and Mitchell, AIES 2018) modifies the predictor update by removing its projection onto the adversary's gradient and subtracting α times the adversary's gradient.
- **Medium-to-high confidence.** VQGAN (Esser, Rombach and Ommer, CVPR 2021) uses an adaptive weight equal to the norm of the reconstruction-loss gradient divided by the norm of the adversarial-loss gradient (plus a small constant) at the generator's last layer. That is the closest analogue I know to per-step norm matching of an adversarial term against a task term, with a single shared layer rather than per-encoder.
- **Medium confidence.** Uncertainty weighting (Kendall, Gal and Cipolla, CVPR 2018) is another multi-task weighting scheme, by learned homoscedastic noise. It is less related.

Normalising an auxiliary gradient against a task gradient is therefore established practice in several literatures. This study's rule is a per-encoder, declared-ratio variant used as an experimental control, and no novelty is claimed for it or for the controller.
