# Protocol: matched update strength and active local feedback

**Status.** This is exploratory method development on already-exposed Adult rows.
- NEW_DEVELOPMENT_ASSESSMENT is withheld from this procedure until EVALUATION_LOCK.json. It is not fresh confirmation (`EXPOSURE_LEDGER.md`).
- Earlier verdicts stay closed. Neither the old 5,243-row assessment nor the refreshed study's 3,397-row assessment is read.

**Source.** research/pcrl-refreshed-guarded-joint-v1 @ `ccdcc5a`.

**Staged locks.** Each is pushed before the stages it governs:

| Lock | Pushed before |
|---|---|
| DATA_AND_ENGINEERING_LOCK | warm starts, parity, U, raw controls |
| PHASE_A_PROTOCOL_LOCK | Phase A fits |
| PHASE_A_FREEZE (`PHASE_A_SELECTION.json`) | Phase B |
| PHASE_B_PROTOCOL_LOCK | the full Phase B queue, after the controller preflight |
| EVALUATION_LOCK | the single assessment |

Dated amendments (`AMENDMENT_A<n>.json`) are the only way to change locked code.

## 1. Data and roles (`smf/data.py`, `ROLE_MANIFEST.json`)

| Role | Source | Rows | Use |
|---|---|---|---|
| NEW_DEFENSE_FIT | 80% of the refreshed DEFENSE_FIT groups (hash seed 20261005, salt `assess`) | 15,434 | Warm starts, encoders, training heads, critics, controller probes, LEACE maps, FARE trees, numeric scaling |
| ⤷ CRITIC_FIT / CRITIC_VAL / CONTROLLER_CALIB | 70/15/15 of NEW_DEFENSE_FIT groups (salt `critic`) | 10,764 / 2,343 / 2,327 | Critic and probe fitting / selection / controller AUC |
| NEW_DEVELOPMENT_ASSESSMENT | 20% of those groups | 3,796 | Labels sealed (−1) at load until EVALUATION_LOCK |
| HEAD_VALIDATION | preserved | 1,500 | Task-head C selection only |
| AUDIT_FIT / INNER_SELECTION | preserved | 6,065 / 2,235 | Attacker fitting / selection, inner audit, nomination |

**Excluded everywhere:** the old assessment, cert and exclusions, and the refreshed DEVELOPMENT_ASSESSMENT.

**Encoder training rows.** Encoders and training heads train on all NEW_DEFENSE_FIT rows, including the critic and controller subroles' labels. The subroles are held out only from critic and probe fitting.

**Preprocessing.** The 5 numeric columns are re-standardised on NEW_DEFENSE_FIT only. Raw integers are recovered exactly from the admitted normalisation; the one-hot vocabularies are fixed.

**Contract.** The 83-column contract and the Husband/Wife proxies are kept. SEX is primary; race is a secondary, supported-class audit.

**Fresh fits only.** Warm starts, U, the raw controls, LEACE and FARE are all refit from scratch on NEW_DEFENSE_FIT. Old checkpoints contain the new assessment rows and are never reused as controls.

## 2. Model, views and the update rule (`smf/train.py`, `METHOD_CARD.md`)

**Architecture.**
- Encoders 83-64-64-16 (ReLU) with affine heads; warm start Adam 1e-3 for 20 epochs; SGD 0.05; batch 256; global-norm clip 5.
- Critic views are v_i = [r_i, centred(W_w r_i + b_w)], using the fixed warm-start head; pair = [v1, v2]. There is no moving-head view.
- Releases are [r_i, centred logits of a head refit on r_i only, with C chosen on HEAD_VALIDATION].
- No mandatory LEACE. Official LEACE is a baseline.

**Surrogate.** R = (CE_const − min(CE_const, CE_A, CE_B)) / H on the same rows. It is a fitted surrogate, not mutual information.

**Norm-controlled update.** Per encoder i:
- t_i = ∂L_task/∂enc_i;
- p_i = ∂P/∂enc_i;
- a_i = stop_grad(ρ s_i ‖t_i‖/‖p_i‖), computed in float64 and capped at 100;
- q_i = a_i p_i, set to 0 with the event logged if ‖p_i‖ ≤ 1e-12 or ‖t_i‖ = 0;
- update: −lr(t_i + q_i) on encoders and −lr·(task gradient) on heads, then the global clip.

The coefficients of P are normalised to sum 1:
- joint (w1, w2, 1) on (R1, R2, R_pair);
- local (w1, w2) on (R1, R2), with the pair bank a trained shadow carrying zero encoder gradient.

**Strength grid.** ρ ∈ {0.25, 0.75, 1.5}. These are privacy/task encoder-gradient norm ratios, not budgets.
- Phase A: s_i = 1.
- Phase B: s_i = w_i / sqrt((w1² + w2²)/2), so the RMS of the two relative strengths equals ρ before caps and clipping.

**Logging.** Pre-clip ratios, cosines, cap and zero events, clip hits and post-clip update norms are archived (`GRADIENT_MATCHING.csv`). Norm matching does not match directions, trajectories or task interference.

**Raw controls (RAW-J / RAW-L).** `rgj.train.train_run` "J-O" / "L-O", called unchanged:
- base weights β/3 ×3 (joint) or β/2 ×2 (local);
- online critics;
- β ∈ {0.1, 0.3}, 40 epochs, archived at epochs 20 and 40.

**U.** Task-only for 40 epochs after the warm start, archived at epochs 20 and 40.

## 3. Critic schedules

- **ONLINE.** Before every encoder step, a floored ZCA is recomputed on a fixed 4,096-row CRITIC_FIT subset; 5 critic Adam steps follow.
- **REFRESHED.** Bounded refits at epochs 0, 4, 8, 12 and 16, plus a diagnostic refit at 20. Each refit chooses between a continued (transported) critic and a deterministic restart on CRITIC_VAL (at most 15 epochs, patience 3). There are 5 online steps between refits, with the transform held within each block.
- **ONLINE_MATCHED.** ONLINE plus extra Adam updates per bank. Their number equals the paired REFRESHED unit's refit receipt for epochs 0–16 (continued + restart attempts, including discarded restarts; the epoch-20 diagnostic refit excluded).
  - Spreading: at step s, the bank receives ⌊E·s/S⌋ − ⌊E·(s−1)/S⌋ extra updates, on its own RNG stream.
  - Dependency: the paired REFRESHED unit runs first.
  - What is matched: critic optimizer-update counts.
  - What still differs: validation-based restarts, transform schedule, initialisation, Adam state.

## 4. Phase A (`smf/select.py`, frozen in PHASE_A_PROTOCOL_LOCK)

**Fits.** On seeds 0, 1 and 2, with paired warm starts:
- U (40 epochs);
- RAW-J and RAW-L, β ∈ {0.1, 0.3} (40 epochs);
- normalized joint (NJ) and local (NL) × 3 schedules × 3 ρ (20 epochs).

Only epoch 20 of the normalized arms is eligible.

**Gates.** Gates are applied on INNER_SELECTION against U epoch 20:
- G1: Acc ≥ Acc(U) − 0.01;
- G2: retention ≥ 80%;
- G3: gain over the constant ≥ 0.03.

**Schedule rule.**
1. For each schedule and seed, take the task-feasible NJ point with the lowest inner coalition AUC (ties to lower ρ).
2. Among schedules with a point on every seed, take the lowest mean coalition AUC. Ties go to lower actual critic-update compute, then ONLINE, ONLINE_MATCHED, REFRESHED.
3. Fallback, if no schedule is complete: minimise the sum over seeds of the smallest joint-point gate shortfall, where a point's shortfall is Σ max(0, −margin) in accuracy units. Status FALLBACK.

**Local reference per seed** (selected schedule): the task-feasible NL point with the lowest worse-local AUC, then mean local AUC, then lower ρ.
- U epoch 20 is a disclosed fallback (TASK_ONLY_ALIAS).
- If U lacks utility, the status is NO_VALID_REFERENCE.

`PHASE_A_SELECTION.json` (= PHASE_A_FREEZE) is pushed before Phase B.

## 5. Active local feedback (Phase B)

**Controller probes.** At each measurement the probes run on frozen local views:
- a scale-aware whitened LR on the r block only (float64 SVD; directions below 1e-9 × s_max are rank-null; review R2 — the centred logits are affine in r, so the linear function class is unchanged, and a rotated 1e-6 clue stays readable);
- an MLP(64).

Both are fitted on CRITIC_FIT. One is selected, and its orientation chosen, on CRITIC_VAL; its AUC is then measured on CONTROLLER_CALIB. The probes never touch AUDIT_FIT or the assessment.

**Targets and updates.**
- Target: b_i = max(0.5, AUC_i(frozen reference) − 0.01).
- Initial weights: w1 = w2 = 1, with the pair weight fixed at 1.
- Updates happen at epoch 0, using the reference receipt itself (an intended +0.01 violation), and after epochs 1–19. The epoch-20 measurement is diagnostic only.
- Rule: v_i = AUC_i − b_i, and w_i ← clip(w_i + clip(v_i/0.01, −1, 1), 0.25, 8).

**What the weights do.**
- Local feedback: p_i ∝ w_i ∇R_i, so the weights act only through the allocation s_i. A common multiplier is an exact symmetry.
- Joint feedback: the weights also change the balance between the local and pair directions.

**Twins.** No-feedback twins (J-N, L-N) run the same probes and record the hypothetical weights without applying them.

**Preflight** (before PHASE_B_PROTOCOL_LOCK). Synthetic binding and asymmetric fixtures, plus a real fitting-only receipt on each frozen reference:
- the initial violation;
- the weight change;
- the allocation and update change;
- that untouched recipients are unaffected.

A failed preflight allows at most two dated, reviewed repairs (update frequency, target offset or gain) before Phase B. Never tune from the assessment.

**Activation statuses.** Phase B stores the effective allocation and direction changes per unit.
- INACTIVE_OR_ALIAS: a feedback arm that only ever experienced common-mode violations.
- FEEDBACK_NOT_TESTABLE: no feedback arm was ever effective.

## 6. Phase B arms and nomination

**Arms.** From the frozen reference on each seed, with the selected schedule, ρ ∈ {0.25, 0.75, 1.5}, for 20 more epochs (final epoch only is eligible):
- J-F (joint + feedback; the candidate);
- L-F (local + feedback; the coupling control);
- J-N and L-N (no-feedback twins).

If the schedule is ONLINE_MATCHED, a REFRESHED template on the same initialisation, ρ and rule runs first. Templates are never candidates.

**References.** U epoch 40, RAW-J and RAW-L at epoch 40, official LEACE on U e40, and official FARE plus a same-budget zero-fairness tree, all refit on NEW_DEFENSE_FIT and reselected on inner roles.

**Controls and C\*.** Each control is selected at its task-feasible point with the lowest coalition AUC (ties to lower ρ/β). C\* is the task-feasible control with the lowest coalition AUC among L-F, J-N, L-N, RAW-J, RAW-L, E, F, F0 and U. There is no local guard, and task-only C\* is disclosed.

**J-F nomination.** A point is eligible if it is task-feasible and AUC_i(J-F) ≤ AUC_i(L-F) + 0.005 and AUC_i(J-F) ≤ AUC_i(C\*) + 0.005 for i = 1, 2, against the available valid comparators. Among eligible points, take the lowest coalition AUC, then lower ρ.
- No eligible point means NO_FEASIBLE_NOMINEE.
- The descriptive fallback minimises the summed positive shortfalls of all nomination gates, then coalition AUC, then lower ρ.

## 7. Assessment, endpoints and inference

**Attackers.** Final attackers (LR, MLPs, HGB and a defense-aware scaled reader with an eps-justified float64 rank tolerance) are fitted on AUDIT_FIT. Selection uses INNER_SELECTION, with an AUC attacker (primary) and a CE attacker (proper loss) chosen separately. Coalition banks include the ignore-recipient candidates.

**Controls.** A rotated 1e-6 clue must be detected through serialisation, planted leaks must be detected, and selection-aware nulls are evaluated on a separate split (`smf/audit.py`).

**Primary family (18 slots, z = 2.991316).**
- Claim A compares J-F with L-F; claim B compares J-F with C\*.
- Each claim has nine clauses:
  - coalition improvement: lower bound > 0.02;
  - local increase on each recipient: upper bound < 0.01;
  - accuracy versus U e40 on each task: lower bound > −0.01;
  - retention on each task: lower bound > 0;
  - gain over the constant on each task: lower bound > 0.03.
- Slots P13–P18 alias P04–P09.
- Both claims need every clause and valid nominees on every seed; claim A also needs nontrivial J-F and L-F.

**Secondary family.** Enumerated in `PRIMARY_FAMILY.json` at the Phase B lock.

**Bootstrap.** Paired exact-record-group bootstrap, B = 1999, seed 20261005, with the same groups for all arms and seeds.

Only frozen selected arms, the registered fixed-ρ component points and the official controls are scored.

## 8. Resources and repairs

**Ceilings.** 10 h (from 00:42Z; reserve from 08:42Z), 20 CPU-h, 2 heavy workers, < 8 GiB, ≥ 5 GiB free, $0. A watchdog stops only this study's processes.

**Engineering timing** (seed 0, real fitting rows, 4 epochs):

| Schedule | Time | Estimate for 20 epochs |
|---|---|---|
| ONLINE | 2.2 s | about 11 s |
| REFRESHED | 4.7 s | about 25 s, including refits |
| ONLINE_MATCHED | 3.8 s | — |

A probe receipt takes 2.3 s, so a Phase B run takes about 70 s.

**Projected core** (the full grid, which needs no reduction):
- Phase A: 54 normalized runs plus 12 raw runs plus U, about 0.5 CPU-h.
- Phase B: 36 runs, plus 36 templates if ONLINE_MATCHED is selected, about 1.5 CPU-h.
- Audits and verification: about 1.5 CPU-h.

**Retries.** One half-learning-rate retry for nonfinite training, applied to every eligible arm. Poor results are never a retry trigger.

## 9. Registered predictions (before any nonzero fit)

1. Under norm matching, the Phase A schedules differ little: the mean selected NJ coalition AUCs are within 0.01 of each other. So the earlier J-O advantage was mainly strength.
2. ρ = 1.5 fails a task gate (occupation G1) on at least one seed, and the selected joint ρ is mostly 0.75.
3. The controller activates. On most feedback runs some weight moves by more than 0.5 from 1, and at least half of the J-F runs reach an asymmetric allocation (|s1 − s2| > 0.2).
4. Claim A is NOT_ESTABLISHED: the coalition margin over L-F has a lower bound ≤ 0.02.
5. Claim B is NOT_ESTABLISHED, with C\* = RAW-J (β 0.3) or J-N on most seeds.
6. The selected J-F stays within 1 accuracy point of U e40 on both tasks.

## 10. Limits

- Development evidence on reused rows.
- Attacker-based evidence only, with intervals conditional on fitted models.
- No population, DP, MI, optimality or linear-guardedness claim, and no novelty claim.
- The controller is a heuristic; it is not Cotter et al.'s proxy-Lagrangian and has none of its guarantees.
- Norm matching is not direction matching.

## 11. Phase B registration (written before any Phase B fit; PHASE_B_PROTOCOL_LOCK)

**Phase A freeze.** Pushed at `15a9db9`: schedule = REFRESHED (status SELECTED; every schedule had task-feasible joint points on all seeds).

| Schedule | Mean selected inner coalition AUC |
|---|---|
| ONLINE | 0.8586 |
| REFRESHED | 0.8573 |
| ONLINE_MATCHED | 0.8652 |

Frozen local references:
- seed 0: NL-REFRESHED ρ = 0.75;
- seed 1: NL-REFRESHED ρ = 0.75;
- seed 2: NL-REFRESHED ρ = 1.5.

**Controller preflight** (`PREFLIGHT.json`, under amendment A3, which fixed the preflight's own crash on a zero-direction step).

Controller behaviour:
- **Initial violation.** +0.01 on every seed and recipient; no floor is active.
- **Epoch-0 update.** It moves w to (2, 2) on all seeds (common mode).
- **Selectivity.** Untouched recipients are unaffected.
- **Local arms.** Equal weights are an exact symmetry (identical updates).
- **Joint arms.** Equal weights still change the encoder direction (cos 0.987–0.995).
- **Asymmetric weights.** w = (2, 1) reallocates the relative strengths to 0.95 / 0.47 (RMS 0.75).
- **2-epoch real runs.** The applied weights became asymmetric in 5 of 6 feedback runs, for example (1, 3) and (3, 1), and the parameters differ from the twins.
- **Common-mode alias.** One local run stayed common-mode (3, 3); its parameters equal the twin's exactly, which is the designed INACTIVE_OR_ALIAS case.

Zero-direction finding:
- The single failed check ("asymmetric weights reallocate", seed 2) is not a controller defect.
- On the fixed minibatch, recipient 1's reference critics lost to the constant predictor, so p_1 = 0 (a zero-direction event).
- This is a real data finding: under strong norm-controlled updates the refreshed local reference at ρ = 1.5 pushed its critics below the prior, and protection on that recipient stopped while information remained (v1 inner AUC 0.852). Zero-direction counts are reported per unit in `GRADIENT_MATCHING.csv`.

**Decision.** The preflight did not show dead feedback, so no repair is authorised. The registered additive rule is kept.

The §5 probe description was corrected to the reviewed reader (r block, 1e-9), and the Phase B lock was rewritten before any Phase B fit. The first Phase B lock write (`a1186eb`) carried the stale text.

The reviewer's proposed multiplicative step (P1: w ← clip(w · 2^c, 0.25, 8)) addresses wind-up after common-mode saturation. It is recorded as an advisory and **not adopted**. Saturation and asymmetry counts are reported per unit, so any wind-up shows up in the results.

**Registered activation criterion** (review A1, `CONTROLLER_ACTIVATION.csv` / `CONTROLLER_SUMMARY.json`):

| Arm | Status | Condition |
|---|---|---|
| Local feedback | ACTIVE | some applied update makes \|s1 − s2\| > 0.05 |
| Local feedback | INACTIVE_OR_ALIAS | otherwise |
| Joint feedback | ACTIVE_ASYMMETRIC | same asymmetry condition |
| Joint feedback | ACTIVE_COMMON_ONLY | weights moved but stayed equal (only the local/pair balance changed) |
| Joint feedback | INACTIVE | weights never moved |

- No-feedback twins report their hypothetical weights.
- If no feedback arm is ACTIVE, the component is FEEDBACK_NOT_TESTABLE.

**Notes kept from review.**
- **A5:** within a joint p_i, the local/pair mix still follows each term's gradient size, so this part of the scale confound remains.
- **A7:** the cap at 100 is practically inactive.
- **A10:** ONLINE_MATCHED still differs from REFRESHED in validation-based restarts, the transform schedule, initialisation, Adam state, refit batch order and the per-block coordinate system.
- **A12:** a J-F nominee that uses the +0.005 buffer will often fail its final +0.01 clause.
- Critic Adam state is reset for every Phase B arm alike when the reference critics are inherited.

**Phase B queue.** REFRESHED, so no ONLINE_MATCHED templates are needed.
- Fits: J-F, L-F, J-N, L-N × ρ {0.25, 0.75, 1.5} × seeds 0–2 (36 runs, 21 probe measurements each).
- Then: LEACE and FARE refits, the inner audit, selection (controls, C\*, J-F nomination), tracking, EVALUATION_LOCK and the single assessment.
