# Mathematical review: ordinary penalty without erasure (PN, LN), before the lock

**Role:** read-only mathematical reviewer, 2026-10-03. **Scope:** `PROTOCOL.md`, `METHOD_DELTA.md`, `pnx/{train,run,select,family,infer,outer,critic_gap}.py`, against the predecessor's `jcv/{train,select,finalize}.py`, its theory and diagnostics notes and its `review/THEORY_REVIEW.md`.

**Checks run** (scratch scripts, not committed; nothing in the repository or the unit cache was written): re-ran the predecessor's already-opened U (seed 0) and JP (β ∈ {0.1, 1}, seed 0) through `pnx.train.train_arm`, with **no PN or LN fit**; simulated `pnx.select.select_seed` and `pnx.infer.Ctx.lab` on synthetic inner records; recomputed z and `decide()`.

**Verdict:** the β = 0 parity logic, the selection order and the family arithmetic are sound. Fix before the lock:
- the critic-gap diagnostic as coded measures a whitening artefact, not critic weakness, and PROTOCOL §6 does not match its code;
- three selection edge cases;
- two overclaims in the method text.

## REQUIRED (each is a concrete change; all are cheap and pre-fit)

**R1. Critic-gap: evaluate at the state the saved critics and whitener belong to** (`pnx/train.py`, `pnx/critic_gap.py`).
- **The lag.** In every step, the whitener is refitted and the critics are updated at θ_{t−1}, and only then is the encoder/head update applied. The saved `critics_final.pt` (critics + whitener) therefore belongs to θ_{T−1}, while `critic_gap.diagnose` builds views from the final θ_T. Its docstring claim "exactly what the online critics saw" is false.
- **Why it is not negligible.** The ZCA whitener floors exactly-null and near-null view directions at 1e-8·λmax, i.e. it amplifies them by about 10⁴/√λmax. The null directions are row-centring, logits affine in r, and (JP only) the LEACE direction. Any change of encoder or *training head* between the whitener fit and the evaluation leaks into these directions.
- **Measured on predecessor JP β = 1, seed 0** (online critics; attacker_val CE in nats; prior 0.629):

  | Evaluation state | v1 [A, B] | v2 [A, B] | pair [A, B] |
  |---|---|---|---|
  | θ_{T−1} + saved whitener (correct) | 0.624, 0.621 | 0.617, 0.616 | 0.618, 0.614 |
  | θ_T + saved whitener (as coded) | 1.244, 0.704 | **73.9**, 1.407 | 15.6, 8.63 |
  | θ_T + whitener refitted at θ_T | 0.624, 0.620 | 0.618, 0.617 | 0.619, 0.616 |

  Floored directions at θ_{T−1}: 3 (v1) and 7 (v2). PN will have 2 / 6 / 8 (v1 / v2 / pair), so it is equally exposed.
- **Effect.** As coded, "fresh beats online" (prediction 6) would be confirmed by construction. The fresh critics would also be trained on inputs with ~10²-scale spurious coordinates.
- **Fix.**
  1. In `train_arm`, after the critic updates of each step, keep `pre = {k: v.detach().clone() for k, v in model.state_dict().items()}` (it does not touch the trajectory). Return it as `final_critics["model_state_at_last_critic_update"]`.
  2. In `critic_gap.diagnose`, build the training-time views from that state.
  3. Add one sensitivity row: θ_T + whitener refitted on `guard_idx` at θ_T.
  4. Never apply the saved whitener to θ_T or deployed views (the deployed-view slate already does not).
  5. Re-run the fixture parity tests: the β = 0 bitwise result must still hold.

**R2. Align PROTOCOL §6 and prediction 6 with the code, and define the statistic before the lock.**
- **Text.** §6 describes one fresh variant trained on attacker_fit and a slate on whitened views. The code has:
  - `fresh_def`: same rows as the online critics (defense_train), so it isolates the optimiser;
  - `fresh_att`: attacker_fit rows, so `fresh_att − fresh_def` measures the encoder overfitting its obfuscation to training rows;
  - a slate on the unwhitened view with its own preprocessing.
- **Statistic to register.** Prediction 6 is the mean over PN/LN units × views × critic kinds of the per-kind paired difference CE(online) − CE(fresh_def), on attacker_val at θ_{T−1} with the saved whitener, and the claim is that this mean is > 0.
  - It must be per kind, because "best of bank" and the slate both select on attacker_val, the evaluation rows. That makes them optimistic, by more for the larger slate, so they are secondary.

**R3. METHOD_DELTA, the PN vs JP row: "gradients … are identical" is false.**
- `arm_spec("PN")` differs from `arm_spec("JP")` in two fields:
  - `erasure` (False vs True);
  - `guards` ([] vs [0, 1]), which is inert, since guards are read only in `mode == "projected"` branches.
- The warm start, `perm`/`crng`, critic seeds (`_seed("critic", seed, v, k)`), lr, clip and head-refit procedure are identical.
- The 21 `fit_maps` calls (20 epoch starts plus 1 end-of-stage refit) do more than replace the map. `maps` is consumed by:
  - `task_losses`: the task gradient is pulled back through `I − P_l P_rᵀ`, and the head is trained on erased inputs;
  - `views` for the whitening reference `Vref`, which gains one exact null direction per encoder;
  - the critic-update inputs;
  - the penalty views and gradient;
  - epoch-end guard logging (in penalty mode, not acceptance).

  Each refit also re-bases the head discontinuously.
- **Finalisation.** JP refits official LEACE (float64) on defense_train, releases `LEACE(g)` and refits the heads on it. PN releases `g` and refits the heads on it, with `assert not maps`.
- **Replacement text.** "PN vs JP contrasts the whole LEACE package (in-loop maps, which change task and penalty gradients, critic inputs and whitening, plus the final map before the head refit) against none, at matched warm start, data order, critic initialisation, β, schedule and head procedure. Trajectories diverge from the first step; in-loop and final effects are not separated."

**R4. METHOD_DELTA's "PN vs LN isolates coalition coupling", and PROTOCOL §7's implication that nonzero nominees isolate the coupling component, overclaim.**
- At equal β, PN and LN differ in four ways:
  - the number of surrogate terms (three vs two);
  - local bank size (two vs three critics per local view);
  - the presence of a pair critic;
  - and, because the global clip binds, the task share of each step. Measured on JP seed 0: median ‖βp‖ is 1.2 at β = 0.1 and 5.3 at β = 1, against ‖t‖ of 1.1 and 1.8; the clip binds on about 3% and about 70% of steps; U never clips.
- The training pair bank has no ignore-other-view critics. `R_pair` can then act as extra local pressure (predecessor review §3, §5).
- The local clauses are one-sided, so PN may have *lower* local recovery than LN, and P01 can pass with zero synergy change, since Δcoalition = Δ(max local) + Δ(synergy).
- **Replacement text.** "Claim A tests whether adding the coalition term, with β tuned on inner roles, lowers coalition recovery by ≥ 0.02 relative to the selected local-penalty control without worse local recovery. It does not attribute the difference to coupling alone: PN also carries more penalty mass, a different critic bank and a different task share under clipping. The decomposition Δcoalition = Δ(max local) + Δ(synergy) is reported descriptively."

**R5. The "U fails G3 ⇒ no valid reference" rule (PROTOCOL §4) is not implemented.**
- In simulation with Acc(U) − const = 0.02 on income, a more accurate LN at β = 0.1 became NOMINEE, PN became NOMINEE and C\* = LN, so claim A could pass.
- **Fix.** In `select_seed`, if `out["arms"]["U"]["gates_ok"]` is False, set `out["valid_reference"] = False`, mark every arm `NO_VALID_REFERENCE` and set the comparator to None. In `infer.main`, require `valid_reference` on every seed for claims A and B.

**R6. The descriptive closest-utility fallback for PN (and LN) returns U.**
- β = 0 ≡ U is in the candidate list, and U's worst-gate margin is the G1 maximum (0.01). So `_closest` almost always picks `nn__s{k}__U`.
- In simulation, PN failed only the allowance, and "PN (INFEASIBLE)" was scored as U in every PN row (`lab` → "U").
- **Fix.** Take `_closest` over the nonzero β only for PN and LN, or else label the fallback `TASK_ONLY_ALIAS`.

**R7. The comparator does not carry alias status.**
- If LN is `TASK_ONLY_ALIAS` and wins C\* (the LN-before-U tie order on identical U values), `comparator.arm = "LN"` while the release is U.
- **Fix.** Add `comparator.status = out["arms"][arm]["status"]` and `comparator.unit`. Report it as "LN (TASK_ONLY_ALIAS ≡ U)".

## 1. β = 0 parity

- **The code path is the same.** `diff` of the two `train_arm` bodies shows only the docstring and the extra return value.
- **β = 0 skips the penalty.** `beta > 0` gates the penalty, so `d_enc = zeros(ne)` in both U and PN/LN, and `u = cat(t_enc + d_enc, t_head)` is literally the same expression. The clip and `assign_add` are the same.
- **No erasure.** Both specs have `erasure: False`, and `maps` stays `[None, None]`.
- **Critics never reach the encoder.** Critic views are built under `no_grad`; the critic `backward`/Adam touch only critic parameters; the encoder step uses `autograd.grad`, so stray `.grad` buffers would be ignored anyway.
- **RNG.** `perm` comes from `default_rng([seed, si, ep])`; `crng` is a separate Generator. `torch.manual_seed` at critic creation reseeds the global torch RNG, but nothing on the encoder path draws from it (no dropout or sampling; the `Model` initialisation is overwritten by the warm state).
- **Floating point.** One thread, deterministic CPU kernels and 64-byte-aligned PyTorch allocations, so extra critic allocations cannot change kernel choice.
- **Conclusion.** Bitwise equality is a sound expectation. In this environment, U seed 0 and JP β = 1 seed 0 both reproduced the predecessor's model state bitwise.
- **Tolerance.** It is appropriate. With a bitwise model and a deterministic finaliser the release should also be bitwise, and the 1e-12 slack is harmless.
- **Advisory A1.**
  - Record `release_bitwise`.
  - Add "U re-trained here == predecessor U", so that a failure separates environment drift from critic interference.
  - Pre-register what a failed parity check triggers (diagnose only; no tolerance change).

## 2. Erasure contrast

See R3 for the full list of differences.

**Advisory A3.** One cheap descriptive arm separates the two effects: PN→E, i.e. PN's encoders, then the final official LEACE, then the head refit, as E is for U. Then:
- JP vs PN→E is the in-loop effect;
- PN→E vs PN is the final-map effect.

Register it before the lock, or omit it.

## 3. Selection (`pnx/select.py` vs PROTOCOL §4)

**Steps that match the protocol.** LN first over β ∈ {0, 0.1, 1, 10}, minimising (max local, mean local, β), frozen before PN; PN with the gates and the +0.01 allowance versus frozen LN, minimising (R_pair, β); JP by the same rule over {0.1, 1, 10}; E gates only; U REFERENCE; F per purpose by (round(R_local, 12), config); C\* over {LN, U, E, JP, F, F0} with the allowance, ties in that order. F0 has its own gate status and a separate `pairing_to_F`, which fixes the predecessor's mislabel (all three seeds had gates_ok True but were labelled GATES_FAILED).

**Edge cases (simulated):**

| Case | Outcome | `lab()` | A | B |
|---|---|---|---|---|
| LN = TASK_ONLY_ALIAS | LN unit `nn__s{k}__U` | "U" | NOT_ESTABLISHED (status) | Defined; C\* label needs R7 |
| PN alias | — | "U" | NOT_ESTABLISHED | NOT_ESTABLISHED; clauses defined (PN−U can be exactly 0, giving SE 0 and a valid decision) |
| PN infeasible | Fallback = U | "U" | NOT_ESTABLISHED | R6 |
| C\* = LN (nonzero) | P10–P12 ≡ P01–P03 | LN_b | as registered | Same statistics as A |
| C\* = U with LN aliased | **Unreachable**: identical values, so LN wins the tie | — | — | — |
| No feasible LN | Only if Acc(U) < const + 0.03 (β = 0 always passes G1, and G2 iff Acc(U) ≥ const) | — | — | Needs R5 |

`label_of` and `lab` map aliases correctly (`nn__s{k}__U` → U, `pn__s{k}__PN__b0.1` → PN_b0.1, `nn__s{k}__JP__b1` → JP_b1; F, F0 and U are passed through).

**Advisory A6: prediction 5 conflicts with the already-known inner values.**
- On every seed, F0 passes its own gates, sits within the allowance of an aliased LN, and has a lower inner R_pair than U (0.866 vs 0.872–0.880).
- So C\* is F0 whenever LN aliases, and U can essentially never be C\*.
- Restate the prediction as "LN (nonzero) or F0".
- C\* may also differ by seed; claim B then averages heterogeneous contrasts, so say so.

**Advisory A5.** Secondary rows using the selected JP, F or LN (S-ctrl, S-syn, S-out, S-race) silently use INFEASIBLE fallbacks. Write each seed's comparator status into the CSV.

## 4. Claims language

- **"Equal conditional means of affine scores do not imply equal thresholded decision rates": true.**
  - For binary S with both classes present, `Cov(w·r + b, 1[S=1]) = p(1−p)(E[w·r|S=1] − E[w·r|S=0])`. So LEACE's zero cross-covariance is exactly "equal SEX-conditional means of every affine score" on the fitting rows.
  - Rates `P(w·r > τ | S)` depend on the whole conditional law. Example: r|S=1 ≡ 0; r|S=0 = −1 with probability ¾ and +3 with probability ¼. The means are equal, but P(r > ½) is ¼ vs 0 (numerically 0.248 vs 0).
  - It holds even on the fitting rows. Suggested wording: "...so E/JP decisions need not satisfy demographic parity, even on the fitting rows."
- **Guarantees lost.** On the defense_train rows at the final refit, E/JP have:
  - (a) Cov(r_i, S) = 0;
  - (b) by affine closure, the same for the view [r_i, centred logits];
  - (c) by stacking, the same for the coalition [v1, v2];
  - (d) hence no affine predictor under a convex loss beats the constant there (Belrose et al., Thm 3.4).

  PN/LN lose (a)–(d) **and nothing else**: no held-out, population, nonlinear, probability or decision guarantee existed. The METHOD_DELTA bullets are correct; add "(c) the coalition" explicitly.
- **Fit-row cross-covariance as a diagnostic (Advisory A2).**
  - For PN/LN, "fitting rows" are just the encoder's training rows.
  - `fit_crosscov` and `held_out_linear` report the max per-coordinate |corr|. That is not rotation-invariant and can understate linear leakage without bound. Example: x1 = z + 0.1s, x2 = z gives max |corr| ≈ 0.05, yet the OLS R² of s on (x1, x2) is 1.
  - Add the multiple-correlation R² (OLS, which is LEACE's own target) for r_1, r_2 and [r_1, r_2], on fit and assessment rows, with null references:
    - max |corr| over 16 coordinates ≈ 0.015 at n ≈ 19.4k;
    - R² ≈ d/n.
  - Replace "exactly about 0" with "0 up to rounding (≈ 1e-15 relative)".
- **Penalty mass.** See R4 for the replacement text. The β-matched rows (S-bm) should be described as "PN vs LN at equal nominal β (different total penalty mass)".

## 5. Critic-gap design: what must be held fixed (minimal correct design)

**Fixed:** the encoder and training-head state θ_{T−1} (at the final critic update); the saved whitener; the identity map; the evaluation rows (attacker_val); the loss (CE in nats, defense_train prior as the common baseline); the architecture per kind; and no model selection on attacker_val for the primary statistic.

**Varied:** the critic parameters' provenance, online vs fresh_def (primary); and the training rows, fresh_def vs fresh_att.

**Separate family.** The inner slate on the unwhitened θ_{T−1} view, labelled "own preprocessing; selected on attacker_val". Also the deployed-view slate.

**Sensitivity.** θ_T with a whitener refitted at θ_T.

**Pitfalls to name in the protocol text.**
1. The whitener is fitted on the fitting-role `guard_idx` rows. That is harmless as a fixed affine map, provided the state matches (R1).
2. Online critics saw a moving encoder on defense_train rows; `fresh_att` adds a row shift, hence `fresh_def`.
3. Training-time logits come from the training head, not the refitted deployed head. The two views are affinely equivalent given r (same information), but the saved whitener must never touch deployed logits, because their null space differs.

**Advisory A4.** Record per-step ‖t‖, ‖βp‖ and whether the clip was hit in `diag`. Also record the share of the penalty-gradient energy that lies along the bottom whitener eigendirections. Measured on JP:
- the penalty gradient is about 3× the task gradient at β = 1 (median 5.3 vs 1.8);
- the surrogate is only about 0.02–0.05;
- the online critics recover once the whitener is refitted (table in R1).

That pattern is consistent with the penalty spending its step on whitener-amplified low-variance directions. The diagnostic is exploratory and authorises no grid.

## 6. Family arithmetic

- z₁₈ = Φ⁻¹(1 − 0.05/36) = 2.991316115 and z₃₀ = Φ⁻¹(1 − 0.05/60) = 3.143980287, matching `family.py`.
- Two-sided Bonferroni per slot is valid (conservative) for the two-sided rows and doubly conservative for one-sided rows.
- The six aliases P13–P18 make the primary family conservative. Each claim is an intersection–union test, so the conjunction does not need further correction.
- **Erasure endpoints.** Each pools 3 seeds × 3 β = 9 paired differences on the same assessment rows. The `mean` op weights them equally, which equals the mean over seeds of per-seed β-means. The bootstrap resamples record groups jointly, so cross-β and cross-seed correlation enters the SE.
- **`decide()`, checked on a grid:** lower> passes iff lo > target; upper< iff hi < target; two_sided gives ABOVE iff lo > target, BELOW iff hi < target, else NOT_RESOLVED. NaN bounds never pass. Nonfinite replicates are dropped (`n_finite_replicates` is recorded); say so in the text.
