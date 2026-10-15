# MATH REVIEW (pre-fit) — two-recipient task-wire adapter

Reviewer role: MATH REVIEW. Static review only; no numerical worker run.
Files: two_recipient_adapter/{adapter,guard,recipe,train_worker,test_engineering}.py,
diffed against the verified predecessor minimal_task_wire_adapter/{adapter,guard,train_worker}.py.

Verdict: **NO BLOCKERS.** The registered design is implemented faithfully in the
numerical path. The weak spots are in the tests: several "claimed" properties are
checked vacuously or not at all (SHOULD-FIX).

## BLOCKERS

None.

Items checked and found correct:

- **Arithmetic / zero witness** (adapter.py:114-123). `q = (q0 + hbar R^T) + b`, then `z = q B`, then `own_wire`.
  The head is float64 (useful_service.py:46), so the logits path is float64 after the
  float32 trunk. With R=0, b=0: `hbar @ 0 = +0` and `q0 + 0 + 0 = q0`. The only possible
  change is -0.0 becoming +0.0, which gives the same `q B`. RAW_PROJ computes
  `(float64(z) B^T) B`, so the two wires are identical (tested at test_engineering.py:25-42).
- **K layouts.** guard.pool_pass uses `adapter.K` for the logp0 shape check and for
  one_hot (guard.py:114-130). wire_contract_check slices `16:16+K` and `16+K:16+2K`
  (adapter.py:201-213). Critic widths are 21/29/50, which equals 16+2K+1 for K=2 and K=6
  (train_worker.py:92). There is no leftover `6`, `16:22` or occupation-only slice in
  train_worker, guard or adapter.
- **KL** (guard.py:132-134). It is `sum p0 (log p0 - log p_theta)`, which is KL(p0||p_theta)
  and is agnostic to K. p0 is computed once per recipient in build_context
  (train_worker.py:601-604) and never changes. Start KL is asserted to be at most 1e-12.
- **privacy_gradients** (train_worker.py:195-220). Parameters are collected for the active
  recipients only. `order = sorted(params)` and the grad list is
  `[R_o0, b_o0, R_o1, b_o1]`, read back as `grads[2k], grads[2k+1]`, so the indexing is
  consistent. The objective is the full arm objective (joint: inc+occ+pair; local: inc+occ).
  The gradient with respect to recipient i therefore picks up only its own view (plus pair
  for joint), as registered. Inactive or frozen recipients enter through no-grad wires, so
  S12/S21/I_ONLY get the joint objective restricted to the active residual. allow_unused
  with zero fill is only reachable on a truly disconnected graph. Even the
  entropy-constant branch is `view.sum()*0 + H`, which gives zeros rather than None. A
  disconnection would show up as a zero median, which leads to
  UNAVAILABLE_FIXED_NORMALIZATION, so it cannot fail silently.
- **Simultaneous macro-step** (train_worker.py:477-484).
  - Rows are drawn once and pre_vecs are captured.
  - privacy_gradients is computed once for both recipients from the shared state and the
    current critic state.
  - Both `preps` (pool metrics, guard CE gradient, task gradient, proposal) are fully built
    in a dict comprehension before any `apply_step`.
  - Each `apply_step` evaluates only its own recipient's pool (`evaluate` closes over
    `ad = adapters[i]`). Applying income then occupation is therefore equivalent to applying
    them as one macro-step.
- **Per-recipient budgets / anchors** (build_context, train_worker.py:607-610;
  guard.Anchor).
  - Each recipient has its own anchor at its own zero start, with trust=True in every arm.
  - Anchor objects are never mutated and are shared, unchanged, across arms and phases.
  - `correct_floor = start - int(0.005*10223 + 1e-9) = start - 51`, which is the registered
    floor.
  - The QP constraints are NLL, Brier and KL, plus the ball r=0.01 per recipient.
  - The ladder is 1, 1/2, 1/4, 1/8 with exact rollback, and the rollback is asserted in
    apply_step (train_worker.py:361-362).
- **Critic accounting.**
  - Simultaneous: 10 rounds per macro-step. Sequential and single: 5 rounds per step.
    Warm-up is 80 rounds.
  - Critics are created once per arm and continue across rejections and phases.
  - End-of-arm assertion (train_worker.py:546-552): `80 + rps*320*len(phases)`, which gives
    3280 / 3280 / 1680. This matches RECIPE['expected_critic_rounds'], and optimizer
    steps = 9 × rounds.
  - Opportunities: 320 per active recipient per phase.
- **Sequential freezing** (train_worker.py:462-488).
  - At each phase start the inactive residual is hashed and stored as a flat copy.
  - After every step it is compared bitwise.
  - Anchors are not reset.
  - The reference is refreshed at every epoch start in both phases.
  - Checkpoints are P1E*/P2E*. Only P2 is selectable and the fallback is P2E10.
- **Coefficients** (normalization, train_worker.py:273-310).
  - Inputs: the warm zero state (same critic seed and warm rng as every arm; sha asserted
    in run()), 16 draws, and the task gradient = w_U·minibatch-CE gradient + w_G·whole-GUARD
    CE gradient.
  - λ = 3·median task norm / median privacy norm, computed per recipient and per kind.
  - Joint coefficients are used for J_BOTH/S12/S21/I_ONLY and local coefficients for
    L_BOTH (train_worker.py:669-671).
  - Unavailability rule: a coefficient is unavailable if it is ≤1e-12, nonfinite or >1e4.
- **Counterfactual** (train_worker.py:366-417).
  - It is called with both adapters at post_vecs. post_wire is read from that state.
  - Both recipients are restored to pre_vecs bitwise, via a float64 flat copy and
    `torch.from_numpy(...copy())`.
  - Repeat noise is computed at the pre-state, against the original proposal: task gradient
    and pool recomputed, privacy gradient recomputed with the same critic state. Critics are
    not updated and batch_rng is not consumed in between, and `terms()` only toggles
    requires_grad.
  - The counterfactual proposal is `-lr*prep['g_task']` with the saved `prep['current']`
    pool metrics/gradients, the same anchor, solver and ladder.
  - The post state is then restored and checked bitwise for both recipients. A
    counterfactual for i=1 in the same macro-step starts from the restored post state,
    rebuilds the joint pre-state, and gets the correct joint-pre privacy gradient.
  - The J_BOTH trajectory is not perturbed (no critic, RNG or residual side effect).
  - Scope: first 8 accepted nonzero updates per recipient, plus up to 16 active-budget
    cases. When a case qualifies for both, it is computed once and appended to both lists.

## SHOULD-FIX

1. **test_engineering.py:303-304 — the counterfactual assertion is vacuous.**
   - What the test does: it asserts `all('privacy_minus_counterfactual_max_abs' in r for r in sc['income'] + sc['occupation'])`.
   - Failure: if J_BOTH never produces an accepted nonzero update, or a regression stops
     the scan from being called, both lists are empty and the test still passes. The
     docstring claims "J_BOTH counterfactuals for both recipients".
   - Fix: assert `len(sc['income']) > 0 and len(sc['occupation']) > 0` (with 6 opportunities
     this should hold; otherwise raise the synthetic budget). Also assert that each scan
     records `repeat_computation_proposal_max_abs_noise` ≤ ~1e-12, which proves the repeat
     really happened at the pre-state.

2. **Missing test that the counterfactual does not perturb J_BOTH** (no current test).
   - Failure: a future edit that leaves a residual, critic state or batch_rng draw changed
     after `counterfactual()` would silently alter J_BOTH's trajectory, and only J_BOTH's.
     That breaks matching with T_BOTH/L_BOTH. The in-function restore assertion covers the
     residuals but not critics or RNG.
   - Fix: run J_BOTH twice on the synthetic context, once with the counterfactual call
     monkeypatched to a no-op. Assert that STATE_TRACE.npz and the final critic `state_sha()`
     are bitwise equal.

3. **test_engineering.py:312-332 — the "uses one pre state" test does not exercise run_arm, and can pass vacuously.**
   - What the test does: it builds `preps` by hand and checks order independence. That
     property follows trivially from separability and says nothing about whether run_arm
     itself computes both proposals before applying either.
   - Failure (a): if both steps are REJECTED or INFEASIBLE (R set to 1e-3·(i+1) on all 64
     coordinates may already push KL past 0.001 for some random fixtures), the
     equal-results assertion holds trivially.
   - Failure (b): a regression in run_arm that interleaves `recipient_step(1)` after
     `apply_step(0)` is not caught. For J_BOTH the occupation privacy gradient would then
     still come from the old state, but under a refactor it could be recomputed.
   - Fix:
     - Assert both statuses are ACCEPTED (or use a smaller planted R).
     - Add a run_arm-level check, e.g. log a hash of `pre_vecs` plus a hash of the gradient
       inputs in UPDATE_LOG, and in the test recompute privacy and task gradients from the
       logged pre-state and compare them to the logged proposals for step 1.

4. **Missing test of per-recipient privacy-gradient correctness.**
   - Failure: no test checks that the income gradient with `active=(0,)` equals the income
     gradient with `active=(0,1)` (needed for I_ONLY/S12/S21 vs J_BOTH comparability and
     for the normalization being computed with both active). No test checks that the local
     income gradient equals ∂terms['income']/∂θ_0 only, or that the gradients match a
     finite difference. An ordering bug in `grads[2k]` (for example, if params were built
     in a different order than `sorted`) would go unnoticed.
   - Fix: on the synthetic context with warm critics and a nonzero residual:
     - compare `privacy_gradients(..., 'joint', (0,))[0]` with the `(0,1)` result to ~1e-14;
     - compare `privacy_gradients(..., 'local', (0,1))[1]` against a manual autograd of
       `terms['occupation']`;
     - run one central finite difference on a b coordinate per recipient.

## NOTES

1. **guard.py:265-308 — `active_set` indices refer to the rows kept after dropping zero-norm gradients, not to `constraint_keys`.**
   Today the KL row is the only one plausibly dropped, and it is last, so indices 0 and 1
   still mean NLL and Brier. The logs are still ambiguous. Consider also recording
   `[keys[keep][j] for j in active_set]` in the projection record.
2. **Repeat noise compares an `active=(i,)` recomputation against the original `active=(0,1)` computation** (train_worker.py:377-379).
   The two are mathematically identical. If autograd orders the sum differently, the noise,
   and therefore the contribution tolerance, can only get larger. This is conservative
   (fewer "meaningful" scans), never anti-conservative. It is acceptable but should be
   disclosed: the counterfactual_rule says "repeat".
3. **Critic reference floor on the decision column** (inherited DiagonalReference).
   Suppose a reference draw has a constant decision column, which is more likely for
   income K=2 with rank 1. The floor is then `max(initial*1e-8, 1e-12)`, and a later
   minibatch with a single flipped decision maps to ~1e12. That is loud (nonfinite critic
   loss raises, or the critic saturates), not silent. It is inherited from the predecessor;
   just be aware of it when reading the critic CEs.
4. **run() summaries** (train_worker.py:686-689).
   Unavailable arms get `{'status': 'UNAVAILABLE_FIXED_NORMALIZATION'}`, but trained arms
   have no `status` key. Downstream code that reads `summaries[arm]['status']` will
   KeyError. Consider adding `status='TRAINED'` to trained arms.
5. **test_engineering.py:156 "never_reset" is misnamed.**
   The test checks limits and trust only. The never-reset property is covered indirectly at
   test_engineering.py:309 (one nll_limit per recipient per arm), which is adequate for
   sequential arms, since the same Anchor objects are reused.
6. **Cached and deployed features are computed differently.**
   build_context computes `frozen_features` on the full 8180/2043 tensors in one call,
   while forward_raw batches 1024 rows. BLAS blocking can differ in the last bits between
   the two. Admission records `cached_vs_raw_path_probe_bitwise`, which covers the probe.
   Make sure the release/export path re-derives from raw83 (it does via forward_raw), so no
   training-time cache leaks into deployment.
7. **Separability reasoning holds only because the anchors use the task pool and the own head.**
   The cross-recipient coupling is entirely in the critic and pair terms, which enter only
   through the proposal. This is what makes the separate per-recipient QP plus ladder exact
   for the macro-step. If anyone later adds a joint (pair) task constraint, apply_step would
   have to become joint.
8. **test_engineering.py:281-286 — the RECIPE monkeypatch is shallow, and two places read checkpoint epochs differently.**
   Restoring with `clear()/update(old)` works. However, `recipe.checkpoint_labels` uses the
   module constant CHECKPOINT_EPOCHS while run_arm reads `RECIPE['checkpoint_epochs']`, so
   the test's label assertion relies on run_arm and not on checkpoint_labels. This does not
   affect production, where the two are equal.
