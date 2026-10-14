# MATH_REVIEW — pcrl_minimal_task_wire_adapter_v1 (code-only, nothing executed)

Verdict: NO BLOCKERS. 3 SHOULD-FIX items, 8 NOTEs. The core math matches the spec.

## Verified correct
- Trunk split: hidden64 = enc[1][0:4] and z_from_hidden = enc[1][4] are the same calls Sequential makes, so z0 is bit-equal to encode(1,x). The heads are float64 (useful_service.py:46), so everything after the trunk is float64. At R=0, b=0, q0+0+0 == q0 exactly, which gives the RAW_PROJ arithmetic (adapter.py:113, wires.py:115).
- Geometry: A = W - mean_rows(W), B = top-rank right singular vectors, A·B^T·B = A checked at 1e-10 (geometry.py:40-54).
- QP (guard.py:21-91): KKT algebra checked by hand. Ball off: mu = G^+(A_S p - b_S). Ball on: mu = G^-1(A_S p - (1+nu) b_S) and d = Pp/(1+nu) + q, with ||d|| = r because Pp is orthogonal to q. nu >= 0 is enforced via one_plus >= 1-tol, and mu >= 0, primal feasibility, equality and the ball are all checked.
- QP edge cases: with exactly duplicate rows and unequal slacks, the pair set fails the equality check, so the singleton covers it. Anti-parallel rows are handled by the singleton. When p lies in row(A_S), the ball-off candidate covers it.
- Round-off KL row (~1e-19): after normalisation its slack is ~1e16. Any active set containing it gives ||q|| >> r, or a ball-off d with norm >> r, so it is never selected and does not distort the result. Exactly-zero rows are dropped, which is correct because b >= 0 keeps d = 0 feasible. guarded_step uses the same normalisation (guard.py:198-200).
- pool_pass (guard.py:99-147):
  - CE and KL use log_softmax on the centred logits. Centring does not change the result.
  - KL = Σ p0(log p0 - log p) with p0 = exp(log_softmax of the start wire).
  - Brier is summed over the 6 classes. NLL uses the 1e-12 floor.
  - Gradients are summed per chunk and divided by n.
  - Anchor floor = int(0.005N) = 51.
  - Task weights are 8180/10223 and 2043/10223. The GUARD CE gradient is a mean over the whole pool.
- Critics: trained on occ.detach() under no_grad (train_worker.py:131). Reference stats are detached (kernels.py:61). Critic parameters are set requires_grad=False inside terms, and autograd.grad is taken w.r.t. (R, b) only. Income and hbar/q0 come from no_grad features. Source parameters are frozen and the hash is re-checked after every step (train_worker.py:339, 505).
- Risk = 1 - min(CE_k, H)/H. Joint = occ + pair (pair view = income ‖ occ). Lambda = 3·median_task/median_priv. Unavailable when the median is <= 1e-12, non-finite, or the coefficient is > 1e4 (train_worker.py:256-264).
- Matched arms:
  - Critics are seeded per (view, kind) with torch.manual_seed. The warm state's SHA is asserted equal to normalisation's (train_worker.py:563).
  - batch_rng uses one seed per seed (not per arm) and consumes 6 draws per proposal whatever the status.
  - The counterfactual consumes no RNG and changes no critic state.
  - The critic step count is asserted (train_worker.py:388).
- Counterfactual (train_worker.py:409-458): it reuses the same `current` (metrics and gradients from before the step), anchor, g_task, max_norm, ladder and solver, with privacy removed. `before` is captured before guarded_step (line 330). The state is restored from the float64 `after` vector, an exact copy, and the restore is asserted bit-equal (line 432).
- Leakage: the released occ wire is own_wire(q B), which lies in row(B). lp0 enters only the KL loss. h64, z0 and q0 are never concatenated into the wire.

## SHOULD-FIX
1. guard.py:236-239 (with guard.py:75-79): a zero step is mislabelled. When project_slack falls back to d=0 (`conservative_zero_fallback`), or when the QP solution itself is legitimately zero (proposal points into a zero-slack constraint), guarded_step accepts fraction 1.0 and reports `ROUNDING_ZERO_STEP`. That conflates a solver failure, a projection to zero and float rounding. Fix: derive the status from qp['conservative_zero_fallback'] and from ||d||==0 (e.g. FALLBACK_ZERO_STEP and PROJECTED_ZERO_STEP), so step-accounting tables are not misread.
2. guard.py:51,70: near-parallel rows can drop the solve to the zero fallback. NLL and Brier gradients are likely to be highly collinear. With cosine 1-δ, cond(G) ≈ 1/δ. The pair candidate's equality check (10·QP_TOL = 1e-9 absolute) can fail while each singleton slightly breaks the other halfspace (by more than 1e-10). The result is a zero fallback, which is safe but silent (see #1). Fix: count fallbacks per arm and report them. Alternatively, merge rows whose cosine is above 1 - 1e-8, keeping the tighter normalised slack.
3. test_engineering.py:89-120: the tests do not cover the degenerate cases above. Random Gaussian rows never produce duplicate, anti-parallel or near-parallel (cos ≥ 1-1e-10) pairs, or both halfspaces active together with the ball. Add those cases and assert there is no fallback, or that the fallback is labelled.

## NOTE
- train_worker.py:448: `privacy_accepted_fraction` is hard-coded to None. Pass in result['accepted_fraction'] so the privacy and counterfactual ladders can be compared row by row.
- train_worker.py:351: `zero_displacement` overlaps with the rejected and infeasible counts. The categories are not a partition, so do not sum them.
- guard.py:125: clamp_min gives zero NLL gradient on rows where p < 1e-12. In those cases the linearised NLL halfspace is slightly optimistic, but the finite recheck still enforces the budget.
- KL is convex with ∇K(start) ≈ 0, so the linearised KL halfspace always underestimates K. Early on, TRUST arms rely entirely on the finite ladder (1 to 1/8). Expect shortened or rejected steps near the K boundary. This is expected behaviour, not a bug.
- adapter.py:113: q0 + 0 can turn -0.0 into +0.0. Values stay equal, but tensor_sha distinguishes signed zero. The chance of this is negligible, and the test compares with array_equal, which treats ±0 as equal.
- train_worker.py:534-536: the start-KL assert compares the chunked pool_pass against the unchunked lp0. It fails loudly, not silently, if BLAS results depend on batch size. Acceptable.
- Information re-entry, a design point rather than a code bug: q = q0 + R·hbar lets R put any linear function of h64 into the 5-d row space. That includes the content of the discarded z0 off-row-space residual W4⊥·h64. wire_contract_check (adapter.py:178) only verifies literal row-space membership and head consistency. Any "no residual leak" claim must therefore rest on attacks plus the budgets, not on the contract check.
- Status semantics are otherwise sound. Infeasibility (< -1e-10) and acceptance (<= limit + 1e-10) use the same FINITE_TOL. The linearised check uses the actual post-rounding displacement. Norm repair only shrinks the step, which keeps it feasible since b >= 0. Rollback is asserted bit-exact (train_worker.py:337).
