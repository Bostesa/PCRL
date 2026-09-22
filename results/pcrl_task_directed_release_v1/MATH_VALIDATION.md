# Finite-channel mathematical validation draft

Checked 2026-09-21 in the isolated `pcrl-task-directed-release-v1` worktree. Scope: `finite.py` and its focused test file only; no ACS fitting, holdout outcomes, or commits.

## Interface and objective

`joint_table(s,c,t,n_s,n_c,n_t,weights)` produces the normalized empirical law `p[s,c,t]`, retaining declared empty cells. Optional nonnegative weights are normalized once. `cost_table(t,y,action_probabilities,n_t,weights)` computes

`d[t,z] = sum_i normalized_weight_i * 1[t_i=t] * CE(y_i,prediction_i,z)`.

Consequently, `solve` minimizes `sum_(t,z) d[t,z] Q[t,z]`. It never multiplies by `p(t)` again. A balanced U/PWGTP utility is formed by the caller before this function. Predictions must be frozen probabilities strictly in `(0,1)`; clipping is part of the declared decoder upstream. Each role law may have a different sensitive/context alphabet but must share the input alphabet.

## Positive and zero budgets

For each role, let `a[s,c,z]=sum_t p[s,c,t]Q[t,z]` and `b[s,c,z]=p(s|c)sum_s' a[s',c,z]`. The positive-budget constraint is `sum rel_entr(a,b) <= budget`. Both arguments are affine in Q, so the objective and feasible set are convex. Zero-mass `(s,c)` rows are algebraically zero, without pseudocounts. The implementation uses [CVXPY relative entropy](https://www.cvxpy.org/api_reference/cvxpy.atoms.elementwise.html#rel-entr) and documented [solver settings](https://www.cvxpy.org/tutorial/solvers/index.html).

At zero budget, the LP uses `A@Q=0`, with `A[s,c,t]=p[s,c,t]-p(s|c)p(c,t)`. Every role contributes equations. Nonzero rows are rescaled for numerical conditioning; no empirical population is removed. Entire rows at floating-point cancellation scale are set to numerical zero before rescaling. The threshold is `16*machine_epsilon*(max_t p[s,c,t] + max_t p(s|c)p(c,t))`. This prevents order-`1e-18` product-law roundoff from becoming an order-one false restriction. The reported rank is numerical. All accepted channels are also checked against CMI recomputed from the original, unmodified joint law.

The independent checker uses NumPy tensor contraction and SciPy relative entropy, outside the optimizer expression. Information units are natural-log nats. It handles empty context/output cells using the closed relative-entropy convention.

## Unsupported states and nesting

Pass unweighted estimation frequencies explicitly as `state_mass`. An unsupported child is constrained to the frequency-weighted convex combination of its supported siblings' channel rows. If its entire coarse parent is unsupported, it is fixed to the designated zero action. With `parent=None`, an unsupported state uses the same zero-action rule. These are affine equations in the LP and cone problem.

`embedded_q` accepts an expanded fine-state channel or coarse rows indexed by `parent`. It must pass independent feasibility checks before use. Its objective gives an explicit upper-bound constraint and a fallback witness if optimization fails. A returned witness has status `feasible_witness` and `optimal=False`; it is not promoted to an optimum. Refinement and coalition ordering are checked externally under identical cost/decoder/role definitions.

## Solver acceptance

Zero-budget and unconstrained programs use SciPy HiGHS with primal/dual tolerance `1e-9`. Positive-budget programs try CLARABEL first (`max_iter=500`; absolute/relative gap and feasibility tolerances `1e-9`) and SCS second (`eps=1e-7`, `max_iters=100000`, acceleration lookback 10, normalization enabled). No retry alters a privacy budget or omits a role.

The public `ACCEPTANCE_TOLERANCES` and `SOLVER_SETTINGS` dictionaries are the configuration source. Acceptance tolerances are `1e-7` for simplex, support, CMI excess, maximum channel cleanup, and witness objective comparison; `1e-8` for nonnegativity and normalized zero-budget linear residual. These are numerical acceptance limits, not statistical privacy margins or changes to optimization budgets. Roundoff cleanup clips tiny negative entries, normalizes rows, and restores support rules; all resulting changes and original-law leakages are then checked. Cleanup above the fixed tolerance is rejected. A solver's `optimal_inaccurate` status is retained. Failed attempts, warnings, raw simplex residuals, and repairs remain recorded; metadata excluding the NumPy channel array serializes as strict JSON.

## Executed verification

Command from the worktree:

```sh
results/pcrl_task_directed_release_v1/private/venv/bin/python -m pytest tests/pcrl_task_directed_release_v1/test_finite.py -q
```

Result: **20 passed in 0.52 seconds**. Initial tests failed on the absent implementation before code was written. Two additional regressions were observed failing before their fixes: cancellation-induced false LP infeasibility and nonfinite failure metadata breaking strict JSON serialization.

The suite covers normalized U/weighted tables, exact joint-mass CE values with observation-specific predictions, constant and nonconstant perfectly private channels, independent CMI convexity, zero-budget LP verification, an analytic positive-budget binary symmetric optimum, multiple local/coalition roles with empty contexts, distinct weighted constraints, exact refinement aggregation and embedding, unsupported-child mixtures and wholly absent parents, local/coalition ordering, invalid witnesses, input validation, and real CLARABEL-to-SCS failure/retry behavior. The XOR fixture gives zero CMI under coarse context but `log(2)` under full context, explicitly demonstrating the conditional-context limitation.

These checks establish finite mathematical and numerical behavior on controlled fixtures. They do not establish privacy on a population, validate a learned teacher, or justify changing the frozen programme after outcomes are inspected.

## Dated addendum: primary ACS replay, 2026-09-22

The preceding draft records the earlier synthetic checks. It is retained unchanged. Independent replay has now checked all 81 primary ACS maps, using their actual frozen encoders, empirical laws, costs and release arrays. The three `MATH_REPLAY_PRIMARY_anchor*.json` files are the machine-readable evidence, before the registered numerical recovery described below. Evaluation remained sealed.

Across those maps, the maximum recomputed CMI budget excess was `1.6421322326905877e-13` nats, the largest simplex residual was `4.440892098500626e-16`, and recomputed objective disagreement was zero. Coarse-to-refined empirical aggregation disagreed by at most `4.163336342344337e-17`; the same embedded channels reproduced deployment laws and predictions. H byte parity and runtime input checks passed. All realized input states had mechanism-estimation support; this does not remove the need for the registered unsupported-state rule in deployment.

Feasibility is distinct from optimization. Returned statuses were 55 `optimal`, 22 `optimal_inaccurate`, and four `feasible_witness`. The four witnesses were anchor 0's `Ttask_C_0.0005_a17`, `Ttask_C_0.01_a17`, `Ttask_L_0.002_a17`, and anchor 2's `Trisk_C_0.01_a17`. In particular, the returned anchor-0 local Ttask middle-budget objective was `0.00539235351065942` nats worse than its coalition counterpart. This is a numerical optimization limitation, not a reversal of feasible-set inclusion. Neither an inaccurate status nor a fallback is an independently certified optimum. The registered Branch D permits one equivalent-formulation retry of these four failed optimization slots with identical laws, constraints, costs and tolerances; originals and failed attempts are preserved.

`ZERO_PRIVACY_CERTIFICATES.json` separately reconstructs the zero-budget systems with exact integer/rational equations from the observed counts and survey weights. All six T0 systems have supported dimension 32 and rank 31. All six refined coalition systems have dimension 64 and rank 63. Their only row-stochastic perfectly private kernels are constant over the supported input alphabet. The six refined local systems have rank 34 (anchors 0 and 1) or 32 (anchor 2), leaving nonconstant feasible kernels. These are finite empirical support results, not population impossibility or task-utility claims.

`ENCODER_DIAGNOSTICS.json` records realized code counts (32, 64, 64), 17/33 action counts, fitting-pool sizes, risk-model out-of-fit loss, and absent-class support. Native RAC1P category 4 was absent from all three risk-model training pools; its mechanism counts were 1, 1 and 0. No smoothing is used to manufacture assessed joint-law support.
