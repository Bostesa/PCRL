# STAGE 4 — finite-channel machinery, built and validated

`experiments/pcrl_stochastic_channel_v1/channel.py`, validated by
`tests/pcrl_stochastic_channel_v1/test_channel.py` (20 tests). Numbers in
`VALIDATION_FINITE_CHANNEL.json`. No ACS pool read, no release fitted, 2016 sealed.

## What was built

| function | purpose |
|---|---|
| `cmi(p, q)` | `I(S;Z\|C)` for a finite model `p[s,c,t]` under channel `q[t,z]`. Computes task information too, by passing a task model. |
| `constraint_rows(p)` | `A[(s,c),t] = p(t,s,c) − p(s\|c)p(t,c)`. Conditional independence of a binary release ⟺ `Aq = 0`. |
| `feasible_direction(roles, task)` | The **simultaneous-view** nullspace test: stack `A_r` over roles, take the numerical nullspace, ask whether the task rows vanish on it. |
| `solve(...)` | The convex program: linear cost, `I(S_r;Z\|C_r) ≤ δ_r` per role. |
| `solve_lp(...)` | LP reference: pointwise likelihood-ratio bounds `e^{−ε} p(z\|c) ≤ p(z\|s,c) ≤ e^{ε} p(z\|c)`. |
| `sample_release` / `is_sampled_token` | The released object is a sampled one-hot token (gate G3). |

DCP formulation: `I(S;Z|C) = Σ kl_div(a, b)` exactly, because `Σa = Σb = 1` makes the `−Σa + Σb`
correction terms cancel. `a` and `b` are affine in `Q`, `kl_div` is jointly convex, so each
constraint is a convex sublevel set and the program is convex with a linear objective.

## Validation results

**The randomization payoff, through the solver.** On the review's interior-probability toy with
cost `0` iff `Z = Y`, the only perfectly-private *deterministic* map is the constant one, costing
`P(Y=1) = 1/3`. The solver at `δ = 0` reaches **0.0512 — a 6.5× improvement at exactly zero
leakage** (recomputed `cmi = 1.9e-9`).

The solver independently recovered the review's hand-constructed channel **up to relabelling of
`Z`**: reviewed `P(Z=1|T) = (1, 0, 11/13) = (1, 0, 0.846154)`; solved
`P(Z=0|T) = (0.999999, 0, 0.846256)`. Task information 0.485552 against the review's exact
0.4854855270534466. Two independent routes to the same channel.

**Feasibility test.** On the same toy: feasible, `null_dim = 2`, singular spectrum
`[0.2694, 0, 0]` — exact rank 1 — and the answer is stable across tolerances from `1e-14` to
`1e-6`, which is the sampling-sensitivity report the estimated case will need. When `T` determines
`S`, the test correctly returns infeasible with `null_dim = 1` (the constants only).

**Both role constraints are necessary, and neither implies the other.** With `Z = S xor C`:

| quantity | value (nats) |
|---|---|
| local `I(S;Z)` | 0.000000 |
| coalition `I(S;Z\|C)` | 0.693147 (`= log 2`) |
| coalition with `C` binned into one bin | **0.000000** |

and the reverse case (`B = S`, `Z = S`) gives coalition `0` with local `log 2`. Through the solver:
the XOR channel costs **0.000000** under the local constraint alone and **0.499957** once the
coalition role is bound — the task becomes unreachable, which is the correct answer.

The middle row of that table is the study's central caution, now executable: **a conditional
constraint verified on a coarsened view can be satisfied while the true conditional leak is
complete.** This is why REGISTRATION G5 requires attacks on full continuous `H` and why no fitted
model result here may be described as a certificate.

**LP reference.** At `ε = 0.05` the LP reaches objective 0.0321 with realised max log-ratio exactly
`0.0500` and achieved `cmi = 0.0006`. Compared at *matched achieved CMI*, the MI-budget program is
never more expensive — i.e. the pointwise constraint is the stricter one, as expected. The LP is
kept as a reference formulation, not built out into a parallel pipeline.

**Solver honesty (G4).** A status string is not accepted. Every returned channel is re-verified by
recomputing `cmi`, and `solve` retries on a second solver when the recomputation fails the budget.
The `δ = 0` runs sit on the exponential-cone boundary and CLARABEL routinely reports
`optimal_inaccurate` while being numerically sound (verified violation `8e-12`); the record keeps
every attempt with its status, its independently verified violation and its objective, and the
accepted answer is the feasible one with the lower objective.

## Limits carried forward

* Convexity, feasibility and optimality are properties of **the specified finite model**.
* Nothing here transfers automatically to continuous `H` or to the ACS population.
* Optimizing the codebook, optimizing decoders jointly, or maximizing a utility mutual information
  would each destroy convexity and are out of scope.
* No rate constraint is imposed, and none may be added without voiding the registration.

## Dependency added

`cvxpy` (1.9.3, CLARABEL + SCS). Added for this study only; it is not imported by any historical
module. Recorded in `REPRODUCE.md` for this study.
